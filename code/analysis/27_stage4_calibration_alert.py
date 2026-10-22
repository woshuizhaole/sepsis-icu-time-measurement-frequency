#!/usr/bin/env python3
"""Stage 4/5 calibration and alert operating metrics.

This script is deliberately descriptive on held-out test predictions. Stage-4
files contain no validation probabilities, so the official WBM validation
threshold is reported only as an external threshold-transfer diagnostic for
Stage 4, never as a model-specific tuned threshold. Pretext count/breadth
calibration is marked not estimable because pretext head logits/predictions
were not retained.
"""
from __future__ import annotations
import argparse, json, math, re, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

P = Path('/data/workplace/wyx/measurement_intensity_ssl_20261015')
D = P / 'data' / 'processed'
R = P / 'results'
L = P / 'logs' / 'qc'
_WORK_WINDOWS: dict[int, pd.DataFrame] = {}

SEEDS = [101, 202, 303, 404, 505]
FRACS = [0.01, 0.05, 0.10, 0.25, 1.00]
METHODS = ['random', 'nextmask', 'nextintensity', 'valueforecast']
ALERT_THRESHOLDS = [round(x, 2) for x in np.arange(0.01, 0.11, 0.01)] + [0.15, 0.20, 0.30, 0.40, 0.50]


def parse_stage4_name(path: Path) -> tuple[str, int, float]:
    m = re.match(r'stage4_full_pred_(.+)_seed(\d+)_f(\d+\.\d+)\.parquet$', path.name)
    if not m:
        raise ValueError(f'unexpected stage4 prediction name: {path.name}')
    return m.group(1), int(m.group(2)), float(m.group(3))


def fit_calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Fit y ~ intercept + slope*logit(p) by stable IRLS."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    x = np.column_stack([np.ones(len(p), dtype=float), np.log(p / (1.0 - p))])
    beta = np.zeros(2, dtype=float)
    for _ in range(80):
        eta = np.clip(x @ beta, -35.0, 35.0)
        mu = expit(eta)
        w = np.maximum(mu * (1.0 - mu), 1e-8)
        h = (x.T * w) @ x + np.eye(2) * 1e-8
        g = x.T @ (y - mu)
        try:
            step = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(h, g, rcond=None)[0]
        beta_new = beta + step
        if not np.all(np.isfinite(beta_new)):
            return float('nan'), float('nan')
        beta = beta_new
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return float(beta[0]), float(beta[1])


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict[str, Any]]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    q = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(q)
    if len(edges) < 2:
        idx = np.zeros(len(p), dtype=int)
        edges = np.array([float(p.min()), float(p.max()) + 1e-12])
    else:
        idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        m = idx == b
        if not np.any(m):
            continue
        out.append({
            'bin': int(b),
            'n': int(m.sum()),
            'p_lo': float(edges[b]),
            'p_hi': float(edges[b + 1]),
            'mean_pred': float(p[m].mean()),
            'observed_rate': float(y[m].mean()),
            'abs_gap': float(abs(y[m].mean() - p[m].mean())),
        })
    return out


def refractory_alerts(p: np.ndarray, hours: np.ndarray, threshold: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    hours = np.asarray(hours, dtype=float)
    z = np.zeros(len(p), dtype=np.int8)
    last = -np.inf
    for i, (h, prob) in enumerate(zip(hours, p)):
        if prob >= threshold and h > last + 12.0:
            z[i] = 1
            last = h
    return z


def patient_operating_arrays(
    pid: np.ndarray,
    hour: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    first_positive: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Return cohort operating metrics and patient-level summaries."""
    frame = pd.DataFrame({
        'patient_id': np.asarray(pid, dtype=object),
        'hour': np.asarray(hour, dtype=float),
        'y': np.asarray(y, dtype=np.int8),
        'p': np.asarray(p, dtype=float),
        'first_positive_proxy': np.asarray(first_positive, dtype=float),
    }).sort_values(['patient_id', 'hour'], kind='mergesort')
    alerts = np.zeros(len(frame), dtype=np.int8)
    for _, ix in frame.groupby('patient_id', sort=False).groups.items():
        pos = np.asarray(ix, dtype=int)
        alerts[pos] = refractory_alerts(frame.loc[pos, 'p'].to_numpy(), frame.loc[pos, 'hour'].to_numpy(), threshold)
    frame['alert'] = alerts
    patient_rows = []
    for pid_val, g in frame.groupby('patient_id', sort=False):
        g = g.sort_values('hour')
        positive = bool(g['y'].max() > 0)
        true_alert_rows = g[(g.alert == 1) & (g.y == 1)]
        first_true_hour = float(true_alert_rows.hour.iloc[0]) if len(true_alert_rows) else float('nan')
        fp = float(g.first_positive_proxy.iloc[0])
        lead = (fp - first_true_hour) if np.isfinite(first_true_hour) and np.isfinite(fp) and fp > 0 else float('nan')
        exposure_days = float((g.hour.max() - g.hour.min() + 1.0) / 24.0)
        if exposure_days <= 0:
            exposure_days = float(len(g) / 24.0) if len(g) else float('nan')
        n_alert = int(g.alert.sum())
        n_false = int(((g.alert == 1) & (g.y == 0)).sum())
        patient_rows.append({
            'patient_id': pid_val,
            'positive_patient': int(positive),
            'true_alert': int(len(true_alert_rows) > 0),
            'n_alert': n_alert,
            'n_false': n_false,
            'exposure_days': exposure_days,
            'lead_valid': int(np.isfinite(lead)),
            'lead_h': lead,
        })
    pa = pd.DataFrame(patient_rows)
    n_pat = len(pa)
    n_pos_pat = int(pa.positive_patient.sum())
    n_true_alert_pat = int(pa.true_alert.sum())
    n_alert = int(pa.n_alert.sum())
    n_false = int(pa.n_false.sum())
    exp_days = float(pa.exposure_days.sum())
    lead_vals = pa.loc[pa.lead_valid == 1, 'lead_h'].to_numpy(dtype=float)
    metrics = {
        'threshold': float(threshold),
        'n_patients': int(n_pat),
        'n_positive_patients': int(n_pos_pat),
        'n_alerts': int(n_alert),
        'n_false_alarms': int(n_false),
        'alerted_patients': int((pa.n_alert > 0).sum()),
        'sensitivity_patient': float(n_true_alert_pat / n_pos_pat) if n_pos_pat else float('nan'),
        'alerts_per_patient': float(n_alert / n_pat) if n_pat else float('nan'),
        'false_alarms_per_patient': float(n_false / n_pat) if n_pat else float('nan'),
        'alerts_per_patient_day': float(n_alert / exp_days) if exp_days > 0 else float('nan'),
        'false_alarms_per_patient_day': float(n_false / exp_days) if exp_days > 0 else float('nan'),
        'mean_true_alert_lead_h': float(np.mean(lead_vals)) if len(lead_vals) else float('nan'),
        'median_true_alert_lead_h': float(np.median(lead_vals)) if len(lead_vals) else float('nan'),
        'n_true_alert_lead': int(len(lead_vals)),
    }
    arrays = {
        'n_alert': pa.n_alert.to_numpy(dtype=float),
        'n_false': pa.n_false.to_numpy(dtype=float),
        'exposure_days': pa.exposure_days.to_numpy(dtype=float),
        'positive_patient': pa.positive_patient.to_numpy(dtype=float),
        'true_alert': pa.true_alert.to_numpy(dtype=float),
        'lead_valid': pa.lead_valid.to_numpy(dtype=float),
        'lead_sum': np.nan_to_num(pa.lead_h.to_numpy(dtype=float), nan=0.0),
    }
    return metrics, arrays


def bootstrap_summary(
    y: np.ndarray,
    p: np.ndarray,
    pid: np.ndarray,
    alert_arrays: dict[str, np.ndarray],
    bins: list[dict[str, Any]],
    b: int,
    seed: int,
) -> dict[str, float]:
    """Patient bootstrap for calibration prevalence/mean/Brier and operations."""
    pid_codes, _ = pd.factorize(np.asarray(pid, dtype=object), sort=False)
    n = int(pid_codes.max()) + 1
    sum_y = np.bincount(pid_codes, weights=np.asarray(y, dtype=float), minlength=n)
    sum_p = np.bincount(pid_codes, weights=np.asarray(p, dtype=float), minlength=n)
    sum_brier = np.bincount(pid_codes, weights=(np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2, minlength=n)
    n_rows = np.bincount(pid_codes, minlength=n).astype(float)
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in ['prevalence', 'mean_pred', 'brier', 'sensitivity_patient', 'alerts_per_patient_day', 'false_alarms_per_patient_day', 'mean_true_alert_lead_h']}
    chunk = 100
    for start in range(0, b, chunk):
        m = min(chunk, b - start)
        ix = rng.integers(0, n, size=(m, n))
        den_rows = n_rows[ix].sum(axis=1)
        vals['prevalence'].extend((sum_y[ix].sum(axis=1) / den_rows).tolist())
        vals['mean_pred'].extend((sum_p[ix].sum(axis=1) / den_rows).tolist())
        vals['brier'].extend((sum_brier[ix].sum(axis=1) / den_rows).tolist())
        pos = alert_arrays['positive_patient'][ix].sum(axis=1)
        true = alert_arrays['true_alert'][ix].sum(axis=1)
        vals['sensitivity_patient'].extend(np.divide(true, pos, out=np.full(m, np.nan), where=pos > 0).tolist())
        exp = alert_arrays['exposure_days'][ix].sum(axis=1)
        vals['alerts_per_patient_day'].extend(np.divide(alert_arrays['n_alert'][ix].sum(axis=1), exp, out=np.full(m, np.nan), where=exp > 0).tolist())
        vals['false_alarms_per_patient_day'].extend(np.divide(alert_arrays['n_false'][ix].sum(axis=1), exp, out=np.full(m, np.nan), where=exp > 0).tolist())
        lead_den = alert_arrays['lead_valid'][ix].sum(axis=1)
        vals['mean_true_alert_lead_h'].extend(np.divide(alert_arrays['lead_sum'][ix].sum(axis=1), lead_den, out=np.full(m, np.nan), where=lead_den > 0).tolist())
    out: dict[str, float] = {}
    for key, arr in vals.items():
        a = np.asarray(arr, dtype=float)
        out[f'{key}_lo'] = float(np.nanpercentile(a, 2.5))
        out[f'{key}_hi'] = float(np.nanpercentile(a, 97.5))
    return out


def init_worker() -> None:
    global _WORK_WINDOWS
    windows = pd.read_parquet(D / 'cinc2019_windows.parquet')
    parts = json.loads((D / 'partitions.json').read_text())['cinc2019']
    _WORK_WINDOWS = {}
    for seed in SEEDS:
        testp = {p for p, v in parts[str(seed)].items() if v == 'test'}
        _WORK_WINDOWS[seed] = windows[windows.patient_id.isin(testp)].sort_values(['patient_id', 'anchor_hour']).reset_index(drop=True)

def stage4_worker(path_str: str, thresholds: dict[int, float], bootstrap_b: int) -> dict[str, Any]:
    path = Path(path_str)
    method, seed, fraction = parse_stage4_name(path)
    pred = pd.read_parquet(path)
    if pred.empty or pred[['patient_id', 'y', 'p']].isna().any().any():
        raise ValueError(f'empty or missing prediction values: {path}')
    w = _WORK_WINDOWS[seed]
    if len(w) != len(pred):
        raise ValueError(f'row mismatch {path}: pred={len(pred)} windows={len(w)}')
    # Existing Stage-4 writer is deterministic patient/hour order; verify labels and IDs.
    if not np.array_equal(w.patient_id.to_numpy(dtype=object), pred.patient_id.to_numpy(dtype=object)):
        raise ValueError(f'patient order mismatch: {path}')
    if not np.array_equal(w.y_h6.to_numpy(dtype=np.int8), pred.y.to_numpy(dtype=np.int8)):
        raise ValueError(f'label order mismatch: {path}')
    y = pred.y.to_numpy(dtype=np.int8)
    p = np.clip(pred.p.to_numpy(dtype=float), 1e-7, 1 - 1e-7)
    pid = pred.patient_id.to_numpy(dtype=object)
    hours = w.anchor_hour.to_numpy(dtype=float)
    first_pos = w.first_positive_proxy.to_numpy(dtype=float)
    prevalence = float(y.mean())
    mean_pred = float(p.mean())
    brier = float(np.mean((p - y) ** 2))
    intercept, slope = fit_calibration(y, p)
    bins = calibration_bins(y, p, 10)
    ece = float(sum(x['n'] * x['abs_gap'] for x in bins) / len(y)) if bins else float('nan')
    mce = float(max((x['abs_gap'] for x in bins), default=float('nan')))
    threshold = float(thresholds[seed])
    op, arrays = patient_operating_arrays(pid, hours, y, p, first_pos, threshold)
    boot = bootstrap_summary(y, p, pid, arrays, bins, bootstrap_b, seed + int(round(fraction * 1000)) + METHODS.index(method) * 100000)
    row = {
        'stage': 'Stage4_full_test', 'method': method, 'seed': seed, 'fraction': fraction,
        'threshold': threshold, 'threshold_source': 'Stage5_WBM_validation_utility_external_transfer',
        'n_test_rows': int(len(y)), 'n_test_patients': int(pd.unique(pid).size), 'positive_test_rows': int(y.sum()),
        'prevalence': prevalence, 'mean_pred': mean_pred, 'brier': brier,
        'calibration_intercept': intercept, 'calibration_slope': slope, 'ece_10': ece, 'mce_10': mce,
    }
    row.update({f'cal_{k}': v for k, v in boot.items() if k in ['prevalence_lo','prevalence_hi','mean_pred_lo','mean_pred_hi','brier_lo','brier_hi']})
    row.update({f'alert_{k}': v for k, v in op.items()})
    row.update({f'alert_{k}': v for k, v in boot.items() if k in ['sensitivity_patient_lo','sensitivity_patient_hi','alerts_per_patient_day_lo','alerts_per_patient_day_hi','false_alarms_per_patient_day_lo','false_alarms_per_patient_day_hi','mean_true_alert_lead_h_lo','mean_true_alert_lead_h_hi']})
    # Point curve for fixed, predeclared thresholds. No threshold is selected from test labels.
    curve = []
    for th in ALERT_THRESHOLDS:
        cm, _ = patient_operating_arrays(pid, hours, y, p, first_pos, float(th))
        curve.append({
            'stage': 'Stage4_full_test', 'method': method, 'seed': seed, 'fraction': fraction,
            'threshold': float(th), 'threshold_source': 'predeclared_test_descriptive_grid',
            'n_test_rows': int(len(y)), 'n_test_patients': int(pd.unique(pid).size), 'positive_test_rows': int(y.sum()),
            **cm,
        })
    return {'summary': row, 'bins': [dict(stage='Stage4_full_test', method=method, seed=seed, fraction=fraction, **x) for x in bins], 'curve': curve}


def read_official_thresholds() -> dict[int, float]:
    d = pd.read_csv(R / 'stage5_official_utility_by_seed.csv')
    return {int(r.seed): float(r.threshold) for _, r in d.iterrows()}


def official_wbm_metrics(seed: int, threshold: float, raw: pd.DataFrame) -> dict[str, Any]:
    root = R / f'stage5_utility_seed{seed}'
    pred_dir = root / 'predictions'
    labels_dir = root / 'labels'
    rows = []
    for pf in sorted(pred_dir.glob('*.psv')):
        pid = pf.stem
        ptab = pd.read_csv(pf, sep='|')
        rr = raw[raw.patient_id == pid].sort_values('ICULOS')
        if len(rr) != len(ptab):
            raise ValueError(f'official trajectory length mismatch seed={seed} pid={pid}: raw={len(rr)} pred={len(ptab)}')
        rows.append(pd.DataFrame({
            'patient_id': pid,
            'hour': rr.ICULOS.to_numpy(dtype=float),
            'y': rr.SepsisLabel.to_numpy(dtype=np.int8),
            'p': ptab.PredictedProbability.to_numpy(dtype=float),
            'first_positive_proxy': float(rr.loc[rr.SepsisLabel == 1, 'ICULOS'].min()) if (rr.SepsisLabel == 1).any() else -1.0,
        }))
    allf = pd.concat(rows, ignore_index=True)
    # Full-trajectory operating metrics; official scorer uses this same refractory rule.
    full, _ = patient_operating_arrays(allf.patient_id.to_numpy(), allf.hour.to_numpy(), allf.y.to_numpy(), allf.p.to_numpy(), allf.first_positive_proxy.to_numpy(), threshold)
    # Pre-first-positive sensitivity/alert burden is the proposal's main operating cohort.
    keep = (allf.first_positive_proxy.to_numpy() < 0) | (allf.hour.to_numpy() < allf.first_positive_proxy.to_numpy())
    pre, _ = patient_operating_arrays(allf.patient_id.to_numpy()[keep], allf.hour.to_numpy()[keep], allf.y.to_numpy()[keep], allf.p.to_numpy()[keep], allf.first_positive_proxy.to_numpy()[keep], threshold)
    return {
        'stage': 'Stage5_WBM_official_test', 'seed': int(seed), 'fraction': np.nan, 'method': 'WBM',
        'threshold': float(threshold), 'threshold_source': 'Stage5_WBM_validation_utility',
        'n_test_rows': int(len(allf)), 'n_test_patients': int(allf.patient_id.nunique()), 'positive_test_rows': int(allf.y.sum()),
        **{f'full_{k}': v for k, v in full.items()},
        **{f'pre_first_positive_{k}': v for k, v in pre.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--bootstrap', type=int, default=2000)
    args = ap.parse_args()
    t0 = time.time()
    R.mkdir(parents=True, exist_ok=True); L.mkdir(parents=True, exist_ok=True)
    thresholds = read_official_thresholds()
    paths = []
    for seed in SEEDS:
        for method in METHODS:
            for fraction in FRACS:
                path = R / f'stage4_full_pred_{method}_seed{seed}_f{fraction:.2f}.parquet'
                if path.exists(): paths.append(path)
    expected = len(SEEDS) * len(METHODS) * len(FRACS)
    if len(paths) != expected:
        raise RuntimeError(f'expected {expected} Stage4 full predictions, found {len(paths)}')
    log_lines = [f'start={time.strftime("%Y-%m-%d %H:%M:%S")}', f'workers={args.workers}', f'bootstrap={args.bootstrap}', f'files={len(paths)}']
    summaries = []; bins = []; curves = []; failures = []
    # Windows are immutable and copied to worker processes through fork on Linux; each worker verifies row/order.
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as ex:
        futs = {ex.submit(stage4_worker, str(path), thresholds, int(args.bootstrap)): path for path in paths}
        for fut in as_completed(futs):
            path = futs[fut]
            try:
                out = fut.result(); summaries.append(out['summary']); bins.extend(out['bins']); curves.extend(out['curve'])
                log_lines.append(f'PASS {path.name}')
            except Exception as e:
                failures.append({'file': path.name, 'error': repr(e), 'traceback': traceback.format_exc()}); log_lines.append(f'FAIL {path.name} {e!r}')
    if failures:
        (L / 'stage4_calibration_alert_failures.json').write_text(json.dumps(failures, indent=2))
        raise RuntimeError(f'{len(failures)} worker failures; see stage4_calibration_alert_failures.json')
    s4 = pd.DataFrame(summaries).sort_values(['method', 'seed', 'fraction']).reset_index(drop=True)
    b4 = pd.DataFrame(bins).sort_values(['method', 'seed', 'fraction', 'bin']).reset_index(drop=True)
    c4 = pd.DataFrame(curves).sort_values(['method', 'seed', 'fraction', 'threshold']).reset_index(drop=True)
    s4.to_csv(R / 'stage4_calibration_alert_metrics.csv', index=False)
    b4.to_csv(R / 'stage4_calibration_bins.csv', index=False)
    c4.to_csv(R / 'stage4_alert_curve.csv', index=False)
    raw = pd.read_parquet(D / 'cinc2019_hourly' / 'data.parquet', columns=['patient_id', 'ICULOS', 'SepsisLabel'])
    official = pd.DataFrame([official_wbm_metrics(seed, thresholds[seed], raw) for seed in SEEDS]).sort_values('seed')
    official.to_csv(R / 'stage5_official_alert_metrics.csv', index=False)
    pretext_status = {
        'stage': 'Stage4_pretext_calibration',
        'status': 'NOT_ESTIMABLE',
        'reason': 'Executed pretext artifacts retain encoder weights and aggregate loss metadata only; no held-out target logits/probabilities or per-channel count/breadth prediction distributions were saved.',
        'checked_files': sorted(str(x.relative_to(P)) for x in R.glob('stage4_pretrain_*.json')),
        'required_for_valid_calibration': ['held-out pretext targets', 'per-channel Binomial/Categorical probabilities or logits', 'fixed pretext evaluation split'],
        'do_not_infer_from': ['downstream sepsis probabilities', 'terminal/IPW point estimates', 'pretraining loss alone'],
    }
    (R / 'stage4_pretext_calibration_status.json').write_text(json.dumps(pretext_status, indent=2))
    qc = {
        'stage': 'Stage4/7 calibration_alert',
        'status': 'PASS' if len(s4) == expected and len(official) == len(SEEDS) and len(failures) == 0 else 'PARTIAL',
        'stage4_expected_files': expected, 'stage4_files_processed': int(len(s4)),
        'stage4_methods': METHODS, 'stage4_seeds': SEEDS, 'stage4_fractions': FRACS,
        'stage4_test_alignment': 'patient_id and y_h6 sequence exactly matched fixed windows per seed',
        'stage4_calibration': 'test descriptive; no test recalibration or threshold selection',
        'stage4_alert_threshold': 'external WBM validation utility threshold transferred by seed; not model-specific tuning',
        'stage4_alert_curve': 'predeclared threshold grid, descriptive on test, no threshold selected from test labels',
        'stage5_official_alert': 'full-trajectory and pre-first-positive WBM operating metrics from scorer-aligned test predictions',
        'pretext_count_breadth_calibration': 'NOT_ESTIMABLE; see results/stage4_pretext_calibration_status.json',
        'bootstrap_replicates': int(args.bootstrap), 'bootstrap_unit': 'patient',
        'runtime_seconds': float(time.time() - t0),
        'limitations': [
            'Stage4 test files contain no validation probabilities, so external WBM threshold transfer cannot be interpreted as Stage4 validation-optimal operating point.',
            'Stage4 full predictions are pre-first-positive window cohort; they do not reproduce full-trajectory official utility.',
            'Calibration slope/intercept are descriptive test estimates; no model recalibration was performed.',
            'Frozen-head has no retained patient-level test prediction files and therefore is not included in calibration/alert tables.',
        ],
    }
    (R / 'stage4_calibration_alert_qc.json').write_text(json.dumps(qc, indent=2))
    (L / 'stage4_calibration_alert.log').write_text('\n'.join(log_lines) + f'\nend={time.strftime("%Y-%m-%d %H:%M:%S")} runtime_s={time.time()-t0:.2f}\n')
    print(json.dumps({'qc': qc, 'summary_rows': len(s4), 'bin_rows': len(b4), 'curve_rows': len(c4), 'official_rows': len(official)}, indent=2))


if __name__ == '__main__':
    main()
