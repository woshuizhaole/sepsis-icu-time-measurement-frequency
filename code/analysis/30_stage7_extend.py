from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import numpy as np
import pandas as pd

P = Path('/data/workplace/wyx/measurement_intensity_ssl_20261015')
R = P / 'results'
S = R / 'stats'
S.mkdir(parents=True, exist_ok=True)

# ---- required source evidence ----
stage3_path = R / 'stage3_grud_cde_results.csv'
stage3_qc_path = R / 'stage3_grud_cde_qc.json'
stage4_path = R / 'stage4_calibration_alert_metrics.csv'
stage4_qc_path = R / 'stage4_calibration_alert_qc.json'
stage4_warning_path = R / 'stage4_calibration_alert_warning_audit.json'
stage5_alert_path = R / 'stage5_official_alert_metrics.csv'
stage5_alert_qc_path = R / 'stage5_official_alert_qc.json'
transfer_path = R / 'stage6_transfer_results_all_seeds.csv'
transfer_qc_path = R / 'stage6_transfer_qc.json'
for p in [stage3_path, stage3_qc_path, stage4_path, stage4_qc_path, stage4_warning_path, stage5_alert_path, stage5_alert_qc_path, transfer_path, transfer_qc_path]:
    if not p.exists():
        raise FileNotFoundError(p)

stage3 = pd.read_csv(stage3_path)
stage3_qc = json.loads(stage3_qc_path.read_text())
if len(stage3) != 20 or stage3.duplicated(['architecture', 'condition', 'seed']).any():
    raise RuntimeError('Stage3 GRU-D/CDE source is incomplete or duplicated')
if stage3_qc.get('status') != 'PASS':
    raise RuntimeError(f'Stage3 QC not PASS: {stage3_qc.get("status")}')

stage4 = pd.read_csv(stage4_path)
stage4_qc = json.loads(stage4_qc_path.read_text())
if len(stage4) != 100 or stage4.duplicated(['method', 'seed', 'fraction']).any():
    raise RuntimeError('Stage4 calibration source is incomplete or duplicated')
if stage4_qc.get('computed_scope_status') != 'PASS':
    raise RuntimeError(f'Stage4 calibration computed scope not PASS: {stage4_qc}')

stage5_alert = pd.read_csv(stage5_alert_path)
stage5_alert_qc = json.loads(stage5_alert_qc_path.read_text())
if len(stage5_alert) != 5 or stage5_alert_qc.get('status') != 'PASS':
    raise RuntimeError('Stage5 official alert source is incomplete or not PASS')

transfer = pd.read_csv(transfer_path)
transfer_qc = json.loads(transfer_qc_path.read_text())
if len(transfer) != 45 or transfer.duplicated(['seed', 'direction', 'task', 'cutoff_hour', 'mode']).any():
    raise RuntimeError('Stage6 transfer source is incomplete or duplicated')
if transfer_qc.get('status') != 'PASS':
    raise RuntimeError(f'Stage6 transfer QC not PASS: {transfer_qc.get("status")}')
# Critical scientific gate: every rerun must use the seed-specific P19 cache.
source_records = []
for seed in [101, 202, 303, 404, 505]:
    sp = R / f'stage6_transfer_source_seed{seed}.json'
    if not sp.exists():
        raise RuntimeError(f'missing transfer source record: {sp}')
    rec = json.loads(sp.read_text())
    source_records.append(rec)
if not all(bool(x.get('p19_sequence_seed_specific')) for x in source_records):
    raise RuntimeError('Stage6 transfer source cache gate failed: not all seeds are seed-specific')
if transfer_qc.get('seed_specific_cache_all') is False:
    raise RuntimeError('Stage6 transfer QC explicitly reports a non-seed-specific cache')

# ---- append point estimates and available patient-bootstrap CIs ----
metrics_path = R / 'metrics.parquet'
bootstrap_path = R / 'bootstrap_ci.csv'
metrics = pd.read_parquet(metrics_path)
bootstrap = pd.read_csv(bootstrap_path)
new_stages = {'Stage3_GRUD_CDE', 'Stage4_Calibration_Alert', 'Stage5_WBM_Alert', 'Stage6_Transfer'}
metrics = metrics[~metrics.stage.isin(new_stages)].copy()
bootstrap = bootstrap[~bootstrap.stage.isin(new_stages)].copy()
metric_rows = []
ci_rows = []

def add_metric(stage, model, seed, fraction, metric, value):
    if value is None or not np.isfinite(float(value)):
        return
    metric_rows.append({'stage': stage, 'model': str(model), 'seed': int(seed), 'fraction': float(fraction) if np.isfinite(fraction) else np.nan, 'metric': metric, 'value': float(value)})

def add_ci(stage, model, seed, fraction, metric, estimate, lo, hi):
    if estimate is None or not np.isfinite(float(estimate)):
        return
    if lo is None or hi is None or not np.isfinite(float(lo)) or not np.isfinite(float(hi)):
        return
    ci_rows.append({'stage': stage, 'model': str(model), 'seed': int(seed), 'fraction': float(fraction) if np.isfinite(fraction) else np.nan, 'metric': metric, 'estimate': float(estimate), 'lo': float(lo), 'hi': float(hi)})

# Stage 3 GRU-D/CDE
for r in stage3.itertuples(index=False):
    model = f'{r.architecture}:{r.condition}'
    for m in ['auprc', 'auroc', 'brier']:
        add_metric('Stage3_GRUD_CDE', model, r.seed, np.nan, m, getattr(r, m))
        add_ci('Stage3_GRUD_CDE', model, r.seed, np.nan, m, getattr(r, m), getattr(r, m + '_lo'), getattr(r, m + '_hi'))

# Stage 4 held-out calibration and external-threshold operating diagnostics.
stage4_point_names = {
    'prevalence': 'prevalence', 'mean_pred': 'mean_pred', 'brier': 'brier',
    'calibration_intercept': 'calibration_intercept', 'calibration_slope': 'calibration_slope',
    'ece_10': 'ece_10', 'mce_10': 'mce_10',
    'alert_sensitivity_patient': 'alert_sensitivity_patient',
    'alert_alerts_per_patient_day': 'alerts_per_patient_day',
    'alert_false_alarms_per_patient_day': 'false_alarms_per_patient_day',
    'alert_mean_true_alert_lead_h': 'mean_true_alert_lead_h',
}
stage4_ci_names = {
    'prevalence': 'prevalence', 'mean_pred': 'mean_pred', 'brier': 'brier',
    'alert_sensitivity_patient': 'sensitivity_patient',
    'alert_alerts_per_patient_day': 'alerts_per_patient_day',
    'alert_false_alarms_per_patient_day': 'false_alarms_per_patient_day',
    'alert_mean_true_alert_lead_h': 'mean_true_alert_lead_h',
}
for r in stage4.itertuples(index=False):
    for src, dst in stage4_point_names.items():
        add_metric('Stage4_Calibration_Alert', r.method, r.seed, r.fraction, dst, getattr(r, src))
    for src, dst in stage4_ci_names.items():
        add_ci('Stage4_Calibration_Alert', r.method, r.seed, r.fraction, dst, getattr(r, src), getattr(r, 'cal_' + dst + '_lo') if dst in ['prevalence','mean_pred','brier'] else getattr(r, 'alert_' + dst + '_lo'), getattr(r, 'cal_' + dst + '_hi') if dst in ['prevalence','mean_pred','brier'] else getattr(r, 'alert_' + dst + '_hi'))

# Official WBM operating metrics. These use the corrected y_h6 first-positive-aligned stream.
for r in stage5_alert.itertuples(index=False):
    for prefix in ['pre_first_positive', 'full']:
        for m in ['brier', 'sensitivity_patient', 'alerts_per_patient_day', 'false_alarms_per_patient_day', 'mean_true_alert_lead_h']:
            src = f'{prefix}_{m}'
            if hasattr(r, src):
                add_metric('Stage5_WBM_Alert', 'WBM_official', r.seed, np.nan, f'{prefix}_{m}', getattr(r, src))
                lo = f'{prefix}_boot_{m}_lo'; hi = f'{prefix}_boot_{m}_hi'
                if hasattr(r, lo) and hasattr(r, hi):
                    add_ci('Stage5_WBM_Alert', 'WBM_official', r.seed, np.nan, f'{prefix}_{m}', getattr(r, src), getattr(r, lo), getattr(r, hi))

# Stage 6 corrected five-seed transfer matrix.
for r in transfer.itertuples(index=False):
    cut = float(r.cutoff_hour) if np.isfinite(r.cutoff_hour) else np.nan
    cut_label = 'nan' if not np.isfinite(cut) else f'{cut:g}'
    model = f'{r.direction}|{r.task}|cut{cut_label}|{r.mode}'
    for m in ['auprc', 'auroc', 'brier', 'patient_auprc', 'patient_auroc', 'patient_brier']:
        if hasattr(r, m):
            add_metric('Stage6_Transfer', model, r.seed, cut, m, getattr(r, m))
            add_ci('Stage6_Transfer', model, r.seed, cut, m, getattr(r, m), getattr(r, m + '_lo'), getattr(r, m + '_hi'))

new_metrics = pd.DataFrame(metric_rows, columns=metrics.columns)
new_ci = pd.DataFrame(ci_rows, columns=bootstrap.columns)
metrics = pd.concat([metrics, new_metrics], ignore_index=True)
bootstrap = pd.concat([bootstrap, new_ci], ignore_index=True)
metrics.to_parquet(metrics_path, index=False)
bootstrap.to_csv(bootstrap_path, index=False)

summary = metrics[metrics.stage.isin(new_stages)].groupby(['stage','model','metric'], dropna=False)['value'].agg(n='count', median='median', mean='mean', min='min', max='max').reset_index()
summary.to_csv(R / 'stage7_new_evidence_summary.csv', index=False)

# QC for the extension itself.
gap_qc = {
    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'PASS',
    'stage3_grud_cde_rows': int(len(stage3)),
    'stage4_calibration_rows': int(len(stage4)),
    'stage5_official_alert_rows': int(len(stage5_alert)),
    'stage6_transfer_rows': int(len(transfer)),
    'stage6_seed_specific_cache_all': True,
    'new_metrics_rows': int(len(new_metrics)),
    'new_bootstrap_rows': int(len(new_ci)),
    'metrics_total_rows': int(len(metrics)),
    'bootstrap_total_rows': int(len(bootstrap)),
    'nonestimable_boundaries': {
        'stage4_pretext_count_breadth': 'NOT_ESTIMABLE',
        'stage4_lead_time_no_true_alert_rows': int(json.loads(stage4_warning_path.read_text()).get('lead_time_point_not_estimable', 0)),
        'stage4_frozen_head_calibration_alert': 'NOT_ESTIMABLE',
    },
    'transfer_invalid_basecache_excluded': (R / 'stage6_transfer_invalid_basecache_20261017' / 'INVALID_REASON.md').exists(),
    'no_new_superiority_claim': True,
}
(R / 'stats/stage7_gap_qc.json').write_text(json.dumps(gap_qc, indent=2) + '\n')
print(json.dumps(gap_qc, indent=2))
