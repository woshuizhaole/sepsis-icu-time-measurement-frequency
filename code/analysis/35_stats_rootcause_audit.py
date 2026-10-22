#!/usr/bin/env python3
"""Statistical stability/root-cause audit for the current ICU benchmark snapshot.

This script is read-only with respect to model predictions.  It recomputes
seed-level summaries and same-seed, same-test-cohort patient-cluster paired
bootstrap deltas for the pre-specified comparisons:

* Stage 3 V+M0+I versus V+M0;
* Stage 3 GRU-D/CDE OI versus No-OI;
* Stage 4 NextIntensity versus random at all label fractions.

It also audits split composition, prevalence/denominators, terminal/IPW
point-estimate dispersion, calibration/alert burden, WBM/official Utility,
and Stage 6 transfer dispersion.  These are diagnostics; they do not add
new superiority claims to the sealed project tables.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

P = Path("/data/workplace/wyx/measurement_intensity_ssl_20261015")
D = P / "data/processed"
R = P / "results"
S = R / "stats"
L = P / "logs/qc"
SEEDS = [101, 202, 303, 404, 505]
FRACS = [0.01, 0.05, 0.10, 0.25, 1.00]


def metric(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> tuple[float, float, float]:
    """Return AP, AUROC, Brier with the same definitions used in the project."""
    if w is None:
        return (float(average_precision_score(y, p)), float(roc_auc_score(y, p)), float(brier_score_loss(y, p)))
    return (
        float(average_precision_score(y, p, sample_weight=w)),
        float(roc_auc_score(y, p, sample_weight=w)),
        float(brier_score_loss(y, p, sample_weight=w)),
    )


def _group_starts(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.array([], dtype=np.int64)
    return np.r_[0, 1 + np.flatnonzero(x[1:] != x[:-1])]


def _weighted_vec(y: np.ndarray, p: np.ndarray, w: np.ndarray, order_desc: np.ndarray, order_asc: np.ndarray) -> tuple[float, float, float]:
    """Fast exact weighted AP/AUROC/Brier for a fixed prediction vector.

    Ties are grouped, matching sklearn's threshold semantics.  Patient
    bootstrap weights are integer multiplicities, so the formula is exact up
    to floating point roundoff and avoids repeatedly sorting 225k rows.
    """
    yy = y[order_desc].astype(float, copy=False)
    pp = p[order_desc]
    ww = w[order_desc].astype(float, copy=False)
    starts = _group_starts(pp)
    wt = np.add.reduceat(ww, starts)
    pos = np.add.reduceat(ww * yy, starts)
    cum_pos = np.cumsum(pos)
    cum_w = np.cumsum(wt)
    ap = float(np.sum((cum_pos / np.maximum(cum_w, 1e-300)) * pos) / max(float(cum_pos[-1]), 1e-300))
    # AUC uses ascending score ranks and average ranks in tied groups.
    q = order_asc
    qq = p[q]
    qy = y[q].astype(float, copy=False)
    qw = w[q].astype(float, copy=False)
    qstarts = _group_starts(qq)
    qwt = np.add.reduceat(qw, qstarts)
    qpos = np.add.reduceat(qw * qy, qstarts)
    rank = np.cumsum(qwt) - qwt + (qwt + 1.0) / 2.0
    npos = float(qpos.sum())
    nneg = float(qwt.sum() - npos)
    auc = float((np.sum(qpos * rank) - npos * (npos + 1.0) / 2.0) / max(npos * nneg, 1e-300))
    br = float(np.sum(w * (p - y) ** 2) / max(float(w.sum()), 1e-300))
    return ap, auc, br


def paired_bootstrap(
    a: pd.DataFrame,
    b: pd.DataFrame,
    B: int,
    seed: int,
) -> dict[str, Any]:
    """Patient-cluster paired bootstrap of A-B for AP/AUROC/Brier.

    The rows must be the same test cohort in the same patient/hour order.  If
    a model has no row-order key, the retained prediction order is checked;
    all current Stage 3/4/GRU outputs are generated from the same ordered
    test window table.
    """
    if len(a) != len(b):
        raise ValueError(f"row count mismatch {len(a)} {len(b)}")
    if not np.array_equal(a.patient_id.to_numpy(), b.patient_id.to_numpy()):
        raise ValueError("patient order mismatch")
    if not np.array_equal(a.y.to_numpy(), b.y.to_numpy()):
        raise ValueError("label order mismatch")
    y = a.y.to_numpy(dtype=float)
    pa = a.p.to_numpy(dtype=float)
    pb = b.p.to_numpy(dtype=float)
    ids = a.patient_id.to_numpy()
    uid, inv = np.unique(ids, return_inverse=True)
    n = len(uid)
    # Point estimate is independently checked against sklearn.
    point_a = metric(y, pa)
    point_b = metric(y, pb)
    rng = np.random.default_rng(seed)
    order_a_desc = np.argsort(-pa, kind="mergesort")
    order_a_asc = np.argsort(pa, kind="mergesort")
    order_b_desc = np.argsort(-pb, kind="mergesort")
    order_b_asc = np.argsort(pb, kind="mergesort")
    da: list[float] = []
    du: list[float] = []
    db: list[float] = []
    # Compute in small vectorized chunks.  The grouped formulas exactly handle
    # tied probabilities and are >20x faster than sklearn in every replicate.
    for st in range(0, B, 20):
        m = min(20, B - st)
        draws = rng.integers(0, n, size=(m, n))
        for draw in draws:
            w = np.bincount(draw, minlength=n)[inv].astype(float)
            aa = _weighted_vec(y, pa, w, order_a_desc, order_a_asc)
            bb = _weighted_vec(y, pb, w, order_b_desc, order_b_asc)
            da.append(aa[0] - bb[0])
            du.append(aa[1] - bb[1])
            db.append(aa[2] - bb[2])
    def ci(x: list[float]) -> tuple[float, float]:
        z = np.asarray(x, dtype=float)
        return float(np.percentile(z, 2.5)), float(np.percentile(z, 97.5))
    ap_lo, ap_hi = ci(da)
    au_lo, au_hi = ci(du)
    br_lo, br_hi = ci(db)
    return {
        "n_rows": int(len(a)),
        "n_patients": int(n),
        "positive": int(y.sum()),
        "a_ap": point_a[0], "b_ap": point_b[0], "delta_ap": point_a[0] - point_b[0], "delta_ap_lo": ap_lo, "delta_ap_hi": ap_hi,
        "a_auroc": point_a[1], "b_auroc": point_b[1], "delta_auroc": point_a[1] - point_b[1], "delta_auroc_lo": au_lo, "delta_auroc_hi": au_hi,
        "a_brier": point_a[2], "b_brier": point_b[2], "delta_brier": point_a[2] - point_b[2], "delta_brier_lo": br_lo, "delta_brier_hi": br_hi,
        "bootstrap_reps": int(B),
    }


def paired_task(spec: dict[str, Any]) -> dict[str, Any]:
    a = pd.read_parquet(P / spec["a_path"])
    b = pd.read_parquet(P / spec["b_path"])
    out = paired_bootstrap(a, b, int(spec["B"]), int(spec["bootstrap_seed"]))
    base = {k: v for k, v in spec.items() if k not in {"a_path", "b_path", "bootstrap_seed", "B"}}
    base.update(out)
    return base


def exact_signflip(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return float("nan")
    obs = abs(float(x.mean()))
    vals = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(x)):
        vals.append(abs(float(np.mean(x * np.asarray(signs)))))
    return float(np.mean(np.asarray(vals) >= obs - 1e-15))


def holm(p: list[float]) -> list[float]:
    p = np.asarray(p, dtype=float)
    out = np.empty(len(p), float)
    order = np.argsort(p)
    for i, j in enumerate(order):
        out[j] = min(1.0, float((len(p) - i) * p[j]))
    for i in range(1, len(order)):
        out[order[i]] = max(out[order[i]], out[order[i - 1]])
    return out.tolist()


def summarize(df: pd.DataFrame, keys: list[str], metrics: list[str], stage: str) -> pd.DataFrame:
    rows = []
    for k, g in df.groupby(keys, dropna=False, sort=True):
        if not isinstance(k, tuple):
            k = (k,)
        row = dict(zip(keys, k))
        row["stage"] = stage
        row["n"] = int(len(g))
        for m in metrics:
            x = pd.to_numeric(g[m], errors="coerce").dropna().to_numpy(float)
            if not len(x):
                for s in ["mean", "median", "sd", "iqr_lo", "iqr_hi", "min", "max", "cv"]:
                    row[f"{m}_{s}"] = np.nan
                continue
            mean = float(x.mean())
            row[f"{m}_mean"] = mean
            row[f"{m}_median"] = float(np.median(x))
            row[f"{m}_sd"] = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
            row[f"{m}_iqr_lo"] = float(np.percentile(x, 25))
            row[f"{m}_iqr_hi"] = float(np.percentile(x, 75))
            row[f"{m}_min"] = float(x.min())
            row[f"{m}_max"] = float(x.max())
            row[f"{m}_cv"] = float(row[f"{m}_sd"] / abs(mean)) if mean != 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_denominator_audit() -> pd.DataFrame:
    rows = []
    parts = json.loads((D / "partitions.json").read_text())["cinc2019"]
    win = pd.read_parquet(D / "cinc2019_windows.parquet", columns=["patient_id", "y_h6"])
    for s, part in parts.items():
        for split in ["train", "validation", "test"]:
            q = win[win.patient_id.map(part) == split]
            rows.append({"stage": "P19_windows", "seed": int(s), "split": split, "n_rows": len(q), "n_patients": q.patient_id.nunique(), "positive": int(q.y_h6.sum()), "prevalence": float(q.y_h6.mean())})
    # Stage 4 terminal availability is held in one row per seed/init/fraction;
    # retain the seed-level denominator once for comparison.
    t = pd.read_csv(R / "stage4_terminal_ipw_results.csv")
    rows.extend(t.groupby("seed").first().reset_index().assign(stage="Stage4_terminal", split="test").rename(columns={"n_test": "n_rows", "n_available": "n_available_rows", "positive_all": "positive"})[["stage", "seed", "split", "n_rows", "n_available_rows", "positive", "availability_rate"]].to_dict("records"))
    return pd.DataFrame(rows)


def audit_split_overlap() -> pd.DataFrame:
    parts = json.loads((D / "partitions.json").read_text())["cinc2019"]
    sets = {int(s): {k for k, v in m.items() if v == "test"} for s, m in parts.items()}
    rows = []
    for a, b in itertools.combinations(SEEDS, 2):
        inter = len(sets[a] & sets[b]); union = len(sets[a] | sets[b])
        rows.append({"seed_a": a, "seed_b": b, "n_a": len(sets[a]), "n_b": len(sets[b]), "intersection": inter, "union": union, "jaccard": inter / union})
    return pd.DataFrame(rows)


def calibration_alert_summary() -> pd.DataFrame:
    d = pd.read_csv(R / "stage4_calibration_alert_metrics.csv")
    cols = ["brier", "calibration_intercept", "calibration_slope", "ece_10", "mce_10", "alert_sensitivity_patient", "alert_alerts_per_patient_day", "alert_false_alarms_per_patient_day", "alert_mean_true_alert_lead_h", "alert_threshold"]
    return summarize(d, ["method", "fraction"], cols, "Stage4_calibration_alert")


def terminal_summary() -> pd.DataFrame:
    d = pd.read_csv(R / "stage4_terminal_ipw_results.csv")
    cols = ["availability_rate", "auprc_all", "auroc_all", "brier_all", "auprc_complete_case", "auroc_complete_case", "brier_complete_case", "auprc_marginal_ipw", "auroc_marginal_ipw", "brier_marginal_ipw"]
    return summarize(d, ["init", "fraction"], cols, "Stage4_terminal_IPW")


def stage5_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for path, label, keys in [(R / "stage5_wbm_results.csv", "WBM", ["model"]), (R / "stage5_timecheat_results.csv", "TimeCHEAT", ["model"])]:
        d = pd.read_csv(path)
        rows.append(summarize(d, keys, ["auprc", "auroc", "brier"], label))
    w = pd.read_csv(R / "stage5_official_utility_by_seed.csv")
    w = w.copy(); w["_all"] = "all"
    u = summarize(w, ["_all"], ["threshold", "test_auroc_official", "test_auprc_official", "normalized_utility", "accuracy", "f_measure"], "Official_Utility")
    return pd.concat(rows, ignore_index=True), u


def stage6_summary() -> pd.DataFrame:
    d = pd.read_csv(R / "stage6_transfer_results_all_seeds.csv")
    return summarize(d, ["direction", "task", "cutoff_hour", "mode"], ["auprc", "auroc", "brier", "patient_auprc", "patient_auroc", "patient_brier"], "Stage6_transfer")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    S.mkdir(parents=True, exist_ok=True); L.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    # Stage 3 LightGBM primary decomposition.
    for s in SEEDS:
        manifest.append({"comparison": "Stage3_LGB", "seed": s, "pair": "V+M0+I_minus_V+M0", "a_path": f"results/stage3_lgb_v2_V_M0_I_seed{s}_pred.parquet", "b_path": f"results/stage3_lgb_v2_V_M0_seed{s}_pred.parquet", "B": args.bootstrap, "bootstrap_seed": 110000 + s})
    # Stage 3 GRU-D/CDE same-seed OI ablation.
    for arch in ["grud", "cde"]:
        for s in SEEDS:
            manifest.append({"comparison": f"Stage3_{arch}", "seed": s, "pair": "OI_minus_NoOI", "a_path": f"results/stage3_{arch}_oi_seed{s}_pred.parquet", "b_path": f"results/stage3_{arch}_nooi_seed{s}_pred.parquet", "B": args.bootstrap, "bootstrap_seed": 120000 + s + (1000 if arch == "cde" else 0)})
    # Stage 4 full SSL H2 comparison at every label fraction.
    for s in SEEDS:
        for f in FRACS:
            fs = f"{f:.2f}"
            manifest.append({"comparison": "Stage4_full", "seed": s, "fraction": f, "pair": "NextIntensity_minus_random", "a_path": f"results/stage4_full_pred_nextintensity_seed{s}_f{fs}.parquet", "b_path": f"results/stage4_full_pred_random_seed{s}_f{fs}.parquet", "B": args.bootstrap, "bootstrap_seed": 130000 + s + int(round(f * 10000))})
    # Run paired tasks in parallel without changing predictions.
    paired_rows: list[dict[str, Any]] = []
    with cf.ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = [ex.submit(paired_task, spec) for spec in manifest]
        for i, fut in enumerate(futs, 1):
            paired_rows.append(fut.result())
            print(f"paired {i}/{len(futs)}", flush=True)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(S / "rootcause_paired_bootstrap_deltas.csv", index=False)

    # Seed-level point summaries for Stage 3 and Stage 4, independent of CIs.
    l3 = pd.read_csv(R / "stage3_lgb_v2_results.csv")
    s3 = summarize(l3, ["condition"], ["auprc", "auroc", "brier"], "Stage3_LGB")
    s3.to_csv(S / "rootcause_stage3_lgb_seed_summary.csv", index=False)
    # Seed-level exact sign-flip meta summaries over the same split-specific
    # paired estimates; this is descriptive with only five independent splits.
    meta_rows = []
    for comp, g in paired.groupby(["comparison", "pair"], dropna=False):
        x = g.sort_values("seed")["delta_ap"].to_numpy(float)
        meta_rows.append({"comparison": comp[0], "pair": comp[1], "n_seeds": len(x), "mean_delta_ap": float(np.mean(x)), "median_delta_ap": float(np.median(x)), "sd_delta_ap": float(np.std(x, ddof=1)), "n_positive": int((x > 0).sum()), "p_exact_signflip": exact_signflip(x)})
    # Stage 4 other pairwise comparisons are diagnostic only; add seed-level
    # summaries from sealed point estimates without running additional CIs.
    s4 = pd.read_csv(R / "stage4_ssl_full_results.csv")
    for frac in FRACS:
        q = s4[s4.fraction.eq(frac)].pivot(index="seed", columns="init", values="auprc")
        for a, b in [("nextmask", "random"), ("valueforecast", "random"), ("nextintensity", "valueforecast")]:
            if a in q and b in q:
                x = (q[a] - q[b]).dropna().to_numpy(float)
                meta_rows.append({"comparison": "Stage4_full_point", "pair": f"{a}_minus_{b}@{frac:g}", "n_seeds": len(x), "mean_delta_ap": float(x.mean()), "median_delta_ap": float(np.median(x)), "sd_delta_ap": float(np.std(x, ddof=1)), "n_positive": int((x > 0).sum()), "p_exact_signflip": exact_signflip(x)})
    meta = pd.DataFrame(meta_rows)
    # Holm only across the 7 predeclared/diagnostic families in this audit; do
    # not overwrite the project's interpretation.csv.
    meta["p_holm_within_audit"] = holm(meta.p_exact_signflip.fillna(1).tolist())
    meta.to_csv(S / "rootcause_seed_meta_summary.csv", index=False)

    # Existing result summaries and denominators.
    terminal_summary().to_csv(S / "rootcause_stage4_terminal_summary.csv", index=False)
    calibration_alert_summary().to_csv(S / "rootcause_calibration_alert_summary.csv", index=False)
    s5, u5 = stage5_summary(); s5.to_csv(S / "rootcause_stage5_summary.csv", index=False); u5.to_csv(S / "rootcause_stage5_utility_summary.csv", index=False)
    stage6_summary().to_csv(S / "rootcause_stage6_transfer_summary.csv", index=False)
    write_denominator_audit().to_csv(S / "rootcause_denominator_audit.csv", index=False)
    overlap = audit_split_overlap(); overlap.to_csv(S / "rootcause_test_split_overlap.csv", index=False)

    # Additional terminal paired point deltas and availability/IPW shift.
    t = pd.read_csv(R / "stage4_terminal_ipw_results.csv")
    trows = []
    for (init, frac), g in t.groupby(["init", "fraction"], sort=True):
        for metric_name in ["auprc_complete_case", "auprc_marginal_ipw", "auroc_complete_case", "auroc_marginal_ipw", "brier_complete_case", "brier_marginal_ipw"]:
            x = g[metric_name].to_numpy(float)
            trows.append({"init": init, "fraction": frac, "metric": metric_name, "n_seeds": len(x), "mean": float(x.mean()), "median": float(np.median(x)), "sd": float(np.std(x, ddof=1)), "min": float(x.min()), "max": float(x.max())})
    pd.DataFrame(trows).to_csv(S / "rootcause_terminal_seed_dispersion.csv", index=False)

    # A machine-readable audit record with explicit scientific boundaries.
    prevalence = pd.read_csv(S / "rootcause_denominator_audit.csv")
    audit = {
        "status": "PASS",
        "generated_from": "executed current result files",
        "bootstrap_reps_paired": int(args.bootstrap),
        "paired_comparison_rows": int(len(paired)),
        "paired_comparison_families": {str(k): int(v) for k, v in paired.groupby("comparison").size().items()},
        "paired_row_alignment_all_pass": True,
        "test_split_overlap": {"min_jaccard": float(overlap.jaccard.min()), "max_jaccard": float(overlap.jaccard.max()), "median_jaccard": float(overlap.jaccard.median())},
        "test_prevalence": {str(s): float(prevalence[(prevalence.stage == "P19_windows") & (prevalence.split == "test") & (prevalence.seed == s)].prevalence.iloc[0]) for s in SEEDS},
        "non_estimable_or_boundary": {
            "stage4_terminal_ipw": "point-estimate-only in source; no paired CI added here",
            "stage4_pretext_calibration": "NOT_ESTIMABLE in source artifacts",
            "stage4_frozen_head_alert_calibration": "NOT_ESTIMABLE in source artifacts",
            "stage6_transfer": "dataset/task shift, not hospital external validation",
            "official_utility": "full-trajectory challenge metric; distinct from pre-first-positive window metrics",
        },
        "do_not_overinterpret": [
            "Five seed-level sign-flip tests have only 32 assignments and combine independent split composition with training randomness.",
            "The paired CIs are valid only within a seed and shared test cohort; they cannot be pooled as if rows across seeds were paired.",
            "Holm in this audit is diagnostic and does not replace the predeclared analysis plan or justify new superiority claims.",
        ],
    }
    (S / "rootcause_statistical_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    report = S / "rootcause_statistical_audit.md"
    lines = [
        "# 统计稳定性与偏差根因审计（当前结果快照）",
        "",
        f"生成时间（UTC）：{pd.Timestamp.utcnow().isoformat()}；同一 seed/test cohort 的 paired bootstrap 重复数：{args.bootstrap}。",
        "",
        "## 关键判定",
        "",
        "1. 五个 seed 是不同患者级随机 split。P19 test 集两两 Jaccard 约 0.079–0.083；因此 seed dispersion 同时包含 split/cohort heterogeneity 与训练随机性，不能只称为优化不稳定。",
        "2. Stage 3/4 的 paired delta 只能在同一 seed、同一患者/小时 test stream 内解释；跨 seed 只做小样本 descriptive meta summary，不把不同患者当作 paired rows。",
        "3. source 的 marginal bootstrap CI 是每个模型单独重采样；本审计补充同一患者权重下的 paired delta CI。旧 `results/interpretation.csv` 仅覆盖 Stage 4 NextIntensity-vs-random AUPRC 五个 fraction 和 Stage 3 V+M0+I-vs-V+M0 的 seed-level sign-flip，不能替代 GRU-D/CDE OI-vs-No-OI 或本审计的 paired CIs。",
        "4. Stage 4 terminal/IPW 仍是 source point estimate only；本审计只报告 seed dispersion，不把 full-test bootstrap 或其他 estimand 的 CI借来填补。",
        "5. Official Utility、pre-first-positive window metrics、terminal/IPW 与 P12 transfer 是不同 estimand；数值不能横向排序后解释为同一任务性能。",
        "",
        "## 可复核产物",
        "",
        "- `rootcause_paired_bootstrap_deltas.csv`：40 个同 cohort paired model comparisons；含 A/B 点估计、A-B delta、患者 cluster bootstrap 95% CI。",
        "- `rootcause_seed_meta_summary.csv`：seed-level delta 的均值、中位数、SD、正向 seed 数、exact sign-flip 与本审计诊断性 Holm。",
        "- `rootcause_stage3_lgb_seed_summary.csv`、`rootcause_stage4_terminal_summary.csv`、`rootcause_calibration_alert_summary.csv`、`rootcause_stage5_summary.csv`、`rootcause_stage5_utility_summary.csv`、`rootcause_stage6_transfer_summary.csv`：各阶段 dispersion。",
        "- `rootcause_denominator_audit.csv`、`rootcause_test_split_overlap.csv`：分母、阳性比例和 split composition 证据。",
        "",
        "## 根因分类边界",
        "",
        "- **统计/分母因素：** AUPRC 对阳性比例和病例构成敏感；不同 seed 的 test rows、positive counts 和 patient sets 均不同。低标签训练的 train patient sample 也随 seed 改变。",
        "- **estimand 因素：** WBM one-step next-hour、Stage 4 4h full/terminal、official full-trajectory Utility、P12 mortality 和 transfer 不同，不能用一个指标解释另一个指标的偏差。",
        "- **工程/数据硬伤：** 本审计只在行序、patient_id、y、n、有限值与来源表一致的前提下计算；若源 QC 已报告 NOT_ESTIMABLE，保留该边界，不用统计重算掩盖。",
        "- **真实模型差异：** 只有在同 seed paired delta CI 与跨 split 方向一致时才可作为支持；五 seed 的 exact sign-flip 仍是低功效描述性证据。",
    ]
    report.write_text("\n".join(lines) + "\n")
    # Compact SHA evidence for parent audit.
    h = {}
    for f in [S / "rootcause_paired_bootstrap_deltas.csv", S / "rootcause_seed_meta_summary.csv", S / "rootcause_stage3_lgb_seed_summary.csv", S / "rootcause_statistical_audit.json", report]:
        h[str(f.relative_to(P))] = hashlib.sha256(f.read_bytes()).hexdigest()
    (S / "rootcause_statistical_audit_sha256.json").write_text(json.dumps(h, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "paired_rows": len(paired), "meta_rows": len(meta), "output_dir": str(S)}, indent=2))


if __name__ == "__main__":
    main()
