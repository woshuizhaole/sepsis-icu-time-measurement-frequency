#!/usr/bin/env python3
"""Adversarial diagnostics for the current P19 measurement-intensity results.

This script is deliberately read-only with respect to the frozen analysis
tables: it writes only new ``improvement_adversarial_*`` artifacts.  It uses
the current patient-level partitions and the already-fitted Stage-3 models to
test alternative explanations (time/cohort composition, label association,
patient-specific trajectories, and channel dependence).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

P = Path("/data/workplace/wyx/measurement_intensity_ssl_20261015")
D = P / "data" / "processed"
R = P / "results"

DYN = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
BASE = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
V_COLS = [f"v__{c}" for c in DYN]
M_COLS = [f"m0__{c}" for c in DYN]
I_CH_COLS = [f"i_count4__{c}" for c in DYN]
I_SUM_COLS = ["i_breadth4", "i_repeat4", "i_valid_hours4"]
I_COLS = I_CH_COLS + I_SUM_COLS
VMI_COLS = BASE + V_COLS + M_COLS + I_COLS

VITALS = {"HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "FiO2", "SaO2"}
LABS = set(DYN) - VITALS
SEEDS = [101, 202, 303, 404, 505]


def safe_metric(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    keep = np.isfinite(p)
    y, p = y[keep], p[keep]
    out = {"n": int(len(y)), "positive": int(y.sum()), "prevalence": float(y.mean()) if len(y) else np.nan}
    out["auprc"] = float(average_precision_score(y, p)) if y.sum() and len(np.unique(y)) > 1 else np.nan
    out["auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan
    out["brier"] = float(brier_score_loss(y, p)) if len(y) else np.nan
    return out


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    z = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            z += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(z)


def calib_summary(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    out = {"ece10": ece(y, p)}
    # Calibration slope/intercept are descriptive and use no test fitting.
    pp = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    x = np.log(pp / (1 - pp)).reshape(-1, 1)
    yy = np.asarray(y, dtype=int)
    try:
        m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(x, yy)
        out.update({"calibration_intercept": float(m.intercept_[0]), "calibration_slope": float(m.coef_[0, 0])})
    except Exception:
        out.update({"calibration_intercept": np.nan, "calibration_slope": np.nan})
    return out


def load_frame(seed: int) -> tuple[pd.DataFrame, dict[str, str]]:
    # Read only columns needed by this diagnostic, keeping memory and I/O low.
    w = pd.read_parquet(D / "cinc2019_windows.parquet", columns=[
        "patient_id", "anchor_hour", "next_window_available", "y_h6", "y_h12",
    ])
    f = pd.read_parquet(D / "stage3_p19_features_v2.parquet", columns=["patient_id", "ICULOS"] + VMI_COLS[0:5] + V_COLS + M_COLS + I_COLS)
    f = f.rename(columns={"ICULOS": "anchor_hour"})
    d = w.merge(f, on=["patient_id", "anchor_hour"], how="left", validate="one_to_one")
    d = d.reset_index(drop=True)
    d["ICULOS"] = d["anchor_hour"]
    part = json.loads((D / "partitions.json").read_text())["cinc2019"][str(seed)]
    d["split"] = d.patient_id.map(part)
    if d["split"].isna().any():
        raise RuntimeError("unassigned patients")
    lens = d.groupby("patient_id", sort=False).size()
    d["trajectory_length"] = d.patient_id.map(lens).astype(int)
    bins = [-np.inf, 6, 12, 24, 48, 72, np.inf]
    names = ["1-6", "7-12", "13-24", "25-48", "49-72", ">72"]
    d["anchor_bin"] = pd.cut(d["anchor_hour"], bins=bins, labels=names, right=True).astype(str)
    d["length_bin"] = pd.qcut(d["trajectory_length"], q=5, duplicates="drop").astype(str)
    return d, part


def prevalence_baselines(d: pd.DataFrame, seed: int) -> list[dict]:
    tr, te = d[d.split == "train"], d[d.split == "test"]
    rows = []
    global_p = float(tr.y_h6.mean())
    for name, p in [("constant_train_prevalence", np.full(len(te), global_p))]:
        o = safe_metric(te.y_h6.to_numpy(), p); o.update({"seed": seed, "diagnostic": "baseline", "baseline": name}); rows.append(o)
    # Causal anchor-hour prior estimated from train only. Unseen hours fall back
    # to the global prevalence.
    byh = tr.groupby("anchor_hour").y_h6.mean()
    p = te.anchor_hour.map(byh).fillna(global_p).to_numpy(float)
    o = safe_metric(te.y_h6.to_numpy(), p); o.update({"seed": seed, "diagnostic": "baseline", "baseline": "train_anchor_hour_prevalence"}); rows.append(o)
    # Cohort-composition-only prior: trajectory length bin and anchor hour.
    tab = tr.groupby(["anchor_bin", "length_bin"], observed=True).y_h6.mean()
    p = [tab.get((a, l), global_p) for a, l in zip(te.anchor_bin, te.length_bin)]
    o = safe_metric(te.y_h6.to_numpy(), np.asarray(p, float)); o.update({"seed": seed, "diagnostic": "baseline", "baseline": "train_anchor_length_prevalence"}); rows.append(o)
    return rows


def strata_from_predictions(seed: int, d: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    te = d[d.split == "test"][["patient_id", "anchor_hour", "y_h6", "y_h12", "next_window_available", "anchor_bin", "length_bin"]].copy()
    for cond in ["V", "V_M0", "V_M0_I", "I"]:
        f = R / f"stage3_lgb_v2_{cond}_seed{seed}_pred.parquet"
        if not f.exists():
            continue
        pr = pd.read_parquet(f)
        z = te.merge(pr[["patient_id", "anchor_hour", "p"]], on=["patient_id", "anchor_hour"], how="inner", validate="one_to_one")
        for outcome, ycol in [("h6", "y_h6"), ("h12", "y_h12")]:
            for stratum, vals in [("all", np.ones(len(z), bool)),
                                  ("next_available=0", z.next_window_available.to_numpy() == 0),
                                  ("next_available=1", z.next_window_available.to_numpy() == 1)]:
                m = vals
                o = safe_metric(z.loc[m, ycol].to_numpy(), z.loc[m, "p"].to_numpy()); o.update({"seed": seed, "diagnostic": "stratum", "condition": cond, "outcome": outcome, "stratum": stratum}); rows.append(o)
            for key, g in z.groupby("anchor_bin", observed=True):
                o = safe_metric(g[ycol].to_numpy(), g.p.to_numpy()); o.update({"seed": seed, "diagnostic": "stratum", "condition": cond, "outcome": outcome, "stratum": f"anchor_bin={key}"}); rows.append(o)
            for key, g in z.groupby("length_bin", observed=True):
                o = safe_metric(g[ycol].to_numpy(), g.p.to_numpy()); o.update({"seed": seed, "diagnostic": "stratum", "condition": cond, "outcome": outcome, "stratum": f"length_bin={key}"}); rows.append(o)
        # Use the held-out predictions only; all calibration is descriptive.
        o = safe_metric(z.y_h6.to_numpy(), z.p.to_numpy()); o.update(calib_summary(z.y_h6.to_numpy(), z.p.to_numpy())); o.update({"seed": seed, "diagnostic": "calibration", "condition": cond, "outcome": "h6", "stratum": "all"}); rows.append(o)
    return rows


def rotate_patient_values(d: pd.DataFrame, cols: list[str], seed: int) -> pd.DataFrame:
    """Permute complete trajectories within exact length and split buckets.

    Every patient remains in the same split and every row remains in the test
    set; only the selected intensity channels are reassigned. This fixes the
    singleton exclusion in the old negative-control script.
    """
    out = d.copy()
    rng = np.random.default_rng(20261017 + seed)
    # Work on one compact NumPy block per exact-length bucket.  The former
    # implementation performed one pandas ``.loc`` assignment per patient and
    # became needlessly slow on ~40k trajectories.
    for (split, n), g in d.groupby(["split", "trajectory_length"], sort=False):
        pids = np.array(sorted(g.patient_id.unique()), dtype=object)
        if len(pids) < 2:
            continue
        source = pids[rng.permutation(len(pids))]
        mapping = dict(zip(pids.tolist(), source.tolist()))
        gs = g.sort_values(["patient_id", "anchor_hour"])
        target_pid = gs.patient_id.to_numpy(dtype=object)
        target_pos = gs.groupby("patient_id", sort=False).cumcount().to_numpy(dtype=np.int64)
        starts = gs.groupby("patient_id", sort=False).indices
        start = {pid: int(idx[0]) for pid, idx in starts.items()}
        source_idx = np.asarray([start[mapping[pid]] + pos for pid, pos in zip(target_pid, target_pos)], dtype=np.int64)
        # ``gs.index`` and ``source_idx`` are positional labels in the original
        # frame because the merge frame retains a RangeIndex.
        out.loc[gs.index, cols] = d.iloc[source_idx][cols].to_numpy(copy=False)
    return out


def fit_permuted(seed: int, d: pd.DataFrame, y_train: pd.Series, tag: str, cols: list[str]) -> dict:
    tr, va, te = d[d.split == "train"], d[d.split == "validation"], d[d.split == "test"]
    med = tr[cols].median(numeric_only=True)
    clean = lambda x: x[cols].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0)
    clf = lgb.LGBMClassifier(n_estimators=220, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, n_jobs=1, verbosity=-1)
    clf.fit(clean(tr), y_train.loc[tr.index].astype(int), eval_set=[(clean(va), y_train.loc[va.index].astype(int))], callbacks=[lgb.early_stopping(30, verbose=False)])
    p = clf.predict_proba(clean(te))[:, 1]
    o = safe_metric(te.y_h6.to_numpy(), p); o.update({"seed": seed, "diagnostic": tag, "condition": "V_M0_I", "outcome": "h6", "stratum": "all", "best_iteration": int(clf.best_iteration_ or 0), "n_features": len(cols)})
    o.update(calib_summary(te.y_h6.to_numpy(), p))
    return o


def shuffled_labels_by_hour(d: pd.DataFrame, seed: int) -> pd.Series:
    out = d.y_h6.copy()
    rng = np.random.default_rng(20261018 + seed)
    for split in ["train", "validation"]:
        for _, g in d[d.split == split].groupby("anchor_hour", sort=False):
            out.loc[g.index] = rng.permutation(g.y_h6.to_numpy())
    return out


def shuffled_patient_labels(d: pd.DataFrame, seed: int) -> tuple[pd.Series, int]:
    out = d.y_h6.copy(); rng = np.random.default_rng(20261019 + seed); changed = 0
    for split in ["train", "validation"]:
        g0 = d[d.split == split]
        for _, g in g0.groupby("trajectory_length", sort=False):
            pids = np.array(sorted(g.patient_id.unique()), dtype=object)
            if len(pids) < 2:
                continue
            source = pids[rng.permutation(len(pids))]
            mapping = dict(zip(pids.tolist(), source.tolist()))
            gs = g.sort_values(["patient_id", "anchor_hour"])
            target_pid = gs.patient_id.to_numpy(dtype=object)
            target_pos = gs.groupby("patient_id", sort=False).cumcount().to_numpy(dtype=np.int64)
            starts = gs.groupby("patient_id", sort=False).indices
            start = {pid: int(idx[0]) for pid, idx in starts.items()}
            source_idx = np.asarray([start[mapping[pid]] + pos for pid, pos in zip(target_pid, target_pos)], dtype=np.int64)
            out.loc[gs.index] = d.iloc[source_idx].y_h6.to_numpy()
            changed += int(np.sum(pids != source))
    return out, changed


def fixed_model_occlusion(seed: int, d: pd.DataFrame) -> list[dict]:
    """Perturb I channels at inference in the already-fitted Stage-3 model."""
    model_path = R / f"stage3_lgb_v2_V_M0_I_seed{seed}.txt"
    if not model_path.exists(): return []
    tr, te = d[d.split == "train"], d[d.split == "test"]
    med = tr[VMI_COLS].median(numeric_only=True)
    clean = lambda x: x[VMI_COLS].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0)
    base = clean(te)
    booster = lgb.Booster(model_file=str(model_path))
    rows=[]
    groups = {
        "none": [],
        "drop_vitals": [f"i_count4__{c}" for c in sorted(VITALS)],
        "drop_labs": [f"i_count4__{c}" for c in sorted(LABS)],
        "drop_lactate_wbc": ["i_count4__Lactate", "i_count4__WBC"],
        "drop_all_I": I_COLS,
    }
    for tag, drop in groups.items():
        x = base.copy()
        for c in drop:
            x[c] = med[c]
        p = booster.predict(x)
        o = safe_metric(te.y_h6.to_numpy(), p); o.update(calib_summary(te.y_h6.to_numpy(), p)); o.update({"seed": seed, "diagnostic": "fixed_model_channel_occlusion", "condition": "V_M0_I", "outcome": "h6", "stratum": tag}); rows.append(o)
    return rows


def main():
    baseline_rows=[]; strata_rows=[]; null_rows=[]; occ_rows=[]; perm_rows=[]
    data_only = os.environ.get("ADV_DATA_ONLY", "0") == "1"
    for seed in SEEDS:
        d, _ = load_frame(seed)
        baseline_rows.extend(prevalence_baselines(d, seed))
        strata_rows.extend(strata_from_predictions(seed, d))
        occ_rows.extend(fixed_model_occlusion(seed, d))
        if data_only:
            continue
        # Full exact-length patient trajectory permutation; keep all patients.
        dp = rotate_patient_values(d, I_COLS, seed)
        perm_rows.append(fit_permuted(seed, dp, dp.y_h6, "patient_trajectory_I_permutation", VMI_COLS))
        # Two null label permutations.  The hourly version preserves the
        # target prevalence by anchor hour; the patient version preserves
        # within-patient label shape and trajectory length.
        if seed in [101, 303, 505]:
            ys = shuffled_labels_by_hour(d, seed)
            null_rows.append(fit_permuted(seed, d, ys, "label_shuffle_within_anchor_hour", VMI_COLS))
            yp, changed = shuffled_patient_labels(d, seed)
            o = fit_permuted(seed, d, yp, "label_shuffle_patient_trajectory", VMI_COLS); o["changed_patients"] = int(changed); null_rows.append(o)
    pd.DataFrame(baseline_rows).to_csv(R / "improvement_adversarial_baselines.csv", index=False)
    pd.DataFrame(strata_rows).to_csv(R / "improvement_adversarial_strata.csv", index=False)
    pd.DataFrame(perm_rows).to_csv(R / "improvement_adversarial_patient_permutation.csv", index=False)
    pd.DataFrame(null_rows).to_csv(R / "improvement_adversarial_label_nulls.csv", index=False)
    pd.DataFrame(occ_rows).to_csv(R / "improvement_adversarial_channel_occlusion.csv", index=False)
    summary = {
        "seeds": SEEDS,
        "outputs": [
            "improvement_adversarial_baselines.csv",
            "improvement_adversarial_strata.csv",
            "improvement_adversarial_patient_permutation.csv",
            "improvement_adversarial_label_nulls.csv",
            "improvement_adversarial_channel_occlusion.csv",
        ],
        "nulls": "within-anchor-hour and exact-trajectory patient label permutations; no test labels permuted",
        "trajectory_permutation": "all intensity columns rotated within split and exact patient-trajectory-length buckets; singleton buckets retained unchanged",
        "channel_occlusion": "fixed trained V+M0+I booster, test-time intensity group set to train medians; diagnostic, not retrained performance",
        "data_only": data_only,
    }
    (R / "improvement_adversarial_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
