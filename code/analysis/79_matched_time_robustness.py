#!/usr/bin/env python3
"""Robustness of matched-time evaluation to the matching granularity and the pooling weights (Suppl. Fig. 2).

No model is trained. Every prediction file is a sealed full-test output, aligned to the clinical-decomposition
prediction order exactly as in 71_matched_time_discrimination.py (alignment is asserted).

For one split (--seed) the script writes, under results/matched_time/robustness/:
  binned_metrics_seed{s}.csv      model x granularity: hour-matched AUROC and clock dependence
                                  (granularity: exact ICU hour, 4-h bins capped at >96 h, the six reporting strata)
  bootstrap_summary_seed{s}.csv   paired increments (charting intensity added to values; context fields added to
                                  values + indicators + intensity) under five matching schemes with 2,000-draw CIs
  bootstrap_draws_seed{s}.csv     the 2,000 paired bootstrap deltas behind those CIs
  robustness_qc_seed{s}.json      agreement of the exact-hour numbers with the sealed per-split metrics and the
                                  sealed paired bootstrap (same draws: seed * 7919 + 31)
Hours remaining in the record are not used here.
Usage: 79_matched_time_robustness.py --seed 101 [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from matched_time import (OUT_ROOT, PROCESSED, RESULTS, MatchedAUC, clock_share, partitions,  # noqa: E402
                          patient_draws, time_bin)

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--n-boot", type=int, default=2000)
args = ap.parse_args()
s = args.seed
OUT = OUT_ROOT / "robustness"
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()

part = partitions("cinc2019")
win = pd.read_parquet(PROCESSED / "cinc2019_windows.parquet", columns=["patient_id", "anchor_hour", "y_h6"])
cd = pd.read_parquet(RESULTS / f"rootcause_stage0_2/p19_clinical_decomposition_pred_seed{s}.parquet")
te = win[win.patient_id.map(part[str(s)]) == "test"].sort_values(["patient_id", "anchor_hour"]).reset_index(drop=True)
assert len(te) == len(cd) and (te.patient_id.to_numpy() == cd.patient_id.to_numpy()).all() and (te.anchor_hour.to_numpy() == cd.anchor_hour.to_numpy()).all()
y = cd.y_h6.to_numpy().astype(int)
assert (y == te.y_h6.to_numpy()).all()
hr = cd.anchor_hour.to_numpy()
b4 = np.minimum((hr - 1) // 4, 24)            # 4-h bins; everything beyond 96 h forms the last bin
b6 = time_bin(hr)                             # the six reporting strata
GRAN = {"exact_hour": hr, "bin_4h": b4, "strata_6": b6}

# ---------------------------------------------------------------- predictions (same sources as script 71)
preds = {}
tr = win[win.patient_id.map(part[str(s)]) == "train"]
preds["ICU-hour prevalence lookup"] = te.anchor_hour.map(tr.groupby("anchor_hour").y_h6.mean()).fillna(tr.y_h6.mean()).to_numpy()
for c in ["admin_time_only", "anchor_hour_only", "clinical_V", "clinical_V+M0", "clinical_V+I", "clinical_V+M0+I",
          "clinical_V+I_equalized_noise", "full_V+M0", "full_V+I", "full_V+M0+I"]:
    preds[c] = cd["p__" + c].to_numpy()


def load(path):
    x = pd.read_parquet(path)
    assert (x.patient_id.to_numpy() == cd.patient_id.to_numpy()).all() and (x.y.to_numpy().astype(int) == y).all(), path
    return x.p.to_numpy()


for c in ["V", "M0", "Mh", "I", "V_I", "V_M0", "V_M0_I", "V_Mh", "V_Mh_I"]:
    preds["context+" + c] = load(RESULTS / f"stage3_lgb_v2_{c}_seed{s}_pred.parquet")
for name, pat in [("GRU-D", "stage3_grud_nooi_seed{s}_pred.parquet"), ("GRU-D + intensity", "stage3_grud_oi_seed{s}_pred.parquet"),
                  ("Neural CDE", "stage3_cde_nooi_seed{s}_pred.parquet"), ("Neural CDE + intensity", "stage3_cde_oi_seed{s}_pred.parquet"),
                  ("GRU + next-measurement auxiliary", "stage5_wbm_pred_seed{s}.parquet"), ("TimeCHEAT", "stage5_timecheat_pred_seed{s}.parquet")]:
    f = RESULTS / pat.format(s=s)
    if f.exists():
        preds[name] = load(f)
for tag, root in (("pretrain", "improvement_fair_ssl"), ("pretrain_with_continuation", "improvement_fair_ssl/terminal_controls_all")):
    for pth in sorted(glob.glob(str(RESULTS / root / f"fair_pred_*_seed{s}_f*.parquet"))):
        m = re.search(r"fair_pred_(.+)_seed\d+_f([\d.]+)\.parquet", pth)
        preds[f"{tag}:{m.group(1)}|{m.group(2)}"] = load(pth)
print(f"seed {s}: {len(preds)} models, {len(y)} rows, {time.time() - t0:.0f}s", flush=True)

# ---------------------------------------------------------------- point estimates at three granularities
rows = []
for k, p in preds.items():
    base = dict(seed=s, model=k, auroc_pooled=float(roc_auc_score(y, p)))
    for g, key in GRAN.items():
        rows.append(dict(base, granularity=g, n_strata=int(len(np.unique(key))), auroc_matched=MatchedAUC(y, p, [key]).auc(),
                         clock_dependence=clock_share(p, key)))
pm = pd.DataFrame(rows)
pm.to_csv(OUT / f"binned_metrics_seed{s}.csv", index=False)

# ---------------------------------------------------------------- paired bootstrap under five matching schemes
CONTRASTS = [("charting_intensity_on_values", "clinical_V+I", "clinical_V"),
             ("context_on_values_indicators_intensity", "full_V+M0+I", "clinical_V+M0+I")]
SCHEMES = [("exact_hour", "pairs", "exact_hour"), ("exact_hour", "cases", "exact_hour"), ("exact_hour", "equal", "exact_hour"),
           ("bin_4h", "pairs", "bin_4h"), ("strata_6", "pairs", "strata_6")]
SCHEME_NAME = {("exact_hour", "pairs"): "exact hour, pair-weighted", ("exact_hour", "cases"): "exact hour, case-weighted",
               ("exact_hour", "equal"): "exact hour, equal-weighted", ("bin_4h", "pairs"): "4-h bins, pair-weighted",
               ("strata_6", "pairs"): "six strata, pair-weighted"}
_, inv, draws = patient_draws(cd.patient_id.to_numpy(), s, args.n_boot)
n_pat = int(inv.max()) + 1
need = sorted({m for _, a, b in CONTRASTS for m in (a, b)})
objs = {(m, g): MatchedAUC(y, preds[m], [key]) for m in need for g, key in GRAN.items()}


def val(m, g, wt, w=None):
    return objs[(m, g)].auc(w, weighting=wt)


point, boot = {}, {}
for cname, a, b in CONTRASTS:
    for g, wt, _ in SCHEMES:
        point[(cname, g, wt)] = val(a, g, wt) - val(b, g, wt)
        boot[(cname, g, wt)] = np.empty(args.n_boot)
for i in range(args.n_boot):
    w = np.bincount(draws[i], minlength=n_pat)[inv].astype(float)
    for cname, a, b in CONTRASTS:
        for g, wt, _ in SCHEMES:
            boot[(cname, g, wt)][i] = val(a, g, wt, w) - val(b, g, wt, w)
    if i % 400 == 0:
        print(f"seed {s} draw {i} {time.time() - t0:.0f}s", flush=True)
summ, drow = [], []
for (cname, g, wt), d in boot.items():
    summ.append(dict(seed=s, contrast=cname, matching=g, weighting=wt, scheme=SCHEME_NAME[(g, wt)], delta=point[(cname, g, wt)],
                     ci_lo=float(np.percentile(d, 2.5)), ci_hi=float(np.percentile(d, 97.5)), n_boot=args.n_boot))
    drow.append(pd.DataFrame(dict(seed=s, contrast=cname, matching=g, weighting=wt, draw=np.arange(args.n_boot), delta=d)))
pd.DataFrame(summ).to_csv(OUT / f"bootstrap_summary_seed{s}.csv", index=False)
pd.concat(drow).to_csv(OUT / f"bootstrap_draws_seed{s}.csv", index=False)

# ---------------------------------------------------------------- QC against the sealed exact-hour numbers
sealed = pd.read_csv(OUT_ROOT / "discrimination/per_seed_metrics.csv")
sealed = sealed[sealed.seed == s].set_index("model")
ex = pm[pm.granularity == "exact_hour"].set_index("model")
common = [m for m in ex.index if m in sealed.index]
d_auc = float(np.abs(ex.loc[common, "auroc_matched"] - sealed.loc[common, "auroc_hour_matched"]).max())
d_clk = float(np.abs(ex.loc[common, "clock_dependence"] - sealed.loc[common, "clock_dependence"]).max())
d_pool = float(np.abs(ex.loc[common, "auroc_pooled"] - sealed.loc[common, "auroc"]).max())
pbs = pd.read_csv(OUT_ROOT / f"paired_bootstrap/paired_bootstrap_seed{s}.csv")
ref = pbs[(pbs.contrast == "clin_VI - clin_V") & (pbs.metric == "hour")].iloc[0]
mine = pd.DataFrame(summ)
mine = mine[(mine.contrast == "charting_intensity_on_values") & (mine.scheme == SCHEME_NAME[("exact_hour", "pairs")])].iloc[0]
d_bs = max(abs(mine.delta - ref.delta), abs(mine.ci_lo - ref.ci_lo), abs(mine.ci_hi - ref.ci_hi))
case_ref = sealed.loc["clinical_V+I", "auroc_hour_matched_case_weighted"] - sealed.loc["clinical_V", "auroc_hour_matched_case_weighted"]
eq_ref = sealed.loc["clinical_V+I", "auroc_hour_matched_equal_weighted"] - sealed.loc["clinical_V", "auroc_hour_matched_equal_weighted"]
d_w = max(abs(point[("charting_intensity_on_values", "exact_hour", "cases")] - case_ref), abs(point[("charting_intensity_on_values", "exact_hour", "equal")] - eq_ref))
qc = dict(seed=s, n_models=len(preds), n_rows=int(len(y)), n_boot=args.n_boot, models_compared=len(common),
          max_abs_diff_matched_auroc=d_auc, max_abs_diff_clock_dependence=d_clk, max_abs_diff_pooled_auroc=d_pool,
          max_abs_diff_paired_bootstrap_vs_sealed=float(d_bs), max_abs_diff_weighted_increments=float(d_w), runtime_s=round(time.time() - t0, 1))
qc["status"] = "PASS" if max(d_auc, d_clk, d_pool, d_bs, d_w) < 1e-8 else "FAILED"
(OUT / f"robustness_qc_seed{s}.json").write_text(json.dumps(qc, indent=2))
print(json.dumps(qc, indent=2), flush=True)
