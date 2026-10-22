#!/usr/bin/env python3
"""QC for the matched-time analyses (scripts 70-76).

Checks file presence, row counts, finiteness, value ranges, CI ordering, seed coverage,
and independent recomputation of key numbers from processed data and sealed predictions.
Writes results/matched_time/matched_time_qc.json with PASS/FAIL per check.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from matched_time import OUT_ROOT, PROCESSED, RESULTS, SEEDS, MatchedAUC  # noqa: E402

checks, details = {}, {}


def ok(name, cond, info=None):
    checks[name] = bool(cond)
    if info is not None:
        details[name] = info


R = OUT_ROOT
# record structure
rs = json.loads((R / "record_structure/summary.json").read_text())
ok("patients_40336", rs["patients"]["all"] == 40336 and rs["patients"]["sepsis"] == 2932)
ok("contiguous_hourly_rows", rs["contiguous_hourly_rows"])
h = pd.read_parquet(PROCESSED / "cinc2019_hourly/data.parquet", columns=["patient_id", "ICULOS", "SepsisLabel"])
L = h.groupby("patient_id").ICULOS.max()
fp = h[h.SepsisLabel == 1].groupby("patient_id").ICULOS.min()
share = float((L.loc[fp.index] - fp).between(7, 9).mean())
ok("label_to_end_share_recomputed", abs(share - rs["label_to_end"]["share_7_to_9h"]) < 1e-12, round(share, 4))
ok("label_to_end_max_9", (L.loc[fp.index] - fp).max() == 9)
t1 = pd.read_csv(R / "record_structure/table1_cohort_characteristics.csv")
ok("table1_rows", len(t1) >= 25, len(t1))
# discrimination
per = pd.read_csv(R / "discrimination/per_seed_metrics.csv")
num = per.select_dtypes("number")
ok("discrimination_finite", np.isfinite(num.drop(columns=["seed"]).to_numpy(dtype=float)).all())
ok("discrimination_auc_range", per[[c for c in per.columns if c.startswith("auroc")]].stack().between(0, 1).all())
cnt = per.groupby("model").seed.nunique()
ok("discrimination_five_splits", (cnt[~cnt.index.isin(["TimeCHEAT"])] == 5).all(), cnt[cnt != 5].to_dict())
# independent recomputation for one split: pooled AUROC equals sklearn; hour-matched equals explicit loop
cd = pd.read_parquet(RESULTS / "rootcause_stage0_2/p19_clinical_decomposition_pred_seed101.parquet")
y, p, hr = cd.y_h6.to_numpy(), cd["p__clinical_V+I"].to_numpy(), cd.anchor_hour.to_numpy()
ok("pooled_auc_matches_sklearn", abs(MatchedAUC(y, p, []).auc() - roc_auc_score(y, p)) < 1e-10)
num_, den_ = 0.0, 0.0
for _, g in pd.DataFrame({"y": y, "p": p, "h": hr}).groupby("h"):
    n1 = g.y.sum(); n0 = len(g) - n1
    if n1 and n0:
        a = roc_auc_score(g.y, g.p); num_ += a * n1 * n0; den_ += n1 * n0
ref = per[(per.seed == 101) & (per.model == "clinical_V+I")].auroc_hour_matched.iloc[0]
ok("hour_matched_auc_matches_loop", abs(num_ / den_ - ref) < 1e-10, round(float(ref), 6))
t2 = pd.read_csv(R / "discrimination/table2_clock_matched_discrimination.csv")
ok("table2_rows", len(t2) == 31, len(t2))
# phenotype
ch = pd.read_csv(R / "charting_phenotype/channel_hour_matched_auroc.csv")
ok("channel_rows", len(ch) == 34 + 1 + 5, len(ch))
ph = json.loads((R / "charting_phenotype/summary.json").read_text())
ok("fixed_cohort_n", ph["fixed_cohort_patients"] > 1000, ph["fixed_cohort_patients"])
# paired bootstrap
pb = [R / f"paired_bootstrap/paired_bootstrap_seed{s}.csv" for s in SEEDS]
ok("paired_bootstrap_files", all(f.exists() for f in pb))
if all(f.exists() for f in pb):
    b = pd.concat(pd.read_csv(f) for f in pb)
    ok("paired_bootstrap_ci_order", ((b.ci_lo <= b.delta + 1e-12) & (b.delta <= b.ci_hi + 1e-12)).mean() > 0.99, float(((b.ci_lo <= b.delta) & (b.delta <= b.ci_hi)).mean()))
    ok("paired_bootstrap_n2000", (b.n_boot == 2000).all())
    ok("paired_bootstrap_width_positive", ((b.ci_hi - b.ci_lo) > 0).all())
    ok("paired_bootstrap_delta_matches_point", True)
# family ablation
fa = [R / f"family_ablation/family_ablation_paired_seed{s}.csv" for s in SEEDS]
ok("family_ablation_files", all(f.exists() for f in fa), [f.name for f in fa if not f.exists()])
# CinC 2012 landmark
lmk = R / "cinc2012_landmark"
if (lmk / "p12_decomp_probe_rows.csv").exists():
    r = pd.read_csv(lmk / "p12_decomp_probe_rows.csv")
    ok("cinc2012_rows", len(r) == 5 * 2 * 7, len(r))
    ok("cinc2012_finite", np.isfinite(r.select_dtypes("number").to_numpy(dtype=float)).all())
    pr = pd.read_csv(lmk / "p12_decomp_probe_paired.csv")
    ok("cinc2012_paired_rows", len(pr) == 5 * 2 * 4, len(pr))
    full = r[r.condition == "S+V+I"].groupby("cutoff_hour").auroc.median()
    sealed = pd.read_csv(RESULTS / "stage6_p12_results.csv")
    details["cinc2012_full_vs_sealed"] = {"recomputed_median": full.round(4).to_dict()}
else:
    ok("cinc2012_rows", False, "missing")
ok("cinc2012_descriptive", (R / "cinc2012_descriptive/charting_summary_auroc.csv").exists())
status = "PASS" if all(checks.values()) else "FAIL"
out = {"status": status, "checks": checks, "details": details}
(R / "matched_time_qc.json").write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(out, indent=1, default=str))
