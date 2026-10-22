#!/usr/bin/env python3
"""Measurement-family charting ablation on top of physiological values (Fig. 4B).

LightGBM with the same configuration as the clinical decomposition
(n_estimators=300, lr=0.05, num_leaves=31, min_child_samples=100, subsample=0.8,
colsample_bytree=0.8, reg_lambda=1, early stopping 30 on validation), train-only
median imputation. Conditions add 4-h charting counts of one family (or all) to the
34 physiological values. Paired hour-matched AUROC deltas use 2,000 shared
patient-cluster draws (seed*7919+31).
Usage: 74_matched_time_family_ablation.py --seed 101 [--threads 8]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from matched_time import DYN, FAMILIES, OUT_ROOT, PROCESSED, MatchedAUC, partitions, patient_draws  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--n-boot", type=int, default=2000)
args = ap.parse_args()
s = args.seed
OUT = OUT_ROOT / "family_ablation"
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()

V = [f"v__{c}" for c in DYN]
C = lambda cs: [f"i_count4__{c}" for c in cs]  # noqa: E731
FAM = {"vital": FAMILIES["Vital signs"], "bloodgas": FAMILIES["Blood gas and respiratory support"],
       "chemistry": FAMILIES["Chemistry"], "haematology": FAMILIES["Haematology and coagulation"]}
COND = [("values", V), ("values+vital", V + C(FAM["vital"])), ("values+bloodgas", V + C(FAM["bloodgas"])),
        ("values+chemistry", V + C(FAM["chemistry"])), ("values+haematology", V + C(FAM["haematology"])),
        ("values+vital+bloodgas", V + C(FAM["vital"] + FAM["bloodgas"])),
        ("values+all_but_bloodgas", V + C(FAM["vital"] + FAM["chemistry"] + FAM["haematology"])),
        ("values+all_counts", V + C(DYN))]

win = pd.read_parquet(PROCESSED / "cinc2019_windows.parquet", columns=["patient_id", "anchor_hour", "y_h6"])
feat = pd.read_parquet(PROCESSED / "stage3_p19_features_v2.parquet", columns=["patient_id", "ICULOS"] + V + C(DYN)).rename(columns={"ICULOS": "anchor_hour"})
full = win.merge(feat, on=["patient_id", "anchor_hour"], how="left", validate="one_to_one")
sp = full.patient_id.map(partitions("cinc2019")[str(s)])
tr, va = full[sp == "train"], full[sp == "validation"]
te = full[sp == "test"].sort_values(["patient_id", "anchor_hour"]).reset_index(drop=True)
y = te.y_h6.to_numpy().astype(int)
hr = te.anchor_hour.to_numpy()
_, inv, draws = patient_draws(te.patient_id.to_numpy(), s, args.n_boot)
n_pat = int(inv.max()) + 1

rows, objs = [], {}
for name, cols in COND:
    med = tr[cols].median()
    clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.8,
                         colsample_bytree=0.8, reg_lambda=1.0, random_state=s, n_jobs=args.threads, verbosity=-1)
    clf.fit(tr[cols].fillna(med), tr.y_h6.astype(int), eval_set=[(va[cols].fillna(med), va.y_h6.astype(int))],
            callbacks=[early_stopping(30, verbose=False)])
    p = clf.predict_proba(te[cols].fillna(med))[:, 1]
    objs[name] = MatchedAUC(y, p, [hr])
    rows.append(dict(seed=s, condition=name, n_features=len(cols), best_iteration=int(clf.best_iteration_ or 0),
                     auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                     auroc_hour_matched=objs[name].auc()))
    print(f"seed {s} {name} AUROC {rows[-1]['auroc']:.4f} hour-matched {rows[-1]['auroc_hour_matched']:.4f} {time.time() - t0:.0f}s", flush=True)
pd.DataFrame(rows).to_csv(OUT / f"family_ablation_seed{s}.csv", index=False)

boot = {k: np.empty(args.n_boot) for k in objs}
for i in range(args.n_boot):
    w = np.bincount(draws[i], minlength=n_pat)[inv].astype(float)
    for k, o in objs.items():
        boot[k][i] = o.auc(w)
paired = []
for name, _ in COND[1:]:
    d = boot[name] - boot["values"]
    paired.append(dict(seed=s, contrast=f"{name} - values", delta_auroc_hour_matched=objs[name].auc() - objs["values"].auc(),
                       ci_lo=float(np.percentile(d, 2.5)), ci_hi=float(np.percentile(d, 97.5)), n_boot=args.n_boot))
pd.DataFrame(paired).to_csv(OUT / f"family_ablation_paired_seed{s}.csv", index=False)
print(f"seed {s} done {time.time() - t0:.0f}s", flush=True)
