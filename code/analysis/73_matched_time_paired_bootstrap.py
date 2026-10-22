#!/usr/bin/env python3
"""Paired patient-cluster bootstrap (2,000 draws) for matched-time contrasts (Fig. 3, 6; Suppl. Table 3).

All contrasts are computed on sealed full-test predictions of the same split, so the
two arms share test rows and bootstrap draws (seed*7919+31, identical to the sealed
paired analyses). Metrics: pooled, hour-matched and position-matched AUROC.
Usage: 73_matched_time_paired_bootstrap.py --seed 101 [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from matched_time import OUT_ROOT, PROCESSED, RESULTS, MatchedAUC, patient_draws  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--n-boot", type=int, default=2000)
args = ap.parse_args()
s = args.seed
OUT = OUT_ROOT / "paired_bootstrap"
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()

END = pd.read_parquet(PROCESSED / "cinc2019_hourly/data.parquet", columns=["patient_id", "ICULOS"]).groupby("patient_id").ICULOS.max()
cd = pd.read_parquet(RESULTS / f"rootcause_stage0_2/p19_clinical_decomposition_pred_seed{s}.parquet")
y = cd.y_h6.to_numpy().astype(int)
hr = cd.anchor_hour.to_numpy()
hte = (cd.patient_id.map(END).to_numpy() - hr).astype(int)
_, inv, draws = patient_draws(cd.patient_id.to_numpy(), s, args.n_boot)
n_pat = int(inv.max()) + 1

preds = {"clin_V": cd["p__clinical_V"], "clin_VM0": cd["p__clinical_V+M0"], "clin_VI": cd["p__clinical_V+I"],
         "clin_VM0I": cd["p__clinical_V+M0+I"], "clin_noise": cd["p__clinical_V+I_equalized_noise"],
         "full_VM0I": cd["p__full_V+M0+I"]}
preds = {k: v.to_numpy() for k, v in preds.items()}


def load(path, key):
    x = pd.read_parquet(path)
    assert (x.patient_id.to_numpy() == cd.patient_id.to_numpy()).all() and (x.y.to_numpy().astype(int) == y).all(), path
    preds[key] = x.p.to_numpy()


for c in ["V", "V_I", "V_Mh", "V_Mh_I"]:
    load(RESULTS / f"stage3_lgb_v2_{c}_seed{s}_pred.parquet", "ctx_" + c)
for k, f in [("grud", "stage3_grud_nooi"), ("grud_oi", "stage3_grud_oi"), ("cde", "stage3_cde_nooi"), ("cde_oi", "stage3_cde_oi")]:
    load(RESULTS / f"{f}_seed{s}_pred.parquet", k)
for pth in sorted(glob.glob(str(RESULTS / f"improvement_fair_ssl/fair_pred_*_seed{s}_f*.parquet"))):
    m = re.search(r"fair_pred_(.+)_seed\d+_f([\d.]+)\.parquet", pth)
    load(pth, f"pt_{m.group(1)}_{m.group(2)}")
for pth in sorted(glob.glob(str(RESULTS / f"improvement_fair_ssl/terminal_controls_all/fair_pred_*_seed{s}_f*.parquet"))):
    m = re.search(r"fair_pred_(.+)_seed\d+_f([\d.]+)\.parquet", pth)
    load(pth, f"ptc_{m.group(1)}_{m.group(2)}")

ALL3 = ("pooled", "hour", "position")
CONTRASTS = [("clin_VI", "clin_V", ALL3), ("clin_VM0I", "clin_VM0", ALL3), ("full_VM0I", "clin_VM0I", ALL3),
             ("clin_noise", "clin_VM0I", ALL3), ("ctx_V_I", "ctx_V", ALL3), ("ctx_V_Mh_I", "ctx_V_Mh", ALL3),
             ("grud_oi", "grud", ALL3), ("cde_oi", "cde", ALL3)]
FR = ["0.01", "0.05", "0.10", "0.25", "1.00"]
for fr in FR:
    for m in ["nextmask4", "nextintensity4", "valueforecast4"]:
        CONTRASTS.append((f"pt_{m}_{fr}", f"pt_random_{fr}", ("pooled", "hour")))
    for m in ["terminal_only", "nextmask4", "nextintensity4", "valueforecast4"]:
        if f"ptc_{m}_{fr}" in preds and f"ptc_random_{fr}" in preds:
            CONTRASTS.append((f"ptc_{m}_{fr}", f"ptc_random_{fr}", ("hour",)))
KEYS = {"pooled": [], "hour": [hr], "position": [hr, hte]}
need = sorted({(k, met) for a, b, ms in CONTRASTS for met in ms for k in (a, b)})
objs = {k: MatchedAUC(y, preds[k[0]], KEYS[k[1]]) for k in need}
point = {k: o.auc() for k, o in objs.items()}
boot = {k: np.empty(args.n_boot) for k in need}
for i in range(args.n_boot):
    w = np.bincount(draws[i], minlength=n_pat)[inv].astype(float)
    for k, o in objs.items():
        boot[k][i] = o.auc(w)
    if i % 250 == 0:
        print(f"seed {s} draw {i} {time.time() - t0:.0f}s", flush=True)
rows = []
for a, b, ms in CONTRASTS:
    for met in ms:
        d = boot[(a, met)] - boot[(b, met)]
        rows.append(dict(seed=s, contrast=f"{a} - {b}", metric=met, arm=point[(a, met)], reference=point[(b, met)],
                         delta=point[(a, met)] - point[(b, met)], ci_lo=float(np.percentile(d, 2.5)),
                         ci_hi=float(np.percentile(d, 97.5)), n_boot=args.n_boot))
pd.DataFrame(rows).to_csv(OUT / f"paired_bootstrap_seed{s}.csv", index=False)
print(f"seed {s} done: {len(rows)} contrasts, {time.time() - t0:.0f}s", flush=True)
