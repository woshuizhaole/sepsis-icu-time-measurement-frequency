#!/usr/bin/env python3
"""Descriptive tabulation of the three record-continuation pretraining targets (Fig. 6A).

Targets follow code/analysis/40_improvement_fair_ssl.py::pre_items exactly:
  anchors  = ICU hours 1..24 of training patients (windows with a complete 4 h future only)
  mask     = any measurement of the channel in the next 4 h
  count    = number of hours (0-4) with a measurement of the channel in the next 4 h
  value    = standardised mean of measured values in the next 4 h (training median / std)
No model is trained here; the script only counts and extracts one real example patient.

Outputs (results/matched_time/pretraining_targets/):
  count_class_distribution.csv   seed, count_class, share, n  (pooled over the 34 channels)
  mask_prevalence.csv            seed, share of (anchor, channel) pairs with any measurement
  example_patient_hourly.csv     one patient, hours 1-24, channel, measured, standardised value
  targets_qc.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/data/workplace/wyx/measurement_intensity_ssl_20261015")
ap.add_argument("--example", default="p009862")
args = ap.parse_args()

P = Path(args.root)
D = P / "data/processed"
OUT = P / "results/matched_time/pretraining_targets"
OUT.mkdir(parents=True, exist_ok=True)

DYN = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2',
       'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate',
       'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen',
       'Platelets']
assert len(DYN) == 34
cols = ["patient_id", "ICULOS"] + [f"v__{c}" for c in DYN] + [f"m0__{c}" for c in DYN]
df = pd.read_parquet(D / "stage3_p19_features.parquet", columns=cols).sort_values(["patient_id", "ICULOS"]).reset_index(drop=True)
parts = json.loads((D / "partitions.json").read_text())["cinc2019"]

cls_rows, mask_rows, qc = [], [], {"seeds": {}}
std_example = None
for seed in sorted(parts, key=int):
    train = {k for k, v in parts[seed].items() if v == "train"}
    tr = df[df.patient_id.isin(train)]
    med = tr[[f"v__{c}" for c in DYN]].median(numeric_only=True).fillna(0).to_numpy("float32")
    sd = tr[[f"v__{c}" for c in DYN]].std(numeric_only=True).fillna(1.0).replace(0, 1.0).to_numpy("float32")
    counts = np.zeros(5, dtype=np.int64)
    n_anchor = 0
    ids = tr.patient_id.to_numpy()
    cuts = np.r_[0, 1 + np.flatnonzero(ids[1:] != ids[:-1]), len(ids)]
    M = tr[[f"m0__{c}" for c in DYN]].to_numpy("float32")
    hours = tr.ICULOS.to_numpy().astype(int)
    for a, b in zip(cuts[:-1], cuts[1:]):
        minh = hours[a]
        n = b - a
        lo, hi = max(1, minh), min(hours[b - 1], 24)
        if hi < lo:
            continue
        hs = np.arange(lo, hi + 1)
        idx = hs - minh
        idx = idx[(idx + 4) < n]
        if not len(idx):
            continue
        fm = np.stack([M[a + idx + j] for j in range(1, 5)], axis=1)
        cnt = np.rint(fm.sum(axis=1)).astype(int)
        counts += np.bincount(cnt.ravel(), minlength=5)[:5]
        n_anchor += len(idx)
    tot = counts.sum()
    for k in range(5):
        cls_rows.append({"seed": int(seed), "count_class": k, "share": counts[k] / tot, "n": int(counts[k])})
    mask_rows.append({"seed": int(seed), "mask_positive_share": float(1 - counts[0] / tot), "n_anchor_windows": n_anchor, "n_pairs": int(tot),
                      "n_train_patients": len(train)})
    qc["seeds"][seed] = {"n_anchor_windows": n_anchor, "pairs": int(tot), "class_share": [round(c / tot, 4) for c in counts]}
    if seed == "101":
        std_example = (med, sd)

pd.DataFrame(cls_rows).to_csv(OUT / "count_class_distribution.csv", index=False)
pd.DataFrame(mask_rows).to_csv(OUT / "mask_prevalence.csv", index=False)

# one real example patient: hours 1-24, measured flag and standardised value for every channel
med, sd = std_example
ex = df[(df.patient_id == args.example) & (df.ICULOS <= 24)]
assert len(ex) > 0, "example patient missing"
rows = []
for r in ex.itertuples(index=False):
    for j, c in enumerate(DYN):
        v = getattr(r, f"v__{c}")
        m = getattr(r, f"m0__{c}")
        rows.append({"patient_id": r.patient_id, "icu_hour": int(r.ICULOS), "channel": c, "measured": int(m),
                     "value_std": float((v - med[j]) / sd[j]) if np.isfinite(v) else np.nan})
pd.DataFrame(rows).to_csv(OUT / "example_patient_hourly.csv", index=False)
qc.update({"example_patient": args.example, "example_hours": int(ex.ICULOS.nunique()), "status": "PASS"})
(OUT / "targets_qc.json").write_text(json.dumps(qc, indent=2))
print(json.dumps(qc, indent=2))
