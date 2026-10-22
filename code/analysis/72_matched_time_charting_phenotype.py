#!/usr/bin/env python3
"""Charting-intensity phenotype in CinC 2019 (Fig. 4; Suppl. Fig. 3 inputs).

Descriptive only. Uses the hourly table, the pre-first-positive window index and the
stored 4-h per-channel charting counts (i_count4__*). Hours remaining in the record
are used only for the end-of-record alignment curves.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from matched_time import DYN, FAMILIES, OUT_ROOT, PROCESSED, MatchedAUC  # noqa: E402

OUT = OUT_ROOT / "charting_phenotype"
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
summary: dict = {}
FAM_OF = {c: f for f, cs in FAMILIES.items() for c in cs}


def auc(y, s):
    y = np.asarray(y)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def mean_ci(df, keys, val):
    g = df.groupby(keys)[val]
    m, sd, n = g.mean(), g.std(), g.size()
    out = pd.DataFrame({"mean": m, "ci_lo": m - 1.96 * sd / np.sqrt(n), "ci_hi": m + 1.96 * sd / np.sqrt(n), "n_rows": n})
    return out.reset_index()


# ---------------------------------------------------------------- hourly charting by family
h = pd.read_parquet(PROCESSED / "cinc2019_hourly/data.parquet", columns=["patient_id", "ICULOS", "SepsisLabel"] + DYN)
for f, cs in FAMILIES.items():
    h[f] = h[cs].notna().sum(1)
h["All channels"] = h[list(FAMILIES)].sum(1)
END = h.groupby("patient_id").ICULOS.max()
ONSET = h[h.SepsisLabel == 1].groupby("patient_id").ICULOS.min()
h["record_end"] = h.patient_id.map(END)
h["first_label"] = h.patient_id.map(ONSET)
h["group"] = np.where(h.first_label.notna(), "Sepsis", "No sepsis")
pre = h[h.first_label.isna() | (h.ICULOS < h.first_label)]
fams = list(FAMILIES) + ["All channels"]

curves = []
for f in fams:
    c = mean_ci(pre[pre.ICULOS <= 120], ["group", "ICULOS"], f)
    c["family"] = f
    curves.append(c)
pd.concat(curves).to_csv(OUT / "charting_by_icu_hour.csv", index=False)

fx = h[h.first_label >= 49].copy()
fx["hours_from_first_label"] = fx.ICULOS - fx.first_label
fx = fx[fx.hours_from_first_label.between(-48, 9)]
cur = []
for f in fams:
    c = mean_ci(fx, ["hours_from_first_label"], f)
    c["family"] = f
    cur.append(c)
cur = pd.concat(cur)
cur["n_patients"] = int(fx.patient_id.nunique())
cur.to_csv(OUT / "charting_around_first_label_fixed_cohort.csv", index=False)
summary["fixed_cohort_patients"] = int(fx.patient_id.nunique())

ng = h[h.first_label.isna()].copy()
ng["hours_to_record_end"] = ng.record_end - ng.ICULOS
ng = ng[ng.hours_to_record_end <= 48]
cur = []
for f in fams:
    c = mean_ci(ng, ["hours_to_record_end"], f)
    c["family"] = f
    cur.append(c)
pd.concat(cur).to_csv(OUT / "charting_before_record_end_no_sepsis.csv", index=False)

# between-patient variance share and patient-level means (exclude hour 1, which is never charted)
p2 = pre[pre.ICULOS >= 2].copy()
p2["sepsis"] = p2.first_label.notna().astype(int)
var_rows, pm_rows = [], []
for f in fams:
    gm = p2.groupby("patient_id")[f]
    within = ((p2[f] - gm.transform("mean")) ** 2).sum() / (len(p2) - 1)
    pmean = gm.mean()
    sep = p2.groupby("patient_id").sepsis.first().reindex(pmean.index)
    var_rows.append({"family": f, "between_patient_share": float(1 - within / p2[f].var()),
                     "patient_mean_auroc_for_sepsis": auc(sep.to_numpy(), pmean.to_numpy()),
                     "median_sepsis": float(pmean[sep == 1].median()), "q25_sepsis": float(pmean[sep == 1].quantile(.25)), "q75_sepsis": float(pmean[sep == 1].quantile(.75)),
                     "median_no_sepsis": float(pmean[sep == 0].median()), "q25_no_sepsis": float(pmean[sep == 0].quantile(.25)), "q75_no_sepsis": float(pmean[sep == 0].quantile(.75))})
    pm_rows.append(pd.DataFrame({"patient_id": pmean.index, "family": f, "mean_channels_per_hour": pmean.to_numpy(), "sepsis": sep.to_numpy()}))
pd.DataFrame(var_rows).to_csv(OUT / "between_patient_variance.csv", index=False)
pd.concat(pm_rows).to_parquet(OUT / "patient_mean_charting.parquet", index=False)

# ---------------------------------------------------------------- channel- and family-level hour-matched AUROC of 4-h counts
cc = [f"i_count4__{c}" for c in DYN]
F = pd.read_parquet(PROCESSED / "stage3_p19_features_v2.parquet", columns=["patient_id", "ICULOS"] + cc + ["i_breadth4"]).rename(columns={"ICULOS": "anchor_hour"})
w = pd.read_parquet(PROCESSED / "cinc2019_windows.parquet", columns=["patient_id", "anchor_hour", "y_h6"])
d = w.merge(F, on=["patient_id", "anchor_hour"], how="left", validate="one_to_one")
assert d[cc].notna().all().all()
for f, cs in FAMILIES.items():
    d[f] = d[[f"i_count4__{c}" for c in cs]].sum(1)
d["All channels"] = d[list(FAMILIES)].sum(1)
d["hours_to_record_end"] = d.patient_id.map(END) - d.anchor_hour
d["ever_sepsis"] = d.patient_id.isin(ONSET.index)
y = d.y_h6.to_numpy().astype(int)
hr = d.anchor_hour.to_numpy()
keep = d.hours_to_record_end.to_numpy() > 8
nt = d[~d.ever_sepsis]
end8 = (nt.hours_to_record_end.to_numpy() <= 8).astype(int)
rows = []
for col in cc + ["i_breadth4"] + fams:
    v = d[col].to_numpy()
    name = col.replace("i_count4__", "")
    rows.append({"feature": "Measurement breadth" if col == "i_breadth4" else name,
                 "family": FAM_OF.get(name, "Summary"),
                 "auroc_overall": auc(y, v), "auroc_hour_matched": MatchedAUC(y, v, [hr]).auc(),
                 "auroc_hour_matched_excluding_last_8h": MatchedAUC(y[keep], v[keep], [hr[keep]]).auc(),
                 "auroc_for_record_end_within_8h_no_sepsis": MatchedAUC(end8, nt[col].to_numpy(), [nt.anchor_hour.to_numpy()]).auc(),
                 "mean_4h_count_event": float(v[y == 1].mean()), "mean_4h_count_no_event": float(v[y == 0].mean()),
                 "share_nonzero": float((v > 0).mean())})
ch = pd.DataFrame(rows).sort_values("auroc_hour_matched", ascending=False)
ch.to_csv(OUT / "channel_hour_matched_auroc.csv", index=False)
summary["family_hour_matched_auroc"] = ch[ch.feature.isin(fams)].set_index("feature").auroc_hour_matched.round(3).to_dict()
summary["top_channels"] = ch[~ch.feature.isin(fams + ["Measurement breadth"])].head(6).set_index("feature").auroc_hour_matched.round(3).to_dict()
summary["runtime_s"] = round(time.time() - t0, 1)
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=1))
