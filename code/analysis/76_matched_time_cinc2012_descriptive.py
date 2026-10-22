#!/usr/bin/env python3
"""CinC 2012 landmark charting description and cross-cohort family concordance (Fig. 5E-G, Suppl. Fig. 3).

Descriptive only. Per-patient measurement breadth, event totals and per-parameter
event counts before the 12 h / 24 h cutoff (icu_hour < cutoff), univariate AUROC for
in-hospital death overall and within ICU type; concordance with the CinC 2019
hour-matched channel AUROCs written by 72_matched_time_charting_phenotype.py.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from matched_time import OUT_ROOT, PROCESSED  # noqa: E402

OUT = OUT_ROOT / "cinc2012_descriptive"
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
lm = pd.read_parquet(PROCESSED / "cinc2012_landmarks.parquet")
df = pd.read_parquet(PROCESSED / "cinc2012_hourly/data.parquet")
cnt = [c for c in df.columns if c.startswith("event_count__")]
icu = df.groupby("patient_id").ICUType.first()
ICU = {1: "Coronary care", 2: "Cardiac surgery recovery", 3: "Medical", 4: "Surgical"}
rows, per_param, dist = [], [], []
for c in (12, 24):
    h = df[df.icu_hour < c]
    tot = h.groupby("patient_id")[cnt].sum()
    feat = pd.DataFrame({"total_events": tot.sum(1), "breadth": (tot > 0).sum(1),
                         "repeat": (tot - 1).clip(lower=0).sum(1)})
    z = lm[lm.cutoff_hour == c].set_index("patient_id").join(feat, how="left").join(icu, how="left")
    z[feat.columns] = z[feat.columns].fillna(0)
    y = z.mortality_label.astype(int)
    for f in feat.columns:
        r = {"cutoff_h": c, "feature": f, "median_died": z.loc[y == 1, f].median(), "median_survived": z.loc[y == 0, f].median(),
             "auroc": roc_auc_score(y, z[f])}
        for k, lab in ICU.items():
            m = z.ICUType == k
            r[f"auroc_{lab.lower().replace(' ', '_')}"] = roc_auc_score(y[m], z.loc[m, f]) if y[m].nunique() == 2 else np.nan
        rows.append(r)
    for k, lab in ICU.items():
        m = z.ICUType == k
        dist.append(pd.DataFrame({"cutoff_h": c, "icu_type": lab, "died": y[m].to_numpy(), "breadth": z.loc[m, "breadth"].to_numpy()}))
    zt = lm[lm.cutoff_hour == c].set_index("patient_id").join(tot, how="left").fillna(0)
    for x in cnt:
        per_param.append({"cutoff_h": c, "parameter": x.replace("event_count__", ""), "auroc": roc_auc_score(zt.mortality_label, zt[x]),
                          "mean_died": zt.loc[zt.mortality_label == 1, x].mean(), "mean_survived": zt.loc[zt.mortality_label == 0, x].mean()})
pd.DataFrame(rows).to_csv(OUT / "charting_summary_auroc.csv", index=False)
pp = pd.DataFrame(per_param)
pp.to_csv(OUT / "parameter_count_auroc.csv", index=False)
pd.concat(dist).to_csv(OUT / "breadth_by_icu_type.csv", index=False)

MAP = {"HR": "HR", "Temp": "Temp", "SBP": "SysABP", "MAP": "MAP", "DBP": "DiasABP", "Resp": "RespRate", "HCO3": "HCO3", "FiO2": "FiO2",
       "pH": "pH", "PaCO2": "PaCO2", "AST": "AST", "BUN": "BUN", "Creatinine": "Creatinine", "Glucose": "Glucose", "Lactate": "Lactate",
       "Magnesium": "Mg", "Potassium": "K", "TroponinI": "TroponinI", "Hct": "HCT", "WBC": "WBC", "Platelets": "Platelets",
       "SaO2": "SaO2", "Alkalinephos": "ALP", "Bilirubin_total": "Bilirubin"}
ch_path = OUT_ROOT / "charting_phenotype/channel_hour_matched_auroc.csv"
summary = {"runtime_s": None}
if ch_path.exists():
    ch = pd.read_csv(ch_path).set_index("feature")
    con = []
    for a, b in MAP.items():
        r = pp[(pp.cutoff_h == 24) & (pp.parameter == b)]
        con.append({"cinc2019_channel": a, "cinc2012_parameter": b, "family": ch.loc[a, "family"],
                    "cinc2019_hour_matched_auroc": ch.loc[a, "auroc_hour_matched"], "cinc2012_24h_auroc": float(r.auroc.iloc[0])})
    con = pd.DataFrame(con)
    con.to_csv(OUT / "channel_concordance.csv", index=False)
    fam = con.groupby("family")[["cinc2019_hour_matched_auroc", "cinc2012_24h_auroc"]].mean()
    fam.to_csv(OUT / "family_concordance.csv")
    summary["channel_spearman"] = float(spearmanr(con.cinc2019_hour_matched_auroc, con.cinc2012_24h_auroc)[0])
    summary["family_means"] = fam.round(3).to_dict()
summary["runtime_s"] = round(time.time() - t0, 1)
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=1))
