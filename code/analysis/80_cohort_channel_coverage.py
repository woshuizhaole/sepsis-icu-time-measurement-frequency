#!/usr/bin/env python3
"""Share of patients with at least one measurement of a channel in the first 24 ICU hours, both cohorts (Suppl. Fig. 3 margins).

Descriptive only. The 24 channels are the CinC 2019 / CinC 2012 pairs used in
76_matched_time_cinc2012_descriptive.py::MAP (channel_concordance.csv).
  CinC 2019: all 40,336 patients, hours ICULOS <= 24 (a patient with a shorter record contributes the hours it has)
  CinC 2012: patients with a record at the 24 h landmark (cutoff_hour == 24), events before icu_hour < 24
Output: results/matched_time/cinc2012_descriptive/channel_coverage_24h.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from matched_time import OUT_ROOT, PROCESSED  # noqa: E402

OUT = OUT_ROOT / "cinc2012_descriptive"
MAP = {"HR": "HR", "Temp": "Temp", "SBP": "SysABP", "MAP": "MAP", "DBP": "DiasABP", "Resp": "RespRate", "HCO3": "HCO3", "FiO2": "FiO2",
       "pH": "pH", "PaCO2": "PaCO2", "AST": "AST", "BUN": "BUN", "Creatinine": "Creatinine", "Glucose": "Glucose", "Lactate": "Lactate",
       "Magnesium": "Mg", "Potassium": "K", "TroponinI": "TroponinI", "Hct": "HCT", "WBC": "WBC", "Platelets": "Platelets",
       "SaO2": "SaO2", "Alkalinephos": "ALP", "Bilirubin_total": "Bilirubin"}

h19 = pd.read_parquet(PROCESSED / "cinc2019_hourly/data.parquet", columns=["patient_id", "ICULOS"] + list(MAP))
h19 = h19[h19.ICULOS <= 24]
n19 = h19.patient_id.nunique()
any19 = h19.groupby("patient_id")[list(MAP)].agg(lambda s: s.notna().any()).mean()

lm = pd.read_parquet(PROCESSED / "cinc2012_landmarks.parquet")
keep = set(lm[lm.cutoff_hour == 24].patient_id)
h12 = pd.read_parquet(PROCESSED / "cinc2012_hourly/data.parquet")
h12 = h12[h12.patient_id.isin(keep) & (h12.icu_hour < 24)]
cols12 = {b: f"event_count__{b}" for b in MAP.values()}
tot12 = h12.groupby("patient_id")[list(cols12.values())].sum().reindex(sorted(keep)).fillna(0)
any12 = (tot12 > 0).mean()

rows = [{"cinc2019_channel": a, "cinc2012_parameter": b, "share_patients_cinc2019": float(any19[a]),
         "share_patients_cinc2012": float(any12[cols12[b]]), "n_patients_cinc2019": int(n19), "n_patients_cinc2012": int(len(keep))}
        for a, b in MAP.items()]
out = pd.DataFrame(rows)
assert len(out) == 24 and out[["share_patients_cinc2019", "share_patients_cinc2012"]].notna().all().all()
out.to_csv(OUT / "channel_coverage_24h.csv", index=False)
print(out.round(3).to_string())
print(f"n19={n19}  n12={len(keep)}")
