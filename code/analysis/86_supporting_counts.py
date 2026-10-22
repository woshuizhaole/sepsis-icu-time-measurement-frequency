#!/usr/bin/env python3
"""Supporting counts quoted in the Results text (Fig. 1G paragraph and Fig. 5F paragraph).

Descriptive only; no model is trained.
1. CinC 2019: distribution of the hours between each positive prediction time and the last
   recorded hour, and the event rate inside and outside the 8-15 h band.
2. CinC 2012: mechanical ventilation in the first 24 h and its relation to the number of
   manually recorded respiratory-rate measurements, overall and by in-hospital death,
   in the 24-h landmark cohort.
Outputs: results/matched_time/supporting_counts/*.csv and summary.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from matched_time import OUT_ROOT, PROCESSED  # noqa: E402

OUT = OUT_ROOT / "supporting_counts"
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
summary: dict = {}

# ---------------------------------------------------------------- CinC 2019: hours to end of record
hourly = pd.read_parquet(PROCESSED / "cinc2019_hourly/data.parquet", columns=["patient_id", "ICULOS"])
end = hourly.groupby("patient_id").ICULOS.max()
win = pd.read_parquet(PROCESSED / "cinc2019_windows.parquet", columns=["patient_id", "anchor_hour", "y_h6"])
win["hours_to_end"] = win.patient_id.map(end) - win.anchor_hour
pos = win[win.y_h6 == 1]
dist = pos.hours_to_end.value_counts().sort_index().rename_axis("hours_to_end").reset_index(name="positive_prediction_times")
dist["share"] = dist.positive_prediction_times / len(pos)
dist.to_csv(OUT / "cinc2019_positive_hours_to_end.csv", index=False)
band = win.hours_to_end.between(8, 15)
summary["cinc2019"] = {
    "prediction_times": int(len(win)), "positive_prediction_times": int(len(pos)),
    "max_hours_to_end_positive": int(pos.hours_to_end.max()),
    "share_positive_within_15h": float((pos.hours_to_end <= 15).mean()),
    "share_positive_8_to_15h": float(pos.hours_to_end.between(8, 15).mean()),
    "event_rate_8_to_15h": float(win.loc[band, "y_h6"].mean()),
    "event_rate_outside_8_to_15h": float(win.loc[~band, "y_h6"].mean()),
}

# ---------------------------------------------------------------- CinC 2012: ventilation and respiratory rate
lm = pd.read_parquet(PROCESSED / "cinc2012_landmarks.parquet")
lm24 = lm[lm.cutoff_hour == 24].set_index("patient_id")
df = pd.read_parquet(PROCESSED / "cinc2012_hourly/data.parquet",
                     columns=["patient_id", "icu_hour", "event_count__MechVent", "value__MechVent", "event_count__RespRate"])
h = df[df.icu_hour < 24]
g = h.groupby("patient_id")
pt = pd.DataFrame({"vent": (g.event_count__MechVent.sum() > 0), "resp_count": g.event_count__RespRate.sum()})
z = lm24.join(pt, how="left")
z["vent"] = z.vent.fillna(False).astype(bool)
z["resp_count"] = z.resp_count.fillna(0)
y = z.mortality_label.astype(int)
z.assign(died=y)[["died", "vent", "resp_count"]].to_csv(OUT / "cinc2012_ventilation_resprate_24h.csv")
summary["cinc2012_24h"] = {
    "patients": int(len(z)), "died": int(y.sum()),
    "share_ventilated": float(z.vent.mean()),
    "ventilated_with_any_resprate": int((z.vent & (z.resp_count > 0)).sum()),
    "median_resprate_count_ventilated": float(z.loc[z.vent, "resp_count"].median()),
    "median_resprate_count_not_ventilated": float(z.loc[~z.vent, "resp_count"].median()),
    "share_ventilated_died": float(z.loc[y == 1, "vent"].mean()),
    "share_ventilated_survived": float(z.loc[y == 0, "vent"].mean()),
    "mechvent_values_observed": sorted(map(float, df.value__MechVent.dropna().unique()))[:10],
}
summary["runtime_s"] = round(time.time() - t0, 1)
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
