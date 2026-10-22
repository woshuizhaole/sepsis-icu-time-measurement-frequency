#!/usr/bin/env python
"""Table 1, CinC 2012 block, with in-hospital death taken from the official Outcomes-a/b files.

Script 70 took death from the landmark table, so the 15 patients without any landmark row
were counted under "All" but in neither outcome group. This script recomputes the same rows
for all 8,000 patients and rewrites the CinC 2012 block of table1_cohort_characteristics.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from matched_time import OUT_ROOT, PROCESSED, PROJECT  # noqa: E402

OUT = OUT_ROOT / "record_structure"
RAW = PROJECT / "data/raw/cinc2012"


def q(s):
    s = pd.Series(s).dropna()
    return f"{s.median():.1f} ({s.quantile(.25):.1f}–{s.quantile(.75):.1f})"


def pc(s):
    return f"{100 * pd.Series(s).mean():.1f}"


h2 = pd.read_parquet(PROCESSED / "cinc2012_hourly/data.parquet")
outc = pd.concat([pd.read_csv(RAW / f"Outcomes-{s}.txt") for s in ("a", "b")])
outc["patient_id"] = outc.RecordID.astype(str)
ec = [c for c in h2.columns if c.startswith("event_count__")]
pp = h2.groupby("patient_id").agg(age=("Age", "first"), male=("Gender", "first"), icu=("ICUType", "first"))
pp["died"] = outc.set_index("patient_id")["In-hospital_death"].reindex(pp.index)
assert pp.died.notna().all() and len(pp) == 8000
for cut in (12, 24):
    tot = h2[h2.icu_hour < cut].groupby("patient_id")[ec].sum().reindex(pp.index).fillna(0)
    pp[f"breadth{cut}"] = (tot > 0).sum(1)
    pp[f"events{cut}"] = tot.sum(1)
rows = []
for lab, x in [("All", pp), ("Group 1", pp[pp.died == 1]), ("Group 2", pp[pp.died == 0])]:
    rows.append({"Patients, n": f"{len(x):,}", "Age, years": q(x.age), "Male sex, %": pc(x.male == 1),
                 "Coronary care unit, %": pc(x.icu == 1), "Cardiac surgery recovery unit, %": pc(x.icu == 2),
                 "Medical ICU, %": pc(x.icu == 3), "Surgical ICU, %": pc(x.icu == 4),
                 "Parameters measured in the first 12 h, n": q(x.breadth12),
                 "Parameters measured in the first 24 h, n": q(x.breadth24),
                 "Measurement events in the first 24 h, n": q(x.events24)})
t12 = pd.DataFrame(rows, index=["All", "Group 1", "Group 2"]).T.rename_axis("Characteristic").reset_index()
t12.insert(0, "Cohort", "CinC 2012 mortality cohort (Group 1 = in-hospital death, Group 2 = survived)")
t12["Summary"] = ["count", "median (IQR)", "percentage", "percentage", "percentage", "percentage", "percentage",
                  "median (IQR)", "median (IQR)", "median (IQR)"]
path = OUT / "table1_cohort_characteristics.csv"
t1 = pd.read_csv(path)
t1 = pd.concat([t1[~t1.Cohort.str.startswith("CinC 2012")], t12], ignore_index=True)
t1.to_csv(path, index=False)
qc = {"patients": int(len(pp)), "deaths": int(pp.died.sum()), "survivors": int((pp.died == 0).sum()),
      "no_events_before_12h": int((pp.events12 == 0).sum()), "no_events_before_24h": int((pp.events24 == 0).sum())}
(OUT / "table1_cinc2012_outcomes_qc.json").write_text(json.dumps(qc, indent=2))
print(t12.to_string())
print(qc)
