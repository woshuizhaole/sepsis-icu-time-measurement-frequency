#!/usr/bin/env python3
"""Record structure of the CinC 2019 cohort and cohort characteristics (Fig. 1, Table 1, Suppl. Fig. 1).

Descriptive only: no model is trained. Inputs are the processed hourly table, the
pre-first-positive window index and the CinC 2012 hourly/landmark tables.
Outputs: results/matched_time/record_structure/*.csv and summary.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from matched_time import DYN, FAMILIES, OUT_ROOT, PROCESSED, wilson  # noqa: E402

OUT = OUT_ROOT / "record_structure"
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
summary: dict = {}

# ---------------------------------------------------------------- CinC 2019 patient table
cols = ["patient_id", "source_set", "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS", "SepsisLabel"] + DYN
h = pd.read_parquet(PROCESSED / "cinc2019_hourly/data.parquet", columns=cols).sort_values(["patient_id", "ICULOS"])
vit = FAMILIES["Vital signs"]
labs = [c for c in DYN if c not in vit]
h["obs"] = h[DYN].notna().sum(1)
h["obs_vital"] = h[vit].notna().sum(1)
h["obs_lab"] = h[labs].notna().sum(1)
g = h.groupby("patient_id")
pat = pd.DataFrame({
    "source": g.source_set.first(), "age": g.Age.first(), "male": g.Gender.first(),
    "unit1": g.Unit1.max(), "unit2": g.Unit2.max(), "hosp_to_icu_h": -g.HospAdmTime.first(),
    "first_hour": g.ICULOS.min(), "record_end": g.ICULOS.max(), "n_rows": g.size(),
    "sepsis": g.SepsisLabel.max().astype(int), "obs": g.obs.mean(), "obs_vital": g.obs_vital.mean(),
    "obs_lab": g.obs_lab.mean(),
})
pat["first_label"] = h[h.SepsisLabel == 1].groupby("patient_id").ICULOS.min()
pat["label_to_end_h"] = pat.record_end - pat.first_label
pat["label_at_first_record"] = (pat.first_label == pat.first_hour)
sep = pat[pat.sepsis == 1]
non = pat[pat.sepsis == 0]
summary["patients"] = {"all": int(len(pat)), "sepsis": int(len(sep)), "no_sepsis": int(len(non))}
summary["contiguous_hourly_rows"] = bool((pat.n_rows == pat.record_end - pat.first_hour + 1).all())

# first label -> end of record
vc = sep.label_to_end_h.value_counts().sort_index()
lab_end = pd.DataFrame({"hours_first_label_to_record_end": vc.index.astype(int), "patients": vc.values})
lab_end["share"] = lab_end.patients / len(sep)
lab_end.to_csv(OUT / "first_label_to_record_end.csv", index=False)
summary["label_to_end"] = {
    "median": float(sep.label_to_end_h.median()), "max": float(sep.label_to_end_h.max()),
    "share_7_to_9h": float(sep.label_to_end_h.between(7, 9).mean()),
    "share_9h": float((sep.label_to_end_h == 9).mean()),
    "label_at_first_record": int(sep.label_at_first_record.sum()),
}

# record length distribution and sepsis share by length
rl = pat.groupby(["n_rows", "sepsis"]).size().unstack(fill_value=0).rename(columns={0: "no_sepsis", 1: "sepsis"})
rl["sepsis_share"] = rl.sepsis / (rl.sepsis + rl.no_sepsis)
rl = rl.reset_index().rename(columns={"n_rows": "record_length_h"})
rl.to_csv(OUT / "record_length_distribution.csv", index=False)
summary["record_length_gt60_share"] = {"sepsis": float((sep.n_rows > 60).mean()), "no_sepsis": float((non.n_rows > 60).mean())}
summary["sepsis_share_at_length"] = {int(r.record_length_h): float(r.sepsis_share) for r in rl[rl.record_length_h.between(55, 65)].itertuples()}

# hospital systems
sysrows = []
for s, x in pat.groupby("source"):
    xs, xn = x[x.sepsis == 1], x[x.sepsis == 0]
    sysrows.append({"hospital_system": "A" if s.endswith("A") else "B", "patients": len(x), "sepsis": len(xs),
                    "sepsis_share_records_end_7_9h_after_label": xs.label_to_end_h.between(7, 9).mean(),
                    "no_sepsis_records_longer_than_60h": (xn.n_rows > 60).mean(),
                    "sepsis_records_longer_than_60h": (xs.n_rows > 60).mean()})
pd.DataFrame(sysrows).to_csv(OUT / "hospital_system_structure.csv", index=False)

# ---------------------------------------------------------------- prediction-time index
w = pd.read_parquet(PROCESSED / "cinc2019_windows.parquet", columns=["patient_id", "anchor_hour", "y_h6"])
w["record_end"] = w.patient_id.map(pat.record_end)
w["sepsis"] = w.patient_id.map(pat.sepsis)
er = w[w.anchor_hour <= 120].groupby("anchor_hour").agg(n=("y_h6", "size"), events=("y_h6", "sum"),
                                                       sepsis_share=("sepsis", "mean")).reset_index()
er["event_rate"] = er.events / er.n
er["ci_lo"], er["ci_hi"] = wilson(er.events, er.n)
er.to_csv(OUT / "event_rate_by_icu_hour.csv", index=False)
summary["event_rate"] = {"min_hour": int(er.loc[er.event_rate.idxmin(), "anchor_hour"]), "min_rate": float(er.event_rate.min()),
                         "hour48": float(er.set_index("anchor_hour").event_rate.get(48, np.nan)),
                         "hour60": float(er.set_index("anchor_hour").event_rate.get(60, np.nan))}

hb = [0, 6, 12, 24, 36, 48, 60, 72, 96, 10 ** 6]
hl = ["1-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61-72", "73-96", ">96"]
lb = [0, 12, 24, 36, 48, 60, 72, 96, 10 ** 6]
ll = ["8-12", "13-24", "25-36", "37-48", "49-60", "61-72", "73-96", ">96"]
w["hour_bin"] = pd.cut(w.anchor_hour, hb, labels=hl)
w["length_bin"] = pd.cut(w.record_end, lb, labels=ll)
hm = w.groupby(["length_bin", "hour_bin"], observed=True).y_h6.agg(event_rate="mean", events="sum", n="size").reset_index()
hm.to_csv(OUT / "event_rate_by_hour_and_record_length.csv", index=False)

# horizon evaluability (Suppl. Fig. 1): anchor evaluable if the record extends to anchor + H
hz = []
for H in [1, 4, 6, 8, 12, 24]:
    ok = w.record_end >= w.anchor_hour + H
    for grp, lab in [(1, "Sepsis"), (0, "No sepsis")]:
        m = w.sepsis == grp
        hz.append({"horizon_h": H, "group": lab, "evaluable_fraction": float(ok[m].mean()), "anchors": int(m.sum())})
    hz.append({"horizon_h": H, "group": "Sepsis share among evaluable anchors", "evaluable_fraction": float(w[ok].sepsis.mean()), "anchors": int(ok.sum())})
pd.DataFrame(hz).to_csv(OUT / "horizon_evaluability.csv", index=False)

# ---------------------------------------------------------------- representative records (Fig. 1A)
fam_of = {c: f for f, cs in FAMILIES.items() for c in cs}
cand_s = sep[(sep.first_label.between(40, 60)) & (sep.label_to_end_h == 9) & (~sep.label_at_first_record)].copy()
cand_s["dist"] = (cand_s.obs - sep.obs.median()).abs()
cand_n = non[non.n_rows == int(non.n_rows.median())].copy()
cand_n["dist"] = (cand_n.obs - non.obs.median()).abs()
ex_ids = [cand_s.sort_values(["dist"]).index[0], cand_n.sort_values(["dist"]).index[0]]
ex = h[h.patient_id.isin(ex_ids)][["patient_id", "ICULOS"] + DYN].melt(id_vars=["patient_id", "ICULOS"], var_name="channel", value_name="value")
ex = ex[ex.value.notna()].drop(columns="value")
ex["family"] = ex.channel.map(fam_of)
ex["group"] = ex.patient_id.map(lambda p: "Sepsis" if pat.loc[p, "sepsis"] == 1 else "No sepsis")
ex.to_csv(OUT / "example_records_cinc2019.csv", index=False)
summary["example_records_cinc2019"] = {str(p): {"group": "Sepsis" if pat.loc[p, "sepsis"] else "No sepsis",
                                               "record_end": int(pat.loc[p, "record_end"]),
                                               "first_label": (None if pd.isna(pat.loc[p, "first_label"]) else int(pat.loc[p, "first_label"]))}
                                       for p in ex_ids}

# ---------------------------------------------------------------- Table 1, CinC 2019
def q(s):
    s = pd.Series(s).dropna()
    return f"{s.median():.1f} ({s.quantile(.25):.1f}–{s.quantile(.75):.1f})"


def pc(s):
    return f"{100 * pd.Series(s).mean():.1f}"


wpat = w.groupby("patient_id").agg(n_win=("y_h6", "size"), n_win_pos=("y_h6", "sum"))
pat = pat.join(wpat)
t1 = []
for lab, x in [("All", pat), ("Group 1", pat[pat.sepsis == 1]), ("Group 2", pat[pat.sepsis == 0])]:
    has_label = x.first_label.notna().any()
    t1.append({
        "Patients, n": f"{len(x):,}",
        "Contributing hospital system A, %": pc(x.source.str.endswith("A")),
        "Age, years": q(x.age), "Male sex, %": pc(x.male == 1),
        "Medical ICU, %": pc(x.unit1 == 1), "Surgical ICU, %": pc(x.unit2 == 1),
        "ICU type not recorded, %": pc(x.unit1.isna() & x.unit2.isna()),
        "Hospital-to-ICU interval, h": q(x.hosp_to_icu_h),
        "Recorded ICU stay, h": q(x.n_rows),
        "Recorded ICU stay longer than 60 h, %": pc(x.n_rows > 60),
        "ICU hour of first sepsis label": q(x.first_label) if (has_label and lab == "Group 1") else "—",
        "Hours from first sepsis label to end of record": q(x.label_to_end_h) if (has_label and lab == "Group 1") else "—",
        "Sepsis label present at the first recorded hour, n (%)": (f"{int(x.label_at_first_record.sum())} ({100 * x.label_at_first_record.mean():.1f})" if lab == "Group 1" else "—"),
        "Records ending 7–9 h after the first sepsis label, %": (pc(x.label_to_end_h.between(7, 9)) if lab == "Group 1" else "—"),
        "Channels charted per hour, all": q(x.obs),
        "Channels charted per hour, vital signs": q(x.obs_vital),
        "Channels charted per hour, laboratory": q(x.obs_lab),
        "Hourly prediction times before the first sepsis label, n": f"{int(x.n_win.fillna(0).sum()):,}",
        "Prediction times followed by a sepsis label within 6 h, n": f"{int(x.n_win_pos.fillna(0).sum()):,}",
    })
t19 = pd.DataFrame(t1, index=["All", "Group 1", "Group 2"]).T.rename_axis("Characteristic").reset_index()
t19.insert(0, "Cohort", "CinC 2019 sepsis cohort (Group 1 = sepsis, Group 2 = no sepsis)")

# ---------------------------------------------------------------- Table 1, CinC 2012
h2 = pd.read_parquet(PROCESSED / "cinc2012_hourly/data.parquet")
lm = pd.read_parquet(PROCESSED / "cinc2012_landmarks.parquet")
ec = [c for c in h2.columns if c.startswith("event_count__")]
pp = h2.groupby("patient_id").agg(age=("Age", "first"), male=("Gender", "first"), icu=("ICUType", "first"))
pp["died"] = lm.drop_duplicates("patient_id").set_index("patient_id").mortality_label
for cut in (12, 24):
    sub = h2[h2.icu_hour < cut]
    tot = sub.groupby("patient_id")[ec].sum()
    pp[f"breadth{cut}"] = (tot > 0).sum(1)
    pp[f"events{cut}"] = tot.sum(1)
summary["cinc2012"] = {"patients": int(len(pp)), "deaths": int((pp.died == 1).sum()), "no_landmark_label": int(pp.died.isna().sum()),
                       "landmark_excluded": {int(c): int(len(pp) - lm[lm.cutoff_hour == c].patient_id.nunique()) for c in (12, 24)}}
t2 = []
for lab, x in [("All", pp), ("Group 1", pp[pp.died == 1]), ("Group 2", pp[pp.died == 0])]:
    t2.append({"Patients, n": f"{len(x):,}", "Age, years": q(x.age), "Male sex, %": pc(x.male == 1),
               "Coronary care unit, %": pc(x.icu == 1), "Cardiac surgery recovery unit, %": pc(x.icu == 2),
               "Medical ICU, %": pc(x.icu == 3), "Surgical ICU, %": pc(x.icu == 4),
               "Parameters measured in the first 12 h, n": q(x.breadth12),
               "Parameters measured in the first 24 h, n": q(x.breadth24),
               "Measurement events in the first 24 h, n": q(x.events24)})
t12 = pd.DataFrame(t2, index=["All", "Group 1", "Group 2"]).T.rename_axis("Characteristic").reset_index()
t12.insert(0, "Cohort", "CinC 2012 mortality cohort (Group 1 = in-hospital death, Group 2 = survived)")
table1 = pd.concat([t19, t12], ignore_index=True)
table1["Summary"] = np.where(table1.Characteristic.str.endswith("%"), "percentage",
                             np.where(table1.Characteristic.str.endswith("n (%)"), "count (percentage)",
                                      np.where(table1.Characteristic.str.contains("Patients, n|prediction times|Prediction times"), "count", "median (IQR)")))
table1.to_csv(OUT / "table1_cohort_characteristics.csv", index=False)

# representative CinC 2012 records (Fig. 5A): breadth at 24 h equal to group median
ex2 = []
for grp, lab in [(1, "In-hospital death"), (0, "Survived")]:
    x = pp[(pp.died == grp) & (pp.icu == 3)]
    target = int(pp[pp.died == grp].breadth24.median())
    pid = x[x.breadth24 == target].index.sort_values()[0]
    e = h2[(h2.patient_id == pid) & (h2.icu_hour < 24)][["patient_id", "icu_hour"] + ec]
    e = e.melt(id_vars=["patient_id", "icu_hour"], var_name="parameter", value_name="events")
    e = e[e.events.fillna(0) > 0]
    e["parameter"] = e.parameter.str.replace("event_count__", "", regex=False)
    e["group"] = lab
    ex2.append(e)
pd.concat(ex2).to_csv(OUT / "example_records_cinc2012.csv", index=False)

summary["runtime_s"] = round(time.time() - t0, 1)
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps(summary, indent=1, default=str))
