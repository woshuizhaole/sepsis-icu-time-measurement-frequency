#!/usr/bin/env python3
"""Matched-time discrimination of reference constructions and sealed test predictions (Fig. 2, 3, 6; Table 2).

No model is trained. Every prediction file is a sealed full-test output; rows are
aligned to the clinical-decomposition prediction order (patient_id, anchor_hour)
and alignment is asserted. Hours remaining in the record are used only as an
evaluation stratum and in two reference lookups that are labelled as using
information after the prediction time.
"""
from __future__ import annotations

import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from matched_time import (OUT_ROOT, PROCESSED, RESULTS, SEEDS, TIME_LABELS, MatchedAUC,  # noqa: E402
                          full_metrics, partitions, time_bin)

OUT = OUT_ROOT / "discrimination"
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()

hourly = pd.read_parquet(PROCESSED / "cinc2019_hourly/data.parquet", columns=["patient_id", "ICULOS", "source_set"])
END = hourly.groupby("patient_id").ICULOS.max()
SRC = hourly.groupby("patient_id").source_set.first().map(lambda s: "A" if s.endswith("A") else "B")
part = partitions("cinc2019")
win = pd.read_parquet(PROCESSED / "cinc2019_windows.parquet", columns=["patient_id", "anchor_hour", "y_h6"])
win["hours_to_end"] = win.patient_id.map(END) - win.anchor_hour
win["n_windows"] = win.groupby("patient_id").anchor_hour.transform("size")

CLIN = ["admin_time_only", "anchor_hour_only", "clinical_V", "clinical_V+M0", "clinical_V+I", "clinical_V+M0+I",
        "clinical_V+I_equalized_noise", "full_V+M0", "full_V+I", "full_V+M0+I"]
CTX = ["V", "M0", "Mh", "I", "V_I", "V_M0", "V_M0_I", "V_Mh", "V_Mh_I"]
SEQ = [("GRU-D", "stage3_grud_nooi_seed{s}_pred.parquet"), ("GRU-D + intensity", "stage3_grud_oi_seed{s}_pred.parquet"),
       ("Neural CDE", "stage3_cde_nooi_seed{s}_pred.parquet"), ("Neural CDE + intensity", "stage3_cde_oi_seed{s}_pred.parquet"),
       ("GRU + next-measurement auxiliary", "stage5_wbm_pred_seed{s}.parquet"), ("TimeCHEAT", "stage5_timecheat_pred_seed{s}.parquet")]
ABIN = [0, 6, 12, 24, 48, 72, 10 ** 6]


def reference_predictions(seed: int, te: pd.DataFrame) -> dict:
    sp = win.patient_id.map(part[str(seed)])
    tr = win[sp == "train"]
    g = tr.y_h6.mean()
    out = {"ICU-hour prevalence lookup": te.anchor_hour.map(tr.groupby("anchor_hour").y_h6.mean()).fillna(g).to_numpy()}
    qs = np.quantile(tr.n_windows, [.2, .4, .6, .8])
    ab = lambda v: np.digitize(v, ABIN[1:-1], right=True)  # noqa: E731
    tab = pd.Series(tr.y_h6.to_numpy(), index=pd.MultiIndex.from_arrays([ab(tr.anchor_hour.to_numpy()), np.digitize(tr.n_windows.to_numpy(), qs)])).groupby(level=[0, 1]).mean()
    keys = zip(ab(te.anchor_hour.to_numpy()), np.digitize(te.n_windows.to_numpy(), qs))
    out["ICU-hour x record-length lookup"] = np.array([tab.get(k, g) for k in keys])
    out["Hours-to-record-end lookup"] = te.hours_to_end.map(tr.groupby("hours_to_end").y_h6.mean()).fillna(g).to_numpy()
    return out


def budget_alerts(p, hours, budget, rng, hour_specific):
    jit = p + rng.uniform(0, 1e-12, len(p))
    if not hour_specific:
        return jit > np.quantile(jit, 1 - budget)
    r = pd.Series(jit).groupby(np.asarray(hours)).rank(pct=True).to_numpy()
    return r > 1 - budget


def calibration_slope(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lg = np.log(p / (1 - p))
    b0, b1 = 0.0, 1.0
    for _ in range(50):
        z = b0 + b1 * lg
        q = 1 / (1 + np.exp(-z))
        W = q * (1 - q)
        g0, g1 = (y - q).sum(), ((y - q) * lg).sum()
        H = np.array([[W.sum(), (W * lg).sum()], [(W * lg).sum(), (W * lg * lg).sum()]])
        st = np.linalg.solve(H, [g0, g1])
        b0, b1 = b0 + st[0], b1 + st[1]
        if np.abs(st).max() < 1e-9:
            break
    return float(b1)


rows, strata_rows, system_rows, tail_rows, budget_rows, alloc_rows, curve_rows, roc_rows, cov_rows, cal_rows, cal_bins = ([] for _ in range(11))
examples = None
for s in SEEDS:
    cd = pd.read_parquet(RESULTS / f"rootcause_stage0_2/p19_clinical_decomposition_pred_seed{s}.parquet")
    te = win[win.patient_id.map(part[str(s)]) == "test"].sort_values(["patient_id", "anchor_hour"]).reset_index(drop=True)
    assert len(te) == len(cd) and (te.patient_id.to_numpy() == cd.patient_id.to_numpy()).all() and (te.anchor_hour.to_numpy() == cd.anchor_hour.to_numpy()).all()
    y = cd.y_h6.to_numpy().astype(int)
    assert (y == te.y_h6.to_numpy()).all()
    hr = cd.anchor_hour.to_numpy()
    hte = te.hours_to_end.to_numpy().astype(int)
    sysv = cd.patient_id.map(SRC).to_numpy()
    preds = {}
    fam = {}
    for k, v in reference_predictions(s, te).items():
        preds[k], fam[k] = v, "reference"
    for c in CLIN:
        preds[c], fam[c] = cd["p__" + c].to_numpy(), "gbm_clinical_or_context"
    for c in CTX:
        x = pd.read_parquet(RESULTS / f"stage3_lgb_v2_{c}_seed{s}_pred.parquet")
        assert (x.patient_id.to_numpy() == cd.patient_id.to_numpy()).all() and (x.anchor_hour.to_numpy() == hr).all() and (x.y.to_numpy() == y).all()
        preds["context+" + c], fam["context+" + c] = x.p.to_numpy(), "gbm_with_context"
    for name, pat in SEQ:
        f = RESULTS / pat.format(s=s)
        if not f.exists():
            continue
        x = pd.read_parquet(f)
        assert (x.patient_id.to_numpy() == cd.patient_id.to_numpy()).all() and (x.y.to_numpy().astype(int) == y).all()
        preds[name], fam[name] = x.p.to_numpy(), "sequence"
    for pth in sorted(glob.glob(str(RESULTS / f"improvement_fair_ssl/fair_pred_*_seed{s}_f*.parquet"))):
        m = re.search(r"fair_pred_(.+)_seed\d+_f([\d.]+)\.parquet", pth)
        x = pd.read_parquet(pth)
        assert (x.patient_id.to_numpy() == cd.patient_id.to_numpy()).all() and (x.y.to_numpy().astype(int) == y).all()
        k = f"pretrain:{m.group(1)}|{m.group(2)}"
        preds[k], fam[k] = x.p.to_numpy(), "pretrained_gru"
    for pth in sorted(glob.glob(str(RESULTS / f"improvement_fair_ssl/terminal_controls_all/fair_pred_*_seed{s}_f*.parquet"))):
        m = re.search(r"fair_pred_(.+)_seed\d+_f([\d.]+)\.parquet", pth)
        x = pd.read_parquet(pth)
        assert (x.patient_id.to_numpy() == cd.patient_id.to_numpy()).all() and (x.y.to_numpy().astype(int) == y).all()
        k = f"pretrain_with_continuation:{m.group(1)}|{m.group(2)}"
        preds[k], fam[k] = x.p.to_numpy(), "pretrained_gru_with_record_continuation"

    for k, p in preds.items():
        r = full_metrics(y, p, hr, hte)
        r.update(seed=s, model=k, family=fam[k])
        rows.append(r)

    tb = time_bin(hr)
    focus = CLIN + ["GRU-D", "GRU-D + intensity", "Neural CDE", "Neural CDE + intensity"]
    for k in [k for k in focus if k in preds]:
        p = preds[k]
        for lab in TIME_LABELS:
            mk = tb == lab
            strata_rows.append(dict(seed=s, model=k, stratum=lab, n_rows=int(mk.sum()), n_pos=int(y[mk].sum()),
                                    prevalence=float(y[mk].mean()), auroc=float(roc_auc_score(y[mk], p[mk])),
                                    auprc=float(average_precision_score(y[mk], p[mk]))))
        for sv in ["A", "B"]:
            mk = sysv == sv
            system_rows.append(dict(seed=s, model=k, hospital_system=sv, n_rows=int(mk.sum()), n_pos=int(y[mk].sum()),
                                    auroc=float(roc_auc_score(y[mk], p[mk])), auroc_hour_matched=MatchedAUC(y[mk], p[mk], [hr[mk]]).auc()))
        mk = hte > 8
        tail_rows.append(dict(seed=s, model=k, rows_excluded=int((~mk).sum()), positives_excluded=int(y[~mk].sum()),
                              auroc_all=float(roc_auc_score(y, p)), auroc_excl_last8h=float(roc_auc_score(y[mk], p[mk])),
                              auroc_hour_matched_all=MatchedAUC(y, p, [hr]).auc(),
                              auroc_hour_matched_excl_last8h=MatchedAUC(y[mk], p[mk], [hr[mk]]).auc()))

    rng = np.random.default_rng(s)
    pid = cd.patient_id.to_numpy()
    sep_pat = np.unique(pid[y == 1])
    for k in ["admin_time_only", "clinical_V", "clinical_V+M0", "clinical_V+I", "clinical_V+M0+I", "full_V+M0+I"]:
        p = preds[k]
        for b in (0.05, 0.10, 0.20):
            for mode in ("pooled", "hour_specific"):
                a = budget_alerts(p, hr, b, rng, mode == "hour_specific")
                hit_pat = np.unique(pid[a & (y == 1)])
                budget_rows.append(dict(seed=s, model=k, budget=b, budget_mode=mode, alert_rate=float(a.mean()),
                                        event_time_sensitivity=float(a[y == 1].mean()), ppv=float(y[a].mean()),
                                        patient_sensitivity=float(len(hit_pat) / len(sep_pat))))
                if b == 0.10 and mode == "pooled":
                    for lab in TIME_LABELS:
                        mk = tb == lab
                        alloc_rows.append(dict(seed=s, model=k, stratum=lab, row_share=float(mk.mean()), event_share=float(y[mk].sum() / y.sum()),
                                               alert_share=float(a[mk].sum() / a.sum()), alert_rate_in_stratum=float(a[mk].mean())))
    for k in ["admin_time_only", "clinical_V", "clinical_V+I", "full_V+M0+I"]:
        q = pd.DataFrame({"h": hr, "p": preds[k]})
        q = q[q.h <= 120].groupby("h").p
        curve_rows.append(pd.DataFrame({"seed": s, "series": k, "icu_hour": q.median().index, "median": q.median().to_numpy(),
                                        "q25": q.quantile(.25).to_numpy(), "q75": q.quantile(.75).to_numpy()}))
    q = pd.DataFrame({"h": hr, "y": y})
    q = q[q.h <= 120].groupby("h").y.mean()
    curve_rows.append(pd.DataFrame({"seed": s, "series": "observed_event_rate", "icu_hour": q.index, "median": q.to_numpy(), "q25": np.nan, "q75": np.nan}))
    grid = np.linspace(0, 1, 101)
    mk = tb == "25-48"
    for k in ["admin_time_only", "clinical_V", "clinical_V+I", "full_V+M0+I"]:
        fpr, tpr, _ = roc_curve(y[mk], preds[k][mk])
        roc_rows.append(pd.DataFrame({"seed": s, "model": k, "fpr": grid, "tpr": np.interp(grid, fpr, tpr),
                                      "auroc": roc_auc_score(y[mk], preds[k][mk])}))
    for k in ["clinical_V", "clinical_V+I", "full_V+M0+I"]:
        cv = MatchedAUC(y, preds[k], [hr, hte]).coverage()
        ch = MatchedAUC(y, preds[k], [hr]).coverage()
        cov_rows.append(dict(seed=s, model=k, position_pair_share=cv["pair_share"], position_case_share=cv["case_share"],
                             position_strata=cv["strata"], position_evaluable_strata=cv["evaluable_strata"],
                             hour_pair_share=ch["pair_share"], hour_case_share=ch["case_share"]))
    for k in [k for k in preds if k.startswith("pretrain:")]:
        p = np.asarray(preds[k], dtype=float)
        init, frac = k.split(":")[1].split("|")
        cal_rows.append(dict(seed=s, init=init, fraction=float(frac), prevalence=float(y.mean()), mean_pred=float(p.mean()),
                             mean_pred_over_prevalence=float(p.mean() / y.mean()), brier=float(np.mean((p - y) ** 2)),
                             calibration_slope=calibration_slope(y, p)))
        edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
        bi = np.clip(np.searchsorted(edges[1:-1], p, side="right"), 0, len(edges) - 2)
        for j in np.unique(bi):
            mm = bi == j
            cal_bins.append(dict(seed=s, init=init, fraction=float(frac), bin=int(j), n=int(mm.sum()), mean_pred=float(p[mm].mean()), observed=float(y[mm].mean())))
    if s == 101:
        pick = []
        rs = np.random.default_rng(2026)
        for hh in (8, 16, 24, 36, 60, 80):
            for yy in (1, 0):
                idx = np.flatnonzero((hr == hh) & (y == yy))
                if len(idx):
                    pick.extend(rs.choice(idx, size=min(5, len(idx)), replace=False))
        examples = pd.DataFrame({"anchor_hour": hr[pick], "y": y[pick], "p_context_fields": preds["admin_time_only"][pick],
                                 "p_physiological_values": preds["clinical_V"][pick]})
    print(f"seed {s}: {len(preds)} prediction sets, {time.time() - t0:.0f}s", flush=True)

per = pd.DataFrame(rows)
per.to_csv(OUT / "per_seed_metrics.csv", index=False)
pd.DataFrame(strata_rows).to_csv(OUT / "stratum_metrics.csv", index=False)
pd.DataFrame(system_rows).to_csv(OUT / "hospital_system_metrics.csv", index=False)
pd.DataFrame(tail_rows).to_csv(OUT / "record_end_exclusion.csv", index=False)
pd.DataFrame(budget_rows).to_csv(OUT / "alert_budgets.csv", index=False)
pd.DataFrame(alloc_rows).to_csv(OUT / "pooled_alert_allocation.csv", index=False)
pd.concat(curve_rows).to_csv(OUT / "risk_by_icu_hour.csv", index=False)
pd.concat(roc_rows).to_csv(OUT / "roc_25_48h.csv", index=False)
pd.DataFrame(cov_rows).to_csv(OUT / "matching_coverage.csv", index=False)
pd.DataFrame(cal_rows).to_csv(OUT / "pretraining_calibration.csv", index=False)
pd.DataFrame(cal_bins).to_csv(OUT / "pretraining_reliability_bins.csv", index=False)
examples.to_csv(OUT / "matched_comparison_examples_seed101.csv", index=False)

# ---------------------------------------------------------------- Table 2
CONTEXT = "age, sex, ICU unit, hospital-to-ICU interval, ICU hour"
T2 = [
    ("Reference constructions", "ICU hour only (training-set event rate by hour)", "ICU hour", "1", "No", "ICU-hour prevalence lookup"),
    ("Reference constructions", "ICU hour and total record length", "ICU hour, record length", "2", "Yes", "ICU-hour x record-length lookup"),
    ("Reference constructions", "Hours remaining in the record", "Hours to end of record", "1", "Yes", "Hours-to-record-end lookup"),
    ("Gradient boosting", "Context fields", CONTEXT, "6", "No", "admin_time_only"),
    ("Gradient boosting", "Physiological values", "34 values", "34", "No", "clinical_V"),
    ("Gradient boosting", "Values + measurement indicators", "34 values, 34 indicators", "68", "No", "clinical_V+M0"),
    ("Gradient boosting", "Values + charting intensity", "34 values, 37 charting summaries", "71", "No", "clinical_V+I"),
    ("Gradient boosting", "Values + indicators + charting intensity", "34 values, 34 indicators, 37 charting summaries", "105", "No", "clinical_V+M0+I"),
    ("Gradient boosting", "Values + charting intensity + 34 random columns", "Capacity-matched control", "105", "No", "clinical_V+I_equalized_noise"),
    ("Gradient boosting", "Context + values", f"{CONTEXT}; 34 values", "40", "No", "context+V"),
    ("Gradient boosting", "Context + measurement indicators", f"{CONTEXT}; 34 current-hour indicators", "40", "No", "context+M0"),
    ("Gradient boosting", "Context + charting intensity", f"{CONTEXT}; 37 charting summaries", "43", "No", "context+I"),
    ("Gradient boosting", "Context + 12-h measurement history", f"{CONTEXT}; 12 x 34 hourly indicators", "414", "No", "context+Mh"),
    ("Gradient boosting", "Context + values + indicators", f"{CONTEXT}; 34 values; 34 indicators", "74", "No", "full_V+M0"),
    ("Gradient boosting", "Context + values + charting intensity", f"{CONTEXT}; 34 values; 37 charting summaries", "77", "No", "full_V+I"),
    ("Gradient boosting", "Context + values + indicators + charting intensity", f"{CONTEXT}; 34 values; 34 indicators; 37 charting summaries", "111", "No", "full_V+M0+I"),
    ("Gradient boosting", "Context + values + 12-h measurement history", f"{CONTEXT}; 34 values; 12 x 34 hourly indicators", "448", "No", "context+V_Mh"),
    ("Gradient boosting", "Context + values + history + charting intensity", f"{CONTEXT}; 34 values; 12 x 34 hourly indicators; 37 charting summaries", "485", "No", "context+V_Mh_I"),
    ("Sequence models", "GRU-D", "12-h sequence of values, indicators and time since last measurement; context", "", "No", "GRU-D"),
    ("Sequence models", "GRU-D + charting intensity", "GRU-D inputs plus 37 charting summaries per hour", "", "No", "GRU-D + intensity"),
    ("Sequence models", "Neural CDE", "12-h interpolated value path; context", "", "No", "Neural CDE"),
    ("Sequence models", "Neural CDE + charting intensity", "Neural CDE inputs plus 37 charting summaries per hour", "", "No", "Neural CDE + intensity"),
    ("Sequence models", "GRU trained jointly with next-hour measurement prediction", "12-h sequence of values and indicators; context", "", "No", "GRU + next-measurement auxiliary"),
]
for frac, lab in (("0.10", "10% labels"), ("1.00", "all labels")):
    for init, name in (("random", "No pretraining"), ("nextmask4", "Next 4-h measurement indicators"),
                       ("nextintensity4", "Next 4-h measurement counts"), ("valueforecast4", "Next 4-h values")):
        T2.append((f"Pretrained GRU, {lab}", name, "12-h sequence of values, current and 4-h indicators; context", "108 per hour", "No", f"pretrain:{init}|{frac}"))


def fmt(v, d=3):
    v = pd.Series(v).dropna()
    return "" if v.empty else f"{v.median():.{d}f} ({v.min():.{d}f}-{v.max():.{d}f})"


t2 = []
for fam_lab, model, inputs, nf, future, key in T2:
    x = per[per.model == key]
    assert len(x) == 5, key
    t2.append({"Model family": fam_lab, "Model": model, "Inputs": inputs, "Input features, n": nf,
               "Uses information after the prediction time": future, "Splits, n": int(x.seed.nunique()),
               "AUROC": fmt(x.auroc), "AUPRC": fmt(x.auprc), "AUPRC / event rate": fmt(x.auprc_lift, 2),
               "AUPRC / event rate within ICU-hour strata": fmt(x.auprc_lift_within_time_strata, 2),
               "Hour-matched AUROC": fmt(x.auroc_hour_matched), "Position-matched AUROC": fmt(x.auroc_position_matched),
               "Clock dependence of predictions": fmt(x.clock_dependence, 2)})
pd.DataFrame(t2).to_csv(OUT / "table2_clock_matched_discrimination.csv", index=False)

# ---------------------------------------------------------------- summary of key contrasts (point estimates per split)
pv = per.pivot_table(index="seed", columns="model", values=["auroc", "auroc_hour_matched", "auroc_position_matched",
                                                            "auroc_hour_matched_case_weighted", "auroc_hour_matched_equal_weighted"])
summary = {"runtime_s": round(time.time() - t0, 1), "contrasts": {}}
for a, b in [("clinical_V+I", "clinical_V"), ("clinical_V+M0+I", "clinical_V+M0"), ("full_V+M0+I", "clinical_V+M0+I"),
             ("clinical_V+I_equalized_noise", "clinical_V+M0+I"), ("context+V_I", "context+V"), ("context+V_Mh_I", "context+V_Mh"),
             ("GRU-D + intensity", "GRU-D"), ("Neural CDE + intensity", "Neural CDE")]:
    d = {}
    for met in ["auroc", "auroc_hour_matched", "auroc_position_matched", "auroc_hour_matched_case_weighted", "auroc_hour_matched_equal_weighted"]:
        v = (pv[(met, a)] - pv[(met, b)]).dropna()
        d[met] = {"median": float(v.median()), "positive_splits": int((v > 0).sum()), "per_split": [round(float(z), 4) for z in v]}
    summary["contrasts"][f"{a} - {b}"] = d
nonref = per[(per.family != "reference") & (~per.model.str.startswith("pretrain")) & (per.model != "anchor_hour_only")]
med = nonref.groupby("model")[["clock_dependence", "auroc", "auroc_hour_matched"]].median()
med["inflation"] = med.auroc - med.auroc_hour_matched
from scipy.stats import spearmanr  # noqa: E402
summary["clock_dependence_vs_inflation"] = {"n_models": int(len(med)), "spearman": float(spearmanr(med.clock_dependence, med.inflation)[0])}
allm = per[(per.family != "reference") & (per.model != "anchor_hour_only") & (~per.model.str.startswith("pretrain_with"))].groupby("model")[["clock_dependence", "auroc", "auroc_hour_matched"]].median()
summary["clock_dependence_vs_inflation_including_pretrained"] = {"n_models": int(len(allm)), "spearman": float(spearmanr(allm.clock_dependence, allm.auroc - allm.auroc_hour_matched)[0])}
med.to_csv(OUT / "clock_dependence_vs_inflation.csv")
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=1))
