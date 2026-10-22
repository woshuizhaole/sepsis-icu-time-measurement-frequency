#!/usr/bin/env python3
"""Per-split 95% patient-cluster bootstrap intervals for every sealed model (Suppl. Table S2; full version of Table 2).

No model is trained. Predictions are the sealed full-test outputs, aligned to the clinical-decomposition prediction
order exactly as in 71_matched_time_discrimination.py (alignment asserted). For each split and model, seven metrics
are recomputed under 2,000 patient-cluster bootstrap draws (draws = seed * 7919 + 31, the same draws as the paired
analyses in 73_matched_time_paired_bootstrap.py):
  auroc, auprc, auprc_lift, auprc_lift_within_time_strata, auroc_hour_matched, auroc_position_matched, clock_dependence
Point estimates are recomputed with unit weights and asserted equal (< 1e-9) to results/matched_time/discrimination/
per_seed_metrics.csv. Hours remaining in the record define the position-matching strata only; it is never a model input.
Usage: 81_per_split_ci.py --seed 101 --chunk 0 --n-chunks 3 [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from matched_time import (OUT_ROOT, PROCESSED, RESULTS, TIME_LABELS, MatchedAUC, partitions, patient_draws,  # noqa: E402
                          time_bin)

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--chunk", type=int, default=0)
ap.add_argument("--n-chunks", type=int, default=1)
ap.add_argument("--n-boot", type=int, default=2000)
args = ap.parse_args()
s = args.seed
OUT = OUT_ROOT / "per_split_ci"
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
METRICS = ["auroc", "auprc", "auprc_lift", "auprc_lift_within_time_strata", "auroc_hour_matched", "auroc_position_matched", "clock_dependence"]
ABIN = [0, 6, 12, 24, 48, 72, 10 ** 6]


class WeightedAP:
    """Average precision with optional row weights and sklearn's handling of tied scores."""

    def __init__(self, y, p):
        y = np.asarray(y, float)
        p = np.asarray(p, float)
        order = np.argsort(-p, kind="stable")
        ps = p[order]
        self.order = order
        self.ys = y[order]
        self.gs = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]])

    def ap(self, w=None):
        ws = 1.0 if w is None else np.asarray(w, float)[self.order]
        gp = np.add.reduceat(ws * self.ys, self.gs)
        gn = np.add.reduceat(ws * (1.0 - self.ys), self.gs)
        tp, fp = np.cumsum(gp), np.cumsum(gn)
        if tp[-1] <= 0:
            return float("nan")
        return float(np.sum(np.diff(np.r_[0.0, tp / tp[-1]]) * (tp / np.maximum(tp + fp, 1e-300))))


class Metrics:
    def __init__(self, y, p, hr, hte):
        self.y = np.asarray(y, float)
        self.p = np.asarray(p, float)
        self.pooled = MatchedAUC(y, p, [])
        self.hour = MatchedAUC(y, p, [hr])
        self.pos = MatchedAUC(y, p, [hr, hte])
        self.ap_all = WeightedAP(y, p)
        b = time_bin(hr)
        self.strata = []
        for lab in TIME_LABELS:
            idx = np.flatnonzero(b == lab)
            if len(idx) == 0:
                continue
            self.strata.append((idx, WeightedAP(y[idx], p[idx])))
        self.hcode = np.unique(hr, return_inverse=True)[1]
        self.nh = int(self.hcode.max()) + 1
        pc = np.clip(self.p, 1e-6, 1 - 1e-6)
        self.lg = np.log(pc / (1 - pc))

    def all(self, w=None):
        y = self.y
        ws = np.ones(len(y)) if w is None else w
        prev = float((ws * y).sum() / ws.sum())
        ap = self.ap_all.ap(w)
        num = den = 0.0
        for idx, a in self.strata:
            wi = ws[idx]
            npos = float((wi * y[idx]).sum())
            n = float(wi.sum())
            if npos <= 0 or npos >= n:
                continue
            num += a.ap(None if w is None else wi) / (npos / n) * npos
            den += npos
        lg = self.lg
        sw = ws.sum()
        m = (ws * lg).sum() / sw
        gsw = np.bincount(self.hcode, ws, minlength=self.nh)
        gm = np.bincount(self.hcode, ws * lg, minlength=self.nh) / np.maximum(gsw, 1e-300)
        tot = (ws * (lg - m) ** 2).sum()
        clk = float(1 - (ws * (lg - gm[self.hcode]) ** 2).sum() / tot) if tot > 0 else float("nan")
        return dict(auroc=self.pooled.auc(w), auprc=ap, auprc_lift=ap / prev, auprc_lift_within_time_strata=num / den if den > 0 else float("nan"),
                    auroc_hour_matched=self.hour.auc(w), auroc_position_matched=self.pos.auc(w), clock_dependence=clk)


# ---------------------------------------------------------------- data and predictions (same sources as script 71)
hourly = pd.read_parquet(PROCESSED / "cinc2019_hourly/data.parquet", columns=["patient_id", "ICULOS"])
END = hourly.groupby("patient_id").ICULOS.max()
part = partitions("cinc2019")
win = pd.read_parquet(PROCESSED / "cinc2019_windows.parquet", columns=["patient_id", "anchor_hour", "y_h6"])
win["hours_to_end"] = win.patient_id.map(END) - win.anchor_hour
win["n_windows"] = win.groupby("patient_id").anchor_hour.transform("size")
cd = pd.read_parquet(RESULTS / f"rootcause_stage0_2/p19_clinical_decomposition_pred_seed{s}.parquet")
te = win[win.patient_id.map(part[str(s)]) == "test"].sort_values(["patient_id", "anchor_hour"]).reset_index(drop=True)
assert len(te) == len(cd) and (te.patient_id.to_numpy() == cd.patient_id.to_numpy()).all() and (te.anchor_hour.to_numpy() == cd.anchor_hour.to_numpy()).all()
y = cd.y_h6.to_numpy().astype(int)
assert (y == te.y_h6.to_numpy()).all()
hr = cd.anchor_hour.to_numpy()
hte = te.hours_to_end.to_numpy().astype(int)


def reference_predictions(seed: int) -> dict:
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


def load(path):
    x = pd.read_parquet(path)
    assert (x.patient_id.to_numpy() == cd.patient_id.to_numpy()).all() and (x.y.to_numpy().astype(int) == y).all(), path
    return x.p.to_numpy()


preds = dict(reference_predictions(s))
for c in ["admin_time_only", "anchor_hour_only", "clinical_V", "clinical_V+M0", "clinical_V+I", "clinical_V+M0+I",
          "clinical_V+I_equalized_noise", "full_V+M0", "full_V+I", "full_V+M0+I"]:
    preds[c] = cd["p__" + c].to_numpy()
for c in ["V", "M0", "Mh", "I", "V_I", "V_M0", "V_M0_I", "V_Mh", "V_Mh_I"]:
    preds["context+" + c] = load(RESULTS / f"stage3_lgb_v2_{c}_seed{s}_pred.parquet")
for name, pat in [("GRU-D", "stage3_grud_nooi_seed{s}_pred.parquet"), ("GRU-D + intensity", "stage3_grud_oi_seed{s}_pred.parquet"),
                  ("Neural CDE", "stage3_cde_nooi_seed{s}_pred.parquet"), ("Neural CDE + intensity", "stage3_cde_oi_seed{s}_pred.parquet"),
                  ("GRU + next-measurement auxiliary", "stage5_wbm_pred_seed{s}.parquet"), ("TimeCHEAT", "stage5_timecheat_pred_seed{s}.parquet")]:
    f = RESULTS / pat.format(s=s)
    if f.exists():
        preds[name] = load(f)
for tag, root in (("pretrain", "improvement_fair_ssl"), ("pretrain_with_continuation", "improvement_fair_ssl/terminal_controls_all")):
    for pth in sorted(glob.glob(str(RESULTS / root / f"fair_pred_*_seed{s}_f*.parquet"))):
        m = re.search(r"fair_pred_(.+)_seed\d+_f([\d.]+)\.parquet", pth)
        preds[f"{tag}:{m.group(1)}|{m.group(2)}"] = load(pth)

sealed = pd.read_csv(OUT_ROOT / "discrimination/per_seed_metrics.csv")
sealed = sealed[sealed.seed == s].set_index("model")
missing = sorted(set(sealed.index) - set(preds))
extra = sorted(set(preds) - set(sealed.index))
assert not missing and not extra, (missing, extra)
names = sorted(preds)
mine = names[args.chunk::args.n_chunks]
print(f"seed {s} chunk {args.chunk}/{args.n_chunks}: {len(mine)} of {len(names)} models, {len(y)} rows, {time.time() - t0:.0f}s", flush=True)

# ---------------------------------------------------------------- point estimates, asserted against the sealed table
objs, point, qc_diff = {}, {}, 0.0
for k in mine:
    objs[k] = Metrics(y, preds[k], hr, hte)
    point[k] = objs[k].all(None)
    for mname in METRICS:
        d = abs(point[k][mname] - sealed.loc[k, mname])
        qc_diff = max(qc_diff, float(d))
        assert d < 1e-9, (k, mname, point[k][mname], sealed.loc[k, mname])
print(f"point estimates match the sealed table (max diff {qc_diff:.2e}), {time.time() - t0:.0f}s", flush=True)

# ---------------------------------------------------------------- bootstrap
_, inv, draws = patient_draws(cd.patient_id.to_numpy(), s, args.n_boot)
n_pat = int(inv.max()) + 1
boot = {k: {m: np.empty(args.n_boot) for m in METRICS} for k in mine}
for i in range(args.n_boot):
    w = np.bincount(draws[i], minlength=n_pat)[inv].astype(float)
    for k in mine:
        r = objs[k].all(w)
        for m in METRICS:
            boot[k][m][i] = r[m]
    if i % 100 == 0:
        print(f"seed {s} chunk {args.chunk} draw {i} {time.time() - t0:.0f}s", flush=True)
rows = []
for k in mine:
    for m in METRICS:
        v = boot[k][m]
        rows.append(dict(seed=s, model=k, metric=m, estimate=point[k][m], ci_lo=float(np.nanpercentile(v, 2.5)),
                         ci_hi=float(np.nanpercentile(v, 97.5)), n_boot=args.n_boot, n_nan_draws=int(np.isnan(v).sum())))
pd.DataFrame(rows).to_csv(OUT / f"per_split_ci_seed{s}_chunk{args.chunk}.csv", index=False)
qc = dict(seed=s, chunk=args.chunk, n_chunks=args.n_chunks, n_models=len(mine), n_boot=args.n_boot, max_abs_diff_point_vs_sealed=qc_diff,
          runtime_s=round(time.time() - t0, 1), status="PASS")
(OUT / f"per_split_ci_qc_seed{s}_chunk{args.chunk}.json").write_text(json.dumps(qc, indent=2))
print(json.dumps(qc), flush=True)
