"""Shared metrics for matched-time evaluation.

Hour-matched AUROC compares cases and controls only within the same ICU hour;
position-matched AUROC additionally matches on hours remaining in the record.
Stratum AUROCs are pooled by case-control pair counts (default), case counts,
or equal stratum weights. All functions accept optional per-row weights so the
same code serves point estimates and patient-cluster bootstrap replicates.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT = Path("/data/workplace/wyx/measurement_intensity_ssl_20261015")
PROCESSED = PROJECT / "data/processed"
RESULTS = PROJECT / "results"
OUT_ROOT = RESULTS / "matched_time"
SEEDS = [101, 202, 303, 404, 505]

DYN = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3", "FiO2", "pH",
       "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
       "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct",
       "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets"]
# Measurement families used throughout the charting-intensity analyses.
FAMILIES = {
    "Vital signs": ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"],
    "Blood gas and respiratory support": ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "Lactate"],
    "Chemistry": ["AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
                  "Glucose", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI"],
    "Haematology and coagulation": ["Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets"],
}
TIME_BINS = [0, 6, 12, 24, 48, 96, 10 ** 6]
TIME_LABELS = ["1-6", "7-12", "13-24", "25-48", "49-96", ">96"]


def partitions(dataset: str = "cinc2019") -> dict:
    return json.loads((PROCESSED / "partitions.json").read_text())[dataset]


def time_bin(hours) -> np.ndarray:
    """Map ICU hours to the six reporting strata (1-6, 7-12, 13-24, 25-48, 49-96, >96 h)."""
    idx = np.digitize(np.asarray(hours), TIME_BINS[1:-1], right=True)
    return np.asarray(TIME_LABELS, dtype=object)[idx]


class MatchedAUC:
    """Stratified AUROC with exact tie handling and optional row weights.

    keys: list of arrays defining strata (empty list -> pooled AUROC).
    """

    def __init__(self, y, p, keys=None):
        y = np.asarray(y, dtype=float)
        p = np.asarray(p, dtype=float)
        if not keys:
            keys = [np.zeros(len(y), dtype=np.int8)]
        keys = [np.asarray(k) for k in keys]
        order = np.lexsort((p,) + tuple(keys[::-1]))
        ys, ps = y[order], p[order]
        new_s = np.zeros(len(ys), dtype=bool)
        new_s[0] = True
        for k in keys:
            ks = k[order]
            new_s[1:] |= ks[1:] != ks[:-1]
        new_g = new_s.copy()
        new_g[1:] |= ps[1:] != ps[:-1]
        self.order = order
        self.ys = ys
        self.gs = np.flatnonzero(new_g)
        sid = np.cumsum(new_s) - 1
        self.gsid = sid[self.gs]
        self.first_g = np.flatnonzero(np.r_[True, self.gsid[1:] != self.gsid[:-1]])
        self.n_strata = int(sid[-1]) + 1

    def strata(self, w=None):
        ws = np.ones(len(self.ys)) if w is None else np.asarray(w, dtype=float)[self.order]
        gn = np.add.reduceat(ws * (1.0 - self.ys), self.gs)
        gp = np.add.reduceat(ws * self.ys, self.gs)
        cn = np.cumsum(gn) - gn
        below = cn - cn[self.first_g][self.gsid] + 0.5 * gn
        num = np.bincount(self.gsid, gp * below, minlength=self.n_strata)
        n1 = np.bincount(self.gsid, gp, minlength=self.n_strata)
        n0 = np.bincount(self.gsid, gn, minlength=self.n_strata)
        return num, n1, n0

    def auc(self, w=None, weighting: str = "pairs") -> float:
        num, n1, n0 = self.strata(w)
        ok = (n1 > 0) & (n0 > 0)
        if not ok.any():
            return float("nan")
        if weighting == "pairs":
            return float(num[ok].sum() / (n1[ok] * n0[ok]).sum())
        a = num[ok] / (n1[ok] * n0[ok])
        if weighting == "cases":
            return float((a * n1[ok]).sum() / n1[ok].sum())
        if weighting == "equal":
            return float(a.mean())
        raise ValueError(weighting)

    def coverage(self) -> dict:
        """Share of pooled case-control pairs and of cases that fall in evaluable strata."""
        num, n1, n0 = self.strata()
        ok = (n1 > 0) & (n0 > 0)
        total_pairs = n1.sum() * n0.sum()
        return {"pair_share": float((n1[ok] * n0[ok]).sum() / total_pairs),
                "case_share": float(n1[ok].sum() / n1.sum()),
                "evaluable_strata": int(ok.sum()), "strata": int(self.n_strata)}


def clock_share(p, hours) -> float:
    """Share of variance in predicted log-odds explained by exact ICU hour (eta squared)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    lg = np.log(p / (1 - p))
    df = pd.DataFrame({"l": lg, "h": np.asarray(hours)})
    gm = df.groupby("h").l.transform("mean")
    tot = float(((df.l - df.l.mean()) ** 2).sum())
    if tot <= 0:
        return float("nan")
    return float(1 - ((df.l - gm) ** 2).sum() / tot)


def lift_within_strata(y, p, hours) -> float:
    """Average precision divided by stratum event rate, pooled by stratum positives."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    b = time_bin(hours)
    num = den = 0.0
    for lab in TIME_LABELS:
        m = b == lab
        npos = y[m].sum()
        if npos == 0 or npos == m.sum():
            continue
        num += average_precision_score(y[m], p[m]) / y[m].mean() * npos
        den += npos
    return float(num / den)


def full_metrics(y, p, hours, hours_to_end) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    hm = MatchedAUC(y, p, [hours])
    pm = MatchedAUC(y, p, [hours, hours_to_end])
    em = MatchedAUC(y, p, [hours_to_end])
    return {
        "n_rows": int(len(y)), "n_pos": int(y.sum()), "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
        "auprc_lift": float(average_precision_score(y, p) / y.mean()),
        "auprc_lift_within_time_strata": lift_within_strata(y, p, hours),
        "brier": float(np.mean((p - y) ** 2)),
        "auroc_hour_matched": hm.auc(), "auroc_hour_matched_case_weighted": hm.auc(weighting="cases"),
        "auroc_hour_matched_equal_weighted": hm.auc(weighting="equal"),
        "auroc_position_matched": pm.auc(), "auroc_end_matched": em.auc(),
        "clock_dependence": clock_share(p, hours),
    }


def wilson(k, n, z=1.96):
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    ph = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    den = 1 + z ** 2 / n
    centre = (ph + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / den
    return centre - half, centre + half


def patient_draws(patient_ids, seed: int, n_boot: int = 2000):
    """Patient-cluster bootstrap draws shared with the sealed paired analyses (seed*7919+31)."""
    uniq, inv = np.unique(np.asarray(patient_ids), return_inverse=True)
    draws = np.random.default_rng(seed * 7919 + 31).integers(0, len(uniq), size=(n_boot, len(uniq)), dtype=np.int32)
    return uniq, inv, draws
