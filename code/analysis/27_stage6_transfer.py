#!/usr/bin/env python3
"""Stage 6 cross-dataset transfer audit.

This script uses a deliberately narrow common-channel interface.  It trains a
small GRU self-supervised encoder on one corpus without reading target labels,
then evaluates frozen-head, fine-tuned, and target-only random baselines on the
other corpus.  All split maps are patient-level and all target windows are
constructed from observations strictly before the target landmark/hour.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


P = Path("/data/workplace/wyx/measurement_intensity_ssl_20261015")
D = P / "data/processed"
R = P / "results"
L = P / "logs/analysis"
R.mkdir(parents=True, exist_ok=True)
L.mkdir(parents=True, exist_ok=True)

P19_DYN = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]

# Strictly matched clinical channels.  O2Sat and SaO2 are deliberately not
# duplicated: P12 exposes SaO2, which is mapped to the P19 O2Sat slot.  Channels
# without a defensible semantic counterpart are zero-padded in the common
# interface and never used by either source encoder.
P12_MAP = {
    "HR": "HR",
    "O2Sat": "SaO2",
    "Temp": "Temp",
    "SBP": "SysABP",
    "MAP": "MAP",
    "DBP": "DiasABP",
    "Resp": "RespRate",
    "HCO3": "HCO3",
    "FiO2": "FiO2",
    "pH": "pH",
    "PaCO2": "PaCO2",
    "AST": "AST",
    "BUN": "BUN",
    "Creatinine": "Creatinine",
    "Glucose": "Glucose",
    "Lactate": "Lactate",
    "Magnesium": "Mg",
    "Potassium": "K",
    "TroponinI": "TroponinI",
    "Hct": "HCT",
    "WBC": "WBC",
    "Platelets": "Platelets",
}
COMMON = list(P12_MAP)
HIDDEN = 64
SEQ_LEN = 12
TARGET_W = 4
INPUT_D = 3 + 3 * len(COMMON)  # Age, Gender, ICU hour + value/mask/4h count.


class Encoder(nn.Module):
    def __init__(self, d: int = INPUT_D, h: int = HIDDEN):
        super().__init__()
        self.gru = nn.GRU(d, h, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gru(x)[0][:, -1]


class IntensityHead(nn.Module):
    def __init__(self, h: int = HIDDEN, k: int = len(COMMON)):
        super().__init__()
        self.out = nn.Linear(h, k * (TARGET_W + 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.out(z).view(-1, len(COMMON), TARGET_W + 1)


class SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return torch.from_numpy(self.x[i]), torch.from_numpy(self.y[i])


def _source_indices() -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Indices in the 108-dim existing P19 sequence cache."""
    st = [0, 1, 5]
    v = np.asarray([6 + P19_DYN.index(c) for c in COMMON], dtype=np.int64)
    m = np.asarray([6 + len(P19_DYN) + P19_DYN.index(c) for c in COMMON], dtype=np.int64)
    c4 = np.asarray([6 + 2 * len(P19_DYN) + P19_DYN.index(c) for c in COMMON], dtype=np.int64)
    return st, v, m, c4


def p19_common(x: np.ndarray) -> np.ndarray:
    st, v, m, c4 = _source_indices()
    idx = np.concatenate([np.asarray(st), v, m, c4])
    return np.asarray(x[..., idx], dtype=np.float32)


def _finite_fill(x: np.ndarray, med: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        x = x.copy()
        bad = ~np.isfinite(x)
        x[bad] = np.take(med, np.where(bad)[-1])
    return x


def fit_medians(x: np.ndarray) -> np.ndarray:
    # Median over batch and time; static/mask/count columns are finite already.
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.asarray(x, dtype=np.float32), axis=(0, 1))
    med[~np.isfinite(med)] = 0.0
    return med.astype(np.float32)


def boot_cluster(y: np.ndarray, p: np.ndarray, pid: np.ndarray, B: int, seed: int) -> Dict[str, float]:
    """Patient-cluster bootstrap intervals; rows within a patient share weight."""
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    pid = np.asarray(pid)
    u, inv = np.unique(pid, return_inverse=True)
    n = len(u)
    if n == 0:
        return {f"{m}_{q}": np.nan for m in ("auprc", "auroc", "brier") for q in ("lo", "hi")}
    rng = np.random.default_rng(seed)
    vals = []
    for st in range(0, B, 100):
        ns = min(100, B - st)
        samp = rng.integers(0, n, size=(ns, n))
        for s in samp:
            w = np.bincount(s, minlength=n)[inv]
            try:
                ap = average_precision_score(y, p, sample_weight=w)
                au = roc_auc_score(y, p, sample_weight=w)
                br = brier_score_loss(y, p, sample_weight=w)
            except ValueError:
                ap = au = br = np.nan
            vals.append((ap, au, br))
    a = np.asarray(vals, dtype=float)
    out = {}
    for j, m in enumerate(("auprc", "auroc", "brier")):
        out[f"{m}_lo"] = float(np.nanpercentile(a[:, j], 2.5))
        out[f"{m}_hi"] = float(np.nanpercentile(a[:, j], 97.5))
    return out


def _patient_select(w: pd.DataFrame, max_rows: int, seed: int, cap: int = 8) -> pd.DataFrame:
    """Select a bounded number of anchors per patient before global cap."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in w.groupby("patient_id", sort=False):
        n = min(cap, len(g))
        if len(g) <= n:
            parts.append(g)
        else:
            ii = np.linspace(0, len(g) - 1, n).round().astype(int)
            parts.append(g.iloc[np.unique(ii)])
    z = pd.concat(parts, ignore_index=True) if parts else w.iloc[:0].copy()
    if len(z) > max_rows:
        # Sample patients, then retain at most cap anchors, to avoid long-stay
        # patients dominating the source/target head.
        keep = rng.choice(len(z), size=max_rows, replace=False)
        z = z.iloc[np.sort(keep)].reset_index(drop=True)
    return z


def load_p19(seed: int):
    # Each seed cache imputes values with that seed's train-only medians.  The
    # unqualified cache is retained only as a last-resort legacy fallback.
    seq_path = D / f"p19_sequences_12h_seed{seed}.npy"
    if not seq_path.exists():
        seq_path = D / "p19_sequences_12h.npy"
    seq = np.load(seq_path, mmap_mode="r")
    meta = pd.read_parquet(D / "p19_sequence_meta.parquet", columns=["patient_id", "ICULOS", "row_idx"])
    win = pd.read_parquet(D / "cinc2019_windows.parquet")
    parts = json.loads((D / "partitions.json").read_text())["cinc2019"][str(seed)]
    # A row-index PID vector makes next-window target extraction leak-proof.
    pid_by_row = meta.sort_values("row_idx").patient_id.to_numpy()
    hour_by_row = meta.sort_values("row_idx").ICULOS.to_numpy()
    return seq, meta, win, parts, pid_by_row, hour_by_row, seq_path


def p19_pretrain(seed: int, seq, meta: pd.DataFrame, parts: Dict[str, str], max_n: int, rng: np.random.Generator):
    train = meta[meta.patient_id.map(parts).eq("train") & (meta.ICULOS <= 20)].copy()
    # Need four future rows; contiguous row_idx within patient is checked here.
    rid = train.row_idx.to_numpy(dtype=np.int64)
    pid = train.patient_id.to_numpy()
    good = np.ones(len(rid), dtype=bool)
    _, _, m, _ = _source_indices()
    m_full = m
    if len(rid):
        lim = len(seq)
        for j in range(1, TARGET_W + 1):
            q = rid + j
            good &= q < lim
        # Explicit PID equality avoids crossing a patient boundary.
        pid_arr = meta.sort_values("row_idx").patient_id.to_numpy()
        for j in range(1, TARGET_W + 1):
            q = rid + j
            ok = q < len(pid_arr)
            eq = np.zeros(len(rid), dtype=bool)
            eq[ok] = pid_arr[q[ok]] == pid[ok]
            good &= eq
    rid = rid[good]
    if len(rid) > max_n:
        rid = rng.choice(rid, size=max_n, replace=False)
    x = p19_common(np.asarray(seq[rid]))
    # The existing cache stores m0 at positions 40..73; use the same slots.
    full_m = np.asarray(seq[rid[:, None] + np.arange(1, TARGET_W + 1)[None, :], -1, :], dtype=np.float32)
    m_idx = 6 + len(P19_DYN) + np.asarray([P19_DYN.index(c) for c in COMMON])
    counts = np.rint(full_m[..., m_idx].sum(axis=1)).astype(np.int64)
    return x, counts, {"n_pool": int(len(train)), "n_used": int(len(rid))}


def build_p12_cache() -> Path:
    out = D / "stage6_transfer_p12_common.parquet"
    if out.exists() and out.stat().st_size > 1000:
        return out
    cols = ["patient_id", "icu_hour", "Age", "Gender"]
    for src in P12_MAP.values():
        cols.extend([f"value__{src}", f"event_count__{src}"])
    # Keep only existing columns; all mapped channels are present in the audited
    # 2012 hourly table, but this makes the cache robust to a schema refresh.
    raw = pd.read_parquet(D / "cinc2012_hourly/data.parquet")
    cols = [c for c in cols if c in raw.columns]
    raw = raw[cols].copy()
    raw["icu_hour"] = pd.to_numeric(raw["icu_hour"], errors="coerce").fillna(0)
    raw["Age"] = pd.to_numeric(raw["Age"], errors="coerce")
    raw["Gender"] = pd.to_numeric(raw["Gender"], errors="coerce")
    # Build the common representation in one pass over patients so m4 never
    # includes future observations.
    rows = []
    for pid, g in raw.groupby("patient_id", sort=False):
        g = g.sort_values("icu_hour")
        z = np.zeros((len(g), INPUT_D), dtype=np.float32)
        z[:, 0] = g["Age"].iloc[0] if len(g) else np.nan
        z[:, 1] = g["Gender"].iloc[0] if len(g) else np.nan
        z[:, 2] = g["icu_hour"].to_numpy(dtype=np.float32) + 1.0
        for j, p19 in enumerate(COMMON):
            p12 = P12_MAP[p19]
            vc = f"value__{p12}"; mc = f"event_count__{p12}"
            if vc in g:
                val = pd.to_numeric(g[vc], errors="coerce").to_numpy(dtype=np.float32)
                val[val == -1] = np.nan
            else:
                val = np.full(len(g), np.nan, dtype=np.float32)
            if mc in g:
                m0 = pd.to_numeric(g[mc], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
            else:
                m0 = np.zeros(len(g), dtype=np.float32)
            z[:, 3 + j] = val
            z[:, 3 + len(COMMON) + j] = (m0 > 0).astype(np.float32)
            z[:, 3 + 2 * len(COMMON) + j] = pd.Series(m0).rolling(4, min_periods=1).sum().to_numpy(dtype=np.float32)
        q = pd.DataFrame(z, columns=[f"x{i}" for i in range(INPUT_D)])
        q.insert(0, "icu_hour", g["icu_hour"].to_numpy(dtype=np.int16))
        q.insert(0, "patient_id", pid)
        rows.append(q)
    pd.concat(rows, ignore_index=True).to_parquet(out, index=False, compression="zstd")
    return out


def load_p12():
    cache = build_p12_cache()
    tab = pd.read_parquet(cache)
    xcols = [f"x{i}" for i in range(INPUT_D)]
    groups = {pid: g.sort_values("icu_hour")[xcols].to_numpy(dtype=np.float32) for pid, g in tab.groupby("patient_id", sort=False)}
    hours = {pid: g.sort_values("icu_hour")["icu_hour"].to_numpy(dtype=np.int16) for pid, g in tab.groupby("patient_id", sort=False)}
    lm = pd.read_parquet(D / "cinc2012_landmarks.parquet")
    parts_all = json.loads((D / "partitions.json").read_text())["cinc2012"]
    return groups, hours, lm, parts_all, tab


def seq_until(arr: np.ndarray, hrs: np.ndarray, end_hour: int, length: int = SEQ_LEN) -> np.ndarray | None:
    # Strictly before end_hour.  End hour is a cutoff (P12) or anchor+1.
    ii = np.where(hrs < end_hour)[0]
    if len(ii) == 0:
        return None
    i = int(ii[-1]); lo = max(0, i - length + 1)
    x = arr[lo:i + 1]
    if len(x) < length:
        x = np.concatenate([np.repeat(x[:1], length - len(x), axis=0), x], axis=0)
    return x.astype(np.float32)


def p12_pretrain(groups, hours, parts, seed: int, max_n: int, rng: np.random.Generator):
    train_ids = [p for p, s in parts.items() if s == "train" and p in groups]
    items = []
    for pid in train_ids:
        h = hours[pid]
        # Label-blind, fixed first-24-hour source pool.
        for t in h.tolist():
            if t <= 20 and (h >= t + TARGET_W).any():
                items.append((pid, int(t)))
    if len(items) > max_n:
        items = [items[i] for i in rng.choice(len(items), size=max_n, replace=False)]
    x = []; y = []
    m0s = slice(3 + len(COMMON), 3 + 2 * len(COMMON))
    for pid, t in items:
        q = seq_until(groups[pid], hours[pid], t + 1)
        if q is None: continue
        future = []
        for j in range(1, TARGET_W + 1):
            z = groups[pid][np.where(hours[pid] == t + j)[0][0]]
            future.append(z[m0s])
        x.append(q); y.append(np.rint(np.sum(np.asarray(future), axis=0)).astype(np.int64))
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64), {"n_pool": len(items), "n_used": len(x)}


def train_ssl(x: np.ndarray, y: np.ndarray, seed: int, threads: int, epochs: int = 2) -> Tuple[dict, Dict[str, float]]:
    torch.manual_seed(seed)
    med = fit_medians(x)
    x = _finite_fill(x, med)
    enc = Encoder(); head = IntensityHead()
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    dl = DataLoader(SeqDataset(x, y), batch_size=1024, shuffle=True, num_workers=0)
    enc.train(); head.train(); losses = []
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            z = enc(xb.float()); logits = head(z)
            loss = ce(logits.reshape(-1, TARGET_W + 1), yb.long().reshape(-1))
            loss.backward(); opt.step(); losses.append(float(loss.detach()))
    return enc.state_dict(), {"pretrain_loss": float(np.mean(losses)) if losses else np.nan, "median_finite": int(np.isfinite(med).sum())}


def make_p12_target(groups, hours, lm, parts: Dict[str, str], cutoff: int, seed: int):
    z = lm[(lm.cutoff_hour == cutoff) & lm.landmark_available.astype(bool)].copy()
    z["split"] = z.patient_id.map(parts)
    def make(split: str):
        rows = []; ys = []; pids = []
        for r in z[z.split == split].itertuples(index=False):
            if r.patient_id not in groups: continue
            q = seq_until(groups[r.patient_id], hours[r.patient_id], int(cutoff))
            if q is None: continue
            rows.append(q); ys.append(int(r.mortality_label)); pids.append(r.patient_id)
        return np.asarray(rows, dtype=np.float32), np.asarray(ys, dtype=np.int8), np.asarray(pids)
    return make("train"), make("validation"), make("test")


def make_p19_target(seq, meta, win, parts: Dict[str, str], seed: int, max_train: int, max_test: int):
    w = win[win.next_window_available.astype(bool)].copy()
    w["split"] = w.patient_id.map(parts)
    w = w[w.split.isin(["train", "validation", "test"])]
    tr = _patient_select(w[w.split == "train"], max_train, seed)
    va = _patient_select(w[w.split == "validation"], min(max_test, 80000), seed + 1)
    te = _patient_select(w[w.split == "test"], max_test, seed + 2)
    m = meta.set_index(["patient_id", "ICULOS"])["row_idx"]
    def get(z):
        rid = np.asarray([m.get((r.patient_id, r.anchor_hour), -1) for r in z.itertuples(index=False)], dtype=np.int64)
        good = rid >= 0
        return p19_common(np.asarray(seq[rid[good]])), z.loc[good, "y_h6"].to_numpy(dtype=np.int8), z.loc[good, "patient_id"].to_numpy()
    return get(tr), get(va), get(te)


def _metrics(y, p, pid, prefix=""):
    y = np.asarray(y, dtype=np.int8); p = np.asarray(p, dtype=float); pid = np.asarray(pid)
    out = {f"{prefix}n": int(len(y)), f"{prefix}n_patients": int(len(np.unique(pid))), f"{prefix}positive": int(y.sum())}
    out[f"{prefix}auprc"] = float(average_precision_score(y, p))
    out[f"{prefix}auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan
    out[f"{prefix}brier"] = float(brier_score_loss(y, p))
    out.update({f"{prefix}{k}": v for k, v in boot_cluster(y, p, pid, 1000, 99173).items()})
    return out


def train_target(source_state, xtr, ytr, pidtr, xva, yva, xte, yte, pidte, mode: str, seed: int, source_med: np.ndarray, target_med: np.ndarray):
    # Frozen/fine-tuned runs use source-train medians; target-only uses target
    # train medians.  Both are fitted without target labels.
    med = source_med if mode != "random" else target_med
    xtr = _finite_fill(xtr, med); xva = _finite_fill(xva, med); xte = _finite_fill(xte, med)
    enc = Encoder(); head = nn.Linear(HIDDEN, 1)
    if source_state is not None:
        enc.load_state_dict(source_state)
    if mode == "frozen":
        for p in enc.parameters(): p.requires_grad_(False)
        params = list(head.parameters())
    else:
        params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=2e-3 if mode == "frozen" else 1e-3)
    lossf = nn.BCEWithLogitsLoss()
    best = -np.inf; best_state = None
    dl = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr.astype(np.float32))), batch_size=1024, shuffle=True)
    for _ in range(3):
        enc.train(); head.train()
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True); loss = lossf(head(enc(xb.float())).squeeze(-1), yb); loss.backward(); opt.step()
        enc.eval(); head.eval()
        with torch.no_grad():
            pv = torch.sigmoid(head(enc(torch.from_numpy(xva).float())).squeeze(-1)).numpy()
        sc = average_precision_score(yva, pv) if len(np.unique(yva)) > 1 else -np.inf
        if sc > best:
            best = sc; best_state = {"enc": {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}, "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
    enc.load_state_dict(best_state["enc"]); head.load_state_dict(best_state["head"]); enc.eval(); head.eval()
    with torch.no_grad():
        p = torch.sigmoid(head(enc(torch.from_numpy(xte).float())).squeeze(-1)).numpy()
    return p, float(best)


def run_seed(seed: int, args):
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(seed)
    groups, hours, lm, p12parts, p12tab = load_p12()
    p19seq, p19meta, p19win, p19parts, pid_by_row, hour_by_row, p19_seq_path = load_p19(seed)
    rows = []; pred_rows = []

    # Source P19 -> target P12.
    xpre, ypre, preinfo = p19_pretrain(seed, p19seq, p19meta, p19parts, args.max_pretrain, rng)
    p19_state, pre_stats = train_ssl(xpre, ypre, seed, args.threads, args.ssl_epochs)
    p19_med = fit_medians(xpre)
    for cutoff in (12, 24):
        (xtr, ytr, pidtr), (xva, yva, pidva), (xte, yte, pidte) = make_p12_target(groups, hours, lm, p12parts[str(seed)], cutoff, seed)
        tmed = fit_medians(xtr)
        for mode in ("frozen", "finetune", "random"):
            state = p19_state if mode != "random" else None
            p, vbest = train_target(state, xtr, ytr, pidtr, xva, yva, xte, yte, pidte, mode, seed, p19_med, tmed)
            met = _metrics(yte, p, pidte)
            row = {"seed": seed, "direction": "p19_to_p12", "source": "cinc2019", "target": "cinc2012", "task": "landmark_mortality", "cutoff_hour": cutoff, "mode": mode, "n_common_dynamic": len(COMMON), "estimand": "P12 mortality conditional on observed history before cutoff", "val_best_auprc": vbest, **met, "source_pretrain_n": preinfo["n_used"], **pre_stats}
            rows.append(row)
            pred_rows.append(pd.DataFrame({"seed": seed, "direction": "p19_to_p12", "cutoff_hour": cutoff, "mode": mode, "patient_id": pidte, "y": yte, "pred": p}))

    # Source P12 -> target P19.
    xpre2, ypre2, preinfo2 = p12_pretrain(groups, hours, p12parts[str(seed)], seed, args.max_pretrain, rng)
    p12_state, pre_stats2 = train_ssl(xpre2, ypre2, seed + 17, args.threads, args.ssl_epochs)
    p12_med = fit_medians(xpre2)
    (xtr, ytr, pidtr), (xva, yva, pidva), (xte, yte, pidte) = make_p19_target(p19seq, p19meta, p19win, p19parts, seed, args.max_train_p19, args.max_test_p19)
    tmed = fit_medians(xtr)
    for mode in ("frozen", "finetune", "random"):
        state = p12_state if mode != "random" else None
        p, vbest = train_target(state, xtr, ytr, pidtr, xva, yva, xte, yte, pidte, mode, seed + 31, p12_med, tmed)
        met = _metrics(yte, p, pidte)
        # A second patient-level summary is useful because the hourly target
        # contains repeated rows per patient.
        zz = pd.DataFrame({"pid": pidte, "y": yte, "p": p}).groupby("pid", sort=False).agg(y=("y", "max"), p=("p", "max")).reset_index()
        pmet = _metrics(zz.y.to_numpy(), zz.p.to_numpy(), zz.pid.to_numpy(), prefix="patient_")
        row = {"seed": seed, "direction": "p12_to_p19", "source": "cinc2012", "target": "cinc2019", "task": "pre_first_positive_h6", "cutoff_hour": np.nan, "mode": mode, "n_common_dynamic": len(COMMON), "estimand": "P19 hourly pre-first-positive h6 warning; complete next-window target rows", "val_best_auprc": vbest, **met, **pmet, "source_pretrain_n": preinfo2["n_used"], **pre_stats2}
        rows.append(row)
        pred_rows.append(pd.DataFrame({"seed": seed, "direction": "p12_to_p19", "cutoff_hour": np.nan, "mode": mode, "patient_id": pidte, "y": yte, "pred": p}))

    pd.DataFrame(rows).to_csv(R / f"stage6_transfer_results_seed{seed}.csv", index=False)
    pd.concat(pred_rows, ignore_index=True).to_parquet(R / f"stage6_transfer_predictions_seed{seed}.parquet", index=False)
    (R / f"stage6_transfer_source_seed{seed}.json").write_text(json.dumps({"seed": seed, "common_channels": COMMON, "p12_map": P12_MAP, "input_dim": INPUT_D, "p19_sequence_file": str(p19_seq_path), "p19_sequence_seed_specific": p19_seq_path.name == f"p19_sequences_12h_seed{seed}.npy", "p19_pretrain": preinfo, "p12_pretrain": preinfo2}, indent=2))
    print(json.dumps({"seed": seed, "rows": len(rows), "p19_pretrain": preinfo, "p12_pretrain": preinfo2}, sort_keys=True), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-pretrain", type=int, default=20000)
    ap.add_argument("--max-train-p19", type=int, default=100000)
    ap.add_argument("--max-test-p19", type=int, default=80000)
    ap.add_argument("--ssl-epochs", type=int, default=2)
    args = ap.parse_args()
    run_seed(args.seed, args)


if __name__ == "__main__":
    main()
