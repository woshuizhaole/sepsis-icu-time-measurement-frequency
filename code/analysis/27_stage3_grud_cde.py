#!/usr/bin/env python3
"""Stage 3 GRU-D and Neural-CDE baselines on the causal CinC2019 windows.

This script uses the frozen Stage-2 12-hour sequence cache.  The cache contains
train-split median-imputed values, current observation masks and the 4-hour
observation counts.  GRU-D reconstructs the standard decay input from the mask
and time since observation.  The CDE implementation uses an explicit Euler
discretisation of the controlled differential equation on the same 1-hour grid;
the observational-intensity (OI) condition appends the mask and 4-hour count
channels, while No-OI receives values and static covariates only.  Both models
use the same patient split, labels, hidden size, optimiser, batches and early
stopping protocol.

The code intentionally keeps the model-specific comparison separate from the
ordinary GRU and LightGBM results.  It never reads labels when constructing the
input path; labels are joined only for the supervised train/validation/test
indices.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn

P = Path("/data/workplace/wyx/measurement_intensity_ssl_20261015")
D = P / "data/processed"
R = P / "results"
L = P / "logs/analysis"
R.mkdir(parents=True, exist_ok=True)
L.mkdir(parents=True, exist_ok=True)

DYN = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
BASE = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
N_BASE = len(BASE)
N_DYN = len(DYN)
V0, V1 = N_BASE, N_BASE + N_DYN
M0, M1 = V1, V1 + N_DYN
I0, I1 = M1, M1 + N_DYN
N_CACHE = I1


def _condition_channels(condition: str) -> int:
    """Number of path channels (before the explicit normalized time channel)."""
    if condition == "nooi":
        return N_BASE + N_DYN
    if condition == "oi":
        return N_BASE + N_DYN + N_DYN + N_DYN
    raise KeyError(condition)


def _cache_path(seed: int) -> Path:
    p = D / f"p19_sequences_12h_seed{seed}.npy"
    if not p.exists():
        # Seed 101 is the original cache and is identical in schema; only use
        # it as an explicit fallback for a smoke test, never silently in a
        # final multi-seed run.
        raise FileNotFoundError(p)
    return p


def _load_split(seed: int) -> Dict[str, Dict[str, np.ndarray]]:
    """Join causal windows to the one-row-per-hour sequence cache index."""
    part = json.loads((D / "partitions.json").read_text())["cinc2019"][str(seed)]
    win = pd.read_parquet(
        D / "cinc2019_windows.parquet",
        columns=["patient_id", "anchor_hour", "y_h6"],
    )
    meta = pd.read_parquet(D / "p19_sequence_meta.parquet")
    z = win.merge(
        meta,
        left_on=["patient_id", "anchor_hour"],
        right_on=["patient_id", "ICULOS"],
        how="left",
        validate="one_to_one",
    )
    if z.row_idx.isna().any():
        raise RuntimeError("window-to-sequence join has missing row_idx")
    z["split"] = z.patient_id.map(part)
    if z.split.isna().any():
        raise RuntimeError("unassigned patient in fixed partition")
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for split in ("train", "validation", "test"):
        q = z[z.split == split].sort_values(["patient_id", "anchor_hour"])
        out[split] = {
            "idx": q.row_idx.to_numpy(dtype=np.int64),
            "y": q.y_h6.to_numpy(dtype=np.float32),
            "patient_id": q.patient_id.to_numpy(dtype=str),
            "anchor_hour": q.anchor_hour.to_numpy(dtype=np.int32),
        }
    return out


def _stats_path(seed: int) -> Path:
    return R / f"stage3_grud_cde_norm_seed{seed}.json"


def prepare_stats(seed: int, splits: Mapping[str, Mapping[str, np.ndarray]], seq: np.ndarray) -> Dict[str, np.ndarray]:
    """Estimate train-only normalization from observed train endpoint rows.

    Values are summarized only where the Stage-2 observation mask is one.  The
    static channels use all train endpoints.  This avoids using validation or
    test rows to set scale while keeping the calculation lightweight.
    """
    path = _stats_path(seed)
    if path.exists():
        raw = json.loads(path.read_text())
        return {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}
    idx = splits["train"]["idx"]
    n = len(idx)
    sums_v = np.zeros(N_DYN, dtype=np.float64)
    sums2_v = np.zeros(N_DYN, dtype=np.float64)
    count_v = np.zeros(N_DYN, dtype=np.float64)
    sums_b = np.zeros(N_BASE, dtype=np.float64)
    sums2_b = np.zeros(N_BASE, dtype=np.float64)
    # Iterate in chunks so a cache scan never allocates a second multi-GB array.
    for st in range(0, n, 8192):
        x = np.asarray(seq[idx[st: st + 8192], -1, :], dtype=np.float32)
        v = x[:, V0:V1].astype(np.float64)
        m = x[:, M0:M1] > 0.5
        sums_v += np.where(m, v, 0.0).sum(axis=0)
        sums2_v += np.where(m, v * v, 0.0).sum(axis=0)
        count_v += m.sum(axis=0)
        b = x[:, :N_BASE].astype(np.float64)
        sums_b += b.sum(axis=0)
        sums2_b += (b * b).sum(axis=0)
    vmean = sums_v / np.maximum(count_v, 1.0)
    vvar = np.maximum(sums2_v / np.maximum(count_v, 1.0) - vmean * vmean, 1e-6)
    bmean = sums_b / max(n, 1)
    bvar = np.maximum(sums2_b / max(n, 1) - bmean * bmean, 1e-6)
    out = {
        "value_mean": vmean.astype(np.float32),
        "value_std": np.sqrt(vvar).astype(np.float32),
        "base_mean": bmean.astype(np.float32),
        "base_std": np.sqrt(bvar).astype(np.float32),
        "observed_count": count_v.astype(np.float32),
    }
    path.write_text(json.dumps({k: v.tolist() for k, v in out.items()}, indent=2))
    return out


def _batch_iter(
    seq: np.ndarray,
    indices: np.ndarray,
    y: np.ndarray,
    pids: np.ndarray,
    batch: int,
    stats: Mapping[str, np.ndarray],
    shuffle: bool,
    seed: int,
    max_rows: int = 0,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, np.ndarray]]:
    """Yield normalized sequence batches from a read-only memmap."""
    if max_rows and len(indices) > max_rows:
        rng = np.random.default_rng(seed)
        take = rng.choice(len(indices), size=max_rows, replace=False)
        indices, y, pids = indices[take], y[take], pids[take]
    order = np.arange(len(indices))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    bmean = stats["base_mean"].reshape(1, 1, N_BASE)
    bstd = np.maximum(stats["base_std"], 1e-6).reshape(1, 1, N_BASE)
    vmean = stats["value_mean"].reshape(1, 1, N_DYN)
    vstd = np.maximum(stats["value_std"], 1e-6).reshape(1, 1, N_DYN)
    for st in range(0, len(order), batch):
        take = order[st: st + batch]
        # np.asarray forces a compact writable array for torch; only one batch
        # is materialized at a time.
        x = np.array(seq[indices[take]], dtype=np.float32, copy=True)
        x[:, :, :N_BASE] = (x[:, :, :N_BASE] - bmean) / bstd
        x[:, :, V0:V1] = (x[:, :, V0:V1] - vmean) / vstd
        x[:, :, I0:I1] = x[:, :, I0:I1] / 4.0
        yield torch.from_numpy(x), torch.from_numpy(y[take].astype(np.float32)), pids[take]


def _delta_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Regular-hour time since the last observation for every channel."""
    b, t, c = mask.shape
    delta = torch.zeros_like(mask)
    delta[:, 0] = 1.0 - mask[:, 0]
    for k in range(1, t):
        delta[:, k] = (1.0 - mask[:, k]) * (delta[:, k - 1] + 1.0)
    return delta


class GRUD(nn.Module):
    """GRU-D with learnable input decay and optional exogenous intensity."""

    def __init__(self, hidden: int = 64, use_oi: bool = False) -> None:
        super().__init__()
        self.use_oi = bool(use_oi)
        self.decay_w = nn.Parameter(torch.zeros(N_DYN))
        self.decay_b = nn.Parameter(torch.zeros(N_DYN))
        in_dim = N_BASE + N_DYN + N_DYN + N_DYN + (N_DYN if use_oi else 0)
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x[:, :, :N_BASE]
        value = x[:, :, V0:V1]
        mask = x[:, :, M0:M1]
        intensity = x[:, :, I0:I1]
        delta = _delta_from_mask(mask)
        last = torch.zeros_like(value[:, 0])
        chunks: List[torch.Tensor] = []
        # Input values are standardized, so the train-only population mean is
        # zero and is the GRU-D fallback for never-observed channels.
        for k in range(value.shape[1]):
            mk = mask[:, k]
            dk = delta[:, k]
            gamma = torch.exp(-torch.relu(self.decay_w * dk + self.decay_b))
            decayed = mk * value[:, k] + (1.0 - mk) * (gamma * last)
            last = mk * value[:, k] + (1.0 - mk) * last
            z = [base[:, k], decayed, mk, dk]
            if self.use_oi:
                z.append(intensity[:, k])
            chunks.append(torch.cat(z, dim=-1))
        h, _ = self.gru(torch.stack(chunks, dim=1))
        return self.head(h[:, -1]).squeeze(-1)


class CDEFunc(nn.Module):
    def __init__(self, hidden: int, channels: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.channels = channels
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.Tanh(),
            nn.Linear(hidden * 2, hidden * channels),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(z.shape[0], self.hidden, self.channels)


class NeuralCDE(nn.Module):
    """Neural CDE on the hourly piecewise-linear path.

    The explicit Euler update is the fixed-grid numerical solver
    ``z_{t+1}=z_t+f(z_t) dX_t``.  It is a controlled differential equation
    discretisation, with the same path and endpoint as a standard CDE solver,
    and avoids an unbounded adaptive solver cost on 1.5M ICU windows.
    """

    def __init__(self, channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.channels = channels
        self.hidden = hidden
        self.initial = nn.Linear(channels, hidden)
        self.func = CDEFunc(hidden, channels)
        self.head = nn.Linear(hidden, 1)

    def forward(self, path: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.initial(path[:, 0]))
        for k in range(path.shape[1] - 1):
            dx = path[:, k + 1] - path[:, k]
            z = z + torch.bmm(self.func(z), dx.unsqueeze(-1)).squeeze(-1)
        return self.head(z).squeeze(-1)


def _make_path(x: torch.Tensor, condition: str) -> torch.Tensor:
    base = x[:, :, :N_BASE]
    value = x[:, :, V0:V1]
    time_channel = torch.linspace(0.0, 1.0, x.shape[1], device=x.device, dtype=x.dtype)
    time_channel = time_channel.view(1, -1, 1).expand(x.shape[0], -1, -1)
    if condition == "nooi":
        return torch.cat([base, value, time_channel], dim=-1)
    if condition == "oi":
        return torch.cat([base, value, x[:, :, M0:M1], x[:, :, I0:I1], time_channel], dim=-1)
    raise KeyError(condition)


def _metric(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    if np.unique(y).size < 2:
        return {"auprc": float("nan"), "auroc": float("nan"), "brier": float(brier_score_loss(y, p))}
    return {
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def _bootstrap(y: np.ndarray, p: np.ndarray, pids: np.ndarray, b: int, seed: int) -> Dict[str, float]:
    """Patient-level paired bootstrap percentile interval."""
    if b <= 0:
        return {}
    uniq, inverse = np.unique(pids, return_inverse=True)
    rng = np.random.default_rng(seed)
    vals = np.empty((b, 3), dtype=np.float64)
    for st in range(0, b, 100):
        nrep = min(100, b - st)
        sampled = rng.integers(0, len(uniq), size=(nrep, len(uniq)))
        for j, s in enumerate(sampled):
            w = np.bincount(s, minlength=len(uniq))[inverse]
            vals[st + j, 0] = average_precision_score(y, p, sample_weight=w)
            vals[st + j, 1] = roc_auc_score(y, p, sample_weight=w)
            vals[st + j, 2] = brier_score_loss(y, p, sample_weight=w)
    out: Dict[str, float] = {}
    for j, name in enumerate(("auprc", "auroc", "brier")):
        out[f"{name}_lo"] = float(np.percentile(vals[:, j], 2.5))
        out[f"{name}_hi"] = float(np.percentile(vals[:, j], 97.5))
    return out


def _forward(model: nn.Module, architecture: str, condition: str, xb: torch.Tensor) -> torch.Tensor:
    if architecture == "grud":
        return model(xb)
    return model(_make_path(xb, condition))


def run(args: argparse.Namespace) -> Dict[str, object]:
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(max(1, min(2, args.threads // 4)))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    splits = _load_split(args.seed)
    seq = np.load(_cache_path(args.seed), mmap_mode="r")
    if seq.ndim != 3 or seq.shape[2] != N_CACHE:
        raise RuntimeError(f"unexpected sequence cache shape {seq.shape}")
    stats = prepare_stats(args.seed, splits, seq)
    condition = args.condition.lower()
    architecture = args.architecture.lower()
    if architecture == "grud":
        model: nn.Module = GRUD(hidden=args.hidden, use_oi=(condition == "oi"))
    elif architecture == "cde":
        model = NeuralCDE(_condition_channels(condition) + 1, hidden=args.hidden)
    else:
        raise ValueError(architecture)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best_score = -math.inf
    best_state = None
    start = time.time()
    history: List[Dict[str, float]] = []

    def epoch(split: str, epoch_no: int, train: bool) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        model.train(train)
        rec = splits[split]
        losses: List[float] = []
        allp: List[np.ndarray] = []
        ally: List[np.ndarray] = []
        allpid: List[np.ndarray] = []
        iterator = _batch_iter(
            seq, rec["idx"], rec["y"], rec["patient_id"], args.batch, stats,
            shuffle=train, seed=args.seed + epoch_no * 7919,
            max_rows=(args.max_train if split == "train" else args.max_val if split == "validation" else args.max_test),
        )
        with torch.set_grad_enabled(train):
            for xb, yb, pid in iterator:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                if train:
                    opt.zero_grad(set_to_none=True)
                logits = _forward(model, architecture, condition, xb)
                loss = loss_fn(logits, yb)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                losses.append(float(loss.detach().cpu()))
                allp.append(torch.sigmoid(logits.detach()).cpu().numpy())
                ally.append(yb.detach().cpu().numpy())
                allpid.append(pid)
        p = np.concatenate(allp) if allp else np.empty(0, dtype=np.float32)
        y = np.concatenate(ally) if ally else np.empty(0, dtype=np.float32)
        pid = np.concatenate(allpid) if allpid else np.empty(0, dtype=str)
        return float(np.mean(losses) if losses else float("nan")), p, y, pid

    for ep in range(args.epochs):
        tr_loss, _, _, _ = epoch("train", ep, True)
        with torch.no_grad():
            va_loss, vp, vy, _ = epoch("validation", ep, False)
        va_score = float(average_precision_score(vy, vp))
        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss, "val_auprc": va_score})
        print({"architecture": architecture, "condition": condition, **history[-1]}, flush=True)
        if va_score > best_score:
            best_score = va_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        _, tp, ty, tpid = epoch("test", args.epochs, False)
    met = _metric(ty, tp)
    ci = _bootstrap(ty, tp, tpid, args.bootstrap, args.seed + 100000)
    stem = f"stage3_{architecture}_{condition}_seed{args.seed}"
    pred = pd.DataFrame({
        "patient_id": tpid,
        "y": ty.astype(np.int8),
        "p": tp.astype(np.float32),
        "architecture": architecture,
        "condition": condition,
        "seed": args.seed,
    })
    pred.to_parquet(R / f"{stem}_pred.parquet", index=False)
    torch.save({"model": model.state_dict(), "architecture": architecture, "condition": condition,
                "seed": args.seed, "hidden": args.hidden, "history": history}, R / f"{stem}.pt")
    row: Dict[str, object] = {
        "architecture": architecture,
        "condition": condition,
        "seed": int(args.seed),
        "n_train": int(min(len(splits["train"]["idx"]), args.max_train) if args.max_train else len(splits["train"]["idx"])),
        "n_val": int(min(len(splits["validation"]["idx"]), args.max_val) if args.max_val else len(splits["validation"]["idx"])),
        "n_test": int(min(len(splits["test"]["idx"]), args.max_test) if args.max_test else len(splits["test"]["idx"])),
        "positive_test": int(ty.sum()),
        "val_best_auprc": float(best_score),
        "hidden": int(args.hidden),
        "epochs": int(args.epochs),
        "batch": int(args.batch),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "elapsed_sec": float(time.time() - start),
        **met,
        **ci,
    }
    pd.DataFrame([row]).to_csv(R / f"{stem}_metrics.csv", index=False)
    print({"done": stem, **row}, flush=True)
    return row


def merge_results() -> None:
    files = sorted(list(R.glob("stage3_grud_*_metrics.csv")) + list(R.glob("stage3_cde_*_metrics.csv")))
    if not files:
        raise SystemExit("no per-run metrics files")
    out = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out = out.sort_values(["architecture", "condition", "seed"]).reset_index(drop=True)
    out.to_csv(R / "stage3_grud_cde_results.csv", index=False)
    print(out.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--architecture", choices=["grud", "cde"])
    ap.add_argument("--condition", choices=["nooi", "oi"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge_results()
    else:
        if args.architecture is None or args.condition is None or args.seed is None:
            ap.error("--architecture, --condition and --seed are required unless --merge")
        run(args)


if __name__ == "__main__":
    main()
