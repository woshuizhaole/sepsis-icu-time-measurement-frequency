#!/usr/bin/env python3
"""Recompute Stage 6 patient-cluster intervals with the locked 2,000 draws.

The model predictions are immutable inputs; this pass only updates the
confidence intervals in each seed result and then the ordinary aggregation
script can be rerun.
"""
from __future__ import annotations

import importlib.util
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

P = Path("/data/workplace/wyx/measurement_intensity_ssl_20261015")
R = P / "results"
SEEDS = [101, 202, 303, 404, 505]
B = 2000


def load_transfer_module():
    path = P / "code/analysis/27_stage6_transfer.py"
    spec = importlib.util.spec_from_file_location("stage6_transfer", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main():
    mod = load_transfer_module()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    seeds = [args.seed] if args.seed is not None else SEEDS
    for s in seeds:
        rp = R / f"stage6_transfer_results_seed{s}.csv"
        pp = R / f"stage6_transfer_predictions_seed{s}.parquet"
        d = pd.read_csv(rp)
        p = pd.read_parquet(pp)
        for i, row in d.iterrows():
            mask = p["direction"].eq(row["direction"]) & p["mode"].eq(row["mode"])
            if pd.isna(row["cutoff_hour"]):
                mask &= p.cutoff_hour.isna()
            else:
                mask &= p.cutoff_hour.eq(row["cutoff_hour"])
            q = p.loc[mask]
            ci = mod.boot_cluster(q.y.to_numpy(), q.pred.to_numpy(), q.patient_id.to_numpy(), B, 99173)
            for k, v in ci.items():
                d.loc[i, k] = v
            if row["direction"] == "p12_to_p19":
                z = q.groupby("patient_id", sort=False).agg(y=("y", "max"), pred=("pred", "max")).reset_index()
                pci = mod.boot_cluster(z.y.to_numpy(), z.pred.to_numpy(), z.patient_id.to_numpy(), B, 99173)
                for k, v in pci.items():
                    d.loc[i, f"patient_{k}"] = v
        d.to_csv(rp, index=False)
        print({"seed": s, "rows": len(d), "bootstrap_reps": B}, flush=True)


if __name__ == "__main__":
    main()
