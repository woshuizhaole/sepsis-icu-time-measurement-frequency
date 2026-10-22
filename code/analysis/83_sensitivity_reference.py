#!/usr/bin/env python3
"""Reference model for the complete-window sensitivity analyses (Supplementary Table 4).

The sensitivity runs in 16_stage6_sensitivity.py (first 4 ICU hours excluded; lactate,
white-cell count, creatinine and total bilirubin removed) use prediction times whose next
4-h window is recorded. This script fits the same LightGBM model on the same rows with no
exclusion and the full feature set, so the sensitivity results have a like-for-like
reference. It reuses the functions of 16_stage6_sensitivity.py unchanged (an unknown mode
name applies no exclusion) and writes to results/sensitivity_reference/ only.

Usage: 83_sensitivity_reference.py --seed 101 [--bootstrap 2000]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import pandas as pd

P = Path("/data/workplace/wyx/measurement_intensity_ssl_20261015")
OUT = P / "results/sensitivity_reference"


def load_module():
    spec = importlib.util.spec_from_file_location("stage6_sensitivity", P / "code/analysis/16_stage6_sensitivity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mod = load_module()
    ns = argparse.Namespace(modes="reference_complete_window", bootstrap=args.bootstrap, seed=args.seed)
    rows = mod.p19(args.seed, ns)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"reference_seed{args.seed}.csv", index=False)
    qc = {"seed": args.seed, "rows": len(df), "estimable": bool(df.estimable.all()),
          "n_features": int(df.n_features.iloc[0]), "n_test": int(df.n_test.iloc[0]),
          "positive_test": int(df.positive_test.iloc[0]), "bootstrap": args.bootstrap,
          "runtime_s": round(time.time() - t0, 1)}
    (OUT / f"reference_seed{args.seed}_qc.json").write_text(json.dumps(qc, indent=2))
    print(json.dumps(qc), flush=True)


if __name__ == "__main__":
    main()
