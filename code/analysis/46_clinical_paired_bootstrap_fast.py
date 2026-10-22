#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

P = Path('/data/workplace/wyx/measurement_intensity_ssl_20261015')
O = P / 'results/rootcause_stage0_2'
PAIRS = [
    ("clinical_V+I", "clinical_V"),
    ("clinical_V+M0+I", "clinical_V+M0"),
    ("clinical_V+I_equalized_noise", "clinical_V+M0+I"),
    ("full_V+I", "clinical_V+I"),
    ("full_V+M0+I", "clinical_V+M0+I"),
]
Y = None
INV = None
DRAWS = None
PRED = None

def metric(y, p, w):
    return np.array([
        average_precision_score(y, p, sample_weight=w),
        roc_auc_score(y, p, sample_weight=w),
        brier_score_loss(y, p, sample_weight=w),
    ], dtype=float)

def worker(pair):
    comp, base = pair
    pc, pb = PRED[comp], PRED[base]
    vals = np.empty((len(DRAWS), 3), dtype=float)
    n = int(INV.max()) + 1
    for i, s in enumerate(DRAWS):
        w = np.bincount(s, minlength=n)[INV]
        vals[i] = metric(Y, pc, w) - metric(Y, pb, w)
    point = metric(Y, pc, np.ones_like(Y, dtype=np.float64)) - metric(Y, pb, np.ones_like(Y, dtype=np.float64))
    return {
        "comparison": f"{comp}_minus_{base}", "component": comp, "reference": base,
        "delta_auprc": float(point[0]), "delta_auprc_lo": float(np.percentile(vals[:,0],2.5)), "delta_auprc_hi": float(np.percentile(vals[:,0],97.5)),
        "delta_auroc": float(point[1]), "delta_auroc_lo": float(np.percentile(vals[:,1],2.5)), "delta_auroc_hi": float(np.percentile(vals[:,1],97.5)),
        "delta_brier": float(point[2]), "delta_brier_lo": float(np.percentile(vals[:,2],2.5)), "delta_brier_hi": float(np.percentile(vals[:,2],97.5)),
        "bootstrap": int(len(DRAWS)), "paired_patient_bootstrap": True,
    }

def main(seed, workers):
    global Y, INV, DRAWS, PRED
    t0 = time.time()
    pred_path = O / f'p19_clinical_decomposition_pred_seed{seed}.parquet'
    d = pd.read_parquet(pred_path)
    Y = d.y_h6.astype(int).to_numpy()
    pids = d.patient_id.to_numpy()
    _, INV = np.unique(pids, return_inverse=True)
    rng = np.random.default_rng(seed * 7919 + 31)
    DRAWS = rng.integers(0, int(INV.max())+1, size=(2000, int(INV.max())+1), dtype=np.int32)
    all_conditions = sorted({x for pair in PAIRS for x in pair}); PRED = {c: d["p__"+c].to_numpy(dtype=float) for c in all_conditions}
    # Fork shares immutable arrays; each worker handles distinct contrasts.
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers) as pool:
        out = pool.map(worker, PAIRS)
    for row in out:
        row.update({"seed": seed, "n_test": int(len(d)), "n_test_patients": int(INV.max()+1), "positive_test": int(Y.sum())})
    out = pd.DataFrame(out)
    out = out[["seed","comparison","component","reference","n_test","n_test_patients","positive_test",
               "delta_auprc","delta_auprc_lo","delta_auprc_hi","delta_auroc","delta_auroc_lo","delta_auroc_hi",
               "delta_brier","delta_brier_lo","delta_brier_hi","bootstrap","paired_patient_bootstrap"]]
    out.to_csv(O / f'p19_clinical_decomposition_paired_seed{seed}.csv', index=False)
    (O / f'p19_clinical_decomposition_seed{seed}.json').write_text(json.dumps({
        "seed": seed, "runtime_seconds": time.time()-t0, "n_pair_rows": int(len(out)),
        "bootstrap": 2000, "workers": workers, "draw_seed": seed*7919+31,
        "paired_path": str(O / f'p19_clinical_decomposition_paired_seed{seed}.csv')
    }, indent=2))
    print(json.dumps({"seed":seed,"runtime_seconds":time.time()-t0,"rows":len(out)}), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    main(a.seed, a.workers)
