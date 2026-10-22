#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

P = Path('/data/workplace/wyx/measurement_intensity_ssl_20261015')
D = P / 'data/processed'
O = P / 'results/rootcause_stage0_2'
O.mkdir(parents=True, exist_ok=True)
DYN = ["HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2","BaseExcess","HCO3","FiO2","pH","PaCO2","SaO2","AST","BUN","Alkalinephos","Calcium","Chloride","Creatinine","Bilirubin_direct","Glucose","Lactate","Magnesium","Phosphate","Potassium","Bilirubin_total","TroponinI","Hct","Hgb","PTT","WBC","Fibrinogen","Platelets"]
BASE = ["Age","Gender","Unit1","Unit2","HospAdmTime","anchor_hour"]
V = [f"v__{c}" for c in DYN]
M0 = [f"m0__{c}" for c in DYN]
I = [f"i_count4__{c}" for c in DYN] + ["i_breadth4","i_repeat4","i_valid_hours4"]
NOISE = [f"eqnoise_{j:02d}" for j in range(len(DYN))]
CONDITIONS = [
    ("admin_time_only", BASE),
    ("anchor_hour_only", ["anchor_hour"]),
    ("clinical_V", V),
    ("clinical_V+M0", V + M0),
    ("clinical_V+I", V + I),
    ("clinical_V+M0+I", V + M0 + I),
    ("clinical_V+I_equalized_noise", V + I + NOISE),
    ("full_V+M0", BASE + V + M0),
    ("full_V+I", BASE + V + I),
    ("full_V+M0+I", BASE + V + M0 + I),
]
PRIMARY_PAIRS = [
    ("clinical_V+I", "clinical_V"),
    ("clinical_V+M0+I", "clinical_V+M0"),
    ("clinical_V+I_equalized_noise", "clinical_V+M0+I"),
    ("full_V+I", "clinical_V+I"),
    ("full_V+M0+I", "clinical_V+M0+I"),
]

def metric(y, p, w=None):
    return np.array([
        average_precision_score(y, p, sample_weight=w),
        roc_auc_score(y, p, sample_weight=w),
        brier_score_loss(y, p, sample_weight=w),
    ], dtype=float)

def bootstrap_values(y, p, inv, draws):
    n = int(inv.max()) + 1
    out = np.empty((len(draws), 3), dtype=float)
    for i, s in enumerate(draws):
        w = np.bincount(s, minlength=n)[inv]
        out[i] = metric(y, p, w)
    return out

def main(seed):
    t0 = time.time()
    win = pd.read_parquet(D / "cinc2019_windows.parquet", columns=["patient_id","anchor_hour","y_h6"])
    feat = pd.read_parquet(D / "stage3_p19_features_v2.parquet")
    full = win.merge(feat.rename(columns={"ICULOS":"anchor_hour"}), on=["patient_id","anchor_hour"], how="left", validate="one_to_one")
    part = json.loads((D / "partitions.json").read_text())["cinc2019"][str(seed)]
    full["split"] = full.patient_id.map(part)
    # Train-only/label-blind nuisance columns make V+I and V+M0+I have the same feature count.
    rng = np.random.default_rng(seed * 1009 + 17)
    noise = np.empty((len(full), len(NOISE)), dtype=np.float32)
    for split_name in ["train","validation","test"]:
        idx = np.flatnonzero(full["split"].to_numpy() == split_name)
        noise[idx] = rng.normal(0.0, 1.0, size=(len(idx), len(NOISE))).astype(np.float32)
    full = pd.concat([full.reset_index(drop=True), pd.DataFrame(noise, columns=NOISE)], axis=1)
    tr = full[full.split == "train"]
    va = full[full.split == "validation"]
    te = full[full.split == "test"]
    ytr = tr.y_h6.astype(int).to_numpy()
    yva = va.y_h6.astype(int).to_numpy()
    yte = te.y_h6.astype(int).to_numpy()
    pids = te.patient_id.to_numpy()
    uniq, inv = np.unique(pids, return_inverse=True)
    draws = np.random.default_rng(seed * 7919 + 31).integers(0, len(uniq), size=(2000, len(uniq)), dtype=np.int32)
    preds = {}
    rows = []
    for condition, cols in CONDITIONS:
        med = tr[cols].replace([np.inf,-np.inf],np.nan).median(numeric_only=True)
        def clean(q):
            return q[cols].replace([np.inf,-np.inf], np.nan).fillna(med).fillna(0)
        clf = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=seed, n_jobs=1, verbosity=-1
        )
        clf.fit(clean(tr), ytr, eval_set=[(clean(va), yva)],
                callbacks=[early_stopping(30, verbose=False)])
        p = clf.predict_proba(clean(te))[:,1]
        preds[condition] = p
        point = metric(yte, p)
        boots = bootstrap_values(yte, p, inv, draws)
        rows.append({
            "seed": seed, "condition": condition, "n_features": len(cols),
            "n_train": len(tr), "n_validation": len(va), "n_test": len(te),
            "n_test_patients": len(uniq), "positive_test": int(yte.sum()),
            "best_iteration": int(clf.best_iteration_ or 0),
            "auprc": float(point[0]), "auroc": float(point[1]), "brier": float(point[2]),
            "auprc_lo": float(np.percentile(boots[:,0], 2.5)),
            "auprc_hi": float(np.percentile(boots[:,0], 97.5)),
            "auroc_lo": float(np.percentile(boots[:,1], 2.5)),
            "auroc_hi": float(np.percentile(boots[:,1], 97.5)),
            "brier_lo": float(np.percentile(boots[:,2], 2.5)),
            "brier_hi": float(np.percentile(boots[:,2], 97.5)),
        })
        print(seed, condition, rows[-1]["auprc"], flush=True)
    metric_df = pd.DataFrame(rows)
    pred_df = pd.DataFrame({"patient_id": pids, "anchor_hour": te.anchor_hour.to_numpy(), "y_h6": yte})
    for k, p in preds.items():
        pred_df["p__" + k] = p
    metric_df.to_csv(O / f"p19_clinical_decomposition_seed{seed}.csv", index=False)
    pred_df.to_parquet(O / f"p19_clinical_decomposition_pred_seed{seed}.parquet", index=False)
    pair_rows = []
    for comp, base in PRIMARY_PAIRS:
        pc, pb = preds[comp], preds[base]
        vals = np.empty((len(draws), 3), dtype=float)
        for i, s in enumerate(draws):
            w = np.bincount(s, minlength=len(uniq))[inv]
            vals[i] = metric(yte, pc, w) - metric(yte, pb, w)
        point = metric(yte, pc) - metric(yte, pb)
        pair_rows.append({
            "seed": seed, "comparison": f"{comp}_minus_{base}",
            "component": comp, "reference": base,
            "n_test": len(te), "n_test_patients": len(uniq), "positive_test": int(yte.sum()),
            "delta_auprc": float(point[0]), "delta_auprc_lo": float(np.percentile(vals[:,0],2.5)), "delta_auprc_hi": float(np.percentile(vals[:,0],97.5)),
            "delta_auroc": float(point[1]), "delta_auroc_lo": float(np.percentile(vals[:,1],2.5)), "delta_auroc_hi": float(np.percentile(vals[:,1],97.5)),
            "delta_brier": float(point[2]), "delta_brier_lo": float(np.percentile(vals[:,2],2.5)), "delta_brier_hi": float(np.percentile(vals[:,2],97.5)),
            "bootstrap": 2000, "paired_patient_bootstrap": True,
        })
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(O / f"p19_clinical_decomposition_paired_seed{seed}.csv", index=False)
    (O / f"p19_clinical_decomposition_seed{seed}.json").write_text(json.dumps({
        "seed": seed, "runtime_seconds": time.time()-t0, "conditions": [x[0] for x in CONDITIONS],
        "primary_pairs": [f"{a}_minus_{b}" for a,b in PRIMARY_PAIRS],
        "n_rows": int(len(metric_df)), "n_pair_rows": int(len(pair_df)),
        "bootstrap": 2000, "noise_equalized_feature_count": len(NOISE),
        "outputs": [str(O / f"p19_clinical_decomposition_seed{seed}.csv"),
                    str(O / f"p19_clinical_decomposition_paired_seed{seed}.csv"),
                    str(O / f"p19_clinical_decomposition_pred_seed{seed}.parquet")]
    }, indent=2))
    print(json.dumps({"seed":seed,"runtime_seconds":time.time()-t0,"rows":len(metric_df),"pair_rows":len(pair_df)}), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    main(ap.parse_args().seed)
