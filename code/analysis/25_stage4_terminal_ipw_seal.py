from pathlib import Path
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd

P = Path("/data/workplace/wyx/measurement_intensity_ssl_20261015")
R = P / "results"
S = R / "stats"
S.mkdir(parents=True, exist_ok=True)

seeds = [101, 202, 303, 404, 505]
inits = ["random", "nextmask", "nextintensity", "valueforecast"]
fracs = [0.01, 0.05, 0.10, 0.25, 1.00]
keys = [(s, i, f) for s in seeds for i in inits for f in fracs]
point_metrics = [
    "auprc_complete_case", "auroc_complete_case", "brier_complete_case",
    "auprc_marginal_ipw", "auroc_marginal_ipw", "brier_marginal_ipw",
]
required = {
    "seed", "init", "fraction", "n_test", "n_available", "availability_rate",
    "positive_all", "positive_available", "auprc_all", "auroc_all", "brier_all",
    *point_metrics,
}

paths = sorted(R.glob("stage4_terminal_ipw_eval_seed*_*.csv"))
parts = []
file_errors = []
for path in paths:
    try:
        d = pd.read_csv(path)
    except Exception as exc:
        file_errors.append(f"{path.name}: read_error={exc}")
        continue
    if len(d) != 1:
        file_errors.append(f"{path.name}: rows={len(d)}")
    missing = sorted(required.difference(d.columns))
    if missing:
        file_errors.append(f"{path.name}: missing={missing}")
    parts.append(d)
if not parts:
    raise RuntimeError("No terminal/IPW result files found")
out = pd.concat(parts, ignore_index=True)
key_tuples = [(int(r.seed), str(r.init), float(r.fraction)) for r in out.itertuples()]
expected = set(keys)
got = set(key_tuples)
missing_keys = sorted(expected - got)
extra_keys = sorted(got - expected)
if missing_keys or extra_keys or file_errors:
    raise RuntimeError({"missing": missing_keys, "extra": extra_keys, "file_errors": file_errors})
if len(out) != len(keys):
    raise RuntimeError(f"rows={len(out)} expected={len(keys)}")

out = out.sort_values(["seed", "init", "fraction"]).reset_index(drop=True)
for col in ["seed", "n_test", "n_available", "positive_all", "positive_available"]:
    out[col] = out[col].astype(int)
out["fraction"] = out["fraction"].astype(float)
finite_counts = {m: int(np.isfinite(out[m].to_numpy(dtype=float)).sum()) for m in point_metrics}
all_finite = all(v == len(out) for v in finite_counts.values())
# The all-test metrics are already sealed in Stage 4 full; verify exact agreement.
full = pd.read_csv(R / "stage4_ssl_full_results.csv")
full_key = full.set_index(["seed", "init", "fraction"])
max_abs = {}
for m in ["auprc", "auroc", "brier"]:
    diffs = []
    for r in out.itertuples():
        ref = float(full_key.loc[(int(r.seed), str(r.init), float(r.fraction)), m])
        diffs.append(abs(float(getattr(r, m + "_all")) - ref))
    max_abs[m] = float(max(diffs))

agg_path = R / "stage4_terminal_ipw_results.csv"
out.to_csv(agg_path, index=False)
summary_rows = []
for metric in point_metrics:
    g = out.groupby(["init", "fraction"], as_index=False)[metric].agg(
        n="count", mean="mean", median="median", min="min", max="max"
    )
    g["metric"] = metric
    summary_rows.append(g)
summary = pd.concat(summary_rows, ignore_index=True)
summary = summary[["init", "fraction", "metric", "n", "mean", "median", "min", "max"]]
summary.to_csv(R / "stage4_terminal_ipw_summary.csv", index=False)

# Add only the newly estimable conditional/IPW point metrics to the existing sealed table.
metrics_path = R / "metrics.parquet"
metrics = pd.read_parquet(metrics_path)
metrics = metrics[metrics.stage != "Stage4_terminal_IPW"]
new_rows = []
for r in out.itertuples():
    for metric in point_metrics:
        new_rows.append({
            "stage": "Stage4_terminal_IPW",
            "model": str(r.init),
            "seed": int(r.seed),
            "fraction": float(r.fraction),
            "metric": metric,
            "value": float(getattr(r, metric)),
        })
metrics = pd.concat([metrics, pd.DataFrame(new_rows)], ignore_index=True)
metrics.to_parquet(metrics_path, index=False)

qc = {
    "stage": "Stage4 terminal/IPW downstream point estimates",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "result_files": len(paths),
    "rows": int(len(out)),
    "expected_rows": int(len(keys)),
    "seeds": seeds,
    "initializations": inits,
    "fractions": fracs,
    "point_metrics": point_metrics,
    "finite_counts": finite_counts,
    "n_test_unique": sorted(set(out.n_test.tolist())),
    "n_available_min": int(out.n_available.min()),
    "n_available_max": int(out.n_available.max()),
    "availability_rate_min": float(out.availability_rate.min()),
    "availability_rate_max": float(out.availability_rate.max()),
    "positive_available_min": int(out.positive_available.min()),
    "full_metric_max_abs_difference": max_abs,
    "patient_bootstrap_for_new_metrics": False,
    "bootstrap_note": "The six newly added complete-case/IPW metrics are point estimates only; existing Stage 4 full-test bootstrap rows are retained separately.",
    "status": "PASS" if all_finite and all(v < 1e-6 for v in max_abs.values()) else "FAILED",
}
(R / "stats/stage4_terminal_ipw_qc.json").write_text(json.dumps(qc, indent=2) + "\n")
print(json.dumps({"aggregate_rows": len(out), "metrics_rows": len(metrics), "qc_status": qc["status"], "full_metric_max_abs_difference": max_abs}, indent=2))
