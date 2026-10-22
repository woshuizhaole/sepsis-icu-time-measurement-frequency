#!/usr/bin/env python3
"""Read-only QC for the Stage 3 GRU-D/Neural-CDE run matrix."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

P = Path("/data/workplace/wyx/measurement_intensity_ssl_20261015")
D, R, L = P / "data/processed", P / "results", P / "logs"
SEEDS = [101, 202, 303, 404, 505]
ARCH = ["grud", "cde"]
COND = ["nooi", "oi"]


def main() -> None:
    checks = []
    errors = []
    metrics_path = R / "stage3_grud_cde_results.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    m = pd.read_csv(metrics_path)
    expected = {(a, c, s) for a in ARCH for c in COND for s in SEEDS}
    got = set(zip(m.architecture, m.condition, m.seed))
    checks.append({"check": "metrics_matrix_20", "observed": len(got), "expected": len(expected), "pass": got == expected})
    if got != expected:
        errors.append(f"metrics combinations mismatch: missing={sorted(expected-got)} extra={sorted(got-expected)}")
    numeric = m.select_dtypes(include=[np.number]).to_numpy()
    finite = bool(np.isfinite(numeric).all())
    checks.append({"check": "metrics_finite", "pass": finite})
    if not finite:
        errors.append("non-finite metric value")
    ci_ok = bool(((m.auprc >= m.auprc_lo) & (m.auprc <= m.auprc_hi) &
                  (m.auroc >= m.auroc_lo) & (m.auroc <= m.auroc_hi) &
                  (m.brier >= m.brier_lo) & (m.brier <= m.brier_hi)).all())
    checks.append({"check": "bootstrap_intervals_contain_point", "pass": ci_ok})
    if not ci_ok:
        errors.append("one or more bootstrap intervals do not contain point estimate")

    part_all = json.loads((D / "partitions.json").read_text())["cinc2019"]
    win = pd.read_parquet(D / "cinc2019_windows.parquet", columns=["patient_id", "anchor_hour", "y_h6"])
    meta = pd.read_parquet(D / "p19_sequence_meta.parquet")
    z = win.merge(meta, left_on=["patient_id", "anchor_hour"], right_on=["patient_id", "ICULOS"], validate="one_to_one")
    for a, c, s in sorted(expected):
        row = m[(m.architecture == a) & (m.condition == c) & (m.seed == s)]
        if len(row) != 1:
            errors.append(f"missing unique metric row {a}/{c}/{s}")
            continue
        r = row.iloc[0]
        pred_path = R / f"stage3_{a}_{c}_seed{s}_pred.parquet"
        ckpt_path = R / f"stage3_{a}_{c}_seed{s}.pt"
        if not pred_path.exists() or not ckpt_path.exists():
            errors.append(f"missing prediction/checkpoint {a}/{c}/{s}")
            continue
        p = pd.read_parquet(pred_path)
        expected_test = set(k for k, v in part_all[str(s)].items() if v == "test")
        bad_pid = set(p.patient_id.astype(str)) - expected_test
        n_test = int((z.patient_id.map(part_all[str(s)]) == "test").sum())
        checks.append({"check": "prediction", "architecture": a, "condition": c, "seed": int(s),
                       "rows": len(p), "expected_rows": n_test,
                       "positive": int(p.y.sum()), "metric_positive": int(r.positive_test),
                       "finite": bool(np.isfinite(p[["y", "p"]].to_numpy()).all()),
                       "test_patient_set_only": not bad_pid,
                       "pass": len(p) == n_test and int(p.y.sum()) == int(r.positive_test) and
                               bool(np.isfinite(p[["y", "p"]].to_numpy()).all()) and not bad_pid})
        if len(p) != n_test:
            errors.append(f"row count mismatch {a}/{c}/{s}: {len(p)} != {n_test}")
        if int(p.y.sum()) != int(r.positive_test):
            errors.append(f"positive count mismatch {a}/{c}/{s}")
        if not np.isfinite(p[["y", "p"]].to_numpy()).all():
            errors.append(f"non-finite prediction {a}/{c}/{s}")
        if bad_pid:
            errors.append(f"prediction patient outside test split {a}/{c}/{s}: {sorted(bad_pid)[:3]}")

    for s in SEEDS:
        st = R / f"stage3_grud_cde_norm_seed{s}.json"
        if not st.exists():
            errors.append(f"missing train-only norm stats seed{s}")
    logs = list(L.glob("analysis/stage3_grud_cde_*_seed*.log"))
    traceback_logs = [str(x) for x in logs if "Traceback" in x.read_text(errors="replace")]
    checks.append({"check": "logs_no_traceback", "n_logs": len(logs), "pass": not traceback_logs})
    if traceback_logs:
        errors.append(f"traceback text in logs: {traceback_logs}")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAILED",
        "n_checks": len(checks),
        "errors": errors,
        "checks": checks,
        "protocol": {
            "architecture": "GRU-D and explicit fixed-grid Euler Neural CDE",
            "conditions": "No-OI value/static path vs OI value+mask+4h-intensity path",
            "hidden": 64, "epochs": 2, "batch": 4096, "seeds": SEEDS,
            "patient_bootstrap": 2000,
        },
    }
    out = R / "stage3_grud_cde_qc.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({"status": report["status"], "n_checks": len(checks), "n_errors": len(errors), "path": str(out)}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
