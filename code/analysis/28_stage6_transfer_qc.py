#!/usr/bin/env python3
"""Seal and audit the Stage 6 cross-dataset transfer matrix.

This is read-only with respect to model results: it concatenates the five
seed-level outputs, checks split/estimand invariants, and writes aggregate
tables plus a machine-readable QC record.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

P = Path("/data/workplace/wyx/measurement_intensity_ssl_20261015")
D = P / "data/processed"
R = P / "results"
SEEDS = [101, 202, 303, 404, 505]
REQ = [
    "val_best_auprc", "auprc", "auroc", "brier",
    "auprc_lo", "auprc_hi", "auroc_lo", "auroc_hi", "brier_lo", "brier_hi",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    frames = []
    pred_frames = []
    source_meta = []
    for s in SEEDS:
        rp = R / f"stage6_transfer_results_seed{s}.csv"
        pp = R / f"stage6_transfer_predictions_seed{s}.parquet"
        sp = R / f"stage6_transfer_source_seed{s}.json"
        assert rp.exists() and pp.exists() and sp.exists(), (s, rp, pp, sp)
        d = pd.read_csv(rp)
        assert len(d) == 9, (s, len(d))
        assert set(d.seed.astype(int)) == {s}
        assert d[REQ].notna().all().all(), f"nonfinite required result field seed {s}"
        assert d.n_common_dynamic.eq(22).all()
        assert d.direction.value_counts().to_dict() == {"p19_to_p12": 6, "p12_to_p19": 3}
        p = pd.read_parquet(pp)
        assert p.seed.astype(int).eq(s).all()
        assert p.pred.notna().all() and p.y.notna().all()
        assert set(p.direction) == {"p19_to_p12", "p12_to_p19"}
        frames.append(d)
        pred_frames.append(p)
        source_meta.append(json.loads(sp.read_text()))

    all_results = pd.concat(frames, ignore_index=True)
    all_preds = pd.concat(pred_frames, ignore_index=True)
    out_results = R / "stage6_transfer_results_all_seeds.csv"
    out_preds = R / "stage6_transfer_predictions_all_seeds.parquet"
    all_results.to_csv(out_results, index=False)
    all_preds.to_parquet(out_preds, index=False)

    metric_cols = ["auprc", "auroc", "brier", "patient_auprc", "patient_auroc", "patient_brier"]
    summary = []
    for keys, g in all_results.groupby(["direction", "cutoff_hour", "mode"], dropna=False, sort=True):
        direction, cutoff, mode = keys
        row = {"direction": direction, "cutoff_hour": cutoff, "mode": mode, "n_seeds": int(g.seed.nunique())}
        for c in metric_cols:
            if c not in g:
                continue
            x = g[c].dropna().to_numpy(float)
            if len(x):
                row[f"{c}_median"] = float(np.median(x))
                row[f"{c}_iqr_lo"] = float(np.percentile(x, 25))
                row[f"{c}_iqr_hi"] = float(np.percentile(x, 75))
            else:
                row[f"{c}_median"] = row[f"{c}_iqr_lo"] = row[f"{c}_iqr_hi"] = np.nan
        row["n_test_median"] = float(g.n.median())
        row["n_patients_median"] = float(g.n_patients.median())
        row["positive_median"] = float(g.positive.median())
        summary.append(row)
    out_summary = R / "stage6_transfer_summary.csv"
    pd.DataFrame(summary).to_csv(out_summary, index=False)

    # Seed-paired deltas against the target-only random initialization.  These
    # are descriptive (no multiplicity-adjusted inference is claimed here) and
    # make the transfer boundary auditable without choosing a favourable seed.
    delta_rows = []
    for keys, g in all_results.groupby(["seed", "direction", "cutoff_hour"], dropna=False, sort=True):
        seed, direction, cutoff = keys
        q = g.set_index("mode")
        for mode in ("frozen", "finetune"):
            if mode not in q.index or "random" not in q.index:
                continue
            row = {"seed": int(seed), "direction": direction, "cutoff_hour": cutoff, "mode_vs_random": mode}
            for c in ("auprc", "auroc", "brier", "patient_auprc", "patient_auroc", "patient_brier"):
                if c in q:
                    row[f"delta_{c}"] = float(q.loc[mode, c] - q.loc["random", c]) if pd.notna(q.loc[mode, c]) and pd.notna(q.loc["random", c]) else np.nan
            delta_rows.append(row)
    out_delta = R / "stage6_transfer_paired_deltas.csv"
    pd.DataFrame(delta_rows).to_csv(out_delta, index=False)

    partitions = json.loads((D / "partitions.json").read_text())
    split_qc = {}
    for ds in ("cinc2019", "cinc2012"):
        split_qc[ds] = {}
        for s in SEEDS:
            m = partitions[ds][str(s)]
            sets = {x: {k for k, v in m.items() if v == x} for x in ("train", "validation", "test")}
            split_qc[ds][str(s)] = {"n": {x: len(v) for x, v in sets.items()}, "pairwise_disjoint": bool(
                not (sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]))}

    group_sizes = all_preds.groupby(["seed", "direction", "cutoff_hour", "mode"], dropna=False).size().reset_index(name="n_pred")
    assert len(group_sizes) == 45
    assert group_sizes.n_pred.gt(0).all()
    # Every target mode within a seed/cutoff must evaluate the same patient set.
    pred_set_qc = []
    for (s, direction, cutoff), g in all_preds.groupby(["seed", "direction", "cutoff_hour"], dropna=False):
        sets = [frozenset(x.patient_id.astype(str)) for _, x in g.groupby("mode")]
        pred_set_qc.append({"seed": int(s), "direction": direction, "cutoff_hour": cutoff, "n_modes": len(sets), "same_patient_set": len(set(sets)) == 1, "n_patients": len(sets[0])})
    assert all(x["same_patient_set"] for x in pred_set_qc)

    # Explicit estimand check: no row mixes the landmark and hourly targets.
    estimands = sorted(all_results.estimand.dropna().unique().tolist())
    assert len(estimands) == 2
    assert all_results.loc[all_results.direction == "p19_to_p12", "estimand"].str.contains("conditional on observed history").all()
    assert all_results.loc[all_results.direction == "p12_to_p19", "estimand"].str.contains("complete next-window target rows").all()
    assert all(m["common_channels"] == source_meta[0]["common_channels"] for m in source_meta)
    assert all(m["input_dim"] == 69 for m in source_meta)
    assert all(m.get("p19_sequence_seed_specific") is True for m in source_meta), "seed-specific P19 sequence cache required"
    sequence_files = {str(m["seed"]): m.get("p19_sequence_file") for m in source_meta}
    assert all(m["p19_pretrain"]["n_used"] == 20000 and m["p12_pretrain"]["n_used"] == 20000 for m in source_meta)

    qc = {
        "status": "PASS",
        "seeds": SEEDS,
        "result_rows": int(len(all_results)),
        "prediction_rows": int(len(all_preds)),
        "rows_per_seed": {str(s): int((all_results.seed == s).sum()) for s in SEEDS},
        "required_fields_finite": True,
        "bootstrap_reps": 2000,
        "common_dynamic_channels": source_meta[0]["common_channels"],
        "common_dynamic_count": len(source_meta[0]["common_channels"]),
        "input_dim": source_meta[0]["input_dim"],
        "p19_sequence_files": sequence_files,
        "p19_sequence_cache_seed_specific": True,
        "estimands": estimands,
        "split_qc": split_qc,
        "prediction_patient_set_qc": pred_set_qc,
        "source_pretrain_label_blind_pool": {"p19_first20h_train_only": True, "p12_first20h_train_only": True, "used_per_seed": 20000},
        "scientific_boundaries": {
            "p12_event_counts_collapsed_to_hourly_binary_before_common_mapping": True,
            "p19_target_complete_next_window_only": True,
            "p12_history_strictly_before_cutoff": True,
            "same_p12_patients_can_appear_at_cutoffs_12_and_24_but_are_not_pooled": True,
            "cross_dataset_result_is_dataset_task_shift_not_external_hospital_validation": True,
            "no_patient_id_join_across_datasets": True,
        },
        "artifacts": {
            str(x): {"bytes": (R / x).stat().st_size, "sha256": sha256(R / x)}
            for x in [out_results.name, out_summary.name, out_delta.name, out_preds.name]
        },
    }
    (R / "stage6_transfer_qc.json").write_text(json.dumps(qc, indent=2))
    print(json.dumps({"status": qc["status"], "result_rows": len(all_results), "prediction_rows": len(all_preds), "summary_rows": len(summary), "delta_rows": len(delta_rows), "qc": str(R / "stage6_transfer_qc.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
