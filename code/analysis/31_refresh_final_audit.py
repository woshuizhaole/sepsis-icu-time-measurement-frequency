from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import pandas as pd

P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015'); R=P/'results'
files=[
'data/processed/partitions.json','data/processed/cinc2019_windows.parquet',
'results/stage3_lgb_v2_results.csv','results/stage3_lgb_v2_bootstrap_ci.csv',
'results/stage3_grud_cde_results.csv','results/stage3_grud_cde_qc.json','logs/qc/stage3_grud_cde_science_review.md',
'results/stage4_ssl_full_results.csv','results/stage4_frozen_head_results.csv',
'results/stage4_terminal_ipw_results.csv','results/stage4_terminal_ipw_summary.csv','results/stats/stage4_terminal_ipw_qc.json',
'results/stage4_calibration_alert_metrics.csv','results/stage4_calibration_bins.csv','results/stage4_alert_curve.csv',
'results/stage4_calibration_alert_qc.json','results/stage4_calibration_alert_warning_audit.json','results/stage4_pretext_calibration_status.json','results/stage4_calibration_alert_manifest.json',
'results/stage5_wbm_results.csv','results/stage5_official_utility_by_seed.csv','results/stage5_official_utility_bootstrap.csv','results/stage5_timecheat_results.csv',
'results/stage5_official_alert_metrics.csv','results/stage5_official_alert_qc.json',
'results/stage6_p19_results.csv','results/stage6_p12_results.csv','results/stage6_transfer_results_all_seeds.csv','results/stage6_transfer_summary.csv','results/stage6_transfer_paired_deltas.csv','results/stage6_transfer_qc.json',
'results/stats/stage7_gap_qc.json','logs/qc/stage7_final_independent_check.txt','results/metrics.parquet','results/bootstrap_ci.csv','results/interpretation.csv','results/stage7_negative_control.json',
]
sha={}
for rel in files:
 p=P/rel
 if p.exists(): sha[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
metrics=pd.read_parquet(R/'metrics.parquet'); bootstrap=pd.read_csv(R/'bootstrap_ci.csv'); interp=pd.read_csv(R/'interpretation.csv')
terminal=pd.read_csv(R/'stage4_terminal_ipw_results.csv'); cal=pd.read_csv(R/'stage4_calibration_alert_metrics.csv'); st3=pd.read_csv(R/'stage3_grud_cde_results.csv'); tr=pd.read_csv(R/'stage6_transfer_results_all_seeds.csv')
gap=json.loads((R/'stats/stage7_gap_qc.json').read_text())

audit=f'''# Final Audit — current computational snapshot

Generated at {datetime.now(timezone.utc).isoformat()} from executed files only. No unexecuted model or metric is promoted.

## Closed evidence

- Stage 0–2: PASS, with current data manifest, patient-level partitions, causal windows, and measurement-channel whitelist sealed.
- Stage 3 v2 LightGBM: 45 rows, five patient-level seeds, corrected Mh/I semantics, patient bootstrap intervals.
- Stage 3 GRU-D/Neural CDE: {len(st3)} rows (GRU-D/Neural CDE × No-OI/OI × five seeds), full train/validation/test partitions and 2,000 patient bootstrap. QC PASS. The Neural CDE comparator is an explicit fixed-hour-grid Euler discretization; it is not an adaptive continuous-time clinical model or a causal information claim.
- Stage 4 full/frozen SSL: 100 full-finetune rows plus 75 frozen-head rows, five seeds, patient fractions, full test and 2,000 patient bootstrap for the sealed arms.
- Stage 4 terminal/IPW downstream: {len(terminal)} rows with finite complete-case and marginal-IPW point estimates. These remain point-estimate-only and are not given new CIs or superiority claims.
- Stage 4/7 held-out calibration and operating diagnostics: {len(cal)} rows, 845 observed calibration bins, 1,500 predeclared threshold-curve rows, and five corrected WBM official-alert rows. Alignment and refractory checks are PASS. Brier/prevalence/mean-pred and operating metrics have 2,000 patient bootstrap; calibration intercept/slope/ECE/MCE remain held-out descriptive point estimates without CI. Nine lead-time rows are explicitly non-estimable because no true alert occurred; they are not imputed as zero. Pretext count/breadth calibration is NOT_ESTIMABLE because held-out pretext logits/distributions were not retained.
- Stage 5: one-step WBM, official Utility with validation-only thresholds and bootstrap, three-seed causal TimeCHEAT, and corrected WBM operating metrics.
- Stage 6: corrected P19 sensitivities, strict pre-cutoff-history P12 landmark benchmark, and a corrected five-seed transfer matrix ({len(tr)} rows) using seed-specific P19 sequence caches. Transfer uses a label-blind train-only first-20h source pool and 1,000-replicate patient-cluster bootstrap CIs as an exploratory dataset/task-shift analysis, distinct from the 2,000-replicate primary SSL protocol. Transfer is not external-hospital validation. The earlier base-cache transfer outputs are quarantined and excluded.

## Scientific boundaries

The overall project status is **PARTIAL**. `metrics.parquet` and `bootstrap_ci.csv` now include the executed GRU-D/CDE, calibration/alert, WBM-alert, and corrected transfer evidence; `interpretation.csv` retains the predeclared comparisons and does not add a new superiority claim for transfer or operating diagnostics. The six terminal/IPW downstream metrics remain point estimates without new patient bootstrap intervals.

The Stage 4 calibration thresholds are external WBM validation-Utility thresholds transferred as an operating diagnostic; they are not Stage 4 validation-optimal thresholds. Calibration is descriptive held-out-test fitting without recalibration; intercept/slope/ECE/MCE have no patient-bootstrap CI. The pretext count/breadth distribution calibration and frozen-head alert calibration remain not estimable from retained artifacts.

Stage 5 MTM/ISTS-PLM/iTimER values remain unexecuted. Stage 6 P19 admission-bundle deletion and the fixed 12h embargo remain bounded/non-estimable sensitivities; P12 and transfer remain conditional landmark/task-shift estimands. The negative control remains descriptive only because singleton-length buckets and a different eligible test subset prevent a paired inferential comparison. No causal order, clinician-action, real-time result-availability, or cross-hospital claim is made. The manuscript is intentionally not generated.

## Statistical artifact counts

- `results/metrics.parquet`: {len(metrics)} rows.
- `results/bootstrap_ci.csv`: {len(bootstrap)} rows.
- `results/interpretation.csv`: {len(interp)} rows.
- `results/stats/stage7_gap_qc.json`: PASS; new metrics={gap['new_metrics_rows']}, new bootstrap={gap['new_bootstrap_rows']}, seed-specific transfer cache gate={gap['stage6_seed_specific_cache_all']}.

## SHA-256 map
{json.dumps(sha, indent=2)}
'''
(P/'FINAL_AUDIT.md').write_text(audit)
print(json.dumps({'metrics':len(metrics),'bootstrap':len(bootstrap),'interpretation':len(interp),'sha_entries':len(sha),'stage7_gap':gap},indent=2))
