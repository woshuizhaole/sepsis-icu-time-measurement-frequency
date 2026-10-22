from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import pandas as pd

P = Path('/data/workplace/wyx/measurement_intensity_ssl_20261015')
R = P / 'results'
files = [
    'data/processed/partitions.json',
    'data/processed/cinc2019_windows.parquet',
    'results/stage3_lgb_v2_results.csv',
    'results/stage3_lgb_v2_bootstrap_ci.csv',
    'results/stage4_ssl_full_results.csv',
    'results/stage4_frozen_head_results.csv',
    'results/stage4_terminal_ipw_results.csv',
    'results/stage4_terminal_ipw_summary.csv',
    'results/stats/stage4_terminal_ipw_qc.json',
    'results/stage5_wbm_results.csv',
    'results/stage5_official_utility_by_seed.csv',
    'results/stage5_official_utility_bootstrap.csv',
    'results/stage5_timecheat_results.csv',
    'results/stage6_p19_results.csv',
    'results/stage6_p12_results.csv',
    'results/metrics.parquet',
    'results/bootstrap_ci.csv',
    'results/interpretation.csv',
    'results/stage7_negative_control.json',
]
sha = {}
for rel in files:
    p = P / rel
    if p.exists():
        sha[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
metrics = pd.read_parquet(R / 'metrics.parquet')
bootstrap = pd.read_csv(R / 'bootstrap_ci.csv')
interpretation = pd.read_csv(R / 'interpretation.csv')
terminal = pd.read_csv(R / 'stage4_terminal_ipw_results.csv')
qc = json.loads((R / 'stats/stage4_terminal_ipw_qc.json').read_text())
qc_status = qc['status']

audit = f'''# Final Audit — current computational snapshot

Generated at {datetime.now(timezone.utc).isoformat()} from executed files only. No unexecuted model or metric is promoted.

## Closed evidence

- Stage 0–2: PASS.
- Stage 3 v2 LightGBM decomposition: 45 rows, five patient-level seeds, corrected Mh/I semantics, patient bootstrap CI.
- Stage 4 corrected SSL: 100 full-finetuning rows plus 75 frozen-head rows, five seeds, patient fractions, full test, and 2,000 patient bootstrap for the sealed full/frozen arms.
- Stage 4 terminal/IPW downstream point-estimate matrix: {len(terminal)} rows covering 5 seeds × 4 initializations × 5 fractions. The six new complete-case and hour-stratified marginal-IPW metrics are finite; all-test metrics reproduce the sealed Stage 4 full-test metrics within the recorded floating-point tolerance. QC: results/stats/stage4_terminal_ipw_qc.json ({qc_status}).
- Stage 5: one-step WBM auxiliary baseline, official Utility with validation-only threshold and Utility bootstrap, and three-seed causal TimeCHEAT comparator.
- Stage 6: corrected complete-case P19 sensitivities and strict pre-cutoff-history P12 landmark benchmark.
- Stage 7: metrics, existing bootstrap intervals, Holm/sign-flip interpretation, and a descriptive seed-101 trajectory-permutation negative control.

## Scientific boundaries

The overall project status is **PARTIAL**. The newly sealed terminal/IPW rows are downstream point estimates conditional on next-window availability (complete-case) or weighted by an hour-stratified observation model (marginal IPW). The six new conditional/IPW metrics do not yet have patient-bootstrap intervals; existing Stage 4 full-test bootstrap rows remain separate and are not reused to claim uncertainty for these new estimands. No causal order, clinician-action, or real-time result-availability interpretation is made.

Stage 4 pretext calibration/count-breadth calibration, same-protocol Stage 3 GRU-D/Neural CDE, full cross-dataset frozen/fine-tune transfer, and alert-burden/calibration analyses remain unexecuted. Stage 5 WBM is one-step next-hour and TimeCHEAT is a three-seed rolling sequence-to-one comparator with static fields omitted. Stage 6 P19 rows are complete-case conditional sensitivities; the 12h embargo has no positive estimable windows; P12 is conditional on retrospective landmark availability.

The negative control remains descriptive only: the executed trajectory permutation contains singleton-length buckets and a slightly different eligible test subset, so its delta is not a paired inferential comparison. The manuscript is intentionally not generated.

## Statistical artifact counts

- `results/metrics.parquet`: {len(metrics)} rows, including terminal/IPW point metrics.
- `results/bootstrap_ci.csv`: {len(bootstrap)} rows; this count intentionally excludes CIs for the six new terminal/IPW metrics.
- `results/interpretation.csv`: {len(interpretation)} rows; no new terminal/IPW superiority claim is added.

## SHA-256 map
{json.dumps(sha, indent=2)}
'''
(P / 'FINAL_AUDIT.md').write_text(audit)
print(json.dumps({'metrics': len(metrics), 'bootstrap': len(bootstrap), 'interpretation': len(interpretation), 'terminal': len(terminal), 'sha_entries': len(sha)}, indent=2))
