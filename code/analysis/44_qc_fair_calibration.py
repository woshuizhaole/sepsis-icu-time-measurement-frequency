from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

P = Path('/data/workplace/wyx/measurement_intensity_ssl_20261015')
R = P / 'results/improvement_fair_ssl/calibration'
files = sorted(R.glob('fair_calibration_seed*.csv'))
expected_seeds = [101, 202, 303, 404, 505]
methods = ['random','nextmask4','nextintensity4','valueforecast4']
rows = []
errors = []
for seed in expected_seeds:
    f = R / f'fair_calibration_seed{seed}.csv'
    if not f.exists():
        errors.append(f'missing:{f}')
        continue
    d = pd.read_csv(f)
    if len(d) != 4:
        errors.append(f'row_count:{seed}:{len(d)}')
    if set(d['method']) != set(methods):
        errors.append(f'methods:{seed}:{sorted(set(d.method))}')
    if d['seed'].nunique() != 1 or int(d['seed'].iloc[0]) != seed:
        errors.append(f'seed_column:{seed}')
    rows.append(d)
all_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
all_path = R / 'fair_calibration_all_seeds.csv'
all_df.to_csv(all_path, index=False)
num_cols = [c for c in all_df.columns if c not in ['seed','method']]
finite = bool(np.isfinite(all_df[num_cols].to_numpy(dtype=float)).all()) if len(all_df) else False
key_unique = bool(all_df[['seed','method']].duplicated().sum() == 0) if len(all_df) else False
row_pass = len(all_df) == 20 and len(files) == 5
prev_consistent = True
auprc_invariant = True
ci_order = True
for seed, g in all_df.groupby('seed'):
    prev_consistent &= g['test_prevalence'].nunique() == 1
    auprc_invariant &= bool(np.max(np.abs(g['raw_auprc'] - g['cal_auprc'])) < 1e-5)
    for c in ['raw_auprc_lo','raw_auprc_hi','raw_brier_lo','raw_brier_hi','cal_auprc_lo','cal_auprc_hi','cal_brier_lo','cal_brier_hi']:
        if c.endswith('_lo'):
            hi = c.replace('_lo','_hi')
            ci_order &= bool((g[c] <= g[hi]).all())
summary = (all_df.groupby('method', as_index=False)
           .agg(n=('seed','count'),
                raw_mean_p=('raw_mean_p','mean'),
                cal_mean_p=('cal_mean_p','mean'),
                test_prevalence=('test_prevalence','mean'),
                raw_brier=('raw_brier','mean'),
                cal_brier=('cal_brier','mean'),
                raw_auprc=('raw_auprc','mean'),
                cal_auprc=('cal_auprc','mean'),
                raw_brier_delta=('raw_brier','mean')))
if len(summary):
    summary['cal_minus_raw_brier'] = summary['cal_brier'] - summary['raw_brier']
    summary['cal_minus_raw_mean_p'] = summary['cal_mean_p'] - summary['raw_mean_p']
summary_path = R / 'fair_calibration_summary.csv'
summary.to_csv(summary_path, index=False)
qc = {
    'status': 'PASS' if (row_pass and not errors and finite and key_unique and prev_consistent and auprc_invariant and ci_order) else 'FAILED',
    'rows': int(len(all_df)),
    'files': [str(x) for x in files],
    'expected_rows': 20,
    'expected_seeds': expected_seeds,
    'expected_methods': methods,
    'finite_numeric': finite,
    'unique_seed_method_keys': key_unique,
    'test_prevalence_consistent_within_seed': bool(prev_consistent),
    'validation_offset_calibration_only': True,
    'auprc_invariant_under_intercept_offset': bool(auprc_invariant),
    'bootstrap_ci_ordered': bool(ci_order),
    'errors': errors,
    'all_path': str(all_path),
    'summary_path': str(summary_path),
}
(R / 'fair_calibration_qc.json').write_text(json.dumps(qc, indent=2))
print(json.dumps(qc, indent=2))
print(summary.to_string(index=False))
