from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015')
O=P/'results/rootcause_stage0_2'
seeds=[101,202,303,404,505]
conditions=['admin_time_only','anchor_hour_only','clinical_V','clinical_V+M0','clinical_V+I','clinical_V+M0+I','clinical_V+I_equalized_noise','full_V+M0','full_V+I','full_V+M0+I']
pairs=['clinical_V+I_minus_clinical_V','clinical_V+M0+I_minus_clinical_V+M0','clinical_V+I_equalized_noise_minus_clinical_V+M0+I','full_V+I_minus_clinical_V+I','full_V+M0+I_minus_clinical_V+M0+I']
metrics=[]
paired=[]
errors=[]
for s in seeds:
    f=O/f'p19_clinical_decomposition_seed{s}.csv'
    g=O/f'p19_clinical_decomposition_paired_seed{s}.csv'
    if not f.exists(): errors.append(f'missing_metrics:{s}')
    else: metrics.append(pd.read_csv(f))
    if not g.exists(): errors.append(f'missing_paired:{s}')
    else: paired.append(pd.read_csv(g))
m=pd.concat(metrics,ignore_index=True) if metrics else pd.DataFrame()
p=pd.concat(paired,ignore_index=True) if paired else pd.DataFrame()
m.to_csv(O/'p19_clinical_decomposition_all_seeds.csv',index=False)
p.to_csv(O/'p19_clinical_decomposition_paired_all_seeds.csv',index=False)
num_m=[c for c in m.columns if c not in ['seed','condition']]
num_p=[c for c in p.columns if c not in ['seed','comparison','component','reference','paired_patient_bootstrap']]
finite_m=bool(len(m) and np.isfinite(m[num_m].to_numpy(dtype=float)).all())
finite_p=bool(len(p) and np.isfinite(p[num_p].to_numpy(dtype=float)).all())
unique_m=bool(len(m) and not m[['seed','condition']].duplicated().any())
unique_p=bool(len(p) and not p[['seed','comparison']].duplicated().any())
shape_ok=len(m)==50 and len(p)==25
if set(m.condition)!=set(conditions): errors.append('condition_set')
if set(p.comparison)!=set(pairs): errors.append('pair_set')
if set(m.seed)!=set(seeds) or set(p.seed)!=set(seeds): errors.append('seed_set')
for s,g in m.groupby('seed'):
    if len(g)!=10: errors.append(f'metric_rows:{s}:{len(g)}')
    if g.n_test.nunique()!=1 or g.n_test_patients.nunique()!=1 or g.positive_test.nunique()!=1: errors.append(f'test_cohort:{s}')
    a=g.loc[g.condition=='clinical_V+M0+I','n_features'].iloc[0]
    b=g.loc[g.condition=='clinical_V+I_equalized_noise','n_features'].iloc[0]
    if a!=105 or b!=105: errors.append(f'equalized_features:{s}:{a}:{b}')
for c in ['auprc','auroc','brier']:
    for lo,hi in [(c+'_lo',c+'_hi')]:
        if not (m[lo] <= m[hi]).all(): errors.append(f'metric_ci:{c}')
for c in ['delta_auprc','delta_auroc','delta_brier']:
    if not (p[c+'_lo'] <= p[c+'_hi']).all(): errors.append(f'paired_ci:{c}')
summary=(m.groupby('condition',as_index=False)
    .agg(n=('seed','count'), auprc_median=('auprc','median'), auprc_q1=('auprc',lambda x: x.quantile(.25)), auprc_q3=('auprc',lambda x: x.quantile(.75)),
         auprc_mean=('auprc','mean'), auroc_median=('auroc','median'), brier_median=('brier','median'),
         n_features=('n_features','first')))
summary.to_csv(O/'p19_clinical_decomposition_summary.csv',index=False)
ps=(p.groupby('comparison',as_index=False)
    .agg(n=('seed','count'), delta_auprc_median=('delta_auprc','median'), delta_auprc_q1=('delta_auprc',lambda x:x.quantile(.25)), delta_auprc_q3=('delta_auprc',lambda x:x.quantile(.75)),
         delta_auprc_mean=('delta_auprc','mean'), directions_positive=('delta_auprc',lambda x:int((x>0).sum())),
         paired_ci_cross_zero=('delta_auprc',lambda x:True)))
# CI crossing is per split; add a summary fraction and no superiority claim.
for comp,g in p.groupby('comparison'):
    i=ps.index[ps.comparison==comp][0]
    ps.loc[i,'ci_cross_zero_fraction']=float(((g.delta_auprc_lo<=0)&(g.delta_auprc_hi>=0)).mean())
    ps.loc[i,'all_split_ci_exclude_zero']=bool(((g.delta_auprc_lo>0)|(g.delta_auprc_hi<0)).all())
ps.to_csv(O/'p19_clinical_decomposition_paired_summary.csv',index=False)
qc={
    'status':'PASS' if shape_ok and not errors and finite_m and finite_p and unique_m and unique_p else 'FAILED',
    'metric_rows':int(len(m)), 'paired_rows':int(len(p)),
    'expected_metric_rows':50, 'expected_paired_rows':25,
    'seeds':seeds, 'conditions':conditions, 'pairs':pairs,
    'finite_metrics':finite_m, 'finite_paired':finite_p,
    'unique_metric_keys':unique_m, 'unique_paired_keys':unique_p,
    'same_test_cohort_within_seed':not any(x.startswith('test_cohort:') for x in errors),
    'equalized_feature_count_105':not any(x.startswith('equalized_features:') for x in errors),
    'bootstrap':2000, 'paired_patient_bootstrap':True, 'errors':errors,
    'metrics_path':str(O/'p19_clinical_decomposition_all_seeds.csv'),
    'paired_path':str(O/'p19_clinical_decomposition_paired_all_seeds.csv'),
    'summary_path':str(O/'p19_clinical_decomposition_summary.csv'),
    'paired_summary_path':str(O/'p19_clinical_decomposition_paired_summary.csv')
}
(O/'p19_clinical_decomposition_qc.json').write_text(json.dumps(qc,indent=2))
print(json.dumps(qc,indent=2))
print(summary.to_string(index=False))
print(ps.to_string(index=False))
