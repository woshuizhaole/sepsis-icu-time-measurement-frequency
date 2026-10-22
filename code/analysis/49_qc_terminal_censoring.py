from pathlib import Path
import json, numpy as np, pandas as pd
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015')
O=P/'results/improvement_fair_ssl/terminal_censoring'
seeds=[101,202,303,404,505]; methods=['random','nextmask4','nextintensity4','valueforecast4']; fracs=[0.01,0.05,0.10,0.25,1.0]; models=['hour_empirical','history_logistic','history_gbm']
files=sorted(O.glob('terminal_censoring_metrics_seed*.csv'))
parts=[pd.read_csv(f) for f in files]
d=pd.concat(parts,ignore_index=True)
expected={(s,m,f,t) for s in seeds for m in methods for f in fracs for t in models}
got={(int(r.seed),str(r.method),float(r.fraction),str(r.terminal_model)) for r in d.itertuples()}
errors=[]
if len(d)!=300: errors.append(f'rows:{len(d)}')
if expected-got: errors.append(f'missing:{sorted(expected-got)[:10]}')
if got-expected: errors.append(f'extra:{sorted(got-expected)[:10]}')
if d.duplicated(['seed','method','fraction','terminal_model']).any(): errors.append('duplicate_keys')
num=[c for c in d.columns if c not in ['seed','method','fraction','terminal_model']]
finite=bool(np.isfinite(d[num].to_numpy(dtype=float)).all())
if not finite: errors.append('nonfinite')
for s,g in d.groupby('seed'):
    if g.n_available.nunique()!=1 or g.n_test.nunique()!=1 or g.positive_all.nunique()!=1 or g.positive_available.nunique()!=1: errors.append(f'alignment:{s}')
if not all((d[c+'_lo']<=d[c+'_hi']).all() for c in []): pass
# validation files and terminal logits are required artifacts.
val=sorted(O.glob('terminal_model_validation_seed*.csv')); logits=sorted(O.glob('terminal_logits_seed*.parquet'))
if len(val)!=5: errors.append(f'validation_files:{len(val)}')
if len(logits)!=5: errors.append(f'logit_files:{len(logits)}')
for f in logits:
    q=pd.read_parquet(f)
    if len(q)==0 or not np.isfinite(q.filter(regex='^(p_available|logit_available)').to_numpy(dtype=float)).all(): errors.append(f'logit_invalid:{f.name}')
d.to_csv(O/'terminal_censoring_metrics_all_seeds.csv',index=False)
summary=(d.groupby(['terminal_model','method','fraction'],as_index=False)
    .agg(n=('seed','count'), auprc_complete_case_median=('auprc_complete_case','median'), auprc_raw_ipw_median=('auprc_raw_ipw','median'), auprc_stabilized_ipw_median=('auprc_stabilized_ipw','median'),
         raw_weight_p99_median=('raw_weight_p99','median'), raw_weight_max_max=('raw_weight_max','max'), raw_ess_median=('raw_effective_sample_size','median'),
         stabilized_weight_max_max=('stabilized_weight_max','max'), stabilized_ess_median=('stabilized_effective_sample_size','median')))
summary.to_csv(O/'terminal_censoring_summary.csv',index=False)
qc={'status':'PASS' if not errors and finite and len(d)==300 else 'FAILED','rows':int(len(d)),'expected_rows':300,'seeds':seeds,'methods':methods,'fractions':fracs,'terminal_models':models,'finite_numeric':finite,'validation_files':len(val),'logit_files':len(logits),'point_estimates_only':True,'patient_bootstrap':False,'errors':errors,'metrics_path':str(O/'terminal_censoring_metrics_all_seeds.csv'),'summary_path':str(O/'terminal_censoring_summary.csv')}
(O/'terminal_censoring_qc.json').write_text(json.dumps(qc,indent=2))
print(json.dumps(qc,indent=2)); print(summary.to_string(index=False))
