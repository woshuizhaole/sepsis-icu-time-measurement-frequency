from pathlib import Path
import json,numpy as np,pandas as pd
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015'); R=P/'results/improvement_fair_ssl/terminal_controls_all'
seeds=[101,202,303,404,505]; inits=['random','terminal_only','nextmask4','nextintensity4','valueforecast4']; models=['hour_empirical','history_logistic','history_gbm']
files=sorted(R.glob('terminal_aux_bootstrap_seed*.csv')); d=pd.concat([pd.read_csv(f) for f in files],ignore_index=True); errors=[]
expected={(s,i,m) for s in seeds for i in inits for m in models}; got={(int(r.seed),str(r['init']),str(r.terminal_model)) for _,r in d.iterrows()}
if len(d)!=75: errors.append(f'rows:{len(d)}')
if expected-got: errors.append(f'missing:{sorted(expected-got)[:10]}')
if got-expected: errors.append(f'extra:{sorted(got-expected)[:10]}')
num=d.select_dtypes(include=[np.number]).columns; finite=bool(np.isfinite(d[num].to_numpy(dtype=float)).all())
if not finite: errors.append('nonfinite')
for c in ['complete_case','raw_ipw','stabilized_ipw']:
 for m in ['auprc','brier']:
  if not (d[f'{c}_{m}_lo']<=d[f'{c}_{m}_hi']).all(): errors.append(f'ci:{c}:{m}')
d.to_csv(R/'terminal_aux_bootstrap_all_seeds.csv',index=False)
summary=(d.groupby(['init','terminal_model'],as_index=False).agg(n=('seed','count'),raw_ipw_auprc_lo_median=('raw_ipw_auprc_lo','median'),raw_ipw_auprc_hi_median=('raw_ipw_auprc_hi','median'),stabilized_ipw_auprc_lo_median=('stabilized_ipw_auprc_lo','median'),stabilized_ipw_auprc_hi_median=('stabilized_ipw_auprc_hi','median'),raw_ipw_brier_lo_median=('raw_ipw_brier_lo','median'),raw_ipw_brier_hi_median=('raw_ipw_brier_hi','median')))
summary.to_csv(R/'terminal_aux_bootstrap_summary.csv',index=False)
qc={'status':'PASS' if len(d)==75 and not errors and finite else 'FAILED','rows':int(len(d)),'expected_rows':75,'seeds':seeds,'inits':inits,'terminal_models':models,'bootstrap':2000,'metrics':['auprc','brier'],'finite_numeric':finite,'errors':errors,'path':str(R/'terminal_aux_bootstrap_all_seeds.csv'),'summary_path':str(R/'terminal_aux_bootstrap_summary.csv')}
(R/'terminal_aux_bootstrap_qc.json').write_text(json.dumps(qc,indent=2)); print(json.dumps(qc,indent=2)); print(summary.to_string(index=False))
