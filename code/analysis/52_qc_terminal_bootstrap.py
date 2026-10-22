from pathlib import Path
import json, numpy as np, pandas as pd
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015')
O=P/'results/improvement_fair_ssl/terminal_censoring'
seeds=[101,202,303,404,505]; methods=['random','nextmask4','nextintensity4','valueforecast4']; models=['hour_empirical','history_logistic','history_gbm']
files=sorted(O.glob('terminal_censoring_bootstrap_seed*.csv'))
d=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
errors=[]
if len(d)!=60: errors.append(f'rows:{len(d)}')
expected={(s,m,t) for s in seeds for m in methods for t in models}
got={(int(r.seed),str(r.method),str(r.terminal_model)) for r in d.itertuples()}
if expected-got: errors.append(f'missing:{sorted(expected-got)}')
if got-expected: errors.append(f'extra:{sorted(got-expected)}')
if d.duplicated(['seed','method','terminal_model']).any(): errors.append('duplicate')
num=[c for c in d.columns if c not in ['seed','method','terminal_model','fraction','bootstrap']]
finite=bool(np.isfinite(d[num].to_numpy(dtype=float)).all())
if not finite: errors.append('nonfinite')
for c in ['complete_case','raw_ipw','stabilized_ipw']:
    for metric in ['auprc','brier']:
        if not (d[f'{c}_{metric}_lo']<=d[f'{c}_{metric}_hi']).all(): errors.append(f'ci_order:{c}:{metric}')
d.to_csv(O/'terminal_censoring_bootstrap_all_seeds.csv',index=False)
summary=(d.groupby(['terminal_model','method'],as_index=False)
 .agg(n=('seed','count'),complete_case_auprc_median=('complete_case_auprc_lo','median'),raw_ipw_auprc_lo_median=('raw_ipw_auprc_lo','median'),raw_ipw_auprc_hi_median=('raw_ipw_auprc_hi','median'),stabilized_ipw_auprc_lo_median=('stabilized_ipw_auprc_lo','median'),stabilized_ipw_auprc_hi_median=('stabilized_ipw_auprc_hi','median'),raw_ipw_brier_lo_median=('raw_ipw_brier_lo','median'),raw_ipw_brier_hi_median=('raw_ipw_brier_hi','median')))
summary.to_csv(O/'terminal_censoring_bootstrap_summary.csv',index=False)
qc={'status':'PASS' if len(d)==60 and not errors and finite else 'FAILED','rows':int(len(d)),'expected_rows':60,'seeds':seeds,'methods':methods,'terminal_models':models,'bootstrap':2000,'metrics':['auprc','brier'],'finite_numeric':finite,'errors':errors,'bootstrap_path':str(O/'terminal_censoring_bootstrap_all_seeds.csv'),'summary_path':str(O/'terminal_censoring_bootstrap_summary.csv')}
(O/'terminal_censoring_bootstrap_qc.json').write_text(json.dumps(qc,indent=2))
print(json.dumps(qc,indent=2)); print(summary.to_string(index=False))
