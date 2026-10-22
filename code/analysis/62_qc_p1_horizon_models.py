from pathlib import Path
import json,numpy as np,pandas as pd
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015'); O=P/'results/rootcause_stage6'
seeds=[101,202,303,404,505]; H=[1,4,6,8,12,24]; d=pd.concat([pd.read_csv(f) for f in sorted(O.glob('p1_horizon_model_results_seed*.csv'))],ignore_index=True); errors=[]
if len(d)!=30: errors.append(f'rows:{len(d)}')
if d.duplicated(['seed','horizon_hours']).any(): errors.append('duplicate')
num=[c for c in ['n_train','n_val','n_test','positive_test','availability_rate','auprc','auroc','brier','auprc_2.5','auprc_97.5','auroc_2.5','auroc_97.5','brier_2.5','brier_97.5'] if c in d.columns]
if not np.isfinite(d[num].to_numpy(float)).all(): errors.append('nonfinite')
for c in ['auprc','auroc','brier']:
 if not (d[c+'_2.5']<=d[c+'_97.5']).all(): errors.append(f'ci:{c}')
d.to_csv(O/'p1_horizon_model_results_all_seeds.csv',index=False)
summary=(d.groupby('horizon_hours',as_index=False).agg(n=('seed','count'),auprc_median=('auprc','median'),auprc_q1=('auprc',lambda x:x.quantile(.25)),auprc_q3=('auprc',lambda x:x.quantile(.75)),auroc_median=('auroc','median'),brier_median=('brier','median'),positive_test_median=('positive_test','median'),n_test_median=('n_test','median'),availability_rate_median=('availability_rate','median')))
summary.to_csv(O/'p1_horizon_model_summary.csv',index=False)
qc={'status':'PASS' if len(d)==30 and not errors else 'FAILED','rows':int(len(d)),'expected_rows':30,'seeds':seeds,'horizons':H,'bootstrap':2000,'errors':errors,'results_path':str(O/'p1_horizon_model_results_all_seeds.csv'),'summary_path':str(O/'p1_horizon_model_summary.csv')}
(O/'p1_horizon_model_qc.json').write_text(json.dumps(qc,indent=2)); print(json.dumps(qc,indent=2)); print(summary.to_string(index=False))
