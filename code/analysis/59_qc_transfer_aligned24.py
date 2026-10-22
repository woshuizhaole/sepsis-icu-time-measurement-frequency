from pathlib import Path
import json,numpy as np,pandas as pd
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015'); R=P/'results/stage6_transfer_aligned24'
seeds=[101,202,303,404,505]; modes=['frozen','finetune','random']; errors=[]
files=sorted(R.glob('stage6_transfer_results_seed*.csv')); d=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
if len(d)!=45: errors.append(f'rows:{len(d)}')
if d.duplicated(['seed','direction','cutoff_hour','mode']).any(): errors.append('duplicate')
num=[c for c in ["n_train","n_val","n_test","positive_test","auprc","auroc","brier","auprc_lo","auprc_hi","auroc_lo","auroc_hi","brier_lo","brier_hi"] if c in d.columns]
if not np.isfinite(d[num].to_numpy(float)).all(): errors.append("nonfinite_metrics")
d.to_csv(R/'stage6_transfer_aligned24_all_seeds.csv',index=False)
summary=(d.groupby(['direction','cutoff_hour','mode'],dropna=False,as_index=False).agg(n=('seed','count'),auprc_median=('auprc','median'),auprc_q1=('auprc',lambda x:x.quantile(.25)),auprc_q3=('auprc',lambda x:x.quantile(.75)),auroc_median=('auroc','median'),brier_median=('brier','median')))
summary.to_csv(R/'stage6_transfer_aligned24_summary.csv',index=False)
qc={'status':'PASS' if len(d)==45 and not errors else 'FAILED','rows':int(len(d)),'expected_rows':45,'seeds':seeds,'modes':modes,'source_hours':24,'bootstrap':2000,'errors':errors,'result_path':str(R/'stage6_transfer_aligned24_all_seeds.csv'),'summary_path':str(R/'stage6_transfer_aligned24_summary.csv')}
(R/'stage6_transfer_aligned24_qc.json').write_text(json.dumps(qc,indent=2)); print(json.dumps(qc,indent=2)); print(summary.to_string(index=False))
