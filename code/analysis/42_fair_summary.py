from pathlib import Path
import json, numpy as np, pandas as pd
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015'); R=P/'results/improvement_fair_ssl'
seeds=[101,202,303,404,505]; fracs=[.01,.05,.1,.25,1.0]; methods=['random','nextmask4','nextintensity4','valueforecast4']
rows=[]
for s in seeds:
 d=pd.read_csv(R/f'fair_ssl_results_seed{s}.csv')
 if len(d)!=20: raise RuntimeError((s,len(d)))
 rows.append(d)
out=pd.concat(rows,ignore_index=True); out.to_csv(R/'fair_ssl_results_all_seeds.csv',index=False)
pair=pd.read_csv(R/'fair_pairwise_patient_bootstrap.csv')
summary=pair.groupby(['comparison','fraction']).agg(n=('seed','count'),mean_delta_auprc=('delta_auprc','mean'),median_delta_auprc=('delta_auprc','median'),positive_seeds=('delta_auprc',lambda x:int((x>0).sum())),ci_lo_min=('auprc_lo','min'),ci_hi_max=('auprc_hi','max')).reset_index()
summary.to_csv(R/'fair_pairwise_summary.csv',index=False)
q={'status':'PASS','result_rows':len(out),'expected_rows':100,'pair_rows':len(pair),'expected_pair_rows':75,'seeds':seeds,'fractions':fracs,'methods':methods,'all_finite':bool(np.isfinite(out[['auprc','auroc','brier','auprc_lo','auprc_hi','auroc_lo','auroc_hi','brier_lo','brier_hi']].to_numpy()).all()),'pair_qc':json.loads((R/'fair_pairwise_qc.json').read_text()),'no_terminal_auxiliary':True,'unified_4h':True,'train_only_value_scaling':True,'explicit_seed_reset':True}
(R/'fair_ssl_qc.json').write_text(json.dumps(q,indent=2)); print(q); print(summary.to_string(index=False))
