import json
from pathlib import Path
import numpy as np
import pandas as pd
R=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015/results')
checks=[]
s3=pd.read_csv(R/'stage3_grud_cde_results.csv'); k3=['architecture','condition','seed']
checks += [('stage3_rows',len(s3)==20),('stage3_unique',s3.duplicated(k3).sum()==0),('stage3_finite',np.isfinite(s3[['auprc','auroc','brier','auprc_lo','auprc_hi','auroc_lo','auroc_hi','brier_lo','brier_hi']].to_numpy()).all()),('stage3_ci_order',(s3.auprc_lo<=s3.auprc_hi).all() and (s3.auroc_lo<=s3.auroc_hi).all() and (s3.brier_lo<=s3.brier_hi).all())]
s4=pd.read_csv(R/'stage4_calibration_alert_metrics.csv'); k4=['method','seed','fraction']; core=['prevalence','mean_pred','brier','calibration_intercept','calibration_slope','ece_10','mce_10']
checks += [('stage4_cal_rows',len(s4)==100),('stage4_cal_unique',s4.duplicated(k4).sum()==0),('stage4_cal_finite',np.isfinite(s4[core].to_numpy()).all()),('stage4_lead_nonestimable',int((s4.alert_n_true_alert_lead==0).sum())==9),('stage4_curve_rows',len(pd.read_csv(R/'stage4_alert_curve.csv'))==1500),('stage4_bins_rows',len(pd.read_csv(R/'stage4_calibration_bins.csv'))==845)]
of=pd.read_csv(R/'stage5_official_alert_metrics.csv'); checks += [('official_rows',len(of)==5),('official_finite',np.isfinite(of.select_dtypes(include=['number']).to_numpy()).all())]
tr=pd.read_csv(R/'stage6_transfer_results_all_seeds.csv'); kt=['seed','direction','task','cutoff_hour','mode']; req=['auprc','auroc','brier','auprc_lo','auprc_hi','auroc_lo','auroc_hi','brier_lo','brier_hi']
checks += [('transfer_rows',len(tr)==45),('transfer_unique',tr.duplicated(kt).sum()==0),('transfer_finite',np.isfinite(tr[req].to_numpy()).all())]
src=[json.loads((R/f'stage6_transfer_source_seed{s}.json').read_text()) for s in [101,202,303,404,505]]
checks += [('seed_specific_cache',all(x.get('p19_sequence_seed_specific') is True for x in src))]
m=pd.read_parquet(R/'metrics.parquet'); b=pd.read_csv(R/'bootstrap_ci.csv');
checks += [('metrics_rows',len(m)==2693),('bootstrap_rows',len(b)==1693),('metrics_finite',np.isfinite(m.value.to_numpy()).all()),('bootstrap_finite',np.isfinite(b[['estimate','lo','hi']].to_numpy()).all()),('bootstrap_order',(b.lo<=b.hi).all()),('metrics_unique',m.duplicated(['stage','model','seed','fraction','metric']).sum()==0),('bootstrap_unique',b.duplicated(['stage','model','seed','fraction','metric']).sum()==0)]
checks += [('invalid_basecache_excluded',(R/'stage6_transfer_invalid_basecache_20261017'/'INVALID_REASON.md').exists())]
for k,v in checks: print(k,bool(v))
print('status','PASS' if all(v for _,v in checks) else 'FAILED')
