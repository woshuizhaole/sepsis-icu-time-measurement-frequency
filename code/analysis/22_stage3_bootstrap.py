import argparse
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015');R=P/'results'
def boot(y,p,pids,B,seed):
 u,inv=np.unique(pids,return_inverse=True);n=len(u);rng=np.random.default_rng(seed);v=[]
 for st in range(0,B,100):
  for s in rng.integers(0,n,size=(min(100,B-st),n)):
   w=np.bincount(s,minlength=n)[inv]
   v.append((average_precision_score(y,p,sample_weight=w),roc_auc_score(y,p,sample_weight=w),brier_score_loss(y,p,sample_weight=w)))
 a=np.asarray(v);return [float(np.percentile(a[:,j],q)) for j in range(3) for q in [2.5,97.5]]
def main(seed,condition=None):
 rows=[]
 files=sorted(R.glob(f'stage3_lgb_v2_*_seed{seed}_pred.parquet'))
 if condition: files=[f for f in files if f'stage3_lgb_v2_{condition}_seed{seed}_pred.parquet' in str(f)]
 for f in files:
  cond=f.name.split('stage3_lgb_v2_')[1].split(f'_seed{seed}')[0].replace('_','+')
  d=pd.read_parquet(f);ci=boot(d.y.to_numpy(),d.p.to_numpy(),d.patient_id.to_numpy(),2000,seed+7000)
  rows.append({'seed':seed,'condition':cond,'auprc_lo':ci[0],'auprc_hi':ci[1],'auroc_lo':ci[2],'auroc_hi':ci[3],'brier_lo':ci[4],'brier_hi':ci[5]})
 pd.DataFrame(rows).to_csv(R/f'stage3_lgb_v2_bootstrap_seed{seed}_{condition or "all"}.csv',index=False);print(seed,len(rows),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--condition',default=None);args=ap.parse_args();main(args.seed,args.condition)

