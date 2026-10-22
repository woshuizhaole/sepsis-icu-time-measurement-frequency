from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json, os
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015'); R=P/'results/improvement_fair_ssl'; B=2000

def met(y,p,w): return np.array([average_precision_score(y,p,sample_weight=w),roc_auc_score(y,p,sample_weight=w),brier_score_loss(y,p,sample_weight=w)],float)
def one(task):
 seed,frac,m=task
 a=pd.read_parquet(R/f'fair_pred_{m}_seed{seed}_f{frac:.2f}.parquet'); b=pd.read_parquet(R/f'fair_pred_random_seed{seed}_f{frac:.2f}.parquet')
 if not (a.patient_id.equals(b.patient_id) and a.y.equals(b.y)): raise RuntimeError((task,'order'))
 y=a.y.to_numpy(); pa=a.p.to_numpy(); pb=b.p.to_numpy(); pid=a.patient_id.to_numpy(dtype=object)
 point=met(y,pa,np.ones(len(y)))-met(y,pb,np.ones(len(y)))
 uniq,inv=np.unique(pid,return_inverse=True); n=len(uniq); rng=np.random.default_rng(seed+int(frac*1000)+sum(map(ord,m)))
 vals=[]
 for st in range(0,B,100):
  k=min(100,B-st); ss=rng.integers(0,n,size=(k,n))
  for s in ss:
   w=np.bincount(s,minlength=n)[inv]
   vals.append(met(y,pa,w)-met(y,pb,w))
 z=np.asarray(vals,float)
 return {'seed':seed,'fraction':frac,'comparison':m+'-random','n_test':len(y),'positive':int(y.sum()),'delta_auprc':point[0],'delta_auroc':point[1],'delta_brier':point[2],'auprc_lo':np.nanpercentile(z[:,0],2.5),'auprc_hi':np.nanpercentile(z[:,0],97.5),'auroc_lo':np.nanpercentile(z[:,1],2.5),'auroc_hi':np.nanpercentile(z[:,1],97.5),'brier_lo':np.nanpercentile(z[:,2],2.5),'brier_hi':np.nanpercentile(z[:,2],97.5)}

def main():
 tasks=[(s,f,m) for s in [101,202,303,404,505] for f in [0.01,0.05,0.10,0.25,1.0] for m in ['nextmask4','nextintensity4','valueforecast4']]
 rows=[]
 with ProcessPoolExecutor(max_workers=2) as ex:
  fs={ex.submit(one,t):t for t in tasks}
  for i,f in enumerate(as_completed(fs),1):
   rows.append(f.result()); print(i,'/',len(tasks),fs[f],flush=True)
 out=pd.DataFrame(rows).sort_values(['seed','fraction','comparison']); out.to_csv(R/'fair_pairwise_patient_bootstrap.csv',index=False)
 qc={'rows':len(out),'expected':75,'status':'PASS' if len(out)==75 and np.isfinite(out.select_dtypes('number').to_numpy()).all() and (out.auprc_lo<=out.auprc_hi).all() else 'FAILED','paired_unit':'patient within identical seed/fraction test cohort','bootstrap':B,'workers':4}
 (R/'fair_pairwise_qc.json').write_text(json.dumps(qc,indent=2)); print(qc)
if __name__=='__main__': main()
