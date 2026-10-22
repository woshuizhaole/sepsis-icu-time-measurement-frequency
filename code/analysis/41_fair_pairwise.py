from pathlib import Path
import itertools, json
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015'); R=P/'results/improvement_fair_ssl'
seeds=[101,202,303,404,505]; fracs=[0.01,0.05,0.10,0.25,1.0]; methods=['nextmask4','nextintensity4','valueforecast4']; B=2000
def metrics(y,p,w=None):
    return np.array([average_precision_score(y,p,sample_weight=w),roc_auc_score(y,p,sample_weight=w),brier_score_loss(y,p,sample_weight=w)],float)
def boot_pair(y,p_a,p_b,pids,seed):
    uniq,inv=np.unique(np.asarray(pids,dtype=object),return_inverse=True); n=len(uniq); rng=np.random.default_rng(seed)
    vals=[]
    for st in range(0,B,100):
        m=min(100,B-st); ss=rng.integers(0,n,size=(m,n))
        for s in ss:
            c=np.bincount(s,minlength=n); w=c[inv]
            try: vals.append(metrics(y,p_a,w)-metrics(y,p_b,w))
            except ValueError: vals.append([np.nan]*3)
    return np.asarray(vals,float)
rows=[]
for seed in seeds:
  for frac in fracs:
    paths={m:R/f'fair_pred_{m}_seed{seed}_f{frac:.2f}.parquet' for m in ['random']+methods}
    if not all(p.exists() for p in paths.values()): raise FileNotFoundError(paths)
    ds={m:pd.read_parquet(p) for m,p in paths.items()}
    for m in methods:
      a=ds[m]; b=ds['random']
      if not (a.patient_id.equals(b.patient_id) and a.y.equals(b.y)): raise RuntimeError((seed,frac,m,'order mismatch'))
      d=metrics(a.y.to_numpy(),a.p.to_numpy())-metrics(b.y.to_numpy(),b.p.to_numpy())
      boot=boot_pair(a.y.to_numpy(),a.p.to_numpy(),b.p.to_numpy(),a.patient_id.to_numpy(),seed+int(frac*1000)+sum(map(ord,m)))
      rows.append({'seed':seed,'fraction':frac,'comparison':f'{m}-random','n_test':len(a),'positive':int(a.y.sum()),'delta_auprc':d[0],'delta_auroc':d[1],'delta_brier':d[2],'auprc_lo':np.nanpercentile(boot[:,0],2.5),'auprc_hi':np.nanpercentile(boot[:,0],97.5),'auroc_lo':np.nanpercentile(boot[:,1],2.5),'auroc_hi':np.nanpercentile(boot[:,1],97.5),'brier_lo':np.nanpercentile(boot[:,2],2.5),'brier_hi':np.nanpercentile(boot[:,2],97.5)})
out=pd.DataFrame(rows); out.to_csv(R/'fair_pairwise_patient_bootstrap.csv',index=False)
qc={'rows':len(out),'expected':len(seeds)*len(fracs)*len(methods),'status':'PASS' if len(out)==75 and np.isfinite(out.select_dtypes('number').to_numpy()).all() and (out.auprc_lo<=out.auprc_hi).all() else 'FAILED','paired_unit':'patient within identical seed/fraction test cohort','bootstrap':B}
(R/'fair_pairwise_qc.json').write_text(json.dumps(qc,indent=2)); print(qc)
