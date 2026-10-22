import json,os,sys
from pathlib import Path
import numpy as np,pandas as pd
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015');sys.path.insert(0,str(P/'external/evaluation-2019'))
from evaluate_sepsis_score import compute_prediction_utility
def read(path,header,sep='|'):return pd.read_csv(path,sep=sep)[header].to_numpy()
rows=[];rng=np.random.default_rng(20261016)
for d in sorted((P/'results').glob('stage5_utility_seed*/')):
 seed=int(d.name.split('seed')[-1]);ld=d/'labels';pdst=d/'predictions';obs=[];best=[];ina=[];npids=[]
 for lf in sorted(ld.glob('*.psv')):
  pid=lf.stem;lab=read(lf,'SepsisLabel');pr=pd.read_csv(pdst/f'{pid}.psv',sep='|');pred=pr.PredictedLabel.to_numpy()
  obs.append(compute_prediction_utility(lab,pred))
  if np.any(lab):
   ts=int(np.argmax(lab))-(-6);bp=np.zeros(len(lab),int);bp[max(0,ts-12):min(ts+3+1,len(lab))]=1
  else:bp=np.zeros(len(lab),int)
  best.append(compute_prediction_utility(lab,bp));ina.append(compute_prediction_utility(lab,np.zeros(len(lab),int)));npids.append(pid)
 obs=np.asarray(obs);best=np.asarray(best);ina=np.asarray(ina);n=len(obs);vals=[]
 for st in range(0,2000,100):
  for s in rng.integers(0,n,size=(min(100,2000-st),n)):
   vals.append((obs[s].sum()-ina[s].sum())/(best[s].sum()-ina[s].sum()))
 vals=np.asarray(vals);rows.append({'seed':seed,'utility_point':float((obs.sum()-ina.sum())/(best.sum()-ina.sum())),'utility_lo':float(np.percentile(vals,2.5)),'utility_hi':float(np.percentile(vals,97.5)),'n_patients':n})
pd.DataFrame(rows).to_csv(P/'results/stage5_official_utility_bootstrap.csv',index=False);print(rows)

