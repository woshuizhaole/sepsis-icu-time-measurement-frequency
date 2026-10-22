#!/usr/bin/env python3
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
from lightgbm import LGBMClassifier,early_stopping
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015'); D=P/'data/processed'; O=P/'results/rootcause_stage6'; O.mkdir(parents=True,exist_ok=True)
DYN=['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets']
BASE=['Age','Gender','Unit1','Unit2','HospAdmTime','anchor_hour']
USE=BASE+[f'v__{c}' for c in DYN]+[f'm0__{c}' for c in DYN]+[f'i_count4__{c}' for c in DYN]+['i_breadth4','i_repeat4','i_valid_hours4']
H=[1,4,6,8,12,24]
def boot(y,p,pid,B,seed):
 u,inv=np.unique(pid,return_inverse=True); n=len(u); rng=np.random.default_rng(seed); vals=[]
 for st in range(0,B,100):
  for s in rng.integers(0,n,size=(min(100,B-st),n)):
   w=np.bincount(s,minlength=n)[inv]
   try: vals.append((average_precision_score(y,p,sample_weight=w),roc_auc_score(y,p,sample_weight=w),brier_score_loss(y,p,sample_weight=w)))
   except ValueError: vals.append((np.nan,np.nan,np.nan))
 a=np.asarray(vals)
 return {f'{m}_{q}':float(np.nanpercentile(a[:,j],q)) for j,m in enumerate(['auprc','auroc','brier']) for q in [2.5,97.5]}
def main(seed):
 t0=time.time()
 hourly=pd.read_parquet(D/'cinc2019_hourly/data.parquet',columns=['patient_id','ICULOS','SepsisLabel'])
 fp=hourly.loc[hourly.SepsisLabel.astype(int)>0].groupby('patient_id').ICULOS.min().to_dict(); maxh=hourly.groupby('patient_id').ICULOS.max().to_dict()
 w=pd.read_parquet(D/'cinc2019_windows.parquet',columns=['patient_id','anchor_hour','next_window_available'])
 feat=pd.read_parquet(D/'stage3_p19_features_v2.parquet',columns=['patient_id','ICULOS']+[c for c in USE if c!='anchor_hour']).rename(columns={'ICULOS':'anchor_hour'})
 z=w.merge(feat,on=['patient_id','anchor_hour'],how='left',validate='one_to_one'); z['first_positive_proxy']=z.patient_id.map(fp).fillna(-1).astype(int); z['max_hour']=z.patient_id.map(maxh)
 part=json.loads((D/'partitions.json').read_text())['cinc2019'][str(seed)]
 out=[]
 for Hh in H:
  q=z[((z.first_positive_proxy<0)|(z.anchor_hour<z.first_positive_proxy)) & (z.max_hour>=z.anchor_hour+Hh)].copy()
  q['y_h']=((q.first_positive_proxy>=0)&(q.first_positive_proxy>q.anchor_hour)&(q.first_positive_proxy<=q.anchor_hour+Hh)).astype(int)
  tr=q[q.patient_id.map(part)=='train']; va=q[q.patient_id.map(part)=='validation']; te=q[q.patient_id.map(part)=='test']
  if len(te)==0 or te.y_h.sum()==0: out.append({'seed':seed,'horizon_hours':Hh,'estimable':False,'n_train':len(tr),'n_val':len(va),'n_test':len(te),'positive_test':int(te.y_h.sum())}); continue
  med=tr[USE].median(numeric_only=True)
  clean=lambda d:d[USE].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0)
  model=LGBMClassifier(n_estimators=300,learning_rate=.05,num_leaves=31,min_child_samples=100,subsample=.8,colsample_bytree=.8,reg_lambda=1.0,random_state=seed,n_jobs=4,verbosity=-1)
  model.fit(clean(tr),tr.y_h.astype(int),eval_set=[(clean(va),va.y_h.astype(int))],callbacks=[early_stopping(30,verbose=False)])
  y=te.y_h.astype(int).to_numpy(); p=model.predict_proba(clean(te))[:,1]; row={'seed':seed,'horizon_hours':Hh,'estimand':'pre_first_positive_complete_H_window','estimable':True,'n_features':len(USE),'n_train':len(tr),'n_val':len(va),'n_test':len(te),'positive_test':int(y.sum()),'availability_rate':float(len(q)/len(z)),'best_iteration':int(model.best_iteration_ or 0),'auprc':float(average_precision_score(y,p)),'auroc':float(roc_auc_score(y,p)),'brier':float(brier_score_loss(y,p))}; row.update(boot(y,p,te.patient_id.to_numpy(),2000,seed+Hh*100)); out.append(row)
  print(seed,Hh,out[-1].get('auprc'),flush=True)
 pd.DataFrame(out).to_csv(O/f'p1_horizon_model_results_seed{seed}.csv',index=False); (O/f'p1_horizon_model_seed{seed}.json').write_text(json.dumps({'seed':seed,'runtime_seconds':time.time()-t0,'rows':len(out),'horizons':H,'bootstrap':2000},indent=2)); print(json.dumps({'seed':seed,'runtime_seconds':time.time()-t0,'rows':len(out)}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); main(ap.parse_args().seed)
