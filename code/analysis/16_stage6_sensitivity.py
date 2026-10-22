#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
from lightgbm import LGBMClassifier,early_stopping
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015');D=P/'data/processed';R=P/'results'
DYN=['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets']
BASE=['Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS']
def boot(y,p,pids,B,seed):
 u,inv=np.unique(pids,return_inverse=True);n=len(u);vals=[];rng=np.random.default_rng(seed)
 for st in range(0,B,100):
  for s in rng.integers(0,n,size=(min(100,B-st),n)):
   w=np.bincount(s,minlength=n)[inv]
   try:a=average_precision_score(y,p,sample_weight=w);au=roc_auc_score(y,p,sample_weight=w)
   except ValueError:a=np.nan;au=np.nan
   vals.append((a,au,brier_score_loss(y,p,sample_weight=w)))
 a=np.asarray(vals);return {'auprc_lo':float(np.nanpercentile(a[:,0],2.5)),'auprc_hi':float(np.nanpercentile(a[:,0],97.5)),'auroc_lo':float(np.nanpercentile(a[:,1],2.5)),'auroc_hi':float(np.nanpercentile(a[:,1],97.5)),'brier_lo':float(np.nanpercentile(a[:,2],2.5)),'brier_hi':float(np.nanpercentile(a[:,2],97.5))}
def lgb(Xtr,ytr,Xv,yv,Xte,seed):
 m=LGBMClassifier(n_estimators=300,learning_rate=.05,num_leaves=31,min_child_samples=100,subsample=.8,colsample_bytree=.8,reg_lambda=1,random_state=seed,n_jobs=8,verbosity=-1)
 m.fit(Xtr,ytr,eval_set=[(Xv,yv)],callbacks=[early_stopping(30,verbose=False)]);return m
def p19(seed,args):
 df=pd.read_parquet(D/'stage3_p19_features_v2.parquet');w=pd.read_parquet(D/'cinc2019_windows.parquet');part=json.loads((D/'partitions.json').read_text())['cinc2019'][str(seed)]
 cols=BASE+[f'v__{c}' for c in DYN]+[f'm0__{c}' for c in DYN]+[f'i_count4__{c}' for c in DYN]+['i_breadth4','i_repeat4','i_valid_hours4']
 dat=w.merge(df.rename(columns={'ICULOS':'anchor_hour'}),on=['patient_id','anchor_hour'],how='left',validate='one_to_one');dat['ICULOS']=dat.anchor_hour
 dat=dat[dat.next_window_available.astype(bool)].copy()
 out=[]
 for mode in args.modes.split(','):
  z=dat.copy()
  if mode=='admission_excluded':z=z[z.anchor_hour>4]
  elif mode=='embargo12':z=z[~((z.first_positive_proxy>=0)&((z.first_positive_proxy-z.anchor_hour)<=12))]
  use=cols.copy()
  if mode=='drop_lactate_wbc':
   keep=[c for c in DYN if c not in {'Lactate','WBC','Creatinine','Bilirubin_total'}]
   use=BASE+[f'v__{c}' for c in keep]+[f'm0__{c}' for c in keep]+[f'i_count4__{c}' for c in keep]+['i_breadth4_drop','i_repeat4_drop','i_valid_hours4_drop']
   counts=z[[f'i_count4__{c}' for c in keep]].to_numpy(dtype='float32')
   z['i_breadth4_drop']=(counts>0).sum(axis=1).astype('float32'); z['i_repeat4_drop']=np.maximum(counts-1,0).sum(axis=1).astype('float32')
   mh=np.stack([z[[f'mh__lag{l}__{c}' for c in keep]].to_numpy(dtype='float32') for l in range(4)],axis=1)
   z['i_valid_hours4_drop']=(mh.sum(axis=2)>0).sum(axis=1).astype('float32')/4.0
  tr=z[z.patient_id.map(part)=='train'];va=z[z.patient_id.map(part)=='validation'];te=z[z.patient_id.map(part)=='test']
  if len(te)==0 or te.y_h6.sum()==0 or len(tr)==0 or tr.y_h6.sum()==0:
   out.append({'seed':seed,'mode':mode,'estimable':False,'n_train':len(tr),'n_val':len(va),'n_test':len(te),'positive_test':int(te.y_h6.sum())});continue
  med=tr[use].median(numeric_only=True)
  clean=lambda q:q[use].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0)
  m=lgb(clean(tr),tr.y_h6.astype(int),clean(va),va.y_h6.astype(int),clean(te),seed);p=m.predict_proba(clean(te))[:,1];y=te.y_h6.astype(int).to_numpy();row={'seed':seed,'mode':mode,'estimable':True,'n_train':len(tr),'n_val':len(va),'n_test':len(te),'positive_test':int(y.sum()),'n_features':len(use),'auprc':float(average_precision_score(y,p)),'auroc':float(roc_auc_score(y,p)),'brier':float(brier_score_loss(y,p))};row.update(boot(y,p,te.patient_id.to_numpy(),args.bootstrap,seed+1000));out.append(row)
 return out
def p12(seed,args):
 df=pd.read_parquet(D/'cinc2012_hourly/data.parquet'); lm=pd.read_parquet(D/'cinc2012_landmarks.parquet'); part=json.loads((D/'partitions.json').read_text())['cinc2012'][str(seed)]
 value=[c for c in df.columns if c.startswith('value__')]; count=[c for c in df.columns if c.startswith('event_count__')]; static=['Age','Gender','Height','ICUType','Weight']; use=static+value+count
 groups={pid:g.sort_values('icu_hour') for pid,g in df.groupby('patient_id',sort=False)}; records=[]
 for r in lm.itertuples(index=False):
  if not int(r.landmark_available): continue
  g=groups.get(r.patient_id); hist=g[g.icu_hour<float(r.cutoff_hour)] if g is not None else g
  rec={'patient_id':r.patient_id,'cutoff_hour':int(r.cutoff_hour),'mortality_label':int(r.mortality_label),'split':part.get(r.patient_id),'n_pre_cutoff_rows':0 if hist is None else len(hist),'dynamic_zero_history':int(hist is None or len(hist)==0 or hist[value+count].fillna(0).to_numpy().sum()==0)}
  if g is None or hist is None or len(hist)==0:
   for c in static+value+count: rec[c]=np.nan
  else:
   first=g.iloc[0]
   for c in static: rec[c]=first[c]
   for c in value:
    q=hist[c].dropna(); rec[c]=q.iloc[-1] if len(q) else np.nan
   for c in count: rec[c]=float(hist[c].fillna(0).sum())
  records.append(rec)
 zall=pd.DataFrame(records);out=[]
 for cutoff in [12,24]:
  z=zall[zall.cutoff_hour==cutoff]; tr=z[z.split=='train'];va=z[z.split=='validation'];te=z[z.split=='test'];med=tr[use].median(numeric_only=True);clean=lambda q:q[use].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0);m=lgb(clean(tr),tr.mortality_label.astype(int),clean(va),va.mortality_label.astype(int),clean(te),seed);p=m.predict_proba(clean(te))[:,1];y=te.mortality_label.astype(int).to_numpy();row={'seed':seed,'dataset':'cinc2012','cutoff_hour':cutoff,'n_train':len(tr),'n_val':len(va),'n_test':len(te),'positive_test':int(y.sum()),'n_zero_history':int(te.dynamic_zero_history.sum()),'auprc':float(average_precision_score(y,p)),'auroc':float(roc_auc_score(y,p)),'brier':float(brier_score_loss(y,p))};row.update(boot(y,p,te.patient_id.to_numpy(),args.bootstrap,seed+2000));out.append(row)
 return out
def main(args):
 out=p19(args.seed,args.modes.split(',')[0] and args) if args.dataset=='p19' else p12(args.seed,args)
 pd.DataFrame(out).to_csv(R/f'stage6_{args.dataset}_results_seed{args.seed}.csv',index=False);print(out,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['p19','p12'],required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--modes',default='admission_excluded,drop_lactate_wbc,embargo12');ap.add_argument('--bootstrap',type=int,default=2000);main(ap.parse_args())

