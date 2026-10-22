#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from lightgbm import LGBMClassifier, early_stopping

P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015')
D=P/'data/processed'; R=P/'results'; LOG=P/'logs/analysis'
D.mkdir(exist_ok=True); R.mkdir(exist_ok=True); LOG.mkdir(exist_ok=True)
DYN=['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets']
STATIC=['Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS']

def build_features():
    out=D/'stage3_p19_features.parquet'
    if out.exists(): return pd.read_parquet(out)
    df=pd.read_parquet(D/'cinc2019_hourly/data.parquet')
    df=df.sort_values(['patient_id','ICULOS']).reset_index(drop=True)
    for c in DYN:
        df[c]=pd.to_numeric(df[c],errors='coerce')
        df[f'm0__{c}']=df[c].notna().astype('int8')
        df[f'm12__{c}']=df.groupby('patient_id')[f'm0__{c}'].transform(lambda x:x.rolling(12,min_periods=1).sum()).astype('float32')
        df[f'm4__{c}']=df.groupby('patient_id')[f'm0__{c}'].transform(lambda x:x.rolling(4,min_periods=1).sum()).astype('float32')
        df[f'v__{c}']=df.groupby('patient_id')[c].ffill()
    keep=['patient_id','ICULOS']+STATIC[:-1]+['ICULOS','SepsisLabel']
    keep += [f'v__{c}' for c in DYN] + [f'm0__{c}' for c in DYN] + [f'm12__{c}' for c in DYN] + [f'm4__{c}' for c in DYN]
    # avoid duplicate ICULOS
    keep=list(dict.fromkeys(keep))
    df=df[keep]
    df.to_parquet(out,index=False,compression='zstd')
    return df

def feature_columns(cond):
    base=['Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS']
    if cond=='V': return base+[f'v__{c}' for c in DYN]
    if cond=='M0': return base+[f'm0__{c}' for c in DYN]
    if cond=='Mh': return base+[f'm12__{c}' for c in DYN]
    if cond=='I': return base+[f'm4__{c}' for c in DYN]
    if cond=='R': return base+[f'm12__{c}' for c in DYN]
    if cond=='V+I': return base+[f'v__{c}' for c in DYN]+[f'm4__{c}' for c in DYN]
    if cond=='V+M0': return base+[f'v__{c}' for c in DYN]+[f'm0__{c}' for c in DYN]
    if cond=='V+M0+I': return base+[f'v__{c}' for c in DYN]+[f'm0__{c}' for c in DYN]+[f'm4__{c}' for c in DYN]
    if cond=='V+Mh': return base+[f'v__{c}' for c in DYN]+[f'm12__{c}' for c in DYN]
    if cond=='V+Mh+I': return base+[f'v__{c}' for c in DYN]+[f'm12__{c}' for c in DYN]+[f'm4__{c}' for c in DYN]
    raise KeyError(cond)

def run(seed, conditions, max_rows=None):
    feat=build_features()
    win=pd.read_parquet(D/'cinc2019_windows.parquet',columns=['patient_id','anchor_hour','y_h6','next_window_available'])
    feat=feat.rename(columns={'ICULOS':'anchor_hour'})
    dat=win.merge(feat,on=['patient_id','anchor_hour'],how='left',validate='one_to_one')
    partitions=json.loads((D/'partitions.json').read_text())['cinc2019'][str(seed)]
    dat['split']=dat.patient_id.map(partitions)
    if dat.split.isna().any(): raise RuntimeError('unassigned patients')
    if max_rows and len(dat)>max_rows:
        rng=np.random.default_rng(seed)
        # retain all positives and sample negatives deterministically
        pos=dat.y_h6==1; p=dat[pos]; n=dat[~pos].sample(max(1,max_rows-len(p)),random_state=seed)
        dat=pd.concat([p,n]).sample(frac=1,random_state=seed)
    out=[]
    for cond in conditions:
        cols=feature_columns(cond)
        cols=[c for c in cols if c in dat.columns]
        tr=dat[dat.split=='train']; va=dat[dat.split=='validation']; te=dat[dat.split=='test']
        med=tr[cols].median(numeric_only=True)
        Xtr=tr[cols].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0)
        Xva=va[cols].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0)
        Xte=te[cols].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0)
        clf=LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=100,
                           subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=seed,
                           n_jobs=16,verbosity=-1)
        clf.fit(Xtr,tr.y_h6.astype(int),eval_set=[(Xva,va.y_h6.astype(int))],callbacks=[early_stopping(30,verbose=False)])
        p=clf.predict_proba(Xte)[:,1]; y=te.y_h6.astype(int).to_numpy()
        metrics={'seed':seed,'condition':cond,'n_train':len(tr),'n_val':len(va),'n_test':len(te),'positive_test':int(y.sum()),'best_iteration':int(clf.best_iteration_ or 0),
                 'auprc':float(average_precision_score(y,p)),'auroc':float(roc_auc_score(y,p)),'brier':float(brier_score_loss(y,p))}
        out.append(metrics)
        pred=pd.DataFrame({'patient_id':te.patient_id.to_numpy(),'anchor_hour':te.anchor_hour.to_numpy(),'y':y,'p':p})
        pred.to_parquet(R/f'stage3_lgb_{cond.replace("+","_")}_seed{seed}_pred.parquet',index=False)
        clf.booster_.save_model(str(R/f'stage3_lgb_{cond.replace("+","_")}_seed{seed}.txt'))
        print(metrics,flush=True)
    return out

ap=argparse.ArgumentParser(); ap.add_argument('--seeds',default='101'); ap.add_argument('--conditions',default='V,M0,Mh,I,V+M0,V+M0+I,V+Mh,V+Mh+I'); ap.add_argument('--max-rows',type=int,default=0); args=ap.parse_args()
allout=[]
for seed in [int(x) for x in args.seeds.split(',') if x]:
 allout.extend(run(seed,args.conditions.split(','),args.max_rows or None))
pd.DataFrame(allout).to_csv(R/'stage3_lgb_results.csv',index=False)
print(json.dumps(allout,indent=2))
