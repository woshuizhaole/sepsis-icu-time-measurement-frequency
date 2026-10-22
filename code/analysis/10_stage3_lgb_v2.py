#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from lightgbm import LGBMClassifier, early_stopping

P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015')
D=P/'data/processed'; R=P/'results'; LOG=P/'logs/analysis'
D.mkdir(exist_ok=True); R.mkdir(exist_ok=True); LOG.mkdir(exist_ok=True)
DYN=['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets']
BASE=['Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS']

def build_features():
    out=D/'stage3_p19_features_v2.parquet'
    if out.exists(): return pd.read_parquet(out)
    src=D/'stage3_p19_features.parquet'
    df=pd.read_parquet(src).sort_values(['patient_id','ICULOS']).reset_index(drop=True)
    mcols=[f'm0__{c}' for c in DYN]; c4cols=[f'm4__{c}' for c in DYN]
    pid=df['patient_id']; grp=df.groupby('patient_id',sort=False)
    keep=['patient_id','ICULOS']+BASE[:-1]+['SepsisLabel']+[f'v__{c}' for c in DYN]+mcols
    outdf=df[keep].copy()
    m0=df[mcols].to_numpy(dtype='float32',copy=False)
    m4=df[c4cols].to_numpy(dtype='float32',copy=False)
    # I: per-channel 4 h counts plus breadth, repeat count and valid-hour fraction.
    for j,c in enumerate(DYN): outdf[f'i_count4__{c}']=m4[:,j]
    outdf['i_breadth4']=(m4>0).sum(axis=1).astype('float32')
    outdf['i_repeat4']=np.maximum(m4-1.0,0).sum(axis=1).astype('float32')
    anyhour=m0.any(axis=1).astype('float32')
    outdf['i_valid_hours4']=(pd.Series(anyhour).groupby(pid,sort=False).rolling(4,min_periods=1).sum().reset_index(level=0,drop=True).to_numpy(dtype='float32')/4.0)
    # Mh: complete 12 hourly masks, lag 0=current; pre-admission history is zero.
    mh_parts=[]
    for lag in range(12):
        shifted=grp[mcols].shift(lag).fillna(0).astype('float32')
        shifted.columns=[f'mh__lag{lag}__{c}' for c in DYN]
        mh_parts.append(shifted)
    outdf=pd.concat([outdf]+mh_parts,axis=1)
    # R: causal recency in hours, capped at 12.
    rownum=grp.cumcount().astype('int64')
    rparts=[]
    for j,c in enumerate(DYN):
        last=rownum.where(df[mcols[j]]>0).groupby(pid,sort=False).ffill()
        rec=(rownum-last).fillna(12).clip(0,12).astype('float32').rename(f'r__{c}')
        rparts.append(rec)
    outdf=pd.concat([outdf]+rparts,axis=1)
    outdf.to_parquet(out,index=False,compression='zstd')
    return outdf

def feature_columns(cond):
    if cond=='V': return BASE+[f'v__{c}' for c in DYN]
    if cond=='M0': return BASE+[f'm0__{c}' for c in DYN]
    if cond=='Mh': return BASE+[f'mh__lag{l}__{c}' for l in range(12) for c in DYN]
    if cond=='I': return BASE+[f'i_count4__{c}' for c in DYN]+['i_breadth4','i_repeat4','i_valid_hours4']
    if cond=='R': return BASE+[f'r__{c}' for c in DYN]
    if cond=='V+I': return BASE+[f'v__{c}' for c in DYN]+[f'i_count4__{c}' for c in DYN]+['i_breadth4','i_repeat4','i_valid_hours4']
    if cond=='V+M0': return BASE+[f'v__{c}' for c in DYN]+[f'm0__{c}' for c in DYN]
    if cond=='V+M0+I': return feature_columns('V+M0')+[f'i_count4__{c}' for c in DYN]+['i_breadth4','i_repeat4','i_valid_hours4']
    if cond=='V+Mh': return BASE+[f'v__{c}' for c in DYN]+feature_columns('Mh')[len(BASE):]
    if cond=='V+Mh+I': return feature_columns('V+Mh')+[f'i_count4__{c}' for c in DYN]+['i_breadth4','i_repeat4','i_valid_hours4']
    raise KeyError(cond)

def run(seed, conditions):
    feat=build_features()
    win=pd.read_parquet(D/'cinc2019_windows.parquet',columns=['patient_id','anchor_hour','y_h6','next_window_available'])
    dat=win.merge(feat.rename(columns={'ICULOS':'anchor_hour'}),on=['patient_id','anchor_hour'],how='left',validate='one_to_one')
    dat['ICULOS']=dat['anchor_hour']
    partitions=json.loads((D/'partitions.json').read_text())['cinc2019'][str(seed)]
    dat['split']=dat.patient_id.map(partitions)
    if dat.split.isna().any(): raise RuntimeError('unassigned patients')
    out=[]
    for cond in conditions:
        cols=feature_columns(cond)
        tr=dat[dat.split=='train']; va=dat[dat.split=='validation']; te=dat[dat.split=='test']
        med=tr[cols].median(numeric_only=True)
        def clean(x): return x[cols].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0)
        Xtr=clean(tr); Xva=clean(va); Xte=clean(te)
        clf=LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=100,
                           subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=seed,
                           n_jobs=32,verbosity=-1)
        clf.fit(Xtr,tr.y_h6.astype(int),eval_set=[(Xva,va.y_h6.astype(int))],callbacks=[early_stopping(30,verbose=False)])
        p=clf.predict_proba(Xte)[:,1]; y=te.y_h6.astype(int).to_numpy()
        metrics={'seed':seed,'condition':cond,'n_train':len(tr),'n_val':len(va),'n_test':len(te),'positive_test':int(y.sum()),'best_iteration':int(clf.best_iteration_ or 0),
                 'auprc':float(average_precision_score(y,p)),'auroc':float(roc_auc_score(y,p)),'brier':float(brier_score_loss(y,p)),
                 'n_features':len(cols)}
        out.append(metrics)
        pred=pd.DataFrame({'patient_id':te.patient_id.to_numpy(),'anchor_hour':te.anchor_hour.to_numpy(),'y':y,'p':p})
        pred.to_parquet(R/f'stage3_lgb_v2_{cond.replace("+","_")}_seed{seed}_pred.parquet',index=False)
        clf.booster_.save_model(str(R/f'stage3_lgb_v2_{cond.replace("+","_")}_seed{seed}.txt'))
        print(metrics,flush=True)
    return out

ap=argparse.ArgumentParser()
ap.add_argument('--seeds',default='101')
ap.add_argument('--conditions',default='V,M0,Mh,I,V+I,V+M0,V+M0+I,V+Mh,V+Mh+I')
args=ap.parse_args()
allout=[]
for seed in [int(x) for x in args.seeds.split(',') if x]:
    allout.extend(run(seed,args.conditions.split(',')))
pd.DataFrame(allout).to_csv(R/'stage3_lgb_v2_results.csv',index=False)
print(json.dumps(allout,indent=2))

