#!/usr/bin/env python3
from pathlib import Path
import json,numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
from lightgbm import LGBMClassifier,early_stopping
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015');D=P/'data/processed';R=P/'results'
DYN=['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets'];BASE=['Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS']
def boot(y,p,pids,B,seed):
 u,inv=np.unique(pids,return_inverse=True);n=len(u);rng=np.random.default_rng(seed);v=[]
 for st in range(0,B,100):
  for s in rng.integers(0,n,size=(min(100,B-st),n)):
   w=np.bincount(s,minlength=n)[inv];v.append((average_precision_score(y,p,sample_weight=w),roc_auc_score(y,p,sample_weight=w),brier_score_loss(y,p,sample_weight=w)))
 a=np.asarray(v);return {'auprc_lo':float(np.percentile(a[:,0],2.5)),'auprc_hi':float(np.percentile(a[:,0],97.5)),'auroc_lo':float(np.percentile(a[:,1],2.5)),'auroc_hi':float(np.percentile(a[:,1],97.5)),'brier_lo':float(np.percentile(a[:,2],2.5)),'brier_hi':float(np.percentile(a[:,2],97.5))}
def main(seed=101):
 df=pd.read_parquet(D/'stage3_p19_features_v2.parquet');w=pd.read_parquet(D/'cinc2019_windows.parquet');part=json.loads((D/'partitions.json').read_text())['cinc2019'][str(seed)]
 cols=BASE+[f'v__{c}' for c in DYN]+[f'm0__{c}' for c in DYN]+[f'i_count4__{c}' for c in DYN]+['i_breadth4','i_repeat4','i_valid_hours4'];ic=[f'i_count4__{c}' for c in DYN]+['i_breadth4','i_repeat4','i_valid_hours4']
 dat=w.merge(df.rename(columns={'ICULOS':'anchor_hour'}),on=['patient_id','anchor_hour'],how='left',validate='one_to_one');dat['ICULOS']=dat.anchor_hour;rng=np.random.default_rng(20261016);eligible_sets={}
 for split in ['train','validation','test']:
  base=dat[dat.patient_id.map(part)==split];sizes=base.groupby('patient_id').size();eligible=[p for p,n in sizes.items() if n>1];eligible_sets[split]=set(eligible);sub=base[base.patient_id.isin(eligible)];by={pid:g.sort_values('anchor_hour').index.to_numpy() for pid,g in sub.groupby('patient_id',sort=False)};buckets={}
  for pid,idx in by.items():buckets.setdefault(len(idx),[]).append(pid)
  for n,pids in buckets.items():
   arr=np.array(sorted(pids),dtype=object);perm=np.roll(arr,1)
   for target,source in zip(arr,perm):dat.loc[by[target],ic]=dat.loc[by[source],ic].to_numpy()
 mask=dat.patient_id.map(lambda p: p in eligible_sets.get(part.get(p),set())).to_numpy();dat=dat[mask].copy()
 tr=dat[dat.patient_id.map(part)=='train'];va=dat[dat.patient_id.map(part)=='validation'];te=dat[dat.patient_id.map(part)=='test'];med=tr[cols].median(numeric_only=True);clean=lambda q:q[cols].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0)
 m=LGBMClassifier(n_estimators=300,learning_rate=.05,num_leaves=31,min_child_samples=100,subsample=.8,colsample_bytree=.8,reg_lambda=1,random_state=seed,n_jobs=16,verbosity=-1);m.fit(clean(tr),tr.y_h6.astype(int),eval_set=[(clean(va),va.y_h6.astype(int))],callbacks=[early_stopping(30,verbose=False)]);p=m.predict_proba(clean(te))[:,1];y=te.y_h6.to_numpy();o={'seed':seed,'negative_control':'patient_level_equal_length_intensity_trajectory_derangement','eligible_test_patients':int(te.patient_id.nunique()),'excluded_singleton_test_patients':int(len(set(dat[dat.patient_id.map(part)=='test'].patient_id.unique()))-te.patient_id.nunique()),'n_test':len(te),'positive_test':int(y.sum()),'auprc':float(average_precision_score(y,p)),'auroc':float(roc_auc_score(y,p)),'brier':float(brier_score_loss(y,p))};o.update(boot(y,p,te.patient_id.to_numpy(),2000,seed+9000));(R/'stage7_negative_control.json').write_text(json.dumps(o,indent=2));print(o)
if __name__=='__main__':main()

