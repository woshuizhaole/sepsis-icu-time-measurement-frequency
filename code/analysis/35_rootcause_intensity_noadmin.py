#!/usr/bin/env python3
"""Root-cause diagnostics: proxy proximity/intensity relation and no-admin controls.
Writes only results/rootcause_stage0_2 and logs/qc; never overwrites main results.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from lightgbm import LGBMClassifier, early_stopping
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

P=Path("/data/workplace/wyx/measurement_intensity_ssl_20261015"); D=P/"data/processed"; R=P/"results"; O=R/"rootcause_stage0_2"; O.mkdir(parents=True,exist_ok=True)
DYN=["HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2","BaseExcess","HCO3","FiO2","pH","PaCO2","SaO2","AST","BUN","Alkalinephos","Calcium","Chloride","Creatinine","Bilirubin_direct","Glucose","Lactate","Magnesium","Phosphate","Potassium","Bilirubin_total","TroponinI","Hct","Hgb","PTT","WBC","Fibrinogen","Platelets"]
SEED=101

def boot_metric(y,p,pids,B=500,seed=99101):
    u,inv=np.unique(np.asarray(pids),return_inverse=True); n=len(u); rng=np.random.default_rng(seed); vals=[]
    for _ in range(B):
        s=rng.integers(0,n,size=n); w=np.bincount(s,minlength=n)[inv]
        vals.append([average_precision_score(y,p,sample_weight=w),roc_auc_score(y,p,sample_weight=w),brier_score_loss(y,p,sample_weight=w)])
    a=np.asarray(vals); return {f"{name}_{q}":float(np.percentile(a[:,j],q)) for j,name in enumerate(["auprc","auroc","brier"]) for q in [2.5,97.5]}

t0=time.time()
# Build causal row table using existing Stage2 windows and v2 features.
win=pd.read_parquet(D/"cinc2019_windows.parquet")
feat_cols=["patient_id","ICULOS"]+[f"i_count4__{c}" for c in DYN]
feat=pd.read_parquet(D/"stage3_p19_features_v2.parquet",columns=feat_cols)
z=win.merge(feat.rename(columns={"ICULOS":"anchor_hour"}),on=["patient_id","anchor_hour"],how="left",validate="one_to_one")
ic=[f"i_count4__{c}" for c in DYN]
z["intensity_total4"]=z[ic].sum(axis=1).astype(float)
z["distance_to_proxy"]=z["first_positive_proxy"]-z["anchor_hour"]
z["source_set"]=z.patient_id.map(pd.read_parquet(D/"cinc2019_hourly/data.parquet",columns=["patient_id","source_set"]).drop_duplicates("patient_id").set_index("patient_id").source_set)
part=json.loads((D/"partitions.json").read_text())["cinc2019"][str(SEED)]; z["split"]=z.patient_id.map(part)
# Relationship table by proximity and target.
bins=[-np.inf,0,1,2,3,4,6,12,24,48,np.inf]
zb=z.copy(); zb["distance_bin"]=pd.cut(zb.distance_to_proxy,bins=bins,right=False,include_lowest=True)
rel=zb.groupby("distance_bin",observed=False).agg(n=("y_h6","size"),positive=("y_h6","sum"),mean_intensity=("intensity_total4","mean"),median_intensity=("intensity_total4","median"),mean_intensity_y1=("intensity_total4",lambda x: float(x[zb.loc[x.index,"y_h6"].to_numpy()==1].mean()) if (zb.loc[x.index,"y_h6"].to_numpy()==1).any() else np.nan),mean_intensity_y0=("intensity_total4",lambda x: float(x[zb.loc[x.index,"y_h6"].to_numpy()==0].mean()) if (zb.loc[x.index,"y_h6"].to_numpy()==0).any() else np.nan)).reset_index()
rel["rate"]=rel.positive/rel.n
rel.to_csv(O/"p19_intensity_proximity_relation.csv",index=False)
# Split/test metrics for raw intensity, anchor hour, and combined [intensity, hour].
train=z[z.split=="train"].copy(); test=z[z.split=="test"].copy(); ytr=train.y_h6.astype(int).to_numpy(); yte=test.y_h6.astype(int).to_numpy()
def rank_empirical(train_key,test_key):
    # rank/empirical target encoding fitted on train only, with stable fallback.
    q=pd.qcut(train_key,q=20,duplicates="drop"); rates=pd.Series(ytr).groupby(q,observed=True).mean()
    edges=q.cat.categories
    # use bin membership generated from categories; nearest category mean fallback
    cats=pd.cut(test_key,bins=np.r_[-np.inf,[x.right for x in edges]],right=True,include_lowest=True)
    p=cats.map(rates).astype(float).fillna(float(ytr.mean())).to_numpy()
    return p
pi=rank_empirical(train.intensity_total4,test.intensity_total4)
ph=rank_empirical(train.anchor_hour,test.anchor_hour)
rel_metrics=[{"predictor":"intensity_total4_empirical20","n_train":len(train),"n_test":len(test),"positive_test":int(yte.sum()),"auprc":float(average_precision_score(yte,pi)),"auroc":float(roc_auc_score(yte,pi)),"brier":float(brier_score_loss(yte,pi))},
{"predictor":"anchor_hour_empirical20","n_train":len(train),"n_test":len(test),"positive_test":int(yte.sum()),"auprc":float(average_precision_score(yte,ph)),"auroc":float(roc_auc_score(yte,ph)),"brier":float(brier_score_loss(yte,ph))}]
# logistic conditional association; scale only from training
sc=StandardScaler().fit(train[["intensity_total4","anchor_hour"]]); Xtr=sc.transform(train[["intensity_total4","anchor_hour"]]); Xte=sc.transform(test[["intensity_total4","anchor_hour"]]); lr=LogisticRegression(max_iter=300,class_weight=None,n_jobs=1).fit(Xtr,ytr); pp=lr.predict_proba(Xte)[:,1]
rm={"predictor":"logistic_intensity_plus_anchor_hour","n_train":len(train),"n_test":len(test),"positive_test":int(yte.sum()),"auprc":float(average_precision_score(yte,pp)),"auroc":float(roc_auc_score(yte,pp)),"brier":float(brier_score_loss(yte,pp)),"coef_intensity":float(lr.coef_[0,0]),"coef_anchor_hour":float(lr.coef_[0,1])}; rm.update(boot_metric(yte,pp,test.patient_id.to_numpy(),B=500,seed=SEED+7000)); rel_metrics.append(rm)
pd.DataFrame(rel_metrics).to_csv(O/"p19_intensity_proximity_metrics.csv",index=False)
# Intensity means by target in test and first-positive cohort.
means=z.groupby(["split","y_h6"]).intensity_total4.agg(["size","mean","median","std"]).reset_index(); means.to_csv(O/"p19_intensity_target_summary.csv",index=False)
pos=z[(z.first_positive_proxy>=0)&(z.distance_to_proxy>0)]
from scipy.stats import spearmanr
rho,pval=spearmanr(pos.distance_to_proxy.to_numpy(),pos.intensity_total4.to_numpy())
(O/"p19_intensity_spearman.json").write_text(json.dumps({"n_positive_patient_anchors":len(pos),"spearman_distance_intensity":float(rho),"p_value":float(pval)},indent=2))

# No-administrative/static controls. Exact LGBM config of Stage3 v2; only feature set changes.
full=pd.read_parquet(D/"stage3_p19_features_v2.parquet")
full=win.merge(full.rename(columns={"ICULOS":"anchor_hour"}),on=["patient_id","anchor_hour"],how="left",validate="one_to_one")
full["split"]=full.patient_id.map(part)
base=["Age","Gender","Unit1","Unit2","HospAdmTime","ICULOS"]
def cols(cond,variant):
    if variant=="current": b=base
    elif variant=="demographic_only": b=["Age","Gender"]
    elif variant=="clinical_only": b=[]
    else: raise ValueError(variant)
    if cond=="V": return b+[f"v__{c}" for c in DYN]
    if cond=="M0": return b+[f"m0__{c}" for c in DYN]
    if cond=="Mh": return b+[f"mh__lag{l}__{c}" for l in range(12) for c in DYN]
    if cond=="I": return b+[f"i_count4__{c}" for c in DYN]+["i_breadth4","i_repeat4","i_valid_hours4"]
    if cond=="V+I": return b+[f"v__{c}" for c in DYN]+[f"i_count4__{c}" for c in DYN]+["i_breadth4","i_repeat4","i_valid_hours4"]
    if cond=="V+M0": return b+[f"v__{c}" for c in DYN]+[f"m0__{c}" for c in DYN]
    if cond=="V+M0+I": return cols("V+M0",variant)+[f"i_count4__{c}" for c in DYN]+["i_breadth4","i_repeat4","i_valid_hours4"]
    if cond=="V+Mh": return b+[f"v__{c}" for c in DYN]+[f"mh__lag{l}__{c}" for l in range(12) for c in DYN]
    if cond=="V+Mh+I": return cols("V+Mh",variant)+[f"i_count4__{c}" for c in DYN]+["i_breadth4","i_repeat4","i_valid_hours4"]
    raise KeyError(cond)
conditions=["V","I","V+I","V+M0","V+M0+I"]
rows=[]
for variant in ["clinical_only"]:
    for cond in conditions:
        c=cols(cond,variant); tr=full[full.split=="train"]; va=full[full.split=="validation"]; te=full[full.split=="test"]
        med=tr[c].median(numeric_only=True)
        clean=lambda q:q[c].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0)
        clf=LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=100,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=SEED,n_jobs=4,verbosity=-1)
        clf.fit(clean(tr),tr.y_h6.astype(int),eval_set=[(clean(va),va.y_h6.astype(int))],callbacks=[early_stopping(30,verbose=False)])
        p=clf.predict_proba(clean(te))[:,1]; y=te.y_h6.astype(int).to_numpy()
        row={"variant":variant,"condition":cond,"n_features":len(c),"n_train":len(tr),"n_test":len(te),"positive_test":int(y.sum()),"best_iteration":int(clf.best_iteration_ or 0),"auprc":float(average_precision_score(y,p)),"auroc":float(roc_auc_score(y,p)),"brier":float(brier_score_loss(y,p))}
        row.update(boot_metric(y,p,te.patient_id.to_numpy(),B=500,seed=SEED+len(rows)+1)); rows.append(row)
        print(row,flush=True)
pd.DataFrame(rows).to_csv(O/"p19_no_admin_static_lgb_seed101.csv",index=False)
# direct deltas relative to current per condition
d=pd.DataFrame(rows)
d.to_csv(O/"p19_no_admin_static_lgb_seed101.csv",index=False)
summary={"runtime_seconds":time.time()-t0,"seed":SEED,"n_rows_relation":len(z),"outputs":str(O),"relation_metrics":rel_metrics}
(O/"p19_rootcause_diagnostic_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=str))
print(json.dumps(summary,indent=2,ensure_ascii=False,default=str))

