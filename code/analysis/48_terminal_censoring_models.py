#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015')
D=P/'data/processed'; R=P/'results'; O=R/'improvement_fair_ssl/terminal_censoring'; O.mkdir(parents=True,exist_ok=True)
DYN=['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets']
MODEL_FEATURES=['anchor_hour']+[f'v__{c}' for c in DYN]+[f'm0__{c}' for c in DYN]+[f'i_count4__{c}' for c in DYN]+['i_breadth4','i_repeat4','i_valid_hours4']
METHODS=['random','nextmask4','nextintensity4','valueforecast4']
FRACS=[0.01,0.05,0.10,0.25,1.0]

def safe_metric(y,p,w=None):
    try: ap=float(average_precision_score(y,p,sample_weight=w)); auc=float(roc_auc_score(y,p,sample_weight=w)); br=float(brier_score_loss(y,p,sample_weight=w))
    except ValueError: ap=auc=br=float('nan')
    return ap,auc,br

def ess(w):
    w=np.asarray(w,dtype=float)
    return float((w.sum()**2)/max((w*w).sum(),1e-12))

def weight_stats(w):
    w=np.asarray(w,dtype=float)
    return {'weight_min':float(np.min(w)),'weight_p50':float(np.percentile(w,50)),'weight_p95':float(np.percentile(w,95)),'weight_p99':float(np.percentile(w,99)),'weight_max':float(np.max(w)),'effective_sample_size':ess(w)}

def main(seed):
    t0=time.time()
    win=pd.read_parquet(D/'cinc2019_windows.parquet',columns=['patient_id','anchor_hour','y_h6','next_window_available'])
    feat=pd.read_parquet(D/'stage3_p19_features_v2.parquet',columns=['patient_id','ICULOS']+MODEL_FEATURES[1:])
    feat=feat.rename(columns={'ICULOS':'anchor_hour'})
    z=win.merge(feat,on=['patient_id','anchor_hour'],how='left',validate='one_to_one',sort=False)
    part=json.loads((D/'partitions.json').read_text())['cinc2019'][str(seed)]
    z['split']=z.patient_id.map(part)
    tr=z[z.split=='train'].reset_index(drop=True); va=z[z.split=='validation'].reset_index(drop=True); te=z[z.split=='test'].reset_index(drop=True)
    med=tr[MODEL_FEATURES].replace([np.inf,-np.inf],np.nan).median(numeric_only=True)
    def clean(d): return d[MODEL_FEATURES].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0)
    Xtr=clean(tr); Xva=clean(va); Xte=clean(te); ytr=tr.next_window_available.astype(int).to_numpy(); yva=va.next_window_available.astype(int).to_numpy(); yte=te.next_window_available.astype(int).to_numpy()
    # Empirical hour-only model, estimated on train only.
    hour_tab=tr.groupby('anchor_hour').next_window_available.agg(['sum','count'])
    p_hour_tr=tr.anchor_hour.map((hour_tab['sum']+1)/(hour_tab['count']+2)).fillna(ytr.mean()).to_numpy()
    p_hour_va=va.anchor_hour.map((hour_tab['sum']+1)/(hour_tab['count']+2)).fillna(ytr.mean()).to_numpy()
    p_hour_te=te.anchor_hour.map((hour_tab['sum']+1)/(hour_tab['count']+2)).fillna(ytr.mean()).to_numpy()
    terminal_probs={'hour_empirical':p_hour_te}
    terminal_logits={'hour_empirical':np.log(np.clip(p_hour_te,1e-6,1-1e-6)/(1-np.clip(p_hour_te,1e-6,1-1e-6)))}
    validation_rows=[]
    for name in ['history_logistic','history_gbm']:
        if name=='history_logistic':
            sc=StandardScaler().fit(Xtr)
            model=LogisticRegression(max_iter=500,n_jobs=1,solver='lbfgs').fit(sc.transform(Xtr),ytr)
            pv=model.predict_proba(sc.transform(Xva))[:,1]; pt=model.predict_proba(sc.transform(Xte))[:,1]
            logit=np.log(np.clip(pt,1e-6,1-1e-6)/(1-np.clip(pt,1e-6,1-1e-6)))
        else:
            model=LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=100,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=seed,n_jobs=1,verbosity=-1)
            model.fit(Xtr,ytr,eval_set=[(Xva,yva)],callbacks=[early_stopping(30,verbose=False)])
            pv=model.predict_proba(Xva)[:,1]; pt=model.predict_proba(Xte)[:,1]
            logit=np.log(np.clip(pt,1e-6,1-1e-6)/(1-np.clip(pt,1e-6,1-1e-6)))
        terminal_probs[name]=pt; terminal_logits[name]=logit
        ap,auc,br=safe_metric(yva,pv)
        validation_rows.append({'seed':seed,'terminal_model':name,'n_train':len(tr),'n_validation':len(va),'train_prevalence':float(ytr.mean()),'validation_prevalence':float(yva.mean()),'validation_auprc':ap,'validation_auroc':auc,'validation_brier':br,'validation_mean_p':float(np.mean(pv))})
    ap,auc,br=safe_metric(yva,p_hour_va)
    validation_rows.insert(0,{'seed':seed,'terminal_model':'hour_empirical','n_train':len(tr),'n_validation':len(va),'train_prevalence':float(ytr.mean()),'validation_prevalence':float(yva.mean()),'validation_auprc':ap,'validation_auroc':auc,'validation_brier':br,'validation_mean_p':float(np.mean(p_hour_va))})
    pd.DataFrame(validation_rows).to_csv(O/f'terminal_model_validation_seed{seed}.csv',index=False)
    logits=pd.DataFrame({'patient_id':te.patient_id.to_numpy(),'anchor_hour':te.anchor_hour.to_numpy(),'next_window_available':yte})
    for name in terminal_probs:
        logits['p_available__'+name]=terminal_probs[name]
        logits['logit_available__'+name]=terminal_logits[name]
    logits.to_parquet(O/f'terminal_logits_seed{seed}.parquet',index=False)
    rows=[]
    for method in METHODS:
        for frac in FRACS:
            path=R/f'improvement_fair_ssl/fair_pred_{method}_seed{seed}_f{frac:.2f}.parquet'
            pred=pd.read_parquet(path)
            if len(pred)!=len(te) or not np.array_equal(pred.y.to_numpy(dtype=int),te.y_h6.to_numpy(dtype=int)) or not np.array_equal(pred.patient_id.to_numpy(),te.patient_id.to_numpy()):
                raise RuntimeError(f'prediction_alignment_failed seed={seed} method={method} frac={frac}')
            y=pred.y.to_numpy(dtype=int); p=pred.p.to_numpy(dtype=float); avail=yte.astype(bool)
            apc,aucc,brc=safe_metric(y[avail],p[avail])
            pnum=np.clip(p_hour_te,0.05,0.999)
            for name,phat in terminal_probs.items():
                phat=np.clip(np.asarray(phat,dtype=float),0.05,0.999)
                raw_w=1.0/phat[avail]
                stab_w=(pnum/phat)[avail]
                apr,aucr,brr=safe_metric(y[avail],p[avail],raw_w)
                aps,aucs,brs=safe_metric(y[avail],p[avail],stab_w)
                row={'seed':seed,'method':method,'fraction':frac,'terminal_model':name,'n_test':len(te),'n_available':int(avail.sum()),'availability_rate':float(avail.mean()),'positive_all':int(y.sum()),'positive_available':int(y[avail].sum()),'auprc_complete_case':apc,'auroc_complete_case':aucc,'brier_complete_case':brc,'auprc_raw_ipw':apr,'auroc_raw_ipw':aucr,'brier_raw_ipw':brr,'auprc_stabilized_ipw':aps,'auroc_stabilized_ipw':aucs,'brier_stabilized_ipw':brs,'raw_positive_support':int(np.sum(raw_w>0)),'stabilized_positive_support':int(np.sum(stab_w>0))}
                row.update({'raw_'+k:v for k,v in weight_stats(raw_w).items()}); row.update({'stabilized_'+k:v for k,v in weight_stats(stab_w).items()})
                rows.append(row)
    out=pd.DataFrame(rows); out.to_csv(O/f'terminal_censoring_metrics_seed{seed}.csv',index=False)
    (O/f'terminal_censoring_seed{seed}.json').write_text(json.dumps({'seed':seed,'runtime_seconds':time.time()-t0,'terminal_models':list(terminal_probs),'rows':len(out),'bootstrap':False,'features':MODEL_FEATURES},indent=2))
    print({'seed':seed,'runtime_seconds':time.time()-t0,'rows':len(out)},flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); main(ap.parse_args().seed)
