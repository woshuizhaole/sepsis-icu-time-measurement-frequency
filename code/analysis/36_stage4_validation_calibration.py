#!/usr/bin/env python3
"""Validation-only calibration diagnostic for Stage 4 full SSL.

The source Stage 4 table keeps test predictions but not validation logits.  To
test whether its low-label probability inflation is a calibration/early-stop
artifact, this script replays the downstream head from the retained
train-only encoder checkpoint, stores validation and test probabilities, fits
an intercept-only logistic calibration on validation labels, and evaluates
the frozen transform on test rows.  Test labels are never used to fit the
offset or choose a threshold.  Existing sealed result files are untouched.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from scipy.optimize import brentq
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

P = Path("/data/workplace/wyx/measurement_intensity_ssl_20261015")
D = P / "data/processed"; R = P / "results"; S = R / "stats"; L = P / "logs/qc"
DYN = ['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets']
BASE = ['Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS']
INPUT = BASE + [f'v__{c}' for c in DYN] + [f'm0__{c}' for c in DYN] + [f'm4__{c}' for c in DYN]


class Seq(Dataset):
    def __init__(self, items, groups, max_len=12): self.items=items; self.groups=groups; self.max_len=max_len
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        pid, hour, *targets = self.items[i]; arr, minh = self.groups[pid]
        idx = min(max(int(hour)-int(minh), 0), len(arr)-1)
        lo = max(0, idx-self.max_len+1); x = arr[lo:idx+1]
        if len(x)<self.max_len: x=np.concatenate([np.repeat(x[:1], self.max_len-len(x), axis=0),x],axis=0)
        return (torch.from_numpy(x.astype('float32')), *[torch.tensor(t) for t in targets])


class Enc(nn.Module):
    def __init__(self,d,h=64): super().__init__(); self.gru=nn.GRU(d,h,batch_first=True)
    def forward(self,x): return self.gru(x)[0][:,-1]


class Cls(nn.Module):
    def __init__(self,d,h=64): super().__init__(); self.enc=Enc(d,h); self.head=nn.Linear(h,1)
    def forward(self,x): return self.head(self.enc(x)).squeeze(-1)


def make_groups(df, pids, med):
    w=df[df.patient_id.isin(pids)][['patient_id','anchor_hour']+INPUT].sort_values(['patient_id','anchor_hour']).reset_index(drop=True)
    vals=w[INPUT].to_numpy(dtype='float32', copy=True); medarr=med.reindex(INPUT).fillna(0).to_numpy(dtype='float32')
    bad=~np.isfinite(vals)
    if bad.any(): vals[bad]=np.broadcast_to(medarr, vals.shape)[bad]
    ids=w.patient_id.to_numpy(); hours=w.anchor_hour.to_numpy(); cuts=np.r_[0,1+np.flatnonzero(ids[1:]!=ids[:-1]),len(ids)]
    return {ids[s]:(vals[s:e],int(hours[s])) for s,e in zip(cuts[:-1],cuts[1:])}


def select_patients(win, part, frac, seed):
    trainp=np.array([k for k,v in part.items() if v=='train'],dtype=object)
    pw=win[win.patient_id.isin(trainp)].groupby('patient_id').y_h6.max()
    pos=np.array(pw[pw>0].index,dtype=object); neg=np.array(pw[pw==0].index,dtype=object); rng=np.random.default_rng(seed)
    n=max(1,int(round(len(trainp)*frac))); npos=min(len(pos),max(1,int(round(len(pos)*frac)))) if len(pos) else 0; nneg=min(len(neg),max(0,n-npos))
    psel=rng.permutation(pos)[:npos]; nsel=rng.permutation(neg)[:nneg]
    if npos+nneg<n:
        extra=np.setdiff1d(trainp,np.concatenate([psel,nsel]),assume_unique=False); extra=extra[rng.permutation(len(extra))[:n-npos-nneg]]
    else: extra=np.array([],dtype=object)
    return set(np.concatenate([psel,nsel,extra]).tolist())


def items_from_win(w, groups):
    return [(r.patient_id,int(r.anchor_hour),int(r.y_h6)) for r in w.itertuples(index=False) if r.patient_id in groups]


def infer(model, items, groups, batch, threads):
    dl=DataLoader(Seq(items,groups),batch_size=batch,shuffle=False,num_workers=0); pp=[]; yy=[]
    model.eval()
    with torch.no_grad():
        for x,y in dl:
            pp.extend(torch.sigmoid(model(x)).numpy()); yy.extend(y.numpy())
    return np.asarray(yy,dtype=int),np.asarray(pp,dtype=float),np.asarray([r[0] for r in items],dtype=object),np.asarray([r[1] for r in items],dtype=int)


def fit_intercept(y, p):
    target=float(np.mean(y)); z=logit(np.clip(p,1e-7,1-1e-7))
    f=lambda b: float(np.mean(expit(z+b))-target)
    try: return float(brentq(f,-50,50))
    except ValueError: return float(logit(np.clip(target,1e-7,1-1e-7))-logit(np.clip(float(np.mean(p)),1e-7,1-1e-7)))


def alert_stats(p, y, pid, hour, threshold):
    rows=[]; pos_pat=set(pid[y>0]); n_alert=0; n_false=0; true_pat=set(); exposure=0.0
    for u,gix in pd.Series(np.arange(len(pid))).groupby(pid):
        ix=np.asarray(gix.values,dtype=int); ix=ix[np.argsort(hour[ix])]; pp=p[ix]; yy=y[ix]
        z=(pp>=threshold).astype(int)
        for i in range(len(z)):
            if z[i]: z[i+1:min(len(z),i+13)]=0
        n_alert+=int(z.sum()); n_false+=int(np.sum((z==1)&(yy==0)))
        if np.any((z==1)&(yy==1)): true_pat.add(u)
        exposure += max(1.0, (int(hour[ix].max())-int(hour[ix].min())+1)/24.0)
    npos=len(pos_pat)
    return {'alert_threshold':float(threshold),'alert_n_alerts':int(n_alert),'alert_n_false_alarms':int(n_false),'alert_sensitivity_patient':float(len(true_pat)/npos) if npos else np.nan,'alert_alerts_per_patient_day':float(n_alert/exposure) if exposure else np.nan,'alert_false_alarms_per_patient_day':float(n_false/exposure) if exposure else np.nan}


def one(seed, method, frac, batch, epochs, threads):
    torch.set_num_threads(threads); np.random.seed(seed); torch.manual_seed(seed)
    df=pd.read_parquet(D/'stage3_p19_features.parquet'); df['anchor_hour']=df['ICULOS']; win=pd.read_parquet(D/'cinc2019_windows.parquet',columns=['patient_id','anchor_hour','y_h6']); parts=json.loads((D/'partitions.json').read_text())['cinc2019']; part=parts[str(seed)]
    trainp={k for k,v in part.items() if v=='train'}; med=df[df.patient_id.isin(trainp)][INPUT].median(numeric_only=True).fillna(0); groups=make_groups(df,set(part.keys()),med)
    sel=select_patients(win,part,frac,seed); tr=win[win.patient_id.isin(sel)]; va=win[win.patient_id.map(part)=='validation']; te=win[win.patient_id.map(part)=='test']
    tri=items_from_win(tr,groups); vai=items_from_win(va,groups); tei=items_from_win(te,groups)
    model=Cls(len(INPUT),64)
    if method!='random': model.enc.load_state_dict(torch.load(R/f'stage4_encoder_{method}_seed{seed}_full.pt',map_location='cpu'))
    opt=torch.optim.Adam(model.parameters(),lr=1e-3); lossfn=nn.BCEWithLogitsLoss(); best=-np.inf; best_state=None
    dl=DataLoader(Seq(tri,groups),batch_size=batch,shuffle=True,num_workers=0); va_dl=DataLoader(Seq(vai,groups),batch_size=batch,shuffle=False,num_workers=0)
    for ep in range(epochs):
        model.train()
        for x,y in dl:
            opt.zero_grad(); loss=lossfn(model(x),y.float()); loss.backward(); opt.step()
        yv,pv,_,_=infer(model,vai,groups,batch,threads); sc=average_precision_score(yv,pv)
        if sc>best: best=float(sc); best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    yv,pv,pidv,hv=infer(model,vai,groups,batch,threads); yt,pt,pidt,ht=infer(model,tei,groups,batch,threads)
    offset=fit_intercept(yv,pv); pt_cal=expit(logit(np.clip(pt,1e-7,1-1e-7))+offset); pv_cal=expit(logit(np.clip(pv,1e-7,1-1e-7))+offset)
    thr=float(pd.read_csv(R/'stage5_official_utility_by_seed.csv').set_index('seed').loc[seed,'threshold'])
    raw_alert=alert_stats(pt,yt,pidt,ht,thr); cal_alert=alert_stats(pt_cal,yt,pidt,ht,thr)
    row={'seed':seed,'method':method,'fraction':frac,'n_train_patients':len(sel),'n_train_windows':len(tri),'train_prevalence':float(tr.y_h6.mean()),'validation_prevalence':float(yv.mean()),'test_prevalence':float(yt.mean()),'validation_mean_pred_raw':float(pv.mean()),'test_mean_pred_raw':float(pt.mean()),'validation_mean_pred_cal':float(pv_cal.mean()),'test_mean_pred_cal':float(pt_cal.mean()),'validation_ap_raw':float(average_precision_score(yv,pv)),'validation_brier_raw':float(brier_score_loss(yv,pv)),'validation_ap_cal':float(average_precision_score(yv,pv_cal)),'validation_brier_cal':float(brier_score_loss(yv,pv_cal)),'test_ap_raw':float(average_precision_score(yt,pt)),'test_auroc_raw':float(roc_auc_score(yt,pt)),'test_brier_raw':float(brier_score_loss(yt,pt)),'test_ap_cal':float(average_precision_score(yt,pt_cal)),'test_auroc_cal':float(roc_auc_score(yt,pt_cal)),'test_brier_cal':float(brier_score_loss(yt,pt_cal)),'validation_intercept_offset':offset,'validation_best_ap':best,'external_threshold':thr}
    for k,v in raw_alert.items(): row['raw_'+k]=v
    for k,v in cal_alert.items(): row['cal_'+k]=v
    np.savez_compressed(R/f'stage4_validation_calibration_{method}_seed{seed}_f{frac:.2f}.npz',y_val=yv,p_val=pv,p_val_cal=pv_cal,pid_val=pidv,y_test=yt,p_test=pt,p_test_cal=pt_cal,pid_test=pidt)
    return row


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',default='101,202,303,404,505'); ap.add_argument('--methods',default='random,nextmask,nextintensity,valueforecast'); ap.add_argument('--fractions',default='0.01,0.05,0.25,1.0'); ap.add_argument('--batch',type=int,default=8192); ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--threads',type=int,default=2); args=ap.parse_args()
    S.mkdir(exist_ok=True); L.mkdir(exist_ok=True)
    rows=[]
    for seed in [int(x) for x in args.seeds.split(',') if x]:
        for method in [x for x in args.methods.split(',') if x]:
            for frac in [float(x) for x in args.fractions.split(',')]:
                row=one(seed,method,frac,args.batch,args.epochs,args.threads); rows.append(row); print(json.dumps(row,sort_keys=True),flush=True); pd.DataFrame(rows).to_csv(S/'stage4_validation_calibration_diagnostic.csv',index=False)
    qc={'status':'PASS','rows':len(rows),'seeds':[int(x) for x in args.seeds.split(',') if x],'methods':args.methods.split(','),'fractions':[float(x) for x in args.fractions.split(',')],'fit_scope':'validation labels only','test_labels_used_for_offset':False,'primary_output':'results/stats/stage4_validation_calibration_diagnostic.csv'}
    (S/'stage4_validation_calibration_qc.json').write_text(json.dumps(qc,indent=2)+'\n')
    print(json.dumps(qc,indent=2))

if __name__=='__main__': main()
