#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015'); D=P/'data/processed'; R=P/'results'
DYN=['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets']
BASE=['Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS']; INPUT=BASE+[f'v__{c}' for c in DYN]+[f'm0__{c}' for c in DYN]+[f'm4__{c}' for c in DYN]
MASK_SL=slice(len(BASE)+len(DYN),len(BASE)+2*len(DYN))

def make_groups(df,pids,med):
    w=df[df.patient_id.isin(pids)][['patient_id','anchor_hour']+INPUT].sort_values(['patient_id','anchor_hour']).reset_index(drop=True)
    vals=w[INPUT].to_numpy(dtype='float32',copy=True); medarr=med.reindex(INPUT).fillna(0).to_numpy(dtype='float32')
    bad=~np.isfinite(vals)
    if bad.any(): vals[bad]=np.broadcast_to(medarr,vals.shape)[bad]
    ids=w.patient_id.to_numpy(); hours=w.anchor_hour.to_numpy()
    cuts=np.r_[0,1+np.flatnonzero(ids[1:]!=ids[:-1]),len(ids)]
    return {ids[s]:(vals[s:e],int(hours[s])) for s,e in zip(cuts[:-1],cuts[1:])}

class WinDS(Dataset):
    def __init__(self,items,groups,max_len=12): self.items=items;self.groups=groups;self.max_len=max_len
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        pid,h,y,nmask,avail=self.items[i]; arr,minh=self.groups[pid]; idx=min(max(h-minh,0),len(arr)-1); lo=max(0,idx-self.max_len+1)
        x=arr[lo:idx+1]
        if len(x)<self.max_len: x=np.concatenate([np.repeat(x[:1],self.max_len-len(x),axis=0),x],axis=0)
        return torch.from_numpy(x.astype('float32')),torch.tensor(y,dtype=torch.float32),torch.from_numpy(nmask.astype('float32')),torch.tensor(avail,dtype=torch.float32)

class WBMGRU(nn.Module):
    def __init__(self,d,h=64): super().__init__(); self.gru=nn.GRU(d,h,batch_first=True); self.sepsis=nn.Linear(h,1); self.wbm=nn.Linear(h,len(DYN)); self.term=nn.Linear(h,1)
    def forward(self,x): z=self.gru(x)[0][:,-1]; return self.sepsis(z).squeeze(-1),self.wbm(z),self.term(z).squeeze(-1)

def items_from_win(win,groups):
    out=[]
    for r in win.itertuples(index=False):
        pid=r.patient_id; h=int(r.anchor_hour); arr,minh=groups[pid]; idx=h-minh; avail=int(idx+1<len(arr))
        nmask=arr[idx+1,MASK_SL].astype('float32') if avail else np.zeros(len(DYN),dtype='float32')
        out.append((pid,h,int(r.y_h6),nmask,avail))
    return out

def bootstrap(y,p,pids,B,seed):
    y=np.asarray(y);p=np.asarray(p);pids=np.asarray(pids);u,inv=np.unique(pids,return_inverse=True);n=len(u); vals=[]
    rng=np.random.default_rng(seed)
    for st in range(0,B,100):
        for s in rng.integers(0,n,size=(min(100,B-st),n)):
            w=np.bincount(s,minlength=n)[inv]
            try: a=average_precision_score(y,p,sample_weight=w); au=roc_auc_score(y,p,sample_weight=w)
            except ValueError: a=np.nan;au=np.nan
            vals.append((a,au,brier_score_loss(y,p,sample_weight=w)))
    a=np.asarray(vals)
    return {'auprc_lo':float(np.nanpercentile(a[:,0],2.5)),'auprc_hi':float(np.nanpercentile(a[:,0],97.5)),'auroc_lo':float(np.nanpercentile(a[:,1],2.5)),'auroc_hi':float(np.nanpercentile(a[:,1],97.5)),'brier_lo':float(np.nanpercentile(a[:,2],2.5)),'brier_hi':float(np.nanpercentile(a[:,2],97.5))}

def run(seed,args):
    torch.set_num_threads(args.threads); np.random.seed(seed); torch.manual_seed(seed); device=torch.device(args.device)
    df=pd.read_parquet(D/'stage3_p19_features.parquet'); df['anchor_hour']=df.ICULOS
    win=pd.read_parquet(D/'cinc2019_windows.parquet',columns=['patient_id','anchor_hour','y_h6'])
    part=json.loads((D/'partitions.json').read_text())['cinc2019'][str(seed)]
    trainp={k for k,v in part.items() if v=='train'}; med=df[df.patient_id.isin(trainp)][INPUT].median(numeric_only=True).fillna(0); groups=make_groups(df,set(part.keys()),med)
    trw=win[win.patient_id.map(part)=='train']; vaw=win[win.patient_id.map(part)=='validation']; tew=win[win.patient_id.map(part)=='test']
    tr=DataLoader(WinDS(items_from_win(trw,groups),groups),batch_size=args.batch,shuffle=True,num_workers=0); va=DataLoader(WinDS(items_from_win(vaw,groups),groups),batch_size=args.batch,shuffle=False,num_workers=0); teitems=items_from_win(tew,groups); te=DataLoader(WinDS(teitems,groups),batch_size=args.batch,shuffle=False,num_workers=0)
    model=WBMGRU(len(INPUT),args.hidden).to(device); opt=torch.optim.Adam(model.parameters(),lr=args.lr); bce=nn.BCEWithLogitsLoss(); best=-np.inf; best_state=None
    for ep in range(args.epochs):
        model.train(); losses=[]
        for x,y,nmask,avail in tr:
            x=x.to(device);y=y.to(device);nmask=nmask.to(device);avail=avail.to(device); opt.zero_grad(); sp,wm,tm=model(x)
            lsepsis=bce(sp,y); lterm=bce(tm,avail)
            valid=avail>0.5
            lw=torch.nn.functional.binary_cross_entropy_with_logits(wm[valid],nmask[valid]) if bool(valid.any()) else torch.tensor(0.,device=device)
            loss=lsepsis+lterm+args.wbm_weight*lw; loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        model.eval(); vp=[];vy=[]
        with torch.no_grad():
            for x,y,nmask,avail in va:
                sp,_,_=model(x.to(device));vp.extend(torch.sigmoid(sp).cpu().numpy());vy.extend(y.numpy())
        score=average_precision_score(vy,vp); print({'seed':seed,'epoch':ep,'train_loss':float(np.mean(losses)),'val_auprc':float(score)},flush=True)
        if score>best: best=score;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval(); pp=[];yy=[]
    with torch.no_grad():
        for x,y,nmask,avail in te:
            sp,_,_=model(x.to(device));pp.extend(torch.sigmoid(sp).cpu().numpy());yy.extend(y.numpy())
    pids=np.array([x[0] for x in teitems],dtype=object);yy=np.asarray(yy);pp=np.asarray(pp);row={'seed':seed,'model':'Warpformer_WBM_joint','n_train_windows':len(trw),'n_val_windows':len(vaw),'n_test_windows':len(teitems),'positive_test':int(yy.sum()),'best_val_auprc':float(best),'auprc':float(average_precision_score(yy,pp)),'auroc':float(roc_auc_score(yy,pp)),'brier':float(brier_score_loss(yy,pp))};row.update(bootstrap(yy,pp,pids,args.bootstrap,seed+7000));pd.DataFrame({'patient_id':pids,'anchor_hour':[x[1] for x in teitems],'y':yy,'p':pp}).to_parquet(R/f'stage5_wbm_pred_seed{seed}.parquet',index=False);torch.save(best_state,R/f'stage5_wbm_seed{seed}.pt');print(row,flush=True);return row

def main(args):
    row=run(args.seed,args);pd.DataFrame([row]).to_csv(R/f'stage5_wbm_results_seed{args.seed}.csv',index=False)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--epochs',type=int,default=3);ap.add_argument('--batch',type=int,default=2048);ap.add_argument('--hidden',type=int,default=64);ap.add_argument('--lr',type=float,default=1e-3);ap.add_argument('--wbm-weight',type=float,default=1.0);ap.add_argument('--threads',type=int,default=10);ap.add_argument('--bootstrap',type=int,default=2000);ap.add_argument('--device',default='cpu');main(ap.parse_args())

