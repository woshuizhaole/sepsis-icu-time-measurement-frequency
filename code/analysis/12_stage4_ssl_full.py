#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015')
D=P/'data/processed'; R=P/'results'; L=P/'logs/analysis'
R.mkdir(exist_ok=True); L.mkdir(exist_ok=True)
DYN=['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets']
BASE=['Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS']
INPUT=BASE+[f'v__{c}' for c in DYN]+[f'm0__{c}' for c in DYN]+[f'm4__{c}' for c in DYN]
MASK_SL=slice(len(BASE)+len(DYN),len(BASE)+2*len(DYN))

class Seq(Dataset):
    def __init__(self, items, groups, max_len=12):
        self.items=items; self.groups=groups; self.max_len=max_len
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        pid,hour,*targets=self.items[i]
        arr,minh=self.groups[pid]
        idx=min(max(int(hour)-int(minh),0),len(arr)-1)
        lo=max(0,idx-self.max_len+1); x=arr[lo:idx+1]
        if len(x)<self.max_len:
            x=np.concatenate([np.repeat(x[:1],self.max_len-len(x),axis=0),x],axis=0)
        return (torch.from_numpy(x.astype('float32')),*[torch.tensor(t) for t in targets])

class Enc(nn.Module):
    def __init__(self,d,h=64): super().__init__(); self.gru=nn.GRU(d,h,batch_first=True)
    def forward(self,x): return self.gru(x)[0][:,-1]

class SSLHead(nn.Module):
    def __init__(self,h,k=34):
        super().__init__(); self.mask=nn.Linear(h,k); self.intensity=nn.Linear(h,k*5)
        self.value=nn.Linear(h,k); self.terminal=nn.Linear(h,1)
    def forward(self,z):
        return self.mask(z), self.intensity(z).view(-1,34,5), self.value(z), self.terminal(z).squeeze(-1)

class Cls(nn.Module):
    def __init__(self,d,h=64): super().__init__(); self.enc=Enc(d,h); self.head=nn.Linear(h,1)
    def forward(self,x): return self.head(self.enc(x)).squeeze(-1)

def make_groups(df, pids, med):
    # One vectorized conversion followed by contiguous patient slices; no per-group pandas apply.
    w=df[df.patient_id.isin(pids)][['patient_id','anchor_hour']+INPUT].sort_values(['patient_id','anchor_hour']).reset_index(drop=True)
    vals=w[INPUT].to_numpy(dtype='float32',copy=True)
    medarr=med.reindex(INPUT).fillna(0).to_numpy(dtype='float32')
    bad=~np.isfinite(vals)
    if bad.any(): vals[bad]=np.broadcast_to(medarr,vals.shape)[bad]
    ids=w.patient_id.to_numpy(); hours=w.anchor_hour.to_numpy()
    cuts=np.r_[0,1+np.flatnonzero(ids[1:]!=ids[:-1]),len(ids)]
    groups={}
    for s,e in zip(cuts[:-1],cuts[1:]): groups[ids[s]]=(vals[s:e],int(hours[s]))
    return groups

def pre_items(groups, max_hour=24):
    # Vectorized within each patient; avoids per-anchor Python slicing while preserving causal targets.
    raw=[]
    for pid,(arr,minh) in groups.items():
        maxh=minh+len(arr)-1
        lo=max(1,minh); hi=min(maxh,max_hour)
        if hi<lo: continue
        hours=np.arange(lo,hi+1,dtype='int64'); idx=hours-int(minh); n=len(hours)
        valid=(idx+4)<len(arr)
        nmask=np.zeros((n,len(DYN)),dtype='float32'); nval=np.zeros((n,len(DYN)),dtype='float32'); count=np.zeros((n,len(DYN)),dtype='int64')
        vi=np.flatnonzero(valid)
        if len(vi):
            nmask[vi]=arr[idx[vi]+1,MASK_SL]
            nval[vi]=arr[idx[vi]+1,len(BASE):len(BASE)+len(DYN)]
            m=arr[:,MASK_SL].astype('float32')
            cs=np.concatenate([np.zeros((1,len(DYN)),dtype='float32'),np.cumsum(m,axis=0)],axis=0)
            count[vi]=np.rint(cs[idx[vi]+5]-cs[idx[vi]+1]).astype('int64')
        for i,h in enumerate(hours): raw.append((pid,int(h),int(valid[i]),nmask[i],count[i],nval[i]))
    by_h={}
    for _,h,a,*_ in raw: by_h.setdefault(h,[]).append(a)
    ph={h:float((sum(v)+1)/(len(v)+2)) for h,v in by_h.items()}
    items=[]
    for pid,h,a,nm,c,nv in raw:
        w=float(1.0/max(ph[h],0.05)) if a else 0.0
        items.append((pid,h,nm,c,nv,int(a),w))
    return items,ph

def pretrain(method,df,part,seed,args):
    device=torch.device(args.device)
    trainp={k for k,v in part.items() if v=='train'}
    med=df[df.patient_id.isin(trainp)][INPUT].median(numeric_only=True).fillna(0)
    groups=make_groups(df,trainp,med)
    items,ph=pre_items(groups,args.pretrain_hours)
    ds=Seq(items,groups)
    dl=DataLoader(ds,batch_size=args.batch,shuffle=True,num_workers=0)
    enc=Enc(len(INPUT),args.hidden).to(device); head=SSLHead(args.hidden).to(device)
    opt=torch.optim.Adam(list(enc.parameters())+list(head.parameters()),lr=args.lr)
    bce=nn.BCEWithLogitsLoss(); ce=nn.CrossEntropyLoss(reduction='none'); mse=nn.MSELoss(reduction='none')
    enc.train(); head.train(); losses=[]
    for ep in range(args.pretrain_epochs):
        for x,nmask,count,nval,avail,weight in dl:
            x=x.to(device); nmask=nmask.to(device); count=count.to(device); nval=nval.to(device); avail=avail.to(device); weight=weight.to(device)
            opt.zero_grad(); z=enc(x); lm,li,lv,lt=head(z)
            la=bce(lt,avail.float())
            valid=avail>0.5
            if bool(valid.any()):
                li_loss=ce(li.reshape(-1,5),count.long().reshape(-1)).view(-1,len(DYN)).mean(dim=1)
                lm_loss=torch.nn.functional.binary_cross_entropy_with_logits(lm,nmask.float(),reduction='none').mean(dim=1)
                lv_err=mse(lv*nmask.float(),nval.float()*nmask.float()); lv_loss=lv_err.sum(dim=1)/nmask.float().sum(dim=1).clamp_min(1.0)
                iw=weight.float()
                lc=(li_loss[valid]*iw[valid]).mean(); lmask=(lm_loss[valid]*iw[valid]).mean(); lval=(lv_loss[valid]*iw[valid]).mean()
                # Target-isolated SSL: terminal-state is a shared registered auxiliary;
                # each named method optimizes only its own pretext head.
                if method=='nextmask': loss=la+lmask
                elif method=='nextintensity': loss=la+lc
                elif method=='valueforecast': loss=la+lval
                else: raise ValueError(method)
            else: loss=la
            loss.backward(); opt.step(); losses.append(float(loss.detach()))
    path=R/f'stage4_encoder_{method}_seed{seed}_full.pt'; torch.save({k:v.detach().cpu() for k,v in enc.state_dict().items()},path)
    meta={'seed':seed,'method':method,'n_train_patients':len(trainp),'n_pretrain_anchors':len(items),'n_complete_targets':int(sum(i[5] for i in items)),'availability_rate':float(np.mean([i[5] for i in items])),'hour_ipcw':ph,'loss_mean':float(np.mean(losses))}
    (R/f'stage4_pretrain_{method}_seed{seed}_full.json').write_text(json.dumps(meta,indent=2))
    print({'pretrain':method,**meta},flush=True)
    return path

def select_patients(win,part,frac,seed):
    trainp=np.array([k for k,v in part.items() if v=='train'],dtype=object)
    pw=win[win.patient_id.isin(trainp)].groupby('patient_id').y_h6.max()
    pos=np.array(pw[pw>0].index,dtype=object); neg=np.array(pw[pw==0].index,dtype=object)
    rng=np.random.default_rng(seed)
    n=max(1,int(round(len(trainp)*frac)))
    np_rng=rng.permutation(pos); nn_rng=rng.permutation(neg)
    npos=min(len(pos),max(1,int(round(len(pos)*frac)))) if len(pos) else 0
    nneg=min(len(neg),max(0,n-npos))
    if npos+nneg<n:
        extra=np.setdiff1d(trainp,np.concatenate([np_rng[:npos],nn_rng[:nneg]]),assume_unique=False)
        extra=extra[rng.permutation(len(extra))[:n-npos-nneg]]
    else: extra=np.array([],dtype=object)
    return set(np.concatenate([np_rng[:npos],nn_rng[:nneg],extra]).tolist())

def items_from_win(w,groups):
    return [(r.patient_id,int(r.anchor_hour),int(r.y_h6)) for r in w.itertuples(index=False) if r.patient_id in groups]

def patient_bootstrap(y,p,pids,B,rng):
    y=np.asarray(y); p=np.asarray(p); pids=np.asarray(pids); uniq,inv=np.unique(pids,return_inverse=True); n=len(uniq)
    vals=[]
    for start in range(0,B,100):
        m=min(100,B-start); samples=rng.integers(0,n,size=(m,n)); 
        for s in samples:
            counts=np.bincount(s,minlength=n); w=counts[inv]
            try: ap=average_precision_score(y,p,sample_weight=w); auc=roc_auc_score(y,p,sample_weight=w)
            except ValueError: ap=np.nan; auc=np.nan
            vals.append((ap,auc,brier_score_loss(y,p,sample_weight=w)))
    a=np.asarray(vals,float)
    return {'auprc_lo':float(np.nanpercentile(a[:,0],2.5)),'auprc_hi':float(np.nanpercentile(a[:,0],97.5)),
            'auroc_lo':float(np.nanpercentile(a[:,1],2.5)),'auroc_hi':float(np.nanpercentile(a[:,1],97.5)),
            'brier_lo':float(np.nanpercentile(a[:,2],2.5)),'brier_hi':float(np.nanpercentile(a[:,2],97.5))}

def downstream(init_path,method,df,win,part,seed,frac,args,groups):
    device=torch.device(args.device)
    trainp=select_patients(win,part,frac,seed)
    trw=win[win.patient_id.isin(trainp)]; vaw=win[win.patient_id.map(part)=='validation']; tew=win[win.patient_id.map(part)=='test']
    tritems=items_from_win(trw,groups); vaitems=items_from_win(vaw,groups); teitems=items_from_win(tew,groups)
    tr=DataLoader(Seq(tritems,groups),batch_size=args.batch,shuffle=True,num_workers=0)
    va=DataLoader(Seq(vaitems,groups),batch_size=args.batch,shuffle=False,num_workers=0)
    te=DataLoader(Seq(teitems,groups),batch_size=args.batch,shuffle=False,num_workers=0)
    model=Cls(len(INPUT),args.hidden)
    if init_path is not None: model.enc.load_state_dict(torch.load(init_path,map_location='cpu'))
    model=model.to(device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr); lossfn=nn.BCEWithLogitsLoss()
    best=-np.inf; best_state=None
    for ep in range(args.downstream_epochs):
        model.train(); losses=[]
        for x,y in tr:
            x=x.to(device); y=y.to(device)
            opt.zero_grad(); loss=lossfn(model(x),y.float()); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        model.eval(); vp=[]; vy=[]
        with torch.no_grad():
            for x,y in va:
                vp.extend(torch.sigmoid(model(x.to(device))).detach().cpu().numpy()); vy.extend(y.numpy())
        sc=average_precision_score(vy,vp)
        if sc>best: best=sc; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    model.eval(); pp=[]; yy=[]
    with torch.no_grad():
        for x,y in te:
            pp.extend(torch.sigmoid(model(x.to(device))).detach().cpu().numpy()); yy.extend(y.numpy())
    # Reconstruct test patient IDs from the same deterministic item order.
    pids=np.array([x[0] for x in teitems],dtype=object); yy=np.asarray(yy); pp=np.asarray(pp)
    rng=np.random.default_rng(seed+int(frac*10000)+{'random':0,'nextmask':101,'nextintensity':202,'valueforecast':303}.get(method,404))
    boot=patient_bootstrap(yy,pp,pids,args.bootstrap,rng)
    row={'seed':seed,'init':method,'fraction':frac,'n_train_patients':len(trainp),'n_train_windows':len(tritems),'n_test_windows':len(teitems),'positive_test':int(yy.sum()),'train_positive_windows':int(trw.y_h6.sum()),'val_best_auprc':float(best),'auprc':float(average_precision_score(yy,pp)),'auroc':float(roc_auc_score(yy,pp)),'brier':float(brier_score_loss(yy,pp)),'input_dim':len(INPUT)}
    row.update(boot)
    pred=pd.DataFrame({'patient_id':pids,'y':yy,'p':pp})
    pred.to_parquet(R/f'stage4_full_pred_{method}_seed{seed}_f{frac:.2f}.parquet',index=False)
    print(row,flush=True)
    return row

def main(args):
    torch.set_num_threads(args.threads); np.random.seed(args.seed0); torch.manual_seed(args.seed0)
    df=pd.read_parquet(D/'stage3_p19_features.parquet'); df['anchor_hour']=df['ICULOS']
    win=pd.read_parquet(D/'cinc2019_windows.parquet',columns=['patient_id','anchor_hour','y_h6'])
    seeds=[int(x) for x in args.seeds.split(',') if x]; fracs=[float(x) for x in args.fractions.split(',')]
    allrows=[]
    for seed in seeds:
        np.random.seed(seed); torch.manual_seed(seed)
        part=json.loads((D/'partitions.json').read_text())['cinc2019'][str(seed)]
        trainp={k for k,v in part.items() if v=='train'}
        med=df[df.patient_id.isin(trainp)][INPUT].median(numeric_only=True).fillna(0)
        groups=make_groups(df,set(part.keys()),med)
        paths={'random':None}
        for method in args.methods.split(','):
            paths[method]=pretrain(method,df,part,seed,args)
        seedrows=[]
        for method,path in paths.items():
            for frac in fracs:
                row=downstream(path,method,df,win,part,seed,frac,args,groups)
                allrows.append(row); seedrows.append(row)
        pd.DataFrame(seedrows).to_csv(R/f'stage4_ssl_full_results_seed{seed}.csv',index=False)
        pd.DataFrame(allrows).to_csv(R/'stage4_ssl_full_results.csv',index=False)
    pd.DataFrame(allrows).to_csv(R/'stage4_ssl_full_results.csv',index=False)
    qc={'stage':'Stage 4 SSL full','rows':len(allrows),'expected_rows':len(seeds)*len(fracs)*(1+len(args.methods.split(','))),'seeds':seeds,'fractions':fracs,'full_test':True,'patient_bootstrap':args.bootstrap,'status':'PASS' if len(allrows)==len(seeds)*len(fracs)*(1+len(args.methods.split(','))) else 'PARTIAL'}
    (R/'stage4_ssl_full_qc.json').write_text(json.dumps(qc,indent=2))
    print(qc,flush=True)
ap=argparse.ArgumentParser()
ap.add_argument('--seeds',default='101,202,303,404,505'); ap.add_argument('--fractions',default='0.01,0.05,0.10,0.25,1.0')
ap.add_argument('--methods',default='nextmask,nextintensity,valueforecast'); ap.add_argument('--pretrain-hours',type=int,default=24)
ap.add_argument('--pretrain-epochs',type=int,default=1); ap.add_argument('--downstream-epochs',type=int,default=3)
ap.add_argument('--batch',type=int,default=1024); ap.add_argument('--hidden',type=int,default=64); ap.add_argument('--lr',type=float,default=1e-3)
ap.add_argument('--threads',type=int,default=40); ap.add_argument('--bootstrap',type=int,default=2000); ap.add_argument('--seed0',type=int,default=101); ap.add_argument('--device',default='cpu')
args=ap.parse_args(); main(args)

