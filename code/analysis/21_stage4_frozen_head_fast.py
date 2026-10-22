#!/usr/bin/env python3
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015');D=P/'data/processed';R=P/'results'
DYN=['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets'];BASE=['Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS'];INPUT=BASE+[f'v__{c}' for c in DYN]+[f'm0__{c}' for c in DYN]+[f'm4__{c}' for c in DYN]
class Enc(nn.Module):
 def __init__(self,d,h=64):super().__init__();self.gru=nn.GRU(d,h,batch_first=True)
 def forward(self,x):return self.gru(x)[0][:,-1]
def make_groups(df,pids,med):
 w=df[df.patient_id.isin(pids)][['patient_id','anchor_hour']+INPUT].sort_values(['patient_id','anchor_hour']).reset_index(drop=True);a=w[INPUT].to_numpy('float32',copy=True);m=med.reindex(INPUT).fillna(0).to_numpy('float32');bad=~np.isfinite(a)
 if bad.any():a[bad]=np.broadcast_to(m,a.shape)[bad]
 ids=w.patient_id.to_numpy();h=w.anchor_hour.to_numpy();cuts=np.r_[0,1+np.flatnonzero(ids[1:]!=ids[:-1]),len(ids)]
 out={}
 for s,e in zip(cuts[:-1],cuts[1:]):
  arr=a[s:e];pad=np.repeat(arr[:1],11,axis=0);p=np.concatenate([pad,arr],axis=0);seq=np.lib.stride_tricks.sliding_window_view(p,(12,p.shape[1]))[:,0,:,:];out[ids[s]]=(seq,int(h[s]))
 return out
def build_split(win,part,groups,sp):
 w=win[win.patient_id.map(part)==sp].sort_values(['patient_id','anchor_hour']);xs=[];ys=[];pids=[]
 for pid,g in w.groupby('patient_id',sort=False):
  seq,mh=groups[pid];idx=(g.anchor_hour.to_numpy()-mh).astype(int);xs.append(seq[idx]);ys.append(g.y_h6.to_numpy(dtype='float32'));pids.extend([pid]*len(g))
 return np.concatenate(xs).astype('float32'),np.concatenate(ys).astype('float32'),np.asarray(pids,dtype=object)
def choose(win,part,frac,seed):
 train=np.array([k for k,v in part.items() if v=='train'],dtype=object);pw=win[win.patient_id.isin(train)].groupby('patient_id').y_h6.max();pos=np.array(pw[pw>0].index);neg=np.array(pw[pw==0].index);rng=np.random.default_rng(seed);n=max(1,int(round(len(train)*frac)));np_=min(len(pos),max(1,int(round(len(pos)*frac))));nn_=min(len(neg),max(0,n-np_));return set(np.concatenate([rng.permutation(pos)[:np_],rng.permutation(neg)[:nn_]]).tolist())
def boot(y,p,pids,B,seed):
 u,inv=np.unique(pids,return_inverse=True);n=len(u);rng=np.random.default_rng(seed);v=[]
 for st in range(0,B,100):
  for s in rng.integers(0,n,size=(min(100,B-st),n)):
   w=np.bincount(s,minlength=n)[inv];v.append((average_precision_score(y,p,sample_weight=w),roc_auc_score(y,p,sample_weight=w),brier_score_loss(y,p,sample_weight=w)))
 a=np.asarray(v);return [float(np.percentile(a[:,j],q)) for j in range(3) for q in [2.5,97.5]]
def main(args):
 torch.set_num_threads(args.threads);np.random.seed(args.seed);torch.manual_seed(args.seed)
 df=pd.read_parquet(D/'stage3_p19_features.parquet');df['anchor_hour']=df.ICULOS;win=pd.read_parquet(D/'cinc2019_windows.parquet',columns=['patient_id','anchor_hour','y_h6']);part=json.loads((D/'partitions.json').read_text())['cinc2019'][str(args.seed)];trainp={k for k,v in part.items() if v=='train'};med=df[df.patient_id.isin(trainp)][INPUT].median(numeric_only=True).fillna(0);groups=make_groups(df,set(part.keys()),med)
 Xtr,ytr,pidtr=build_split(win,part,groups,'train');Xva,yva,pidva=build_split(win,part,groups,'validation');Xte,yte,pidte=build_split(win,part,groups,'test');rows=[]
 for method in ['nextmask','nextintensity','valueforecast']:
  encoder=Enc(len(INPUT),64);encoder.load_state_dict(torch.load(R/f'stage4_encoder_{method}_seed{args.seed}_full.pt',map_location='cpu'));encoder.eval();features={}
  for name,X in [('train',Xtr),('validation',Xva),('test',Xte)]:
   out=[]
   for (xb,) in DataLoader(TensorDataset(torch.from_numpy(X)),batch_size=args.batch,shuffle=False): 
    with torch.no_grad():out.append(encoder(xb))
   features[name]=torch.cat(out).numpy()
  for frac in [.01,.05,.10,.25,1.0]:
   selected=choose(win,part,frac,args.seed);idx=np.array([p in selected for p in pidtr]);head=nn.Linear(64,1);opt=torch.optim.Adam(head.parameters(),lr=1e-3);lossf=nn.BCEWithLogitsLoss();best=-1;state=None;dl=DataLoader(TensorDataset(torch.from_numpy(features['train'][idx]).float(),torch.from_numpy(ytr[idx]).float()),batch_size=args.head_batch,shuffle=True)
   for ep in range(3):
    head.train()
    for xb,yb in dl:opt.zero_grad();loss=lossf(head(xb).squeeze(-1),yb);loss.backward();opt.step()
    head.eval();vp=torch.sigmoid(head(torch.from_numpy(features['validation']).float()).squeeze(-1)).detach().numpy();sc=average_precision_score(yva,vp)
    if sc>best:best=sc;state={k:v.detach().clone() for k,v in head.state_dict().items()}
   head.load_state_dict(state);head.eval();p=torch.sigmoid(head(torch.from_numpy(features['test']).float()).squeeze(-1)).detach().numpy();ci=boot(yte,p,pidte,args.bootstrap,args.seed+int(frac*10000));row={'seed':args.seed,'init':method,'fraction':frac,'n_train_patients':len(selected),'n_train_windows':int(idx.sum()),'n_test_windows':len(yte),'positive_test':int(yte.sum()),'val_best_auprc':float(best),'auprc':float(average_precision_score(yte,p)),'auroc':float(roc_auc_score(yte,p)),'brier':float(brier_score_loss(yte,p))};row.update(dict(zip(['auprc_lo','auprc_hi','auroc_lo','auroc_hi','brier_lo','brier_hi'],ci)));rows.append(row)
 pd.DataFrame(rows).to_csv(R/f'stage4_frozen_head_results_seed{args.seed}.csv',index=False);print({'seed':args.seed,'rows':len(rows)},flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--batch',type=int,default=8192);ap.add_argument('--head-batch',type=int,default=16384);ap.add_argument('--threads',type=int,default=8);ap.add_argument('--bootstrap',type=int,default=2000);main(ap.parse_args())

