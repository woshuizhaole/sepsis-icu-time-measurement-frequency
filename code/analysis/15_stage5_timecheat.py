#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015');D=P/'data/processed';R=P/'results'
sys.path.insert(0,str(P/'external/TimeCHEAT'))
from models.model import BackBone
DYN=['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets']
BASE=['Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS']; INPUT=BASE+[f'v__{c}' for c in DYN]+[f'm0__{c}' for c in DYN]+[f'm4__{c}' for c in DYN]
class DS(Dataset):
 def __init__(self,items,groups):self.items=items;self.groups=groups
 def __len__(self):return len(self.items)
 def __getitem__(self,i):
  pid,h,y=self.items[i];arr,mh=self.groups[pid];idx=min(max(h-mh,0),len(arr)-1);lo=max(0,idx-11);x=arr[lo:idx+1]
  if len(x)<12:x=np.concatenate([np.repeat(x[:1],12-len(x),axis=0),x],0)
  v=x[:,len(BASE):len(BASE)+len(DYN)];m=x[:,len(BASE)+len(DYN):len(BASE)+2*len(DYN)];t=np.arange(12,dtype='float32')[:,None]
  return torch.from_numpy(np.concatenate([v,m,t],1).astype('float32')),torch.tensor(y,dtype=torch.long)
def groups(df,pids,med):
 w=df[df.patient_id.isin(pids)][['patient_id','anchor_hour']+INPUT].sort_values(['patient_id','anchor_hour']).reset_index(drop=True);a=w[INPUT].to_numpy('float32',copy=True);ma=med.reindex(INPUT).fillna(0).to_numpy('float32');bad=~np.isfinite(a)
 if bad.any():a[bad]=np.broadcast_to(ma,a.shape)[bad]
 ids=w.patient_id.to_numpy();h=w.anchor_hour.to_numpy();cuts=np.r_[0,1+np.flatnonzero(ids[1:]!=ids[:-1]),len(ids)]
 return {ids[s]:(a[s:e],int(h[s])) for s,e in zip(cuts[:-1],cuts[1:])}
def boot(y,p,pids,B,seed):
 u,inv=np.unique(pids,return_inverse=True);n=len(u);vals=[];rng=np.random.default_rng(seed)
 for st in range(0,B,100):
  for s in rng.integers(0,n,size=(min(100,B-st),n)):
   w=np.bincount(s,minlength=n)[inv]
   vals.append((average_precision_score(y,p,sample_weight=w),roc_auc_score(y,p,sample_weight=w),brier_score_loss(y,p,sample_weight=w)))
 a=np.asarray(vals);return {'auprc_lo':float(np.percentile(a[:,0],2.5)),'auprc_hi':float(np.percentile(a[:,0],97.5)),'auroc_lo':float(np.percentile(a[:,1],2.5)),'auroc_hi':float(np.percentile(a[:,1],97.5)),'brier_lo':float(np.percentile(a[:,2],2.5)),'brier_hi':float(np.percentile(a[:,2],97.5))}
def main(args):
 torch.set_num_threads(args.threads);np.random.seed(args.seed);torch.manual_seed(args.seed);dev=torch.device(args.device)
 df=pd.read_parquet(D/'stage3_p19_features.parquet');df['anchor_hour']=df.ICULOS;win=pd.read_parquet(D/'cinc2019_windows.parquet',columns=['patient_id','anchor_hour','y_h6']);part=json.loads((D/'partitions.json').read_text())['cinc2019'][str(args.seed)]
 trp={k for k,v in part.items() if v=='train'};med=df[df.patient_id.isin(trp)][INPUT].median(numeric_only=True).fillna(0);g=groups(df,set(part.keys()),med)
 def its(sp):return [(r.patient_id,int(r.anchor_hour),int(r.y_h6)) for r in win[win.patient_id.map(part)==sp].itertuples(index=False)]
 tri,vi,tei=its('train'),its('validation'),its('test');tr=DataLoader(DS(tri,g),batch_size=args.batch,shuffle=True,num_workers=0);va=DataLoader(DS(vi,g),batch_size=args.batch,shuffle=False);te=DataLoader(DS(tei,g),batch_size=args.batch,shuffle=False)
 cfg={'obs':12,'n_class':2};model=BackBone(channels=len(DYN),attn_head=1,latent_dim=args.latent,n_layers=1,ref_points=args.ref,n_patches=4,dropout=.1,former_factor=1,former_dff=64,former_output_attention=False,former_layers=1,former_heads=4,former_activation='gelu',downstream='classification',config=cfg).to(dev)
 opt=torch.optim.AdamW(model.parameters(),lr=args.lr);lossf=nn.CrossEntropyLoss();best=-np.inf;state=None
 for ep in range(args.epochs):
  model.train();ls=[]
  for x,y in tr:opt.zero_grad();z=model(x.to(dev),None);loss=lossf(z,y.to(dev));loss.backward();opt.step();ls.append(float(loss.detach().cpu()))
  model.eval();vp=[];vy=[]
  with torch.no_grad():
   for x,y in va:vp.extend(torch.softmax(model(x.to(dev),None),1)[:,1].cpu().numpy());vy.extend(y.numpy())
  sc=average_precision_score(vy,vp);print({'seed':args.seed,'epoch':ep,'loss':float(np.mean(ls)),'val_auprc':float(sc)},flush=True)
  if sc>best:best=sc;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 model.load_state_dict(state);model.eval();pp=[];yy=[]
 with torch.no_grad():
  for x,y in te:pp.extend(torch.softmax(model(x.to(dev),None),1)[:,1].cpu().numpy());yy.extend(y.numpy())
 yy=np.asarray(yy);pp=np.asarray(pp);pids=np.array([x[0] for x in tei],dtype=object);row={'seed':args.seed,'model':'TimeCHEAT_adapted_causal','n_train_windows':len(tri),'n_val_windows':len(vi),'n_test_windows':len(tei),'positive_test':int(yy.sum()),'best_val_auprc':float(best),'auprc':float(average_precision_score(yy,pp)),'auroc':float(roc_auc_score(yy,pp)),'brier':float(brier_score_loss(yy,pp))};row.update(boot(yy,pp,pids,args.bootstrap,args.seed+9000));pd.DataFrame({'patient_id':pids,'anchor_hour':[x[1] for x in tei],'y':yy,'p':pp}).to_parquet(R/f'stage5_timecheat_pred_seed{args.seed}.parquet',index=False);torch.save(state,R/f'stage5_timecheat_seed{args.seed}.pt');pd.DataFrame([row]).to_csv(R/f'stage5_timecheat_results_seed{args.seed}.csv',index=False);print(row,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--epochs',type=int,default=2);ap.add_argument('--batch',type=int,default=256);ap.add_argument('--latent',type=int,default=32);ap.add_argument('--ref',type=int,default=24);ap.add_argument('--lr',type=float,default=1e-3);ap.add_argument('--threads',type=int,default=10);ap.add_argument('--bootstrap',type=int,default=2000);ap.add_argument('--device',default='cpu');main(ap.parse_args())

