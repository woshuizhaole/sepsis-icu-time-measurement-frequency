#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015'); D=P/'data/processed'; R=P/'results'; L=P/'logs/analysis'
D.mkdir(exist_ok=True);R.mkdir(exist_ok=True);L.mkdir(exist_ok=True)
DYN=['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets']
BASE=['Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS']
INPUT=BASE+[f'v__{c}' for c in DYN]+[f'm0__{c}' for c in DYN]+[f'm4__{c}' for c in DYN]
VMASK=[f'm0__{c}' for c in DYN]; VVAL=[f'v__{c}' for c in DYN]
class Seq(Dataset):
 def __init__(self,items,groups,max_len=12): self.items=items;self.groups=groups;self.max_len=max_len
 def __len__(self):return len(self.items)
 def __getitem__(self,i):
  pid,hour,*targets=self.items[i];arr,minhour=self.groups[pid];idx=min(max(int(hour)-int(minhour),0),len(arr)-1);lo=max(0,idx-self.max_len+1);x=arr[lo:idx+1]
  if len(x)<self.max_len:x=np.concatenate([np.repeat(x[:1],self.max_len-len(x),axis=0),x],axis=0)
  return (torch.from_numpy(x.astype('float32')),*[torch.tensor(t) for t in targets])
class Enc(nn.Module):
 def __init__(self,d,h=64):super().__init__();self.gru=nn.GRU(d,h,batch_first=True)
 def forward(self,x):return self.gru(x)[0][:,-1]
class SSLHead(nn.Module):
 def __init__(self,h,k=34):super().__init__();self.mask=nn.Linear(h,k);self.intensity=nn.Linear(h,k*5);self.value=nn.Linear(h,k)
 def forward(self,z):return self.mask(z),self.intensity(z).view(-1,34,5),self.value(z)
GROUPS_CACHE=None
class Cls(nn.Module):
 def __init__(self,d,h=64):super().__init__();self.enc=Enc(d,h);self.head=nn.Linear(h,1)
 def forward(self,x):return self.head(self.enc(x)).squeeze(-1)
def groups_and_items(df,part,split,max_pre=None,seed=101,pre=False):
 pids={k for k,v in part.items() if v==split}; groups={}; trainp={k for k,v in part.items() if v=='train'}; med=df[df.patient_id.isin(trainp)][INPUT].median(numeric_only=True).fillna(0)
 for pid,g in df[df.patient_id.isin(pids)].groupby('patient_id',sort=False):
  g=g.sort_values('anchor_hour'); arr=g[INPUT].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0).to_numpy(dtype='float32'); groups[pid]=(arr,int(g.anchor_hour.min()))
 rng=np.random.default_rng(seed); items=[]
 for pid,(arr,minh) in groups.items():
  maxh=minh+len(arr)-1
  for hour in range(max(1,minh),min(maxh-4,24)+1 if pre else maxh+1):
   idx=hour-minh
   if pre:
    # targets next 4 hour mask counts, and next-hour mask/value
    m=[];v=[]
    for j in range(1,5):
     if idx+j<len(arr): m.append(arr[idx+j, len(BASE)+len(DYN):len(BASE)+2*len(DYN)]);v.append(arr[idx+j,len(BASE):len(BASE)+len(DYN)])
    if len(m)<4:continue
    count=np.rint(np.sum(np.asarray(m),axis=0)).astype('int64'); nextmask=np.asarray(m[0],dtype='float32'); nextval=np.asarray(v[0],dtype='float32')
    items.append((pid,hour,nextmask,count,nextval))
   else: items.append((pid,hour))
 if max_pre and len(items)>max_pre:
  idx=rng.choice(len(items),size=max_pre,replace=False);items=[items[i] for i in idx]
 return groups,items
def pretrain(method,df,part,args):
 groups,items=groups_and_items(df,part,'train',args.max_pretrain,args.seed,pre=True)
 dl=DataLoader(Seq(items,groups),batch_size=args.batch,shuffle=True,num_workers=0)
 enc=Enc(len(INPUT),args.hidden);head=SSLHead(args.hidden);opt=torch.optim.Adam(list(enc.parameters())+list(head.parameters()),lr=args.lr)
 bce=nn.BCEWithLogitsLoss();ce=nn.CrossEntropyLoss();mse=nn.MSELoss()
 enc.train();head.train();losses=[]
 for x,nmask,count,nval in dl:
  opt.zero_grad();z=enc(x);lm,li,lv=head(z)
  if method=='nextmask':loss=bce(lm,nmask.float())
  elif method=='nextintensity':loss=ce(li.reshape(-1,5),count.long().reshape(-1))
  elif method=='valueforecast':loss=mse(lv*nmask.float(),nval.float()*nmask.float())
  else: raise ValueError(method)
  loss.backward();opt.step();losses.append(float(loss))
 path=R/f'stage4_encoder_{method}_seed{args.seed}.pt';torch.save(enc.state_dict(),path)
 return path,float(np.mean(losses)),len(items)
def downstream(init_path,df,part,args,frac):
 global GROUPS_CACHE
 trainp={k for k,v in part.items() if v=='train'}; med=df[df.patient_id.isin(trainp)][INPUT].median(numeric_only=True).fillna(0); valp={k for k,v in part.items() if v=='validation'};testp={k for k,v in part.items() if v=='test'}
 if GROUPS_CACHE is None:
  GROUPS_CACHE={}
  for pid,g in df.groupby('patient_id',sort=False):
   if pid not in trainp|valp|testp:continue
   g=g.sort_values('anchor_hour'); arr=g[INPUT].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0).to_numpy(dtype='float32'); GROUPS_CACHE[pid]=(arr,int(g.anchor_hour.min()))
 groups=GROUPS_CACHE
 win=pd.read_parquet(D/'cinc2019_windows.parquet',columns=['patient_id','anchor_hour','y_h6'])
 rng=np.random.default_rng(args.seed);items=[]
 tr=win[win.patient_id.isin(trainp)];pos=tr[tr.y_h6==1];neg=tr[tr.y_h6==0];n=max(1,int(len(tr)*frac));nneg=max(0,n-len(pos));neg=neg.iloc[rng.choice(len(neg),size=min(nneg,len(neg)),replace=False)];tr=pd.concat([pos,neg])
 def make(w):return [(r.patient_id,int(r.anchor_hour),int(r.y_h6)) for r in w.itertuples(index=False)]
 tra=make(tr);va=make(win[win.patient_id.isin(valp)]);test_w=win[win.patient_id.isin(testp)]
 if args.max_test and len(test_w)>args.max_test: test_w=test_w.sample(args.max_test,random_state=args.seed)
 te=make(test_w)
 model=Cls(len(INPUT),args.hidden)
 if init_path:model.enc.load_state_dict(torch.load(init_path,map_location='cpu'))
 opt=torch.optim.Adam(model.parameters(),lr=args.lr);lossfn=nn.BCEWithLogitsLoss();dl=DataLoader(Seq(tra,groups),batch_size=args.batch,shuffle=True)
 model.train();ls=[]
 for x,y in dl:opt.zero_grad();loss=lossfn(model(x),y.float());loss.backward();opt.step();ls.append(float(loss))
 model.eval();pred=[];yy=[]
 with torch.no_grad():
  for x,y in DataLoader(Seq(te,groups),batch_size=args.batch):pred.extend(torch.sigmoid(model(x)).numpy());yy.extend(y.numpy())
 return {'init':init_path or 'random','fraction':frac,'n_train':len(tra),'n_test':len(te),'positive_test':int(sum(yy)),'auprc':average_precision_score(yy,pred),'auroc':roc_auc_score(yy,pred),'brier':brier_score_loss(yy,pred),'train_loss':float(np.mean(ls))}
ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=101);ap.add_argument('--max-pretrain',type=int,default=20000);ap.add_argument('--max-label',type=int,default=10000);ap.add_argument('--fractions',default='0.01,0.05,0.10,0.25,1.0');ap.add_argument('--methods',default='nextmask,nextintensity,valueforecast');ap.add_argument('--epochs',type=int,default=1);ap.add_argument('--batch',type=int,default=512);ap.add_argument('--hidden',type=int,default=64);ap.add_argument('--lr',type=float,default=1e-3);ap.add_argument('--threads',type=int,default=32);ap.add_argument('--max-test',type=int,default=20000);ap.add_argument('--skip-pretrain',action='store_true');args=ap.parse_args();torch.set_num_threads(args.threads);np.random.seed(args.seed);torch.manual_seed(args.seed)
df=pd.read_parquet(D/'stage3_p19_features.parquet'); df['anchor_hour']=df['ICULOS']; part=json.loads((D/'partitions.json').read_text())['cinc2019'][str(args.seed)]
pre={};pre['random']=None
if args.skip_pretrain:
 for method in args.methods.split(','):
  pre[method]=R/f'stage4_encoder_{method}_seed{args.seed}.pt'
else:
 for method in args.methods.split(','):
  path,loss,n=pretrain(method,df,part,args);pre[method]=path;print({'pretrain':method,'loss':loss,'n':n},flush=True)
rows=[]
for init,path in pre.items():
 for frac in [float(x) for x in args.fractions.split(',')]:
  rows.append(downstream(path,df,part,args,frac));print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(R/'stage4_ssl_results.csv',index=False)
