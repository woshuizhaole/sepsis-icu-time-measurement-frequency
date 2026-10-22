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

def cols(cond):
 if cond=='V': return BASE+[f'v__{c}' for c in DYN]
 if cond=='M0': return BASE+[f'm0__{c}' for c in DYN]
 if cond=='Mh': return BASE+[f'm12__{c}' for c in DYN]
 if cond=='I': return BASE+[f'm4__{c}' for c in DYN]
 if cond=='V+M0': return BASE+[f'v__{c}' for c in DYN]+[f'm0__{c}' for c in DYN]
 if cond=='V+M0+I': return BASE+[f'v__{c}' for c in DYN]+[f'm0__{c}' for c in DYN]+[f'm4__{c}' for c in DYN]
 if cond=='V+Mh': return BASE+[f'v__{c}' for c in DYN]+[f'm12__{c}' for c in DYN]
 if cond=='V+Mh+I': return BASE+[f'v__{c}' for c in DYN]+[f'm12__{c}' for c in DYN]+[f'm4__{c}' for c in DYN]
 raise KeyError(cond)
class SeqDataset(Dataset):
 def __init__(self, items, groups, max_len=12): self.items=items; self.groups=groups; self.max_len=max_len
 def __len__(self): return len(self.items)
 def __getitem__(self,i):
  pid,hour,y=self.items[i]; arr,min_hour=self.groups[pid]; idx=min(max(int(hour)-int(min_hour),0),len(arr)-1); lo=max(0,idx-self.max_len+1); x=arr[lo:idx+1]
  if len(x)<self.max_len: x=np.concatenate([np.repeat(x[:1],self.max_len-len(x),axis=0),x],axis=0)
  return torch.from_numpy(x.astype('float32')),torch.tensor(y,dtype=torch.float32)
class GRUModel(nn.Module):
 def __init__(self,d,h=64): super().__init__(); self.gru=nn.GRU(d,h,batch_first=True); self.head=nn.Linear(h,1)
 def forward(self,x): z,_=self.gru(x); return self.head(z[:,-1]).squeeze(-1)
def build_groups(df,selected,train_pids):
 med=df.loc[df.patient_id.isin(train_pids),selected].median(numeric_only=True).fillna(0)
 work=df[['patient_id','anchor_hour']+selected].copy().sort_values(['patient_id','anchor_hour'])
 for c in selected: work[c]=pd.to_numeric(work[c],errors='coerce').fillna(med.get(c,0)).astype('float32')
 return {pid:(g[selected].to_numpy(dtype='float32'), int(g.anchor_hour.min())) for pid,g in work.groupby('patient_id',sort=False)}
def index_items(win,part,groups,max_n,seed):
 w=win[win.patient_id.isin(groups.keys())].copy(); w=w[w.patient_id.map(part).notna()]
 pos=w[w.y_h6==1]; neg=w[w.y_h6==0]
 rng=np.random.default_rng(seed)
 if max_n and len(w)>max_n:
  nneg=max(0,max_n-len(pos)); neg=neg.iloc[rng.choice(len(neg),size=min(nneg,len(neg)),replace=False)]
  w=pd.concat([pos,neg])
 items=[]
 for r in w.itertuples(index=False):
  # stage3 feature rows have hourly rows; find index by arithmetic from min hour (ICULOS starts 1)
  items.append((r.patient_id,int(r.anchor_hour),int(r.y_h6)))
 return items
def run(cond,args):
 torch.set_num_threads(args.threads); seed=args.seed; np.random.seed(seed); torch.manual_seed(seed)
 df=pd.read_parquet(D/'stage3_p19_features.parquet'); df['anchor_hour']=df['ICULOS']; win=pd.read_parquet(D/'cinc2019_windows.parquet')
 part=json.loads((D/'partitions.json').read_text())['cinc2019'][str(seed)]
 train_pids={k for k,v in part.items() if v=='train'}; val_pids={k for k,v in part.items() if v=='validation'}; test_pids={k for k,v in part.items() if v=='test'}
 selected=cols(cond); groups=build_groups(df,selected,train_pids)
 # Include val/test groups with train medians via rebuilding all groups from train medians is unnecessary for this smoke; build with same schema and fill zero where missing.
 for pid,g in df.groupby('patient_id',sort=False):
  if pid in groups: continue
  x=g.sort_values('anchor_hour')[selected].copy()
  train_med=df[df.patient_id.isin(train_pids)][selected].median(numeric_only=True).fillna(0)
  x=x.apply(pd.to_numeric,errors='coerce').fillna(train_med).fillna(0).astype('float32'); groups[pid]=(x.to_numpy(),min_hour)
 tritems=index_items(win[win.patient_id.map(part)=='train'],part,{k:v for k,v in groups.items() if k in train_pids},args.max_train,seed)
 vaitems=index_items(win[win.patient_id.map(part)=='validation'],part,{k:v for k,v in groups.items() if k in val_pids},args.max_val,seed)
 teitems=index_items(win[win.patient_id.map(part)=='test'],part,{k:v for k,v in groups.items() if k in test_pids},args.max_test,seed)
 tr=DataLoader(SeqDataset(tritems,{k:groups[k] for k in train_pids}),batch_size=args.batch,shuffle=True,num_workers=0)
 va=DataLoader(SeqDataset(vaitems,{k:groups[k] for k in val_pids}),batch_size=args.batch,shuffle=False,num_workers=0)
 te=DataLoader(SeqDataset(teitems,{k:groups[k] for k in test_pids}),batch_size=args.batch,shuffle=False,num_workers=0)
 model=GRUModel(len(selected),args.hidden); opt=torch.optim.Adam(model.parameters(),lr=args.lr); lossfn=nn.BCEWithLogitsLoss()
 best=None; best_state=None
 for ep in range(args.epochs):
  model.train(); losses=[]
  for x,y in tr:
   opt.zero_grad(); loss=lossfn(model(x),y); loss.backward(); opt.step(); losses.append(float(loss))
  model.eval(); yp=[]; yy=[]
  with torch.no_grad():
   for x,y in va: yp.extend(torch.sigmoid(model(x)).numpy()); yy.extend(y.numpy())
  score=average_precision_score(yy,yp); print({'condition':cond,'epoch':ep,'train_loss':float(np.mean(losses)),'val_auprc':score},flush=True)
  if best is None or score>best: best=score; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
 if best_state: model.load_state_dict(best_state)
 model.eval(); yp=[]; yy=[]
 with torch.no_grad():
  for x,y in te: yp.extend(torch.sigmoid(model(x)).numpy()); yy.extend(y.numpy())
 metrics={'condition':cond,'seed':seed,'n_train':len(tritems),'n_val':len(vaitems),'n_test':len(teitems),'positive_test':int(sum(yy)),'val_best_auprc':best,'test_auprc':average_precision_score(yy,yp),'test_auroc':roc_auc_score(yy,yp),'test_brier':brier_score_loss(yy,yp),'input_dim':len(selected),'epochs':args.epochs}
 pd.DataFrame({'y':yy,'p':yp}).to_parquet(R/f'stage3_gru_{cond.replace("+","_")}_seed{seed}_pred.parquet',index=False)
 torch.save(model.state_dict(),R/f'stage3_gru_{cond.replace("+","_")}_seed{seed}.pt')
 print(metrics,flush=True); return metrics
ap=argparse.ArgumentParser();ap.add_argument('--conditions',default='V');ap.add_argument('--seed',type=int,default=101);ap.add_argument('--max-train',type=int,default=50000);ap.add_argument('--max-val',type=int,default=50000);ap.add_argument('--max-test',type=int,default=100000);ap.add_argument('--epochs',type=int,default=1);ap.add_argument('--batch',type=int,default=512);ap.add_argument('--hidden',type=int,default=64);ap.add_argument('--lr',type=float,default=1e-3);ap.add_argument('--threads',type=int,default=32);args=ap.parse_args()
res=[run(c,args) for c in args.conditions.split(',')];pd.DataFrame(res).to_csv(R/'stage3_gru_results.csv',index=False)
