from pathlib import Path
import argparse, json, random
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015'); D=P/'data/processed'; R=P/'results/improvement_fair_ssl'; OUT=R/'calibration'; OUT.mkdir(parents=True,exist_ok=True)
DYN=['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets']; BASE=['Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS']; INPUT=BASE+[f'v__{c}' for c in DYN]+[f'm0__{c}' for c in DYN]+[f'm4__{c}' for c in DYN]
class Seq(Dataset):
 def __init__(self,items,groups): self.items=items; self.groups=groups
 def __len__(self): return len(self.items)
 def __getitem__(self,i):
  pid,h,y=self.items[i]; arr,minh=self.groups[pid]; idx=min(max(int(h)-int(minh),0),len(arr)-1); lo=max(0,idx-11); x=arr[lo:idx+1]
  if len(x)<12:x=np.concatenate([np.repeat(x[:1],12-len(x),0),x],0)
  return torch.from_numpy(x.astype('float32')),torch.tensor(y)
class Enc(nn.Module):
 def __init__(self,d,h=64): super().__init__(); self.gru=nn.GRU(d,h,batch_first=True)
 def forward(self,x): return self.gru(x)[0][:,-1]
class Cls(nn.Module):
 def __init__(self,d,h=64): super().__init__(); self.enc=Enc(d,h); self.head=nn.Linear(h,1)
 def forward(self,x): return self.head(self.enc(x)).squeeze(-1)
def seed(x): random.seed(x); np.random.seed(x); torch.manual_seed(x); torch.cuda.manual_seed_all(x)
def groups(df,pids,med,scale):
 w=df[df.patient_id.isin(pids)][['patient_id','anchor_hour']+INPUT].sort_values(['patient_id','anchor_hour']).reset_index(drop=True); a=w[INPUT].to_numpy('float32'); medarr=med.reindex(INPUT).fillna(0).to_numpy('float32'); sc=scale.reindex(INPUT).fillna(1).to_numpy('float32'); bad=~np.isfinite(a); a[bad]=np.broadcast_to(medarr,a.shape)[bad]; vs=len(BASE); ve=vs+len(DYN); a[:,vs:ve]=(a[:,vs:ve]-medarr[vs:ve])/np.maximum(sc[vs:ve],1e-6); ids=w.patient_id.to_numpy(); h=w.anchor_hour.to_numpy(); cuts=np.r_[0,1+np.flatnonzero(ids[1:]!=ids[:-1]),len(ids)]; return {ids[x]:(a[x:y],int(h[x])) for x,y in zip(cuts[:-1],cuts[1:])}
def items(win,pids): return [(r.patient_id,int(r.anchor_hour),int(r.y_h6)) for r in win.itertuples() if r.patient_id in pids]
def solve_offset(logit,y):
 target=float(np.mean(y)); lo,hi=-20.,20.
 for _ in range(80):
  mid=(lo+hi)/2; m=float(np.mean(1/(1+np.exp(-np.clip(logit+mid,-40,40)))))
  if m>target: hi=mid
  else: lo=mid
 return (lo+hi)/2
def boot(y,p,pids,b,rng):
 uniq,inv=np.unique(pids,return_inverse=True); n=len(uniq); vals=[]
 for st in range(0,b,100):
  k=min(100,b-st); ss=rng.integers(0,n,size=(k,n))
  for s in ss:
   w=np.bincount(s,minlength=n)[inv]; vals.append((average_precision_score(y,p,sample_weight=w),brier_score_loss(y,p,sample_weight=w)))
 z=np.asarray(vals); return np.nanpercentile(z,2.5,0),np.nanpercentile(z,97.5,0)
def run(seedv,device):
 seed(seedv); df=pd.read_parquet(D/'stage3_p19_features.parquet'); df['anchor_hour']=df['ICULOS']; win=pd.read_parquet(D/'cinc2019_windows.parquet',columns=['patient_id','anchor_hour','y_h6']); part=json.loads((D/'partitions.json').read_text())['cinc2019'][str(seedv)]; trainp={k for k,v in part.items() if v=='train'}; med=df[df.patient_id.isin(trainp)][INPUT].median(numeric_only=True).fillna(0); sc=pd.Series(1.,index=INPUT); sc.loc[[f'v__{c}' for c in DYN]]=df[df.patient_id.isin(trainp)][[f'v__{c}' for c in DYN]].std().fillna(1).replace(0,1); g=groups(df,set(part),med,sc); trw=win[win.patient_id.isin(trainp)]; vaw=win[win.patient_id.map(part)=='validation']; tew=win[win.patient_id.map(part)=='test']; methods=['random','nextmask4','nextintensity4','valueforecast4']; rows=[]
 for method in methods:
  seed(seedv*1000+sum(map(ord,method))); model=Cls(len(INPUT)); ep=R/f'fair_encoder_{method}_seed{seedv}.pt'
  if method!='random': model.enc.load_state_dict(torch.load(ep,map_location='cpu'))
  model.to(device); opt=torch.optim.Adam(model.parameters(),1e-3); loss=nn.BCEWithLogitsLoss(); tr=DataLoader(Seq(items(trw,set(trw.patient_id)),g),batch_size=1024,shuffle=True); va=DataLoader(Seq(items(vaw,set(vaw.patient_id)),g),batch_size=2048); te=DataLoader(Seq(items(tew,set(tew.patient_id)),g),batch_size=2048); best=-1; beststate=None
  for e in range(3):
   model.train()
   for x,y in tr: opt.zero_grad(); z=loss(model(x.to(device)),y.float().to(device)); z.backward(); opt.step()
   model.eval(); vp=[];vy=[]
   with torch.no_grad():
    for x,y in va: vp.extend(model(x.to(device)).cpu().numpy()); vy.extend(y.numpy())
   scv=average_precision_score(vy,1/(1+np.exp(-np.clip(vp,-40,40))))
   if scv>best: best=scv; beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  model.load_state_dict(beststate); model.eval(); vl=[];vy=[]; tl=[];ty=[]
  with torch.no_grad():
   for x,y in va: vl.extend(model(x.to(device)).cpu().numpy()); vy.extend(y.numpy())
   for x,y in te: tl.extend(model(x.to(device)).cpu().numpy()); ty.extend(y.numpy())
  tpids=np.array([r.patient_id for r in tew.itertuples()],object); vl=np.asarray(vl); tl=np.asarray(tl); vy=np.asarray(vy); ty=np.asarray(ty); off=solve_offset(vl,vy); raw=1/(1+np.exp(-np.clip(tl,-40,40))); cal=1/(1+np.exp(-np.clip(tl+off,-40,40))); rng=np.random.default_rng(seedv+sum(map(ord,method))); rlo,rhi=boot(ty,raw,tpids,2000,rng); clo,chi=boot(ty,cal,tpids,2000,rng)
  rows += [{'seed':seedv,'method':method,'val_prevalence':float(vy.mean()),'offset':off,'test_prevalence':float(ty.mean()),'raw_mean_p':float(raw.mean()),'cal_mean_p':float(cal.mean()),'raw_auprc':average_precision_score(ty,raw),'cal_auprc':average_precision_score(ty,cal),'raw_brier':brier_score_loss(ty,raw),'cal_brier':brier_score_loss(ty,cal),'raw_auprc_lo':rlo[0],'raw_auprc_hi':rhi[0],'raw_brier_lo':rlo[1],'raw_brier_hi':rhi[1],'cal_auprc_lo':clo[0],'cal_auprc_hi':chi[0],'cal_brier_lo':clo[1],'cal_brier_hi':chi[1],'val_best_auprc':best}]
 pd.DataFrame(rows).to_csv(OUT/f'fair_calibration_seed{seedv}.csv',index=False); print(seedv,rows,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--device',default='cpu'); a=ap.parse_args(); run(a.seed,torch.device(a.device))
