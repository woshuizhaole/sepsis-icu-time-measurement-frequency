from pathlib import Path
import argparse,json
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015');D=P/'data/processed';R=P/'results'
class Enc(nn.Module):
 def __init__(self,d,h=64):super().__init__();self.gru=nn.GRU(d,h,batch_first=True)
 def forward(self,x):return self.gru(x)[0][:,-1]
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
 part=json.loads((D/'partitions.json').read_text())['cinc2019'][str(args.seed)];win=pd.read_parquet(D/'cinc2019_windows.parquet',columns=['patient_id','anchor_hour','y_h6']);meta=pd.read_parquet(D/'p19_sequence_meta.parquet');seq=np.load(D/f'p19_sequences_12h_seed{args.seed}.npy',mmap_mode='r')
 def split(sp):
  w=win[win.patient_id.map(part)==sp].sort_values(['patient_id','anchor_hour']);z=w.merge(meta,left_on=['patient_id','anchor_hour'],right_on=['patient_id','ICULOS'],how='left',validate='one_to_one');return seq[z.row_idx.to_numpy()],z.y_h6.to_numpy('float32'),z.patient_id.to_numpy()
 splitdata={s:split(s) for s in ['train','validation','test']};rows=[]
 for method in ['nextmask','nextintensity','valueforecast']:
  enc=Enc(108,64);enc.load_state_dict(torch.load(R/f'stage4_encoder_{method}_seed{args.seed}_full.pt',map_location='cpu'));enc.eval();feat={}
  for sp,(X,y,pids) in splitdata.items():
   out=[]
   for (xb,) in DataLoader(TensorDataset(torch.from_numpy(np.asarray(X))),batch_size=args.batch,shuffle=False):
    with torch.no_grad():out.append(enc(xb))
   feat[sp]=torch.cat(out).numpy()
  ytr=splitdata['train'][1];yva=splitdata['validation'][1];yte=splitdata['test'][1];pids_tr=splitdata['train'][2];pids_te=splitdata['test'][2]
  for frac in [.01,.05,.10,.25,1.0]:
   sel=choose(win,part,frac,args.seed);idx=np.array([p in sel for p in pids_tr]);head=nn.Linear(64,1);opt=torch.optim.Adam(head.parameters(),lr=1e-3);lossf=nn.BCEWithLogitsLoss();best=-1;state=None;dl=DataLoader(TensorDataset(torch.from_numpy(feat['train'][idx]).float(),torch.from_numpy(ytr[idx]).float()),batch_size=args.head_batch,shuffle=True)
   for ep in range(3):
    head.train()
    for xb,yb in dl:opt.zero_grad();loss=lossf(head(xb).squeeze(-1),yb);loss.backward();opt.step()
    head.eval();vp=torch.sigmoid(head(torch.from_numpy(feat['validation']).float()).squeeze(-1)).detach().numpy();sc=average_precision_score(yva,vp)
    if sc>best:best=sc;state={k:v.detach().clone() for k,v in head.state_dict().items()}
   head.load_state_dict(state);head.eval();p=torch.sigmoid(head(torch.from_numpy(feat['test']).float()).squeeze(-1)).detach().numpy();ci=boot(yte,p,pids_te,args.bootstrap,args.seed+int(frac*10000));row={'seed':args.seed,'init':method,'fraction':frac,'n_train_patients':len(sel),'n_train_windows':int(idx.sum()),'n_test_windows':len(yte),'positive_test':int(yte.sum()),'val_best_auprc':float(best),'auprc':float(average_precision_score(yte,p)),'auroc':float(roc_auc_score(yte,p)),'brier':float(brier_score_loss(yte,p))};row.update(dict(zip(['auprc_lo','auprc_hi','auroc_lo','auroc_hi','brier_lo','brier_hi'],ci)));rows.append(row)
 pd.DataFrame(rows).to_csv(R/f'stage4_frozen_head_results_seed{args.seed}.csv',index=False);print({'seed':args.seed,'rows':len(rows)},flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--batch',type=int,default=8192);ap.add_argument('--head-batch',type=int,default=16384);ap.add_argument('--threads',type=int,default=8);ap.add_argument('--bootstrap',type=int,default=2000);main(ap.parse_args())

