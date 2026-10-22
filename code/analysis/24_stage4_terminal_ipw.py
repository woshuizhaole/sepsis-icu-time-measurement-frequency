from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015');D=P/'data/processed';R=P/'results'
rows=[]
win=pd.read_parquet(D/'cinc2019_windows.parquet',columns=['patient_id','anchor_hour','y_h6','next_window_available'])
parts=json.loads((D/'partitions.json').read_text())['cinc2019']
res=pd.read_csv(R/'stage4_ssl_full_results.csv')
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=None);ap.add_argument('--init',default=None);ap.add_argument('--fraction',type=float,default=None);args=ap.parse_args()
if args.seed is not None: res=res[res.seed==args.seed]
if args.init is not None: res=res[res['init']==args.init]
if args.fraction is not None: res=res[np.isclose(res.fraction,args.fraction)]
for _,meta in res.iterrows():
 seed=int(meta.seed);init=str(meta['init']);frac=float(meta.fraction);part=parts[str(seed)]
 tew=win[win.patient_id.map(part)=='test']
 pred=pd.read_parquet(R/f"stage4_full_pred_{init}_seed{seed}_f{frac:.2f}.parquet")
 if len(pred)!=len(tew): raise RuntimeError((seed,init,frac,len(pred),len(tew)))
 z=pred.copy();z['avail']=tew.next_window_available.to_numpy(dtype=bool);z['hour']=tew.anchor_hour.to_numpy(dtype=int);y=z.y.to_numpy(dtype=int);p=z.p.to_numpy(dtype=float)
 trw=win[win.patient_id.map(part)=='train'];ph=trw.groupby('anchor_hour').next_window_available.mean().to_dict();w=np.array([1.0/max(float(ph.get(h,trw.next_window_available.mean())),0.05) for h in z.hour],float);w[~z.avail]=0;valid=z.avail.to_numpy()
 base={'seed':seed,'init':init,'fraction':frac,'n_test':len(z),'n_available':int(valid.sum()),'availability_rate':float(valid.mean()),'positive_all':int(y.sum()),'positive_available':int(y[valid].sum())}
 base.update({'auprc_all':float(average_precision_score(y,p)),'auroc_all':float(roc_auc_score(y,p)),'brier_all':float(brier_score_loss(y,p))})
 if valid.sum()>0 and y[valid].sum()>0:
  ww=w[valid];base.update({'auprc_complete_case':float(average_precision_score(y[valid],p[valid])),'auroc_complete_case':float(roc_auc_score(y[valid],p[valid])),'brier_complete_case':float(brier_score_loss(y[valid],p[valid])),'auprc_marginal_ipw':float(average_precision_score(y[valid],p[valid],sample_weight=ww)),'auroc_marginal_ipw':float(roc_auc_score(y[valid],p[valid],sample_weight=ww)),'brier_marginal_ipw':float(brier_score_loss(y[valid],p[valid],sample_weight=ww))})
 else:
  for k in ['auprc_complete_case','auroc_complete_case','brier_complete_case','auprc_marginal_ipw','auroc_marginal_ipw','brier_marginal_ipw']:base[k]=np.nan
 rows.append(base)
out=pd.DataFrame(rows);out.to_csv(R/(f'stage4_terminal_ipw_eval_seed{args.seed}_{args.init}_{args.fraction:.2f}.csv' if args.init is not None else (f'stage4_terminal_ipw_eval_seed{args.seed}.csv' if args.seed is not None else 'stage4_terminal_ipw_eval.csv')),index=False);print(out.shape);print(out[['init','fraction','availability_rate','auprc_complete_case','auprc_marginal_ipw']].groupby(['init','fraction']).median().to_string())

