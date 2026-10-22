#!/usr/bin/env python3
from pathlib import Path
import argparse,json,time,multiprocessing as mp
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,brier_score_loss
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015')
D=P/'data/processed'; C=P/'results/improvement_fair_ssl/terminal_censoring'; R=P/'results/improvement_fair_ssl/terminal_controls_all'
INITS=['random','terminal_only','nextmask4','nextintensity4','valueforecast4']; MODELS=['hour_empirical','history_logistic','history_gbm']
Y=None; INV=None; AVAIL=None; TERM=None; PREDS=None; DRAWS=None
def est(y,p,w=None): return np.array([average_precision_score(y,p,sample_weight=w),brier_score_loss(y,p,sample_weight=w)],float)
def worker(task):
    init,model=task; p=PREDS[init]; n=int(INV.max())+1; out=np.empty((len(DRAWS),2,3),float)
    for i,s in enumerate(DRAWS):
        c=np.bincount(s,minlength=n); wc=c[INV][AVAIL]; raw=wc/np.clip(TERM[model][AVAIL],.05,.999); stab=raw*np.clip(TERM['hour_empirical'][AVAIL],.05,.999)
        yy=Y[AVAIL]; pp=p[AVAIL]; out[i,:,0]=est(yy,pp,wc); out[i,:,1]=est(yy,pp,raw); out[i,:,2]=est(yy,pp,stab)
    row={'init':init,'terminal_model':model,'fraction':1.0,'bootstrap':len(DRAWS)}
    for j,nm in enumerate(['complete_case','raw_ipw','stabilized_ipw']):
        row[nm+'_auprc_lo']=float(np.percentile(out[:,0,j],2.5)); row[nm+'_auprc_hi']=float(np.percentile(out[:,0,j],97.5)); row[nm+'_brier_lo']=float(np.percentile(out[:,1,j],2.5)); row[nm+'_brier_hi']=float(np.percentile(out[:,1,j],97.5))
    return row
def main(seed,workers):
    global Y,INV,AVAIL,TERM,PREDS,DRAWS
    t=time.time(); lg=pd.read_parquet(C/f'terminal_logits_seed{seed}.parquet'); AVAIL=lg.next_window_available.to_numpy(dtype=bool); _,INV=np.unique(lg.patient_id.to_numpy(),return_inverse=True); rng=np.random.default_rng(seed*27011+7); DRAWS=rng.integers(0,int(INV.max())+1,size=(2000,int(INV.max())+1),dtype=np.int32)
    TERM={m:np.clip(lg[f'p_available__{m}'].to_numpy(float),.05,.999) for m in MODELS}; PREDS={}
    for init in INITS:
        pred=pd.read_parquet(R/f'fair_pred_{init}_seed{seed}_f1.00.parquet')
        if len(pred)!=len(lg) or not np.array_equal(pred.patient_id.to_numpy(),lg.patient_id.to_numpy()): raise RuntimeError(f'align:{seed}:{init}')
        Y=pred.y.to_numpy(int); PREDS[init]=pred.p.to_numpy(float)
    with mp.get_context('fork').Pool(processes=workers) as pool: rows=pool.map(worker,[(i,m) for i in INITS for m in MODELS])
    out=pd.DataFrame(rows); out.insert(0,'seed',seed); out.to_csv(R/f'terminal_aux_bootstrap_seed{seed}.csv',index=False); (R/f'terminal_aux_bootstrap_seed{seed}.json').write_text(json.dumps({'seed':seed,'runtime_seconds':time.time()-t,'rows':len(out),'bootstrap':2000,'workers':workers},indent=2)); print({'seed':seed,'runtime_seconds':time.time()-t,'rows':len(out)},flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--workers',type=int,default=2); a=ap.parse_args(); main(a.seed,a.workers)
