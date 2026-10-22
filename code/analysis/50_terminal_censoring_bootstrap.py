#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time, multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015')
D=P/'data/processed'; R=P/'results/improvement_fair_ssl/terminal_censoring'
METHODS=['random','nextmask4','nextintensity4','valueforecast4']
MODELS=['hour_empirical','history_logistic','history_gbm']
Y=None; INV=None; AVAIL=None; PHAT=None; DRAWS=None

def est(y,p,w=None):
    return np.array([average_precision_score(y,p,sample_weight=w),brier_score_loss(y,p,sample_weight=w)],dtype=float)

def worker(task):
    method,model=task
    p=PHAT[method]
    n=int(INV.max())+1
    out=np.empty((len(DRAWS),2,3),dtype=float)
    # columns: estimand complete/raw/stabilized; metrics AP/Brier
    for i,s in enumerate(DRAWS):
        counts=np.bincount(s,minlength=n)
        wc=counts[INV][AVAIL]
        raw=wc/np.clip(PHAT[model][AVAIL],0.05,0.999)
        stab=raw*np.clip(PHAT['hour_empirical'][AVAIL],0.05,0.999)
        y=Y[AVAIL]; pp=p[AVAIL]
        out[i,:,0]=est(y,pp,wc)
        out[i,:,1]=est(y,pp,raw)
        out[i,:,2]=est(y,pp,stab)
    names=['complete_case','raw_ipw','stabilized_ipw']
    row={'method':method,'terminal_model':model,'fraction':1.0,'bootstrap':len(DRAWS)}
    for j,nm in enumerate(names):
        row[nm+'_auprc_lo']=float(np.percentile(out[:,0,j],2.5)); row[nm+'_auprc_hi']=float(np.percentile(out[:,0,j],97.5))
        row[nm+'_brier_lo']=float(np.percentile(out[:,1,j],2.5)); row[nm+'_brier_hi']=float(np.percentile(out[:,1,j],97.5))
    return row

def main(seed,workers):
    global Y,INV,AVAIL,PHAT,DRAWS
    t0=time.time()
    pred_dir=P/'results/improvement_fair_ssl'
    logits=pd.read_parquet(R/f'terminal_logits_seed{seed}.parquet')
    AVAIL=logits.next_window_available.to_numpy(dtype=bool)
    _,INV=np.unique(logits.patient_id.to_numpy(),return_inverse=True)
    rng=np.random.default_rng(seed*27011+7)
    DRAWS=rng.integers(0,int(INV.max())+1,size=(2000,int(INV.max())+1),dtype=np.int32)
    PHAT={}
    for m in MODELS:
        PHAT[m]={}
    # PHAT[model] stores model availability probabilities, PHAT[method] stores outcome predictions.
    model_p={m:logits['p_available__'+m].to_numpy(dtype=float) for m in MODELS}
    PHAT.update(model_p)
    PHAT['hour_empirical']=model_p['hour_empirical']
    tasks=[]
    for method in METHODS:
        pfile=pred_dir/f'fair_pred_{method}_seed{seed}_f1.00.parquet'
        pred=pd.read_parquet(pfile)
        if len(pred)!=len(logits) or not np.array_equal(pred.patient_id.to_numpy(),logits.patient_id.to_numpy()):
            raise RuntimeError(f'alignment:{seed}:{method}')
        Y=pred.y.to_numpy(dtype=int); PHAT[method]=pred.p.to_numpy(dtype=float)
        for model in MODELS: tasks.append((method,model))
    ctx=mp.get_context('fork')
    with ctx.Pool(processes=workers) as pool: rows=pool.map(worker,tasks)
    out=pd.DataFrame(rows); out.insert(0,'seed',seed)
    out.to_csv(R/f'terminal_censoring_bootstrap_seed{seed}.csv',index=False)
    (R/f'terminal_censoring_bootstrap_seed{seed}.json').write_text(json.dumps({'seed':seed,'runtime_seconds':time.time()-t0,'rows':len(out),'bootstrap':2000,'workers':workers},indent=2))
    print(json.dumps({'seed':seed,'runtime_seconds':time.time()-t0,'rows':len(out)}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--workers',type=int,default=2); a=ap.parse_args(); main(a.seed,a.workers)
