#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,shutil
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015');D=P/'data/processed';R=P/'results';EXT=P/'external/evaluation-2019'
sys.path.insert(0,str(P/'code/analysis')); sys.path.insert(0,str(EXT))
from stage5_wbm_import import WBMGRU,WinDS,make_groups,BASE,INPUT,DYN,MASK_SL
from evaluate_sepsis_score import evaluate_sepsis_score,compute_prediction_utility

def refractory(p,thr):
    z=(np.asarray(p)>=thr).astype(int)
    for i in range(len(z)):
        if z[i]: z[i+1:min(len(z),i+13)]=0
    return z

def main(args):
    device=torch.device(args.device); torch.set_num_threads(args.threads)
    df=pd.read_parquet(D/'stage3_p19_features.parquet'); df['anchor_hour']=df.ICULOS
    raw=pd.read_parquet(D/'cinc2019_hourly/data.parquet',columns=['patient_id','ICULOS','SepsisLabel']).sort_values(['patient_id','ICULOS'])
    part=json.loads((D/'partitions.json').read_text())['cinc2019'][str(args.seed)]
    trainp={k for k,v in part.items() if v=='train'}; med=df[df.patient_id.isin(trainp)][INPUT].median(numeric_only=True).fillna(0); groups=make_groups(df,set(part.keys()),med)
    model=WBMGRU(len(INPUT),args.hidden).to(device); model.load_state_dict(torch.load(R/f'stage5_wbm_seed{args.seed}.pt',map_location='cpu')); model.eval()
    def split_pred(split):
        w=df[df.patient_id.map(part)==split][['patient_id','anchor_hour']].sort_values(['patient_id','anchor_hour'])
        items=[]
        for r in w.itertuples(index=False):
            arr,minh=groups[r.patient_id]; idx=int(r.anchor_hour)-minh; av=int(idx+1<len(arr)); nm=np.zeros(len(DYN),dtype='float32'); items.append((r.patient_id,int(r.anchor_hour),0,nm,av))
        dl=DataLoader(WinDS(items,groups),batch_size=args.batch,shuffle=False,num_workers=0); pp=[]
        with torch.no_grad():
            for x,y,nm,av in dl:
                sp,_,_=model(x.to(device));pp.extend(torch.sigmoid(sp).cpu().numpy())
        return pd.DataFrame({'patient_id':[x[0] for x in items],'ICULOS':[x[1] for x in items],'p':np.asarray(pp)})
    valp=split_pred('validation'); testp=split_pred('test')
    lab=raw[raw.patient_id.isin(testp.patient_id)|raw.patient_id.isin(valp.patient_id)].copy()
    lab['patient_id']=lab.patient_id.astype(str)
    valm=valp.merge(raw,on=['patient_id','ICULOS'],how='left',validate='one_to_one')
    best=(-np.inf,0.5)
    for thr in np.linspace(0.01,0.99,99):
        u=0.0
        for pid,g in valm.groupby('patient_id',sort=False):
            u+=compute_prediction_utility(g.SepsisLabel.to_numpy(dtype=int),refractory(g.p.to_numpy(),thr))
        if u>best[0]: best=(u,float(thr))
    thr=best[1]; outroot=R/f'stage5_utility_seed{args.seed}'; ld=outroot/'labels';pdst=outroot/'predictions'
    if outroot.exists(): shutil.rmtree(outroot)
    ld.mkdir(parents=True);pdst.mkdir(parents=True)
    testm=testp.merge(raw,on=['patient_id','ICULOS'],how='left',validate='one_to_one')
    for pid,g in testm.groupby('patient_id',sort=False):
        g=g.sort_values('ICULOS'); (ld/f'{pid}.psv').write_text('SepsisLabel\n'+'\n'.join(str(int(x)) for x in g.SepsisLabel)+'\n')
        z=refractory(g.p.to_numpy(),thr); (pdst/f'{pid}.psv').write_text('PredictedProbability|PredictedLabel\n'+'\n'.join(f'{float(p):.10g}|{int(q)}' for p,q in zip(g.p,z))+'\n')
    auroc,auprc,acc,fm,util=evaluate_sepsis_score(str(ld),str(pdst))
    out={'seed':args.seed,'threshold_validation_utility':best[0],'threshold':thr,'test_auroc_official':float(auroc),'test_auprc_official':float(auprc),'accuracy':float(acc),'f_measure':float(fm),'normalized_utility':float(util),'n_test_patients':int(testm.patient_id.nunique()),'n_test_rows':int(len(testm))}
    (outroot/'official_scores.json').write_text(json.dumps(out,indent=2)); print(out,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--device',default='cpu');ap.add_argument('--threads',type=int,default=10);ap.add_argument('--batch',type=int,default=2048);ap.add_argument('--hidden',type=int,default=64);main(ap.parse_args())

