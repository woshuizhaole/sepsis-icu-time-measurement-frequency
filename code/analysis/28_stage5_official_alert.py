#!/usr/bin/env python3
from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np,pandas as pd
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015');D=P/'data/processed';R=P/'results';L=P/'logs/qc'
sys.path.insert(0,str(P/'code/analysis'))
from importlib import import_module
m=import_module('27_stage4_calibration_alert')

def calib_metrics(y,p):
    y=np.asarray(y,dtype=np.int8);p=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7)
    bins=m.calibration_bins(y,p,10)
    return {'n_rows':int(len(y)),'positive_rows':int(y.sum()),'prevalence':float(y.mean()),'mean_pred':float(p.mean()),'brier':float(np.mean((p-y)**2)), 'calibration_intercept':m.fit_calibration(y,p)[0], 'calibration_slope':m.fit_calibration(y,p)[1], 'ece_10':float(sum(x['n']*x['abs_gap'] for x in bins)/len(y)) if bins else float('nan'), 'mce_10':float(max((x['abs_gap'] for x in bins),default=float('nan')))}

def one(seed,thr,raw_groups,windows_by_seed,B=2000):
    root=R/f'stage5_utility_seed{seed}';rows=[];match=0;total=0
    for pf in sorted((root/'predictions').glob('*.psv')):
        pid=pf.stem;pt=pd.read_csv(pf,sep='|');rr=raw_groups.get(pid)
        if rr is None or len(rr)!=len(pt): raise ValueError(f'length mismatch seed={seed} pid={pid}')
        p=pt.PredictedProbability.to_numpy(dtype=float);y=rr.SepsisLabel.to_numpy(dtype=np.int8);h=rr.ICULOS.to_numpy(dtype=float)
        custom=m.refractory_alerts(p,h,float(thr));official=pt.PredictedLabel.to_numpy(dtype=np.int8)
        match += int(np.array_equal(custom,official)); total += 1
        fp=float(rr.loc[rr.SepsisLabel==1,'ICULOS'].min()) if (rr.SepsisLabel==1).any() else -1.0
        rows.append(pd.DataFrame({'patient_id':pid,'hour':h,'y':y,'p':p,'first_positive_proxy':fp}))
    allf=pd.concat(rows,ignore_index=True)
    first=allf.first_positive_proxy.to_numpy(dtype=float);hours=allf.hour.to_numpy(dtype=float)
    full,fa=m.patient_operating_arrays(allf.patient_id.to_numpy(),hours,allf.y.to_numpy(),allf.p.to_numpy(),first,float(thr))
    full_boot=m.bootstrap_summary(allf.y.to_numpy(),allf.p.to_numpy(),allf.patient_id.to_numpy(),fa,m.calibration_bins(allf.y.to_numpy(),allf.p.to_numpy(),10),B,seed+777000)
    # WBM retained parquet predictions are the causal y_h6 window target. The
    # official PSV stream is for current SepsisLabel utility and must not be
    # reused as a pre-first-positive y_h6 calibration target.
    wbm=pd.read_parquet(R/f'stage5_wbm_pred_seed{seed}.parquet')
    win=windows_by_seed[seed]
    if len(wbm)!=len(win) or not np.array_equal(wbm.patient_id.to_numpy(dtype=object),win.patient_id.to_numpy(dtype=object)) or not np.array_equal(wbm.anchor_hour.to_numpy(dtype=int),win.anchor_hour.to_numpy(dtype=int)) or not np.array_equal(wbm.y.to_numpy(dtype=np.int8),win.y_h6.to_numpy(dtype=np.int8)):
        raise ValueError(f'WBM window alignment mismatch seed={seed}')
    predf=pd.DataFrame({'patient_id':wbm.patient_id.astype(str),'hour':wbm.anchor_hour.astype(float),'y':wbm.y.astype(np.int8),'p':wbm.p.astype(float),'first_positive_proxy':win.first_positive_proxy.to_numpy(dtype=float)})
    pre,pa=m.patient_operating_arrays(predf.patient_id.to_numpy(),predf.hour.to_numpy(),predf.y.to_numpy(),predf.p.to_numpy(),predf.first_positive_proxy.to_numpy(),float(thr))
    pre_boot=m.bootstrap_summary(predf.y.to_numpy(),predf.p.to_numpy(),predf.patient_id.to_numpy(),pa,m.calibration_bins(predf.y.to_numpy(),predf.p.to_numpy(),10),B,seed+888000)
    base={'stage':'Stage5_WBM_official_test','method':'WBM','seed':int(seed),'threshold':float(thr),'threshold_source':'Stage5_WBM_validation_utility','scorer_refractory_match_patients':int(match),'scorer_refractory_total_patients':int(total),'n_test_rows':int(len(allf)),'n_test_patients':int(allf.patient_id.nunique()),'full_target':'SepsisLabel_for_official_utility','pre_target':'y_h6_from_retained_stage5_wbm_pred'}
    out=dict(base)
    out.update({f'full_{k}':v for k,v in calib_metrics(allf.y.to_numpy(),allf.p.to_numpy()).items()})
    out.update({f'full_alert_{k}':v for k,v in full.items()})
    out.update({f'full_boot_{k}':v for k,v in full_boot.items()})
    out.update({f'pre_first_positive_{k}':v for k,v in calib_metrics(predf.y.to_numpy(),predf.p.to_numpy()).items()})
    out.update({f'pre_first_positive_alert_{k}':v for k,v in pre.items()})
    out.update({f'pre_first_positive_boot_{k}':v for k,v in pre_boot.items()})
    return out

def main():
    t=time.time();raw=pd.read_parquet(D/'cinc2019_hourly/data.parquet',columns=['patient_id','ICULOS','SepsisLabel'])
    raw_groups={str(pid):g.sort_values('ICULOS').reset_index(drop=True) for pid,g in raw.groupby('patient_id',sort=False)}
    thresholds={int(r.seed):float(r.threshold) for _,r in pd.read_csv(R/'stage5_official_utility_by_seed.csv').iterrows()}
    winall=pd.read_parquet(D/'cinc2019_windows.parquet')
    parts=json.loads((D/'partitions.json').read_text())['cinc2019']
    windows_by_seed={}
    for seed in [101,202,303,404,505]:
        testp={p for p,v in parts[str(seed)].items() if v=='test'}
        windows_by_seed[seed]=winall[winall.patient_id.isin(testp)].sort_values(['patient_id','anchor_hour']).reset_index(drop=True)
    rows=[one(s,thresholds[s],raw_groups,windows_by_seed,2000) for s in [101,202,303,404,505]]
    pd.DataFrame(rows).sort_values('seed').to_csv(R/'stage5_official_alert_metrics.csv',index=False)
    qc={'stage':'Stage5 official alert operating metrics','status':'PASS' if all(r['scorer_refractory_match_patients']==r['scorer_refractory_total_patients'] for r in rows) else 'FAILED','rows':len(rows),'bootstrap_replicates':2000,'bootstrap_unit':'patient','threshold_source':'validation-only official WBM Utility threshold','runtime_seconds':time.time()-t,'raw_patient_groups':len(raw_groups),'notes':['full trajectory metrics use official test predictions and labels','pre_first_positive metrics use retained Stage5 WBM y_h6 predictions aligned to fixed windows and the same validation-only threshold','official full trajectory uses current SepsisLabel for utility; pre-first-positive uses y_h6 and is a distinct estimand']}
    (R/'stage5_official_alert_qc.json').write_text(json.dumps(qc,indent=2));(L/'stage5_official_alert_metrics.log').write_text(json.dumps(qc,indent=2)+'\n')
    print(json.dumps(qc,indent=2))
if __name__=='__main__':main()
