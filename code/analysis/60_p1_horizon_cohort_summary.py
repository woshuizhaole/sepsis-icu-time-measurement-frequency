from pathlib import Path
import json, numpy as np, pandas as pd
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015'); D=P/'data/processed'; O=P/'results/rootcause_stage6'; O.mkdir(parents=True,exist_ok=True)
H=[1,4,6,8,12,24]
h=pd.read_parquet(D/'cinc2019_hourly/data.parquet',columns=['patient_id','ICULOS','SepsisLabel'])
h=h.sort_values(['patient_id','ICULOS']).reset_index(drop=True)
fp=h.loc[h.SepsisLabel.astype(int)>0].groupby('patient_id').ICULOS.min().to_dict()
h['first_positive_proxy']=h.patient_id.map(fp).fillna(-1).astype(int)
maxh=h.groupby('patient_id').ICULOS.max().to_dict(); minh=h.groupby('patient_id').ICULOS.min().to_dict()
rows=[]
for cohort in ['all_record_first_event','pre_first_positive']:
  for horizon in H:
    z=h if cohort=='all_record_first_event' else h[(h.first_positive_proxy<0)|(h.ICULOS<h.first_positive_proxy)]
    target=(z.first_positive_proxy>=0)&(z.ICULOS<z.first_positive_proxy)&(z.first_positive_proxy<=z.ICULOS+horizon)
    avail=z.apply(lambda r: maxh.get(r.patient_id,-1)>=int(r.ICULOS)+horizon,axis=1).to_numpy()
    rows.append({'dataset':'p19','cohort':cohort,'horizon_hours':horizon,'n_rows':len(z),'n_patients':z.patient_id.nunique(),'positive_rows':int(target.sum()),'positive_patients':int(z.loc[target,'patient_id'].nunique()),'availability_rate':float(avail.mean()),'complete_rows':int(avail.sum()),'positive_complete_rows':int((target.to_numpy()&avail).sum())})
# Verify H6 proxy equals stored y_h6 on pre-first windows.
w=pd.read_parquet(D/'cinc2019_windows.parquet',columns=['patient_id','anchor_hour','y_h6','next_window_available'])
wm=w.merge(h[['patient_id','ICULOS','first_positive_proxy']].rename(columns={'ICULOS':'anchor_hour'}),on=['patient_id','anchor_hour'],how='left',validate='one_to_one')
proxy=((wm.first_positive_proxy>=0)&(wm.anchor_hour<wm.first_positive_proxy)&(wm.first_positive_proxy<=wm.anchor_hour+6)).astype(int)
h6_diff=int((proxy.to_numpy()!=wm.y_h6.astype(int).to_numpy()).sum())
# Admission-bundle descriptive removal: all later anchors retained, first 4h history bundle removed from any downstream history.
admission=pd.DataFrame([{'n_all_rows':len(g),'n_after_anchor_gt4':int((g.ICULOS>4).sum()),'patient_id':pid} for pid,g in h.groupby('patient_id',sort=False)])
# deterministic measurement thinning summaries from existing 4h intensity counts.
# Use hourly measurement masks to quantify retained observation burden.
all_cols=pd.read_parquet(D/'cinc2019_hourly/data.parquet',columns=None).columns.tolist()
dyn=[c for c in all_cols if c not in ['patient_id','source_set','Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS','SepsisLabel']]
mask_data=pd.read_parquet(D/'cinc2019_hourly/data.parquet',columns=['patient_id','ICULOS']+dyn)
rng=np.random.default_rng(20261018)
thin=[]
for keep in [1.0,0.5,0.25,0.10]:
  m=mask_data[dyn].notna().to_numpy(dtype=np.int8)
  if keep<1:
    m=(m*(rng.random(m.shape)<keep)).astype(np.int8)
  thin.append({'keep_probability':keep,'n_rows':len(m),'mean_observed_channels_per_hour':float(m.sum(axis=1).mean()),'zero_observation_hour_fraction':float((m.sum(axis=1)==0).mean()),'mean_observed_channels_per_patient_hour':float(m.sum()/len(m))})
out=pd.DataFrame(rows); out.to_csv(O/'p1_horizon_cohort_summary.csv',index=False)
admission.to_csv(O/'p1_admission_bundle_summary.csv',index=False)
pd.DataFrame(thin).to_csv(O/'p1_thinning_summary.csv',index=False)
qc={'status':'PASS' if h6_diff==0 and len(out)==12 else 'FAILED','horizons':H,'cohorts':['all_record_first_event','pre_first_positive'],'rows':int(len(out)),'h6_proxy_diff_vs_stored':h6_diff,'admission_bundle_patients':int(len(admission)),'thinning_rows':len(thin),'outputs':[str(O/'p1_horizon_cohort_summary.csv'),str(O/'p1_admission_bundle_summary.csv'),str(O/'p1_thinning_summary.csv')]}
(O/'p1_horizon_cohort_qc.json').write_text(json.dumps(qc,indent=2)); print(json.dumps(qc,indent=2)); print(out.to_string(index=False)); print(pd.DataFrame(thin).to_string(index=False))
