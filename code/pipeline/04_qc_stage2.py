#!/usr/bin/env python3
from pathlib import Path
import json, yaml, numpy as np, pandas as pd, pyarrow.parquet as pq
from collections import Counter
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015')
PROC=P/'data/processed'; STATS=P/'results/stats'; LOG=P/'logs/qc'
part=json.loads((PROC/'partitions.json').read_text())
issues=[]; checks={}
for ds in ['cinc2019','cinc2012']:
 for seed,m in part[ds].items():
  inv={}
  for pid,sp in m.items(): inv.setdefault(sp,set()).add(pid)
  overlaps=[]
  keys=list(inv)
  for i,a in enumerate(keys):
   for b in keys[i+1:]: overlaps.extend(list(inv[a]&inv[b]))
  checks[f'{ds}_partition_{seed}']={'n':len(m),'counts':{k:len(v) for k,v in inv.items()},'overlap':len(overlaps)}
  if overlaps: issues.append(f'{ds} partition {seed} overlap')
# P19 hourly and windows
p19h=pq.read_table(PROC/'cinc2019_hourly/data.parquet').to_pandas()
p19w=pd.read_parquet(PROC/'cinc2019_windows.parquet')
checks['p19_hourly']={'rows':len(p19h),'patients':p19h.patient_id.nunique(),'duplicate_patient_hour':int(p19h.duplicated(['patient_id','ICULOS']).sum()),'columns':len(p19h.columns)}
checks['p19_windows']={'rows':len(p19w),'patients':p19w.patient_id.nunique(),'anchors_ge_first':int(((p19w.first_positive_proxy>=0) & (p19w.anchor_hour>=p19w.first_positive_proxy)).sum()),'target_not_future':int((p19w.intensity_target_start<=p19w.anchor_hour).sum()),'next_available_rate':float(p19w.next_window_available.mean()),'y_h6_rate':float(p19w.y_h6.mean()),'y_h12_rate':float(p19w.y_h12.mean())}
if checks['p19_hourly']['duplicate_patient_hour']: issues.append('P19 duplicate patient-hour')
if checks['p19_windows']['target_not_future']: issues.append('P19 future target starts at/before anchor')
if checks['p19_windows']['anchors_ge_first']: issues.append('P19 anchor on/after first positive')
# P12
p12h=pq.read_table(PROC/'cinc2012_hourly/data.parquet').to_pandas()
p12l=pd.read_parquet(PROC/'cinc2012_landmarks.parquet')
checks['p12_hourly']={'rows':len(p12h),'patients':p12h.patient_id.nunique(),'duplicate_patient_hour':int(p12h.duplicated(['patient_id','icu_hour']).sum()),'columns':len(p12h.columns)}
checks['p12_landmarks']={'rows':len(p12l),'patients':p12l.patient_id.nunique(),'cutoffs':p12l.cutoff_hour.value_counts().to_dict(),'invalid_availability':int((p12l.max_observed_hour<p12l.cutoff_hour).sum()),'labels':p12l.mortality_label.value_counts().to_dict()}
if checks['p12_hourly']['duplicate_patient_hour']: issues.append('P12 duplicate patient-hour')
if checks['p12_landmarks']['invalid_availability']: issues.append('P12 landmark not available')
# channel yaml
channels=yaml.safe_load((PROC/'measurement_channels.yaml').read_text())
checks['channels']={k:len(v['measurement_channels']) for k,v in channels.items()}
for k,v in channels.items():
 if any(x in v['measurement_channels'] for x in ['SepsisLabel','ICULOS','RecordID','Age','Gender','ICUType']): issues.append(f'{k} administrative channel in measurement list')
summary={'stage':'Stage 2','checks':checks,'issues':issues,'status':'PASS' if not issues else 'FAILED'}
(STATS/'stage2_qc.json').write_text(json.dumps(summary,indent=2,default=str))
print(json.dumps(summary,indent=2,default=str))
raise SystemExit(1 if issues else 0)
