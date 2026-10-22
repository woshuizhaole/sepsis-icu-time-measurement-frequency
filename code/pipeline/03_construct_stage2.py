#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math, os, time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from sklearn.model_selection import train_test_split

PROJECT=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015')
RAW=PROJECT/'data/raw'; PROC=PROJECT/'data/processed'; STATS=PROJECT/'results/stats'; TABLES=PROJECT/'results/tables'; LOGS=PROJECT/'logs/qc'
for d in [PROC,STATS,TABLES,LOGS]: d.mkdir(parents=True,exist_ok=True)
P19_DYNAMIC=['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets']
P19_STATIC=['Age','Gender','Unit1','Unit2','HospAdmTime']
P12_STATIC=['RecordID','Age','Gender','Height','ICUType','Weight']
P12_FILES=sorted((RAW/'cinc2012').glob('set-*/*.txt'))
P19_FILES=sorted((RAW/'cinc2019'/'training'/'training_setA').glob('*.psv'))+sorted((RAW/'cinc2019'/'training'/'training_setB').glob('*.psv'))

def p19_label(path):
    first=path.stem
    with path.open() as f:
        h=f.readline().rstrip().split('|'); i=h.index('SepsisLabel')
        pos=0; n=0
        for line in f:
            vals=line.rstrip().split('|'); n+=1; pos += int(vals[i]=='1')
    return first, int(pos>0), path.parent.name

def make_partition(ids, labels, seed):
    ids=np.array(ids); labels=np.array(labels)
    tr, rest, ytr, yrest=train_test_split(ids, labels, test_size=0.30, stratify=labels, random_state=seed)
    va, te, _, _=train_test_split(rest, yrest, test_size=0.50, stratify=yrest, random_state=seed)
    return {str(x):'train' for x in tr} | {str(x):'validation' for x in va} | {str(x):'test' for x in te}

# Splits based on patient-level outcomes/positive presence only.
p19_meta=[p19_label(p) for p in P19_FILES]
p19_ids=[x[0] for x in p19_meta]; p19_y=[x[1] for x in p19_meta]
p12_outcomes={}
for out in ['Outcomes-a.txt','Outcomes-b.txt']:
    f=RAW/'cinc2012'/out
    with f.open() as fh:
        for row in csv.DictReader(fh): p12_outcomes[row['RecordID']]=int(row['In-hospital_death'])
p12_ids=[p.stem for p in P12_FILES]
p12_y=[p12_outcomes.get(i) for i in p12_ids]
assert all(y is not None for y in p12_y)
partitions={'cinc2019':{},'cinc2012':{}}
for seed in [101,202,303,404,505]:
    partitions['cinc2019'][str(seed)]=make_partition(p19_ids,p19_y,seed)
    partitions['cinc2012'][str(seed)]=make_partition(p12_ids,p12_y,seed)
(PROC/'partitions.json').write_text(json.dumps(partitions,indent=2))
part_summary={}
for ds in partitions:
    part_summary[ds]={}
    for seed,m in partitions[ds].items(): part_summary[ds][seed]=dict(Counter(m.values()))
(STATS/'stage2_partition_summary.json').write_text(json.dumps(part_summary,indent=2))

# Channel whitelist.
p12_params=set()
for path in P12_FILES[:100]:
    with path.open(errors='replace') as fh:
        for line in fh:
            if line.startswith('Time,Parameter,Value') or not line.strip(): continue
            parts=line.rstrip().split(',',2)
            if len(parts)==3: p12_params.add(parts[1])
# Scan all parameter names cheaply from already downloaded primary tracks.
for path in P12_FILES:
    with path.open(errors='replace') as fh:
        for line in fh:
            if line.startswith('Time,Parameter,Value') or not line.strip(): continue
            parts=line.rstrip().split(',',2)
            if len(parts)==3: p12_params.add(parts[1])
p12_dynamic=sorted(p12_params-set(P12_STATIC))
channels={'cinc2019':{'measurement_channels':P19_DYNAMIC,'static_channels':P19_STATIC,'excluded_channels':['ICULOS','SepsisLabel']},'cinc2012':{'measurement_channels':p12_dynamic,'static_channels':P12_STATIC,'excluded_channels':['Time','RecordID']}}
(PROC/'measurement_channels.yaml').write_text(yaml.safe_dump(channels,sort_keys=False,allow_unicode=True))

def normalize_batch(batch):
    for c in batch.columns:
        if c in ('patient_id','source_set'):
            continue
        batch[c] = pd.to_numeric(batch[c], errors='coerce').astype('float64')
    return batch

# P19 hourly table in parquet batches.
p19_out=PROC/'cinc2019_hourly'; p19_out.mkdir(exist_ok=True)
def read_p19(path):
    df=pd.read_csv(path,sep='|')
    df['patient_id']=path.stem; df['source_set']=path.parent.name
    keep=['patient_id','source_set']+P19_STATIC+['ICULOS','SepsisLabel']+P19_DYNAMIC
    return df[[c for c in keep if c in df.columns]]
writer=None; part=0; p19_rows=0
for start in range(0,len(P19_FILES),500):
    frames=[read_p19(p) for p in P19_FILES[start:start+500]]
    batch=normalize_batch(pd.concat(frames,ignore_index=True))
    table=pa.Table.from_pandas(batch,preserve_index=False)
    if writer is None: writer=pq.ParquetWriter(str(p19_out/'data.parquet'),table.schema,compression='zstd')
    writer.write_table(table); p19_rows += len(batch)
if writer: writer.close()

# P12 hourly table. Event rows count as recorded events; value=-1 is retained in raw event logic and becomes NaN for value columns.
p12_hourly_out=PROC/'cinc2012_hourly'; p12_hourly_out.mkdir(exist_ok=True)
def read_p12(path):
    events=defaultdict(list); static={}
    max_hour=0
    with path.open(errors='replace') as fh:
        for line in fh:
            line=line.rstrip('\n\r')
            if not line or line.startswith('Time,Parameter,Value'): continue
            parts=line.split(',',2)
            if len(parts)!=3: continue
            t,param,val=parts; hh,mm=t.split(':'); minute=int(hh)*60+int(mm); hour=minute//60; max_hour=max(max_hour,hour)
            try: value=float(val)
            except ValueError: value=np.nan
            if param in P12_STATIC: static[param]=value
            else: events[(hour,param)].append((minute,value))
    rows=[]
    for hour in range(max_hour+1):
        row={'patient_id':path.stem,'source_set':path.parent.name,'icu_hour':hour}
        row.update(static)
        for param in p12_dynamic:
            vals=events.get((hour,param),[])
            row[f'event_count__{param}']=len(vals)
            valid=[(m,v) for m,v in vals if np.isfinite(v) and v!=-1]
            row[f'value__{param}']=valid[-1][1] if valid else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
writer=None; p12_rows=0
for start in range(0,len(P12_FILES),500):
    batch=normalize_batch(pd.concat([read_p12(p) for p in P12_FILES[start:start+500]],ignore_index=True))
    table=pa.Table.from_pandas(batch,preserve_index=False)
    if writer is None: writer=pq.ParquetWriter(str(p12_hourly_out/'data.parquet'),table.schema,compression='zstd')
    writer.write_table(table); p12_rows += len(batch)
if writer: writer.close()

# Window indices. Main P19 anchors are before first positive-label proxy; P12 is landmark conditional on observed history through cutoff.
p19_index=[]
for path in P19_FILES:
    df=pd.read_csv(path,sep='|',usecols=['ICULOS','SepsisLabel'])
    first=int(df.loc[df.SepsisLabel.astype(int)==1,'ICULOS'].min()) if (df.SepsisLabel.astype(int)==1).any() else None
    maxh=int(df.ICULOS.max())
    for t in df.ICULOS.astype(int).tolist():
        if first is not None and t>=first: continue
        p19_index.append({'patient_id':path.stem,'anchor_hour':t,'input_start_hour':max(0,t-11),'input_end_hour':t,'intensity_target_start':t+1,'intensity_target_end':t+4,'next_window_available':int(maxh>=t+4),'first_positive_proxy':first if first is not None else -1,'y_h6':int(first is not None and first<=t+6),'y_h12':int(first is not None and first<=t+12)})
pd.DataFrame(p19_index).to_parquet(PROC/'cinc2019_windows.parquet',index=False)
# P12 landmarks from hourly tables and Outcomes A/B.
p12_index=[]
for path in P12_FILES:
    pid=path.stem; death=p12_outcomes[pid]
    # max observed hour from raw event time
    maxh=-1
    with path.open(errors='replace') as fh:
        for line in fh:
            if line.startswith('Time,') or not line.strip(): continue
            try: hh,mm=line.split(',',1)[0].split(':'); maxh=max(maxh,int(hh)*60//60)
            except Exception: pass
    for cutoff in [12,24]:
        if maxh>=cutoff:
            p12_index.append({'patient_id':pid,'cutoff_hour':cutoff,'max_observed_hour':maxh,'mortality_label':death,'landmark_available':1})
pd.DataFrame(p12_index).to_parquet(PROC/'cinc2012_landmarks.parquet',index=False)

summary={'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'p19_files':len(P19_FILES),'p19_hourly_rows':p19_rows,'p19_windows':len(p19_index),'p12_files':len(P12_FILES),'p12_hourly_rows':p12_rows,'p12_landmarks':len(p12_index),'p12_dynamic_channels':len(p12_dynamic),'status':'PASS'}
(STATS/'stage2_construct_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
