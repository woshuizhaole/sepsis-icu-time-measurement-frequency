#!/usr/bin/env python3
"""Independent read-only audit of Stage 0-2 data/window/label lineage.

Does not import project pipeline code. Writes only its own audit artifacts.
"""
from __future__ import annotations
import csv, json, time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

P = Path("/data/workplace/wyx/measurement_intensity_ssl_20261015")
RAW = P/"data/raw"; PROC = P/"data/processed"
OUT = P/"results/rootcause_stage0_2"; LOG = P/"logs/qc/rootcause_stage0_2_audit.log"
OUT.mkdir(parents=True, exist_ok=True); LOG.parent.mkdir(parents=True, exist_ok=True)
P19_DYN = ["HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2","BaseExcess","HCO3","FiO2","pH","PaCO2","SaO2","AST","BUN","Alkalinephos","Calcium","Chloride","Creatinine","Bilirubin_direct","Glucose","Lactate","Magnesium","Phosphate","Potassium","Bilirubin_total","TroponinI","Hct","Hgb","PTT","WBC","Fibrinogen","Platelets"]
MISSING = {"", "nan", "NaN", "NAN", "null", "NULL"}

def p19_raw(path):
    with path.open("r", newline="") as fh:
        header = fh.readline().rstrip("\r\n").split("|"); ix={x:i for i,x in enumerate(header)}
        ih, il = ix["ICULOS"], ix["SepsisLabel"]; hours=[]; labels=[]; nmiss=Counter(); bad=0
        for line in fh:
            vals=line.rstrip("\r\n").split("|")
            if len(vals)!=len(header): bad += 1; continue
            try: h=int(float(vals[ih])); hours.append(h)
            except Exception: continue
            labels.append(int(float(vals[il])) if vals[il] not in MISSING else 0)
            for c,v in zip(header,vals):
                if v in MISSING: nmiss[c]+=1
        hs=set(hours); first=min((h for h,y in zip(hours,labels) if y==1),default=None); pos=[h for h,y in zip(hours,labels) if y==1]
        return {"patient_id":path.stem,"source_set":path.parent.name,"n":len(hours),"min_hour":min(hours) if hours else None,"max_hour":max(hours) if hours else None,"hours":hours,"hour_set":hs,"labels":labels,"first":first,"n_pos":sum(labels),"last_pos":max(pos) if pos else None,"transitions":sum(int(a!=b) for a,b in zip(labels,labels[1:])),"gaps":sum(1 for a,b in zip(sorted(hs),sorted(hs)[1:]) if b!=a+1),"missing":dict(nmiss),"bad":bad}

def p12_raw(path):
    events=dups=maxh=-1; minh=None; neg1=0; params=Counter(); seen=set(); events=0; dups=0; maxh=-1
    with path.open(errors="replace") as fh:
        for line in fh:
            line=line.rstrip("\r\n")
            if not line or line.startswith("Time,Parameter,Value"): continue
            parts=line.split(",",2)
            if len(parts)!=3: continue
            t,param,val=parts
            try: hh,mm=t.split(":"); minute=int(hh)*60+int(mm); h=int(hh)
            except Exception: continue
            events+=1; maxh=max(maxh,h); minh=minute if minh is None else min(minh,minute); params[param]+=1
            if val.strip()=="-1": neg1+=1
            k=(t,param)
            if k in seen: dups+=1
            seen.add(k)
    return {"patient_id":path.stem,"source_set":path.parent.name,"events":events,"max_hour":maxh,"min_minute":minh,"duplicate_events":dups,"neg1_events":neg1,"params":dict(params)}

def wjson(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str))
def wcsv(name,rows):
    if rows: pd.DataFrame(rows).to_csv(OUT/name,index=False)

t0=time.time()
# P19 raw lineage
p19_files=sorted((RAW/"cinc2019"/"training"/"training_setA").glob("*.psv"))+sorted((RAW/"cinc2019"/"training"/"training_setB").glob("*.psv"))
p19=[p19_raw(x) for x in p19_files]; p19_by={x["patient_id"]:x for x in p19}
p19_summary={"n_files":len(p19),"n_rows":sum(x["n"] for x in p19),"n_positive_rows":sum(x["n_pos"] for x in p19),"n_patients_with_positive":sum(x["first"] is not None for x in p19),"n_contiguous_hour_tracks":sum(x["gaps"]==0 for x in p19),"n_tracks_with_hour_gaps":sum(x["gaps"]>0 for x in p19),"total_hour_gaps":sum(x["gaps"] for x in p19),"min_first_positive":min((x["first"] for x in p19 if x["first"] is not None),default=None),"median_first_positive":float(np.median([x["first"] for x in p19 if x["first"] is not None])),"n_label_nonmonotone":sum(x["transitions"]>1 for x in p19),"n_label_transitions":sum(x["transitions"] for x in p19),"n_positive_tracks_with_noncontiguous_positive":sum(x["first"] is not None and x["last_pos"]-x["first"]+1 != x["n_pos"] for x in p19)}
wjson("raw_p19_summary.json",p19_summary)
wcsv("raw_p19_patient_lineage.csv",[{k:v for k,v in x.items() if k not in ("hours","hour_set","labels","missing")} for x in p19])

# P19 independent windows and availability
w=pd.read_parquet(PROC/"cinc2019_windows.parquet"); w.patient_id=w.patient_id.astype(str)
rows=[]; mismatch=[]
for x in p19:
    hs=x["hour_set"]; first=x["first"]; maxh=x["max_hour"]
    for t in sorted(h for h in hs if first is None or h<first):
        actual=all((t+k) in hs for k in range(1,5)); proxy=bool(maxh is not None and maxh>=t+4)
        if actual!=proxy: mismatch.append((x["patient_id"],t,int(actual),int(proxy),maxh))
        rows.append({"patient_id":x["patient_id"],"anchor_hour":t,"expected_first":first if first is not None else -1,"expected_start":t+1,"expected_end":t+4,"expected_avail":int(actual),"expected_y_h6":int(first is not None and first<=t+6),"expected_y_h12":int(first is not None and first<=t+12)})
ind=pd.DataFrame(rows).sort_values(["patient_id","anchor_hour"]).reset_index(drop=True); ww=w.sort_values(["patient_id","anchor_hour"]).reset_index(drop=True)
ydiff=[]
if len(ind)==len(ww) and np.array_equal(ind[["patient_id","anchor_hour"]].to_numpy(),ww[["patient_id","anchor_hour"]].to_numpy()):
    for a,b in [("expected_first","first_positive_proxy"),("expected_start","intensity_target_start"),("expected_end","intensity_target_end"),("expected_y_h6","y_h6"),("expected_y_h12","y_h12"),("expected_avail","next_window_available")]:
        y=ind[a].astype(int).to_numpy(); z=ww[b].astype(int).to_numpy(); ydiff.append({"field":b,"n_diff":int((y!=z).sum())})
else: ydiff.append({"field":"row_alignment","n_diff":-1,"stored_rows":len(ww),"recomputed_rows":len(ind)})
wjson("p19_window_recompute.json",{"recomputed_rows":len(ind),"stored_rows":len(ww),"field_differences":ydiff,"actual_next_window_rate":float(ind.expected_avail.mean()),"stored_next_window_rate":float(ww.next_window_available.mean()),"proxy_availability_mismatches_n":len(mismatch),"proxy_availability_mismatch_examples":mismatch[:20],"expected_y_h6_rate":float(ind.expected_y_h6.mean()),"stored_y_h6_rate":float(ww.y_h6.mean()),"expected_y_h12_rate":float(ind.expected_y_h12.mean()),"stored_y_h12_rate":float(ww.y_h12.mean())})
wcsv("p19_window_availability_mismatch.csv",[{"patient_id":a,"anchor_hour":b,"actual":c,"proxy":d,"max_hour":e} for a,b,c,d,e in mismatch])

# Splits
parts=json.loads((PROC/"partitions.json").read_text()); split_summary=[]; partrows=[]
for seed,mp in parts["cinc2019"].items():
    for pid,sp in mp.items():
        x=p19_by.get(pid,{}); partrows.append({"seed":int(seed),"patient_id":pid,"split":sp,"source_set":x.get("source_set"),"patient_positive":int(x.get("first") is not None)})
    for sp in ["train","validation","test"]:
        ids=[pid for pid,s in mp.items() if s==sp]; xs=[p19_by[pid] for pid in ids]; z=ind[ind.patient_id.isin(ids)]
        split_summary.append({"seed":int(seed),"split":sp,"patients":len(ids),"positive_patients":sum(x["first"] is not None for x in xs),"patient_pos_rate":float(np.mean([x["first"] is not None for x in xs])),"windows":len(z),"positive_windows":int(z.expected_y_h6.sum()),"window_pos_rate":float(z.expected_y_h6.mean()),"source_A":sum(x["source_set"]=="training_setA" for x in xs),"source_B":sum(x["source_set"]=="training_setB" for x in xs)})
wcsv("p19_split_lineage.csv",partrows); wcsv("p19_split_summary.csv",split_summary)

# Feature exact semantics
feat=pd.read_parquet(PROC/"stage3_p19_features.parquet"); rawh=pd.read_parquet(PROC/"cinc2019_hourly/data.parquet")
feat=feat.sort_values(["patient_id","ICULOS"]).reset_index(drop=True); rawh=rawh.sort_values(["patient_id","ICULOS"]).reset_index(drop=True)
sem={"row_alignment":bool(np.array_equal(feat[["patient_id","ICULOS"]].to_numpy(),rawh[["patient_id","ICULOS"]].to_numpy())),"feature_rows":len(feat),"raw_hourly_rows":len(rawh)}
for c in P19_DYN:
    sem[f"m0_diff_{c}"]=int((feat[f"m0__{c}"].astype(float).to_numpy()!=rawh[c].notna().astype(float).to_numpy()).sum())
    exp=rawh[c].notna().astype("int8").groupby(rawh.patient_id,sort=False).rolling(4,min_periods=1).sum().reset_index(level=0,drop=True).to_numpy()
    sem[f"m4_diff_{c}"]=int((np.abs(exp-feat[f"m4__{c}"].astype(float).to_numpy())>1e-9).sum())
f2=pd.read_parquet(PROC/"stage3_p19_features_v2.parquet",columns=["patient_id","ICULOS"]+[f"i_count4__{c}" for c in P19_DYN]).sort_values(["patient_id","ICULOS"]).reset_index(drop=True)
for c in P19_DYN: sem[f"v2_i_count4_diff_{c}"]=int((f2[f"i_count4__{c}"].astype(float).to_numpy()!=feat[f"m4__{c}"].astype(float).to_numpy()).sum())
sem["feature_has_label_column"]=bool("SepsisLabel" in feat.columns)
wjson("p19_feature_semantics.json",sem)

# P12 raw, outcomes and hourly/landmark
p12_files=sorted((RAW/"cinc2012").glob("set-*/*.txt")); p12=[p12_raw(x) for x in p12_files]
wjson("raw_p12_summary.json",{"n_files":len(p12),"events":sum(x["events"] for x in p12),"duplicate_events":sum(x["duplicate_events"] for x in p12),"neg1_events":sum(x["neg1_events"] for x in p12),"parameter_count":len(set().union(*[set(x["params"]) for x in p12])),"n_with_hour12":sum(x["max_hour"]>=12 for x in p12),"n_with_hour24":sum(x["max_hour"]>=24 for x in p12),"max_hour_median":float(np.median([x["max_hour"] for x in p12]))})
wcsv("raw_p12_patient_lineage.csv",p12)
outcomes={}
for fn in ["Outcomes-a.txt","Outcomes-b.txt","Outcomes-c.txt"]:
    path=RAW/"cinc2012"/fn
    if path.exists():
        outcomes[fn]={r["RecordID"]:int(r["In-hospital_death"]) for r in csv.DictReader(path.open())}
labdiff=[pid for pid in outcomes.get("Outcomes-a.txt",{}) if pid in outcomes.get("Outcomes-b.txt",{}) and outcomes["Outcomes-a.txt"][pid]!=outcomes["Outcomes-b.txt"][pid]]
p12l=pd.read_parquet(PROC/"cinc2012_landmarks.parquet").sort_values(["patient_id","cutoff_hour"]).reset_index(drop=True)
allout={}; [allout.update(outcomes.get(fn,{})) for fn in ["Outcomes-a.txt","Outcomes-b.txt","Outcomes-c.txt"]]
p12_expected=[{"patient_id":x["patient_id"],"cutoff_hour":cutoff,"expected_label":allout.get(x["patient_id"])} for x in p12 for cutoff in [12,24] if x["max_hour"]>=cutoff]
pe=pd.DataFrame(p12_expected).sort_values(["patient_id","cutoff_hour"]).reset_index(drop=True)
p12_cmp={"raw_files":len(p12),"stored_landmarks":len(p12l),"expected_available_rows":len(pe),"outcomes_ab_label_disagreements":len(labdiff),"landmark_key_alignment":bool(len(pe)==len(p12l) and np.array_equal(pe[["patient_id","cutoff_hour"]].to_numpy(),p12l[["patient_id","cutoff_hour"]].to_numpy())),"landmark_availability_diff":int((p12l.landmark_available.astype(int).to_numpy()!=1).sum()) if len(pe)==len(p12l) else None,"landmark_label_diff":int((p12l.mortality_label.astype(int).to_numpy()!=pe.expected_label.fillna(-99).astype(int).to_numpy()).sum()) if len(pe)==len(p12l) else None}
wjson("p12_landmark_recompute.json",p12_cmp)
p12h=pd.read_parquet(PROC/"cinc2012_hourly/data.parquet"); hc=p12h.groupby("patient_id").size().to_dict(); hd=[{"patient_id":x["patient_id"],"expected_rows":x["max_hour"]+1,"stored_rows":int(hc.get(x["patient_id"],-1)),"max_hour":x["max_hour"]} for x in p12 if x["max_hour"]+1!=hc.get(x["patient_id"],-1)]
wjson("p12_hourly_recompute.json",{"stored_rows":len(p12h),"stored_patients":p12h.patient_id.nunique(),"hour_range_differences_n":len(hd),"examples":hd[:20]})

# Label-proximity vs intensity descriptives
m4=[f"m4__{c}" for c in P19_DYN]; joined=w.merge(feat[["patient_id","ICULOS"]+m4].rename(columns={"ICULOS":"anchor_hour"}),on=["patient_id","anchor_hour"],how="left"); joined["intensity_total4"]=joined[m4].sum(axis=1); joined["hours_to_first"]=joined.first_positive_proxy.sub(joined.anchor_hour)
g=joined.groupby(pd.cut(joined.hours_to_first,[-np.inf,0,1,2,3,4,6,12,24,48,np.inf],right=False,include_lowest=True),observed=False).agg(n=("y_h6","size"),positive=("y_h6","sum"),mean_intensity=("intensity_total4","mean"),median_intensity=("intensity_total4","median")).reset_index(); g["rate"]=g["positive"]/g["n"]; g.to_csv(OUT/"p19_intensity_by_proxy_proximity.csv",index=False)

result={"runtime_seconds":time.time()-t0,"p19_summary":p19_summary,"p19_window":json.loads((OUT/"p19_window_recompute.json").read_text()),"p19_feature_semantics_status":"PASS" if sem["row_alignment"] and all(v==0 for k,v in sem.items() if "_diff_" in k) else "FAILED","p12_landmark":p12_cmp,"p12_hourly_diff_n":len(hd),"output_dir":str(OUT)}
wjson("rootcause_stage0_2_summary.json",result); LOG.write_text(json.dumps(result,indent=2,ensure_ascii=False,default=str)+"\n"); print(json.dumps(result,indent=2,ensure_ascii=False,default=str))

