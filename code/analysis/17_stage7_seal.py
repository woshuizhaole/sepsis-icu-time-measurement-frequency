from pathlib import Path
import json,hashlib,itertools
import numpy as np,pandas as pd
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015');R=P/'results';D=P/'data/processed'
def signflip(d):
 d=np.asarray(d,float);obs=abs(d.mean());v=[]
 for sg in itertools.product([-1,1],repeat=len(d)):v.append(abs((d*np.asarray(sg)).mean()))
 return float(np.mean(np.asarray(v)>=obs))
def holm(p):
 p=np.asarray(p,float);o=np.argsort(p);a=np.empty(len(p))
 for i,j in enumerate(o):a[j]=min(1,(len(p)-i)*p[j])
 for i in range(1,len(o)):a[o[i]]=max(a[o[i]],a[o[i-1]])
 return a
rows=[];ci=[]
l=pd.read_csv(R/'stage3_lgb_v2_results.csv')
for _,r in l.iterrows():
 for m in ['auprc','auroc','brier']:rows.append({'stage':'Stage3','model':str(r.condition),'seed':int(r.seed),'fraction':np.nan,'metric':m,'value':float(r[m])})
s4=pd.read_csv(R/'stage4_ssl_full_results.csv')
for _,r in s4.iterrows():
 for m in ['auprc','auroc','brier']:
  rows.append({'stage':'Stage4','model':str(r['init']),'seed':int(r.seed),'fraction':float(r.fraction),'metric':m,'value':float(r[m])})
  ci.append({'stage':'Stage4','model':str(r['init']),'seed':int(r.seed),'fraction':float(r.fraction),'metric':m,'estimate':float(r[m]),'lo':float(r[f'{m}_lo']),'hi':float(r[f'{m}_hi'])})
for path,stage in [(R/'stage5_wbm_results.csv','Stage5_WBM'),(R/'stage5_timecheat_results.csv','Stage5_TimeCHEAT')]:
 d=pd.read_csv(path)
 for _,r in d.iterrows():
  for m in ['auprc','auroc','brier']:
   rows.append({'stage':stage,'model':str(r['model']),'seed':int(r.seed),'fraction':np.nan,'metric':m,'value':float(r[m])})
   ci.append({'stage':stage,'model':str(r['model']),'seed':int(r.seed),'fraction':np.nan,'metric':m,'estimate':float(r[m]),'lo':float(r[f'{m}_lo']),'hi':float(r[f'{m}_hi'])})
for path,stage in [(R/'stage6_p19_results.csv','Stage6_P19'),(R/'stage6_p12_results.csv','Stage6_P12')]:
 d=pd.read_csv(path)
 for _,r in d.iterrows():
  if 'estimable' in r and not bool(r.estimable):continue
  model=str(r.get('mode',r.get('cutoff_hour','target_only')))
  for m in ['auprc','auroc','brier']:
   rows.append({'stage':stage,'model':model,'seed':int(r.seed),'fraction':float(r.get('cutoff_hour',np.nan)),'metric':m,'value':float(r[m])})
   ci.append({'stage':stage,'model':model,'seed':int(r.seed),'fraction':float(r.get('cutoff_hour',np.nan)),'metric':m,'estimate':float(r[m]),'lo':float(r[f'{m}_lo']),'hi':float(r[f'{m}_hi'])})
pd.DataFrame(rows).to_parquet(R/'metrics.parquet',index=False);pd.DataFrame(ci).to_csv(R/'bootstrap_ci.csv',index=False)
comp=[]
for frac in sorted(s4.fraction.unique()):
 a=s4[(s4['init']=='nextintensity')&(s4.fraction==frac)].set_index('seed').auprc
 b=s4[(s4['init']=='random')&(s4.fraction==frac)].set_index('seed').auprc;d=(a-b).dropna()
 comp.append({'comparison':f'NextIntensity-random@{frac:g}','n_seeds':len(d),'mean_delta_auprc':float(d.mean()),'median_delta_auprc':float(d.median()),'p_exact_signflip':signflip(d),'direction':'positive' if d.mean()>0 else 'negative'})
p=[x['p_exact_signflip'] for x in comp]
if p:
 for x,a in zip(comp,holm(p)):x['p_holm']=float(a)
a=l[l.condition=='V+M0+I'].set_index('seed').auprc;b=l[l.condition=='V+M0'].set_index('seed').auprc;d=(a-b).dropna();q=signflip(d)
comp.append({'comparison':'Stage3 V+M0+I-V+M0','n_seeds':len(d),'mean_delta_auprc':float(d.mean()),'median_delta_auprc':float(d.median()),'p_exact_signflip':q,'p_holm':q,'direction':'positive' if d.mean()>0 else 'negative'})
pd.DataFrame(comp).to_csv(R/'interpretation.csv',index=False)
sha={}
for f in [D/'partitions.json',D/'cinc2019_windows.parquet',R/'stage3_lgb_v2_results.csv',R/'stage4_ssl_full_results.csv',R/'stage5_wbm_results.csv',R/'stage5_official_utility_by_seed.csv',R/'stage6_p19_results.csv',R/'stage6_p12_results.csv',R/'metrics.parquet',R/'bootstrap_ci.csv',R/'interpretation.csv']:
 if f.exists():sha[str(f.relative_to(P))]=hashlib.sha256(f.read_bytes()).hexdigest()
audit='# Final Audit (current computational snapshot)\\n\\nGenerated from executed files only. No unexecuted model or metric is promoted.\\n\\nStage 3 v2 LightGBM and Stage 4 corrected SSL have fresh scientific PASS. Stage 5 is PARTIAL: WBM is one-step next-hour auxiliary and not the frozen 4h terminal/IPCW estimand; TimeCHEAT is a 3-seed causal architecture comparator. Stage 6 is conditional sensitivity/target-only task shift; embargo12 is non-estimable under the fixed 6h first-positive proxy.\\n\\nStatistical files: results/metrics.parquet, results/bootstrap_ci.csv, results/interpretation.csv. P19 pre-first-positive AUPRC, P12 landmark mortality, and official full-trajectory Utility are distinct estimands. Observation intensity is a recorded-process representation, not a causal order signal.\\n\\nSHA256\\n'+json.dumps(sha,indent=2)+'\\n'
(P/'FINAL_AUDIT.md').write_text(audit)
print('metrics',len(rows),'ci',len(ci),'comparisons',len(comp),'sha',len(sha))

