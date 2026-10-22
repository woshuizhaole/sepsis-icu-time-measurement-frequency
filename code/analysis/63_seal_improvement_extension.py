from pathlib import Path
import json, hashlib
import numpy as np, pandas as pd
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015'); R=P/'results'; C=R/'improvement_fair_ssl/terminal_censoring'; A=R/'improvement_fair_ssl/terminal_controls_all'
# preserve pre-extension files
for src,name in [(R/'metrics.parquet','metrics_pre_improvement_extension.parquet'),(R/'bootstrap_ci.csv','bootstrap_ci_pre_improvement_extension.csv')]:
    dst=R/name
    if not dst.exists(): dst.write_bytes(src.read_bytes())
metrics=pd.read_parquet(R/'metrics.parquet'); boot=pd.read_csv(R/'bootstrap_ci.csv')
new_metrics=[]; new_boot=[]
def add_group(point_path,boot_path,stage,init_col,model_col,metric_prefix,fr=1.0):
    p=pd.read_csv(point_path); b=pd.read_csv(boot_path)
    for _,r in b.iterrows():
        seed=int(r['seed']); init=str(r[init_col]); model=str(r[model_col]); estrow=p[(p.seed==seed)&(p[init_col]==init)&(p[model_col]==model)&(np.isclose(p.fraction,fr))].iloc[0]
        for estimand in ['complete_case','raw_ipw','stabilized_ipw']:
            for met in ['auprc','brier']:
                tag=f'{init}|{model}|{estimand}'
                estcol=f'{met}_{estimand}'; col=f'{estimand}_{met}'
                lo=f'{col}_lo'; hi=f'{col}_hi'
                new_boot.append({'stage':stage,'model':tag,'seed':seed,'fraction':fr,'metric':met,'estimate':float(estrow[estcol]),'lo':float(r[lo]),'hi':float(r[hi])})
                new_metrics.append({'stage':stage,'model':tag,'seed':seed,'fraction':fr,'metric':met,'value':float(estrow[estcol])})
add_group(C/'terminal_censoring_metrics_all_seeds.csv',C/'terminal_censoring_bootstrap_all_seeds.csv','P0_terminal_censoring','method','terminal_model','terminal_censoring')
add_group(P/'results/improvement_fair_ssl/terminal_aux_censoring_metrics_all.csv',A/'terminal_aux_bootstrap_all_seeds.csv','P0_terminal_aux','init','terminal_model','terminal_aux')
nm=pd.DataFrame(new_metrics); nb=pd.DataFrame(new_boot)
metrics2=pd.concat([metrics,nm],ignore_index=True); boot2=pd.concat([boot,nb],ignore_index=True)
# Validate extension keys and all finite/order.
errors=[]
if nm.duplicated(['stage','model','seed','fraction','metric']).any(): errors.append('new_metric_duplicates')
if nb.duplicated(['stage','model','seed','fraction','metric']).any(): errors.append('new_boot_duplicates')
if not np.isfinite(nm.value.to_numpy(float)).all(): errors.append('new_metric_nonfinite')
if not np.isfinite(nb[['estimate','lo','hi']].to_numpy(float)).all(): errors.append('new_boot_nonfinite')
if not (nb.lo<=nb.hi).all(): errors.append('new_boot_ci_order')
if metrics2.duplicated(['stage','model','seed','fraction','metric']).any(): errors.append('sealed_metric_duplicates')
if boot2.duplicated(['stage','model','seed','fraction','metric']).any(): errors.append('sealed_boot_duplicates')
metrics2.to_parquet(R/'metrics.parquet',index=False); boot2.to_csv(R/'bootstrap_ci.csv',index=False)
qc={'status':'PASS' if not errors else 'FAILED','metrics_added':int(len(nm)),'bootstrap_added':int(len(nb)),'metrics_total':int(len(metrics2)),'bootstrap_total':int(len(boot2)),'errors':errors,'fixed_weight_bootstrap':True,'sources':['terminal_censoring_bootstrap_all_seeds.csv','terminal_aux_bootstrap_all_seeds.csv']}
(R/'stats/stage7_improvement_extension_qc.json').write_text(json.dumps(qc,indent=2))
for path in [R/'metrics.parquet',R/'bootstrap_ci.csv',R/'stats/stage7_improvement_extension_qc.json']:
 print(path,hashlib.sha256(path.read_bytes()).hexdigest())
print(json.dumps(qc,indent=2))
