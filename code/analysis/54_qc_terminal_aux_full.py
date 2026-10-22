from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score,brier_score_loss
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015')
R=P/'results/improvement_fair_ssl/terminal_controls_all'
seeds=[101,202,303,404,505]; inits=['random','terminal_only','nextmask4','nextintensity4','valueforecast4']; fracs=[0.01,0.05,0.1,0.25,1.0]
files=sorted(R.glob('fair_ssl_results_seed*.csv'))
parts=[pd.read_csv(f) for f in files]
d=pd.concat(parts,ignore_index=True)
errors=[]
expected={(s,i,f) for s in seeds for i in inits for f in fracs}
got={(int(r.seed),str(r['init']),float(r.fraction)) for _,r in d.iterrows()}
if len(d)!=125: errors.append(f'rows:{len(d)}')
if expected-got: errors.append(f'missing:{sorted(expected-got)[:10]}')
if got-expected: errors.append(f'extra:{sorted(got-expected)[:10]}')
if d.duplicated(['seed','init','fraction']).any(): errors.append('duplicate_keys')
num=d.select_dtypes(include=[np.number]).columns
finite=bool(np.isfinite(d[num].to_numpy(dtype=float)).all())
if not finite: errors.append('nonfinite_results')
if d.input_dim.nunique()!=1 or int(d.input_dim.iloc[0])!=108: errors.append('input_dim')
d.to_csv(P/'results/improvement_fair_ssl/terminal_aux_full_results_all.csv',index=False)
summary=(d.groupby(['init','fraction'],as_index=False)
 .agg(n=('seed','count'),auprc_median=('auprc','median'),auprc_q1=('auprc',lambda x:x.quantile(.25)),auprc_q3=('auprc',lambda x:x.quantile(.75)),auroc_median=('auroc','median'),brier_median=('brier','median')))
summary.to_csv(P/'results/improvement_fair_ssl/terminal_aux_full_summary.csv',index=False)
# terminal logits summary
log_rows=[]
for s in seeds:
    for init in ['terminal_only','nextmask4','nextintensity4','valueforecast4']:
        f=R/f'terminal_logits_{init}_seed{s}.parquet'
        if not f.exists(): errors.append(f'missing_logits:{s}:{init}'); continue
        q=pd.read_parquet(f); y=q.next_window_available.to_numpy(dtype=int); p=q.terminal_probability.to_numpy(dtype=float)
        log_rows.append({'seed':s,'init':init,'n_test':len(q),'availability_prevalence':float(y.mean()),'mean_probability':float(p.mean()),'auroc':float(roc_auc_score(y,p)),'brier':float(brier_score_loss(y,p)),'finite':bool(np.isfinite(p).all())})
logdf=pd.DataFrame(log_rows); logdf.to_csv(P/'results/improvement_fair_ssl/terminal_aux_full_logits_summary.csv',index=False)
if len(logdf)!=20 or not logdf.finite.all(): errors.append('logit_summary')
# Random/pretext rows should reproduce fair full matrix under explicit seed reset.
fair=pd.read_csv(P/'results/improvement_fair_ssl/fair_ssl_results_all_seeds.csv')
left=d[d['init'].isin(['random','nextmask4','nextintensity4','valueforecast4'])][['seed','init','fraction','auprc','auroc','brier']].copy()
right=fair[['seed','init','fraction','auprc','auroc','brier']].copy()
m=left.merge(right,on=['seed','init','fraction'],suffixes=('_aux','_fair'))
maxdiff={c:float(np.max(np.abs(m[c+'_aux']-m[c+'_fair']))) for c in ['auprc','auroc','brier']} if len(m) else {'auprc':float('nan'),'auroc':float('nan'),'brier':float('nan')}
if len(m)!=100: errors.append(f'random_pretext_compare_rows:{len(m)}')
qc={'status':'PASS' if not errors and finite else 'FAILED','rows':int(len(d)),'expected_rows':125,'seeds':seeds,'inits':inits,'fractions':fracs,'bootstrap':2000,'terminal_logits_rows':int(len(logdf)),'terminal_logits_required':20,'finite_numeric':finite,'random_pretext_max_abs_difference_vs_fair':maxdiff,'errors':errors,'results_path':str(P/'results/improvement_fair_ssl/terminal_aux_full_results_all.csv'),'summary_path':str(P/'results/improvement_fair_ssl/terminal_aux_full_summary.csv'),'logits_summary_path':str(P/'results/improvement_fair_ssl/terminal_aux_full_logits_summary.csv')}
(P/'results/improvement_fair_ssl/terminal_aux_full_qc.json').write_text(json.dumps(qc,indent=2))
print(json.dumps(qc,indent=2)); print(summary.to_string(index=False)); print(logdf.to_string(index=False))
