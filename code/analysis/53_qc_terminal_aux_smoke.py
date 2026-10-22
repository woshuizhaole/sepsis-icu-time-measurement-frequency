from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score,brier_score_loss
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015')
base=P/'results/improvement_fair_ssl'
modes=['none','all']; methods=['random','terminal_only','nextmask4','nextintensity4','valueforecast4']; fracs=[0.01,0.1,1.0]
rows=[]; logits_rows=[]; errors=[]
for mode in modes:
    d=pd.read_csv(base/f'terminal_controls_{mode}/fair_ssl_results_seed101.csv')
    d['terminal_mode']=mode; rows.append(d)
    for meth in ['terminal_only','nextmask4','nextintensity4','valueforecast4']:
        f=base/f'terminal_controls_{mode}/terminal_logits_{meth}_seed101.parquet'
        if not f.exists(): errors.append(f'missing:{f}'); continue
        q=pd.read_parquet(f); y=q.next_window_available.to_numpy(dtype=int); p=q.terminal_probability.to_numpy(dtype=float)
        logits_rows.append({'terminal_mode':mode,'method':meth,'n_test':len(q),'availability_prevalence':float(y.mean()),'mean_probability':float(p.mean()),'auroc':float(roc_auc_score(y,p)),'brier':float(brier_score_loss(y,p)),'finite':bool(np.isfinite(p).all())})
allm=pd.concat(rows,ignore_index=True)
allm.to_csv(base/'terminal_aux_smoke_results_all.csv',index=False)
logdf=pd.DataFrame(logits_rows); logdf.to_csv(base/'terminal_aux_smoke_logits_summary.csv',index=False)
expected=2*5*3
if len(allm)!=expected: errors.append(f'rows:{len(allm)}')
if set(allm.terminal_mode)!=set(modes) or set(allm.init)!=set(methods): errors.append('key_set')
if allm.duplicated(['terminal_mode','init','fraction']).any(): errors.append('duplicate_keys')
if not np.isfinite(allm.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all(): errors.append('metric_nonfinite')
if len(logdf)!=8 or not logdf.finite.all(): errors.append('logit_summary')
qc={'status':'PASS' if not errors else 'FAILED','rows':int(len(allm)),'expected_rows':expected,'logit_rows':int(len(logdf)),'modes':modes,'methods':methods,'fractions':fracs,'bootstrap':500,'errors':errors,'result_path':str(base/'terminal_aux_smoke_results_all.csv'),'logit_summary_path':str(base/'terminal_aux_smoke_logits_summary.csv')}
(base/'terminal_aux_smoke_qc.json').write_text(json.dumps(qc,indent=2))
print(json.dumps(qc,indent=2)); print(logdf.to_string(index=False)); print(allm.groupby(['terminal_mode','init'],as_index=False).auprc.median().to_string(index=False))
