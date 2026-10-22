from pathlib import Path
import json, numpy as np, pandas as pd
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015'); R=P/'results'; checks=[]; details={}
def j(rel):
    return json.loads((P/rel).read_text())
# Existing final QC
m=pd.read_parquet(R/'metrics.parquet'); b=pd.read_csv(R/'bootstrap_ci.csv')
checks += [('metrics_rows_3503',len(m)==3503),('bootstrap_rows_2503',len(b)==2503),('metrics_finite',np.isfinite(m.value.to_numpy()).all()),('bootstrap_finite',np.isfinite(b[['estimate','lo','hi']].to_numpy()).all()),('bootstrap_order',(b.lo<=b.hi).all()),('metrics_unique',not m.duplicated(['stage','model','seed','fraction','metric']).any()),('bootstrap_unique',not b.duplicated(['stage','model','seed','fraction','metric']).any())]
# QC JSONs
qc_paths=['results/improvement_fair_ssl/calibration/fair_calibration_qc.json','results/rootcause_stage0_2/p19_clinical_decomposition_qc.json','results/improvement_fair_ssl/terminal_censoring/terminal_censoring_qc.json','results/improvement_fair_ssl/terminal_censoring/terminal_censoring_bootstrap_qc.json','results/improvement_fair_ssl/terminal_aux_full_qc.json','results/improvement_fair_ssl/terminal_controls_all/terminal_aux_bootstrap_qc.json','results/rootcause_stage6/p1_horizon_cohort_qc.json','results/rootcause_stage6/p1_horizon_model_qc.json','results/stage6_transfer_aligned24/stage6_transfer_aligned24_qc.json','results/stats/stage7_improvement_extension_qc.json']
for rel in qc_paths:
    q=j(rel); checks.append((rel,q.get('status')=='PASS')); details[rel]=q.get('status')
# Corrected complete-case CIs have nonzero widths.
for rel in ['results/improvement_fair_ssl/terminal_censoring/terminal_censoring_bootstrap_all_seeds.csv','results/improvement_fair_ssl/terminal_controls_all/terminal_aux_bootstrap_all_seeds.csv']:
    d=pd.read_csv(P/rel)
    for c in ['complete_case_auprc','complete_case_brier','raw_ipw_auprc','stabilized_ipw_auprc']:
        w=d[c+'_hi']-d[c+'_lo']; checks.append((rel+':'+c+'_positive_width',bool((w>0).all())))
# Cohort name boundary and no old unqualified label.
co=pd.read_csv(R/'rootcause_stage6/p1_horizon_cohort_summary.csv'); checks += [('cohort_name_boundary','all_record_first_event' in set(co.cohort) and 'all_record' not in set(co.cohort))]
# Audit and local docs mirrored
checks += [('final_audit_exists',(P/'FINAL_AUDIT.md').exists()),('worklog_exists',(P/'docs/WORKLOG.md').exists()),('improvement_exists',(P/'improvement.md').exists())]
out={'status':'PASS' if all(v for _,v in checks) else 'FAILED','checks':dict(checks),'details':details}
out={'status':'PASS' if all(bool(v) for _,v in checks) else 'FAILED','checks':{k:bool(v) for k,v in checks},'details':details}
(P/'results/stats/final_improvement_qc.json').write_text(json.dumps(out,indent=2))
for k,v in checks: print(k,v)
print('status',out['status'])
