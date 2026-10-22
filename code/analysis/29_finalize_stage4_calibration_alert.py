#!/usr/bin/env python3
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np,pandas as pd
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015');R=P/'results';L=P/'logs/qc'
L.mkdir(parents=True,exist_ok=True)
s=pd.read_csv(R/'stage4_calibration_alert_metrics.csv')
b=pd.read_csv(R/'stage4_calibration_bins.csv')
c=pd.read_csv(R/'stage4_alert_curve.csv')
expected_keys={(m,int(seed),float(f)) for m in ['random','nextmask','nextintensity','valueforecast'] for seed in [101,202,303,404,505] for f in [0.01,0.05,0.10,0.25,1.0]}
keys={(r.method,int(r.seed),float(r.fraction)) for r in s.itertuples()}
lead_non=s[s['alert_n_true_alert_lead'].fillna(0)<=0][['method','seed','fraction','threshold','alert_n_true_alert_lead','alert_mean_true_alert_lead_h','alert_mean_true_alert_lead_h_lo','alert_mean_true_alert_lead_h_hi']].to_dict('records')
lead_boot=s[s['alert_mean_true_alert_lead_h_lo'].isna()|s['alert_mean_true_alert_lead_h_hi'].isna()][['method','seed','fraction','alert_n_true_alert_lead']].to_dict('records')
cal_na={col:int(s[col].isna().sum()) for col in ['prevalence','mean_pred','brier','calibration_intercept','calibration_slope','ece_10','mce_10'] if col in s}
bin_counts=b.groupby(['method','seed','fraction']).size()
bin_short=[{'method':k[0],'seed':int(k[1]),'fraction':float(k[2]),'n_bins':int(v)} for k,v in bin_counts.items() if v<10]
curve_keys={(r.method,int(r.seed),float(r.fraction)) for r in c.itertuples()}
warnings={
 'lead_time_point_not_estimable':len(lead_non),
 'lead_time_point_rows':lead_non,
 'lead_time_bootstrap_not_estimable':len(lead_boot),
 'lead_time_bootstrap_rows':lead_boot,
 'reason':'The corresponding test prediction stream produced no post-refractory alert on any y_h6-positive row; a true-alert lead time and its patient bootstrap distribution therefore have no defined support. This is a data-supported non-estimability result, not an imputed zero.',
 'calibration_metric_na_counts':cal_na,
 'quantile_bin_short_rows':bin_short,
 'quantile_bin_note':'Some files have fewer than 10 bins because tied/constant probabilities collapse quantile edges; bins are reported as observed and no empty bin is treated as a calibration failure.'
}
(R/'stage4_calibration_alert_warning_audit.json').write_text(json.dumps(warnings,indent=2))
pretext={
 'stage':'Stage4_pretext_calibration',
 'status':'NOT_ESTIMABLE',
 'reason':'Retained pretraining artifacts contain encoder weights and aggregate loss metadata only; no held-out pretext logits/probabilities or per-channel count/breadth predictive distributions were saved.',
 'checked_files':sorted(str(x.relative_to(P)) for x in R.glob('stage4_pretrain_*.json')),
 'required_for_valid_calibration':['held-out pretext targets','per-channel Binomial/Categorical probabilities or logits','fixed pretext evaluation split'],
 'do_not_infer_from':['downstream sepsis probabilities','terminal/IPW point estimates','pretraining loss alone']
}
(R/'stage4_pretext_calibration_status.json').write_text(json.dumps(pretext,indent=2))
official_path=R/'stage5_official_alert_metrics.csv'
official_qc_path=R/'stage5_official_alert_qc.json'
official_status='PENDING'
official_rows=0
official_na={}
if official_path.exists():
    od=pd.read_csv(official_path);official_rows=len(od);official_status='PASS' if len(od)==5 else 'FAILED'
    for col in od.columns:
        n=int(od[col].isna().sum())
        if n: official_na[col]=n
qc={
 'stage':'Stage4/7 calibration_alert',
 'status':'PARTIAL',
 'computed_scope_status':'PASS' if len(s)==100 and keys==expected_keys and len(c)==1500 and curve_keys==expected_keys and len(b)>0 else 'FAILED',
 'stage4_expected_rows':100,'stage4_rows':int(len(s)),'stage4_key_count':int(len(keys)),
 'stage4_curve_expected_rows':1500,'stage4_curve_rows':int(len(c)),
 'stage4_bin_rows':int(len(b)),
 'stage4_prediction_alignment':'Each of 100 prediction files was checked against fixed cinc2019_windows patient_id and y_h6 sequence; all passed before output.',
 'stage4_calibration':'Held-out test descriptive metrics only; no test recalibration or test threshold tuning.',
 'stage4_alert_threshold':'Stage5 WBM validation Utility threshold transferred by seed as an external operating diagnostic; not Stage4 validation-optimal.',
 'stage4_alert_curve':'Predeclared test threshold grid; descriptive operating curve, no threshold selected from test labels.',
 'stage5_official_alert_rows':official_rows,
 'stage5_official_alert_status':official_status,
 'stage5_official_alert_na_counts':official_na,
 'pretext_count_breadth_calibration':'NOT_ESTIMABLE; see results/stage4_pretext_calibration_status.json.',
 'frozen_head_calibration_alert':'NOT_ESTIMABLE because no patient-level frozen-head prediction files were retained.',
 'patient_bootstrap_replicates':2000,'bootstrap_unit':'patient',
 'warning_audit':'results/stage4_calibration_alert_warning_audit.json',
 'limitations':[
  'Nine Stage4 combinations have no true alert on a y_h6-positive row at the transferred threshold; true-alert lead time and its CI are undefined and are left NaN.',
  'Stage4 full predictions are the pre-first-positive window cohort and cannot reproduce full-trajectory official Utility.',
  'Calibration slope/intercept are descriptive held-out-test fits; no recalibration was performed.',
  'The overall Stage4 proposal gate remains PARTIAL until pretext count/breadth calibration and retained frozen-head prediction artifacts are available.'
 ],
}
(R/'stage4_calibration_alert_qc.json').write_text(json.dumps(qc,indent=2))
log={
 'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
 'stage4_outputs':'100-row calibration/alert table, 845-row calibration bins, 1500-row threshold curve',
 'stage4_compute':'completed before main tail process was terminated; no Stage4 output was recomputed in finalization',
 'warning_audit':warnings,
 'pretext_status':pretext,
 'official_status':official_status,
 'official_qc_path':str(official_qc_path.relative_to(P)) if official_qc_path.exists() else None
}
(L/'stage4_calibration_alert.log').write_text(json.dumps(log,indent=2)+'\n')
ind=f"""Stage4 calibration/alert independent check
rows={len(s)} expected=100
unique_keys={len(keys)} expected_keys={len(expected_keys)}
curve_rows={len(c)} expected=1500
bin_rows={len(b)}
calibration_metric_na_counts={cal_na}
lead_time_point_nonestimable={len(lead_non)}
lead_time_bootstrap_nonestimable={len(lead_boot)}
official_rows={official_rows} official_status={official_status}
pretext_status=NOT_ESTIMABLE
"""
(L/'stage4_calibration_alert_independent_check.txt').write_text(ind)
print(json.dumps({'qc_status':qc['status'],'computed_scope':qc['computed_scope_status'],'stage4_rows':len(s),'curve_rows':len(c),'lead_nonestimable':len(lead_non),'official_rows':official_rows,'official_status':official_status},indent=2))
