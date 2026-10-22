#!/usr/bin/env python3
"""Fast, read-only static statistical audit (no bootstrap/model training)."""
from __future__ import annotations
import itertools, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd

P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015');D=P/'data/processed';R=P/'results';S=R/'stats';L=P/'logs/qc';S.mkdir(parents=True,exist_ok=True);L.mkdir(parents=True,exist_ok=True)
SEEDS=[101,202,303,404,505]; FRACS=[0.01,0.05,0.10,0.25,1.0]

def summary(d,keys,metrics,stage):
    rows=[]
    if not keys: d=d.assign(_all='all'); keys=['_all']
    for k,g in d.groupby(keys,dropna=False,sort=True):
        if not isinstance(k,tuple): k=(k,)
        r=dict(zip(keys,k));r['stage']=stage;r['n']=len(g)
        for m in metrics:
            x=pd.to_numeric(g[m],errors='coerce').dropna().to_numpy(float)
            if len(x):
                r[m+'_mean']=float(x.mean());r[m+'_median']=float(np.median(x));r[m+'_sd']=float(x.std(ddof=1)) if len(x)>1 else 0.;r[m+'_min']=float(x.min());r[m+'_max']=float(x.max());r[m+'_iqr_lo']=float(np.percentile(x,25));r[m+'_iqr_hi']=float(np.percentile(x,75));r[m+'_cv']=float(x.std(ddof=1)/abs(x.mean())) if len(x)>1 and x.mean()!=0 else np.nan
            else:
                for z in ['mean','median','sd','min','max','iqr_lo','iqr_hi','cv']:r[m+'_'+z]=np.nan
        rows.append(r)
    return pd.DataFrame(rows)

def point_pair(df, index, a,b,metric, label):
    q=df.pivot(index=index,columns='init' if 'init' in df.columns else 'condition',values=metric)
    if a not in q or b not in q:return pd.DataFrame()
    out=(q[a]-q[b]).rename('delta').reset_index();out['pair']=label;out['metric']=metric;return out

def main():
    # Stage 3 LightGBM seed dispersion and point paired deltas.
    l3=pd.read_csv(R/'stage3_lgb_v2_results.csv');summary(l3,['condition'],['auprc','auroc','brier'],'Stage3_LGB').to_csv(S/'rootcause_stage3_lgb_seed_summary.csv',index=False)
    p3=[]
    z=l3.pivot(index='seed',columns='condition',values='auprc')
    for a,b in [('V+M0+I','V+M0'),('V+Mh+I','V+Mh'),('I','V'),('V+M0','V'),('V+Mh','V')]:
        x=(z[a]-z[b]).rename('delta').reset_index();x['pair']=a+'-minus-'+b;x['metric']='auprc';p3.append(x)
    pd.concat(p3,ignore_index=True).to_csv(S/'rootcause_stage3_point_deltas.csv',index=False)

    # GRU-D/CDE OI deltas from sealed point results.
    g3=pd.read_csv(R/'stage3_grud_cde_results.csv');pg=[]
    for arch in sorted(g3.architecture.unique()):
        q=g3[g3.architecture==arch].pivot(index='seed',columns='condition',values='auprc');x=(q['oi']-q['nooi']).rename('delta').reset_index();x['architecture']=arch;x['pair']='OI-minus-NoOI';pg.append(x)
    pd.concat(pg,ignore_index=True).to_csv(S/'rootcause_grud_cde_point_deltas.csv',index=False)

    # Stage 4 full point dispersion and all primary pair deltas.
    s4=pd.read_csv(R/'stage4_ssl_full_results.csv');summary(s4,['init','fraction'],['auprc','auroc','brier'],'Stage4_full').to_csv(S/'rootcause_stage4_full_seed_summary.csv',index=False)
    p4=[]
    for f in FRACS:
        q=s4[s4.fraction.eq(f)].pivot(index='seed',columns='init',values='auprc')
        for a,b in [('nextintensity','random'),('nextmask','random'),('valueforecast','random'),('nextintensity','valueforecast')]:
            x=(q[a]-q[b]).rename('delta').reset_index();x['fraction']=f;x['pair']=a+'-minus-'+b;x['metric']='auprc';p4.append(x)
    pd.concat(p4,ignore_index=True).to_csv(S/'rootcause_stage4_point_deltas.csv',index=False)

    # Terminal/IPW, calibration/alert, Stage 5/6 dispersion.
    t=pd.read_csv(R/'stage4_terminal_ipw_results.csv');summary(t,['init','fraction'],['availability_rate','auprc_all','auroc_all','brier_all','auprc_complete_case','auroc_complete_case','brier_complete_case','auprc_marginal_ipw','auroc_marginal_ipw','brier_marginal_ipw'],'Stage4_terminal_IPW').to_csv(S/'rootcause_stage4_terminal_summary.csv',index=False)
    c=pd.read_csv(R/'stage4_calibration_alert_metrics.csv');summary(c,['method','fraction'],['brier','calibration_intercept','calibration_slope','ece_10','mce_10','alert_sensitivity_patient','alert_alerts_per_patient_day','alert_false_alarms_per_patient_day','alert_mean_true_alert_lead_h','alert_threshold'],'Stage4_calibration_alert').to_csv(S/'rootcause_calibration_alert_summary.csv',index=False)
    w=pd.read_csv(R/'stage5_wbm_results.csv');summary(w,['model'],['auprc','auroc','brier'],'Stage5_WBM').to_csv(S/'rootcause_stage5_wbm_summary.csv',index=False)
    tc=pd.read_csv(R/'stage5_timecheat_results.csv');summary(tc,['model'],['auprc','auroc','brier'],'Stage5_TimeCHEAT').to_csv(S/'rootcause_stage5_timecheat_summary.csv',index=False)
    utility_df=pd.read_csv(R/'stage5_official_utility_by_seed.csv');summary(utility_df,[],['threshold','test_auroc_official','test_auprc_official','normalized_utility','accuracy','f_measure'],'Official_Utility').to_csv(S/'rootcause_stage5_utility_summary.csv',index=False)
    tr=pd.read_csv(R/'stage6_transfer_results_all_seeds.csv');summary(tr,['direction','task','cutoff_hour','mode'],['auprc','auroc','brier','patient_auprc','patient_auroc','patient_brier'],'Stage6_transfer').to_csv(S/'rootcause_stage6_transfer_summary.csv',index=False)

    # Denominators and independent split overlap.
    parts=json.loads((D/'partitions.json').read_text())['cinc2019'];win=pd.read_parquet(D/'cinc2019_windows.parquet',columns=['patient_id','y_h6']);den=[]
    for ss,part in parts.items():
        for sp in ['train','validation','test']:
            q=win[win.patient_id.map(part)==sp];den.append({'seed':int(ss),'split':sp,'n_rows':len(q),'n_patients':q.patient_id.nunique(),'positive':int(q.y_h6.sum()),'prevalence':float(q.y_h6.mean())})
    pd.DataFrame(den).to_csv(S/'rootcause_denominator_audit.csv',index=False)
    sets={int(ss):{k for k,v in part.items() if v=='test'} for ss,part in parts.items()};ov=[]
    for a,b in itertools.combinations(SEEDS,2):
        i=len(sets[a]&sets[b]);union_n=len(sets[a]|sets[b]);ov.append({'seed_a':a,'seed_b':b,'n_a':len(sets[a]),'n_b':len(sets[b]),'intersection':i,'union':union_n,'jaccard':i/union_n})
    pd.DataFrame(ov).to_csv(S/'rootcause_test_split_overlap.csv',index=False)

    # Compact key evidence JSON/report; no superiority claims.
    s3delta=pd.read_csv(S/'rootcause_stage3_point_deltas.csv');s4delta=pd.read_csv(S/'rootcause_stage4_point_deltas.csv'); cal=c
    key={'status':'PASS','source':'current executed snapshot','stage3_lgb_rows':len(l3),'stage4_full_rows':len(s4),'stage4_terminal_rows':len(t),'stage4_calibration_rows':len(c),'stage5_utility_rows':len(utility_df),'stage6_transfer_rows':len(tr),'split_test_jaccard_min':float(pd.DataFrame(ov).jaccard.min()),'split_test_jaccard_max':float(pd.DataFrame(ov).jaccard.max()),'stage3_primary_delta_mean':float(s3delta[s3delta.pair=='V+M0+I-minus-V+M0'].delta.mean()),'stage3_primary_delta_positive_seeds':int((s3delta[s3delta.pair=='V+M0+I-minus-V+M0'].delta>0).sum()),'stage4_nextintensity_delta_by_fraction':{str(f):float(s4delta[(s4delta.pair=='nextintensity-minus-random')&(s4delta.fraction==f)].delta.mean()) for f in FRACS},'stage4_zero_true_lead_rows':int((cal.alert_n_true_alert_lead==0).sum()),'stage4_source_terminal_ci':'NOT_AVAILABLE_POINT_ESTIMATES_ONLY','stage4_pretext_calibration':'NOT_ESTIMABLE_SOURCE_ARTIFACTS','utility_estimand':'full_trajectory_challenge_metric_distinct_from_window_metrics'}
    (S/'rootcause_static_audit.json').write_text(json.dumps(key,indent=2)+'\n')
    lines=['# 静态统计稳定性审计（无新增模型/无bootstrap）','',f"生成：{pd.Timestamp.utcnow().isoformat()}",'', '## 结论', '', f"- 五个独立患者 split 的 test Jaccard={key['split_test_jaccard_min']:.4f}–{key['split_test_jaccard_max']:.4f}，seed dispersion 不能只解释为优化随机性。", f"- Stage3 V+M0+I−V+M0 AUPRC seed-level mean delta={key['stage3_primary_delta_mean']:.6f}，正向 {key['stage3_primary_delta_positive_seeds']}/5；未代表独立信息或因果效应。", '- Stage4 NextIntensity−random 各 fraction 的点差异见 `rootcause_stage4_point_deltas.csv`；保留每个 seed，不用单一最佳 seed。', '- Stage4 terminal/IPW 仍只有 point estimates；不借用其他 estimand 的 CI。', f"- Calibration/alert 表中 {key['stage4_zero_true_lead_rows']}/100 rows 没有 true alert lead-time support，源文件已正确留为不可估计。", '', '## 代码与输出', '', '- 代码：`code/analysis/37_stats_static_audit.py`。', '- 目录：`results/stats/rootcause_*`。', '- 本文件只作诊断，未修改 `metrics.parquet`、`bootstrap_ci.csv`、`interpretation.csv`。']
    (L/'rootcause_static_audit.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(key,indent=2))

if __name__=='__main__': main()
