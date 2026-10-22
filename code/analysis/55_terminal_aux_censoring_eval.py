from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
P=Path("/data/workplace/wyx/measurement_intensity_ssl_20261015")
D=P/"data/processed"; C=P/"results/improvement_fair_ssl/terminal_censoring"; R=P/"results/improvement_fair_ssl/terminal_controls_all"
seeds=[101,202,303,404,505]; inits=["random","terminal_only","nextmask4","nextintensity4","valueforecast4"]; fracs=[0.01,0.05,0.10,0.25,1.0]; models=["hour_empirical","history_logistic","history_gbm"]
def met(y,p,w=None):
    try: return float(average_precision_score(y,p,sample_weight=w)),float(roc_auc_score(y,p,sample_weight=w)),float(brier_score_loss(y,p,sample_weight=w))
    except ValueError: return float("nan"),float("nan"),float("nan")
def ess(w): return float((w.sum()**2)/max((w*w).sum(),1e-12))
rows=[]; errors=[]
for s in seeds:
    lg=pd.read_parquet(C/f"terminal_logits_seed{s}.parquet")
    avail=lg.next_window_available.to_numpy(dtype=bool); pterm={m:np.clip(lg[f"p_available__{m}"].to_numpy(dtype=float),.05,.999) for m in models}
    pnum=pterm["hour_empirical"]
    for init in inits:
        for frac in fracs:
            f=R/f"fair_pred_{init}_seed{s}_f{frac:.2f}.parquet"
            if not f.exists(): errors.append(f"missing:{f}"); continue
            pred=pd.read_parquet(f)
            if len(pred)!=len(lg) or not np.array_equal(pred.patient_id.to_numpy(),lg.patient_id.to_numpy()): errors.append(f"align:{s}:{init}:{frac}")
            y=pred.y.to_numpy(dtype=int); p=pred.p.to_numpy(dtype=float); apc,aucc,brc=met(y[avail],p[avail])
            for model in models:
                raw=1/pterm[model][avail]; stab=(pnum/pterm[model])[avail]
                apr,aucr,brr=met(y[avail],p[avail],raw); aps,aucs,brs=met(y[avail],p[avail],stab)
                rows.append({"seed":s,"init":init,"fraction":frac,"terminal_model":model,"n_test":len(pred),"n_available":int(avail.sum()),"availability_rate":float(avail.mean()),"positive_all":int(y.sum()),"positive_available":int(y[avail].sum()),"auprc_complete_case":apc,"auroc_complete_case":aucc,"brier_complete_case":brc,"auprc_raw_ipw":apr,"auroc_raw_ipw":aucr,"brier_raw_ipw":brr,"auprc_stabilized_ipw":aps,"auroc_stabilized_ipw":aucs,"brier_stabilized_ipw":brs,"raw_weight_p99":float(np.percentile(raw,99)),"raw_weight_max":float(raw.max()),"raw_ess":ess(raw),"stabilized_weight_p99":float(np.percentile(stab,99)),"stabilized_weight_max":float(stab.max()),"stabilized_ess":ess(stab)})
out=pd.DataFrame(rows); out.to_csv(P/"results/improvement_fair_ssl/terminal_aux_censoring_metrics_all.csv",index=False)
summary=(out.groupby(["init","terminal_model","fraction"],as_index=False).agg(n=("seed","count"),auprc_complete_case_median=("auprc_complete_case","median"),auprc_raw_ipw_median=("auprc_raw_ipw","median"),auprc_stabilized_ipw_median=("auprc_stabilized_ipw","median"),raw_ess_median=("raw_ess","median"),stabilized_ess_median=("stabilized_ess","median")))
summary.to_csv(P/"results/improvement_fair_ssl/terminal_aux_censoring_summary.csv",index=False)
qc={"status":"PASS" if len(out)==375 and not errors and np.isfinite(out.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all() else "FAILED","rows":int(len(out)),"expected_rows":375,"seeds":seeds,"inits":inits,"fractions":fracs,"terminal_models":models,"errors":errors,"point_estimates_only":True,"path":str(P/"results/improvement_fair_ssl/terminal_aux_censoring_metrics_all.csv")}
(P/"results/improvement_fair_ssl/terminal_aux_censoring_qc.json").write_text(json.dumps(qc,indent=2)); print(json.dumps(qc,indent=2)); print(summary.to_string(index=False))
