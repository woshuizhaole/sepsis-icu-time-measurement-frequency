from pathlib import Path
import json
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015')
R=P/'results'
p=R/'stage6_transfer_qc.json'; q=json.loads(p.read_text()); q['bootstrap_replicates']=1000; q['bootstrap_unit']='patient_cluster'; q['bootstrap_protocol_note']='Exploratory 1000-replicate patient-cluster bootstrap for transfer; not the 2000-replicate primary protocol.'; q['source_pretrain_protocol_note']='Both directions use a label-blind train-only first-20h source pool for this dataset/task-shift transfer, distinct from the primary Stage4 first-24h SSL pool.'; p.write_text(json.dumps(q,indent=2)+'\n')
p=R/'stats/stage7_gap_qc.json'; q=json.loads(p.read_text()); q['stage6_transfer_bootstrap_replicates']=1000; q['stage6_transfer_bootstrap_note']='Exploratory patient-cluster bootstrap; transfer CIs are not 2000-replicate primary-protocol CIs.'; q['stage4_calibration_ci_scope']='Brier/prevalence/mean-pred and operating metrics have patient bootstrap; calibration intercept/slope/ECE/MCE are held-out descriptive point estimates without CI.'; p.write_text(json.dumps(q,indent=2)+'\n')
print('patched')
