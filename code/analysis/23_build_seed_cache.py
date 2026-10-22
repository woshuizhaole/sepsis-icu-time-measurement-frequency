import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
P=Path('/data/workplace/wyx/measurement_intensity_ssl_20261015');D=P/'data/processed'
DYN=['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets'];BASE=['Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS'];INPUT=BASE+[f'v__{c}' for c in DYN]+[f'm0__{c}' for c in DYN]+[f'm4__{c}' for c in DYN]
def main(seed):
 part=json.loads((D/'partitions.json').read_text())['cinc2019'][str(seed)]
 df=pd.read_parquet(D/'stage3_p19_features.parquet').sort_values(['patient_id','ICULOS']).reset_index(drop=True)
 tr={k for k,v in part.items() if v=='train'}
 med=df[df.patient_id.isin(tr)][INPUT].median(numeric_only=True).fillna(0)
 a=df[INPUT].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0).to_numpy('float32')
 path=D/f'p19_sequences_12h_seed{seed}.npy'
 mm=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(df),12,len(INPUT)))
 st=0
 for pid,g in df.groupby('patient_id',sort=False):
  n=len(g);x=a[st:st+n];p=np.concatenate([np.repeat(x[:1],11,0),x],0)
  mm[st:st+n]=np.lib.stride_tricks.sliding_window_view(p,(12,p.shape[1]))[:,0,:,:];st+=n
 mm.flush();print(seed,path,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);main(ap.parse_args().seed)

