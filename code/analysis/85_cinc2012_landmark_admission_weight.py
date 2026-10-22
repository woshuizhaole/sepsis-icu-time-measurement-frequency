#!/usr/bin/env python3
"""CinC 2012 landmark decomposition with weight fixed at its admission value.

The hourly table keeps the last Weight row of each record as the static descriptor, and for
about two thirds of patients that row lies after the 12-h or 24-h landmark. This script reads
the 00:00 general-descriptor Weight from each raw record (-1 -> missing) and re-runs the
unchanged model code of script 75 with that value, writing to a separate output directory.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

P = Path('/data/workplace/wyx/measurement_intensity_ssl_20261015')
RAW = P / 'data/raw/cinc2012'
OUT = P / 'results/matched_time/cinc2012_landmark_admission_weight'
OUT.mkdir(parents=True, exist_ok=True)

adm = {}
for s in ('set-a', 'set-b'):
    for f in sorted((RAW / s).glob('*.txt')):
        w = np.nan
        for line in open(f):
            parts = line.strip().split(',')
            if len(parts) == 3 and parts[1] == 'Weight':
                if parts[0] == '00:00':
                    v = float(parts[2])
                    w = np.nan if v == -1 else v
                break
        adm[f.stem] = w
assert len(adm) == 8000
pd.Series(adm, name='admission_weight').rename_axis('patient_id').to_csv(OUT / 'admission_weight.csv')

src = (P / 'code/analysis/75_matched_time_cinc2012_landmark.py').read_text()
anchor = "df = pd.read_parquet(D/'cinc2012_hourly/data.parquet')\n"
assert src.count(anchor) == 1
src = src.replace(anchor, anchor + "df['Weight'] = df.patient_id.map(ADMISSION_WEIGHT).astype(float)\n")
sys.argv = [sys.argv[0], str(OUT)]
exec(compile(src, '75_matched_time_cinc2012_landmark.py (admission weight)', 'exec'), {'__name__': '__main__', 'ADMISSION_WEIGHT': adm})
