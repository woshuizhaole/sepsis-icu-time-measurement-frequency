#!/usr/bin/env python3
"""CinC 2012 fixed-landmark representation decomposition (Fig. 5B-D).

Static (S), last values (V) and charting intensity (I: per-parameter event counts,
breadth, repeat, observed hours) before the 12 h / 24 h cutoff (strict icu_hour < cutoff);
LightGBM with the sealed landmark configuration; 2,000 shared patient bootstrap draws.
The model code below is identical to the run of 2026-10-18 (2,360 s on inspur) whose
outputs are archived in results/matched_time/cinc2012_landmark/.
Usage: 75_matched_time_cinc2012_landmark.py [OUT_DIR]
"""
import json, time, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from lightgbm import LGBMClassifier, early_stopping
P = Path('/data/workplace/wyx/measurement_intensity_ssl_20261015'); D = P/'data/processed'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else P / 'results/matched_time/cinc2012_landmark'
OUT.mkdir(parents=True, exist_ok=True)
B = 2000; SEEDS = [101, 202, 303, 404, 505]
t0 = time.time()
df = pd.read_parquet(D/'cinc2012_hourly/data.parquet')
lm = pd.read_parquet(D/'cinc2012_landmarks.parquet')
parts = json.loads((D/'partitions.json').read_text())['cinc2012']
value = [c for c in df.columns if c.startswith('value__')]
count = [c for c in df.columns if c.startswith('event_count__')]
static = ['Age', 'Gender', 'Height', 'ICUType', 'Weight']
df = df.sort_values(['patient_id', 'icu_hour'], kind='mergesort')
first_static = df.groupby('patient_id', sort=False)[static].head(1).set_index(df.groupby('patient_id', sort=False).head(1).patient_id)
feat = {}
for cutoff in (12, 24):
    h = df[df.icu_hour < float(cutoff)]
    g = h.groupby('patient_id', sort=False)
    last = g[value].last()                       # last non-null value per parameter (sorted by icu_hour)
    cnt = g[count].apply(lambda q: q.fillna(0).sum())
    obs_hours = (h[count].fillna(0).to_numpy().sum(1) > 0)
    oh = pd.Series(obs_hours, index=h.index).groupby(h.patient_id).sum().rename('obs_hours')
    nrow = g.size().rename('n_pre_rows')
    z = lm[(lm.cutoff_hour == cutoff) & (lm.landmark_available == 1)][['patient_id', 'mortality_label']].copy()
    z = z.merge(nrow, left_on='patient_id', right_index=True, how='left')
    z = z.merge(first_static, left_on='patient_id', right_index=True, how='left')
    z = z.merge(last, left_on='patient_id', right_index=True, how='left')
    z = z.merge(cnt, left_on='patient_id', right_index=True, how='left')
    z = z.merge(oh, left_on='patient_id', right_index=True, how='left')
    nohist = z.n_pre_rows.isna() | (z.n_pre_rows == 0)
    z.loc[nohist, static + value + count + ['obs_hours']] = np.nan   # mirror p12(): no history -> all NaN
    c = z[count].fillna(0)
    z['breadth'] = (c > 0).sum(1).astype(float)
    z['repeat'] = c.sum(1) - z['breadth']
    z.loc[nohist, ['breadth', 'repeat']] = np.nan
    z['nohist'] = nohist.astype(int)
    feat[cutoff] = z
print('features built', {k: v.shape for k, v in feat.items()}, f'{time.time()-t0:.0f}s', flush=True)
I_cols = count + ['breadth', 'repeat', 'obs_hours']
COND = {'S': static, 'V': value, 'I': I_cols, 'V+I': value + I_cols, 'S+V': static + value,
        'S+V+I': static + value + I_cols, 'S+I': static + I_cols}
PAIRS = [('V+I', 'V'), ('S+V+I', 'S+V'), ('I', 'S'), ('S+I', 'S')]
ICU = {1: 'CCU', 2: 'CSRU', 3: 'MICU', 4: 'SICU'}
rows, prow, strat = [], [], []
for seed in SEEDS:
    part = parts[str(seed)]
    for cutoff in (12, 24):
        z = feat[cutoff].copy(); z['split'] = z.patient_id.map(part)
        tr, va, te = z[z.split == 'train'], z[z.split == 'validation'], z[z.split == 'test']
        y = te.mortality_label.astype(int).to_numpy(); preds = {}
        for name, cols in COND.items():
            med = tr[cols].median(numeric_only=True)
            clean = lambda q: q[cols].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0)
            m = LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=31, min_child_samples=100, subsample=.8,
                               colsample_bytree=.8, reg_lambda=1, random_state=seed, n_jobs=8, verbosity=-1)
            m.fit(clean(tr), tr.mortality_label.astype(int), eval_set=[(clean(va), va.mortality_label.astype(int))],
                  callbacks=[early_stopping(30, verbose=False)])
            preds[name] = m.predict_proba(clean(te))[:, 1]
        # shared patient bootstrap draws (one row per patient in test)
        n = len(te); rng = np.random.default_rng(seed * 7919 + cutoff)
        W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)])
        bm = {}
        for name, p in preds.items():
            bm[name] = np.array([[average_precision_score(y, p, sample_weight=w), roc_auc_score(y, p, sample_weight=w),
                                  brier_score_loss(y, p, sample_weight=w)] for w in W])
            pt = (average_precision_score(y, p), roc_auc_score(y, p), brier_score_loss(y, p))
            lo, hi = np.percentile(bm[name], 2.5, 0), np.percentile(bm[name], 97.5, 0)
            rows.append(dict(seed=seed, cutoff_hour=cutoff, condition=name, n_features=len(COND[name]), n_train=len(tr),
                             n_val=len(va), n_test=n, positive_test=int(y.sum()), n_test_nohist=int(te.nohist.sum()),
                             auprc=pt[0], auroc=pt[1], brier=pt[2], auprc_lo=lo[0], auprc_hi=hi[0],
                             auroc_lo=lo[1], auroc_hi=hi[1], brier_lo=lo[2], brier_hi=hi[2]))
        for a, b in PAIRS:
            d = bm[a] - bm[b]
            pa = (average_precision_score(y, preds[a]) - average_precision_score(y, preds[b]),
                  roc_auc_score(y, preds[a]) - roc_auc_score(y, preds[b]),
                  brier_score_loss(y, preds[a]) - brier_score_loss(y, preds[b]))
            lo, hi = np.percentile(d, 2.5, 0), np.percentile(d, 97.5, 0)
            prow.append(dict(seed=seed, cutoff_hour=cutoff, comparison=f'{a} minus {b}', bootstrap=B,
                             delta_auprc=pa[0], delta_auprc_lo=lo[0], delta_auprc_hi=hi[0],
                             delta_auroc=pa[1], delta_auroc_lo=lo[1], delta_auroc_hi=hi[1],
                             delta_brier=pa[2], delta_brier_lo=lo[2], delta_brier_hi=hi[2]))
        icu = te.ICUType.to_numpy()
        for k, lab in ICU.items():
            mk = icu == k
            if mk.sum() < 20 or y[mk].sum() < 2 or y[mk].sum() == mk.sum():
                continue
            strat.append(dict(seed=seed, cutoff_hour=cutoff, icu_type=lab, n=int(mk.sum()), deaths=int(y[mk].sum()),
                              auroc_V=roc_auc_score(y[mk], preds['V'][mk]), auroc_VI=roc_auc_score(y[mk], preds['V+I'][mk]),
                              auprc_V=average_precision_score(y[mk], preds['V'][mk]),
                              auprc_VI=average_precision_score(y[mk], preds['V+I'][mk])))
        print(f'seed {seed} cutoff {cutoff} done {time.time()-t0:.0f}s', flush=True)
pd.DataFrame(rows).to_csv(OUT/'p12_decomp_probe_rows.csv', index=False)
pd.DataFrame(prow).to_csv(OUT/'p12_decomp_probe_paired.csv', index=False)
s = pd.DataFrame(strat); s['delta_auroc'] = s.auroc_VI - s.auroc_V; s['delta_auprc'] = s.auprc_VI - s.auprc_V
s.to_csv(OUT/'p12_decomp_probe_icutype.csv', index=False)
print('total', f'{time.time()-t0:.0f}s')
