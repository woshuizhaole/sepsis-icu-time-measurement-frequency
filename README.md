# Sepsis prediction, ICU time and measurement frequency

Analysis code for the study *Adjusting for ICU time reveals shortcut learning in sepsis prediction and identifies measurement frequency as a risk marker*.

The study asks whether the record structure of public ICU cohorts favours models that read ICU time, and whether how often each variable is measured carries information about sepsis risk beyond physiological values. It uses the PhysioNet/Computing in Cardiology Challenge 2019 sepsis cohort and the Challenge 2012 mortality cohort.

## Data

Both datasets are public and downloaded directly from PhysioNet:

- PhysioNet/CinC Challenge 2019, version 1.0.0: https://doi.org/10.13026/v64v-d857
- PhysioNet/CinC Challenge 2012, version 1.0.0: https://doi.org/10.13026/C2V88N

Raw data are not included in this repository. Place them as expected by the scripts:

```
data/raw/cinc2019/training_setA/*.psv
data/raw/cinc2019/training_setB/*.psv
data/raw/cinc2012/set-a/*.txt
data/raw/cinc2012/set-b/*.txt
data/raw/cinc2012/Outcomes-a.txt
data/raw/cinc2012/Outcomes-b.txt
```

The 2019 cohort contains 40,336 patients (sets A and B). The analysis of the 2012 cohort uses the 8,000 patients in sets A and B; set C was never used for model selection, training or evaluation.

## Environment

Python 3.11 with the packages in `environment.yml` (or `pip_freeze.txt`). The main dependencies are numpy, pandas, pyarrow, scikit-learn, LightGBM, torch and scipy.

```
conda env create -f environment.yml
conda activate measurement-ssl
```

## Code

`code/pipeline/` builds the analysis tables from the raw files:

| Script | Purpose |
|---|---|
| `01_download_data.sh` | Download both PhysioNet trees |
| `02_schema_audit.py` | Parse and audit schemas, labels and record structure |
| `03_construct_stage2.py` | Hourly tables, causal windows, patient-level splits, landmark cohorts |
| `04_qc_stage2.py` | Checks for split overlap, anchor and target timing, channel whitelist |

`code/analysis/` contains the analysis scripts in the order used in the paper. Each script writes CSV/JSON outputs and is restartable from the processed tables.

| Scripts | Analysis |
|---|---|
| `10_stage3_lgb_v2.py` | Gradient-boosted decomposition of values, missingness, history and measurement frequency |
| `27_stage3_grud_cde.py`, `28_qc_stage3_grud_cde.py` | GRU-D and neural controlled differential equation comparators |
| `12_stage4_ssl_full.py`, `21_stage4_frozen_head*.py`, `40_improvement_fair_ssl.py` | Pretraining objectives and label-fraction experiments |
| `13_stage5_wbm.py`, `14_stage5_utility.py`, `15_stage5_timecheat.py` | Strong baselines and the official challenge utility |
| `16_stage6_sensitivity.py`, `27_stage6_transfer.py`, `58_stage6_transfer_aligned24.py` | Window, cohort and transfer sensitivity analyses |
| `34_stage0_2_rootcause.py`, `45_clinical_decomposition_5seed.py` | Demographic, ICU-time and clinical decomposition |
| `70_matched_time_record_structure.py` | Record structure and cohort table |
| `71_matched_time_discrimination.py` | Pooled, adjusted and fully adjusted evaluation |
| `72_matched_time_charting_phenotype.py` | Measurement frequency by variable group and patient |
| `73_matched_time_paired_bootstrap.py` | Patient-cluster paired bootstrap intervals |
| `74_matched_time_family_ablation.py` | Variable-group ablation |
| `75_matched_time_cinc2012_landmark.py`, `76_matched_time_cinc2012_descriptive.py` | Mortality-cohort landmark analysis |
| `77_qc_matched_time.py` | Quality control for the matched-time results |
| `83_sensitivity_reference.py`, `85_cinc2012_landmark_admission_weight.py` | Sensitivity reference model and admission-weight landmark analysis |

`code/utils/matched_time.py` implements the covariate-adjusted AUROC, η², within-stratum AUPRC and patient-cluster bootstrap.

## Reproducing the analysis

```
bash code/pipeline/01_download_data.sh
python code/pipeline/02_schema_audit.py
python code/pipeline/03_construct_stage2.py
python code/pipeline/04_qc_stage2.py

python code/analysis/70_matched_time_record_structure.py
python code/analysis/71_matched_time_discrimination.py
python code/analysis/72_matched_time_charting_phenotype.py
python code/analysis/73_matched_time_paired_bootstrap.py
python code/analysis/74_matched_time_family_ablation.py
python code/analysis/75_matched_time_cinc2012_landmark.py
python code/analysis/76_matched_time_cinc2012_descriptive.py
python code/analysis/77_qc_matched_time.py
python code/analysis/83_sensitivity_reference.py
python code/analysis/85_cinc2012_landmark_admission_weight.py
```

All patient-level splits use seeds 101, 202, 303, 404 and 505. Preprocessing statistics are fitted on the training split only. The scripts write to `results/` and `logs/`; paths are defined at the top of each file.

## Citation

If you use this code, please cite the paper. Contact: Yao Wang, w.y0629@163.com.
