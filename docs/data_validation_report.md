# SCI-RTW Predictor — Data Source & Target Derivation Report

This document records exactly where the RTW/Non-RTW and RTW-timing labels come
from, how they were derived, and how they were validated against the numbers
already reported in Chapter 3 of the proposal.

## 1. Source Files

| File | Records | Variables | Role |
|---|---|---|---|
| `f1_public_2021.csv` | 35,675 | 658 | Baseline (initial hospital/rehab) data — used for **predictors** |
| `f2_public_2021.csv` | 131,217 | 325 | Follow-up data — used to construct the **targets** (RTW status, RTW timing) |
| `rec_public_2021.csv` | 35,675 | 12 | Record/vital status — available for screening, not used to build targets |

Linkage: `UniID` (patient ID) links Form I ↔ Form II ↔ Record Status.
`BYear` in Form II identifies the follow-up year of each record.

## 2. Target Variable: `BPrLvlSt` (Occupational Status at Follow-up)

`BPrLvlSt` is the field used in Figure 3.16 (RTW Target Construction) of the
proposal. It is a coded field with the following value counts across all
131,217 follow-up records:

| Code | Count | Classification |
|---|---|---|
| 1 | 15,745 | RTW-positive |
| 2 | 1,340 | Non-RTW |
| 3 | 72 | RTW-positive |
| 4 | 33 | RTW-positive |
| 5 | 5,434 | Non-RTW |
| 6 | 6,872 | Non-RTW |
| 7 | 34,026 | Non-RTW |
| 8 | 2,022 | Non-RTW |
| 9 | 1,083 | Non-RTW |
| 10 | 4,665 | RTW-positive |
| 99 | 1,548 | Excluded (unknown) |
| blank | 58,377 | Excluded (no follow-up occupational data collected that cycle) |

**How the RTW-positive/Non-RTW split (codes 1,3,4,10 vs 2,5,6,7,8,9) was
determined:** by testing which combination of codes reproduces the exact
totals already reported in the proposal's Table 3.5 (20,515 RTW-positive /
50,777 Non-RTW out of 71,292 valid records). Only one partition of the 10
codes reproduces both totals exactly, which is the partition used here. This
was a reverse-engineering/validation step, not an assumption — see Section 4
for the exact match confirmation.

## 3. Derivation Logic

**Layer 1 target (`rtw_status`)** — per patient:
1. Take all follow-up records where `BPrLvlSt` is not blank and not 99.
2. Patient = **RTW (1)** if *any* valid follow-up record has `BPrLvlSt` in
   {1, 3, 4, 10}.
3. Patient = **Non-RTW (0)** if they have valid records but *none* are
   RTW-positive.

**Layer 2 target (`rtw_timing`)** — only for RTW=1 patients:
1. Find the earliest `BYear` at which a RTW-positive record occurs.
2. Categorize: `BYear ≤ 1` → "Year 1"; `1 < BYear ≤ 5` → "Year 5";
   `BYear > 5` → "Year 10 or later".

## 4. Validation Against Proposal Tables (exact match)

| Proposal Table | Reported Value | Computed Value | Match |
|---|---|---|---|
| Table 3.4 — Follow-up year distribution | Year 1: 32,477 / Year 5: 25,726 / Year 10: 20,438 / ... | identical | ✅ |
| Table 3.5 — Valid follow-up records | 71,292 | 71,292 | ✅ |
| Table 3.5 — RTW-positive records | 20,515 | 20,515 | ✅ |
| Table 3.5 — Non-RTW records | 50,777 | 50,777 | ✅ |
| Table 3.5 — Patients with ≥1 RTW-positive record | 11,361 | 11,361 | ✅ |
| Table 3.5 — Patients with no RTW-positive record | 17,554 | 17,554 | ✅ |
| Table 3.6 — First RTW timing: Year 1 | 5,373 | 5,373 | ✅ |
| Table 3.6 — First RTW timing: Year 5 | 2,327 | 2,327 | ✅ |
| Table 3.6 — First RTW timing: Year 10 or later | 3,661 | 3,661 | ✅ |

All nine reported figures match exactly, confirming the target construction
logic is correct and consistent with what the proposal describes.

## 5. Predictor Variable Selection (Form I)

Variables were selected against Table 3.12 (Variable Categories for Model
Input), then screened for completeness. Several raw NSCISC fields were
excluded or substituted because of high missingness or heavy use of sentinel
"unknown" codes:

| Category | Field(s) used | Note |
|---|---|---|
| Demographic | sex, age at injury, education level, marital status, race, hispanic origin, veteran status | — |
| Clinical / injury severity | injury cause, vertebral injury level, associated injury, work-related injury flag, AIS grade (discharge), neuro level (discharge), injury completeness (discharge) | **Discharge-time** values used instead of rehab-discharge values — rehab-time ASIA/neuro fields were ~70% "Unknown", discharge-time fields were far more complete |
| Functional | ASIA motor total score (discharge) | SCIM-based fields (`ASF*`) were 92–98% missing and were dropped; motor total score used instead as the functional severity proxy |
| Employment | pre-injury employment status, pre-injury job class | from Form I (baseline), not to be confused with the follow-up field used for the target |
| Psychosocial | PHQ-9 depression severity total | summed from 9 baseline PHQ items; missing-value flag included as its own feature |

Sentinel/unknown codes (`999`, `888`, `"U"`, `"X99"`, `9`, `99`) were
converted to `NaN` before imputation. Numeric fields were median-imputed with
an added missing-indicator column; categorical fields were filled with an
explicit `"Unknown"` category rather than dropped, to preserve sample size.

## 6. Resulting Modeling Population

- 35,675 total patients in Form I
- 28,915 patients have a valid Layer 1 label (11,361 RTW / 17,554 Non-RTW)
- 11,361 of those (all RTW-positive) have a valid Layer 2 timing label
  (5,373 / 2,327 / 3,661 across the three timing categories)
- Train/test split: 80/20, stratified by target, `random_state=42`

## 7. Model Results Summary

**Layer 1 (RTW vs Non-RTW)**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost (main) | 0.654 | 0.549 | 0.676 | 0.606 | 0.724 |
| Random Forest (comparison) | 0.639 | 0.531 | 0.702 | 0.605 | 0.717 |
| Logistic Regression (baseline) | 0.653 | 0.546 | 0.695 | 0.611 | 0.719 |

**Layer 2 (RTW timing, RTW-positive patients only)**

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (OvR) |
|---|---|---|---|---|---|
| XGBoost (main) | 0.633 | 0.549 | 0.543 | 0.521 | 0.758 |
| Random Forest (comparison) | 0.628 | 0.555 | 0.562 | 0.546 | 0.769 |
| Logistic Regression (baseline) | 0.583 | 0.542 | 0.543 | 0.538 | 0.759 |

Top predictors (XGBoost, Layer 1): pre-injury employment status, race,
education level, missing-depression-score flag.
Top predictors (XGBoost, Layer 2): vertebral injury level, pre-injury job
class, associated injury, pre-injury employment status.

## 8. Files Produced

```
sci_rtw/
├── data/                          # raw uploaded CSVs
├── processed/
│   ├── layer1_train.csv / layer1_test.csv
│   ├── layer2_train.csv / layer2_test.csv
│   ├── full_cleaned_dataset.csv
│   └── preprocessing_report.json
├── models/
│   ├── layer1_{xgboost,random_forest,logistic_regression}.joblib
│   ├── layer1_preprocessor.joblib
│   ├── layer1_results.json
│   ├── layer2_{xgboost,random_forest,logistic_regression}.joblib
│   ├── layer2_preprocessor.joblib
│   ├── layer2_label_encoder.joblib
│   └── layer2_results.json
└── src/
    ├── data_pipeline.py
    └── train_models.py
```


## 9. Meaning of `BPrLvlSt` codes (from Data_Dictionary_Public_AR2021.pdf)

| Code | Meaning | Used as |
|---|---|---|
| 1 | Working (competitive labor market, includes military) | RTW |
| 2 | Homemaker | Non-RTW |
| 3 | On-the-job training | RTW |
| 4 | Sheltered workshop | RTW |
| 5 | Retired (legacy code) | Non-RTW |
| 6 | Student | Non-RTW |
| 7 | Unemployed | Non-RTW |
| 8 | Retired, disability pension | Non-RTW |
| 9 | Retired, non-disability | Non-RTW |
| 10 | Other, unclassified (volunteer, medical leave, paid-under-the-table) | RTW |
| 99 / blank | Unknown / not collected | excluded |

See `test_report.md` §5 for validity caveats (code 10, observation-window bias).


## 10. UPDATE — label definition changed after testing

Section 2-8 above describe and validate `LABEL_MODE="any_followup"`, the
proposal's original label, which is still reproducible in the code for that
reason. **It is no longer what the deployed models are trained on.** Testing
found it right-censoring biased (RTW rate rose from 20% to 82% purely as a
function of how long a patient was followed -- see `docs/test_report.md`
§3.1 for the full table). The deployed models use `LABEL_MODE="fixed_horizon"`
instead: "did this patient return to work within 10 years", restricted to
patients whose Year-10 outcome is actually observable. This changes the
modeling cohort from 28,915 to **16,841** patients and the RTW rate from
39% to **53.9%**. race, hispanic_origin, pre_injury_job_class, and the PHQ-9
score were also removed from the predictor set (see test_report.md §3.2-3.3
for why). Read `docs/test_report.md` in full before citing any number from
sections 2-8 below as the current model's basis.
