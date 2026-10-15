# SCI-RTW Predictor — Test Report & Fix Log

This documents what was broken, what was changed, and what was verified,
across three rounds: an initial UI/prediction-quality pass, a deeper pass
triggered by label-validity findings that required retraining, and a third
round (§6) that added login, roles, a server-side database and a more
rigorous model evaluation. **§6.8 holds the current model numbers**: the
models were retrained in §6.7 and again in §6.8 after data fixes, so the
figures in §6.2, §6.5 and §6.7 are the ones before those fixes, and those
in §3 are from round two. They are kept as history.

## 1. Summary

| Suite | Scope | Result |
|---|---|---|
| `tests/test_api.py` | 51 automated API tests: prediction, validation, model report, confidence level, suggested considerations (pytest) | **51 / 51 pass** |
| `tests/test_auth_records.py` | 36 automated tests: login, roles, user accounts, patient profiles, saved history and its paging, model version, database upgrade (pytest) | **36 / 36 pass** |
| `tests/test_security.py` | 15 automated tests: sign-in lockout, forced change of the default password, session revocation, audit log, browser-facing headers (pytest) | **15 / 15 pass** |
| `tests/ui_walkthrough.py` | 69 checks driving the real React UI in headless Chromium, as administrator, doctor and patient | **69 / 69 pass** |
| Regression check | 400 held-out patients pushed through the live HTTP path | within the expected range of the accuracy reported at training (see §3.4 for why this check exists) |

Run them yourself:
```bash
pip install pytest httpx
python -m pytest tests -v
pip install playwright && playwright install chromium
# needs API on :8000 and frontend on :5173 running:
python tests/ui_walkthrough.py     # screenshots go to docs/screenshots/
```

## 2. Mapping to proposal Table 3.18

| ID | Function | Status | Evidence / note |
|---|---|---|---|
| TC01 | User login | Pass | `test_tc01_*` (7 tests): login, wrong password, logout, anonymous access refused, and each role limited to its own functions; walkthrough signs in as administrator, doctor and patient (see §6.1) |
| TC02 | Enter patient data | Pass | `test_tc02_*`; UI walkthrough completes the patient step and all 3 data-entry steps |
| TC03 | Missing-data validation | Pass | API rejects each of the 9 required fields when absent (HTTP 422); the 5 optional fields may be omitted or sent as `null`; UI disables **Next** until required fields are filled |
| TC04 | Generate RTW prediction | Pass | Shape, probability range, status/probability consistency, determinism, sensible ranking |
| TC05 | Generate RTW timing | Pass | Timing returned only when RTW is predicted; probabilities sum to 1; new "By Year 10" label (see §3) |
| TC06 | View prediction history | Pass | `test_tc06_*` (8 tests): stored in the server database against a patient profile, shared between clinicians, and a patient account sees only its own linked record (see §6.1) |
| TC07 | View model accuracy report | Pass | `/model-info` + Model Insights screen show all 3 models per layer, with cross-validation, the overfitting check and the majority-class reference point (see §6.2) |

## 3. What changed since the first pass (fixing the problems found)

The first version of this prototype used the proposal's original label
("RTW = any follow-up record shows it, whenever observed") and kept race,
Hispanic origin, and a PHQ-9 depression score as predictors. Testing found
that label right-censoring-biased (RTW rate rose from 20% to 82% purely as
a function of how long a patient had been followed), and that PHQ acted as
a proxy for "recently injured" rather than a clinical signal. The fix,
implemented in full:

### 3.1 Fixed-horizon label (replaces the original)

`data_pipeline.py`, `LABEL_MODE = "fixed_horizon"` (default): a patient is
now labelled **"returned to work within 10 years"**, restricted to a cohort
where that's actually knowable --- included if they either had an
RTW-positive record at Year ≤ 10 (event observed), or have a record at
Year ≥ 10 with no such event (followed long enough to say "no" with
confidence). Patients last seen before Year 10 with no event are excluded
(their Year-10 outcome is genuinely unknown, not "no").

Effect: cohort shrinks from 28,915 to **16,841** patients, but the RTW rate
goes from a censoring-distorted 39% to a clean **53.9%**, and Layer 2's
third timing bucket is renamed **"By Year 10"** (previously "Year 10 or
later", which is no longer accurate under a 10-year horizon). The original
"any follow-up" definition is kept as `LABEL_MODE = "any_followup"` purely
to keep the proposal's Table 3.4-3.6 numbers reproducible; it is not used
for the deployed models. See `data_pipeline.py`'s module docstring for the
full derivation.

### 3.2 Dropped predictors: race, hispanic_origin, pre_injury_job_class, PHQ-9

- **race / hispanic_origin**: removing them changed Layer 1 accuracy/AUC by
  ~0 (0.723 → 0.719 AUC). No benefit, clear fairness risk in a clinical
  tool, and outside the proposal's Table 3.12 variable list anyway.
- **PHQ-9 depression score**: entering any value changed the same patient's
  RTW probability by ~17 points regardless of severity (blank: 0.70,
  score=2: 0.53, score=20: 0.47) -- it was encoding "PHQ was collected",
  which correlates with recent injury, not depression severity.
- **pre_injury_job_class**: after the fix below, this field's "N/A, not
  working" code alone carried 4-10x the importance of the next real
  feature. It turned out to still be era-correlated (patients with this
  code: mean 5.6 years followed vs. 11.5 for others) and near-redundant
  with `pre_injury_employment_status` (its code 88 never co-occurs with
  "Working"). Dropping it cost no accuracy (0.681→0.682).

### 3.3 Missingness is NOT kept as its own category any more (dataset-wide fix)

The job-class finding above turned out to be one instance of a systemic
pattern: checked across all 13 categorical predictors, **every field with a
non-trivial unknown rate showed the same pattern** -- "unknown" patients
were followed for several years longer or shorter, on average, than
"known" patients (e.g. `vertebral_injury_level`/`associated_injury`:
78% of the cohort coded "Unknown", with a 9.5-year mean follow-up gap
between the known and unknown groups; `work_related_injury`: 7.0-year gap;
even the numeric ASIA motor score's missingness carried a 6.0-year gap).
This happens because NSCISC's intake form collected different fields in
different eras, so "was this ever collected" is structurally a proxy for
"how long ago was this patient enrolled" -- which is exactly the quantity
the fixed-horizon label's cohort-inclusion rule depends on.

**Fix**: every missing categorical value is now imputed with that field's
population **mode**, and every missing numeric value with its **median**
(`data_pipeline.py`, persisted to `imputation_modes.json`) -- with **no
separate "Unknown" category or "_missing" indicator flag** exposed to the
model. This removes the channel the model was using to infer "how long has
this patient been followed" instead of learning real clinical patterns.
The tradeoff, stated plainly: genuinely clinical missingness (e.g. an AIS
grade that couldn't be assessed because a patient was unstable) is no
longer distinguishable from administrative non-collection. The properly
rigorous fix would be either (a) obtaining an enrollment/calendar-year
field to model era explicitly and test for residual bias, or (b) treating
this as what it structurally is -- a right-censored survival problem --
with a discrete-time hazard or Cox model rather than a hard-cutoff
classification. Both are reasonable future-work items beyond this
prototype's scope.

**Result after all of §3.1-3.3**: Layer 1 accuracy 0.683, AUC 0.758, with
importance spread naturally across ten-plus features (max single-feature
importance 0.046, vs. 0.169 for the worst leaking feature beforehand) and
every top feature clinically plausible (education level, AIS grade,
vertebral/associated injury, pre-injury employment status).

### 3.4 Critical bug found by the regression test: train/serve dtype mismatch

After the retrain above, the automated accuracy-regression test
(`test_api_reproduces_reported_layer1_accuracy`) failed badly: **53.5%**
accuracy through the live API vs. **68.3%** reported by the training
script on the same held-out patients, with class-level accuracy of **5.5%**
for Non-RTW vs 97.4% for RTW -- the model was essentially always predicting
RTW. Root cause: `pd.read_csv()` (used when training) auto-infers
numeric-looking categorical columns like `sex` as `float64`, so
`OneHotEncoder` was fit on values like `1.0` (a Python float). The API
receives JSON and Pydantic declares these same fields as `str`, so it
always sends the *string* `"1.0"`. In Python, `"1.0" != 1.0`, so
`OneHotEncoder` (with `handle_unknown="ignore"`) silently treated every
single categorical value arriving through the API as "never seen before"
and zeroed it out -- the model was making real predictions on almost no
information, and happened to fall back to roughly the (now-majority) RTW
class. Caught entirely by the population-level accuracy-regression test;
individual hand-picked example patients still produced plausible-*looking*
single numbers and did not reveal it, which is why that regression test
exists at all rather than relying on spot checks.

**Fix**: `train_models.py` now explicitly casts every categorical column to
`str` (`cast_categoricals()`) before fit/transform, for both train and test
splits, so training-time dtype matches serving-time dtype by construction
rather than by coincidence of how pandas happened to infer a column's
dtype. Verified: `test_api_reproduces_reported_layer1_accuracy` now passes
at 68.5% (API) vs 68.3% (training script) -- matching within noise.

## 4. Other differences from the proposal

- **3-step form, not 5.** "Vocational factors" has no backing NSCISC
  variables (omitted, not shown non-functional). "Psychosocial" is also
  gone -- its only backing variable, PHQ-9, was removed per §3.2.
- **Login, user-account management and the patient-profile database were
  missing in rounds one and two and are now implemented** (§6.1). The form
  gained a leading "Patient" step, so it is now 4 steps: Patient plus the
  3 data-entry steps.
- **Dashboard and Model Insights show real values**, not the proposal's
  illustrative mock-up numbers (Figs. 3.7/3.13/3.14 show e.g. 87.4%
  accuracy, AUC 0.914, "97% RTW probability" -- do not present these as
  results; the trained models achieve ~68% accuracy / 0.76 AUC on Layer 1).
- **"Top predictive features" are per-patient** (XGBoost per-prediction
  contributions, `pred_contribs=True`), closer to the SHAP panel in the
  proposal than a single global importance list.
- Optional fields with no value are described to the user as "not
  specified" and silently imputed with the trained population mode/median
  server-side -- there is no "Unknown" option to select any more (§3.3).

## 5. Bugs found and fixed, in order found

1. Misleading dropdown defaults (fixed, then superseded by §3.3's removal
   of "Unknown" as a category entirely).
2. Job-class unknown code mismatch (`9` vs `99`) -- fixed, then the field
   was dropped entirely per §3.2.
3. Raw feature names in the UI (`"pre_injury_employment_status_5.0"`)
   replaced with dictionary labels throughout.
4. Pydantic `.dict()` deprecation → `.model_dump()`.
5. **Right-censoring label bias** -- fixed via the fixed-horizon
   relabeling (§3.1).
6. **PHQ-9 / race as injury-era and no-benefit proxies** -- dropped (§3.2).
7. **Dataset-wide missingness-as-era-proxy** -- fixed via mode/median
   imputation with no missingness indicator (§3.3).
8. **Train/serve categorical dtype mismatch** -- the most severe bug found;
   silently made the deployed model near-useless despite looking fine on
   individual examples. Fixed via explicit string-casting (§3.4).

Every one of 1-8 above was caught by an automated test before being
described as fixed, not by inspection alone -- several (5, 6, 8 especially)
would not have been visible from the UI or from a handful of manually-tried
patients.

## 6. Round three: login, database, and a more rigorous evaluation

### 6.1 Login, roles and the database (FR1, FR2, FR7; TC01, TC06)

The three functions the proposal required that rounds one and two did not
have are now implemented, following the use case diagram (Fig. 3.4) and the
ERD (Fig. 3.6):

| Role | Can do | Cannot do |
|---|---|---|
| Doctor / Therapist | Manage patient profiles, run and save predictions, add clinician notes, view all history, view the model report | Manage user accounts |
| Patient | View the results and history of the patient profile linked to their own account | See any other patient, run predictions, view the model report |
| Administrator | Create, approve, edit, deactivate and remove user accounts; view the model report | See patient records or run predictions |

- **Database**: SQLite (`api/sci_rtw.db`, created on first run), with one
  table per ERD entity: `users`, `patients`, `assessments`, `predictions`,
  `prediction_reports`. Model Evaluation stays in `models/*.json`, written
  by the training script, so the reported metrics have one source.
- **Registration**: a patient account works immediately but sees nothing
  until a clinician links it to a patient profile. A doctor or therapist
  account is created inactive and needs administrator approval, because
  those roles can read every patient record.
- **Passwords** are stored as salted PBKDF2-SHA256 hashes and login tokens
  as SHA-256 hashes (checked by `test_passwords_are_not_stored_in_plain_text`).
- **First administrator**: username `admin`, password `admin123` unless the
  `SCI_RTW_ADMIN_PASSWORD` environment variable is set before the first
  run. This default is for demonstration only and should be changed.

Not implemented, stated plainly: there is no password-reset-by-email, no
audit log, and the API has no HTTPS or rate limiting of its own. These are
appropriate for a real deployment but outside a prototype's scope.

**Bug 9, found by the browser walkthrough, not the unit tests**: with the
real server, a page that fires several requests at once (the dashboard
loads history and model info together) intermittently got HTTP 500. FastAPI
can run a request's database dependency and its endpoint on different
worker threads, which SQLite refuses by default. All 30 one-at-a-time API
tests passed while this was broken. Fixed in `api/db.py`
(`check_same_thread=False`, safe because each connection belongs to one
request), with `test_concurrent_requests_do_not_fail` starting a real
server to guard it. That test was confirmed to fail before the fix.

### 6.2 Model evaluation (proposal §3.6.1)

`src/train_models.py` now does what §3.6.1 describes: randomised
hyperparameter search and 5-fold stratified cross-validation on the
training split only, one final evaluation on the held-out test split, a
training-versus-test accuracy comparison as the overfitting check,
bootstrap 95% intervals, calibration, and permutation importance per input
field. Full numbers are in `models/evaluation_report.json`; figures are in
`docs/figures/` (`python src/make_figures.py`).

The processed dataset was regenerated from the NSCISC 2021 public files and
matches the previous version cell for cell (16,841 patients), and the
imputation values are the same whether computed from all patients or from
training patients only, so no information leaks from the test split there.

**Layer 1 (RTW within 10 years), test set of 3,369 patients**

| Model | Test accuracy (95% interval) | ROC-AUC | 5-fold CV accuracy | Train − test gap |
|---|---|---|---|---|
| XGBoost (main) | 68.0% (66.4 – 69.5) | 0.763 | 68.3% ± 0.9 | 2.4 pts |
| Random Forest | 68.3% (66.7 – 69.9) | 0.758 | 68.3% ± 0.6 | 3.3 pts |
| Logistic Regression | 68.1% (66.4 – 69.6) | 0.759 | 68.1% ± 0.6 | 0.5 pts |
| Always predict "RTW" | 53.9% | 0.500 | | |

What this supports, and what it does not:
- All three models beat the majority-class reference by about 14 points,
  well outside the interval. The model has real predictive value.
- **The three models are not distinguishable from each other.** Their
  intervals overlap almost entirely. XGBoost is kept as the deployed model
  as the proposal specifies and because it gives per-patient explanations,
  but the report should not claim it is more accurate than the baseline.
- Tuning changed little (Logistic Regression kept its defaults), and
  cross-validation agrees with the test result, so 68% is a stable estimate
  for these 14 inputs, not an artefact of one split.
- Overfitting is small (gap of 0.5 to 3.3 points).
- Predicted probabilities are close to observed rates (Brier score about
  0.20), slightly under-predicting RTW in the middle of the range.

**Layer 2 (timing of first RTW), test set of 1,817 patients**

| Model | Test accuracy | Macro-F1 | Recall: Year 1 / Year 5 / By Year 10 | Train − test gap |
|---|---|---|---|---|
| XGBoost (main, class-weighted) | 53.8% | 0.465 | 65% / 29% / 54% | 5.8 pts |
| Random Forest | 57.3% | 0.474 | 74% / 28% / 43% | 17.6 pts |
| Logistic Regression | 53.7% | 0.464 | 64% / 27% / 57% | 2.9 pts |
| XGBoost, unweighted (round two) | 60.8% | 0.443 | 87% / 23% / 22% | |
| Always predict "Year 1" | 59.2% | | 100% / 0% / 0% | |

- **Layer 2 is weak, and this is a limitation to report, not hide.** No
  model beats "always predict Year 1" on accuracy. Round two's 60.6%
  looked better only because that XGBoost was the one model trained
  without class weighting, so it predicted "Year 1" for most patients.
- Class weighting was added to XGBoost so the three models are compared
  like for like. It trades accuracy for balance: recall for "By Year 10"
  rises from 22% to 54%. Which is preferable depends on the use; macro-F1
  and per-class recall are the honest metrics to quote for this layer, not
  accuracy alone.
- "Year 5" is poorly identified by every model (27 to 29% recall).
- Random Forest overfits Layer 2 clearly (17.6-point gap).
- The likely cause is the data, not the algorithm: three models reach the
  same ceiling, and the timing classes come from follow-up visits at years
  1, 5 and 10, which is a coarse measurement of when someone returned.

**Objective (a): most influential factors** (drop in ROC-AUC when the field
is shuffled in the test set, XGBoost)

- Layer 1: education level (0.062), AIS grade at discharge (0.030),
  pre-injury employment status (0.030), ASIA motor score (0.026), injury
  completeness (0.016). Sex, veteran status and work-related injury have
  no measurable effect.
- Layer 2: age at injury (0.053), pre-injury employment status (0.026),
  ASIA motor score (0.019), education level (0.010).

These are associations in NSCISC data, not causes.

### 6.3 Still different from the proposal

- **Functional variables**: only the ASIA motor score is used. FIM was
  tested and rejected on evidence (§6.4). SCIM, walking ability and
  wheelchair use (Table 3.12, Fig. 3.10) are recorded for under 3% of the
  cohort in the public files, so they cannot be used.
- **Psychosocial and vocational pages** (Figs. 3.11, 3.12) remain dropped,
  for the reasons in §3.2 and §4.
- **Tables 3.5 and 3.6** in the proposal describe the original label
  (28,915 patients, "Year 10 or later"). The deployed models use the
  fixed-horizon label (16,841 patients, "By Year 10"); §3.1 gives the
  justification to carry into the final report.
- **U.S. veteran status** is not asked on the form. It is still a model
  input, but it does not apply to patients outside the U.S. and has no
  measurable effect (§6.2: shuffling it changes ROC-AUC by about 0.0001),
  so the API fills in the usual value, "No", as it does for any optional
  field left blank. The models and every figure in this report are
  unchanged.

### 6.4 FIM motor score: tested, not adopted

The proposal lists FIM as a functional predictor. The FIM motor total at
discharge (`AFScorDs`) is recorded for 59% of the Layer 1 training
patients, so it was tested rather than assumed (`python
src/experiment_fim.py`, 5-fold cross-validation on the training split):

| Layer 1 feature set | XGBoost ROC-AUC | Logistic Regression ROC-AUC |
|---|---|---|
| Deployed (no FIM) | 0.756 ± 0.007 | 0.750 ± 0.005 |
| + FIM, median-imputed where missing | 0.759 ± 0.007 | 0.751 ± 0.005 |
| + FIM with the real scores scrambled | 0.756 ± 0.007 | 0.750 ± 0.005 |
| + only a flag for "FIM not recorded" | 0.757 ± 0.006 | 0.752 ± 0.005 |
| Patients with a recorded FIM only: without → with FIM | | 0.735 → 0.736 |

- The gain from adding FIM (+0.0025 AUC) is about a third of the
  fold-to-fold variation, so it is not distinguishable from noise.
- A bare "not recorded" flag does about as well as the real score, and
  among patients who all have a score, FIM adds nothing. Whatever small
  gain exists is therefore not coming from the clinical content of FIM.
- Whether FIM was recorded is strongly tied to enrolment era: patients
  with a recorded score were followed for 12.1 years on average against
  22.9 for the rest, and their RTW rates are 63% against 41%. This is the
  same artefact removed in §3.3, so including FIM risks reintroducing it.
- Layer 2 shows the same pattern (ROC-AUC 0.728 with or without FIM).

Conclusion: FIM is left out. It would add an input for the clinician to
enter with no measurable benefit. A plausible reason is that the ASIA
motor score, AIS grade and injury completeness already carry most of the
functional information FIM would add.

### 6.5 Confidence level and suggested considerations (Fig. 3.13)

Both were shown on the proposal's results screen and are now implemented.

**Confidence level.** Defined by how far the estimated probability is from
50%, and reported with the accuracy measured at that level on the held-out
test set (XGBoost, 3,369 patients), so the label is a measured claim:

| Level | Estimated RTW probability | Share of test patients | Predictions correct |
|---|---|---|---|
| Low | between 35% and 65% | 42.7% | 54.5% |
| Moderate | 65–80%, or 20–35% | 36.3% | 71.7% |
| High | 80% or more, or 20% or less | 21.0% | 89.2% |

This is worth stating in the report: the overall 68% accuracy hides a wide
spread. For about a fifth of patients the model is right nine times in
ten; for over two fifths it is barely better than chance, and the
interface says so instead of presenting every estimate as equally solid.

**Suggested considerations.** Up to four planning prompts chosen by fixed
rules in `api/recommendations.py`, not by the model: one for the overall
direction (or a statement of uncertainty at Low confidence), then prompts
matched to the fields that lowered this patient's estimate (education,
pre-injury employment, injury severity and motor function, age), then one
on timing when RTW is predicted. Each is shown with the reason it
appeared. They are worded as things to consider and are labelled as not
being treatment recommendations, in keeping with proposal §1.4 and §3.6.4;
`test_recommendations_never_give_medical_instructions` checks every
combination of inputs for this. The clinician's own notes remain separate.

Both are stored with each saved prediction. A database created before this
change is upgraded automatically at startup: the two columns are added and
existing records are completed by re-running their stored inputs, but only
where that reproduces the stored probability, so a record made by a since
retrained model is never given suggestions that do not belong to it
(`test_database_from_earlier_version_is_upgraded_without_losing_records`).

Limitations: the rules are the author's, informed by the literature in
Chapter 2, and have not been reviewed by clinicians. The bands are fixed
at 15 and 30 points from 50% as a simple, explainable choice.

### 6.6 Split ratio, number of folds, and significance testing

The proposal leaves the train-test ratio and the number of folds open
(§3.6.1, Table 3.16), to be settled by experiment, and says the most
appropriate model will be chosen (§3.5.4). `python
src/experiment_splits.py` runs those experiments; full numbers are in
`docs/split_experiment_results.json`. Every run uses the tuned settings
from §6.2, so only the split or the fold count varies. It changes nothing
in the deployed models. The tables below are from the rerun after the
motor score fix of §6.8.

**Split ratio** (XGBoost, mean ± spread over 10 random splits per ratio)

| Train/test | Layer 1 accuracy | Layer 1 ROC-AUC | Layer 2 accuracy | Layer 2 macro-F1 |
|---|---|---|---|---|
| 60/40 | 67.1% ± 0.5 | 0.741 | 55.7% ± 0.7 | 0.480 |
| 70/30 | 67.4% ± 0.5 | 0.743 | 55.9% ± 1.0 | 0.482 |
| 80/20 (deployed) | 67.3% ± 0.6 | 0.742 | 56.1% ± 0.9 | 0.484 |
| 90/10 | 67.3% ± 0.9 | 0.742 | 55.4% ± 1.5 | 0.473 |

**Number of folds** (XGBoost, cross-validation on the training split)

| Folds | Layer 1 accuracy | Layer 2 accuracy |
|---|---|---|
| 3 | 67.2% ± 0.9 | 57.2% ± 0.7 |
| 5 (deployed) | 67.2% ± 1.1 | 56.9% ± 0.7 |
| 10 | 67.2% ± 1.2 | 57.5% ± 1.2 |

- No ratio and no fold count gives a better result. The differences
  between ratios are under one point, which is no more than the spread
  between two random splits of the same ratio. Random Forest and Logistic
  Regression show the same pattern.
- 80/20 and 5 folds are therefore kept as conventional choices, not
  winning ones. Picking a ratio because it happened to score highest
  would make the reported figure optimistic, since the test set would
  then have helped choose it.

**McNemar's exact test** on the deployed held-out test split (pairwise
p-values Holm-adjusted for the three comparisons)

| Comparison | Layer 1 (3,369 patients) | Layer 2 (1,817 patients) |
|---|---|---|
| Logistic Regression vs Random Forest | no difference (p = 1.0) | Random Forest better (p = 0.008) |
| Logistic Regression vs XGBoost | no difference (p = 1.0) | no difference (p = 0.20) |
| Random Forest vs XGBoost | no difference (p = 0.91) | Random Forest better (p = 0.02) |
| Logistic Regression vs majority class | model better (p < 10⁻²⁵) | majority class better (p < 0.0001) |
| Random Forest vs majority class | model better (p < 10⁻²⁵) | majority class better (p = 0.01) |
| XGBoost vs majority class | model better (p < 10⁻²⁵) | majority class better (p = 0.0002) |

- Layer 1: all three clearly beat the majority-class reference, and no
  pair of models differs (67.6%, 67.1% and 67.3%). After the AIS grade
  fix alone XGBoost had been ahead of Random Forest (p = 0.04); that
  small lead went with the motor score fix. The test gives no reason to
  prefer one algorithm on accuracy. XGBoost stays the deployed model
  because it has the highest ROC-AUC and the best-calibrated estimates
  (Brier score 0.204), and those margins are small.
- Layer 2: Random Forest is significantly more accurate than the other
  two (55.8% against 54.1% and 52.9%), but it is significantly worse
  than always answering "Year 1", it overfits the most (a train − test
  gap of about 11 points), and its macro-F1 is about the same (0.470
  against 0.460).
  XGBoost stays the deployed Layer 2 model for those reasons; on
  accuracy alone, the proposal's primary metric, Random Forest would be
  chosen, and the report should say so.

### 6.7 Input validation, and one coding for AIS grade

**Input validation.** Probing the API by hand showed it accepted values
that cannot be real and still returned a normal-looking prediction: an age
at injury of 500 or −5, a motor score of 900. A value it did not recognise
in a category field (`sex: "female"`, AIS grade `"d"`) was swapped
silently for that field's most common value, so a typing mistake produced
a confident estimate for what was in effect a different patient.

- Age at injury must now be 0 to 110 and the motor score 0 to 100.
- A category value the model was not trained on is refused with a 422
  that names the field and lists the valid values. A blank optional field
  is still filled with the population value, as before.
- The form applies the same ranges before sending, and shows the allowed
  range under the field.
- Tests: `test_unrecognised_category_value_rejected`,
  `test_out_of_range_number_rejected`,
  `test_number_at_range_limit_accepted`. The old
  `test_unseen_category_value_falls_back_to_mode_not_crash` asserted the
  behaviour that was removed and was replaced by the first of these.

**AIS grade was recorded two ways.** The `AASAImDs` field holds the old
numeric Frankel code for older injuries and the AIS letter for newer ones.
The pipeline kept both, so the model saw ten categories:

| Coding | Patients | Injury year (median, range) | Returned to work |
|---|---|---|---|
| Frankel 1–5 | 7,593 | 1983 (1972–1993) | 40.2% |
| AIS A–E | 8,657 | 2004 (1973–2020) | 65.2% |

The same clinical grade had very different outcomes depending only on how
it was written down: "Complete" was followed by return to work in 37% of
patients coded `5` and 58% of those coded `A`. This is the era-proxy
problem of §3.2 and §3.3 again. The coding told the model when the patient
was injured, and under the fixed-horizon label that predicts the outcome
for reasons that have nothing to do with the injury.

Fix: `data_pipeline.py` maps each Frankel code to the AIS letter for the
same grade (5→A, 1→B, 2→C, 3→D, 4→E) before imputation, and all models
were retrained with the same split, seed and search. The form now offers
A to E only, and a blank AIS grade is filled with `A` instead of `5`.
Predictions saved earlier keep the code that was entered.

**Layer 1 after the fix** (test set of 3,369 patients; before → after)

| Model | Test accuracy (95% interval) | ROC-AUC | 5-fold CV accuracy | Train − test gap |
|---|---|---|---|---|
| XGBoost (main) | 68.0% → 68.2% (66.5 – 69.7) | 0.763 → 0.758 | 67.8% ± 1.3 | 1.5 pts |
| Random Forest | 68.3% → 66.8% (65.2 – 68.5) | 0.758 → 0.750 | 67.2% ± 1.2 | 4.7 pts |
| Logistic Regression | 68.1% → 67.4% (65.8 – 69.0) | 0.759 → 0.742 | 67.0% ± 1.0 | 0.2 pts |
| Always predict "RTW" | 53.9% | 0.500 | | |

- The cost is small: XGBoost's accuracy is unchanged within noise and
  its ROC-AUC falls by 0.006. Logistic Regression loses the most (0.016
  AUC), because a separate coefficient per code was its only way to use
  the era signal.
- **AIS grade was mostly an era signal.** Its permutation importance fell
  from 0.030 (second) to 0.005 (ninth). The current Layer 1 ranking is
  education level (0.065), ASIA motor score (0.048), pre-injury employment
  status (0.031), injury completeness (0.018), associated injury (0.013).
- Confidence levels, measured again: Low 56.7% correct (1,520 patients),
  Moderate 71.4% (1,173), High 88.3% (676). Brier score 0.199.
- Layer 2 did not move: XGBoost 53.8% accuracy and 0.464 macro-F1,
  Random Forest 57.2% and 0.472, Logistic Regression 52.8% and 0.461,
  against 59.2% for always answering "Year 1". It remains weak.

**Not fixed, and a limitation to report.** The motor score's importance
rose from 0.026 to 0.048 when AIS grade's fell. Part of that is likely the
same era signal moving to the next place it can be read: the motor score
is missing for 71% of Frankel-coded patients against 10% of the rest, and a missing score is filled with the median, 50. A motor score of
exactly 50 therefore still marks an older record (median injury year 1981
when missing, 2001 when recorded; 41% against 62% returned to work).
Median imputation removed the explicit missing flag in §3.3 but cannot
remove this. Dealing with it properly means modelling the era directly or
restricting the cohort to patients with a recorded score, either of which
changes the cohort and is left as future work. (A third way that keeps the
cohort was found and applied: §6.8.)

### 6.8 Security hardening, and the motor score fill

**Security.** A review of the login and record-keeping code found five
gaps. None had been exploited; each is now closed and covered by
`tests/test_security.py`.

| Gap | Change |
|---|---|
| Sign-in could be tried without limit, so a password could be guessed. | Five wrong passwords in a row lock that username for 15 minutes (HTTP 429). An unknown username is counted and timed the same way, so neither reveals which usernames exist. |
| The administrator could stay on the published default password `admin123`. | An account still on that password can reach only "change password": every other endpoint answers 403 and the frontend shows a "Choose a new password" screen. A database made by an earlier version is checked at startup. |
| Changing a password left other sessions of the account signed in. | They are signed out. |
| Nothing recorded who did what. Any clinician can remove a patient profile with all its predictions. | An audit log records sign-ins (including failed ones), account changes, and patient profiles created, changed or removed, with the user and time. The administrator reads it on the User Accounts screen (`GET /audit-log`). It never holds a password and keeps the username after an account is removed. |
| Any website could call the API from a visitor's browser (`allow_origins=["*"]`), and responses carried no protective headers. | Browser calls are accepted only from a frontend on the same machine unless `SCI_RTW_CORS_ORIGINS` lists others. Every response forbids framing and content-type sniffing, JSON is marked `no-store`, and the app is served with a content security policy. |

Not changed: a clinician can still see and remove every patient profile
(the proposal gives both clinical roles the same access), self-registration
is open, and the login token is kept in the browser's local storage. The
`Dockerfile` still runs the service as root; that was left because the
image could not be rebuilt and tested here.
**The motor score fill.** §6.7 left one limitation open: a missing ASIA
motor score was filled with the median, 50, so the model read exactly 50
as an older record. The prediction form fills a blank the same way, so
leaving the field blank was not neutral for a new patient, and neither was
a real score of 50. `python src/experiment_motor_score.py` measures this
on the Layer 1 algorithm and compares three treatments. The score is
recorded for 62% of the cohort; 61% of those patients returned to work
against 42% of the rest. The "dip" is the mean estimate with every test
patient's score set to 50, minus the mean with it set to 49 and 51.

| Training data | Dip at 50 | Accuracy / ROC-AUC, test patients with a recorded score (2,083) |
|---|---|---|
| Missing score filled with the median (as before) | −5.9 points | 67.7% / 0.741 |
| Only patients with a recorded score (8,403) | −1.0 points | 66.6% / 0.743 |
| Missing score filled with a random recorded score | +0.4 points | 67.2% / 0.741 |

- The dip was an artefact, not a clinical effect: it disappears when no
  single value marks a missing score, and patients with a recorded score
  are predicted equally well either way (ROC-AUC 0.741 to 0.743).
- Filling with a random recorded score removes the dip and keeps the
  whole cohort, so it was adopted.

Fix: `data_pipeline.py` now fills each missing motor score, after the
split, with a score drawn at random from the training patients who have
one (seeded, so the result is reproducible). The draws for the test split
also come from training patients, so no test patient's score reaches the
training data. Every other column and both splits are unchanged cell for
cell. All models were retrained with the same seed and search. A blank on
the form is still sent to the model as 50, the median recorded score, but
50 no longer carries any other meaning.

**Layer 1 after the fix** (test set of 3,369 patients; before → after)

| Model | Test accuracy (95% interval) | ROC-AUC | 5-fold CV accuracy | Train − test gap |
|---|---|---|---|---|
| XGBoost (main) | 68.2% → 67.6% (65.9 – 69.3) | 0.758 → 0.744 | 67.2% ± 1.1 | 1.8 pts |
| Random Forest | 66.8% → 67.1% (65.5 – 68.8) | 0.750 → 0.740 | 67.2% ± 1.1 | 4.5 pts |
| Logistic Regression | 67.4% → 67.3% (65.8 – 69.0) | 0.742 → 0.741 | 66.9% ± 1.1 | 0.2 pts |
| Always predict "RTW" | 53.9% | 0.500 | | |

- XGBoost gives up 0.6 points of accuracy and 0.014 of ROC-AUC. That is
  the price of no longer recognising a missing score: the earlier figure
  was partly earned on the patients whose score is missing, by reading
  that it was missing. Logistic Regression barely moves, because a linear
  model could not single out one value in the first place.
- **The motor score was mostly an era signal.** Its permutation importance
  fell from 0.048 (second) to 0.010 (seventh). The current Layer 1 ranking
  is education level (0.071), pre-injury employment status (0.034), injury
  completeness (0.018), associated injury (0.015), age at injury (0.012).
  With §6.7 this means the two leading clinical predictors of the first
  models, AIS grade and motor score, owed most of their rank to the era
  they were recorded in. The honest reading for Objective (a) is that
  education and employment before injury predict return to work far more
  strongly in this dataset than the injury itself.
- Confidence levels, measured again: Low 57.0% correct (1,644 patients),
  Moderate 72.4% (1,133), High 87.8% (592). Brier score 0.204.
- Layer 2 did not improve: XGBoost 54.1% accuracy and 0.460 macro-F1
  (recall 67% / 27% / 51% for Year 1 / Year 5 / By Year 10), Random
  Forest 55.8% and 0.470, Logistic Regression 52.9% and 0.459, against
  59.2% for always answering "Year 1". It remains weak.

### 6.9 Model version, and accuracy group by group

**Model version.** Predictions saved before a retrain stay in a patient's
history next to newer ones, and nothing told them apart. Each saved
prediction now carries a 12-character fingerprint of the model files that
produced it (`GET /health` gives the current one). A record from an
earlier model, including any saved before this change, opens with a notice
that the same details may give a different estimate today.

**Accuracy group by group.** `python src/evaluate_subgroups.py` scores the
deployed Layer 1 model separately for groups of the 3,369 test patients
(full table in `docs/subgroup_results.json`). Race and Hispanic origin are
not model inputs; they are read from the raw data only for this check.
"Sensitivity" is the share of patients who returned to work that the model
predicted would; "specificity" the share of those who did not that it
predicted would not.

| Group | Patients | Accuracy | Sensitivity | Specificity | Returned to work | Mean estimate |
|---|---|---|---|---|---|---|
| All | 3,369 | 67.6% | 63% | 73% | 53.9% | 50.4% |
| Male | 2,709 | 67.2% | 63% | 73% | 54.3% | 50.7% |
| Female | 660 | 69.1% | 64% | 75% | 52.6% | 49.2% |
| Age under 25 | 1,279 | 65.1% | 44% | 82% | 43.8% | 41.7% |
| Age 25 to 34 | 893 | 65.6% | 58% | 74% | 52.5% | 48.7% |
| Age 35 to 44 | 511 | 66.0% | 68% | 62% | 62.4% | 55.5% |
| Age 45 and over | 686 | 75.9% | 87% | 53% | 68.4% | 64.9% |
| Below high school | 966 | 64.7% | 16% | 93% | 37.3% | 32.1% |
| High school diploma | 1,804 | 63.9% | 62% | 66% | 53.8% | 51.0% |
| Above high school | 579 | 83.8% | 99% | 16% | 81.9% | 78.2% |
| AIS A | 1,650 | 65.6% | 50% | 81% | 49.1% | 43.8% |
| AIS B | 404 | 62.1% | 52% | 72% | 49.3% | 48.0% |
| AIS C | 405 | 69.4% | 65% | 74% | 53.3% | 51.0% |
| AIS D | 909 | 72.7% | 84% | 52% | 64.9% | 63.0% |
| White | 2,475 | 66.5% | 66% | 68% | 58.6% | 53.1% |
| Black | 607 | 70.8% | 53% | 83% | 39.7% | 42.6% |
| Hispanic | 300 | 70.7% | 52% | 86% | 45.3% | 42.6% |
| Motor score recorded | 2,083 | 67.2% | 67% | 67% | 62.9% | 54.3% |
| Motor score not recorded | 1,286 | 68.3% | 52% | 79% | 39.4% | 44.0% |

- **No group is served clearly worse on accuracy.** It runs from 62% to
  84%, and sex makes no difference. Removing race and Hispanic origin as
  inputs did not leave the model less accurate for Black or Hispanic
  patients.
- **The kind of error differs sharply, mainly by education.** For patients
  without a high school diploma the model almost always answers "will not
  return": it identified only 16% of those who did. For patients with more
  than a high school diploma it almost always answers "will return", and
  identified only 16% of those who did not. The accuracy of 84% in that
  group is little better than the 82% reached by answering "will return"
  for everyone in it. The same pattern, less strongly, runs by age and by
  AIS grade. A clinician should read a "Non-RTW" result for a young or
  less-educated patient as the group's usual outcome, not as a finding
  about that patient; the per-patient factors on the results page show
  when education is what drove an estimate.
- **Estimates run a little low overall** (50.4% against 53.9% observed),
  which follows from class weighting, and the gap is not even: 5.5 points
  low for White patients and 2.9 points high for Black patients.
- **The largest gap is by whether the motor score was recorded**, and it
  is the unsolved part of the problem in §3.3. Patients with a recorded
  score are the more recent ones, and they returned to work far more often
  (62.9% against 39.4%). Much of that difference is not real: under the
  fixed-horizon label a recent patient who has not yet been followed for
  ten years is in the cohort only if they returned to work. Before §6.8
  the model reproduced this gap by reading the missing score; now it
  cannot, so it sits between the two groups, 8.6 points low for one and
  4.6 points high for the other. Neither is the true rate for a patient
  injured today. That needs the method named in §3.3, a survival model
  that keeps the patients who are still being followed, and it is the
  first thing to do if this work is continued.

### 6.10 Response time, and the timing wording

**Response time (non-functional requirement "Performance", Table 3.8).**
The proposal asks for results "within a reasonable time" without a
figure. `python src/measure_performance.py` measures the API on a
throwaway database (full table in `docs/performance_results.json`). It
was run on an ordinary Windows laptop (Python 3.14), with requests made
in-process, so the network is left out.

| Request | Requests timed | Median | 95th percentile | Slowest |
|---|---|---|---|---|
| Prediction, not saved (`POST /predict`) | 300 | 30 ms | 37 ms | 98 ms |
| Prediction saved to a patient (the form's path) | 200 | 39 ms | 48 ms | 88 ms |
| History, one page of 50 | 20 | 19 ms | 23 ms | 27 ms |
| History, all 200 saved predictions | 20 | 48 ms | 72 ms | 78 ms |

- A prediction, with both layers, the per-patient explanation and the
  database write, takes about 40 ms. Even the slowest of 500 was under a
  tenth of a second, well inside any reasonable reading of the
  requirement.
- Start-up, loading the models and opening the database, takes about
  3.5 seconds and happens once.
- On the hosted copy the network round trip comes on top. The free plan
  also puts the service to sleep when idle, and the first visit after
  that waits about a minute for it to wake (README, section 8). That is a
  property of the free hosting, not of the system.

**Timing wording.** The results page described the three timing bars as
the "chance of first returning by each follow-up point", which reads as a
running total. They are the shares of one prediction across three
separate answers, and add up to 100%. The page now says that the bars
show which follow-up is most likely to be the first to find the patient
working.

