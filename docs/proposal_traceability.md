# What the proposal promised, and where the evidence is

A map from each objective and requirement in the CSP600 proposal to what
was built and where it is shown, for writing the results and conclusion
chapters. Section numbers (§) refer to `docs/test_report.md`. Figures are
the current ones, after the retraining in §6.8.

## Research objectives (proposal §1.3)

| Objective | Status | Evidence |
|---|---|---|
| (a) Identify which clinical and personal factors most influence return to work | Met, with a finding to state plainly | Permutation importance per input field on held-out patients (§6.2, current ranking in §6.8). Layer 1: education level, pre-injury employment status, injury completeness, associated injury, age at injury. AIS grade and motor score ranked high at first but owed most of that rank to the era the record came from (§6.7, §6.8). Figures: `docs/figures/layer1_factor_importance.png`, `layer2_factor_importance.png`. |
| (b) Develop a prototype using two-layer predictive models | Met | Layer 1 predicts RTW or Non-RTW; Layer 2 predicts timing only when RTW is predicted (TC04, TC05). The full system: login, roles, patient profiles, saved history, printable report (§6.1). |
| (c) Evaluate the accuracy of the two layers | Met | Layer 1 XGBoost: 67.6% accuracy, ROC-AUC 0.744, against 53.9% for always answering "RTW". Cross-validation, 95% intervals, overfitting check, calibration (§6.2), split ratios, fold counts and McNemar tests (§6.6), accuracy for each patient group (§6.9). Layer 2 XGBoost: 54.1% accuracy, below the 59.2% of always answering "Year 1"; reported as a limitation (§6.2, §6.8). |

## Functional requirements (proposal Table 3.7)

| ID | Requirement | Status | Evidence |
|---|---|---|---|
| FR1 | Register and login | Met | TC01; §6.1; sign-in lockout and forced change of the default password (§6.8) |
| FR2 | Manage patient profile | Met | `test_fr2_*`; Patients screen (screenshot 14) |
| FR3 | Enter SCI patient data | Met, with fewer sections | TC02, TC03; demographic, injury and functional steps. The vocational and psychosocial sections were dropped for lack of usable data (§4, §6.3) |
| FR4 | Generate RTW prediction | Met | TC04; screenshots 10, 11 |
| FR5 | Estimate RTW timing | Met, weak | TC05; accuracy below the simple reference (objective c) |
| FR6 | View prediction result | Met | Probability, confidence level, timing, per-patient factors and suggested considerations (§6.5); patients see their own (screenshot 19) |
| FR7 | View prediction history | Met | TC06; paging and model version on each record (§6.9) |
| FR8 | View model accuracy report | Met | TC07; Model Insights screen for clinicians and the administrator (screenshots 15, 16) |

## Non-functional requirements (proposal Table 3.8)

| Requirement | Status | Evidence |
|---|---|---|
| Usability | **Not yet measured** | Planned for the final weeks: `docs/usability_test_kit.md` (tasks, comprehension questions, SUS). Until then only the automated walkthrough (69 checks) shows the screens work, not that people find them easy. |
| Reliability (evaluation shown with suitable metrics) | Met | Objective (c) above; Model Insights shows the comparison, validation and confidence levels |
| Maintainability | Met | Data pipeline, training, figures and experiments are scripts that rerun end to end (README, "Retraining"); pinned dependency versions; 102 automated tests run on every push (`.github/workflows/tests.yml`) |
| Security and privacy | Met for a prototype | Hashed passwords, roles, lockout, audit log, protective headers (§6.8). The repository is private; the hosted copy holds invented patients only |
| Performance | Met | About 40 ms for a saved prediction, under 0.1 s for the slowest of 500 (§6.10) |

## Proposal test cases (Table 3.18)

All seven pass: TC01 to TC07, with the tests behind each listed in §2.

## Differences from the proposal to explain in the report

- **Outcome label.** Return to work within ten years, for patients whose
  ten-year outcome is known (16,841 patients), instead of "any follow-up
  shows it" (28,915). The original was biased by how long each patient was
  followed (§3.1).
- **Predictors removed.** Race and Hispanic origin (no accuracy benefit,
  fairness risk), PHQ-9 and pre-injury job class (stood in for the injury
  era), FIM (tested, not adopted, §6.4). U.S. veteran status is still a
  model input but is not asked on the form (§6.3).
- **Form sections.** Vocational and psychosocial pages dropped; a patient
  step added first (§4).
- **Missing values** are filled with the usual value instead of kept as
  "Unknown", because missingness itself gave away the era (§3.3); the motor
  score is filled with a random recorded score (§6.8).
- **AIS grade** recorded two ways in the source data was merged into one
  coding (§6.7).
- **Real figures replace the mock-up numbers** in the proposal's
  screen designs (e.g. 87.4% accuracy, AUC 0.914 in Figs. 3.7, 3.13,
  3.14): the trained models reach about 68% and 0.74 (§4).
- **Added beyond the proposal:** confidence level with measured accuracy,
  rule-based suggested considerations, printable report, clinician notes,
  audit log, model version on saved predictions.

## Limitations and future work (for the conclusion chapter)

1. **Timing (Layer 2) is weak.** No model beats always answering "Year 1"
   on accuracy. The follow-up visits at years 1, 5 and 10 measure timing
   coarsely.
2. **The data's era effect is reduced, not removed.** Patients who have
   not yet been followed for ten years enter the cohort only if they
   returned to work, so recent patients look more likely to return than
   they are (§6.9). The fix is a survival model that keeps patients still
   being followed; this is the first piece of future work.
3. **U.S. data only.** Not validated on Malaysian patients, whose job
   market, welfare support and education system differ.
4. **Errors are not even across groups.** For patients without a high
   school diploma the model almost always predicts Non-RTW (§6.9).
5. **The suggested considerations are the author's rules** and have not
   been reviewed by clinicians (§6.5).
6. **Usability is unmeasured** until the planned test is run.
