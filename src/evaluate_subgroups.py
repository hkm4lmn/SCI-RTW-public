"""
SCI-RTW Predictor: does the deployed model work equally well for everyone?

One accuracy figure for the whole test set can hide a group the model
serves badly. This script scores the deployed Layer 1 model (XGBoost)
separately for groups of the held-out test patients and prints the
comparison. It changes nothing; results are written to
docs/subgroup_results.json and discussed in docs/test_report.md, 6.9.

Groups: sex, age at injury, education, level of injury, AIS grade, whether
the ASIA motor score was recorded, and race and Hispanic origin. The last
two are NOT inputs to the model (they were removed, see data_pipeline.py);
they are read from the raw data here only to check that removing them did
not leave the model working worse for one group.

For each group:
  accuracy, ROC-AUC  - as for the whole test set, with a bootstrap 95%
                       interval on accuracy
  sensitivity        - of those who returned to work, the share predicted to
  specificity        - of those who did not, the share predicted not to
  observed / mean    - the actual return-to-work rate against the average
  estimate             estimate; a gap means the model is systematically
                       too high or too low for that group

Groups of fewer than MIN_GROUP test patients are listed but not scored: a
figure from so few patients would mostly be noise.

Run:  python src/evaluate_subgroups.py      (a few seconds)
"""

import json

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import roc_auc_score

import data_pipeline as dp
import train_models as tm

MIN_GROUP = 100
N_BOOTSTRAP = 1000


def age_band(age):
    return "Under 25" if age < 25 else "25 to 34" if age < 35 else "35 to 44" if age < 45 else "45 and over"


def education_band(code):
    level = int(float(code))
    return {1: "Below high school", 2: "Below high school", 3: "High school diploma"}.get(
        level, "Other, unclassified" if level == 8 else "Above high school")


def race_band(code):
    return {1: "White", 2: "Black"}.get(code, "Other or unknown")


def hispanic_band(code):
    return {0: "Not Hispanic", 1: "Hispanic"}.get(code, "Unknown")


LEVELS = {"C": "Cervical", "T": "Thoracic", "L": "Lumbar or sacral", "S": "Lumbar or sacral"}


def load():
    X_train, _, X_test, y_test = tm.load_layer(1)
    ids = pd.read_csv(str(tm.PROCESSED_DIR / "layer1_test.csv"), usecols=["UniID"])["UniID"]
    raw = pd.read_csv(str(dp.find_data_dir() / "f1_public_2021.csv"), usecols=["UniID", "ARace", "AHispnic"],
                      low_memory=False).set_index("UniID")
    motor_recorded = dp.build_predictors().set_index("UniID")[dp.MOTOR_SCORE].notna()

    preprocessor = joblib.load(str(tm.MODEL_DIR / "layer1_preprocessor.joblib"))
    booster = xgb.Booster()
    booster.load_model(str(tm.MODEL_DIR / "layer1_xgboost.json"))
    proba = booster.predict(xgb.DMatrix(preprocessor.transform(X_test)))

    groups = pd.DataFrame({
        "Sex": X_test["sex"].map({"1.0": "Male", "2.0": "Female"}).fillna("Other").to_numpy(),
        "Age at injury": X_test["age_at_injury"].map(age_band).to_numpy(),
        "Education": X_test["education_level"].map(education_band).to_numpy(),
        "Level of injury": X_test["neuro_level_discharge"].str[0].map(LEVELS).fillna("Other").to_numpy(),
        "AIS grade": ("AIS " + X_test["aisgrade_discharge"]).to_numpy(),
        "ASIA motor score": ids.map(motor_recorded).map({True: "Recorded", False: "Not recorded"}).to_numpy(),
        "Race (not a model input)": ids.map(raw["ARace"]).map(race_band).to_numpy(),
        "Hispanic origin (not a model input)": ids.map(raw["AHispnic"]).map(hispanic_band).to_numpy(),
    })
    return groups, y_test.to_numpy(), proba


def score(y, proba, rng):
    pred = (proba >= 0.5).astype(int)
    correct = pred == y
    n = len(y)
    boot = [correct[rng.integers(0, n, n)].mean() for _ in range(N_BOOTSTRAP)]
    return {
        "n": int(n),
        "accuracy": round(float(correct.mean()), 4),
        "accuracy_ci": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
        "roc_auc": round(float(roc_auc_score(y, proba)), 4) if 0 < y.sum() < n else None,
        "sensitivity": round(float(pred[y == 1].mean()), 4) if (y == 1).any() else None,
        "specificity": round(float(1 - pred[y == 0].mean()), 4) if (y == 0).any() else None,
        "observed_rtw_rate": round(float(y.mean()), 4),
        "mean_estimate": round(float(proba.mean()), 4),
    }


def pct(value):
    return "    -" if value is None else f"{value:5.1%}"


if __name__ == "__main__":
    groups, y, proba = load()
    rng = np.random.default_rng(tm.RANDOM_STATE)
    results = {"min_group": MIN_GROUP, "all_test_patients": score(y, proba, rng), "groups": {}}

    header = (f"{'':24} {'n':>6} {'accuracy (95% interval)':>26} {'AUC':>6} {'sens.':>6} {'spec.':>6} "
              f"{'observed':>9} {'estimate':>9}")
    print(header)

    def show(label, s):
        ci = f"{s['accuracy']:.1%} ({s['accuracy_ci'][0]:.1%} - {s['accuracy_ci'][1]:.1%})"
        auc = "     -" if s["roc_auc"] is None else f"{s['roc_auc']:6.3f}"
        print(f"{label:24} {s['n']:>6,} {ci:>26} {auc} {pct(s['sensitivity']):>6} {pct(s['specificity']):>6} "
              f"{pct(s['observed_rtw_rate']):>9} {pct(s['mean_estimate']):>9}")

    show("All test patients", results["all_test_patients"])
    for column in groups.columns:
        print(f"\n{column}")
        results["groups"][column] = {}
        counts = groups[column].value_counts()
        for value in sorted(counts.index, key=lambda v: -counts[v]):
            mask = (groups[column] == value).to_numpy()
            if mask.sum() < MIN_GROUP:
                results["groups"][column][value] = {"n": int(mask.sum()), "too_few_to_score": True}
                print(f"  {value:22} {int(mask.sum()):>6,}   (too few to score)")
                continue
            results["groups"][column][value] = score(y[mask], proba[mask], rng)
            show("  " + value, results["groups"][column][value])

    out = dp.PROJECT_ROOT / "docs" / "subgroup_results.json"
    with open(str(out), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")
