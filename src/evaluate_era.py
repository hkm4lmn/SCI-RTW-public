"""
SCI-RTW Predictor: how much do recently injured patients flatter the results?
(sensitivity check for Chapter 4)

Under the 10-year RTW label, a patient injured too recently to have been
followed for 10 years (the public data run to 2021) is in the cohort only if
they have already returned to work. Their outcome is therefore almost always
"returned to work", which makes them easy to predict.

This script scores the DEPLOYED models (Layer 1 XGBoost, Layer 2 Random
Forest, as saved in models/) on the held-out test patients, split into those
injured up to 2010, whose 10-year outcome is known whatever it was, and those
injured from 2011. The first group gives the fairer estimate for a newly
injured patient. Nothing is retrained.

Run:  python src/evaluate_era.py      (a few seconds)
Writes docs/era_sensitivity_results.json.
"""

import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import data_pipeline as dp
import train_models as tm

LAST_FULL_FOLLOWUP_YEAR = 2010
N_BOOTSTRAP = 1000
OUTPUT_PATH = tm.PROJECT_ROOT / "docs" / "era_sensitivity_results.json"


def injury_years(layer):
    ids = pd.read_csv(str(tm.PROCESSED_DIR / f"layer{layer}_test.csv"), usecols=["UniID"])["UniID"]
    f1 = pd.read_csv(str(dp.find_data_dir() / "f1_public_2021.csv"), usecols=["UniID", "AInjDt"],
                     low_memory=False).set_index("UniID")
    return pd.to_numeric(f1["AInjDt"], errors="coerce").reindex(ids).to_numpy()


def accuracy_ci(y, pred, rng):
    n = len(y)
    stats = [accuracy_score(y[i], pred[i]) for i in (rng.integers(0, n, n) for _ in range(N_BOOTSTRAP))]
    return [round(float(np.percentile(stats, 2.5)), 4), round(float(np.percentile(stats, 97.5)), 4)]


def layer1(rng):
    _, y_train, X_test, y_test = tm.load_layer(1)
    pre = joblib.load(str(tm.MODEL_DIR / "layer1_preprocessor.joblib"))
    booster = xgb.Booster()
    booster.load_model(str(tm.MODEL_DIR / "layer1_xgboost.json"))
    proba = booster.predict(xgb.DMatrix(pre.transform(X_test)))
    y, pred = np.asarray(y_test), (proba >= 0.5).astype(int)
    majority = int(np.bincount(np.asarray(y_train)).argmax())
    years = injury_years(1)
    out = {}
    for name, mask in [("All test patients", np.ones(len(y), bool)),
                       (f"Injured up to {LAST_FULL_FOLLOWUP_YEAR}", years <= LAST_FULL_FOLLOWUP_YEAR),
                       (f"Injured {LAST_FULL_FOLLOWUP_YEAR + 1} or later", years > LAST_FULL_FOLLOWUP_YEAR)]:
        yy, pp, pr = y[mask], pred[mask], proba[mask]
        out[name] = {
            "n": int(mask.sum()),
            "observed_rtw_rate": round(float(yy.mean()), 4),
            "mean_estimate": round(float(pr.mean()), 4),
            "accuracy": round(float(accuracy_score(yy, pp)), 4),
            "accuracy_95ci": accuracy_ci(yy, pp, rng),
            "roc_auc": round(float(roc_auc_score(yy, pr)), 4) if len(set(yy)) == 2 else None,
            "majority_baseline": round(float((yy == majority).mean()), 4),
        }
    return out


def layer2(rng):
    _, _, X_test, y_test = tm.load_layer(2)
    pre = joblib.load(str(tm.MODEL_DIR / "layer2_preprocessor.joblib"))
    model = joblib.load(str(tm.MODEL_DIR / "layer2_random_forest.joblib"))
    encoder = joblib.load(str(tm.MODEL_DIR / "layer2_label_encoder.joblib"))
    pred = encoder.inverse_transform(model.predict(pre.transform(X_test)))
    y = np.asarray(y_test)
    years = injury_years(2)
    out = {}
    for name, mask in [("All test patients", np.ones(len(y), bool)),
                       (f"Injured up to {LAST_FULL_FOLLOWUP_YEAR}", years <= LAST_FULL_FOLLOWUP_YEAR),
                       (f"Injured {LAST_FULL_FOLLOWUP_YEAR + 1} or later", years > LAST_FULL_FOLLOWUP_YEAR)]:
        yy, pp = y[mask], pred[mask]
        out[name] = {
            "n": int(mask.sum()),
            "class_share": {c: round(float((yy == c).mean()), 4) for c in tm.TIMING_ORDER},
            "accuracy": round(float(accuracy_score(yy, pp)), 4),
            "accuracy_95ci": accuracy_ci(yy, pp, rng),
            "f1_macro": round(float(f1_score(yy, pp, average="macro", labels=tm.TIMING_ORDER)), 4),
            "baseline_always_year1": round(float((yy == "Year 1").mean()), 4),
        }
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(tm.RANDOM_STATE)
    results = {"layer1_xgboost": layer1(rng), "layer2_random_forest": layer2(rng)}
    for layer, groups in results.items():
        print(f"\n{layer}")
        for name, r in groups.items():
            print(f"  {name:28} " + "  ".join(f"{k}={v}" for k, v in r.items()))
    with open(str(OUTPUT_PATH), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUTPUT_PATH}")
