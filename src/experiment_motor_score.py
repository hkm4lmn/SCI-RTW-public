"""
SCI-RTW Predictor: how should a missing ASIA motor score be filled? (experiment)

The motor score is unrecorded for 38% of the cohort, mostly older records.
The pipeline first filled every missing score with the median, 50. A score
of exactly 50 then marked an older record, and the model lowered its
estimate there. The prediction form fills a blank with the same 50, so
leaving the field blank was not neutral for a new patient
(docs/test_report.md, sections 6.7 and 6.8).

This script measures that dip and compares three ways of treating the
missing scores, for the deployed Layer 1 algorithm (XGBoost, tuned settings
from train_models.py). It changes nothing: it only prints a comparison.
It starts from the scores as recorded, not from the filled values in
processed/, so its result does not depend on what the pipeline now does.

  1. median fill (previous)   - missing scores filled with the median
  2. recorded scores only     - trained only on patients who have a score
  3. random-draw fill (now    - each missing training score replaced by a
     used by the pipeline)      score drawn from patients who have one, so
                                no single value marks a missing score

Each variant is scored on the same held-out test patients, twice: those
with a recorded score (where the fill plays no part in the input), and all
of them, with a missing test score filled the way the API fills a blank.
The "dip" is the mean estimate with every test patient's score set to the
fill value, minus the mean with it set one point either side.

Run:  python src/experiment_motor_score.py      (about 1 minute)
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

import data_pipeline as dp
import train_models as tm

warnings.filterwarnings("ignore")

FIELD = dp.MOTOR_SCORE


def load():
    """The deployed split, with each motor score as recorded (missing = NaN)."""
    X_train, y_train, X_test, y_test = tm.load_layer(1)
    recorded = dp.build_predictors().set_index("UniID")[FIELD]
    for split, X in (("train", X_train), ("test", X_test)):
        ids = pd.read_csv(str(tm.PROCESSED_DIR / f"layer1_{split}.csv"), usecols=["UniID"])["UniID"]
        X[FIELD] = ids.map(recorded).to_numpy()
    return X_train, y_train.to_numpy(), X_test, y_test.to_numpy()


def tuned_xgboost(y):
    with open(str(tm.MODEL_DIR / "evaluation_report.json")) as f:
        best = json.load(f)["layer1"]["tuning"]["xgboost"]["best_params"]
    return tm.base_models(1, y)["xgboost"].set_params(n_jobs=-1, **best)


def mean_estimate(pipe, X, score):
    forced = X.copy()
    forced[FIELD] = float(score)
    return float(pipe.predict_proba(forced)[:, 1].mean())


def report(label, pipe, X_test, y_test, fill):
    known = X_test[FIELD].notna().to_numpy()
    X_all = X_test.fillna({FIELD: fill})
    out = []
    for name, mask in (("recorded-score test patients", known), ("all test patients", np.ones(len(y_test), bool))):
        proba = pipe.predict_proba(X_all[mask])[:, 1]
        out.append(f"{name}: accuracy={accuracy_score(y_test[mask], proba >= 0.5):.4f} "
                   f"auc={roc_auc_score(y_test[mask], proba):.4f} (n={int(mask.sum()):,})")
    sweep = {s: mean_estimate(pipe, X_all, s) for s in (fill - 2, fill - 1, fill, fill + 1, fill + 2)}
    dip = sweep[fill] - (sweep[fill - 1] + sweep[fill + 1]) / 2
    print(f"\n{label}  (a blank is filled with {fill:.0f})")
    for line in out:
        print("  " + line)
    print("  mean estimate by forced score: " + "  ".join(f"{s:.0f}: {p:.3f}" for s, p in sweep.items()))
    print(f"  dip at the fill value: {dip * 100:+.1f} points")


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load()
    known = X_train[FIELD].notna().to_numpy()
    median = float(X_train[FIELD].median())
    print(f"Layer 1: {len(y_train):,} training patients, motor score recorded for {known.mean():.1%}; "
          f"{len(y_test):,} test patients, recorded for {X_test[FIELD].notna().mean():.1%}")
    print(f"RTW rate in training: score recorded {y_train[known].mean():.1%}, "
          f"not recorded {y_train[~known].mean():.1%}")
    print(f"Median of recorded training scores: {median:.0f}")

    X_median = X_train.fillna({FIELD: median})
    pipe = tm.make_pipeline(tuned_xgboost(y_train)).fit(X_median, y_train)
    report("1. median fill (previous)", pipe, X_test, y_test, median)

    X_known, y_known = X_train[known], y_train[known]
    pipe = tm.make_pipeline(tuned_xgboost(y_known)).fit(X_known, y_known)
    report(f"2. recorded scores only (n={len(y_known):,})", pipe, X_test, y_test, median)

    X_drawn = X_train.copy()
    rng = np.random.default_rng(tm.RANDOM_STATE)
    X_drawn.loc[~known, FIELD] = rng.choice(X_train.loc[known, FIELD].to_numpy(), size=int((~known).sum()))
    pipe = tm.make_pipeline(tuned_xgboost(y_train)).fit(X_drawn, y_train)
    report("3. random-draw fill (now used by the pipeline)", pipe, X_test, y_test, median)
