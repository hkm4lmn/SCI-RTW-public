"""
SCI-RTW Predictor: should the FIM motor score be a predictor? (experiment)

The proposal lists FIM among the functional variables (Table 3.12,
Fig. 3.10). This script tests whether adding the FIM motor total at
discharge (Form I, AFScorDs) improves either layer, using 5-fold
cross-validation on the training split only. It changes nothing: it only
prints a comparison. Result and conclusion: docs/test_report.md, section 6.4.

Four variants are compared, because FIM is unrecorded (code 99) for about
40% of patients and whether it was recorded is strongly tied to enrolment
era - the artefact described in data_pipeline.py:
  1. baseline              - the deployed feature set
  2. + FIM                 - real scores, median-imputed where missing
  3. + FIM shuffled        - same patients missing, real scores scrambled
                             among those who have one (clinical value removed)
  4. + missing flag only   - no score at all, just "was FIM recorded"
If (2) beats (1) but (3) or (4) do just as well, the gain is the artefact.
A last pair repeats the comparison among only patients who HAVE a score,
where missingness cannot play any part.

Run:  python src/experiment_fim.py      (about 2 minutes)
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import data_pipeline as dp
import train_models as tm

warnings.filterwarnings("ignore")

FIM_UNKNOWN = 99


def preprocessor(numeric_cols):
    return ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), tm.CATEGORICAL_COLS),
    ])


def models(layer, y):
    """The tuned settings train_models.py selected, so only the feature set varies."""
    if layer == 1:
        xgb = XGBClassifier(n_estimators=600, max_depth=3, learning_rate=0.04, subsample=0.85,
                            colsample_bytree=0.8, reg_lambda=3, random_state=tm.RANDOM_STATE, n_jobs=1)
        return {"xgboost": (xgb, {}),
                "logistic_regression": (LogisticRegression(max_iter=3000, class_weight="balanced"), {})}
    xgb = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.04, subsample=0.85,
                        colsample_bytree=0.6, reg_lambda=10, min_child_weight=6,
                        random_state=tm.RANDOM_STATE, n_jobs=1)
    return {"xgboost": (xgb, {"model__sample_weight": compute_sample_weight("balanced", y)}),
            "logistic_regression": (LogisticRegression(max_iter=3000, class_weight="balanced", C=0.01), {})}


def score(X, y, numeric_cols, model, params, scoring):
    s = cross_validate(Pipeline([("prep", preprocessor(numeric_cols)), ("model", model)]), X, y,
                       cv=tm.cv_splitter(), scoring=scoring, params=params, n_jobs=-1)
    return "  ".join(f"{k}={s['test_' + k].mean():.4f}±{s['test_' + k].std():.4f}" for k in scoring)


def run(layer, fim_by_id, followup_years):
    X_train, y_raw, _, _ = tm.load_layer(layer)
    ids = pd.read_csv(str(tm.PROCESSED_DIR / f"layer{layer}_train.csv"), usecols=["UniID"])["UniID"].values
    y = np.asarray(y_raw) if layer == 1 else pd.Series(y_raw).map(
        {"By Year 10": 0, "Year 1": 1, "Year 5": 2}).values
    scoring = ["accuracy", "roc_auc"] if layer == 1 else ["accuracy", "f1_macro", "roc_auc_ovr"]

    fim = pd.Series(ids).map(fim_by_id).astype(float).values
    known = fim != FIM_UNKNOWN
    median = np.median(fim[known])
    print(f"\n=== Layer {layer}: {len(ids):,} training patients, FIM recorded for {known.mean():.1%} "
          f"(median {median:.0f})")
    print(f"    mean years of follow-up: FIM recorded {followup_years.reindex(ids[known]).mean():.1f}, "
          f"not recorded {followup_years.reindex(ids[~known]).mean():.1f}")
    if layer == 1:
        print(f"    RTW rate: FIM recorded {y[known].mean():.1%}, not recorded {y[~known].mean():.1%}")

    imputed = np.where(known, fim, median)
    shuffled = imputed.copy()
    shuffled[known] = np.random.default_rng(0).permutation(fim[known])
    variants = {
        "1. baseline (no FIM)": None,
        "2. + FIM (median-imputed)": imputed,
        "3. + FIM shuffled among recorded": shuffled,
        "4. + missing flag only": (~known).astype(float),
    }
    for label, column in variants.items():
        X, numeric = X_train.copy(), list(tm.NUMERIC_COLS)
        if column is not None:
            X["extra"] = column
            numeric.append("extra")
        for name, (model, params) in models(layer, y).items():
            print(f"  {label:34} {name:20} {score(X, y, numeric, model, params, scoring)}")

    # Among patients who all have a FIM score, missingness cannot contribute.
    X_known, y_known = X_train[known].copy(), y[known]
    lr = models(layer, y_known)["logistic_regression"][0]
    print(f"  {'recorded-FIM patients only, no FIM':34} {'logistic_regression':20} "
          f"{score(X_known, y_known, list(tm.NUMERIC_COLS), lr, {}, scoring)}  (n={len(y_known):,})")
    X_known["extra"] = fim[known]
    print(f"  {'recorded-FIM patients only, + FIM':34} {'logistic_regression':20} "
          f"{score(X_known, y_known, tm.NUMERIC_COLS + ['extra'], lr, {}, scoring)}  (n={len(y_known):,})")


if __name__ == "__main__":
    f1 = pd.read_csv(str(dp.find_data_dir() / "f1_public_2021.csv"), usecols=["UniID", "AFScorDs"])
    f2 = pd.read_csv(str(dp.find_data_dir() / "f2_public_2021.csv"), usecols=["UniID", "BYear", "BPrLvlSt"])
    followup = f2[f2["BPrLvlSt"].notna() & (f2["BPrLvlSt"] != 99)].groupby("UniID")["BYear"].max()
    for layer in (1, 2):
        run(layer, f1.set_index("UniID")["AFScorDs"], followup)
