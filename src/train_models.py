"""
SCI-RTW Predictor: Model Development and Evaluation
Chapter 3, Section 3.5.4 (Model Development) and 3.6.1 (Model Evaluation)

Trains, for each layer:
  - XGBoost             (main model)
  - Random Forest       (comparison model)
  - Logistic Regression (baseline model)

Layer 1: binary classification (RTW vs Non-RTW within 10 years)
Layer 2: multiclass classification (Year 1 / Year 5 / By Year 10),
         trained only on RTW-positive patients.

What this script does, in order, per layer (all on the TRAINING split only
until the final step):
  1. Hyperparameter tuning  - randomised search, 5-fold stratified CV
  2. Cross-validation       - 5-fold scores (mean +/- std) for the tuned models
  3. Final fit on the full training split
  4. Held-out test evaluation - Table 3.17 metrics, computed once, on data
     that played no part in steps 1-3
  5. Overfitting check       - training accuracy vs test accuracy (3.6.1)
  6. Extras for the report   - majority-class baseline, bootstrap 95% CIs,
     calibration (Layer 1), per-class scores (Layer 2), and permutation
     importance per input field (Objective a: which factors matter most)

Outputs (models/):
  layer{1,2}_results.json   - test metrics + feature importance (read by the API)
  evaluation_report.json    - everything else above (read by the API and
                              by src/make_figures.py)
  model + preprocessor files loaded by api/main.py

Run:  python src/train_models.py          (about 5-10 minutes)
      python src/train_models.py --quick  (skips tuning; for a fast smoke test)
"""

import argparse
import json
import os
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_fscore_support,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# Project root = parent folder of this script's "src" folder.
# Expects <project_root>/processed/ (from data_pipeline.py) and creates
# <project_root>/models/ automatically.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS = 5
N_BOOTSTRAP = 1000
N_PERMUTATION_REPEATS = 10
MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]
# Layer 2 timing classes in chronological order (LabelEncoder sorts them
# alphabetically, which puts "By Year 10" first; reports use this order).
TIMING_ORDER = ["Year 1", "Year 5", "By Year 10"]

# race, hispanic_origin, and the PHQ-9 depression score were dropped after
# testing (see docs/test_report.md and data_pipeline.py docstring): PHQ acted
# as an injury-era proxy, not a clinical signal, and race/hispanic_origin gave
# no accuracy benefit while posing a fairness risk in a clinical tool.
# The "_missing" indicator flags were also removed (not just for PHQ): testing
# showed missingness across this dataset is pervasive and strongly correlated
# with how long a patient was followed, so a missing-flag became an
# era/cohort-pathway proxy rather than a clinical signal (see data_pipeline.py).
NUMERIC_COLS = ["age_at_injury", "asia_motor_total_discharge"]

# pre_injury_job_class dropped: still era-correlated after mode-imputation
# (its "N/A, not working" code alone showed a 5.6 vs 11.5-year follow-up gap)
# and near-redundant with pre_injury_employment_status. See data_pipeline.py.
CATEGORICAL_COLS = ["sex", "injury_cause", "vertebral_injury_level", "associated_injury",
                     "marital_status", "education_level",
                     "veteran_status", "pre_injury_employment_status",
                     "work_related_injury", "neuro_level_discharge", "aisgrade_discharge",
                     "injury_completeness_discharge"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# Randomised-search spaces. Kept modest on purpose: this dataset has only 14
# input fields, so the aim is to show the defaults were not badly chosen, not
# to squeeze out the last decimal.
SEARCH_SPACES = {
    "logistic_regression": {"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    "random_forest": {
        "n_estimators": [300, 500],
        "max_depth": [8, 12, 16],
        "min_samples_leaf": [1, 3, 5, 10],
        "max_features": ["sqrt", 0.3],
    },
    "xgboost": {
        "n_estimators": [200, 400, 600],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.02, 0.04, 0.08, 0.12],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 6],
        "reg_lambda": [1, 3, 10],
    },
}
SEARCH_ITERATIONS = {"logistic_regression": 6, "random_forest": 12, "xgboost": 25}


def cast_categoricals(df):
    """CRITICAL: force every categorical column to string dtype before
    fit/transform.

    Bug found by testing (see docs/test_report.md): plain pd.read_csv()
    auto-infers numeric-looking categorical columns (sex, education_level,
    etc.) as float64, so OneHotEncoder was fit on float 1.0/2.0/3.0. The API
    receives JSON and Pydantic declares these fields as `str`, so it always
    sends the *string* "1.0". "1.0" (str) != 1.0 (float) in Python, so every
    single categorical value arriving through the API was silently treated
    as "unseen" and one-hot encoded to all zeros -- the model was making
    real predictions on almost no information. This was caught by an
    end-to-end regression test (API accuracy vs. training-time accuracy on
    the same held-out patients), NOT by single hand-picked example patients,
    which still produced plausible-*looking* numbers because the model
    fell back to roughly the majority-class rate either way. Casting both
    train and serve sides to string dtype makes them match by construction
    instead of by coincidence of how pandas happened to infer a dtype.
    """
    df = df.copy()
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def build_preprocessor():
    # Numeric fields are standardised so Logistic Regression converges (left
    # raw, lbfgs hit its iteration limit). Tree models are unaffected by this:
    # their splits depend only on the ordering of values.
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
    ])


def get_feature_names(preprocessor):
    num_names = NUMERIC_COLS
    cat_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_COLS))
    return num_names + cat_names


def base_models(layer, y_train):
    """Untuned starting points; the search overrides the listed parameters.
    Every model is class-weighted so the comparison is like for like."""
    if layer == 1:
        xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08,
                            subsample=0.9, colsample_bytree=0.9,
                            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1)
    else:
        # XGBoost has no class_weight option for multiclass, so the balanced
        # weighting is passed as per-row sample weights at fit time instead
        # (see fit_params()). The first version of this script left Layer 2
        # XGBoost unweighted while the other two models were weighted, which
        # made it look more accurate only because it predicted the majority
        # "Year 1" class far more often.
        xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08,
                            subsample=0.9, colsample_bytree=0.9,
                            objective="multi:softprob", num_class=3,
                            eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=1)
    return {
        "logistic_regression": LogisticRegression(max_iter=3000, class_weight="balanced",
                                                  random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=12,
                                                class_weight="balanced",
                                                random_state=RANDOM_STATE, n_jobs=1),
        "xgboost": xgb,
    }


def fit_params(layer, name, y):
    if layer == 2 and name == "xgboost":
        return {"model__sample_weight": compute_sample_weight("balanced", y)}
    return {}


def make_pipeline(model):
    # The encoder sits inside the pipeline so that, during cross-validation,
    # it is refit on each training fold and never sees the validation fold.
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def cv_splitter():
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def cv_scoring(layer):
    if layer == 1:
        return {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}
    return {"accuracy": "accuracy", "f1_macro": "f1_macro", "roc_auc_ovr": "roc_auc_ovr"}


def tuning_metric(layer):
    # Layer 1: ROC-AUC (threshold-free, stable). Layer 2: macro-F1, because
    # plain accuracy rewards always predicting the majority "Year 1" class.
    return "roc_auc" if layer == 1 else "f1_macro"


def tune(layer, name, model, X, y):
    search = RandomizedSearchCV(
        make_pipeline(model),
        {f"model__{k}": v for k, v in SEARCH_SPACES[name].items()},
        n_iter=SEARCH_ITERATIONS[name], scoring=tuning_metric(layer), cv=cv_splitter(),
        random_state=RANDOM_STATE, n_jobs=-1, refit=False)
    search.fit(X, y, **fit_params(layer, name, y))
    best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    return best, float(search.best_score_)


def cross_validate_model(layer, name, model, X, y):
    scores = cross_validate(make_pipeline(model), X, y, cv=cv_splitter(), scoring=cv_scoring(layer),
                            params=fit_params(layer, name, y), n_jobs=-1)
    return {metric: {"mean": round(float(scores[f"test_{metric}"].mean()), 4),
                     "std": round(float(scores[f"test_{metric}"].std()), 4)}
            for metric in cv_scoring(layer)}


# ----------------------------------------------------------------------------
# Test-set metrics (Table 3.17)
# ----------------------------------------------------------------------------
def binary_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def multiclass_metrics(y_true, y_pred, y_proba, classes):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro"), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "roc_auc_ovr": round(roc_auc_score(y_true, y_proba, multi_class="ovr"), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "classes": list(classes),
    }


def bootstrap_ci(y_true, y_pred, y_proba, layer):
    """95% confidence intervals for the headline test metrics, by resampling
    the test set. Shows how much of a gap between two models is just noise."""
    rng = np.random.default_rng(RANDOM_STATE)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    acc, second = [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        acc.append(accuracy_score(y_true[idx], y_pred[idx]))
        if layer == 1:
            second.append(roc_auc_score(y_true[idx], y_proba[idx]))
        else:
            second.append(f1_score(y_true[idx], y_pred[idx], average="macro"))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"accuracy": ci(acc), ("roc_auc" if layer == 1 else "f1_macro"): ci(second)}


def feature_importance_from_model(model, feature_names, model_name):
    """Extract feature importance in a consistent format across model types."""
    if model_name == "logistic_regression":
        coef = model.coef_
        importance = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef[0])
    else:  # tree-based models expose feature_importances_
        importance = model.feature_importances_

    pairs = sorted(zip(feature_names, importance), key=lambda x: -x[1])
    return [{"feature": f, "importance": round(float(i), 5)} for f, i in pairs[:20]]


def permutation_importance_by_field(model, preprocessor, X_test, y_test, layer):
    """Objective (a): which clinical and personal factors matter most.

    Shuffles ONE input field at a time in the held-out test set and measures
    how much the score drops. Unlike the per-encoded-column importances
    above, this gives one number per field the clinician actually enters
    (all of a field's one-hot columns move together), and it is measured on
    unseen patients rather than read off the model's internals."""
    rng = np.random.default_rng(RANDOM_STATE)

    def score(X):
        proba = model.predict_proba(preprocessor.transform(X))
        if layer == 1:
            return roc_auc_score(y_test, proba[:, 1])
        return roc_auc_score(y_test, proba, multi_class="ovr")

    baseline = score(X_test)
    rows = []
    for field in FEATURE_COLS:
        drops = []
        for _ in range(N_PERMUTATION_REPEATS):
            shuffled = X_test.copy()
            shuffled[field] = rng.permutation(shuffled[field].values)
            drops.append(baseline - score(shuffled))
        rows.append({"field": field, "auc_drop_mean": round(float(np.mean(drops)), 5),
                     "auc_drop_std": round(float(np.std(drops)), 5)})
    return {"metric": "roc_auc" if layer == 1 else "roc_auc_ovr", "baseline": round(float(baseline), 4),
            "repeats": N_PERMUTATION_REPEATS, "fields": sorted(rows, key=lambda r: -r["auc_drop_mean"])}


def roc_points(y_true, y_proba, n_points=101):
    """ROC curve resampled to a fixed grid so the report file stays small."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    grid = np.linspace(0, 1, n_points)
    return {"fpr": [round(float(v), 4) for v in grid],
            "tpr": [round(float(v), 4) for v in np.interp(grid, fpr, tpr)]}


# Confidence level shown with each prediction (proposal Fig. 3.13). It is
# defined by how far the estimated probability is from 50%, and each level
# is reported with how often predictions at that level were actually right
# on the held-out test set - so "High" is a measured claim, not a label.
# api/main.py reads these same boundaries from evaluation_report.json.
CONFIDENCE_BANDS = [("Low", 0.0), ("Moderate", 0.15), ("High", 0.30)]  # (level, min distance from 0.5)


def confidence_report(y_true, y_proba):
    y_true, y_proba = np.asarray(y_true), np.asarray(y_proba)
    distance = np.abs(y_proba - 0.5)
    correct = (y_proba >= 0.5).astype(int) == y_true
    bands = []
    for i, (level, lower) in enumerate(CONFIDENCE_BANDS):
        upper = CONFIDENCE_BANDS[i + 1][1] if i + 1 < len(CONFIDENCE_BANDS) else 0.5 + 1e-9
        mask = (distance >= lower) & (distance < upper)
        bands.append({"level": level, "min_distance": lower,
                      "n_test": int(mask.sum()), "share_of_test": round(float(mask.mean()), 4),
                      "accuracy": round(float(correct[mask].mean()), 4)})
    return bands


def calibration_report(y_true, y_proba):
    observed, predicted = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    return {"brier_score": round(float(brier_score_loss(y_true, y_proba)), 4),
            "curve": [{"mean_predicted": round(float(p), 4), "observed_rate": round(float(o), 4)}
                      for p, o in zip(predicted, observed)]}


# ----------------------------------------------------------------------------
# Train + evaluate one layer
# ----------------------------------------------------------------------------
def load_layer(layer):
    target = "rtw_status" if layer == 1 else "rtw_timing"
    train = pd.read_csv(str(PROCESSED_DIR / f"layer{layer}_train.csv"))
    test = pd.read_csv(str(PROCESSED_DIR / f"layer{layer}_test.csv"))
    X_train = cast_categoricals(train.drop(columns=["UniID", target]))[FEATURE_COLS]
    X_test = cast_categoricals(test.drop(columns=["UniID", target]))[FEATURE_COLS]
    return X_train, train[target], X_test, test[target]


def train_layer(layer, quick=False):
    X_train, y_train_raw, X_test, y_test_raw = load_layer(layer)

    label_encoder = None
    if layer == 1:
        y_train, y_test = y_train_raw.to_numpy(), y_test_raw.to_numpy()
        class_labels = ["Non-RTW", "RTW"]
    else:
        label_encoder = LabelEncoder()
        label_encoder.fit(TIMING_ORDER)
        y_train = label_encoder.transform(y_train_raw)
        y_test = label_encoder.transform(y_test_raw)
        class_labels = list(label_encoder.classes_)
    classes = list(range(len(class_labels)))

    counts = np.bincount(y_test, minlength=len(classes))
    majority = int(np.argmax(np.bincount(y_train, minlength=len(classes))))
    report = {
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "class_labels": class_labels,
        "test_class_counts": {class_labels[i]: int(counts[i]) for i in classes},
        # What you would score by always predicting the commonest training
        # class - the number every model's accuracy has to be read against.
        "majority_baseline": {"always_predicts": class_labels[majority],
                              "accuracy": round(float((y_test == majority).mean()), 4)},
        "tuning": {}, "cross_validation": {}, "train_vs_test": {}, "test_ci": {},
        "tuning_metric": tuning_metric(layer),
    }
    if layer == 1:
        report["calibration"], report["roc_curves"] = {}, {}
    else:
        report["per_class"] = {}

    # The deployed preprocessor is fit on the full training split.
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)

    results, fitted = {}, {}
    for name in MODEL_NAMES:
        model = base_models(layer, y_train)[name]
        print(f"\n--- {name} (Layer {layer}) ---")

        # 1-2. Tuning + cross-validation, training split only.
        default_cv = cross_validate_model(layer, name, model, X_train, y_train)
        if quick:
            best_params, tuned_cv = {}, default_cv
        else:
            best_params, best_score = tune(layer, name, model, X_train, y_train)
            default_score = default_cv[tuning_metric(layer)]["mean"]
            if best_score > default_score:
                model.set_params(**best_params)
                tuned_cv = cross_validate_model(layer, name, model, X_train, y_train)
            else:  # the search found nothing better than the starting point
                best_params, tuned_cv = {}, default_cv
        report["tuning"][name] = {"best_params": best_params, "default_cv": default_cv,
                                  "kept_defaults": not best_params}
        report["cross_validation"][name] = tuned_cv
        print("  tuned params:", best_params or "(defaults kept)")
        print("  5-fold CV   :", json.dumps(tuned_cv))

        # 3. Final fit on the full training split.
        model.set_params(n_jobs=-1) if name != "logistic_regression" else None
        extra = {k.replace("model__", ""): v for k, v in fit_params(layer, name, y_train).items()}
        model.fit(X_train_t, y_train, **extra)
        fitted[name] = model

        # 4. Held-out test evaluation.
        y_pred, y_proba = model.predict(X_test_t), model.predict_proba(X_test_t)
        if layer == 1:
            metrics = binary_metrics(y_test, y_pred, y_proba[:, 1])
            report["calibration"][name] = calibration_report(y_test, y_proba[:, 1])
            report["roc_curves"][name] = roc_points(y_test, y_proba[:, 1])
            report["test_ci"][name] = bootstrap_ci(y_test, y_pred, y_proba[:, 1], layer)
            if name == "xgboost":  # the deployed model
                report["confidence_bands"] = confidence_report(y_test, y_proba[:, 1])
        else:
            metrics = multiclass_metrics(y_test, y_pred, y_proba, classes)
            metrics["class_labels"] = class_labels
            p, r, f, s = precision_recall_fscore_support(y_test, y_pred, labels=classes, zero_division=0)
            report["per_class"][name] = {
                class_labels[i]: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                                  "f1": round(float(f[i]), 4), "support": int(s[i])} for i in classes}
            report["test_ci"][name] = bootstrap_ci(y_test, y_pred, y_proba, layer)
        print("  test        :", json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}))

        # 5. Overfitting check.
        train_acc = float(accuracy_score(y_train, model.predict(X_train_t)))
        report["train_vs_test"][name] = {"train_accuracy": round(train_acc, 4),
                                         "test_accuracy": metrics["accuracy"],
                                         "gap": round(train_acc - metrics["accuracy"], 4)}

        results[name] = {"metrics": metrics,
                         "feature_importance": feature_importance_from_model(model, feature_names, name)}
        if name == "xgboost":
            # Native XGBoost format - portable across XGBoost versions/machines.
            # Pickling the sklearn wrapper (joblib.dump) is NOT reliably
            # cross-version compatible and can raise "input stream corrupted".
            model.get_booster().save_model(str(MODEL_DIR / f"layer{layer}_xgboost.json"))
        else:
            joblib.dump(model, str(MODEL_DIR / f"layer{layer}_{name}.joblib"))

    # 6. Objective (a): factor importance for the deployed (XGBoost) model.
    report["factor_importance"] = permutation_importance_by_field(
        fitted["xgboost"], preprocessor, X_test, y_test, layer)

    if layer == 2:
        # Record what the old, unweighted Layer 2 XGBoost scores on the same
        # test set, so the report can show exactly what weighting changed.
        unweighted = base_models(layer, y_train)["xgboost"].set_params(n_jobs=-1)
        unweighted.fit(X_train_t, y_train)
        up, uproba = unweighted.predict(X_test_t), unweighted.predict_proba(X_test_t)
        um = multiclass_metrics(y_test, up, uproba, classes)
        _, ur, _, _ = precision_recall_fscore_support(y_test, up, labels=classes, zero_division=0)
        report["xgboost_unweighted_reference"] = {
            "accuracy": um["accuracy"], "f1_macro": um["f1_macro"], "recall_macro": um["recall_macro"],
            "recall_per_class": {class_labels[i]: round(float(ur[i]), 4) for i in classes}}
        joblib.dump(label_encoder, str(MODEL_DIR / "layer2_label_encoder.joblib"))

    joblib.dump(preprocessor, str(MODEL_DIR / f"layer{layer}_preprocessor.joblib"))
    with open(str(MODEL_DIR / f"layer{layer}_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="skip hyperparameter tuning")
    args = parser.parse_args()

    evaluation = {
        "settings": {
            "random_state": RANDOM_STATE, "cv_folds": CV_FOLDS, "tuned": not args.quick,
            "bootstrap_resamples": N_BOOTSTRAP,
            "python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    for layer, title in ((1, "LAYER 1 MODELS (RTW vs Non-RTW)"), (2, "LAYER 2 MODELS (RTW Timing)")):
        print("\n" + "=" * 60 + f"\nTRAINING {title}\n" + "=" * 60)
        evaluation[f"layer{layer}"] = train_layer(layer, quick=args.quick)

    with open(str(MODEL_DIR / "evaluation_report.json"), "w") as f:
        json.dump(evaluation, f, indent=2)
    print(f"\nAll models and evaluation_report.json saved to {MODEL_DIR}")
