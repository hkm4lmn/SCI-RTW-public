"""
SCI-RTW Predictor: split ratio, number of folds, and model comparison (experiment)

The proposal (3.6.1, Table 3.16) leaves the train-test ratio and the number
of cross-validation folds open, to be settled by experiment, and says the
most appropriate of the three models will be chosen (3.5.4). This script
runs those experiments. It changes nothing: the deployed models stay as
train_models.py left them. It prints the comparison and writes
docs/split_experiment_results.json.

Three parts, per layer:
  A. Split ratio   - 60/40, 70/30, 80/20 and 90/10, each repeated with
                     several different random splits, so that a difference
                     between two ratios can be read against how much the
                     score moves between two splits of the SAME ratio.
  B. Fold count    - 3-, 5- and 10-fold stratified cross-validation on the
                     deployed training split.
  C. Significance  - McNemar's exact test between each pair of models, and
                     between each model and the majority-class reference,
                     on the deployed held-out test split. Pairwise p-values
                     are Holm-adjusted for the three comparisons.

Every model uses the tuned settings train_models.py selected (read from
models/evaluation_report.json), so only the split or fold count varies.

A caution for the report: choosing the ratio that happens to give the
highest test score makes that score optimistic, because the test set then
helped choose it. Part A is there to show whether the result depends on
the ratio at all, not to pick a winner by the last decimal.

Run:  python src/experiment_splits.py              (about 10 minutes)
      python src/experiment_splits.py --repeats 2  (fast smoke test)
"""

import argparse
import json
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder

import train_models as tm

warnings.filterwarnings("ignore")

TEST_SIZES = [0.4, 0.3, 0.2, 0.1]
FOLD_COUNTS = [3, 5, 10]
DEFAULT_REPEATS = 10
OUTPUT_PATH = tm.PROJECT_ROOT / "docs" / "split_experiment_results.json"

with open(str(tm.MODEL_DIR / "evaluation_report.json")) as f:
    EVALUATION = json.load(f)


def second_metric(layer):
    return "roc_auc" if layer == 1 else "f1_macro"


def tuned_models(layer, y_train, n_jobs=-1):
    """The three models with the settings train_models.py selected."""
    models = tm.base_models(layer, y_train)
    for name, model in models.items():
        model.set_params(**EVALUATION[f"layer{layer}"]["tuning"][name]["best_params"])
        if name != "logistic_regression":
            model.set_params(n_jobs=n_jobs)
    return models


def encode(layer, y_raw):
    if layer == 1:
        return np.asarray(y_raw)
    return LabelEncoder().fit(tm.TIMING_ORDER).transform(y_raw)


def load_split(layer):
    """The deployed split, exactly as train_models.py uses it."""
    X_train, y_train, X_test, y_test = tm.load_layer(layer)
    return X_train, encode(layer, y_train), X_test, encode(layer, y_test)


def fit_and_predict(layer, name, model, X_train, y_train, X_test):
    pipeline = tm.make_pipeline(model)
    pipeline.fit(X_train, y_train, **tm.fit_params(layer, name, y_train))
    return pipeline, pipeline.predict(X_test), pipeline.predict_proba(X_test)


def score(layer, y_true, y_pred, y_proba):
    if layer == 1:
        return roc_auc_score(y_true, y_proba[:, 1])
    return f1_score(y_true, y_pred, average="macro")


def mean_std(values):
    return {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4)}


def show(stat, percent=True):
    if percent:
        return f"{stat['mean']:.1%} ± {stat['std'] * 100:.1f}"
    return f"{stat['mean']:.3f} ± {stat['std']:.3f}"


# ----------------------------------------------------------------------------
# A. Split ratio
# ----------------------------------------------------------------------------
def split_ratio_experiment(layer, repeats):
    X_train, y_train, X_test, y_test = load_split(layer)
    X = pd.concat([X_train, X_test], ignore_index=True)
    y = np.concatenate([y_train, y_test])
    metric = second_metric(layer)
    print(f"\nA. Split ratio - {len(y):,} patients, {repeats} random splits per ratio")
    print(f"   {'train/test':11} {'model':20} {'test accuracy':16} {metric:16} {'train-test gap':16} majority")

    rows = []
    for test_size in TEST_SIZES:
        collected = {name: {"accuracy": [], metric: [], "gap": []} for name in tm.MODEL_NAMES}
        majority = []
        for seed in range(repeats):
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y,
                                                      random_state=seed)
            majority.append(float((y_te == np.argmax(np.bincount(y_tr))).mean()))
            for name, model in tuned_models(layer, y_tr).items():
                pipeline, y_pred, y_proba = fit_and_predict(layer, name, model, X_tr, y_tr, X_te)
                accuracy = accuracy_score(y_te, y_pred)
                collected[name]["accuracy"].append(accuracy)
                collected[name][metric].append(score(layer, y_te, y_pred, y_proba))
                collected[name]["gap"].append(accuracy_score(y_tr, pipeline.predict(X_tr)) - accuracy)
        ratio = f"{round((1 - test_size) * 100)}/{round(test_size * 100)}"
        for name in tm.MODEL_NAMES:
            row = {"ratio": ratio, "model": name, "n_test": int(round(len(y) * test_size)),
                   "majority_accuracy": mean_std(majority),
                   **{k: mean_std(v) for k, v in collected[name].items()}}
            rows.append(row)
            print(f"   {ratio:11} {name:20} {show(row['accuracy']):16} {show(row[metric], False):16} "
                  f"{show(row['gap']):16} {row['majority_accuracy']['mean']:.1%}")
    return rows


# ----------------------------------------------------------------------------
# B. Number of folds
# ----------------------------------------------------------------------------
def fold_count_experiment(layer):
    X_train, y_train, _, _ = load_split(layer)
    metric = second_metric(layer)
    print(f"\nB. Number of folds - cross-validation on the {len(y_train):,} training patients")
    print(f"   {'folds':6} {'model':20} {'accuracy':16} {metric}")

    rows = []
    for k in FOLD_COUNTS:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=tm.RANDOM_STATE)
        for name, model in tuned_models(layer, y_train, n_jobs=1).items():
            scores = cross_validate(tm.make_pipeline(model), X_train, y_train, cv=cv,
                                    scoring=tm.cv_scoring(layer),
                                    params=tm.fit_params(layer, name, y_train), n_jobs=-1)
            row = {"folds": k, "model": name, "accuracy": mean_std(scores["test_accuracy"]),
                   metric: mean_std(scores[f"test_{metric}"])}
            rows.append(row)
            print(f"   {k:<6} {name:20} {show(row['accuracy']):16} {show(row[metric], False)}")
    return rows


# ----------------------------------------------------------------------------
# C. Significance
# ----------------------------------------------------------------------------
def mcnemar(correct_a, correct_b):
    """Exact McNemar test on the patients the two sets of predictions
    disagree about: if neither is better, each should win about half."""
    only_a = int((correct_a & ~correct_b).sum())
    only_b = int((~correct_a & correct_b).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return only_a, only_b, float(p)


def holm(p_values):
    """Holm step-down adjustment for several comparisons made together."""
    order = np.argsort(p_values)
    adjusted, running = np.empty(len(p_values)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def significance_experiment(layer):
    X_train, y_train, X_test, y_test = load_split(layer)
    majority = int(np.argmax(np.bincount(y_train)))
    correct = {"majority_class": y_test == majority}
    accuracy = {"majority_class": float((y_test == majority).mean())}
    for name, model in tuned_models(layer, y_train).items():
        _, y_pred, _ = fit_and_predict(layer, name, model, X_train, y_train, X_test)
        correct[name] = y_pred == y_test
        accuracy[name] = float(accuracy_score(y_test, y_pred))

    print(f"\nC. McNemar's exact test - {len(y_test):,} held-out test patients")
    print("   test accuracy: " + ", ".join(f"{k} {v:.1%}" for k, v in accuracy.items()))
    print(f"   {'A':20} {'B':20} {'only A right':13} {'only B right':13} {'p':10} verdict")

    pairs = list(combinations(tm.MODEL_NAMES, 2))
    tests = [mcnemar(correct[a], correct[b]) for a, b in pairs]
    adjusted = holm([t[2] for t in tests])
    rows = []
    for (a, b), (only_a, only_b, p), p_adj in zip(pairs, tests, adjusted):
        rows.append({"a": a, "b": b, "only_a_right": only_a, "only_b_right": only_b,
                     "p_value": p, "p_holm": p_adj, "significant": p_adj < 0.05})
    for name in tm.MODEL_NAMES:
        only_a, only_b, p = mcnemar(correct[name], correct["majority_class"])
        rows.append({"a": name, "b": "majority_class", "only_a_right": only_a, "only_b_right": only_b,
                     "p_value": p, "p_holm": None, "significant": p < 0.05})
    for row in rows:
        p = row["p_value"] if row["p_holm"] is None else row["p_holm"]
        verdict = "no significant difference"
        if row["significant"]:
            verdict = f"{row['a'] if row['only_a_right'] > row['only_b_right'] else row['b']} better"
        print(f"   {row['a']:20} {row['b']:20} {row['only_a_right']:<13} {row['only_b_right']:<13} "
              f"{p:<10.4g} {verdict}")
    return {"test_accuracy": {k: round(v, 4) for k, v in accuracy.items()}, "tests": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help="random splits per ratio in part A")
    args = parser.parse_args()

    results = {"settings": {"test_sizes": TEST_SIZES, "fold_counts": FOLD_COUNTS,
                            "repeats": args.repeats, "random_state": tm.RANDOM_STATE}}
    for layer, title in ((1, "LAYER 1 (RTW vs Non-RTW)"), (2, "LAYER 2 (RTW timing)")):
        print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60, flush=True)
        results[f"layer{layer}"] = {
            "significance": significance_experiment(layer),
            "fold_count": fold_count_experiment(layer),
            "split_ratio": split_ratio_experiment(layer, args.repeats),
        }

    with open(str(OUTPUT_PATH), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")
