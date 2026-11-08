"""
SCI-RTW Predictor: would more data help? (learning-curve experiment)

Retrains the deployed model of each layer (Layer 1 XGBoost, Layer 2 Random
Forest, with the tuned settings) on growing slices of the training split and
scores every model on the same held-out test split. If the test score is
still rising at 100%, more patients of the same kind should help; if it has
flattened, more of the same data will add little and better features are
needed instead. Training accuracy is shown too: a large, stable train-test gap
points to noise in the data rather than too few patients.

Changes nothing on disk except docs/learning_curve_results.json.

Run:  python src/experiment_learning_curve.py   (a few minutes)
"""

import json
import warnings

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import experiment_splits as es
import train_models as tm

warnings.filterwarnings("ignore")

FRACTIONS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
REPEATS = 5
DEPLOYED = {1: "xgboost", 2: "random_forest"}
OUTPUT_PATH = tm.PROJECT_ROOT / "docs" / "learning_curve_results.json"


def run_layer(layer):
    X_train, y_train, X_test, y_test = es.load_split(layer)
    name = DEPLOYED[layer]
    metric = es.second_metric(layer)
    print(f"\nLayer {layer} ({name}) - {len(y_train):,} training / {len(y_test):,} test patients")
    print(f"   {'share':>6} {'patients':>9} {'test accuracy':>16} {metric:>16} {'train accuracy':>16}")
    rows = []
    for frac in FRACTIONS:
        acc, sec, tr = [], [], []
        for seed in range(REPEATS if frac < 1.0 else 1):
            if frac < 1.0:
                X_s, _, y_s, _ = train_test_split(X_train, y_train, train_size=frac,
                                                  stratify=y_train, random_state=seed)
            else:
                X_s, y_s = X_train, y_train
            model = es.tuned_models(layer, y_s)[name]
            pipeline, y_pred, y_proba = es.fit_and_predict(layer, name, model, X_s, y_s, X_test)
            acc.append(accuracy_score(y_test, y_pred))
            sec.append(es.score(layer, y_test, y_pred, y_proba))
            tr.append(accuracy_score(y_s, pipeline.predict(X_s)))
        row = {"share": frac, "n_train": int(len(y_s)), "test_accuracy": es.mean_std(acc),
               metric: es.mean_std(sec), "train_accuracy": es.mean_std(tr)}
        rows.append(row)
        print(f"   {frac:>6.0%} {row['n_train']:>9,} {es.show(row['test_accuracy']):>16} "
              f"{es.show(row[metric], False):>16} {es.show(row['train_accuracy']):>16}")
    return {"model": name, "metric": metric, "rows": rows}


if __name__ == "__main__":
    results = {f"layer{layer}": run_layer(layer) for layer in (1, 2)}
    with open(str(OUTPUT_PATH), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUTPUT_PATH}")
