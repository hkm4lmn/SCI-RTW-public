"""
SCI-RTW Predictor: can unused NSCISC variables add information? (experiment)

The learning curve (experiment_learning_curve.py) showed that more patients of
the same kind add little, so the limit is what the model is told about each
patient. This script tests seven Form I variables that the model does not use
and that are recorded at or before discharge (so they are known when a
clinician would use the tool):

  primary_payer          APrimPay   who pays for care (insurance type)
  residence_discharge    APResDis   where the patient went (home, nursing home...)
  spinal_surgery         ASpinSrg   spinal surgery done (added to NSCISC 10/2006)
  bladder_discharge      ABdMMDis   bladder management method at discharge
  ventilator_discharge   AUMVDis    mechanical ventilation at discharge
  days_injury_to_rehab   AI2RhADa   days from injury to rehab admission
  rehab_days             AHDaSyRb   days in the system's rehab unit

Left out on purpose: depression, anxiety, resilience and family income. The
data dictionary says these were collected only from 10/2016; such patients
cannot have a 10-year follow-up yet, so they are in the cohort only if they
returned to work early, and the variable would leak the outcome.

The trap this guards against (same as the FIM experiment, docs/test_report.md
6.4): the cohort's RTW rate depends strongly on injury era (37% for injuries
up to 1985, 99% from 2011, because recent patients are only included if they
have already returned to work). A variable whose values drift with era, such
as rehab length of stay, can look predictive only because it reveals the era.
So each variable is scored three ways, with 5-fold cross-validation on the
training split only (the test split is not touched):

  baseline        the deployed feature set
  + variable      the real values (unknowns filled with mode/median, no flag)
  + era-shuffled  the same values shuffled among patients injured in the same
                  5-year period: era information kept, patient information gone

A variable carries real patient information only if "+ variable" beats both
"baseline" and "+ era-shuffled" by more than the fold-to-fold noise. The
comparison is repeated on patients injured up to 2010 only, whose 10-year
outcome was observable whatever it was, so the selection effect is absent.

Run:  python src/experiment_extra_features.py   (about 10-15 minutes)
Writes docs/extra_features_results.json; changes nothing else.
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import data_pipeline as dp
import experiment_splits as es
import train_models as tm

warnings.filterwarnings("ignore")

CANDIDATES = {
    # name: (NSCISC column, kind, unknown / not-applicable codes)
    "primary_payer": ("APrimPay", "cat", [77, 99]),
    "residence_discharge": ("APResDis", "cat", [99]),
    "spinal_surgery": ("ASpinSrg", "cat", [9]),
    "bladder_discharge": ("ABdMMDis", "cat", [99]),
    "ventilator_discharge": ("AUMVDis", "cat", [9]),
    "days_injury_to_rehab": ("AI2RhADa", "num", [888, 999, 8888, 9999]),
    "rehab_days": ("AHDaSyRb", "num", [888, 999, 8888, 9999]),
}
DEPLOYED = {1: "xgboost", 2: "random_forest"}
LAST_FULL_FOLLOWUP_YEAR = 2010          # data run to 2021, so 10 years is observable
OUTPUT_PATH = tm.PROJECT_ROOT / "docs" / "extra_features_results.json"


def load_extra():
    cols = ["UniID", "AInjDt"] + [c for c, _, _ in CANDIDATES.values()]
    f1 = pd.read_csv(str(dp.find_data_dir() / "f1_public_2021.csv"), usecols=cols, low_memory=False)
    f1 = f1.set_index("UniID")
    out = pd.DataFrame(index=f1.index)
    out["injury_year"] = pd.to_numeric(f1["AInjDt"], errors="coerce")
    for name, (col, kind, unknown) in CANDIDATES.items():
        values = pd.to_numeric(f1[col], errors="coerce")
        out[name] = values.where(~values.isin(unknown))
    return out


def fill(series, kind):
    """Pipeline-style fill: no missing flag, so missingness cannot leak."""
    if kind == "num":
        return series.fillna(series.median())
    return series.fillna(series.mode().iloc[0]).astype(int).astype(str)


def era_shuffle(values, era, seed=0):
    rng = np.random.default_rng(seed)
    shuffled = np.array(values.to_numpy(), copy=True)
    for e in np.unique(era):
        idx = np.where(era == e)[0]
        shuffled[idx] = shuffled[rng.permutation(idx)]
    return pd.Series(shuffled, index=values.index)


def pipeline(model, extra_num, extra_cat):
    prep = ColumnTransformer([
        ("num", StandardScaler(), tm.NUMERIC_COLS + extra_num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         tm.CATEGORICAL_COLS + extra_cat),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def cv_score(layer, X, y, extra_num=(), extra_cat=()):
    name = DEPLOYED[layer]
    model = es.tuned_models(layer, y, n_jobs=1)[name]
    metric = es.second_metric(layer)
    scores = cross_validate(pipeline(model, list(extra_num), list(extra_cat)), X, y,
                            cv=tm.cv_splitter(), scoring=["accuracy", metric],
                            params=tm.fit_params(layer, name, y), n_jobs=-1)
    return {"accuracy": es.mean_std(scores["test_accuracy"]),
            metric: es.mean_std(scores[f"test_{metric}"])}


def run_cohort(layer, X, y, extra, label):
    metric = es.second_metric(layer)
    era = (extra["injury_year"] // 5 * 5).fillna(0).to_numpy()
    base = cv_score(layer, X, y)
    noise = base[metric]["std"]
    print(f"\n  {label}: {len(y):,} patients, RTW-class balance {np.bincount(y) / len(y)}")
    print(f"    {'variant':34} {'accuracy':>16} {metric:>16}   verdict")
    print(f"    {'baseline':34} {es.show(base['accuracy']):>16} {es.show(base[metric], False):>16}")
    rows = {"baseline": base}
    passed = []
    for name, (_, kind, _) in CANDIDATES.items():
        values = fill(extra[name], kind)
        recorded = float(extra[name].notna().mean())
        real, shuf = X.copy(), X.copy()
        real[name] = values.to_numpy()
        shuf[name] = era_shuffle(values, era).to_numpy()
        num, cat = ([name], []) if kind == "num" else ([], [name])
        with_var = cv_score(layer, real, y, num, cat)
        with_shuffled = cv_score(layer, shuf, y, num, cat)
        gain = with_var[metric]["mean"] - base[metric]["mean"]
        over_shuffle = with_var[metric]["mean"] - with_shuffled[metric]["mean"]
        ok = gain > noise and over_shuffle > noise
        if ok:
            passed.append(name)
        verdict = ("PASS" if ok else "no") + f"  (gain {gain:+.4f}, over era-shuffled {over_shuffle:+.4f}," \
                                             f" noise {noise:.4f}, recorded {recorded:.0%})"
        print(f"    {'+ ' + name:34} {es.show(with_var['accuracy']):>16} "
              f"{es.show(with_var[metric], False):>16}   {verdict}")
        print(f"    {'  era-shuffled':34} {es.show(with_shuffled['accuracy']):>16} "
              f"{es.show(with_shuffled[metric], False):>16}")
        rows[name] = {"with": with_var, "era_shuffled": with_shuffled, "recorded": round(recorded, 3),
                      "gain": round(gain, 4), "gain_over_shuffled": round(over_shuffle, 4), "passed": ok}
    if passed:
        combo = X.copy()
        num = [n for n in passed if CANDIDATES[n][1] == "num"]
        cat = [n for n in passed if CANDIDATES[n][1] == "cat"]
        for n in passed:
            combo[n] = fill(extra[n], CANDIDATES[n][1]).to_numpy()
        together = cv_score(layer, combo, y, num, cat)
        print(f"    {'+ all passing: ' + ', '.join(passed):34} {es.show(together['accuracy']):>16} "
              f"{es.show(together[metric], False):>16}")
        rows["all_passing"] = {"variables": passed, "scores": together}
    return rows


def run_layer(layer, extra_all):
    X, y, _, _ = es.load_split(layer)
    ids = pd.read_csv(str(tm.PROCESSED_DIR / f"layer{layer}_train.csv"), usecols=["UniID"])["UniID"]
    extra = extra_all.reindex(ids.to_numpy()).reset_index(drop=True)
    X = X.reset_index(drop=True)
    print(f"\n=== Layer {layer} ({DEPLOYED[layer]}), 5-fold CV on the training split")
    results = {"all": run_cohort(layer, X, y, extra, "All training patients")}
    early = (extra["injury_year"] <= LAST_FULL_FOLLOWUP_YEAR).to_numpy()
    results["injured_to_2010"] = run_cohort(layer, X[early].reset_index(drop=True), y[early],
                                            extra[early].reset_index(drop=True),
                                            f"Injured up to {LAST_FULL_FOLLOWUP_YEAR} only")
    return results


if __name__ == "__main__":
    extra_all = load_extra()
    results = {f"layer{layer}": run_layer(layer, extra_all) for layer in (1, 2)}
    with open(str(OUTPUT_PATH), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUTPUT_PATH}")
