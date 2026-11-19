"""
SCI-RTW Predictor: SHAP explanation of the deployed Layer 1 model (Chapter 4)

Computes SHAP values for the deployed Layer 1 XGBoost model on the held-out
test patients. XGBoost's own TreeSHAP (predict(..., pred_contribs=True)) is
used, which gives exact SHAP values for tree models and is the same method
the app uses for each patient's "key factors". The one-hot columns of each
input field are summed back onto that field, so the 14 fields match the rest
of the report. Values are in log-odds of returning to work: positive pushes
the prediction towards RTW, negative away from it.

Nothing is retrained. Run:  python src/explain_shap.py   (a few seconds)
Writes docs/shap_results.json, docs/figures/layer1_shap_summary.png and
docs/figures/layer1_shap_example.png.
"""

import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

import train_models as tm
from make_figures import (BLUE, FIELD_LABELS, INK, INK_MUTED, INK_SECONDARY, MODEL_DIR, plt,
                          save)

OUTPUT_PATH = tm.PROJECT_ROOT / "docs" / "shap_results.json"
ORANGE = "#eb6834"
NUMERIC_FIELDS = ["age_at_injury", "asia_motor_total_discharge"]
SEED = 42

# Invented patient (not from the NSCISC data); the same one the API tests use.
EXAMPLE_PATIENT = {
    "sex": "2.0", "age_at_injury": 34, "education_level": "5.0", "marital_status": "2.0",
    "veteran_status": "0.0", "pre_injury_employment_status": "1.0", "injury_cause": "30.0",
    "vertebral_injury_level": "1.0", "associated_injury": "0.0", "work_related_injury": "0.0",
    "neuro_level_discharge": "L02", "aisgrade_discharge": "D",
    "injury_completeness_discharge": "1.0", "asia_motor_total_discharge": 90,
}
EXAMPLE_TEXT = ("Invented example: female, 34, bachelor's degree, married, working before injury, "
                "fall, L2, AIS D, motor score 90")


def field_of(feature_name, fields):
    """Map an encoded column (e.g. 'cat__education_level_5.0') to its input field."""
    prefix, name = feature_name.split("__", 1)
    if prefix == "num":
        return name
    for f in sorted(fields, key=len, reverse=True):
        if name.startswith(f + "_"):
            return f
    return name


def grouped_contributions(booster, pre, X):
    """Per-patient SHAP values summed by input field, plus the base value."""
    contribs = booster.predict(xgb.DMatrix(pre.transform(X)), pred_contribs=True)
    names = list(pre.get_feature_names_out())
    fields = list(X.columns)
    out = pd.DataFrame(0.0, index=range(len(X)), columns=fields)
    for j, name in enumerate(names):
        out[field_of(name, fields)] += contribs[:, j]
    return out, contribs[:, -1]


def summary_figure(shap_by_field, X, order):
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(order) + 1.6))
    cmap = plt.get_cmap("coolwarm")
    for y, field in enumerate(order[::-1]):
        vals = shap_by_field[field].to_numpy()
        jitter = rng.uniform(-0.28, 0.28, len(vals))
        if field in NUMERIC_FIELDS:
            v = X[field].to_numpy(dtype=float)
            lo, hi = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
            colours = cmap(np.clip((v - lo) / (hi - lo), 0, 1))
        else:
            colours = "#9db7d8"
        ax.scatter(vals, y + jitter, s=3, c=colours, alpha=0.55, linewidths=0, rasterized=True)
    ax.axvline(0, color=INK_MUTED, linewidth=0.9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([FIELD_LABELS[f] for f in order[::-1]])
    ax.set_xlabel("SHAP value (change in log-odds of returning to work)")
    ax.set_title("Layer 1 (XGBoost): SHAP values for each test patient")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.annotate("Each dot is one test patient. Right of 0 = pushes towards RTW.\n"
            "Age and motor score: blue = low value, red = high value. Other fields are categories.",
                xy=(1, 0), xycoords="axes fraction", xytext=(0, -38), textcoords="offset points",
                ha="right", va="top", fontsize=8, color=INK_SECONDARY)
    save(fig, "layer1_shap_summary.png")


def example_figure(contrib, base, prob):
    items = sorted(contrib.items(), key=lambda kv: abs(kv[1]))
    labels = [FIELD_LABELS[f] for f, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7.4, 0.36 * len(items) + 1.6))
    ax.barh(labels, vals, color=[BLUE if v > 0 else ORANGE for v in vals], height=0.62)
    span = max(abs(v) for v in vals)
    for y, v in enumerate(vals):
        ax.text(v + (span * 0.02 if v >= 0 else -span * 0.02), y, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5, color=INK_SECONDARY)
    ax.set_xlim(-span * 1.25, span * 1.25)
    ax.axvline(0, color=INK_MUTED, linewidth=0.9)
    ax.set_xlabel("SHAP value (change in log-odds of returning to work)")
    ax.set_title("Layer 1: why the model predicts RTW for one patient")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.annotate(f"{EXAMPLE_TEXT}.\nAverage starting point {base:+.2f} log-odds; sum of all bars "
            f"{sum(vals):+.2f}; final {base + sum(vals):+.2f} log-odds = {prob:.0%} chance of RTW.\n"
            "Blue raises the estimate, orange lowers it.",
                xy=(0, 0), xycoords="axes fraction", xytext=(0, -38), textcoords="offset points",
                ha="left", va="top", fontsize=8, color=INK)
    save(fig, "layer1_shap_example.png")


def main():
    _, _, X_test, _ = tm.load_layer(1)
    pre = joblib.load(str(MODEL_DIR / "layer1_preprocessor.joblib"))
    booster = xgb.Booster()
    booster.load_model(str(MODEL_DIR / "layer1_xgboost.json"))

    shap_by_field, base = grouped_contributions(booster, pre, X_test)
    # SHAP values plus the base value add up to the model's log-odds output.
    margin = booster.predict(xgb.DMatrix(pre.transform(X_test)), output_margin=True)
    max_gap = float(np.abs(shap_by_field.sum(axis=1).to_numpy() + base - margin).max())

    mean_abs = shap_by_field.abs().mean().sort_values(ascending=False)
    order = list(mean_abs.index)
    perm = json.load(open(MODEL_DIR / "evaluation_report.json"))["layer1"]["factor_importance"]["fields"]
    perm_rank = [f["field"] for f in perm]
    rho = spearmanr([order.index(f) for f in perm_rank], range(len(perm_rank))).correlation

    direction = {}
    for f in NUMERIC_FIELDS:
        r = spearmanr(X_test[f].astype(float), shap_by_field[f], nan_policy="omit").correlation
        direction[f] = round(float(r), 3)
    # Age is not one-directional, so it is summarised by band as well.
    bands = {}
    for field, edges, names in [("age_at_injury", [0, 25, 45, 60, 200], ["under 25", "25-44", "45-59", "60+"]),
                                ("asia_motor_total_discharge", [-1, 25, 50, 75, 101],
                                 ["0-25", "26-50", "51-75", "76-100"])]:
        cut = pd.cut(X_test[field].astype(float), edges, labels=names, right=False if field == "age_at_injury" else True)
        g = shap_by_field[field].groupby(cut, observed=False)
        bands[field] = {str(k): {"n": int(len(v)), "mean_shap": round(float(v.mean()), 3)} for k, v in g}

    example = pd.DataFrame([EXAMPLE_PATIENT])
    example = tm.cast_categoricals(example)[list(X_test.columns)]
    ex_by_field, ex_base = grouped_contributions(booster, pre, example)
    ex_contrib = {f: float(ex_by_field.loc[0, f]) for f in X_test.columns}
    ex_prob = float(booster.predict(xgb.DMatrix(pre.transform(example)))[0])

    summary_figure(shap_by_field, X_test, order)
    example_figure(ex_contrib, float(ex_base[0]), ex_prob)

    results = {
        "model": "Layer 1 XGBoost (deployed)",
        "method": "XGBoost TreeSHAP (pred_contribs), one-hot columns summed by input field",
        "units": "log-odds of RTW",
        "n_test_patients": int(len(X_test)),
        "base_value_log_odds": round(float(base[0]), 4),
        "additivity_max_error": max_gap,
        "mean_abs_shap_by_field": {f: round(float(v), 4) for f, v in mean_abs.items()},
        "rank_shap": order,
        "rank_permutation_importance": perm_rank,
        "spearman_rank_correlation_shap_vs_permutation": round(float(rho), 3),
        "numeric_value_vs_shap_spearman": direction,
        "mean_shap_by_band": bands,
        "example_patient": {"description": EXAMPLE_TEXT, "predicted_rtw_probability": round(ex_prob, 4),
                            "base_value_log_odds": round(float(ex_base[0]), 4),
                            "shap_by_field": {f: round(v, 4) for f, v in
                                              sorted(ex_contrib.items(), key=lambda kv: -abs(kv[1]))}},
    }
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print("saved", OUTPUT_PATH.relative_to(tm.PROJECT_ROOT))
    print(json.dumps({k: results[k] for k in ["rank_shap", "spearman_rank_correlation_shap_vs_permutation",
                                               "mean_shap_by_band", "additivity_max_error"]},
                     indent=1))


if __name__ == "__main__":
    main()
