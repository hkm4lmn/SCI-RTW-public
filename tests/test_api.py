"""
SCI-RTW Predictor - automated API tests.
Maps to Table 3.18 (Test Case Generation) in the proposal.

Run from the project root:
    python -m pytest tests/test_api.py -v
"""
import math
import os
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "api"))

import main as m  # noqa: E402
from main import app  # noqa: E402
from conftest import make_user  # noqa: E402

# /predict and /model-info now require a logged-in clinician (FR1), so every
# request in this file is sent as a doctor. Role rules themselves are tested
# in tests/test_auth_records.py.
client = TestClient(app)
client.headers.update(make_user(client, "dr_api_tests", "doctor"))

# sex/education/marital/injury_cause/employment all carry ".0" suffixes (see
# labelMaps.js header for why); aisgrade_discharge and neuro_level_discharge
# do not, since those source columns were never purely numeric.
#
# Both fixtures below are REAL patients pulled from the held-out test set
# (not hand-picked guesses) whose model prediction was checked to be both
# correct and confident (pred_proba 0.28 / 0.80 on the current model) before being hardcoded here
# -- an earlier invented "older/severe" profile turned out to sit just above
# the 0.5 threshold on the real model and made TC05 flaky. See
# docs/test_report.md for why a plausible-sounding profile isn't a safe
# substitute for checking an actual model output.
RTW_LIKELY = {  # confidently-correct RTW case, pred_proba=0.797
    "sex": "1.0", "age_at_injury": 18, "education_level": "3.0", "marital_status": "1.0",
    "veteran_status": "0.0", "pre_injury_employment_status": "1.0", "injury_cause": "1.0",
    "vertebral_injury_level": "1.0", "associated_injury": "0.0", "work_related_injury": "0.0",
    "neuro_level_discharge": "L03", "aisgrade_discharge": "D",
    "injury_completeness_discharge": "1.0", "asia_motor_total_discharge": 92,
}
RTW_UNLIKELY = {  # confidently-correct Non-RTW case, pred_proba=0.280
    "sex": "1.0", "age_at_injury": 26, "education_level": "1.0", "marital_status": "1.0",
    "veteran_status": "0.0", "pre_injury_employment_status": "1.0", "injury_cause": "1.0",
    "vertebral_injury_level": "1.0", "associated_injury": "0.0", "work_related_injury": "0.0",
    "neuro_level_discharge": "C04", "aisgrade_discharge": "D",
    "injury_completeness_discharge": "4.0", "asia_motor_total_discharge": 46,
}
REQUIRED_FIELDS = ["sex", "age_at_injury", "education_level", "marital_status",
                   "pre_injury_employment_status", "injury_cause", "neuro_level_discharge",
                   "aisgrade_discharge", "injury_completeness_discharge"]
# processed/ holds patient-level rows derived from the NSCISC data, so it is
# not in the public copy of this repository; tests that read it are skipped
# there. Run src/data_pipeline.py on the raw NSCISC files to create it.
PROCESSED_DIR = os.path.join(ROOT, "processed")
needs_processed_data = pytest.mark.skipif(
    not os.path.exists(os.path.join(PROCESSED_DIR, "layer1_test.csv")),
    reason="processed/ not present: run src/data_pipeline.py on the NSCISC files")

OPTIONAL_FIELDS = ["veteran_status", "vertebral_injury_level", "associated_injury",
                   "work_related_injury", "asia_motor_total_discharge"]


# ---------------- TC02: enter patient data ----------------
def test_tc02_valid_patient_data_accepted():
    assert client.post("/predict", json=RTW_LIKELY).status_code == 200


# ---------------- TC03: missing data validation ----------------
@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_tc03_missing_required_field_rejected(field):
    payload = {k: v for k, v in RTW_LIKELY.items() if k != field}
    assert client.post("/predict", json=payload).status_code == 422, f"missing '{field}' should be rejected"


@pytest.mark.parametrize("field", OPTIONAL_FIELDS)
def test_tc03_optional_fields_may_be_omitted(field):
    payload = {k: v for k, v in RTW_LIKELY.items() if k != field}
    assert client.post("/predict", json=payload).status_code == 200


def test_tc03_optional_fields_may_be_explicit_null():
    payload = dict(RTW_LIKELY, veteran_status=None, vertebral_injury_level=None,
                   associated_injury=None, work_related_injury=None, asia_motor_total_discharge=None)
    assert client.post("/predict", json=payload).status_code == 200


# ---------------- TC04: generate RTW prediction ----------------
def test_tc04_prediction_shape_and_ranges():
    body = client.post("/predict", json=RTW_LIKELY).json()
    assert body["rtw_status"] in ("RTW", "Non-RTW")
    assert 0.0 <= body["rtw_probability"] <= 1.0
    assert len(body["top_factors"]) == 5


def test_tc04_status_consistent_with_probability():
    for p in (RTW_LIKELY, RTW_UNLIKELY):
        body = client.post("/predict", json=p).json()
        assert (body["rtw_status"] == "RTW") == (body["rtw_probability"] >= 0.5)


def test_tc04_contrasting_profiles_rank_sensibly():
    a = client.post("/predict", json=RTW_LIKELY).json()["rtw_probability"]
    b = client.post("/predict", json=RTW_UNLIKELY).json()["rtw_probability"]
    assert a > b, "young/mild profile should score higher than older/severe"


def test_tc04_deterministic():
    a = client.post("/predict", json=RTW_LIKELY).json()
    b = client.post("/predict", json=RTW_LIKELY).json()
    assert a == b


# ---------------- TC05: generate RTW timing ----------------
def test_tc05_timing_present_only_when_rtw():
    body = client.post("/predict", json=RTW_LIKELY).json()
    assert body["rtw_status"] == "RTW"
    assert body["rtw_timing"] in ("Year 1", "Year 5", "By Year 10")
    probs = body["rtw_timing_probabilities"]
    assert set(probs) == {"Year 1", "Year 5", "By Year 10"}
    assert abs(sum(probs.values()) - 1.0) < 0.01

    body = client.post("/predict", json=RTW_UNLIKELY).json()
    assert body["rtw_status"] == "Non-RTW"
    assert body["rtw_timing"] is None
    assert body["rtw_timing_probabilities"] is None


# ---------------- TC07: model accuracy report ----------------
def test_tc07_model_info_reports_all_models_and_metrics():
    info = client.get("/model-info").json()
    for layer in ("layer1", "layer2"):
        models = info[layer]["models_compared"]
        assert set(models) == {"xgboost", "random_forest", "logistic_regression"}
        for mm in models.values():
            assert 0 < mm["accuracy"] <= 1
            assert "confusion_matrix" in mm
        assert len(info[layer]["top_features"]) > 0


def test_tc07_model_info_reports_validation_evidence():
    """Proposal 3.6.1: cross-validation, an overfitting check and the
    majority-class reference point must be reported alongside test accuracy."""
    info = client.get("/model-info").json()
    for layer in ("layer1", "layer2"):
        ev = info[layer]["evaluation"]
        assert ev is not None, "run src/train_models.py to generate models/evaluation_report.json"
        assert 0 < ev["majority_baseline"]["accuracy"] < 1
        for name, mm in info[layer]["models_compared"].items():
            cv = ev["cross_validation"][name]["accuracy"]
            assert 0 < cv["mean"] <= 1 and cv["std"] >= 0
            tv = ev["train_vs_test"][name]
            # The overfitting check must describe the same model the metrics table does.
            assert tv["test_accuracy"] == mm["accuracy"]
            assert abs(tv["gap"] - (tv["train_accuracy"] - tv["test_accuracy"])) < 1e-3
            lo, hi = ev["test_ci"][name]["accuracy"]
            assert lo <= mm["accuracy"] <= hi
        fields = [f["field"] for f in ev["factor_importance"]["fields"]]
        assert sorted(fields) == sorted(m.NUMERIC_COLS + m.CATEGORICAL_COLS)
        drops = [f["auc_drop_mean"] for f in ev["factor_importance"]["fields"]]
        assert drops == sorted(drops, reverse=True)
    assert info["layer1"]["evaluation"]["test_ci"]["xgboost"]["accuracy"][0] > \
        info["layer1"]["evaluation"]["majority_baseline"]["accuracy"], \
        "Layer 1 should beat the majority-class baseline beyond sampling noise"


def test_health_and_category_options():
    assert client.get("/health").json()["status"] == "ok"
    opts = client.get("/category-options").json()
    assert len(opts) == 12
    assert all(len(v) > 0 for v in opts.values())


def test_dropped_fields_are_gone():
    """race, hispanic_origin, pre_injury_job_class, depression_severity_phq_total
    were removed after testing (era-proxy / no accuracy benefit) -- confirm
    they don't silently linger anywhere a demo or marker could notice."""
    opts = client.get("/category-options").json()
    for f in ("race", "hispanic_origin", "pre_injury_job_class", "depression_severity_phq_total"):
        assert f not in opts
    schema_fields = app.openapi()["components"]["schemas"]["PatientInput"]["properties"]
    for f in ("race", "hispanic_origin", "pre_injury_job_class", "depression_severity_phq_total"):
        assert f not in schema_fields


# ---------------- Mode-imputation (replaces old "Unknown"-category tests) ----------------
def test_omitted_optional_field_uses_population_mode():
    """Omitting an optional field must give the exact same prediction as
    explicitly sending that field's trained population mode -- this is the
    whole point of removing the separate 'Unknown' category (see
    data_pipeline.py / test_report.md Finding #2)."""
    base = dict(RTW_LIKELY)
    omitted = client.post("/predict", json=base).json()["rtw_probability"]
    with_modes = dict(base)
    for field, mode_val in m.IMPUTATION_MODES["categorical"].items():
        if field in OPTIONAL_FIELDS:
            with_modes[field] = mode_val
    explicit = client.post("/predict", json=with_modes).json()["rtw_probability"]
    assert omitted == explicit


@pytest.mark.parametrize("field,value", [
    ("injury_cause", "not-a-real-code"),   # required field
    ("sex", "female"),                     # a label instead of its code
    ("aisgrade_discharge", "d"),           # wrong case
    ("work_related_injury", "maybe"),      # optional field
])
def test_unrecognised_category_value_rejected(field, value):
    """A value the model was never trained on used to be swapped silently for
    the field's most common value, so a typo still returned a confident-looking
    prediction for a different patient. Only a blank is imputed now."""
    r = client.post("/predict", json=dict(RTW_LIKELY, **{field: value}))
    assert r.status_code == 422
    assert field in r.text


@pytest.mark.parametrize("field,value", [
    ("age_at_injury", -5), ("age_at_injury", 500),
    ("asia_motor_total_discharge", -1), ("asia_motor_total_discharge", 900),
])
def test_out_of_range_number_rejected(field, value):
    r = client.post("/predict", json=dict(RTW_LIKELY, **{field: value}))
    assert r.status_code == 422
    assert field in r.text


@pytest.mark.parametrize("field,value", [
    ("age_at_injury", 0), ("age_at_injury", 110),
    ("asia_motor_total_discharge", 0), ("asia_motor_total_discharge", 100),
])
def test_number_at_range_limit_accepted(field, value):
    assert client.post("/predict", json=dict(RTW_LIKELY, **{field: value})).status_code == 200


def test_no_unknown_or_sentinel_codes_in_category_options():
    """Every field's own unknown sentinel (per data_pipeline.UNKNOWN_CODES)
    should have been converted to NaN and mode-imputed away during
    preprocessing -- none should appear as a selectable category any more.
    Checked per-field, not as a blanket banned-value set: injury_cause's
    real code "9.0" means All-terrain vehicle, not unknown (its sentinel is
    "99.0"), so a generic "9"/"99" blocklist would false-positive on it."""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import data_pipeline as dp

    opts = client.get("/category-options").json()
    for field, codes in dp.UNKNOWN_CODES.items():
        if field not in opts:
            continue
        sentinel_strs = {f"{float(c):.1f}" if str(c).lstrip("-").isdigit() else str(c) for c in codes}
        bad = sentinel_strs & set(opts[field])
        assert not bad, f"{field} still has unconverted sentinel value(s): {bad}"
    assert "Unknown" not in [v for values in opts.values() for v in values]


def test_ais_grade_has_one_coding():
    """AIS grade used to carry the old numeric Frankel codes next to the AIS
    letters, which let the model tell older patients from newer ones. The
    pipeline maps them onto the letters, so only the letters are valid."""
    assert client.get("/category-options").json()["aisgrade_discharge"] == ["A", "B", "C", "D", "E"]
    assert m.IMPUTATION_MODES["categorical"]["aisgrade_discharge"] == "A"
    assert client.post("/predict", json=dict(RTW_LIKELY, aisgrade_discharge="5")).status_code == 422


@needs_processed_data
def test_training_data_has_one_ais_coding():
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "layer1_train.csv"), dtype=str)
    assert set(train["aisgrade_discharge"]) <= set("ABCDE")


# ---------------- Patient-specific explanations ----------------
def test_patient_factors_are_consistent_with_probability():
    for p in (RTW_LIKELY, RTW_UNLIKELY):
        body = client.post("/predict", json=p).json()
        assert 1 <= len(body["patient_factors"]) <= 6
        for f in body["patient_factors"]:
            assert (f["contribution"] > 0) == (f["effect"] == "increases")
        mags = [abs(f["contribution"]) for f in body["patient_factors"]]
        assert mags == sorted(mags, reverse=True)


def test_patient_factors_sum_to_model_margin():
    """sigmoid(bias + sum of ALL field contributions) must equal the probability."""
    import xgboost as xgb
    payload = m.PatientInput(**RTW_LIKELY)
    X = m.build_feature_row(payload)
    d = xgb.DMatrix(m.layer1_preprocessor.transform(X))
    contribs = m.layer1_model.predict(d, pred_contribs=True)[0]
    prob = float(m.layer1_model.predict(d)[0])
    assert abs(1 / (1 + math.exp(-contribs.sum())) - prob) < 1e-4


def test_patient_factors_differ_between_patients():
    a = client.post("/predict", json=RTW_LIKELY).json()["patient_factors"]
    b = client.post("/predict", json=RTW_UNLIKELY).json()["patient_factors"]
    assert a != b


# ---------------- Confidence level + suggested considerations (Fig. 3.13) ----------------
def profile(**changes):
    return client.post("/predict", json=dict(RTW_LIKELY, **changes)).json()


def test_confidence_level_follows_distance_from_fifty_percent():
    bands = m.CONFIDENCE_BANDS
    assert [b["level"] for b in bands] == ["Low", "Moderate", "High"]
    for p, expected in ((0.50, "Low"), (0.58, "Low"), (0.36, "Low"), (0.66, "Moderate"), (0.30, "Moderate"),
                        (0.80, "High"), (0.20, "High"), (0.99, "High"), (0.01, "High")):
        assert m.recommendations.confidence_for(p, bands)["level"] == expected, p
    for body in (profile(), client.post("/predict", json=RTW_UNLIKELY).json()):
        expected = m.recommendations.confidence_for(body["rtw_probability"], bands)
        assert body["confidence"] == expected


def test_confidence_levels_are_backed_by_measured_accuracy():
    """'High' must mean something: accuracy on held-out patients has to rise
    with the level, and each level must rest on a real number of patients."""
    bands = m.CONFIDENCE_BANDS
    accuracies = [b["accuracy"] for b in bands]
    assert accuracies == sorted(accuracies) and accuracies[-1] - accuracies[0] > 0.2
    assert all(b["n_test"] >= 100 for b in bands)
    assert sum(b["n_test"] for b in bands) == m.EVALUATION["layer1"]["n_test"]
    body = profile()
    assert f"{body['confidence']['test_accuracy']:.0%}" in body["confidence"]["basis"]


def test_recommendations_shape_and_cap():
    for body in (profile(), client.post("/predict", json=RTW_UNLIKELY).json()):
        recs = body["recommendations"]
        assert 1 <= len(recs) <= m.recommendations.MAX_SUGGESTIONS
        for r in recs:
            assert set(r) == {"title", "text", "reason"} and all(r.values())
        assert len({r["title"] for r in recs}) == len(recs)


def test_recommendations_follow_the_prediction_direction():
    likely = [r["title"] for r in profile()["recommendations"]]
    unlikely = [r["title"] for r in client.post("/predict", json=RTW_UNLIKELY).json()["recommendations"]]
    assert likely[0] == "Begin return-to-work planning early"
    assert unlikely[0] == "Consider early vocational rehabilitation referral"
    # Timing prompts only make sense when return to work is predicted.
    assert not any("first year" in t or "follow-up" in t for t in unlikely)


def test_recommendation_rules_use_only_factors_that_lowered_the_estimate():
    suggest = m.recommendations.suggest
    titles = lambda **kw: [r["title"] for r in suggest(**kw)]
    base = dict(rtw_status="Non-RTW", confidence_level="High", rtw_timing=None)

    assert titles(**base, contributions_by_field={"education_level": 0.9}) == \
        ["Consider early vocational rehabilitation referral"], "a factor that RAISED the estimate must not trigger advice"
    assert "Discuss education and skills training" in titles(**base, contributions_by_field={"education_level": -0.4})
    assert titles(**base, contributions_by_field={"education_level": -0.01}) == \
        ["Consider early vocational rehabilitation referral"], "negligible contributions are ignored"
    # Several injury fields give ONE accommodation prompt, not one each.
    injury = {"asia_motor_total_discharge": -0.3, "aisgrade_discharge": -0.2, "neuro_level_discharge": -0.1}
    assert titles(**base, contributions_by_field=injury).count("Plan workplace accommodations") == 1
    # Largest negative factor comes first among the specific prompts.
    ordered = titles(**base, contributions_by_field={"education_level": -0.1, "age_at_injury": -0.8})
    assert ordered.index("Discuss options suited to career stage") < ordered.index("Discuss education and skills training")
    # No field the model does not use, and no medical instruction, can appear.
    everything = suggest("RTW", "High", "Year 5", {f: -1.0 for f in m.NUMERIC_COLS + m.CATEGORICAL_COLS})
    assert len(everything) == m.recommendations.MAX_SUGGESTIONS


def test_low_confidence_is_stated_instead_of_a_direction():
    for status in ("RTW", "Non-RTW"):
        first = m.recommendations.suggest(status, "Low", None, {})[0]
        assert first["title"] == "Treat this estimate as uncertain"


def test_recommendations_never_give_medical_instructions():
    """Proposal 1.4 / 3.6.4: no treatment recommendations, diagnoses or
    prescriptions. Every prompt is phrased as something to consider."""
    seen = []
    for status in ("RTW", "Non-RTW"):
        for level in ("Low", "Moderate", "High"):
            for timing in (None, "Year 1", "Year 5", "By Year 10"):
                seen += m.recommendations.suggest(
                    status, level, timing, {f: -1.0 for f in m.NUMERIC_COLS + m.CATEGORICAL_COLS})
    for text in {f"{r['title']}. {r['text']}" for r in seen}:
        assert "onsider" in text or "Base planning" in text, text
        assert not any(w in text.lower() for w in ("prescribe", "medication", "dose", "diagnos", "must ", "surgery"))


# ---------------- Regression: API reproduces training-time accuracy ----------------
@needs_processed_data
def test_api_reproduces_reported_layer1_accuracy():
    """Push 400 held-out test patients through the real HTTP path and check
    accuracy is close to what train_models.py reported. Guards against
    train/serve skew (encoding, imputation, Booster loading)."""
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "layer1_test.csv"), dtype=str)
    df = df.sample(min(400, len(df)), random_state=7)
    cat_cols = [c for c in df.columns if c not in ("UniID", "rtw_status", "age_at_injury",
                                                     "asia_motor_total_discharge")]
    correct = 0
    for _, row in df.iterrows():
        payload = {c: row[c] for c in cat_cols}
        payload["age_at_injury"] = float(row["age_at_injury"])
        payload["asia_motor_total_discharge"] = float(row["asia_motor_total_discharge"])
        pred = client.post("/predict", json=payload).json()["rtw_status"]
        correct += int((pred == "RTW") == (row["rtw_status"] == "1"))
    acc = correct / len(df)
    print(f"\nAPI accuracy on {len(df)} held-out patients: {acc:.3f}")
    assert 0.60 <= acc <= 0.76, f"accuracy {acc:.3f} far from reported ~0.676"
