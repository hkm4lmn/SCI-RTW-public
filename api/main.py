"""
SCI-RTW Predictor: Backend API
Chapter 3, Section 3.4.1 (System Architecture) implemented as a real service.

Run locally:
    uvicorn main:app --reload --port 8000

Endpoints (role in brackets; see auth.py for the role rules):
    GET  /health                       - service + model status check          [public]
    GET  /category-options              - valid dropdown values for the form    [public]
    POST /auth/register, /auth/login    - FR1 register and login                [public]
    GET  /model-info                    - metrics + feature importance          [clinician, admin]
    POST /predict                       - run Layer 1 + Layer 2, not saved      [clinician]
    GET/POST/PUT/DELETE /patients       - FR2 manage patient profiles           [clinician]
    POST /patients/{id}/predictions     - run a prediction and save it          [clinician]
    GET  /predictions                   - FR7 prediction history (limit, offset) [clinician, patient]
    GET/POST/PATCH/DELETE /users        - manage user accounts                  [admin]
    GET  /audit-log                     - who did what and when                 [admin]

--------------------------------------------------------------------------
NOTE on missing/unknown values (see data_pipeline.py for the full story)
--------------------------------------------------------------------------
Earlier versions of this API let optional fields default to a literal
"Unknown" category, mirrored from the training data. Testing found that
"was this field even collected" was itself strongly correlated with how
long a patient had been followed (an artifact of NSCISC's intake forms
changing over the decades), so keeping "Unknown" as its own category let
the model use it as a shortcut for the label rather than real clinical
signal. The training pipeline now mode/median-imputes every missing value
instead of keeping an "Unknown" bucket - so an omitted field here is filled
with that field's population mode/median (IMPUTATION_MODES), exactly
mirroring what training did. There is no "Unknown" option to pass any more.
"""

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import pandas as pd
import numpy as np
import hashlib
import joblib
import json
import os
import sqlite3
import xgboost as xgb

import auth
import recommendations
from auth import current_user, require_clinician, require_staff
from db import CLINICIAN_ROLES, get_db, init_db, now_iso, record_audit

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

app = FastAPI(title="SCI-RTW Predictor API", version="1.2")

# Which other web addresses may call this API from a browser. By default
# only a frontend running on this same machine (the Vite dev server, on any
# port). The hosted build is served by this API itself, so it needs none.
# To allow others, list them in SCI_RTW_CORS_ORIGINS, separated by commas.
CORS_ORIGINS = [o.strip() for o in os.environ.get("SCI_RTW_CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=None if CORS_ORIGINS else r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

# The interactive API pages load their script from a CDN, so they are left
# out of the content security policy.
DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data:; "
    "frame-ancestors 'none'; base-uri 'self'; form-action 'self'")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    if not request.url.path.startswith(DOCS_PATHS):
        response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
    # Patient records and tokens must not be kept by a browser or proxy cache.
    if response.headers.get("content-type", "").startswith("application/json"):
        response.headers["Cache-Control"] = "no-store"
    return response


init_db()
auth.ensure_default_admin()
app.include_router(auth.router)

# ----------------------------------------------------------------------------
# Load models + artifacts once at startup
# ----------------------------------------------------------------------------
layer1_preprocessor = joblib.load(f"{MODEL_DIR}/layer1_preprocessor.joblib")
layer1_model = xgb.Booster()
layer1_model.load_model(f"{MODEL_DIR}/layer1_xgboost.json")

layer2_preprocessor = joblib.load(f"{MODEL_DIR}/layer2_preprocessor.joblib")
layer2_model = xgb.Booster()
layer2_model.load_model(f"{MODEL_DIR}/layer2_xgboost.json")
layer2_label_encoder = joblib.load(f"{MODEL_DIR}/layer2_label_encoder.joblib")



def model_fingerprint() -> str:
    """Identifies the deployed models by their content, so that it changes
    whenever they are retrained and never otherwise. Saved with each
    prediction: an estimate in a patient's history can then be told apart
    from one the current models would give."""
    digest = hashlib.sha256()
    for layer in (1, 2):
        with open(f"{MODEL_DIR}/layer{layer}_xgboost.json", "rb") as f:
            digest.update(f.read())
    return digest.hexdigest()[:12]


MODEL_VERSION = model_fingerprint()

with open(f"{MODEL_DIR}/imputation_modes.json") as f:
    IMPUTATION_MODES = json.load(f)  # {"numeric": {...}, "categorical": {...}}

with open(f"{MODEL_DIR}/category_options.json") as f:
    CATEGORY_OPTIONS = json.load(f)

with open(f"{MODEL_DIR}/layer1_results.json") as f:
    LAYER1_RESULTS = json.load(f)

with open(f"{MODEL_DIR}/layer2_results.json") as f:
    LAYER2_RESULTS = json.load(f)

# Cross-validation, overfitting check, baseline and factor importance
# written by train_models.py. Optional so the API still starts against
# models trained with an older version of that script.
EVALUATION = {}
if os.path.exists(f"{MODEL_DIR}/evaluation_report.json"):
    with open(f"{MODEL_DIR}/evaluation_report.json") as f:
        EVALUATION = json.load(f)

# Accuracy measured at each confidence level on the held-out test set.
if "confidence_bands" not in EVALUATION.get("layer1", {}):
    raise RuntimeError("models/evaluation_report.json has no confidence bands. "
                       "Run: python src/train_models.py")
CONFIDENCE_BANDS = EVALUATION["layer1"]["confidence_bands"]

NUMERIC_COLS = ["age_at_injury", "asia_motor_total_discharge"]

CATEGORICAL_COLS = ["sex", "injury_cause", "vertebral_injury_level", "associated_injury",
                     "marital_status", "education_level", "veteran_status",
                     "pre_injury_employment_status", "work_related_injury",
                     "neuro_level_discharge", "aisgrade_discharge",
                     "injury_completeness_discharge"]

L1_FEATURE_NAMES = list(layer1_preprocessor.get_feature_names_out())


def normalise_category(field: str, value):
    """A value the user leaves blank (None) is filled with that field's
    population mode - exactly what training does for genuinely missing data.
    This is a safe fallback, not a special 'unknown' category (see module
    docstring). Anything else has already been checked against
    CATEGORY_OPTIONS by PatientInput."""
    if value is None:
        return IMPUTATION_MODES["categorical"][field]
    return str(value)


def field_of(feature_name: str) -> str:
    """Map an encoded feature (e.g. 'cat__pre_injury_employment_status_5.0')
    back to its input field."""
    prefix, name = feature_name.split("__", 1)
    if prefix == "num":
        return name
    for col in sorted(CATEGORICAL_COLS, key=len, reverse=True):
        if name.startswith(col + "_"):
            return col
    return name


# ----------------------------------------------------------------------------
# Request / response schemas (mirrors the 3-step "New Prediction" form:
# Demographics, Injury Profile, Functional Status)
# ----------------------------------------------------------------------------
class PatientInput(BaseModel):
    # Demographics
    sex: str
    age_at_injury: float = Field(ge=0, le=110, allow_inf_nan=False)
    education_level: str
    marital_status: str
    veteran_status: Optional[str] = None
    pre_injury_employment_status: str

    # Injury profile
    injury_cause: str
    vertebral_injury_level: Optional[str] = None
    associated_injury: Optional[str] = None
    work_related_injury: Optional[str] = None
    neuro_level_discharge: str
    aisgrade_discharge: str
    injury_completeness_discharge: str

    # Functional status
    asia_motor_total_discharge: Optional[float] = Field(default=None, ge=0, le=100, allow_inf_nan=False)

    # A value the model was never trained on is refused rather than replaced:
    # replacing it returned a confident-looking prediction for what was in
    # effect a different patient. Only a blank is imputed (normalise_category).
    @field_validator(*CATEGORICAL_COLS)
    @classmethod
    def known_category(cls, value, info):
        if value is not None and value not in CATEGORY_OPTIONS[info.field_name]:
            raise ValueError("not a recognised value; expected one of: "
                             + ", ".join(CATEGORY_OPTIONS[info.field_name]))
        return value


def build_feature_row(patient: PatientInput) -> pd.DataFrame:
    row = patient.model_dump()

    if row["asia_motor_total_discharge"] is None:
        row["asia_motor_total_discharge"] = IMPUTATION_MODES["numeric"]["asia_motor_total_discharge"]

    df = pd.DataFrame([row])
    for c in CATEGORICAL_COLS:
        df[c] = df[c].map(lambda v, c=c: normalise_category(c, v))
    return df[NUMERIC_COLS + CATEGORICAL_COLS]


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "layer1_model_loaded": layer1_model is not None,
        "layer2_model_loaded": layer2_model is not None,
        "model_version": MODEL_VERSION,
    }


@app.get("/category-options")
def category_options():
    return CATEGORY_OPTIONS


@app.get("/model-info")
def model_info(user: dict = Depends(require_staff)):
    def layer_info(results, key):
        ev = EVALUATION.get(key)
        return {
            "models_compared": {name: res["metrics"] for name, res in results.items()},
            "active_model": "xgboost",
            "top_features": results["xgboost"]["feature_importance"][:10],
            "evaluation": None if ev is None else {
                "n_train": ev["n_train"],
                "n_test": ev["n_test"],
                "majority_baseline": ev["majority_baseline"],
                "cross_validation": ev["cross_validation"],
                "train_vs_test": ev["train_vs_test"],
                "test_ci": ev["test_ci"],
                "factor_importance": ev["factor_importance"],
                "per_class": ev.get("per_class"),
                "confidence_bands": ev.get("confidence_bands"),
                "calibration": {m: c["brier_score"] for m, c in ev["calibration"].items()}
                               if "calibration" in ev else None,
            },
        }

    return {
        "layer1": layer_info(LAYER1_RESULTS, "layer1"),
        "layer2": layer_info(LAYER2_RESULTS, "layer2"),
        "settings": EVALUATION.get("settings"),
    }


def run_prediction(patient: PatientInput) -> dict:
    try:
        X = build_feature_row(patient)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    # ---- Layer 1: RTW status ----
    X1 = layer1_preprocessor.transform(X)
    dmat1 = xgb.DMatrix(X1)
    rtw_proba = float(layer1_model.predict(dmat1)[0])
    rtw_pred = int(rtw_proba >= 0.5)

    result = {
        "rtw_status": "RTW" if rtw_pred == 1 else "Non-RTW",
        "rtw_probability": round(rtw_proba, 4),
        "layer1_model_used": "xgboost",
        "model_version": MODEL_VERSION,
        "from_current_model": True,
    }

    # ---- Layer 2: RTW timing (only if RTW predicted) ----
    if rtw_pred == 1:
        X2 = layer2_preprocessor.transform(X)
        dmat2 = xgb.DMatrix(X2)
        timing_proba = layer2_model.predict(dmat2)[0]
        timing_idx = int(np.argmax(timing_proba))
        timing_label = layer2_label_encoder.inverse_transform([timing_idx])[0]

        result["rtw_timing"] = timing_label
        result["rtw_timing_probabilities"] = {
            cls: round(float(p), 4)
            for cls, p in zip(layer2_label_encoder.classes_, timing_proba)
        }
    else:
        result["rtw_timing"] = None
        result["rtw_timing_probabilities"] = None

    # ---- Global top features (same for every patient) ----
    result["top_factors"] = LAYER1_RESULTS["xgboost"]["feature_importance"][:5]

    # ---- Patient-specific factors: XGBoost per-prediction contributions
    # (SHAP-style, in log-odds), summed back onto the input field ----
    contribs = layer1_model.predict(dmat1, pred_contribs=True)[0]
    by_field = {}
    for name, c in zip(L1_FEATURE_NAMES, contribs[:-1]):
        f = field_of(name)
        by_field[f] = by_field.get(f, 0.0) + float(c)
    entered = patient.model_dump()
    ranked = sorted(by_field.items(), key=lambda kv: -abs(kv[1]))
    result["patient_factors"] = [
        {
            "field": f,
            "value": entered.get(f),
            "contribution": round(c, 4),
            "effect": "increases" if c > 0 else "decreases",
        }
        for f, c in ranked[:6]
        if abs(c) > 1e-6
    ]
    result["baseline_contribution"] = round(float(contribs[-1]), 4)

    # ---- Confidence level + suggested considerations (Fig. 3.13) ----
    result["confidence"] = recommendations.confidence_for(rtw_proba, CONFIDENCE_BANDS)
    result["recommendations"] = recommendations.suggest(
        result["rtw_status"], result["confidence"]["level"], result["rtw_timing"], by_field)

    return result


@app.post("/predict")
def predict(patient: PatientInput, user: dict = Depends(require_clinician)):
    """Stateless prediction: nothing is stored. Use
    POST /patients/{id}/predictions to save the result to a patient's history."""
    return run_prediction(patient)


# ----------------------------------------------------------------------------
# Patient profiles (FR2) - Doctor / Therapist
# ----------------------------------------------------------------------------
class PatientProfileInput(BaseModel):
    patient_code: str = Field(min_length=1, max_length=40)
    full_name: str = Field(min_length=1, max_length=120)
    notes: str = Field(default="", max_length=2000)
    # Username of a patient-role account allowed to view this profile's results.
    linked_username: Optional[str] = None


class NotesInput(BaseModel):
    clinician_notes: str = Field(max_length=4000)


PATIENT_SELECT = """
    SELECT p.*, u.username AS linked_username,
           (SELECT COUNT(*) FROM assessments a WHERE a.patient_id = p.id) AS assessment_count
    FROM patients p LEFT JOIN users u ON u.id = p.user_id
"""


def patient_out(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "patient_code": row["patient_code"],
        "full_name": row["full_name"],
        "notes": row["notes"],
        "linked_username": row["linked_username"],
        "assessment_count": row["assessment_count"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def resolve_linked_user(conn, username: Optional[str]) -> Optional[int]:
    if not username or not username.strip():
        return None
    row = conn.execute("SELECT id, role FROM users WHERE username = ?", (username.strip(),)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No account with username '{username.strip()}'")
    if row["role"] != "patient":
        raise HTTPException(status_code=422, detail="Only a patient-role account can be linked to a patient profile")
    return row["id"]


def get_patient_or_404(conn, patient_id: int) -> sqlite3.Row:
    row = conn.execute(PATIENT_SELECT + " WHERE p.id = ?", (patient_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Patient profile not found")
    return row


@app.get("/patients")
def list_patients(q: Optional[str] = None, user: dict = Depends(require_clinician),
                  conn: sqlite3.Connection = Depends(get_db)):
    if q and q.strip():
        like = f"%{q.strip()}%"
        rows = conn.execute(PATIENT_SELECT + " WHERE p.patient_code LIKE ? OR p.full_name LIKE ? "
                            "ORDER BY p.updated_at DESC", (like, like)).fetchall()
    else:
        rows = conn.execute(PATIENT_SELECT + " ORDER BY p.updated_at DESC").fetchall()
    return [patient_out(r) for r in rows]


@app.post("/patients", status_code=201)
def create_patient(data: PatientProfileInput, user: dict = Depends(require_clinician),
                   conn: sqlite3.Connection = Depends(get_db)):
    linked_id = resolve_linked_user(conn, data.linked_username)
    ts = now_iso()
    try:
        cur = conn.execute(
            "INSERT INTO patients (patient_code, full_name, notes, user_id, created_by, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (data.patient_code.strip(), data.full_name.strip(), data.notes, linked_id, user["id"], ts, ts))
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="A patient profile with that patient ID already exists")
    record_audit(conn, user, "patient.created", data.patient_code.strip())
    return patient_out(get_patient_or_404(conn, cur.lastrowid))


@app.get("/patients/{patient_id}")
def get_patient(patient_id: int, user: dict = Depends(require_clinician),
                conn: sqlite3.Connection = Depends(get_db)):
    return patient_out(get_patient_or_404(conn, patient_id))


@app.put("/patients/{patient_id}")
def update_patient(patient_id: int, data: PatientProfileInput, user: dict = Depends(require_clinician),
                   conn: sqlite3.Connection = Depends(get_db)):
    get_patient_or_404(conn, patient_id)
    linked_id = resolve_linked_user(conn, data.linked_username)
    try:
        conn.execute(
            "UPDATE patients SET patient_code = ?, full_name = ?, notes = ?, user_id = ?, updated_at = ? WHERE id = ?",
            (data.patient_code.strip(), data.full_name.strip(), data.notes, linked_id, now_iso(), patient_id))
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="A patient profile with that patient ID already exists")
    record_audit(conn, user, "patient.updated", data.patient_code.strip())
    return patient_out(get_patient_or_404(conn, patient_id))


@app.delete("/patients/{patient_id}")
def delete_patient(patient_id: int, user: dict = Depends(require_clinician),
                   conn: sqlite3.Connection = Depends(get_db)):
    """Removes the profile together with its assessments and predictions."""
    row = get_patient_or_404(conn, patient_id)
    conn.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
    record_audit(conn, user, "patient.removed", row["patient_code"],
                 f"with {row['assessment_count']} saved prediction(s)")
    return {"status": "removed"}


# ----------------------------------------------------------------------------
# Saved predictions + history (FR4-FR7)
# ----------------------------------------------------------------------------
PREDICTION_SELECT = """
    SELECT pr.id, pr.rtw_status, pr.rtw_probability, pr.rtw_timing, pr.timing_probabilities_json,
           pr.model_used, pr.model_version, pr.created_at, a.input_json, a.patient_id,
           p.patient_code, p.full_name AS patient_name, eu.full_name AS entered_by_name,
           rp.factors_json, rp.clinician_notes, rp.confidence_json, rp.recommendations_json
    FROM predictions pr
    JOIN assessments a ON a.id = pr.assessment_id
    JOIN patients p ON p.id = a.patient_id
    JOIN prediction_reports rp ON rp.prediction_id = pr.id
    LEFT JOIN users eu ON eu.id = a.entered_by
"""


def prediction_out(row: sqlite3.Row) -> dict:
    timing = row["timing_probabilities_json"]
    # A record that backfill_reports() could not complete (see there) still
    # shows a confidence level, which depends only on the stored probability,
    # but no suggestions rather than ones rebuilt from partial data.
    confidence = (json.loads(row["confidence_json"]) if row["confidence_json"]
                  else recommendations.confidence_for(row["rtw_probability"], CONFIDENCE_BANDS))
    suggestions = json.loads(row["recommendations_json"]) if row["recommendations_json"] else []
    return {
        "id": row["id"],
        "timestamp": row["created_at"],
        "patient": {"id": row["patient_id"], "patient_code": row["patient_code"], "full_name": row["patient_name"]},
        "entered_by": row["entered_by_name"],
        "input": json.loads(row["input_json"]),
        "result": {
            "rtw_status": row["rtw_status"],
            "rtw_probability": row["rtw_probability"],
            "rtw_timing": row["rtw_timing"],
            "rtw_timing_probabilities": json.loads(timing) if timing else None,
            "layer1_model_used": row["model_used"],
            # A record saved before versions were kept has none, and so is
            # also from an earlier model.
            "model_version": row["model_version"],
            "from_current_model": row["model_version"] == MODEL_VERSION,
            "patient_factors": json.loads(row["factors_json"]),
            "confidence": confidence,
            "recommendations": suggestions,
        },
        "clinician_notes": row["clinician_notes"],
    }


@app.post("/patients/{patient_id}/predictions", status_code=201)
def predict_and_save(patient_id: int, patient: PatientInput, user: dict = Depends(require_clinician),
                     conn: sqlite3.Connection = Depends(get_db)):
    profile = get_patient_or_404(conn, patient_id)
    result = run_prediction(patient)
    ts = now_iso()
    assessment_id = conn.execute(
        "INSERT INTO assessments (patient_id, entered_by, input_json, created_at) VALUES (?, ?, ?, ?)",
        (patient_id, user["id"], json.dumps(patient.model_dump()), ts)).lastrowid
    timing_probs = result["rtw_timing_probabilities"]
    prediction_id = conn.execute(
        "INSERT INTO predictions (assessment_id, rtw_status, rtw_probability, rtw_timing, "
        "timing_probabilities_json, model_used, model_version, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (assessment_id, result["rtw_status"], result["rtw_probability"], result["rtw_timing"],
         json.dumps(timing_probs) if timing_probs else None, result["layer1_model_used"],
         MODEL_VERSION, ts)).lastrowid
    conn.execute(
        "INSERT INTO prediction_reports (prediction_id, factors_json, confidence_json, recommendations_json) "
        "VALUES (?, ?, ?, ?)",
        (prediction_id, json.dumps(result["patient_factors"]), json.dumps(result["confidence"]),
         json.dumps(result["recommendations"])))
    conn.execute("UPDATE patients SET updated_at = ? WHERE id = ?", (ts, patient_id))
    record_audit(conn, user, "prediction.saved", profile["patient_code"], f"prediction {prediction_id}")
    result["prediction_id"] = prediction_id
    return result


@app.get("/predictions")
def list_predictions(response: Response, patient_id: Optional[int] = None,
                     limit: Optional[int] = Query(default=None, ge=1, le=500),
                     offset: int = Query(default=0, ge=0), user: dict = Depends(current_user),
                     conn: sqlite3.Connection = Depends(get_db)):
    """Clinicians see every saved prediction; a patient sees only those of
    the profile(s) linked to their own account. Newest first. Without a
    limit every match is returned; with one, the X-Total-Count header gives
    the number of matches so a caller can page through them."""
    where, params = [], []
    if user["role"] == "patient":
        where.append("p.user_id = ?")
        params.append(user["id"])
    elif user["role"] not in CLINICIAN_ROLES:
        raise HTTPException(status_code=403, detail="Your role does not have access to this function")
    if patient_id is not None:
        where.append("a.patient_id = ?")
        params.append(patient_id)
    condition = " WHERE " + " AND ".join(where) if where else ""
    total = conn.execute("SELECT COUNT(*) FROM predictions pr JOIN assessments a ON a.id = pr.assessment_id "
                         "JOIN patients p ON p.id = a.patient_id" + condition, params).fetchone()[0]
    response.headers["X-Total-Count"] = str(total)
    # In SQLite a limit of -1 means no limit.
    sql = PREDICTION_SELECT + condition + " ORDER BY pr.id DESC LIMIT ? OFFSET ?"
    return [prediction_out(r) for r in conn.execute(sql, params + [-1 if limit is None else limit, offset]).fetchall()]


@app.patch("/predictions/{prediction_id}/notes")
def update_prediction_notes(prediction_id: int, data: NotesInput, user: dict = Depends(require_clinician),
                            conn: sqlite3.Connection = Depends(get_db)):
    cur = conn.execute("UPDATE prediction_reports SET clinician_notes = ? WHERE prediction_id = ?",
                       (data.clinician_notes, prediction_id))
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Prediction not found")
    row = conn.execute(PREDICTION_SELECT + " WHERE pr.id = ?", (prediction_id,)).fetchone()
    record_audit(conn, user, "prediction.notes_updated", row["patient_code"], f"prediction {prediction_id}")
    return prediction_out(row)


def backfill_reports() -> int:
    """Give records saved by an earlier version their confidence level and
    suggested considerations.

    The stored inputs are run through the model again. The result is used
    only if it reproduces the stored probability, which proves the same
    model produced the record; if the model has since been retrained the
    record is left as it was, because suggestions from a different model
    would not belong to that prediction. The stored prediction itself is
    never changed."""
    from db import connect

    conn = connect()
    done = 0
    try:
        rows = conn.execute(
            "SELECT pr.id, pr.rtw_probability, a.input_json FROM predictions pr "
            "JOIN assessments a ON a.id = pr.assessment_id "
            "JOIN prediction_reports rp ON rp.prediction_id = pr.id "
            "WHERE rp.recommendations_json IS NULL").fetchall()
        for row in rows:
            try:
                again = run_prediction(PatientInput(**json.loads(row["input_json"])))
            except Exception:
                continue
            if abs(again["rtw_probability"] - row["rtw_probability"]) > 1e-4:
                continue
            conn.execute(
                "UPDATE prediction_reports SET confidence_json = ?, recommendations_json = ? WHERE prediction_id = ?",
                (json.dumps(again["confidence"]), json.dumps(again["recommendations"]), row["id"]))
            done += 1
        conn.commit()
    finally:
        conn.close()
    return done


backfill_reports()

# ----------------------------------------------------------------------------
# Hosted deployment: serve the built frontend from this same service
# ----------------------------------------------------------------------------
# Off unless SCI_RTW_FRONTEND_DIR points at a built frontend (the Dockerfile
# sets it), so local development with the Vite dev server is unaffected.
# Mounted last so every API route above takes priority.
FRONTEND_DIR = os.environ.get("SCI_RTW_FRONTEND_DIR")
if FRONTEND_DIR:
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
