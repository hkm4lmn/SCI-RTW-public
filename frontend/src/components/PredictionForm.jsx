import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { getLabel } from "../constants/labelMaps.js";
import PredictionResults from "./PredictionResults.jsx";

// 3 data-entry steps, matching exactly what the trained model uses (see
// docs/test_report.md: the proposal's Vocational and Psychosocial sections
// were dropped -- Psychosocial because its one usable field, PHQ-9, turned
// out to act as an injury-era proxy rather than a clinical signal, and
// Vocational because no NSCISC variables back it).
// The flow starts by choosing whose assessment this is (proposal Fig. 3.5:
// "create or select patient profile" comes before data entry), so every
// prediction is saved against a patient profile.
const STEPS = [
  { key: "patient", label: "Patient", fields: [] },
  {
    key: "demographics",
    label: "Demographics",
    fields: [
      { name: "sex", label: "Biological sex", type: "select", required: true },
      { name: "age_at_injury", label: "Age at injury", type: "number", required: true, min: 0, max: 110 },
      { name: "education_level", label: "Highest education level", type: "select", required: true },
      { name: "marital_status", label: "Marital status", type: "select", required: true },
      // veteran_status (U.S. military) is a model input but is not asked:
      // it does not apply outside the U.S. and has no measurable effect on
      // the prediction, so the API fills in its usual value, "No".
      { name: "pre_injury_employment_status", label: "Pre-injury employment status", type: "select", required: true },
    ],
  },
  {
    key: "injury",
    label: "Injury Profile",
    fields: [
      { name: "injury_cause", label: "Cause of injury", type: "select", required: true },
      { name: "vertebral_injury_level", label: "Vertebral fracture/dislocation", type: "select" },
      { name: "associated_injury", label: "Associated injury", type: "select" },
      { name: "work_related_injury", label: "Work-related injury", type: "select" },
      { name: "neuro_level_discharge", label: "Neurologic level (at discharge)", type: "select", required: true },
      { name: "aisgrade_discharge", label: "AIS grade (at discharge)", type: "select", required: true },
      { name: "injury_completeness_discharge", label: "Injury completeness category", type: "select", required: true },
    ],
  },
  {
    key: "functional",
    label: "Functional Status",
    fields: [
      {
        name: "asia_motor_total_discharge", label: "ASIA motor total score (at discharge)",
        type: "number", min: 0, max: 100,
        hint: "0–100. Leave blank if not assessed: 50, the typical recorded score, is used instead.",
      },
    ],
  },
];

const OPTIONAL_FIELDS = STEPS.flatMap((st) => st.fields)
  .filter((f) => !f.required)
  .map((f) => f.name);

const DEFAULTS = {
  sex: "", age_at_injury: "", education_level: "", marital_status: "",
  pre_injury_employment_status: "",
  injury_cause: "", vertebral_injury_level: "", associated_injury: "",
  work_related_injury: "", neuro_level_discharge: "", aisgrade_discharge: "",
  injury_completeness_discharge: "", asia_motor_total_discharge: "",
};

const NEW_PATIENT = { patient_code: "", full_name: "" };

// The API refuses a number outside the field's range, so the form says so
// before the request is sent. A blank is not out of range.
function outOfRange(field, value) {
  if (field.type !== "number" || value === "") return false;
  const n = Number(value);
  return Number.isNaN(n) || n < field.min || n > field.max;
}

export default function PredictionForm({ onSaved, initialPatientId }) {
  const [options, setOptions] = useState(null);
  const [optionsError, setOptionsError] = useState(null);
  const [patients, setPatients] = useState(null);
  const [patientId, setPatientId] = useState(initialPatientId ? String(initialPatientId) : "");
  const [newPatient, setNewPatient] = useState(NEW_PATIENT);
  const [creatingPatient, setCreatingPatient] = useState(false);
  const [patientError, setPatientError] = useState(null);
  // Arriving from a patient's row skips straight to data entry.
  const [stepIdx, setStepIdx] = useState(initialPatientId ? 1 : 0);
  const [form, setForm] = useState(DEFAULTS);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [submitError, setSubmitError] = useState(null);

  useEffect(() => {
    api.categoryOptions().then(setOptions).catch((e) => setOptionsError(e.message));
    api.listPatients().then(setPatients).catch((e) => setOptionsError(e.message));
  }, []);

  function update(name, value) {
    setForm((f) => ({ ...f, [name]: value }));
  }

  function currentStepValid() {
    const step = STEPS[stepIdx];
    if (step.key === "patient") return patientId !== "";
    return step.fields.every((f) => {
      const value = form[f.name];
      const blank = value === "" || value === null || value === undefined;
      return blank ? !f.required : !outOfRange(f, value);
    });
  }

  async function createPatient() {
    setCreatingPatient(true);
    setPatientError(null);
    try {
      const created = await api.createPatient({ ...newPatient, notes: "", linked_username: null });
      setPatients((list) => [created, ...(list || [])]);
      setPatientId(String(created.id));
      setNewPatient(NEW_PATIENT);
    } catch (e) {
      setPatientError(e.message);
    } finally {
      setCreatingPatient(false);
    }
  }

  function startOver() {
    setForm(DEFAULTS);
    setStepIdx(0);
    setResult(null);
    setSubmitError(null);
  }

  async function runPrediction() {
    setSubmitting(true);
    setSubmitError(null);
    try {
      const payload = { ...form };
      payload.age_at_injury = Number(payload.age_at_injury);
      payload.asia_motor_total_discharge =
        payload.asia_motor_total_discharge === "" ? null : Number(payload.asia_motor_total_discharge);
      // Optional selects left blank -> null, so the API applies the same
      // population-mode imputation it used during training (no "Unknown"
      // category exists in the model any more -- see labelMaps.js header).
      OPTIONAL_FIELDS.forEach((name) => {
        if (payload[name] === "") payload[name] = null;
      });
      const res = await api.savePrediction(patientId, payload);
      setResult(res);
      onSaved();
    } catch (e) {
      setSubmitError(e.message);
    } finally {
      setSubmitting(false);
    }
  }

  if (optionsError) {
    return (
      <div className="error-banner">
        Couldn't load form options from the API: {optionsError}. Confirm the backend
        is running at the configured URL.
      </div>
    );
  }

  if (!options || !patients) {
    return <p className="loading-note">Loading form…</p>;
  }

  const patient = patients.find((p) => String(p.id) === patientId);

  if (result) {
    return <PredictionResults result={result} patient={patient} onNewPrediction={startOver} />;
  }

  const step = STEPS[stepIdx];
  const isLast = stepIdx === STEPS.length - 1;

  return (
    <div>
      <div className="page-header">
        <h2>New RTW Prediction</h2>
        <p>Complete each section for the probability and timeline estimate.</p>
      </div>

      <div className="step-tabs">
        {STEPS.map((s, i) => (
          <div key={s.key} className={`step-tab ${i === stepIdx ? "active" : i < stepIdx ? "done" : ""}`}>
            {s.label}
          </div>
        ))}
      </div>

      <div className="panel">
        <h3>{step.label}</h3>
        {step.key !== "patient" && patient && (
          <p className="sub">Assessment for {patient.full_name} ({patient.patient_code}).</p>
        )}
        {step.key === "patient" && (
          <div>
            <div className="field-grid">
              <div className="field">
                <label htmlFor="select_patient">Patient profile *</label>
                <select id="select_patient" value={patientId} onChange={(e) => setPatientId(e.target.value)}>
                  <option value="">Select…</option>
                  {patients.map((p) => (
                    <option key={p.id} value={p.id}>{p.patient_code} — {p.full_name}</option>
                  ))}
                </select>
                <span className="hint">The prediction is saved to this patient's history.</span>
              </div>
            </div>

            <h4 className="subhead">Or create a new patient profile</h4>
            <div className="field-grid">
              <div className="field">
                <label htmlFor="new_patient_code">Patient ID</label>
                <input id="new_patient_code" value={newPatient.patient_code} maxLength={40}
                       onChange={(e) => setNewPatient({ ...newPatient, patient_code: e.target.value })} />
              </div>
              <div className="field">
                <label htmlFor="new_patient_name">Patient name</label>
                <input id="new_patient_name" value={newPatient.full_name} maxLength={120}
                       onChange={(e) => setNewPatient({ ...newPatient, full_name: e.target.value })} />
              </div>
            </div>
            {patientError && <div className="error-banner" style={{ marginTop: 18 }}>{patientError}</div>}
            <button
              className="btn ghost" style={{ marginTop: 16 }}
              disabled={creatingPatient || !newPatient.patient_code.trim() || !newPatient.full_name.trim()}
              onClick={createPatient}
            >
              {creatingPatient ? "Creating…" : "Create profile"}
            </button>
          </div>
        )}
        <div className="field-grid">
          {step.fields.map((f) => (
            <div className="field" key={f.name}>
              <label htmlFor={`field_${f.name}`}>
                {f.label}
                {f.required && " *"}
              </label>
              {f.type === "select" ? (
                <select id={`field_${f.name}`} value={form[f.name]} onChange={(e) => update(f.name, e.target.value)}>
                  <option value="">{f.required ? "Select…" : "Not specified"}</option>
                  {(options[f.name] || []).map((v) => (
                    <option key={v} value={v}>{getLabel(f.name, v)}</option>
                  ))}
                </select>
              ) : (
                <input
                  id={`field_${f.name}`}
                  type="number"
                  min={f.min}
                  max={f.max}
                  value={form[f.name]}
                  onChange={(e) => update(f.name, e.target.value)}
                />
              )}
              {outOfRange(f, form[f.name])
                ? <span className="hint field-error">Enter a value from {f.min} to {f.max}.</span>
                : f.hint && <span className="hint">{f.hint}</span>}
            </div>
          ))}
        </div>

        {submitError && <div className="error-banner" style={{ marginTop: 18 }}>Prediction failed: {submitError}</div>}

        <div className="form-actions">
          <button
            className="btn ghost"
            disabled={stepIdx === 0}
            onClick={() => setStepIdx((i) => Math.max(0, i - 1))}
          >
            Previous
          </button>
          {!isLast ? (
            <button
              className="btn primary"
              disabled={!currentStepValid()}
              onClick={() => setStepIdx((i) => Math.min(STEPS.length - 1, i + 1))}
            >
              Next
            </button>
          ) : (
            <button className="btn primary" disabled={submitting || !currentStepValid()} onClick={runPrediction}>
              {submitting ? "Running prediction…" : "Run Prediction"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
