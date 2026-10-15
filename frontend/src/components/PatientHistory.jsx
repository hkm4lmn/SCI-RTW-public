import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { getLabel } from "../constants/labelMaps.js";
import PredictionResults from "./PredictionResults.jsx";

function NotesPanel({ record, canEdit, onChanged }) {
  const [notes, setNotes] = useState(record.clinician_notes);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState(null);

  // Reset only when a different record is opened; re-running this after a
  // save would wipe the "Notes saved." confirmation.
  useEffect(() => { setNotes(record.clinician_notes); setStatus(null); }, [record.id]);

  async function save() {
    setSaving(true);
    setStatus(null);
    try {
      await api.updateNotes(record.id, notes);
      await onChanged();
      setStatus("Notes saved.");
    } catch (e) {
      setStatus(`Couldn't save notes: ${e.message}`);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="panel">
      <h3>Clinician notes</h3>
      <p className="sub">
        Recorded {new Date(record.timestamp).toLocaleString()}
        {record.entered_by ? ` by ${record.entered_by}` : ""}.
      </p>
      {/* A text box prints clipped, so the printed report shows the notes as text. */}
      {canEdit && <p className="notes-text print-only">{notes || "No notes have been added for this assessment."}</p>}
      {canEdit ? (
        <div className="field no-print">
          <textarea rows={4} value={notes} maxLength={4000} aria-label="Clinician notes"
                    placeholder="Recommendations or follow-up notes for this assessment"
                    onChange={(e) => setNotes(e.target.value)} />
          <div>
            <button className="btn ghost" disabled={saving || notes === record.clinician_notes} onClick={save}>
              {saving ? "Saving…" : "Save notes"}
            </button>
            {status && <span className="hint" style={{ marginLeft: 12 }}>{status}</span>}
          </div>
        </div>
      ) : (
        <p className="notes-text">{record.clinician_notes || "No notes have been added for this assessment."}</p>
      )}
    </div>
  );
}

export default function PatientHistory({ history, role, initialPatientId, onChanged }) {
  const isPatient = role === "patient";
  const [patientFilter, setPatientFilter] = useState(initialPatientId ? String(initialPatientId) : "");
  const [openId, setOpenId] = useState(null);

  const patientsInHistory = [...new Map(history.map((h) => [h.patient.id, h.patient])).values()];
  const shown = patientFilter ? history.filter((h) => String(h.patient.id) === patientFilter) : history;
  const open = history.find((h) => h.id === openId);

  if (open) {
    return (
      <PredictionResults
        result={open.result}
        patient={open.patient}
        onNewPrediction={() => setOpenId(null)}
        actionLabel="Back to history"
        assessedAt={open.timestamp}
        enteredBy={open.entered_by}
      >
        <NotesPanel record={open} canEdit={!isPatient} onChanged={onChanged} />
      </PredictionResults>
    );
  }

  return (
    <div>
      <div className="page-header">
        <h2>{isPatient ? "My Results" : "Patient Assessment History"}</h2>
        <p>
          {isPatient
            ? "Return-to-work assessments your care team has recorded for you."
            : `${shown.length} saved assessment${shown.length !== 1 ? "s" : ""}, newest first.`}
        </p>
      </div>

      {isPatient && (
        <div className="disclaimer">
          These are statistical estimates to support discussion with your care team. They are
          not a diagnosis or a guarantee of outcome.
        </div>
      )}

      {history.length === 0 ? (
        <div className="panel">
          <div className="empty-state">
            {isPatient
              ? "No results yet. Your doctor or therapist needs to link your account to your patient record."
              : "No assessments yet. Run a prediction from \"New Prediction\" to see it here."}
          </div>
        </div>
      ) : (
        <div className="panel">
          {!isPatient && (
            <div className="panel-toolbar">
              <select className="search" value={patientFilter} aria-label="Filter by patient"
                      onChange={(e) => setPatientFilter(e.target.value)}>
                <option value="">All patients</option>
                {patientsInHistory.map((p) => (
                  <option key={p.id} value={p.id}>{p.patient_code} — {p.full_name}</option>
                ))}
              </select>
            </div>
          )}
          <table className="history">
            <thead>
              <tr>
                <th>Date</th>
                {!isPatient && <th>Patient</th>}
                <th>Age at injury</th>
                <th>AIS Grade</th>
                <th>Neuro Level</th>
                <th>RTW Prediction</th>
                <th>Probability</th>
                <th>Timing</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {shown.map((h) => (
                <tr key={h.id}>
                  <td>{new Date(h.timestamp).toLocaleString()}</td>
                  {!isPatient && <td>{h.patient.patient_code} — {h.patient.full_name}</td>}
                  <td>{h.input.age_at_injury}</td>
                  <td>{getLabel("aisgrade_discharge", h.input.aisgrade_discharge)}</td>
                  <td>{getLabel("neuro_level_discharge", h.input.neuro_level_discharge)}</td>
                  <td>
                    <span className={`pill ${h.result.rtw_status === "RTW" ? "rtw" : "non-rtw"}`}>
                      {h.result.rtw_status}
                    </span>
                  </td>
                  <td>{Math.round(h.result.rtw_probability * 100)}%</td>
                  <td>{h.result.rtw_timing || "—"}</td>
                  <td className="row-actions">
                    <button className="link" onClick={() => setOpenId(h.id)}>View</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
