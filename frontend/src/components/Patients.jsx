import React, { useEffect, useState } from "react";
import { api } from "../api.js";

const EMPTY = { patient_code: "", full_name: "", notes: "", linked_username: "" };

export default function Patients({ onPredict, onViewHistory, onChanged }) {
  const [patients, setPatients] = useState(null);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState("");
  const [editingId, setEditingId] = useState(null); // null = closed, "new" = creating, number = editing
  const [form, setForm] = useState(EMPTY);
  const [formError, setFormError] = useState(null);
  const [saving, setSaving] = useState(false);

  function load() {
    return api.listPatients().then(setPatients).catch((e) => setError(e.message));
  }

  useEffect(() => { load(); }, []);

  function openNew() {
    setForm(EMPTY);
    setFormError(null);
    setEditingId("new");
  }

  function openEdit(p) {
    setForm({
      patient_code: p.patient_code, full_name: p.full_name,
      notes: p.notes, linked_username: p.linked_username || "",
    });
    setFormError(null);
    setEditingId(p.id);
  }

  async function save(e) {
    e.preventDefault();
    setSaving(true);
    setFormError(null);
    const payload = { ...form, linked_username: form.linked_username.trim() || null };
    try {
      if (editingId === "new") await api.createPatient(payload);
      else await api.updatePatient(editingId, payload);
      setEditingId(null);
      await load();
      onChanged();
    } catch (err) {
      setFormError(err.message);
    } finally {
      setSaving(false);
    }
  }

  async function remove(p) {
    const count = p.assessment_count;
    const warning = count
      ? `Remove ${p.full_name} (${p.patient_code}) and their ${count} saved assessment${count !== 1 ? "s" : ""}? This cannot be undone.`
      : `Remove the profile for ${p.full_name} (${p.patient_code})?`;
    if (!window.confirm(warning)) return;
    try {
      await api.deletePatient(p.id);
      await load();
      onChanged();
    } catch (err) {
      setError(err.message);
    }
  }

  if (!patients && !error) return <p className="loading-note">Loading patients…</p>;

  const term = search.trim().toLowerCase();
  const shown = (patients || []).filter(
    (p) => !term || p.patient_code.toLowerCase().includes(term) || p.full_name.toLowerCase().includes(term)
  );

  return (
    <div>
      <div className="page-header">
        <h2>Patient Profiles</h2>
        <p>Create a profile for each patient, then run and review their return-to-work assessments.</p>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {editingId !== null && (
        <form className="panel" onSubmit={save}>
          <h3>{editingId === "new" ? "New patient profile" : "Edit patient profile"}</h3>
          <div className="field-grid">
            <div className="field">
              <label htmlFor="patient_code">Patient ID *</label>
              <input id="patient_code" value={form.patient_code} required maxLength={40}
                     onChange={(e) => setForm({ ...form, patient_code: e.target.value })} />
              <span className="hint">Your clinic's record number, e.g. MRN-1042.</span>
            </div>
            <div className="field">
              <label htmlFor="patient_name">Patient name *</label>
              <input id="patient_name" value={form.full_name} required maxLength={120}
                     onChange={(e) => setForm({ ...form, full_name: e.target.value })} />
            </div>
            <div className="field">
              <label htmlFor="linked_username">Linked patient account</label>
              <input id="linked_username" value={form.linked_username}
                     onChange={(e) => setForm({ ...form, linked_username: e.target.value })} />
              <span className="hint">Optional. Username of the patient's own login, so they can view their results.</span>
            </div>
            <div className="field">
              <label htmlFor="patient_notes">Notes</label>
              <textarea id="patient_notes" rows={3} value={form.notes} maxLength={2000}
                        onChange={(e) => setForm({ ...form, notes: e.target.value })} />
            </div>
          </div>
          {formError && <div className="error-banner" style={{ marginTop: 18 }}>{formError}</div>}
          <div className="form-actions">
            <button type="button" className="btn ghost" onClick={() => setEditingId(null)}>Cancel</button>
            <button type="submit" className="btn primary" disabled={saving}>
              {saving ? "Saving…" : "Save profile"}
            </button>
          </div>
        </form>
      )}

      <div className="panel">
        <div className="panel-toolbar">
          <input className="search" placeholder="Search by patient ID or name" value={search}
                 onChange={(e) => setSearch(e.target.value)} aria-label="Search patients" />
          <button className="btn primary" onClick={openNew}>New patient</button>
        </div>

        {shown.length === 0 ? (
          <div className="empty-state">
            {patients?.length ? "No patients match that search." : "No patient profiles yet. Create one to start an assessment."}
          </div>
        ) : (
          <table className="history">
            <thead>
              <tr>
                <th>Patient ID</th>
                <th>Name</th>
                <th>Linked account</th>
                <th>Assessments</th>
                <th>Last updated</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {shown.map((p) => (
                <tr key={p.id}>
                  <td>{p.patient_code}</td>
                  <td>{p.full_name}</td>
                  <td>{p.linked_username || "—"}</td>
                  <td>{p.assessment_count}</td>
                  <td>{new Date(p.updated_at).toLocaleDateString()}</td>
                  <td className="row-actions">
                    <button className="link" onClick={() => onPredict(p.id)}>New prediction</button>
                    <button className="link" disabled={!p.assessment_count} onClick={() => onViewHistory(p.id)}>History</button>
                    <button className="link" onClick={() => openEdit(p)}>Edit</button>
                    <button className="link danger" onClick={() => remove(p)}>Remove</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
