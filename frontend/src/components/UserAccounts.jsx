import React, { useEffect, useState } from "react";
import { api } from "../api.js";

const ROLES = [
  { value: "doctor", label: "Doctor" },
  { value: "therapist", label: "Therapist" },
  { value: "patient", label: "Patient" },
  { value: "admin", label: "Administrator" },
];
const ROLE_LABEL = Object.fromEntries(ROLES.map((r) => [r.value, r.label]));

const EMPTY = { username: "", full_name: "", role: "doctor", password: "", is_active: true };

const ACTION_LABEL = {
  "login.success": "Signed in",
  "login.failed": "Failed sign-in",
  "login.locked": "Sign-in locked after repeated failures",
  "password.changed": "Changed own password",
  "account.register": "Registered",
  "account.created": "Created account",
  "account.updated": "Updated account",
  "account.removed": "Removed account",
  "patient.created": "Created patient profile",
  "patient.updated": "Updated patient profile",
  "patient.removed": "Removed patient profile",
  "prediction.saved": "Saved prediction",
  "prediction.notes_updated": "Updated clinician notes",
};

export default function UserAccounts({ currentUser }) {
  const [users, setUsers] = useState(null);
  const [error, setError] = useState(null);
  const [editingId, setEditingId] = useState(null); // null = closed, "new" = creating, number = editing
  const [form, setForm] = useState(EMPTY);
  const [formError, setFormError] = useState(null);
  const [saving, setSaving] = useState(false);
  const [activity, setActivity] = useState([]);

  function load() {
    // The log is refreshed with the accounts, so an action just taken shows up.
    api.auditLog().then(setActivity).catch(() => setActivity([]));
    return api.listUsers().then(setUsers).catch((e) => setError(e.message));
  }

  useEffect(() => { load(); }, []);

  function openNew() {
    setForm(EMPTY);
    setFormError(null);
    setEditingId("new");
  }

  function openEdit(u) {
    setForm({ username: u.username, full_name: u.full_name, role: u.role, password: "", is_active: u.is_active });
    setFormError(null);
    setEditingId(u.id);
  }

  async function run(action) {
    setError(null);
    try {
      await action();
      await load();
    } catch (err) {
      setError(err.message);
    }
  }

  async function save(e) {
    e.preventDefault();
    setSaving(true);
    setFormError(null);
    try {
      if (editingId === "new") {
        await api.createUser(form);
      } else {
        const changes = { full_name: form.full_name, role: form.role, is_active: form.is_active };
        if (form.password) changes.password = form.password;
        await api.updateUser(editingId, changes);
      }
      setEditingId(null);
      await load();
    } catch (err) {
      setFormError(err.message);
    } finally {
      setSaving(false);
    }
  }

  function remove(u) {
    if (!window.confirm(`Remove the account "${u.username}"? This cannot be undone.`)) return;
    run(() => api.deleteUser(u.id));
  }

  if (!users && !error) return <p className="loading-note">Loading user accounts…</p>;

  const pending = (users || []).filter((u) => !u.is_active);
  const isNew = editingId === "new";

  return (
    <div>
      <div className="page-header">
        <h2>User Accounts</h2>
        <p>Approve new clinician registrations and manage who can access the system and in which role.</p>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {pending.length > 0 && (
        <div className="disclaimer">
          {pending.length} account{pending.length !== 1 ? "s are" : " is"} inactive or awaiting approval.
        </div>
      )}

      {editingId !== null && (
        <form className="panel" onSubmit={save}>
          <h3>{isNew ? "New user account" : `Edit account: ${form.username}`}</h3>
          <div className="field-grid">
            {isNew && (
              <div className="field">
                <label htmlFor="u_username">Username *</label>
                <input id="u_username" value={form.username} required minLength={3} maxLength={40}
                       pattern="[A-Za-z0-9_.\-]+"
                       onChange={(e) => setForm({ ...form, username: e.target.value })} />
                <span className="hint">Letters, numbers, dot, dash and underscore.</span>
              </div>
            )}
            <div className="field">
              <label htmlFor="u_full_name">Full name *</label>
              <input id="u_full_name" value={form.full_name} required maxLength={120}
                     onChange={(e) => setForm({ ...form, full_name: e.target.value })} />
            </div>
            <div className="field">
              <label htmlFor="u_role">Role *</label>
              <select id="u_role" value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })}>
                {ROLES.map((r) => <option key={r.value} value={r.value}>{r.label}</option>)}
              </select>
            </div>
            <div className="field">
              <label htmlFor="u_password">{isNew ? "Password *" : "New password"}</label>
              <input id="u_password" type="password" value={form.password} required={isNew} minLength={8}
                     autoComplete="new-password"
                     onChange={(e) => setForm({ ...form, password: e.target.value })} />
              <span className="hint">
                {isNew ? "At least 8 characters." : "Leave blank to keep the current password."}
              </span>
            </div>
            <div className="field">
              <label className="check">
                <input type="checkbox" checked={form.is_active}
                       onChange={(e) => setForm({ ...form, is_active: e.target.checked })} />
                Account is active (can sign in)
              </label>
            </div>
          </div>
          {formError && <div className="error-banner" style={{ marginTop: 18 }}>{formError}</div>}
          <div className="form-actions">
            <button type="button" className="btn ghost" onClick={() => setEditingId(null)}>Cancel</button>
            <button type="submit" className="btn primary" disabled={saving}>
              {saving ? "Saving…" : "Save account"}
            </button>
          </div>
        </form>
      )}

      <div className="panel">
        <div className="panel-toolbar">
          <span className="sub" style={{ margin: 0 }}>{users?.length || 0} accounts</span>
          <button className="btn primary" onClick={openNew}>New user</button>
        </div>
        <table className="history">
          <thead>
            <tr>
              <th>Username</th>
              <th>Full name</th>
              <th>Role</th>
              <th>Status</th>
              <th>Created</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {(users || []).map((u) => {
              const isSelf = u.id === currentUser.id;
              return (
                <tr key={u.id}>
                  <td>{u.username}{isSelf && " (you)"}</td>
                  <td>{u.full_name}</td>
                  <td>{ROLE_LABEL[u.role]}</td>
                  <td>
                    <span className={`pill ${u.is_active ? "rtw" : "non-rtw"}`}>
                      {u.is_active ? "Active" : "Inactive"}
                    </span>
                  </td>
                  <td>{new Date(u.created_at).toLocaleDateString()}</td>
                  <td className="row-actions">
                    {!u.is_active && (
                      <button className="link" onClick={() => run(() => api.updateUser(u.id, { is_active: true }))}>
                        Approve
                      </button>
                    )}
                    {u.is_active && !isSelf && (
                      <button className="link" onClick={() => run(() => api.updateUser(u.id, { is_active: false }))}>
                        Deactivate
                      </button>
                    )}
                    <button className="link" onClick={() => openEdit(u)}>Edit</button>
                    {!isSelf && <button className="link danger" onClick={() => remove(u)}>Remove</button>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="panel">
        <h3>Activity log</h3>
        <p className="sub">
          The last {activity.length} recorded actions: sign-ins, account changes and changes to patient records.
        </p>
        <table className="activity-log">
          <thead>
            <tr>
              <th>When</th>
              <th>User</th>
              <th>Action</th>
              <th>Record</th>
              <th>Detail</th>
            </tr>
          </thead>
          <tbody>
            {activity.map((a) => (
              <tr key={a.id}>
                <td>{new Date(a.at).toLocaleString()}</td>
                <td>{a.username || "—"}</td>
                <td>{ACTION_LABEL[a.action] || a.action}</td>
                <td>{a.target}</td>
                <td>{a.detail}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
