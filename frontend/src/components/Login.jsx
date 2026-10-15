import React, { useState } from "react";
import { api } from "../api.js";

const REGISTER_ROLES = [
  { value: "doctor", label: "Doctor" },
  { value: "therapist", label: "Therapist" },
  { value: "patient", label: "Patient" },
];

const EMPTY = { full_name: "", username: "", password: "", role: "doctor" };

export default function Login({ onSignedIn }) {
  const [mode, setMode] = useState("login"); // login | register
  const [form, setForm] = useState(EMPTY);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);

  function update(name, value) {
    setForm((f) => ({ ...f, [name]: value }));
  }

  function switchMode(next) {
    setMode(next);
    setError(null);
    setNotice(null);
  }

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      if (mode === "register") {
        const res = await api.register(form);
        if (res.pending_approval) {
          // Clinician accounts can read every patient record, so the
          // administrator has to approve them before first login.
          setNotice("Account created. An administrator must approve it before you can sign in.");
          setMode("login");
          setForm((f) => ({ ...EMPTY, username: f.username }));
          return;
        }
      }
      onSignedIn(await api.login(form.username, form.password));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const isRegister = mode === "register";

  return (
    <div>
      <div className="auth-brand">
        <h1>SCI-RTW Predict</h1>
        <p>Return-to-work decision support for spinal cord injury rehabilitation</p>
      </div>

      <div className="model-tabs">
        <button type="button" className={`model-tab ${!isRegister ? "active" : ""}`} onClick={() => switchMode("login")}>
          Sign in
        </button>
        <button type="button" className={`model-tab ${isRegister ? "active" : ""}`} onClick={() => switchMode("register")}>
          Register
        </button>
      </div>

      {notice && <div className="disclaimer">{notice}</div>}
      {error && <div className="error-banner">{error}</div>}

      <form onSubmit={submit} className="auth-form">
        {isRegister && (
          <div className="field">
            <label htmlFor="full_name">Full name</label>
            <input id="full_name" value={form.full_name} required
                   onChange={(e) => update("full_name", e.target.value)} />
          </div>
        )}
        <div className="field">
          <label htmlFor="username">Username</label>
          <input id="username" value={form.username} required autoComplete="username"
                 onChange={(e) => update("username", e.target.value)} />
        </div>
        <div className="field">
          <label htmlFor="password">Password</label>
          <input id="password" type="password" value={form.password} required
                 minLength={isRegister ? 8 : undefined}
                 autoComplete={isRegister ? "new-password" : "current-password"}
                 onChange={(e) => update("password", e.target.value)} />
          {isRegister && <span className="hint">At least 8 characters.</span>}
        </div>
        {isRegister && (
          <div className="field">
            <label htmlFor="role">I am a</label>
            <select id="role" value={form.role} onChange={(e) => update("role", e.target.value)}>
              {REGISTER_ROLES.map((r) => <option key={r.value} value={r.value}>{r.label}</option>)}
            </select>
            <span className="hint">
              {form.role === "patient"
                ? "Your clinician links your account to your record so you can see your results."
                : "Doctor and therapist accounts need administrator approval before first sign-in."}
            </span>
          </div>
        )}
        <button type="submit" className="btn primary" disabled={busy}>
          {busy ? "Please wait…" : isRegister ? "Create account" : "Sign in"}
        </button>
      </form>
    </div>
  );
}
