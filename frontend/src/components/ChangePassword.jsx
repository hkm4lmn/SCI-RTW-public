import React, { useState } from "react";
import { api } from "../api.js";

// Shown instead of the app while the account is still on the built-in
// default password.
export default function ChangePassword({ user, onChanged, onCancel }) {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [repeat, setRepeat] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  async function submit(e) {
    e.preventDefault();
    if (next !== repeat) {
      setError("The two new passwords do not match.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await api.changePassword(current, next);
      await onChanged();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <div className="auth-brand">
        <h1>Choose a new password</h1>
        <p>
          The account "{user.username}" is still using the default password. Change it to continue.
        </p>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <form onSubmit={submit} className="auth-form">
        <div className="field">
          <label htmlFor="current_password">Current password</label>
          <input id="current_password" type="password" value={current} required
                 autoComplete="current-password" onChange={(e) => setCurrent(e.target.value)} />
        </div>
        <div className="field">
          <label htmlFor="new_password">New password</label>
          <input id="new_password" type="password" value={next} required minLength={8}
                 autoComplete="new-password" onChange={(e) => setNext(e.target.value)} />
          <span className="hint">At least 8 characters.</span>
        </div>
        <div className="field">
          <label htmlFor="repeat_password">New password again</label>
          <input id="repeat_password" type="password" value={repeat} required minLength={8}
                 autoComplete="new-password" onChange={(e) => setRepeat(e.target.value)} />
        </div>
        <button type="submit" className="btn primary" disabled={busy}>
          {busy ? "Please wait…" : "Change password"}
        </button>
        <button type="button" className="btn ghost" onClick={onCancel}>Sign out</button>
      </form>
    </div>
  );
}
