import React, { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import Login from "./components/Login.jsx";
import ChangePassword from "./components/ChangePassword.jsx";
import Dashboard from "./components/Dashboard.jsx";
import Patients from "./components/Patients.jsx";
import PredictionForm from "./components/PredictionForm.jsx";
import ModelInsights from "./components/ModelInsights.jsx";
import PatientHistory from "./components/PatientHistory.jsx";
import UserAccounts from "./components/UserAccounts.jsx";
import Icon from "./components/icons.jsx";

// What each role can reach, per the proposal's use case diagram (Fig. 3.4).
const CLINICIAN_NAV = [
  { key: "dashboard", label: "Dashboard" },
  { key: "patients", label: "Patients" },
  { key: "predict", label: "New Prediction" },
  { key: "insights", label: "Model Insights" },
  { key: "history", label: "Patient History" },
];
const NAV_BY_ROLE = {
  doctor: CLINICIAN_NAV,
  therapist: CLINICIAN_NAV,
  patient: [{ key: "history", label: "My Results" }],
  admin: [
    { key: "users", label: "User Accounts" },
    { key: "insights", label: "Model Insights" },
  ],
};

const ROLE_LABELS = { doctor: "Doctor", therapist: "Therapist", patient: "Patient", admin: "Administrator" };

export default function App() {
  const [user, setUser] = useState(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [page, setPage] = useState(null);
  const [apiStatus, setApiStatus] = useState("checking"); // checking | ok | down
  const [history, setHistory] = useState([]);
  const [historyError, setHistoryError] = useState(null);
  const [predictPatientId, setPredictPatientId] = useState(null);
  const [historyPatientId, setHistoryPatientId] = useState(null);

  const signedIn = useCallback((u) => {
    setUser(u);
    setPage(NAV_BY_ROLE[u.role][0].key);
  }, []);

  const signedOut = useCallback(() => {
    setUser(null);
    setPage(null);
    setHistory([]);
  }, []);

  // Restore the session after a page reload.
  useEffect(() => {
    api.setSessionExpiredHandler(signedOut);
    if (!api.hasToken()) {
      setAuthChecked(true);
      return;
    }
    // `stale` stops a superseded run (React StrictMode mounts effects twice
    // in development) from resetting the page after the user has navigated.
    let stale = false;
    api.me()
      .then((u) => { if (!stale) signedIn(u); })
      .catch(() => {})
      .finally(() => { if (!stale) setAuthChecked(true); });
    return () => { stale = true; };
  }, [signedIn, signedOut]);

  useEffect(() => {
    api.health().then(() => setApiStatus("ok")).catch(() => setApiStatus("down"));
  }, [page, user]);

  const refreshHistory = useCallback(() => {
    return api.listPredictions()
      .then((rows) => { setHistory(rows); setHistoryError(null); })
      .catch((e) => setHistoryError(e.message));
  }, []);

  const role = user?.role;
  useEffect(() => {
    if (role && role !== "admin") refreshHistory();
  }, [role, refreshHistory]);

  function logout() {
    api.logout().catch(() => {}).finally(signedOut);
  }

  function navigate(key) {
    setPredictPatientId(null);
    setHistoryPatientId(null);
    setPage(key);
  }

  const apiDownBanner = apiStatus === "down" && (
    <div className="error-banner">
      Can't reach the prediction API at the configured URL. Make sure the
      backend is running (<code>uvicorn main:app --reload --port 8000</code>
      {" "}from the <code>api/</code> folder).
    </div>
  );

  if (!authChecked) return <p className="loading-note" style={{ padding: 40 }}>Loading…</p>;

  if (!user) {
    return (
      <div className="auth-shell">
        <div className="auth-card">
          {apiDownBanner}
          <Login onSignedIn={signedIn} />
        </div>
      </div>
    );
  }

  // An account still on the built-in default password can do nothing else
  // until it is changed (the API enforces the same rule).
  if (user.must_change_password) {
    return (
      <div className="auth-shell">
        <div className="auth-card">
          <ChangePassword user={user} onChanged={() => api.me().then(signedIn)} onCancel={logout} />
        </div>
      </div>
    );
  }

  const navItems = NAV_BY_ROLE[user.role];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <h1>SCI-RTW Predict</h1>
          <p>Clinical decision support</p>
        </div>
        <nav>
          {navItems.map((item) => (
            <button
              key={item.key}
              className={page === item.key ? "active" : ""}
              onClick={() => navigate(item.key)}
            >
              <Icon name={item.key} />
              {item.label}
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className="sidebar-user">
            <strong>{user.full_name}</strong>
            <span>{ROLE_LABELS[user.role]}</span>
          </div>
          <button className="logout" onClick={logout}><Icon name="logout" />Log out</button>
          <div className="api-status">
            <span className={`status-dot ${apiStatus === "ok" ? "ok" : "down"}`} />
            {apiStatus === "checking" && "Checking API…"}
            {apiStatus === "ok" && "Model API connected"}
            {apiStatus === "down" && "API unreachable"}
          </div>
        </div>
      </aside>

      <main className="main">
        {apiDownBanner}
        {historyError && page !== "users" && page !== "insights" && (
          <div className="error-banner">Couldn't load prediction history: {historyError}</div>
        )}
        {page === "dashboard" && <Dashboard history={history} onNavigate={navigate} />}
        {page === "patients" && (
          <Patients
            onPredict={(id) => { setPredictPatientId(id); setPage("predict"); }}
            onViewHistory={(id) => { setHistoryPatientId(id); setPage("history"); }}
            onChanged={refreshHistory}
          />
        )}
        {page === "predict" && (
          <PredictionForm key={predictPatientId ?? "new"} initialPatientId={predictPatientId} onSaved={refreshHistory} />
        )}
        {page === "insights" && <ModelInsights />}
        {page === "history" && (
          <PatientHistory
            key={historyPatientId ?? "all"}
            history={history}
            role={user.role}
            initialPatientId={historyPatientId}
            onChanged={refreshHistory}
          />
        )}
        {page === "users" && <UserAccounts currentUser={user} />}
      </main>
    </div>
  );
}
