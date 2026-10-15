// In development the API runs separately on port 8000. A production build is
// served by the API itself (see main.py), so it calls its own address.
const BASE_URL = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? "http://localhost:8000" : "");
const TOKEN_KEY = "sci_rtw_auth_token";

let token = null;
try {
  token = localStorage.getItem(TOKEN_KEY);
} catch {
  token = null;
}
let onSessionExpired = null;

function setToken(value) {
  token = value;
  try {
    if (value) localStorage.setItem(TOKEN_KEY, value);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    // storage unavailable: the session just won't survive a reload
  }
}

// FastAPI sends `detail` as a string for our own errors and as a list of
// {loc, msg} objects for request-validation (422) errors.
function errorMessage(body, status) {
  const d = body?.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) {
    return d.map((e) => `${(e.loc || []).slice(1).join(".")}: ${e.msg}`).join("; ");
  }
  return `Request failed (${status})`;
}

async function request(path, options = {}) {
  const headers = { "Content-Type": "application/json" };
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(`${BASE_URL}${path}`, { ...options, headers });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    if (res.status === 401 && token && path !== "/auth/login") {
      setToken(null);
      if (onSessionExpired) onSessionExpired();
    }
    throw new Error(errorMessage(body, res.status));
  }
  return res.json();
}

const send = (method, path, body) =>
  request(path, { method, body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  hasToken: () => Boolean(token),
  setSessionExpiredHandler: (fn) => { onSessionExpired = fn; },

  health: () => request("/health"),
  categoryOptions: () => request("/category-options"),
  modelInfo: () => request("/model-info"),

  login: async (username, password) => {
    const res = await send("POST", "/auth/login", { username, password });
    setToken(res.token);
    return res.user;
  },
  register: (data) => send("POST", "/auth/register", data),
  me: () => request("/auth/me"),
  changePassword: (current_password, new_password) =>
    send("POST", "/auth/change-password", { current_password, new_password }),
  logout: async () => {
    try {
      await send("POST", "/auth/logout");
    } finally {
      setToken(null);
    }
  },

  listPatients: () => request("/patients"),
  createPatient: (data) => send("POST", "/patients", data),
  updatePatient: (id, data) => send("PUT", `/patients/${id}`, data),
  deletePatient: (id) => send("DELETE", `/patients/${id}`),

  savePrediction: (patientId, patient) => send("POST", `/patients/${patientId}/predictions`, patient),
  listPredictions: () => request("/predictions"),
  updateNotes: (predictionId, clinician_notes) =>
    send("PATCH", `/predictions/${predictionId}/notes`, { clinician_notes }),

  listUsers: () => request("/users"),
  createUser: (data) => send("POST", "/users", data),
  updateUser: (id, data) => send("PATCH", `/users/${id}`, data),
  deleteUser: (id) => send("DELETE", `/users/${id}`),
  auditLog: () => request("/audit-log"),
};
