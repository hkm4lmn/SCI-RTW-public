"""
SCI-RTW Predictor: application database (SQLite)
Chapter 3, Section 3.4.5 (Entity Relationship Diagram) implemented.

Tables map to the entities in Table 3.10:
    users               - User Account        (login details, role, access)
    patients            - Patient Profile     (patient background information)
    assessments         - Clinical Assessment (the SCI input variables entered)
    predictions         - Prediction Record   (RTW output, probability, timing, date)
    prediction_reports  - Prediction Report   (important factors, confidence level,
                                               suggested considerations, clinician notes)

Two further tables support security rather than the ERD:
    login_failures      - consecutive failed sign-ins per username (lockout)
    audit_log           - who did what and when (sign-ins, account changes,
                          patient records created, changed or removed)

The sixth entity, Model Evaluation, is not a table: it is produced by
train_models.py and served read-only from models/layer{1,2}_results.json,
so there is exactly one source of truth for the reported metrics.

Uses only the standard library (sqlite3) - no extra dependency to install.
The file location can be overridden with the SCI_RTW_DB environment variable
(the automated tests point it at a temporary file).
"""

import os
import sqlite3
from datetime import datetime, timezone

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sci_rtw.db")

ROLES = ("doctor", "therapist", "patient", "admin")
CLINICIAN_ROLES = ("doctor", "therapist")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    full_name     TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('doctor', 'therapist', 'patient', 'admin')),
    password_hash TEXT NOT NULL,
    password_salt TEXT NOT NULL,
    is_active     INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS patients (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_code TEXT NOT NULL UNIQUE COLLATE NOCASE,
    full_name    TEXT NOT NULL,
    notes        TEXT NOT NULL DEFAULT '',
    user_id      INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_by   INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assessments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    entered_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    input_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_id             INTEGER NOT NULL UNIQUE REFERENCES assessments(id) ON DELETE CASCADE,
    rtw_status                TEXT NOT NULL,
    rtw_probability           REAL NOT NULL,
    rtw_timing                TEXT,
    timing_probabilities_json TEXT,
    model_used                TEXT NOT NULL,
    created_at                TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prediction_reports (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id   INTEGER NOT NULL UNIQUE REFERENCES predictions(id) ON DELETE CASCADE,
    factors_json    TEXT NOT NULL,
    clinician_notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS login_failures (
    username     TEXT PRIMARY KEY COLLATE NOCASE,
    failures     INTEGER NOT NULL,
    locked_until TEXT,
    updated_at   TEXT NOT NULL
);

-- user_id is deliberately not a foreign key: the entry must outlive the
-- account, so the username is stored alongside it.
CREATE TABLE IF NOT EXISTS audit_log (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    at       TEXT NOT NULL,
    user_id  INTEGER,
    username TEXT NOT NULL DEFAULT '',
    action   TEXT NOT NULL,
    target   TEXT NOT NULL DEFAULT '',
    detail   TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_assessments_patient ON assessments(patient_id);
CREATE INDEX IF NOT EXISTS idx_patients_user ON patients(user_id);
"""


def db_path() -> str:
    return os.environ.get("SCI_RTW_DB", DEFAULT_DB_PATH)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    # check_same_thread=False: FastAPI may run the get_db dependency and the
    # endpoint that uses it on different worker threads. That is safe here
    # because a connection belongs to exactly one request and is never used
    # by two threads at once (see test_concurrent_requests_do_not_fail).
    conn = sqlite3.connect(db_path(), check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    # SQLite leaves foreign keys off unless asked, per connection.
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def get_db():
    """FastAPI dependency: one connection per request, committed on success."""
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# Columns added after the first release. CREATE TABLE IF NOT EXISTS leaves an
# existing table alone, so a database made by an earlier version needs these
# added explicitly. Adding a nullable column keeps every existing row.
ADDED_COLUMNS = [
    ("prediction_reports", "confidence_json", "TEXT"),
    ("prediction_reports", "recommendations_json", "TEXT"),
    ("users", "must_change_password", "INTEGER NOT NULL DEFAULT 0"),
    ("predictions", "model_version", "TEXT"),
]


def record_audit(conn, actor, action, target="", detail=""):
    """Add an entry to the audit log. `actor` is the signed-in user, or None
    for something done without a session (a failed sign-in). The entry is
    committed with the request, so an action that fails leaves no entry
    unless the caller commits first."""
    conn.execute(
        "INSERT INTO audit_log (at, user_id, username, action, target, detail) VALUES (?, ?, ?, ?, ?, ?)",
        (now_iso(), actor["id"] if actor else None, actor["username"] if actor else "", action, target, detail))


def init_db():
    conn = connect()
    try:
        conn.executescript(SCHEMA)
        for table, column, definition in ADDED_COLUMNS:
            existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        conn.commit()
    finally:
        conn.close()
