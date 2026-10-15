"""
SCI-RTW Predictor - tests for the security protections: sign-in lockout,
the forced change of the default administrator password, session
revocation, the audit log, and the browser-facing headers.

Run from the project root:
    python -m pytest tests -v
"""
import os
import sqlite3

from conftest import ADMIN, login_headers, make_user


def db():
    conn = sqlite3.connect(os.environ["SCI_RTW_DB"])
    conn.row_factory = sqlite3.Row
    return conn


def attempt(client, username, password):
    return client.post("/auth/login", json={"username": username, "password": password})


def audit_entries(client, admin_headers, action):
    rows = client.get("/audit-log?limit=500", headers=admin_headers).json()
    return [r for r in rows if r["action"] == action]


# ---------------- sign-in lockout ----------------
def test_five_wrong_passwords_lock_the_username(client):
    make_user(client, "sec_lock", "doctor")
    for _ in range(5):
        assert attempt(client, "sec_lock", "wrong-password").status_code == 401
    # Locked: even the right password is refused, and says why.
    r = attempt(client, "sec_lock", "password123")
    assert r.status_code == 429 and "Too many failed" in r.json()["detail"]
    # Another account is unaffected.
    make_user(client, "sec_lock_other", "doctor")


def test_lock_ends_after_the_lockout_period(client):
    make_user(client, "sec_lock_expiry", "doctor")
    for _ in range(5):
        attempt(client, "sec_lock_expiry", "wrong-password")
    assert attempt(client, "sec_lock_expiry", "password123").status_code == 429
    conn = db()
    conn.execute("UPDATE login_failures SET locked_until = '2000-01-01T00:00:00+00:00', "
                 "updated_at = '2000-01-01T00:00:00+00:00' WHERE username = 'sec_lock_expiry'")
    conn.commit()
    conn.close()
    assert attempt(client, "sec_lock_expiry", "password123").status_code == 200


def test_successful_sign_in_resets_the_failure_count(client):
    make_user(client, "sec_reset", "doctor")
    for _ in range(2):
        for _ in range(4):
            assert attempt(client, "sec_reset", "wrong-password").status_code == 401
        assert attempt(client, "sec_reset", "password123").status_code == 200


def test_unknown_username_is_locked_the_same_way(client):
    # Otherwise the difference would reveal which usernames exist.
    for _ in range(5):
        assert attempt(client, "sec_no_such_user", "wrong-password").status_code == 401
    assert attempt(client, "sec_no_such_user", "wrong-password").status_code == 429


def test_oversized_login_fields_are_refused(client):
    assert attempt(client, "x" * 41, "password123").status_code == 422
    assert attempt(client, "admin", "x" * 129).status_code == 422


# ---------------- default administrator password ----------------
def test_account_on_default_password_can_only_change_it(client, admin_headers):
    make_user(client, "sec_default", "admin")
    conn = db()
    conn.execute("UPDATE users SET must_change_password = 1 WHERE username = 'sec_default'")
    conn.commit()
    conn.close()

    r = attempt(client, "sec_default", "password123")
    assert r.json()["user"]["must_change_password"] is True
    headers = {"Authorization": f"Bearer {r.json()['token']}"}
    assert client.get("/auth/me", headers=headers).status_code == 200
    blocked = client.get("/users", headers=headers)
    assert blocked.status_code == 403 and "Change your password" in blocked.json()["detail"]
    assert client.get("/model-info", headers=headers).status_code == 403

    same = client.post("/auth/change-password", headers=headers,
                       json={"current_password": "password123", "new_password": "password123"})
    assert same.status_code == 422
    changed = client.post("/auth/change-password", headers=headers,
                          json={"current_password": "password123", "new_password": "a-better-password"})
    assert changed.status_code == 200
    assert client.get("/auth/me", headers=headers).json()["must_change_password"] is False
    assert client.get("/users", headers=headers).status_code == 200


def test_default_admin_is_flagged_only_when_on_the_built_in_password(tmp_path, monkeypatch):
    import auth
    import db as appdb

    def admin_flag():
        conn = appdb.connect()
        try:
            return conn.execute("SELECT must_change_password FROM users WHERE username = 'admin'").fetchone()[0]
        finally:
            conn.close()

    # No password supplied: the built-in default is used and must be changed.
    monkeypatch.setenv("SCI_RTW_DB", str(tmp_path / "default.db"))
    monkeypatch.delenv("SCI_RTW_ADMIN_PASSWORD")
    appdb.init_db()
    auth.ensure_default_admin()
    assert admin_flag() == 1

    # A password supplied by the operator is left alone.
    monkeypatch.setenv("SCI_RTW_DB", str(tmp_path / "chosen.db"))
    monkeypatch.setenv("SCI_RTW_ADMIN_PASSWORD", "operator-chosen-pw")
    appdb.init_db()
    auth.ensure_default_admin()
    assert admin_flag() == 0


# ---------------- sessions ----------------
def test_changing_password_signs_out_other_sessions(client):
    first = make_user(client, "sec_sessions", "doctor")
    second = login_headers(client, "sec_sessions", "password123")
    r = client.post("/auth/change-password", headers=first,
                    json={"current_password": "password123", "new_password": "another-password"})
    assert r.status_code == 200
    assert client.get("/auth/me", headers=first).status_code == 200
    assert client.get("/auth/me", headers=second).status_code == 401


# ---------------- audit log ----------------
def test_audit_log_is_for_administrators_only(client, admin_headers, doctor_headers):
    assert client.get("/audit-log", headers=admin_headers).status_code == 200
    assert client.get("/audit-log", headers=doctor_headers).status_code == 403
    assert client.get("/audit-log").status_code == 401


def test_removing_a_patient_profile_is_recorded_with_who_did_it(client, admin_headers, doctor_headers):
    profile = client.post("/patients", headers=doctor_headers,
                          json={"patient_code": "SEC-AUDIT-1", "full_name": "Audit Patient"}).json()
    assert client.delete(f"/patients/{profile['id']}", headers=doctor_headers).status_code == 200
    removed = [r for r in audit_entries(client, admin_headers, "patient.removed") if r["target"] == "SEC-AUDIT-1"]
    assert len(removed) == 1 and removed[0]["username"] == "dr_test"
    created = [r for r in audit_entries(client, admin_headers, "patient.created") if r["target"] == "SEC-AUDIT-1"]
    assert len(created) == 1


def test_failed_sign_in_is_recorded_although_the_request_fails(client, admin_headers):
    assert attempt(client, "sec_audit_fail", "wrong-password").status_code == 401
    assert any(r["target"] == "sec_audit_fail" for r in audit_entries(client, admin_headers, "login.failed"))


def test_account_changes_are_recorded_and_outlive_the_account(client, admin_headers):
    created = client.post("/users", headers=admin_headers, json={
        "username": "sec_audit_user", "password": "password123", "role": "patient", "full_name": "Audit User"}).json()
    client.patch(f"/users/{created['id']}", headers=admin_headers, json={"is_active": False})
    client.delete(f"/users/{created['id']}", headers=admin_headers)
    for action in ("account.created", "account.updated", "account.removed"):
        assert any(r["target"] == "sec_audit_user" and r["username"] == ADMIN["username"]
                   for r in audit_entries(client, admin_headers, action)), action
    update = [r for r in audit_entries(client, admin_headers, "account.updated") if r["target"] == "sec_audit_user"]
    assert update[0]["detail"] == "deactivated"


def test_audit_log_never_contains_a_password(client, admin_headers):
    make_user(client, "sec_audit_pw", "doctor", password="very-secret-pw-1")
    attempt(client, "sec_audit_pw", "a-wrong-secret-pw")
    text = client.get("/audit-log?limit=500", headers=admin_headers).text
    assert "very-secret-pw-1" not in text and "a-wrong-secret-pw" not in text


# ---------------- browser-facing headers ----------------
def test_only_a_local_frontend_may_call_the_api_from_a_browser(client):
    local = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert local.headers.get("access-control-allow-origin") == "http://localhost:5173"
    other = client.get("/health", headers={"Origin": "https://example.com"})
    assert "access-control-allow-origin" not in other.headers
    lookalike = client.get("/health", headers={"Origin": "http://localhost.example.com"})
    assert "access-control-allow-origin" not in lookalike.headers


def test_responses_carry_security_headers(client, doctor_headers):
    r = client.get("/patients", headers=doctor_headers)
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["cache-control"] == "no-store"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    # The interactive API page loads its script from a CDN, so no policy there.
    assert "content-security-policy" not in client.get("/docs").headers
