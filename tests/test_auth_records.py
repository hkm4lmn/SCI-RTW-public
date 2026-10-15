"""
SCI-RTW Predictor - tests for login, roles, patient profiles and saved history.
Covers TC01 (user login / role-based access) and TC06 (prediction history
stored server-side), plus FR2 (manage patient profile) and the
Administrator's "Manage User Account" use case.

Run from the project root:
    python -m pytest tests -v
"""
import sqlite3
import os

from conftest import login_headers, make_user

PATIENT_DATA = {
    "sex": "1.0", "age_at_injury": 18, "education_level": "3.0", "marital_status": "1.0",
    "veteran_status": "0.0", "pre_injury_employment_status": "1.0", "injury_cause": "1.0",
    "vertebral_injury_level": "1.0", "associated_injury": "0.0", "work_related_injury": "0.0",
    "neuro_level_discharge": "L03", "aisgrade_discharge": "D",
    "injury_completeness_discharge": "1.0", "asia_motor_total_discharge": 92,
}


def new_profile(client, headers, code, **extra):
    r = client.post("/patients", headers=headers, json={"patient_code": code, "full_name": f"Patient {code}", **extra})
    assert r.status_code == 201, r.text
    return r.json()


# ---------------- TC01: user login ----------------
def test_tc01_login_returns_token_and_role(client):
    make_user(client, "tc01_doctor", "doctor")
    r = client.post("/auth/login", json={"username": "tc01_doctor", "password": "password123"})
    assert r.status_code == 200
    body = r.json()
    assert body["user"]["role"] == "doctor" and len(body["token"]) > 20
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {body['token']}"})
    assert me.json()["username"] == "tc01_doctor"


def test_tc01_wrong_password_and_unknown_user_rejected(client):
    make_user(client, "tc01_wrongpw", "doctor")
    assert client.post("/auth/login", json={"username": "tc01_wrongpw", "password": "nope-nope"}).status_code == 401
    assert client.post("/auth/login", json={"username": "nobody_here", "password": "password123"}).status_code == 401


def test_tc01_login_username_is_case_insensitive(client):
    make_user(client, "tc01_case", "doctor")
    assert client.post("/auth/login", json={"username": "TC01_Case", "password": "password123"}).status_code == 200


def test_tc01_protected_endpoints_reject_anonymous(client):
    assert client.post("/predict", json=PATIENT_DATA).status_code == 401
    for path in ("/patients", "/predictions", "/model-info", "/users", "/auth/me"):
        assert client.get(path).status_code == 401, path
    assert client.get("/patients", headers={"Authorization": "Bearer not-a-real-token"}).status_code == 401


def test_tc01_public_endpoints_need_no_login(client):
    assert client.get("/health").status_code == 200
    assert client.get("/category-options").status_code == 200


def test_tc01_logout_invalidates_token(client):
    headers = make_user(client, "tc01_logout", "doctor")
    assert client.get("/auth/me", headers=headers).status_code == 200
    assert client.post("/auth/logout", headers=headers).status_code == 200
    assert client.get("/auth/me", headers=headers).status_code == 401


def test_tc01_access_is_based_on_role(client, admin_headers, doctor_headers, therapist_headers):
    patient_headers = make_user(client, "tc01_patient", "patient")
    # Patient: results/history only.
    assert client.get("/predictions", headers=patient_headers).status_code == 200
    assert client.get("/patients", headers=patient_headers).status_code == 403
    assert client.post("/predict", headers=patient_headers, json=PATIENT_DATA).status_code == 403
    assert client.get("/model-info", headers=patient_headers).status_code == 403
    assert client.get("/users", headers=patient_headers).status_code == 403
    # Doctor and therapist: clinical functions, not account management.
    for h in (doctor_headers, therapist_headers):
        assert client.get("/patients", headers=h).status_code == 200
        assert client.post("/predict", headers=h, json=PATIENT_DATA).status_code == 200
        assert client.get("/model-info", headers=h).status_code == 200
        assert client.get("/users", headers=h).status_code == 403
    # Administrator: accounts + model report, not patient records.
    assert client.get("/users", headers=admin_headers).status_code == 200
    assert client.get("/model-info", headers=admin_headers).status_code == 200
    assert client.get("/patients", headers=admin_headers).status_code == 403
    assert client.get("/predictions", headers=admin_headers).status_code == 403
    assert client.post("/predict", headers=admin_headers, json=PATIENT_DATA).status_code == 403


def test_concurrent_requests_do_not_fail(client, doctor_headers):
    """The browser fires several requests at once (dashboard loads history
    and model info together). FastAPI may run a request's database
    dependency and its endpoint on different worker threads, which SQLite
    refuses by default -- found by tests/ui_walkthrough.py, where it showed
    up as intermittent 500s. TestClient serves one request at a time and
    cannot reproduce it, so this test starts a real server."""
    import socket
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import httpx
    import uvicorn
    from main import app

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 15
        while not server.started:
            assert time.time() < deadline, "test server did not start"
            time.sleep(0.05)
        paths = ["/predictions", "/patients", "/auth/me", "/model-info"] * 15
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", headers=doctor_headers, timeout=30) as http:
            with ThreadPoolExecutor(max_workers=8) as pool:
                statuses = list(pool.map(lambda p: http.get(p).status_code, paths))
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    assert statuses == [200] * len(paths)


# ---------------- Registration ----------------
def test_register_patient_is_active_immediately(client):
    r = client.post("/auth/register", json={"username": "reg_patient", "password": "password123",
                                             "full_name": "Reg Patient", "role": "patient"})
    assert r.status_code == 201 and r.json()["pending_approval"] is False
    assert client.post("/auth/login", json={"username": "reg_patient", "password": "password123"}).status_code == 200


def test_register_clinician_needs_admin_approval(client, admin_headers):
    r = client.post("/auth/register", json={"username": "reg_doctor", "password": "password123",
                                             "full_name": "Reg Doctor", "role": "doctor"})
    assert r.status_code == 201 and r.json()["pending_approval"] is True
    creds = {"username": "reg_doctor", "password": "password123"}
    assert client.post("/auth/login", json=creds).status_code == 403

    user_id = r.json()["user"]["id"]
    assert client.patch(f"/users/{user_id}", headers=admin_headers, json={"is_active": True}).status_code == 200
    assert client.post("/auth/login", json=creds).status_code == 200


def test_register_cannot_self_assign_admin(client):
    r = client.post("/auth/register", json={"username": "reg_admin", "password": "password123",
                                             "full_name": "Sneaky", "role": "admin"})
    assert r.status_code == 422


def test_register_rejects_duplicate_username_and_weak_password(client):
    payload = {"username": "reg_dupe", "password": "password123", "full_name": "Dupe", "role": "patient"}
    assert client.post("/auth/register", json=payload).status_code == 201
    assert client.post("/auth/register", json=payload).status_code == 409
    assert client.post("/auth/register", json=dict(payload, username="reg_short", password="short")).status_code == 422


def test_passwords_are_not_stored_in_plain_text(client):
    make_user(client, "pw_storage", "doctor", password="a-very-secret-pw")
    conn = sqlite3.connect(os.environ["SCI_RTW_DB"])
    row = conn.execute("SELECT password_hash, password_salt FROM users WHERE username = 'pw_storage'").fetchone()
    tokens = [t[0] for t in conn.execute("SELECT token_hash FROM sessions")]
    conn.close()
    assert "a-very-secret-pw" not in row[0] and len(row[0]) == 64 and len(row[1]) == 32
    assert all(len(t) == 64 for t in tokens)


def test_change_password(client):
    headers = make_user(client, "pw_change", "doctor")
    bad = client.post("/auth/change-password", headers=headers,
                      json={"current_password": "wrong-one", "new_password": "new-password-1"})
    assert bad.status_code == 400
    ok = client.post("/auth/change-password", headers=headers,
                     json={"current_password": "password123", "new_password": "new-password-1"})
    assert ok.status_code == 200
    assert client.post("/auth/login", json={"username": "pw_change", "password": "password123"}).status_code == 401
    assert client.post("/auth/login", json={"username": "pw_change", "password": "new-password-1"}).status_code == 200


# ---------------- Manage user accounts (Administrator) ----------------
def test_admin_can_create_update_deactivate_and_remove_user(client, admin_headers):
    r = client.post("/users", headers=admin_headers, json={
        "username": "managed_user", "password": "password123", "full_name": "Managed", "role": "therapist"})
    assert r.status_code == 201
    uid = r.json()["id"]
    headers = login_headers(client, "managed_user", "password123")

    r = client.patch(f"/users/{uid}", headers=admin_headers, json={"role": "doctor", "full_name": "Managed Renamed"})
    assert r.json()["role"] == "doctor" and r.json()["full_name"] == "Managed Renamed"
    # A role change ends existing sessions so the new role applies at once.
    assert client.get("/auth/me", headers=headers).status_code == 401

    assert client.patch(f"/users/{uid}", headers=admin_headers, json={"is_active": False}).status_code == 200
    assert client.post("/auth/login", json={"username": "managed_user", "password": "password123"}).status_code == 403

    assert client.delete(f"/users/{uid}", headers=admin_headers).status_code == 200
    assert uid not in [u["id"] for u in client.get("/users", headers=admin_headers).json()]


def test_admin_cannot_remove_self_or_last_admin(client, admin_headers):
    me = client.get("/auth/me", headers=admin_headers).json()
    assert client.delete(f"/users/{me['id']}", headers=admin_headers).status_code == 409
    assert client.patch(f"/users/{me['id']}", headers=admin_headers, json={"is_active": False}).status_code == 409
    assert client.patch(f"/users/{me['id']}", headers=admin_headers, json={"role": "doctor"}).status_code == 409
    assert client.get("/auth/me", headers=admin_headers).status_code == 200


def test_user_with_patient_records_cannot_be_removed(client, admin_headers):
    headers = make_user(client, "has_records", "doctor")
    new_profile(client, headers, "REC-OWNER-1")
    uid = client.get("/auth/me", headers=headers).json()["id"]
    assert client.delete(f"/users/{uid}", headers=admin_headers).status_code == 409


# ---------------- FR2: manage patient profile ----------------
def test_fr2_patient_profile_create_read_update_delete(client, doctor_headers):
    p = new_profile(client, doctor_headers, "MRN-1001", notes="Initial note")
    assert p["assessment_count"] == 0 and p["linked_username"] is None

    assert client.get(f"/patients/{p['id']}", headers=doctor_headers).json()["notes"] == "Initial note"
    r = client.put(f"/patients/{p['id']}", headers=doctor_headers,
                   json={"patient_code": "MRN-1001", "full_name": "Renamed Patient", "notes": "Updated"})
    assert r.status_code == 200 and r.json()["full_name"] == "Renamed Patient"

    assert client.delete(f"/patients/{p['id']}", headers=doctor_headers).status_code == 200
    assert client.get(f"/patients/{p['id']}", headers=doctor_headers).status_code == 404


def test_fr2_duplicate_patient_code_rejected(client, doctor_headers):
    new_profile(client, doctor_headers, "MRN-DUPE")
    r = client.post("/patients", headers=doctor_headers, json={"patient_code": "mrn-dupe", "full_name": "Other"})
    assert r.status_code == 409


def test_fr2_search_patients(client, doctor_headers):
    new_profile(client, doctor_headers, "SEARCH-ALPHA")
    new_profile(client, doctor_headers, "SEARCH-BETA")
    codes = [p["patient_code"] for p in client.get("/patients?q=search-alp", headers=doctor_headers).json()]
    assert codes == ["SEARCH-ALPHA"]


def test_fr2_only_patient_accounts_can_be_linked(client, doctor_headers):
    make_user(client, "link_not_patient", "therapist")
    r = client.post("/patients", headers=doctor_headers, json={
        "patient_code": "LINK-BAD", "full_name": "X", "linked_username": "link_not_patient"})
    assert r.status_code == 422
    r = client.post("/patients", headers=doctor_headers, json={
        "patient_code": "LINK-MISSING", "full_name": "X", "linked_username": "no_such_account"})
    assert r.status_code == 404


# ---------------- TC06: prediction history (server-side) ----------------
def test_tc06_saved_prediction_appears_in_history(client, doctor_headers):
    p = new_profile(client, doctor_headers, "HIST-1")
    r = client.post(f"/patients/{p['id']}/predictions", headers=doctor_headers, json=PATIENT_DATA)
    assert r.status_code == 201
    saved = r.json()

    history = client.get(f"/predictions?patient_id={p['id']}", headers=doctor_headers).json()
    assert len(history) == 1
    rec = history[0]
    assert rec["id"] == saved["prediction_id"]
    assert rec["patient"]["patient_code"] == "HIST-1"
    assert rec["entered_by"] == "Dr Test"
    assert rec["input"]["neuro_level_discharge"] == "L03"
    assert rec["result"]["rtw_status"] == saved["rtw_status"]
    assert rec["result"]["rtw_probability"] == saved["rtw_probability"]
    assert rec["result"]["rtw_timing_probabilities"] == saved["rtw_timing_probabilities"]
    assert rec["result"]["patient_factors"] == saved["patient_factors"]
    assert client.get(f"/patients/{p['id']}", headers=doctor_headers).json()["assessment_count"] == 1


def test_tc06_saved_prediction_matches_stateless_predict(client, doctor_headers):
    p = new_profile(client, doctor_headers, "HIST-SAME")
    saved = client.post(f"/patients/{p['id']}/predictions", headers=doctor_headers, json=PATIENT_DATA).json()
    plain = client.post("/predict", headers=doctor_headers, json=PATIENT_DATA).json()
    saved.pop("prediction_id")
    assert saved == plain


def test_tc06_history_is_shared_between_clinicians_and_newest_first(client, doctor_headers, therapist_headers):
    p = new_profile(client, doctor_headers, "HIST-ORDER")
    first = client.post(f"/patients/{p['id']}/predictions", headers=doctor_headers, json=PATIENT_DATA).json()
    second = client.post(f"/patients/{p['id']}/predictions", headers=therapist_headers,
                         json=dict(PATIENT_DATA, age_at_injury=40)).json()
    ids = [h["id"] for h in client.get(f"/predictions?patient_id={p['id']}", headers=therapist_headers).json()]
    assert ids == [second["prediction_id"], first["prediction_id"]]


def test_tc06_stateless_predict_is_not_saved(client, doctor_headers):
    before = len(client.get("/predictions", headers=doctor_headers).json())
    client.post("/predict", headers=doctor_headers, json=PATIENT_DATA)
    assert len(client.get("/predictions", headers=doctor_headers).json()) == before


def test_tc06_prediction_for_unknown_patient_rejected(client, doctor_headers):
    assert client.post("/patients/999999/predictions", headers=doctor_headers, json=PATIENT_DATA).status_code == 404


def test_tc06_invalid_data_is_not_saved(client, doctor_headers):
    p = new_profile(client, doctor_headers, "HIST-INVALID")
    bad = {k: v for k, v in PATIENT_DATA.items() if k != "sex"}
    assert client.post(f"/patients/{p['id']}/predictions", headers=doctor_headers, json=bad).status_code == 422
    assert client.get(f"/predictions?patient_id={p['id']}", headers=doctor_headers).json() == []


def test_tc06_patient_sees_only_their_own_linked_history(client, doctor_headers):
    mine_headers = make_user(client, "pt_mine", "patient")
    other_headers = make_user(client, "pt_other", "patient")
    mine = new_profile(client, doctor_headers, "PT-MINE", linked_username="pt_mine")
    other = new_profile(client, doctor_headers, "PT-OTHER", linked_username="pt_other")
    unlinked = new_profile(client, doctor_headers, "PT-UNLINKED")
    for p in (mine, other, unlinked):
        assert client.post(f"/patients/{p['id']}/predictions", headers=doctor_headers,
                           json=PATIENT_DATA).status_code == 201

    seen = client.get("/predictions", headers=mine_headers).json()
    assert [h["patient"]["patient_code"] for h in seen] == ["PT-MINE"]
    # Asking for someone else's profile by id must not leak it.
    assert client.get(f"/predictions?patient_id={other['id']}", headers=mine_headers).json() == []
    assert [h["patient"]["patient_code"] for h in client.get("/predictions", headers=other_headers).json()] == ["PT-OTHER"]


def test_history_can_be_read_a_page_at_a_time(client, doctor_headers):
    p = new_profile(client, doctor_headers, "PAGE-1")
    ids = [client.post(f"/patients/{p['id']}/predictions", headers=doctor_headers, json=PATIENT_DATA).json()["prediction_id"]
           for _ in range(5)]
    url = f"/predictions?patient_id={p['id']}"
    everything = client.get(url, headers=doctor_headers)
    assert [h["id"] for h in everything.json()] == ids[::-1]
    assert everything.headers["x-total-count"] == "5"

    pages = [client.get(f"{url}&limit=2&offset={offset}", headers=doctor_headers) for offset in (0, 2, 4, 6)]
    assert [[h["id"] for h in page.json()] for page in pages] == [ids[4:2:-1], ids[2:0:-1], ids[:1], []]
    assert all(page.headers["x-total-count"] == "5" for page in pages)

    for bad in ("limit=0", "limit=501", "offset=-1"):
        assert client.get(f"/predictions?{bad}", headers=doctor_headers).status_code == 422


def test_page_total_counts_only_what_a_patient_may_see(client, doctor_headers):
    headers = make_user(client, "pt_page", "patient")
    mine = new_profile(client, doctor_headers, "PAGE-MINE", linked_username="pt_page")
    other = new_profile(client, doctor_headers, "PAGE-OTHER")
    for p in (mine, mine, other):
        client.post(f"/patients/{p['id']}/predictions", headers=doctor_headers, json=PATIENT_DATA)
    r = client.get("/predictions?limit=1", headers=headers)
    assert r.headers["x-total-count"] == "2" and len(r.json()) == 1


def test_tc06_unlinked_patient_account_sees_nothing(client):
    headers = make_user(client, "pt_nolink", "patient")
    assert client.get("/predictions", headers=headers).json() == []


def test_clinician_notes_saved_on_prediction_report(client, doctor_headers):
    p = new_profile(client, doctor_headers, "NOTES-1")
    pid = client.post(f"/patients/{p['id']}/predictions", headers=doctor_headers, json=PATIENT_DATA).json()["prediction_id"]
    r = client.patch(f"/predictions/{pid}/notes", headers=doctor_headers,
                     json={"clinician_notes": "Refer to vocational rehabilitation."})
    assert r.status_code == 200 and r.json()["clinician_notes"] == "Refer to vocational rehabilitation."
    history = client.get(f"/predictions?patient_id={p['id']}", headers=doctor_headers).json()
    assert history[0]["clinician_notes"] == "Refer to vocational rehabilitation."
    assert client.patch("/predictions/999999/notes", headers=doctor_headers,
                        json={"clinician_notes": "x"}).status_code == 404


def test_saved_record_keeps_its_confidence_and_recommendations(client, doctor_headers):
    p = new_profile(client, doctor_headers, "REPORT-1")
    saved = client.post(f"/patients/{p['id']}/predictions", headers=doctor_headers, json=PATIENT_DATA).json()
    rec = client.get(f"/predictions?patient_id={p['id']}", headers=doctor_headers).json()[0]
    assert rec["result"]["confidence"] == saved["confidence"]
    assert rec["result"]["recommendations"] == saved["recommendations"] and saved["recommendations"]


def test_database_from_earlier_version_is_upgraded_without_losing_records(client, doctor_headers):
    """An existing sci_rtw.db made before confidence/recommendations existed
    must keep working: the columns are added, old rows survive, and startup
    backfill completes them from their stored inputs."""
    import db
    import main

    p = new_profile(client, doctor_headers, "UPGRADE-1")
    pid = client.post(f"/patients/{p['id']}/predictions", headers=doctor_headers, json=PATIENT_DATA).json()["prediction_id"]
    retrained = new_profile(client, doctor_headers, "UPGRADE-2")
    other = client.post(f"/patients/{retrained['id']}/predictions", headers=doctor_headers, json=PATIENT_DATA).json()["prediction_id"]
    before = client.get(f"/predictions?patient_id={p['id']}", headers=doctor_headers).json()[0]

    # Put the table back to its earlier shape, and make the second record look
    # as if a different (since retrained) model had produced it.
    conn = sqlite3.connect(os.environ["SCI_RTW_DB"])
    n_before = conn.execute("SELECT COUNT(*) FROM prediction_reports").fetchone()[0]
    conn.execute("ALTER TABLE prediction_reports DROP COLUMN confidence_json")
    conn.execute("ALTER TABLE prediction_reports DROP COLUMN recommendations_json")
    conn.execute("UPDATE predictions SET rtw_probability = 0.61 WHERE id = ?", (other,))
    conn.commit()
    conn.close()

    db.init_db()
    conn = sqlite3.connect(os.environ["SCI_RTW_DB"])
    assert conn.execute("SELECT COUNT(*) FROM prediction_reports").fetchone()[0] == n_before
    assert conn.execute("SELECT COUNT(*) FROM prediction_reports WHERE recommendations_json IS NOT NULL").fetchone()[0] == 0
    conn.close()
    db.init_db()  # running the upgrade twice must be harmless

    assert main.backfill_reports() == n_before - 1
    after = client.get(f"/predictions?patient_id={p['id']}", headers=doctor_headers).json()[0]
    assert after == before, "an upgraded record must read exactly as it did when first saved"

    # The record from a 'different model' keeps its prediction, shows a
    # confidence level for its own probability, and gets no invented suggestions.
    stale = client.get(f"/predictions?patient_id={retrained['id']}", headers=doctor_headers).json()[0]
    assert stale["result"]["rtw_probability"] == 0.61
    assert stale["result"]["confidence"]["level"] == "Low"
    assert stale["result"]["recommendations"] == []
    assert main.backfill_reports() == 0


def test_deleting_profile_removes_its_history(client, doctor_headers):
    p = new_profile(client, doctor_headers, "CASCADE-1")
    client.post(f"/patients/{p['id']}/predictions", headers=doctor_headers, json=PATIENT_DATA)
    client.delete(f"/patients/{p['id']}", headers=doctor_headers)
    assert client.get(f"/predictions?patient_id={p['id']}", headers=doctor_headers).json() == []
    conn = sqlite3.connect(os.environ["SCI_RTW_DB"])
    orphans = conn.execute(
        "SELECT (SELECT COUNT(*) FROM assessments WHERE patient_id = ?) + "
        "(SELECT COUNT(*) FROM predictions WHERE assessment_id NOT IN (SELECT id FROM assessments)) + "
        "(SELECT COUNT(*) FROM prediction_reports WHERE prediction_id NOT IN (SELECT id FROM predictions))",
        (p["id"],)).fetchone()[0]
    conn.close()
    assert orphans == 0


# ---------------- Model version kept with each saved prediction ----------------
def test_saved_prediction_records_which_model_produced_it(client, doctor_headers):
    current = client.get("/health").json()["model_version"]
    assert len(current) == 12
    profile = new_profile(client, doctor_headers, "TC06-VERSION")
    saved = client.post(f"/patients/{profile['id']}/predictions", headers=doctor_headers, json=PATIENT_DATA).json()
    assert saved["model_version"] == current and saved["from_current_model"] is True

    def stored():
        rows = client.get(f"/predictions?patient_id={profile['id']}", headers=doctor_headers).json()
        return rows[0]["result"]

    assert stored()["model_version"] == current and stored()["from_current_model"] is True

    # After a retrain the record still says which model it came from, and a
    # record from before versions were kept counts as an earlier model too.
    for earlier in ("0123456789ab", None):
        conn = sqlite3.connect(os.environ["SCI_RTW_DB"])
        conn.execute("UPDATE predictions SET model_version = ? WHERE id = ?", (earlier, saved["prediction_id"]))
        conn.commit()
        conn.close()
        assert stored()["model_version"] == earlier and stored()["from_current_model"] is False
