"""
SCI-RTW Predictor: fill the database with demonstration data.

Creates demo user accounts, patient profiles and saved predictions in
api/sci_rtw.db so the screens have something to show. Patient profiles
and predictions go through the real API, so every prediction is genuine
model output.

    python api/seed_demo_data.py

- All names are invented. The clinical inputs are rows from the held-out
  NSCISC test split (already de-identified public data), so the profiles
  are realistic without describing any real, identifiable person.
- Only adds records; anything already in the database is left as it is.
  Refuses to run a second time (it checks for the demo patient IDs), so it
  cannot create duplicates. To reseed from scratch: stop the API, delete
  api/sci_rtw.db, and run this again.
- Stop the API before running (or restart it afterwards) - not required,
  but it avoids two programs writing to the file at the same moment.
"""

import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
from fastapi.testclient import TestClient

API_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, API_DIR)

import auth  # noqa: E402
import db  # noqa: E402
from main import app  # noqa: E402

DEMO_PASSWORD = "demo12345"

USERS = [  # username, full name, role, active
    ("dr.aisyah", "Dr. Aisyah Rahman", "doctor", True),
    ("dr.kumar", "Dr. Kumar Selvam", "doctor", True),
    ("therapist.lim", "Lim Wei Jie", "therapist", True),
    ("patient.farid", "Farid Hakimi", "patient", True),
    ("patient.mei", "Tan Mei Ling", "patient", True),
    ("dr.pending", "Dr. Nurul Izzah", "doctor", False),  # shows the approval queue
]

PATIENT_NAMES = ["Farid Hakimi", "Tan Mei Ling", "Ahmad Zulkifli", "Siti Nadia", "Rajesh Kumar",
                 "Nur Amira", "Daniel Wong", "Hafiz Roslan", "Priya Nair", "Lee Jia Hui",
                 "Azman Ismail", "Chong Kai Xin"]
LINKED_ACCOUNTS = {"Farid Hakimi": "patient.farid", "Tan Mei Ling": "patient.mei"}

NOTES_RTW = [
    "Good candidate for early vocational rehabilitation referral. Review at 3 months.",
    "Discussed result with patient and family. Employer contacted about workplace adjustments.",
    "Refer to occupational therapy for work-readiness assessment.",
]
NOTES_NON_RTW = [
    "Lower likelihood on current profile. Focus on functional goals first; reassess after rehabilitation block.",
    "Explained that this is an estimate, not a decision. Referred to social work for financial support options.",
]

CATEGORICAL = ["sex", "education_level", "marital_status", "veteran_status", "pre_injury_employment_status",
               "injury_cause", "vertebral_injury_level", "associated_injury", "work_related_injury",
               "neuro_level_discharge", "aisgrade_discharge", "injury_completeness_discharge"]


def login(client, username, password):
    r = client.post("/auth/login", json={"username": username, "password": password})
    if r.status_code != 200:
        sys.exit(f"Could not sign in as '{username}': {r.json().get('detail')}")
    return {"Authorization": f"Bearer {r.json()['token']}"}


def must(response, expected):
    if response.status_code != expected:
        sys.exit(f"Seeding failed ({response.status_code}): {response.text}")
    return response.json()


def clinical_inputs(n):
    """n real rows from the held-out test split, as /predict payloads."""
    path = os.path.join(API_DIR, "..", "processed", "layer1_test.csv")
    rows = pd.read_csv(path, dtype=str).sample(n, random_state=11)
    payloads = []
    for _, row in rows.iterrows():
        p = {c: row[c] for c in CATEGORICAL}
        p["age_at_injury"] = float(row["age_at_injury"])
        p["asia_motor_total_discharge"] = float(row["asia_motor_total_discharge"])
        payloads.append(p)
    return payloads


def backdate(first_id, spread_days=45):
    """The demo records were all created just now; spread their dates over
    recent weeks so the history looks like real use. Only records this run
    created (assessment id >= first_id) are touched."""
    conn = db.connect()
    try:
        now = datetime.now(timezone.utc)
        ids = [r["id"] for r in conn.execute("SELECT id FROM assessments WHERE id >= ? ORDER BY id", (first_id,))]
        for i, assessment_id in enumerate(ids):
            when = now - timedelta(days=spread_days * (1 - (i + 1) / len(ids)), hours=(i * 5) % 9)
            ts = when.isoformat(timespec="seconds")
            conn.execute("UPDATE assessments SET created_at = ? WHERE id = ?", (ts, assessment_id))
            conn.execute("UPDATE predictions SET created_at = ? WHERE assessment_id = ?", (ts, assessment_id))
        conn.execute(
            "UPDATE patients SET "
            "created_at = (SELECT MIN(created_at) FROM assessments WHERE patient_id = patients.id), "
            "updated_at = (SELECT MAX(created_at) FROM assessments WHERE patient_id = patients.id) "
            "WHERE id IN (SELECT patient_id FROM assessments WHERE id >= ?)", (first_id,))
        conn.commit()
    finally:
        conn.close()


def main():
    conn = sqlite3.connect(db.db_path())
    codes = [f"MRN-{1001 + i}" for i in range(len(PATIENT_NAMES))]
    clash = conn.execute(f"SELECT COUNT(*) FROM patients WHERE patient_code IN ({','.join('?' * len(codes))})",
                         codes).fetchone()[0]
    first_new_assessment = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM assessments").fetchone()[0]
    conn.close()
    if clash:
        sys.exit(f"{db.db_path()} already contains the demo patients ({codes[0]} to {codes[-1]}); "
                 "nothing was changed. Delete the file first if you want to reseed.")

    # Accounts are written straight to the database, so seeding works whatever
    # the administrator's password is and before it has been changed.
    conn = db.connect()
    taken = {row["username"].lower() for row in conn.execute("SELECT username FROM users")}
    for username, full_name, role, active in USERS:
        if username not in taken:
            auth.insert_user(conn, auth.RegisterInput(username=username, full_name=full_name, role=role,
                                                      password=DEMO_PASSWORD), db.ROLES, active)
    conn.commit()
    conn.close()

    client = TestClient(app)

    clinicians = [login(client, u, DEMO_PASSWORD) for u in ("dr.aisyah", "therapist.lim", "dr.kumar")]
    inputs = clinical_inputs(len(PATIENT_NAMES))
    saved = 0
    for i, (name, payload) in enumerate(zip(PATIENT_NAMES, inputs)):
        who = clinicians[i % len(clinicians)]
        patient = must(client.post("/patients", headers=who, json={
            "patient_code": f"MRN-{1001 + i}", "full_name": name,
            "notes": "Demonstration record." if i % 3 else "Demonstration record. Referred from inpatient rehabilitation.",
            "linked_username": LINKED_ACCOUNTS.get(name)}), 201)

        assessments = [payload]
        if i % 4 == 0:
            # A later reassessment for some patients: motor score improved.
            better = min(100.0, payload["asia_motor_total_discharge"] + 12)
            assessments.append(dict(payload, asia_motor_total_discharge=better))
        for j, body in enumerate(assessments):
            result = must(client.post(f"/patients/{patient['id']}/predictions", headers=who, json=body), 201)
            saved += 1
            if (i + j) % 2 == 0:
                pool = NOTES_RTW if result["rtw_status"] == "RTW" else NOTES_NON_RTW
                must(client.patch(f"/predictions/{result['prediction_id']}/notes", headers=who,
                                  json={"clinician_notes": pool[(i + j) % len(pool)]}), 200)

    backdate(first_new_assessment)

    conn = sqlite3.connect(db.db_path())
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("users", "patients", "assessments", "predictions", "prediction_reports")}
    rtw = conn.execute("SELECT COUNT(*) FROM predictions WHERE rtw_status = 'RTW' AND assessment_id >= ?",
                       (first_new_assessment,)).fetchone()[0]
    conn.close()

    print(f"Seeded {db.db_path()}")
    for table, n in counts.items():
        print(f"  {table:20} {n}")
    print(f"  (table totals include anything already in the database; "
          f"{rtw} of the {saved} demo predictions are RTW)")
    print(f"\nDemo accounts (password for all: {DEMO_PASSWORD}):")
    for username, full_name, role, active in USERS:
        print(f"  {username:16} {role:10} {full_name}{'' if active else '   <- inactive, awaiting approval'}")
    print(f"  {auth.DEFAULT_ADMIN_USERNAME:16} {'admin':10} (password unchanged)")


if __name__ == "__main__":
    main()
