"""
SCI-RTW Predictor: how long does the system take to answer?

The proposal's performance requirement (Table 3.8) is that the prototype
"display results within a reasonable time". This script measures it, so
the report can give a figure instead of an impression. It changes nothing:
it works on a throwaway database and writes docs/performance_results.json
(discussed in docs/test_report.md, 6.10).

Measured, through the API's full request handling (validation, login
check, both model layers, the explanation, the database):
  start-up          - loading the models and opening the database once
  predict           - POST /predict for held-out test patients, not saved
  predict and save  - POST /patients/{id}/predictions, the form's real path
  history           - GET /predictions, one page of 50 and the full list,
                      once the database holds every saved prediction

Requests are made in-process (FastAPI's TestClient), so the figures leave
out the network. On the same machine that adds well under a millisecond;
over the internet it adds the connection's round-trip time.

Run:  python src/measure_performance.py      (under a minute)
"""

import json
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP_DIR = tempfile.mkdtemp(prefix="sci_rtw_perf_")
# Set before main.py is imported, so the real api/sci_rtw.db is never touched.
os.environ["SCI_RTW_DB"] = os.path.join(TMP_DIR, "perf.db")
sys.path.insert(0, str(ROOT / "api"))

import pandas as pd  # noqa: E402

N_PREDICT = 300
N_SAVE = 200
PASSWORD = "perf-check-pw1"


def summary(ms):
    ms = sorted(ms)
    return {"n": len(ms), "median_ms": round(statistics.median(ms), 1),
            "p95_ms": round(ms[int(0.95 * (len(ms) - 1))], 1), "max_ms": round(ms[-1], 1)}


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1000


def main():
    start = time.perf_counter()
    import auth
    import db
    from main import app
    startup_ms = (time.perf_counter() - start) * 1000

    from fastapi.testclient import TestClient
    client = TestClient(app)

    conn = db.connect()
    auth.insert_user(conn, auth.RegisterInput(username="perf_doctor", full_name="Perf Doctor",
                                              role="doctor", password=PASSWORD), db.ROLES, True)
    conn.commit()
    conn.close()
    token = client.post("/auth/login", json={"username": "perf_doctor", "password": PASSWORD}).json()["token"]
    client.headers.update({"Authorization": f"Bearer {token}"})

    rows = pd.read_csv(ROOT / "processed" / "layer1_test.csv", dtype=str).sample(N_PREDICT, random_state=3)
    numeric = ("age_at_injury", "asia_motor_total_discharge")
    payloads = [{c: (float(r[c]) if c in numeric else r[c])
                 for c in rows.columns if c not in ("UniID", "rtw_status")} for _, r in rows.iterrows()]

    client.post("/predict", json=payloads[0])  # first call warms caches; not counted
    predict = []
    for p in payloads:
        r, ms = timed(lambda: client.post("/predict", json=p))
        assert r.status_code == 200, r.text
        predict.append(ms)

    patient_id = client.post("/patients", json={"patient_code": "PERF-1", "full_name": "Perf Patient"}).json()["id"]
    save = []
    for p in payloads[:N_SAVE]:
        r, ms = timed(lambda: client.post(f"/patients/{patient_id}/predictions", json=p))
        assert r.status_code == 201, r.text
        save.append(ms)

    page, full = [], []
    for _ in range(20):
        r, ms = timed(lambda: client.get("/predictions", params={"limit": 50}))
        assert r.status_code == 200 and len(r.json()) == 50
        page.append(ms)
        r, ms = timed(lambda: client.get("/predictions"))
        assert r.status_code == 200 and len(r.json()) == N_SAVE
        full.append(ms)

    results = {
        "machine": {"system": f"{platform.system()} {platform.release()}", "processor": platform.processor(),
                    "cpu_count": os.cpu_count(), "python": platform.python_version()},
        "startup_ms": round(startup_ms),
        "predict": summary(predict),
        "predict_and_save": summary(save),
        "history_page_of_50": summary(page),
        f"history_all_{N_SAVE}": summary(full),
    }
    out = ROOT / "docs" / "performance_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Start-up (models + database): {results['startup_ms']:,} ms")
    print(f"{'request':28} {'n':>4} {'median':>9} {'95th pct':>9} {'max':>9}")
    for name in ("predict", "predict_and_save", "history_page_of_50", f"history_all_{N_SAVE}"):
        s = results[name]
        print(f"{name:28} {s['n']:>4} {s['median_ms']:>7} ms {s['p95_ms']:>7} ms {s['max_ms']:>7} ms")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
