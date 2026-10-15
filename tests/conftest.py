"""
Shared pytest setup. Points the API at a throwaway SQLite file BEFORE main.py
is imported, so running the tests never touches the real api/sci_rtw.db.
"""
import os
import sys
import tempfile

import pytest

_TMP_DIR = tempfile.mkdtemp(prefix="sci_rtw_test_")
os.environ["SCI_RTW_DB"] = os.path.join(_TMP_DIR, "test.db")
os.environ["SCI_RTW_ADMIN_PASSWORD"] = "test-admin-pw"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "api"))

ADMIN = {"username": "admin", "password": "test-admin-pw"}


def login_headers(client, username, password):
    r = client.post("/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def make_user(client, username, role, password="password123", full_name=None):
    """Create an active account as the administrator and return its auth headers.
    Safe to call repeatedly for the same username."""
    admin = login_headers(client, **ADMIN)
    r = client.post("/users", headers=admin, json={
        "username": username, "password": password, "role": role,
        "full_name": full_name or username.title()})
    assert r.status_code in (201, 409), r.text
    return login_headers(client, username, password)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)


@pytest.fixture(scope="session")
def admin_headers(client):
    return login_headers(client, **ADMIN)


@pytest.fixture(scope="session")
def doctor_headers(client):
    return make_user(client, "dr_test", "doctor", full_name="Dr Test")


@pytest.fixture(scope="session")
def therapist_headers(client):
    return make_user(client, "therapist_test", "therapist")
