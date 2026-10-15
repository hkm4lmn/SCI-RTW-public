"""
SCI-RTW Predictor: authentication, roles and user-account management
Implements FR1 (register and login), TC01 (role-based access) and the
Administrator's "Manage User Account" use case (Table 3.9).

Roles and what each may do (Figure 3.4, Use Case Diagram):
    doctor / therapist - manage patient profiles, enter data, run predictions,
                         view history, view the model accuracy report
    patient            - view the prediction results and history of the
                         patient profile linked to their own account
    admin              - manage user accounts, view the model accuracy report

Self-registration: a patient account is usable immediately but sees nothing
until a clinician links it to a patient profile. A doctor/therapist account
is created INACTIVE and must be approved by the administrator, because those
roles can read every patient record.

Passwords are stored as salted PBKDF2-SHA256 hashes, never in plain text.
Login returns a random bearer token; only its SHA-256 hash is stored.
Standard library only (hashlib, hmac, secrets).

Further protections:
    - Five wrong passwords in a row for one username lock sign-in for that
      username for fifteen minutes (password guessing).
    - An administrator still on the built-in default password can do
      nothing except change it.
    - Changing a password signs out every other session of that account.
    - Sign-ins, account changes and patient-record changes are written to
      the audit log, which the administrator can read.
"""

import hashlib
import hmac
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from db import CLINICIAN_ROLES, connect, get_db, now_iso, record_audit

PBKDF2_ITERATIONS = 200_000
SESSION_HOURS = 12
MAX_FAILED_LOGINS = 5
LOCKOUT_MINUTES = 15
SELF_REGISTER_ROLES = ("doctor", "therapist", "patient")

DEFAULT_ADMIN_USERNAME = "admin"
# Demo default so the prototype can be started with no setup. Set the
# SCI_RTW_ADMIN_PASSWORD environment variable before first run to override.
DEFAULT_ADMIN_PASSWORD = "admin123"

router = APIRouter()


# ----------------------------------------------------------------------------
# Password + token helpers
# ----------------------------------------------------------------------------
def hash_password(password: str, salt: Optional[str] = None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                 bytes.fromhex(salt), PBKDF2_ITERATIONS)
    return digest.hex(), salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    candidate, _ = hash_password(password, salt)
    return hmac.compare_digest(candidate, stored_hash)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# Checked when the username does not exist, so that sign-in takes as long
# for an unknown username as for a wrong password and the response time
# does not reveal which usernames are real.
_DUMMY_HASH, _DUMMY_SALT = hash_password(secrets.token_hex(16))


def public_user(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "username": row["username"],
        "full_name": row["full_name"],
        "role": row["role"],
        "is_active": bool(row["is_active"]),
        "must_change_password": bool(row["must_change_password"]),
        "created_at": row["created_at"],
    }


def ensure_default_admin():
    """Create the first administrator if the system has none.

    An administrator whose password is the built-in default must change it
    at first sign-in. That includes one created by an earlier version."""
    conn = connect()
    try:
        if not conn.execute("SELECT 1 FROM users WHERE role = 'admin'").fetchone():
            password = os.environ.get("SCI_RTW_ADMIN_PASSWORD", DEFAULT_ADMIN_PASSWORD)
            pw_hash, salt = hash_password(password)
            conn.execute(
                "INSERT INTO users (username, full_name, role, password_hash, password_salt, is_active, created_at) "
                "VALUES (?, ?, 'admin', ?, ?, 1, ?)",
                (DEFAULT_ADMIN_USERNAME, "System Administrator", pw_hash, salt, now_iso()),
            )
        row = conn.execute("SELECT * FROM users WHERE username = ?", (DEFAULT_ADMIN_USERNAME,)).fetchone()
        if row and verify_password(DEFAULT_ADMIN_PASSWORD, row["password_hash"], row["password_salt"]):
            conn.execute("UPDATE users SET must_change_password = 1 WHERE id = ?", (row["id"],))
        conn.commit()
    finally:
        conn.close()


# ----------------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------------
def session_user(authorization: Optional[str] = Header(default=None),
                 conn: sqlite3.Connection = Depends(get_db)) -> dict:
    """The signed-in user. Used directly only by the few endpoints that must
    work before a required password change; everything else uses current_user."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Not logged in")
    token = authorization.split(" ", 1)[1].strip()
    row = conn.execute(
        "SELECT u.*, s.expires_at FROM sessions s JOIN users u ON u.id = s.user_id "
        "WHERE s.token_hash = ?", (hash_token(token),)).fetchone()
    if row is None or row["expires_at"] < now_iso():
        raise HTTPException(status_code=401, detail="Session expired, please log in again")
    if not row["is_active"]:
        raise HTTPException(status_code=403, detail="This account is not active")
    return public_user(row)


def current_user(user: dict = Depends(session_user)) -> dict:
    if user["must_change_password"]:
        raise HTTPException(status_code=403, detail="Change your password before using the system")
    return user


def require_roles(*roles):
    def dependency(user: dict = Depends(current_user)) -> dict:
        if user["role"] not in roles:
            raise HTTPException(status_code=403, detail="Your role does not have access to this function")
        return user
    return dependency


require_clinician = require_roles(*CLINICIAN_ROLES)
require_admin = require_roles("admin")
require_staff = require_roles(*CLINICIAN_ROLES, "admin")


# ----------------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------------
class RegisterInput(BaseModel):
    username: str = Field(min_length=3, max_length=40, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(min_length=1, max_length=120)
    role: str


class LoginInput(BaseModel):
    username: str = Field(max_length=40)
    password: str = Field(max_length=128)


class UserCreateInput(RegisterInput):
    is_active: bool = True


class UserUpdateInput(BaseModel):
    full_name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    role: Optional[str] = None
    is_active: Optional[bool] = None
    password: Optional[str] = Field(default=None, min_length=8, max_length=128)


class ChangePasswordInput(BaseModel):
    current_password: str = Field(max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


def insert_user(conn, data: RegisterInput, allowed_roles, is_active: bool) -> dict:
    if data.role not in allowed_roles:
        raise HTTPException(status_code=422, detail=f"Role must be one of: {', '.join(allowed_roles)}")
    pw_hash, salt = hash_password(data.password)
    try:
        cur = conn.execute(
            "INSERT INTO users (username, full_name, role, password_hash, password_salt, is_active, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (data.username, data.full_name.strip(), data.role, pw_hash, salt, int(is_active), now_iso()),
        )
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="That username is already taken")
    return public_user(conn.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone())


# ----------------------------------------------------------------------------
# Register / login (FR1)
# ----------------------------------------------------------------------------
@router.post("/auth/register", status_code=201)
def register(data: RegisterInput, conn: sqlite3.Connection = Depends(get_db)):
    # Clinician accounts can read every patient record, so they wait for
    # administrator approval; patient accounts are active straight away.
    user = insert_user(conn, data, SELF_REGISTER_ROLES, is_active=(data.role == "patient"))
    record_audit(conn, user, "account.register", user["username"], f"role {user['role']}")
    return {"user": user, "pending_approval": not user["is_active"]}


def note_failed_login(conn, username: str, previous_failures: int) -> None:
    """Count a wrong password and lock the username at the limit. Committed
    here because the 401 that follows rolls the request's transaction back."""
    failures = previous_failures + 1
    locked_until = None
    if failures >= MAX_FAILED_LOGINS:
        locked_until = (datetime.now(timezone.utc) + timedelta(minutes=LOCKOUT_MINUTES)).isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO login_failures (username, failures, locked_until, updated_at) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(username) DO UPDATE SET failures = excluded.failures, "
        "locked_until = excluded.locked_until, updated_at = excluded.updated_at",
        (username, failures, locked_until, now_iso()))
    record_audit(conn, None, "login.locked" if locked_until else "login.failed", username)
    conn.commit()


@router.post("/auth/login")
def login(data: LoginInput, conn: sqlite3.Connection = Depends(get_db)):
    username = data.username.strip()
    # Failures count towards a lock only while they keep coming: a username
    # left alone for the lockout period starts again from zero.
    window_start = (datetime.now(timezone.utc) - timedelta(minutes=LOCKOUT_MINUTES)).isoformat(timespec="seconds")
    conn.execute("DELETE FROM login_failures WHERE updated_at < ?", (window_start,))
    failed = conn.execute("SELECT failures, locked_until FROM login_failures WHERE username = ?",
                          (username,)).fetchone()
    if failed and failed["locked_until"] and failed["locked_until"] > now_iso():
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed sign-in attempts. Try again in {LOCKOUT_MINUTES} minutes.")

    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if row is None:
        verify_password(data.password, _DUMMY_HASH, _DUMMY_SALT)
        password_ok = False
    else:
        password_ok = verify_password(data.password, row["password_hash"], row["password_salt"])
    if not password_ok:
        note_failed_login(conn, username, failed["failures"] if failed else 0)
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    if not row["is_active"]:
        raise HTTPException(status_code=403, detail="This account is awaiting administrator approval")
    conn.execute("DELETE FROM login_failures WHERE username = ?", (username,))
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(hours=SESSION_HOURS)
    conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now_iso(),))
    conn.execute("INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                 (hash_token(token), row["id"], now_iso(), expires.isoformat(timespec="seconds")))
    user = public_user(row)
    record_audit(conn, user, "login.success")
    return {"token": token, "user": user}


@router.post("/auth/logout")
def logout(authorization: Optional[str] = Header(default=None),
           user: dict = Depends(session_user), conn: sqlite3.Connection = Depends(get_db)):
    token = authorization.split(" ", 1)[1].strip()
    conn.execute("DELETE FROM sessions WHERE token_hash = ?", (hash_token(token),))
    return {"status": "logged out"}


@router.get("/auth/me")
def me(user: dict = Depends(session_user)):
    return user


@router.post("/auth/change-password")
def change_password(data: ChangePasswordInput, authorization: Optional[str] = Header(default=None),
                    user: dict = Depends(session_user), conn: sqlite3.Connection = Depends(get_db)):
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone()
    if not verify_password(data.current_password, row["password_hash"], row["password_salt"]):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    if data.new_password == data.current_password:
        raise HTTPException(status_code=422, detail="The new password must be different from the current one")
    pw_hash, salt = hash_password(data.new_password)
    conn.execute("UPDATE users SET password_hash = ?, password_salt = ?, must_change_password = 0 WHERE id = ?",
                 (pw_hash, salt, user["id"]))
    # Anyone else signed in with the old password is signed out.
    token = authorization.split(" ", 1)[1].strip()
    conn.execute("DELETE FROM sessions WHERE user_id = ? AND token_hash != ?", (user["id"], hash_token(token)))
    record_audit(conn, user, "password.changed")
    return {"status": "password changed"}


# ----------------------------------------------------------------------------
# Manage user accounts (Administrator)
# ----------------------------------------------------------------------------
def active_admin_count(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM users WHERE role = 'admin' AND is_active = 1").fetchone()[0]


@router.get("/users")
def list_users(admin: dict = Depends(require_admin), conn: sqlite3.Connection = Depends(get_db)):
    rows = conn.execute("SELECT * FROM users ORDER BY is_active, created_at DESC").fetchall()
    return [public_user(r) for r in rows]


@router.post("/users", status_code=201)
def create_user(data: UserCreateInput, admin: dict = Depends(require_admin),
                conn: sqlite3.Connection = Depends(get_db)):
    user = insert_user(conn, data, ("doctor", "therapist", "patient", "admin"), data.is_active)
    record_audit(conn, admin, "account.created", user["username"], f"role {user['role']}")
    return user


@router.patch("/users/{user_id}")
def update_user(user_id: int, data: UserUpdateInput, admin: dict = Depends(require_admin),
                conn: sqlite3.Connection = Depends(get_db)):
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="User not found")
    if data.role is not None and data.role not in ("doctor", "therapist", "patient", "admin"):
        raise HTTPException(status_code=422, detail="Unknown role")

    new_role = data.role if data.role is not None else row["role"]
    new_active = data.is_active if data.is_active is not None else bool(row["is_active"])
    losing_admin = row["role"] == "admin" and row["is_active"] and (new_role != "admin" or not new_active)
    if losing_admin and active_admin_count(conn) <= 1:
        raise HTTPException(status_code=409, detail="The system must keep at least one active administrator")

    conn.execute("UPDATE users SET full_name = ?, role = ?, is_active = ? WHERE id = ?",
                 (data.full_name.strip() if data.full_name else row["full_name"], new_role, int(new_active), user_id))
    if data.password:
        pw_hash, salt = hash_password(data.password)
        conn.execute("UPDATE users SET password_hash = ?, password_salt = ?, must_change_password = 0 WHERE id = ?",
                     (pw_hash, salt, user_id))
    if not new_active or data.password or new_role != row["role"]:
        # Force a fresh login so the change takes effect immediately.
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
    changes = [text for changed, text in (
        (new_role != row["role"], f"role {row['role']} to {new_role}"),
        (new_active != bool(row["is_active"]), "activated" if new_active else "deactivated"),
        (bool(data.password), "password reset"),
        (bool(data.full_name) and data.full_name.strip() != row["full_name"], "name changed"),
    ) if changed]
    record_audit(conn, admin, "account.updated", row["username"], ", ".join(changes))
    return public_user(conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone())


@router.delete("/users/{user_id}")
def delete_user(user_id: int, admin: dict = Depends(require_admin),
                conn: sqlite3.Connection = Depends(get_db)):
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="User not found")
    if row["id"] == admin["id"]:
        raise HTTPException(status_code=409, detail="You cannot remove your own account")
    has_records = conn.execute(
        "SELECT 1 FROM assessments WHERE entered_by = ? UNION SELECT 1 FROM patients WHERE created_by = ?",
        (user_id, user_id)).fetchone()
    if has_records:
        raise HTTPException(
            status_code=409,
            detail="This user has entered patient records. Deactivate the account instead of removing it.")
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    record_audit(conn, admin, "account.removed", row["username"], f"role {row['role']}")
    return {"status": "removed"}


# ----------------------------------------------------------------------------
# Audit log (Administrator)
# ----------------------------------------------------------------------------
@router.get("/audit-log")
def audit_log(limit: int = Query(default=100, ge=1, le=500), admin: dict = Depends(require_admin),
              conn: sqlite3.Connection = Depends(get_db)):
    """Most recent entries first."""
    rows = conn.execute("SELECT id, at, username, action, target, detail FROM audit_log "
                        "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]
