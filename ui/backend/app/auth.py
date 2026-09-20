"""OAuth2 password grant with JWT bearer tokens.

This is RFC 6749 section 4.3 as FastAPI models it (`OAuth2PasswordRequestForm` ->
`Authorization: Bearer <jwt>`), with a local user store rather than an external identity
provider. That choice is deliberate: the machine serving the model may have no inbound
internet route, so an authorization-code flow against Google/Entra could not complete its
callback. Everything here works offline.

Design notes:

* **Passwords** are stored as scrypt hashes (stdlib `hashlib.scrypt`, RFC 7914) with a
  per-user 16-byte salt. Verification is constant-time. No third-party KDF dependency.
* **Access tokens** are short-lived (30 min default) HS256 JWTs. **Refresh tokens** are
  long-lived (14 days) and carry `typ: refresh`, so a refresh token cannot be replayed as
  an access token or vice versa.
* **The signing key** is generated on first run and persisted with owner-only ACLs. It is
  never derived from a password, so rotating it invalidates every outstanding token --
  which is the intended "log everyone out" lever.
* Tokens are bearer credentials: anyone holding one is the user. Over a plain-HTTP LAN
  bind they are sniffable, which is why `Settings.validate()` requires TLS or loopback.

The store lives in `ui/backend/auth/` and is git-ignored.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from .config import settings

AUTH_DIR = Path(__file__).resolve().parent.parent / "auth"
USERS_FILE = AUTH_DIR / "users.json"
SECRET_FILE = AUTH_DIR / "secret.key"

ALGORITHM = "HS256"
ACCESS_TTL_S = 30 * 60
REFRESH_TTL_S = 14 * 24 * 3600

# scrypt parameters. N=2**15 costs ~50-100 ms per verification on this class of machine:
# slow enough to make offline cracking expensive, fast enough for an interactive login.
_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1
_DK_LEN = 32


# ---------------------------------------------------------------------------------------
# On-disk state
# ---------------------------------------------------------------------------------------
def _restrict_permissions(path: Path) -> None:
    """Strip inherited ACLs so only the owner and SYSTEM can read a secret file.

    On Windows a file in a user profile is usually already owner-only, but the repo may
    live on a shared drive (it does here: C:\\temp), where Users has read by default.
    icacls is the only reliable way to fix that from Python; failure is non-fatal because
    the alternative is refusing to run at all.
    """
    if sys.platform != "win32":
        try:
            path.chmod(0o600)
        except OSError:
            pass
        return
    try:
        subprocess.run(
            ["icacls", str(path), "/inheritance:r",
             "/grant:r", f"{os.environ.get('USERNAME', '')}:F",
             "/grant:r", "SYSTEM:F"],
            capture_output=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def _signing_key() -> bytes:
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    if SECRET_FILE.is_file():
        data = SECRET_FILE.read_bytes().strip()
        if len(data) >= 32:
            return data
    key = secrets.token_bytes(64)
    SECRET_FILE.write_bytes(key)
    _restrict_permissions(SECRET_FILE)
    return key


def _load_users() -> dict[str, dict[str, Any]]:
    if not USERS_FILE.is_file():
        return {}
    try:
        data = json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data.get("users", {}) if isinstance(data, dict) else {}


def _save_users(users: dict[str, dict[str, Any]]) -> None:
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps({"users": users}, indent=2), encoding="utf-8")
    _restrict_permissions(USERS_FILE)


def user_count() -> int:
    return len(_load_users())


# ---------------------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------------------
def hash_password(password: str) -> dict[str, str]:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P,
        dklen=_DK_LEN, maxmem=64 * 1024 * 1024,
    )
    return {
        "algo": "scrypt",
        "n": str(_SCRYPT_N), "r": str(_SCRYPT_R), "p": str(_SCRYPT_P),
        "salt": salt.hex(),
        "hash": dk.hex(),
    }


def verify_password(password: str, record: dict[str, Any]) -> bool:
    try:
        salt = bytes.fromhex(record["salt"])
        expected = bytes.fromhex(record["hash"])
        n, r, p = int(record["n"]), int(record["r"]), int(record["p"])
    except (KeyError, ValueError, TypeError):
        return False
    try:
        dk = hashlib.scrypt(
            password.encode("utf-8"), salt=salt, n=n, r=r, p=p,
            dklen=len(expected), maxmem=64 * 1024 * 1024,
        )
    except ValueError:
        return False
    return hmac.compare_digest(dk, expected)


def create_user(username: str, password: str) -> None:
    username = username.strip().lower()
    if not username:
        raise ValueError("username must not be empty")
    if len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    users = _load_users()
    users[username] = {"username": username, "password": hash_password(password),
                       "created_at": time.time()}
    _save_users(users)


def delete_user(username: str) -> bool:
    users = _load_users()
    if users.pop(username.strip().lower(), None) is None:
        return False
    _save_users(users)
    return True


def authenticate(username: str, password: str) -> str | None:
    """Return the canonical username on success, None otherwise.

    A miss still runs a scrypt hash so the response time does not reveal whether the
    account exists.
    """
    username = username.strip().lower()
    users = _load_users()
    record = users.get(username)
    if record is None:
        hashlib.scrypt(b"absent", salt=b"0" * 16, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P,
                       dklen=_DK_LEN, maxmem=64 * 1024 * 1024)
        return None
    if not verify_password(password, record.get("password", {})):
        return None
    return username


# ---------------------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------------------
@dataclass
class TokenPair:
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TTL_S


def _encode(subject: str, ttl: int, typ: str) -> str:
    now = int(time.time())
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + ttl,
        "typ": typ,
        "jti": secrets.token_hex(8),
    }
    return jwt.encode(payload, _signing_key(), algorithm=ALGORITHM)


def issue_tokens(username: str) -> TokenPair:
    return TokenPair(
        access_token=_encode(username, ACCESS_TTL_S, "access"),
        refresh_token=_encode(username, REFRESH_TTL_S, "refresh"),
    )


def decode_token(token: str, expected_typ: str) -> str:
    """Return the subject, or raise HTTPException(401)."""
    try:
        payload = jwt.decode(token, _signing_key(), algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="token expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

    # A refresh token must never be accepted where an access token is required.
    if payload.get("typ") != expected_typ:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"expected a {expected_typ} token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    subject = payload.get("sub")
    if not subject or subject not in _load_users():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="unknown subject",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return str(subject)


# ---------------------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------------------
# auto_error=False so we can distinguish "no credential supplied" from "bad credential",
# and so the auth-disabled path below can short-circuit without a 403 from the scheme.
_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token", auto_error=False)


def auth_enabled() -> bool:
    """Auth is on whenever any account exists.

    Making it presence-based rather than a flag means a loopback-only install stays
    zero-friction, while `create_user` is the single, obvious act that turns it on --
    and Settings.validate() refuses a non-loopback bind until that has happened.
    """
    return user_count() > 0


async def require_user(token: str | None = Depends(_oauth2_scheme)) -> str | None:
    if not auth_enabled():
        return None
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_token(token, "access")
