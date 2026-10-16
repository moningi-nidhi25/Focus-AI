import logging
import os
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

_log = logging.getLogger(__name__)
_fallback_secret = secrets.token_urlsafe(48)
TOKEN_HOURS = 12


def _secret() -> str:
    s = os.getenv("JWT_SECRET", "").strip()
    if s:
        return s
    _log.warning("JWT_SECRET is not set: using a temporary secret (logins reset on every restart)")
    return _fallback_secret


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode()[:72], bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode()[:72], hashed.encode())
    except ValueError:
        return False


def create_token(user_id: str, role: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(hours=TOKEN_HOURS)
    return jwt.encode({"sub": user_id, "role": role, "exp": exp}, _secret(), algorithm="HS256")


def decode_token(token: str) -> dict:
    return jwt.decode(token, _secret(), algorithms=["HS256"])