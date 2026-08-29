from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import AuthSession, User

_password_hasher = PasswordHasher()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_email(email: str) -> str:
    value = email.strip().lower()
    if not value or "@" not in value or value.startswith("@") or value.endswith("@"):
        raise ValueError("invalid email")
    return value


def _validate_password(password: str) -> None:
    if len(password) < 10:
        raise ValueError("password must be at least 10 characters")


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def create_user(session: Session, email: str, password: str) -> User:
    normalized = normalize_email(email)
    _validate_password(password)
    if session.scalar(select(User).where(User.email == normalized)) is not None:
        raise ValueError("user already exists")
    user = User(
        email=normalized,
        password_hash=_password_hasher.hash(password),
        is_active=True,
        must_change_password=True,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def authenticate_user(session: Session, email: str, password: str) -> User | None:
    try:
        normalized = normalize_email(email)
    except ValueError:
        return None
    user = session.scalar(select(User).where(User.email == normalized))
    if user is None or not user.is_active:
        return None
    try:
        if not _password_hasher.verify(user.password_hash, password):
            return None
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return None
    if _password_hasher.check_needs_rehash(user.password_hash):
        user.password_hash = _password_hasher.hash(password)
        session.commit()
    return user


def create_session(session: Session, user: User, *, session_days: int = 7) -> str:
    token = secrets.token_urlsafe(32)
    auth_session = AuthSession(
        user_id=user.id,
        token_hash=_token_hash(token),
        expires_at=_now() + timedelta(days=session_days),
    )
    session.add(auth_session)
    session.commit()
    return token


def resolve_session(session: Session, token: str | None) -> User | None:
    if not token:
        return None
    auth_session = session.scalar(select(AuthSession).where(AuthSession.token_hash == _token_hash(token)))
    if auth_session is None or auth_session.revoked_at is not None:
        return None
    if _utc(auth_session.expires_at) <= _now():
        return None
    user = auth_session.user
    if not user.is_active:
        return None
    return user


def revoke_session(session: Session, token: str | None) -> None:
    if not token:
        return
    auth_session = session.scalar(select(AuthSession).where(AuthSession.token_hash == _token_hash(token)))
    if auth_session is None or auth_session.revoked_at is not None:
        return
    auth_session.revoked_at = _now()
    session.commit()


def change_password(session: Session, user: User, new_password: str) -> User:
    _validate_password(new_password)
    user.password_hash = _password_hasher.hash(new_password)
    user.must_change_password = False
    session.commit()
    session.refresh(user)
    return user
