from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.services.auth import (
    authenticate_user,
    change_password,
    create_session,
    create_user,
    resolve_session,
    revoke_session,
)


def make_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path/'auth.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    return Session()


def test_create_and_authenticate_user_normalizes_email_and_requires_initial_password_change(tmp_path):
    with make_session(tmp_path) as session:
        user = create_user(session, " User@Example.COM ", "Initial-pass-123!")
        assert user.email == "user@example.com"
        assert user.must_change_password is True
        assert user.password_hash != "Initial-pass-123!"
        assert authenticate_user(session, "USER@example.com", "Initial-pass-123!").id == user.id
        assert authenticate_user(session, "user@example.com", "wrong") is None


def test_duplicate_email_is_rejected(tmp_path):
    with make_session(tmp_path) as session:
        create_user(session, "user@example.com", "Initial-pass-123!")
        with pytest.raises(ValueError, match="already exists"):
            create_user(session, " USER@example.com ", "Other-pass-123!")


def test_session_token_is_opaque_resolvable_and_revocable(tmp_path):
    with make_session(tmp_path) as session:
        user = create_user(session, "user@example.com", "Initial-pass-123!")
        token = create_session(session, user, session_days=7)
        assert len(token) >= 32
        assert resolve_session(session, token).id == user.id
        revoke_session(session, token)
        assert resolve_session(session, token) is None


def test_expired_session_is_rejected(tmp_path):
    with make_session(tmp_path) as session:
        user = create_user(session, "user@example.com", "Initial-pass-123!")
        token = create_session(session, user, session_days=7)
        auth_session = user.auth_sessions[0]
        auth_session.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()
        assert resolve_session(session, token) is None


def test_change_password_clears_initial_flag_and_invalidates_old_password(tmp_path):
    with make_session(tmp_path) as session:
        user = create_user(session, "user@example.com", "Initial-pass-123!")
        change_password(session, user, "New-pass-456!")
        assert user.must_change_password is False
        assert authenticate_user(session, user.email, "Initial-pass-123!") is None
        assert authenticate_user(session, user.email, "New-pass-456!").id == user.id


def test_password_minimum_is_eight_characters(tmp_path):
    with make_session(tmp_path) as session:
        user = create_user(session, "short@example.com", "Abcd1234")
        assert authenticate_user(session, user.email, "Abcd1234").id == user.id
        with pytest.raises(ValueError, match="at least 8"):
            create_user(session, "too-short@example.com", "Abc1234")
