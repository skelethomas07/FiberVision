from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.cli.create_user import main
from app.db import Base
from app.models import User


def make_session_factory(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path/'cli.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def password_reader(values):
    iterator = iter(values)
    return lambda prompt: next(iterator)


def test_cli_creates_normalized_user_with_initial_password_flag(tmp_path, capsys):
    Session = make_session_factory(tmp_path)
    result = main(["User@Example.COM"], password_reader=password_reader(["Initial-pass-123!", "Initial-pass-123!"]), session_factory=Session)
    assert result == 0
    with Session() as session:
        user = session.scalar(select(User))
        assert user.email == "user@example.com"
        assert user.must_change_password is True
    assert "Created user: user@example.com" in capsys.readouterr().out


def test_cli_rejects_password_confirmation_mismatch(tmp_path, capsys):
    Session = make_session_factory(tmp_path)
    result = main(["user@example.com"], password_reader=password_reader(["Initial-pass-123!", "different-pass-123!"]), session_factory=Session)
    assert result == 2
    with Session() as session:
        assert session.scalar(select(User)) is None
    assert "do not match" in capsys.readouterr().err


def test_cli_rejects_duplicate_user(tmp_path, capsys):
    Session = make_session_factory(tmp_path)
    reader = password_reader(["Initial-pass-123!", "Initial-pass-123!"])
    assert main(["user@example.com"], password_reader=reader, session_factory=Session) == 0
    reader2 = password_reader(["Other-pass-123!", "Other-pass-123!"])
    assert main(["USER@example.com"], password_reader=reader2, session_factory=Session) == 2
    assert "already exists" in capsys.readouterr().err
