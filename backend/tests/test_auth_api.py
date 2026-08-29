from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.main import create_app
from app.services.auth import create_user
from app.storage import LocalObjectStorage


class NoopQueue:
    def enqueue(self, job_id: str):
        pass


def make_client(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path/'auth-api.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as session:
        create_user(session, "user@example.com", "Initial-pass-123!")
    app = create_app(session_factory=Session, storage=LocalObjectStorage(tmp_path / "objects"), queue=NoopQueue())
    return TestClient(app), Session


def test_anonymous_analysis_api_is_protected(tmp_path):
    client, _ = make_client(tmp_path)
    response = client.post("/api/analyses", json={"image_id": "missing"})
    assert response.status_code == 401


def test_login_sets_session_cookie_and_reports_initial_password_change(tmp_path):
    client, _ = make_client(tmp_path)
    response = client.post("/api/auth/login", json={"email": "USER@example.com", "password": "Initial-pass-123!"})
    assert response.status_code == 200
    assert response.json()["email"] == "user@example.com"
    assert response.json()["must_change_password"] is True
    assert "fibervision_session" in response.cookies

    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["email"] == "user@example.com"

    blocked = client.post("/api/analyses", json={"image_id": "missing"})
    assert blocked.status_code == 403
    assert blocked.json()["detail"] == "PASSWORD_CHANGE_REQUIRED"


def test_password_change_unlocks_application_and_logout_revokes_session(tmp_path):
    client, _ = make_client(tmp_path)
    login = client.post("/api/auth/login", json={"email": "user@example.com", "password": "Initial-pass-123!"})
    assert login.status_code == 200

    change = client.post("/api/auth/change-password", json={"new_password": "New-pass-456!"})
    assert change.status_code == 200
    assert change.json()["must_change_password"] is False

    unlocked = client.post("/api/analyses", json={"image_id": "missing"})
    assert unlocked.status_code == 404

    logout = client.post("/api/auth/logout")
    assert logout.status_code == 200
    assert client.get("/api/auth/me").status_code == 401

    old_login = client.post("/api/auth/login", json={"email": "user@example.com", "password": "Initial-pass-123!"})
    assert old_login.status_code == 401
    new_login = client.post("/api/auth/login", json={"email": "user@example.com", "password": "New-pass-456!"})
    assert new_login.status_code == 200
