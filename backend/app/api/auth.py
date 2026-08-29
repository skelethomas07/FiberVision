from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel

from ..models import User
from ..services.auth import (
    authenticate_user,
    change_password as change_user_password,
    create_session,
    resolve_session,
    revoke_session,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: str
    password: str


class ChangePasswordRequest(BaseModel):
    new_password: str


def _public_user(user: User) -> dict[str, object]:
    return {
        "id": user.id,
        "email": user.email,
        "must_change_password": bool(user.must_change_password),
    }


def _session_token(request: Request) -> str | None:
    return request.cookies.get(request.app.state.settings.auth_cookie_name)


def require_user(request: Request) -> User:
    with request.app.state.Session() as session:
        user = resolve_session(session, _session_token(request))
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="AUTHENTICATION_REQUIRED")
        session.expunge(user)
        return user


def require_ready_user(request: Request) -> User:
    user = require_user(request)
    if user.must_change_password:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="PASSWORD_CHANGE_REQUIRED")
    return user


@router.post("/login")
def login(payload: LoginRequest, request: Request, response: Response):
    with request.app.state.Session() as session:
        user = authenticate_user(session, payload.email, payload.password)
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="INVALID_CREDENTIALS")
        token = create_session(session, user, session_days=request.app.state.settings.auth_session_days)
        data = _public_user(user)
    response.set_cookie(
        key=request.app.state.settings.auth_cookie_name,
        value=token,
        max_age=request.app.state.settings.auth_session_days * 24 * 60 * 60,
        httponly=True,
        secure=request.app.state.settings.auth_cookie_secure,
        samesite="lax",
        path="/",
    )
    return data


@router.get("/me")
def me(user: User = Depends(require_user)):
    return _public_user(user)


@router.post("/change-password")
def change_password(payload: ChangePasswordRequest, request: Request, user: User = Depends(require_user)):
    with request.app.state.Session() as session:
        persistent = session.get(User, user.id)
        if persistent is None or not persistent.is_active:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="AUTHENTICATION_REQUIRED")
        try:
            changed = change_user_password(session, persistent, payload.new_password)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
        return _public_user(changed)


@router.post("/logout")
def logout(request: Request, response: Response):
    token = _session_token(request)
    with request.app.state.Session() as session:
        revoke_session(session, token)
    response.delete_cookie(
        key=request.app.state.settings.auth_cookie_name,
        path="/",
        secure=request.app.state.settings.auth_cookie_secure,
        httponly=True,
        samesite="lax",
    )
    return {"status": "ok"}
