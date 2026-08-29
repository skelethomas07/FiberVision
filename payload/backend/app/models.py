from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class AnalysisStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    ANALYZING = "ANALYZING"
    POSTPROCESSING = "POSTPROCESSING"
    DONE = "DONE"
    FAILED = "FAILED"


class ReviewStatus(str, enum.Enum):
    OPEN = "OPEN"
    APPROVED = "APPROVED"


class ImageAsset(Base):
    __tablename__ = "image_assets"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_id)
    original_filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    storage_key: Mapped[str] = mapped_column(String(500), unique=True)
    size_bytes: Mapped[int] = mapped_column(Integer)
    nm_per_pixel: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_id)
    image_id: Mapped[str] = mapped_column(ForeignKey("image_assets.id", ondelete="CASCADE"), index=True)
    status: Mapped[AnalysisStatus] = mapped_column(String(32), default=AnalysisStatus.QUEUED)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_version: Mapped[str] = mapped_column(String(100), default="v6.11")
    summary_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    image: Mapped[ImageAsset] = relationship()
    measurements: Mapped[list["ModelMeasurement"]] = relationship(back_populates="analysis", cascade="all, delete-orphan")


class ModelMeasurement(Base):
    __tablename__ = "model_measurements"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_id)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analysis_jobs.id", ondelete="CASCADE"), index=True)
    external_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)
    width_px: Mapped[float] = mapped_column(Float)
    width_nm: Mapped[float | None] = mapped_column(Float, nullable=True)
    angle_deg: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(64), default="ai")
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    analysis: Mapped[AnalysisJob] = relationship(back_populates="measurements")


class ReviewSession(Base):
    __tablename__ = "review_sessions"
    __table_args__ = (UniqueConstraint("analysis_id", name="uq_review_analysis"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_id)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analysis_jobs.id", ondelete="CASCADE"), index=True)
    status: Mapped[ReviewStatus] = mapped_column(String(32), default=ReviewStatus.OPEN)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    measurements: Mapped[list["ReviewMeasurement"]] = relationship(cascade="all, delete-orphan")


class ReviewMeasurement(Base):
    __tablename__ = "review_measurements"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_id)
    review_id: Mapped[str] = mapped_column(ForeignKey("review_sessions.id", ondelete="CASCADE"), index=True)
    source_model_measurement_id: Mapped[str | None] = mapped_column(ForeignKey("model_measurements.id"), nullable=True, index=True)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)
    width_px: Mapped[float] = mapped_column(Float)
    width_nm: Mapped[float | None] = mapped_column(Float, nullable=True)
    angle_deg: Mapped[float] = mapped_column(Float)
    active: Mapped[bool] = mapped_column(default=True)
    edited: Mapped[bool] = mapped_column(default=False)
    source: Mapped[str] = mapped_column(String(32), default="model")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class ReviewEvent(Base):
    __tablename__ = "review_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_id)
    review_id: Mapped[str] = mapped_column(ForeignKey("review_sessions.id", ondelete="CASCADE"), index=True)
    action: Mapped[str] = mapped_column(String(32))
    measurement_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class TrainingExample(Base):
    __tablename__ = "training_examples"
    __table_args__ = (UniqueConstraint("review_id", "review_measurement_id", "label", name="uq_training_example"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_id)
    review_id: Mapped[str] = mapped_column(ForeignKey("review_sessions.id", ondelete="CASCADE"), index=True)
    image_id: Mapped[str] = mapped_column(ForeignKey("image_assets.id", ondelete="CASCADE"), index=True)
    review_measurement_id: Mapped[str] = mapped_column(String(36))
    model_measurement_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    label: Mapped[str] = mapped_column(String(32), index=True)
    is_fiber: Mapped[bool | None] = mapped_column(nullable=True)
    measure_here: Mapped[bool] = mapped_column()
    geometry_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    original_geometry_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_id)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    checkpoint_uri: Mapped[str] = mapped_column(String(500))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_id)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(512))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    auth_sessions: Mapped[list["AuthSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class AuthSession(Base):
    __tablename__ = "auth_sessions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    user: Mapped[User] = relationship(back_populates="auth_sessions")
