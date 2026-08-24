from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from .review_labels import supervision_label

from ..models import (
    AnalysisJob,
    AnalysisStatus,
    ModelMeasurement,
    ReviewEvent,
    ReviewMeasurement,
    ReviewSession,
    ReviewStatus,
    TrainingExample,
)


def _now():
    return datetime.now(timezone.utc)


def _geometry(x1: float, y1: float, x2: float, y2: float, nm_per_pixel: float | None = None) -> dict[str, float | None]:
    width = float(math.hypot(x2 - x1, y2 - y1))
    angle = float((math.degrees(math.atan2(-(y2 - y1), x2 - x1)) + 180.0) % 360.0 - 180.0)
    return {
        "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
        "width_px": width,
        "width_nm": width * float(nm_per_pixel) if nm_per_pixel is not None else None,
        "angle_deg": angle,
    }


def _model_geometry(model: ModelMeasurement) -> dict[str, float | None]:
    return {
        "x1": model.x1, "y1": model.y1, "x2": model.x2, "y2": model.y2,
        "width_px": model.width_px, "width_nm": model.width_nm,
        "angle_deg": model.angle_deg,
    }


def get_or_create_review(session: Session, analysis_id: str) -> ReviewSession:
    existing = session.scalar(select(ReviewSession).where(ReviewSession.analysis_id == analysis_id))
    if existing is not None:
        _ = list(existing.measurements)
        return existing

    analysis = session.get(AnalysisJob, analysis_id)
    if analysis is None:
        raise LookupError(f"analysis not found: {analysis_id}")
    if analysis.status != AnalysisStatus.DONE:
        raise ValueError("analysis must be DONE before review")

    review = ReviewSession(analysis_id=analysis_id, status=ReviewStatus.OPEN)
    session.add(review)
    session.flush()
    models = session.scalars(
        select(ModelMeasurement).where(ModelMeasurement.analysis_id == analysis_id).order_by(ModelMeasurement.created_at, ModelMeasurement.id)
    ).all()
    for model in models:
        session.add(ReviewMeasurement(
            review_id=review.id,
            source_model_measurement_id=model.id,
            x1=model.x1, y1=model.y1, x2=model.x2, y2=model.y2,
            width_px=model.width_px, width_nm=model.width_nm,
            angle_deg=model.angle_deg, active=True, edited=False, source=model.source,
        ))
    session.commit()
    session.refresh(review)
    _ = list(review.measurements)
    return review


def _require_open(session: Session, review_id: str) -> ReviewSession:
    review = session.get(ReviewSession, review_id)
    if review is None:
        raise LookupError(f"review not found: {review_id}")
    if review.status != ReviewStatus.OPEN:
        raise RuntimeError("review is already approved")
    return review


def apply_review_changes(
    session: Session,
    review_id: str,
    *,
    removed_ids: Iterable[str] = (),
    corrected: Iterable[dict[str, Any]] = (),
    added: Iterable[dict[str, Any]] = (),
) -> ReviewSession:
    review = _require_open(session, review_id)
    analysis = session.get(AnalysisJob, review.analysis_id)
    nm_per_pixel = analysis.image.nm_per_pixel

    for measurement_id in removed_ids:
        row = session.get(ReviewMeasurement, str(measurement_id))
        if row is None or row.review_id != review_id:
            raise LookupError(f"review measurement not found: {measurement_id}")
        if row.active:
            row.active = False
            session.add(ReviewEvent(review_id=review_id, action="REMOVE", measurement_id=row.id))

    for item in corrected:
        measurement_id = str(item["id"])
        row = session.get(ReviewMeasurement, measurement_id)
        if row is None or row.review_id != review_id:
            raise LookupError(f"review measurement not found: {measurement_id}")
        before = {"x1": row.x1, "y1": row.y1, "x2": row.x2, "y2": row.y2}
        geom = _geometry(float(item["x1"]), float(item["y1"]), float(item["x2"]), float(item["y2"]), nm_per_pixel)
        for key in ("x1", "y1", "x2", "y2", "width_px", "width_nm", "angle_deg"):
            setattr(row, key, geom[key])
        row.active = True
        row.edited = True
        session.add(ReviewEvent(
            review_id=review_id, action="CORRECT", measurement_id=row.id,
            payload_json={"before": before, "after": geom},
        ))

    for item in added:
        geom = _geometry(float(item["x1"]), float(item["y1"]), float(item["x2"]), float(item["y2"]), nm_per_pixel)
        row = ReviewMeasurement(
            review_id=review_id,
            source_model_measurement_id=None,
            **geom,
            active=True,
            edited=False,
            source="manual",
        )
        session.add(row)
        session.flush()
        session.add(ReviewEvent(review_id=review_id, action="ADD", measurement_id=row.id, payload_json=geom))

    session.commit()
    session.refresh(review)
    _ = list(review.measurements)
    return review


def approve_review(session: Session, review_id: str) -> dict[str, Any]:
    review = session.get(ReviewSession, review_id)
    if review is None:
        raise LookupError(f"review not found: {review_id}")
    existing_count = len(session.scalars(
        select(TrainingExample).where(TrainingExample.review_id == review_id)
    ).all())
    if review.status == ReviewStatus.APPROVED:
        return {"review_id": review.id, "status": ReviewStatus.APPROVED, "training_examples": existing_count}

    analysis = session.get(AnalysisJob, review.analysis_id)
    image_id = analysis.image_id
    rows = session.scalars(
        select(ReviewMeasurement).where(ReviewMeasurement.review_id == review_id).order_by(ReviewMeasurement.created_at, ReviewMeasurement.id)
    ).all()

    count = 0
    for row in rows:
        # A manual line that was later removed was a transient editing action, not final supervision.
        if row.source_model_measurement_id is None and not row.active:
            continue

        model = session.get(ModelMeasurement, row.source_model_measurement_id) if row.source_model_measurement_id else None
        original = _model_geometry(model) if model else None
        geometry = {
            "x1": row.x1, "y1": row.y1, "x2": row.x2, "y2": row.y2,
            "width_px": row.width_px, "width_nm": row.width_nm, "angle_deg": row.angle_deg,
        }

        label, is_fiber, measure_here = supervision_label(
            has_model=model is not None,
            model_source=model.source if model else None,
            active=row.active,
            edited=row.edited,
        )

        session.add(TrainingExample(
            review_id=review_id,
            image_id=image_id,
            review_measurement_id=row.id,
            model_measurement_id=model.id if model else None,
            label=label,
            is_fiber=is_fiber,
            measure_here=measure_here,
            geometry_json=geometry,
            original_geometry_json=original,
        ))
        session.add(ReviewEvent(review_id=review_id, action="KEEP" if label == "AUTO_KEEP" else label, measurement_id=row.id))
        count += 1

    review.status = ReviewStatus.APPROVED
    review.approved_at = _now()
    session.commit()
    return {"review_id": review.id, "status": ReviewStatus.APPROVED, "training_examples": count}
