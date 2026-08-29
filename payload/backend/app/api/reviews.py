from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select

from ..models import ModelMeasurement, ReviewMeasurement, ReviewSession
from ..services.review import apply_review_changes, approve_review

router = APIRouter(prefix="/api/reviews", tags=["reviews"])


class CorrectedLine(BaseModel):
    id: str
    x1: float
    y1: float
    x2: float
    y2: float


class AddedLine(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class ReviewPatch(BaseModel):
    removed_ids: list[str] = Field(default_factory=list)
    corrected: list[CorrectedLine] = Field(default_factory=list)
    added: list[AddedLine] = Field(default_factory=list)


def serialize_review(session, review: ReviewSession):
    rows = session.scalars(
        select(ReviewMeasurement)
        .where(ReviewMeasurement.review_id == review.id)
        .order_by(ReviewMeasurement.created_at, ReviewMeasurement.id)
    ).all()
    model_ids = [row.source_model_measurement_id for row in rows if row.source_model_measurement_id]
    models = session.scalars(select(ModelMeasurement).where(ModelMeasurement.id.in_(model_ids))).all() if model_ids else []
    model_by_id = {model.id: model for model in models}
    return {
        "id": review.id,
        "analysis_id": review.analysis_id,
        "status": str(review.status),
        "measurements": [
            {
                "id": row.id,
                "source_model_measurement_id": row.source_model_measurement_id,
                "x1": row.x1, "y1": row.y1, "x2": row.x2, "y2": row.y2,
                "width_px": row.width_px, "width_nm": row.width_nm,
                "angle_deg": row.angle_deg,
                "active": row.active,
                "edited": row.edited,
                "source": row.source,
                "confidence": (
                    None
                    if row.source_model_measurement_id is None or row.source in {"manual", "visionflux_manual"}
                    else (model_by_id[row.source_model_measurement_id].confidence if row.source_model_measurement_id in model_by_id else None)
                ),
            }
            for row in rows
        ],
    }


@router.patch("/{review_id}")
def patch_review(review_id: str, payload: ReviewPatch, request: Request):
    with request.app.state.Session() as session:
        try:
            review = apply_review_changes(
                session,
                review_id,
                removed_ids=payload.removed_ids,
                corrected=[item.model_dump() for item in payload.corrected],
                added=[item.model_dump() for item in payload.added],
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return serialize_review(session, review)


@router.post("/{review_id}/approve")
def approve(review_id: str, request: Request):
    with request.app.state.Session() as session:
        try:
            return approve_review(session, review_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
