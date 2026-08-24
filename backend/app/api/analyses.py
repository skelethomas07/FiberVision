from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import select

from ..models import AnalysisJob, AnalysisStatus, ModelMeasurement
from ..services.analysis import create_analysis
from ..services.review import get_or_create_review
from .reviews import serialize_review

router = APIRouter(prefix="/api/analyses", tags=["analyses"])


class AnalysisCreate(BaseModel):
    image_id: str


def _job(job: AnalysisJob):
    return {
        "id": job.id,
        "image_id": job.image_id,
        "status": str(job.status),
        "progress": job.progress,
        "error_message": job.error_message,
        "model_version": job.model_version,
        "summary": job.summary_json or {},
    }


@router.post("", status_code=status.HTTP_202_ACCEPTED)
def create(payload: AnalysisCreate, request: Request):
    with request.app.state.Session() as session:
        try:
            job = create_analysis(session, payload.image_id, request.app.state.queue, request.app.state.model_version)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _job(job)


@router.get("/{analysis_id}")
def get_analysis(analysis_id: str, request: Request):
    with request.app.state.Session() as session:
        job = session.get(AnalysisJob, analysis_id)
        if job is None:
            raise HTTPException(status_code=404, detail="analysis not found")
        return _job(job)


@router.get("/{analysis_id}/result")
def get_result(analysis_id: str, request: Request):
    with request.app.state.Session() as session:
        job = session.get(AnalysisJob, analysis_id)
        if job is None:
            raise HTTPException(status_code=404, detail="analysis not found")
        if job.status != AnalysisStatus.DONE:
            raise HTTPException(status_code=409, detail=f"analysis is {job.status}")
        rows = session.scalars(
            select(ModelMeasurement)
            .where(ModelMeasurement.analysis_id == analysis_id)
            .order_by(ModelMeasurement.created_at, ModelMeasurement.id)
        ).all()
        return {
            "analysis_id": job.id,
            "image_id": job.image_id,
            "image_url": f"/api/images/{job.image_id}/content",
            "summary": job.summary_json or {},
            "measurements": [
                {
                    "id": row.id,
                    "external_id": row.external_id,
                    "x1": row.x1, "y1": row.y1, "x2": row.x2, "y2": row.y2,
                    "width_px": row.width_px, "width_nm": row.width_nm,
                    "angle_deg": row.angle_deg,
                    "confidence": row.confidence,
                    "source": row.source,
                    "metadata": row.metadata_json or {},
                }
                for row in rows
            ],
        }


@router.get("/{analysis_id}/review")
def get_review(analysis_id: str, request: Request):
    with request.app.state.Session() as session:
        try:
            review = get_or_create_review(session, analysis_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return serialize_review(session, review)
