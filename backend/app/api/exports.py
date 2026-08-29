from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from sqlalchemy import select

from ..models import AnalysisJob, AnalysisStatus, ReviewMeasurement, ReviewSession
from ..services.exports import build_csv, build_export_zip, render_overlay
from ..services.review import get_or_create_review
from ..services.visionflux_import import preview_storage_key

router = APIRouter(prefix="/api/analyses", tags=["exports"])


def _context(analysis_id: str, request: Request):
    session = request.app.state.Session()
    try:
        analysis = session.get(AnalysisJob, analysis_id)
        if analysis is None:
            raise HTTPException(status_code=404, detail="analysis not found")
        if analysis.status != AnalysisStatus.DONE:
            raise HTTPException(status_code=409, detail=f"analysis is {analysis.status}")
        try:
            review = get_or_create_review(session, analysis_id)
        except (LookupError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        rows = session.scalars(
            select(ReviewMeasurement)
            .where(ReviewMeasurement.review_id == review.id)
            .order_by(ReviewMeasurement.created_at, ReviewMeasurement.id)
        ).all()
        image = analysis.image
        try:
            image_bytes = request.app.state.storage.get_bytes(preview_storage_key(image.id))
        except Exception:
            image_bytes = request.app.state.storage.get_bytes(image.storage_key)
        return rows, image_bytes
    finally:
        session.close()


def _attachment(content: bytes, media_type: str, filename: str) -> Response:
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{analysis_id}/exports/csv")
def export_csv(analysis_id: str, request: Request):
    rows, _ = _context(analysis_id, request)
    return _attachment(build_csv(rows), "text/csv; charset=utf-8", "fiber_measurements.csv")


@router.get("/{analysis_id}/exports/overlay")
def export_overlay(analysis_id: str, request: Request):
    rows, image_bytes = _context(analysis_id, request)
    return _attachment(render_overlay(image_bytes, rows, labeled=False), "image/png", "fiber_measurements.png")


@router.get("/{analysis_id}/exports/labeled")
def export_labeled(analysis_id: str, request: Request):
    rows, image_bytes = _context(analysis_id, request)
    return _attachment(render_overlay(image_bytes, rows, labeled=True), "image/png", "fiber_measurements_labeled.png")


@router.get("/{analysis_id}/exports/bundle")
def export_bundle(analysis_id: str, request: Request):
    rows, image_bytes = _context(analysis_id, request)
    return _attachment(build_export_zip(image_bytes, rows), "application/zip", "fiber_analysis_exports.zip")
