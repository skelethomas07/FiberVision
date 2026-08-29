from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

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
        return rows, image_bytes, _safe_base_name(image.original_filename)
    finally:
        session.close()


def _safe_base_name(filename: str) -> str:
    stem = Path(filename).stem.strip()
    cleaned = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("._")
    return cleaned or "sem_image"


def _attachment(content: bytes, media_type: str, filename: str) -> Response:
    fallback = re.sub(r"[^A-Za-z0-9._-]+", "_", filename) or "download"
    encoded = quote(filename)
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{fallback}"; filename*=UTF-8\'\'{encoded}'},
    )


@router.get("/{analysis_id}/exports/csv")
def export_csv(analysis_id: str, request: Request):
    rows, _, base_name = _context(analysis_id, request)
    return _attachment(build_csv(rows), "text/csv; charset=utf-8", f"{base_name}_measurements.csv")


@router.get("/{analysis_id}/exports/overlay")
def export_overlay(analysis_id: str, request: Request):
    rows, image_bytes, base_name = _context(analysis_id, request)
    return _attachment(render_overlay(image_bytes, rows, labeled=False), "image/png", f"{base_name}_overlay.png")


@router.get("/{analysis_id}/exports/labeled")
def export_labeled(analysis_id: str, request: Request):
    rows, image_bytes, base_name = _context(analysis_id, request)
    return _attachment(render_overlay(image_bytes, rows, labeled=True), "image/png", f"{base_name}_labeled.png")


@router.get("/{analysis_id}/exports/bundle")
def export_bundle(analysis_id: str, request: Request):
    rows, image_bytes, base_name = _context(analysis_id, request)
    return _attachment(build_export_zip(image_bytes, rows, base_name=base_name), "application/zip", f"{base_name}_exports.zip")
