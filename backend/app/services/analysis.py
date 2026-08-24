from __future__ import annotations

from datetime import datetime, timezone
from math import hypot

from sqlalchemy.orm import Session

from ..models import AnalysisJob, AnalysisStatus, ImageAsset, ModelMeasurement
from ..queue import AnalysisQueue


def create_analysis(session: Session, image_id: str, queue: AnalysisQueue, model_version: str = "v6.11") -> AnalysisJob:
    image = session.get(ImageAsset, image_id)
    if image is None:
        raise LookupError(f"image not found: {image_id}")
    job = AnalysisJob(image_id=image_id, status=AnalysisStatus.QUEUED, progress=0, model_version=model_version)
    session.add(job)
    session.commit()
    session.refresh(job)
    try:
        queue.enqueue(job.id)
    except Exception:
        job.status = AnalysisStatus.FAILED
        job.error_message = "failed to enqueue analysis"
        session.commit()
        raise
    return job


def create_imported_analysis(
    session: Session,
    image_id: str,
    measurements: list[dict],
    model_version: str = "v6.11",
) -> AnalysisJob:
    image = session.get(ImageAsset, image_id)
    if image is None:
        raise LookupError(f"image not found: {image_id}")
    now = datetime.now(timezone.utc)
    job = AnalysisJob(
        image_id=image_id,
        status=AnalysisStatus.DONE,
        progress=100,
        model_version=f"VisionFlux import · {model_version}",
        summary_json={
            "source": "visionflux_import",
            "units": "nm" if image.nm_per_pixel is not None else "px",
            "imported_measurements": len(measurements),
        },
        started_at=now,
        completed_at=now,
    )
    session.add(job)
    session.flush()
    for item in measurements:
        width_px = float(
            item.get("width_px")
            or hypot(float(item["x2"]) - float(item["x1"]), float(item["y2"]) - float(item["y1"]))
        )
        session.add(
            ModelMeasurement(
                analysis_id=job.id,
                external_id=None,
                x1=float(item["x1"]),
                y1=float(item["y1"]),
                x2=float(item["x2"]),
                y2=float(item["y2"]),
                width_px=width_px,
                width_nm=(width_px * float(image.nm_per_pixel) if image.nm_per_pixel is not None else None),
                angle_deg=float(item.get("angle_deg", 0.0)),
                confidence=1.0,
                source=str(item.get("source", "visionflux_auto")),
                metadata_json={"imported_from": "visionflux_annotated_image"},
            )
        )
    session.commit()
    session.refresh(job)
    return job
