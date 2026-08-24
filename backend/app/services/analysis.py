from __future__ import annotations

from sqlalchemy.orm import Session

from ..models import AnalysisJob, AnalysisStatus, ImageAsset
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
