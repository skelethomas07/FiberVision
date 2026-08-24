from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import delete

from ..inference.contracts import InferenceEngine
from ..models import AnalysisJob, AnalysisStatus, ModelMeasurement
from ..storage import ObjectStorage


def _now():
    return datetime.now(timezone.utc)


def run_analysis_job(job_id, session_factory, storage: ObjectStorage, engine: InferenceEngine, work_root: Path) -> None:
    work_root = Path(work_root)
    with session_factory() as session:
        job = session.get(AnalysisJob, job_id)
        if job is None:
            raise LookupError(f"analysis job not found: {job_id}")
        if job.status == AnalysisStatus.DONE:
            return
        image = job.image
        job.status = AnalysisStatus.ANALYZING
        job.progress = 10
        job.started_at = _now()
        job.error_message = None
        session.commit()
        image_key = image.storage_key
        image_name = image.original_filename
        nm_per_pixel = image.nm_per_pixel

    job_dir = work_root / str(job_id)
    shutil.rmtree(job_dir, ignore_errors=True)
    job_dir.mkdir(parents=True, exist_ok=True)
    image_path = job_dir / "input" / Path(image_name).name
    output_dir = job_dir / "output"

    try:
        storage.get_to_path(image_key, image_path)
        result = engine.analyze(image_path, output_dir, nm_per_pixel)
        with session_factory() as session:
            job = session.get(AnalysisJob, job_id)
            job.status = AnalysisStatus.POSTPROCESSING
            job.progress = 85
            session.commit()

        with session_factory() as session:
            job = session.get(AnalysisJob, job_id)
            session.execute(delete(ModelMeasurement).where(ModelMeasurement.analysis_id == job_id))
            for item in result.measurements:
                session.add(ModelMeasurement(
                    analysis_id=job_id,
                    external_id=item.external_id,
                    x1=item.x1, y1=item.y1, x2=item.x2, y2=item.y2,
                    width_px=item.width_px, width_nm=item.width_nm,
                    angle_deg=item.angle_deg, confidence=item.confidence,
                    source=item.source, metadata_json=item.metadata,
                ))
            for artifact_name, artifact_path in result.artifacts.items():
                storage.put_file(f"analyses/{job_id}/{artifact_path.name}", artifact_path)
            job.summary_json = result.summary
            job.status = AnalysisStatus.DONE
            job.progress = 100
            job.completed_at = _now()
            session.commit()
    except Exception as exc:
        with session_factory() as session:
            session.rollback()
            session.execute(delete(ModelMeasurement).where(ModelMeasurement.analysis_id == job_id))
            job = session.get(AnalysisJob, job_id)
            if job is not None:
                job.status = AnalysisStatus.FAILED
                job.progress = 0
                job.error_message = str(exc)
                job.completed_at = _now()
            session.commit()
