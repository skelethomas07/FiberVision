from __future__ import annotations

from .config import get_settings
from .db import SessionLocal
from .inference.sem_fiber_engine import SemFiberEngine
from .storage import build_storage
from .workers.analysis import run_analysis_job

_engine = None
_storage = None


def _get_engine():
    global _engine
    settings = get_settings()
    if _engine is None:
        _engine = SemFiberEngine(
            settings.model_checkpoint,
            device=settings.model_device,
            width_calibration_path=settings.width_calibration_path,
            calibration_table_path=settings.calibration_table_path,
            thick_recovery=True,
        )
    return _engine


def _get_storage():
    global _storage
    if _storage is None:
        _storage = build_storage(get_settings())
    return _storage


def run_job(job_id: str) -> None:
    settings = get_settings()
    run_analysis_job(job_id, SessionLocal, _get_storage(), _get_engine(), settings.work_dir)
