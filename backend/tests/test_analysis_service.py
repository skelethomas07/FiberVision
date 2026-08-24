from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.inference.contracts import AnalysisResult, MeasurementPrediction
from app.models import AnalysisJob, AnalysisStatus, ImageAsset, ModelMeasurement
from app.services.analysis import create_analysis
from app.workers.analysis import run_analysis_job


class RecordingQueue:
    def __init__(self):
        self.jobs = []

    def enqueue(self, job_id: str) -> None:
        self.jobs.append(job_id)


class MemoryStorage:
    def __init__(self, data: bytes = b"image"):
        self.data = data

    def get_to_path(self, key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.data)

    def put_file(self, key: str, source: Path, content_type: str | None = None) -> None:
        pass


class FakeEngine:
    def analyze(self, image_path: Path, output_dir: Path, nm_per_pixel=None):
        return AnalysisResult(
            measurements=[
                MeasurementPrediction(
                    external_id="p1", x1=1, y1=2, x2=4, y2=6,
                    width_px=5, width_nm=10, angle_deg=53.1,
                    confidence=0.9, source="ai",
                )
            ],
            summary={"n_predictions": 1},
        )


class FailingEngine:
    def analyze(self, image_path: Path, output_dir: Path, nm_per_pixel=None):
        raise RuntimeError("boom")


def make_session_factory(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path/'test.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def seed_image(Session):
    with Session() as session:
        image = ImageAsset(
            original_filename="sample.jpg",
            content_type="image/jpeg",
            storage_key="images/sample.jpg",
            size_bytes=5,
        )
        session.add(image)
        session.commit()
        return image


def test_create_analysis_queues_new_job(tmp_path):
    Session = make_session_factory(tmp_path)
    image = seed_image(Session)
    queue = RecordingQueue()

    with Session() as session:
        job = create_analysis(session, image.id, queue)

    assert job.status == AnalysisStatus.QUEUED
    assert queue.jobs == [job.id]


def test_worker_persists_measurements_and_marks_done(tmp_path):
    Session = make_session_factory(tmp_path)
    image = seed_image(Session)
    queue = RecordingQueue()
    with Session() as session:
        job = create_analysis(session, image.id, queue)

    run_analysis_job(job.id, Session, MemoryStorage(), FakeEngine(), tmp_path / "work")

    with Session() as session:
        saved = session.get(AnalysisJob, job.id)
        measurements = session.scalars(
            select(ModelMeasurement).where(ModelMeasurement.analysis_id == job.id)
        ).all()
    assert saved.status == AnalysisStatus.DONE
    assert saved.progress == 100
    assert saved.summary_json["n_predictions"] == 1
    assert len(measurements) == 1
    assert measurements[0].source == "ai"


def test_worker_failure_is_atomic_and_marks_failed(tmp_path):
    Session = make_session_factory(tmp_path)
    image = seed_image(Session)
    with Session() as session:
        job = create_analysis(session, image.id, RecordingQueue())

    run_analysis_job(job.id, Session, MemoryStorage(), FailingEngine(), tmp_path / "work")

    with Session() as session:
        saved = session.get(AnalysisJob, job.id)
        count = len(session.scalars(
            select(ModelMeasurement).where(ModelMeasurement.analysis_id == job.id)
        ).all())
    assert saved.status == AnalysisStatus.FAILED
    assert "boom" in saved.error_message
    assert count == 0
