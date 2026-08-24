from io import BytesIO
from PIL import Image
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.inference.contracts import AnalysisResult, MeasurementPrediction
from app.main import create_app
from app.storage import LocalObjectStorage
from app.workers.analysis import run_analysis_job


class RecordingQueue:
    def __init__(self): self.jobs = []
    def enqueue(self, job_id: str): self.jobs.append(job_id)


class FakeEngine:
    def analyze(self, image_path: Path, output_dir: Path, nm_per_pixel=None):
        return AnalysisResult(measurements=[
            MeasurementPrediction(
                external_id="p1", x1=10, y1=10, x2=20, y2=10,
                width_px=10, width_nm=20, angle_deg=0,
                confidence=.95, source="ai",
            )
        ], summary={"n_predictions": 1})

def _valid_jpeg_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("L", (32, 32), color=128).save(buffer, format="JPEG")
    return buffer.getvalue()

def test_upload_analyze_review_and_approve(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path/'api.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    queue = RecordingQueue()
    storage = LocalObjectStorage(tmp_path / "objects")
    app = create_app(session_factory=Session, storage=storage, queue=queue)
    client = TestClient(app)

    upload = client.post(
        "/api/images",
        files={"file": ("sample.jpg", _valid_jpeg_bytes(), "image/jpeg")},
        data={"nm_per_pixel": "2.0"},
    )
    assert upload.status_code == 201
    image_id = upload.json()["id"]

    create = client.post("/api/analyses", json={"image_id": image_id})
    assert create.status_code == 202
    job_id = create.json()["id"]
    assert create.json()["status"] == "QUEUED"
    assert queue.jobs == [job_id]

    run_analysis_job(job_id, Session, storage, FakeEngine(), tmp_path / "work")

    result = client.get(f"/api/analyses/{job_id}/result")
    assert result.status_code == 200
    payload = result.json()
    assert payload["measurements"][0]["width_px"] == 10
    assert payload["image_url"] == f"/api/images/{image_id}/content"

    review = client.get(f"/api/analyses/{job_id}/review")
    assert review.status_code == 200
    review_payload = review.json()
    review_id = review_payload["id"]
    measurement_id = review_payload["measurements"][0]["id"]

    patch = client.patch(
        f"/api/reviews/{review_id}",
        json={
            "removed_ids": [],
            "corrected": [{"id": measurement_id, "x1": 10, "y1": 8, "x2": 10, "y2": 20}],
            "added": [{"x1": 30, "y1": 5, "x2": 30, "y2": 15}],
        },
    )
    assert patch.status_code == 200
    assert len(patch.json()["measurements"]) == 2

    approved = client.post(f"/api/reviews/{review_id}/approve")
    assert approved.status_code == 200
    assert approved.json()["training_examples"] == 2
