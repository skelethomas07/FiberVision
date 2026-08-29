from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile

from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.main import create_app
from app.models import AnalysisJob, AnalysisStatus, ImageAsset, ReviewMeasurement, ReviewSession, ReviewStatus
from app.services.auth import create_user
from app.services.exports import build_csv, build_export_zip, render_overlay
from app.services.visionflux_import import preview_storage_key
from app.storage import LocalObjectStorage


class NoopQueue:
    def enqueue(self, job_id: str):
        pass


def sample_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (64, 48), "black").save(buffer, format="PNG")
    return buffer.getvalue()


def row(**overrides):
    data = dict(
        id="r1", source_model_measurement_id="m1", x1=5.0, y1=10.0, x2=40.0, y2=10.0,
        width_px=35.0, width_nm=70.0, angle_deg=0.0, active=True, edited=False, source="ai",
    )
    data.update(overrides)
    return SimpleNamespace(**data)


def test_csv_contains_only_active_rows_with_stable_index_and_status():
    content = build_csv([
        row(),
        row(id="r2", active=False),
        row(id="r3", source_model_measurement_id=None, source="manual", edited=False, y1=20.0, y2=20.0),
        row(id="r4", edited=True, y1=30.0, y2=30.0),
    ]).decode("utf-8-sig")
    lines = content.strip().splitlines()
    assert lines[0].startswith("index,id,x1,y1,x2,y2,width_px,width_nm,measurement_angle_deg,fiber_angle_deg,source,status")
    assert len(lines) == 4
    assert ",r1," in lines[1] and ",0.000000,90.000000,ai,KEEP" in lines[1]
    assert ",r3," in lines[2] and lines[2].endswith(",manual,MANUAL_ADD")
    assert ",r4," in lines[3] and lines[3].endswith(",ai,CORRECTED")


def test_overlay_preserves_image_size_and_uses_yellow_and_blue_lines():
    output = render_overlay(sample_png(), [row(), row(id="r2", source_model_measurement_id=None, source="manual", y1=20.0, y2=20.0)])
    image = Image.open(BytesIO(output)).convert("RGB")
    assert image.size == (64, 48)
    yellow = image.getpixel((20, 10))
    blue = image.getpixel((20, 20))
    assert yellow[0] > 200 and yellow[1] > 150 and yellow[2] < 150
    assert blue[2] > 200 and blue[0] < 150


def test_labeled_overlay_differs_and_zip_contains_all_exports():
    plain = render_overlay(sample_png(), [row()], labeled=False)
    labeled = render_overlay(sample_png(), [row()], labeled=True)
    assert plain != labeled
    bundle = build_export_zip(sample_png(), [row()], base_name="sample_image_1")
    with ZipFile(BytesIO(bundle)) as archive:
        assert set(archive.namelist()) == {"sample_image_1_measurements.csv", "sample_image_1_overlay.png", "sample_image_1_labeled.png"}
        assert archive.read("sample_image_1_measurements.csv").startswith(b"\xef\xbb\xbf")


def make_export_client(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path/'exports.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    storage = LocalObjectStorage(tmp_path / "objects")
    with Session() as session:
        create_user(session, "user@example.com", "Initial-pass-123!")
        image = ImageAsset(original_filename="sample image (1).tif", content_type="image/png", storage_key="images/sample.png", size_bytes=10, nm_per_pixel=2.0)
        session.add(image); session.flush()
        analysis = AnalysisJob(image_id=image.id, status=AnalysisStatus.DONE, progress=100, model_version="v6.11")
        session.add(analysis); session.flush()
        review = ReviewSession(analysis_id=analysis.id, status=ReviewStatus.OPEN)
        session.add(review); session.flush()
        session.add(ReviewMeasurement(
            review_id=review.id, source_model_measurement_id=None,
            x1=5, y1=10, x2=40, y2=10, width_px=35, width_nm=70,
            angle_deg=0, active=True, edited=False, source="manual",
        ))
        session.commit()
        analysis_id = analysis.id
        image_id = image.id
    storage.put_bytes(preview_storage_key(image_id), sample_png(), "image/png")
    app = create_app(session_factory=Session, storage=storage, queue=NoopQueue())
    return TestClient(app), analysis_id


def unlock(client):
    assert client.post("/api/auth/login", json={"email": "user@example.com", "password": "Initial-pass-123!"}).status_code == 200
    assert client.post("/api/auth/change-password", json={"new_password": "New-pass-456!"}).status_code == 200


def test_export_endpoints_are_authenticated_and_downloadable(tmp_path):
    client, analysis_id = make_export_client(tmp_path)
    anonymous = client.get(f"/api/analyses/{analysis_id}/exports/csv")
    assert anonymous.status_code == 401
    unlock(client)
    csv_response = client.get(f"/api/analyses/{analysis_id}/exports/csv")
    assert csv_response.status_code == 200
    assert csv_response.headers["content-type"].startswith("text/csv")
    assert "attachment" in csv_response.headers["content-disposition"]
    assert "sample_image_1_measurements.csv" in csv_response.headers["content-disposition"]
    bundle = client.get(f"/api/analyses/{analysis_id}/exports/bundle")
    assert bundle.status_code == 200
    assert bundle.headers["content-type"] == "application/zip"
    assert "sample_image_1_exports.zip" in bundle.headers["content-disposition"]
    with ZipFile(BytesIO(bundle.content)) as archive:
        assert set(archive.namelist()) == {"sample_image_1_measurements.csv", "sample_image_1_overlay.png", "sample_image_1_labeled.png"}
