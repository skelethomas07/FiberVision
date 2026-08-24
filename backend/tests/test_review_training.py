import json
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import (
    AnalysisJob, AnalysisStatus, ImageAsset, ModelMeasurement, TrainingExample,
)
from app.services.review import apply_review_changes, approve_review, get_or_create_review
from app.training.export import export_approved_dataset


def make_session_factory(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path/'review.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def seed_analysis(Session):
    with Session() as session:
        image = ImageAsset(
            original_filename="sem.jpg", content_type="image/jpeg",
            storage_key="images/sem.jpg", size_bytes=10, nm_per_pixel=2.0,
        )
        session.add(image); session.flush()
        analysis = AnalysisJob(image_id=image.id, status=AnalysisStatus.DONE, progress=100)
        session.add(analysis); session.flush()
        rows = []
        for i, x in enumerate((10.0, 30.0, 50.0)):
            row = ModelMeasurement(
                analysis_id=analysis.id, external_id=f"p{i}",
                x1=x, y1=10, x2=x+10, y2=10,
                width_px=10, width_nm=20, angle_deg=0,
                confidence=.9, source="ai",
            )
            session.add(row); rows.append(row)
        session.commit()
        return analysis.id


def test_approval_materializes_all_four_supervision_types(tmp_path):
    Session = make_session_factory(tmp_path)
    analysis_id = seed_analysis(Session)

    with Session() as session:
        review = get_or_create_review(session, analysis_id)
        original = list(review.measurements)
        assert original[0].source == "ai"
        keep_id, remove_id, correct_id = [m.id for m in original]
        apply_review_changes(
            session,
            review.id,
            removed_ids=[remove_id],
            corrected=[{"id": correct_id, "x1": 50, "y1": 8, "x2": 50, "y2": 20}],
            added=[{"x1": 70, "y1": 5, "x2": 70, "y2": 17}],
        )
        result = approve_review(session, review.id)
        again = approve_review(session, review.id)

    with Session() as session:
        examples = session.scalars(
            select(TrainingExample).where(TrainingExample.review_id == review.id)
        ).all()

    labels = sorted(e.label for e in examples)
    assert labels == ["AUTO_KEEP", "AUTO_REMOVE", "MANUAL_ADD", "MANUAL_CORRECT"]
    removed = next(e for e in examples if e.label == "AUTO_REMOVE")
    assert removed.measure_here is False
    assert removed.is_fiber is None
    corrected = next(e for e in examples if e.label == "MANUAL_CORRECT")
    assert corrected.original_geometry_json["width_px"] == 10
    assert corrected.geometry_json["width_px"] == 12
    assert result["training_examples"] == 4
    assert again["training_examples"] == 4


def test_export_writes_jsonl_with_image_reference(tmp_path):
    Session = make_session_factory(tmp_path)
    analysis_id = seed_analysis(Session)
    with Session() as session:
        review = get_or_create_review(session, analysis_id)
        approve_review(session, review.id)

    output = tmp_path / "approved.jsonl"
    with Session() as session:
        count = export_approved_dataset(session, output)

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert count == 3
    assert len(rows) == 3
    assert rows[0]["image"]["storage_key"] == "images/sem.jpg"
    assert rows[0]["measure_here"] is True


def test_saved_manual_measurement_can_be_corrected_before_approval(tmp_path):
    Session = make_session_factory(tmp_path)
    analysis_id = seed_analysis(Session)
    with Session() as session:
        review = get_or_create_review(session, analysis_id)
        apply_review_changes(session, review.id, added=[{"x1": 70, "y1": 0, "x2": 75, "y2": 0}])
        manual = next(m for m in review.measurements if m.source_model_measurement_id is None)
        apply_review_changes(
            session,
            review.id,
            corrected=[{"id": manual.id, "x1": 70, "y1": 0, "x2": 77, "y2": 0}],
        )
        approve_review(session, review.id)
        example = session.scalar(
            select(TrainingExample).where(
                TrainingExample.review_id == review.id,
                TrainingExample.review_measurement_id == manual.id,
            )
        )
        assert example.label == "MANUAL_ADD"
        assert example.geometry_json["width_px"] == 7
