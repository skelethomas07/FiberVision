from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import AnalysisJob, AnalysisStatus, ImageAsset, ModelMeasurement
from app.services.review import approve_review, get_or_create_review
from app.training.bundle import prepare_training_bundle


class Storage:
    def get_to_path(self, key: str, destination: Path):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"sem")


def test_prepare_training_bundle_writes_v611_labels_and_images(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path/'bundle.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as session:
        image = ImageAsset(original_filename="field.jpg", content_type="image/jpeg", storage_key="images/field.jpg", size_bytes=3, nm_per_pixel=2.0)
        session.add(image); session.flush()
        job = AnalysisJob(image_id=image.id, status=AnalysisStatus.DONE, progress=100)
        session.add(job); session.flush()
        keep = ModelMeasurement(analysis_id=job.id, x1=0, y1=0, x2=10, y2=0, width_px=10, width_nm=20, angle_deg=0, confidence=.9, source="ai")
        remove = ModelMeasurement(analysis_id=job.id, x1=0, y1=10, x2=8, y2=10, width_px=8, width_nm=16, angle_deg=0, confidence=.4, source="ai")
        session.add_all([keep, remove]); session.commit()
        review = get_or_create_review(session, job.id)
        row_by_model = {m.source_model_measurement_id: m for m in review.measurements}
        row_by_model[remove.id].active = False
        session.commit()
        approve_review(session, review.id)

    out = tmp_path / "bundle"
    with Session() as session:
        result = prepare_training_bundle(session, Storage(), out)

    labels = pd.read_csv(result.labels_csv)
    assert len(labels) == 2
    assert set(labels["is_negative"].astype(bool)) == {False, True}
    assert set(labels["image_id"]) == {image.id}
    assert labels.loc[~labels["is_negative"].astype(bool), "center_x_px"].iloc[0] == 5
    assert (result.image_dir / f"{image.id}.jpg").read_bytes() == b"sem"
