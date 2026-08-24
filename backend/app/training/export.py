from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import ImageAsset, TrainingExample


def export_approved_dataset(session: Session, output_path: str | Path) -> int:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    examples = session.scalars(select(TrainingExample).order_by(TrainingExample.created_at, TrainingExample.id)).all()
    with output_path.open("w", encoding="utf-8") as handle:
        for item in examples:
            image = session.get(ImageAsset, item.image_id)
            row = {
                "training_example_id": item.id,
                "review_id": item.review_id,
                "label": item.label,
                "is_fiber": item.is_fiber,
                "measure_here": item.measure_here,
                "geometry": item.geometry_json,
                "original_geometry": item.original_geometry_json,
                "model_measurement_id": item.model_measurement_id,
                "image": {
                    "id": image.id,
                    "filename": image.original_filename,
                    "storage_key": image.storage_key,
                    "nm_per_pixel": image.nm_per_pixel,
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(examples)
