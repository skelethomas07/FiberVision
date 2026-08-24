from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import ImageAsset, TrainingExample
from ..storage import ObjectStorage


@dataclass(frozen=True)
class TrainingBundle:
    root: Path
    labels_csv: Path
    image_dir: Path
    n_examples: int
    n_images: int


def _fiber_angle_from_measurement(angle_deg: float) -> float:
    value = float(angle_deg) + 90.0
    return float((value + 90.0) % 180.0 - 90.0)


def prepare_training_bundle(
    session: Session,
    storage: ObjectStorage,
    output_dir: str | Path,
) -> TrainingBundle:
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    examples = session.scalars(
        select(TrainingExample).order_by(TrainingExample.created_at, TrainingExample.id)
    ).all()
    rows: list[dict] = []
    copied_images: set[str] = set()

    for item in examples:
        image = session.get(ImageAsset, item.image_id)
        if image is None:
            continue
        ext = Path(image.original_filename).suffix.lower() or ".png"
        local_image = image_dir / f"{image.id}{ext}"
        if image.id not in copied_images:
            storage.get_to_path(image.storage_key, local_image)
            copied_images.add(image.id)

        # AUTO_REMOVE is a reviewer-rejected measurement site. Use the original
        # model geometry as a negative site; do not reinterpret the whole local
        # image region as non-fiber.
        geom = item.original_geometry_json if item.label == "AUTO_REMOVE" else item.geometry_json
        if not geom:
            continue
        x1, y1 = float(geom["x1"]), float(geom["y1"])
        x2, y2 = float(geom["x2"]), float(geom["y2"])
        width = float(geom.get("width_px") or math.hypot(x2 - x1, y2 - y1))
        angle = float(geom.get("angle_deg") or 0.0)
        rows.append({
            "image_id": image.id,
            "center_x_px": 0.5 * (x1 + x2),
            "center_y_px": 0.5 * (y1 + y2),
            "x1_px": x1,
            "y1_px": y1,
            "x2_px": x2,
            "y2_px": y2,
            "width_px": width,
            "width_nm": geom.get("width_nm"),
            "nm_per_pixel": image.nm_per_pixel,
            "measurement_angle_deg": angle,
            "local_fiber_angle_deg": _fiber_angle_from_measurement(angle),
            "annotation_confidence": 1.0,
            "ambiguous_crossing": False,
            "is_negative": item.label == "AUTO_REMOVE",
            "supervision": item.label,
            "measure_here": bool(item.measure_here),
            "training_example_id": item.id,
        })

    labels_csv = output_dir / "labels.csv"
    pd.DataFrame(rows).to_csv(labels_csv, index=False)
    return TrainingBundle(
        root=output_dir,
        labels_csv=labels_csv,
        image_dir=image_dir,
        n_examples=len(rows),
        n_images=len(copied_images),
    )
