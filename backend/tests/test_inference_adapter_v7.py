from pathlib import Path

import pandas as pd

from app.inference.sem_fiber_engine import SemFiberEngine, map_predictions


def test_map_predictions_maps_only_accepted_v7_sites():
    frame = pd.DataFrame([
        {
            "site_id": 7,
            "x1_px": 10.0,
            "y1_px": 20.0,
            "x2_px": 16.0,
            "y2_px": 28.0,
            "width_px": 10.0,
            "width_nm": 50.0,
            "measurement_angle_raster_deg": 53.13,
            "fiber_angle_raster_deg": -36.87,
            "confidence": 0.76,
            "validity": 0.88,
            "width_sigma_px": 1.4,
            "boundary_disagreement": 0.11,
            "junction_distance_px": 18.0,
            "coherence": 0.91,
            "branch_id": 3,
            "measurement_source": "model_geometry",
            "rejected_reason": "",
        },
        {
            "site_id": 8,
            "x1_px": 1.0,
            "y1_px": 2.0,
            "x2_px": 3.0,
            "y2_px": 4.0,
            "width_px": 3.0,
            "width_nm": 15.0,
            "measurement_angle_raster_deg": 45.0,
            "fiber_angle_raster_deg": -45.0,
            "confidence": 0.40,
            "validity": 0.20,
            "width_sigma_px": 1.0,
            "boundary_disagreement": 0.5,
            "junction_distance_px": 1.0,
            "coherence": 0.2,
            "branch_id": 4,
            "measurement_source": "model_geometry",
            "rejected_reason": "low_segmentation_confidence",
        },
    ])

    result = map_predictions(frame)

    assert len(result) == 1
    item = result[0]
    assert item.external_id == "7"
    assert (item.x1, item.y1, item.x2, item.y2) == (10.0, 20.0, 16.0, 28.0)
    assert item.width_px == 10.0
    assert item.width_nm == 50.0
    assert item.angle_deg == -53.13
    assert item.confidence == 0.76
    assert item.source == "model_geometry"
    assert item.metadata["validity"] == 0.88
    assert item.metadata["fiber_angle_deg"] == 36.87
    assert item.metadata["boundary_disagreement"] == 0.11
    assert item.metadata["branch_id"] == 3


def test_v7_engine_uses_checkpoint_parent_as_run_directory(tmp_path: Path):
    checkpoint = tmp_path / "run" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"placeholder")

    engine = SemFiberEngine(checkpoint, device="cpu")

    assert engine.run_dir == checkpoint.parent
    assert engine.checkpoint_path == checkpoint
