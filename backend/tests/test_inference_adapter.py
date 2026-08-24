from pathlib import Path

import pandas as pd

from app.inference.sem_fiber_engine import map_predictions


def test_map_predictions_normalizes_v611_dataframe():
    frame = pd.DataFrame([
        {
            "prediction_id": "p-1",
            "x1_px": 10.0,
            "y1_px": 20.0,
            "x2_px": 16.0,
            "y2_px": 28.0,
            "width_px": 10.0,
            "width_nm": 20.0,
            "measurement_angle_deg": 53.13,
            "confidence": 0.91,
            "validity": 0.88,
            "width_sigma_px": 1.4,
            "measurement_source": "thick_recovery",
            "measurement_method": "profile",
        }
    ])

    result = map_predictions(frame)

    assert len(result) == 1
    item = result[0]
    assert item.external_id == "p-1"
    assert (item.x1, item.y1, item.x2, item.y2) == (10.0, 20.0, 16.0, 28.0)
    assert item.width_px == 10.0
    assert item.width_nm == 20.0
    assert item.angle_deg == 53.13
    assert item.confidence == 0.91
    assert item.source == "thick_recovery"
    assert item.metadata["validity"] == 0.88
    assert item.metadata["measurement_method"] == "profile"
