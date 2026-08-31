from pathlib import Path
import pandas as pd
from app.inference.sem_fiber_engine import SemFiberEngine, map_predictions


def test_map_predictions_maps_v612_ai_and_thick_rows():
    frame = pd.DataFrame([
        {
            "prediction_id": 7,
            "x1_px": 10.0, "y1_px": 20.0, "x2_px": 16.0, "y2_px": 28.0,
            "width_px": 10.0, "width_nm": 50.0,
            "measurement_angle_deg": 53.13, "local_fiber_angle_deg": -36.87,
            "confidence": 0.76, "validity": 0.88, "width_sigma_px": 1.4,
            "measurement_source": "ai", "rejected_reason": "",
        },
        {
            "prediction_id": 8,
            "x1_px": 30.0, "y1_px": 40.0, "x2_px": 45.0, "y2_px": 40.0,
            "width_px": 15.0, "width_nm": 75.0,
            "measurement_angle_deg": 0.0, "local_fiber_angle_deg": -90.0,
            "confidence": 0.67, "validity": 0.72, "width_sigma_px": 2.0,
            "measurement_source": "thick_recovery", "measurement_method": "profile",
            "scale_sigma_px": 8.0, "edt_width_px": 14.0, "profile_width_px": 15.0,
            "rejected_reason": "",
        },
    ])
    result = map_predictions(frame)
    assert len(result) == 2
    ai, thick = result
    assert ai.external_id == "7"
    assert ai.angle_deg == -53.13
    assert ai.source == "ai"
    assert ai.metadata["fiber_angle_deg"] == 36.87
    assert ai.metadata["validity"] == 0.88
    assert thick.source == "thick_recovery"
    assert thick.metadata["measurement_method"] == "profile"
    assert thick.metadata["scale_sigma_px"] == 8.0


def test_v612_engine_uses_notebook_selected_thresholds_and_thick_recovery(tmp_path: Path):
    checkpoint = tmp_path / "best_full.pt"
    checkpoint.write_bytes(b"placeholder")
    engine = SemFiberEngine(checkpoint, device="cpu")
    assert engine.peak_threshold == 0.4
    assert engine.min_validity == 0.5
    assert engine.tta is True
    assert engine.recover_thick is True
    assert engine.thick_min_width_px == 18.0
    assert engine.thick_spacing_px == 14.0
    assert engine.thick_min_coherence == 0.45
