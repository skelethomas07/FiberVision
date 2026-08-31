import numpy as np
import cv2
import pandas as pd

from sem_fiber_ai.src.thick_fiber import (
    ThickRecoveryConfig, recover_thick_measurements, merge_with_ai
)
from sem_fiber_ai.src.utils import line_endpoints


def _synthetic():
    h, w = 320, 480
    img = np.full((h, w), 30, np.uint8)
    # thin fibres
    cv2.line(img, (10, 60), (460, 80), 170, 8, cv2.LINE_AA)
    cv2.line(img, (20, 250), (450, 230), 170, 10, cv2.LINE_AA)
    # wide fibres
    cv2.line(img, (30, 170), (450, 150), 220, 42, cv2.LINE_AA)
    cv2.line(img, (120, 10), (180, 310), 205, 56, cv2.LINE_AA)
    return cv2.GaussianBlur(img, (0, 0), 0.8)


def test_recovers_wide_not_thin():
    img = _synthetic()
    cfg = ThickRecoveryConfig(
        min_width_px=26, min_sigma=8, spacing_px=30,
        min_ridge_coherence=0.10, segment_support=0.0,
        sigmas=(1, 2, 3, 4, 6, 8, 12, 16, 24),
    )
    df, diag = recover_thick_measurements(img, image_id="syn", cfg=cfg)
    assert len(df) > 2
    assert (df["width_px"] >= 26).all()
    assert df["width_px"].max() > 35
    assert diag["n_candidate_pixels"] > 0


def test_merge_replaces_same_direction_flanks():
    thick = pd.DataFrame([{
        "image_id": "x", "prediction_id": 1,
        "center_x_px": 100.0, "center_y_px": 100.0,
        "x1_px": 100.0, "y1_px": 75.0, "x2_px": 100.0, "y2_px": 125.0,
        "measurement_angle_deg": 90.0, "local_fiber_angle_deg": 0.0,
        "width_px": 50.0, "width_nm": np.nan, "nm_per_pixel": np.nan,
        "confidence": 0.9, "validity": 0.9, "width_sigma_px": 2.0,
        "rejected_reason": "", "measurement_source": "thick_recovery",
        "recovered_thick": True, "scale_sigma_px": 16.0,
        "edt_width_px": 48.0, "profile_width_px": 50.0,
        "profile_contrast": 0.3, "measurement_method": "profile_fwhm",
        "width_calibrated": False,
    }])
    rows = []
    # same-direction narrow flank -> should be replaced
    for y in (85.0, 115.0):
        x1, y1, x2, y2 = line_endpoints(100.0, y, 90.0, 12.0, 1.0)
        rows.append({
            "image_id": "x", "prediction_id": len(rows)+1,
            "center_x_px": 100.0, "center_y_px": y,
            "x1_px": x1, "y1_px": y1, "x2_px": x2, "y2_px": y2,
            "measurement_angle_deg": 90.0, "local_fiber_angle_deg": 0.0,
            "width_px": 12.0, "width_nm": np.nan, "nm_per_pixel": np.nan,
            "confidence": 0.8, "validity": 0.8, "width_sigma_px": 1.0,
            "rejected_reason": "", "width_calibrated": False,
        })
    # crossing thin fibre -> must survive
    x1, y1, x2, y2 = line_endpoints(100.0, 100.0, 0.0, 10.0, 1.0)
    rows.append({
        "image_id": "x", "prediction_id": 3,
        "center_x_px": 100.0, "center_y_px": 100.0,
        "x1_px": x1, "y1_px": y1, "x2_px": x2, "y2_px": y2,
        "measurement_angle_deg": 0.0, "local_fiber_angle_deg": 90.0,
        "width_px": 10.0, "width_nm": np.nan, "nm_per_pixel": np.nan,
        "confidence": 0.8, "validity": 0.8, "width_sigma_px": 1.0,
        "rejected_reason": "", "width_calibrated": False,
    })
    ai = pd.DataFrame(rows)
    combined, stats = merge_with_ai(ai, thick, ThickRecoveryConfig())
    assert stats["n_ai_replaced"] == 2
    assert stats["n_thick_added"] == 1
    assert len(combined) == 2
    assert (combined["measurement_source"] == "thick_recovery").sum() == 1
    assert np.isclose(combined.loc[combined["measurement_source"] == "ai", "width_px"].iloc[0], 10.0)
