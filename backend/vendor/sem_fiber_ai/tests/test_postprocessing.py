"""Decoding, duplicate suppression and matching."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sem_fiber_ai.src.matching import MatchConfig, match_measurements
from sem_fiber_ai.src.postprocess import PostConfig, decode_predictions, find_peaks
from sem_fiber_ai.src.utils import line_endpoints


def _maps(h=64, w=64, sites=((20, 20, 10.0, 0.0), (40, 45, 24.0, 60.0))):
    logit = np.full((h, w), -8.0, np.float32)
    width = np.full((h, w), np.log(10.0), np.float32)
    orient = np.zeros((2, h, w), np.float32)
    orient[0] = 1.0
    validity = np.full((h, w), 4.0, np.float32)
    for (y, x, wid, ang) in sites:
        logit[y, x] = 4.0
        width[y - 1:y + 2, x - 1:x + 2] = np.log(wid)
        t = np.deg2rad(ang)
        orient[0, y - 1:y + 2, x - 1:x + 2] = np.cos(2 * t)
        orient[1, y - 1:y + 2, x - 1:x + 2] = np.sin(2 * t)
    return {"center_logit": logit, "segment_logit": np.zeros((h, w), np.float32),
            "orient": orient, "width": width,
            "validity_logit": validity, "logvar": np.full((h, w), -2.0, np.float32)}


def test_peaks_and_decoding():
    df = decode_predictions(_maps(), image_id="t", cfg=PostConfig(peak_threshold=0.5))
    assert len(df) == 2
    assert set(df["width_px"].round(1)) == {10.0, 24.0}
    # measurement line must be perpendicular to the fiber axis
    d = (df["measurement_angle_deg"] - df["local_fiber_angle_deg"]).abs() % 180
    assert np.allclose(d, 90.0)


def test_width_nm_is_nan_without_calibration():
    df = decode_predictions(_maps(), cfg=PostConfig(peak_threshold=0.5))
    assert df["width_nm"].isna().all()
    df2 = decode_predictions(_maps(), nm_per_pixel=2.0,
                             cfg=PostConfig(peak_threshold=0.5))
    assert df2["width_nm"].to_numpy() == pytest.approx(
        df2["width_px"].to_numpy() * 2.0)


def test_implausible_widths_are_rejected():
    m = _maps(sites=((20, 20, 1.0, 0.0), (40, 45, 500.0, 0.0)))
    df = decode_predictions(m, cfg=PostConfig(peak_threshold=0.5))
    assert len(df) == 0


def test_duplicate_suppression_respects_orientation():
    cfg = PostConfig()
    same = pd.DataFrame({
        "center_x_px": [10.0, 14.0], "center_y_px": [10.0, 10.0],
        "local_fiber_angle_deg": [0.0, 2.0], "width_px": [12.0, 12.0],
        "confidence": [0.9, 0.8]})
    crossing = same.copy()
    crossing["local_fiber_angle_deg"] = [0.0, 80.0]
    from sem_fiber_ai.src.postprocess import suppress_duplicates
    assert len(suppress_duplicates(same, cfg)) == 1        # same fiber -> one
    assert len(suppress_duplicates(crossing, cfg)) == 2    # crossing -> keep both


def _table(xs, ys, widths, angles, prefix="annotation"):
    rows = []
    for i, (x, y, w, a) in enumerate(zip(xs, ys, widths, angles), start=1):
        x1, y1, x2, y2 = line_endpoints(x, y, a, w)
        rows.append({f"{prefix}_id": i, "center_x_px": x, "center_y_px": y,
                     "x1_px": x1, "y1_px": y1, "x2_px": x2, "y2_px": y2,
                     "measurement_angle_deg": a, "width_px": w,
                     "width_nm": np.nan, "confidence": 0.9})
    return pd.DataFrame(rows)


def test_matching_is_one_to_one():
    gt = _table([10, 12], [10, 10], [10, 10], [0, 0])
    pred = _table([11], [10], [10], [0], prefix="prediction")
    m = match_measurements(gt, pred, MatchConfig())
    assert len(m) == 1
    assert m["pred_index"].nunique() == 1


def test_matching_respects_the_distance_limit():
    gt = _table([10], [10], [10], [0])
    pred = _table([200], [200], [10], [0], prefix="prediction")
    assert len(match_measurements(gt, pred, MatchConfig())) == 0


def test_matching_can_use_angle_agreement():
    gt = _table([10], [10], [10], [0.0])
    pred = _table([11], [10], [10], [80.0], prefix="prediction")
    strict = match_measurements(gt, pred, MatchConfig(use_angle=True,
                                                      max_angle_deg=20))
    loose = match_measurements(gt, pred, MatchConfig(use_angle=False))
    assert len(strict) == 0 and len(loose) == 1


def test_empty_inputs_do_not_crash():
    empty = pd.DataFrame(columns=["center_x_px", "center_y_px", "width_px",
                                  "measurement_angle_deg"])
    assert len(match_measurements(empty, empty, MatchConfig())) == 0
