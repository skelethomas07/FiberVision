"""Augmentation must transform coordinates, widths and angles consistently."""
import numpy as np
import pandas as pd
import pytest

from sem_fiber_ai.src import coords as C
from sem_fiber_ai.src.augmentations import AugConfig, GeometricAug


def _table(n=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        ang = rng.uniform(-90, 90)
        w = rng.uniform(6, 30)
        cx, cy = rng.uniform(60, 190), rng.uniform(60, 190)
        x1, y1, x2, y2 = C.chord_endpoints(cx, cy, ang, w)
        rows.append({"annotation_id": len(rows), "center_x_px": cx, "center_y_px": cy,
                     "x1_px": x1, "y1_px": y1, "x2_px": x2, "y2_px": y2,
                     "measurement_angle_deg": ang,
                     "measurement_angle_raster_deg": ang,
                     "fiber_angle_raster_deg": float(C.fiber_angle_from_measurement(ang)),
                     "tensor_fiber_angle_raster_deg": float(C.fiber_angle_from_measurement(ang)),
                     "local_fiber_angle_deg": float(C.fiber_angle_from_measurement(ang)),
                     "width_px": w, "nm_per_pixel": 2.0,
                     "angle_source_convention": "imagej_y_up"})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("seed", range(6))
def test_geometric_aug_keeps_angles_and_widths_consistent(seed):
    img = np.zeros((256, 256), np.float32)
    ann = _table(seed=seed)
    cfg = AugConfig(hflip=0.5, vflip=0.5, rot90=0.5, rotate_deg=30.0,
                    scale_range=(0.7, 1.4), translate_frac=0.05)
    aug = GeometricAug(cfg, np.random.default_rng(seed))
    _out, new, _ = aug(img, ann)
    M = aug.last_M
    s = C.transform_scale(M)
    assert s == pytest.approx(aug.last_scale, rel=1e-6)
    # keep the rows that survived (inside the frame) and compare by index
    for _, r in new.iterrows():
        o = ann.loc[ann.annotation_id == r.annotation_id].iloc[0]
        ex, ey = C.transform_points(M, o.center_x_px, o.center_y_px)
        assert r.center_x_px == pytest.approx(ex, abs=1e-6)
        assert r.center_y_px == pytest.approx(ey, abs=1e-6)
        assert r.width_px == pytest.approx(o.width_px * s, rel=1e-6)
        assert r.nm_per_pixel == pytest.approx(2.0 / s, rel=1e-6)
        exp_ang = C.transform_angle(M, o.measurement_angle_raster_deg)
        assert C.angular_diff_180(r.measurement_angle_raster_deg, exp_ang) < 1e-6
        assert C.angular_diff_180(r.measurement_angle_deg, exp_ang) < 1e-6
        assert C.angular_diff_180(r.fiber_angle_raster_deg,
                                  C.fiber_angle_from_measurement(exp_ang)) < 1e-6
        # fibre angle columns must move with the LINEAR part (flip-safe)
        exp_fib = C.transform_angle(M, o.fiber_angle_raster_deg)
        assert C.angular_diff_180(r.tensor_fiber_angle_raster_deg, exp_fib) < 1e-6
        assert C.angular_diff_180(r.local_fiber_angle_deg, exp_fib) < 1e-6
        # and the measurement stays perpendicular to the fibre
        assert C.angular_diff_180(r.fiber_angle_raster_deg,
                                  r.measurement_angle_raster_deg) == pytest.approx(90.0)
    assert (new["angle_source_convention"] == "raster_y_down").all()


def test_pure_flip_negates_angles():
    img = np.zeros((256, 256), np.float32)
    ann = _table(seed=3)
    cfg = AugConfig(hflip=1.0, vflip=0.0, rot90=0.0, rotate_deg=0.0,
                    scale_range=(1.0, 1.0), translate_frac=0.0)
    aug = GeometricAug(cfg, np.random.default_rng(0))
    _o, new, _ = aug(img, ann)
    assert len(new) == len(ann)
    for _, r in new.iterrows():
        o = ann.loc[ann.annotation_id == r.annotation_id].iloc[0]
        assert C.angular_diff_180(r.fiber_angle_raster_deg, -o.fiber_angle_raster_deg) < 1e-6
        assert r.width_px == pytest.approx(o.width_px)
