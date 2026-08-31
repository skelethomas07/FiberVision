"""Angles, endpoints and augmentation must stay mutually consistent."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sem_fiber_ai.src.augmentations import AugConfig, GeometricAug
from sem_fiber_ai.src.utils import (angular_diff_180, endpoints_to_center_angle_len,
                                    line_endpoints, vec2_to_angle, angle_to_vec2,
                                    wrap_deg_180)


@pytest.mark.parametrize("angle", [-170.0, -90.0, -33.3, 0.0, 45.0, 121.0, 179.0])
@pytest.mark.parametrize("length", [3.0, 17.5, 61.0])
def test_endpoint_roundtrip(angle, length):
    x1, y1, x2, y2 = line_endpoints(100.0, 50.0, angle, length)
    cx, cy, a, l = endpoints_to_center_angle_len(x1, y1, x2, y2)
    assert (cx, cy) == pytest.approx((100.0, 50.0))
    assert l == pytest.approx(length)
    assert angular_diff_180(a, angle) == pytest.approx(0.0, abs=1e-6)


def test_orientation_encoding_is_pi_periodic():
    a = np.array([89.0, -89.0])
    c, s = angle_to_vec2(a)
    back = vec2_to_angle(c, s)
    assert angular_diff_180(back, a) == pytest.approx([0.0, 0.0], abs=1e-6)
    # 89 and -89 are 2 degrees apart, not 178
    assert angular_diff_180(89.0, -89.0) == pytest.approx(2.0)


def test_wrap_keeps_range():
    v = wrap_deg_180(np.array([-180.0, -91.0, 0.0, 90.0, 271.0]))
    assert np.all(v >= -90.0) and np.all(v < 90.0)


def _ann():
    rows = []
    for x, y, ang, w in [(60.0, 40.0, 30.0, 12.0), (150.0, 120.0, -70.0, 20.0)]:
        x1, y1, x2, y2 = line_endpoints(x, y, ang, w)
        rows.append({"center_x_px": x, "center_y_px": y, "x1_px": x1, "y1_px": y1,
                     "x2_px": x2, "y2_px": y2, "measurement_angle_deg": ang,
                     "local_fiber_angle_deg": ang - 90.0, "width_px": w,
                     "nm_per_pixel": 1.0})
    return pd.DataFrame(rows)


def test_augmentation_preserves_width_under_pure_flip():
    img = np.random.default_rng(0).random((200, 200)).astype(np.float32) * 255
    cfg = AugConfig(hflip=1.0, vflip=0.0, rot90=0.0, rotate_deg=0.0,
                    scale_range=(1.0, 1.0), translate_frac=0.0)
    ann = _ann()
    out_img, out_ann, _ = GeometricAug(cfg, np.random.default_rng(0))(img, ann)
    assert out_img.shape == img.shape
    assert len(out_ann) == len(ann)
    assert out_ann["width_px"].to_numpy() == pytest.approx(
        ann["width_px"].to_numpy(), rel=1e-3)


def test_augmentation_scales_width_and_pixel_size_together():
    img = np.zeros((256, 256), np.float32)
    scale = 1.25
    cfg = AugConfig(hflip=0.0, vflip=0.0, rot90=0.0, rotate_deg=0.0,
                    scale_range=(scale, scale), translate_frac=0.0)
    ann = _ann()
    _out, out_ann, _ = GeometricAug(cfg, np.random.default_rng(1))(img, ann)
    assert out_ann["width_px"].to_numpy() == pytest.approx(
        ann["width_px"].to_numpy() * scale, rel=1e-2)
    # physical size must be unchanged: width_px * nm_per_pixel is invariant
    before = ann["width_px"] * ann["nm_per_pixel"]
    after = out_ann["width_px"] * out_ann["nm_per_pixel"]
    assert after.to_numpy() == pytest.approx(before.to_numpy(), rel=1e-2)


def test_augmentation_endpoints_stay_consistent_with_angle():
    img = np.zeros((256, 256), np.float32)
    cfg = AugConfig(hflip=0.5, vflip=0.5, rot90=0.5, rotate_deg=15.0,
                    scale_range=(0.95, 1.05), translate_frac=0.02)
    ann = _ann()
    _out, out_ann, _ = GeometricAug(cfg, np.random.default_rng(3))(img, ann)
    for _, r in out_ann.iterrows():
        _cx, _cy, a, l = endpoints_to_center_angle_len(
            r["x1_px"], r["y1_px"], r["x2_px"], r["y2_px"])
        assert l == pytest.approx(r["width_px"], rel=1e-6)
        assert angular_diff_180(a, r["measurement_angle_deg"]) == pytest.approx(
            0.0, abs=1e-6)
