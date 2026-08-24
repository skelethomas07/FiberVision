"""Calibration must never invent a scale, and must find one when it exists."""
from __future__ import annotations

import json

import numpy as np
import pytest

from sem_fiber_ai.src.calibration import (Calibration, detect_footer_row,
                                          detect_scale_bar_px, from_fov_text,
                                          resolve_calibration, strip_footer)


def _image_with_footer(h=960, w=1280, footer=64, bar=100):
    rng = np.random.default_rng(0)
    img = (rng.normal(120, 30, (h + footer, w))).clip(0, 255).astype(np.float32)
    img[h:] = 0.0                      # black info panel
    img[h + 20:h + 26, 900:900 + bar] = 255.0   # scale bar
    return img


def test_footer_is_found_geometrically():
    img = _image_with_footer()
    assert detect_footer_row(img) == 960
    body, row = strip_footer(img)
    assert body.shape == (960, 1280) and row == 960


def test_no_footer_returns_none():
    img = np.full((512, 512), 120.0, np.float32)
    assert detect_footer_row(img) is None


def test_footer_detection_is_not_hard_coded_to_one_size():
    img = _image_with_footer(h=700, w=900, footer=40)
    assert detect_footer_row(img) == 700


def test_scale_bar_length():
    img = _image_with_footer(bar=137)
    assert detect_scale_bar_px(img, 960) == 137


def test_fov_text_parsing():
    nmpp, _ = from_fov_text("SED 1.00kV FOV:1280x960nm HV", 1280)
    assert nmpp == pytest.approx(1.0)
    nmpp, _ = from_fov_text("FOV: 2.56 x 1.92 um", 1280)
    assert nmpp == pytest.approx(2.0)
    assert from_fov_text("no field of view here", 1280)[0] is None


def test_unknown_scale_is_reported_not_guessed(tmp_path):
    img = np.full((512, 512), 120.0, np.float32)
    p = tmp_path / "img.png"
    p.write_bytes(b"")
    cal = resolve_calibration(p, img, image_id="x")
    assert cal.source == "unknown"
    assert cal.nm_per_pixel is None and not cal.known
    assert np.isnan(cal.px_to_nm(10.0))


def test_override_wins_and_is_recorded(tmp_path):
    img = _image_with_footer()
    cal = resolve_calibration(tmp_path / "a.png", img, override=2.5)
    assert cal.nm_per_pixel == 2.5 and cal.source == "override"
    assert cal.px_to_nm(4.0) == pytest.approx(10.0)
    assert cal.nm_to_px(10.0) == pytest.approx(4.0)


def test_sidecar_is_used(tmp_path):
    img = np.full((300, 300), 100.0, np.float32)
    p = tmp_path / "img.png"
    (tmp_path / "img.calib.json").write_text(json.dumps({"nm_per_pixel": 0.75}))
    cal = resolve_calibration(p, img)
    assert cal.source == "sidecar" and cal.nm_per_pixel == pytest.approx(0.75)


def test_negative_override_rejected(tmp_path):
    with pytest.raises(ValueError):
        resolve_calibration(tmp_path / "a.png", np.zeros((10, 10), np.float32),
                            override=-1.0)


# --------------------------------------------------------------------------- #
# measurement-unit handling
# --------------------------------------------------------------------------- #
def _synthetic_fibers(h=400, w=400, width_px=16.0, spacing=80, sigma=4.0):
    """Horizontal bright bars of a known width, on a dark background."""
    import cv2
    img = np.full((h, w), 40.0, np.float32)
    centres = list(range(spacing // 2, h, spacing))
    for cy in centres:
        img[int(cy - width_px / 2):int(cy + width_px / 2), :] = 200.0
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    xs, ys = [], []
    for cy in centres:
        for cx in range(40, w - 40, 60):
            xs.append(float(cx))
            ys.append(float(cy))
    return img, np.asarray(xs), np.asarray(ys)


def test_length_to_original_px_conversions():
    from sem_fiber_ai.src.annotation_extraction import length_to_original_px

    # pixels of the overlay, overlay is 1.25x the original
    assert length_to_original_px(20.0, "pixels", reg_scale=1.25) == pytest.approx(25.0)
    # physical units need the calibration
    assert length_to_original_px(50.0, "nm", nm_per_pixel=2.0) == pytest.approx(25.0)
    assert length_to_original_px(0.05, "um", nm_per_pixel=2.0) == pytest.approx(25.0)
    # no calibration -> NaN, never a silent fallback to pixels
    assert np.isnan(length_to_original_px(50.0, "nm", nm_per_pixel=None))


def test_units_inferred_as_pixels_when_lengths_are_pixels():
    from sem_fiber_ai.src.annotation_extraction import infer_length_units

    img, xs, ys = _synthetic_fibers(width_px=16.0)
    angles = np.full(xs.shape, 90.0)          # chords run across horizontal bars
    lengths = np.full(xs.shape, 16.0)         # already in pixels
    res = infer_length_units(img, xs, ys, angles, lengths,
                             reg_scale=1.0, nm_per_pixel=4.0, search=4.0, step=2.0)
    assert res["best"] is not None
    assert res["best"]["units"] == "pixels"
    assert res["best"]["fwhm_over_width_median"] == pytest.approx(1.0, abs=0.35)


def test_units_inferred_as_nm_when_lengths_are_physical():
    from sem_fiber_ai.src.annotation_extraction import infer_length_units

    img, xs, ys = _synthetic_fibers(width_px=16.0)
    angles = np.full(xs.shape, 90.0)
    # same fibers, but the table reports nanometres at 4 nm/px
    lengths = np.full(xs.shape, 16.0 * 4.0)
    res = infer_length_units(img, xs, ys, angles, lengths,
                             reg_scale=1.0, nm_per_pixel=4.0, search=4.0, step=2.0)
    assert res["best"] is not None
    assert res["best"]["units"] == "nm"


def test_unit_inference_reports_untestable_hypotheses():
    from sem_fiber_ai.src.annotation_extraction import infer_length_units

    img, xs, ys = _synthetic_fibers(width_px=16.0)
    angles = np.full(xs.shape, 90.0)
    lengths = np.full(xs.shape, 16.0)
    res = infer_length_units(img, xs, ys, angles, lengths, nm_per_pixel=None,
                             search=4.0, step=2.0)
    by_unit = {c["units"]: c for c in res["candidates"]}
    assert by_unit["pixels"]["ok"] is True
    assert by_unit["nm"]["ok"] is False        # cannot be tested without a scale


def test_coordinate_frame_scale_is_recovered_and_verified():
    """Coordinates written in a resized frame must be detected, not trusted."""
    import cv2
    from sem_fiber_ai.src.annotation_extraction import infer_csv_coordinate_scale

    # bright horizontal bars of known width in a 1280x960 image
    h, w, width_px, spacing = 960, 1280, 16.0, 80
    img = np.full((h, w), 40.0, np.float32)
    centres = list(range(spacing // 2, h, spacing))
    for cy in centres:
        img[int(cy - width_px / 2):int(cy + width_px / 2), :] = 200.0
    img = cv2.GaussianBlur(img, (0, 0), 1.2)

    xs, ys = [], []
    for cy in centres:
        for cx in range(60, w - 60, 70):
            xs.append(float(cx))
            ys.append(float(cy))
    xs, ys = np.asarray(xs), np.asarray(ys)
    ang = np.full(xs.shape, 90.0)          # chords cut across the bars
    wid = np.full(xs.shape, width_px)

    # the table was written in a 1200x900 frame, i.e. 15/16 of true position
    k = 1200 / w
    res = infer_csv_coordinate_scale(img, xs * k, ys * k, ang, wid * k)
    assert res["ok"]
    assert res["scale"] == pytest.approx(1 / k, rel=0.02)
    assert res["contrast"] > 20

    # coordinates already in image pixels must be left alone
    res2 = infer_csv_coordinate_scale(img, xs, ys, ang, wid)
    assert res2["ok"]
    assert res2["scale"] == pytest.approx(1.0, abs=0.02)
