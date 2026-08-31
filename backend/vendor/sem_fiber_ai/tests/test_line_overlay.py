"""The preferred extraction path: geometry read straight off a line overlay."""
from __future__ import annotations

import numpy as np
import pytest

from sem_fiber_ai.src.annotation_extraction import (detect_measurement_segments,
                                                    infer_units_from_segments,
                                                    match_segments_to_csv)
from sem_fiber_ai.src.utils import angular_diff_180, line_endpoints

YELLOW = (255, 211, 70)


def _draw(specs, h=400, w=400, stroke=3, color=YELLOW):
    """Render chords the way an annotation tool would: stroked lines on an image."""
    import cv2
    rng = np.random.default_rng(0)
    img = (rng.normal(120, 18, (h, w, 3))).clip(0, 255).astype(np.uint8)
    img[..., 1] = img[..., 0]
    img[..., 2] = img[..., 0]                 # keep the background grayscale
    for (cx, cy, ang, ln) in specs:
        x1, y1, x2, y2 = line_endpoints(cx, cy, ang, ln)
        cv2.line(img, (int(round(x1)), int(round(y1))),
                 (int(round(x2)), int(round(y2))), color, stroke, cv2.LINE_8)
    return img


def _specs():
    rng = np.random.default_rng(3)
    out = []
    for gx in range(50, 360, 60):
        for gy in range(50, 360, 60):
            out.append((float(gx), float(gy), float(rng.uniform(-89, 89)),
                        float(rng.uniform(22, 46))))
    return out


def test_chords_are_recovered_with_correct_geometry():
    specs = _specs()
    img = _draw(specs)
    segs, stroke = detect_measurement_segments(img)
    assert len(segs) == len(specs)
    assert stroke == pytest.approx(4.0, abs=2.0)

    # pair recovered chords back to the specs by nearest centre
    for s in segs:
        best = min(specs, key=lambda t: (t[0] - s.cx) ** 2 + (t[1] - s.cy) ** 2)
        assert abs(best[0] - s.cx) < 2.0 and abs(best[1] - s.cy) < 2.0
        assert angular_diff_180(best[2], s.angle_deg) < 8.0
        assert s.length_px == pytest.approx(best[3], abs=3.0)


def test_matching_to_a_table_needs_no_ocr():
    specs = _specs()
    img = _draw(specs)
    segs, _ = detect_measurement_segments(img)
    order = np.random.default_rng(7).permutation(len(specs))
    lengths = np.array([specs[i][3] for i in order])
    angles = np.array([specs[i][2] for i in order])

    idx, diag = match_segments_to_csv(segs, lengths, angles)
    assert diag["n_matched"] >= int(0.9 * len(specs))
    assert diag["median_length_residual_px"] < 2.5
    assert diag["length_correlation"] > 0.95


def test_units_are_read_off_the_drawing():
    specs = _specs()
    img = _draw(specs)
    segs, _ = detect_measurement_segments(img)
    nm_per_pixel = 4.0
    lengths_nm = np.array([s[3] * nm_per_pixel for s in specs])
    angles = np.array([s[2] for s in specs])

    res = infer_units_from_segments(segs, lengths_nm, angles,
                                    nm_per_pixel=nm_per_pixel)
    assert res["best"]["units"] == "nm"

    lengths_px = np.array([s[3] for s in specs])
    res2 = infer_units_from_segments(segs, lengths_px, angles,
                                     nm_per_pixel=nm_per_pixel)
    assert res2["best"]["units"] == "pixels"


def test_marker_dots_are_not_mistaken_for_chords():
    import cv2
    rng = np.random.default_rng(1)
    img = (rng.normal(120, 15, (200, 200, 3))).clip(0, 255).astype(np.uint8)
    img[..., 1] = img[..., 0]
    img[..., 2] = img[..., 0]
    for (x, y) in [(40, 40), (120, 60), (80, 150)]:
        cv2.circle(img, (x, y), 4, YELLOW, -1)
    segs, _ = detect_measurement_segments(img)
    assert segs == []          # round blobs are rejected by the aspect test
