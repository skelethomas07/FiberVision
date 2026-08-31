"""Regression tests for the v3 fixes.

Each test names the failure it prevents, because every one of them is something
that happened in a real run and cost hours before it was visible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sem_fiber_ai.src.consistency import recovery_gate, width_scale_report
from sem_fiber_ai.src.fiber_prior import (FiberPrior, PriorConfig, audit_prior,
                                          mask_separability_auc,
                                          tune_prior_config)
from sem_fiber_ai.src.utils import duplicate_image_ids, image_id_from_path


# --------------------------------------------------------------------------- #
# image ids
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,expect", [
    # the bug: 462_1 .. 462_9 all collapsed onto "462", merging 36 fields into 4
    ("462_1.png", "462_1"),
    ("462_9.png", "462_9"),
    ("462_10.png", "462_10"),
    ("462.png", "462"),
    ("58_17.png", "58_17"),
    ("40s_48-1.tif", "40s_48-1"),
    # role suffixes still strip, longest form first
    ("3-8_labeled_thickness.png", "3-8"),
    ("2-21_corrected_measurements.csv", "2-21"),
    ("2-21_visionflux_review.png", "2-21"),
    # genuine copy markers still strip
    ("2-11__1_.jpg", "2-11"),
    ("field (1).png", "field"),
    ("field_copy.png", "field"),
])
def test_field_numbers_survive_id_parsing(name, expect):
    assert image_id_from_path(name) == expect


def test_duplicate_ids_are_reported():
    dupes = duplicate_image_ids(["a_1.png", "a_1.tif", "a_2.png"])
    assert set(dupes) == {"a_1"}
    assert len(dupes["a_1"]) == 2


def test_load_records_refuses_colliding_ids(tmp_path):
    import cv2

    from sem_fiber_ai.src.dataset import load_records

    img_dir = tmp_path / "img"
    img_dir.mkdir()
    for name in ("x_1.png", "x_1.jpg"):
        cv2.imwrite(str(img_dir / name), np.zeros((32, 32), np.uint8))
    csv = tmp_path / "labels.csv"
    pd.DataFrame({"image_id": ["x_1"], "center_x_px": [4.0], "center_y_px": [4.0],
                  "width_px": [3.0], "nm_per_pixel": [1.0]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="more than one file"):
        load_records(csv, img_dir)


# --------------------------------------------------------------------------- #
# fiber prior
# --------------------------------------------------------------------------- #
def _mat(n_fibers: int, size: int = 192, seed: int = 0) -> np.ndarray:
    """Synthetic fiber mat: bright straight fibers on a dark background."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), np.float32)
    for _ in range(n_fibers):
        x0, y0 = rng.integers(0, size, 2)
        ang = rng.uniform(0.0, np.pi)
        for t in np.linspace(-size, size, 4 * size):
            x, y = int(x0 + t * np.cos(ang)), int(y0 + t * np.sin(ang))
            if 0 <= x < size and 0 <= y < size:
                img[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = 1.0
    return (img * 160 + 40 + rng.normal(0, 6, img.shape)).clip(0, 255).astype(np.uint8)


def _centres(img: np.ndarray, n: int = 200, seed: int = 1) -> pd.DataFrame:
    ys, xs = np.where(img > 150)
    idx = np.random.default_rng(seed).choice(len(xs), min(n, len(xs)), replace=False)
    return pd.DataFrame({"center_x_px": xs[idx], "center_y_px": ys[idx]})


def test_mask_does_not_swallow_the_frame():
    """v2 flood-filled the bright component, giving 70-100% mask area."""
    img = _mat(30)
    ann = _centres(img)
    prior = FiberPrior.compute(img, PriorConfig(), ann=ann)
    area = float(prior.mask.mean())
    assert area < 0.85, f"mask covers {area:.0%} of the frame"
    # and it must still hold the annotations
    h, w = prior.mask.shape
    xs = ann["center_x_px"].to_numpy(int).clip(0, w - 1)
    ys = ann["center_y_px"].to_numpy(int).clip(0, h - 1)
    assert prior.mask[ys, xs].mean() > 0.9


def test_audit_returns_a_verdict_not_just_a_log():
    """check_priors can only gate on something it is given."""
    img = _mat(25)
    ann = _centres(img)
    rep = audit_prior(FiberPrior.compute(img, PriorConfig(), ann=ann), ann, img)
    assert "ok" in rep and "failures" in rep
    assert isinstance(rep["ok"], bool)


def test_audit_fails_a_leaked_mask():
    """A mask covering everything separates nothing; that must not pass."""
    img = _mat(25)
    ann = _centres(img)
    prior = FiberPrior.compute(img, PriorConfig(), ann=ann)
    prior.mask = np.ones_like(prior.mask)
    prior.mask[:2] = False                      # a sliver of negatives left
    rep = audit_prior(prior, ann, img)
    assert rep["ok"] is False
    assert any("mask_area_fraction" in f for f in rep["failures"])


def test_separability_distinguishes_dense_from_leaked():
    """Area alone cannot: 80% area at AUC 0.97 is a dense mat, not a failure."""
    img = _mat(30)
    good = FiberPrior.compute(img, PriorConfig(), ann=_centres(img)).mask
    # a mask that has grown into the pores: still 95% of the frame, but its
    # boundary no longer has anything to do with brightness
    leaked = np.ones_like(good)
    rng = np.random.default_rng(0)
    leaked[rng.random(leaked.shape) < 0.05] = False
    assert (mask_separability_auc(img, good)
            > mask_separability_auc(img, leaked))


def test_tune_prior_config_returns_usable_knobs():
    img = _mat(25)
    ann = _centres(img)
    out = tune_prior_config([(img, ann)], PriorConfig(),
                            flank_grid=(1.0, 2.0), area_grid=(0.5, 0.8),
                            min_coverage=0.9)
    assert set(out["config"]) == {"flank_scale", "area_max"}
    assert len(out["grid"]) == 4


def test_check_priors_raises_when_strict(tmp_path):
    """The August run logged 'tune config['prior'] first' and trained anyway."""
    import cv2

    from sem_fiber_ai.src.dataset import ImageRecord
    from sem_fiber_ai.src.train import PriorAuditError, check_priors

    img = _mat(25)
    path = tmp_path / "f_1.png"
    cv2.imwrite(str(path), img)
    ann = _centres(img)
    rec = ImageRecord("f_1", path, ann, 1.0)
    rec._image = img
    prior = FiberPrior.compute(img, PriorConfig(), ann=ann)
    prior.mask = np.ones_like(prior.mask)
    prior.mask[:2] = False
    rec._prior = prior

    with pytest.raises(PriorAuditError):
        check_priors([rec], tmp_path, strict=True)
    # and it must still be possible to proceed deliberately
    rep = check_priors([rec], tmp_path, strict=False)
    assert rep["f_1"]["ok"] is False


# --------------------------------------------------------------------------- #
# label-table consistency
# --------------------------------------------------------------------------- #
def test_width_scale_report_flags_magnification_tracking(tmp_path):
    """The real signature: constant width in px, so nm width follows nm/px."""
    rows = []
    for iid, nmpp, med in [("3-8", 5.0, 8.0), ("3-9", 3.34, 9.6), ("3-10", 2.0, 11.2)]:
        for w in np.random.default_rng(0).normal(med, 0.4, 100):
            rows.append({"image_id": iid, "width_px": w, "nm_per_pixel": nmpp})
    csv = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    rep = width_scale_report(csv)
    assert "3" in rep["suspect_specimens"]


def test_width_scale_report_passes_a_consistent_specimen(tmp_path):
    """Same physical width imaged at two magnifications: px scales, nm does not."""
    rows = []
    for iid, nmpp in [("4-1", 2.0), ("4-2", 4.0)]:
        for w in np.random.default_rng(0).normal(40.0 / nmpp, 0.4, 100):
            rows.append({"image_id": iid, "width_px": w, "nm_per_pixel": nmpp})
    csv = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    assert width_scale_report(csv)["suspect_specimens"] == []


def test_recovery_gate_catches_what_rate_alone_misses():
    """rate == 1.000 by construction when the CSV carries coordinates."""
    meta = [
        {"image_id": "A", "recovery_rate": 1.0,
         "orientation_convention": {"median_deviation_deg": 40.2},
         "line_overlay": {"median_length_residual_px": 0.26}},
        {"image_id": "2-21", "recovery_rate": 1.0,
         "orientation_convention": {"median_deviation_deg": 3.4},
         "line_overlay": {}},
    ]
    out = recovery_gate(meta)
    assert out["keep"] == ["2-21"]
    assert "A" in out["drop"]


# --------------------------------------------------------------------------- #
# things that crashed only AFTER a full training run
# --------------------------------------------------------------------------- #
def test_string_annotation_ids_survive_matching():
    """VisionFlux writes ids like 'auto-r3-s0'; int() on one killed cell 8."""
    from sem_fiber_ai.src.matching import MatchConfig, match_measurements

    gt = pd.DataFrame({"annotation_id": ["auto-r3-s0", "auto-r3-s1"],
                       "center_x_px": [10.0, 40.0], "center_y_px": [10.0, 40.0],
                       "width_px": [6.0, 7.0],
                       "measurement_angle_deg": [10.0, 20.0]})
    pred = gt.rename(columns={"annotation_id": "prediction_id"}).copy()
    pred["confidence"] = 0.9
    m = match_measurements(gt, pred, MatchConfig())
    assert len(m) == 2
    assert m["gt_id"].tolist() == ["auto-r3-s0", "auto-r3-s1"]


def test_numeric_ids_stay_numeric():
    from sem_fiber_ai.src.matching import _as_id

    assert _as_id(3) == 3 and _as_id("7") == 7
    assert _as_id("auto-r3-s0") == "auto-r3-s0"


def test_overlay_draws_without_endpoint_columns():
    """A table with only centre/angle/width must still render, not crash."""
    from sem_fiber_ai.src.visualization import draw_measurements

    canvas = np.zeros((64, 64, 3), np.uint8)
    df = pd.DataFrame({"center_x_px": [32.0], "center_y_px": [32.0],
                       "width_px": [10.0], "measurement_angle_deg": [0.0]})
    out = draw_measurements(canvas, df, color=(255, 0, 0))
    assert out.any(), "nothing was drawn"


def test_overlay_prefers_endpoints_when_present():
    from sem_fiber_ai.src.visualization import draw_measurements

    canvas = np.zeros((64, 64, 3), np.uint8)
    df = pd.DataFrame({"center_x_px": [32.0], "center_y_px": [32.0],
                       "width_px": [10.0], "measurement_angle_deg": [0.0],
                       "x1_px": [5.0], "y1_px": [5.0],
                       "x2_px": [5.0], "y2_px": [58.0]})
    out = draw_measurements(canvas, df, color=(255, 0, 0))
    assert out[:, 3:8].any() and not out[:, 40:].any(), "drew the wrong line"
