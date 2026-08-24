"""Regression tests for the v2 changes.

Each test here pins down a specific failure that actually occurred during the
rebuild, so that a future refactor cannot quietly reintroduce it.  The
comments name the failure rather than restating the assertion.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from sem_fiber_ai.src.angle_audit import audit_angles
from sem_fiber_ai.src.dataset import stratified_grouped_split
from sem_fiber_ai.src.fiber_metrics import (distribution_distance,
                                            fiber_level_recall, skeleton_coverage)
from sem_fiber_ai.src.fiber_prior import FiberPrior, PriorConfig, audit_prior
from sem_fiber_ai.src.postprocess import refine_width_from_image
from sem_fiber_ai.src.pseudolabel import (PseudoConfig, calibrate_widths,
                                          measure_image)
from sem_fiber_ai.src.targets import TargetConfig, encode_targets
from sem_fiber_ai.src.utils import angular_diff_180


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _bars(angle_deg: float, size: int = 256, width: int = 7,
          spacing: int = 28) -> np.ndarray:
    """A field of parallel bars at a known raster angle."""
    img = np.zeros((size, size), np.float32)
    t = np.deg2rad(angle_deg)
    ux, uy = np.cos(t), np.sin(t)
    px, py = -uy, ux
    c = size / 2
    for k in range(-size, size, spacing):
        x, y = c + px * k, c + py * k
        cv2.line(img, (int(x - ux * 2 * size), int(y - uy * 2 * size)),
                 (int(x + ux * 2 * size), int(y + uy * 2 * size)), 1.0, width)
    return cv2.GaussianBlur(img, (0, 0), 1.2) * 200 + 20


def _network(n: int = 30, size: int = 384, seed: int = 0,
             per_fiber: int = 3, angle_col: str = "chord"):
    """Fiber network plus a *sparse* sample of annotations, as the real CSVs are."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), np.float32)
    rows = []
    for _ in range(n):
        a = rng.uniform(-90, 90)
        t = np.deg2rad(a)
        ux, uy = np.cos(t), np.sin(t)
        cx, cy = rng.uniform(0, size), rng.uniform(0, size)
        w = rng.uniform(5, 11)
        cv2.line(img, (int(cx - ux * 2 * size), int(cy - uy * 2 * size)),
                 (int(cx + ux * 2 * size), int(cy + uy * 2 * size)), 1.0,
                 int(round(w)))
        for k in rng.uniform(-size, size, per_fiber):
            x, y = cx + ux * k, cy + uy * k
            if 0 <= x < size and 0 <= y < size:
                ang = {"chord": a + 90.0, "fiber": a, "yup": -(a + 90.0)}[angle_col]
                rows.append({"center_x_px": x, "center_y_px": y, "width_px": w,
                             "measurement_angle_deg": ang,
                             "local_fiber_angle_deg": np.nan,
                             "annotation_confidence": 1.0,
                             "ambiguous_crossing": False})
    img = (cv2.GaussianBlur(img, (0, 0), 1.2) * 190 + 25
           + rng.normal(0, 4, (size, size)))
    return img.astype(np.float32), pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# fiber_prior
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("angle", [0.0, 30.0, -45.0, 80.0])
def test_orientation_field_matches_raster_convention(angle):
    """The structure tensor must agree with utils.angle_to_direction(y_sign=+1).

    A sign or 90-degree error here is invisible in the loss (both look like a
    consistent target) and silently corrupts every orientation output.
    """
    from sem_fiber_ai.src.fiber_prior import orientation_field

    ang, coh = orientation_field(_bars(angle), PriorConfig())
    sel = coh > 0.5
    assert sel.mean() > 0.5
    err = float(np.median(angular_diff_180(ang[sel], angle)))
    assert err < 3.0, f"orientation off by {err:.1f} deg at true {angle}"


def test_fiber_mask_covers_annotated_centres():
    """Frangi vesselness alone misses fiber centre-lines.

    Vesselness measures Hessian curvature, so on a fiber several pixels thick it
    peaks at the two flanks and goes weak along the middle -- exactly where an
    annotator puts the measurement point.  Thresholding the response directly
    covered 28% of annotated centres; the ridge-seeded / intensity-grown mask
    covers essentially all of them.  The ignore map is built from this mask, so
    a mask that misses fibers reinstates the bug it exists to prevent.
    """
    img, ann = _network(seed=1)
    prior = FiberPrior.compute(img, PriorConfig(), ann=ann)
    rep = audit_prior(prior, ann)
    assert rep["centre_coverage"] > 0.9, rep


def test_prior_polarity_autodetected_for_dark_fibers():
    img, ann = _network(seed=2)
    inverted = (255.0 - img).astype(np.float32)
    prior = FiberPrior.compute(inverted, PriorConfig(polarity="auto"), ann=ann)
    assert prior.polarity == "dark"
    assert audit_prior(prior, ann)["centre_coverage"] > 0.9


# --------------------------------------------------------------------------- #
# targets -- the central fix
# --------------------------------------------------------------------------- #
def test_unlabelled_fiber_is_ignored_not_negative():
    """The bug that produced 1% recall.

    With a sparse annotation sample, every unannotated fiber pixel was being
    scored as a hard negative -- 56% of the image on a representative field.
    The network's only way to satisfy that is to suppress its confidence
    everywhere, which is what the 0.56 heatmap maximum was.
    """
    img, ann = _network(seed=3)
    prior = FiberPrior.compute(img, PriorConfig(), ann=ann)
    fiber = prior.mask

    old = encode_targets(img.shape, ann, TargetConfig())
    new = encode_targets(img.shape, ann, TargetConfig(), prior=prior)

    def negative_fiber_fraction(t):
        return float((fiber & (t["center"] < 1e-3) & (t["ignore"] < 0.5)).mean())

    assert negative_fiber_fraction(old) > 0.3      # the bug, reproduced
    assert negative_fiber_fraction(new) == 0.0     # and removed
    # supervision must go UP, not merely become safer
    assert new["reg_mask"].mean() > 2 * old["reg_mask"].mean()


def test_focal_label_balance_on_fiber_centreline_improves():
    """The effect of the ignore map on what the loss actually sees.

    Weighting each negative by the focal penalty-reduction factor (1-target)^4,
    the positive-to-negative ratio along the fiber centre-line improves by more
    than an order of magnitude from the identical annotations.
    """
    from skimage.morphology import skeletonize

    img, ann = _network(seed=4)
    prior = FiberPrior.compute(img, PriorConfig(), ann=ann)
    skel = skeletonize(prior.mask)

    def effective_ratio(t):
        keep = skel & (t["ignore"] < 0.5)
        tgt = t["center"][keep]
        n_pos = float((tgt >= 0.999).sum())
        w_neg = float((((1 - tgt) ** 4) * (tgt < 0.999)).sum())
        return n_pos / max(n_pos + w_neg, 1e-9)

    old = effective_ratio(encode_targets(img.shape, ann, TargetConfig()))
    new = effective_ratio(encode_targets(img.shape, ann, TargetConfig(),
                                         prior=prior))
    assert new > 10 * old, f"old={old:.5f} new={new:.5f}"


def test_reviewer_rejections_become_weighted_negatives():
    """A rejected site must override the blanket fiber ignore.

    The reviewer looked at that exact location and said no, which is strictly
    more information than 'unmeasured'.
    """
    img, ann = _network(seed=5)
    prior = FiberPrior.compute(img, PriorConfig(), ann=ann)
    on_fiber = np.argwhere(prior.mask)
    y, x = on_fiber[len(on_fiber) // 2]
    neg = pd.DataFrame([{"center_x_px": float(x), "center_y_px": float(y),
                         "width_px": 8.0}])
    t = encode_targets(img.shape, ann, TargetConfig(), prior=prior, negatives=neg)
    assert t["neg_boost"][y, x] > 1.0
    assert t["ignore"][y, x] == 0.0        # not ignored: it is a known negative
    assert t["center"][y, x] == 0.0


def test_orientation_target_comes_from_the_image_not_the_chord():
    """Chord scatter must not reach the orientation head.

    A hand-drawn chord is only roughly perpendicular to its fiber.  With
    orientation_source='image' the target should track the true fiber direction
    even when the recorded chords are noisy.
    """
    rng = np.random.default_rng(0)
    img = _bars(30.0, size=256)
    ann = pd.DataFrame([{
        "center_x_px": float(x), "center_y_px": float(y), "width_px": 7.0,
        # chords scattered by +/-25 deg around the true perpendicular
        "measurement_angle_deg": 120.0 + rng.normal(0, 25),
        "local_fiber_angle_deg": np.nan, "annotation_confidence": 1.0,
        "ambiguous_crossing": False}
        for x, y in rng.uniform(40, 210, (60, 2))])
    prior = FiberPrior.compute(img, PriorConfig(), ann=ann)
    t = encode_targets(img.shape, ann, TargetConfig(orientation_source="image"),
                       prior=prior)
    sel = t["reg_mask"] > 0.5
    assert sel.sum() > 500
    ang = 0.5 * np.rad2deg(np.arctan2(t["sin2t"][sel], t["cos2t"][sel]))
    assert float(np.median(angular_diff_180(ang, 30.0))) < 5.0


# --------------------------------------------------------------------------- #
# fiber-level metrics
# --------------------------------------------------------------------------- #
def test_fiber_recall_credits_the_same_fiber_at_a_different_point():
    """Predictions on the right fibers at other positions must count.

    The manual chords sit where the annotator happened to click; demanding the
    model reproduce those coordinates measures agreement with an arbitrary
    sampling.  But an orientation-blind version of this metric would credit any
    prediction that merely lands nearby, so the crossing-fiber case is pinned
    too.
    """
    rng = np.random.default_rng(1)
    gt, pred, mask = [], [], np.zeros((512, 512), bool)
    for _ in range(20):
        a = rng.uniform(-90, 90)
        t = np.deg2rad(a)
        ux, uy = np.cos(t), np.sin(t)
        cx, cy = rng.uniform(120, 392, 2)
        w = rng.uniform(5, 11)
        m = np.zeros((512, 512), np.uint8)
        cv2.line(m, (int(cx - ux * 300), int(cy - uy * 300)),
                 (int(cx + ux * 300), int(cy + uy * 300)), 1, int(round(w)))
        mask |= m.astype(bool)
        for k in rng.uniform(-110, 110, 4):
            gt.append({"center_x_px": cx + ux * k, "center_y_px": cy + uy * k,
                       "width_px": w, "local_fiber_angle_deg": a})
        for k in rng.uniform(-110, 110, 8):
            pred.append({"center_x_px": cx + ux * k, "center_y_px": cy + uy * k,
                         "width_px": w, "local_fiber_angle_deg": a})
    gt, pred = pd.DataFrame(gt), pd.DataFrame(pred)

    assert fiber_level_recall(gt, pred)["fiber_recall"] > 0.5
    # same positions, wrong orientation -> must not be credited
    wrong = pred.assign(local_fiber_angle_deg=pred["local_fiber_angle_deg"] + 60)
    assert fiber_level_recall(gt, wrong)["fiber_recall"] < 0.25
    assert skeleton_coverage(pred, mask)["coverage"] > 0.2


def test_distribution_distance_reports_in_input_units():
    rng = np.random.default_rng(0)
    a = rng.normal(10.0, 2.0, 500)
    d_same = distribution_distance(a, rng.normal(10.0, 2.0, 500))
    d_shift = distribution_distance(a, rng.normal(13.0, 2.0, 500))
    assert d_same["wasserstein"] < 0.5
    assert 2.5 < d_shift["wasserstein"] < 3.5      # reads in the input's units
    assert abs(d_shift["median_error"] - 3.0) < 0.5


# --------------------------------------------------------------------------- #
# angle audit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("column,expected", [
    ("chord", "chord, y down (current default)"),
    ("fiber", "fiber direction, y down"),
    ("yup", "chord, y up"),
])
def test_angle_audit_identifies_the_convention(column, expected):
    """A convention error and annotator scatter demand opposite responses.

    One is a bug no amount of training fixes; the other is why orientation is
    supervised from the image.  The audit has to tell them apart.
    """
    img, ann = _network(n=40, size=512, per_fiber=4, seed=6, angle_col=column)
    res = audit_angles(img, ann)
    assert res.best == expected, res.results


def test_angle_audit_flags_large_scatter_without_blaming_the_convention():
    rng = np.random.default_rng(0)
    img, ann = _network(n=40, size=512, per_fiber=4, seed=7)
    ann = ann.assign(measurement_angle_deg=ann["measurement_angle_deg"]
                     + rng.normal(0, 22, len(ann)))
    res = audit_angles(img, ann)
    assert res.best == "chord, y down (current default)"
    assert res.scatter_deg > 8.0
    assert "scatter" in res.verdict


# --------------------------------------------------------------------------- #
# pseudo-labelling
# --------------------------------------------------------------------------- #
def test_pseudo_label_calibration_removes_the_fwhm_offset():
    """FWHM is not the estimator a human uses.

    On a blurred edge it exceeds the drawn width by an amount set by the
    instrument rather than the specimen, so it is close to one multiplicative
    constant -- which the manual CSVs pin down.  This is the best use of the
    answer key: fixing the one systematic degree of freedom the classical
    estimator cannot know.
    """
    frames, manual = [], []
    for s in range(3):
        img, ann = _network(n=25, size=384, per_fiber=8, seed=20 + s)
        d = measure_image(img, image_id=f"f{s}", cfg=PseudoConfig())
        if not len(d):
            pytest.skip("no pseudo-labels on the synthetic field")
        frames.append(d)
        manual.append(pd.DataFrame({"image_id": f"f{s}",
                                    "width_px": ann["width_px"].to_numpy()}))
    pseudo = pd.concat(frames, ignore_index=True)
    manual = pd.concat(manual, ignore_index=True)

    def bias(p):
        return abs(float(np.median(p["width_px"]) / np.median(manual["width_px"]) - 1))

    before = bias(pseudo)
    cal, info = calibrate_widths(pseudo, manual)
    assert bias(cal) < max(0.10, before / 2), (before, bias(cal), info)


def test_pseudo_labels_are_marked_as_generated():
    """Nothing downstream may mistake these for human measurement."""
    img, _ = _network(n=20, size=256, seed=8)
    df = measure_image(img, image_id="x", cfg=PseudoConfig())
    if not len(df):
        pytest.skip("no pseudo-labels on the synthetic field")
    assert (df["annotation_confidence"] < 1.0).all()
    assert df["source_csv"].str.startswith("pseudo:").all()


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #
def test_stratified_split_fills_val_and_test_with_singleton_strata():
    """Per-stratum fractions round to zero when a stratum holds one field.

    The dataset has four calibration regimes across eleven fields, so several
    strata are singletons; taking a fixed fraction inside each one produced
    empty val and test sets.
    """
    ids = list("abcdefghijk")
    strata = {"a": "u", "b": "u", "c": "u", "d": "u", "e": "n1", "f": "n1",
              "g": "n2", "h": "n2", "i": "n2", "j": "n3", "k": "n3"}
    out = stratified_grouped_split(ids, strata, val_frac=0.2, test_frac=0.2)
    assert out["val"] and out["test"]
    assert sorted(sum(out.values(), [])) == sorted(ids)
    assert not (set(out["train"]) & set(out["val"]) & set(out["test"]))
    # test must not sit entirely inside one calibration regime
    assert len({strata[i] for i in out["test"]}) > 1


def test_specimen_grouping_keeps_fields_of_one_sample_together():
    """Five fields of one specimen are not five independent observations.

    Near-duplicate detection groups images that look alike; it does not know
    that 40s_48-1 and 40s_48-5 came from the same sample. Splitting them across
    train and test leaks the specimen and flatters every held-out number.
    """
    from sem_fiber_ai.src.dataset import merge_groups, specimen_groups

    ids = ["40s_48-1", "40s_48-2", "40s_48-3", "40s_48-4", "40s_48-5",
           "2-10", "2-21", "2-22", "3-8", "3-13", "7-2"]
    g = specimen_groups(ids)
    assert len(set(g.values())) == 4          # 11 images, 4 specimens
    assert g["40s_48-1"] == g["40s_48-5"]
    assert g["2-10"] != g["3-8"]

    merged = merge_groups(g, {i: i for i in ids})
    strata = {i: ("uncalibrated" if i.startswith("40s") else "nmpp1.5-3")
              for i in ids}
    out = stratified_grouped_split(ids, strata, val_frac=0.2, test_frac=0.2,
                                   groups=merged)
    # no specimen may appear in two splits
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not ({g[i] for i in out[a]} & {g[i] for i in out[b]})


def test_prior_audit_detects_an_overgrown_mask():
    """Coverage only catches a mask that is too small.

    Once unlabelled fiber is ignored, a mask that has grown into the pores marks
    real background as ignore and leaves the loss with almost no negatives. That
    failure raises coverage rather than lowering it, so it needs its own signal:
    intensity stops separating mask from non-mask.
    """
    img, ann = _network(seed=9)
    good = FiberPrior.compute(img, PriorConfig(), ann=ann)
    over = FiberPrior.compute(img, PriorConfig(dilate_px=14), ann=ann)

    rg = audit_prior(good, ann, img)
    ro = audit_prior(over, ann, img)

    # coverage is blind to over-growth: both look fine by that measure
    assert rg["centre_coverage"] > 0.9 and ro["centre_coverage"] > 0.9
    # separability and the negative budget are not
    assert rg["intensity_separability_auc"] > ro["intensity_separability_auc"]
    assert ro["negative_budget_fraction"] < rg["negative_budget_fraction"] / 2
    assert ro["mask_area_fraction"] > 0.75


# --------------------------------------------------------------------------- #
# postprocess
# --------------------------------------------------------------------------- #
def test_profile_width_returns_nan_when_the_profile_never_returns():
    """It used to return (scan span) x width instead.

    That is a number that looks like a measurement and is not one: on a field of
    known widths it made a correct width head appear 89% wrong.  A cross-check
    that fails this way is worse than no cross-check, because it discredits a
    working model.
    """
    # A fiber with a neighbour 10 px away -- the ordinary situation in a dense
    # separator network.  The scan runs into the neighbour, so on that side the
    # profile never falls back to half maximum.  The old code walked to the end
    # of the array and returned the scan span times the width it was supposed to
    # be checking: a fabricated number that looks like a measurement.
    img = np.zeros((160, 160), np.float32)
    cv2.line(img, (0, 80), (159, 80), 1.0, 9)
    cv2.line(img, (0, 90), (159, 90), 1.0, 9)
    img = cv2.GaussianBlur(img, (0, 0), 1.0) * 200 + 20
    df = pd.DataFrame([{"center_x_px": 80.0, "center_y_px": 80.0,
                        "measurement_angle_deg": 90.0, "width_px": 6.0}])
    assert refine_width_from_image(df, img)["width_px_profile"].isna().all(), (
        "a profile that never returns to half maximum must yield NaN, not a "
        "number derived from the width it was meant to be checking")
    # without the guard the old value comes back, which is what made a correct
    # width head look 89% wrong on a dense field
    naive = refine_width_from_image(df, img, require_return=False)
    assert np.isfinite(naive["width_px_profile"].iloc[0])

    # a uniformly bright field has no peak at all
    flat = np.full((160, 160), 200.0, np.float32)
    assert refine_width_from_image(df, flat)["width_px_profile"].isna().all()

    # an isolated bar still measures
    bar = np.zeros((128, 128), np.float32)
    cv2.line(bar, (0, 64), (127, 64), 1.0, 9)
    bar = cv2.GaussianBlur(bar, (0, 0), 1.0) * 200 + 20
    df2 = pd.DataFrame([{"center_x_px": 64.0, "center_y_px": 64.0,
                         "measurement_angle_deg": 90.0, "width_px": 9.0}])
    got = float(refine_width_from_image(df2, bar)["width_px_profile"].iloc[0])
    assert 7.0 < got < 14.0
