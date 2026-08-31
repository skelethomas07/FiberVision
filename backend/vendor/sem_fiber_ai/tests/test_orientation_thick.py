"""Orientation parity in the raster convention, and thick-branch gating."""
import numpy as np
import pandas as pd
import pytest

from src.coords import angular_diff_180
from src.orientation import (fibre_orientation_summary, orientation_field, orientation_summary,
                             synthetic_parity_check)
from src.synthetic import make_field
from src.thick_experimental import NOT_VALIDATED, measure_thick_from_maps, thick_config, validate_against_gt


def test_structure_tensor_backend_passes_parity_and_a_flipped_one_fails():
    rep = synthetic_parity_check(lambda g: orientation_field(g))
    assert rep["passed"], rep
    assert rep["median_error_deg"] < 2.0
    # the mirrored convention is visibly wrong at oblique angles (about 2x|angle|)
    assert rep["median_error_if_sign_flipped_deg"] > 20.0

    def flipped(g):
        a, c = orientation_field(g)
        return -a, c

    bad = synthetic_parity_check(flipped)
    assert not bad["passed"] and bad["median_error_deg"] > 20.0


def test_wrapped_angles_near_90_are_handled_by_the_parity_check():
    rep = synthetic_parity_check(lambda g: orientation_field(g), angles=(-89, -85, 85, 89))
    assert rep["passed"], rep


def test_orientation_summary_recovers_aligned_synthetic_field():
    fld = make_field(3, H=256, W=256, n_fibres=25, mean_angle=30.0, angle_spread=6.0, n_annotations=30)
    s = orientation_summary(fld.image, fld.mask)
    assert abs(angular_diff_180(s["mean_angle_deg"], 30.0)) < 6.0, s
    assert s["S"] > 0.6
    iso = make_field(4, H=256, W=256, n_fibres=25, angle_spread=90.0, n_annotations=30)
    si = orientation_summary(iso.image, iso.mask)
    assert si["S"] < s["S"]
    fs = fibre_orientation_summary(fld.fibres.rename(columns={"angle": "fiber_angle_raster_deg"})
                                   .assign(length_px=fld.fibres["L"]))
    assert abs(angular_diff_180(fs["mean_angle_deg"], 30.0)) < 8.0


# --------------------------------------------------------------------------- thick gating
def _fake_maps(fld):
    from scipy import ndimage as ndi

    mask = fld.mask
    dist = ndi.distance_transform_edt(mask).astype(np.float32)
    ang, coh = orientation_field(fld.image)
    th = np.deg2rad(ang)
    orient = np.stack([np.cos(2 * th), np.sin(2 * th)]).astype(np.float32)
    seg_logit = np.where(mask, 6.0, -6.0).astype(np.float32)
    return {"segment_logit": seg_logit, "dist": dist, "orient": orient}


def test_thick_branch_is_disabled_by_default_and_labelled_not_validated():
    cfg = {"thick_experimental": {}}
    tc = thick_config(cfg)
    assert tc["enabled"] is False
    fld = make_field(11, H=320, W=320, n_fibres=8, width_median=28.0, width_sigma=0.15,
                     width_min=20.0, width_max=45.0, n_annotations=20)
    out = measure_thick_from_maps(_fake_maps(fld), fld.image, cfg, image_id="T", nm_per_px=None)
    t = out["table"]
    assert out["summary"]["validation_status"] == NOT_VALIDATED
    assert (t["validation_status"] == NOT_VALIDATED).all()
    assert (t["measurement_source"] == "thick_experimental").all()
    assert t["width_nm"].isna().all()
    if len(t):
        assert (t["width_px"] >= tc["min_width_px"]).all()


def test_thick_certificate_only_from_real_gt_that_passes_every_threshold():
    cfg = {"thick_experimental": {"validation": {"min_sites": 10, "min_precision": 0.7, "min_recall": 0.5,
                                                 "max_median_rel_error": 0.15, "max_p90_rel_error": 0.2}}}
    rng = np.random.default_rng(5)
    n = 20
    gt = pd.DataFrame({"center_x_px": rng.uniform(50, 400, n), "center_y_px": rng.uniform(50, 400, n),
                       "width_px": rng.uniform(20, 40, n), "fiber_angle_raster_deg": rng.uniform(-90, 90, n)})
    good = gt.copy()
    good["width_px"] *= 1.05
    assert validate_against_gt(good, gt, cfg, fields=["x"], synthetic=True)["status"] == NOT_VALIDATED
    assert validate_against_gt(good, gt.head(5), cfg, fields=["x"])["status"] == NOT_VALIDATED
    cert = validate_against_gt(good, gt, cfg, fields=["x"])
    assert cert["status"].startswith("VALIDATED") and cert["passed"] and all(cert["checks"].values())
    bad = gt.copy()
    bad["width_px"] *= 1.5
    cert_bad = validate_against_gt(bad, gt, cfg, fields=["x"])
    assert cert_bad["status"] == NOT_VALIDATED and not cert_bad["checks"]["max_median_rel_error"]
    missing = good.head(5)          # low recall
    cert_missing = validate_against_gt(missing, gt, cfg, fields=["x"])
    assert cert_missing["status"] == NOT_VALIDATED and not cert_missing["checks"]["min_recall"]
