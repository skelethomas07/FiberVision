"""Calibration status rules, the hard gate, and physical resampling."""
import numpy as np
import pandas as pd
import pytest

from sem_fiber_ai.src import calib_audit as CA
from sem_fiber_ai.src import physical as P
from sem_fiber_ai.src import coords as C


def test_resolved_when_bar_and_annotator_agree():
    a = CA.audit_field("f1", physical_nm_per_px=2.0, physical_source="scale_bar",
                       annotator_implied_nm_per_px=2.05, annotator_length_units="nm")
    assert a.calibration_valid and a.status == CA.STATUS_RESOLVED and a.nm_per_px == 2.0


def test_contradictory_when_annotator_scale_disagrees():
    """A_4/A_8-type fields: table implies ~1.06 nm/px, bar says ~6 nm/px."""
    a = CA.audit_field("A_4", physical_nm_per_px=6.25, physical_source="scale_bar",
                       annotator_implied_nm_per_px=1.067, annotator_length_units="nm")
    assert not a.calibration_valid and a.status == CA.STATUS_CONTRADICTORY
    assert a.nm_per_px is None and a.agreement_ratio == pytest.approx(1.067 / 6.25)
    assert a.reason.startswith("annotator_scale_disagrees")


def test_unresolved_when_no_physical_scale_and_never_uses_labels():
    a = CA.audit_field("574_3", physical_nm_per_px=None, physical_source="unknown",
                       annotator_implied_nm_per_px=3.7, annotator_length_units="nm")
    assert not a.calibration_valid and a.status == CA.STATUS_UNRESOLVED
    assert a.nm_per_px is None      # the label-implied 3.7 is NOT substituted
    b = CA.audit_field("x", physical_nm_per_px=2.0, physical_source="overlay_scale_fit")
    assert not b.calibration_valid


def test_disputed_footer_is_contradictory():
    a = CA.audit_field("40s_48-4", physical_nm_per_px=3.1, physical_source="fov_text_disputed")
    assert not a.calibration_valid and a.status == CA.STATUS_CONTRADICTORY


def test_single_source_allowed_and_manual_wins():
    a = CA.audit_field("B_2", physical_nm_per_px=4.673, physical_source="scale_bar",
                       annotator_implied_nm_per_px=None, annotator_length_units="pixels")
    assert a.calibration_valid and a.status == CA.STATUS_SINGLE
    m = CA.audit_field("B_2", physical_nm_per_px=4.673, physical_source="scale_bar",
                       annotator_implied_nm_per_px=1.0, annotator_length_units="nm",
                       manual_nm_per_px=4.7)
    assert m.calibration_valid and m.status == CA.STATUS_MANUAL and m.nm_per_px == 4.7


def test_apply_calibration_sets_nan_nm_for_invalid_fields_and_gate():
    df = pd.DataFrame({"image_id": ["ok", "ok", "bad"], "width_px": [10.0, 20.0, 15.0],
                       "width_nm": [1, 1, 1], "nm_per_pixel": [9, 9, 9]})
    audits = [CA.audit_field("ok", physical_nm_per_px=2.0, physical_source="fov_text"),
              CA.audit_field("bad", physical_nm_per_px=None, physical_source="unknown")]
    out = CA.apply_calibration_to_labels(df, audits)
    assert np.allclose(out.loc[out.image_id == "ok", "width_nm"], [20.0, 40.0])
    assert out.loc[out.image_id == "bad", "width_nm"].isna().all()
    assert out.loc[out.image_id == "bad", "nm_per_pixel"].isna().all()
    assert (out.loc[out.image_id == "bad", "calibration_status"] == CA.STATUS_UNRESOLVED).all()
    gate = CA.calibration_gate(audits)
    assert not gate["passed"] and gate["invalid"][0]["image_id"] == "bad"
    assert CA.calibration_gate(audits, image_ids=["ok"])["passed"]


def test_audit_from_extraction_meta_treats_overlay_fit_as_provenance():
    v7 = {"image_id": "3-8", "calibration": {"nm_per_pixel": 1.0, "source": "fov_text",
                                              "detail": "FOV"},
          "physical_nm_per_px": 1.0, "length_units": "nm", "annotator_units_per_px": 1.02}
    legacy = {"image_id": "3-8", "calibration": {"nm_per_pixel": 1.0, "source": "fov_text",
                                                  "detail": "FOV"},
              "length_units": "nm", "calibration_applied_source": "overlay_scale_fit",
              "nm_per_pixel_applied": 1.02, "scale_fit": {}}
    for meta in (v7, legacy):
        a = CA.audit_from_extraction_meta([meta])[0]
        assert a.status == CA.STATUS_RESOLVED and a.nm_per_px == 1.0
        assert a.annotator_implied_nm_per_px == pytest.approx(1.02)
    # a table in pixels carries no physical evidence and can never resolve a field
    px = {"image_id": "A_8", "calibration": {"nm_per_pixel": None, "source": "unknown"},
          "length_units": "pixels", "annotator_units_per_px": 1.0}
    a = CA.audit_from_extraction_meta([px])[0]
    assert a.status == CA.STATUS_UNRESOLVED and a.annotator_implied_nm_per_px is None


def test_reference_resolution_from_training_only():
    ref = P.reference_nm_per_px({"a": 1.0, "b": 2.0, "c": 5.0, "d": None, "e": float("nan")})
    assert ref == 2.0
    assert P.reference_nm_per_px({"a": None}) is None


def test_resample_plan_and_transforms():
    d = P.plan_resample("f", (960, 1280), 2.0, 1.0, calibration_valid=True)
    assert d.included and d.factor_applied == pytest.approx(2.0)
    assert d.shape_resampled == (1920, 2560)
    assert d.nm_per_px_resampled == pytest.approx(1.0)
    bad = P.plan_resample("f", (960, 1280), 2.0, 1.0, calibration_valid=False)
    assert not bad.included and bad.reason == "calibration_invalid"
    far = P.plan_resample("f", (960, 1280), 10.0, 1.0, calibration_valid=True)
    assert not far.included and far.reason.startswith("resample_factor_out_of_range")
    # rounding: the applied factor is what the image size actually got
    odd = P.plan_resample("f", (101, 203), 1.37, 1.0, calibration_valid=True)
    assert odd.shape_resampled == (round(101 * 1.37), round(203 * 1.37))
    assert odd.factor_applied == pytest.approx(round(203 * 1.37) / 203)
    assert odd.nm_per_px_resampled == pytest.approx(1.37 / odd.factor_applied)

    df = pd.DataFrame({"center_x_px": [100.0], "center_y_px": [50.0], "x1_px": [95.0],
                       "y1_px": [50.0], "x2_px": [105.0], "y2_px": [50.0],
                       "width_px": [10.0], "width_nm": [20.0], "nm_per_pixel": [2.0],
                       "measurement_angle_raster_deg": [0.0]})
    r = P.resample_labels(df, 2.0, 1.0)
    assert r.width_px.iloc[0] == 20.0 and r.center_x_px.iloc[0] == 200.0
    assert r.width_nm.iloc[0] == pytest.approx(20.0)     # physical width unchanged
    assert r.nm_per_pixel.iloc[0] == 1.0 and r.nm_per_pixel_original.iloc[0] == 2.0
    back = P.unresample_predictions(r, 2.0)
    assert back.width_px.iloc[0] == pytest.approx(10.0)
    assert back.center_x_px.iloc[0] == pytest.approx(100.0)


def test_resample_image_scales_line_width():
    import cv2
    from scipy import ndimage as ndi

    img = np.zeros((200, 200), np.float32)
    cv2.line(img, (20, 100), (180, 100), 255.0, 10)
    big = P.resample_image(img, 2.0)
    assert big.shape == (400, 400)
    w_src = int((img[:, 100] > 127).sum())        # cv2 renders 10 or 11 px
    prof = big[:, 200] > 127
    assert abs(int(prof.sum()) - 2 * w_src) <= 1
    # an EDT half-width at the ridge doubles with the factor
    edt = ndi.distance_transform_edt(big > 127)
    assert abs(2 * edt.max() - 2 * w_src) <= 2
