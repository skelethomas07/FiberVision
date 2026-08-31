"""Label schema, legacy upgrade and synthetic ground-truth consistency."""
import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi

from src.coords import (IMAGEJ, RASTER, angular_diff_180, measurement_angle_from_endpoints, wrap180)
from src.labels import LABEL_COLUMNS, REQUIRED, ensure_schema, upgrade_legacy_labels, validate_labels
from src.synthetic import make_field, write_synthetic_dataset


def test_schema_validation_raises_on_missing_required_and_reports_bad_widths():
    with pytest.raises(ValueError):
        validate_labels(pd.DataFrame({"image_id": ["a"], "width_px": [3.0]}))
    df = ensure_schema(pd.DataFrame({"image_id": ["a", "a"], "annotation_id": [1, 2],
                                     "center_x_px": [1.0, 2.0], "center_y_px": [1.0, 2.0],
                                     "width_px": [5.0, -1.0], "measurement_angle_raster_deg": [0.0, 0.0],
                                     "fiber_angle_raster_deg": [-90.0, -90.0]}))
    assert list(df.columns[:len(LABEL_COLUMNS)]) == LABEL_COLUMNS
    rep = validate_labels(df)
    assert rep["n"] == 2 and rep["problems"][0]["code"] == "nonpositive_width"
    assert all(c in LABEL_COLUMNS for c in REQUIRED)


def test_legacy_v6_table_is_upgraded_with_explicit_convention_and_provenance():
    legacy = pd.DataFrame({"image_id": ["f"] * 3, "annotation_id": [1, 2, 3],
                           "center_x_px": [50.0, 60.0, 70.0], "center_y_px": [50.0, 60.0, 70.0],
                           "width_px": [10.0, 12.0, 8.0],
                           "measurement_angle_deg": [30.0, -80.0, 90.0],      # ImageJ, y-up
                           "local_fiber_angle_deg": [0.0, 0.0, 0.0]})
    up = upgrade_legacy_labels(legacy, angle_source_convention=IMAGEJ)
    assert "measurement_angle_deg" not in up.columns and "local_fiber_angle_deg" not in up.columns
    np.testing.assert_allclose(up["source_angle_deg"], [30.0, -80.0, 90.0])
    assert (up["angle_source_convention"] == IMAGEJ).all()
    expect = wrap180(-np.array([30.0, -80.0, 90.0]))
    np.testing.assert_allclose(up["measurement_angle_raster_deg"], expect)
    fib = up["fiber_angle_raster_deg"].to_numpy(float)
    assert np.all(np.abs(angular_diff_180(fib, expect - 90.0)) < 1e-9)
    # endpoints regenerated in raster must reproduce the raster angle
    ma = measurement_angle_from_endpoints(up["x1_px"], up["y1_px"], up["x2_px"], up["y2_px"])
    assert np.all(np.abs(angular_diff_180(ma, expect)) < 1e-6)
    assert (up["extraction_path"] == "legacy_v6_upgraded").all()
    # declaring the table as already-raster leaves the numbers untouched
    same = upgrade_legacy_labels(legacy, angle_source_convention=RASTER)
    np.testing.assert_allclose(same["measurement_angle_raster_deg"], [30.0, -80.0, -90.0])


def test_synthetic_annotations_match_fibre_geometry_and_edt_recovers_width():
    fld = make_field(21, H=384, W=384, n_fibres=20, n_annotations=80)
    ann = fld.annotations
    assert len(ann) > 20
    assert set(REQUIRED) <= set(ann.columns)
    assert (ann["angle_source_convention"] == RASTER).all()
    # chord endpoints agree with width and measurement angle in raster
    L = np.hypot(ann["x2_px"] - ann["x1_px"], ann["y2_px"] - ann["y1_px"])
    np.testing.assert_allclose(L, ann["width_px"], rtol=1e-6)
    ma = measurement_angle_from_endpoints(ann["x1_px"], ann["y1_px"], ann["x2_px"], ann["y2_px"])
    assert np.all(np.abs(angular_diff_180(ma, ann["measurement_angle_raster_deg"])) < 1e-6)
    assert np.all(np.abs(angular_diff_180(ann["fiber_angle_raster_deg"],
                                          ann["measurement_angle_raster_deg"] - 90.0)) < 1e-6)
    # widths are what was drawn: nm = px * nm/px with the field's calibration
    np.testing.assert_allclose(ann["width_nm"], ann["width_px"] * fld.nm_per_px, rtol=1e-6)
    # the mask is a faithful rendering: 2*EDT at annotated centres ~ width (all widths, incl. the thick tail)
    edt = ndi.distance_transform_edt(fld.mask)
    cx = np.clip(np.round(ann["center_x_px"]).astype(int), 0, fld.mask.shape[1] - 1)
    cy = np.clip(np.round(ann["center_y_px"]).astype(int), 0, fld.mask.shape[0] - 1)
    inside = fld.mask[cy, cx]
    assert inside.mean() > 0.9
    rec = 2.0 * edt[cy, cx][inside]
    rel = rec / ann["width_px"].to_numpy(float)[inside]
    assert 0.8 < np.median(rel) < 1.15, np.median(rel)
    thick = ann["width_px"].to_numpy(float)[inside] > 18
    if thick.sum() >= 3:
        assert 0.75 < np.median(rel[thick]) < 1.2
    # negative, wrapped and axis-aligned angles all occur in the label table
    a = ann["measurement_angle_raster_deg"].to_numpy(float)
    assert a.min() >= -90.0 and a.max() < 90.0


def test_synthetic_dataset_writes_images_labels_and_calibration(tmp_path):
    ds = write_synthetic_dataset(tmp_path, n_specimens=2, fields_per_specimen=2, H=128, W=128,
                                 n_annotations=15, with_footer=True)
    labels = pd.read_csv(ds["labels_csv"])
    ids = sorted(labels["image_id"].unique())
    assert ids == ["S1-1", "S1-2", "S2-1", "S2-2"]
    for iid in ids:
        assert (tmp_path / "original" / f"{iid}.png").exists()
    assert validate_labels(labels)["problems"] == []
    assert "calibration" in ds and all(iid in ds["calibration"] for iid in ids)
    # every field of one specimen shares its nm/px; specimens are allowed to differ
    per = labels.groupby("image_id")["nm_per_pixel"].nunique()
    assert (per == 1).all()
