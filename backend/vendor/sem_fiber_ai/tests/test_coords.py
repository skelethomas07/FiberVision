"""Angle-convention tests: fixed transforms, round trips, wraparound, tensor."""
import numpy as np
import pandas as pd
import pytest

from sem_fiber_ai.src import coords as C

KNOWN = [-89.9, -75.0, -45.0, -30.0, -1.0, 0.0, 1.0, 30.0, 45.0, 60.0, 89.0]


def test_wrap180_range_and_periodicity():
    a = np.array([-270, -180, -90, -89.999, 0, 89.999, 90, 180, 270, 450])
    w = C.wrap180(a)
    assert np.all(w >= -90) and np.all(w < 90)
    assert np.allclose(C.wrap180(a + 180), w)
    assert C.wrap180(90.0) == -90.0 and C.wrap180(-90.0) == -90.0


@pytest.mark.parametrize("ang", KNOWN)
def test_endpoints_round_trip_raster(ang):
    x1, y1, x2, y2 = C.chord_endpoints(100.0, 50.0, ang, 20.0)
    back = float(C.measurement_angle_from_endpoints(x1, y1, x2, y2))
    assert C.angular_diff_180(back, ang) < 1e-9
    assert np.isclose(np.hypot(x2 - x1, y2 - y1), 20.0)


def test_horizontal_and_vertical_lines():
    assert float(C.measurement_angle_from_endpoints(0, 0, 10, 0)) == 0.0
    assert float(C.measurement_angle_from_endpoints(10, 0, 0, 0)) == 0.0     # undirected
    assert float(C.measurement_angle_from_endpoints(0, 0, 0, 10)) == -90.0   # wraps to -90
    assert float(C.measurement_angle_from_endpoints(0, 10, 0, 0)) == -90.0
    # a chord pointing down-right in raster has POSITIVE raster angle
    assert float(C.measurement_angle_from_endpoints(0, 0, 10, 10)) == 45.0
    # ...and ImageJ would call that same drawn line -45 (y up)
    assert float(C.imagej_angle_from_endpoints_yup(0, 0, 10, 10)) == -45.0


@pytest.mark.parametrize("ang", KNOWN)
def test_imagej_raster_round_trip(ang):
    ij = C.raster_to_imagej(ang)
    assert C.angular_diff_180(C.imagej_to_raster(ij), ang) < 1e-9
    r = C.imagej_to_raster(ang)
    assert C.angular_diff_180(C.raster_to_imagej(r), ang) < 1e-9


def test_imagej_conversion_matches_endpoint_geometry():
    """The finding behind v7: measurement_angle_deg agreed with endpoints under
    y-up.  Converting that column with the FIXED transform must reproduce the
    raster endpoint angle to numerical precision on every angle."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        ang = rng.uniform(-90, 90)
        x1, y1, x2, y2 = C.chord_endpoints(300, 200, ang, rng.uniform(4, 40))
        ij = C.imagej_angle_from_endpoints_yup(x1, y1, x2, y2)
        assert C.angular_diff_180(C.imagej_to_raster(ij), ang) < 1e-9


def test_wraparound_near_90():
    assert C.angular_diff_180(89.5, -89.5) == pytest.approx(1.0)
    assert C.angular_diff_180(-89.9, 89.9) == pytest.approx(0.2)
    assert C.angular_diff_180(0.0, 90.0) == pytest.approx(90.0)
    assert C.angular_diff_180(45.0, -45.0) == pytest.approx(90.0)


def test_fiber_perpendicular_to_measurement():
    for ang in KNOWN:
        f = C.fiber_angle_from_measurement(ang)
        assert C.angular_diff_180(f, ang) == pytest.approx(90.0)
        assert C.angular_diff_180(C.measurement_angle_from_fiber(f), ang) < 1e-9


def test_transform_angle_rotation_and_reflection():
    th = np.deg2rad(20.0)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    for ang in KNOWN:
        assert C.angular_diff_180(C.transform_angle(R, ang), ang + 20.0) < 1e-9
    # horizontal flip: x -> -x mirrors the direction, theta -> 180 - theta = -theta (mod 180)
    F = np.array([[-1, 0, 99], [0, 1, 0], [0, 0, 1]], float)
    for ang in KNOWN:
        assert C.angular_diff_180(C.transform_angle(F, ang), -ang) < 1e-9
    # vertical flip also negates the raster angle
    Fv = np.array([[1, 0, 0], [0, -1, 99], [0, 0, 1]], float)
    for ang in KNOWN:
        assert C.angular_diff_180(C.transform_angle(Fv, ang), -ang) < 1e-9
    # a 90-degree rotation (k90 in the augmenter): [[0,-1],[1,0]]
    R90 = np.array([[0, -1, 99], [1, 0, 0], [0, 0, 1]], float)
    assert C.angular_diff_180(C.transform_angle(R90, 0.0), 90.0) < 1e-9
    assert C.transform_scale(2.5 * np.eye(3)) == pytest.approx(2.5)


def test_doubled_angle_and_order_parameter():
    for ang in KNOWN:
        c, s = C.angle_to_vec2(ang)
        assert C.angular_diff_180(C.vec2_to_angle(c, s), ang) < 1e-9
    assert C.order_parameter_2d([30.0] * 10) == pytest.approx(1.0)
    iso = np.linspace(-90, 90, 3600, endpoint=False)
    assert C.order_parameter_2d(iso) < 1e-6
    assert C.angular_diff_180(C.circular_mean_180([80.0, -80.0]), -90.0) < 1e-9


@pytest.mark.parametrize("ang", [-80, -45, -20, 0, 15, 40, 60, 85])
def test_structure_tensor_orientation_is_raster(ang):
    """A bright line drawn at raster angle ``ang`` must be reported at ``ang``."""
    import cv2

    img = np.zeros((160, 160), np.float32)
    x1, y1, x2, y2 = C.chord_endpoints(80, 80, ang, 120)
    cv2.line(img, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))),
             255.0, 5, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    fib, coh = C.structure_tensor_orientation(img, sigma=3.0)
    sel = coh > 0.5
    sel &= img > 40
    got = C.circular_mean_180(fib[sel])
    assert C.angular_diff_180(got, ang) < 3.0


def test_standardize_label_table_from_imagej_export():
    rng = np.random.default_rng(1)
    rows = []
    for ang in KNOWN:
        w = float(rng.uniform(5, 30))
        x1, y1, x2, y2 = C.chord_endpoints(200.0, 150.0, ang, w)
        rows.append({"x1_px": x1, "y1_px": y1, "x2_px": x2, "y2_px": y2,
                     "measurement_angle_deg": float(C.imagej_angle_from_endpoints_yup(
                         x1, y1, x2, y2)), "width_px": w})
    df = pd.DataFrame(rows)
    out = C.standardize_label_table(df, angle_source_convention=C.IMAGEJ)
    for c in C.LABEL_ANGLE_COLUMNS:
        assert c in out.columns
    assert np.allclose(C.angular_diff_180(out["measurement_angle_raster_deg"], KNOWN), 0, atol=1e-9)
    assert np.allclose(out["imagej_angle_deg"], df["measurement_angle_deg"])
    assert (out["angle_convention_residual_deg"] < 1e-6).all()
    assert (out["angle_source_convention"] == C.IMAGEJ).all()
    # a table whose angle column is NOT ImageJ but raster must be declared as such
    raw = C.standardize_label_table(df.assign(measurement_angle_deg=KNOWN),
                                    angle_source_convention=C.RASTER)
    assert raw["imagej_angle_deg"].isna().all()
    assert (raw["angle_convention_residual_deg"] < 1e-6).all()
    with pytest.raises(ValueError):
        C.standardize_label_table(df, angle_source_convention="whatever")
