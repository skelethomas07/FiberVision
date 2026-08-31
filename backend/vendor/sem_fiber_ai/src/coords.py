"""ONE coordinate convention for every angle in the project (v7).

Internal convention: **raster** coordinates.  +x points right, +y points DOWN
(row index increases downwards).  A direction angle is measured from +x towards
+y, i.e. clockwise on screen, and fibre/measurement orientations are
pi-periodic and wrapped to ``[-90, 90)``.

    measurement_angle_raster_deg = wrap180(degrees(atan2(y2 - y1, x2 - x1)))
    fiber_angle_raster_deg       = wrap180(measurement_angle_raster_deg - 90)

ImageJ's *Angle* column for a line ROI uses the mathematical convention with
+y UP (counter-clockwise from +x, in (-180, 180]).  For the same drawn line the
two conventions differ only by the sign of the sine term, so the conversion is
a fixed transformation, not something to be fitted per field:

    raster = wrap180(-imagej)          imagej = wrap180(-raster)

Every function below is covered by ``tests/test_coords.py`` at known angles,
including negative angles, vertical and horizontal lines and wraparound at
+-90 deg.  Nothing in this module reads image data or annotation tables, so it
cannot be biased by them.
"""
from __future__ import annotations

from typing import Any

import numpy as np

RASTER = "raster_y_down"
IMAGEJ = "imagej_y_up"
CONVENTIONS = (RASTER, IMAGEJ)


def wrap180(angle):
    """Wrap pi-periodic angle(s) in degrees to ``[-90, 90)``."""
    return (np.asarray(angle, dtype=np.float64) + 90.0) % 180.0 - 90.0


def wrap360(angle):
    """Wrap 2pi-periodic angle(s) in degrees to ``(-180, 180]``."""
    a = np.asarray(angle, dtype=np.float64)
    out = (a + 180.0) % 360.0 - 180.0
    return np.where(out == -180.0, 180.0, out)


def angular_diff_180(a, b):
    """Smallest absolute difference between two pi-periodic angles (deg)."""
    d = np.abs(wrap180(np.asarray(a, np.float64) - np.asarray(b, np.float64)))
    return np.minimum(d, 180.0 - d)


# --------------------------------------------------------------------------- #
# endpoints <-> angles, raster convention only
# --------------------------------------------------------------------------- #
def measurement_angle_from_endpoints(x1, y1, x2, y2):
    """Raster measurement angle of the chord (x1,y1)->(x2,y2), in [-90, 90)."""
    dx = np.asarray(x2, np.float64) - np.asarray(x1, np.float64)
    dy = np.asarray(y2, np.float64) - np.asarray(y1, np.float64)
    return wrap180(np.degrees(np.arctan2(dy, dx)))


def fiber_angle_from_measurement(measurement_angle_raster_deg):
    """The fibre runs perpendicular to the chord drawn across it."""
    return wrap180(np.asarray(measurement_angle_raster_deg, np.float64) - 90.0)


def measurement_angle_from_fiber(fiber_angle_raster_deg):
    return wrap180(np.asarray(fiber_angle_raster_deg, np.float64) + 90.0)


def direction_vector(angle_raster_deg):
    """Unit vector (ux, uy) in raster coordinates for a raster angle."""
    t = np.deg2rad(np.asarray(angle_raster_deg, np.float64))
    return np.cos(t), np.sin(t)


def chord_endpoints(cx, cy, measurement_angle_raster_deg, length_px):
    """Endpoints of a chord of ``length_px`` centred at (cx, cy), raster frame."""
    ux, uy = direction_vector(measurement_angle_raster_deg)
    hx = ux * np.asarray(length_px, np.float64) / 2.0
    hy = uy * np.asarray(length_px, np.float64) / 2.0
    cx = np.asarray(cx, np.float64)
    cy = np.asarray(cy, np.float64)
    return cx - hx, cy - hy, cx + hx, cy + hy


# --------------------------------------------------------------------------- #
# ImageJ (y-up) <-> raster (y-down)
# --------------------------------------------------------------------------- #
def imagej_to_raster(angle_imagej_deg):
    """ImageJ y-up line angle -> raster y-down angle (pi-periodic)."""
    return wrap180(-np.asarray(angle_imagej_deg, np.float64))


def raster_to_imagej(angle_raster_deg):
    """Raster y-down angle -> ImageJ y-up angle (pi-periodic, in [-90, 90))."""
    return wrap180(-np.asarray(angle_raster_deg, np.float64))


def to_raster(angle_deg, source_convention: str):
    """Convert an angle from ``source_convention`` to the internal raster one."""
    if source_convention == RASTER:
        return wrap180(angle_deg)
    if source_convention == IMAGEJ:
        return imagej_to_raster(angle_deg)
    raise ValueError(f"unknown angle convention {source_convention!r}; "
                     f"expected one of {CONVENTIONS}")


def imagej_angle_from_endpoints_yup(x1, y1, x2, y2):
    """What ImageJ would report for a line drawn between two raster points.

    ImageJ measures the angle with +y up, so the raster dy is negated.  The
    result is wrapped to the pi-periodic range because a measurement line has
    no direction.
    """
    dx = np.asarray(x2, np.float64) - np.asarray(x1, np.float64)
    dy = np.asarray(y2, np.float64) - np.asarray(y1, np.float64)
    return wrap180(np.degrees(np.arctan2(-dy, dx)))


# --------------------------------------------------------------------------- #
# doubled-angle encoding (pi-periodic vectors)
# --------------------------------------------------------------------------- #
def angle_to_vec2(angle_raster_deg):
    t = np.deg2rad(np.asarray(angle_raster_deg, np.float64))
    return np.cos(2.0 * t), np.sin(2.0 * t)


def vec2_to_angle(cos2t, sin2t):
    return wrap180(np.degrees(0.5 * np.arctan2(sin2t, cos2t)))


def circular_mean_180(angles_deg, weights=None):
    """Mean of pi-periodic angles via the doubled-angle vector."""
    a = np.asarray(angles_deg, np.float64)
    if a.size == 0:
        return float("nan")
    w = np.ones_like(a) if weights is None else np.asarray(weights, np.float64)
    c, s = angle_to_vec2(a)
    return float(vec2_to_angle(np.sum(w * c), np.sum(w * s)))


def order_parameter_2d(angles_deg, weights=None):
    """Nematic order parameter S2 = |<exp(2 i theta)>| in [0, 1]."""
    a = np.asarray(angles_deg, np.float64)
    if a.size == 0:
        return float("nan")
    w = np.ones_like(a) if weights is None else np.asarray(weights, np.float64)
    c, s = angle_to_vec2(a)
    return float(np.hypot(np.sum(w * c), np.sum(w * s)) / max(np.sum(w), 1e-12))


# --------------------------------------------------------------------------- #
# linear transforms of directions (augmentation, resampling)
# --------------------------------------------------------------------------- #
def transform_points(M, x, y):
    """Apply a 2x3 or 3x3 affine matrix to raster points."""
    M = np.asarray(M, np.float64)
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    nx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
    ny = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    return nx, ny


def transform_angle(M, angle_raster_deg):
    """Transform a pi-periodic raster direction by the linear part of ``M``.

    Correct for rotations AND reflections (a flip mirrors the direction; adding
    the matrix's rotation angle would be wrong under a reflection).
    """
    A = np.asarray(M, np.float64)[:2, :2]
    ux, uy = direction_vector(angle_raster_deg)
    nx = A[0, 0] * ux + A[0, 1] * uy
    ny = A[1, 0] * ux + A[1, 1] * uy
    return wrap180(np.degrees(np.arctan2(ny, nx)))


def transform_scale(M) -> float:
    """Isotropic scale factor of a similarity matrix (sqrt|det| of linear part)."""
    A = np.asarray(M, np.float64)[:2, :2]
    return float(np.sqrt(abs(np.linalg.det(A))))


# --------------------------------------------------------------------------- #
# structure-tensor fibre orientation in the raster convention (fixed mapping)
# --------------------------------------------------------------------------- #
def structure_tensor_orientation(gray: np.ndarray, sigma: float = 2.0,
                                 grad_sigma: float = 1.0
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """Dense fibre orientation (raster deg, [-90, 90)) and coherency.

    The dominant gradient direction is ``0.5 * atan2(2 Jxy, Jxx - Jyy)`` with
    ``Jxy = <Ix Iy>`` computed on raster axes (x = column, y = row); the fibre
    runs perpendicular to it.  This is the ONE mapping used everywhere (targets,
    audits, orientation reports).  It is verified on synthetic lines at known
    raster angles in ``tests/test_coords.py`` and never fitted to annotations.
    """
    from scipy import ndimage as ndi

    g = np.asarray(gray, np.float64)
    ix = ndi.gaussian_filter(g, grad_sigma, order=(0, 1), mode="nearest")
    iy = ndi.gaussian_filter(g, grad_sigma, order=(1, 0), mode="nearest")
    jxx = ndi.gaussian_filter(ix * ix, sigma, mode="nearest")
    jyy = ndi.gaussian_filter(iy * iy, sigma, mode="nearest")
    jxy = ndi.gaussian_filter(ix * iy, sigma, mode="nearest")
    grad_ang = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    fiber = wrap180(np.degrees(grad_ang) + 90.0)
    tr = jxx + jyy
    diff = np.sqrt(np.clip((jxx - jyy) ** 2 + 4.0 * jxy ** 2, 0.0, None))
    coh = np.where(tr > 1e-12, diff / np.clip(tr, 1e-12, None), 0.0)
    return fiber.astype(np.float32), coh.astype(np.float32)


def orientationpy_theta_to_raster(theta_deg):
    """orientationpy reports theta with +y up (mathematical); negate for raster.

    Verified numerically over -75..75 deg in the v6.10 work; kept as a fixed
    conversion here so no field ever decides its own sign.
    """
    return wrap180(-np.asarray(theta_deg, np.float64))


# --------------------------------------------------------------------------- #
# annotation-table standardisation
# --------------------------------------------------------------------------- #
LABEL_ANGLE_COLUMNS = ("measurement_angle_raster_deg", "fiber_angle_raster_deg",
                       "angle_source_convention", "imagej_angle_deg")


def standardize_label_table(df: "Any", *, angle_source_convention: str,
                            raw_angle_column: str = "measurement_angle_deg") -> "Any":
    """Attach the explicit raster angle columns to an extracted label table.

    The recovered endpoints ``x1_px..y2_px`` are always raster coordinates (they
    are pixel positions on the original image), so the raster measurement angle
    is derived from them.  The raw angle column, whatever convention the export
    used, is preserved verbatim in ``imagej_angle_deg`` when the export was an
    ImageJ (y-up) table, and ``angle_source_convention`` records what it was.
    Where endpoints are missing the raw angle is converted with the fixed
    transformation instead.
    """
    import pandas as pd

    if angle_source_convention not in CONVENTIONS:
        raise ValueError(f"angle_source_convention must be one of {CONVENTIONS}")
    out = df.copy()
    n = len(out)
    have_ep = {"x1_px", "y1_px", "x2_px", "y2_px"} <= set(out.columns)
    raw = (pd.to_numeric(out[raw_angle_column], errors="coerce").to_numpy(np.float64)
           if raw_angle_column in out.columns else np.full(n, np.nan))
    if have_ep:
        meas = measurement_angle_from_endpoints(
            out["x1_px"].to_numpy(np.float64), out["y1_px"].to_numpy(np.float64),
            out["x2_px"].to_numpy(np.float64), out["y2_px"].to_numpy(np.float64))
        degenerate = ~np.isfinite(meas)
        if degenerate.any():
            meas = np.where(degenerate, to_raster(raw, angle_source_convention), meas)
    else:
        meas = to_raster(raw, angle_source_convention)
    out["measurement_angle_raster_deg"] = meas
    out["fiber_angle_raster_deg"] = fiber_angle_from_measurement(meas)
    out["angle_source_convention"] = angle_source_convention
    out["imagej_angle_deg"] = raw if angle_source_convention == IMAGEJ else np.nan
    # consistency of the raw column with the endpoints under the fixed transform
    conv = to_raster(raw, angle_source_convention)
    dev = angular_diff_180(conv, meas)
    out["angle_convention_residual_deg"] = np.where(np.isfinite(dev), dev, np.nan)
    return out
