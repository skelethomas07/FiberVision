"""Physical-resolution resampling (v7).

Fields are acquired at 1-5 nm/px.  For nm-based training every image and its
annotations are resampled to ONE reference resolution so that a given physical
width is the same number of pixels in every field and the network's receptive
field covers the same physical extent everywhere.

* The reference nm/px is the median physical nm/px of the TRAINING fields only
  (validation and test never influence it) and is stored in the split manifest.
* Coordinates, widths and nm/px are transformed exactly with the factor that
  was actually applied (after integer rounding of the image size), and both the
  original and the transformed calibration are stored on every row.
* A field whose calibration is invalid, or whose factor falls outside the
  allowed range, is excluded from nm-based training with a machine-readable
  reason instead of being silently used at the wrong scale.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .utils import get_logger

LOG = get_logger(__name__)

COORD_X = ("center_x_px", "x1_px", "x2_px")
COORD_Y = ("center_y_px", "y1_px", "y2_px")
LENGTH_COLS = ("width_px",)


@dataclass
class ResampleDecision:
    image_id: str
    included: bool
    reason: str
    nm_per_px_original: float | None
    nm_per_px_resampled: float | None
    factor_requested: float | None
    factor_applied: float | None
    shape_original: tuple[int, int] | None
    shape_resampled: tuple[int, int] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reference_nm_per_px(train_nm_per_px: dict[str, float | None]) -> float | None:
    vals = np.array([v for v in train_nm_per_px.values()
                     if v is not None and np.isfinite(v) and v > 0], np.float64)
    if vals.size == 0:
        return None
    return float(np.median(vals))


def plan_resample(image_id: str, shape_hw: tuple[int, int], nm_per_px: float | None,
                  ref_nm_per_px: float | None, *, calibration_valid: bool,
                  factor_range: tuple[float, float] = (0.33, 3.0)) -> ResampleDecision:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if not calibration_valid or nm_per_px is None or not np.isfinite(nm_per_px):
        return ResampleDecision(image_id, False, "calibration_invalid", nm_per_px, None,
                                None, None, (h, w), None)
    if ref_nm_per_px is None or not np.isfinite(ref_nm_per_px) or ref_nm_per_px <= 0:
        return ResampleDecision(image_id, False, "no_reference_resolution", nm_per_px,
                                None, None, None, (h, w), None)
    f = float(nm_per_px) / float(ref_nm_per_px)
    if not (factor_range[0] <= f <= factor_range[1]):
        return ResampleDecision(image_id, False, f"resample_factor_out_of_range_{f:.3f}",
                                nm_per_px, None, f, None, (h, w), None)
    nh, nw = max(8, int(round(h * f))), max(8, int(round(w * f)))
    # the factor that is ACTUALLY applied after rounding, so nm/px stays exact
    fa = float(nw) / float(w)
    return ResampleDecision(image_id, True, "ok", float(nm_per_px),
                            float(nm_per_px) / fa, f, fa, (h, w), (nh, nw))


def resample_image(gray: np.ndarray, factor: float) -> np.ndarray:
    import cv2
    h, w = gray.shape[:2]
    nh, nw = max(8, int(round(h * factor))), max(8, int(round(w * factor)))
    interp = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(np.asarray(gray, np.float32), (nw, nh), interpolation=interp)


def resample_labels(df: "Any", factor: float, nm_per_px_resampled: float | None) -> "Any":
    """Scale coordinates and pixel widths by ``factor``; angles are unchanged
    under isotropic scaling; nm widths are unchanged by construction."""
    out = df.copy()
    for c in COORD_X + COORD_Y:
        if c in out.columns:
            out[c] = out[c].to_numpy(np.float64) * factor
    for c in LENGTH_COLS:
        if c in out.columns:
            out[c] = out[c].to_numpy(np.float64) * factor
    out["resample_factor"] = factor
    if "nm_per_pixel" in out.columns:
        out["nm_per_pixel_original"] = out["nm_per_pixel"]
    out["nm_per_pixel"] = (nm_per_px_resampled if nm_per_px_resampled is not None
                           else np.nan)
    if "width_nm" in out.columns and nm_per_px_resampled is not None:
        out["width_nm"] = out["width_px"].to_numpy(np.float64) * nm_per_px_resampled
    return out


def unresample_predictions(df: "Any", factor: float) -> "Any":
    """Map predictions made on a resampled image back to original pixels."""
    out = df.copy()
    inv = 1.0 / float(factor)
    for c in COORD_X + COORD_Y:
        if c in out.columns:
            out[c] = out[c].to_numpy(np.float64) * inv
    for c in ("width_px", "width_px_dist", "width_px_head", "width_px_edt",
              "width_sigma_px", "length_px"):
        if c in out.columns:
            out[c] = out[c].to_numpy(np.float64) * inv
    out["resample_factor"] = factor
    return out
