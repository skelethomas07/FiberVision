"""Turn dense prediction maps into a measurement table.

Steps: peak detection on the centre heatmap -> NMS -> read width, orientation
and uncertainty at each peak -> build the chord perpendicular to the local fiber
axis -> reject implausible detections -> suppress duplicates that sit on the
same local stretch of the same fiber.

The last step is the subtle one.  Ordinary NMS by distance is not enough: two
peaks 20 px apart on the *same* fiber are duplicates, while two peaks 20 px
apart on two *crossing* fibers are both legitimate.  We therefore suppress only
when the predicted orientations also agree.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .utils import angular_diff_180, get_logger, line_endpoints, vec2_to_angle

LOG = get_logger(__name__)

PRED_COLUMNS = [
    "image_id", "prediction_id", "center_x_px", "center_y_px",
    "x1_px", "y1_px", "x2_px", "y2_px",
    "measurement_angle_deg", "local_fiber_angle_deg",
    "width_px", "width_nm", "nm_per_pixel",
    "confidence", "validity", "width_sigma_px", "rejected_reason",
]


@dataclass
class PostConfig:
    peak_threshold: float = 0.30
    nms_radius: int = 7
    duplicate_radius: float = 0.9      # in units of predicted width
    duplicate_angle_deg: float = 25.0
    min_width_px: float = 2.0
    max_width_px: float = 200.0
    min_validity: float = 0.3
    max_sigma_px: float | None = None
    top_k: int = 5000
    y_sign: float = 1.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def find_peaks(heat: np.ndarray, threshold: float, radius: int, top_k: int
               ) -> np.ndarray:
    """Local maxima of the heatmap, returned as (y, x) sorted by score."""
    from skimage.feature import peak_local_max

    pk = peak_local_max(heat, min_distance=max(1, radius),
                        threshold_abs=threshold, exclude_border=False)
    if pk.size == 0:
        return pk.reshape(0, 2)
    scores = heat[pk[:, 0], pk[:, 1]]
    order = np.argsort(-scores)[:top_k]
    return pk[order]


def decode_predictions(maps: dict[str, np.ndarray], *, image_id: str = "",
                       nm_per_pixel: float | None = None,
                       cfg: PostConfig | None = None,
                       log_width: bool = True,
                       keep_rejected: bool = False) -> "Any":
    """Convert model output maps into a prediction DataFrame.

    ``maps`` holds raw head outputs as 2-D arrays: ``center_logit``,
    ``segment_logit``, ``orient`` (2, H, W), ``width``, ``validity_logit``,
    ``logvar``.
    """
    import pandas as pd

    cfg = cfg or PostConfig()
    heat = _sigmoid(maps["center_logit"])
    validity = _sigmoid(maps["validity_logit"])
    orient = maps["orient"]
    width_map = np.exp(maps["width"]) if log_width else maps["width"]
    sigma_map = np.exp(0.5 * maps["logvar"])
    if log_width:
        # a variance learned in log space becomes a relative error in pixels
        sigma_map = sigma_map * width_map

    peaks = find_peaks(heat, cfg.peak_threshold, cfg.nms_radius, cfg.top_k)
    rows: list[dict[str, Any]] = []
    for (py, px) in peaks:
        conf = float(heat[py, px])
        val = float(validity[py, px])
        width = float(width_map[py, px])
        sigma = float(sigma_map[py, px])
        fiber_ang = float(vec2_to_angle(orient[0, py, px], orient[1, py, px]))
        meas_ang = fiber_ang + 90.0     # chord runs across the fiber

        reason = ""
        if not np.isfinite(width) or width < cfg.min_width_px:
            reason = "width_too_small"
        elif width > cfg.max_width_px:
            reason = "width_too_large"
        elif val < cfg.min_validity:
            reason = "low_validity"
        elif cfg.max_sigma_px is not None and sigma > cfg.max_sigma_px:
            reason = "uncertain"

        x1, y1, x2, y2 = line_endpoints(float(px), float(py), meas_ang, width,
                                        cfg.y_sign)
        rows.append({
            "image_id": image_id, "prediction_id": len(rows) + 1,
            "center_x_px": float(px), "center_y_px": float(py),
            "x1_px": x1, "y1_px": y1, "x2_px": x2, "y2_px": y2,
            "measurement_angle_deg": meas_ang,
            "local_fiber_angle_deg": fiber_ang,
            "width_px": width,
            "width_nm": width * nm_per_pixel if nm_per_pixel else np.nan,
            "nm_per_pixel": nm_per_pixel if nm_per_pixel else np.nan,
            "confidence": conf, "validity": val, "width_sigma_px": sigma,
            "rejected_reason": reason,
        })

    df = pd.DataFrame(rows, columns=PRED_COLUMNS)
    if len(df):
        kept = df[df["rejected_reason"] == ""].copy()
        kept = suppress_duplicates(kept, cfg)
        dropped = df[df["rejected_reason"] != ""].copy()
        LOG.info("%s: %d peaks -> %d predictions (%d rejected, %d duplicates)",
                 image_id or "image", len(df), len(kept), len(dropped),
                 len(df) - len(dropped) - len(kept))
        df = pd.concat([kept, dropped], ignore_index=True) if keep_rejected else kept
        df["prediction_id"] = np.arange(1, len(df) + 1)
    return df.reset_index(drop=True)


def suppress_duplicates(df: "Any", cfg: PostConfig) -> "Any":
    """Drop peaks that measure the same stretch of the same fiber twice."""
    if len(df) <= 1:
        return df
    order = df.sort_values("confidence", ascending=False).index.to_list()
    xs = df["center_x_px"].to_numpy()
    ys = df["center_y_px"].to_numpy()
    ang = df["local_fiber_angle_deg"].to_numpy()
    wid = df["width_px"].to_numpy()
    keep: list[int] = []
    taken = np.zeros(len(df), bool)
    pos = {idx: i for i, idx in enumerate(df.index)}
    for idx in order:
        i = pos[idx]
        if taken[i]:
            continue
        keep.append(idx)
        d = np.hypot(xs - xs[i], ys - ys[i])
        same_dir = angular_diff_180(ang, ang[i]) < cfg.duplicate_angle_deg
        radius = cfg.duplicate_radius * np.maximum(wid, wid[i])
        taken |= (d < radius) & same_dir
        taken[i] = True
    return df.loc[keep].sort_values("confidence", ascending=False)


def refine_width_from_image(df: "Any", gray: np.ndarray, *, half_span: float = 2.5,
                            n_samples: int = 101,
                            require_return: bool = True) -> "Any":
    """Measure width from the intensity profile, independently of the network.

    Useful as a cross-check rather than as the primary output: if the regressed
    width and this disagree systematically, the width head is the thing to look
    at.

    Two corrections over the first version, both of which mattered.  The scan
    used to span only +/-1.6 widths, which in a dense network does not reach
    background before it runs into the neighbouring fiber.  More seriously, when
    the profile never fell back to half maximum the walk simply hit the end of
    the array and the function returned ``(ts[-1] - ts[0]) * width``, i.e. a
    fixed multiple of the width it was supposed to be checking -- a number that
    looks like a measurement and is not one.  On a synthetic field with known
    widths that made a *correct* width head appear 89% wrong.  A cross-check
    that fails this way is worse than none, because it discredits a working
    model.  Those sites now return NaN and are excluded.
    """
    import cv2

    from .utils import angle_to_direction

    if not len(df):
        return df
    H, W = gray.shape
    img = gray.astype(np.float32)
    ts = np.linspace(-half_span, half_span, n_samples)
    ux, uy = angle_to_direction(df["measurement_angle_deg"].to_numpy(), 1.0)
    cx = df["center_x_px"].to_numpy()
    cy = df["center_y_px"].to_numpy()
    wid = df["width_px"].to_numpy()
    prof = np.full((len(df), ts.size), np.nan, np.float32)
    for i, t in enumerate(ts):
        x = (cx + ux * wid * t).astype(np.float32)
        y = (cy + uy * wid * t).astype(np.float32)
        m = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if m.any():
            prof[m, i] = cv2.remap(img, x[m], y[m], cv2.INTER_LINEAR).ravel()
    centre = int(np.argmin(np.abs(ts)))
    out = np.full(len(df), np.nan)
    for j, p in enumerate(prof):
        if not np.isfinite(p).all():
            continue
        peak, base = p[np.abs(ts) < 0.3].max(), np.percentile(p, 10)
        if peak <= base:
            continue
        half = 0.5 * (peak + base)
        if p[centre] < half:
            continue
        lo = centre
        while lo > 0 and p[lo] > half:
            lo -= 1
        hi = centre
        while hi < ts.size - 1 and p[hi] > half:
            hi += 1
        if require_return and (lo == 0 or hi == ts.size - 1):
            # the profile never came back down inside the scan: a neighbouring
            # fiber, a fold, or a site sitting on a crossing. There is no
            # half-maximum crossing to measure, so report nothing rather than
            # the span of the scan window.
            continue
        out[j] = (ts[hi] - ts[lo]) * wid[j]
    df = df.copy()
    df["width_px_profile"] = out
    return df
