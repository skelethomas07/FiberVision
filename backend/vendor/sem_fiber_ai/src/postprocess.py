"""Dense maps -> measurement-site table (v7).

``geometry`` mode (default)
    fibre mask = segmentation head; sites = medial-axis pixels of that mask at
    a CONTROLLED spacing along each branch; width = 2 x distance-to-boundary
    read from the ``dist`` head at the site (max over a 3x3 neighbourhood so a
    one-pixel skeleton offset does not bias it low); orientation = ``orient``
    head; the ``width`` head is a cross-check, and its disagreement with the
    distance width is reported per site (``boundary_disagreement``).

``baseline`` mode
    the v6 decoding: local maxima of the centre heatmap, width from the
    ``width`` head, duplicate suppression along the fibre.  Kept for
    comparison only.

Every site carries ``rejected_reason`` (machine-readable, '' = accepted) and
the separate quality quantities: ``confidence`` (segmentation probability at
the site), ``validity`` (validity head), ``width_sigma_px`` (aleatoric width
uncertainty), ``boundary_disagreement`` (|2 dist - width_head| / width),
``junction_distance_px``, ``coherence`` (structure-tensor coherency at the
site).  Confidence is NOT accuracy and is never used as one.

Nanometre policy: ``width_nm`` is filled only when ``calibration_valid`` is
True; otherwise NaN and the reason is ``calibration_invalid``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .coords import (angular_diff_180, chord_endpoints, measurement_angle_from_fiber,
                     vec2_to_angle)
from .utils import get_logger

LOG = get_logger(__name__)

SITE_COLUMNS = [
    "image_id", "site_id", "center_x_px", "center_y_px", "x1_px", "y1_px", "x2_px", "y2_px",
    "measurement_angle_raster_deg", "fiber_angle_raster_deg",
    "width_px", "width_px_dist", "width_px_head", "width_nm", "nm_per_pixel",
    "calibration_valid", "confidence", "validity", "width_sigma_px", "boundary_disagreement",
    "junction_distance_px", "coherence", "branch_id", "measurement_source", "rejected_reason",
]

REJECT_CODES = ("low_validity", "width_too_small", "width_too_large", "junction_zone",
                "boundary_disagreement", "near_image_border", "low_segmentation_confidence",
                "outside_segmentation", "uncertain_width", "calibration_invalid")


@dataclass
class PostConfig:
    mode: str = "geometry"
    seg_threshold: float = 0.5
    min_validity: float = 0.3
    spacing_px: float = 12.0
    min_width_px: float = 2.0
    max_width_px: float = 200.0
    boundary_tol: float = 0.35          # relative |2 dist - width_head| tolerance
    border_px: int = 6
    junction_clear_scale: float = 0.6
    min_seg_confidence: float = 0.5
    max_sigma_rel: float | None = None  # reject when width_sigma / width exceeds this
    # baseline only
    peak_threshold: float = 0.3
    nms_radius: int = 7
    duplicate_radius: float = 0.9
    duplicate_angle_deg: float = 25.0
    top_k: int = 8000


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def maps_from_torch(maps: dict[str, Any]) -> dict[str, np.ndarray]:
    """(1,C,H,W) tensors -> numpy 2-D / (2,H,W) arrays."""
    out = {}
    for k, v in maps.items():
        a = v.detach().float().cpu().numpy()
        out[k] = a[0] if a.shape[1] > 1 else a[0, 0]
    return out


def decode_predictions(maps: dict[str, np.ndarray], *, image_id: str = "",
                       nm_per_pixel: float | None = None, calibration_valid: bool = False,
                       cfg: PostConfig | None = None, coherency: np.ndarray | None = None,
                       keep_rejected: bool = True) -> "Any":
    cfg = cfg or PostConfig()
    if cfg.mode == "baseline":
        df = _decode_baseline(maps, image_id=image_id, cfg=cfg, coherency=coherency)
    else:
        df = _decode_geometry(maps, image_id=image_id, cfg=cfg, coherency=coherency)
    df = _apply_nm(df, nm_per_pixel, calibration_valid)
    if not keep_rejected and len(df):
        df = df[df["rejected_reason"] == ""].reset_index(drop=True)
    return df


def _apply_nm(df: "Any", nm_per_pixel: float | None, calibration_valid: bool) -> "Any":
    valid = bool(calibration_valid) and nm_per_pixel is not None and np.isfinite(nm_per_pixel) \
        and nm_per_pixel > 0
    df["calibration_valid"] = valid
    if valid:
        df["nm_per_pixel"] = float(nm_per_pixel)
        df["width_nm"] = df["width_px"].to_numpy(np.float64) * float(nm_per_pixel)
    else:
        df["nm_per_pixel"] = np.nan
        df["width_nm"] = np.nan
    return df


def _decode_geometry(maps, *, image_id: str, cfg: PostConfig, coherency) -> "Any":
    import pandas as pd
    from scipy import ndimage as ndi

    from .skeleton import branch_structure, spaced_sites

    seg = _sigmoid(maps["segment_logit"])
    validity = _sigmoid(maps["validity_logit"])
    dist = np.clip(maps["dist"], 0.0, None)
    width_head = np.exp(maps["width"])
    sigma_map = np.exp(0.5 * maps["logvar"]) * width_head
    orient = maps["orient"]
    h, w = seg.shape
    mask = seg >= cfg.seg_threshold
    if not mask.any():
        return pd.DataFrame(columns=SITE_COLUMNS)
    bs = branch_structure(mask)
    dist_max = ndi.maximum_filter(dist, size=3)
    sites = spaced_sites(bs, cfg.spacing_px, score=dist_max)
    rows = []
    for (y, x, lb) in sites:
        wd = float(2.0 * dist_max[y, x])
        wh = float(width_head[y, x])
        val = float(validity[y, x])
        conf = float(seg[y, x])
        fa = float(vec2_to_angle(orient[0, y, x], orient[1, y, x]))
        ma = float(measurement_angle_from_fiber(fa))
        jd = float(bs.junction_dist[y, x])
        coh = float(coherency[y, x]) if coherency is not None else np.nan
        sig = float(sigma_map[y, x])
        disagree = abs(wd - wh) / max(wd, 1e-6)
        reason = ""
        if wd < cfg.min_width_px:
            reason = "width_too_small"
        elif wd > cfg.max_width_px:
            reason = "width_too_large"
        elif min(x, y, w - 1 - x, h - 1 - y) < cfg.border_px:
            reason = "near_image_border"
        elif conf < cfg.min_seg_confidence:
            reason = "low_segmentation_confidence"
        elif val < cfg.min_validity:
            reason = "low_validity"
        elif jd < cfg.junction_clear_scale * wd:
            reason = "junction_zone"
        elif disagree > cfg.boundary_tol:
            reason = "boundary_disagreement"
        elif cfg.max_sigma_rel is not None and sig / max(wd, 1e-6) > cfg.max_sigma_rel:
            reason = "uncertain_width"
        x1, y1, x2, y2 = chord_endpoints(float(x), float(y), ma, wd)
        rows.append({
            "image_id": image_id, "site_id": len(rows) + 1,
            "center_x_px": float(x), "center_y_px": float(y),
            "x1_px": float(x1), "y1_px": float(y1), "x2_px": float(x2), "y2_px": float(y2),
            "measurement_angle_raster_deg": ma, "fiber_angle_raster_deg": fa,
            "width_px": wd, "width_px_dist": wd, "width_px_head": wh,
            "width_nm": np.nan, "nm_per_pixel": np.nan, "calibration_valid": False,
            "confidence": conf, "validity": val, "width_sigma_px": sig,
            "boundary_disagreement": float(disagree), "junction_distance_px": jd,
            "coherence": coh, "branch_id": int(lb), "measurement_source": "model_geometry",
            "rejected_reason": reason,
        })
    df = pd.DataFrame(rows, columns=SITE_COLUMNS)
    n_ok = int((df["rejected_reason"] == "").sum()) if len(df) else 0
    LOG.info("%s: %d medial-axis sites at spacing %.0f px -> %d accepted (%s)",
             image_id or "image", len(df), cfg.spacing_px, n_ok,
             dict(df["rejected_reason"].value_counts()) if len(df) else {})
    return df


def _decode_baseline(maps, *, image_id: str, cfg: PostConfig, coherency) -> "Any":
    import pandas as pd
    from skimage.feature import peak_local_max

    heat = _sigmoid(maps["center_logit"])
    validity = _sigmoid(maps["validity_logit"])
    seg = _sigmoid(maps["segment_logit"])
    width_map = np.exp(maps["width"])
    sigma_map = np.exp(0.5 * maps["logvar"]) * width_map
    orient = maps["orient"]
    h, w = heat.shape
    pk = peak_local_max(heat, min_distance=max(1, cfg.nms_radius),
                        threshold_abs=cfg.peak_threshold, exclude_border=False)
    if pk.size:
        scores = heat[pk[:, 0], pk[:, 1]]
        pk = pk[np.argsort(-scores)[:cfg.top_k]]
    rows = []
    for (y, x) in pk:
        wd = float(width_map[y, x])
        val = float(validity[y, x])
        fa = float(vec2_to_angle(orient[0, y, x], orient[1, y, x]))
        ma = float(measurement_angle_from_fiber(fa))
        reason = ""
        if not np.isfinite(wd) or wd < cfg.min_width_px:
            reason = "width_too_small"
        elif wd > cfg.max_width_px:
            reason = "width_too_large"
        elif val < cfg.min_validity:
            reason = "low_validity"
        elif min(x, y, w - 1 - x, h - 1 - y) < cfg.border_px:
            reason = "near_image_border"
        x1, y1, x2, y2 = chord_endpoints(float(x), float(y), ma, wd)
        rows.append({
            "image_id": image_id, "site_id": len(rows) + 1,
            "center_x_px": float(x), "center_y_px": float(y),
            "x1_px": float(x1), "y1_px": float(y1), "x2_px": float(x2), "y2_px": float(y2),
            "measurement_angle_raster_deg": ma, "fiber_angle_raster_deg": fa,
            "width_px": wd, "width_px_dist": float(2.0 * maps["dist"][y, x]) if "dist" in maps else np.nan,
            "width_px_head": wd, "width_nm": np.nan, "nm_per_pixel": np.nan,
            "calibration_valid": False, "confidence": float(heat[y, x]), "validity": val,
            "width_sigma_px": float(sigma_map[y, x]),
            "boundary_disagreement": np.nan, "junction_distance_px": np.nan,
            "coherence": float(coherency[y, x]) if coherency is not None else np.nan,
            "branch_id": 0, "measurement_source": "model_baseline", "rejected_reason": reason,
        })
    df = pd.DataFrame(rows, columns=SITE_COLUMNS)
    if len(df):
        kept = df[df["rejected_reason"] == ""].copy()
        kept = suppress_duplicates(kept, cfg)
        dup = df.index.difference(kept.index).difference(df.index[df["rejected_reason"] != ""])
        df.loc[dup, "rejected_reason"] = "duplicate_same_fiber"
    return df


def suppress_duplicates(df: "Any", cfg: PostConfig) -> "Any":
    if len(df) <= 1:
        return df
    order = df.sort_values("confidence", ascending=False).index.to_list()
    xs, ys = df["center_x_px"].to_numpy(), df["center_y_px"].to_numpy()
    ang, wid = df["fiber_angle_raster_deg"].to_numpy(), df["width_px"].to_numpy()
    keep, taken = [], np.zeros(len(df), bool)
    pos = {idx: i for i, idx in enumerate(df.index)}
    for idx in order:
        i = pos[idx]
        if taken[i]:
            continue
        keep.append(idx)
        d = np.hypot(xs - xs[i], ys - ys[i])
        same = angular_diff_180(ang, ang[i]) < cfg.duplicate_angle_deg
        taken |= (d < cfg.duplicate_radius * np.maximum(wid, wid[i])) & same
        taken[i] = True
    return df.loc[keep]


def rejection_summary(df: "Any") -> dict[str, int]:
    if not len(df):
        return {"accepted": 0}
    vc = df["rejected_reason"].fillna("").replace("", "accepted").value_counts()
    return {str(k): int(v) for k, v in vc.items()}
