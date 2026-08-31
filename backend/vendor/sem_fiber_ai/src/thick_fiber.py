"""Classical recovery of thick fibres that the learned centre heatmap misses.

The neural network remains the primary detector.  This module only supplements it
at wide, ridge-like structures where three independent image cues agree:

1. the Otsu fibre body is wide enough in its Euclidean distance transform;
2. a gamma-normalised Hessian bank says the local ridge scale is large;
3. the Hessian is anisotropic (a line/ridge, not an isotropic crossing/blob).

Widths are then read from the SEM intensity profile across the local fibre axis,
with the EDT width kept as an auditable cross-check.  The merger removes only
same-orientation, narrow AI detections sitting on the flanks of an accepted wide
measurement; crossing thin fibres are retained.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .utils import line_endpoints, wrap_deg_180


@dataclass
class ThickRecoveryConfig:
    enabled: bool = True
    sigmas: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0,
                                 12.0, 16.0, 24.0)
    gamma: float = 2.0
    min_sigma: float = 8.0
    min_width_px: float = 26.0
    max_width_px: float = 160.0
    response_percentile: float = 40.0
    min_ridge_coherence: float = 0.22
    segment_support: float = 0.15
    spacing_px: float = 28.0
    max_candidates: int = 600
    profile_step_px: float = 0.5
    profile_span_scale: float = 1.20
    profile_min_contrast: float = 0.06
    profile_edt_ratio_min: float = 0.45
    profile_edt_ratio_max: float = 2.20
    duplicate_angle_deg: float = 28.0
    replace_width_ratio: float = 0.72
    replace_normal_scale: float = 0.55
    replace_tangent_scale: float = 0.90


def _to_float(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray, np.float32)
    lo, hi = np.percentile(g[np.isfinite(g)], (1.0, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(g)), float(np.nanmax(g))
    return np.clip((g - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def fibre_body_mask(gray: np.ndarray) -> np.ndarray:
    """Bright SEM fibre body from a light Gaussian blur + Otsu threshold."""
    import cv2
    from skimage.filters import threshold_otsu

    g = _to_float(gray)
    sm = cv2.GaussianBlur(g, (0, 0), 1.0)
    try:
        thr = float(threshold_otsu(sm))
    except ValueError:  # flat image
        thr = float(np.percentile(sm, 50.0))
    body = sm >= thr
    # Fill tiny one-pixel holes but do not dilate: EDT width must stay physical.
    body = ndi.binary_closing(body, structure=np.ones((3, 3), bool), iterations=1)
    return body.astype(bool)


def ridge_scale_bank(gray: np.ndarray,
                     sigmas: tuple[float, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24),
                     gamma: float = 2.0) -> dict[str, np.ndarray]:
    """Multi-scale bright-ridge response, winning sigma, axis and anisotropy.

    A bright ridge is concave across its width, so the signal is the *minimum*
    algebraic Hessian eigenvalue, negated.  The eigenvector of the other
    eigenvalue is the fibre axis; ``0.5*atan2(2Hxy, Hxx-Hyy)`` gives that axis
    directly in image x/y coordinates for this polarity.
    """
    import cv2

    g = _to_float(gray)
    h, w = g.shape
    best = np.zeros((h, w), np.float32)
    best_sigma = np.full((h, w), float(sigmas[0]), np.float32)
    best_angle = np.zeros((h, w), np.float32)
    best_coh = np.zeros((h, w), np.float32)

    for s0 in sigmas:
        s = float(s0)
        k = max(3, int(6 * s) | 1)
        sm = cv2.GaussianBlur(g, (k, k), s, borderType=cv2.BORDER_REFLECT)
        gxx = cv2.Sobel(sm, cv2.CV_32F, 2, 0, ksize=3,
                        borderType=cv2.BORDER_REFLECT)
        gyy = cv2.Sobel(sm, cv2.CV_32F, 0, 2, ksize=3,
                        borderType=cv2.BORDER_REFLECT)
        gxy = cv2.Sobel(sm, cv2.CV_32F, 1, 1, ksize=3,
                        borderType=cv2.BORDER_REFLECT)

        disc = np.sqrt(np.maximum((gxx - gyy) ** 2 + 4.0 * gxy ** 2, 0.0))
        lam_min = 0.5 * (gxx + gyy - disc)
        lam_max = 0.5 * (gxx + gyy + disc)
        r = (s ** float(gamma)) * np.maximum(-lam_min, 0.0)

        # Line-likeness: 1 when one principal curvature dominates, 0 for a blob.
        a = np.abs(lam_min)
        b = np.abs(lam_max)
        coh = np.clip((a - b) / (a + b + 1e-8), 0.0, 1.0).astype(np.float32)

        # For H=[negative,0;0,0] (vertical fibre), this is +90 deg = fibre axis.
        ang = np.rad2deg(0.5 * np.arctan2(2.0 * gxy, gxx - gyy)).astype(np.float32)

        take = r > best
        best[take] = r[take]
        best_sigma[take] = s
        best_angle[take] = ang[take]
        best_coh[take] = coh[take]

    hi = float(np.percentile(best[best > 0], 99.5)) if np.any(best > 0) else 0.0
    norm = np.clip(best / hi, 0.0, 1.0).astype(np.float32) if hi > 0 else best
    return {"response": best, "response_norm": norm,
            "argmax_sigma": best_sigma,
            "orientation_deg": wrap_deg_180(best_angle).astype(np.float32),
            "coherence": best_coh}


def _angle_diff_180(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % 180.0
    return np.minimum(d, 180.0 - d)


def _profile_width(gray_f: np.ndarray, x: float, y: float, fiber_angle_deg: float,
                   edt_width: float, cfg: ThickRecoveryConfig
                   ) -> tuple[float, float]:
    """FWHM-like bright-ridge width across the fibre.

    Returns (width, contrast).  NaN width means the profile did not return to
    background cleanly inside the scan.
    """
    import cv2

    max_half = min(cfg.max_width_px * 0.75,
                   max(cfg.min_width_px, 0.5 * edt_width) * cfg.profile_span_scale)
    max_half = max(max_half, 0.75 * cfg.min_width_px)
    step = max(0.25, float(cfg.profile_step_px))
    t = np.arange(-max_half, max_half + 0.5 * step, step, dtype=np.float32)

    a = np.deg2rad(float(fiber_angle_deg) + 90.0)
    xs = (x + t * np.cos(a)).astype(np.float32)
    ys = (y + t * np.sin(a)).astype(np.float32)
    h, w = gray_f.shape
    valid = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
    if valid.mean() < 0.98:
        return np.nan, 0.0

    p = cv2.remap(gray_f, xs[None, :], ys[None, :], cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_REFLECT).reshape(-1)
    # Light smoothing removes SEM grain without changing a 26+ px width.
    p = ndi.gaussian_filter1d(p.astype(np.float32), sigma=max(0.5, 1.0 / step))

    centre0 = int(np.argmin(np.abs(t)))
    search = max(2, int(round(min(4.0, 0.15 * edt_width) / step)))
    lo0, hi0 = max(0, centre0 - search), min(len(p), centre0 + search + 1)
    centre = lo0 + int(np.argmax(p[lo0:hi0]))

    # Local background is the lower tail of this cross-section.  Using the ends
    # alone is brittle in a dense mat because another fibre can sit there.
    background = float(np.percentile(p, 18.0))
    peak = float(np.mean(p[max(0, centre - 1):min(len(p), centre + 2)]))
    contrast = peak - background
    if contrast < cfg.profile_min_contrast:
        return np.nan, contrast
    half = background + 0.5 * contrast
    if p[centre] <= half:
        return np.nan, contrast

    li = centre
    while li > 0 and p[li] > half:
        li -= 1
    ri = centre
    while ri < len(p) - 1 and p[ri] > half:
        ri += 1
    if li == 0 or ri == len(p) - 1:
        return np.nan, contrast

    def cross(i0: int, i1: int) -> float:
        y0, y1 = float(p[i0]), float(p[i1])
        x0, x1 = float(t[i0]), float(t[i1])
        if abs(y1 - y0) < 1e-8:
            return 0.5 * (x0 + x1)
        q = np.clip((half - y0) / (y1 - y0), 0.0, 1.0)
        return x0 + q * (x1 - x0)

    left = cross(li, li + 1)
    right = cross(ri - 1, ri)
    width = float(right - left)
    return width if width > 0 else np.nan, contrast


def _select_spaced(mask: np.ndarray, score: np.ndarray, spacing: float,
                   max_n: int) -> list[tuple[int, int]]:
    """Greedy high-score sampling with a circular exclusion radius."""
    import cv2

    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return []
    vals = score[ys, xs]
    order = np.argsort(-vals)
    blocked = np.zeros(mask.shape, np.uint8)
    picks: list[tuple[int, int]] = []
    rad = max(3, int(round(spacing)))
    for j in order:
        y, x = int(ys[j]), int(xs[j])
        if blocked[y, x]:
            continue
        picks.append((y, x))
        cv2.circle(blocked, (x, y), rad, 1, -1)
        if len(picks) >= int(max_n):
            break
    return picks


def recover_thick_measurements(gray: np.ndarray, *, image_id: str = "",
                               nm_per_pixel: float | None = None,
                               segment_prob: np.ndarray | None = None,
                               cfg: ThickRecoveryConfig | None = None
                               ) -> tuple["Any", dict[str, Any]]:
    """Return classical wide-fibre measurements and diagnostic maps."""
    import pandas as pd
    from skimage.morphology import skeletonize

    cfg = cfg or ThickRecoveryConfig()
    g = _to_float(gray)
    body = fibre_body_mask(gray)
    edt = ndi.distance_transform_edt(body).astype(np.float32)
    width_map = 2.0 * edt

    bank = ridge_scale_bank(gray, cfg.sigmas, cfg.gamma)
    resp_vals = bank["response"][body]
    floor = (float(np.percentile(resp_vals, cfg.response_percentile))
             if resp_vals.size else 0.0)

    support = np.ones_like(body, bool)
    if segment_prob is not None and np.shape(segment_prob) == body.shape:
        support = np.asarray(segment_prob, np.float32) >= float(cfg.segment_support)

    skel = skeletonize(body)
    candidate = (
        skel
        & (width_map >= float(cfg.min_width_px))
        & (width_map <= float(cfg.max_width_px))
        & (bank["argmax_sigma"] >= float(cfg.min_sigma))
        & (bank["response"] >= floor)
        & (bank["coherence"] >= float(cfg.min_ridge_coherence))
        & support
    )

    score = (width_map
             * (0.35 + 0.65 * bank["response_norm"])
             * (0.35 + 0.65 * bank["coherence"]))
    picks = _select_spaced(candidate, score, cfg.spacing_px, cfg.max_candidates)

    rows: list[dict[str, Any]] = []
    rejected_profile = rejected_ratio = 0
    for py, px in picks:
        edt_w = float(width_map[py, px])
        sigma = float(bank["argmax_sigma"][py, px])
        fiber_ang = float(bank["orientation_deg"][py, px])
        prof_w, contrast = _profile_width(g, float(px), float(py), fiber_ang,
                                          edt_w, cfg)

        method = "profile_fwhm"
        if np.isfinite(prof_w):
            ratio = prof_w / max(edt_w, 1e-6)
            if ratio < cfg.profile_edt_ratio_min or ratio > cfg.profile_edt_ratio_max:
                rejected_ratio += 1
                continue
            width = float(prof_w)
        else:
            # EDT is a conservative fallback only at an already scale/coherence
            # validated medial-axis site.
            rejected_profile += 1
            width = edt_w
            method = "edt_fallback"

        if not (cfg.min_width_px <= width <= cfg.max_width_px):
            continue

        meas_ang = float(wrap_deg_180(fiber_ang + 90.0))
        x1, y1, x2, y2 = line_endpoints(float(px), float(py), meas_ang, width, 1.0)
        conf = float(bank["response_norm"][py, px])
        coh = float(bank["coherence"][py, px])
        sigma_est = (0.5 * abs(float(prof_w) - edt_w)
                     if np.isfinite(prof_w) else 0.15 * edt_w)

        rows.append({
            "image_id": image_id,
            "prediction_id": len(rows) + 1,
            "center_x_px": float(px), "center_y_px": float(py),
            "x1_px": x1, "y1_px": y1, "x2_px": x2, "y2_px": y2,
            "measurement_angle_deg": meas_ang,
            "local_fiber_angle_deg": fiber_ang,
            "width_px": width,
            "width_nm": (width * float(nm_per_pixel)
                         if nm_per_pixel is not None and np.isfinite(nm_per_pixel)
                         else np.nan),
            "nm_per_pixel": (float(nm_per_pixel)
                             if nm_per_pixel is not None and np.isfinite(nm_per_pixel)
                             else np.nan),
            "confidence": conf,
            "validity": coh,
            "width_sigma_px": float(sigma_est),
            "rejected_reason": "",
            "measurement_source": "thick_recovery",
            "recovered_thick": True,
            "scale_sigma_px": sigma,
            "edt_width_px": edt_w,
            "profile_width_px": float(prof_w) if np.isfinite(prof_w) else np.nan,
            "profile_contrast": float(contrast),
            "measurement_method": method,
            "width_calibrated": False,
        })

    df = pd.DataFrame(rows)
    diag: dict[str, Any] = {
        "body_mask": body,
        "skeleton": skel,
        "candidate_mask": candidate,
        "width_map": width_map,
        "scale_sigma": bank["argmax_sigma"],
        "scale_response": bank["response_norm"],
        "scale_coherence": bank["coherence"],
        "n_candidate_pixels": int(candidate.sum()),
        "n_sampled_sites": int(len(picks)),
        "n_accepted": int(len(df)),
        "n_profile_fallback": int(rejected_profile),
        "n_profile_ratio_rejected": int(rejected_ratio),
        "response_floor": floor,
    }
    return df, diag


def merge_with_ai(ai_df: "Any", thick_df: "Any",
                  cfg: ThickRecoveryConfig | None = None
                  ) -> tuple["Any", dict[str, int]]:
    """Merge recovered wide measurements with learned detections.

    A recovered site is skipped if the AI already has a comparable-width,
    same-orientation measurement there.  Otherwise only narrow same-orientation
    AI detections lying on that wide fibre's flanks are removed.  Different-angle
    detections (e.g. a thin crossing fibre) survive.
    """
    import pandas as pd

    cfg = cfg or ThickRecoveryConfig()
    ai = ai_df.copy()
    if "measurement_source" not in ai.columns:
        ai["measurement_source"] = "ai"
    else:
        ai["measurement_source"] = ai["measurement_source"].fillna("ai")
    ai["recovered_thick"] = ai.get("recovered_thick", False)
    for c in ("scale_sigma_px", "edt_width_px", "profile_width_px",
              "profile_contrast", "measurement_method"):
        if c not in ai.columns:
            ai[c] = np.nan if c != "measurement_method" else "network"

    if thick_df is None or len(thick_df) == 0:
        ai["prediction_id"] = np.arange(1, len(ai) + 1)
        return ai.reset_index(drop=True), {
            "n_ai_input": int(len(ai)), "n_ai_replaced": 0,
            "n_thick_candidates": 0, "n_thick_added": 0,
            "n_combined": int(len(ai)),
        }

    drop = np.zeros(len(ai), bool)
    add_rows = []
    ax = ai["center_x_px"].to_numpy(float)
    ay = ai["center_y_px"].to_numpy(float)
    aw = ai["width_px"].to_numpy(float)
    aa = ai["local_fiber_angle_deg"].to_numpy(float)

    for _, r in thick_df.sort_values("width_px", ascending=False).iterrows():
        w = float(r["width_px"])
        theta = np.deg2rad(float(r["local_fiber_angle_deg"]))
        ux, uy = np.cos(theta), np.sin(theta)
        vx, vy = -uy, ux
        dx = ax - float(r["center_x_px"])
        dy = ay - float(r["center_y_px"])
        tangent = np.abs(dx * ux + dy * uy)
        normal = np.abs(dx * vx + dy * vy)
        ad = _angle_diff_180(aa, float(r["local_fiber_angle_deg"]))

        same = ((tangent <= cfg.replace_tangent_scale * w)
                & (normal <= cfg.replace_normal_scale * w)
                & (ad <= cfg.duplicate_angle_deg)
                & (~drop))
        # If a learned chord already spans most of this wide structure, retain
        # the learned measurement and do not double-count the classical one.
        if np.any(same & (aw >= cfg.replace_width_ratio * w)):
            continue

        narrow_flanks = same & (aw < cfg.replace_width_ratio * w)
        drop |= narrow_flanks
        add_rows.append(r.to_dict())

    kept = ai.loc[~drop].copy()
    added = pd.DataFrame(add_rows)
    combined = pd.concat([kept, added], ignore_index=True, sort=False)
    if len(combined):
        combined["prediction_id"] = np.arange(1, len(combined) + 1)
    stats = {
        "n_ai_input": int(len(ai)),
        "n_ai_replaced": int(drop.sum()),
        "n_thick_candidates": int(len(thick_df)),
        "n_thick_added": int(len(added)),
        "n_combined": int(len(combined)),
    }
    return combined.reset_index(drop=True), stats
