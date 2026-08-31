"""Orientation analysis (v7): label-free field statistics + head diagnostics.

* :func:`orientation_field` -- structure-tensor fibre direction and coherency in
  the raster convention (``coords.structure_tensor_orientation``); this is the
  classical OrientationJ-style deliverable and needs no model.
* :func:`orientation_summary` -- coherency-weighted histogram, nematic order
  parameter S2, mean direction and dispersion, inside a mask.
* :func:`fibre_orientation_summary` -- the same from a fibre table (one vote per
  fibre, NUMBER-weighted).
* :func:`head_vs_tensor` -- the model's orientation head against the tensor it
  was trained on, stratified by coherency.
* :func:`orientationpy_orientation` / :func:`orientationj_compare` -- parity
  checks against the published implementations with a FIXED sign conversion
  (``orientationpy`` and OrientationJ report angles with +y up; raster is +y
  down, so ``raster = -theta``).  A synthetic parity test confirms the sign for
  the installed version; nothing is fitted per field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .coords import (angular_diff_180, circular_mean_180, order_parameter_2d,
                     orientationpy_theta_to_raster, structure_tensor_orientation, vec2_to_angle,
                     wrap180)


@dataclass
class TensorConfig:
    sigma_grad: float = 1.5
    sigma_tensor: float = 4.0
    coherency_min: float = 0.25


def orientation_field(gray: np.ndarray, cfg: TensorConfig | None = None):
    cfg = cfg or TensorConfig()
    g = np.asarray(gray, np.float64)
    if g.max() > 1.5:
        g = g / 255.0
    return structure_tensor_orientation(g, sigma=cfg.sigma_tensor, grad_sigma=cfg.sigma_grad)


def orientation_summary(gray: np.ndarray, mask: np.ndarray | None = None,
                        cfg: TensorConfig | None = None, n_bins: int = 36) -> dict[str, Any]:
    cfg = cfg or TensorConfig()
    ang, coh = orientation_field(gray, cfg)
    sel = np.ones(ang.shape, bool) if mask is None else np.asarray(mask, bool)
    sel &= coh >= cfg.coherency_min
    a, w = ang[sel].astype(np.float64), coh[sel].astype(np.float64)
    edges = np.linspace(-90, 90, n_bins + 1)
    hist = np.histogram(a, bins=edges, weights=w)[0] if a.size else np.zeros(n_bins)
    hist = hist / hist.sum() if hist.sum() > 0 else hist
    S = order_parameter_2d(a, w) if a.size else float("nan")
    return {"n_pixels": int(a.size), "coherent_fraction": float(sel.mean()),
            "mean_angle_deg": circular_mean_180(a, w) if a.size else float("nan"),
            "order_parameter_S": S, "S": S,
            "dispersion_deg": float(np.degrees(0.5 * np.sqrt(-2.0 * np.log(max(S, 1e-9)))))
            if np.isfinite(S) and S > 0 else float("nan"),
            "hist_edges_deg": edges.tolist(), "hist_weight": hist.tolist(),
            "convention": "raster_y_down, fibre direction, [-90, 90)"}


def fibre_orientation_summary(fibres: "Any", n_bins: int = 36) -> dict[str, Any]:
    a = np.asarray(fibres["fiber_angle_raster_deg"], float)
    a = a[np.isfinite(a)]
    edges = np.linspace(-90, 90, n_bins + 1)
    hist = np.histogram(a, bins=edges)[0].astype(float)
    hist = hist / hist.sum() if hist.sum() > 0 else hist
    S = order_parameter_2d(a) if a.size else float("nan")
    return {"n_fibres": int(a.size), "mean_angle_deg": circular_mean_180(a) if a.size else float("nan"),
            "order_parameter_S": S, "S": S, "hist_edges_deg": edges.tolist(), "hist_count": hist.tolist(),
            "weighting": "number (one vote per fibre)"}


def histogram_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Circular 1-Wasserstein between two normalised angle histograms (deg)."""
    h1, h2 = np.asarray(h1, float), np.asarray(h2, float)
    if h1.sum() <= 0 or h2.sum() <= 0:
        return float("nan")
    p, q = h1 / h1.sum(), h2 / h2.sum()
    n = p.size
    width = 180.0 / n
    cd = np.cumsum(p - q)
    # circular EMD: subtract the best constant shift
    best = np.min([np.abs(cd - c).sum() for c in np.linspace(cd.min(), cd.max(), 64)])
    return float(best * width)


def head_vs_tensor(sites: "Any", gray: np.ndarray, mask: np.ndarray | None = None,
                   cfg: TensorConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TensorConfig()
    ang, coh = orientation_field(gray, cfg)
    H, W = ang.shape
    cx = np.clip(np.rint(sites["center_x_px"].to_numpy(float)), 0, W - 1).astype(int)
    cy = np.clip(np.rint(sites["center_y_px"].to_numpy(float)), 0, H - 1).astype(int)
    head = sites["fiber_angle_raster_deg"].to_numpy(float)
    d = angular_diff_180(head, ang[cy, cx])
    c = coh[cy, cx]
    out: dict[str, Any] = {"n": int(d.size)}
    if d.size:
        out["median_deg_all"] = float(np.median(d))
        for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
            s = (c >= lo) & (c < hi)
            if s.sum() >= 5:
                out[f"median_deg_coh_{lo:.2f}-{hi:.2f}"] = float(np.median(d[s]))
        s = c >= cfg.coherency_min
        out["median_deg_coherent"] = float(np.median(d[s])) if s.any() else float("nan")
    return out


# --------------------------------------------------------------------------- #
def orientationpy_orientation(gray: np.ndarray, sigma_tensor: float = 4.0,
                              mode: str = "gaussian"):
    """(angle_raster_deg, coherency) from orientationpy with the FIXED conversion."""
    import orientationpy

    img = np.asarray(gray, np.float64)
    if img.max() > 1.5:
        img = img / 255.0
    g = orientationpy.computeGradient(img, mode=mode)
    S = orientationpy.computeStructureTensor(g, sigma=sigma_tensor)
    theta = np.asarray(orientationpy.computeOrientation(S)["theta"], float)
    ang = orientationpy_theta_to_raster(theta)
    Syy, Syx, Sxx = (np.asarray(S[0], float), np.asarray(S[1], float), np.asarray(S[2], float))
    tr = Sxx + Syy
    det = np.sqrt(np.maximum((Sxx - Syy) ** 2 + 4 * Syx ** 2, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.where(tr > 1e-12, det / tr, 0.0)
    return ang.astype(np.float32), coh.astype(np.float32)


def synthetic_parity_check(backend, angles=(-75, -45, -15, 0, 15, 45, 75), H: int = 256,
                           W: int = 256, width: float = 9.0) -> dict[str, Any]:
    """Run ``backend(gray) -> (angle_raster_deg, coherency)`` on bars drawn at
    known RASTER angles and report the median error.  A backend whose sign
    conversion is wrong shows ~2x|angle| errors at oblique angles."""
    from scipy import ndimage as ndi

    yy, xx = np.mgrid[0:H, 0:W]
    errs, errs_flipped = [], []
    for a in angles:
        img = np.zeros((H, W))
        th = np.deg2rad(a)
        d = np.abs(-(xx - W / 2) * np.sin(th) + (yy - H / 2) * np.cos(th))
        img[d < width / 2] = 1.0
        img = ndi.gaussian_filter(img, 1.0) * 255.0
        ang, coh = backend(img)
        sel = (d < width / 4) & (coh > 0.5)
        if sel.sum() < 10:
            continue
        errs.append(float(np.median(angular_diff_180(ang[sel], a))))
        errs_flipped.append(float(np.median(angular_diff_180(-ang[sel], a))))
    ok = bool(errs) and float(np.median(errs)) < 3.0
    return {"median_error_deg": float(np.median(errs)) if errs else float("nan"),
            "median_error_if_sign_flipped_deg": float(np.median(errs_flipped)) if errs_flipped else float("nan"),
            "passed": ok, "angles": list(angles)}


def load_orientationj_tiff(orientation_path, coherency_path=None) -> tuple[np.ndarray, np.ndarray | None]:
    """OrientationJ 32-bit exports.  Orientation is converted with the fixed
    y-up -> raster rule; the caller must verify against ``synthetic_parity_check``
    on a synthetic image processed by the SAME OrientationJ settings."""
    from skimage import io

    theta = np.asarray(io.imread(str(orientation_path)), np.float64)
    if np.nanmax(np.abs(theta)) <= np.pi + 1e-3:
        theta = np.degrees(theta)
    ang = wrap180(-theta).astype(np.float32)
    coh = np.asarray(io.imread(str(coherency_path)), np.float32) if coherency_path else None
    return ang, coh


def sites_orientation_error(sites: "Any", gt: "Any", *, max_dist_scale: float = 1.5,
                            min_dist_px: float = 8.0) -> dict[str, Any]:
    """Angle error at GT sites against the nearest accepted predicted site."""
    if not len(gt) or not len(sites):
        return {"n": 0}
    gx, gy = gt["center_x_px"].to_numpy(float), gt["center_y_px"].to_numpy(float)
    ga = gt["fiber_angle_raster_deg"].to_numpy(float)
    gw = gt["width_px"].to_numpy(float)
    px, py = sites["center_x_px"].to_numpy(float), sites["center_y_px"].to_numpy(float)
    pa = sites["fiber_angle_raster_deg"].to_numpy(float)
    errs = []
    for i in range(len(gt)):
        d = np.hypot(px - gx[i], py - gy[i])
        j = int(np.argmin(d))
        if d[j] <= max(min_dist_px, max_dist_scale * gw[i]):
            errs.append(float(angular_diff_180(pa[j], ga[i])))
    e = np.asarray(errs)
    if not e.size:
        return {"n": 0}
    return {"n": int(e.size), "median_abs_error_deg": float(np.median(e)),
            "within_10deg": float((e <= 10).mean()), "within_20deg": float((e <= 20).mean())}


def vec_maps_to_angle(orient: np.ndarray) -> np.ndarray:
    return vec2_to_angle(orient[0], orient[1]).astype(np.float32)
