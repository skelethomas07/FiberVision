"""Dense supervision maps from sparse manual chords (v7).

Two modes, selected by ``TargetConfig.mode``:

``baseline``
    The v6 encoding, kept for comparison: Gaussian peaks at the annotated
    centres, width/orientation discs at the sites, unlabelled fibre ignored.
    Its known failure is that the centre head has no target on the ~99% of
    fibre pixels nobody clicked, so recall saturates regardless of threshold,
    and the width head only ever sees the annotated discs.

``geometry``
    A geometry-aware representation.  From the image-derived fibre mask:

    * ``segment``   fibre presence;
    * ``center``    the medial axis (ridge) of the mask -- a DENSE target, so
                    every fibre pixel has a defined target;
    * ``dist``      distance to the nearest fibre boundary, in pixels, so the
                    width at a ridge pixel is ``2 * dist``;
    * ``cos2t/sin2t`` structure-tensor orientation where coherent;
    * ``validity``  ridge pixels away from junctions with coherent orientation.

    Manual widths supervise the SCALE of the distance field.  At every annotated
    site the manual width is compared with the mask's own ridge width; the
    ratio is applied along the SAME branch only, with a weight that decays with
    distance from the site (``dist_weight``), and elsewhere the uncorrected
    mask distance is kept at a low ``unverified_weight``.  Nothing dense is
    fabricated far from a measurement without its uncertainty being encoded in
    the weight map that the loss uses.

Width strata.  Thick fibres are rare in pixels and in chords.  ``strata_edges``
and ``strata_weights`` (computed from the training label table, never from
validation/test) give every supervised pixel a weight inversely proportional to
the frequency of its width band, capped, so the thick tail is not drowned out.

All maps are float32 at stride 1.  Angles: raster convention throughout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .coords import direction_vector, wrap180
from .utils import get_logger

LOG = get_logger(__name__)

TARGET_KEYS = ("center", "segment", "cos2t", "sin2t", "width", "validity",
               "validity_mask", "reg_mask", "ignore", "dist", "dist_weight",
               "strata_weight")


@dataclass
class TargetConfig:
    mode: str = "geometry"                 # geometry | baseline
    # -- baseline centre peaks --------------------------------------------
    sigma_scale: float = 0.22
    sigma_min: float = 1.5
    sigma_max: float = 6.0
    # -- annotated-site discs -----------------------------------------------
    reg_radius_scale: float = 0.35
    reg_radius_min: float = 2.0
    log_width: bool = True
    ignore_radius_scale: float = 1.2
    ignore_unlabelled_fiber: bool = True    # baseline only
    # -- orientation ----------------------------------------------------------
    orientation_source: str = "image"       # image | chord
    coherency_min: float = 0.25
    # -- geometry mode -----------------------------------------------------------
    ridge_sigma: float = 1.0                # Gaussian half-width of the ridge target
    propagate_scale: float = 6.0            # e-folding distance along a branch, x width
    propagate_max_scale: float = 20.0       # hard reach along the branch, x width
    unverified_weight: float = 0.3          # dist weight where no manual width reaches
    verified_weight: float = 1.0
    assign_scale: float = 1.2               # site -> branch tolerance, x width
    assign_min_px: float = 6.0
    junction_clear_scale: float = 0.6       # validity = 0 within this x width of a junction
    ratio_clip: tuple[float, float] = (0.5, 2.0)
    # -- width strata (set from the TRAINING labels) -----------------------------
    strata_edges: tuple[float, ...] = (0.0, 6.0, 9.0, 13.0, 18.0, 25.0, 1e9)
    strata_weights: tuple[float, ...] = field(default_factory=lambda: (1.0,) * 6)
    strata_weight_cap: float = 6.0


# --------------------------------------------------------------------------- #
def strata_weights_from_widths(widths: np.ndarray, edges: tuple[float, ...],
                               cap: float = 6.0) -> tuple[float, ...]:
    """Inverse-frequency weight per width band, normalised to mean 1, capped."""
    w = np.asarray(widths, np.float64)
    w = w[np.isfinite(w) & (w > 0)]
    e = np.asarray(edges, np.float64)
    counts = np.histogram(w, bins=e)[0].astype(np.float64)
    if counts.sum() == 0:
        return tuple(1.0 for _ in range(len(e) - 1))
    freq = counts / counts.sum()
    inv = np.where(freq > 0, 1.0 / np.clip(freq, 1e-6, None), 0.0)
    inv = np.where(inv > 0, inv, np.nanmax(inv[inv > 0]) if (inv > 0).any() else 1.0)
    inv = inv / np.average(inv, weights=np.where(freq > 0, freq, 0) + 1e-12)
    return tuple(float(min(cap, max(0.2, v))) for v in inv)


def strata_weight_map(width_px: np.ndarray, cfg: TargetConfig) -> np.ndarray:
    e = np.asarray(cfg.strata_edges, np.float64)
    wts = np.asarray(cfg.strata_weights, np.float64)
    idx = np.clip(np.searchsorted(e, np.nan_to_num(width_px, nan=0.0), side="right") - 1,
                  0, len(wts) - 1)
    return wts[idx].astype(np.float32)


# --------------------------------------------------------------------------- #
def _gaussian_patch(sigma: float, radius: int) -> np.ndarray:
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    return np.outer(g, g)


def draw_gaussian(heat: np.ndarray, cx: float, cy: float, sigma: float) -> None:
    radius = max(1, int(round(3 * sigma)))
    patch = _gaussian_patch(sigma, radius)
    h, w = heat.shape
    x, y = int(round(cx)), int(round(cy))
    l, r = min(x, radius), min(w - x, radius + 1)
    t, b = min(y, radius), min(h - y, radius + 1)
    if r <= -l or b <= -t:
        return
    sub = heat[y - t:y + b, x - l:x + r]
    pat = patch[radius - t:radius + b, radius - l:radius + r]
    if pat.size:
        np.maximum(sub, pat, out=sub)


def _disc(h: int, w: int, cx: float, cy: float, rad: int):
    """Boolean disc and its bounding box slices (None if off-image)."""
    import cv2

    icx, icy = int(round(cx)), int(round(cy))
    r0, r1 = max(0, icy - rad), min(h, icy + rad + 1)
    c0, c1 = max(0, icx - rad), min(w, icx + rad + 1)
    if r0 >= r1 or c0 >= c1:
        return None, None
    sub = np.zeros((r1 - r0, c1 - c0), np.uint8)
    cv2.circle(sub, (icx - c0, icy - r0), rad, 1, -1)
    return sub.astype(bool), (slice(r0, r1), slice(c0, c1))


# --------------------------------------------------------------------------- #
def encode_targets(shape: tuple[int, int], ann: "Any", cfg: TargetConfig | None = None, *,
                   valid_mask: np.ndarray | None = None, prior: "Any" | None = None
                   ) -> dict[str, np.ndarray]:
    """Encode one crop's annotations (raster convention) into dense maps."""
    cfg = cfg or TargetConfig()
    h, w = shape
    out = {k: np.zeros((h, w), np.float32) for k in TARGET_KEYS}
    out["strata_weight"][:] = 1.0
    if valid_mask is not None:
        out["ignore"][~valid_mask] = 1.0

    have_prior = prior is not None and getattr(prior, "mask", None) is not None
    if have_prior and prior.mask.shape != (h, w):
        LOG.error("prior shape %s != target shape %s; ignoring it", prior.mask.shape, (h, w))
        have_prior = False
    fiber = prior.mask.astype(bool) if have_prior else np.zeros((h, w), bool)

    # ------------------------------------------------------------------ #
    # 1. dense, label-free geometry from the prior
    # ------------------------------------------------------------------ #
    bs = None
    if have_prior:
        out["segment"][fiber] = 1.0
        coh_ok = prior.coherency >= cfg.coherency_min
        sel = fiber & coh_ok
        t = np.deg2rad(prior.angle_deg[sel].astype(np.float64))
        out["cos2t"][sel] = np.cos(2 * t)
        out["sin2t"][sel] = np.sin(2 * t)
        if cfg.orientation_source == "image":
            out["reg_mask"][sel] = 1.0
        if cfg.mode == "geometry":
            from .skeleton import branch_structure, nearest_branch

            bs = branch_structure(fiber)
            out["dist"][fiber] = bs.edt[fiber]
            out["dist_weight"][fiber] = cfg.unverified_weight
            # ridge target: Gaussian-blurred medial axis, peak 1 on the axis
            from scipy import ndimage as ndi

            ridge = ndi.gaussian_filter(bs.skeleton.astype(np.float32), cfg.ridge_sigma)
            peak = float(ridge.max()) if ridge.max() > 0 else 1.0
            out["center"] = np.clip(ridge / peak, 0.0, 1.0).astype(np.float32)
            # validity: ridge pixels, coherent, clear of junctions
            local_w = 2.0 * bs.edt
            clear = bs.junction_dist > cfg.junction_clear_scale * np.maximum(local_w, 2.0)
            vpos = bs.skeleton & coh_ok & clear
            vneg = (~fiber) | (bs.skeleton & ~clear)
            out["validity"][vpos] = 1.0
            out["validity_mask"][vpos | vneg] = 1.0
            nb_label, nb_dist = nearest_branch(bs)
        else:
            out["validity_mask"][~fiber] = 1.0
    else:
        out["validity_mask"][:] = 1.0     # no prior: everything unlabelled is background

    if ann is None or len(ann) == 0:
        if cfg.mode == "baseline" and have_prior and cfg.ignore_unlabelled_fiber:
            out["ignore"][fiber] = 1.0
        out["reg_mask"] *= (1.0 - out["ignore"])
        out["width"][:] = 0.0
        if cfg.mode != "geometry":
            out["reg_mask"][:] = 0.0            # orientation known, width unknown: no width target
        return _finish(out)

    # ------------------------------------------------------------------ #
    # 2. per-annotation supervision
    # ------------------------------------------------------------------ #
    cx_all = np.asarray(ann["center_x_px"], np.float64)
    cy_all = np.asarray(ann["center_y_px"], np.float64)
    wd_all = np.asarray(ann["width_px"], np.float64)
    amb = (np.asarray(ann["ambiguous_crossing"]).astype(bool)
           if "ambiguous_crossing" in ann.columns else np.zeros(len(ann), bool))
    conf = (np.nan_to_num(np.asarray(ann["annotation_confidence"], np.float64), nan=1.0)
            if "annotation_confidence" in ann.columns else np.ones(len(ann)))
    fa_all = (np.asarray(ann["fiber_angle_raster_deg"], np.float64)
              if "fiber_angle_raster_deg" in ann.columns else np.full(len(ann), np.nan))
    sw_all = strata_weight_map(wd_all, cfg)

    measured = np.zeros((h, w), bool)
    site_ratio: list[tuple[int, float, float, float, float]] = []   # branch, cx, cy, width, ratio

    for i in range(len(ann)):
        cx, cy, width = float(cx_all[i]), float(cy_all[i]), float(wd_all[i])
        if not (0 <= cx < w and 0 <= cy < h) or not np.isfinite(width) or width <= 0:
            continue
        if amb[i]:
            rad = max(2, int(round(cfg.ignore_radius_scale * width)))
            d, box = _disc(h, w, cx, cy, rad)
            if d is not None:
                out["ignore"][box][d] = 1.0
            continue
        if cfg.mode == "baseline":
            sigma = float(np.clip(cfg.sigma_scale * width, cfg.sigma_min, cfg.sigma_max))
            draw_gaussian(out["center"], cx, cy, sigma)
        rad = int(round(max(cfg.reg_radius_min, cfg.reg_radius_scale * width)))
        d, box = _disc(h, w, cx, cy, rad)
        if d is None:
            continue
        wv = float(np.log(width) if cfg.log_width else width)
        out["width"][box][d] = wv
        out["reg_mask"][box][d] = 1.0
        out["validity"][box][d] = float(conf[i])
        out["validity_mask"][box][d] = 1.0
        out["strata_weight"][box][d] = sw_all[i]
        measured[box] |= d
        if cfg.orientation_source == "chord" and np.isfinite(fa_all[i]):
            tt = np.deg2rad(float(wrap180(fa_all[i])))
            out["cos2t"][box][d] = np.cos(2 * tt)
            out["sin2t"][box][d] = np.sin(2 * tt)
        if cfg.mode == "geometry" and bs is not None and bs.n_branches:
            ix = int(min(max(round(cx), 0), w - 1))
            iy = int(min(max(round(cy), 0), h - 1))
            lb, ndist = int(nb_label[iy, ix]), float(nb_dist[iy, ix])
            if lb > 0 and ndist <= max(cfg.assign_min_px, cfg.assign_scale * width):
                # mask ridge half-width near the site: max EDT on that branch within a small radius
                r = max(3, int(round(width)))
                y0, y1 = max(0, iy - r), min(h, iy + r + 1)
                x0, x1 = max(0, ix - r), min(w, ix + r + 1)
                near = (bs.labels[y0:y1, x0:x1] == lb)
                if near.any():
                    half = float(np.max(bs.edt[y0:y1, x0:x1][near]))
                    if half > 0.5:
                        ratio = float(np.clip(width / (2.0 * half), *cfg.ratio_clip))
                        site_ratio.append((lb, cx, cy, width, ratio))

    # ------------------------------------------------------------------ #
    # 3. geometry: propagate the manual scale along the SAME branch only
    # ------------------------------------------------------------------ #
    if cfg.mode == "geometry" and bs is not None and site_ratio:
        yy, xx = np.mgrid[0:h, 0:w]
        ratio_map = np.ones((h, w), np.float32)
        wgt_map = np.full((h, w), cfg.unverified_weight, np.float32)
        wgt_map[~fiber] = 0.0
        for lb, cx, cy, width, ratio in site_ratio:
            on_branch = (nb_label == lb) & fiber
            if not on_branch.any():
                continue
            dd = np.hypot(xx - cx, yy - cy)
            reach = cfg.propagate_max_scale * width
            tau = cfg.propagate_scale * width
            wgt = cfg.verified_weight * np.exp(-dd / max(tau, 1e-6))
            wgt = np.where(dd <= reach, np.maximum(wgt, cfg.unverified_weight), 0.0)
            upd = on_branch & (wgt > wgt_map)
            ratio_map[upd] = ratio
            wgt_map[upd] = wgt[upd]
            # the width head also learns the manual width along the branch, weighted
            upd2 = upd & (~measured)
            out["width"][upd2] = float(np.log(width) if cfg.log_width else width)
            out["reg_mask"][upd2] = np.maximum(out["reg_mask"][upd2], wgt[upd2])
            out["strata_weight"][upd2] = strata_weight_map(np.array([width]), cfg)[0]
        # global per-image correction for unverified pixels, at low weight
        g_ratio = float(np.median([r for *_, r in site_ratio]))
        unver = fiber & (wgt_map <= cfg.unverified_weight + 1e-6)
        ratio_map[unver] = g_ratio
        out["dist"] = (out["dist"] * ratio_map).astype(np.float32)
        out["dist_weight"] = wgt_map.astype(np.float32)

    # ------------------------------------------------------------------ #
    # 4. baseline: unlabelled fibre is unknown, not background
    # ------------------------------------------------------------------ #
    if cfg.mode == "baseline" and have_prior and cfg.ignore_unlabelled_fiber:
        unknown = fiber & ~(out["center"] > 1e-3)
        out["ignore"][unknown] = 1.0
    return _finish(out)


def _finish(out: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keep = 1.0 - out["ignore"]
    out["center"] = np.clip(out["center"], 0.0, 1.0)
    out["segment"] = np.clip(out["segment"], 0.0, 1.0)
    out["reg_mask"] = (out["reg_mask"] * keep).astype(np.float32)
    out["validity_mask"] = (out["validity_mask"] * keep).astype(np.float32)
    out["dist_weight"] = (out["dist_weight"] * keep).astype(np.float32)
    return out


def decode_width(value, cfg: TargetConfig | None = None):
    cfg = cfg or TargetConfig()
    return np.exp(value) if cfg.log_width else value
