"""Turn sparse manual measurements into dense supervision maps.

Each manual measurement is a single oriented segment: one point, one angle, one
width.  The network predicts dense maps, so every annotation has to be
*painted* into those maps together with a mask saying where the supervision is
actually valid.  Getting this wrong is the most common silent failure in
heatmap regression -- unlabelled fiber pixels look like negatives, and the
model learns to suppress exactly what it should detect.

That failure is not hypothetical here.  The first version of this encoder
painted Gaussians at the annotated centres and left the rest of the image at
zero, and the trained centre head peaked at 0.56 with 1% recall.  The CSV
carries a few hundred chords per field out of tens of thousands of legitimate
measurement sites, so the overwhelming majority of true fibers were being
taught as background.

Three changes fix it, and they are all about *what the labels do and do not
claim*:

``ignore`` covers unannotated fiber
    A pixel on a fiber that nobody measured is not a negative.  It is unknown.
    The fiber mask from :mod:`fiber_prior` marks those pixels ignore, so the
    only true negatives left are pore and background -- which the label set
    really does justify.

orientation comes from the image, not the chord
    A hand-drawn chord is only roughly perpendicular to its fiber; the
    disagreement on this dataset runs to 26 degrees.  Supervising the
    orientation head from ``chord - 90`` teaches that scatter.  The structure
    tensor is both denser and more accurate, so it supplies ``cos2t``/``sin2t``
    wherever coherency is high, and the chord is kept only as a fallback.

reviewer rejections become real negatives
    Sites a reviewer looked at and threw out are the most informative negatives
    in the dataset, because they are the hard cases.  They are painted into
    ``neg_boost`` so the loss can weight them above ordinary background.

Maps produced (all at stride 1 unless the caller downsamples):

``center``      Gaussian peaks at measurement centres
``segment``     fiber-presence supervision (mask-derived when a prior is given)
``cos2t/sin2t`` local fiber orientation, encoded pi-periodically
``width``       fiber thickness (log-pixels by default)
``validity``    1 where a reliable measurement was judged possible
``reg_mask``    where width / orientation losses may be evaluated
``ignore``      pixels excluded from *all* losses
``neg_boost``   extra weight on confirmed-negative sites
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .utils import angle_to_direction, get_logger, wrap_deg_180

LOG = get_logger(__name__)

TARGET_KEYS = ("center", "segment", "cos2t", "sin2t", "width",
               "validity", "reg_mask", "ignore", "neg_boost")


@dataclass
class TargetConfig:
    """All knobs live here so the encoding is reproducible from the YAML."""
    sigma_scale: float = 0.22        # Gaussian sigma as a fraction of width
    sigma_min: float = 1.5
    sigma_max: float = 6.0
    reg_radius_scale: float = 0.35   # radius of the width/angle supervision disc
    reg_radius_min: float = 2.0
    segment_thickness: float = 0.35  # chord paint thickness, fraction of width
    log_width: bool = True
    ignore_radius_scale: float = 1.2  # ring around ambiguous labels
    y_sign: float = 1.0

    # -- prior-driven behaviour ------------------------------------------- #
    #: mark unannotated fiber pixels ignore instead of background.  This is the
    #: single most important switch in the file; turning it off reproduces the
    #: 1%-recall failure.
    ignore_unlabelled_fiber: bool = True
    #: where the orientation target comes from: the structure tensor, the
    #: annotated chord, or the tensor with the chord as fallback.
    orientation_source: str = "image"     # image | chord | blend
    #: below this coherency the tensor orientation is not trusted
    coherency_min: float = 0.25
    #: supervise orientation and width over the whole fiber mask, not just a
    #: disc at each annotated centre.  Widths are propagated from the nearest
    #: annotation on the same fiber; see ``width_propagate_px``.
    dense_orientation: bool = True
    #: how far a measured width may be propagated along its fiber, in units of
    #: that width.  0 disables propagation.
    width_propagate_scale: float = 2.0
    #: extra loss weight for reviewer-rejected sites
    negative_weight: float = 3.0
    negative_radius_px: float = 4.0
    #: paint the segment head from the fiber mask rather than from chords
    segment_from_prior: bool = True


def _gaussian_patch(sigma: float, radius: int) -> np.ndarray:
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    return np.outer(g, g)


def _draw_gaussian(heat: np.ndarray, cx: float, cy: float, sigma: float) -> None:
    """CornerNet-style splat, kept as a max so overlapping peaks do not sum."""
    radius = max(1, int(round(3 * sigma)))
    patch = _gaussian_patch(sigma, radius)
    h, w = heat.shape
    x, y = int(round(cx)), int(round(cy))
    l, r = min(x, radius), min(w - x, radius + 1)
    t, b = min(y, radius), min(h - y, radius + 1)
    if r <= -l or b <= -t:
        return
    masked_heat = heat[y - t:y + b, x - l:x + r]
    masked_patch = patch[radius - t:radius + b, radius - l:radius + r]
    if masked_patch.size:
        np.maximum(masked_heat, masked_patch, out=masked_heat)


def encode_targets(shape: tuple[int, int], ann: "Any", cfg: TargetConfig | None = None,
                   *, valid_mask: np.ndarray | None = None,
                   prior: "Any" | None = None,
                   negatives: "Any" | None = None) -> dict[str, np.ndarray]:
    """Encode one image's annotations into dense maps.

    Parameters
    ----------
    shape : (H, W) of the image the annotations are expressed in.
    ann : DataFrame with the consolidated label schema.
    valid_mask : optional bool array; False where the image content is not real
        (inpainted overlay, footer).  Those pixels go into ``ignore``.
    prior : optional :class:`fiber_prior.FiberPrior` for this crop.  Supplies
        the fiber mask and the orientation field.  Without it the encoder falls
        back to the old chord-only behaviour, which is kept only so the
        regression tests can compare the two.
    negatives : optional DataFrame of reviewer-rejected sites.
    """
    cfg = cfg or TargetConfig()
    h, w = shape
    out = {k: np.zeros((h, w), np.float32) for k in TARGET_KEYS}
    if valid_mask is not None:
        out["ignore"][~valid_mask] = 1.0

    import cv2

    have_prior = prior is not None and getattr(prior, "mask", None) is not None
    if have_prior and prior.mask.shape != (h, w):
        LOG.error("prior shape %s does not match target shape %s; ignoring it",
                  prior.mask.shape, (h, w))
        have_prior = False

    # ------------------------------------------------------------------ #
    # 1. fiber presence, and the ignore mask that stops unlabelled fiber
    #    from being scored as background
    # ------------------------------------------------------------------ #
    if have_prior:
        fiber = prior.mask.astype(bool)
        if cfg.segment_from_prior:
            out["segment"][fiber] = 1.0
        if cfg.dense_orientation:
            coh_ok = prior.coherency >= cfg.coherency_min
            sel = fiber & coh_ok
            t = np.deg2rad(prior.angle_deg[sel].astype(np.float64))
            out["cos2t"][sel] = np.cos(2 * t)
            out["sin2t"][sel] = np.sin(2 * t)
            if cfg.orientation_source in ("image", "blend"):
                out["reg_mask"][sel] = 1.0
                out["validity"][sel] = 1.0

    if ann is None or len(ann) == 0:
        if not have_prior:
            LOG.warning("encoding targets for an image with no annotations "
                        "and no prior: every pixel will be a negative")
        if have_prior and cfg.ignore_unlabelled_fiber:
            # nothing measured here, so nothing on the fibers is known
            out["ignore"][prior.mask.astype(bool)] = 1.0
            out["reg_mask"][:] = 0.0
        out["reg_mask"] *= (1.0 - out["ignore"])
        return out

    # ------------------------------------------------------------------ #
    # 2. per-annotation supervision
    # ------------------------------------------------------------------ #
    measured = np.zeros((h, w), np.uint8)   # where a human actually measured
    width_seed = np.full((h, w), np.nan, np.float32)

    # [v6.2] columns once, not a Series per row
    def _col(name, default, dtype=float):
        if name in ann.columns:
            return ann[name].to_numpy(dtype, na_value=default)                 if hasattr(ann[name], "to_numpy") else np.asarray(ann[name], dtype)
        return np.full(len(ann), default, dtype)

    _cx = np.asarray(ann["center_x_px"], float)
    _cy = np.asarray(ann["center_y_px"], float)
    _wd = np.asarray(ann["width_px"], float)
    _amb = (np.asarray(ann["ambiguous_crossing"]).astype(bool)
            if "ambiguous_crossing" in ann.columns else np.zeros(len(ann), bool))
    _cf = (np.nan_to_num(np.asarray(ann["annotation_confidence"], float), nan=1.0)
           if "annotation_confidence" in ann.columns else np.ones(len(ann)))
    _ang_col = np.asarray(ann["measurement_angle_deg"], float)
    _lfa = (np.asarray(ann["local_fiber_angle_deg"], float)
            if "local_fiber_angle_deg" in ann.columns
            else np.full(len(ann), np.nan))

    for _i in range(len(ann)):
        cx, cy = float(_cx[_i]), float(_cy[_i])
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        width = float(_wd[_i])
        if not np.isfinite(width) or width <= 0:
            continue
        ambiguous = bool(_amb[_i])
        conf = float(_cf[_i])

        sigma = float(np.clip(cfg.sigma_scale * width, cfg.sigma_min, cfg.sigma_max))
        if ambiguous:
            # do not teach a peak here, but do not call it background either
            rad = int(round(cfg.ignore_radius_scale * width))
            cv2.circle(out["ignore"], (int(cx), int(cy)), max(2, rad), 1.0, -1)
            continue

        _draw_gaussian(out["center"], cx, cy, sigma)

        ang = float(_ang_col[_i])
        if not cfg.segment_from_prior or not have_prior:
            # fiber presence along the measured chord (fallback path)
            if np.isfinite(ang):
                ux, uy = angle_to_direction(ang, cfg.y_sign)
                half = width / 2.0
                p1 = (int(round(cx - ux * half)), int(round(cy - uy * half)))
                p2 = (int(round(cx + ux * half)), int(round(cy + uy * half)))
                thick = max(1, int(round(cfg.segment_thickness * width)))
                cv2.line(out["segment"], p1, p2, 1.0, thick, cv2.LINE_8)

        # width / orientation supervision on a small disc around the centre
        rad = int(round(max(cfg.reg_radius_min, cfg.reg_radius_scale * width)))
        cv2.circle(measured, (int(round(cx)), int(round(cy))), rad, 1, -1)
        cv2.circle(width_seed, (int(round(cx)), int(round(cy))), rad,
                   float(np.log(width) if cfg.log_width else width), -1)

        # [v6.2] the disc, and every write through it, inside its own bbox
        _icx, _icy = int(round(cx)), int(round(cy))
        _r0, _r1 = max(0, _icy - rad), min(h, _icy + rad + 1)
        _c0, _c1 = max(0, _icx - rad), min(w, _icx + rad + 1)
        if _r0 >= _r1 or _c0 >= _c1:
            continue
        _sub = np.zeros((_r1 - _r0, _c1 - _c0), np.uint8)
        cv2.circle(_sub, (_icx - _c0, _icy - _r0), rad, 1, -1)
        sel = _sub.astype(bool)
        _box = (slice(_r0, _r1), slice(_c0, _c1))

        # orientation target at the annotated site
        use_chord = cfg.orientation_source == "chord" or not have_prior
        if not use_chord and cfg.orientation_source == "blend":
            cohere = float(prior.coherency[int(round(cy)), int(round(cx))])
            use_chord = cohere < cfg.coherency_min
        if use_chord:
            fiber_ang = _lfa[_i]
            if not np.isfinite(fiber_ang) and np.isfinite(ang):
                fiber_ang = ang - 90.0    # chord is perpendicular to the fiber
            if np.isfinite(fiber_ang):
                t = np.deg2rad(float(wrap_deg_180(fiber_ang)))
                out["cos2t"][_box][sel] = np.cos(2 * t)
                out["sin2t"][_box][sel] = np.sin(2 * t)

        out["width"][_box][sel] = np.log(width) if cfg.log_width else width
        out["reg_mask"][_box][sel] = 1.0
        out["validity"][_box][sel] = conf

    # ------------------------------------------------------------------ #
    # 3. propagate width along the fibers, so the dense orientation
    #    supervision has a width target to go with it
    # ------------------------------------------------------------------ #
    if have_prior and cfg.dense_orientation and cfg.width_propagate_scale > 0:
        from scipy.ndimage import distance_transform_edt

        if measured.any():
            dist, (iy, ix) = distance_transform_edt(
                measured == 0, return_indices=True)
            nearest = width_seed[iy, ix]
            typical = float(np.exp(np.nanmedian(width_seed))) if cfg.log_width \
                else float(np.nanmedian(width_seed))
            reach = cfg.width_propagate_scale * max(typical, 1.0)
            grow = prior.mask.astype(bool) & (dist <= reach) & np.isfinite(nearest)
            fresh = grow & (out["reg_mask"] > 0) & (measured == 0)
            out["width"][fresh] = nearest[fresh]
        else:
            # orientation is known but no width anywhere: do not invent one
            out["reg_mask"][:] = 0.0

    # ------------------------------------------------------------------ #
    # 4. unlabelled fiber is unknown, not background
    # ------------------------------------------------------------------ #
    if have_prior and cfg.ignore_unlabelled_fiber:
        peak = out["center"] > 1e-3
        unknown = prior.mask.astype(bool) & ~peak
        out["ignore"][unknown] = np.maximum(out["ignore"][unknown], 1.0)

    # ------------------------------------------------------------------ #
    # 5. reviewer-rejected sites: the hard negatives
    # ------------------------------------------------------------------ #
    if negatives is not None and len(negatives):
        for _, r in negatives.iterrows():
            cx, cy = float(r["center_x_px"]), float(r["center_y_px"])
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            wpx = float(r.get("width_px", np.nan))
            rad = int(round(max(cfg.negative_radius_px,
                                0.5 * wpx if np.isfinite(wpx) else 0.0)))
            cv2.circle(out["neg_boost"], (int(round(cx)), int(round(cy))),
                       max(2, rad), float(cfg.negative_weight), -1)
        # a confirmed negative overrides the blanket fiber ignore: the reviewer
        # looked at this exact site and said no.
        out["ignore"][out["neg_boost"] > 0] = 0.0
        out["center"][out["neg_boost"] > 0] = 0.0
        out["reg_mask"][out["neg_boost"] > 0] = 0.0

    out["center"] = np.clip(out["center"], 0.0, 1.0)
    out["segment"] = np.clip(out["segment"], 0.0, 1.0)
    out["reg_mask"] *= (1.0 - out["ignore"])
    out["validity"] *= (1.0 - out["ignore"])
    return out


def decode_width(value: np.ndarray | float, cfg: TargetConfig | None = None
                 ) -> np.ndarray | float:
    """Inverse of the width encoding."""
    cfg = cfg or TargetConfig()
    return np.exp(value) if cfg.log_width else value


def sample_negative_centers(shape: tuple[int, int], ann: "Any", n: int, *,
                            min_distance: float = 12.0,
                            rng: np.random.Generator | None = None,
                            valid_mask: np.ndarray | None = None,
                            fiber_mask: np.ndarray | None = None) -> np.ndarray:
    """Random points far enough from every annotation, for the patch baseline.

    When a fiber mask is supplied the sampler also avoids unmeasured fibers, for
    the same reason the dense encoder does: an unmeasured fiber is not a
    negative example of a fiber.
    """
    rng = rng or np.random.default_rng(0)
    h, w = shape
    if ann is not None and len(ann):
        pts = ann[["center_x_px", "center_y_px"]].to_numpy(float)
    else:
        pts = np.zeros((0, 2))
    out: list[tuple[float, float]] = []
    tries = 0
    while len(out) < n and tries < n * 60:
        tries += 1
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        if valid_mask is not None and not valid_mask[int(y), int(x)]:
            continue
        if fiber_mask is not None and fiber_mask[int(y), int(x)]:
            continue
        if pts.size and np.min(np.hypot(pts[:, 0] - x, pts[:, 1] - y)) < min_distance:
            continue
        out.append((x, y))
    if len(out) < n:
        LOG.warning("only %d/%d negative centres could be sampled", len(out), n)
    return np.asarray(out, np.float32).reshape(-1, 2)
