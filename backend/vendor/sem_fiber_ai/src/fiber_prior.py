"""Image-derived priors: where the fibers are, and which way they run.

Two facts about this dataset motivate everything in this module.

1. The manual CSVs are a *sample*, not an exhaustive labelling.  A few hundred
   to a few thousand chords were measured on a field that contains tens of
   thousands of legitimate measurement sites.  Painting a Gaussian at each
   annotation and leaving every other pixel at zero tells the network that
   ~99% of the real fiber pixels are background.  That is why the centre
   heatmap saturates around 0.5 and recall collapses.  The fix is to mark
   unannotated *fiber* pixels as ``ignore`` rather than as negatives, which
   needs a mask of where the fibers are.

2. A human-drawn chord is only approximately perpendicular to the fiber.  On
   this dataset the measured chord-versus-ridge disagreement runs from 3 to 26
   degrees -- the same magnitude as the model's orientation error.  Supervising
   orientation from the chord therefore teaches the annotator's scatter.  The
   image knows the fiber direction far better than the chord does.

So: take the fiber mask and a dense orientation field from the pixels, and use
the CSV for what it is genuinely authoritative about -- the *width*, and which
sites a reviewer accepted or rejected.

Everything is cached to a ``.npz`` next to the image, because the ridge filter
is the slowest thing in the training loop if recomputed per tile.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .utils import angular_diff_180, get_logger, wrap_deg_180

LOG = get_logger(__name__)


@dataclass
class PriorConfig:
    cache_and_warp: bool = True
    """Knobs for the image-derived prior.  All of them are recorded in the run
    manifest so a reconstruction is reproducible."""

    sigmas: tuple[float, ...] = (1.0, 1.6, 2.5, 4.0, 6.0)
    #: how the fiber/background split is chosen on the ridge response
    threshold: str = "otsu"          # otsu | percentile
    percentile: float = 70.0         # used when threshold == "percentile"
    #: dilate the mask so the ignore ring covers the fiber flanks too
    dilate_px: int = 2
    #: drop specks smaller than this many pixels from the mask
    min_object_px: int = 24
    #: gaussian sigma of the structure tensor window, in pixels
    tensor_sigma: float = 2.0
    #: gradient smoothing before the tensor
    grad_sigma: float = 1.0
    #: below this coherency the local orientation is meaningless
    coherency_min: float = 0.25
    #: "auto" decides from the annotations whether fibers are bright or dark
    polarity: str = "auto"           # auto | bright | dark

    # -- [v3] area control ------------------------------------------------- #
    # v2 grew the mask by flood-filling every bright connected component that a
    # ridge touched.  On a dense separator mat that component IS the whole
    # image: the August run produced masks covering 70-86% of the manual fields
    # and 94-100% of the pseudo-labelled ones.  With unlabelled fiber ignored,
    # that leaves 0-20% of pixels as negatives -- v1's recall collapse mirrored
    # into a precision collapse.  The mask is now bounded in two ways.
    #: how far the mask may grow from a ridge seed, in units of max(sigmas)
    flank_scale: float = 1.5
    #: backstop ceiling on mask area.  Deliberately loose: a separator mat can
    #: genuinely be 80% fiber, and on the August data the manual fields
    #: separated at AUC 0.95-0.98, i.e. the mask was tracking real brightness.
    #: Area alone is therefore not evidence of failure -- see ``auc_min``.
    area_max: float = 0.85
    #: the real quality gate.  A mask that has leaked into the pores stops
    #: tracking intensity; the pseudo-labelled fields ran at AUC 0.60-0.67 with
    #: 94-100% area, which is leakage, while 0.97 at 80% area is a dense field.
    auc_min: float = 0.80
    #: below this the mask is suspiciously small and is reported as such
    area_min: float = 0.05
    #: enforce area_max by escalating the seed threshold
    auto_area: bool = True
    #: audit failure threshold for the fraction of annotated centres covered
    min_coverage: float = 0.90


# --------------------------------------------------------------------------- #
# ridge response and mask
# --------------------------------------------------------------------------- #
def _to_float(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray, np.float32)
    lo, hi = np.percentile(g, (1.0, 99.0))
    if hi <= lo:
        lo, hi = float(g.min()), float(g.max()) or 1.0
    return np.clip((g - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def decide_polarity(gray: np.ndarray, ann: "Any" | None) -> str:
    """Are the fibers brighter or darker than their surroundings?

    SEM secondary-electron images of a separator normally show bright fibers on
    a dark pore background, but backscattered and inverted exports exist, and
    guessing wrong silently inverts every ridge.  When annotations are
    available the question is settled empirically: compare the intensity at the
    measured centres against the field as a whole.
    """
    if ann is None or not len(ann):
        return "bright"
    h, w = gray.shape
    xs = np.clip(ann["center_x_px"].to_numpy(float).round().astype(int), 0, w - 1)
    ys = np.clip(ann["center_y_px"].to_numpy(float).round().astype(int), 0, h - 1)
    at_ann = float(np.median(gray[ys, xs]))
    overall = float(np.median(gray))
    return "bright" if at_ann >= overall else "dark"


# --------------------------------------------------------------------------- #
# [v6.2] Fast multiscale ridge filter.
#
# Identical in form to skimage.filters.frangi -- same vesselness expression,
# same "gamma is fixed by the first sigma" behaviour (frangi sets gamma once
# and leaves it non-None for every later scale; reproducing that is what makes
# the two agree) -- but the Hessian is built with separable cv2 filters instead
# of five scipy.ndimage.gaussian_filter passes at truncate=8/100.
#
# The two polarities share everything: for a negated image the Hessian is
# negated, so its eigenvalues only change sign, and the structuredness term
# sqrt(l1^2 + l2^2) is sign-invariant -- which means gamma is common too.
# --------------------------------------------------------------------------- #
def _gauss_deriv_kernels(sigma: float, truncate: float = 6.0):
    """Normalised gaussian and its first derivative, at sigma/sqrt(2)."""
    import math

    s = float(sigma) / math.sqrt(2.0)      # two successive passes give sigma
    r = max(1, int(truncate * s + 0.5))
    x = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-x * x / (2.0 * s * s))
    g /= g.sum()
    d1 = -(x / (s * s)) * g
    return g.astype(np.float32), d1.astype(np.float32)


def _hessian_eigs(img: np.ndarray, sigma: float, truncate: float = 6.0):
    """(smaller, larger) eigenvalue by magnitude, plus the structuredness term."""
    import cv2

    g, d1 = _gauss_deriv_kernels(sigma, truncate)

    def sep(a, kx, ky):
        return cv2.sepFilter2D(a, cv2.CV_32F, kx, ky,
                               borderType=cv2.BORDER_REFLECT)

    gr = sep(img, g, d1)
    gc = sep(img, d1, g)
    hrr, hrc, hcc = sep(gr, g, d1), sep(gr, d1, g), sep(gc, d1, g)

    tr = hrr + hcc
    det = hrr * hcc - hrc * hrc
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))
    a, b = tr / 2.0 + disc, tr / 2.0 - disc
    swap = np.abs(a) < np.abs(b)
    lo = np.where(swap, a, b)
    hi = np.where(swap, b, a)
    return lo, hi, np.sqrt(lo * lo + hi * hi)


def _vesselness(lo, hi, s, gamma, beta=0.5):
    l2 = np.maximum(hi, 1e-10)
    r_b = np.abs(lo) / l2
    return (np.exp(-(r_b ** 2) / (2.0 * beta ** 2))
            * (1.0 - np.exp(-(s ** 2) / (2.0 * gamma ** 2))))


def ridge_response_both(gray: np.ndarray, cfg: "PriorConfig | None" = None
                        ) -> "tuple[np.ndarray, np.ndarray]":
    """Bright- and dark-ridge responses from one pass of Hessians."""
    cfg = cfg or PriorConfig()
    img = np.ascontiguousarray(_to_float(gray), dtype=np.float32)
    sigmas = cfg.sigmas if cfg else (1.0, 2.0, 4.0)
    fb = np.zeros_like(img)
    fd = np.zeros_like(img)
    gamma = None
    for sg in sigmas:
        lo, hi, s = _hessian_eigs(img, float(sg))
        if gamma is None:                       # fixed by the first sigma
            gamma = float(s.max()) / 2.0 or 1.0
        fb = np.maximum(fb, _vesselness(-lo, -hi, s, gamma)).astype(np.float32)
        fd = np.maximum(fd, _vesselness(lo, hi, s, gamma)).astype(np.float32)
    return _norm_response(fb), _norm_response(fd)


def _norm_response(r: np.ndarray) -> np.ndarray:
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    hi = float(np.percentile(r, 99.5))
    return np.clip(r / hi, 0.0, 1.0).astype(np.float32) if hi > 0 else r


def ridge_response(gray: np.ndarray, cfg: PriorConfig | None = None, *,
                   polarity: str = "bright") -> np.ndarray:
    """Multiscale Hessian ridge filter, normalised to [0, 1].

    Frangi's vesselness is used rather than a single-scale Laplacian because the
    fiber diameters in this dataset span roughly 5-25 px once the magnification
    range is taken into account, and a single sigma silently misses one end.
    """
    bright, dark = ridge_response_both(gray, cfg)      # [v6.2]
    return dark if polarity == "dark" else bright


def mask_separability_auc(gray: np.ndarray, mask: np.ndarray, *,
                          polarity: str = "bright") -> float:
    """AUC of intensity as a classifier of mask membership.

    1.0 means the mask sits exactly on the bright (or dark) structure; 0.5
    means it was drawn without reference to brightness.  This is the one mask
    diagnostic that needs no labels, which is why it also gates the pseudo-
    labelled fields, where there are none.
    """
    m = np.asarray(mask, bool)
    if not m.any() or m.all():
        return float("nan")
    g = np.asarray(gray, np.float32).ravel()
    mm = m.ravel()
    idx = np.random.default_rng(0).choice(g.size, size=min(g.size, 200_000),
                                          replace=False)
    gs, ms = g[idx], mm[idx]
    n_pos, n_neg = float(ms.sum()), float((~ms).sum())
    if n_pos < 10 or n_neg < 10:
        return float("nan")
    order = np.argsort(gs, kind="stable")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, gs.size + 1)
    auc = (ranks[ms].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(1.0 - auc if polarity == "dark" else auc)


def _remove_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Version-independent small-object removal (skimage's kwargs keep moving)."""
    if min_px <= 0 or not mask.any():
        return mask
    import cv2

    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[lab]


def fiber_mask(response: np.ndarray, gray: np.ndarray,
               cfg: PriorConfig | None = None, *,
               polarity: str = "bright") -> np.ndarray:
    """Binary fiber mask, seeded on ridges and grown over the fiber body.

    A ridge filter alone is not enough here.  Frangi vesselness measures Hessian
    curvature, so on a fiber several pixels thick it responds strongly at the
    two flanks and *weakly along the centre line* -- precisely where a human
    puts the measurement point.  Thresholding the response directly produced a
    mask that missed 72% of the annotated centres in testing.

    So the two cues are combined the way they should be: ridges supply
    confident seeds, an intensity threshold supplies the fiber body, and the
    mask is the set of bright components that contain at least one seed.  That
    keeps genuine fibers whole while rejecting bright background speckle, which
    has no ridge inside it.
    """
    import cv2
    from skimage.filters import threshold_otsu

    cfg = cfg or PriorConfig()
    r = np.asarray(response, np.float32)
    g = _to_float(gray)
    if polarity == "dark":
        g = 1.0 - g

    # fiber body: bright relative to the field
    try:
        body = g >= float(threshold_otsu(g))
    except ValueError:                                    # pragma: no cover
        body = g >= float(np.percentile(g, 50))

    # ridge seeds
    if cfg.threshold == "percentile":
        seed_thr = float(np.percentile(r, cfg.percentile))
    else:
        pos = r[r > 0]
        try:
            seed_thr = float(threshold_otsu(pos)) if pos.size > 16 else 0.5
        except ValueError:                                # pragma: no cover
            seed_thr = 0.5
    seeds = r >= seed_thr

    # [v3] Grow the mask from the seeds, not across the whole component. A
    # fiber is a few pixels wide, so a body pixel more than ~1.5 fiber radii
    # from any ridge is pore wall or resin, not fiber -- however brightly it
    # connects to one. Bounding the reach is what keeps the mask off the 80%
    # plateau that v2 could only warn about.
    reach_px = max(1, int(round(cfg.flank_scale * max(cfg.sigmas))))
    kr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (2 * reach_px + 1, 2 * reach_px + 1))

    def _grow(seed_mask: np.ndarray) -> np.ndarray:
        reach = cv2.dilate(seed_mask.astype(np.uint8), kr).astype(bool)
        m = (body & reach) | seed_mask
        m = _remove_small(m, cfg.min_object_px)
        if cfg.dilate_px > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * cfg.dilate_px + 1, 2 * cfg.dilate_px + 1))
            m = cv2.dilate(m.astype(np.uint8), k).astype(bool)
        return m

    m = _grow(seeds)

    # Ceiling.  Two conditions, and the second is the one that matters: a mask
    # can legitimately be large on a dense mat, but it cannot be large AND stop
    # separating fiber from pore by brightness.  When either trips, the seed
    # threshold is the remaining lever -- raise it and report by how much, so
    # the rescue is auditable rather than silent.
    def _bad(mask: np.ndarray) -> tuple[bool, float, float]:
        area = float(mask.mean())
        auc = mask_separability_auc(gray, mask, polarity=polarity)
        return (area > cfg.area_max
                or (np.isfinite(auc) and auc < cfg.auc_min)), area, auc

    if cfg.auto_area:
        bad, area0, auc0 = _bad(m)
        if bad:
            pos = r[r > 0]
            best = (m, area0, auc0)
            for q in (60.0, 70.0, 80.0, 88.0, 93.0, 96.0, 98.0,
                      99.0, 99.5, 99.8):   # [1e] the dense fields live
                                            # in the tail
                thr = float(np.percentile(pos, q)) if pos.size else seed_thr
                if thr <= seed_thr:
                    continue
                cand = _grow(r >= thr)
                bad_c, area_c, auc_c = _bad(cand)
                if not np.isfinite(auc_c):
                    continue
                if auc_c > best[2] or (best[1] > cfg.area_max
                                       and area_c < best[1]):
                    best = (cand, area_c, auc_c)
                if not bad_c:
                    LOG.info("prior rescued: area %.2f->%.2f, separability "
                             "%.2f->%.2f (seed threshold %.3f -> %.3f)",
                             area0, area_c, auc0, auc_c, seed_thr, thr)
                    return cand
            m, area_b, auc_b = best
            LOG.warning("the fiber mask could not be brought inside "
                        "area<=%.2f / AUC>=%.2f (best area %.2f, AUC %.2f). "
                        "Either the field is genuinely dense or the polarity "
                        "is inverted -- do not train on it unchecked.",
                        cfg.area_max, cfg.auc_min, area_b, auc_b)
    return m


# --------------------------------------------------------------------------- #
# orientation field
# --------------------------------------------------------------------------- #
def orientation_field(gray: np.ndarray, cfg: PriorConfig | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Dense fiber orientation and coherency from the structure tensor.

    Returns ``(angle_deg, coherency)``.  The angle is pi-periodic in
    ``[-90, 90)`` and expressed in the project's raster convention: 0 deg points
    along +x, +90 deg points along +y (downwards on screen), matching
    :func:`utils.angle_to_direction` with ``y_sign = +1``.

    The fiber runs along the eigenvector of the *smaller* eigenvalue -- the
    direction in which the image changes least.  Coherency is the normalised
    eigenvalue gap, so 1 means a clean unidirectional ridge and 0 means an
    isotropic blob or a crossing, which is exactly where orientation
    supervision should be withheld.
    """
    from skimage.feature import structure_tensor

    cfg = cfg or PriorConfig()
    g = _to_float(gray)
    # skimage returns the tensor in (row, col) order: Arr = <Iy Iy>, Arc = <Iy Ix>
    a_rr, a_rc, a_cc = structure_tensor(
        g, sigma=cfg.tensor_sigma, order="rc",
        mode="nearest")
    jxx, jyy, jxy = a_cc, a_rr, a_rc          # rename into (x = col, y = row)

    # dominant *gradient* orientation; the fiber is perpendicular to it
    grad_ang = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    fiber_ang = np.rad2deg(grad_ang) + 90.0

    tr = jxx + jyy
    diff = np.sqrt(np.clip((jxx - jyy) ** 2 + 4.0 * jxy ** 2, 0.0, None))
    coh = np.where(tr > 1e-12, diff / np.clip(tr, 1e-12, None), 0.0)
    return wrap_deg_180(fiber_ang).astype(np.float32), coh.astype(np.float32)


# --------------------------------------------------------------------------- #
# bundled prior with caching
# --------------------------------------------------------------------------- #
class FiberPrior:
    """Fiber mask + orientation field for one image, cached on disk."""

    __slots__ = ("mask", "angle_deg", "coherency", "response", "polarity")

    def __init__(self, mask: np.ndarray, angle_deg: np.ndarray,
                 coherency: np.ndarray, response: np.ndarray,
                 polarity: str = "bright") -> None:
        self.mask = mask
        self.angle_deg = angle_deg
        self.coherency = coherency
        self.response = response
        self.polarity = polarity

    @property
    def shape(self) -> tuple[int, int]:
        return self.mask.shape

    def crop(self, y0: int, x0: int, h: int, w: int) -> "FiberPrior":
        sl = (slice(y0, y0 + h), slice(x0, x0 + w))
        return FiberPrior(self.mask[sl], self.angle_deg[sl], self.coherency[sl],
                          self.response[sl], self.polarity)



    def pad_to(self, h: int, w: int) -> "FiberPrior":
        """[1e] Reflect-pad to (h, w) so a boundary crop matches its tile."""
        import numpy as _np

        def _p(a):
            dy, dx = h - a.shape[0], w - a.shape[1]
            if dy <= 0 and dx <= 0:
                return a[:h, :w]
            return _np.pad(a, ((0, max(0, dy)), (0, max(0, dx))),
                           mode="reflect")[:h, :w]

        return FiberPrior(_p(self.mask), _p(self.angle_deg),
                          _p(self.coherency), _p(self.response),
                          self.polarity)

    def warp_like(self, M, out_hw, *, angle_only_rotation=None):
        """Apply a similarity transform to this prior.

        ``M`` is the 3x3 matrix the image was warped with.  mask, coherency and
        response are resampled nearest-neighbour; the ANGLE FIELD is resampled
        nearest-neighbour for position and then transformed analytically, so no
        angle value is ever interpolated.  A similarity maps a direction
        exactly: d -> A d with A the linear part, including reflections.
        """
        import cv2
        import numpy as _np

        from .utils import wrap_deg_180

        h, w = out_hw
        A = _np.asarray(M, float)[:2, :2]

        def _nn(a, is_bool=False):
            src = a.astype(_np.float32)
            out = cv2.warpAffine(src, _np.asarray(M, float)[:2], (w, h),
                                 flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_REFLECT_101)
            return out.astype(bool) if is_bool else out

        mask = _nn(self.mask, is_bool=(self.mask.dtype == bool))
        coh = _nn(self.coherency)
        resp = _nn(self.response)
        ang = _nn(self.angle_deg)

        th = _np.deg2rad(ang, dtype=_np.float32)
        vx = _np.cos(th)
        vy = _np.sin(th)
        nx = A[0, 0] * vx + A[0, 1] * vy
        ny = A[1, 0] * vx + A[1, 1] * vy
        ang = wrap_deg_180(_np.rad2deg(_np.arctan2(ny, nx))).astype(_np.float32)

        return FiberPrior(mask, ang, coh, resp, self.polarity)

    # -- construction ------------------------------------------------------ #
    @classmethod
    def compute(cls, gray: np.ndarray, cfg: PriorConfig | None = None, *,
                ann: "Any" | None = None) -> "FiberPrior":
        cfg = cfg or PriorConfig()
        pol = cfg.polarity
        if pol == "auto":
            pol = decide_polarity(gray, ann)
        resp = ridge_response(gray, cfg, polarity=pol)
        mask = fiber_mask(resp, gray, cfg, polarity=pol)
        ang, coh = orientation_field(gray, cfg)
        return cls(mask, ang, coh, resp.astype(np.float32), pol)

    @classmethod
    def load_or_compute(cls, gray: np.ndarray, cache: str | Path | None,
                        cfg: PriorConfig | None = None, *,
                        ann: "Any" | None = None) -> "FiberPrior":
        cfg = cfg or PriorConfig()
        # Cache identity must include the pixels and the resolved polarity.
        # v5 keyed only on config + shape, so replacing an image with another
        # same-sized field (or changing annotations enough to flip auto polarity)
        # could silently reuse a stale mask.
        resolved_pol = cfg.polarity if cfg.polarity != "auto" else decide_polarity(gray, ann)
        pix_hash = hashlib.blake2b(np.ascontiguousarray(gray).view(np.uint8),
                                   digest_size=8).hexdigest()
        key_src = (repr(sorted(cfg.__dict__.items())) + str(gray.shape) +
                   resolved_pol + pix_hash)
        key = hashlib.md5(key_src.encode()).hexdigest()[:12]
        path = Path(cache).with_suffix(f".{key}.npz") if cache else None
        if path is not None and path.exists():
            try:
                z = np.load(path)
                return cls(z["mask"].astype(bool), z["angle"], z["coh"],
                           z["resp"], str(z["polarity"]))
            except Exception as exc:                       # pragma: no cover
                LOG.warning("prior cache %s unreadable (%s); recomputing",
                            path, exc)
        prior = cls.compute(gray, cfg, ann=ann)
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, mask=prior.mask, angle=prior.angle_deg,
                                    coh=prior.coherency, resp=prior.response,
                                    polarity=prior.polarity)
            except OSError as exc:                         # pragma: no cover
                LOG.warning("could not cache prior to %s (%s)", path, exc)
        return prior


# --------------------------------------------------------------------------- #
# self-check
# --------------------------------------------------------------------------- #
def audit_prior(prior: FiberPrior, ann: "Any" | None,
                gray: np.ndarray | None = None,
                cfg: PriorConfig | None = None) -> dict[str, Any]:
    """Is the prior any good?  Answer it with the annotations, not by eye.

    ``centre_coverage`` is the decisive number: the fraction of manually
    measured centres that land inside the fiber mask.  A human put those points
    on fibers, so a mask that misses them is wrong.  Below ~0.9 the ridge
    settings need adjusting before training on them.

    ``chord_vs_prior_deg`` compares each chord's implied fiber direction
    (chord - 90) against the structure tensor.  A large *median* offset means a
    convention error somewhere; a large *spread* with a near-zero median is
    ordinary annotator scatter, and is the reason orientation is supervised
    from the image instead.
    """
    cfg = cfg or PriorConfig()
    out: dict[str, Any] = {
        "polarity": prior.polarity,
        "mask_area_fraction": float(prior.mask.mean()),
        "median_coherency_in_mask": float(np.median(prior.coherency[prior.mask]))
        if prior.mask.any() else float("nan"),
    }
    out.update(_separability(prior, gray))
    if ann is None or not len(ann):
        out.update(_verdict(out, cfg))
        return out
    h, w = prior.shape
    xs = np.clip(ann["center_x_px"].to_numpy(float).round().astype(int), 0, w - 1)
    ys = np.clip(ann["center_y_px"].to_numpy(float).round().astype(int), 0, h - 1)
    out["n_annotations"] = int(len(xs))
    out["centre_coverage"] = float(prior.mask[ys, xs].mean())

    if "measurement_angle_deg" in ann.columns:
        chord = ann["measurement_angle_deg"].to_numpy(float)
        implied = wrap_deg_180(chord - 90.0)
        prior_ang = prior.angle_deg[ys, xs]
        ok = np.isfinite(implied) & (prior.coherency[ys, xs] > 0.2)
        if ok.any():
            d = np.asarray(angular_diff_180(implied[ok], prior_ang[ok]), float)
            out["chord_vs_prior_deg"] = {
                "n": int(ok.sum()),
                "median": float(np.median(d)),
                "p90": float(np.percentile(d, 90)),
                "within_15deg": float((d <= 15).mean()),
            }
    out.update(_verdict(out, cfg))
    _warn_on_audit(out, cfg)
    return out


def _separability(prior: FiberPrior, gray: np.ndarray | None) -> dict[str, Any]:
    """How cleanly does the mask separate fiber from background?

    The coverage check only catches a mask that is too SMALL. Too large is the
    more dangerous direction once unlabelled fiber is being ignored: an
    over-grown mask marks real background as ignore, the loss is left with
    almost no negatives, and the model fires everywhere while precision
    collapses -- a failure that looks like success in the training curve.

    No labels are needed to detect it. If the mask is tracking real fiber, the
    intensity distributions inside and outside it barely overlap; as the mask
    swallows background, the two distributions converge. The AUC of intensity as
    a classifier of mask membership measures exactly that: 1.0 is perfect
    separation, 0.5 is a mask drawn at random with respect to brightness.
    """
    out: dict[str, Any] = {}
    if gray is None:
        return out
    auc = mask_separability_auc(gray, prior.mask, polarity=prior.polarity)
    if np.isfinite(auc):
        out["intensity_separability_auc"] = auc

    # how much of the image is left as usable background once the mask (which
    # becomes the ignore region) is removed
    out["negative_budget_fraction"] = float((~prior.mask).mean())
    return out


def _verdict(out: dict[str, Any], cfg: PriorConfig) -> dict[str, Any]:
    """[v3] Turn the audit numbers into a pass/fail plus named reasons.

    v2 computed all of this and then logged it.  A number that only ever
    reaches a log line does not gate anything, and the August run trained twice
    straight through an ERROR that said not to.  The caller now gets a boolean
    it has to act on.
    """
    fails: list[str] = []
    cov = out.get("centre_coverage")
    if cov is not None and cov < cfg.min_coverage:
        fails.append(f"centre_coverage={cov:.2f} < {cfg.min_coverage:.2f} "
                     "(mask too small: real fibers would be taught as background)")
    area = out.get("mask_area_fraction")
    if area is not None and area > cfg.area_max:
        fails.append(f"mask_area_fraction={area:.2f} > {cfg.area_max:.2f} "
                     "(mask too large: too few negatives, precision collapses)")
    if area is not None and area < cfg.area_min:
        fails.append(f"mask_area_fraction={area:.2f} < {cfg.area_min:.2f} "
                     "(mask nearly empty: check polarity)")
    auc = out.get("intensity_separability_auc")
    if auc is not None and auc < cfg.auc_min:
        fails.append(f"intensity_separability_auc={auc:.2f} < {cfg.auc_min:.2f} "
                     "(mask is not tracking brightness; likely grown into pores)")
    return {"ok": not fails, "failures": fails}


def _warn_on_audit(out: dict[str, Any], cfg: PriorConfig | None = None) -> None:
    """Both directions of mask failure, each with the number that shows it."""
    cfg = cfg or PriorConfig()
    cov = out.get("centre_coverage")
    if cov is not None and cov < cfg.min_coverage:
        LOG.warning("only %.0f%% of annotated centres fall inside the fiber "
                    "mask -- it is too SMALL. Real fibers will be taught as "
                    "background. Raise dilate_px or lower the threshold.",
                    100 * cov)
    area = out.get("mask_area_fraction", 0.0)
    auc = out.get("intensity_separability_auc")
    budget = out.get("negative_budget_fraction", 1.0)
    if area > cfg.area_max:
        LOG.warning("the fiber mask covers %.0f%% of the image -- it is "
                    "probably too LARGE. Only %.0f%% of pixels are left as "
                    "negatives, so the detector has little background to learn "
                    "from and will over-fire.", 100 * area, 100 * budget)
    if auc is not None and auc < cfg.auc_min:
        LOG.warning("intensity separates fiber from background at AUC %.2f. "
                    "Below the configured floor the mask is not tracking brightness, which "
                    "usually means it has grown into the pores; check the "
                    "polarity and the threshold before training.", auc)
    if budget < 0.15:
        LOG.error("only %.0f%% of the image remains as trainable background. "
                  "With unlabelled fiber ignored, that leaves almost no "
                  "negatives and precision will collapse.", 100 * budget)


# --------------------------------------------------------------------------- #
# [v3] choose the knobs from the data instead of guessing them
# --------------------------------------------------------------------------- #
def tune_prior_config(samples: "list[tuple[np.ndarray, Any]]",
                      base: PriorConfig | None = None, *,
                      flank_grid: tuple[float, ...] = (0.8, 1.2, 1.6, 2.2),
                      area_grid: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60),
                      min_coverage: float = 0.95) -> dict[str, Any]:
    """Search ``flank_scale`` x ``area_max`` for the smallest mask that still
    contains the annotated centres.

    The two failure directions pull against each other -- coverage wants a
    bigger mask, the negative budget wants a smaller one -- and v2 left both
    knobs at hand-picked defaults, which is how the August run ended up at 86%
    area with a 0.82 coverage on the worst field.  There is no need to guess:
    the annotations say where fiber is, so score every candidate on
    ``min(coverage)`` across fields and take the smallest median area that
    clears ``min_coverage``.

    ``samples`` is a list of ``(gray, annotations)`` pairs -- three or four
    representative fields is enough, and each candidate costs one ridge filter
    per field, so keep it small.
    """
    base = base or PriorConfig()
    rows: list[dict[str, Any]] = []
    # the ridge response does not depend on flank_scale/area_max, so compute it
    # once per field and reuse it across the grid
    cached = []
    for gray, ann in samples:
        pol = base.polarity if base.polarity != "auto" else decide_polarity(gray, ann)
        cached.append((gray, ann, pol, ridge_response(gray, base, polarity=pol)))

    for flank in flank_grid:
        for area_max in area_grid:
            cfg = PriorConfig(**{**base.__dict__, "flank_scale": float(flank),
                                 "area_max": float(area_max)})
            covs, areas = [], []
            for gray, ann, pol, resp in cached:
                m = fiber_mask(resp, gray, cfg, polarity=pol)
                areas.append(float(m.mean()))
                if ann is not None and len(ann):
                    h, w = m.shape
                    xs = np.clip(ann["center_x_px"].to_numpy(float).round().astype(int), 0, w - 1)
                    ys = np.clip(ann["center_y_px"].to_numpy(float).round().astype(int), 0, h - 1)
                    covs.append(float(m[ys, xs].mean()))
            rows.append({"flank_scale": float(flank), "area_max": float(area_max),
                         "worst_coverage": float(min(covs)) if covs else float("nan"),
                         "median_area": float(np.median(areas)),
                         "worst_area": float(max(areas))})

    ok = [r for r in rows if r["worst_coverage"] >= min_coverage]
    if ok:
        best = min(ok, key=lambda r: (r["median_area"], -r["worst_coverage"]))
        LOG.info("prior tuned: flank_scale=%.1f area_max=%.2f -> worst coverage "
                 "%.2f, median area %.2f", best["flank_scale"], best["area_max"],
                 best["worst_coverage"], best["median_area"])
    else:
        best = max(rows, key=lambda r: r["worst_coverage"])
        LOG.warning("no prior setting reached coverage %.2f; best is %.2f at "
                    "flank_scale=%.1f area_max=%.2f. The mask cannot both hold "
                    "the annotations and leave a negative budget -- suspect the "
                    "polarity or a mis-registered field.", min_coverage,
                    best["worst_coverage"], best["flank_scale"], best["area_max"])
    return {"best": best, "grid": rows,
            "config": {"flank_scale": best["flank_scale"],
                       "area_max": best["area_max"]}}


def best_polarity(gray, cfg=None, ann=None):
    """Choose the polarity that actually separates fiber from background.

    ``decide_polarity`` settles the question from the annotations, but the ~119
    unlabelled fields have none and fall through to "bright".  On several of the
    2-* fields that produced a mask with AUC 0.44-0.64 -- at or below chance --
    and pseudo-labels drawn from a chance mask are noise, not supervision.
    Here both polarities are built and scored, and the better one is returned
    with its audit so the caller can refuse the field outright.
    """
    from dataclasses import replace as _replace

    cfg = cfg or PriorConfig()
    # [v6.2] One Hessian pass and one structure tensor for both polarities.
    _resp = dict(zip(("bright", "dark"), ridge_response_both(gray, cfg)))
    _ang, _coh = orientation_field(gray, cfg)
    best, best_pol, best_auc = None, None, -1.0
    for pol in ("bright", "dark"):
        try:
            _c = _replace(cfg, polarity=pol)
            _m = fiber_mask(_resp[pol], gray, _c, polarity=pol)
            p = FiberPrior(_m, _ang, _coh, _resp[pol], pol)
        except Exception as exc:  # noqa: BLE001
            LOG.debug("polarity %s failed: %s", pol, exc)
            continue
        auc = float(audit_prior(p, ann, gray, cfg).get("intensity_separability_auc", float("nan")))
        if auc == auc and auc > best_auc:
            best, best_pol, best_auc = p, pol, auc
    if best is None:
        return None, "bright", float("nan")
    return best, best_pol, best_auc
