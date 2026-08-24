"""Turn the ~119 unlabelled fields into training data.

Only 11 of the 130 SEM fields carry manual measurements, and that is the
binding constraint on this project: a 25M-parameter network with 7 training
images will memorise them whatever else is fixed.  More human annotation is the
obvious answer and the expensive one.

The cheap answer is that a classical measurement pipeline already works
reasonably on these images -- ridge detection finds the fibers, and the
full-width-at-half-maximum of the intensity profile across a fiber is a
defensible thickness estimate that predates deep learning by decades.  It is
slower and more brittle than the network, and it has no notion of which sites
are ambiguous, but it does not need labels.

So: measure every unlabelled field classically, train on those pseudo-labels,
then fine-tune on the 11 human-measured fields.  The network learns fiber
appearance from 130 images and learns the *annotator's* judgement -- which
sites count, where exactly to place the chord -- from the 11 that carry it.

Two safeguards matter more than the method.

The pseudo-labels are marked
    ``annotation_confidence`` is set below 1 and ``source_csv`` records that
    the row was generated, so nothing downstream can mistake them for human
    measurement, and evaluation can exclude them.

They never enter validation or test
    A metric computed against pseudo-labels measures agreement with the
    classical pipeline, not with the specimen.  Held-out sets are human-labelled
    fields only; :func:`build_pseudo_records` enforces this.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .fiber_prior import FiberPrior, PriorConfig, best_polarity
from .utils import angle_to_direction, get_logger, line_endpoints, wrap_deg_180

LOG = get_logger(__name__)

PSEUDO_SOURCE = "pseudo:classical_fwhm"


@dataclass
class PseudoConfig:
    min_separability_auc: float = 0.65
    #: spacing between measurement sites along the fiber skeleton, in pixels
    spacing_px: float = 14.0
    #: only measure where the structure tensor is confident about direction
    coherency_min: float = 0.45
    #: half-length of the intensity profile, in units of the initial width guess
    half_span: float = 2.5
    n_samples: int = 81
    #: plausible fiber widths; anything outside is dropped rather than clipped
    min_width_px: float = 2.5
    max_width_px: float = 60.0
    #: drop sites whose profile is not a clean single peak
    min_contrast: float = 0.12
    #: confidence written into the labels table for generated rows
    confidence: float = 0.45
    #: cap per image, to stop one dense field dominating the training set
    max_per_image: int = 3000


# --------------------------------------------------------------------------- #
def _profile_fwhm(img: np.ndarray, cx: float, cy: float, chord_deg: float,
                  guess_px: float, cfg: PseudoConfig) -> tuple[float, float]:
    """Full width at half maximum of the intensity profile across one fiber.

    Returns ``(width_px, contrast)``.  Contrast is the peak-to-baseline
    difference in normalised units, and is the quality gate: a site where the
    profile is flat has no fiber to measure, whatever the ridge filter said.
    """
    import cv2

    ts = np.linspace(-cfg.half_span, cfg.half_span, cfg.n_samples)
    ux, uy = angle_to_direction(chord_deg, 1.0)
    xs = (cx + ux * guess_px * ts).astype(np.float32)
    ys = (cy + uy * guess_px * ts).astype(np.float32)
    h, w = img.shape
    if xs.min() < 0 or ys.min() < 0 or xs.max() >= w - 1 or ys.max() >= h - 1:
        return float("nan"), 0.0
    prof = cv2.remap(img, xs.reshape(-1, 1), ys.reshape(-1, 1),
                     cv2.INTER_LINEAR).ravel()
    centre = int(np.argmin(np.abs(ts)))
    peak = float(prof[np.abs(ts) < 0.3].max())
    base = float(np.percentile(prof, 10))
    contrast = peak - base
    if contrast <= 0:
        return float("nan"), 0.0
    half = 0.5 * (peak + base)
    if prof[centre] < half:
        return float("nan"), contrast
    lo = centre
    while lo > 0 and prof[lo] > half:
        lo -= 1
    hi = centre
    while hi < ts.size - 1 and prof[hi] > half:
        hi += 1
    if lo == 0 or hi == ts.size - 1:
        return float("nan"), contrast      # profile never came back down
    return float((ts[hi] - ts[lo]) * guess_px), float(contrast)


def measure_image(gray: np.ndarray, *, image_id: str = "",
                  nm_per_pixel: float | None = None,
                  cfg: PseudoConfig | None = None,
                  prior_cfg: PriorConfig | None = None,
                  prior: FiberPrior | None = None) -> "Any":
    """Classical measurements over one field, in the labels-table schema."""
    import pandas as pd
    from skimage.morphology import skeletonize

    cfg = cfg or PseudoConfig()
    if prior is None:
        # [v4] Unlabelled fields have no annotations, so decide_polarity cannot
        # settle the question and defaults to "bright".  On several 2-* fields
        # that produced a mask at AUC 0.44-0.64 -- chance -- and pseudo-labels
        # from a chance mask are noise dressed as supervision.  Score both.
        prior, _pol, _auc = best_polarity(gray, prior_cfg or PriorConfig())
        if prior is None:
            raise ValueError("could not build a fiber prior at either polarity")
        _floor = float(getattr(cfg or PseudoConfig(), "min_separability_auc", 0.65))
        if _auc == _auc and _auc < _floor:
            # Measured and at (or below) chance: refuse.  A NaN AUC means the
            # audit could not judge -- proceed with a warning rather than
            # discarding a field on a number we do not have.
            raise ValueError(
                f"fiber prior is not separable (best AUC {_auc:.2f} < {_floor:.2f} "
                f"at polarity '{_pol}'); refusing to pseudo-label this field")
        if _auc != _auc:
            LOG.warning("separability could not be scored; using polarity '%s' "
                        "unchecked", _pol)
        else:
            LOG.info("polarity '%s' chosen (separability AUC %.2f)", _pol, _auc)
    g = gray.astype(np.float32)
    g = (g - float(np.percentile(g, 1))) / max(
        float(np.percentile(g, 99) - np.percentile(g, 1)), 1e-6)
    if prior.polarity == "dark":
        g = 1.0 - g

    skel = skeletonize(prior.mask.astype(bool))
    ys, xs = np.nonzero(skel)
    if not len(xs):
        LOG.warning("%s: no fiber skeleton found; no pseudo-labels", image_id)
        return pd.DataFrame()

    # thin the skeleton points to roughly one site every spacing_px, greedily
    order = np.random.default_rng(0).permutation(len(xs))
    taken_x: list[float] = []
    taken_y: list[float] = []
    grid: dict[tuple[int, int], list[int]] = {}
    cell = max(1.0, cfg.spacing_px)
    for i in order:
        x, y = float(xs[i]), float(ys[i])
        gx, gy = int(x // cell), int(y // cell)
        clash = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((gx + dx, gy + dy), ()):
                    if (taken_x[j] - x) ** 2 + (taken_y[j] - y) ** 2 < cfg.spacing_px ** 2:
                        clash = True
                        break
                if clash:
                    break
            if clash:
                break
        if not clash:
            grid.setdefault((gx, gy), []).append(len(taken_x))
            taken_x.append(x)
            taken_y.append(y)
        if len(taken_x) >= cfg.max_per_image:
            break

    # an initial width guess from the local mask thickness, refined by FWHM
    from scipy.ndimage import distance_transform_edt
    thickness = 2.0 * distance_transform_edt(prior.mask.astype(bool))

    rows: list[dict[str, Any]] = []
    for x, y in zip(taken_x, taken_y):
        iy, ix = int(round(y)), int(round(x))
        if prior.coherency[iy, ix] < cfg.coherency_min:
            continue
        fiber_ang = float(prior.angle_deg[iy, ix])
        chord_ang = wrap_deg_180(fiber_ang + 90.0)
        guess = float(thickness[iy, ix])
        if not np.isfinite(guess) or guess < 1.0:
            continue
        width, contrast = _profile_fwhm(g, x, y, chord_ang, max(guess, 3.0), cfg)
        if not np.isfinite(width):
            continue
        if width < cfg.min_width_px or width > cfg.max_width_px:
            continue
        if contrast < cfg.min_contrast:
            continue
        x1, y1, x2, y2 = line_endpoints(x, y, chord_ang, width, 1.0)
        rows.append({
            "image_id": image_id, "annotation_id": len(rows) + 1,
            "center_x_px": x, "center_y_px": y,
            "x1_px": x1, "y1_px": y1, "x2_px": x2, "y2_px": y2,
            "measurement_angle_deg": chord_ang,
            "local_fiber_angle_deg": fiber_ang,
            "width_px": width,
            "width_nm": width * nm_per_pixel if nm_per_pixel else np.nan,
            "nm_per_pixel": nm_per_pixel if nm_per_pixel else np.nan,
            "annotation_confidence": cfg.confidence,
            "ambiguous_crossing": False,
            "source_csv": PSEUDO_SOURCE,
            "is_negative": False,
        })
    df = pd.DataFrame(rows)
    LOG.info("%s: %d pseudo-labels from %d skeleton sites", image_id, len(df),
             len(taken_x))
    return df


# --------------------------------------------------------------------------- #
def build_pseudo_labels(image_dir: str | Path, out_csv: str | Path, *,
                        exclude_ids: Sequence[str] = (),
                        calibration: dict[str, float] | None = None,
                        cfg: PseudoConfig | None = None,
                        prior_cfg: PriorConfig | None = None,
                        limit: int | None = None) -> "Any":
    """Measure every image in ``image_dir`` that has no manual labels."""
    import pandas as pd

    from .utils import image_id_from_path, list_images, read_gray
    from .calibration import strip_footer

    exclude = {str(i) for i in exclude_ids}
    paths = [p for p in list_images(image_dir)
             if image_id_from_path(p) not in exclude]
    if limit:
        paths = paths[:limit]
    LOG.info("pseudo-labelling %d image(s); %d excluded as manually labelled",
             len(paths), len(exclude))

    frames = []
    for i, p in enumerate(paths, 1):
        image_id = image_id_from_path(p)
        try:
            gray, _row = strip_footer(read_gray(p))
            nmpp = (calibration or {}).get(image_id)
            frames.append(measure_image(gray, image_id=image_id,
                                        nm_per_pixel=nmpp, cfg=cfg,
                                        prior_cfg=prior_cfg))
        except Exception as exc:                          # pragma: no cover
            LOG.error("%s: pseudo-labelling failed (%s)", image_id, exc)
        if i % 10 == 0:
            LOG.info("  %d/%d done", i, len(paths))

    frames = [f for f in frames if len(f)]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(df):
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        LOG.info("wrote %d pseudo-labels over %d image(s) -> %s",
                 len(df), df["image_id"].nunique(), out_csv)
    return df


def validate_against_manual(pseudo: "Any", manual: "Any") -> dict[str, Any]:
    """Do the classical measurements agree with the humans where both exist?

    Run this on the 11 labelled fields before trusting the other 119.  If the
    classical width distribution is biased relative to the manual one, that bias
    is about to be baked into the pretrained weights, and the fine-tuning stage
    may not fully remove it.  A median relative error above roughly 15% means
    the pseudo-labels need their parameters adjusted first.
    """
    from .fiber_metrics import distribution_distance

    out: dict[str, Any] = {}
    common = sorted(set(pseudo["image_id"]) & set(manual["image_id"]))
    out["n_common_images"] = len(common)
    if not common:
        return out
    per = {}
    for iid in common:
        d = distribution_distance(
            manual.loc[manual["image_id"] == iid, "width_px"].to_numpy(float),
            pseudo.loc[pseudo["image_id"] == iid, "width_px"].to_numpy(float))
        per[iid] = d
    out["per_image"] = per
    rel = [v.get("median_relative_error") for v in per.values()
           if v.get("median_relative_error") is not None]
    if rel:
        out["median_relative_error_mean"] = float(np.mean(rel))
        out["median_relative_error_worst"] = float(np.max(np.abs(rel)))
        if abs(float(np.mean(rel))) > 0.15:
            LOG.error("classical pseudo-labels are biased by %.0f%% against the "
                      "manual measurements; calibrate them with "
                      "calibrate_widths() before pretraining",
                      100 * float(np.mean(rel)))
    return out


def calibrate_widths(pseudo: "Any", manual: "Any", *,
                     apply: bool = True, strict: bool = False,
                     max_relative_range: float = 0.25,
                     min_images: int = 1) -> tuple["Any", dict[str, Any]]:
    """Rescale pseudo widths onto the manual measurement scale.

    The correction is fit only from the ``manual`` rows supplied by the caller.
    That detail is important for cross-validation: an outer-test specimen must
    never be present in ``manual`` here, otherwise its answer key leaks into the
    pseudo-label teacher before the model ever sees the held-out field.

    ``strict=True`` turns the old warning-only behaviour into a publication
    gate.  The calibration then raises when too few common fields exist or when
    the per-field median-ratio range exceeds ``max_relative_range * factor``.
    """
    common = sorted(set(pseudo["image_id"].astype(str)) &
                    set(manual["image_id"].astype(str)))
    ratios: dict[str, float] = {}
    for iid in common:
        m = manual.loc[manual["image_id"].astype(str) == iid, "width_px"].to_numpy(float)
        p = pseudo.loc[pseudo["image_id"].astype(str) == iid, "width_px"].to_numpy(float)
        m, p = m[np.isfinite(m)], p[np.isfinite(p)]
        if m.size >= 20 and p.size >= 20:
            ratios[iid] = float(np.median(m) / max(np.median(p), 1e-6))

    info: dict[str, Any] = {
        "per_image_ratio": ratios,
        "n_images": len(ratios),
        "strict": bool(strict),
        "max_relative_range": float(max_relative_range),
        "min_images": int(min_images),
    }
    if len(ratios) < int(min_images):
        msg = (f"pseudo-width calibration has {len(ratios)} usable common field(s), "
               f"but {int(min_images)} are required")
        if strict:
            raise RuntimeError(msg)
        LOG.warning("%s; pseudo widths left unscaled", msg)
        info["factor"] = 1.0
        info["passed"] = False
        info["failure"] = msg
        return pseudo, info

    vals = np.array(list(ratios.values()), float)
    factor = float(np.median(vals))
    spread = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    rel_range = float((vals.max() - vals.min()) / max(abs(factor), 1e-12))
    info.update({
        "factor": factor,
        "spread": spread,
        "min": float(vals.min()),
        "max": float(vals.max()),
        "relative_range": rel_range,
        "passed": True,
    })
    if vals.size > 1 and rel_range > float(max_relative_range):
        msg = ("the FWHM-to-manual ratio varies from %.3f to %.3f "
               "(range %.1f%% of the pooled factor %.3f), above the %.1f%% gate. "
               "A single multiplicative calibration is not transferable."
               % (vals.min(), vals.max(), 100 * rel_range, factor,
                  100 * float(max_relative_range)))
        info["passed"] = False
        info["failure"] = msg
        if strict:
            raise RuntimeError(msg)
        LOG.error("%s", msg)

    LOG.info("pseudo-label width calibration factor %.3f from %d field(s); "
             "relative range %.1f%%", factor, vals.size, 100 * rel_range)
    if not apply:
        return pseudo, info

    out = pseudo.copy()
    out["width_px"] = out["width_px"] * factor
    if "width_nm" in out.columns:
        out["width_nm"] = out["width_nm"] * factor
    for a, b in (("x1_px", "x2_px"), ("y1_px", "y2_px")):
        if a in out.columns and b in out.columns:
            mid = 0.5 * (out[a] + out[b])
            out[a] = mid + (out[a] - mid) * factor
            out[b] = mid + (out[b] - mid) * factor
    out["calibration_factor"] = factor
    return out, info

