"""Fiber-level detection metrics, and why chord-to-chord recall is misleading.

The first evaluation of this project reported 1.1% recall and read as a total
failure.  Part of that was a real bug in the target encoder.  The rest is a
measurement artefact worth stating plainly, because it will otherwise keep
producing pessimistic numbers even after the model is good.

A manual CSV contains a few hundred chords drawn at positions the annotator
happened to choose along the fiber network.  Nothing about those positions is
canonical: a second annotator would measure the same specimen at different
points and agree on the *widths*, not the coordinates.  Scoring a prediction as
correct only when it lands within 12 px of one particular chosen point
therefore measures agreement with an arbitrary sampling, not measurement skill.

Three metrics here say what is actually wanted:

``fiber_recall``
    Of the fibers a human measured, how many did the model measure *somewhere*
    on the same stretch, at a compatible orientation?  This is the honest
    detection number.

``skeleton_coverage``
    What fraction of the visible fiber network received at least one
    measurement?  This is the completeness number, and it needs no manual
    labels at all, so it can be computed on unlabelled fields.

``distribution_distance``
    How far apart are the two width *distributions*?  This is the quantity a
    materials paper reports, and the one the thesis claim rests on.

Chord-level precision/recall is still computed and still reported -- it is the
right metric for asking "did the model find this exact site" -- but it should
not be the headline.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .utils import angular_diff_180, get_logger

LOG = get_logger(__name__)


# --------------------------------------------------------------------------- #
def fiber_level_recall(gt: "Any", pred: "Any", *,
                       distance_scale: float = 1.5,
                       min_distance_px: float = 8.0,
                       max_angle_deg: float = 30.0) -> dict[str, float]:
    """Fraction of ground-truth sites with a compatible prediction nearby.

    A prediction counts for a GT chord when it sits within
    ``max(min_distance_px, distance_scale * gt_width)`` of it *and* the two
    fiber orientations agree to ``max_angle_deg``.  The orientation test is what
    keeps this from degenerating into "is there any prediction in the
    neighbourhood": in a dense network two crossing fibers pass within a few
    pixels of each other, and only one of them is the measured one.

    Unlike the one-to-one matching used for chord metrics, this is deliberately
    many-to-one.  Several predictions along one fiber all confirm that the fiber
    was found; that is not double counting, because the question being asked is
    about the fiber, not about the point.
    """
    if not len(gt):
        return {"n_gt": 0, "fiber_recall": float("nan")}
    if not len(pred):
        return {"n_gt": int(len(gt)), "n_pred": 0, "fiber_recall": 0.0}

    gx = gt["center_x_px"].to_numpy(float)
    gy = gt["center_y_px"].to_numpy(float)
    gw = gt["width_px"].to_numpy(float)
    ga = _fiber_angle(gt)

    px = pred["center_x_px"].to_numpy(float)
    py = pred["center_y_px"].to_numpy(float)
    pa = _fiber_angle(pred)

    tol = np.maximum(min_distance_px, distance_scale * np.nan_to_num(gw, nan=0.0))
    hit = np.zeros(len(gt), bool)
    # chunked to keep the pairwise distance matrix bounded on large fields
    step = max(1, int(4_000_000 // max(1, len(pred))))
    for lo in range(0, len(gt), step):
        hi = min(len(gt), lo + step)
        d = np.hypot(gx[lo:hi, None] - px[None, :], gy[lo:hi, None] - py[None, :])
        close = d <= tol[lo:hi, None]
        if np.isfinite(ga[lo:hi]).any() and np.isfinite(pa).any():
            da = angular_diff_180(ga[lo:hi, None], pa[None, :])
            aligned = np.asarray(da, float) <= max_angle_deg
            aligned |= ~np.isfinite(ga[lo:hi, None]) | ~np.isfinite(pa[None, :])
            close &= aligned
        hit[lo:hi] = close.any(axis=1)
    return {"n_gt": int(len(gt)), "n_pred": int(len(pred)),
            "fiber_recall": float(hit.mean()),
            "tolerance_px_median": float(np.median(tol))}


def _fiber_angle(df: "Any") -> np.ndarray:
    """Fiber direction for either table, whichever column carries it."""
    if "local_fiber_angle_deg" in df.columns:
        a = df["local_fiber_angle_deg"].to_numpy(float)
        if np.isfinite(a).any():
            return a
    if "measurement_angle_deg" in df.columns:
        return df["measurement_angle_deg"].to_numpy(float) - 90.0
    return np.full(len(df), np.nan)


# --------------------------------------------------------------------------- #
def skeleton_coverage(pred: "Any", fiber_mask: np.ndarray, *,
                      radius_px: float = 12.0) -> dict[str, float]:
    """How much of the visible fiber network got measured.

    Needs no manual labels, so it is the one completeness number available on
    the ~119 unlabelled fields.  The skeleton is used rather than the mask so
    that thick fibers do not count more than thin ones.
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    if fiber_mask is None or not fiber_mask.any():
        return {"skeleton_px": 0, "coverage": float("nan")}
    skel = skeletonize(fiber_mask.astype(bool))
    n_skel = int(skel.sum())
    if not len(pred) or n_skel == 0:
        return {"skeleton_px": n_skel, "coverage": 0.0}

    h, w = fiber_mask.shape
    seeds = np.ones((h, w), bool)
    xs = np.clip(pred["center_x_px"].to_numpy(float).round().astype(int), 0, w - 1)
    ys = np.clip(pred["center_y_px"].to_numpy(float).round().astype(int), 0, h - 1)
    seeds[ys, xs] = False
    dist = distance_transform_edt(seeds)
    return {"skeleton_px": n_skel,
            "coverage": float((dist[skel] <= radius_px).mean()),
            "median_distance_to_nearest_pred_px": float(np.median(dist[skel]))}


# --------------------------------------------------------------------------- #
def distribution_distance(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Distance between two width distributions, in the units they came in.

    Wasserstein is reported alongside KS because it is the one that carries
    physical meaning: it is the average distance a measurement would have to
    move to turn one distribution into the other, so it reads in pixels or
    nanometres rather than as an abstract statistic.  KS is scale-free and
    over-reacts to a shift in the mode; on a few hundred samples it will call
    two visually identical histograms different.
    """
    from scipy import stats

    gt = np.asarray(gt, float)
    pred = np.asarray(pred, float)
    gt, pred = gt[np.isfinite(gt)], pred[np.isfinite(pred)]
    if gt.size < 3 or pred.size < 3:
        return {"n_gt": int(gt.size), "n_pred": int(pred.size)}
    ks, p = stats.ks_2samp(gt, pred)
    w1 = float(stats.wasserstein_distance(gt, pred))
    med_gt, med_pred = float(np.median(gt)), float(np.median(pred))
    iqr_gt = float(np.subtract(*np.percentile(gt, [75, 25])))
    iqr_pred = float(np.subtract(*np.percentile(pred, [75, 25])))
    return {
        "n_gt": int(gt.size), "n_pred": int(pred.size),
        "wasserstein": w1,
        "wasserstein_relative": w1 / max(abs(med_gt), 1e-6),
        "ks_statistic": float(ks), "ks_pvalue": float(p),
        "median_gt": med_gt, "median_pred": med_pred,
        "median_error": med_pred - med_gt,
        "median_relative_error": (med_pred - med_gt) / max(abs(med_gt), 1e-6),
        "iqr_gt": iqr_gt, "iqr_pred": iqr_pred,
        "iqr_ratio": iqr_pred / max(iqr_gt, 1e-6),
    }


# --------------------------------------------------------------------------- #
def headline(per_image: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The three numbers that should appear in the thesis table.

    Reported as mean and spread across images, never pooled into one figure: a
    pooled number on a dataset this small is carried by whichever field happens
    to have the most annotations.
    """
    def gather(path: tuple[str, ...]) -> list[float]:
        vals = []
        for res in per_image.values():
            node: Any = res
            for k in path:
                node = (node or {}).get(k) if isinstance(node, dict) else None
            if isinstance(node, (int, float)) and np.isfinite(node):
                vals.append(float(node))
        return vals

    out: dict[str, Any] = {"n_images": len(per_image)}
    for name, path in (
            ("fiber_recall", ("fiber_level", "fiber_recall")),
            ("skeleton_coverage", ("coverage", "coverage")),
            ("width_median_relative_error",
             ("distribution_px", "median_relative_error")),
            ("width_wasserstein_px", ("distribution_px", "wasserstein")),
            ("chord_recall", ("detection", "recall")),
            ("chord_precision", ("detection", "precision")),
            ("orientation_median_error_deg",
             ("orientation", "median_abs_angle_error_deg")),
    ):
        v = gather(path)
        if v:
            out[name] = {"mean": float(np.mean(v)),
                         "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                         "min": float(np.min(v)), "max": float(np.max(v)),
                         "n": len(v)}
    return out
