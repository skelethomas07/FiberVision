"""Evaluation metrics: detection, thickness, orientation, calibration, distribution.

Everything here consumes the one-to-one matching produced by
:mod:`matching`, so precision, recall and the error statistics all refer to the
same pairing.  Metrics are reported per image and pooled; the per-image view is
what tells you whether a good pooled number is carried by one easy field.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .utils import angular_diff_180, get_logger

LOG = get_logger(__name__)


# --------------------------------------------------------------------------- #
def detection_metrics(n_gt: int, n_pred: int, n_matched: int) -> dict[str, float]:
    precision = n_matched / n_pred if n_pred else 0.0
    recall = n_matched / n_gt if n_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"n_gt": n_gt, "n_pred": n_pred, "n_matched": n_matched,
            "precision": precision, "recall": recall, "f1": f1}


def detection_rate_by_tolerance(matches: "Any", n_gt: int,
                                tolerances: tuple[float, ...] = (2, 4, 6, 8, 12, 16)
                                ) -> dict[str, float]:
    out: dict[str, float] = {}
    d = matches["distance"].to_numpy() if len(matches) else np.array([])
    for t in tolerances:
        out[f"recall@{t:g}px"] = float((d <= t).sum() / n_gt) if n_gt else 0.0
    return out


def average_precision(pred: "Any", matches: "Any", n_gt: int) -> float:
    """AP from the confidence-ranked precision-recall curve."""
    if not len(pred) or not n_gt:
        return 0.0
    matched_ids = set(matches["pred_index"].tolist()) if len(matches) else set()
    order = np.argsort(-pred["confidence"].to_numpy()) if "confidence" in pred \
        else np.arange(len(pred))
    tp = np.array([1.0 if i in matched_ids else 0.0 for i in order])
    ctp = np.cumsum(tp)
    precision = ctp / np.arange(1, len(tp) + 1)
    recall = ctp / n_gt
    ap, prev_r = 0.0, 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_r)
        prev_r = r
    return float(ap)


# --------------------------------------------------------------------------- #
def thickness_metrics(gt: np.ndarray, pred: np.ndarray, *, unit: str = "px"
                      ) -> dict[str, float]:
    gt = np.asarray(gt, float)
    pred = np.asarray(pred, float)
    ok = np.isfinite(gt) & np.isfinite(pred)
    gt, pred = gt[ok], pred[ok]
    if gt.size == 0:
        return {f"n_{unit}": 0}
    from scipy import stats

    err = pred - gt
    abs_err = np.abs(err)
    rel = abs_err / np.clip(np.abs(gt), 1e-6, None)
    out = {
        f"n": int(gt.size),
        f"mae_{unit}": float(abs_err.mean()),
        f"medae_{unit}": float(np.median(abs_err)),
        f"rmse_{unit}": float(np.sqrt((err ** 2).mean())),
        f"bias_{unit}": float(err.mean()),
        "mean_relative_error": float(rel.mean()),
        "within_5pct": float((rel <= 0.05).mean()),
        "within_10pct": float((rel <= 0.10).mean()),
        "within_20pct": float((rel <= 0.20).mean()),
    }
    if gt.size > 2 and np.ptp(gt) > 0 and np.ptp(pred) > 0:
        out["pearson_r"] = float(stats.pearsonr(gt, pred)[0])
        out["spearman_r"] = float(stats.spearmanr(gt, pred)[0])
        ss_res = float(((gt - pred) ** 2).sum())
        ss_tot = float(((gt - gt.mean()) ** 2).sum())
        out["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        sd = float(err.std(ddof=1))
        out["bland_altman_bias"] = float(err.mean())
        out["bland_altman_loa_low"] = float(err.mean() - 1.96 * sd)
        out["bland_altman_loa_high"] = float(err.mean() + 1.96 * sd)
    return out


def orientation_metrics(gt_deg: np.ndarray, pred_deg: np.ndarray) -> dict[str, float]:
    gt = np.asarray(gt_deg, float)
    pred = np.asarray(pred_deg, float)
    ok = np.isfinite(gt) & np.isfinite(pred)
    if not ok.any():
        return {"n": 0}
    err = np.asarray(angular_diff_180(gt[ok], pred[ok]), float)
    return {
        "n": int(err.size),
        "mean_abs_angle_error_deg": float(err.mean()),
        "median_abs_angle_error_deg": float(np.median(err)),
        "within_5deg": float((err <= 5).mean()),
        "within_10deg": float((err <= 10).mean()),
        "within_20deg": float((err <= 20).mean()),
    }


# --------------------------------------------------------------------------- #
def calibration_metrics(confidence: np.ndarray, error: np.ndarray, *,
                        n_bins: int = 10) -> dict[str, Any]:
    """Reliability of the confidence score against realised error."""
    conf = np.asarray(confidence, float)
    err = np.asarray(error, float)
    ok = np.isfinite(conf) & np.isfinite(err)
    conf, err = conf[ok], err[ok]
    if conf.size < 5:
        return {"n": int(conf.size)}
    from scipy import stats

    edges = np.linspace(conf.min(), conf.max() + 1e-9, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf >= lo) & (conf < hi)
        if sel.sum() >= 3:
            bins.append({"confidence": float(conf[sel].mean()),
                         "mean_abs_error": float(np.abs(err[sel]).mean()),
                         "n": int(sel.sum())})
    out: dict[str, Any] = {"n": int(conf.size), "bins": bins}
    if conf.size > 3 and np.ptp(conf) > 0:
        out["confidence_error_spearman"] = float(stats.spearmanr(conf,
                                                                 np.abs(err))[0])
    return out


def uncertainty_metrics(sigma: np.ndarray, error: np.ndarray) -> dict[str, Any]:
    s = np.asarray(sigma, float)
    e = np.abs(np.asarray(error, float))
    ok = np.isfinite(s) & np.isfinite(e)
    if ok.sum() < 5:
        return {"n": int(ok.sum())}
    from scipy import stats
    z = (np.asarray(error, float)[ok] / np.clip(s[ok], 1e-6, None))
    return {"n": int(ok.sum()),
            "sigma_error_spearman": float(stats.spearmanr(s[ok], e[ok])[0]),
            "z_score_std": float(z.std(ddof=1)),
            "coverage_1sigma": float((np.abs(z) <= 1).mean()),
            "coverage_2sigma": float((np.abs(z) <= 2).mean())}


# --------------------------------------------------------------------------- #
def distribution_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    """Compare the two thickness *distributions*, not just the paired values.

    A model can have mediocre per-measurement error yet reproduce the population
    statistics a materials paper actually reports -- and vice versa.  Both are
    worth knowing.
    """
    from scipy import stats

    gt = np.asarray(gt, float)
    pred = np.asarray(pred, float)
    gt, pred = gt[np.isfinite(gt)], pred[np.isfinite(pred)]
    if gt.size < 3 or pred.size < 3:
        return {"n_gt": int(gt.size), "n_pred": int(pred.size)}
    ks, p = stats.ks_2samp(gt, pred)
    q = (5, 25, 50, 75, 95)
    return {
        "n_gt": int(gt.size), "n_pred": int(pred.size),
        "gt_mean": float(gt.mean()), "pred_mean": float(pred.mean()),
        "gt_median": float(np.median(gt)), "pred_median": float(np.median(pred)),
        "gt_std": float(gt.std(ddof=1)), "pred_std": float(pred.std(ddof=1)),
        "gt_percentiles": {f"p{k}": float(np.percentile(gt, k)) for k in q},
        "pred_percentiles": {f"p{k}": float(np.percentile(pred, k)) for k in q},
        "ks_statistic": float(ks), "ks_pvalue": float(p),
    }


def evaluate_image(gt: "Any", pred: "Any", matches: "Any", *,
                   nm_per_pixel: float | None = None) -> dict[str, Any]:
    """All metric families for one image."""
    out: dict[str, Any] = {"detection": detection_metrics(len(gt), len(pred),
                                                          len(matches))}
    out["detection"].update(detection_rate_by_tolerance(matches, len(gt)))
    out["detection"]["average_precision"] = average_precision(pred, matches, len(gt))

    if len(matches):
        gw = matches["gt_width"].to_numpy()
        pw = matches["pred_width"].to_numpy()
        out["thickness_px"] = thickness_metrics(gw, pw, unit="px")
        if nm_per_pixel:
            out["thickness_nm"] = thickness_metrics(gw * nm_per_pixel,
                                                    pw * nm_per_pixel, unit="nm")
        out["orientation"] = orientation_metrics(matches["gt_angle"].to_numpy(),
                                                 matches["pred_angle"].to_numpy())
        out["calibration"] = calibration_metrics(matches["confidence"].to_numpy(),
                                                 pw - gw)
        out["distribution"] = distribution_metrics(
            gt["width_px"].to_numpy(), pred["width_px"].to_numpy())
    return out


def aggregate(per_image: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pool across images, keeping the per-image spread visible."""
    pooled: dict[str, Any] = {"n_images": len(per_image)}
    if not per_image:
        return pooled
    for family in ("detection", "thickness_px", "thickness_nm", "orientation"):
        vals: dict[str, list[float]] = {}
        for res in per_image.values():
            for k, v in (res.get(family) or {}).items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    vals.setdefault(k, []).append(float(v))
        if vals:
            pooled[family] = {k: {"mean": float(np.mean(v)),
                                  "std": float(np.std(v)) if len(v) > 1 else 0.0,
                                  "min": float(np.min(v)), "max": float(np.max(v))}
                              for k, v in vals.items()}
    return pooled
