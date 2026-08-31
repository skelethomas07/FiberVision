"""Evaluation metrics (v7).

Levels:
* distribution -- number-weighted fibre widths (predicted fibre table vs the
  manual chords rolled up the SAME way): median, IQR, SD, p90, p95, their
  ratios, 1-Wasserstein (px and relative), KS;
* matched sites -- Hungarian one-to-one matching of predicted sites to manual
  chords by centre distance + fibre angle; MAE, median relative error, bias,
  and bias as a function of manual width (binned);
* fibre-level recall -- did a compatible prediction land on the measured
  fibre stretch (many-to-one, angle-gated);
* per specimen -- fields of one specimen are averaged before anything is
  pooled; bootstrap CIs are over specimen groups and are refused when there
  are too few groups.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .coords import angular_diff_180


# --------------------------------------------------------------------------- #
def distribution_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    from scipy import stats

    gt = np.asarray(gt, float)
    pred = np.asarray(pred, float)
    gt, pred = gt[np.isfinite(gt)], pred[np.isfinite(pred)]
    if gt.size < 3 or pred.size < 3:
        return {"n_gt": int(gt.size), "n_pred": int(pred.size)}
    q = lambda a, p: float(np.percentile(a, p))  # noqa: E731
    out = {"n_gt": int(gt.size), "n_pred": int(pred.size),
           "gt_median": q(gt, 50), "pred_median": q(pred, 50),
           "gt_iqr": q(gt, 75) - q(gt, 25), "pred_iqr": q(pred, 75) - q(pred, 25),
           "gt_sd": float(gt.std(ddof=1)), "pred_sd": float(pred.std(ddof=1)),
           "gt_p90": q(gt, 90), "pred_p90": q(pred, 90),
           "gt_p95": q(gt, 95), "pred_p95": q(pred, 95),
           "wasserstein": float(stats.wasserstein_distance(gt, pred))}
    out["wasserstein_relative"] = out["wasserstein"] / max(out["gt_median"], 1e-6)
    out["median_relative_error"] = (out["pred_median"] - out["gt_median"]) / max(out["gt_median"], 1e-6)
    out["sd_ratio"] = out["pred_sd"] / max(out["gt_sd"], 1e-6)
    out["iqr_ratio"] = out["pred_iqr"] / max(out["gt_iqr"], 1e-6)
    out["p90_ratio"] = out["pred_p90"] / max(out["gt_p90"], 1e-6)
    out["p95_ratio"] = out["pred_p95"] / max(out["gt_p95"], 1e-6)
    ks = stats.ks_2samp(gt, pred)
    out["ks_statistic"], out["ks_pvalue"] = float(ks.statistic), float(ks.pvalue)
    return out


# --------------------------------------------------------------------------- #
def match_sites(gt: "Any", pred: "Any", *, max_distance_scale: float = 1.5,
                min_distance_px: float = 8.0, max_angle_deg: float = 30.0,
                angle_weight: float = 0.1) -> "Any":
    """One-to-one Hungarian assignment; returns a DataFrame of matched pairs."""
    import pandas as pd
    from scipy.optimize import linear_sum_assignment

    cols = ["gt_index", "pred_index", "distance", "gt_width", "pred_width", "gt_angle",
            "pred_angle", "angle_error"]
    if not len(gt) or not len(pred):
        return pd.DataFrame(columns=cols)
    g, p = gt.reset_index(drop=True), pred.reset_index(drop=True)
    gxy = g[["center_x_px", "center_y_px"]].to_numpy(float)
    pxy = p[["center_x_px", "center_y_px"]].to_numpy(float)
    gw = g["width_px"].to_numpy(float)
    dist = np.linalg.norm(gxy[:, None, :] - pxy[None, :, :], axis=-1)
    ga = g["fiber_angle_raster_deg"].to_numpy(float)[:, None]
    pa = p["fiber_angle_raster_deg"].to_numpy(float)[None, :]
    ang = np.asarray(angular_diff_180(ga, pa), float)
    tol = np.maximum(min_distance_px, max_distance_scale * np.nan_to_num(gw))[:, None]
    forbidden = (dist > tol) | (ang > max_angle_deg)
    cost = dist + angle_weight * ang
    big = float(cost[~forbidden].max() if (~forbidden).any() else 1.0) * 1e3 + 1e6
    ri, ci = linear_sum_assignment(np.where(forbidden, big, cost))
    rows = [{"gt_index": int(r), "pred_index": int(c), "distance": float(dist[r, c]),
             "gt_width": float(gw[r]), "pred_width": float(p["width_px"].iloc[c]),
             "gt_angle": float(ga[r, 0]), "pred_angle": float(pa[0, c]),
             "angle_error": float(ang[r, c])}
            for r, c in zip(ri, ci) if not forbidden[r, c]]
    return pd.DataFrame(rows, columns=cols)


def matched_site_metrics(matches: "Any", n_gt: int, n_pred: int,
                         width_bins: Sequence[float] = (0, 6, 9, 13, 18, 25, 1e9)) -> dict[str, Any]:
    out: dict[str, Any] = {"n_matched": int(len(matches)), "n_gt": int(n_gt), "n_pred": int(n_pred),
                           "matched_fraction_of_gt": float(len(matches) / max(n_gt, 1))}
    if not len(matches):
        return out
    g = matches["gt_width"].to_numpy(float)
    p = matches["pred_width"].to_numpy(float)
    err = p - g
    rel = err / np.clip(g, 1e-6, None)
    out.update({"mae_px": float(np.abs(err).mean()), "median_abs_error_px": float(np.median(np.abs(err))),
                "median_relative_error": float(np.median(rel)),
                "median_abs_relative_error": float(np.median(np.abs(rel))),
                "bias_px": float(err.mean()), "within_10pct": float((np.abs(rel) <= 0.1).mean()),
                "within_20pct": float((np.abs(rel) <= 0.2).mean()),
                "angle_median_abs_error_deg": float(np.median(matches["angle_error"]))})
    if g.size > 2 and np.ptp(g) > 0 and np.ptp(p) > 0:
        out["pearson_r"] = float(np.corrcoef(g, p)[0, 1])
    bins = list(width_bins)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = (g >= lo) & (g < hi)
        if s.sum() >= 3:
            rows.append({"width_lo": lo, "width_hi": hi, "n": int(s.sum()),
                         "bias_px": float(err[s].mean()),
                         "median_relative_error": float(np.median(rel[s]))})
    out["bias_vs_width"] = rows
    return out


def fiber_level_recall(gt: "Any", pred: "Any", *, distance_scale: float = 1.5,
                       min_distance_px: float = 8.0, max_angle_deg: float = 30.0) -> dict[str, float]:
    if not len(gt):
        return {"n_gt": 0, "fiber_recall": float("nan")}
    if not len(pred):
        return {"n_gt": int(len(gt)), "n_pred": 0, "fiber_recall": 0.0}
    gx, gy = gt["center_x_px"].to_numpy(float), gt["center_y_px"].to_numpy(float)
    gw, ga = gt["width_px"].to_numpy(float), gt["fiber_angle_raster_deg"].to_numpy(float)
    px, py = pred["center_x_px"].to_numpy(float), pred["center_y_px"].to_numpy(float)
    pa = pred["fiber_angle_raster_deg"].to_numpy(float)
    tol = np.maximum(min_distance_px, distance_scale * np.nan_to_num(gw))
    hit = np.zeros(len(gt), bool)
    step = max(1, int(4_000_000 // max(1, len(pred))))
    for lo in range(0, len(gt), step):
        hi = min(len(gt), lo + step)
        d = np.hypot(gx[lo:hi, None] - px[None, :], gy[lo:hi, None] - py[None, :])
        close = d <= tol[lo:hi, None]
        da = np.asarray(angular_diff_180(ga[lo:hi, None], pa[None, :]), float)
        close &= (da <= max_angle_deg) | ~np.isfinite(da)
        hit[lo:hi] = close.any(axis=1)
    return {"n_gt": int(len(gt)), "n_pred": int(len(pred)), "fiber_recall": float(hit.mean())}


def skeleton_coverage(pred: "Any", fibre_mask: np.ndarray, *, radius_px: float = 12.0) -> dict[str, float]:
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    if fibre_mask is None or not fibre_mask.any():
        return {"skeleton_px": 0, "coverage": float("nan")}
    skel = skeletonize(fibre_mask.astype(bool))
    n = int(skel.sum())
    if not len(pred) or n == 0:
        return {"skeleton_px": n, "coverage": 0.0}
    h, w = fibre_mask.shape
    seeds = np.ones((h, w), bool)
    xs = np.clip(pred["center_x_px"].to_numpy(float).round().astype(int), 0, w - 1)
    ys = np.clip(pred["center_y_px"].to_numpy(float).round().astype(int), 0, h - 1)
    seeds[ys, xs] = False
    dist = distance_transform_edt(seeds)
    return {"skeleton_px": n, "coverage": float((dist[skel] <= radius_px).mean())}


# --------------------------------------------------------------------------- #
def _get(d: dict[str, Any], path: Sequence[str]):
    node: Any = d
    for k in path:
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node if isinstance(node, (int, float)) and np.isfinite(node) else None


HEADLINE_PATHS: dict[str, tuple[str, ...]] = {
    "fibre_median_relative_error": ("distribution_fibre_px", "median_relative_error"),
    "fibre_sd_ratio": ("distribution_fibre_px", "sd_ratio"),
    "fibre_iqr_ratio": ("distribution_fibre_px", "iqr_ratio"),
    "fibre_p90_ratio": ("distribution_fibre_px", "p90_ratio"),
    "fibre_p95_ratio": ("distribution_fibre_px", "p95_ratio"),
    "fibre_wasserstein_relative": ("distribution_fibre_px", "wasserstein_relative"),
    "site_mae_px": ("matched_sites", "mae_px"),
    "site_median_abs_relative_error": ("matched_sites", "median_abs_relative_error"),
    "site_bias_px": ("matched_sites", "bias_px"),
    "fiber_recall": ("fiber_level", "fiber_recall"),
    "skeleton_coverage": ("coverage", "coverage"),
    "orientation_median_abs_error_deg": ("orientation_sites", "median_abs_error_deg"),
    "order_parameter_S_pred": ("orientation", "S_pred_fibre"),
    "order_parameter_S_gt": ("orientation", "S_gt_fibre"),
    "unassigned_fraction": ("roll_up", "unassigned_fraction"),
}


def aggregate_by_specimen(per_field: dict[str, dict[str, Any]], specimen_of: dict[str, str],
                          *, n_boot: int = 1000, min_groups_for_ci: int = 5, seed: int = 0
                          ) -> dict[str, Any]:
    """Field metrics -> per-specimen means -> mean over specimens with bootstrap CI."""
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"n_fields": len(per_field),
                           "n_specimens": len(set(specimen_of.get(f, f) for f in per_field)),
                           "metrics": {}}
    warn = out["n_specimens"] < min_groups_for_ci
    out["ci_warning"] = (f"only {out['n_specimens']} independent specimen group(s): bootstrap "
                         f"confidence intervals need >= {min_groups_for_ci}; per-specimen values "
                         "are reported individually instead") if warn else None
    for name, path in HEADLINE_PATHS.items():
        by_spec: dict[str, list[float]] = {}
        for f, m in per_field.items():
            v = _get(m, path)
            if v is not None:
                by_spec.setdefault(specimen_of.get(f, f), []).append(float(v))
        if not by_spec:
            continue
        spec_means = {s: float(np.mean(v)) for s, v in by_spec.items()}
        vals = np.array(list(spec_means.values()))
        entry: dict[str, Any] = {"per_specimen": spec_means, "mean_over_specimens": float(vals.mean()),
                                 "sd_over_specimens": float(vals.std(ddof=1)) if vals.size > 1 else None,
                                 "n_specimens": int(vals.size),
                                 "per_field": {f: _get(m, path) for f, m in per_field.items()
                                               if _get(m, path) is not None}}
        if not warn and vals.size >= 2 and n_boot > 0:
            boots = [float(np.mean(rng.choice(vals, size=vals.size, replace=True)))
                     for _ in range(int(n_boot))]
            entry["ci95_over_specimens"] = [float(np.percentile(boots, 2.5)),
                                            float(np.percentile(boots, 97.5))]
        out["metrics"][name] = entry
    return out
