"""One-to-one matching between predictions and manual measurements.

Greedy nearest-neighbour matching inflates recall: one prediction can be the
nearest neighbour of several ground-truth points, and which pairs you end up
with depends on iteration order.  We solve a global assignment with the
Hungarian algorithm so every manual measurement is matched at most once and the
result is order-independent and reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .utils import angular_diff_180, get_logger

LOG = get_logger(__name__)


@dataclass
class MatchConfig:
    max_center_distance: float = 12.0    # px
    use_angle: bool = True
    angle_weight: float = 0.15           # px of cost per degree of disagreement
    max_angle_deg: float = 45.0


def _as_id(v: "Any") -> "int | str":
    """Keep an id as an int when it is one, otherwise keep it verbatim."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return str(v)


def match_measurements(gt: "Any", pred: "Any", cfg: MatchConfig | None = None
                       ) -> "Any":
    """Hungarian assignment on centre distance (optionally angle-aware).

    Returns a DataFrame with one row per matched pair, carrying both widths and
    both angles so every downstream metric works from the same pairing.
    """
    import pandas as pd
    from scipy.optimize import linear_sum_assignment

    cfg = cfg or MatchConfig()
    cols = ["gt_index", "pred_index", "gt_id", "pred_id", "distance",
            "gt_width", "pred_width", "gt_angle", "pred_angle", "angle_error",
            "confidence", "pred_x", "pred_y", "gt_x", "gt_y",
            "gt_width_nm", "pred_width_nm"]
    if not len(gt) or not len(pred):
        return pd.DataFrame(columns=cols)

    g = gt.reset_index(drop=True)
    p = pred.reset_index(drop=True)
    gxy = g[["center_x_px", "center_y_px"]].to_numpy(float)
    pxy = p[["center_x_px", "center_y_px"]].to_numpy(float)
    dist = np.linalg.norm(gxy[:, None, :] - pxy[None, :, :], axis=-1)

    ang_err = np.zeros_like(dist)
    if cfg.use_angle and "measurement_angle_deg" in g and "measurement_angle_deg" in p:
        ga = g["measurement_angle_deg"].to_numpy(float)[:, None]
        pa = p["measurement_angle_deg"].to_numpy(float)[None, :]
        ang_err = np.asarray(angular_diff_180(ga, pa), float)

    cost = dist + (cfg.angle_weight * ang_err if cfg.use_angle else 0.0)
    forbidden = dist > cfg.max_center_distance
    if cfg.use_angle:
        forbidden |= ang_err > cfg.max_angle_deg
    big = float(cost[~forbidden].max() if (~forbidden).any() else 1.0) * 1000.0 + 1e6
    cost_padded = np.where(forbidden, big, cost)

    ri, ci = linear_sum_assignment(cost_padded)
    rows = []
    for r, c in zip(ri, ci):
        if forbidden[r, c]:
            continue
        rows.append({
            "gt_index": int(r), "pred_index": int(c),
            # [v3] ids are labels, not integers. VisionFlux writes strings like
            # "auto-r3-s0", and int() on one of those killed the whole
            # evaluation at the last step of a finished training run.
            "gt_id": _as_id(g.get("annotation_id", pd.Series(range(len(g)))).iloc[r]),
            "pred_id": _as_id(p.get("prediction_id", pd.Series(range(len(p)))).iloc[c]),
            "distance": float(dist[r, c]),
            "gt_width": float(g["width_px"].iloc[r]),
            "pred_width": float(p["width_px"].iloc[c]),
            "gt_angle": float(g["measurement_angle_deg"].iloc[r]),
            "pred_angle": float(p["measurement_angle_deg"].iloc[c]),
            "angle_error": float(ang_err[r, c]),
            "confidence": float(p["confidence"].iloc[c]) if "confidence" in p else np.nan,
            "pred_x": float(p["center_x_px"].iloc[c]),
            "pred_y": float(p["center_y_px"].iloc[c]),
            "gt_x": float(g["center_x_px"].iloc[r]),
            "gt_y": float(g["center_y_px"].iloc[r]),
            "gt_width_nm": float(g["width_nm"].iloc[r]) if "width_nm" in g else np.nan,
            "pred_width_nm": (float(p["width_nm"].iloc[c])
                              if "width_nm" in p else np.nan),
        })
    out = pd.DataFrame(rows, columns=cols)
    LOG.info("matched %d of %d manual measurements against %d predictions",
             len(out), len(g), len(p))
    return out
