"""EXPERIMENTAL thick-fibre recovery (v7) -- NOT VALIDATED.

Thick fibres (> ~18 px) are rare in the chords, so the learned width head is
weakest there.  This module reads them straight from the geometry heads: the
segmentation mask's medial axis and the distance-to-boundary field, restricted
to wide, coherent, junction-free stretches.  It is the descendant of the
v6.9-v6.11 ``thick_fiber`` / ``thick_from_maps`` experiments.

Status rules (enforced, not advisory):

* ``enabled: false`` by default; when enabled, outputs go to a SEPARATE table
  ``*_thick_EXPERIMENTAL.csv`` and are never pooled with the main sites.
* The table is labelled ``validation_status = "NOT VALIDATED"`` unless
  :func:`validate_against_gt` has been run on REAL manual chords for thick
  fibres and every threshold in ``cfg['thick_experimental']['validation']``
  passed, in which case the returned certificate (with the fields it was
  validated on) is attached.  Synthetic tests do not count.
* Earlier v6 EDT-agreement figures (e.g. "9/27 sites") were internal
  self-consistency checks and are not accuracy evidence.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .coords import chord_endpoints, measurement_angle_from_fiber, vec2_to_angle
from .skeleton import branch_structure, spaced_sites

NOT_VALIDATED = "NOT VALIDATED"


def thick_config(cfg: dict[str, Any]) -> dict[str, Any]:
    d = {"enabled": False, "min_width_px": 18.0, "max_width_px": 160.0, "spacing_px": 14.0,
         "min_coherence": 0.45, "validation": {"min_sites": 30, "min_precision": 0.7,
                                               "min_recall": 0.5, "max_median_rel_error": 0.15,
                                               "max_p90_rel_error": 0.2}}
    d.update(cfg.get("thick_experimental") or {})
    return d


def measure_thick_from_maps(maps: dict[str, np.ndarray], gray: np.ndarray, cfg: dict[str, Any], *,
                            image_id: str = "", nm_per_px: float | None = None,
                            certificate: dict[str, Any] | None = None) -> dict[str, Any]:
    import pandas as pd
    from scipy import ndimage as ndi

    from .orientation import orientation_field

    tc = thick_config(cfg)
    seg = 1.0 / (1.0 + np.exp(-maps["segment_logit"]))
    dist = np.clip(maps["dist"], 0, None)
    mask = seg >= 0.5
    cols = ["image_id", "site_id", "center_x_px", "center_y_px", "x1_px", "y1_px", "x2_px", "y2_px",
            "measurement_angle_raster_deg", "fiber_angle_raster_deg", "width_px", "width_nm",
            "coherence", "branch_id", "measurement_source", "validation_status"]
    if not mask.any():
        return {"table": pd.DataFrame(columns=cols), "summary": {"n": 0, "validation_status": NOT_VALIDATED}}
    bs = branch_structure(mask, min_branch_px=int(tc["spacing_px"]))
    dmax = ndi.maximum_filter(dist, size=5)
    wide = bs.skeleton & (2.0 * dmax >= tc["min_width_px"]) & (2.0 * dmax <= tc["max_width_px"])
    ang, coh = orientation_field(gray)
    rows = []
    if wide.any():
        sub = branch_structure(ndi.binary_dilation(wide, iterations=2) & mask,
                               min_branch_px=int(tc["spacing_px"]))
        for (y, x, lb) in spaced_sites(sub, float(tc["spacing_px"]), score=dmax):
            if not wide[y, x] or coh[y, x] < tc["min_coherence"]:
                continue
            w = float(2.0 * dmax[y, x])
            if bs.junction_dist[y, x] < 0.6 * w:
                continue
            fa = float(vec2_to_angle(maps["orient"][0, y, x], maps["orient"][1, y, x]))
            ma = float(measurement_angle_from_fiber(fa))
            x1, y1, x2, y2 = chord_endpoints(float(x), float(y), ma, w)
            rows.append({"image_id": image_id, "site_id": len(rows) + 1, "center_x_px": float(x),
                         "center_y_px": float(y), "x1_px": x1, "y1_px": y1, "x2_px": x2, "y2_px": y2,
                         "measurement_angle_raster_deg": ma, "fiber_angle_raster_deg": fa,
                         "width_px": w, "width_nm": (w * nm_per_px if nm_per_px else np.nan),
                         "coherence": float(coh[y, x]), "branch_id": int(lb),
                         "measurement_source": "thick_experimental",
                         "validation_status": (certificate or {}).get("status", NOT_VALIDATED)})
    table = pd.DataFrame(rows, columns=cols)
    summary = {"n": int(len(table)), "validation_status": (certificate or {}).get("status", NOT_VALIDATED),
               "certificate": certificate,
               "median_width_px": float(table["width_px"].median()) if len(table) else None,
               "note": "experimental; never pooled with the validated site table"}
    return {"table": table, "summary": summary}


def validate_against_gt(thick_sites: "Any", gt: "Any", cfg: dict[str, Any], *, fields: list[str],
                        synthetic: bool = False) -> dict[str, Any]:
    """Certificate for the thick path from REAL manual chords of thick fibres."""
    from .metrics import match_sites

    tc = thick_config(cfg)
    v = tc["validation"]
    if synthetic:
        return {"status": NOT_VALIDATED, "reason": "synthetic data cannot validate the thick path",
                "fields": fields}
    g = gt[gt["width_px"] >= tc["min_width_px"]]
    if len(g) < v["min_sites"]:
        return {"status": NOT_VALIDATED, "reason": f"only {len(g)} thick manual chords (< {v['min_sites']})",
                "fields": fields, "n_gt_thick": int(len(g))}
    m = match_sites(g, thick_sites, max_distance_scale=1.0, min_distance_px=10.0, max_angle_deg=30.0)
    n_pred = int(len(thick_sites))
    recall = len(m) / max(len(g), 1)
    precision = len(m) / max(n_pred, 1)
    if len(m):
        rel = np.abs(m["pred_width"] - m["gt_width"]) / m["gt_width"]
        med, p90 = float(rel.median()), float(np.percentile(rel, 90))
    else:
        med, p90 = float("nan"), float("nan")
    checks = {"min_sites": len(g) >= v["min_sites"], "min_precision": precision >= v["min_precision"],
              "min_recall": recall >= v["min_recall"],
              "max_median_rel_error": bool(np.isfinite(med) and med <= v["max_median_rel_error"]),
              "max_p90_rel_error": bool(np.isfinite(p90) and p90 <= v["max_p90_rel_error"])}
    passed = all(checks.values())
    return {"status": "VALIDATED (real GT)" if passed else NOT_VALIDATED, "passed": passed,
            "checks": checks, "n_gt_thick": int(len(g)), "n_pred": n_pred, "n_matched": int(len(m)),
            "precision": float(precision), "recall": float(recall), "median_rel_error": med,
            "p90_rel_error": p90, "fields": fields, "thresholds": v}
