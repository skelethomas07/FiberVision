"""Consolidated label schema for v7 and its validation.

All coordinates are raster pixels of the ORIGINAL (footer-stripped) image.
All angles are raster degrees in [-90, 90).  Provenance columns keep what the
export said, so a value can always be traced back.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .coords import (IMAGEJ, RASTER, angular_diff_180, fiber_angle_from_measurement,
                     measurement_angle_from_endpoints, to_raster)

LABEL_COLUMNS = [
    "image_id", "annotation_id",
    "center_x_px", "center_y_px", "x1_px", "y1_px", "x2_px", "y2_px",
    "measurement_angle_raster_deg", "fiber_angle_raster_deg",
    "width_px", "width_nm", "nm_per_pixel", "calibration_status", "calibration_valid",
    "source_angle_deg", "angle_source_convention", "imagej_angle_deg",
    "angle_convention_residual_deg",
    "source_length", "source_length_units", "annotator_units_per_px", "width_px_drawn",
    "tensor_fiber_angle_raster_deg", "orientation_coherence", "chord_vs_tensor_deg",
    "annotation_confidence", "ambiguous_crossing", "is_negative",
    "extraction_path", "source_csv",
]

REQUIRED = ("image_id", "annotation_id", "center_x_px", "center_y_px", "width_px",
            "measurement_angle_raster_deg", "fiber_angle_raster_deg")


def empty_labels() -> "Any":
    import pandas as pd

    return pd.DataFrame(columns=LABEL_COLUMNS)


def ensure_schema(df: "Any") -> "Any":
    import pandas as pd

    out = df.copy()
    for c in LABEL_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    return out[LABEL_COLUMNS + [c for c in out.columns if c not in LABEL_COLUMNS]]


def validate_labels(df: "Any", *, max_residual_deg: float = 2.0) -> dict[str, Any]:
    """Machine-readable consistency report; raises on schema violations."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"label table is missing required columns: {missing}")
    rep: dict[str, Any] = {"n": int(len(df)), "problems": []}
    if not len(df):
        return rep
    w = df["width_px"].to_numpy(np.float64)
    bad_w = ~np.isfinite(w) | (w <= 0)
    if bad_w.any():
        rep["problems"].append({"code": "nonpositive_width", "n": int(bad_w.sum())})
    if {"x1_px", "y1_px", "x2_px", "y2_px"} <= set(df.columns):
        ep = measurement_angle_from_endpoints(df["x1_px"], df["y1_px"], df["x2_px"], df["y2_px"])
        d = angular_diff_180(ep, df["measurement_angle_raster_deg"].to_numpy(np.float64))
        d = d[np.isfinite(d)]
        if d.size and float(np.nanmax(d)) > max_residual_deg:
            rep["problems"].append({"code": "endpoints_disagree_with_angle",
                                    "max_deg": float(np.nanmax(d))})
        L = np.hypot(df["x2_px"] - df["x1_px"], df["y2_px"] - df["y1_px"]).to_numpy(np.float64)
        rel = np.abs(L - w) / np.clip(w, 1e-6, None)
        rel = rel[np.isfinite(rel)]
        if rel.size and float(np.nanmax(rel)) > 0.05:
            rep["problems"].append({"code": "endpoint_length_disagrees_with_width",
                                    "max_rel": float(np.nanmax(rel))})
    fa = fiber_angle_from_measurement(df["measurement_angle_raster_deg"].to_numpy(np.float64))
    d2 = angular_diff_180(fa, df["fiber_angle_raster_deg"].to_numpy(np.float64))
    d2 = d2[np.isfinite(d2)]
    if d2.size and float(np.nanmax(d2)) > 1e-6:
        rep["problems"].append({"code": "fiber_angle_not_perpendicular_to_chord",
                                "max_deg": float(np.nanmax(d2))})
    if "angle_source_convention" in df.columns:
        conv = set(df["angle_source_convention"].dropna().astype(str).unique())
        bad = conv - {RASTER, IMAGEJ}
        if bad:
            rep["problems"].append({"code": "unknown_angle_convention", "values": sorted(bad)})
    rep["ok"] = not rep["problems"]
    return rep


def upgrade_legacy_labels(df: "Any", *, angle_source_convention: str = IMAGEJ) -> "Any":
    """Bring a v6 ``labels.csv`` onto the v7 schema (explicit convention required)."""
    from .coords import standardize_label_table

    out = standardize_label_table(df, angle_source_convention=angle_source_convention)
    out["source_angle_deg"] = (df["measurement_angle_deg"].to_numpy(np.float64)
                               if "measurement_angle_deg" in df.columns else np.nan)
    # v6 tables synthesised endpoints with a per-field fitted sign; regenerate
    # them in the raster frame from (centre, width, raster angle) wherever they
    # are missing so the endpoint geometry and the angle columns agree.
    from .coords import chord_endpoints

    need = np.ones(len(out), bool)
    if {"x1_px", "y1_px", "x2_px", "y2_px"} <= set(out.columns):
        ep = out[["x1_px", "y1_px", "x2_px", "y2_px"]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
        need = ~np.isfinite(ep).all(axis=1)
    if need.any():
        width_col = "width_px_drawn" if "width_px_drawn" in out.columns else "width_px"
        length = pd.to_numeric(out[width_col], errors="coerce").to_numpy(np.float64)
        x1, y1, x2, y2 = chord_endpoints(pd.to_numeric(out["center_x_px"], errors="coerce").to_numpy(np.float64),
                                         pd.to_numeric(out["center_y_px"], errors="coerce").to_numpy(np.float64),
                                         out["measurement_angle_raster_deg"].to_numpy(np.float64), length)
        for col, val in (("x1_px", x1), ("y1_px", y1), ("x2_px", x2), ("y2_px", y2)):
            cur = pd.to_numeric(out[col], errors="coerce").to_numpy(np.float64) if col in out.columns \
                else np.full(len(out), np.nan)
            out[col] = np.where(need, val, cur)
        if "endpoint_source" not in out.columns:
            out["endpoint_source"] = np.where(need, "regenerated_from_angle_v7", "table")
    if "local_fiber_angle_deg" in out.columns:
        out = out.drop(columns=["local_fiber_angle_deg"])
    if "measurement_angle_deg" in out.columns:
        out = out.drop(columns=["measurement_angle_deg"])
    out["extraction_path"] = out.get("extraction_path", "legacy_v6_upgraded")
    return ensure_schema(out)
