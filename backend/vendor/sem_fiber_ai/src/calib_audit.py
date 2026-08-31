"""Physical calibration audit and the pre-publication calibration gate (v7).

Rules enforced here (not merely described):

* nm/px may come ONLY from a scale bar, burned-in acquisition metadata (FOV
  text / magnification) or a manually supplied table entry.  It is never
  inferred or "repaired" from the ground-truth fibre widths.
* The annotator's own scale (CSV ``Length`` units per drawn overlay pixel) is
  a provenance quantity.  It is used for exactly one thing: converting the
  annotator's stated length back into pixels of the overlay it was drawn on.
  It is compared with the physical scale in the audit; it is never substituted
  for it.
* A field whose physical nm/px is unresolved or contradictory keeps its pixel
  measurements and gets ``NaN`` nanometres, status ``calibration_invalid``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .utils import get_logger

LOG = get_logger(__name__)

STATUS_RESOLVED = "resolved"            # physical scale + independent annotator scale agree
STATUS_SINGLE = "single_source"         # physical scale only (annotator table was in pixels)
STATUS_MANUAL = "manual"                # typed in from the acquisition log
STATUS_UNRESOLVED = "unresolved"        # nothing physical could be read
STATUS_CONTRADICTORY = "contradictory"  # physical routes or annotator scale disagree
VALID_STATUSES = (STATUS_RESOLVED, STATUS_SINGLE, STATUS_MANUAL)

PHYSICAL_SOURCES = ("override", "sidecar", "manual", "fov_text", "scale_bar",
                    "magnification")
INVALID_SOURCES = ("unknown", "fov_text_disputed", "overlay_scale_fit", "label_implied")


@dataclass
class FieldCalibration:
    image_id: str
    nm_per_px: float | None                 # value to USE (None when invalid)
    status: str
    calibration_valid: bool
    reason: str
    physical_nm_per_px: float | None        # what the physical route reported
    physical_source: str
    physical_detail: str
    annotator_implied_nm_per_px: float | None
    annotator_length_units: str
    agreement_ratio: float | None           # annotator / physical

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(v) -> bool:
    return v is not None and np.isfinite(v) and v > 0


def audit_field(image_id: str, *, physical_nm_per_px: float | None,
                physical_source: str, physical_detail: str = "",
                annotator_implied_nm_per_px: float | None = None,
                annotator_length_units: str = "unknown",
                manual_nm_per_px: float | None = None,
                tolerance: float = 0.10) -> FieldCalibration:
    """Decide the calibration status of one field from independent evidence.

    ``manual_nm_per_px`` (acquisition log, typed by a person) wins over every
    automatic route and is recorded as such.  Otherwise the physical route must
    be a recognised source, and when the annotator's table carried nanometres
    the two must agree within ``tolerance``.
    """
    ann_ok = _finite(annotator_implied_nm_per_px) and annotator_length_units in ("nm", "um")
    ratio = None
    if _finite(manual_nm_per_px):
        if ann_ok:
            ratio = float(annotator_implied_nm_per_px) / float(manual_nm_per_px)
        return FieldCalibration(
            image_id, float(manual_nm_per_px), STATUS_MANUAL, True,
            "manual_table_entry", physical_nm_per_px, physical_source,
            physical_detail, annotator_implied_nm_per_px if ann_ok else None,
            annotator_length_units, ratio)

    if physical_source in ("override", "sidecar"):
        physical_source = "manual" if physical_source == "override" else physical_source

    if physical_source not in PHYSICAL_SOURCES or not _finite(physical_nm_per_px):
        reason = ("physical_scale_unresolved" if physical_source in ("unknown", "")
                  else f"physical_scale_{physical_source}")
        if physical_source == "fov_text_disputed":
            status = STATUS_CONTRADICTORY
        else:
            status = STATUS_UNRESOLVED
        return FieldCalibration(
            image_id, None, status, False, reason, physical_nm_per_px,
            physical_source, physical_detail,
            annotator_implied_nm_per_px if ann_ok else None,
            annotator_length_units, None)

    phys = float(physical_nm_per_px)
    if ann_ok:
        ratio = float(annotator_implied_nm_per_px) / phys
        if abs(ratio - 1.0) <= tolerance:
            return FieldCalibration(
                image_id, phys, STATUS_RESOLVED, True, "physical_and_annotator_agree",
                phys, physical_source, physical_detail,
                float(annotator_implied_nm_per_px), annotator_length_units, ratio)
        return FieldCalibration(
            image_id, None, STATUS_CONTRADICTORY, False,
            f"annotator_scale_disagrees_ratio_{ratio:.3f}",
            phys, physical_source, physical_detail,
            float(annotator_implied_nm_per_px), annotator_length_units, ratio)
    return FieldCalibration(
        image_id, phys, STATUS_SINGLE, True, "physical_scale_only",
        phys, physical_source, physical_detail, None, annotator_length_units, None)


def audit_from_extraction_meta(metas: list[dict[str, Any]], *,
                               manual_table: dict[str, float] | None = None,
                               tolerance: float = 0.10) -> list[FieldCalibration]:
    """Build the audit from ``labels_meta.json`` records of the v7 extraction.

    The extraction reports the PHYSICAL route in ``meta['calibration']`` /
    ``meta['physical_nm_per_px']`` and, separately, the annotator's own scale
    solved from the drawn overlay (``meta['annotator_units_per_px']``, CSV
    units per overlay pixel).  When the table was in physical units that
    number is the annotator-implied nm/px (or um/px); it is compared with the
    physical route and NEVER used as the calibration itself.  A table whose
    lengths were pixels contributes no physical evidence.
    """
    manual_table = manual_table or {}
    out = []
    for m in metas:
        iid = str(m.get("image_id"))
        cal = m.get("calibration") or {}
        phys = m.get("physical_nm_per_px", cal.get("nm_per_pixel"))
        src = str(cal.get("source", "unknown"))
        detail = str(cal.get("detail", ""))
        units = str(m.get("length_units", "unknown"))
        upp = m.get("annotator_units_per_px")
        if upp is None and str(m.get("calibration_applied_source", "")) == "overlay_scale_fit":
            # legacy v6 meta: the overlay-fitted scale was stored as an *applied*
            # nm/px; v7 keeps it as provenance only.
            upp = m.get("nm_per_pixel_applied")
        implied, ann_units = None, "unknown"
        if units in ("nm", "um", "physical") and _finite(upp):
            upp = float(upp) * float(m.get("overlay_to_original_scale", 1.0) or 1.0)
            if units == "nm":
                implied, ann_units = upp, "nm"
            elif units == "um":
                implied, ann_units = upp * 1000.0, "nm"
            elif _finite(phys):
                # physical but unit unknown: nm or um, whichever is consistent
                cand = {"nm": upp, "um": upp * 1000.0}
                best = min(cand, key=lambda k: abs(np.log(cand[k] / float(phys))))
                implied, ann_units = cand[best], "nm"
                detail = f"{detail}; annotator table read as {best}"
            else:
                implied, ann_units = upp, "physical_unknown_unit"
        elif units == "pixels":
            ann_units = "pixels"
        manual = manual_table.get(iid)
        out.append(audit_field(iid, physical_nm_per_px=phys, physical_source=src,
                               physical_detail=detail,
                               annotator_implied_nm_per_px=implied,
                               annotator_length_units=ann_units,
                               manual_nm_per_px=manual, tolerance=tolerance))
    return out


def apply_calibration_to_labels(df: "Any", audits: list[FieldCalibration]) -> "Any":
    """Rewrite the nm columns of a label table from the audited calibration.

    ``width_px`` is left untouched (it is a pixel quantity).  ``width_nm`` and
    ``nm_per_pixel`` are set from the audited physical scale, NaN when invalid.
    """
    by_id = {a.image_id: a for a in audits}
    out = df.copy()
    nmpp = np.full(len(out), np.nan)
    status = np.array(["unaudited"] * len(out), dtype=object)
    valid = np.zeros(len(out), bool)
    ids = out["image_id"].astype(str).to_numpy()
    for iid, a in by_id.items():
        sel = ids == iid
        status[sel] = a.status
        valid[sel] = a.calibration_valid
        if a.calibration_valid:
            nmpp[sel] = a.nm_per_px
    out["nm_per_pixel"] = nmpp
    out["width_nm"] = out["width_px"].to_numpy(np.float64) * nmpp
    out["calibration_status"] = status
    out["calibration_valid"] = valid
    return out


def calibration_gate(audits: list[FieldCalibration], image_ids=None) -> dict[str, Any]:
    """Hard pre-publication gate: every field quoted in nm must be valid."""
    sel = [a for a in audits if image_ids is None or a.image_id in set(image_ids)]
    bad = [a for a in sel if not a.calibration_valid]
    return {
        "passed": len(bad) == 0 and len(sel) > 0,
        "n_fields": len(sel), "n_invalid": len(bad),
        "invalid": [{"image_id": a.image_id, "status": a.status, "reason": a.reason}
                    for a in bad],
        "rule": "nanometre summaries may only include fields with "
                "calibration_valid == True; pixel summaries are unaffected",
    }


def audit_table(audits: list[FieldCalibration]) -> "Any":
    import pandas as pd
    return pd.DataFrame([a.to_dict() for a in audits])
