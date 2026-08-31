"""[v3] Checks on the label table itself, before any of it reaches a model.

Two things in the August run were invisible to every existing audit because
every existing audit looked at one field at a time.

1. Median width in PIXELS was nearly constant across the dataset (6.4-11.7 px)
   while ``nm_per_pixel`` ranged over 1.0-5.0.  Calibrated width therefore
   tracked magnification: within specimen 3, fields 3-10 / 3-9 / 3-8 gave
   22.4 / 32.0 / 40.0 nm.  Those are three fields of one sample, so the
   physical width cannot differ by 1.8x -- either the calibrations are wrong or
   the annotator measured whatever was resolvable at each zoom.  Until that is
   settled the nm-domain regression target has a per-field scale error, and no
   architecture change touches it.

2. 45 fields resolved no pixel size at all despite a detected scale bar.  They
   are not unusable -- they just need the magnification supplied once, by hand,
   from the acquisition log.

Neither of these is something the code can decide.  What it can do is put the
number in front of you and refuse to pretend.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .utils import get_logger, save_json

LOG = get_logger(__name__)


def _specimen(image_id: str) -> str:
    s = str(image_id)
    return s.rsplit("-", 1)[0] if "-" in s else s


def width_scale_report(labels_csv: str | Path,
                       out_json: str | Path | None = None,
                       *, min_fields: int = 2,
                       tolerance: float = 0.25) -> dict[str, Any]:
    """Is calibrated width independent of magnification, within a specimen?

    For each specimen with at least ``min_fields`` calibrated fields, compare
    the spread of median width in nm against the spread of ``nm_per_pixel``.
    If the physical width is real, the first should be small while the second
    varies.  If the measurement is tracking the pixel grid instead, median
    width in *pixels* is what stays constant, and the nm spread follows the
    magnification -- the signature this function reports.

    ``tolerance`` is the fractional spread of nm widths within one specimen
    that is accepted as ordinary field-to-field variation.
    """
    import pandas as pd

    df = pd.read_csv(labels_csv)
    if "is_negative" in df.columns:
        df = df[~df["is_negative"].fillna(False).astype(bool)]
    need = {"image_id", "width_px", "nm_per_pixel"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{labels_csv} lacks {sorted(missing)}")

    per_field = (df.groupby("image_id")
                 .agg(n=("width_px", "size"),
                      med_px=("width_px", "median"),
                      nm_per_px=("nm_per_pixel", "first"))
                 .reset_index())
    per_field["med_nm"] = per_field["med_px"] * per_field["nm_per_px"]
    per_field["specimen"] = per_field["image_id"].map(_specimen)

    rows: list[dict[str, Any]] = []
    for spec, sub in per_field.groupby("specimen"):
        cal = sub.dropna(subset=["nm_per_px"])
        if len(cal) < min_fields or cal["nm_per_px"].nunique() < 2:
            continue
        nm = cal["med_nm"].to_numpy(float)
        px = cal["med_px"].to_numpy(float)
        scale = cal["nm_per_px"].to_numpy(float)
        spread = lambda a: float(a.max() / a.min()) if a.min() > 0 else float("nan")
        # correlation between the calibration and the calibrated width: if the
        # width is physical this is ~0, if it is following the pixel grid it is ~1
        r = (float(np.corrcoef(np.log(scale), np.log(nm))[0, 1])
             if len(cal) > 2 else float("nan"))
        rows.append({
            "specimen": spec,
            "n_fields": int(len(cal)),
            "nm_per_px_spread": spread(scale),
            "median_nm_spread": spread(nm),
            "median_px_spread": spread(px),
            "log_corr_scale_vs_width": r,
            "fields": {str(i): {"nm_per_px": float(s), "med_px": float(p),
                                "med_nm": float(m)}
                       for i, s, p, m in zip(cal["image_id"], scale, px, nm)},
        })

    suspect = [r for r in rows
               if r["median_nm_spread"] - 1.0 > tolerance
               and r["median_nm_spread"] > r["median_px_spread"]]
    report = {"per_field": per_field.to_dict(orient="records"),
              "per_specimen": rows,
              "suspect_specimens": [r["specimen"] for r in suspect],
              "tolerance": tolerance}
    if out_json:
        save_json(report, out_json)

    if suspect:
        LOG.error(
            "on %d specimen(s) (%s) the calibrated width varies more than the "
            "pixel width does, i.e. it follows the magnification rather than "
            "the material. Fields of one sample cannot have different physical "
            "fiber widths, so either nm_per_pixel is wrong on some of them or "
            "the annotator measured at the resolution limit. Fix this before "
            "training in nm.", len(suspect),
            ", ".join(r["specimen"] for r in suspect))
    elif rows:
        LOG.info("width/scale consistency: %d specimen(s) checked, all within "
                 "%.0f%% spread", len(rows), 100 * tolerance)
    else:
        LOG.warning("no specimen has two calibrated fields at different "
                    "magnifications, so scale consistency cannot be checked. "
                    "Any nm-domain claim rests on unverified calibration.")
    return report


def calibration_template(audit_json: str | Path,
                         out_yaml: str | Path = "calibration.yaml") -> dict[str, Any]:
    """Write a stub calibration table for every field with no pixel size.

    The scale bar was found on these fields -- ``bar_px`` is in the audit --
    only its label was never read.  One number per field from the acquisition
    log turns 45 pixel-only fields into calibrated ones, which is a larger gain
    than anything in the model.
    """
    import json

    import yaml

    rep = json.loads(Path(audit_json).read_text(encoding="utf-8"))
    rows = rep.get("per_image") or rep.get("images") or []
    if isinstance(rows, dict):
        rows = [dict(v, image_id=k) for k, v in rows.items()]

    # Values are written as YAML null, never as placeholder TEXT: this file is
    # fed straight to load_calibration_table, and a string there is not a
    # number. The scale-bar length goes in a comment beside the key instead.
    stub: dict[str, Any] = {}
    for r in rows:
        nmpp = r.get("nm_per_pixel")
        unset = (nmpp is None or nmpp == ""
                 or (isinstance(nmpp, float) and not np.isfinite(nmpp)))
        if unset:
            stub[str(r.get("image_id"))] = r.get("scale_bar_px")

    lines = ["# One entry per field with no pixel size. Replace each null with",
             "# a number:  nm_per_pixel = (bar length in nm) / (bar length in px)",
             "# Leave a line as null if you cannot fill it -- that field stays",
             "# pixel-only. Nothing here is read until it is a number.",
             ""]
    for iid in sorted(stub):
        bar = stub[iid]
        comment = f"   # scale bar measured {bar} px" if bar else ""
        # safe_dump of a bare scalar appends a document-end marker; quote the
        # key ourselves so ids like 462_1 and 2-21 both round-trip
        key = "'" + str(iid).replace("'", "''") + "'"
        lines.append(f"{key}: null{comment}")
    Path(out_yaml).write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOG.info("wrote %s with %d field(s) awaiting a pixel size",
             out_yaml, len(stub))
    return stub


def calibration_table_is_usable(path: str | Path) -> bool:
    """True when the table has at least one filled-in number.

    An all-null stub is passed to nothing: handing the extractor a table with
    no numbers in it buys nothing and only widens the surface for a typo.
    """
    from .calibration import load_calibration_table

    return bool(load_calibration_table(path))


def recovery_gate(meta: "list[dict[str, Any]]", *,
                  min_rate: float = 0.80,
                  max_chord_dev_deg: float = 12.0,
                  max_resid_px: float = 1.0) -> dict[str, Any]:
    """Decide which recovered fields are fit to train on.

    ``recovery_rate`` alone cannot do this.  For a field whose CSV carries
    coordinates the rate is 1.000 *by construction* -- nothing was recovered,
    the positions were copied -- so a rate filter passes every such field
    unexamined.  In the August run it excluded nothing, and field ``A`` went
    into training with a 40.2 deg disagreement between its chords and the
    ridges they are supposed to cross.

    The geometry checks are the ones that discriminate:

    ``chord_vs_ridge_deg``
        median angle between each chord and the fiber it crosses.  A chord is a
        width measurement across a fiber, so this is small when the annotation
        is on the structure it claims to be on.  Tens of degrees means the
        overlay was misregistered, the wrong column was read as the angle, or
        the field is not what it is labelled as.
    ``median_length_residual_px``
        how well the recovered chord lengths reproduce the CSV lengths.  Sub-
        pixel on a good field.
    """
    good, bad = [], {}
    for m in meta:
        iid = str(m.get("image_id"))
        lo = m.get("line_overlay") or {}
        oc = m.get("orientation_convention") or {}
        rate = float(m.get("recovery_rate") or 0.0)
        dev = oc.get("median_deviation_deg")
        resid = lo.get("median_length_residual_px")
        reasons = []
        if rate < min_rate:
            reasons.append(f"recovery_rate={rate:.2f} < {min_rate:.2f}")
        if dev is not None and float(dev) > max_chord_dev_deg:
            reasons.append(f"chord_vs_ridge={float(dev):.1f} deg > "
                           f"{max_chord_dev_deg:.0f} (chords do not cross the "
                           f"fibers they claim to measure)")
        if resid is not None and float(resid) > max_resid_px:
            reasons.append(f"length_residual={float(resid):.2f} px > "
                           f"{max_resid_px:.2f}")
        (bad.setdefault(iid, reasons) if reasons else good.append(iid))
    for iid, reasons in sorted(bad.items()):
        LOG.warning("excluding %s: %s", iid, "; ".join(reasons))
    LOG.info("recovery gate: %d field(s) kept, %d excluded", len(good), len(bad))
    return {"keep": sorted(good), "drop": bad}

# --------------------------------------------------------------------------- #
# ---- v6.4 -----------------------------------------------------------------
#
# calibration_template used to WRITE THE WHOLE FILE from its stub.  Two ways
# that bites:
#
#   * run cell 3b after typing numbers into calibration.yaml and the numbers are
#     gone -- the exact file this notebook keeps telling you to fill in by hand;
#   * run cell 3b after cell 3c and every OCR-resolved entry is gone too, so the
#     next run re-reads 130 footers for nothing.
#
# It now merges.  Anything already numeric is kept verbatim and written back
# first; only fields still missing a pixel size are appended as nulls.  Cell
# order between 3b and 3c stops mattering.
# --------------------------------------------------------------------------- #
def calibration_template(audit_json, out_yaml="calibration.yaml"):
    """Add null stubs for uncalibrated fields, preserving every existing number."""
    import json as _json

    import yaml as _yaml

    rep = _json.loads(Path(audit_json).read_text(encoding="utf-8"))
    rows = rep.get("per_image") or rep.get("images") or []
    if isinstance(rows, dict):
        rows = [dict(v, image_id=k) for k, v in rows.items()]

    keep: dict[str, float] = {}
    if Path(out_yaml).exists():
        try:
            loaded = _yaml.safe_load(Path(out_yaml).read_text(encoding="utf-8"))
            for k, v in (loaded or {}).items():
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(fv) and fv > 0:
                    keep[str(k)] = fv
        except Exception as exc:  # noqa: BLE001
            LOG.warning("could not parse %s (%s) -- NOT overwriting it; "
                        "move it aside and re-run if you want a fresh stub",
                        out_yaml, exc)
            return {}

    stub: dict[str, Any] = {}
    for r in rows:
        iid = str(r.get("image_id"))
        nmpp = r.get("nm_per_pixel")
        unset = (nmpp is None or nmpp == ""
                 or (isinstance(nmpp, float) and not np.isfinite(nmpp)))
        if unset and iid not in keep:
            stub[iid] = r.get("scale_bar_px")

    lines = [
        "# Pixel sizes for this project.",
        "# A number here WINS over OCR on every later run and is never",
        "# overwritten by a notebook cell. nm_per_pixel = bar length in nm",
        "# divided by bar length in px, or FOV width in nm divided by image",
        "# width in px. Leave a line null if you cannot fill it -- that field",
        "# stays pixel-only, which is recoverable. A guess is not.",
        "",
    ]
    for iid in sorted(keep):
        key = "'" + str(iid).replace("'", "''") + "'"
        lines.append(f"{key}: {keep[iid]!r}")
    if keep:
        lines.append("")
    for iid in sorted(stub):
        bar = stub[iid]
        comment = f"   # scale bar measured {bar} px" if bar else ""
        key = "'" + str(iid).replace("'", "''") + "'"
        lines.append(f"{key}: null{comment}")
    Path(out_yaml).write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOG.info("%s: %d number(s) kept, %d field(s) still awaiting a pixel size",
             out_yaml, len(keep), len(stub))
    return stub
