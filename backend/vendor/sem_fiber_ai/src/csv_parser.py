"""Schema-agnostic parser for ImageJ ``Results`` tables.

The project must not hard-code a column layout: different ``Set Measurements``
configurations produce different columns, and some exports carry no coordinates
at all.  :func:`parse_measurement_csv` inspects header names *and* column
contents, maps them onto a canonical schema, and reports what it could not find
so the caller can decide whether coordinates have to be recovered from the
annotated overlay instead.

Canonical fields
----------------
``label``      measurement id (1-based, matches the number drawn on the overlay)
``length``     the measurement itself (fiber thickness) in *source* units
``angle``      measurement-line angle in degrees, ImageJ convention
``x1,y1,x2,y2``  endpoints, if present
``cx,cy``      centre, if present or derivable
``area,mean,min,max,std``  ancillary ImageJ statistics (kept, never a target)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import get_logger

LOG = get_logger(__name__)

# name -> canonical field.  Matched case-insensitively after stripping
# non-alphanumerics, so "BX", "b_x", " BX " all collapse to "bx".
_NAME_MAP: dict[str, str] = {
    "label": "label", "": "label", "index": "label", "n": "label", "id": "label",
    "roi": "label", "number": "label", "no": "label", "slice": "slice",
    "length": "length", "len": "length", "linelength": "length",
    "width": "length", "thickness": "length", "diameter": "length",
    "feretmin": "length", "minferet": "length",
    "angle": "angle", "linangle": "angle", "lineangle": "angle",
    "feretangle": "angle", "orientation": "angle",
    "x": "cx", "y": "cy", "xm": "cx", "ym": "cy",
    "centerx": "cx", "centery": "cy", "centrex": "cx", "centrey": "cy",
    "widthpx": "length", "widthnm": "length_nm",
    "localorientationdeg": "fiber_angle", "directiondeg": "angle",
    "pathdirectiondeg": "path_angle", "confidence": "confidence",
    "grade": "grade", "status": "status", "source": "source",
    "reviewlabel": "review_label", "measurementid": "label",
    "centroidx": "cx", "centroidy": "cy",
    "x1": "x1", "y1": "y1", "x2": "x2", "y2": "y2",
    "xstart": "x1", "ystart": "y1", "xend": "x2", "yend": "y2",
    "startx": "x1", "starty": "y1", "endx": "x2", "endy": "y2",
    "bx": "bx", "by": "by", "bwidth": "bw", "bheight": "bh",
    "width_bounding": "bw", "height": "bh",
    "area": "area", "mean": "mean", "min": "min", "max": "max",
    "stddev": "std", "std": "std", "median": "median", "mode": "mode",
    "intden": "intden", "rawintden": "rawintden", "perim": "perim",
    "major": "major", "minor": "minor", "circ": "circ", "ar": "ar",
    "round": "round", "solidity": "solidity", "feret": "feret",
}

_CANON_NUMERIC = ("length", "length_nm", "angle", "fiber_angle", "path_angle",
                  "cx", "cy", "x1", "y1", "x2", "y2", "bx", "by", "bw", "bh",
                  "area", "mean", "min", "max", "std", "confidence")


def _normalise(name: str) -> str:
    """'  Feret Angle ' -> 'feretangle'; BOM and unicode spaces removed."""
    name = str(name).replace("\ufeff", "").strip().lower()
    return re.sub(r"[^a-z0-9]", "", name)


@dataclass
class ParsedCSV:
    """Result of parsing one ImageJ results file."""
    path: Path
    frame: pd.DataFrame                      # canonical columns
    raw_columns: list[str]
    column_map: dict[str, str]               # raw -> canonical
    has_coordinates: bool
    has_endpoints: bool
    n_rows: int
    n_dropped: int
    derived: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "n_rows": self.n_rows,
            "n_dropped": self.n_dropped,
            "raw_columns": self.raw_columns,
            "column_map": self.column_map,
            "has_coordinates": self.has_coordinates,
            "has_endpoints": self.has_endpoints,
            "derived": self.derived,
            "n_errors": len(self.errors),
        }


def _read_any(path: Path) -> pd.DataFrame:
    """Read csv/tsv/xls with delimiter sniffing and BOM tolerance."""
    if path.suffix.lower() in (".xls", ".xlsx"):
        return pd.read_excel(path)
    for kwargs in ({"sep": None, "engine": "python"}, {"sep": ","}, {"sep": "\t"},
                   {"sep": ";"}):
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", **kwargs)
            if df.shape[1] >= 2:
                return df
        except Exception:  # noqa: BLE001 - try the next dialect
            continue
    raise ValueError(f"could not parse {path} as a delimited table")


def parse_measurement_csv(path: str | Path, *, strict: bool = False) -> ParsedCSV:
    """Parse one ImageJ results table into the canonical schema.

    Parameters
    ----------
    path : file to read.
    strict : if True, raise when no ``length``-like column can be identified;
        otherwise return a ParsedCSV whose ``errors`` explains the failure.
    """
    path = Path(path)
    raw = _read_any(path)
    raw_columns = [str(c) for c in raw.columns]

    column_map: dict[str, str] = {}
    used: set[str] = set()
    for col in raw.columns:
        key = _normalise(col)
        canon = _NAME_MAP.get(key)
        if canon is None:
            continue
        # first winner keeps the canonical slot; a second candidate is ignored
        if canon in used:
            continue
        column_map[str(col)] = canon
        used.add(canon)

    # 'Width'/'Height' are ambiguous: with a bounding-rectangle export they are
    # the box dimensions, but on their own they are the measurement itself.
    # Resolve by context rather than by guessing from the name alone.
    if "bx" in used and "by" in used:
        for rawc, canon in list(column_map.items()):
            key = _normalise(rawc)
            if key == "width" and canon == "length":
                other = [c for c in raw.columns
                         if _NAME_MAP.get(_normalise(c)) == "length" and c != rawc]
                if other:
                    column_map[rawc] = "bw"
                    column_map[other[0]] = "length"
                    used.discard("length")
                    used.update({"bw", "length"})
            if key == "height" and canon in ("bh", "length"):
                column_map[rawc] = "bh"
                used.add("bh")

    out = pd.DataFrame(index=raw.index)
    for rawc, canon in column_map.items():
        out[canon] = raw[rawc]

    # --- unnamed first column is ImageJ's row index -----------------------
    if "label" not in out.columns:
        first = raw.columns[0]
        if _normalise(first) in ("", "unnamed0") or raw[first].dtype.kind in "iu":
            out["label"] = raw[first]
            column_map[str(first)] = "label"
        else:
            out["label"] = np.arange(1, len(raw) + 1)

    # ImageJ appends summary rows (Mean / SD / Min / Max) to the Results table
    # when "Summarize" was used.  They carry a valid-looking Length and would
    # otherwise be trained on as if they were real measurements.
    errors: list[dict[str, Any]] = []
    derived: list[str] = []
    summary_names = {"mean", "sd", "min", "max", "median", "stddev", "total"}
    # pandas >= 3 gives text columns a StringDtype rather than object, so test
    # for "not numeric" instead of "is object"
    label_col = next((c for c in raw.columns
                      if _NAME_MAP.get(_normalise(c)) == "label"
                      and not pd.api.types.is_numeric_dtype(raw[c])), None)
    if label_col is not None:
        is_summary = raw[label_col].astype(str).str.strip().str.lower().isin(summary_names)
        if is_summary.any():
            for idx in raw.index[is_summary]:
                errors.append({"row": int(idx), "reason": "imagej_summary_row",
                               "detail": str(raw.at[idx, label_col])})
            out = out.loc[~is_summary].copy()
            LOG.info("%s: removed %d ImageJ summary row(s)", path.name,
                     int(is_summary.sum()))
        # the Label column often names the source image; keep it for pairing
        srcs = raw.loc[~is_summary, label_col].astype(str).str.strip()
        if len(srcs) and srcs.nunique() <= 3:
            out["source_image"] = srcs

    if "length" not in out.columns:
        msg = (f"{path.name}: no length/width/thickness column among {raw_columns}")
        if strict:
            raise ValueError(msg)
        errors.append({"row": None, "reason": "missing_length_column", "detail": msg})
        LOG.warning(msg)
        out["length"] = np.nan

    if "angle" not in out.columns:
        out["angle"] = np.nan
        errors.append({"row": None, "reason": "missing_angle_column",
                       "detail": f"{path.name}: no angle column; orientation will "
                                 "have to be recovered from the image"})

    for c in _CANON_NUMERIC:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # --- derive whatever geometry the export allows -----------------------
    has_endpoints = all(c in out.columns and out[c].notna().any()
                        for c in ("x1", "y1", "x2", "y2"))
    if has_endpoints:
        if "cx" not in out.columns or out["cx"].isna().all():
            out["cx"] = (out["x1"] + out["x2"]) / 2.0
            out["cy"] = (out["y1"] + out["y2"]) / 2.0
    elif {"bx", "by", "bw", "bh"} <= set(out.columns):
        # bounding rectangle of a line ROI -> centre is exact, endpoints are
        # ambiguous in sign, so we only fill the centre here.
        if "cx" not in out.columns or out["cx"].isna().all():
            out["cx"] = out["bx"] + out["bw"] / 2.0
            out["cy"] = out["by"] + out["bh"] / 2.0

    # When the export gives endpoints, the measurement itself is implied by them.
    # Deriving length and angle from the geometry beats guessing which of several
    # similarly-named columns ("width_px", "sample_length_px", "direction_deg",
    # "path_direction_deg") the tool meant.
    if has_endpoints:
        dx = out["x2"] - out["x1"]
        dy = out["y2"] - out["y1"]
        # Endpoints are authoritative when present. A column merely *named*
        # like an angle may mean something else entirely: VisionFlux's
        # "direction_deg" is the FIBER direction, not the chord's, and taking
        # it at face value rotates every measurement by 90 degrees.
        if "angle" in out.columns and out["angle"].notna().any():
            out["angle_reported"] = out["angle"]
        out["angle"] = np.degrees(np.arctan2(dy, dx))
        derived.append("angle from endpoints (raster convention)")
        if "length" not in out.columns or out["length"].isna().all():
            out["length"] = np.hypot(dx, dy)
            derived.append("length from endpoints")

    has_coordinates = ("cx" in out.columns and "cy" in out.columns
                       and out["cx"].notna().any() and out["cy"].notna().any())

    # --- row validity -----------------------------------------------------
    n_before = len(out) + len([e for e in errors
                               if e["reason"] == "imagej_summary_row"])
    bad = out["length"].isna() | (out["length"] <= 0)
    for idx in out.index[bad]:
        errors.append({"row": int(idx), "label": _safe(out, "label", idx),
                       "reason": "invalid_length",
                       "detail": f"length={_safe(out, 'length', idx)}"})
    out = out.loc[~bad].copy()

    # Some exports use a string id ("auto-r3-s0") rather than a row number.
    # That is not a defect, so keep the original and number the rows positionally
    # instead of discarding every measurement.
    numeric_label = pd.to_numeric(out["label"], errors="coerce")
    if numeric_label.isna().all() and len(out):
        out["label_text"] = out["label"].astype(str)
        out["label"] = np.arange(1, len(out) + 1)
        derived.append("label numbered positionally (source ids are not numeric)")
    else:
        out["label"] = numeric_label
        still_bad = out["label"].isna()
        for idx in out.index[still_bad]:
            errors.append({"row": int(idx), "reason": "unparseable_label",
                           "detail": str(_safe(out, "label_text", idx))})
        out = out.loc[~still_bad].copy()
    out["label"] = out["label"].astype(int)

    dup = out["label"].duplicated(keep="first")
    for idx in out.index[dup]:
        errors.append({"row": int(idx), "label": int(out.at[idx, "label"]),
                       "reason": "duplicate_label", "detail": "kept first occurrence"})
    out = out.loc[~dup].copy()

    out = out.reset_index(drop=True)
    LOG.info("%s: %d rows, coords=%s, endpoints=%s, %d errors",
             path.name, len(out), has_coordinates, has_endpoints, len(errors))

    return ParsedCSV(
        path=path, frame=out, raw_columns=raw_columns, column_map=column_map,
        has_coordinates=bool(has_coordinates), has_endpoints=bool(has_endpoints),
        derived=derived,
        n_rows=len(out), n_dropped=n_before - len(out), errors=errors,
    )


def _safe(df: pd.DataFrame, col: str, idx: Any) -> Any:
    try:
        return df.at[idx, col]
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# unit inference
# --------------------------------------------------------------------------- #
def infer_length_quantum(lengths: np.ndarray, *,
                         candidates: tuple[float, ...] = (
                             0.25, 1 / 3, 0.5, 2 / 3, 0.8, 15 / 16, 1.0,
                             16 / 15, 1.25, 4 / 3, 1.5, 2.0, 2.5, 16 / 5),
                         tol: float = 0.02, min_frac: float = 0.6
                         ) -> dict[str, Any]:
    """Detect whether ``lengths`` lie on an integer lattice.

    A measurement that was taken as an integer number of pixels and then
    multiplied by a calibration factor ``c`` leaves a fingerprint: the values
    are (mostly) integer multiples of ``c``.  Recovering ``c`` tells us the
    nm/pixel of the *image the measurement was made on*, which is the single
    most common source of silent unit errors in this kind of dataset.

    Returns a dict with the best candidate, the fraction of values explained,
    and the full ranking.  Never raises; an inconclusive result is reported as
    ``best=None``.
    """
    x = np.asarray(lengths, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 5:
        return {"best": None, "frac": 0.0, "ranking": [], "n": int(x.size)}

    ranking = []
    for c in candidates:
        r = x / c
        frac = float(np.mean(np.abs(r - np.round(r)) < tol))
        ranking.append({"quantum": float(c), "frac_explained": frac})
    ranking.sort(key=lambda d: (-d["frac_explained"], d["quantum"]))
    best = ranking[0] if ranking[0]["frac_explained"] >= min_frac else None
    return {"best": best, "frac": ranking[0]["frac_explained"],
            "ranking": ranking, "n": int(x.size)}
