"""Pixel-size (nm/px) resolution with explicit provenance and no silent defaults.

Resolution order (first hit wins, each records how it was obtained):

1. ``--nm_per_pixel`` on the command line / ``override`` argument.
2. A sidecar file next to the image: ``<stem>.calib.json`` or an entry keyed by
   image id in a project-level ``calibration.yaml``.
3. The burned-in SEM footer text, e.g. ``FOV:1280x960nm`` combined with the
   pixel width of the image *above* the footer.
4. The footer scale bar: the bar's pixel length combined with its printed
   value.  Requires OCR (pytesseract) or an explicit ``scale_bar_nm``.
5. Nothing.  ``nm_per_pixel`` is ``None``; every downstream number is reported
   in pixels and nanometre fields are written as NaN, never guessed.

Note on step 3: FOV text describes the *acquired* field.  If the file you hold
was resized after acquisition, ``nm_per_pixel`` must be computed against the
current pixel width -- which is what :func:`from_fov_text` does.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from .utils import get_logger

LOG = get_logger(__name__)

# Tesseract reliably confuses a few glyphs in this small burned-in font.
# "@" and "Q" for "0" are by far the most common, and they silently break the
# FOV parse, so we try both the raw text and a digit-corrected copy.
_OCR_DIGIT_FIXES = {"@": "0", "Q": "0", "O0": "00", "l": "1", "|": "1"}

#: a standalone physical length, used for the scale-bar label ("100nm", "0.5um").
#: mm is excluded on purpose: the only mm value in these footers is the working
#: distance, which has nothing to do with the scale bar.
_BAR_RE = re.compile(r"(?<![0-9x\u00d7.])([0-9]+(?:\.[0-9]+)?)\s*(nm|um|\u00b5m|\u03bcm)"
                     r"(?![0-9x\u00d7])", re.IGNORECASE)

_FOV_RE = re.compile(
    r"FOV\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*[x\u00d7]\s*([0-9]+(?:\.[0-9]+)?)\s*"
    r"(nm|um|\u00b5m|mm)", re.IGNORECASE)
_UNIT_TO_NM = {"nm": 1.0, "um": 1000.0, "\u00b5m": 1000.0, "mm": 1e6}


@dataclass
class Calibration:
    """nm per pixel for one image, plus how we know it."""
    image_id: str
    nm_per_pixel: float | None
    source: str                      # override | sidecar | fov_text | scale_bar | unknown
    detail: str = ""
    footer_y: int | None = None      # first row of the burned-in info panel, if any

    @property
    def known(self) -> bool:
        return self.nm_per_pixel is not None and np.isfinite(self.nm_per_pixel)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def px_to_nm(self, px: np.ndarray | float) -> np.ndarray | float:
        if not self.known:
            return np.full_like(np.asarray(px, dtype=np.float64), np.nan)
        return np.asarray(px, dtype=np.float64) * self.nm_per_pixel

    def nm_to_px(self, nm: np.ndarray | float) -> np.ndarray | float:
        if not self.known:
            return np.full_like(np.asarray(nm, dtype=np.float64), np.nan)
        return np.asarray(nm, dtype=np.float64) / self.nm_per_pixel


# --------------------------------------------------------------------------- #
# footer handling
# --------------------------------------------------------------------------- #
def detect_footer_row(gray: np.ndarray, *, dark_thresh: float = 30.0,
                      dark_frac: float = 0.55, min_rows: int = 8) -> int | None:
    """Return the first row index of the burned-in black info panel, or None.

    The panel is a block of consecutive rows at the bottom of the frame that are
    overwhelmingly near-black.  Detected geometrically so it works for any SEM
    vendor and any image height -- nothing here is hard-coded to 1280x1024.
    """
    if gray.ndim != 2:
        raise ValueError("detect_footer_row expects a 2-D grayscale array")
    h = gray.shape[0]
    frac_dark = (gray < dark_thresh).mean(axis=1)
    # walk up from the bottom while rows stay mostly dark
    row = h
    while row - 1 >= 0 and frac_dark[row - 1] >= dark_frac:
        row -= 1
    if h - row < min_rows:
        return None
    if (h - row) > 0.4 * h:            # implausible: refuse rather than crop data
        LOG.warning("footer detection wanted %d of %d rows; ignoring", h - row, h)
        return None
    return int(row)


def strip_footer(gray: np.ndarray) -> tuple[np.ndarray, int | None]:
    """Return (image without footer, footer_row).  Footer row is None if absent."""
    row = detect_footer_row(gray)
    return (gray[:row] if row is not None else gray), row


def _ocr_variants(text: str) -> list[str]:
    """The raw OCR string plus a copy with common digit confusions repaired."""
    text = text or ""
    fixed = text
    for bad, good in _OCR_DIGIT_FIXES.items():
        fixed = fixed.replace(bad, good)
    return [text] if fixed == text else [text, fixed]


def from_fov_text(text: str, image_width_px: int) -> tuple[float | None, str]:
    """Parse ``FOV:1280x960nm`` style text into nm/px for a given pixel width."""
    m = None
    for variant in _ocr_variants(text):
        m = _FOV_RE.search(variant)
        if m:
            break
    if not m:
        return None, "no FOV pattern in text"
    w = float(m.group(1)) * _UNIT_TO_NM[m.group(3).lower()]
    if image_width_px <= 0:
        return None, "invalid image width"
    return w / float(image_width_px), f"FOV width {w:g} nm / {image_width_px} px"


def read_footer_text(gray: np.ndarray, footer_row: int | None) -> str:
    """OCR the footer strip if pytesseract is installed; '' otherwise."""
    if footer_row is None:
        return ""
    try:
        import pytesseract
        from PIL import Image
    except Exception:  # noqa: BLE001
        LOG.debug("pytesseract unavailable; skipping footer OCR")
        return ""
    strip = gray[footer_row:].astype(np.uint8)
    try:
        return pytesseract.image_to_string(Image.fromarray(strip))
    except Exception as exc:  # noqa: BLE001
        LOG.warning("footer OCR failed: %s", exc)
        return ""


def scale_bar_nm_from_text(text: str) -> tuple[float | None, str]:
    """Read the printed scale-bar value (e.g. "100nm") out of the footer text.

    Used only when the FOV string could not be parsed.  The bar plus its printed
    value is an independent route to the pixel size, and on these images the two
    agree to about 1%.
    """
    for variant in _ocr_variants(text):
        stripped = _FOV_RE.sub(" ", variant)          # remove the FOV pair first
        matches = _BAR_RE.findall(stripped)
        if matches:
            value, unit = matches[-1]
            try:
                nm = float(value) * _UNIT_TO_NM[unit.lower().replace("\u03bc", "\u00b5")]
            except (ValueError, KeyError):
                continue
            if 1.0 <= nm <= 1e5:
                return nm, f"scale-bar label '{value}{unit}'"
    return None, "no scale-bar value in text"


def _v3_detect_scale_bar_px(gray: np.ndarray, footer_row: int | None,
                        *, bright_thresh: float = 200.0,
                        min_len: int = 20) -> int | None:
    """Length in pixels of the longest bright horizontal run inside the footer.

    Returns None when no footer is present or no plausible bar is found.  The
    physical value of the bar must come from OCR or from the user; this function
    deliberately does not guess it.
    """
    if footer_row is None:
        return None
    strip = gray[footer_row:]
    best = 0
    for row in strip:
        bright = row > bright_thresh
        if not bright.any():
            continue
        # longest run of True
        idx = np.flatnonzero(np.diff(np.concatenate(([0], bright.view(np.int8), [0]))))
        runs = (idx[1::2] - idx[0::2]) if idx.size else np.array([], dtype=int)
        if runs.size:
            best = max(best, int(runs.max()))
    return best if best >= min_len else None


# --------------------------------------------------------------------------- #
# top-level resolution
# --------------------------------------------------------------------------- #
def _v3_resolve_calibration(image_path: str | Path, gray: np.ndarray, *,
                        image_id: str | None = None,
                        override: float | None = None,
                        table: dict[str, float] | None = None,
                        scale_bar_nm: float | None = None) -> Calibration:
    """Resolve nm/px for one image, recording where the number came from."""
    image_path = Path(image_path)
    image_id = image_id or image_path.stem
    footer_row = detect_footer_row(gray)
    width = int(gray.shape[1])

    if override is not None:
        if override <= 0:
            raise ValueError(f"nm_per_pixel must be > 0, got {override}")
        return Calibration(image_id, float(override), "override",
                           "supplied by caller", footer_row)

    if table and image_id in table:
        return Calibration(image_id, float(table[image_id]), "sidecar",
                           "project calibration table", footer_row)

    sidecar = image_path.with_suffix(".calib.json")
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            if "nm_per_pixel" in data:
                return Calibration(image_id, float(data["nm_per_pixel"]), "sidecar",
                                   str(sidecar), footer_row)
            if "fov_width_nm" in data:
                return Calibration(image_id, float(data["fov_width_nm"]) / width,
                                   "sidecar", f"{sidecar} fov_width_nm", footer_row)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("bad sidecar %s: %s", sidecar, exc)

    text = read_footer_text(gray, footer_row)
    nmpp, detail = from_fov_text(text, width)
    if nmpp is not None:
        return Calibration(image_id, float(nmpp), "fov_text", detail, footer_row)

    bar_px = detect_scale_bar_px(gray, footer_row)
    bar_nm, bar_detail = (scale_bar_nm, "supplied by caller") if scale_bar_nm \
        else scale_bar_nm_from_text(text)
    if bar_px and bar_nm:
        return Calibration(image_id, float(bar_nm) / bar_px, "scale_bar",
                           f"{bar_nm:g} nm / {bar_px} px ({bar_detail})", footer_row)

    LOG.warning("%s: pixel size UNKNOWN -- results will be in pixels only "
                "(bar_px=%s, ocr=%s)", image_id, bar_px, bool(text.strip()))
    return Calibration(image_id, None, "unknown",
                       f"scale_bar_px={bar_px}; footer_ocr={'yes' if text.strip() else 'no'}",
                       footer_row)


def load_calibration_table(path: str | Path | None) -> dict[str, float]:
    """Load ``{image_id: nm_per_pixel}`` from YAML or JSON.  Missing file -> {}."""
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        LOG.warning("calibration table %s not found", p)
        return {}
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    else:
        data = json.loads(p.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in data.items()}


# ---- v4 ----


import re
from pathlib import Path
from typing import Any

import numpy as np

from .utils import get_logger

LOG = get_logger(__name__)

# "1" reads as "T" or "I" in this font at small sizes; "0" as "@"/"Q"/"O".
_OCR_DIGIT_FIXES_V4 = {
    "@": "0", "Q": "0", "O": "0", "o": "0",
    "l": "1", "|": "1", "I": "1", "T": "1", "i": "1",
    "S": "5", "s": "5", "B": "8", "Z": "2", "g": "9",
}

_UNIT_TO_NM = {"nm": 1.0, "um": 1000.0, "\u00b5m": 1000.0, "\u03bcm": 1000.0,
               "mm": 1e6}

# A standalone physical length.  mm excluded: the only mm in these footers is
# the working distance.
_BAR_RE_V4 = re.compile(
    r"(?<![0-9x\u00d7.])([0-9]+(?:\.[0-9]+)?)\s*(nm|um|\u00b5m|\u03bcm)"
    r"(?![0-9x\u00d7])", re.IGNORECASE)


def _ocr_variants_v4(text: str) -> list[str]:
    """Raw text plus a digit-repaired copy, repairs applied only inside tokens
    that already look like a number+unit."""
    text = text or ""
    text = re.sub(r"(?<=[0-9])\s*(nn|nrn|rim|nm)\b", "nm", text, flags=re.I)
    text = re.sub(r"(?<=[0-9])\s*(urn|un|urm|\u00b5n)\b", "um", text, flags=re.I)
    out = [text]
    fixed = []
    for tok in re.split(r"(\s+)", text):
        if re.search(r"(nm|um|\u00b5m|\u03bcm)$", tok, re.IGNORECASE):
            head = tok[: -2]
            unit = tok[-2:]
            for bad, good in _OCR_DIGIT_FIXES_V4.items():
                head = head.replace(bad, good)
            fixed.append(head + unit)
        else:
            fixed.append(tok)
    joined = "".join(fixed)
    if joined != text:
        out.append(joined)
    return out


def detect_scale_bar_box(gray: np.ndarray, footer_row: int | None,
                         *, bright_thresh: float = 200.0,
                         min_len: int = 20, min_aspect: float = 3.0,
                         min_fill: float = 0.90
                         ) -> tuple[int, int, int, int] | None:
    """Bounding box (x, y, w, h) of the scale bar inside the footer, or None.

    The bar is the widest bright connected component that is a SOLID rectangle
    (fill >= min_fill of its box) with a moderate aspect ratio.  Solidity is the
    discriminating feature, not elongation: on real footers the bar fills its
    box exactly (1.00) while text glyph runs reach only 0.27-0.63.  Aspect alone
    is not enough -- a 100 nm bar at 50k is 49x13 px, aspect 3.8.  Text glyphs and
    glyph runs fail the aspect and fill tests, which is the failure mode of the
    row-wise longest-run heuristic this replaces.  Coordinates are relative to
    the footer strip.
    """
    if footer_row is None:
        return None
    import cv2

    strip = gray[footer_row:]
    if strip.size == 0:
        return None
    bw = (strip > bright_thresh).astype(np.uint8)
    if not bw.any():
        return None
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(bw, 8)
    best = None
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i][:5])
        if w < min_len or h < 1 or h > 0.30 * strip.shape[0]:
            continue
        if w / max(h, 1) < min_aspect:
            continue
        if area < min_fill * w * h:
            continue
        if best is None or w > best[2]:
            best = (x, y, w, h)
    return best


def detect_scale_bar_px(gray: np.ndarray, footer_row: int | None,
                        **kw: Any) -> int | None:
    """Backwards-compatible wrapper returning only the bar length in pixels."""
    box = detect_scale_bar_box(gray, footer_row, **kw)
    return box[2] if box else None


def _ocr(img: np.ndarray, psm: int, whitelist: str | None) -> str:
    try:
        import pytesseract
        from PIL import Image
    except Exception:  # noqa: BLE001
        return ""
    cfg = f"--oem 3 --psm {psm}"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    try:
        return pytesseract.image_to_string(Image.fromarray(img), config=cfg)
    except Exception as exc:  # noqa: BLE001
        LOG.debug("ocr failed (psm=%s): %s", psm, exc)
        return ""


def _prep(crop: np.ndarray, scale: int) -> np.ndarray:
    """Upscale, Otsu-binarise and invert to black-on-white for tesseract."""
    import cv2

    if crop.size == 0:
        return crop
    up = cv2.resize(crop.astype(np.uint8), None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC)
    up = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return 255 - up          # SEM footers are white-on-black; tesseract wants the reverse


def read_bar_label_nm(gray: np.ndarray, footer_row: int | None,
                      box: tuple[int, int, int, int] | None = None,
                      *, scale: int = 6) -> tuple[float | None, str]:
    """Read the printed scale-bar value from the neighbourhood of the bar.

    The label sits directly above or below the bar and is far too small to
    survive OCR of the whole strip, which is why the v3 path returned text but
    no length.  Cropping tight and upscaling 6x recovers it.
    """
    if footer_row is None:
        return None, "no footer"
    box = box if box is not None else detect_scale_bar_box(gray, footer_row)
    if box is None:
        return None, "no bar found"
    strip = gray[footer_row:]
    x, y, w, h = box
    wl = "0123456789.numµ\u03bc "
    # The label may sit above, below, LEFT or RIGHT of the bar depending on
    # vendor -- on these JEOL/Hitachi footers it is to the right, outside a
    # tight crop.  Widen the horizontal window progressively rather than
    # assuming a placement.
    for pad_x, pad_y, sc in ((45, 18, scale), (100, 22, scale),
                             (160, 26, scale + 2), (240, 30, scale + 2)):
        y0, y1 = max(0, y - pad_y), min(strip.shape[0], y + h + pad_y)
        x0, x1 = max(0, x - pad_x), min(strip.shape[1], x + w + pad_x)
        crop = strip[y0:y1, x0:x1]
        for psm in (7, 6, 11):
            txt = _ocr(_prep(crop, sc), psm, wl)
            for variant in _ocr_variants_v4(txt):
                for value, unit in _BAR_RE_V4.findall(variant):
                    try:
                        nm = float(value) * _UNIT_TO_NM[unit.lower()]
                    except (ValueError, KeyError):
                        continue
                    if 1.0 <= nm <= 1e5:
                        return nm, (f"bar label '{value}{unit}' "
                                    f"(pad {pad_x}, psm {psm})")
    return None, "bar label unreadable"


def read_footer_text_v4(gray: np.ndarray, footer_row: int | None,
                        *, scale: int = 3) -> str:
    """OCR the whole footer strip, upscaled and inverted.

    Used for the FOV string, which is set in the larger font and survives.
    """
    if footer_row is None:
        return ""
    strip = gray[footer_row:].astype(np.uint8)
    best = ""
    for psm in (6, 11):
        txt = _ocr(_prep(strip, scale), psm, None)
        if len(txt.strip()) > len(best.strip()):
            best = txt
    if not best.strip():                       # last resort: raw, as v3 did
        best = _ocr(strip, 6, None)
    return best

def crosscheck_fov_vs_bar(fov_nmpp: float | None, bar_nmpp: float | None,
                          image_id: str, tol: float = 0.02) -> str:
    """Log when the two independent routes to nm/px disagree.

    They measure different things -- the acquired field width and a drawn bar --
    so agreement is real evidence and disagreement is a real warning.  Neither
    is silently preferred here; the caller keeps its own order.
    """
    if fov_nmpp is None or bar_nmpp is None:
        return "single source"
    rel = abs(fov_nmpp - bar_nmpp) / max(fov_nmpp, bar_nmpp)
    if rel > tol:
        LOG.warning("%s: FOV says %.4f nm/px, scale bar says %.4f (%.1f%% apart)"
                    " -- check before quoting nanometres", image_id,
                    fov_nmpp, bar_nmpp, 100 * rel)
        return f"DISAGREE {100 * rel:.1f}%"
    return f"agree within {100 * rel:.1f}%"


def resolve_calibration(image_path, gray, *, image_id=None, override=None,
                        table=None, scale_bar_nm=None, mag_constant=None):
    """v4: same resolution order, better step 3/4, plus a cross-check."""
    import json
    from pathlib import Path as _P

    image_path = _P(image_path)
    image_id = image_id or image_path.stem
    footer_row = detect_footer_row(gray)
    width = int(gray.shape[1])

    if override is not None:
        if override <= 0:
            raise ValueError(f"nm_per_pixel must be > 0, got {override}")
        return Calibration(image_id, float(override), "override",
                           "supplied by caller", footer_row)

    if table and image_id in table:
        return Calibration(image_id, float(table[image_id]), "sidecar",
                           "project calibration table", footer_row)

    sidecar = image_path.with_suffix(".calib.json")
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            if "nm_per_pixel" in data:
                return Calibration(image_id, float(data["nm_per_pixel"]),
                                   "sidecar", str(sidecar), footer_row)
            if "fov_width_nm" in data:
                return Calibration(image_id, float(data["fov_width_nm"]) / width,
                                   "sidecar", f"{sidecar} fov_width_nm", footer_row)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("bad sidecar %s: %s", sidecar, exc)

    text = read_footer_text_v4(gray, footer_row)
    fov_nmpp, fov_detail = from_fov_text(text, width)
    mag, _mag_detail = read_magnification(text)
    if mag and mag_constant and fov_nmpp is None:
        # [v4] large-font magnification + a constant fitted from the images
        # whose FOV string did parse.  Only used when FOV and bar both failed.
        nmpp_mag = (mag_constant / mag) / float(width)
        _pending_mag = (nmpp_mag, f"x{mag:g}k via fitted constant {mag_constant:.0f}")
    else:
        _pending_mag = (None, "")

    box = detect_scale_bar_box(gray, footer_row)
    bar_px = box[2] if box else None
    if scale_bar_nm:
        bar_nm, bar_detail = float(scale_bar_nm), "supplied by caller"
    else:
        bar_nm, bar_detail = read_bar_label_nm(gray, footer_row, box)
    bar_nmpp = (float(bar_nm) / bar_px) if (bar_px and bar_nm) else None

    agreement = crosscheck_fov_vs_bar(fov_nmpp, bar_nmpp, image_id)

    if fov_nmpp is not None:
        return Calibration(image_id, float(fov_nmpp), "fov_text",
                           f"{fov_detail}; {agreement}", footer_row)
    if bar_nmpp is not None:
        return Calibration(image_id, float(bar_nmpp), "scale_bar",
                           f"{bar_nm:g} nm / {bar_px} px ({bar_detail})",
                           footer_row)

    if _pending_mag[0] is not None:
        return Calibration(image_id, float(_pending_mag[0]), "magnification",
                           _pending_mag[1], footer_row)

    LOG.warning("%s: pixel size UNKNOWN (bar_px=%s, footer_text=%s, %s)",
                image_id, bar_px, bool(text.strip()), bar_detail)
    return Calibration(image_id, None, "unknown",
                       f"bar_px={bar_px}; {bar_detail}; "
                       f"footer_ocr={'yes' if text.strip() else 'no'}",
                       footer_row)


def calibrate_all(image_dir, out_yaml="calibration.yaml", *,
                  pattern=("*.tif", "*.tiff", "*.png", "*.jpg", "*.bmp"),
                  overwrite=False, existing=None):
    """Resolve nm/px for every image in a folder and write calibration.yaml.

    Pay the OCR cost once.  Entries already present in the table are kept unless
    ``overwrite`` is set -- a number you typed by hand is better evidence than a
    number tesseract guessed, and this will not quietly replace it.

    Returns a DataFrame with one row per image: id, nm_per_pixel, source,
    detail.  Read the ``source`` column before believing any nanometre.
    """
    from pathlib import Path as _P

    import pandas as pd
    import yaml

    from .utils import read_gray

    image_dir = _P(image_dir)
    files = sorted({p for pat in pattern for p in image_dir.rglob(pat)})
    if not files:
        LOG.warning("no images under %s", image_dir)
        return pd.DataFrame(columns=["image_id", "nm_per_pixel", "source", "detail"])

    known = dict(existing or {})
    if not known and _P(out_yaml).exists():
        known = {str(k): v for k, v in
                 (yaml.safe_load(_P(out_yaml).read_text(encoding="utf-8")) or {}).items()
                 if v is not None}

    rows = []
    for i, p in enumerate(files, 1):
        iid = p.stem
        if iid in known and not overwrite:
            rows.append({"image_id": iid, "nm_per_pixel": float(known[iid]),
                         "source": "existing", "detail": "kept from table"})
            continue
        try:
            gray = read_gray(p)
        except Exception as exc:  # noqa: BLE001
            rows.append({"image_id": iid, "nm_per_pixel": None,
                         "source": "unreadable", "detail": str(exc)})
            continue
        c = resolve_calibration(p, gray, image_id=iid)
        text = read_footer_text_v4(gray, detect_footer_row(gray))
        mag, _ = read_magnification(text)
        fovw = (c.nm_per_pixel * gray.shape[1]) if c.source == "fov_text" else None
        rows.append({"image_id": iid, "nm_per_pixel": c.nm_per_pixel,
                     "source": c.source, "detail": c.detail,
                     "magnification": mag, "fov_width_nm": fovw,
                     "_path": str(p)})
        if i % 25 == 0:
            LOG.info("calibrated %d/%d", i, len(files))

    # second pass: fit FOV_nm x mag from the resolved images and use it on the
    # ones where both FOV and bar failed.  Nothing is overwritten -- this only
    # fills nulls.
    # Fit PER PREFIX GROUP first, then fall back to a folder-wide constant.
    # A folder holding two instruments (or two export widths) has no single
    # constant, and fitting one across the lot would either be refused -- losing
    # the groups that are internally consistent -- or, worse, average them.
    def _group(iid):
        for sep in ("_", "-"):
            if sep in iid:
                return iid.split(sep)[0]
        return iid

    groups = {}
    for r in rows:
        groups.setdefault(_group(r["image_id"]), []).append(r)

    global_k, global_note = fit_magnification_constant(rows)
    n_filled = 0
    for gname, grows in sorted(groups.items()):
        k, note = fit_magnification_constant(grows)
        if k:
            note = f"group '{gname}': {note}"
        else:
            k, note = global_k, (f"group '{gname}' had no constant of its own; "
                                 f"folder-wide {global_note}" if global_k else "")
        if not k:
            continue
        for r in grows:
            if r["nm_per_pixel"] is not None or not r.get("magnification"):
                continue
            try:
                gray = read_gray(r["_path"])
            except Exception:  # noqa: BLE001
                continue
            r["nm_per_pixel"] = (k / r["magnification"]) / gray.shape[1]
            r["source"] = "magnification"
            r["detail"] = f"x{r['magnification']:g}k; {note}"
            n_filled += 1
    LOG.info("magnification route filled %d field(s)", n_filled)

    df = pd.DataFrame(rows).drop(columns=["_path"])
    df = df.sort_values("image_id").reset_index(drop=True)
    table = {r.image_id: (float(r.nm_per_pixel) if r.nm_per_pixel is not None
                          and np.isfinite(r.nm_per_pixel) else None)
             for r in df.itertuples()}
    _P(out_yaml).write_text(
        "# written by calibrate_all(); null means UNKNOWN, fill it in by hand.\n"
        "# Anything you type here wins over OCR on the next run.\n"
        + yaml.safe_dump(table, sort_keys=True, allow_unicode=True),
        encoding="utf-8")
    got = int(df.nm_per_pixel.notna().sum())
    LOG.info("calibration: %d/%d resolved -> %s", got, len(df), out_yaml)
    LOG.info("by source: %s", df.source.value_counts().to_dict())
    return df


_MAG_RE = re.compile(r"[x\u00d7]\s*([0-9]+(?:\.[0-9]+)?)\s*k\b", re.IGNORECASE)


def read_magnification(text: str) -> tuple[float | None, str]:
    """Parse the magnification token (e.g. ``x50.0k``) out of footer text.

    It is set in the same large font as the kV/WD line, so it survives OCR on
    images where the small scale-bar label does not.  On its own it is not a
    pixel size -- see :func:`fit_magnification_constant`.
    """
    for variant in _ocr_variants_v4(text):
        m = _MAG_RE.search(variant)
        if m:
            try:
                mag = float(m.group(1))
            except ValueError:
                continue
            if 0.1 <= mag <= 1e4:
                return mag, f"magnification x{mag:g}k"
    return None, "no magnification in text"


def fit_magnification_constant(rows) -> tuple[float | None, str]:
    """Fit ``FOV_width_nm x magnification_k`` from images where FOV parsed.

    For one instrument at one export width this product is a constant: 2-21 is
    x50.0k with FOV 2560 nm and 2-22 is x100k with FOV 1280 nm, both giving
    128000.  Fitting it from the images that *do* resolve turns the large-font
    magnification into a pixel size for the ones that do not, instead of leaving
    them pixel-only.

    Deliberately conservative: needs at least three agreeing images and refuses
    if their spread exceeds 5%, because a mixed-instrument folder would
    otherwise produce a confident wrong number for every unresolved field.
    """
    import numpy as _np

    ks = [r["fov_width_nm"] * r["magnification"] for r in rows
          if r.get("fov_width_nm") and r.get("magnification")]
    if len(ks) < 3:
        return None, f"only {len(ks)} image(s) with both FOV and magnification"
    ks = _np.asarray(ks, float)
    med = float(_np.median(ks))
    spread = float((ks.max() - ks.min()) / max(med, 1e-9))
    if spread > 0.05:
        LOG.warning("magnification constant varies by %.1f%% across the folder "
                    "(%.0f to %.0f) -- not fitting one; are these from more than "
                    "one instrument or export size?",
                    100 * spread, ks.min(), ks.max())
        return None, f"spread {100 * spread:.1f}% too large"
    LOG.info("magnification constant: FOV_nm x mag_k = %.0f (n=%d, spread %.2f%%)",
             med, len(ks), 100 * spread)
    return med, f"fitted from {len(ks)} image(s), spread {100 * spread:.2f}%"

# --------------------------------------------------------------------------- #
# ---- v6.4 -----------------------------------------------------------------
#
# Why this block exists.  The v6.3 run resolved 87 of 117 pixel sizes from the
# scale bar, and several of those were nonsense: B_4 and B_5 read their bar
# label as "1nm" over a 160 px bar and produced 0.00625 nm/px, which put six
# training fields at a median fibre width of 0.047 nm.  Four more fields showed
# the FOV route and the bar route 90-95% apart and the code only warned.
#
# Three changes, in order of how much they matter:
#
#   1. The bar can no longer ORIGINATE a pixel size unless its OCR'd label is
#      one of the standard 1-2-5 bar values AND the resulting nm/px is
#      physically plausible.  "71nm" is not a bar anyone prints; "1nm" is, but
#      1 nm over 160 px is finer than the microscope resolves.
#   2. The magnification token becomes a first-class route.  It is set in the
#      same large font as the kV/WD line, so it survives OCR where the small bar
#      label does not -- but v6.3 could not read it, because the digit repair
#      table was only applied to tokens ending in nm/um, so "x10@k" never became
#      "x100k".  With that fixed, FOV_nm x mag_k = 128000 turns the magnification
#      into a pixel size for every field whose FOV string failed.
#   3. Disagreements are arbitrated instead of warned about.  When FOV and bar
#      disagree, the magnification route breaks the tie; when it cannot, the
#      field is marked disputed and excluded from nanometre reporting rather
#      than silently taking one of them.
#
# The footer OCR is also memoised on disk, because cell 3 and cell 3c were each
# paying nine minutes for the same reads.
# --------------------------------------------------------------------------- #

#: Values a microscope actually prints on a scale bar: 1-2-5 per decade.
STANDARD_BAR_NM: tuple[float, ...] = tuple(
    float(m) * (10.0 ** e) for e in range(0, 6) for m in (1, 2, 5)
)

#: Hard plausibility bounds on nm/px for a field-emission SEM micrograph.
#: 0.1 nm/px is below the resolution of any instrument that produced these
#: images; 5000 nm/px is a 6.4 mm field.  Anything outside is an OCR artefact.
PLAUSIBLE_NMPP: tuple[float, float] = (0.1, 5000.0)

_DIGIT_FIX_V64 = {
    "@": "0", "Q": "0", "O": "0", "o": "0", "D": "0",
    "l": "1", "|": "1", "I": "1", "T": "1", "i": "1", "!": "1",
    "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "g": "9", "G": "6",
}

#: magnification token, tolerant of the glyph confusions above INSIDE the number
_MAG_TOKEN_RE_V64 = re.compile(
    r"[x\u00d7\u00d7X]\s*([0-9@QODolIiTS sBZzgG.!|]{1,7}?)\s*[kK]\b")


def _fix_digits_v64(s: str) -> str:
    for bad, good in _DIGIT_FIX_V64.items():
        s = s.replace(bad, good)
    return s


def read_magnification(text: str) -> tuple[float | None, str]:
    """Parse ``x50.0k`` out of footer text, repairing OCR digit confusions.

    Overrides the v4 version, whose repair table was applied only to tokens
    ending in a length unit -- so ``x10@k`` (the single most common misread of
    ``x100k`` in this font) was never recovered and the field silently dropped
    out of the magnification-constant fit.
    """
    text = text or ""
    for variant in (text, _fix_digits_v64(text)):
        for m in _MAG_TOKEN_RE_V64.finditer(variant):
            raw = _fix_digits_v64(m.group(1)).strip().strip(".")
            if not raw or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", raw):
                continue
            try:
                mag = float(raw)
            except ValueError:
                continue
            if 0.1 <= mag <= 1e4:
                return mag, f"magnification x{mag:g}k"
    return None, "no magnification in text"


def snap_bar_nm(bar_px: float, nmpp_estimate: float, *, tol: float = 0.15
                ) -> tuple[float | None, float]:
    """Nearest standard bar value to ``bar_px * nmpp_estimate``.

    Returns ``(nm, relative_residual)``.  ``nm`` is None when the implied length
    is not close to any printed value, which is itself informative: it means the
    estimate and the bar are inconsistent.

    This is used to CORROBORATE an estimate, never to produce one.  Snapping a
    bar against a number and then treating the agreement as evidence for that
    number would be circular; what the residual does say is that an estimate
    landing within a few percent of ``standard_value / bar_px`` is consistent
    with a bar someone actually drew.
    """
    if not (bar_px and np.isfinite(bar_px) and bar_px > 0):
        return None, float("inf")
    implied = float(bar_px) * float(nmpp_estimate)
    best, best_rel = None, float("inf")
    for v in STANDARD_BAR_NM:
        rel = abs(v - implied) / v
        if rel < best_rel:
            best, best_rel = v, rel
    return (best, best_rel) if best_rel <= tol else (None, best_rel)


def _is_standard_bar_nm(nm: float, *, tol: float = 1e-6) -> bool:
    return any(abs(nm - v) <= tol * max(v, 1.0) for v in STANDARD_BAR_NM)


def bar_label_candidates(gray, footer_row, box=None, *, scale: int = 6
                         ) -> list[tuple[float, str]]:
    """Every plausible (nm, detail) reading of the bar label, not just the first.

    v4 returned the first match that fell in [1, 1e5] nm, which is how "1nm"
    won on B_4.  Collecting all of them lets the caller apply the standard-value
    and plausibility tests before choosing.
    """
    if footer_row is None:
        return []
    box = box if box is not None else detect_scale_bar_box(gray, footer_row)
    if box is None:
        return []
    strip = gray[footer_row:]
    x, y, w, h = box
    wl = "0123456789.numµ\u03bc "
    out: list[tuple[float, str]] = []
    seen: set[float] = set()
    for pad_x, pad_y, sc in ((45, 18, scale), (100, 22, scale),
                             (160, 26, scale + 2), (240, 30, scale + 2)):
        y0, y1 = max(0, y - pad_y), min(strip.shape[0], y + h + pad_y)
        x0, x1 = max(0, x - pad_x), min(strip.shape[1], x + w + pad_x)
        crop = strip[y0:y1, x0:x1]
        for psm in (7, 6, 11):
            txt = _ocr(_prep(crop, sc), psm, wl)
            for variant in _ocr_variants_v4(txt):
                for value, unit in _BAR_RE_V4.findall(variant):
                    try:
                        nm = float(value) * _UNIT_TO_NM[unit.lower()]
                    except (ValueError, KeyError):
                        continue
                    if 0.5 <= nm <= 1e6 and nm not in seen:
                        seen.add(nm)
                        out.append((nm, f"bar label '{value}{unit}' "
                                        f"(pad {pad_x}, psm {psm})"))
    return out


# --------------------------------------------------------------------------- #
# footer OCR memo -- cell 3 and cell 3c were paying for the same reads twice
# --------------------------------------------------------------------------- #
_FOOTER_CACHE_PATH = Path("outputs/_footer_ocr_cache.json")
_footer_cache: dict[str, Any] | None = None


def _footer_key(strip) -> str:
    import hashlib
    a = np.ascontiguousarray(strip)
    return hashlib.md5(a.tobytes()).hexdigest() + f"_{a.shape[0]}x{a.shape[1]}"


def _footer_cache_load() -> dict[str, Any]:
    global _footer_cache
    if _footer_cache is None:
        try:
            _footer_cache = json.loads(
                _FOOTER_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            _footer_cache = {}
    return _footer_cache


def footer_cache_save() -> None:
    if _footer_cache is None:
        return
    try:
        _FOOTER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _FOOTER_CACHE_PATH.write_text(json.dumps(_footer_cache), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        LOG.debug("could not save footer cache: %s", exc)


def read_footer_all(gray, footer_row) -> dict[str, Any]:
    """Everything readable from the footer, computed once and memoised.

    Returns ``{text, magnification, bar_box, bar_px, bar_candidates}``.
    """
    if footer_row is None:
        return {"text": "", "magnification": None, "bar_box": None,
                "bar_px": None, "bar_candidates": []}
    cache = _footer_cache_load()
    key = _footer_key(gray[footer_row:])
    hit = cache.get(key)
    if hit is not None:
        return hit
    text = read_footer_text_v4(gray, footer_row)
    box = detect_scale_bar_box(gray, footer_row)
    mag, _ = read_magnification(text)
    rec = {"text": text, "magnification": mag,
           "bar_box": list(box) if box else None,
           "bar_px": int(box[2]) if box else None,
           "bar_candidates": bar_label_candidates(gray, footer_row, box)}
    cache[key] = rec
    return rec


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #
def _plausible(nmpp) -> bool:
    return (nmpp is not None and np.isfinite(nmpp)
            and PLAUSIBLE_NMPP[0] <= nmpp <= PLAUSIBLE_NMPP[1])


def resolve_calibration(image_path, gray, *, image_id=None, override=None,
                        table=None, scale_bar_nm=None, mag_constant=None,
                        tol: float = 0.05, footer=None):
    """v6.4: three routes, arbitrated, with a plausibility floor.

    Order of authority:

    1. ``override`` / a hand-filled entry in ``calibration.yaml`` / a sidecar.
    2. ``FOV:<w>x<h><unit>`` from the footer, divided by the current pixel
       width.  Exact by construction and set in the large font.
    3. Magnification x the fitted ``FOV_nm x mag_k`` constant for this
       instrument.  Independent of the small font.
    4. The scale bar, but ONLY when its OCR'd label is a standard 1-2-5 value
       and the resulting nm/px is inside :data:`PLAUSIBLE_NMPP`.

    When 2 and 4 both produce a number and they disagree by more than ``tol``,
    route 3 arbitrates.  If it cannot, ``source`` becomes ``fov_text_disputed``
    and the caller is expected to keep that field out of nanometre reporting.
    """
    image_path = Path(image_path)
    image_id = image_id or image_path.stem
    footer_row = detect_footer_row(gray)
    width = int(gray.shape[1])

    if override is not None:
        if override <= 0:
            raise ValueError(f"nm_per_pixel must be > 0, got {override}")
        return Calibration(image_id, float(override), "override",
                           "supplied by caller", footer_row)
    if table and image_id in table and table[image_id] is not None:
        return Calibration(image_id, float(table[image_id]), "sidecar",
                           "project calibration table", footer_row)
    sidecar = image_path.with_suffix(".calib.json")
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            if "nm_per_pixel" in data:
                return Calibration(image_id, float(data["nm_per_pixel"]),
                                   "sidecar", str(sidecar), footer_row)
            if "fov_width_nm" in data:
                return Calibration(image_id, float(data["fov_width_nm"]) / width,
                                   "sidecar", f"{sidecar} fov_width_nm", footer_row)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("bad sidecar %s: %s", sidecar, exc)

    f = footer if footer is not None else read_footer_all(gray, footer_row)
    text = f.get("text", "")
    mag = f.get("magnification")
    bar_px = f.get("bar_px")

    # -- route 2 ----------------------------------------------------------
    fov_nmpp, fov_detail = from_fov_text(text, width)
    if fov_nmpp is not None and not _plausible(fov_nmpp):
        LOG.warning("%s: FOV text implies %.4g nm/px, outside %s -- ignoring it",
                    image_id, fov_nmpp, PLAUSIBLE_NMPP)
        fov_nmpp = None

    # -- route 3 ----------------------------------------------------------
    mag_nmpp = None
    if mag and mag_constant:
        cand = (float(mag_constant) / float(mag)) / float(width)
        mag_nmpp = cand if _plausible(cand) else None

    # -- route 4 ----------------------------------------------------------
    bar_nmpp, bar_detail = None, "no bar"
    if scale_bar_nm:
        if bar_px:
            bar_nmpp = float(scale_bar_nm) / bar_px
            bar_detail = f"{scale_bar_nm:g} nm / {bar_px} px (supplied)"
    elif bar_px:
        rejected = []
        for nm, detail in f.get("bar_candidates", []):
            cand = float(nm) / bar_px
            if not _is_standard_bar_nm(float(nm)):
                rejected.append(f"{nm:g}nm not a standard bar value")
                continue
            if not _plausible(cand):
                rejected.append(f"{nm:g}nm/{bar_px}px = {cand:.4g} nm/px implausible")
                continue
            bar_nmpp, bar_detail = cand, f"{nm:g} nm / {bar_px} px ({detail})"
            break
        if bar_nmpp is None and rejected:
            bar_detail = "bar label rejected: " + "; ".join(rejected[:3])
        elif bar_nmpp is None:
            bar_detail = "bar label unreadable"

    # -- arbitrate --------------------------------------------------------
    def _rel(a, b):
        return abs(a - b) / max(a, b) if (a and b) else float("inf")

    notes = []
    if mag:
        notes.append(f"x{mag:g}k")
    if bar_px:
        notes.append(f"bar {bar_px}px")

    if fov_nmpp is not None and bar_nmpp is not None:
        rel = _rel(fov_nmpp, bar_nmpp)
        if rel <= tol:
            return Calibration(image_id, float(fov_nmpp), "fov_text",
                               f"{fov_detail}; bar agrees within {100*rel:.1f}%"
                               f" ({bar_detail}); {', '.join(notes)}", footer_row)
        if mag_nmpp is not None:
            if _rel(mag_nmpp, fov_nmpp) <= tol:
                LOG.warning("%s: FOV and bar disagree by %.0f%%; magnification "
                            "backs the FOV -- taking it, bar ignored (%s)",
                            image_id, 100 * rel, bar_detail)
                return Calibration(image_id, float(fov_nmpp), "fov_text",
                                   f"{fov_detail}; bar DISAGREES {100*rel:.0f}% "
                                   f"but magnification agrees; {', '.join(notes)}",
                                   footer_row)
            if _rel(mag_nmpp, bar_nmpp) <= tol:
                LOG.warning("%s: FOV and bar disagree by %.0f%%; magnification "
                            "backs the BAR -- taking the bar (%s)",
                            image_id, 100 * rel, bar_detail)
                return Calibration(image_id, float(bar_nmpp), "scale_bar",
                                   f"{bar_detail}; FOV disagreed {100*rel:.0f}% "
                                   f"but magnification agrees; {', '.join(notes)}",
                                   footer_row)
        LOG.error("%s: FOV says %.4g nm/px, bar says %.4g (%.0f%% apart) and "
                  "nothing arbitrates. Marked DISPUTED -- do not quote nm for "
                  "this field until you type the number in by hand.",
                  image_id, fov_nmpp, bar_nmpp, 100 * rel)
        return Calibration(image_id, float(fov_nmpp), "fov_text_disputed",
                           f"{fov_detail}; bar says {bar_nmpp:.4g} nm/px "
                           f"({bar_detail}); {100*rel:.0f}% apart", footer_row)

    if fov_nmpp is not None:
        extra = ""
        if bar_px:
            snapped, res = snap_bar_nm(bar_px, fov_nmpp)
            extra = (f"; bar {bar_px}px consistent with a {snapped:g} nm bar "
                     f"({100*res:.1f}%)" if snapped else
                     f"; bar {bar_px}px matches no standard value at this scale")
        return Calibration(image_id, float(fov_nmpp), "fov_text",
                           f"{fov_detail}{extra}; {', '.join(notes)}", footer_row)

    if mag_nmpp is not None:
        extra = ""
        if bar_px:
            snapped, res = snap_bar_nm(bar_px, mag_nmpp)
            extra = (f"; bar {bar_px}px consistent with {snapped:g} nm "
                     f"({100*res:.1f}%)" if snapped else
                     f"; bar {bar_px}px matches no standard value")
        return Calibration(image_id, float(mag_nmpp), "magnification",
                           f"x{mag:g}k x fitted constant{extra}", footer_row)

    if bar_nmpp is not None:
        return Calibration(image_id, float(bar_nmpp), "scale_bar",
                           f"{bar_detail}; no FOV text, no magnification constant",
                           footer_row)

    LOG.warning("%s: pixel size UNKNOWN (bar_px=%s, mag=%s, footer_text=%s, %s)",
                image_id, bar_px, mag, bool(text.strip()), bar_detail)
    return Calibration(image_id, None, "unknown",
                       f"bar_px={bar_px}; mag={mag}; {bar_detail}", footer_row)


def calibrate_all(image_dir, out_yaml="calibration.yaml", *,
                  pattern=("*.tif", "*.tiff", "*.png", "*.jpg", "*.bmp"),
                  overwrite=False, existing=None, report_csv=None):
    """v6.4: read every footer once, fit the constant, then resolve.

    The v4 version resolved each image in isolation and only afterwards fitted
    the magnification constant, so the constant could never help an image whose
    bar had already produced a wrong-but-accepted number.  Here the footers are
    read first, the constant is fitted from the images whose FOV string parsed,
    and only then is each field resolved -- with the constant available to
    arbitrate.
    """
    from pathlib import Path as _P

    import pandas as pd
    import yaml

    from .utils import read_gray

    image_dir = _P(image_dir)
    files = sorted({p for pat in pattern for p in image_dir.rglob(pat)})
    if not files:
        LOG.warning("no images under %s", image_dir)
        return pd.DataFrame(columns=["image_id", "nm_per_pixel", "source", "detail"])

    known = dict(existing or {})
    if not known and _P(out_yaml).exists():
        known = {str(k): v for k, v in
                 (yaml.safe_load(_P(out_yaml).read_text(encoding="utf-8")) or {}).items()
                 if v is not None}

    # ---- pass 1: read every footer once -------------------------------- #
    scan = []
    for i, p in enumerate(files, 1):
        iid = p.stem
        try:
            gray = read_gray(p)
        except Exception as exc:  # noqa: BLE001
            scan.append({"image_id": iid, "_path": str(p), "_bad": str(exc)})
            continue
        frow = detect_footer_row(gray)
        f = read_footer_all(gray, frow)
        fov_nmpp, _ = from_fov_text(f["text"], int(gray.shape[1]))
        scan.append({"image_id": iid, "_path": str(p), "_bad": None,
                     "_footer": f, "_width": int(gray.shape[1]),
                     "magnification": f["magnification"],
                     "fov_width_nm": (fov_nmpp * gray.shape[1]
                                      if fov_nmpp is not None else None)})
        if i % 25 == 0:
            LOG.info("footers read %d/%d", i, len(files))
    footer_cache_save()

    # ---- fit FOV_nm x mag_k, per prefix group then folder-wide ---------- #
    def _group(iid):
        for sep in ("_", "-"):
            if sep in iid:
                return iid.split(sep)[0]
        return iid

    groups: dict[str, list] = {}
    for r in scan:
        groups.setdefault(_group(r["image_id"]), []).append(r)
    global_k, global_note = fit_magnification_constant(scan)
    group_k = {}
    for gname, grows in groups.items():
        k, note = fit_magnification_constant(grows)
        group_k[gname] = (k, f"group '{gname}': {note}") if k else \
            (global_k, f"folder-wide {global_note}" if global_k else "")

    # ---- pass 2: resolve ------------------------------------------------ #
    rows = []
    for r in scan:
        iid = r["image_id"]
        if r.get("_bad"):
            rows.append({"image_id": iid, "nm_per_pixel": None,
                         "source": "unreadable", "detail": r["_bad"],
                         "magnification": None, "bar_px": None})
            continue
        if iid in known and not overwrite:
            rows.append({"image_id": iid, "nm_per_pixel": float(known[iid]),
                         "source": "manual", "detail": "from calibration.yaml",
                         "magnification": r.get("magnification"),
                         "bar_px": r["_footer"].get("bar_px")})
            continue
        k, _note = group_k.get(_group(iid), (global_k, ""))
        gray = read_gray(r["_path"])
        c = resolve_calibration(r["_path"], gray, image_id=iid,
                                mag_constant=k, footer=r["_footer"])
        rows.append({"image_id": iid, "nm_per_pixel": c.nm_per_pixel,
                     "source": c.source, "detail": c.detail,
                     "magnification": r.get("magnification"),
                     "bar_px": r["_footer"].get("bar_px")})

    df = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    disputed = df[df.source == "fov_text_disputed"].image_id.tolist()
    table = {r.image_id: (float(r.nm_per_pixel)
                          if r.nm_per_pixel is not None
                          and np.isfinite(r.nm_per_pixel)
                          and r.source != "fov_text_disputed" else None)
             for r in df.itertuples()}
    _P(out_yaml).write_text(
        "# written by calibrate_all() [v6.4]; null means UNKNOWN or DISPUTED.\n"
        "# Type a number here and it wins over OCR on every later run.\n"
        + yaml.safe_dump(table, sort_keys=True, allow_unicode=True),
        encoding="utf-8")
    if report_csv:
        _P(report_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(report_csv, index=False)
    got = int(sum(v is not None for v in table.values()))
    LOG.info("calibration [v6.4]: %d/%d usable -> %s", got, len(df), out_yaml)
    LOG.info("by source: %s", df.source.value_counts().to_dict())
    if disputed:
        LOG.error("DISPUTED (excluded from nm reporting until you fill them in): %s",
                  disputed)
    return df


# ---- v6.4h ----
def load_calibration_table(path):
    """{image_id: nm_per_pixel} from YAML or JSON, skipping unusable entries.

    A ``null`` here is not an error: calibrate_all writes one for every field
    whose pixel size is unknown or disputed, and the correct response is to fall
    back to that image's own footer rather than to refuse to run. Non-numeric
    junk ("100nm", "?") is skipped the same way, with a warning naming the key,
    because a typo in a file the notebook keeps asking you to edit by hand
    should not take down inference.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        LOG.warning("calibration table %s not found", p)
        return {}
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    else:
        data = json.loads(p.read_text(encoding="utf-8"))

    out, skipped, bad = {}, [], []
    for k, v in data.items():
        if v is None:
            skipped.append(str(k))
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            bad.append(f"{k}={v!r}")
            continue
        if not np.isfinite(fv) or fv <= 0:
            bad.append(f"{k}={v!r}")
            continue
        out[str(k)] = fv
    if skipped:
        LOG.info("%s: %d field(s) still have no pixel size (%s%s) -- those "
                 "images fall back to their own footer",
                 p.name, len(skipped), ", ".join(skipped[:6]),
                 ", ..." if len(skipped) > 6 else "")
    if bad:
        LOG.warning("%s: ignoring %d unusable entr(y/ies): %s. A pixel size "
                    "must be a bare number, e.g.  '574_2': 2.0",
                    p.name, len(bad), "; ".join(bad[:6]))
    return out
