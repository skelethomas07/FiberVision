"""Recover measurement geometry from an annotated SEM overlay.

The problem this module solves
------------------------------
An ImageJ export may contain the *measurements* (length, angle) but not *where*
they were taken.  The only remaining record of position is the annotated PNG:
coloured markers plus a numbered black label box per measurement.  To turn that
back into training data we must

1. find the markers (sub-pixel, even when a label box was painted over them),
2. read the number printed next to each marker,
3. join number -> CSV row,
4. convert (marker, angle, length) into an oriented measurement segment,
5. produce a *clean* image with every overlay pixel removed / inpainted so the
   network never sees the annotation it is supposed to predict.

Reading the numbers
-------------------
The numerals are ~8 px tall, which defeats general-purpose OCR.  They are also
rendered by a deterministic bitmap font, so every instance of a given digit is
pixel-identical.  We exploit that: cluster all cleanly-segmented glyphs, keep
the ten largest clusters, name them once (via Tesseract on unambiguous
single-digit boxes, or from a supplied ``digit_templates.json``), then locate
every digit in the image by exact template correlation.  This handles touching
digits and merged boxes that Tesseract fails on, and it is auditable: the
learned templates are written to disk as PNGs.

Fallbacks are explicit, never silent -- if numbers cannot be read, the caller
is told and the annotations are written with ``annotation_confidence`` reflecting
the weaker association rule that was used instead.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .coords import (IMAGEJ, RASTER, chord_endpoints, fiber_angle_from_measurement,
                     imagej_to_raster, measurement_angle_from_endpoints,
                     structure_tensor_orientation, to_raster, wrap180)
from .utils import (Rect, angle_to_direction, angular_diff_180, ensure_dir,
                    get_logger, line_endpoints, read_gray, read_rgb, wrap_deg_180)

LOG = get_logger(__name__)

# --------------------------------------------------------------------------- #
# 1. overlay palette + marker detection
# --------------------------------------------------------------------------- #
def overlay_mask(rgb: np.ndarray, sat_tol: int = 25) -> np.ndarray:
    """Boolean mask of pixels whose channels disagree -> drawn annotation.

    Grayscale SEM data has R==G==B, so any chromatic pixel is overlay.  This is
    far more reliable than matching specific colours and works for any palette.
    """
    a = rgb.astype(np.int16)
    return ((np.abs(a[..., 0] - a[..., 1]) > sat_tol)
            | (np.abs(a[..., 1] - a[..., 2]) > sat_tol)
            | (np.abs(a[..., 0] - a[..., 2]) > sat_tol))


def discover_palette(rgb: np.ndarray, min_count: int = 30) -> list[tuple[int, int, int]]:
    """Return the distinct annotation colours present, most frequent first."""
    m = overlay_mask(rgb)
    if not m.any():
        return []
    cols, counts = np.unique(rgb[m].reshape(-1, 3), axis=0, return_counts=True)
    order = np.argsort(-counts)
    return [tuple(int(v) for v in cols[i]) for i in order if counts[i] >= min_count]


@dataclass
class Marker:
    x: float
    y: float
    color: tuple[int, int, int]
    area: int
    score: float
    occluded: bool = False


def detect_markers(rgb: np.ndarray, *, radius: float = 3.0, color_tol: int = 20,
                   score_thresh: float = 0.55, min_distance: int = 4,
                   palette: Sequence[tuple[int, int, int]] | None = None
                   ) -> list[Marker]:
    """Locate drawn marker dots by disk-template correlation on the colour mask.

    Correlation (rather than connected components) is used because label boxes
    are painted *on top* of markers: a partially covered dot still produces a
    strong local response, whereas its connected component is fragmented.
    """
    import cv2
    from skimage.feature import peak_local_max

    palette = list(palette) if palette else discover_palette(rgb)
    if not palette:
        LOG.warning("no annotation colours found in overlay")
        return []

    k = int(np.ceil(radius)) * 2 + 3
    yy, xx = np.mgrid[-(k // 2):k // 2 + 1, -(k // 2):k // 2 + 1]
    disk = (np.hypot(yy, xx) <= radius).astype(np.float32)
    disk /= disk.sum()

    markers: list[Marker] = []
    for color in palette:
        sel = (np.abs(rgb.astype(np.int16) - np.array(color, np.int16)).max(-1)
               <= color_tol).astype(np.float32)
        if sel.sum() < 4:
            continue
        resp = cv2.filter2D(sel, -1, disk, borderType=cv2.BORDER_CONSTANT)
        peaks = peak_local_max(resp, min_distance=min_distance,
                               threshold_abs=score_thresh, exclude_border=False)
        for (py, px) in peaks:
            # intensity-weighted centroid in a small window -> sub-pixel centre
            y0, y1 = max(0, py - 4), min(sel.shape[0], py + 5)
            x0, x1 = max(0, px - 4), min(sel.shape[1], px + 5)
            w = sel[y0:y1, x0:x1]
            tot = w.sum()
            if tot <= 0:
                continue
            gy = (w * np.arange(y0, y1)[:, None]).sum() / tot
            gx = (w * np.arange(x0, x1)[None, :]).sum() / tot
            score = float(resp[py, px])
            markers.append(Marker(float(gx), float(gy), tuple(color), int(tot),
                                  score, occluded=score < 0.9))
    LOG.info("detected %d markers across %d colour(s)", len(markers), len(palette))
    return markers



# --------------------------------------------------------------------------- #
# 1b. line-overlay extraction  (the preferred path)
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    """One drawn measurement chord recovered from a line overlay."""
    cx: float
    cy: float
    angle_deg: float
    length_px: float
    stroke_px: float
    n_pixels: int
    color: tuple[int, int, int]


def detect_measurement_segments(rgb: np.ndarray, *, palette: Sequence[tuple] | None = None,
                                min_pixels: int = 12, max_aspect: float = 0.85,
                                color_tol: int = 0) -> tuple[list[Segment], float]:
    """Recover every drawn measurement chord from a line overlay.

    This is the high-quality path.  When the annotation tool exports an overlay
    that draws the *lines* rather than only numbered markers, the geometry we
    need is already in the image: each connected chord gives its centre, its
    orientation and its length directly, with no OCR and no inference about what
    a marker means.

    The chord is stroked with a finite pen, so its connected component is longer
    than the chord by roughly one stroke width (the round end caps).  The stroke
    width is estimated robustly as the median perpendicular extent across all
    components and subtracted, which recovers the true chord length to a few
    tenths of a pixel.

    Returns ``(segments, stroke_width_px)``.
    """
    from scipy import ndimage as ndi

    if palette is None:
        palette = discover_palette(rgb)
    if not palette:
        return [], 0.0

    mask = np.zeros(rgb.shape[:2], bool)
    color_of = np.zeros(rgb.shape[:2], np.int16)
    for ci, color in enumerate(palette, start=1):
        if color_tol <= 0:
            sel = np.all(rgb == np.array(color, rgb.dtype), axis=-1)
        else:
            sel = np.abs(rgb.astype(np.int16)
                         - np.array(color, np.int16)).max(-1) <= color_tol
        mask |= sel
        color_of[sel & (color_of == 0)] = ci

    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    raw: list[dict[str, Any]] = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        sel = lab[sl] == i
        npx = int(sel.sum())
        if npx < min_pixels:
            continue
        ys, xs = np.nonzero(sel)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        pts = np.stack([xs, ys], axis=1).astype(np.float64)
        centre = pts.mean(0)
        centred = pts - centre
        _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
        along = centred @ vt[0]
        across = centred @ vt[1]
        extent = float(along.max() - along.min())
        width = float(across.max() - across.min())
        if extent <= 0 or width > max_aspect * extent:
            continue                       # a blob or a marker dot, not a chord
        cid = int(np.bincount(color_of[sl][sel]).argmax())
        raw.append({
            "cx": float(centre[0]), "cy": float(centre[1]),
            # raster convention: +x right, +y down.  Whether the CSV agrees is
            # decided empirically in match_segments_to_csv, never assumed.
            "angle": float(np.degrees(np.arctan2(vt[0][1], vt[0][0]))),
            "extent": extent, "width": width, "npx": npx,
            "color": palette[max(0, cid - 1)],
        })

    if not raw:
        return [], 0.0
    stroke = float(np.median([r["width"] for r in raw]))
    segments = [Segment(r["cx"], r["cy"], r["angle"],
                        max(0.0, r["extent"] - stroke), stroke, r["npx"],
                        tuple(int(v) for v in r["color"]))
                for r in raw]
    LOG.info("recovered %d measurement chords from the line overlay "
             "(stroke width %.1f px)", len(segments), stroke)
    return segments, stroke


def match_segments_to_csv(segments: Sequence[Segment], lengths: np.ndarray,
                          angles: np.ndarray, *, scale: float = 1.0,
                          y_sign: float = 1.0,
                          length_tol: float = 3.0, angle_tol: float = 25.0,
                          angle_weight: float = 0.1) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign drawn chords to CSV rows using geometry alone -- no OCR.

    Each chord has a measured length and orientation; each CSV row states a
    length and an angle.  A global Hungarian assignment on those two quantities
    pairs them uniquely.  This is both the association *and* its own validation:
    if the recovered lengths did not agree with the table, the match quality
    would collapse and we would know immediately.

    ``scale`` converts a CSV length into pixels (1 / nm_per_pixel when the table
    is in nanometres, 1.0 when it is already in pixels).

    Returns ``(row_index_per_segment, diagnostics)`` with -1 where unmatched.
    """
    from scipy.optimize import linear_sum_assignment

    if not segments or lengths.size == 0:
        return np.zeros(len(segments), int) - 1, {"n_matched": 0}

    seg_len = np.array([s.length_px for s in segments])
    seg_ang = np.array([s.angle_deg for s in segments])
    csv_len = np.asarray(lengths, float) * float(scale)
    # [v7] ``angles`` must ALREADY be raster (converted once with the fixed
    # ImageJ -> raster rule by the caller).  ``y_sign`` is kept only so the
    # opposite-sign match quality can be reported as a diagnostic; it is never
    # searched over to choose a convention.
    csv_ang = np.asarray(angles, float) * float(y_sign)

    d_len = np.abs(seg_len[:, None] - csv_len[None, :])
    d_ang = np.asarray(angular_diff_180(seg_ang[:, None], csv_ang[None, :]), float)
    cost = d_len + angle_weight * d_ang
    forbidden = (d_len > length_tol) | (d_ang > angle_tol)
    big = float(cost[~forbidden].max() if (~forbidden).any() else 1.0) * 1e3 + 1e6
    ri, ci = linear_sum_assignment(np.where(forbidden, big, cost))

    out = np.full(len(segments), -1, int)
    res_len, res_ang = [], []
    for r, c in zip(ri, ci):
        if forbidden[r, c]:
            continue
        out[r] = c
        res_len.append(d_len[r, c])
        res_ang.append(d_ang[r, c])
    diag = {
        "n_segments": len(segments), "n_csv_rows": int(csv_len.size),
        "n_matched": int((out >= 0).sum()),
        "median_length_residual_px": float(np.median(res_len)) if res_len else None,
        "median_angle_residual_deg": float(np.median(res_ang)) if res_ang else None,
        "length_correlation": (float(np.corrcoef(seg_len[out >= 0],
                                                 csv_len[out[out >= 0]])[0, 1])
                               if (out >= 0).sum() > 2 else None),
        "y_sign": float(y_sign),
    }
    LOG.info("geometry matching: %d/%d chords paired, length residual %.2f px, "
             "angle residual %.2f deg, r=%.3f", diag["n_matched"], len(segments),
             diag["median_length_residual_px"] or float("nan"),
             diag["median_angle_residual_deg"] or float("nan"),
             diag["length_correlation"] or float("nan"))
    return out, diag




def infer_csv_coordinate_scale(gray: np.ndarray, cx: np.ndarray, cy: np.ndarray,
                               angle_deg: np.ndarray, width: np.ndarray, *,
                               common_widths: Sequence[int] = (1200, 1024, 800, 640),
                               min_contrast: float = 5.0) -> dict[str, Any]:
    """Find the factor mapping CSV coordinates onto image pixels.

    A measuring tool often works on a *display-sized* copy of the image and
    writes coordinates in that frame.  VisionFlux analyses a 1200x900 resize, so
    its coordinates land at 15/16 of where the feature actually is -- a 6.7%
    error that puts every chord slightly off its fiber, quietly, with no symptom
    except worse training.

    Candidates come from the coordinate extent and from common display sizes;
    each is then *verified against the image* by asking whether chords at that
    scale actually span bright ridges.  Nothing is assumed: a candidate that
    cannot beat ``min_contrast`` is rejected and the scale stays 1.0.
    """
    h, w = gray.shape
    ok = np.isfinite(cx) & np.isfinite(cy) & np.isfinite(width) & (width > 0)
    if ok.sum() < 20:
        return {"scale": 1.0, "ok": False, "reason": "too few coordinates"}

    cands = {1.0}
    span_x, span_y = float(np.nanmax(cx[ok])), float(np.nanmax(cy[ok]))
    if span_x > 1:
        cands.add(w / span_x)
    if span_y > 1:
        cands.add(h / span_y)
    for cw in common_widths:
        if 0.5 < w / cw < 2.5:
            cands.add(w / cw)

    trials = []
    for sc in sorted(cands):
        geom = calibrate_marker_geometry(gray, cx * sc, cy * sc, angle_deg,
                                         width * sc, search=4.0, step=2.0)
        if not geom.get("ok"):
            continue
        ratio = geom.get("fwhm_over_width_median") or 3.0
        score = (geom["contrast"] - 2.0 * geom["asymmetry"]
                 - 40.0 * abs(ratio - 1.0))
        trials.append({"scale": float(sc), "score": float(score),
                       "contrast": geom["contrast"],
                       "asymmetry": geom["asymmetry"],
                       "fwhm_over_width_median": ratio,
                       "fwhm_agreement_frac": geom.get("fwhm_agreement_frac")})

    good = [t for t in trials if t["contrast"] >= min_contrast]
    if not good:
        LOG.warning("no coordinate scale put the chords on fibers "
                    "(best contrast %.1f); leaving coordinates unscaled",
                    max((t["contrast"] for t in trials), default=float("nan")))
        return {"scale": 1.0, "ok": False, "trials": trials,
                "reason": "no candidate reached the contrast floor"}
    best = max(good, key=lambda t: t["score"])
    if abs(best["scale"] - 1.0) > 0.01:
        LOG.info("CSV coordinates are in a %.0fx%.0f frame: scaling by %.4f "
                 "(contrast %.1f, FWHM/width %.2f)",
                 w / best["scale"], h / best["scale"], best["scale"],
                 best["contrast"], best["fwhm_over_width_median"])
    return {**best, "ok": True, "trials": trials}


def infer_scale_from_segments(segments: Sequence[Segment], lengths: np.ndarray,
                              angles: np.ndarray, *, span: float = 1.6,
                              n_grid: int = 121) -> dict[str, Any]:
    """Solve for how many CSV units there are per image pixel.

    The overlay states, in pixels, how long every measurement was; the table
    states the same measurements in its own units.  The ratio between them is a
    single unknown, and it is exactly the pixel size when the table is physical.

    This is a better calibrator than the printed scale bar.  Bar detection
    depends on footer layout and breaks silently on an unfamiliar microscope --
    on one of these images it read 32 px for a 100 nm bar (3.125 nm/px) where
    the chords say 2.25.  A wrong pixel size rescales every training target, so
    it is worth solving for rather than trusting.

    Returns the best units-per-pixel with its match quality.  A value near 1
    means the table was already in pixels.
    """
    if not segments or lengths.size == 0:
        return {"ok": False, "reason": "nothing to fit"}
    seg_med = float(np.median([s.length_px for s in segments]))
    csv_med = float(np.nanmedian(lengths))
    if seg_med <= 0 or not np.isfinite(csv_med) or csv_med <= 0:
        return {"ok": False, "reason": "degenerate lengths"}

    guess = csv_med / seg_med
    grid = np.unique(np.concatenate([
        guess * np.linspace(1.0 / span, span, n_grid), [1.0]]))
    best = None
    for upp in grid:                     # [v7] angles are raster already; no sign search
        _idx, diag = match_segments_to_csv(segments, lengths, angles,
                                           scale=1.0 / float(upp), y_sign=1.0)
        if not diag["n_matched"]:
            continue
        resid = diag["median_length_residual_px"] or 1e9
        score = diag["n_matched"] / (1.0 + resid)
        if best is None or score > best["score"]:
            best = {"units_per_pixel": float(upp), "y_sign": 1.0,
                    "score": float(score), **diag}
    if best is not None:
        # diagnostic only: how well would the MIRRORED convention have matched?
        _i2, d2 = match_segments_to_csv(segments, lengths, angles,
                                        scale=1.0 / best["units_per_pixel"], y_sign=-1.0)
        best["n_matched_if_sign_flipped"] = int(d2.get("n_matched", 0))
    if best is None:
        return {"ok": False, "reason": "no scale matched"}
    best["ok"] = True
    best["looks_like_pixels"] = abs(best["units_per_pixel"] - 1.0) < 0.04
    LOG.info("scale from overlay: %.4f CSV units per pixel (%s), %d chords "
             "matched, residual %.2f px", best["units_per_pixel"],
             "pixels" if best["looks_like_pixels"] else "physical units",
             best["n_matched"], best["median_length_residual_px"] or float("nan"))
    return best


def infer_units_from_segments(segments: Sequence[Segment], lengths: np.ndarray,
                              angles: np.ndarray, *,
                              nm_per_pixel: float | None) -> dict[str, Any]:
    """Pick the unit hypothesis that makes the drawn chords match the table.

    Far more direct than inferring units from image intensity: the overlay
    already states, in pixels, how long each measurement was.  Whichever
    interpretation of the CSV reproduces those pixel lengths is the right one.
    """
    trials: list[dict[str, Any]] = []
    for units, scale in (("pixels", 1.0),
                         ("nm", (1.0 / nm_per_pixel) if nm_per_pixel else None),
                         ("um", (1000.0 / nm_per_pixel) if nm_per_pixel else None)):
        if scale is None:
            trials.append({"units": units, "ok": False,
                           "reason": "no calibration available"})
            continue
        # [v7] angles are raster already (fixed conversion); only the units vary
        _idx, diag = match_segments_to_csv(segments, lengths, angles,
                                           scale=scale, y_sign=1.0)
        trials.append({"units": units, "ok": True, "scale": scale, **diag})
    viable = [t for t in trials if t.get("ok") and t.get("n_matched")]
    if not viable:
        return {"best": None, "candidates": trials}
    best = max(viable, key=lambda t: (t["n_matched"],
                                      -(t["median_length_residual_px"] or 1e9)))
    LOG.info("units from overlay geometry: '%s', y_sign %+.0f (%d chords matched, "
             "length residual %.2f px)", best["units"], best["y_sign"],
             best["n_matched"], best["median_length_residual_px"] or float("nan"))
    return {"best": best, "candidates": trials}


# --------------------------------------------------------------------------- #
# 2. label boxes + glyph-template OCR
# --------------------------------------------------------------------------- #
def detect_label_boxes(gray: np.ndarray, ov: np.ndarray, *, dark: float = 20.0,
                       min_area: int = 100, min_side: int = 10) -> list[Rect]:
    """Find the filled dark rectangles that carry the measurement numbers."""
    from scipy import ndimage as ndi

    mask = (gray < dark) & (~ov)
    lab, n = ndi.label(mask)
    out: list[Rect] = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if h < min_side or w < min_side or (lab[sl] == i).sum() < min_area:
            continue
        out.append(Rect(sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))
    LOG.info("detected %d label boxes", len(out))
    return out


def _segment_glyphs(gray: np.ndarray, box: Rect, text_thresh: float
                    ) -> list[tuple[np.ndarray, int, int]]:
    """Split one box into (patch, x_offset, y_offset) glyph candidates."""
    crop = gray[box.y0:box.y1, box.x0:box.x1]
    txt = crop > text_thresh
    if not txt.any():
        return []
    rows = np.flatnonzero(txt.any(1))
    r0, r1 = int(rows[0]), int(rows[-1])
    cols = txt.any(0)
    segs, cur = [], None
    for i, c in enumerate(cols):
        if c and cur is None:
            cur = i
        elif not c and cur is not None:
            segs.append((cur, i))
            cur = None
    if cur is not None:
        segs.append((cur, len(cols)))
    return [(crop[r0:r1 + 1, s:e], s, r0) for s, e in segs]


def name_template_bitmap(tmpl: np.ndarray, *,
                         scales: tuple[int, ...] = (8, 12, 16, 20, 28),
                         psms: tuple[str, ...] = ("10", "8", "7", "13", "6")
                         ) -> dict[str, int]:
    """Vote on the identity of a single glyph by OCR-ing the bitmap directly.

    Rendering the 8-px template at several scales gives Tesseract a much easier
    target than the original box crop, and majority voting across scale/PSM
    combinations is stable where any single call is not.
    """
    votes: dict[str, int] = {}
    try:
        import cv2
        import pytesseract
    except Exception:  # noqa: BLE001
        return votes
    img = (np.asarray(tmpl, np.uint8) * 255)
    cols = np.flatnonzero(img.any(0))
    if cols.size:
        img = img[:, cols[0]:cols[-1] + 1]
    for scale in scales:
        big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        big = cv2.copyMakeBorder(big, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=0)
        for psm in psms:
            try:
                txt = pytesseract.image_to_string(
                    255 - big,
                    config=f"--psm {psm} -c tessedit_char_whitelist=0123456789").strip()
            except Exception:  # noqa: BLE001
                continue
            txt = "".join(c for c in txt if c.isdigit())
            if len(txt) == 1:
                votes[txt] = votes.get(txt, 0) + 1
    return votes


def learn_digit_templates(gray: np.ndarray, boxes: Sequence[Rect], *,
                          text_thresh: float = 100.0, glyph_height: int = 8,
                          max_glyph_width: int = 6, min_support: int = 8
                          ) -> dict[str, np.ndarray]:
    """Cluster glyph bitmaps and name the ten largest clusters.

    Returns ``{digit_char: binary_template}``.  Naming uses Tesseract on boxes
    that contain exactly one glyph; if Tesseract is unavailable or names fewer
    than 10 clusters the result is returned partially filled and the caller must
    fall back.
    """
    clusters: dict[bytes, list[tuple[int, np.ndarray]]] = {}
    single_glyph_box: dict[bytes, list[int]] = {}

    for bi, box in enumerate(boxes):
        glyphs = _segment_glyphs(gray, box, text_thresh)
        if not glyphs:
            continue
        for patch, _sx, _sy in glyphs:
            if patch.shape[0] != glyph_height or patch.shape[1] > max_glyph_width:
                continue
            binp = np.zeros((glyph_height, max_glyph_width), np.uint8)
            binp[:, :patch.shape[1]] = (patch > text_thresh).astype(np.uint8)
            key = binp.tobytes()
            clusters.setdefault(key, []).append((bi, binp))
            if len(glyphs) == 1:
                single_glyph_box.setdefault(key, []).append(bi)

    ranked = sorted(clusters.items(), key=lambda kv: -len(kv[1]))
    ranked = [(k, v) for k, v in ranked if len(v) >= min_support][:10]
    LOG.info("glyph clustering: %d clusters with >= %d support", len(ranked), min_support)

    named: dict[str, np.ndarray] = {}
    try:
        import cv2
        import pytesseract

        for key, members in ranked:
            votes: dict[str, int] = {}
            for bi in single_glyph_box.get(key, [])[:12]:
                box = boxes[bi]
                crop = gray[box.y0:box.y1, box.x0:box.x1].astype(np.uint8)
                big = cv2.resize(crop, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
                big = cv2.copyMakeBorder(big, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=0)
                txt = pytesseract.image_to_string(
                    255 - big, config="--psm 10 -c tessedit_char_whitelist=0123456789"
                ).strip()
                txt = "".join(c for c in txt if c.isdigit())
                if len(txt) == 1:
                    votes[txt] = votes.get(txt, 0) + 1
            if votes:
                best = max(votes, key=votes.get)
                if best not in named:
                    named[best] = np.frombuffer(key, np.uint8).reshape(
                        glyph_height, max_glyph_width).copy()
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Tesseract naming unavailable (%s)", exc)

    unnamed = [np.frombuffer(k, np.uint8).reshape(glyph_height, max_glyph_width).copy()
               for k, _ in ranked
               if not any(np.array_equal(np.frombuffer(k, np.uint8).reshape(
                   glyph_height, max_glyph_width), t) for t in named.values())]

    # stage 2: whatever the box crops could not name, try naming the bitmap
    still: list[np.ndarray] = []
    for tmpl in unnamed:
        votes = {d: n for d, n in name_template_bitmap(tmpl).items() if d not in named}
        if votes:
            best = max(votes, key=votes.get)
            named[best] = tmpl
            LOG.info("template named '%s' from bitmap OCR (votes=%s)", best, votes)
        else:
            still.append(tmpl)

    LOG.info("named %d/10 digit templates: %s (%d unnamed cluster(s))",
             len(named), sorted(named), len(still))
    return named, still


def resolve_unnamed_templates(named: dict[str, np.ndarray],
                              unnamed: list[np.ndarray],
                              gray: np.ndarray, ov: np.ndarray,
                              valid_labels: set[int],
                              boxes: Sequence[Rect] | None = None, *,
                              max_permutations: int = 120,
                              **read_kwargs: Any) -> dict[str, np.ndarray]:
    """Name leftover glyph clusters by testing which assignment reads best.

    Tesseract sometimes refuses a couple of the tiny glyphs.  Rather than give
    up, we exploit a constraint we know must hold: every number printed on the
    overlay is a row id from the CSV, so the correct digit assignment is the one
    that maximises the count of *unique, in-range* numbers read.  With only a
    handful of unnamed clusters the search is exhaustive and cheap.
    """
    import itertools

    missing = [d for d in "0123456789" if d not in named]
    if not unnamed or not missing:
        return named
    best, best_score = dict(named), -1
    perms = list(itertools.permutations(missing, len(unnamed)))[:max_permutations]
    for perm in perms:
        cand = dict(named)
        for digit, tmpl in zip(perm, unnamed):
            cand[digit] = tmpl
        nums = read_numbers(gray, ov, cand, boxes, **read_kwargs)
        vals = [n.value for n in nums]
        uniq = set(vals)
        score = len(uniq & valid_labels) - 0.25 * (len(vals) - len(uniq))
        LOG.debug("template permutation %s -> score %.1f", perm, score)
        if score > best_score:
            best, best_score = cand, score
    LOG.info("resolved unnamed templates by consistency search (score=%.1f)", best_score)
    return best


def save_templates(templates: dict[str, np.ndarray], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    Path(path).write_text(json.dumps(
        {d: t.tolist() for d, t in templates.items()}), encoding="utf-8")


def load_templates(path: str | Path) -> dict[str, np.ndarray]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {d: np.asarray(v, np.uint8) for d, v in data.items()}


@dataclass
class ReadNumber:
    value: int
    x: float           # left edge of the first digit
    y: float           # top of the glyph row
    x_end: float
    score: float
    n_digits: int
    coverage: float = 1.0      # fraction of the row's ink explained by matched digits
    complete: bool = True


def read_numbers(gray: np.ndarray, ov: np.ndarray, templates: dict[str, np.ndarray],
                 boxes: Sequence[Rect] | None = None, *, text_thresh: float = 100.0,
                 glyph_height: int = 8, max_digits: int = 4) -> list[ReadNumber]:
    """Read the numbers printed on the overlay by exact glyph tiling.

    The overlay font is a deterministic bitmap -- every instance of a digit is
    pixel-identical -- so we use a stricter formulation than template
    *correlation*: within each text row we search for a left-to-right **tiling**
    of the row's ink by exact template matches, leaving no ink unexplained and
    no glyphs overlapping.

    Two failure modes motivate this:

    * the 3-px-wide "1" correlates above 0.9 against the vertical stroke inside
      "4", "7" and even a box edge, inventing numbers that never existed;
    * digits are sometimes rendered touching, so any method that demands a
      blank separator truncates "681" to "68" -- which then steals label 68
      from its true owner.

    This routine is deliberately **high precision, moderate recall**: a row it
    cannot tile exactly is reported as unreadable rather than guessed at.  The
    remaining labels are recovered afterwards by
    :func:`complete_labels_by_order`, which uses a rule these reads themselves
    establish and validate.
    """
    if not templates:
        return []
    if boxes is None:
        LOG.warning("read_numbers called without label boxes; nothing to search")
        return []

    valid = (gray > text_thresh) & (~ov)
    cores: list[tuple[str, np.ndarray]] = []
    for digit, tmpl in templates.items():
        t = np.asarray(tmpl, bool)
        cols = np.flatnonzero(t.any(0))
        if cols.size:
            cores.append((digit, t[:, cols[0]:cols[-1] + 1]))

    def tile(band: np.ndarray) -> list[tuple[str, int, int]] | None:
        ink = np.flatnonzero(band.any(0))
        if ink.size == 0:
            return None
        ink_set = {int(c) for c in ink}
        last = int(ink[-1])
        memo: dict[int, list[tuple[str, int, int]] | None] = {}

        def rec(pos: int) -> list[tuple[str, int, int]] | None:
            if pos > last:
                return []
            if pos in memo:
                return memo[pos]
            if pos not in ink_set:                       # inter-digit blank
                nxt = [c for c in ink_set if c > pos]
                memo[pos] = rec(min(nxt)) if nxt else []
                return memo[pos]
            best = None
            for digit, core in cores:
                w = core.shape[1]
                if pos + w > band.shape[1]:
                    continue
                if not np.array_equal(band[:, pos:pos + w], core):
                    continue
                rest = rec(pos + w)
                if rest is not None:
                    best = [(digit, pos, pos + w)] + rest
                    break
            memo[pos] = best
            return best

        return rec(int(ink[0]))

    numbers: list[ReadNumber] = []
    n_unreadable = 0
    for box in boxes:
        crop = valid[box.y0:box.y1, box.x0:box.x1]
        if not crop.any():
            continue
        rows = list(crop.any(1)) + [False]
        runs, start_r = [], None
        for i, r in enumerate(rows):
            if r and start_r is None:
                start_r = i
            elif not r and start_r is not None:
                runs.append((start_r, i))
                start_r = None
        for (r0, r1) in runs:
            if r1 - r0 != glyph_height:      # clipped or occluded text row
                n_unreadable += 1
                continue
            placed = tile(crop[r0:r1])
            if not placed or len(placed) > max_digits:
                n_unreadable += 1
                continue
            try:
                value = int("".join(d for d, _a, _b in placed))
            except ValueError:
                n_unreadable += 1
                continue
            numbers.append(ReadNumber(
                value, float(box.x0 + placed[0][1]), float(box.y0 + r0),
                float(box.x0 + placed[-1][2]), 1.0, len(placed), 1.0, True))

    LOG.info("read %d numbers exactly (%d text rows unreadable)",
             len(numbers), n_unreadable)
    return numbers


def complete_labels_by_order(assoc: list["Association"], markers: Sequence[Marker],
                             valid_labels: set[int], *,
                             max_inversion_rate: float = 0.25,
                             min_anchors: int = 30
                             ) -> tuple[list["Association"], dict[str, Any]]:
    """Fill in labels that OCR could not read, using the annotation order.

    Many annotation tools emit measurements in a deterministic spatial order.
    We do not assume that -- we *test* it on the exactly-read labels first.  If
    label id increases monotonically with marker y across the anchors, then any
    run of unlabelled markers lying between two anchors must carry exactly the
    labels between them; when the counts match, the assignment is forced and
    unique, so no guessing is involved.  When they do not match, the run is left
    unassigned and reported.
    """
    report: dict[str, Any] = {"applied": False, "reason": "", "n_filled": 0}
    anchors = {(round(a.x, 3), round(a.y, 3)): a.number for a in assoc}
    if len(assoc) < min_anchors:
        report["reason"] = f"only {len(assoc)} anchors (< {min_anchors})"
        return assoc, report

    pinned = sorted(assoc, key=lambda a: a.y)
    seq = [a.number for a in pinned]
    inversions = sum(1 for i in range(len(seq) - 1) if seq[i + 1] < seq[i])
    rate = inversions / max(1, len(seq) - 1)
    report["inversion_rate"] = rate
    if rate > max_inversion_rate:
        report["reason"] = (f"label id is not monotonic in y "
                            f"(inversion rate {rate:.2f}); order rule not applied")
        LOG.warning(report["reason"])
        return assoc, report

    order = sorted(markers, key=lambda m: m.y)
    idx_of_anchor: dict[int, int] = {}
    for i, m in enumerate(order):
        key = (round(m.x, 3), round(m.y, 3))
        if key in anchors:
            idx_of_anchor[i] = anchors[key]

    used = set(anchors.values())
    free = sorted(valid_labels - used)
    filled: list[Association] = []
    anchor_positions = sorted(idx_of_anchor)
    bounds = [(-1, 0)] + [(p, idx_of_anchor[p]) for p in anchor_positions]         + [(len(order), max(valid_labels) + 1)]
    for (i0, l0), (i1, l1) in zip(bounds, bounds[1:]):
        gap_idx = [i for i in range(i0 + 1, i1) if i not in idx_of_anchor]
        gap_labels = [l for l in free if l0 < l < l1]
        if not gap_idx:
            continue
        if len(gap_idx) != len(gap_labels):
            continue                         # ambiguous -- leave it alone
        for i, lab in zip(gap_idx, gap_labels):
            m = order[i]
            filled.append(Association(lab, m.x, m.y, m.score, 0.0, True, m.color))

    report.update(applied=True, n_filled=len(filled),
                  reason=f"order rule verified (inversion rate {rate:.3f})")
    LOG.info("order-based completion added %d labels (inversion rate %.3f)",
             len(filled), rate)
    return assoc + filled, report


# --------------------------------------------------------------------------- #
# 3. association: number -> marker
# --------------------------------------------------------------------------- #
def estimate_label_offset(numbers: Sequence[ReadNumber], markers: Sequence[Marker],
                          max_dist: float = 40.0) -> tuple[float, float]:
    """Median (dx, dy) from a number's top-left corner to its marker.

    The renderer used a constant offset between marker and text, so a robust
    median over all pairs recovers it without knowing the drawing code.
    """
    from scipy.spatial import cKDTree

    if not numbers or not markers:
        return 0.0, 0.0
    pts = np.array([[m.x, m.y] for m in markers])
    tree = cKDTree(pts)
    offs = []
    for nb in numbers:
        d, i = tree.query([nb.x, nb.y], distance_upper_bound=max_dist)
        if np.isfinite(d):
            offs.append(pts[i] - np.array([nb.x, nb.y]))
    if not offs:
        return 0.0, 0.0
    off = np.median(np.asarray(offs), axis=0)
    LOG.info("label->marker offset estimated at dx=%.2f dy=%.2f from %d pairs",
             off[0], off[1], len(offs))
    return float(off[0]), float(off[1])


@dataclass
class Association:
    number: int
    x: float
    y: float
    marker_score: float
    ocr_score: float
    matched_marker: bool
    color: tuple[int, int, int] | None = None


def associate_numbers_to_markers(numbers: Sequence[ReadNumber],
                                 markers: Sequence[Marker],
                                 *, max_dist: float = 18.0
                                 ) -> tuple[list[Association], list[dict[str, Any]]]:
    """One-to-one assignment of read numbers to detected markers.

    Uses the estimated constant label offset as the prior position and solves a
    global assignment so two numbers can never claim the same marker.  Numbers
    whose marker was fully painted over fall back to the predicted position and
    are flagged ``matched_marker=False``.
    """
    from scipy.optimize import linear_sum_assignment

    dx, dy = estimate_label_offset(numbers, markers)
    if not numbers:
        return [], [{"reason": "no_numbers_read", "detail": ""}]

    pred = np.array([[nb.x + dx, nb.y + dy] for nb in numbers])
    errors: list[dict[str, Any]] = []
    out: list[Association] = []

    if markers:
        pts = np.array([[m.x, m.y] for m in markers])
        cost = np.linalg.norm(pred[:, None, :] - pts[None, :, :], axis=-1)
        big = max_dist * 10.0
        cost_padded = np.where(cost <= max_dist, cost, big)
        ri, ci = linear_sum_assignment(cost_padded)
        assigned = {int(r): int(c) for r, c in zip(ri, ci)
                    if cost[r, c] <= max_dist}
    else:
        assigned = {}

    for i, nb in enumerate(numbers):
        if i in assigned:
            m = markers[assigned[i]]
            out.append(Association(nb.value, m.x, m.y, m.score, nb.score, True, m.color))
        else:
            out.append(Association(nb.value, float(pred[i, 0]), float(pred[i, 1]),
                                   0.0, nb.score, False, None))
            errors.append({"label": nb.value, "reason": "marker_not_found",
                           "detail": "position inferred from label offset"})

    seen: dict[int, Association] = {}
    for a in out:
        if a.number in seen:
            errors.append({"label": a.number, "reason": "duplicate_number_read",
                           "detail": f"second detection at ({a.x:.1f}, {a.y:.1f})"})
            if (a.ocr_score, a.marker_score) > (seen[a.number].ocr_score,
                                                 seen[a.number].marker_score):
                seen[a.number] = a
        else:
            seen[a.number] = a
    return list(seen.values()), errors


# --------------------------------------------------------------------------- #
# 4. clean-image reconstruction
# --------------------------------------------------------------------------- #
def clean_from_annotated(rgb: np.ndarray, *, inpaint_radius: int = 4,
                         dilate: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Remove every overlay pixel from an annotated image by inpainting.

    Returns ``(gray_clean, overlay_mask)``.  The mask is essential downstream:
    inpainted pixels are *not* evidence and must be excluded from loss and from
    any measurement made on this image.
    """
    import cv2

    ov = overlay_mask(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark_boxes = (gray < 20) & (~ov)
    from scipy import ndimage as ndi
    lab, n = ndi.label(dark_boxes)
    if n:
        sizes = np.asarray(ndi.sum(dark_boxes, lab, range(1, n + 1)))
        keep = 1 + np.flatnonzero(sizes >= 100)
        dark_boxes = np.isin(lab, keep)
    else:
        dark_boxes = np.zeros_like(ov)
    mask = ov | dark_boxes
    if dilate > 0:
        mask = ndi.binary_dilation(mask, np.ones((2 * dilate + 1, 2 * dilate + 1)))
    filled = cv2.inpaint(gray, mask.astype(np.uint8), inpaint_radius, cv2.INPAINT_TELEA)
    return filled.astype(np.float32), mask


# --------------------------------------------------------------------------- #
# 5. geometry + local fiber orientation
# --------------------------------------------------------------------------- #
def local_fiber_angle(gray: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                      *, sigma: float | np.ndarray = 3.0,
                      valid: np.ndarray | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Structure-tensor ridge orientation and its coherence at each point.

    Returns ``(angle_deg, coherence)``.  Coherence is
    ``(l1 - l2) / (l1 + l2)`` of the structure tensor: near 1 where a single
    ridge dominates, near 0 where several fibers cross or the texture is
    isotropic.  Reporting it matters -- in a dense network at low magnification
    the orientation estimate is often meaningless, and a caller that treats an
    incoherent estimate as fact will mislabel most of its data.

    ``sigma`` may be an array, one value per point, so the smoothing scale can
    follow the local fiber width instead of being fixed.
    """
    from skimage.feature import structure_tensor

    img = gray.astype(np.float64)
    if valid is not None and not valid.all():
        img = img.copy()
        img[~valid] = float(np.median(img[valid])) if valid.any() else 0.0

    yy = np.clip(np.round(ys).astype(int), 0, gray.shape[0] - 1)
    xx = np.clip(np.round(xs).astype(int), 0, gray.shape[1] - 1)
    sig = np.atleast_1d(np.asarray(sigma, float))
    if sig.size == 1:
        sig = np.full(xx.shape, float(sig[0]))

    ang = np.full(xx.shape, np.nan)
    coh = np.zeros(xx.shape)
    for s_val in np.unique(np.round(sig, 1)):
        sel = np.round(sig, 1) == s_val
        axx, axy, ayy = structure_tensor(img, sigma=max(1.0, float(s_val)),
                                         order="rc")
        theta = 0.5 * np.arctan2(2 * axy, (axx - ayy)) + np.pi / 2.0
        tr = axx + ayy
        det = axx * ayy - axy * axy
        disc = np.sqrt(np.maximum(0.0, tr * tr - 4 * det))
        c = np.where(tr > 1e-9, disc / np.maximum(tr, 1e-9), 0.0)
        ang[sel] = wrap_deg_180(-np.rad2deg(theta[yy[sel], xx[sel]]))
        coh[sel] = c[yy[sel], xx[sel]]
    return ang, coh


def resolve_orientation_convention(fiber_raw: np.ndarray, coherence: np.ndarray,
                                   measurement_deg: np.ndarray, *,
                                   coherence_min: float = 0.5
                                   ) -> dict[str, Any]:
    """Align the structure-tensor output with the table's angle convention.

    The sign and the 90-degree offset that relate skimage's structure-tensor
    orientation to "the direction the fiber runs, in the same frame the CSV uses"
    depend on axis order and on which convention the measuring tool wrote.  Both
    are easy to get backwards by reasoning and trivial to settle by measurement:
    a chord drawn across a fiber must come out perpendicular to that fiber, so we
    try the four candidate mappings and keep whichever makes that true.
    """
    sel = np.isfinite(fiber_raw) & np.isfinite(measurement_deg) & (coherence >= coherence_min)
    if sel.sum() < 20:
        return {"ok": False, "sign": 1.0, "offset": 0.0,
                "reason": f"only {int(sel.sum())} coherent sites"}
    target = measurement_deg[sel] - 90.0
    best = None
    for sign in (1.0, -1.0):
        for offset in (0.0, 90.0):
            cand = wrap_deg_180(sign * fiber_raw[sel] + offset)
            dev = np.asarray(angular_diff_180(target, cand), float)
            score = float(np.median(dev))
            if best is None or score < best["median_deviation_deg"]:
                best = {"sign": sign, "offset": offset,
                        "median_deviation_deg": score,
                        "frac_within_30deg": float(np.mean(dev < 30.0))}
    best["ok"] = True
    best["n_used"] = int(sel.sum())
    LOG.info("orientation convention: sign %+.0f, offset %.0f deg -> median "
             "chord/fiber deviation %.1f deg (%.0f%% within 30 deg, n=%d)",
             best["sign"], best["offset"], best["median_deviation_deg"],
             100 * best["frac_within_30deg"], best["n_used"])
    return best



# --------------------------------------------------------------------------- #
# 5b. empirical calibration of the marker/angle convention
# --------------------------------------------------------------------------- #
def calibrate_marker_geometry(gray: np.ndarray, cx: np.ndarray, cy: np.ndarray,
                              angle_deg: np.ndarray, width_px: np.ndarray, *,
                              search: float = 10.0, step: float = 1.0,
                              asymmetry_weight: float = 2.5,
                              y_signs: tuple[float, ...] = (1.0,)
                              ) -> dict[str, Any]:
    """Determine, from the image itself, what the marker and angle actually mean.

    Two conventions are unknowable from the files alone and catastrophic to get
    wrong:

    * whether the drawn marker is the centre of the measurement chord or is
      offset from it (renderers often anchor a glyph by its corner);
    * whether the reported angle uses raster (y down) or mathematical (y up)
      orientation -- a sign error that mirrors every measurement line.

    Both are decidable empirically.  A chord that truly spans a fiber produces a
    mean intensity profile that is *symmetric* about its centre, bright in
    ``|t| < 0.5`` and falling to background beyond it.  We scan candidate
    offsets and both sign conventions and keep the one maximising

        core brightness - background brightness - profile asymmetry

    with ``t`` measured in units of the reported width, so the test is scale
    free.  The chosen values are returned with their score so the decision is
    auditable rather than hidden.
    """
    import cv2

    H, W = gray.shape
    img = gray.astype(np.float32)
    ok = np.isfinite(cx) & np.isfinite(cy) & np.isfinite(angle_deg) & (width_px > 0)
    cx, cy, angle_deg, width_px = cx[ok], cy[ok], angle_deg[ok], width_px[ok]
    if cx.size < 20:
        return {"ok": False, "reason": f"only {cx.size} usable annotations"}

    ts = np.linspace(-1.5, 1.5, 61)
    core_sel = np.abs(ts) < 0.4
    back_sel = (np.abs(ts) > 0.8) & (np.abs(ts) < 1.3)

    def profile(dx: float, dy: float, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
        out = np.empty(ts.size, np.float64)
        for i, t in enumerate(ts):
            x = (cx + dx + ux * width_px * t).astype(np.float32)
            y = (cy + dy + uy * width_px * t).astype(np.float32)
            m = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            out[i] = (cv2.remap(img, x[m], y[m], cv2.INTER_LINEAR).ravel().mean()
                      if m.any() else np.nan)
        return out

    grid = np.arange(-search, search + step / 2, step)
    best: dict[str, Any] | None = None
    for y_sign in y_signs:
        ux, uy = angle_to_direction(angle_deg, y_sign)
        for dx in grid:
            for dy in grid:
                p = profile(float(dx), float(dy), ux, uy)
                if not np.isfinite(p).all():
                    continue
                core = float(p[core_sel].mean())
                back = float(p[back_sel].mean())
                asym = float(np.abs(p - p[::-1]).mean())
                # symmetry is weighted heavily on purpose: a chord that is
                # merely bright can be a fiber grazed off-centre, whereas a
                # symmetric bright core with matched shoulders is what a chord
                # spanning a fiber must look like.
                score = core - back - asymmetry_weight * asym
                if best is None or score > best["score"]:
                    best = {"score": score, "dx": float(dx), "dy": float(dy),
                            "y_sign": float(y_sign), "core": core,
                            "background": back, "asymmetry": asym,
                            "profile": p.tolist()}
    if best is None:
        return {"ok": False, "reason": "no candidate profile could be evaluated"}
    best["ok"] = True
    best["contrast"] = best["core"] - best["background"]
    best["n_used"] = int(cx.size)

    # quality gate: does the profile width of each individual chord actually
    # agree with the width the CSV reports for it?  A good pooled profile with
    # poor per-annotation agreement means the positions are only statistically
    # right, which is not good enough to regress against.
    ux, uy = angle_to_direction(angle_deg, best["y_sign"])
    fw = np.full(cx.shape, np.nan)
    tt = np.linspace(-2.0, 2.0, 81)
    prof_all = np.full((cx.size, tt.size), np.nan)
    for i, t in enumerate(tt):
        x = (cx + best["dx"] + ux * width_px * t).astype(np.float32)
        y = (cy + best["dy"] + uy * width_px * t).astype(np.float32)
        m = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if m.any():
            prof_all[m, i] = cv2.remap(img, x[m], y[m], cv2.INTER_LINEAR).ravel()
    centre = int(np.argmin(np.abs(tt)))
    for j, p in enumerate(prof_all):
        if not np.isfinite(p).all():
            continue
        peak = p[np.abs(tt) < 0.3].max()
        base = np.percentile(p, 10)
        half = 0.5 * (peak + base)
        if p[centre] < half:
            continue
        lo = centre
        while lo > 0 and p[lo] > half:
            lo -= 1
        hi = centre
        while hi < tt.size - 1 and p[hi] > half:
            hi += 1
        fw[j] = tt[hi] - tt[lo]
    good = np.isfinite(fw)
    best["fwhm_over_width_median"] = float(np.median(fw[good])) if good.any() else None
    best["fwhm_agreement_frac"] = (float(np.mean((fw[good] > 0.7) & (fw[good] < 1.4)))
                                   if good.any() else None)
    best["n_fwhm"] = int(good.sum())
    LOG.info("marker geometry: offset=(%.1f, %.1f) y_sign=%+.0f contrast=%.1f "
             "asymmetry=%.1f (n=%d)", best["dx"], best["dy"], best["y_sign"],
             best["contrast"], best["asymmetry"], best["n_used"])
    return best


# --------------------------------------------------------------------------- #
# 5c. what units is the measurement column in?
# --------------------------------------------------------------------------- #
#: How a reported ``Length`` maps onto pixels of the ORIGINAL (clean) image.
#: ``reg_scale`` converts annotated-image pixels to original-image pixels, so a
#: length measured on a resized overlay still lands correctly on the original.
LENGTH_UNITS = ("pixels", "nm", "um")


def length_to_original_px(length: np.ndarray | float, units: str, *,
                          reg_scale: float = 1.0,
                          nm_per_pixel: float | None = None) -> np.ndarray:
    """Convert a reported measurement into pixels of the original image.

    ``pixels`` means pixels *of the annotated overlay*, which is the common case
    when the measuring tool worked on a display copy: the overlay may have been
    resized relative to the original, so the registration scale is applied.
    ``nm`` / ``um`` are physical and need the calibration instead.
    """
    L = np.asarray(length, np.float64)
    if units == "pixels":
        return L * float(reg_scale)
    if nm_per_pixel is None or not np.isfinite(nm_per_pixel) or nm_per_pixel <= 0:
        return np.full_like(L, np.nan)
    factor = 1.0 if units == "nm" else 1000.0
    return L * factor / float(nm_per_pixel)


def infer_length_units(gray_original: np.ndarray, cx: np.ndarray, cy: np.ndarray,
                       angle_deg: np.ndarray, length: np.ndarray, *,
                       reg_scale: float = 1.0, nm_per_pixel: float | None = None,
                       candidates: Sequence[str] = LENGTH_UNITS,
                       search: float = 10.0, step: float = 2.0) -> dict[str, Any]:
    """Decide empirically whether ``Length`` is in pixels, nm or um.

    The header rarely says, the number alone is ambiguous, and guessing wrong
    rescales every training target by a constant factor -- which a network will
    happily learn, producing confident measurements that are uniformly wrong.

    The test is direct: under each hypothesis, convert the reported lengths to
    pixels and ask the image whether chords of that length actually span the
    fibers they sit on.  Only the correct hypothesis yields a symmetric
    intensity profile whose full width at half maximum matches the reported
    width; the others produce chords that are systematically too short or too
    long, which shows up immediately in the FWHM ratio.
    """
    results: list[dict[str, Any]] = []
    for units in candidates:
        w_px = length_to_original_px(length, units, reg_scale=reg_scale,
                                     nm_per_pixel=nm_per_pixel)
        if not np.isfinite(w_px).any():
            results.append({"units": units, "ok": False,
                            "reason": "needs a calibration that is not available"})
            continue
        med = float(np.nanmedian(w_px))
        if not (1.0 < med < 0.5 * min(gray_original.shape)):
            results.append({"units": units, "ok": False, "median_width_px": med,
                            "reason": "implied widths are not physically plausible"})
            continue
        geom = calibrate_marker_geometry(gray_original, cx, cy, angle_deg, w_px,
                                         search=search, step=step)
        if not geom.get("ok"):
            results.append({"units": units, "ok": False,
                            "reason": geom.get("reason", "")})
            continue
        ratio = geom.get("fwhm_over_width_median")
        score = (geom["contrast"]
                 - 40.0 * abs((ratio if ratio else 3.0) - 1.0)
                 - 2.0 * geom["asymmetry"])
        results.append({"units": units, "ok": True, "score": float(score),
                        "median_width_px": med, "contrast": geom["contrast"],
                        "asymmetry": geom["asymmetry"],
                        "fwhm_over_width_median": ratio,
                        "fwhm_agreement_frac": geom.get("fwhm_agreement_frac"),
                        "dx": geom["dx"], "dy": geom["dy"],
                        "y_sign": geom["y_sign"]})

    viable = [r for r in results if r.get("ok")]
    if not viable:
        LOG.warning("could not infer measurement units from the image; "
                    "set annotation_length_units explicitly")
        return {"best": None, "candidates": results}
    best = max(viable, key=lambda r: r["score"])
    runner_up = sorted((r["score"] for r in viable), reverse=True)
    margin = (runner_up[0] - runner_up[1]) if len(runner_up) > 1 else float("inf")
    LOG.info("measurement units inferred as '%s' (score %.1f, margin %.1f, "
             "FWHM/width %.2f)", best["units"], best["score"], margin,
             best["fwhm_over_width_median"] or float("nan"))
    if margin < 5.0:
        LOG.warning("units '%s' won by only %.1f -- treat this as uncertain and "
                    "confirm it against the verification overlay",
                    best["units"], margin)
    return {"best": best, "margin": float(margin), "candidates": results}


from .labels import LABEL_COLUMNS as LABELS_COLUMNS  # v7 schema (raster convention)


# --------------------------------------------------------------------------- #
# 6. driver: annotated image + CSV -> consolidated labels table
# --------------------------------------------------------------------------- #

_COORD_HINTS = ("center_x", "centerx", "centre_x", "x1", "bx", "xm")


def _header_has_coordinates(path: Path) -> bool:
    """Cheap header-only probe: does this table carry measurement positions?"""
    try:
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as fh:
            header = fh.readline().lower()
    except OSError:
        return False
    cells = re.split(r"[,\t;]", header)
    norm = {re.sub(r"[^a-z0-9_]", "", c.strip()) for c in cells}
    return any(h in norm for h in _COORD_HINTS)


def _pair_inputs(original_dir: Path | None, annotated_dir: Path, csv_dir: Path
                 ) -> dict[str, dict[str, Path | None]]:
    """Group original / annotated / csv files by image id."""
    from .utils import image_id_from_path, list_images

    pairs: dict[str, dict[str, Any]] = {}
    for p in list_images(annotated_dir):
        d = pairs.setdefault(image_id_from_path(p), {})
        d.setdefault("overlays", []).append(p)
        # a line-only overlay ("<id>_thickness.png") carries the geometry and is
        # always preferred over the labelled version, whose boxes hide the lines
        cur = d.get("annotated")
        if cur is None or ("labeled" in cur.stem.lower()
                           and "labeled" not in p.stem.lower()):
            d["annotated"] = p
    if original_dir is not None and Path(original_dir).is_dir():
        for p in list_images(original_dir):
            pairs.setdefault(image_id_from_path(p), {})["original"] = p
    for p in sorted(Path(csv_dir).glob("*")):
        if p.suffix.lower() not in (".csv", ".tsv", ".txt", ".xls", ".xlsx"):
            continue
        d = pairs.setdefault(image_id_from_path(p), {})
        d.setdefault("csvs", []).append(p)
        # An image can have several tables (a raw export and a reviewed one).
        # Prefer whichever carries coordinates: it removes the whole overlay
        # recovery step, so it is strictly better evidence.
        cur = d.get("csv")
        if cur is None or (_header_has_coordinates(p)
                           and not _header_has_coordinates(cur)):
            d["csv"] = p
    for d in pairs.values():
        d.setdefault("original", None)
        d.setdefault("annotated", None)
        d.setdefault("csv", None)
        d.setdefault("csvs", [])
        d.setdefault("overlays", [])
    return pairs



def refine_annotator_scale(segments: Sequence[Segment], lengths: np.ndarray,
                           idx: np.ndarray, *, min_pairs: int = 15,
                           max_rel_ci: float = 0.15) -> dict[str, Any]:
    """Annotator units-per-pixel as the Theil-Sen SLOPE of table length vs
    drawn length, with an intercept absorbing the constant drawn-length
    offset (stroke caps / detection shorten every chord by about a pixel).

    ``L_table = upp * (L_drawn + delta)`` gives slope ``upp`` and intercept
    ``upp * delta``; a plain median ratio would carry ``delta`` into the
    scale.  Falls back (``ok=False``) when the pairs are too few or the
    width spread is too small for a stable slope.
    """
    from scipy import stats

    d = np.array([s.length_px for s, j in zip(segments, idx) if j >= 0], float)
    c = np.array([lengths[j] for j in idx if j >= 0], float)
    ok = np.isfinite(d) & np.isfinite(c) & (d > 0) & (c > 0)
    d, c = d[ok], c[ok]
    if d.size < min_pairs or np.ptp(d) < 6.0:
        return {"ok": False, "reason": f"{d.size} pairs, spread {np.ptp(d) if d.size else 0:.1f} px",
                "n_pairs": int(d.size)}
    slope, intercept, lo, hi = stats.theilslopes(c, d)
    if slope <= 0 or (hi - lo) / slope > max_rel_ci:
        return {"ok": False, "reason": f"slope CI too wide ({lo:.3f}, {hi:.3f})",
                "n_pairs": int(d.size), "slope": float(slope)}
    return {"ok": True, "units_per_pixel": float(slope), "ci95": [float(lo), float(hi)],
            "drawn_length_offset_px": float(intercept / slope), "n_pairs": int(d.size),
            "method": "theil_sen_with_intercept"}


def choose_line_overlay(candidates: Sequence[Path], *,
                        lengths: np.ndarray | None = None,
                        angles: np.ndarray | None = None,
                        nm_per_pixel: float | None = None
                        ) -> tuple[Path | None, list[Segment], float, dict[str, Any]]:
    """Pick whichever supplied overlay actually reproduces the measurement table.

    A project often ships two renderings of the same annotation: one with the
    numbered label boxes for humans, and one with just the lines.  The boxes are
    painted over the lines, so the labelled version fragments them -- but it can
    still yield a *higher raw component count* than the clean version, because
    each broken piece counts separately.  Selecting on count therefore picks the
    wrong file.

    We select on the only thing that matters instead: how many chords can be
    matched back to rows of the CSV, and how closely their lengths agree.
    """
    best: tuple[Path | None, list[Segment], float, dict[str, Any]] = (None, [], 0.0, {})
    best_score = (-1, 1e9)
    for path in candidates:
        try:
            segs, stroke = detect_measurement_segments(read_rgb(path))
        except Exception as exc:  # noqa: BLE001
            LOG.warning("could not read overlay %s: %s", path, exc)
            continue
        if not segs:
            continue
        if lengths is None or angles is None:
            score = (len(segs), 0.0)
            report: dict[str, Any] = {}
        else:
            # [v7] free-scale geometric fit (the annotator's own units per pixel);
            # angles must already be raster.  No physical calibration is involved.
            fit = infer_scale_from_segments(segs, lengths, angles)
            report = {"best": fit if fit.get("ok") else None, "scale_fit": fit}
            if not fit.get("ok"):
                continue
            score = (fit["n_matched"], -(fit["median_length_residual_px"] or 1e9))
        if score > best_score:
            best_score = score
            best = (path, segs, stroke, report)
    if best[0] is not None:
        LOG.info("line overlay: %s -> %d chords, %s matched to the table",
                 best[0].name, len(best[1]),
                 best[3].get("best", {}).get("n_matched", "n/a")
                 if best[3] else "n/a")
    return best


def extract_one(image_id: str, annotated: Path, csv_path: Path | None,
                original: Path | None, *, nm_per_pixel: float | None = None,
                calib_table: dict[str, float] | None = None,
                templates: dict[str, np.ndarray] | None = None,
                marker_is_center: bool = True,
                marker_offset: tuple[float, float] = (0.0, 0.0),
                overlays: Sequence[Path] | None = None,
                auto_geometry: bool = True,
                coherence_min: float = 0.35,
                crossing_max_deg: float = 40.0,
                y_sign_default: float = 1.0,
                length_units: str = "auto",
                csv_angle_convention: str = IMAGEJ,
                debug_dir: Path | None = None) -> dict[str, Any]:
    """Recover all annotations for one image.  Returns a record with a DataFrame."""
    import pandas as pd

    from .calibration import Calibration, resolve_calibration, strip_footer
    from .csv_parser import infer_length_quantum, parse_measurement_csv
    from .image_registration import register

    import pandas as pd

    errors: list[dict[str, Any]] = []
    rgb = read_rgb(annotated)
    ov = overlay_mask(rgb)
    gray_clean, painted = clean_from_annotated(rgb)

    # ---- 1. the measurements themselves ----------------------------------
    if csv_path is None:
        raise FileNotFoundError(f"{image_id}: no measurement CSV found")
    parsed = parse_measurement_csv(csv_path)
    errors.extend([{**e, "image_id": image_id, "source_csv": str(csv_path)}
                   for e in parsed.errors])
    csv_df = parsed.frame.set_index("label")
    valid_labels = {int(v) for v in csv_df.index}
    quantum = infer_length_quantum(parsed.frame["length"].to_numpy())

    # ---- 2. geometry recovered from the overlay --------------------------
    markers = detect_markers(rgb)
    gray_rgb = np.asarray(rgb).mean(-1)
    boxes = detect_label_boxes(gray_rgb, ov)
    if templates:
        tmpl = templates
    else:
        named, unnamed = learn_digit_templates(gray_rgb, boxes)
        tmpl = resolve_unnamed_templates(named, unnamed, gray_rgb, ov,
                                         valid_labels, boxes)
    numbers = read_numbers(gray_rgb, ov, tmpl, boxes)
    numbers = [n for n in numbers if n.value in valid_labels]
    assoc, assoc_err = associate_numbers_to_markers(numbers, markers)
    errors.extend([{**e, "image_id": image_id} for e in assoc_err])
    n_exact = len(assoc)
    assoc, order_report = complete_labels_by_order(assoc, markers, valid_labels)
    if not order_report.get("applied"):
        errors.append({"image_id": image_id, "reason": "order_rule_not_applied",
                       "detail": order_report.get("reason", "")})

    # ---- 3. calibration ---------------------------------------------------
    if original is not None:
        orig_gray = read_gray(original)
        calib = resolve_calibration(original, orig_gray, image_id=image_id,
                                    override=nm_per_pixel, table=calib_table)
        orig_body, footer_row = strip_footer(orig_gray)
        reg = register(gray_clean, orig_body, overlay_valid=~painted)
        if not reg.ok:
            errors.append({"image_id": image_id, "reason": "registration_failed",
                           "detail": reg.detail})
    else:
        orig_gray = None
        orig_body = gray_clean
        footer_row = None
        calib = resolve_calibration(annotated, gray_clean, image_id=image_id,
                                    override=nm_per_pixel, table=calib_table)
        from .image_registration import Registration
        reg = Registration(np.array([[1, 0, 0], [0, 1, 0]], float), "no_original",
                           1.0, 1.0, 0.0, (0.0, 0.0), True,
                           "clean image reconstructed from the overlay itself")

    nm_per_px = calib.nm_per_pixel if calib.known else None      # PHYSICAL only
    reg_scale = float(reg.scale) if reg.ok else 1.0
    if csv_angle_convention not in (IMAGEJ, RASTER):
        raise ValueError(f"csv_angle_convention must be {IMAGEJ!r} or {RASTER!r}")

    # [v7] ONE fixed conversion of the table's angle column into raster degrees.
    # Nothing below fits a sign or an offset per field.
    csv_ang_src = csv_df["angle"].to_numpy(float)
    csv_ang_raster = np.asarray(to_raster(csv_ang_src, csv_angle_convention), float)
    csv_df = csv_df.assign(angle_raster=csv_ang_raster)

    def _row_record(label, cx, cy, meas_raster, width_px, *, source_angle, extraction_path,
                    conf=1.0, is_negative=False, annotator_upp=np.nan, width_px_drawn=np.nan,
                    source_len=np.nan, units_str="unknown", endpoints=None):
        meas_raster = float(wrap180(meas_raster)) if np.isfinite(meas_raster) else np.nan
        if endpoints is None:
            if np.isfinite(meas_raster):
                x1, y1, x2, y2 = chord_endpoints(cx, cy, meas_raster, width_px)
            else:
                x1 = y1 = x2 = y2 = np.nan
        else:
            x1, y1, x2, y2 = endpoints
            meas_raster = float(measurement_angle_from_endpoints(x1, y1, x2, y2))
        fib = float(fiber_angle_from_measurement(meas_raster)) if np.isfinite(meas_raster) else np.nan
        return {
            "image_id": image_id, "annotation_id": int(label),
            "center_x_px": float(cx), "center_y_px": float(cy),
            "x1_px": float(x1), "y1_px": float(y1), "x2_px": float(x2), "y2_px": float(y2),
            "measurement_angle_raster_deg": meas_raster, "fiber_angle_raster_deg": fib,
            "width_px": float(width_px), "width_nm": np.nan, "nm_per_pixel": np.nan,
            "calibration_status": "unaudited", "calibration_valid": False,
            "source_angle_deg": float(source_angle) if np.isfinite(source_angle) else np.nan,
            "angle_source_convention": csv_angle_convention,
            "imagej_angle_deg": (float(source_angle) if (csv_angle_convention == IMAGEJ
                                 and np.isfinite(source_angle)) else np.nan),
            "angle_convention_residual_deg": np.nan,
            "source_length": float(source_len), "source_length_units": units_str,
            "annotator_units_per_px": float(annotator_upp), "width_px_drawn": float(width_px_drawn),
            "tensor_fiber_angle_raster_deg": np.nan, "orientation_coherence": np.nan,
            "chord_vs_tensor_deg": np.nan,
            "annotation_confidence": float(conf), "ambiguous_crossing": False,
            "is_negative": bool(is_negative), "extraction_path": extraction_path,
            "source_csv": str(csv_path),
        }

    line_report: dict[str, Any] = {"used": False}
    scale_report: dict[str, Any] = {}
    units_report: dict[str, Any] = {}
    geom: dict[str, Any] = {"ok": False}
    units = length_units
    annotator_upp = np.nan
    order_report = {"applied": False, "reason": ""}
    rows: list[dict[str, Any]] = []

    # ---- 3a. the table already carries coordinates ------------------------
    if parsed.has_coordinates:
        units = "pixels" if length_units == "auto" else length_units
        keep = csv_df["status"].astype(str).str.lower().ne("rejected") \
            if "status" in csv_df.columns else pd.Series(True, index=csv_df.index)
        coord_fit = infer_csv_coordinate_scale(
            orig_body, csv_df.loc[keep, "cx"].to_numpy(float),
            csv_df.loc[keep, "cy"].to_numpy(float),
            csv_df.loc[keep, "angle_raster"].to_numpy(float),
            csv_df.loc[keep, "length"].to_numpy(float))
        coord_scale = float(coord_fit["scale"])
        if not coord_fit.get("ok"):
            errors.append({"image_id": image_id, "reason": "coordinate_scale_unverified",
                           "detail": coord_fit.get("reason", "")})
        have_ep = {"x1", "y1", "x2", "y2"} <= set(csv_df.columns)
        for label, row in csv_df.iterrows():
            status = str(row.get("status", "")).strip().lower()
            is_negative = status == "rejected"
            mx, my = reg.apply(np.array([float(row["cx"]) * coord_scale]),
                               np.array([float(row["cy"]) * coord_scale]))
            cx, cy = float(mx[0]), float(my[0])
            L = float(row["length"]) * coord_scale
            if units == "pixels":
                width_px = L * reg_scale
            else:
                width_px = np.nan      # physical lengths without a drawn chord: unusable
            ep = None
            if have_ep:
                ex1, ey1 = reg.apply(np.array([float(row["x1"]) * coord_scale]),
                                     np.array([float(row["y1"]) * coord_scale]))
                ex2, ey2 = reg.apply(np.array([float(row["x2"]) * coord_scale]),
                                     np.array([float(row["y2"]) * coord_scale]))
                ep = (float(ex1[0]), float(ey1[0]), float(ex2[0]), float(ey2[0]))
                width_px = float(np.hypot(ep[2] - ep[0], ep[3] - ep[1]))
            conf = row.get("confidence", 1.0)
            conf = float(conf) if np.isfinite(pd.to_numeric(conf, errors="coerce")) else 1.0
            rows.append(_row_record(label, cx, cy, float(row["angle_raster"]), width_px,
                                    source_angle=float(row["angle"]), extraction_path="csv_coordinates",
                                    conf=conf, is_negative=is_negative, source_len=float(row["length"]),
                                    units_str=units, endpoints=ep))
        geom = {"ok": True, "source": "csv_coordinates"}
        units_report = {"best": {"units": units, "source": "csv coordinates"},
                        "coordinate_scale": {k: v for k, v in coord_fit.items() if k != "trials"}}
        line_report = {"used": False, "reason": "CSV carries coordinates"}
        n_exact = sum(1 for r in rows if not r["is_negative"])
    else:
        # ---- 3b. the overlay draws the measurement chords --------------------
        csv_len = csv_df["length"].to_numpy(float)
        line_path, segments, stroke, u = choose_line_overlay(
            overlays or [annotated], lengths=csv_len, angles=csv_ang_raster,
            nm_per_pixel=None)
        line_report = {"used": False, "n_segments": len(segments)}
        if len(segments) >= max(20, 0.2 * len(csv_df)):
            # The annotator's own scale: CSV units per OVERLAY pixel, solved from
            # the drawing itself.  It converts the table's lengths back into the
            # pixels they were drawn at.  It is provenance -- it is compared with
            # the physical scale in the audit and NEVER substituted for it.
            fit = infer_scale_from_segments(segments, csv_len, csv_ang_raster)
            scale_report = fit
            if fit.get("ok"):
                annotator_upp = float(fit["units_per_pixel"])
                if length_units == "auto":
                    units = "pixels" if fit["looks_like_pixels"] else "physical"
                scale = 1.0 / annotator_upp
            else:
                u2 = infer_units_from_segments(segments, csv_len, csv_ang_raster,
                                               nm_per_pixel=nm_per_px)
                units_report = u2
                units = u2["best"]["units"] if u2.get("best") else "pixels"
                scale = float(u2["best"]["scale"]) if u2.get("best") else 1.0
                annotator_upp = 1.0 / scale if scale else np.nan
            idx, diag = match_segments_to_csv(segments, csv_len, csv_ang_raster,
                                              scale=scale, y_sign=1.0)
            # [v7] refine the annotator scale with an intercept: the drawn chord
            # is systematically ~1 px shorter than the table length / scale
            # (stroke caps, detection), which biases a plain ratio by 5-10 %.
            refine = refine_annotator_scale(segments, csv_len, idx)
            scale_report = {**scale_report, "refined": refine}
            if refine.get("ok"):
                annotator_upp = float(refine["units_per_pixel"])
                scale = 1.0 / annotator_upp
                idx, diag = match_segments_to_csv(segments, csv_len, csv_ang_raster,
                                                  scale=scale, y_sign=1.0)
            _i2, diag_flip = match_segments_to_csv(segments, csv_len, csv_ang_raster,
                                                   scale=scale, y_sign=-1.0)
            line_report.update(used=True, overlay=str(line_path), stroke_px=stroke,
                               units=units, annotator_units_per_px=annotator_upp,
                               n_matched_if_sign_flipped=int(diag_flip.get("n_matched", 0)),
                               **diag)
            if diag.get("n_matched", 0) < diag_flip.get("n_matched", 0):
                errors.append({"image_id": image_id, "reason": "angle_convention_mismatch",
                               "detail": (f"fixed {csv_angle_convention} conversion matched "
                                          f"{diag.get('n_matched')} chords but the mirrored "
                                          f"convention matched {diag_flip.get('n_matched')}; "
                                          "check the export's angle convention")})
            labels_by_pos = list(csv_df.index)
            seen: set[int] = set()
            residuals = []
            for seg, j in zip(segments, idx):
                if j < 0:
                    errors.append({"image_id": image_id, "reason": "chord_unmatched",
                                   "detail": f"drawn chord at ({seg.cx:.0f}, {seg.cy:.0f}) matched no CSV row"})
                    continue
                label = int(labels_by_pos[j])
                if label in seen:
                    continue
                seen.add(label)
                row = csv_df.loc[label]
                cx, cy = reg.apply(np.array([seg.cx]), np.array([seg.cy]))
                cx, cy = float(cx[0]), float(cy[0])
                width_px = float(row["length"]) * scale * reg_scale
                drawn_px = float(seg.length_px) * reg_scale
                ang_r = float(row["angle_raster"]) if np.isfinite(row["angle_raster"]) else float(seg.angle_deg)
                rec = _row_record(label, cx, cy, ang_r, width_px, source_angle=float(row["angle"]),
                                  extraction_path="line_overlay", annotator_upp=annotator_upp,
                                  width_px_drawn=drawn_px, source_len=float(row["length"]),
                                  units_str=units)
                res = float(angular_diff_180(seg.angle_deg, ang_r))
                rec["angle_convention_residual_deg"] = res
                residuals.append(res)
                rows.append(rec)
            for missing in sorted(set(csv_df.index) - seen):
                errors.append({"image_id": image_id, "label": int(missing),
                               "reason": "annotation_not_located",
                               "detail": "CSV row has no drawn chord in the overlay"})
            line_report["median_angle_convention_residual_deg"] = (
                float(np.median(residuals)) if residuals else None)
            geom = {"ok": True, "source": "line_overlay"}
            n_exact = len(rows)
        else:
            # ---- 4. markers + OCR: positions from markers, angles from the table -----
            probe_x, probe_y, probe_a, probe_L = [], [], [], []
            for a in assoc:
                if a.number in csv_df.index:
                    r = csv_df.loc[a.number]
                    probe_x.append(a.x + marker_offset[0])
                    probe_y.append(a.y + marker_offset[1])
                    probe_a.append(float(r["angle_raster"]))
                    probe_L.append(float(r["length"]))
            probe_x, probe_y = np.asarray(probe_x), np.asarray(probe_y)
            probe_a, probe_L = np.asarray(probe_a), np.asarray(probe_L)
            if len(probe_x):
                probe_x, probe_y = reg.apply(probe_x, probe_y)
            if auto_geometry and len(probe_x) >= 20:
                if units == "auto":
                    units_report = infer_length_units(
                        orig_body, probe_x, probe_y, probe_a, probe_L,
                        reg_scale=reg_scale, nm_per_pixel=nm_per_px)
                    units = units_report["best"]["units"] if units_report.get("best") else "pixels"
                    if not units_report.get("best"):
                        errors.append({"image_id": image_id, "reason": "units_not_inferred",
                                       "detail": "defaulting to pixels"})
                probe_w = length_to_original_px(probe_L, units, reg_scale=reg_scale,
                                                nm_per_pixel=nm_per_px)
                geom = calibrate_marker_geometry(orig_body, probe_x, probe_y, probe_a, probe_w,
                                                 y_signs=(1.0,))
                if geom.get("ok"):
                    marker_offset = (marker_offset[0] + geom["dx"], marker_offset[1] + geom["dy"])
            else:
                geom = {"ok": False, "reason": "auto_geometry disabled or too few probes"}
                if units == "auto":
                    units = "pixels"
            if units not in LENGTH_UNITS:
                raise ValueError(f"length_units must be one of {LENGTH_UNITS} or 'auto'")
            if units in ("nm", "um") and nm_per_px is None:
                errors.append({"image_id": image_id, "reason": "physical_length_without_calibration",
                               "detail": "marker-path table is in physical units but no physical "
                                         "nm/px is known; widths cannot be expressed in pixels"})
            seen: set[int] = set()
            for a in assoc:
                if a.number not in csv_df.index:
                    errors.append({"image_id": image_id, "label": a.number,
                                   "reason": "number_not_in_csv", "detail": "read from overlay but absent"})
                    continue
                seen.add(a.number)
                row = csv_df.loc[a.number]
                mx, my = a.x + marker_offset[0], a.y + marker_offset[1]
                cx, cy = reg.apply(np.array([mx]), np.array([my]))
                cx, cy = float(cx[0]), float(cy[0])
                width_px = float(length_to_original_px(float(row["length"]), units,
                                                       reg_scale=reg_scale, nm_per_pixel=nm_per_px))
                ang_r = float(row["angle_raster"])
                if not marker_is_center and np.isfinite(ang_r):
                    x2, y2, _, _ = chord_endpoints(cx, cy, ang_r, width_px * 2)
                    cx, cy = (cx + x2) / 2.0, (cy + y2) / 2.0
                conf = float(min(1.0, 0.5 * a.ocr_score
                                 + 0.5 * (a.marker_score if a.matched_marker else 0.0)))
                rows.append(_row_record(a.number, cx, cy, ang_r, width_px, source_angle=float(row["angle"]),
                                        extraction_path="marker_ocr", conf=conf,
                                        source_len=float(row["length"]), units_str=units))
            for missing in sorted(set(csv_df.index) - seen):
                errors.append({"image_id": image_id, "label": int(missing),
                               "reason": "annotation_not_located",
                               "detail": "CSV row has no recoverable marker in the overlay"})
            n_exact = len(rows)

    from .labels import LABEL_COLUMNS, ensure_schema
    df = ensure_schema(pd.DataFrame(rows, columns=LABEL_COLUMNS)) if rows else \
        ensure_schema(pd.DataFrame(columns=LABEL_COLUMNS))
    df = df[np.isfinite(df["width_px"].to_numpy(float)) & (df["width_px"].to_numpy(float) > 0)] \
        .reset_index(drop=True)

    # ---- 6. structure-tensor fibre orientation: fixed raster mapping, diagnostic only ----
    orientation_check: dict[str, Any] | None = None
    if len(df):
        body_valid = np.ones(orig_body.shape, bool)
        if original is None:
            body_valid = ~painted[:orig_body.shape[0], :orig_body.shape[1]]
        img_t = orig_body.astype(np.float64)
        if not body_valid.all():
            img_t = img_t.copy()
            img_t[~body_valid] = float(np.median(img_t[body_valid])) if body_valid.any() else 0.0
        t_ang, t_coh = structure_tensor_orientation(img_t / 255.0, sigma=2.0, grad_sigma=1.0)
        xs = np.clip(np.rint(df["center_x_px"].to_numpy(float)), 0, orig_body.shape[1] - 1).astype(int)
        ys = np.clip(np.rint(df["center_y_px"].to_numpy(float)), 0, orig_body.shape[0] - 1).astype(int)
        ta, tc = t_ang[ys, xs].astype(float), t_coh[ys, xs].astype(float)
        fib = df["fiber_angle_raster_deg"].to_numpy(float)
        dev = np.asarray(angular_diff_180(fib, ta), float)
        df["tensor_fiber_angle_raster_deg"] = ta
        df["orientation_coherence"] = tc
        df["chord_vs_tensor_deg"] = dev
        coherent = tc >= max(coherence_min, 0.5)
        df["ambiguous_crossing"] = (tc >= coherence_min) & (dev > crossing_max_deg)
        df.loc[~np.isfinite(df["measurement_angle_raster_deg"]), "ambiguous_crossing"] = True
        if "is_negative" in df.columns:
            df.loc[df["is_negative"].astype(bool), "ambiguous_crossing"] = False
        if coherent.sum() >= 20:
            d_ok = dev[coherent]
            # what the MIRRORED convention would have scored, as a diagnostic
            mirrored = np.asarray(angular_diff_180(wrap180(-fib[coherent]), ta[coherent]), float)
            orientation_check = {"n_coherent": int(coherent.sum()),
                                 "median_chord_vs_tensor_deg": float(np.median(d_ok)),
                                 "frac_within_30deg": float(np.mean(d_ok < 30.0)),
                                 "median_if_mirrored_deg": float(np.median(mirrored)),
                                 "convention_used": csv_angle_convention,
                                 "fitted": False}
            if np.median(mirrored) + 5.0 < np.median(d_ok):
                errors.append({"image_id": image_id, "reason": "angle_convention_mismatch",
                               "detail": (f"chords vs tensor: {np.median(d_ok):.1f} deg with the fixed "
                                          f"{csv_angle_convention} conversion, {np.median(mirrored):.1f} "
                                          "deg if mirrored -- the export may not use that convention")})
        else:
            orientation_check = {"n_coherent": int(coherent.sum()), "reason": "too few coherent sites",
                                 "convention_used": csv_angle_convention, "fitted": False}

    if debug_dir is not None:
        import cv2
        ensure_dir(debug_dir)
        cv2.imwrite(str(Path(debug_dir) / f"{image_id}_clean_reconstructed.png"),
                    gray_clean.astype(np.uint8))
        cv2.imwrite(str(Path(debug_dir) / f"{image_id}_overlay_mask.png"),
                    (painted * 255).astype(np.uint8))

    meta = {
        "image_id": image_id,
        "annotated": str(annotated),
        "original": str(original) if original else None,
        "csv": str(csv_path),
        "n_csv_rows": int(len(csv_df)),
        "n_markers_detected": len(markers),
        "n_label_boxes": len(boxes),
        "n_numbers_read": len(numbers),
        "n_annotations_recovered": int(len(df)),
        "n_read_exactly": int(n_exact),
        "order_completion": order_report,
        "recovery_rate": float(len(df) / max(1, len(csv_df))),
        "digit_templates_named": sorted(tmpl),
        "calibration": calib.to_dict(),                 # PHYSICAL route only (footer / table)
        "physical_nm_per_px": nm_per_px,
        "annotator_units_per_px": (float(annotator_upp) if np.isfinite(annotator_upp) else None),
        "length_units": units,
        "csv_angle_convention": csv_angle_convention,
        "extraction_path": geom.get("source", "marker_ocr" if geom.get("ok") else "unknown"),
        "registration": reg.to_dict(),
        "length_quantum": quantum,
        "marker_geometry": {k: v for k, v in geom.items() if k not in ("profile", "t")},
        "length_units_inference": units_report,
        "overlay_to_original_scale": reg_scale,
        "marker_offset": list(marker_offset),
        "orientation_check": orientation_check,
        "csv_columns": parsed.raw_columns,
        "csv_has_coordinates": parsed.has_coordinates,
        "line_overlay": line_report,
        "scale_fit": {k: v for k, v in scale_report.items() if k != "profile"},
        "footer_row": footer_row,
    }
    return {"labels": df, "meta": meta, "errors": errors,
            "clean": gray_clean, "painted": painted, "markers": markers}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    import pandas as pd

    from .calibration import load_calibration_table
    from .utils import save_json, set_seed

    ap = argparse.ArgumentParser(description="Recover measurement annotations "
                                             "from annotated SEM overlays.")
    ap.add_argument("--annotated_dir", required=True)
    ap.add_argument("--csv_dir", required=True)
    ap.add_argument("--original_dir", default=None,
                    help="clean SEM images; omit to reconstruct them by inpainting")
    ap.add_argument("--output_csv", default="data/processed/labels.csv")
    ap.add_argument("--nm_per_pixel", type=float, default=None)
    ap.add_argument("--calibration_table", default=None)
    ap.add_argument("--marker_at_endpoint", action="store_true",
                    help="markers mark a line endpoint rather than its centre")
    ap.add_argument("--marker_dx", type=float, default=0.0)
    ap.add_argument("--marker_dy", type=float, default=0.0)
    ap.add_argument("--length_units", default="auto",
                    choices=("auto", "pixels", "nm", "um"),
                    help="units of the CSV Length column; 'pixels' means pixels "
                         "of the ANNOTATED overlay")
    ap.add_argument("--no_auto_geometry", action="store_true",
                    help="skip the empirical marker-offset/angle-sign calibration")
    ap.add_argument("--debug_dir", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(argv)

    set_seed(args.seed)
    table = load_calibration_table(args.calibration_table)
    pairs = _pair_inputs(Path(args.original_dir) if args.original_dir else None,
                         Path(args.annotated_dir), Path(args.csv_dir))
    if not pairs:
        LOG.error("no inputs found")
        return 2

    frames, metas, errors = [], [], []
    for image_id, files in sorted(pairs.items()):
        if files["annotated"] is None or files["csv"] is None:
            errors.append({"image_id": image_id, "reason": "incomplete_triplet",
                           "detail": f"annotated={files['annotated']}, csv={files['csv']}"})
            LOG.warning("%s: incomplete (annotated=%s csv=%s) -- skipped",
                        image_id, files["annotated"], files["csv"])
            continue
        try:
            rec = extract_one(image_id, files["annotated"], files["csv"],
                              files["original"], overlays=files.get("overlays"),
                              nm_per_pixel=args.nm_per_pixel,
                              calib_table=table,
                              marker_is_center=not args.marker_at_endpoint,
                              marker_offset=(args.marker_dx, args.marker_dy),
                              auto_geometry=not args.no_auto_geometry,
                              length_units=args.length_units,
                              debug_dir=Path(args.debug_dir) if args.debug_dir else None)
        except Exception as exc:  # noqa: BLE001 - one bad image must not kill the run
            LOG.exception("%s failed", image_id)
            errors.append({"image_id": image_id, "reason": "extraction_exception",
                           "detail": str(exc)})
            continue
        frames.append(rec["labels"])
        metas.append(rec["meta"])
        errors.extend(rec["errors"])

    out = Path(args.output_csv)
    ensure_dir(out.parent)
    all_df = (pd.concat(frames, ignore_index=True) if frames
              else pd.DataFrame(columns=LABELS_COLUMNS))
    all_df.to_csv(out, index=False)
    save_json(metas, out.with_name(out.stem + "_meta.json"))
    pd.DataFrame(errors).to_csv(out.with_name(out.stem + "_errors.csv"), index=False)
    LOG.info("wrote %d annotations from %d image(s) -> %s (%d issues logged)",
             len(all_df), len(frames), out, len(errors))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
