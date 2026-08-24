from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage


YELLOW = np.asarray([255.0, 211.0, 70.0], dtype=np.float32)
CYAN = np.asarray([26.0, 220.0, 235.0], dtype=np.float32)
MEASUREMENT_COLUMNS = [
    "measurement_id", "x1", "y1", "x2", "y2", "center_x", "center_y",
    "width", "angle", "source", "status",
]


@dataclass
class ResumeAnalysisResult:
    """AnalysisResult-compatible container for a previously annotated image.

    Resume mode intentionally does not run the fibre detector again. Existing
    VisionFlux yellow/cyan chords are reconstructed from the raster image and the
    normal review UI can then append new manual measurements.
    """

    image: np.ndarray
    image_name: str
    analysis_scale: float
    original_shape: tuple[int, int]
    summary: dict
    measurements: pd.DataFrame
    regions: pd.DataFrame
    representatives: pd.DataFrame
    candidates: pd.DataFrame
    diagnostic_png: bytes | None = None
    orientation: object | None = None
    calibration: object | None = None
    quality: object | None = None
    footer_start_y: int | None = None
    full_original_shape: tuple[int, int] | None = None
    original_image: np.ndarray | None = None


def _load_rgb(data: bytes, filename: str) -> np.ndarray:
    suffix = Path(filename).suffix.lower()
    if suffix in {".tif", ".tiff"}:
        # PIL supports the common RGB/8-bit TIFF files produced by VisionFlux.
        image = Image.open(BytesIO(data)).convert("RGB")
    else:
        image = Image.open(BytesIO(data)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _annotation_masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(rgb[..., :3], dtype=np.float32)
    spread = arr.max(axis=2) - arr.min(axis=2)
    yellow_distance = np.linalg.norm(arr - YELLOW[None, None, :], axis=2)
    cyan_distance = np.linalg.norm(arr - CYAN[None, None, :], axis=2)

    # Exact PNG exports are captured by the distance term. The loose channel
    # constraints keep the importer usable after a light JPEG recompression while
    # still rejecting ordinary grayscale SEM pixels.
    yellow = (
        (spread >= 45)
        & (yellow_distance <= 90)
        & (arr[..., 0] >= 175)
        & (arr[..., 1] >= 135)
        & (arr[..., 2] <= 165)
    )
    cyan = (
        (spread >= 45)
        & (cyan_distance <= 95)
        & (arr[..., 0] <= 155)
        & (arr[..., 1] >= 145)
        & (arr[..., 2] >= 155)
    )
    return yellow, cyan


def _axial_error(a: float, b: float) -> float:
    d = abs(float(a) - float(b)) % 180.0
    return min(d, 180.0 - d)


def _segment_angle(p1: np.ndarray, p2: np.ndarray) -> float:
    dx, dy = float(p2[0] - p1[0]), float(p2[1] - p1[1])
    return float((math.degrees(math.atan2(-dy, dx)) + 180.0) % 180.0)


def _component_segment(coords_yx: np.ndarray) -> dict | None:
    if len(coords_yx) < 7:
        return None
    points = np.column_stack([coords_yx[:, 1], coords_yx[:, 0]]).astype(float)
    center = points.mean(axis=0)
    centered = points - center
    cov = centered.T @ centered / max(len(points) - 1, 1)
    values, vectors = np.linalg.eigh(cov)
    major = float(np.max(values))
    minor = float(np.min(values))
    anisotropy = major / max(minor, 1e-9)
    axis = vectors[:, int(np.argmax(values))]
    if axis[0] < 0 or (abs(axis[0]) < 1e-9 and axis[1] < 0):
        axis = -axis
    projections = centered @ axis
    lo = float(np.percentile(projections, 1.0))
    hi = float(np.percentile(projections, 99.0))
    raw_length = hi - lo
    if raw_length < 2.5:
        return None
    # VisionFlux draws a 3-pixel endpoint dot. Pull each outer extreme inward a
    # little so the reconstructed chord is close to the original click position.
    shrink = min(3.0, 0.16 * raw_length)
    if raw_length > 2.0 * shrink + 1.0:
        lo += shrink
        hi -= shrink
    p1 = center + lo * axis
    p2 = center + hi * axis
    return {
        "p1": p1,
        "p2": p2,
        "center": center,
        "axis": axis,
        "angle": _segment_angle(p1, p2),
        "length": float(np.linalg.norm(p2 - p1)),
        "pixels": int(len(points)),
        "anisotropy": float(anisotropy),
    }


def _interval_on_axis(segment: dict, axis: np.ndarray) -> tuple[float, float]:
    a = float(np.dot(segment["p1"], axis))
    b = float(np.dot(segment["p2"], axis))
    return (min(a, b), max(a, b))


def _merge_pair(a: dict, b: dict) -> dict:
    a_round = float(a.get("anisotropy", 99.0)) < 1.8
    b_round = float(b.get("anisotropy", 99.0)) < 1.8
    if a_round and not b_round:
        axis = np.asarray(b["axis"], float)
    elif b_round and not a_round:
        axis = np.asarray(a["axis"], float)
    else:
        axis = np.asarray(a["axis"], float) + np.asarray(b["axis"], float)
    if np.linalg.norm(axis) < 1e-9:
        axis = np.asarray(a["axis"], float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    if axis[0] < 0 or (abs(axis[0]) < 1e-9 and axis[1] < 0):
        axis = -axis
    pts_a = np.asarray([a.get("center")], float) if a_round else np.vstack([a["p1"], a["p2"]])
    pts_b = np.asarray([b.get("center")], float) if b_round else np.vstack([b["p1"], b["p2"]])
    points = np.vstack([pts_a, pts_b])
    normal = np.asarray([-axis[1], axis[0]])
    normal_pos = float(np.mean(points @ normal))
    projections = points @ axis
    lo, hi = float(np.min(projections)), float(np.max(projections))
    p1 = axis * lo + normal * normal_pos
    p2 = axis * hi + normal * normal_pos
    return {
        "p1": p1,
        "p2": p2,
        "center": 0.5 * (p1 + p2),
        "axis": axis,
        "angle": _segment_angle(p1, p2),
        "length": float(np.linalg.norm(p2 - p1)),
        "pixels": int(a["pixels"] + b["pixels"]),
        "anisotropy": max(float(a.get("anisotropy", 1.0)), float(b.get("anisotropy", 1.0))),
    }


def _pair_score(a: dict, b: dict) -> float | None:
    a_round = float(a.get("anisotropy", 99.0)) < 1.8
    b_round = float(b.get("anisotropy", 99.0)) < 1.8
    if a_round and b_round:
        return None

    # A label may leave one side as little more than the endpoint dot. In that
    # case use the elongated half's direction and connect the round component if
    # it lies on the same line just beyond the cut.
    if a_round != b_round:
        line = b if a_round else a
        dot = a if a_round else b
        axis = np.asarray(line["axis"], float)
        axis /= max(float(np.linalg.norm(axis)), 1e-9)
        normal = np.asarray([-axis[1], axis[0]])
        midpoint = 0.5 * (np.asarray(line["p1"]) + np.asarray(line["p2"]))
        dot_center = np.asarray(dot.get("center"), float)
        normal_distance = abs(float(np.dot(dot_center - midpoint, normal)))
        if normal_distance > 5.5:
            return None
        lo, hi = _interval_on_axis(line, axis)
        t = float(np.dot(dot_center, axis))
        gap = lo - t if t < lo else (t - hi if t > hi else 0.0)
        if gap <= 0.8 or gap > 34.0:
            return None
        return float(gap + 2.0 * normal_distance)

    if _axial_error(a["angle"], b["angle"]) > 9.0:
        return None
    axis = np.asarray(a["axis"], float) + np.asarray(b["axis"], float)
    if np.linalg.norm(axis) < 1e-9:
        axis = np.asarray(a["axis"], float)
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    normal = np.asarray([-axis[1], axis[0]])
    ma = 0.5 * (np.asarray(a["p1"]) + np.asarray(a["p2"]))
    mb = 0.5 * (np.asarray(b["p1"]) + np.asarray(b["p2"]))
    normal_distance = abs(float(np.dot(mb - ma, normal)))
    if normal_distance > 4.5:
        return None
    ia = _interval_on_axis(a, axis)
    ib = _interval_on_axis(b, axis)
    if ia[1] < ib[0]:
        gap = ib[0] - ia[1]
    elif ib[1] < ia[0]:
        gap = ia[0] - ib[1]
    else:
        gap = 0.0
    if gap <= 1.0 or gap > 28.0:
        return None
    merged_extent = max(ia[1], ib[1]) - min(ia[0], ib[0])
    if merged_extent > 140.0:
        return None
    return float(gap + 2.0 * normal_distance + 0.2 * _axial_error(a["angle"], b["angle"]))


def _segments_from_mask(mask: np.ndarray) -> list[dict]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    segments: list[dict] = []
    for label_id in range(1, int(count) + 1):
        coords = np.argwhere(labels == label_id)
        segment = _component_segment(coords)
        if segment is not None:
            segments.append(segment)

    # The labeled PNG is drawn line -> black label box -> white label. The box can
    # split one colored chord into two components, so pair the best collinear halves.
    while True:
        best = None
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                score = _pair_score(segments[i], segments[j])
                if score is not None and (best is None or score < best[0]):
                    best = (score, i, j)
        if best is None:
            break
        _, i, j = best
        merged = _merge_pair(segments[i], segments[j])
        segments = [seg for k, seg in enumerate(segments) if k not in {i, j}] + [merged]

    # Stable display ordering and a minimum useful chord size.
    segments = [seg for seg in segments if seg["length"] >= 3.0]
    segments.sort(key=lambda seg: (
        float(0.5 * (seg["p1"][1] + seg["p2"][1])),
        float(0.5 * (seg["p1"][0] + seg["p2"][0])),
    ))
    return segments


def _stable_measurement_id(source: str, p1: np.ndarray, p2: np.ndarray) -> str:
    a = np.asarray(p1, float)
    b = np.asarray(p2, float)
    if tuple(a) > tuple(b):
        a, b = b, a
    text = f"{source}:{a[0]:.2f},{a[1]:.2f}:{b[0]:.2f},{b[1]:.2f}"
    return f"resume-{source}-{sha1(text.encode('utf-8')).hexdigest()[:12]}"


def _measurement_row(segment: dict, source: str, index: int, nm_per_px: float | None) -> dict:
    p1 = np.asarray(segment["p1"], float)
    p2 = np.asarray(segment["p2"], float)
    x1, y1 = map(float, p1)
    x2, y2 = map(float, p2)
    width = float(math.hypot(x2 - x1, y2 - y1))
    chord_angle = math.degrees(math.atan2(-(y2 - y1), x2 - x1))
    tangent = float((chord_angle + 180.0) % 180.0 - 90.0)
    measurement_id = _stable_measurement_id(source, p1, p2)
    region_id = f"resume-{index:04d}-{measurement_id[-6:]}"
    return {
        "measurement_id": measurement_id,
        "fiber_region_id": region_id,
        "fiber_path_id": None,
        "direction_segment_id": 0,
        "region_sample_index": 0,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "center_x": 0.5 * (x1 + x2),
        "center_y": 0.5 * (y1 + y2),
        "xm": 0.5 * (x1 + x2),
        "ym": 0.5 * (y1 + y2),
        "width_px": width,
        "width_original_px": width,
        "width_nm": np.nan if nm_per_px is None else width * float(nm_per_px),
        "direction_deg": tangent,
        "local_orientation_deg": np.nan,
        "local_coherency": np.nan,
        "orientation_error_deg": np.nan,
        "orientation_score": 1.0,
        "grade": "IMPORTED",
        "confidence": 1.0,
        "sem_agreement": 1.0,
        "bundle_score": 0.0,
        "status": "active",
        "source": source,
        "review_label": "imported",
        "replacement_for": None,
    }


def _metadata_measurement_row(item: dict, index: int, nm_per_px: float | None) -> dict | None:
    try:
        x1, y1, x2, y2 = (float(item[k]) for k in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        return None
    if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    width = float(math.hypot(x2 - x1, y2 - y1))
    if width <= 0:
        return None
    source = "manual" if str(item.get("source", "auto")) == "manual" else "auto"
    direction = _finite(item.get("direction_deg"))
    if direction is None:
        chord_angle = math.degrees(math.atan2(-(y2 - y1), x2 - x1))
        direction = float((chord_angle + 180.0) % 180.0 - 90.0)
    measurement_id = str(item.get("measurement_id") or _stable_measurement_id(source, np.asarray([x1, y1]), np.asarray([x2, y2])))
    region_id = f"resume-meta-{index:04d}-{measurement_id[-6:]}"
    return {
        "measurement_id": measurement_id,
        "fiber_region_id": region_id,
        "fiber_path_id": None,
        "direction_segment_id": 0,
        "region_sample_index": 0,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "center_x": 0.5 * (x1 + x2), "center_y": 0.5 * (y1 + y2),
        "xm": 0.5 * (x1 + x2), "ym": 0.5 * (y1 + y2),
        "width_px": width, "width_original_px": width,
        "width_nm": np.nan if nm_per_px is None else width * float(nm_per_px),
        "direction_deg": float(direction),
        "local_orientation_deg": np.nan, "local_coherency": np.nan,
        "orientation_error_deg": np.nan, "orientation_score": 1.0,
        "grade": "IMPORTED", "confidence": 1.0, "sem_agreement": 1.0, "bundle_score": 0.0,
        "status": "active", "source": source,
        "review_label": str(item.get("review_label") or "imported_metadata"),
        "replacement_for": item.get("replacement_for"),
    }


def try_build_resume_analysis(
    data: bytes,
    filename: str,
    *,
    nm_per_px: float | None = None,
) -> ResumeAnalysisResult | None:
    """Open a VisionFlux annotated result as an editable continuation session.

    Returns None for a normal grayscale SEM so the caller can run the ordinary
    detector. If VisionFlux yellow/cyan chords are present, no fibre detection is
    performed; the colored measurements are reconstructed and returned directly.
    """
    pil = Image.open(BytesIO(data))
    metadata_text = pil.info.get("visionflux_measurements_v1")
    rgb = np.asarray(pil.convert("RGB"), dtype=np.uint8)

    if metadata_text:
        try:
            payload = json.loads(metadata_text)
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, list):
            rows = []
            for index, item in enumerate(payload, start=1):
                if isinstance(item, dict):
                    row = _metadata_measurement_row(item, index, nm_per_px)
                    if row is not None:
                        rows.append(row)
            if rows:
                measurements = pd.DataFrame(rows)
                h, w = int(rgb.shape[0]), int(rgb.shape[1])
                summary = {
                    "resume_mode": True,
                    "resume_source": "png_metadata",
                    "imported_measurements": int(len(measurements)),
                    "imported_auto": int((measurements["source"] == "auto").sum()),
                    "imported_manual": int((measurements["source"] == "manual").sum()),
                    "nm_per_original_px": None if nm_per_px is None else float(nm_per_px),
                    "footer_removed_px": 0,
                    "detector": "resume-import-no-reanalysis",
                }
                return ResumeAnalysisResult(
                    image=rgb, image_name=str(filename), analysis_scale=1.0,
                    original_shape=(h, w), full_original_shape=(h, w), summary=summary,
                    measurements=measurements, regions=pd.DataFrame(), representatives=pd.DataFrame(),
                    candidates=pd.DataFrame(), diagnostic_png=None, orientation=None, calibration=None,
                    quality=None, footer_start_y=None, original_image=rgb,
                )

    yellow_mask, cyan_mask = _annotation_masks(rgb)
    color_pixels = int(yellow_mask.sum() + cyan_mask.sum())
    if color_pixels < 12:
        return None

    spread = rgb[..., :3].max(axis=2).astype(int) - rgb[..., :3].min(axis=2).astype(int)
    color_mask = yellow_mask | cyan_mask
    background = ~color_mask
    gray_like_fraction = float((spread[background] <= 16).mean()) if background.any() else 0.0
    if gray_like_fraction < 0.82:
        return None

    yellow_segments = _segments_from_mask(yellow_mask)
    cyan_segments = _segments_from_mask(cyan_mask)
    if not yellow_segments and not cyan_segments:
        return None

    rows: list[dict] = []
    index = 1
    for source, segments in (("auto", yellow_segments), ("manual", cyan_segments)):
        for segment in segments:
            rows.append(_measurement_row(segment, source, index, nm_per_px))
            index += 1
    measurements = pd.DataFrame(rows)
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    summary = {
        "resume_mode": True,
        "resume_source": "raster_color",
        "imported_measurements": int(len(measurements)),
        "imported_auto": int((measurements["source"] == "auto").sum()),
        "imported_manual": int((measurements["source"] == "manual").sum()),
        "nm_per_original_px": None if nm_per_px is None else float(nm_per_px),
        "footer_removed_px": 0,
        "detector": "resume-import-no-reanalysis",
    }
    return ResumeAnalysisResult(
        image=rgb,
        image_name=str(filename),
        analysis_scale=1.0,
        original_shape=(h, w),
        full_original_shape=(h, w),
        summary=summary,
        measurements=measurements,
        regions=pd.DataFrame(),
        representatives=pd.DataFrame(),
        candidates=pd.DataFrame(),
        diagnostic_png=None,
        orientation=None,
        calibration=None,
        quality=None,
        footer_start_y=None,
        original_image=rgb,
    )


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _export_status(line: dict) -> str:
    explicit = str(line.get("export_status", "")).strip().lower()
    if explicit in {"keep", "removed", "corrected", "ambiguous"}:
        return explicit
    review = str(line.get("review_label", "")).strip().lower()
    if line.get("replacement_for") not in (None, "") or review in {"corrected", "manual_replaced"}:
        return "corrected"
    if review == "ambiguous":
        return "ambiguous"
    return "keep"


def build_measurement_table(
    lines: Iterable[dict],
    *,
    analysis_scale: float,
    image_coordinates_are_original: bool = False,
) -> pd.DataFrame:
    """Build the canonical 11-column training/review CSV from visible chords."""
    if analysis_scale <= 0:
        raise ValueError("analysis_scale must be positive")
    factor = (1.0 / float(analysis_scale)) if image_coordinates_are_original else 1.0
    rows: list[dict] = []
    for index, raw in enumerate(lines, start=1):
        try:
            x1 = float(raw["x1"]) * factor
            y1 = float(raw["y1"]) * factor
            x2 = float(raw["x2"]) * factor
            y2 = float(raw["y2"]) * factor
        except (KeyError, TypeError, ValueError):
            continue
        if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
            continue
        width = float(math.hypot(x2 - x1, y2 - y1))
        if width <= 0:
            continue
        angle = _finite(raw.get("direction_deg"))
        if angle is None:
            chord = math.degrees(math.atan2(-(y2 - y1), x2 - x1))
            angle = float((chord + 180.0) % 180.0 - 90.0)
        source = "manual" if str(raw.get("source", "auto")) == "manual" else "auto"
        measurement_id = str(raw.get("measurement_id") or raw.get("id") or f"measurement-{index}")
        rows.append({
            "measurement_id": measurement_id,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "center_x": 0.5 * (x1 + x2),
            "center_y": 0.5 * (y1 + y2),
            "width": width,
            "angle": float(angle),
            "source": source,
            "status": _export_status(raw),
        })
    return pd.DataFrame(rows, columns=MEASUREMENT_COLUMNS)
