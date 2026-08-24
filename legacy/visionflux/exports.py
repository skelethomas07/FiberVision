from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
from skimage.measure import profile_line


@dataclass(frozen=True)
class ExportBundle:
    imagej_table: pd.DataFrame
    direction_table: pd.DataFrame
    annotated_labeled_png: bytes
    annotated_unlabeled_png: bytes
    unit_length: str
    unit_area: str

    @property
    def annotated_png(self) -> bytes:
        """Backward-compatible alias for the labeled image."""
        return self.annotated_labeled_png


def _active_lines(lines: Iterable[dict]) -> list[dict]:
    return [dict(line) for line in lines if all(np.isfinite(float(line.get(k, np.nan))) for k in ("x1", "y1", "x2", "y2"))]


def _scale_line(line: dict, factor: float) -> dict:
    out = dict(line)
    for key in ("x1", "y1", "x2", "y2"):
        out[key] = float(out[key]) * float(factor)
    if "path_points" in out:
        out["path_points"] = [
            [float(point[0]) * float(factor), float(point[1]) * float(factor)]
            for point in out.get("path_points", [])
        ]
    return out


def _line_angle_deg(line: dict) -> float:
    dx = float(line["x2"]) - float(line["x1"])
    dy = float(line["y2"]) - float(line["y1"])
    angle = math.degrees(math.atan2(-dy, dx))
    # Match ImageJ-style signed straight-line angles.
    return float((angle + 180.0) % 360.0 - 180.0)


def _sample_profile(image: np.ndarray, line: dict) -> np.ndarray:
    gray = np.asarray(image)
    if gray.ndim == 3:
        rgb = gray[..., :3].astype(np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    gray = np.asarray(gray, np.float32)
    profile = profile_line(
        gray,
        (float(line["y1"]), float(line["x1"])),
        (float(line["y2"]), float(line["x2"])),
        linewidth=1,
        order=1,
        mode="reflect",
        reduce_func=np.mean,
    )
    return np.asarray(profile, float).reshape(-1)


def build_imagej_results(
    image: np.ndarray,
    lines: Iterable[dict],
    *,
    analysis_scale: float,
    nm_per_px: float | None,
    image_coordinates_are_original: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Build one ImageJ-like straight-line ROI row per visible fiber label.

    Mean/Min/Max are grayscale intensities sampled along the thickness chord.
    Area follows ImageJ's area-of-selection meaning. For a one-pixel-wide line ROI
    this is the number of sampled pixels, converted to calibrated square units when
    a spatial calibration is available.
    """
    if analysis_scale <= 0:
        raise ValueError("analysis_scale must be positive")
    rows: list[dict] = []
    direction_rows: list[dict] = []
    unit_length = "nm" if nm_per_px is not None else "px"
    unit_area = "nm^2" if nm_per_px is not None else "px^2"

    for index, raw_line in enumerate(_active_lines(lines), start=1):
        line = (
            _scale_line(raw_line, 1.0 / float(analysis_scale))
            if image_coordinates_are_original
            else raw_line
        )
        label = int(raw_line.get("label", index))
        dx = float(line["x2"]) - float(line["x1"])
        dy = float(line["y2"]) - float(line["y1"])
        length_in_image_px = float(math.hypot(dx, dy))
        length_original = (
            length_in_image_px
            if image_coordinates_are_original
            else length_in_image_px / float(analysis_scale)
        )
        length = length_original * float(nm_per_px) if nm_per_px is not None else length_original
        profile = _sample_profile(image, line)
        if profile.size:
            mean = float(np.mean(profile))
            minimum = float(np.min(profile))
            maximum = float(np.max(profile))
            pixel_area = (
                float(profile.size)
                if image_coordinates_are_original
                else float(profile.size) / (float(analysis_scale) ** 2)
            )
        else:
            mean = minimum = maximum = float("nan")
            pixel_area = 0.0
        area = pixel_area * float(nm_per_px) ** 2 if nm_per_px is not None else pixel_area
        angle = _line_angle_deg(line)
        rows.append({
            "label": label,
            "Area": area,
            "Mean": mean,
            "Min": minimum,
            "Max": maximum,
            "Angle": angle,
            "Length": length,
        })
        direction_rows.append({
            "label": label,
            "fiber_direction_deg": raw_line.get("direction_deg", np.nan),
            "thickness_line_angle_deg": angle,
            "thickness": length,
            "length_unit": unit_length,
            "fiber_region_id": raw_line.get("fiber_region_id"),
            "source": raw_line.get("source"),
        })
    return pd.DataFrame(rows), pd.DataFrame(direction_rows), unit_length, unit_area


def _resume_metadata_rows(lines: Iterable[dict], coordinate_scale: float) -> list[dict]:
    rows: list[dict] = []
    for index, raw_line in enumerate(_active_lines(lines), start=1):
        line = _scale_line(raw_line, coordinate_scale)
        rows.append({
            "measurement_id": str(raw_line.get("measurement_id") or raw_line.get("id") or f"measurement-{index}"),
            "x1": float(line["x1"]),
            "y1": float(line["y1"]),
            "x2": float(line["x2"]),
            "y2": float(line["y2"]),
            "direction_deg": raw_line.get("direction_deg"),
            "source": "manual" if str(raw_line.get("source", "auto")) == "manual" else "auto",
            "review_label": raw_line.get("review_label"),
            "replacement_for": raw_line.get("replacement_for"),
        })
    return rows


def render_annotated_image(
    image: np.ndarray,
    lines: Iterable[dict],
    *,
    coordinate_scale: float = 1.0,
    show_labels: bool = True,
) -> bytes:
    arr = np.asarray(image)
    if arr.ndim == 2:
        rgb = np.repeat(arr[..., None], 3, axis=2)
    else:
        rgb = arr[..., :3]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    pil = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default()

    for index, raw_line in enumerate(_active_lines(lines), start=1):
        line = _scale_line(raw_line, coordinate_scale)
        label = int(raw_line.get("label", index))
        source = str(raw_line.get("source", "auto"))
        color = (26, 220, 235) if source == "manual" else (255, 211, 70)
        p1 = (float(line["x1"]), float(line["y1"]))
        p2 = (float(line["x2"]), float(line["y2"]))
        draw.line([p1, p2], fill=color, width=3)
        radius = 3
        for x, y in (p1, p2):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        if show_labels:
            mx, my = 0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1])
            text = str(label)
            bbox = draw.textbbox((mx, my), text, font=font, stroke_width=2)
            pad = 2
            draw.rectangle((bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad), fill=(0, 0, 0))
            draw.text((mx, my), text, fill=(255, 255, 255), font=font)

    buffer = BytesIO()
    pnginfo = PngInfo()
    pnginfo.add_text(
        "visionflux_measurements_v1",
        json.dumps(_resume_metadata_rows(lines, coordinate_scale), ensure_ascii=False, separators=(",", ":")),
    )
    pil.save(buffer, format="PNG", optimize=True, pnginfo=pnginfo)
    return buffer.getvalue()


def build_export_bundle(
    image: np.ndarray,
    lines: Iterable[dict],
    *,
    analysis_scale: float,
    nm_per_px: float | None,
    image_coordinates_are_original: bool = False,
) -> ExportBundle:
    imagej, direction, unit_length, unit_area = build_imagej_results(
        image,
        lines,
        analysis_scale=analysis_scale,
        nm_per_px=nm_per_px,
        image_coordinates_are_original=image_coordinates_are_original,
    )
    coordinate_scale = 1.0 / float(analysis_scale) if image_coordinates_are_original else 1.0
    return ExportBundle(
        imagej_table=imagej,
        direction_table=direction,
        annotated_labeled_png=render_annotated_image(
            image, lines, coordinate_scale=coordinate_scale, show_labels=True,
        ),
        annotated_unlabeled_png=render_annotated_image(
            image, lines, coordinate_scale=coordinate_scale, show_labels=False,
        ),
        unit_length=unit_length,
        unit_area=unit_area,
    )
