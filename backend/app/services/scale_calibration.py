from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from io import BytesIO
from typing import Callable

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage


@dataclass(frozen=True)
class ScaleBar:
    x0: int
    y0: int
    x1: int
    y1: int
    width_px: float


@dataclass(frozen=True)
class CalibrationResult:
    source: str
    nm_per_pixel: float | None
    scale_value_nm: float | None = None
    scale_bar_px: float | None = None
    scale_label: str | None = None


_SCALE_RE = re.compile(r"(?<![\d.])(\d+(?:[.,]\d+)?)\s*(nm|[uµμ]m|pm)\b", re.IGNORECASE)


def parse_scale_label(text: str) -> tuple[float, str] | None:
    """Parse SEM scale labels and common OCR variants into nanometres."""
    for match in _SCALE_RE.finditer(text.replace("μ", "µ")):
        value = float(match.group(1).replace(",", "."))
        unit = match.group(2).lower()
        if unit == "nm":
            factor = 1.0
        else:
            # Tesseract commonly reads the micro sign in "1µm" as "p".
            # Picometre scale bars are not plausible for these SEM exports,
            # so "pm" is intentionally treated as an OCR variant of µm.
            factor = 1000.0
        if value > 0:
            return value * factor, match.group(0)
    return None


def _image_gray(image_bytes: bytes) -> np.ndarray:
    with Image.open(BytesIO(image_bytes)) as source:
        source.seek(0)
        return np.asarray(source.convert("L"), dtype=np.uint8)


def detect_scale_bar(image_bytes: bytes) -> ScaleBar | None:
    gray = _image_gray(image_bytes)
    height, width = gray.shape
    footer_top = max(0, int(height * 0.70))
    footer = gray[footer_top:, :]

    threshold = max(220, int(np.percentile(footer, 97)))
    bright = footer >= threshold
    labels, count = ndimage.label(bright, structure=np.ones((3, 3), dtype=np.uint8))
    slices = ndimage.find_objects(labels)

    min_width = max(30, int(width * 0.035))
    max_width = max(min_width + 1, int(width * 0.65))
    best: tuple[float, ScaleBar] | None = None

    for label_id, component_slice in enumerate(slices, start=1):
        if component_slice is None:
            continue
        ys, xs = component_slice
        component_width = xs.stop - xs.start
        component_height = ys.stop - ys.start
        if component_width < min_width or component_width > max_width:
            continue
        if component_height < 2 or component_height > max(30, int(height * 0.06)):
            continue
        if component_width / max(component_height, 1) < 4.0:
            continue

        component = labels[component_slice] == label_id
        fill_ratio = float(component.mean())
        if fill_ratio < 0.55:
            continue

        y0 = footer_top + ys.start
        y1 = footer_top + ys.stop
        bar = ScaleBar(
            x0=xs.start,
            y0=y0,
            x1=xs.stop,
            y1=y1,
            width_px=float(component_width),
        )
        lower_bonus = 1.0 + 0.15 * (y0 / max(height, 1))
        score = component_width * fill_ratio * lower_bonus
        if best is None or score > best[0]:
            best = (score, bar)

    return None if best is None else best[1]


def _ocr_footer(image_bytes: bytes, bar: ScaleBar) -> str:
    with Image.open(BytesIO(image_bytes)) as source:
        gray = source.convert("L")
        width, height = gray.size
        top = max(0, bar.y0 - max(24, int(height * 0.035)))
        bottom = min(height, bar.y1 + max(40, int(height * 0.06)))
        crop = gray.crop((0, top, width, bottom))
        crop = ImageOps.autocontrast(crop)
        crop = crop.resize((crop.width * 3, crop.height * 3))
        payload = BytesIO()
        crop.save(payload, format="PNG")

    try:
        completed = subprocess.run(
            ["tesseract", "stdin", "stdout", "--psm", "6"],
            input=payload.getvalue(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=8,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.decode("utf-8", errors="ignore")


def detect_scale_calibration(
    image_bytes: bytes,
    *,
    ocr_runner: Callable[[bytes], str] | None = None,
) -> CalibrationResult | None:
    bar = detect_scale_bar(image_bytes)
    if bar is None:
        return None
    text = ocr_runner(image_bytes) if ocr_runner is not None else _ocr_footer(image_bytes, bar)
    parsed = parse_scale_label(text)
    if parsed is None:
        return None
    scale_value_nm, label = parsed
    if bar.width_px <= 0:
        return None
    nm_per_pixel = scale_value_nm / bar.width_px
    if not np.isfinite(nm_per_pixel) or nm_per_pixel <= 0 or nm_per_pixel > 1_000_000:
        return None
    return CalibrationResult(
        source="scale_bar",
        nm_per_pixel=float(nm_per_pixel),
        scale_value_nm=float(scale_value_nm),
        scale_bar_px=float(bar.width_px),
        scale_label=label,
    )


def resolve_nm_per_pixel(
    image_bytes: bytes,
    manual_nm_per_pixel: float | None,
    *,
    ocr_runner: Callable[[bytes], str] | None = None,
) -> CalibrationResult:
    if manual_nm_per_pixel is not None:
        value = float(manual_nm_per_pixel)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("nm_per_pixel must be greater than zero")
        return CalibrationResult(source="manual", nm_per_pixel=value)

    detected = detect_scale_calibration(image_bytes, ocr_runner=ocr_runner)
    return detected if detected is not None else CalibrationResult(source="none", nm_per_pixel=None)
