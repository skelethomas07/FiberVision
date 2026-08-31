from __future__ import annotations

import csv
import math
from io import BytesIO, StringIO
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageDraw, ImageFont

YELLOW = (255, 211, 77)
BLUE = (74, 163, 255)


def _active_rows(rows):
    return [row for row in rows if bool(row.active)]


def _status(row) -> str:
    if bool(row.edited):
        return "CORRECTED"
    if row.source_model_measurement_id is None:
        return "MANUAL_ADD"
    return "KEEP"


def build_csv(rows) -> bytes:
    buffer = StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow([
        "index", "id", "x1", "y1", "x2", "y2", "width_px", "width_nm",
        "measurement_angle_deg", "fiber_angle_deg", "source", "status",
    ])
    for index, row in enumerate(_active_rows(rows), start=1):
        writer.writerow([
            index,
            row.id,
            f"{float(row.x1):.6f}",
            f"{float(row.y1):.6f}",
            f"{float(row.x2):.6f}",
            f"{float(row.y2):.6f}",
            f"{float(row.width_px):.6f}",
            "" if row.width_nm is None else f"{float(row.width_nm):.6f}",
            f"{float(row.angle_deg):.6f}",
            f"{((float(row.angle_deg) + 90.0) % 180.0):.6f}",
            row.source,
            _status(row),
        ])
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def _fiber_angle_deg(row) -> float:
    return (float(row.angle_deg) + 90.0) % 180.0


def _orientation_degree_bin(row) -> int:
    # Nearest whole degree over the requested inclusive 0..180 presentation.
    # 180 is kept as the upper-edge bin instead of being wrapped back to 0.
    angle = _fiber_angle_deg(row)
    return max(0, min(180, int(math.floor(angle + 0.5))))


def build_orientation_distribution_csv(rows) -> bytes:
    """181 columns (0°..180°), each value is count/total and percentage."""
    active = _active_rows(rows)
    total = len(active)
    counts = [0] * 181
    for row in active:
        counts[_orientation_degree_bin(row)] += 1

    buffer = StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow([f"{degree}°" for degree in range(181)])
    writer.writerow([
        f"{count}/{total} ({(100.0 * count / total if total else 0.0):.2f}%)"
        for count in counts
    ])
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def build_thickness_range_csv(rows, normal_min_nm: float, normal_max_nm: float) -> bytes:
    low = float(normal_min_nm)
    high = float(normal_max_nm)
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError("normal thickness range must be finite")
    if low < 0 or high < 0 or low > high:
        raise ValueError("normal thickness range must satisfy 0 <= min <= max")

    active = _active_rows(rows)
    widths: list[float] = []
    for row in active:
        if row.width_nm is None or not math.isfinite(float(row.width_nm)):
            raise ValueError("nm thickness is unavailable for one or more active measurements")
        widths.append(float(row.width_nm))
    total = len(widths)
    inside = sum(1 for width in widths if low <= width <= high)
    percent = 100.0 * inside / total if total else 0.0

    buffer = StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow([
        "normal_min_nm", "normal_max_nm", "total_fibers",
        "within_range_fibers", "within_range_percent", "within_range_pair",
    ])
    writer.writerow([
        f"{low:.6f}", f"{high:.6f}", total, inside, f"{percent:.2f}",
        f"{inside}/{total} ({percent:.2f}%)",
    ])
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def _line_color(row):
    source = str(row.source or "").lower()
    if row.source_model_measurement_id is None or source in {"manual", "visionflux_manual"}:
        return BLUE
    return YELLOW


def render_overlay(image_bytes: bytes, rows, *, labeled: bool = False) -> bytes:
    with Image.open(BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    line_width = max(2, round(min(image.size) / 500))
    radius = max(2, line_width)

    for index, row in enumerate(_active_rows(rows), start=1):
        color = _line_color(row)
        a = (float(row.x1), float(row.y1))
        b = (float(row.x2), float(row.y2))
        draw.line([a, b], fill=color, width=line_width)
        for x, y in (a, b):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        if labeled:
            mx = (a[0] + b[0]) / 2
            my = (a[1] + b[1]) / 2
            label = str(index)
            bbox = draw.textbbox((mx, my), label, font=font, anchor="mm")
            pad = 2
            draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(12, 16, 24))
            draw.text((mx, my), label, fill=(255, 255, 255), font=font, anchor="mm")

    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def build_export_zip(image_bytes: bytes, rows, *, base_name: str = "fiber") -> bytes:
    active = _active_rows(rows)
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(f"{base_name}_measurements.csv", build_csv(active))
        archive.writestr(f"{base_name}_overlay.png", render_overlay(image_bytes, active, labeled=False))
        archive.writestr(f"{base_name}_labeled.png", render_overlay(image_bytes, active, labeled=True))
    return output.getvalue()
