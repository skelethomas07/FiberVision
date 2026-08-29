from __future__ import annotations

import csv
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


def build_export_zip(image_bytes: bytes, rows) -> bytes:
    active = _active_rows(rows)
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("measurements.csv", build_csv(active))
        archive.writestr("measurements_overlay.png", render_overlay(image_bytes, active, labeled=False))
        archive.writestr("measurements_labeled.png", render_overlay(image_bytes, active, labeled=True))
    return output.getvalue()
