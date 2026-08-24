from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import atan2, degrees, hypot

import numpy as np
from PIL import Image
from scipy import ndimage

try:
    from .visionflux_resume_legacy import try_build_resume_analysis as _legacy_resume
except ImportError:  # bundle tests run before the legacy module is copied into backend
    _legacy_resume = None


@dataclass(frozen=True)
class ImportedMeasurement:
    x1: float
    y1: float
    x2: float
    y2: float
    width_px: float
    angle_deg: float
    source: str


@dataclass(frozen=True)
class UploadInspection:
    preview_bytes: bytes
    preview_content_type: str
    is_visionflux_annotated: bool
    measurements: list[ImportedMeasurement]


YELLOW = np.asarray([255.0, 211.0, 70.0], dtype=np.float32)
CYAN = np.asarray([26.0, 220.0, 235.0], dtype=np.float32)
BLUE = np.asarray([40.0, 130.0, 255.0], dtype=np.float32)


def _colour_masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(rgb[..., :3], dtype=np.float32)
    spread = arr.max(axis=2) - arr.min(axis=2)
    yellow_distance = np.linalg.norm(arr - YELLOW[None, None, :], axis=2)
    cyan_distance = np.linalg.norm(arr - CYAN[None, None, :], axis=2)
    blue_distance = np.linalg.norm(arr - BLUE[None, None, :], axis=2)
    yellow = (
        (spread >= 45) & (yellow_distance <= 90)
        & (arr[..., 0] >= 175) & (arr[..., 1] >= 135) & (arr[..., 2] <= 165)
    )
    manual = (
        (spread >= 45)
        & ((cyan_distance <= 95) | (blue_distance <= 100))
        & (arr[..., 2] >= 150)
        & ((arr[..., 2] - arr[..., 0]) >= 40)
    )
    return yellow, manual


def _components(mask: np.ndarray, min_pixels: int = 10) -> list[np.ndarray]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    out: list[np.ndarray] = []
    for label_id in range(1, int(count) + 1):
        coords = np.argwhere(labels == label_id).astype(np.float64)
        if len(coords) >= min_pixels:
            out.append(coords)
    return out


def _measurement(component_yx: np.ndarray, source: str) -> ImportedMeasurement | None:
    xy = component_yx[:, ::-1]
    center = xy.mean(axis=0)
    centered = xy - center
    if len(xy) < 3:
        return None
    cov = np.cov(centered, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    values = np.maximum(values, 1e-9)
    if float(values.max() / values.min()) < 2.2:
        return None
    axis = vectors[:, int(np.argmax(values))]
    projection = centered @ axis
    lo, hi = np.percentile(projection, [4, 96])
    if hi - lo < 8:
        return None
    a = center + axis * lo
    b = center + axis * hi
    x1, y1 = map(float, a)
    x2, y2 = map(float, b)
    width = hypot(x2 - x1, y2 - y1)
    angle = ((degrees(atan2(-(y2 - y1), x2 - x1)) + 180.0) % 360.0) - 180.0
    return ImportedMeasurement(x1, y1, x2, y2, width, angle, source)


def _neutralised_preview(image: Image.Image, annotation_mask: np.ndarray) -> bytes:
    gray = np.asarray(image.convert('L')).copy()
    if annotation_mask.any():
        for comp in _components(annotation_mask, min_pixels=1):
            ys = comp[:, 0].astype(int)
            xs = comp[:, 1].astype(int)
            y0, y1 = max(0, ys.min() - 5), min(gray.shape[0], ys.max() + 6)
            x0, x1 = max(0, xs.min() - 5), min(gray.shape[1], xs.max() + 6)
            local_mask = annotation_mask[y0:y1, x0:x1]
            local = gray[y0:y1, x0:x1]
            background = local[~local_mask]
            fill = int(np.median(background)) if background.size else int(np.median(gray))
            gray[ys, xs] = fill
    preview = Image.fromarray(gray, mode='L').convert('RGB')
    buf = BytesIO()
    preview.save(buf, format='PNG')
    return buf.getvalue()


def inspect_sem_upload(
    data: bytes,
    filename: str = "sem-image",
    nm_per_pixel: float | None = None,
) -> UploadInspection:
    image = Image.open(BytesIO(data))
    image.seek(0)
    rgb_image = image.convert("RGB")
    rgb = np.asarray(rgb_image)
    yellow, blue = _colour_masks(rgb)

    fallback: list[ImportedMeasurement] = []
    for mask, source in ((yellow, "visionflux_auto"), (blue, "visionflux_manual")):
        for comp in _components(mask):
            item = _measurement(comp, source)
            if item is not None:
                fallback.append(item)

    measurements = fallback
    has_metadata = bool(image.info.get("visionflux_measurements_v1"))
    # The legacy VisionFlux importer knows how to reconnect lines split by number
    # labels and can read exact PNG metadata. Only invoke raster resume when our
    # stricter line-shape gate saw at least one real chord, preventing arbitrary
    # coloured blobs from being treated as measurements.
    if _legacy_resume is not None and (fallback or has_metadata):
        resumed = _legacy_resume(data, filename, nm_per_px=nm_per_pixel)
        if resumed is not None and not resumed.measurements.empty:
            measurements = []
            for row in resumed.measurements.to_dict(orient="records"):
                source = "visionflux_manual" if str(row.get("source")) == "manual" else "visionflux_auto"
                x1, y1, x2, y2 = (float(row[k]) for k in ("x1", "y1", "x2", "y2"))
                measurements.append(ImportedMeasurement(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    width_px=float(row.get("width_px", hypot(x2 - x1, y2 - y1))),
                    angle_deg=((degrees(atan2(-(y2 - y1), x2 - x1)) + 180.0) % 360.0) - 180.0,
                    source=source,
                ))

    annotated = bool(measurements)
    annotation_mask = yellow | blue if annotated else np.zeros(yellow.shape, dtype=bool)
    preview = _neutralised_preview(rgb_image, annotation_mask)
    return UploadInspection(preview, "image/png", annotated, measurements)


def preview_storage_key(image_id: str) -> str:
    return f"previews/{image_id}.png"


def import_storage_key(image_id: str) -> str:
    return f"visionflux-imports/{image_id}.json"


def measurement_payload(measurement: ImportedMeasurement) -> dict[str, float | str]:
    return {
        "x1": measurement.x1,
        "y1": measurement.y1,
        "x2": measurement.x2,
        "y2": measurement.y2,
        "width_px": measurement.width_px,
        "angle_deg": measurement.angle_deg,
        "source": measurement.source,
    }
