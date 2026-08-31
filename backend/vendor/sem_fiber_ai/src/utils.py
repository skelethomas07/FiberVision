"""Shared utilities: logging, seeding, IO, geometry, environment capture.

Nothing in this module imports torch at module level except inside functions,
so the data-audit / annotation-extraction stages run without a DL install.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

LOGGER_NAME = "sem_fiber_ai"

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def get_logger(name: str = LOGGER_NAME, level: int = logging.INFO) -> logging.Logger:
    """Return a configured module logger (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)-7s %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


LOG = get_logger()


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 1337, deterministic: bool = True) -> None:
    """Seed python / numpy / torch and (optionally) force deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # cuBLAS workspace config is required for full determinism on CUDA>=10.2
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:  # pragma: no cover - torch optional for audit stage
        pass


def environment_report() -> dict[str, Any]:
    """Capture package versions / platform for the run manifest."""
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    for mod in ("torch", "cv2", "skimage", "pandas", "scipy", "albumentations"):
        try:
            m = __import__(mod)
            report[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            report[mod] = None
    try:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_device"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception:
        report["cuda_available"] = False
        report["cuda_device"] = None
    return report


def pick_device(prefer: str = "auto") -> "Any":
    """Return a torch.device honouring ``prefer`` in {auto, cuda, cpu}."""
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("config requested CUDA but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(directory: str | Path) -> list[Path]:
    """List image files in a directory, sorted, case-insensitive extension match."""
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"not a directory: {d}")
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def save_json(obj: Any, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    return str(o)


def read_gray(path: str | Path) -> np.ndarray:
    """Read an image as float32 grayscale in [0, 255]. Raises on failure."""
    import cv2

    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr[..., :3], cv2.COLOR_BGR2GRAY)
    if arr.dtype == np.uint16:
        arr = (arr.astype(np.float32) / 257.0)
    return arr.astype(np.float32)


def read_rgb(path: str | Path) -> np.ndarray:
    """Read an image as uint8 RGB."""
    import cv2

    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# geometry / angles
# --------------------------------------------------------------------------- #
def wrap_deg_180(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to [-90, 90). Fiber orientation is pi-periodic."""
    return (np.asarray(a, dtype=np.float64) + 90.0) % 180.0 - 90.0


def angular_diff_180(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    """Smallest absolute difference between two pi-periodic angles, in degrees."""
    d = np.abs(wrap_deg_180(np.asarray(a, np.float64) - np.asarray(b, np.float64)))
    return np.minimum(d, 180.0 - d)


def angle_to_vec2(theta_deg: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Encode a pi-periodic angle as (cos 2t, sin 2t)."""
    t = np.deg2rad(np.asarray(theta_deg, np.float64))
    return np.cos(2 * t), np.sin(2 * t)


def vec2_to_angle(cos2t: np.ndarray | float, sin2t: np.ndarray | float) -> np.ndarray:
    """Decode (cos 2t, sin 2t) back to an angle in [-90, 90)."""
    return wrap_deg_180(np.rad2deg(0.5 * np.arctan2(sin2t, cos2t)))


#: Sign applied to sin(angle) when converting an angle to an image-space
#: direction.  ``+1`` means the angle was written in raster coordinates
#: (y downwards); ``-1`` means the usual mathematical y-up convention that
#: ImageJ documents.  Which one a given export actually used is an empirical
#: question -- see ``calibrate_marker_geometry`` -- so it is configurable and
#: must never be assumed.
Y_SIGN_DOWN = 1.0
Y_SIGN_UP = -1.0


def angle_to_direction(angle_deg: np.ndarray | float, y_sign: float = Y_SIGN_DOWN
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Unit vector in image coordinates for a measurement angle."""
    t = np.deg2rad(np.asarray(angle_deg, np.float64))
    return np.cos(t), y_sign * np.sin(t)


def line_endpoints(cx: float, cy: float, angle_deg: float, length: float,
                   y_sign: float = Y_SIGN_DOWN
                   ) -> tuple[float, float, float, float]:
    """Endpoints of a segment of ``length`` centred at (cx, cy) at ``angle_deg``."""
    ux, uy = angle_to_direction(angle_deg, y_sign)
    dx, dy = ux * length / 2.0, uy * length / 2.0
    return cx - dx, cy - dy, cx + dx, cy + dy


def endpoints_to_center_angle_len(x1: float, y1: float, x2: float, y2: float,
                                  y_sign: float = Y_SIGN_DOWN
                                  ) -> tuple[float, float, float, float]:
    """Inverse of :func:`line_endpoints`; returns (cx, cy, angle_deg, length)."""
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    dx, dy = x2 - x1, y2 - y1
    length = float(np.hypot(dx, dy))
    angle = float(np.rad2deg(np.arctan2(y_sign * dy, dx)))
    return cx, cy, angle, length


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rect:
    """Axis-aligned crop rectangle, inclusive of x0/y0, exclusive of x1/y1."""
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def crop(self, img: np.ndarray) -> np.ndarray:
        return img[self.y0:self.y1, self.x0:self.x1]


def chunked(seq: Sequence[Any], n: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


#: role suffixes, longest first so the longer form always wins
_ID_SUFFIXES: tuple[str, ...] = tuple(sorted(
    ("_labeled_thickness", "_labeled", "_annotated", "_overlay",
     "_corrected_measurements", "_measurements", "_visionflux_review",
     "_visionflux", "_review", "_imagej_results", "_results", "_thickness"),
    key=len, reverse=True))

#: Drive/OS duplicate markers ONLY -- never a bare "_<n>", which is a field
#: number on this dataset (462_1 is a different micrograph from 462_2).
_COPY_MARKER = re.compile(
    r"(?:__\d+_?|_\(\d+\)|\s*\(\d+\)|[ _-]copy(?:[ _-]?\d+)?)$", re.IGNORECASE)


def image_id_from_path(path: str | Path) -> str:
    """Stable image id = filename stem with role and copy suffixes stripped.

    ``2-21_labeled_thickness.png`` -> ``2-21``; ``2-11__1_.jpg`` -> ``2-11``;
    ``462_1.png`` -> ``462_1`` (a field number, NOT a copy marker).
    Used to pair original / annotated / csv files that were named by hand.
    """
    stem = Path(path).stem
    # [v3] copy markers first (they sit outside the role suffix), and longest
    # suffix first, repeatedly: the v2 list order stripped "_review" out of
    # "_visionflux_review" and then no longer matched the longer form.
    stem = _COPY_MARKER.sub("", stem)
    changed = True
    while changed:
        changed = False
        for suffix in _ID_SUFFIXES:
            if stem.lower().endswith(suffix) and len(stem) > len(suffix):
                stem = stem[: -len(suffix)]
                stem = _COPY_MARKER.sub("", stem)
                changed = True
                break
    return stem.rstrip("_-") or Path(path).stem


def duplicate_image_ids(paths: "Iterable[str | Path]") -> dict[str, list[Path]]:
    """Ids that more than one file resolves to.

    Two files sharing an id is never benign: every downstream table is keyed by
    it, so one file's measurements end up attributed to another file's pixels.
    Callers should refuse to proceed rather than pick a winner.
    """
    seen: dict[str, list[Path]] = {}
    for p in paths:
        seen.setdefault(image_id_from_path(p), []).append(Path(p))
    return {k: v for k, v in seen.items() if len(v) > 1}
