"""Shared utilities: logging, seeding, IO, hashing, environment capture (v7).

Angle helpers are re-exported from :mod:`coords`, which is the single home of
the project's coordinate convention.  Nothing here imports torch at module
level so the audit / extraction stages run without a DL install.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .coords import (angle_to_vec2, angular_diff_180, chord_endpoints,  # noqa: F401
                     direction_vector, fiber_angle_from_measurement,
                     measurement_angle_from_endpoints, vec2_to_angle, wrap180)

LOGGER_NAME = "sem_fiber_ai"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# --------------------------------------------------------------------------- #
# version: one file, read everywhere
# --------------------------------------------------------------------------- #
_VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"


def package_version() -> str:
    try:
        return _VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:                                  # pragma: no cover
        return "7.0.0"


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def get_logger(name: str = LOGGER_NAME, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S"))
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
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:                              # pragma: no cover
        pass


def rng_state_snapshot() -> dict[str, Any]:
    """Every random generator the training loop touches, for exact resume."""
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state()}
    try:
        import torch

        state["torch_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:                              # pragma: no cover
        pass
    return state


def rng_state_restore(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    try:
        import torch

        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"].cpu())
        if "torch_cuda" in state and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])
            except Exception as exc:                 # noqa: BLE001
                LOG.warning("could not restore CUDA RNG state: %s", exc)
    except ImportError:                              # pragma: no cover
        pass


def environment_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "package_version": package_version(),
    }
    for mod in ("torch", "cv2", "skimage", "pandas", "scipy", "sklearn", "timm", "yaml"):
        try:
            m = __import__(mod)
            report[mod] = getattr(m, "__version__", "unknown")
        except Exception:                            # noqa: BLE001
            report[mod] = None
    try:
        import torch

        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_version"] = getattr(torch.version, "cuda", None)
        report["cuda_device"] = (torch.cuda.get_device_name(0)
                                 if torch.cuda.is_available() else None)
    except Exception:                                # noqa: BLE001
        report["cuda_available"] = False
        report["cuda_device"] = None
    return report


def pick_device(prefer: str = "auto") -> "Any":
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was required but torch.cuda.is_available() is False")
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
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"not a directory: {d}")
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _json_default(o: Any) -> Any:
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "to_dict"):
        return o.to_dict()
    return str(o)


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write via a temp file + rename so a reader never sees a half file."""
    p = Path(path)
    ensure_dir(p.parent)
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return p


def save_json(obj: Any, path: str | Path) -> Path:
    return atomic_write_text(path, json.dumps(obj, indent=2, default=_json_default,
                                              ensure_ascii=False))


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_tree(root: str | Path, patterns: Sequence[str] = ("*.py", "*.yaml", "VERSION")
                ) -> str:
    """Deterministic digest of a source tree (sorted relative paths + bytes)."""
    root = Path(root)
    h = hashlib.sha256()
    files = sorted({p for pat in patterns for p in root.rglob(pat) if p.is_file()
                    and "__pycache__" not in p.parts})
    for p in files:
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def read_gray(path: str | Path) -> np.ndarray:
    """Read an image as float32 grayscale in [0, 255]."""
    import cv2

    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr[..., :3], cv2.COLOR_BGR2GRAY)
    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 257.0
    return arr.astype(np.float32)


def read_rgb(path: str | Path) -> np.ndarray:
    import cv2

    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# legacy angle helpers kept for the reused extraction / prior modules.  The
# only convention they know is raster (y down); ``y_sign=-1`` converts a y-up
# (ImageJ) angle into a raster direction explicitly.
# --------------------------------------------------------------------------- #
Y_SIGN_DOWN = 1.0
Y_SIGN_UP = -1.0


def wrap_deg_180(a):
    return wrap180(a)


def angle_to_direction(angle_deg, y_sign: float = Y_SIGN_DOWN):
    t = np.deg2rad(np.asarray(angle_deg, np.float64))
    return np.cos(t), y_sign * np.sin(t)


def line_endpoints(cx: float, cy: float, angle_deg: float, length: float,
                   y_sign: float = Y_SIGN_DOWN):
    ux, uy = angle_to_direction(angle_deg, y_sign)
    dx, dy = ux * length / 2.0, uy * length / 2.0
    return cx - dx, cy - dy, cx + dx, cy + dy


def endpoints_to_center_angle_len(x1, y1, x2, y2, y_sign: float = Y_SIGN_DOWN):
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    dx, dy = x2 - x1, y2 - y1
    return cx, cy, float(np.rad2deg(np.arctan2(y_sign * dy, dx))), float(np.hypot(dx, dy))


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rect:
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


_ID_SUFFIXES: tuple[str, ...] = tuple(sorted(
    ("_labeled_thickness", "_labeled", "_annotated", "_overlay",
     "_corrected_measurements", "_measurements", "_visionflux_review",
     "_visionflux", "_review", "_imagej_results", "_results", "_thickness"),
    key=len, reverse=True))
_COPY_MARKER = re.compile(
    r"(?:__\d+_?|_\(\d+\)|\s*\(\d+\)|[ _-]copy(?:[ _-]?\d+)?)$", re.IGNORECASE)


def image_id_from_path(path: str | Path) -> str:
    """Filename stem with role and copy suffixes stripped (``2-21_labeled_thickness`` -> ``2-21``)."""
    stem = Path(path).stem
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
    seen: dict[str, list[Path]] = {}
    for p in paths:
        seen.setdefault(image_id_from_path(p), []).append(Path(p))
    return {k: v for k, v in seen.items() if len(v) > 1}


# --------------------------------------------------------------------------- #
# Safe archive extraction (also embedded verbatim in the notebook setup cell)
# --------------------------------------------------------------------------- #
def safe_extract_tar(archive_path, dest_dir, *, expected_sha256=None):
    """Extract a .tar(.gz) into ``dest_dir`` refusing every unsafe member.

    Rejected: absolute paths, ``..`` components, members that resolve outside
    ``dest_dir``, symlinks/hardlinks, devices/FIFOs.  If ``expected_sha256`` is
    given the archive digest must match first.  Returns the list of extracted
    member names.
    """
    import hashlib
    import os
    import tarfile

    archive_path = str(archive_path)
    dest = os.path.realpath(str(dest_dir))
    if expected_sha256:
        h = hashlib.sha256()
        with open(archive_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != expected_sha256:
            raise RuntimeError(f"archive digest {h.hexdigest()} != expected {expected_sha256}")
    os.makedirs(dest, exist_ok=True)
    names = []
    with tarfile.open(archive_path, "r:*") as tf:
        members = tf.getmembers()
        for m in members:
            name = m.name
            if not name or os.path.isabs(name) or name.startswith(("/", "\\")):
                raise RuntimeError(f"unsafe archive member (absolute path): {name!r}")
            parts = name.replace("\\", "/").split("/")
            if any(p == ".." for p in parts):
                raise RuntimeError(f"unsafe archive member (parent reference): {name!r}")
            if m.issym() or m.islnk():
                raise RuntimeError(f"unsafe archive member (link): {name!r}")
            if not (m.isfile() or m.isdir()):
                raise RuntimeError(f"unsafe archive member (special file): {name!r}")
            target = os.path.realpath(os.path.join(dest, name))
            if os.path.commonpath([dest, target]) != dest:
                raise RuntimeError(f"unsafe archive member (escapes destination): {name!r}")
        for m in members:
            tf.extract(m, dest, set_attrs=False)
            names.append(m.name)
    return names
