from __future__ import annotations

"""
CHEM FRONTIER continual fiber-thickness learning pipeline.

Designed for Google Colab + Google Drive.
Default dataset root:
    /content/drive/MyDrive/CHEM FRONTIER

Expected per-image files:
    2-21.jpg
    2-21_thickness.png
    2-21_labeled_thickness.png
    2-21_ImageJ_results.csv

Ground-truth convention in *_thickness.png:
    yellow (255, 211, 70): VisionFlux proposal kept by a human  -> AUTO_KEEP
    cyan   (26, 220, 235): human-added measurement              -> MANUAL_ADD

VisionFlux proposals that are present in the raw VisionFlux output but absent
from the final yellow overlay are treated as explicit hard negatives
(AUTO_REMOVE). The pipeline never treats arbitrary unlabeled image locations as
negative, because absence of a manual mark does not necessarily mean "not a
fiber".

The script compares several independent approaches and stores a leaderboard:
    - VisionFlux fixed-rule confidence (baseline)
    - ExtraTrees on engineered features
    - RandomForest on engineered features
    - HistGradientBoosting on engineered features
    - MLP on engineered features
    - optional Patch CNN (PyTorch)
    - optional 1-D normal-profile CNN (PyTorch)

Candidate generation is intentionally broader than the original VisionFlux:
    - default VisionFlux
    - wide/relaxed VisionFlux pass for broad fibers
    - relative-contrast multi-scale ridge proposals
    - normal-profile / parallel-edge recovery proposals

The broad-fiber recovery path explicitly uses "failure of the normal detector"
as a feature rather than as a final conclusion. A candidate can have weak
VisionFlux/ridge response but strong bilateral edges, long orientation
continuity, and a broad normal profile; the learned model decides whether that
combination should be trusted.

Re-running the script after adding more complete answer-sheet sets retrains the
models, evaluates them on a stable image-level holdout, and updates the champion
only when the new score improves.
"""

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import pickle
import random
import re
import shutil
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage, signal
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.morphology import closing, disk, remove_small_objects, skeletonize

from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_ROOT = Path("/content/drive/MyDrive/CHEM FRONTIER")
YELLOW = np.array([255, 211, 70], dtype=np.int16)
CYAN = np.array([26, 220, 235], dtype=np.int16)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
TRUTH_SUFFIXES = (
    "_thickness",
    "_labeled_thickness",
    "_ImageJ_results",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    root: Path = DEFAULT_ROOT
    output_dir_name: str = "_fiber_models"
    cache_dir_name: str = "_fiber_cache"
    visionflux_filename_candidates: tuple[str, ...] = (
        "visionflux_thickness_algorithm.py",
        "visionflux_thickness_algorithm(2).py",
    )

    # Ground-truth color extraction.
    color_tolerance: int = 12
    endpoint_peak_min_distance: int = 3
    endpoint_peak_threshold: float = 2.2
    endpoint_max_pair_px: float = 140.0
    endpoint_min_pair_px: float = 2.5
    endpoint_min_coverage: float = 0.60
    truth_count_retry_fraction: float = 0.12

    # Matching VisionFlux proposals to human-kept measurements.
    match_center_tolerance_px: float = 10.0
    match_angle_tolerance_deg: float = 28.0
    match_width_ratio_max: float = 2.4

    # Feature extraction and recovery candidates.
    profile_half_width_px: int = 80
    profile_samples: int = 161
    patch_size: int = 48
    relative_ridge_scales: tuple[tuple[float, float], ...] = (
        (1.2, 3.5),
        (2.0, 6.0),
        (3.5, 10.0),
        (6.0, 18.0),
        (10.0, 30.0),
    )
    relative_ridge_percentile: float = 78.0
    relative_ridge_sample_spacing_px: float = 8.0
    recovery_grid_stride_px: int = 10
    recovery_min_profile_score: float = 0.18
    recovery_max_candidates_per_image: int = 3500

    # Model training.
    seed: int = 20260821
    min_complete_images_for_training: int = 4
    use_patch_cnn: bool = True
    use_profile_cnn: bool = True
    cnn_epochs: int = 8
    cnn_batch_size: int = 128
    cnn_max_samples: int = 18000
    threshold_grid: tuple[float, ...] = tuple(np.linspace(0.15, 0.85, 29))

    # Champion score. F1 alone can hide the two supervision types the user
    # explicitly cares about, so manual-add recall and hard-negative rejection
    # are rewarded separately.
    score_f1_weight: float = 0.50
    score_manual_add_recall_weight: float = 0.25
    score_hard_negative_rejection_weight: float = 0.20
    score_average_precision_weight: float = 0.05

    @property
    def output_dir(self) -> Path:
        return self.root / self.output_dir_name

    @property
    def cache_dir(self) -> Path:
        return self.root / self.cache_dir_name


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def wrap90(angle: Any) -> Any:
    arr = np.asarray(angle, dtype=float)
    out = (arr + 90.0) % 180.0 - 90.0
    if np.ndim(angle) == 0:
        return float(out)
    return out


def axial_error_deg(a: Any, b: Any) -> Any:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    d = np.abs(aa - bb) % 180.0
    out = np.minimum(d, 180.0 - d)
    if np.ndim(a) == 0 and np.ndim(b) == 0:
        value = float(out)
        if abs(value - round(value)) < 1e-12:
            return int(round(value))
        return value
    return out


def line_angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    # Image y increases downward; negate dy to use the ordinary Cartesian angle.
    return wrap90(math.degrees(math.atan2(-(y2 - y1), x2 - x1)))


def robust_normalize(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if not finite.size:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(finite, [0.5, 99.5])
    return np.clip((arr - lo) / max(float(hi - lo), 1e-9), 0.0, 1.0).astype(np.float32)


def load_gray(path: str | Path) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() in {".tif", ".tiff"}:
        try:
            import tifffile

            arr = tifffile.imread(p)
        except Exception:
            arr = np.asarray(Image.open(p))
    else:
        arr = np.asarray(Image.open(p))
    if arr.ndim == 3:
        rgb = arr[..., :3].astype(np.float32)
        arr = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    arr = np.squeeze(np.asarray(arr, dtype=np.float32))
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D SEM image, got shape={arr.shape}: {p}")
    return arr


def file_signature(paths: Sequence[str | Path]) -> str:
    h = hashlib.sha1()
    for path in paths:
        p = Path(path)
        h.update(str(p.resolve()).encode())
        if p.exists():
            st = p.stat()
            h.update(str(st.st_size).encode())
            h.update(str(st.st_mtime_ns).encode())
    return h.hexdigest()[:16]


def _stem_from_truth_name(name: str) -> str | None:
    for suffix in ("_labeled_thickness.png", "_thickness.png", "_ImageJ_results.csv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------


def scan_dataset(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    raw_by_stem: dict[str, Path] = {}
    truth_by_stem: dict[str, dict[str, Path]] = {}

    for p in sorted(root.iterdir()):
        if not p.is_file():
            continue
        truth_stem = _stem_from_truth_name(p.name)
        if truth_stem is not None:
            d = truth_by_stem.setdefault(truth_stem, {})
            if p.name.endswith("_labeled_thickness.png"):
                d["labeled"] = p
            elif p.name.endswith("_thickness.png"):
                d["overlay"] = p
            elif p.name.endswith("_ImageJ_results.csv"):
                d["csv"] = p
            continue

        if p.suffix.lower() in IMAGE_EXTENSIONS:
            # Ignore generated prediction images.
            if any(token in p.stem for token in ("_prediction", "_debug", "_overlay_pred")):
                continue
            raw_by_stem[p.stem] = p

    stems = sorted(set(raw_by_stem) | set(truth_by_stem))
    rows = []
    for stem in stems:
        truth = truth_by_stem.get(stem, {})
        raw = raw_by_stem.get(stem)
        complete = raw is not None and all(k in truth for k in ("overlay", "labeled", "csv"))
        rows.append(
            {
                "stem": stem,
                "raw_path": str(raw) if raw else None,
                "overlay_path": str(truth.get("overlay")) if truth.get("overlay") else None,
                "labeled_path": str(truth.get("labeled")) if truth.get("labeled") else None,
                "csv_path": str(truth.get("csv")) if truth.get("csv") else None,
                "complete_truth": bool(complete),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Truth extraction from yellow/cyan overlays
# ---------------------------------------------------------------------------


def _color_mask(rgb: np.ndarray, target: np.ndarray, tolerance: int) -> np.ndarray:
    diff = np.abs(rgb.astype(np.int16) - target[None, None, :])
    return np.max(diff, axis=2) <= int(tolerance)


def _pair_endpoint_peaks(
    mask: np.ndarray,
    *,
    min_distance: int,
    threshold_abs: float,
    max_pair_px: float,
    min_pair_px: float,
    min_coverage: float,
) -> pd.DataFrame:
    """Recover individual measurement marks from line+endpoint graphics.

    The supplied overlays use a thin colored chord with two thicker endpoint
    dots. Distance-transform peaks are therefore much more stable than connected
    components when nearby measurements touch each other.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return pd.DataFrame(
            columns=[
                "x1",
                "y1",
                "x2",
                "y2",
                "center_x",
                "center_y",
                "overlay_length_px",
                "overlay_angle_deg",
                "coverage",
            ]
        )

    dt = ndimage.distance_transform_edt(mask)
    peaks_yx = peak_local_max(
        dt,
        min_distance=max(1, int(min_distance)),
        threshold_abs=float(threshold_abs),
        exclude_border=False,
    )
    if len(peaks_yx) < 2:
        return pd.DataFrame()
    xy = peaks_yx[:, [1, 0]].astype(np.float32)
    tree = cKDTree(xy)

    pair_hypotheses: list[tuple[float, int, int, float, float]] = []
    h, w = mask.shape
    for i, p in enumerate(xy):
        neighbors = tree.query_ball_point(p, r=float(max_pair_px))
        for j in neighbors:
            if j <= i:
                continue
            q = xy[j]
            dx = float(q[0] - p[0])
            dy = float(q[1] - p[1])
            dist = math.hypot(dx, dy)
            if dist < float(min_pair_px):
                continue

            # Sample the chord densely. True endpoint pairs are connected by the
            # rendered colored line, whereas accidental neighbor pairs are not.
            n = max(7, int(math.ceil(1.5 * dist)))
            t = np.linspace(0.0, 1.0, n, dtype=np.float32)
            xs = np.clip(np.rint(p[0] + dx * t).astype(int), 0, w - 1)
            ys = np.clip(np.rint(p[1] + dy * t).astype(int), 0, h - 1)
            coverage = float(mask[ys, xs].mean())
            if coverage < float(min_coverage):
                continue

            # Prefer high line coverage. A tiny length reward avoids pairing two
            # peaks inside one endpoint dot when a real connected counterpart is
            # also available.
            score = coverage + 0.0008 * min(dist, 100.0)
            pair_hypotheses.append((score, i, j, dist, coverage))

    pair_hypotheses.sort(reverse=True)
    used: set[int] = set()
    rows: list[dict[str, float]] = []
    for score, i, j, dist, coverage in pair_hypotheses:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        p, q = xy[i], xy[j]
        x1, y1, x2, y2 = map(float, (p[0], p[1], q[0], q[1]))
        rows.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": 0.5 * (x1 + x2),
                "center_y": 0.5 * (y1 + y2),
                "overlay_length_px": float(dist),
                "overlay_angle_deg": float(line_angle_deg(x1, y1, x2, y2)),
                "coverage": float(coverage),
            }
        )
    return pd.DataFrame(rows)


def _truth_extraction_once(
    rgb: np.ndarray,
    config: PipelineConfig,
    *,
    min_distance: int,
    threshold_abs: float,
) -> pd.DataFrame:
    blocks = []
    for target, source in ((YELLOW, "AUTO_KEEP"), (CYAN, "MANUAL_ADD")):
        mask = _color_mask(rgb, target, config.color_tolerance)
        frame = _pair_endpoint_peaks(
            mask,
            min_distance=min_distance,
            threshold_abs=threshold_abs,
            max_pair_px=config.endpoint_max_pair_px,
            min_pair_px=config.endpoint_min_pair_px,
            min_coverage=config.endpoint_min_coverage,
        )
        if not frame.empty:
            frame["truth_source"] = source
            blocks.append(frame)
    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, ignore_index=True)


def extract_truth_measurements(
    overlay_path: str | Path,
    labeled_path: str | Path,
    csv_path: str | Path,
    config: PipelineConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract spatial truth from the color overlay and numeric QA from CSV.

    The CSV supplied by the user currently has Length/Angle but no x/y
    coordinates. Therefore the overlay determines spatial coordinates. The CSV
    count is used to tune endpoint extraction and its Length column calibrates
    the overlay's drawn chord length to ImageJ's numeric length scale. The
    labeled image is used as an integrity check (same dimensions); OCR is not
    required.
    """
    config = config or PipelineConfig(root=Path(overlay_path).parent)
    overlay_path = Path(overlay_path)
    labeled_path = Path(labeled_path)
    csv_path = Path(csv_path)

    rgb = np.asarray(Image.open(overlay_path).convert("RGB"))
    labeled_rgb = np.asarray(Image.open(labeled_path).convert("RGB"))
    csv = pd.read_csv(csv_path)
    expected_count = int(len(csv))

    if labeled_rgb.shape[:2] != rgb.shape[:2]:
        raise ValueError(
            f"Overlay/labeled image dimensions differ: {overlay_path.name}={rgb.shape[:2]}, "
            f"{labeled_path.name}={labeled_rgb.shape[:2]}"
        )

    attempts: list[tuple[int, float]] = [
        (config.endpoint_peak_min_distance, config.endpoint_peak_threshold),
        (3, 2.2),
        (2, 2.2),
        (4, 2.2),
        (3, 1.8),
        (3, 2.5),
    ]
    # Remove duplicates while preserving order.
    seen = set()
    attempts = [x for x in attempts if not (x in seen or seen.add(x))]

    best: pd.DataFrame | None = None
    best_gap = float("inf")
    best_params = attempts[0]
    for idx, (min_distance, threshold_abs) in enumerate(attempts):
        truth = _truth_extraction_once(
            rgb,
            config,
            min_distance=min_distance,
            threshold_abs=threshold_abs,
        )
        gap = abs(len(truth) - expected_count)
        if gap < best_gap:
            best = truth
            best_gap = gap
            best_params = (min_distance, threshold_abs)
        if expected_count == 0:
            break
        if gap / expected_count <= config.truth_count_retry_fraction:
            break

    truth = best if best is not None else pd.DataFrame()
    if truth.empty:
        qa = {
            "csv_count": expected_count,
            "extracted_count": 0,
            "count_recall_proxy": 0.0,
            "length_scale": 1.0,
            "endpoint_params": best_params,
        }
        return truth, qa

    # Calibrate drawn pixel chord to ImageJ Length units. Exact row-to-row CSV
    # matching is impossible without CSV coordinates, so we use robust global
    # quantiles rather than inventing identities.
    csv_length = pd.to_numeric(csv.get("Length", pd.Series(dtype=float)), errors="coerce")
    csv_length = csv_length[np.isfinite(csv_length) & (csv_length > 0)]
    overlay_len = pd.to_numeric(truth["overlay_length_px"], errors="coerce")
    overlay_len = overlay_len[np.isfinite(overlay_len) & (overlay_len > 0)]
    if len(csv_length) and len(overlay_len):
        q = np.array([0.25, 0.50, 0.75])
        csv_q = np.quantile(csv_length, q)
        ov_q = np.quantile(overlay_len, q)
        ratios = csv_q / np.maximum(ov_q, 1e-6)
        length_scale = float(np.median(ratios[np.isfinite(ratios)]))
    else:
        length_scale = 1.0

    truth["truth_width"] = truth["overlay_length_px"] * length_scale
    truth["truth_angle_deg"] = truth["overlay_angle_deg"]
    truth["target"] = 1

    qa = {
        "csv_count": expected_count,
        "extracted_count": int(len(truth)),
        "yellow_count": int((truth["truth_source"] == "AUTO_KEEP").sum()),
        "cyan_count": int((truth["truth_source"] == "MANUAL_ADD").sum()),
        "count_recall_proxy": float(len(truth) / expected_count) if expected_count else float("nan"),
        "length_scale": float(length_scale),
        "endpoint_params": {
            "min_distance": int(best_params[0]),
            "threshold_abs": float(best_params[1]),
        },
        "csv_length_median": float(csv_length.median()) if len(csv_length) else float("nan"),
        "truth_width_median": float(truth["truth_width"].median()),
    }
    return truth.reset_index(drop=True), qa


# ---------------------------------------------------------------------------
# VisionFlux integration
# ---------------------------------------------------------------------------


def find_visionflux_file(root: str | Path, config: PipelineConfig) -> Path:
    root = Path(root)
    for name in config.visionflux_filename_candidates:
        candidate = root / name
        if candidate.exists():
            return candidate
    # Colab users often upload the algorithm into /content before moving it to
    # Drive. Accept that as a fallback.
    for name in config.visionflux_filename_candidates:
        for base in (Path.cwd(), Path("/content")):
            candidate = base / name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        "VisionFlux algorithm file not found. Put one of these files in the dataset root: "
        + ", ".join(config.visionflux_filename_candidates)
    )


def load_visionflux_module(path: str | Path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location("visionflux_algorithm", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import VisionFlux module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _use_original_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    out = frame.copy()
    mapping = {
        "center_x_original": "center_x",
        "center_y_original": "center_y",
        "x1_original": "x1",
        "y1_original": "y1",
        "x2_original": "x2",
        "y2_original": "y2",
    }
    for src, dst in mapping.items():
        if src in out.columns:
            out[dst] = pd.to_numeric(out[src], errors="coerce")
    if "width_original_px" in out.columns:
        out["width_px"] = pd.to_numeric(out["width_original_px"], errors="coerce")
    return out


def run_visionflux_passes(
    raw_path: str | Path,
    module: Any,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run default and intentionally broad VisionFlux passes.

    The wide pass raises max width / scale and lowers confidence gates so broad
    fibers are not discarded before the learned model sees them.
    """
    raw_path = Path(raw_path)
    blocks = []
    meta: dict[str, Any] = {}

    default_result = module.analyze_fiber_thickness(
        raw_path,
        max_dimension=None,
        crop_footer=False,
        prefer_gpu=True,
    )
    default_df = _use_original_coordinates(default_result.measurements)
    if not default_df.empty:
        default_df["proposal_source"] = "VISIONFLUX_DEFAULT"
        default_df["default_visionflux_hit"] = 1.0
        blocks.append(default_df)
    meta["default_summary"] = dict(default_result.summary)

    # Keep the existing algorithm, but deliberately create a second expert with
    # a wider scale/width prior. The learned model is free to reject it.
    try:
        wide_cfg = module.FastDetectorConfig(
            ridge_sigmas=(1.6, 2.4, 3.6, 5.2, 7.2, 10.0, 14.0, 20.0),
            orientation_sigma_px=5.0,
            sample_spacing_px=8.0,
            min_path_length_px=16.0,
            min_component_pixels=10,
            max_half_width_px=80.0,
            min_width_px=2.5,
            max_width_px=150.0,
            ridge_percentile=58.0,
            high_ridge_percentile=80.0,
            min_coherency=0.025,
            direction_split_deg=16.0,
            direction_split_persistence=3,
            max_measurements=7000,
            min_confidence=0.10,
            pore_reject_fraction=0.28,
        )
        wide_result = module.analyze_fiber_thickness(
            raw_path,
            max_dimension=None,
            crop_footer=False,
            prefer_gpu=True,
            config=wide_cfg,
        )
        wide_df = _use_original_coordinates(wide_result.measurements)
        if not wide_df.empty:
            wide_df["proposal_source"] = "VISIONFLUX_WIDE"
            wide_df["default_visionflux_hit"] = 0.0
            blocks.append(wide_df)
        meta["wide_summary"] = dict(wide_result.summary)
    except Exception as exc:
        warnings.warn(f"Wide VisionFlux pass failed: {type(exc).__name__}: {exc}")
        wide_result = default_result
        meta["wide_summary"] = {"error": f"{type(exc).__name__}: {exc}"}

    if not blocks:
        proposals = pd.DataFrame()
    else:
        proposals = pd.concat(blocks, ignore_index=True, sort=False)

    # Attach map references for downstream proposal generators without storing
    # huge arrays in the DataFrame.
    meta["orientation"] = default_result.orientation
    meta["ridge_response"] = np.asarray(default_result.ridge_response)
    meta["ridge_scale"] = np.asarray(default_result.ridge_scale)
    meta["pore_core"] = np.asarray(default_result.pore_core)
    return proposals, meta


# ---------------------------------------------------------------------------
# Candidate geometry and local features
# ---------------------------------------------------------------------------


def _candidate_normal_angle(frame: pd.DataFrame) -> np.ndarray:
    if all(c in frame.columns for c in ("x1", "y1", "x2", "y2")):
        vals = [
            line_angle_deg(x1, y1, x2, y2)
            for x1, y1, x2, y2 in frame[["x1", "y1", "x2", "y2"]].to_numpy(float)
        ]
        return np.asarray(vals, dtype=float)
    if "direction_deg" in frame.columns:
        return wrap90(pd.to_numeric(frame["direction_deg"], errors="coerce").to_numpy(float) + 90.0)
    return np.full(len(frame), np.nan, dtype=float)


def _sample_line_profile(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    normal_angle_deg: float,
    *,
    half_width: float,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    angle = math.radians(float(normal_angle_deg))
    # Cartesian angle -> image coordinates.
    dx = math.cos(angle)
    dy = -math.sin(angle)
    distances = np.linspace(-half_width, half_width, int(samples), dtype=np.float32)
    xs = center_x + dx * distances
    ys = center_y + dy * distances
    coords = np.vstack([ys, xs])
    values = ndimage.map_coordinates(
        np.asarray(image, dtype=np.float32),
        coords,
        order=1,
        mode="nearest",
    )
    return distances, np.asarray(values, dtype=np.float32)


def profile_features(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    normal_angle_deg: float,
    *,
    half_width: int = 80,
    samples: int = 161,
    normalized: bool = False,
) -> dict[str, float]:
    """Human-inspired normal-profile features.

    Captures both:
      1) relative 0-1-2-1-0 ridge shape independent of global brightness;
      2) dark/boundary-to-dark/boundary pairing on opposite sides.

    The returned values are evidence only; none is a hard fiber rule.
    """
    if not np.isfinite([center_x, center_y, normal_angle_deg]).all():
        return {k: float("nan") for k in PROFILE_FEATURE_NAMES}

    norm = np.asarray(image, dtype=np.float32) if normalized else robust_normalize(image)
    distances, raw = _sample_line_profile(
        norm,
        center_x,
        center_y,
        normal_angle_deg,
        half_width=half_width,
        samples=samples,
    )
    sm = ndimage.gaussian_filter1d(raw.astype(np.float32), 1.2, mode="nearest")
    grad = np.gradient(sm, distances)
    mid = len(sm) // 2
    left_slice = slice(0, max(1, mid - 2))
    right_slice = slice(min(len(sm) - 1, mid + 2), len(sm))

    left_candidates, _ = signal.find_peaks(-sm[left_slice], distance=3)
    right_candidates, _ = signal.find_peaks(-sm[right_slice], distance=3)
    right_offset = min(len(sm) - 1, mid + 2)
    right_candidates = right_candidates + right_offset

    # If local-minimum finding fails, fall back to darkest points on each side.
    if not len(left_candidates):
        left_candidates = np.asarray([int(np.argmin(sm[:mid]))]) if mid > 0 else np.asarray([], int)
    if not len(right_candidates):
        right_candidates = np.asarray([mid + int(np.argmin(sm[mid:]))]) if mid < len(sm) else np.asarray([], int)

    center_window = sm[max(0, mid - 2) : min(len(sm), mid + 3)]
    center_value = float(np.mean(center_window)) if len(center_window) else float(sm[mid])

    best_score = -np.inf
    best = None
    for li in left_candidates:
        for ri in right_candidates:
            if ri <= li:
                continue
            left_d = abs(float(distances[li]))
            right_d = abs(float(distances[ri]))
            width = left_d + right_d
            if width < 2.0:
                continue
            left_dark = float(sm[li])
            right_dark = float(sm[ri])
            bilateral_dark = 0.5 * (left_dark + right_dark)
            center_prominence = center_value - bilateral_dark
            symmetry = min(left_d, right_d) / max(left_d, right_d, 1e-6)

            # Strong opposite signed gradients near the two boundaries are good
            # evidence even when the fiber center is not globally bright.
            l0, l1 = max(0, li - 2), min(len(grad), li + 3)
            r0, r1 = max(0, ri - 2), min(len(grad), ri + 3)
            left_edge = float(np.max(np.abs(grad[l0:l1]))) if l1 > l0 else 0.0
            right_edge = float(np.max(np.abs(grad[r0:r1]))) if r1 > r0 else 0.0
            edge_pair = math.sqrt(max(left_edge, 0.0) * max(right_edge, 0.0))

            # Reward local ridge shape but do not require positive prominence;
            # thick/flat fibers can be rescued by edge evidence.
            score = 1.9 * edge_pair + 0.8 * max(center_prominence, 0.0) + 0.25 * symmetry
            if score > best_score:
                best_score = score
                best = (li, ri, width, left_d, right_d, left_dark, right_dark, edge_pair, symmetry)

    if best is None:
        best = (0, len(sm) - 1, float(2 * half_width), float(half_width), float(half_width), float(sm[0]), float(sm[-1]), 0.0, 1.0)
    li, ri, width, left_d, right_d, left_dark, right_dark, edge_pair, symmetry = best

    interior = sm[li : ri + 1] if ri >= li else sm
    outside_left = sm[:li] if li > 0 else sm[:1]
    outside_right = sm[ri + 1 :] if ri + 1 < len(sm) else sm[-1:]
    outside = np.concatenate([outside_left, outside_right])

    # 0-1-2-1-0-like relative ridge evidence: compare center to both boundaries
    # and to nearby outside without any absolute brightness threshold.
    boundary_mean = 0.5 * (left_dark + right_dark)
    relative_ridge = center_value - boundary_mean
    outside_mean = float(np.mean(outside)) if len(outside) else boundary_mean
    interior_mean = float(np.mean(interior)) if len(interior) else center_value
    interior_std = float(np.std(interior)) if len(interior) else 0.0

    return {
        "profile_width_px": float(width),
        "profile_left_px": float(left_d),
        "profile_right_px": float(right_d),
        "profile_edge_pair": float(edge_pair),
        "profile_symmetry": float(symmetry),
        "profile_center": float(center_value),
        "profile_left_dark": float(left_dark),
        "profile_right_dark": float(right_dark),
        "relative_ridge": float(relative_ridge),
        "profile_inside_outside": float(interior_mean - outside_mean),
        "profile_interior_std": float(interior_std),
        "profile_score": float(max(best_score, 0.0)),
    }


PROFILE_FEATURE_NAMES = [
    "profile_width_px",
    "profile_left_px",
    "profile_right_px",
    "profile_edge_pair",
    "profile_symmetry",
    "profile_center",
    "profile_left_dark",
    "profile_right_dark",
    "relative_ridge",
    "profile_inside_outside",
    "profile_interior_std",
    "profile_score",
]


def patch_features(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    size: int = 48,
    *,
    normalized: bool = False,
) -> dict[str, float]:
    norm = np.asarray(image, dtype=np.float32) if normalized else robust_normalize(image)
    r = size // 2
    x = int(round(center_x))
    y = int(round(center_y))
    y0, y1 = max(0, y - r), min(norm.shape[0], y + r + 1)
    x0, x1 = max(0, x - r), min(norm.shape[1], x + r + 1)
    patch = norm[y0:y1, x0:x1]
    if patch.size == 0:
        return {"patch_mean": np.nan, "patch_std": np.nan, "patch_p10": np.nan, "patch_p90": np.nan, "patch_grad": np.nan}
    gy, gx = np.gradient(patch)
    gm = np.hypot(gx, gy)
    return {
        "patch_mean": float(np.mean(patch)),
        "patch_std": float(np.std(patch)),
        "patch_p10": float(np.percentile(patch, 10)),
        "patch_p90": float(np.percentile(patch, 90)),
        "patch_grad": float(np.mean(gm)),
    }


def _deduplicate_candidates(frame: pd.DataFrame, radius_px: float = 4.0, angle_tol: float = 15.0) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    frame = frame.reset_index(drop=True).copy()
    if "candidate_score" not in frame.columns:
        frame["candidate_score"] = pd.to_numeric(frame.get("confidence", 0.0), errors="coerce").fillna(0.0)
    order = np.argsort(-pd.to_numeric(frame["candidate_score"], errors="coerce").fillna(0).to_numpy())
    centers = frame[["center_x", "center_y"]].to_numpy(float)
    angles = _candidate_normal_angle(frame)
    kept: list[int] = []
    suppressed = np.zeros(len(frame), dtype=bool)
    tree = cKDTree(centers)
    for idx in order:
        if suppressed[idx]:
            continue
        kept.append(int(idx))
        for j in tree.query_ball_point(centers[idx], r=radius_px):
            if j == idx or suppressed[j]:
                continue
            if np.isfinite(angles[idx]) and np.isfinite(angles[j]) and axial_error_deg(angles[idx], angles[j]) > angle_tol:
                continue
            suppressed[j] = True
    return frame.iloc[kept].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Additional candidate generators
# ---------------------------------------------------------------------------


def generate_relative_ridge_candidates(
    image: np.ndarray,
    orientation: Any,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Generate locally bright candidates without an absolute brightness gate.

    A multi-scale difference of Gaussians responds to a 0-1-2-1-0 local shape
    even when the entire neighborhood is relatively dark in the global SEM.
    """
    img = robust_normalize(image)
    best = np.zeros_like(img, dtype=np.float32)
    best_scale = np.zeros_like(img, dtype=np.float32)
    for small, large in config.relative_ridge_scales:
        a = ndimage.gaussian_filter(img, small, mode="reflect")
        b = ndimage.gaussian_filter(img, large, mode="reflect")
        response = np.maximum(a - b, 0.0).astype(np.float32)
        take = response > best
        best[take] = response[take]
        best_scale[take] = float(large)

    positive = best[best > 0]
    if not len(positive):
        return pd.DataFrame()
    threshold = float(np.percentile(positive, config.relative_ridge_percentile))
    mask = best >= threshold
    mask = closing(mask, disk(1))
    try:
        mask = remove_small_objects(mask, min_size=10)
    except Exception:
        pass
    sk = skeletonize(mask)
    yy, xx = np.where(sk)
    if not len(xx):
        return pd.DataFrame()

    # Keep approximately one point per requested spacing along the raw skeleton.
    stride = max(1, int(round(config.relative_ridge_sample_spacing_px)))
    order = np.lexsort((xx, yy))
    yy, xx = yy[order][::stride], xx[order][::stride]

    theta = np.asarray(getattr(orientation, "theta"))
    coh = np.asarray(getattr(orientation, "coherency"))
    rows = []
    for y, x in zip(yy, xx):
        tangent = float(theta[y, x]) if theta.shape == img.shape else 0.0
        normal = wrap90(tangent + 90.0)
        pf = profile_features(
            img,
            float(x),
            float(y),
            normal,
            half_width=config.profile_half_width_px,
            samples=config.profile_samples,
            normalized=True,
        )
        width = pf["profile_width_px"]
        rad = math.radians(normal)
        dx, dy = math.cos(rad), -math.sin(rad)
        half = width / 2.0
        rows.append(
            {
                "center_x": float(x),
                "center_y": float(y),
                "x1": float(x - dx * half),
                "y1": float(y - dy * half),
                "x2": float(x + dx * half),
                "y2": float(y + dy * half),
                "width_px": float(width),
                "direction_deg": float(tangent),
                "local_orientation_deg": float(tangent),
                "local_coherency": float(coh[y, x]) if coh.shape == img.shape else np.nan,
                "ridge_score": float(best[y, x] / max(float(np.percentile(positive, 99)), 1e-9)),
                "confidence": float(np.clip(pf["profile_score"], 0, 1)),
                "candidate_score": float(best[y, x]),
                "proposal_source": "RELATIVE_RIDGE",
                "default_visionflux_hit": 0.0,
                **pf,
            }
        )
    return pd.DataFrame(rows)


def generate_profile_recovery_candidates(
    image: np.ndarray,
    orientation: Any,
    default_proposals: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Broad/weak-response recovery based on bilateral normal-profile evidence.

    The generator is deliberately permissive. It asks: "Could two opposite
    dark/edge boundaries form a broad fiber here?" The learned classifier later
    decides. Weak default VisionFlux response is included as a feature so the
    model can learn the user's proposed reverse signal for thick fibers.
    """
    img = robust_normalize(image)
    theta = np.asarray(getattr(orientation, "theta"))
    coh = np.asarray(getattr(orientation, "coherency"))
    energy = np.asarray(getattr(orientation, "energy"))
    energy_n = robust_normalize(energy) if energy.shape == img.shape else np.zeros_like(img)

    # Distance to default VisionFlux proposal is a continuous "miss" feature.
    if default_proposals is not None and not default_proposals.empty:
        default_centers = default_proposals.loc[
            default_proposals.get("proposal_source", "") == "VISIONFLUX_DEFAULT",
            ["center_x", "center_y"],
        ].to_numpy(float)
        default_tree = cKDTree(default_centers) if len(default_centers) else None
    else:
        default_tree = None

    rows = []
    stride = int(config.recovery_grid_stride_px)
    h, w = img.shape
    for y in range(stride // 2, h, stride):
        for x in range(stride // 2, w, stride):
            tangent = float(theta[y, x]) if theta.shape == img.shape else 0.0
            normal = wrap90(tangent + 90.0)
            pf = profile_features(
                img,
                float(x),
                float(y),
                normal,
                half_width=config.profile_half_width_px,
                samples=config.profile_samples,
                normalized=True,
            )
            # Fast permissive filter: either a convincing edge pair or a clear
            # relative ridge. This prevents the entire image grid entering the
            # candidate pool while still admitting broad weak-center fibers.
            if (
                pf["profile_score"] < config.recovery_min_profile_score
                and pf["profile_edge_pair"] < 0.015
                and pf["relative_ridge"] < 0.025
            ):
                continue

            if default_tree is not None:
                dist_default = float(default_tree.query([x, y], k=1)[0])
            else:
                dist_default = float("inf")
            missed_default = float(dist_default > max(5.0, 0.25 * pf["profile_width_px"]))

            width = float(pf["profile_width_px"])
            rad = math.radians(normal)
            dx, dy = math.cos(rad), -math.sin(rad)
            half = width / 2.0
            rows.append(
                {
                    "center_x": float(x),
                    "center_y": float(y),
                    "x1": float(x - dx * half),
                    "y1": float(y - dy * half),
                    "x2": float(x + dx * half),
                    "y2": float(y + dy * half),
                    "width_px": width,
                    "direction_deg": float(tangent),
                    "local_orientation_deg": float(tangent),
                    "local_coherency": float(coh[y, x]) if coh.shape == img.shape else np.nan,
                    "orientation_energy": float(energy_n[y, x]),
                    "distance_to_default": float(dist_default if np.isfinite(dist_default) else 999.0),
                    "missed_by_default": missed_default,
                    "confidence": 0.0,
                    "candidate_score": float(
                        pf["profile_score"]
                        + 0.25 * pf["profile_edge_pair"]
                        + 0.05 * min(width / 25.0, 3.0) * missed_default
                    ),
                    "proposal_source": "PROFILE_RECOVERY",
                    "default_visionflux_hit": 0.0,
                    **pf,
                }
            )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    if len(out) > config.recovery_max_candidates_per_image:
        out = out.nlargest(config.recovery_max_candidates_per_image, "candidate_score")
    return out.reset_index(drop=True)


def add_local_features(frame: pd.DataFrame, image: np.ndarray, config: PipelineConfig) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy().reset_index(drop=True)
    norm_image = robust_normalize(image)
    normal_angles = _candidate_normal_angle(out)
    out["normal_angle_deg"] = normal_angles

    # Profile columns may already exist for recovery proposals.
    prof_rows = []
    patch_rows = []
    for row, normal in zip(out.itertuples(index=False), normal_angles):
        cx = float(getattr(row, "center_x"))
        cy = float(getattr(row, "center_y"))
        prof_rows.append(
            profile_features(
                norm_image,
                cx,
                cy,
                normal,
                half_width=config.profile_half_width_px,
                samples=config.profile_samples,
                normalized=True,
            )
        )
        patch_rows.append(patch_features(norm_image, cx, cy, config.patch_size, normalized=True))
    prof = pd.DataFrame(prof_rows)
    patch = pd.DataFrame(patch_rows)
    for col in prof.columns:
        if col not in out.columns:
            out[col] = prof[col]
        else:
            current = pd.to_numeric(out[col], errors="coerce")
            out[col] = current.where(np.isfinite(current), prof[col])
    for col in patch.columns:
        out[col] = patch[col]

    # Relative within-image width rank is learned rather than hard-coding a
    # "thick = 25px" rule.
    width = pd.to_numeric(out.get("profile_width_px", out.get("width_px", np.nan)), errors="coerce")
    out["width_rank_pct"] = width.rank(pct=True, method="average").fillna(0.5)
    out["broad_missed_interaction"] = (
        out["width_rank_pct"]
        * pd.to_numeric(out.get("missed_by_default", 0.0), errors="coerce").fillna(0.0)
        * pd.to_numeric(out.get("profile_edge_pair", 0.0), errors="coerce").fillna(0.0)
    )
    return out


# ---------------------------------------------------------------------------
# Supervision: kept, removed, manually added
# ---------------------------------------------------------------------------


def match_auto_to_truth(
    auto: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    center_tolerance_px: float = 10.0,
    angle_tolerance_deg: float = 28.0,
    width_ratio_max: float = 2.4,
) -> pd.DataFrame:
    """Create explicit supervision without turning all unlabeled space negative."""
    auto = auto.copy().reset_index(drop=True)
    truth = truth.copy().reset_index(drop=True)
    if auto.empty:
        rows = []
        for t in truth.itertuples(index=False):
            d = t._asdict()
            d.update(supervision=str(d.get("truth_source", "MANUAL_ADD")), target=1, matched_truth=1)
            rows.append(d)
        return pd.DataFrame(rows)

    if "proposal_source" in auto.columns:
        base_mask = auto["proposal_source"].astype(str).eq("VISIONFLUX_DEFAULT")
        # Only default proposals are meaningful AUTO_REMOVE negatives because the
        # user reviewed the original VisionFlux output, not our new recovery passes.
        reviewed_auto = auto.loc[base_mask].copy().reset_index(drop=True)
    else:
        reviewed_auto = auto.copy()

    keep_truth = truth.loc[truth["truth_source"].astype(str).eq("AUTO_KEEP")].copy()
    manual_truth = truth.loc[truth["truth_source"].astype(str).eq("MANUAL_ADD")].copy()

    matched_auto: set[int] = set()
    rows: list[dict[str, Any]] = []

    if not reviewed_auto.empty and not keep_truth.empty:
        centers = reviewed_auto[["center_x", "center_y"]].to_numpy(float)
        tree = cKDTree(centers)
        auto_normal = _candidate_normal_angle(reviewed_auto)
        if "width_px" in reviewed_auto.columns:
            auto_width = pd.to_numeric(reviewed_auto["width_px"], errors="coerce").to_numpy(float)
        elif all(c in reviewed_auto.columns for c in ("x1", "y1", "x2", "y2")):
            auto_width = np.hypot(
                reviewed_auto["x2"].to_numpy(float) - reviewed_auto["x1"].to_numpy(float),
                reviewed_auto["y2"].to_numpy(float) - reviewed_auto["y1"].to_numpy(float),
            )
        else:
            auto_width = np.full(len(reviewed_auto), np.nan, dtype=float)

        for t in keep_truth.itertuples(index=False):
            td = t._asdict()
            tc = np.array([float(td["center_x"]), float(td["center_y"])])
            indices = tree.query_ball_point(tc, r=float(center_tolerance_px))
            best_idx = None
            best_cost = float("inf")
            truth_angle = float(td.get("truth_angle_deg", td.get("overlay_angle_deg", np.nan)))
            truth_width = float(td.get("truth_width", td.get("overlay_length_px", np.nan)))
            for idx in indices:
                angle_err = axial_error_deg(auto_normal[idx], truth_angle) if np.isfinite(auto_normal[idx]) and np.isfinite(truth_angle) else 0.0
                if angle_err > angle_tolerance_deg:
                    continue
                aw = auto_width[idx]
                if np.isfinite(aw) and np.isfinite(truth_width) and min(aw, truth_width) > 0:
                    ratio = max(aw, truth_width) / min(aw, truth_width)
                    if ratio > width_ratio_max:
                        continue
                else:
                    ratio = 1.0
                dist = float(np.linalg.norm(centers[idx] - tc))
                cost = dist / max(center_tolerance_px, 1e-6) + angle_err / max(angle_tolerance_deg, 1e-6) + 0.15 * abs(math.log(max(ratio, 1e-6)))
                if cost < best_cost and idx not in matched_auto:
                    best_cost = cost
                    best_idx = idx
            if best_idx is not None:
                matched_auto.add(best_idx)
                row = reviewed_auto.iloc[best_idx].to_dict()
                row.update(
                    supervision="AUTO_KEEP",
                    target=1,
                    matched_truth=1,
                    truth_width=truth_width,
                    truth_angle_deg=truth_angle,
                    truth_center_x=float(td["center_x"]),
                    truth_center_y=float(td["center_y"]),
                )
                rows.append(row)
            else:
                # The kept yellow truth exists but the current detector no longer
                # proposes it. Preserve it as a positive training example.
                row = td.copy()
                row.update(
                    proposal_source="TRUTH_YELLOW_UNMATCHED",
                    supervision="AUTO_KEEP",
                    target=1,
                    matched_truth=1,
                    width_px=truth_width,
                    confidence=0.0,
                    default_visionflux_hit=0.0,
                )
                rows.append(row)

    # Reviewed default proposals not present in yellow are exactly the user's
    # removed VisionFlux measurements: high-value hard negatives.
    for idx, r in reviewed_auto.iterrows():
        if idx in matched_auto:
            continue
        row = r.to_dict()
        row.update(supervision="AUTO_REMOVE", target=0, matched_truth=0)
        rows.append(row)

    # Cyan lines are human-added hard positives regardless of whether another
    # experimental generator happens to be nearby.
    for t in manual_truth.itertuples(index=False):
        row = t._asdict()
        row.update(
            proposal_source="TRUTH_MANUAL_ADD",
            supervision="MANUAL_ADD",
            target=1,
            matched_truth=1,
            width_px=float(row.get("truth_width", row.get("overlay_length_px", np.nan))),
            confidence=0.0,
            default_visionflux_hit=0.0,
        )
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full candidate/supervision table for one image
# ---------------------------------------------------------------------------


def build_image_training_table(
    stem: str,
    raw_path: str | Path,
    overlay_path: str | Path,
    labeled_path: str | Path,
    csv_path: str | Path,
    module: Any,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    image = load_gray(raw_path)
    truth, truth_qa = extract_truth_measurements(overlay_path, labeled_path, csv_path, config)
    proposals, vf_meta = run_visionflux_passes(raw_path, module, config)

    supervised = match_auto_to_truth(
        proposals,
        truth,
        center_tolerance_px=config.match_center_tolerance_px,
        angle_tolerance_deg=config.match_angle_tolerance_deg,
        width_ratio_max=config.match_width_ratio_max,
    )

    # Add new generator proposals only as *unlabeled candidates*. They are useful
    # for inference and model feature distributions, but are never automatically
    # assigned target=0 merely because a human did not see them in VisionFlux.
    orientation = vf_meta["orientation"]
    relative = generate_relative_ridge_candidates(image, orientation, config)
    recovery = generate_profile_recovery_candidates(image, orientation, proposals, config)

    additional = []
    wide_only = proposals.loc[proposals.get("proposal_source", pd.Series(index=proposals.index, dtype=str)).astype(str).eq("VISIONFLUX_WIDE")].copy() if not proposals.empty else pd.DataFrame()
    for frame in (wide_only, relative, recovery):
        if frame is not None and not frame.empty:
            frame = frame.copy()
            frame["target"] = np.nan
            frame["supervision"] = "UNLABELED_PROPOSAL"
            additional.append(frame)

    blocks = [supervised] + additional if not supervised.empty else additional
    if not blocks:
        return pd.DataFrame(), {"truth_qa": truth_qa, "visionflux": vf_meta.get("default_summary", {})}
    table = pd.concat(blocks, ignore_index=True, sort=False)
    table["stem"] = stem
    table["raw_path"] = str(raw_path)
    table = add_local_features(table, image, config)

    # Attach map features at candidate centers where available.
    ridge = np.asarray(vf_meta.get("ridge_response"))
    pore = np.asarray(vf_meta.get("pore_core"))
    orientation = vf_meta.get("orientation")
    theta = np.asarray(getattr(orientation, "theta"))
    coh = np.asarray(getattr(orientation, "coherency"))
    energy = np.asarray(getattr(orientation, "energy"))
    energy_n = robust_normalize(energy) if energy.shape == image.shape else np.zeros_like(image, np.float32)

    map_rows = []
    for r in table.itertuples(index=False):
        x = int(np.clip(round(float(getattr(r, "center_x"))), 0, image.shape[1] - 1))
        y = int(np.clip(round(float(getattr(r, "center_y"))), 0, image.shape[0] - 1))
        map_rows.append(
            {
                "map_ridge": float(ridge[y, x]) if ridge.shape == image.shape else np.nan,
                "map_pore": float(pore[y, x]) if pore.shape == image.shape else np.nan,
                "map_coherency": float(coh[y, x]) if coh.shape == image.shape else np.nan,
                "map_energy": float(energy_n[y, x]),
                "map_theta": float(theta[y, x]) if theta.shape == image.shape else np.nan,
            }
        )
    maps = pd.DataFrame(map_rows)
    for col in maps.columns:
        table[col] = maps[col]

    qa = {
        "stem": stem,
        "truth_qa": truth_qa,
        "visionflux_default": vf_meta.get("default_summary", {}),
        "visionflux_wide": vf_meta.get("wide_summary", {}),
        "supervision_counts": table["supervision"].value_counts(dropna=False).to_dict(),
        "candidate_count": int(len(table)),
    }
    return table, qa


# ---------------------------------------------------------------------------
# Caching / incremental dataset build
# ---------------------------------------------------------------------------


def _cache_paths(config: PipelineConfig, stem: str) -> tuple[Path, Path]:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    return config.cache_dir / f"{safe}.pkl", config.cache_dir / f"{safe}.json"


def build_training_dataset(
    manifest: pd.DataFrame,
    module: Any,
    config: PipelineConfig,
    *,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    blocks = []
    qa_rows: list[dict[str, Any]] = []

    complete = manifest.loc[manifest["complete_truth"].astype(bool)].copy()
    for i, row in enumerate(complete.itertuples(index=False), start=1):
        stem = str(row.stem)
        paths = [row.raw_path, row.overlay_path, row.labeled_path, row.csv_path]
        sig = file_signature(paths)
        cache_pkl, cache_json = _cache_paths(config, stem)
        use_cache = False
        if not force_rebuild and cache_pkl.exists() and cache_json.exists():
            try:
                meta = json.loads(cache_json.read_text(encoding="utf-8"))
                use_cache = meta.get("signature") == sig
            except Exception:
                use_cache = False

        print(f"[{i}/{len(complete)}] {stem}: {'cache' if use_cache else 'analyze'}")
        if use_cache:
            with cache_pkl.open("rb") as f:
                table = pickle.load(f)
            qa = meta.get("qa", {})
        else:
            table, qa = build_image_training_table(
                stem,
                row.raw_path,
                row.overlay_path,
                row.labeled_path,
                row.csv_path,
                module,
                config,
            )
            with cache_pkl.open("wb") as f:
                pickle.dump(table, f, protocol=pickle.HIGHEST_PROTOCOL)
            cache_json.write_text(
                json.dumps({"signature": sig, "qa": qa}, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )
        if table is not None and not table.empty:
            blocks.append(table)
        qa_rows.append(qa)

    if not blocks:
        return pd.DataFrame(), qa_rows
    return pd.concat(blocks, ignore_index=True, sort=False), qa_rows


def _json_default(x: Any):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return str(x)


# ---------------------------------------------------------------------------
# Stable image-level split
# ---------------------------------------------------------------------------


def _group_bucket(text: str, seed: int = 20260821) -> int:
    raw = f"{seed}:{text}".encode()
    return int(hashlib.sha1(raw).hexdigest()[:8], 16) % 10


def stable_group_split(frame: pd.DataFrame, group_col: str = "stem", seed: int = 20260821) -> pd.DataFrame:
    out = frame.copy()
    groups = sorted(out[group_col].dropna().astype(str).unique())
    if len(groups) < 5:
        # Deterministic fallback that still never leaks the same SEM across sets.
        assignment = {}
        for i, g in enumerate(groups):
            if i == len(groups) - 1 and len(groups) >= 3:
                split = "test"
            elif i == len(groups) - 2 and len(groups) >= 3:
                split = "val"
            else:
                split = "train"
            assignment[g] = split
    else:
        assignment = {}
        for g in groups:
            b = _group_bucket(g, seed)
            assignment[g] = "test" if b in (0, 1) else ("val" if b == 2 else "train")
        # Hashes can occasionally leave a split empty on small sets. Fix only at
        # group level while keeping determinism.
        counts = pd.Series(list(assignment.values())).value_counts()
        for missing in ("test", "val"):
            if counts.get(missing, 0) == 0:
                donor = next(g for g in reversed(groups) if assignment[g] == "train")
                assignment[donor] = missing
                counts = pd.Series(list(assignment.values())).value_counts()
    out["split"] = out[group_col].astype(str).map(assignment)
    return out



# ---------------------------------------------------------------------------
# Candidate-generator comparison (before learned classification)
# ---------------------------------------------------------------------------


def _generator_hits_truth(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    center_tolerance_px: float,
    angle_tolerance_deg: float,
) -> np.ndarray:
    if truth.empty or candidates.empty:
        return np.zeros(len(truth), dtype=bool)
    centers = candidates[["center_x", "center_y"]].to_numpy(float)
    valid_centers = np.isfinite(centers).all(axis=1)
    candidates = candidates.loc[valid_centers].reset_index(drop=True)
    centers = centers[valid_centers]
    if not len(candidates):
        return np.zeros(len(truth), dtype=bool)
    tree = cKDTree(centers)
    cand_angle = _candidate_normal_angle(candidates)
    truth_angle = _candidate_normal_angle(truth)
    hits = np.zeros(len(truth), dtype=bool)
    for i, t in enumerate(truth.itertuples(index=False)):
        tc = [float(getattr(t, "center_x")), float(getattr(t, "center_y"))]
        for j in tree.query_ball_point(tc, r=float(center_tolerance_px)):
            if np.isfinite(cand_angle[j]) and np.isfinite(truth_angle[i]):
                if axial_error_deg(cand_angle[j], truth_angle[i]) > angle_tolerance_deg:
                    continue
            hits[i] = True
            break
    return hits


def evaluate_generators(
    dataset: pd.DataFrame,
    config: PipelineConfig,
    *,
    thick_quantile: float = 0.85,
) -> pd.DataFrame:
    """Compare proposal generators independently of the learned classifier.

    "Thick" is defined *within each SEM* using the truth-width quantile, not a
    fixed pixel threshold. This makes the report directly test the user's idea
    that a recovery expert may be especially valuable on the broadest fibers.
    """
    if dataset.empty:
        return pd.DataFrame()
    generators = [
        "VISIONFLUX_DEFAULT",
        "VISIONFLUX_WIDE",
        "RELATIVE_RIDGE",
        "PROFILE_RECOVERY",
    ]
    rows = []
    per_gen = {g: {"truth": 0, "hit": 0, "manual": 0, "manual_hit": 0, "thick": 0, "thick_hit": 0, "removed": 0, "removed_hit": 0} for g in generators}

    for stem, group in dataset.groupby("stem", sort=False):
        positives = group.loc[group["target"].eq(1)].copy()
        # Each positive supervision row is one human-approved measurement. Avoid
        # duplicates if a future cache contains repeated copies.
        if positives.empty:
            continue
        positives = positives.drop_duplicates(subset=["supervision", "center_x", "center_y"])
        widths = pd.to_numeric(positives.get("truth_width", positives.get("width_px", np.nan)), errors="coerce")
        finite_widths = widths[np.isfinite(widths)]
        cutoff = float(np.quantile(finite_widths, thick_quantile)) if len(finite_widths) else float("inf")
        is_thick = np.asarray(widths >= cutoff, dtype=bool)
        is_manual = positives["supervision"].astype(str).eq("MANUAL_ADD").to_numpy()
        removed = group.loc[group["supervision"].astype(str).eq("AUTO_REMOVE")].copy()

        for gen in generators:
            cand = group.loc[group["proposal_source"].astype(str).eq(gen)].copy()
            hit = _generator_hits_truth(
                cand,
                positives,
                center_tolerance_px=config.match_center_tolerance_px,
                angle_tolerance_deg=config.match_angle_tolerance_deg,
            )
            removed_hit = _generator_hits_truth(
                cand,
                removed,
                center_tolerance_px=config.match_center_tolerance_px,
                angle_tolerance_deg=config.match_angle_tolerance_deg,
            ) if not removed.empty else np.zeros(0, dtype=bool)
            d = per_gen[gen]
            d["truth"] += len(positives)
            d["hit"] += int(hit.sum())
            d["manual"] += int(is_manual.sum())
            d["manual_hit"] += int((hit & is_manual).sum())
            d["thick"] += int(is_thick.sum())
            d["thick_hit"] += int((hit & is_thick).sum())
            d["removed"] += len(removed)
            d["removed_hit"] += int(removed_hit.sum())

    for gen, d in per_gen.items():
        truth_recall = d["hit"] / d["truth"] if d["truth"] else float("nan")
        manual_recall = d["manual_hit"] / d["manual"] if d["manual"] else float("nan")
        thick_recall = d["thick_hit"] / d["thick"] if d["thick"] else float("nan")
        removed_hit_rate = d["removed_hit"] / d["removed"] if d["removed"] else float("nan")
        rows.append({
            "generator": gen,
            "truth_recall": float(truth_recall),
            "manual_add_recall": float(manual_recall),
            "thick_recall": float(thick_recall),
            "removed_hit_rate": float(removed_hit_rate),
            "truth_count": int(d["truth"]),
            "manual_count": int(d["manual"]),
            "thick_count": int(d["thick"]),
            "removed_count": int(d["removed"]),
        })
    return pd.DataFrame(rows).sort_values(["thick_recall", "truth_recall"], ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature model competition
# ---------------------------------------------------------------------------


NON_FEATURE_COLUMNS = {
    "target",
    "split",
    "stem",
    "raw_path",
    "supervision",
    "proposal_source",
    "measurement_id",
    "detector",
    "status",
    "grade",
    "reason",
    "truth_source",
}


def choose_feature_columns(frame: pd.DataFrame) -> list[str]:
    cols = []
    preferred = [
        "confidence",
        "sem_agreement",
        "bundle_score",
        "orientation_error_deg",
        "orientation_score",
        "ridge_score",
        "edge_score",
        "pore_fraction",
        "local_coherency",
        "map_ridge",
        "map_pore",
        "map_coherency",
        "map_energy",
        "width_px",
        "profile_width_px",
        "profile_left_px",
        "profile_right_px",
        "profile_edge_pair",
        "profile_symmetry",
        "profile_center",
        "profile_left_dark",
        "profile_right_dark",
        "relative_ridge",
        "profile_inside_outside",
        "profile_interior_std",
        "profile_score",
        "patch_mean",
        "patch_std",
        "patch_p10",
        "patch_p90",
        "patch_grad",
        "distance_to_default",
        "missed_by_default",
        "default_visionflux_hit",
        "width_rank_pct",
        "broad_missed_interaction",
        "orientation_energy",
    ]
    for c in preferred:
        if c in frame.columns:
            cols.append(c)
    # Include other numeric VisionFlux columns automatically, except coordinates
    # and target leakage fields.
    for c in frame.columns:
        if c in cols or c in NON_FEATURE_COLUMNS or c.startswith("truth_"):
            continue
        if c in {"x1", "y1", "x2", "y2", "center_x", "center_y", "xm", "ym"}:
            continue
        if pd.api.types.is_numeric_dtype(frame[c]):
            cols.append(c)
    return cols


def _make_sklearn_models(seed: int) -> dict[str, Any]:
    return {
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                        max_features="sqrt",
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=450,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                        max_features="sqrt",
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=250,
                        learning_rate=0.055,
                        max_leaf_nodes=31,
                        min_samples_leaf=12,
                        l2_regularization=0.8,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "mlp": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(96, 48, 24),
                        alpha=1e-3,
                        learning_rate_init=1e-3,
                        max_iter=350,
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def _predict_proba_binary(model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return np.asarray(p[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        z = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-z))
    return np.asarray(model.predict(X), dtype=float)


def _score_at_threshold(
    frame: pd.DataFrame,
    probability: np.ndarray,
    threshold: float,
    config: PipelineConfig,
) -> dict[str, float]:
    y = frame["target"].astype(int).to_numpy()
    pred = (probability >= threshold).astype(int)
    precision = precision_score(y, pred, zero_division=0)
    recall = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)

    sup = frame["supervision"].astype(str).to_numpy()
    manual = sup == "MANUAL_ADD"
    removed = sup == "AUTO_REMOVE"
    manual_recall = float((pred[manual] == 1).mean()) if manual.any() else recall
    hard_negative_rejection = float((pred[removed] == 0).mean()) if removed.any() else 1.0 - max(0.0, 1.0 - precision)
    try:
        ap = average_precision_score(y, probability)
    except Exception:
        ap = float("nan")
    try:
        auc = roc_auc_score(y, probability) if len(np.unique(y)) > 1 else float("nan")
    except Exception:
        auc = float("nan")

    primary = (
        config.score_f1_weight * f1
        + config.score_manual_add_recall_weight * manual_recall
        + config.score_hard_negative_rejection_weight * hard_negative_rejection
        + config.score_average_precision_weight * (ap if np.isfinite(ap) else f1)
    )
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "manual_add_recall": float(manual_recall),
        "hard_negative_rejection": float(hard_negative_rejection),
        "average_precision": float(ap),
        "roc_auc": float(auc),
        "primary_score": float(primary),
    }


def _best_threshold(frame: pd.DataFrame, probability: np.ndarray, config: PipelineConfig) -> dict[str, float]:
    scored = [_score_at_threshold(frame, probability, t, config) for t in config.threshold_grid]
    return max(scored, key=lambda x: (x["primary_score"], x["f1"], x["precision"]))


def _visionflux_baseline_probability(frame: pd.DataFrame) -> np.ndarray:
    conf = pd.to_numeric(frame.get("confidence", 0.0), errors="coerce").fillna(0.0).to_numpy(float)
    # Manual additions are intentionally not given knowledge of the truth. The
    # baseline receives 0 if VisionFlux had no proposal there, which measures the
    # actual limitation we want learned models to overcome.
    return np.clip(conf, 0.0, 1.0)


def select_champion_name(leaderboard: pd.DataFrame) -> str:
    if leaderboard.empty:
        raise ValueError("Leaderboard is empty")
    idx = pd.to_numeric(leaderboard["primary_score"], errors="coerce").idxmax()
    return str(leaderboard.loc[idx, "model"])


def train_sklearn_competition(
    supervised: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    if supervised.empty:
        raise ValueError("No supervised rows available")
    data = stable_group_split(supervised, "stem", config.seed)
    features = choose_feature_columns(data)
    if not features:
        raise ValueError("No numeric feature columns found")

    train = data.loc[data["split"] == "train"].copy()
    val = data.loc[data["split"] == "val"].copy()
    test = data.loc[data["split"] == "test"].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError(
            f"Need non-empty train/val/test image groups; got {len(train)}/{len(val)}/{len(test)} rows"
        )

    results: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}

    # Fixed-rule VisionFlux baseline.
    val_p = _visionflux_baseline_probability(val)
    best = _best_threshold(val, val_p, config)
    test_p = _visionflux_baseline_probability(test)
    test_metrics = _score_at_threshold(test, test_p, best["threshold"], config)
    results.append(
        {
            "model": "visionflux_fixed_rule",
            **{f"val_{k}": v for k, v in best.items()},
            **test_metrics,
            "train_rows": len(train),
            "val_rows": len(val),
            "test_rows": len(test),
        }
    )
    fitted["visionflux_fixed_rule"] = {
        "baseline": True,
        "model": None,
        "threshold": best["threshold"],
        "features": [],
    }

    for name, model in _make_sklearn_models(config.seed).items():
        model.fit(train[features], train["target"].astype(int))
        val_p = _predict_proba_binary(model, val[features])
        best = _best_threshold(val, val_p, config)
        test_p = _predict_proba_binary(model, test[features])
        test_metrics = _score_at_threshold(test, test_p, best["threshold"], config)
        results.append(
            {
                "model": name,
                **{f"val_{k}": v for k, v in best.items()},
                **test_metrics,
                "train_rows": len(train),
                "val_rows": len(val),
                "test_rows": len(test),
            }
        )
        fitted[name] = {"model": model, "threshold": best["threshold"], "features": features}

    board = pd.DataFrame(results).sort_values("primary_score", ascending=False).reset_index(drop=True)
    return board, fitted, features


# ---------------------------------------------------------------------------
# Optional PyTorch image/profile models
# ---------------------------------------------------------------------------


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _extract_patch(image: np.ndarray, cx: float, cy: float, size: int, *, normalized: bool = False) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32) if normalized else robust_normalize(image)
    r = size // 2
    x = int(round(cx))
    y = int(round(cy))
    pad = r + 2
    padded = np.pad(img, pad, mode="reflect")
    xp, yp = x + pad, y + pad
    patch = padded[yp - r : yp - r + size, xp - r : xp - r + size]
    if patch.shape != (size, size):
        patch = np.asarray(Image.fromarray((patch * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    return patch.astype(np.float32)


def _candidate_profile_vector(
    image: np.ndarray, row: pd.Series, config: PipelineConfig, *, normalized: bool = False
) -> np.ndarray:
    normal = line_angle_deg(float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])) if all(c in row for c in ("x1", "y1", "x2", "y2")) else float(row.get("normal_angle_deg", 0.0))
    _, profile = _sample_line_profile(
        np.asarray(image, dtype=np.float32) if normalized else robust_normalize(image),
        float(row["center_x"]),
        float(row["center_y"]),
        normal,
        half_width=config.profile_half_width_px,
        samples=config.profile_samples,
    )
    lo, hi = np.percentile(profile, [5, 95])
    return np.clip((profile - lo) / max(float(hi - lo), 1e-6), 0, 1).astype(np.float32)


def _build_torch_architecture(mode: str):
    import torch.nn as nn

    if mode == "patch":
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1),
                )
            def forward(self, x):
                return self.net(x).squeeze(1)
        return Model()
    if mode == "profile":
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
                    nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                    nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(), nn.Linear(64, 1),
                )
            def forward(self, x):
                return self.net(x).squeeze(1)
        return Model()
    raise ValueError(f"Unknown torch mode: {mode}")


def train_torch_models(
    supervised: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not _torch_available() or (not config.use_patch_cnn and not config.use_profile_cnn):
        return pd.DataFrame(), {}

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    seed_everything(config.seed)
    split = stable_group_split(supervised, "stem", config.seed)

    # Balance explicit positives and hard negatives to stop the CNN learning only
    # the dominant class. Sampling remains image-level split safe.
    def cap_balanced(frame: pd.DataFrame) -> pd.DataFrame:
        pos = frame.loc[frame["target"] == 1]
        neg = frame.loc[frame["target"] == 0]
        n_each = min(max(len(pos), 1), max(len(neg), 1), config.cnn_max_samples // 2)
        blocks = []
        if len(pos):
            blocks.append(pos.sample(min(len(pos), n_each), random_state=config.seed))
        if len(neg):
            blocks.append(neg.sample(min(len(neg), n_each), random_state=config.seed + 1))
        return pd.concat(blocks).sample(frac=1.0, random_state=config.seed).reset_index(drop=True)

    train_df = cap_balanced(split.loc[split["split"] == "train"])
    val_df = split.loc[split["split"] == "val"].reset_index(drop=True)
    test_df = split.loc[split["split"] == "test"].reset_index(drop=True)

    image_cache: dict[str, np.ndarray] = {}
    def get_image(path: str) -> np.ndarray:
        if path not in image_cache:
            image_cache[path] = robust_normalize(load_gray(path))
        return image_cache[path]

    class CandidateDataset(Dataset):
        def __init__(self, frame: pd.DataFrame, mode: str):
            self.frame = frame.reset_index(drop=True)
            self.mode = mode

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, idx):
            row = self.frame.iloc[idx]
            image = get_image(str(row["raw_path"]))
            if self.mode == "patch":
                x = _extract_patch(image, float(row["center_x"]), float(row["center_y"]), config.patch_size, normalized=True)[None, ...]
            else:
                x = _candidate_profile_vector(image, row, config, normalized=True)[None, ...]
            y = np.float32(row["target"])
            return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

    class PatchCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(1)

    class ProfileCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
                nn.Flatten(), nn.Linear(64, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_model(name: str, mode: str, model: nn.Module):
        model = model.to(device)
        train_loader = DataLoader(CandidateDataset(train_df, mode), batch_size=config.cnn_batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(CandidateDataset(val_df, mode), batch_size=config.cnn_batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(CandidateDataset(test_df, mode), batch_size=config.cnn_batch_size, shuffle=False, num_workers=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        pos = max(int((train_df["target"] == 1).sum()), 1)
        neg = max(int((train_df["target"] == 0).sum()), 1)
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_state = None
        best_val = -np.inf
        patience = 3
        stale = 0
        for epoch in range(config.cnn_epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            val_prob = predict_loader(model, val_loader, device)
            metric = _best_threshold(val_df, val_prob, config)
            if metric["primary_score"] > best_val + 1e-4:
                best_val = metric["primary_score"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        val_prob = predict_loader(model, val_loader, device)
        best = _best_threshold(val_df, val_prob, config)
        test_prob = predict_loader(model, test_loader, device)
        metrics = _score_at_threshold(test_df, test_prob, best["threshold"], config)
        row = {
            "model": name,
            **{f"val_{k}": v for k, v in best.items()},
            **metrics,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
        }
        return row, {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "threshold": best["threshold"], "mode": mode}

    def predict_loader(model, loader, device):
        model.eval()
        probs = []
        with torch.no_grad():
            for xb, _ in loader:
                logits = model(xb.to(device))
                probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probs) if probs else np.empty(0, dtype=float)

    rows = []
    fitted = {}
    if config.use_patch_cnn:
        row, obj = run_model("patch_cnn", "patch", PatchCNN())
        rows.append(row)
        fitted["patch_cnn"] = obj
    if config.use_profile_cnn:
        row, obj = run_model("profile_cnn", "profile", ProfileCNN())
        rows.append(row)
        fitted["profile_cnn"] = obj
    return pd.DataFrame(rows), fitted


# ---------------------------------------------------------------------------
# Champion persistence and incremental retraining
# ---------------------------------------------------------------------------


def save_training_outputs(
    board: pd.DataFrame,
    sklearn_models: dict[str, Any],
    torch_models: dict[str, Any],
    features: list[str],
    config: PipelineConfig,
    manifest: pd.DataFrame,
    qa_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    board_path = config.output_dir / "leaderboard.csv"
    board.to_csv(board_path, index=False)

    complete_stems = sorted(manifest.loc[manifest["complete_truth"], "stem"].astype(str))
    champion = select_champion_name(board)
    champion_row = board.loc[board["model"] == champion].iloc[0].to_dict()

    # Save all sklearn contenders so a future run can compare/inspect them.
    model_dir = config.output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    for name, obj in sklearn_models.items():
        joblib.dump(obj, model_dir / f"{name}.joblib")

    if torch_models:
        import torch

        for name, obj in torch_models.items():
            torch.save(obj, model_dir / f"{name}.pt")

    previous_path = config.output_dir / "champion.json"
    previous = None
    if previous_path.exists():
        try:
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
        except Exception:
            previous = None

    previous_score = float(previous.get("primary_score", -np.inf)) if previous else -np.inf
    new_score = float(champion_row["primary_score"])
    improved = new_score > previous_score + 1e-6

    record = {
        "model": champion,
        "primary_score": new_score,
        "metrics": champion_row,
        "trained_at": timestamp,
        "complete_image_count": len(complete_stems),
        "complete_stems": complete_stems,
        "feature_columns": features,
        "previous_score": previous_score if np.isfinite(previous_score) else None,
        "improved": bool(improved),
    }

    # Do not overwrite the historical champion on a worse run. Save the latest
    # challenger separately so the user can diagnose regressions.
    (config.output_dir / "latest_challenger.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    if improved or previous is None:
        previous_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        if champion in sklearn_models:
            pt_path = config.output_dir / "champion_model.pt"
            if pt_path.exists():
                pt_path.unlink()
            joblib.dump(sklearn_models[champion], config.output_dir / "champion_model.joblib")
        elif champion in torch_models:
            joblib_path = config.output_dir / "champion_model.joblib"
            if joblib_path.exists():
                joblib_path.unlink()
            import torch
            torch.save(torch_models[champion], config.output_dir / "champion_model.pt")

    qa_path = config.output_dir / "dataset_qa.json"
    qa_path.write_text(json.dumps(qa_rows, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return record


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------


def train_all(config: PipelineConfig, *, force_rebuild: bool = False) -> dict[str, Any]:
    seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = scan_dataset(config.root)
    complete_count = int(manifest["complete_truth"].sum()) if not manifest.empty else 0
    print(f"Dataset root: {config.root}")
    print(f"Complete answer-sheet sets: {complete_count}")
    if complete_count < config.min_complete_images_for_training:
        raise RuntimeError(
            f"Need at least {config.min_complete_images_for_training} complete image sets for image-level train/val/test; found {complete_count}."
        )

    visionflux_path = find_visionflux_file(config.root, config)
    print(f"VisionFlux module: {visionflux_path}")
    vf = load_visionflux_module(visionflux_path)

    dataset, qa_rows = build_training_dataset(manifest, vf, config, force_rebuild=force_rebuild)
    if dataset.empty:
        raise RuntimeError("No training candidates were produced")

    # Persist a human-readable candidate table. Pickle cache keeps exact values;
    # CSV is for inspection in Drive.
    dataset.to_csv(config.output_dir / "all_candidates.csv", index=False)
    generator_board = evaluate_generators(dataset, config)
    generator_board.to_csv(config.output_dir / "generator_leaderboard.csv", index=False)
    print("\nGenerator coverage (proposal stage):")
    if not generator_board.empty:
        print(generator_board.to_string(index=False))

    supervised = dataset.loc[dataset["target"].notna()].copy()
    supervised["target"] = supervised["target"].astype(int)
    supervised.to_csv(config.output_dir / "supervised_candidates.csv", index=False)
    print("Supervision counts:")
    print(supervised["supervision"].value_counts())

    board_sklearn, fitted_sklearn, features = train_sklearn_competition(supervised, config)
    board_torch, fitted_torch = train_torch_models(supervised, config)
    board = pd.concat([board_sklearn, board_torch], ignore_index=True, sort=False)
    board = board.sort_values("primary_score", ascending=False).reset_index(drop=True)

    record = save_training_outputs(
        board,
        fitted_sklearn,
        fitted_torch,
        features,
        config,
        manifest,
        qa_rows,
    )
    print("\nLeaderboard:")
    columns = [
        c
        for c in [
            "model",
            "primary_score",
            "precision",
            "recall",
            "f1",
            "manual_add_recall",
            "hard_negative_rejection",
            "average_precision",
            "threshold",
        ]
        if c in board.columns
    ]
    print(board[columns].to_string(index=False))
    print(f"\nCurrent run winner: {record['model']}  score={record['primary_score']:.4f}")
    print(f"Champion updated: {record['improved']}")
    return {"leaderboard": board, "generator_leaderboard": generator_board, "record": record, "dataset": dataset, "manifest": manifest}


# ---------------------------------------------------------------------------
# Inference candidate pool + overlay for new raw SEMs
# ---------------------------------------------------------------------------


def build_inference_candidates(
    raw_path: str | Path,
    module: Any,
    config: PipelineConfig,
) -> pd.DataFrame:
    image = load_gray(raw_path)
    vf, meta = run_visionflux_passes(raw_path, module, config)
    rel = generate_relative_ridge_candidates(image, meta["orientation"], config)
    rec = generate_profile_recovery_candidates(image, meta["orientation"], vf, config)
    blocks = [x for x in (vf, rel, rec) if x is not None and not x.empty]
    if not blocks:
        return pd.DataFrame()
    cand = pd.concat(blocks, ignore_index=True, sort=False)
    cand = add_local_features(cand, image, config)

    # Attach the same map features used during training.
    ridge = np.asarray(meta.get("ridge_response"))
    pore = np.asarray(meta.get("pore_core"))
    orientation = meta["orientation"]
    coh = np.asarray(orientation.coherency)
    energy = robust_normalize(np.asarray(orientation.energy))
    rows = []
    for r in cand.itertuples(index=False):
        x = int(np.clip(round(float(getattr(r, "center_x"))), 0, image.shape[1] - 1))
        y = int(np.clip(round(float(getattr(r, "center_y"))), 0, image.shape[0] - 1))
        rows.append(
            {
                "map_ridge": float(ridge[y, x]) if ridge.shape == image.shape else np.nan,
                "map_pore": float(pore[y, x]) if pore.shape == image.shape else np.nan,
                "map_coherency": float(coh[y, x]) if coh.shape == image.shape else np.nan,
                "map_energy": float(energy[y, x]) if energy.shape == image.shape else np.nan,
            }
        )
    maps = pd.DataFrame(rows)
    for c in maps.columns:
        cand[c] = maps[c]
    cand["candidate_score"] = pd.to_numeric(cand.get("candidate_score", cand.get("confidence", 0.0)), errors="coerce").fillna(0.0)
    return _deduplicate_candidates(cand, radius_px=4.0, angle_tol=15.0)


def _predict_with_saved_sklearn(candidates: pd.DataFrame, obj: dict[str, Any]) -> np.ndarray:
    if obj.get("baseline"):
        return _visionflux_baseline_probability(candidates)
    model = obj["model"]
    features = obj["features"]
    work = candidates.copy()
    for c in features:
        if c not in work.columns:
            work[c] = np.nan
    return _predict_proba_binary(model, work[features])


def _predict_with_saved_torch(
    candidates: pd.DataFrame,
    raw_path: str | Path,
    obj: dict[str, Any],
    config: PipelineConfig,
) -> np.ndarray:
    import torch

    mode = str(obj["mode"])
    model = _build_torch_architecture(mode)
    model.load_state_dict(obj["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    image = robust_normalize(load_gray(raw_path))
    arrays = []
    for _, row in candidates.iterrows():
        if mode == "patch":
            x = _extract_patch(
                image, float(row["center_x"]), float(row["center_y"]), config.patch_size, normalized=True
            )[None, ...]
        else:
            x = _candidate_profile_vector(image, row, config, normalized=True)[None, ...]
        arrays.append(x)
    if not arrays:
        return np.empty(0, dtype=float)
    xall = np.stack(arrays).astype(np.float32)
    probs = []
    with torch.no_grad():
        for i in range(0, len(xall), config.cnn_batch_size):
            xb = torch.from_numpy(xall[i : i + config.cnn_batch_size]).to(device)
            probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(probs).astype(float)


def draw_prediction_overlay(raw_path: str | Path, predictions: pd.DataFrame, out_path: str | Path) -> None:
    base = Image.open(raw_path).convert("RGB")
    draw = ImageDraw.Draw(base)
    for r in predictions.itertuples(index=False):
        p = float(getattr(r, "model_probability", 0.0))
        # Kept predictions cyan; uncertain-but-visible orange is not emitted here
        # because the pipeline intentionally passes on low-confidence thickness.
        color = (26, 220, 235)
        width = 2
        draw.line(
            (
                float(getattr(r, "x1")),
                float(getattr(r, "y1")),
                float(getattr(r, "x2")),
                float(getattr(r, "y2")),
            ),
            fill=color,
            width=width,
        )
        rr = 2
        for x, y in ((getattr(r, "x1"), getattr(r, "y1")), (getattr(r, "x2"), getattr(r, "y2"))):
            draw.ellipse((x - rr, y - rr, x + rr, y + rr), fill=color)
    base.save(out_path)


def predict_unlabeled(config: PipelineConfig, *, only_stems: Sequence[str] | None = None) -> pd.DataFrame:
    manifest = scan_dataset(config.root)
    vf_path = find_visionflux_file(config.root, config)
    vf = load_visionflux_module(vf_path)
    champion_path = config.output_dir / "champion.json"
    if not champion_path.exists():
        raise FileNotFoundError("No champion.json. Run training first.")
    champion = json.loads(champion_path.read_text(encoding="utf-8"))
    model_name = champion["model"]

    sklearn_obj = None
    torch_obj = None
    joblib_path = config.output_dir / "champion_model.joblib"
    torch_path = config.output_dir / "champion_model.pt"
    if joblib_path.exists():
        sklearn_obj = joblib.load(joblib_path)
    elif torch_path.exists():
        import torch
        try:
            torch_obj = torch.load(torch_path, map_location="cpu", weights_only=False)
        except TypeError:
            torch_obj = torch.load(torch_path, map_location="cpu")
    else:
        raise FileNotFoundError("Champion model file is missing; retrain once.")

    rows = []
    work = manifest.loc[manifest["raw_path"].notna()].copy()
    if only_stems:
        work = work.loc[work["stem"].astype(str).isin(set(map(str, only_stems)))]
    # Predict raw-only files by default. Complete truth files can still be forced
    # by passing their stems for diagnostics.
    if not only_stems:
        work = work.loc[~work["complete_truth"]]

    pred_dir = config.output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for r in work.itertuples(index=False):
        print(f"Predict: {r.stem}")
        cand = build_inference_candidates(r.raw_path, vf, config)
        if cand.empty:
            continue
        if sklearn_obj is not None:
            p = _predict_with_saved_sklearn(cand, sklearn_obj)
            threshold = float(sklearn_obj["threshold"])
        else:
            p = _predict_with_saved_torch(cand, r.raw_path, torch_obj, config)
            threshold = float(torch_obj["threshold"])
        cand["model_probability"] = p
        accepted = cand.loc[p >= threshold].copy()
        # Conservative NMS: duplicate experts should not create duplicate widths.
        accepted["candidate_score"] = accepted["model_probability"]
        accepted = _deduplicate_candidates(accepted, radius_px=6.0, angle_tol=18.0)
        accepted["stem"] = r.stem
        accepted["champion_model"] = model_name
        accepted["threshold"] = threshold
        accepted.to_csv(pred_dir / f"{r.stem}_predictions.csv", index=False)
        draw_prediction_overlay(r.raw_path, accepted, pred_dir / f"{r.stem}_prediction.png")
        rows.append(accepted)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Colab helper / CLI
# ---------------------------------------------------------------------------


def mount_google_drive() -> None:
    try:
        from google.colab import drive

        drive.mount("/content/drive")
    except ImportError:
        print("Not running inside Google Colab; skipping drive.mount().")


def print_dataset_status(root: str | Path = DEFAULT_ROOT) -> pd.DataFrame:
    frame = scan_dataset(root)
    if frame.empty:
        print("No dataset files found.")
        return frame
    display_cols = ["stem", "complete_truth", "raw_path", "overlay_path", "csv_path"]
    print(frame[display_cols].to_string(index=False))
    return frame


def make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CHEM FRONTIER continual SEM fiber-thickness learner")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--mode", choices=["train", "predict", "status"], default="train")
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--no-cnn", action="store_true")
    p.add_argument("--stems", nargs="*", default=None, help="Specific stems to predict, e.g. 2-21 2-22")
    p.add_argument("--mount-drive", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = make_arg_parser().parse_args(argv)
    if args.mount_drive:
        mount_google_drive()
    cfg = PipelineConfig(root=args.root)
    if args.no_cnn:
        cfg.use_patch_cnn = False
        cfg.use_profile_cnn = False
    if args.mode == "status":
        print_dataset_status(cfg.root)
    elif args.mode == "train":
        train_all(cfg, force_rebuild=args.force_rebuild)
    elif args.mode == "predict":
        predict_unlabeled(cfg, only_stems=args.stems)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
