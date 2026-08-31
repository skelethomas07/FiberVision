"""Geometry-aware augmentation for SEM fiber measurement data.

Two constraints drive every choice here:

* **Thickness must be preserved or transformed exactly.**  Anisotropic resizing
  changes apparent fiber diameter as a function of orientation, so it is
  forbidden.  Isotropic scaling is allowed only if the width labels are scaled
  by the same factor, which this module does explicitly.
* **Angles are not invariant.**  A flip negates an angle, a rotation adds to it.
  Every geometric op therefore updates centres, endpoints, angles and widths
  together; the transform and the label update live in the same function so
  they cannot drift apart.

Photometric ops stay mild: SEM contrast carries the signal we are measuring, and
aggressive jitter manufactures structures that do not exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .utils import get_logger, wrap_deg_180

LOG = get_logger(__name__)


@dataclass
class AugConfig:
    hflip: float = 0.5
    vflip: float = 0.5
    rot90: float = 0.5
    rotate_deg: float = 20.0
    scale_range: tuple[float, float] = (0.9, 1.1)
    translate_frac: float = 0.05
    brightness: float = 0.12
    contrast: float = 0.12
    gamma: tuple[float, float] = (0.85, 1.18)
    noise_std: float = 4.0
    blur_prob: float = 0.15
    blur_sigma: tuple[float, float] = (0.4, 1.0)
    enabled: bool = True


class GeometricAug:
    """Apply a random similarity transform to an image and its annotations."""

    def __init__(self, cfg: AugConfig, rng: np.random.Generator | None = None) -> None:
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()

    # ---------------------------------------------------------------- #
    def __call__(self, image: np.ndarray, ann: "Any", valid: np.ndarray | None = None
                 ) -> tuple[np.ndarray, "Any", np.ndarray | None]:
        import cv2

        cfg = self.cfg
        if not cfg.enabled:
            self.last_M = np.eye(3)
            self.last_shape = image.shape[:2]
            self.last_scale = 1.0
            return image, ann, valid
        ann = ann.copy()
        h, w = image.shape[:2]

        flip_x = self.rng.random() < cfg.hflip
        flip_y = self.rng.random() < cfg.vflip
        k90 = int(self.rng.integers(0, 4)) if self.rng.random() < cfg.rot90 else 0
        angle = float(self.rng.uniform(-cfg.rotate_deg, cfg.rotate_deg))
        scale = float(self.rng.uniform(*cfg.scale_range))
        tx = float(self.rng.uniform(-cfg.translate_frac, cfg.translate_frac) * w)
        ty = float(self.rng.uniform(-cfg.translate_frac, cfg.translate_frac) * h)

        # build one 2x3 matrix for the whole similarity transform
        M = np.eye(3, dtype=np.float64)
        if flip_x:
            F = np.array([[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]], float)
            M = F @ M
        if flip_y:
            F = np.array([[1, 0, 0], [0, -1, h - 1], [0, 0, 1]], float)
            M = F @ M
        if k90:
            for _ in range(k90):
                R = np.array([[0, -1, h - 1], [1, 0, 0], [0, 0, 1]], float)
                M = R @ M
                h, w = w, h
        R = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        R = np.vstack([R, [0, 0, 1]])
        R[0, 2] += tx
        R[1, 2] += ty
        M = R @ M

        # [v4] record the transform so the cached fiber prior can be warped
        # with exactly this matrix instead of being recomputed per crop.
        self.last_M = M.copy()
        self.last_shape = (h, w)
        self.last_scale = scale

        out = cv2.warpAffine(image, M[:2], (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        out_valid = None
        if valid is not None:
            out_valid = cv2.warpAffine(valid.astype(np.uint8), M[:2], (w, h),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0).astype(bool)

        if len(ann):
            for xcol, ycol in (("center_x_px", "center_y_px"),
                               ("x1_px", "y1_px"), ("x2_px", "y2_px")):
                if xcol in ann.columns:
                    pts = np.stack([ann[xcol].to_numpy(float),
                                    ann[ycol].to_numpy(float),
                                    np.ones(len(ann))], axis=0)
                    new = M @ pts
                    ann[xcol], ann[ycol] = new[0], new[1]

            # angle update: recover it from the transformed endpoints so the
            # matrix is the single source of truth (no duplicated trig)
            if {"x1_px", "y1_px", "x2_px", "y2_px"} <= set(ann.columns):
                dx = ann["x2_px"].to_numpy(float) - ann["x1_px"].to_numpy(float)
                dy = ann["y2_px"].to_numpy(float) - ann["y1_px"].to_numpy(float)
                ann["measurement_angle_deg"] = np.rad2deg(np.arctan2(dy, dx))
                ann["width_px"] = np.hypot(dx, dy)
            else:
                ann["width_px"] = ann["width_px"].to_numpy(float) * scale

            if "local_fiber_angle_deg" in ann.columns:
                # a similarity transform rotates directions by its own rotation
                rot = np.rad2deg(np.arctan2(M[1, 0], M[0, 0]))
                ann["local_fiber_angle_deg"] = wrap_deg_180(
                    ann["local_fiber_angle_deg"].to_numpy(float) + rot)

            if "nm_per_pixel" in ann.columns:
                # pixels changed size, so nm/px must follow or widths in nm break
                ann["nm_per_pixel"] = ann["nm_per_pixel"].to_numpy(float) / scale

            inside = (ann["center_x_px"].between(0, w - 1)
                      & ann["center_y_px"].between(0, h - 1))
            ann = ann.loc[inside].reset_index(drop=True)

        return out, ann, out_valid


class PhotometricAug:
    """Mild intensity jitter that keeps SEM statistics plausible."""

    def __init__(self, cfg: AugConfig, rng: np.random.Generator | None = None) -> None:
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        import cv2

        cfg = self.cfg
        if not cfg.enabled:
            return image
        img = image.astype(np.float32)
        mean = float(img.mean())
        img = (img - mean) * (1.0 + self.rng.uniform(-cfg.contrast, cfg.contrast)) + mean
        img = img + self.rng.uniform(-cfg.brightness, cfg.brightness) * 255.0
        img = np.clip(img, 0, 255)
        g = float(self.rng.uniform(*cfg.gamma))
        img = 255.0 * np.power(img / 255.0, g)
        if cfg.noise_std > 0:
            img = img + self.rng.normal(0, cfg.noise_std, img.shape).astype(np.float32)
        if self.rng.random() < cfg.blur_prob:
            s = float(self.rng.uniform(*cfg.blur_sigma))
            img = cv2.GaussianBlur(img, (0, 0), s)
        return np.clip(img, 0, 255).astype(np.float32)


def normalize(image: np.ndarray, mode: str = "per_image") -> np.ndarray:
    """Intensity normalisation; per-image z-score by default.

    Per-image normalisation is the safe default for SEM: absolute grey level
    depends on detector gain and has no physical meaning, whereas local contrast
    does.
    """
    img = image.astype(np.float32)
    if mode == "per_image":
        m, s = float(img.mean()), float(img.std())
        return (img - m) / (s + 1e-6)
    if mode == "minmax":
        lo, hi = float(img.min()), float(img.max())
        return (img - lo) / (hi - lo + 1e-6)
    if mode == "fixed":
        return img / 255.0
    raise ValueError(f"unknown normalisation mode: {mode}")
