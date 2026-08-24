"""Register an annotated overlay onto its clean SEM original.

Annotated exports are frequently cropped (footer removed), rescaled by a viewer,
or re-saved at a different size.  Coordinates recovered from the overlay are
therefore in *overlay* pixels and must be mapped into *original* pixels before
they can be used as labels.

Strategy, cheapest first:

1. **Identity** -- same shape, high correlation once the footer is stripped.
2. **Similarity via ECC** -- estimates translation+rotation+uniform scale on the
   masked grayscale.  Uniform scale only: an anisotropic fit would silently
   change apparent fiber thickness.
3. **Feature matching (ORB + RANSAC, similarity constrained)** -- fallback for
   large offsets where ECC's basin of attraction is too small.

Every result carries a quality score so the caller can refuse to use a bad fit
rather than train on mis-registered labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .utils import get_logger

LOG = get_logger(__name__)


@dataclass
class Registration:
    """2x3 affine mapping overlay pixels -> original pixels."""
    matrix: np.ndarray
    method: str
    ncc: float
    scale: float
    rotation_deg: float
    translation: tuple[float, float]
    ok: bool
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def apply(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map point arrays from overlay space to original space."""
        pts = np.stack([np.asarray(xs, float), np.asarray(ys, float),
                        np.ones_like(np.asarray(xs, float))], axis=0)
        out = self.matrix @ pts
        return out[0], out[1]

    def to_dict(self) -> dict[str, Any]:
        return {"matrix": self.matrix.tolist(), "method": self.method,
                "ncc": self.ncc, "scale": self.scale,
                "rotation_deg": self.rotation_deg,
                "translation": list(self.translation), "ok": self.ok,
                "detail": self.detail, **self.extra}


def _ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Normalised cross-correlation of two same-shaped images."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if mask is not None:
        a, b = a[mask], b[mask]
    else:
        a, b = a.ravel(), b.ravel()
    if a.size < 16:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def _decompose(M: np.ndarray) -> tuple[float, float, tuple[float, float]]:
    sx = float(np.hypot(M[0, 0], M[1, 0]))
    rot = float(np.rad2deg(np.arctan2(M[1, 0], M[0, 0])))
    return sx, rot, (float(M[0, 2]), float(M[1, 2]))


def register(overlay_gray: np.ndarray, original_gray: np.ndarray, *,
             overlay_valid: np.ndarray | None = None,
             ncc_accept: float = 0.55,
             allow_feature_fallback: bool = True) -> Registration:
    """Estimate the overlay -> original mapping.

    ``overlay_valid`` marks pixels of the overlay that are real image content
    (i.e. not painted over); those pixels alone drive the fit.
    """
    import cv2

    ov = overlay_gray.astype(np.float32)
    orig = original_gray.astype(np.float32)
    valid = (np.ones(ov.shape, bool) if overlay_valid is None else overlay_valid)

    # ---- 1. identity ------------------------------------------------------
    if ov.shape == orig.shape:
        score = _ncc(ov, orig, valid)
        if score >= ncc_accept:
            M = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
            LOG.info("registration: identity accepted (ncc=%.3f)", score)
            return Registration(M, "identity", score, 1.0, 0.0, (0.0, 0.0), True,
                                "same shape, high correlation")

    # ---- 2. ECC similarity ------------------------------------------------
    try:
        sy = orig.shape[0] / ov.shape[0]
        sx = orig.shape[1] / ov.shape[1]
        s0 = float(np.sqrt(sx * sy))
        warp = np.array([[s0, 0, 0], [0, s0, 0]], np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-6)
        ov_n = cv2.normalize(ov, None, 0, 1, cv2.NORM_MINMAX)
        orig_n = cv2.normalize(orig, None, 0, 1, cv2.NORM_MINMAX)
        _cc, warp = cv2.findTransformECC(
            orig_n, ov_n, warp, cv2.MOTION_EUCLIDEAN, crit,
            valid.astype(np.uint8) * 255, 5)
        M = np.vstack([warp.astype(np.float64)])
        warped = cv2.warpAffine(ov, M, (orig.shape[1], orig.shape[0]))
        wmask = cv2.warpAffine(valid.astype(np.uint8), M,
                               (orig.shape[1], orig.shape[0])) > 0
        score = _ncc(warped, orig, wmask)
        s, rot, t = _decompose(M)
        if score >= ncc_accept:
            LOG.info("registration: ECC accepted (ncc=%.3f, scale=%.4f, rot=%.2f deg)",
                     score, s, rot)
            return Registration(M, "ecc_euclidean", score, s, rot, t, True,
                                "ECC on masked grayscale")
        LOG.warning("ECC fit weak (ncc=%.3f)", score)
    except cv2.error as exc:
        LOG.warning("ECC registration failed: %s", exc)
        score, M = 0.0, None

    # ---- 3. ORB + RANSAC --------------------------------------------------
    if allow_feature_fallback:
        try:
            orb = cv2.ORB_create(nfeatures=6000)
            k1, d1 = orb.detectAndCompute(cv2.normalize(ov, None, 0, 255,
                                                        cv2.NORM_MINMAX).astype(np.uint8),
                                          valid.astype(np.uint8))
            k2, d2 = orb.detectAndCompute(cv2.normalize(orig, None, 0, 255,
                                                        cv2.NORM_MINMAX).astype(np.uint8),
                                          None)
            if d1 is not None and d2 is not None and len(k1) >= 8 and len(k2) >= 8:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = sorted(bf.match(d1, d2), key=lambda m: m.distance)[:800]
                if len(matches) >= 8:
                    p1 = np.float32([k1[m.queryIdx].pt for m in matches])
                    p2 = np.float32([k2[m.trainIdx].pt for m in matches])
                    M2, inliers = cv2.estimateAffinePartial2D(
                        p1, p2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                    if M2 is not None:
                        warped = cv2.warpAffine(ov, M2, (orig.shape[1], orig.shape[0]))
                        wmask = cv2.warpAffine(valid.astype(np.uint8), M2,
                                               (orig.shape[1], orig.shape[0])) > 0
                        s2 = _ncc(warped, orig, wmask)
                        s, rot, t = _decompose(M2)
                        ok = s2 >= ncc_accept
                        LOG.info("registration: ORB ncc=%.3f inliers=%d",
                                 s2, int(inliers.sum()) if inliers is not None else 0)
                        return Registration(M2.astype(np.float64), "orb_ransac", s2,
                                            s, rot, t, ok,
                                            f"{int(inliers.sum())} inliers")
        except cv2.error as exc:
            LOG.warning("ORB registration failed: %s", exc)

    LOG.error("registration FAILED -- refusing to guess a transform")
    return Registration(np.array([[1, 0, 0], [0, 1, 0]], float), "failed", 0.0,
                        1.0, 0.0, (0.0, 0.0), False,
                        "no method reached the acceptance threshold")


def perceptual_hash(gray: np.ndarray, size: int = 16) -> np.ndarray:
    """64-bit-style aHash used to flag near-duplicate / overlapping fields."""
    import cv2

    small = cv2.resize(gray.astype(np.float32), (size, size),
                       interpolation=cv2.INTER_AREA)
    dct = cv2.dct(small)[:8, :8]
    flat = dct.flatten()[1:]
    return (flat > np.median(flat)).astype(np.uint8)


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))
