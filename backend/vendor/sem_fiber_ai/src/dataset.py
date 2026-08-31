"""Image records and the tile dataset (v7).

* ``ImageRecord`` carries the field's labels (raster convention), its audited
  calibration and, when physical resampling is on, the factor that was applied
  -- so predictions can be mapped back to original pixels exactly.
* ``TileDataset`` draws training tiles around annotations with a WIDTH-STRATIFIED
  sampler: the probability of centring a tile on a given annotation is inversely
  proportional to the frequency of its width band (computed on the training
  labels), so thick fibres are seen as often as the common thin ones.
* Splits are NOT decided here; see :mod:`specimens`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .augmentations import AugConfig, GeometricAug, PhotometricAug, normalize
from .fiber_prior import FiberPrior, PriorConfig
from .targets import TargetConfig, encode_targets, strata_weight_map
from .utils import get_logger, read_gray

LOG = get_logger(__name__)


@dataclass
class ImageRecord:
    image_id: str
    path: Path
    annotations: Any
    nm_per_pixel: float | None = None
    calibration_valid: bool = False
    calibration_status: str = "unaudited"
    specimen: str = ""
    resample_factor: float = 1.0          # image pixels = original pixels * factor
    prior_cfg: PriorConfig | None = None
    prior_cache_dir: Path | None = None
    _image: np.ndarray | None = field(default=None, repr=False)
    _prior: Any = field(default=None, repr=False)
    _prior_image_only: Any = field(default=None, repr=False)

    def image(self) -> np.ndarray:
        if self._image is None:
            from .calibration import strip_footer
            from .physical import resample_image

            g, _row = strip_footer(read_gray(self.path))
            if abs(self.resample_factor - 1.0) > 1e-9:
                g = resample_image(g, self.resample_factor)
            self._image = g.astype(np.float32)
        return self._image

    def valid(self) -> np.ndarray:
        return np.ones(self.image().shape, bool)

    def prior(self, *, image_only: bool = False) -> FiberPrior:
        """Fibre prior.  ``image_only=True`` never consults annotations (used
        for evaluation, so the answer key cannot shape the mask it is scored on)."""
        cfg = self.prior_cfg or PriorConfig()
        if image_only:
            if self._prior_image_only is None:
                from .fiber_prior import best_polarity

                prior, _pol, _auc = best_polarity(self.image(), cfg, ann=None)
                if prior is None:
                    raise RuntimeError(f"{self.image_id}: image-only prior failed")
                self._prior_image_only = prior
            return self._prior_image_only
        if self._prior is None:
            cache = None
            if self.prior_cache_dir is not None:
                cache = Path(self.prior_cache_dir) / f"{self.image_id}_f{self.resample_factor:.4f}_prior"
            self._prior = FiberPrior.load_or_compute(self.image(), cache, cfg,
                                                     ann=self.annotations)
        return self._prior


def load_records(labels_csv: str | Path, image_dir: str | Path, *,
                 prior_cfg: PriorConfig | None = None,
                 prior_cache_dir: str | Path | None = None,
                 resample_factors: dict[str, float] | None = None,
                 specimen_of: dict[str, str] | None = None,
                 image_ids: Sequence[str] | None = None) -> list[ImageRecord]:
    """Join the label table with the images on disk (raster labels required)."""
    import pandas as pd

    from .labels import validate_labels
    from .physical import resample_labels
    from .utils import duplicate_image_ids, image_id_from_path, list_images

    df = pd.read_csv(labels_csv)
    if "measurement_angle_raster_deg" not in df.columns:
        raise ValueError("labels table is not in the v7 raster schema; run the v7 "
                         "extraction (or labels.upgrade_legacy_labels) first")
    rep = validate_labels(df)
    if not rep.get("ok", True):
        LOG.warning("label table consistency problems: %s", rep["problems"])
    paths = list_images(image_dir)
    clashes = duplicate_image_ids(paths)
    if clashes:
        raise ValueError(f"image ids match more than one file in {image_dir}: "
                         f"{sorted(clashes)[:10]}")
    by_id = {image_id_from_path(p): p for p in paths}
    if "is_negative" not in df.columns:
        df["is_negative"] = False
    df["is_negative"] = df["is_negative"].fillna(False).astype(bool)
    wanted = set(map(str, image_ids)) if image_ids is not None else None
    records: list[ImageRecord] = []
    for iid, sub in df.groupby("image_id"):
        iid = str(iid)
        if wanted is not None and iid not in wanted:
            continue
        path = by_id.get(iid)
        if path is None:
            LOG.error("no image on disk for image_id=%s (looked in %s)", iid, image_dir)
            continue
        sub = sub[~sub["is_negative"]].reset_index(drop=True)
        valid = bool(sub["calibration_valid"].fillna(False).astype(bool).all()) \
            if "calibration_valid" in sub.columns else False
        nmpp = pd.to_numeric(sub.get("nm_per_pixel"), errors="coerce").dropna() \
            if "nm_per_pixel" in sub.columns else pd.Series([], dtype=float)
        nm = float(nmpp.iloc[0]) if (valid and len(nmpp)) else None
        status = str(sub["calibration_status"].iloc[0]) if "calibration_status" in sub.columns \
            else "unaudited"
        f = float((resample_factors or {}).get(iid, 1.0))
        if abs(f - 1.0) > 1e-9:
            sub = resample_labels(sub, f, (nm / f) if nm else None)
            nm_eff = nm / f if nm else None
        else:
            nm_eff = nm
        records.append(ImageRecord(iid, path, sub, nm_eff, valid, status,
                                   (specimen_of or {}).get(iid, iid), f,
                                   prior_cfg=prior_cfg,
                                   prior_cache_dir=Path(prior_cache_dir) if prior_cache_dir else None))
    LOG.info("loaded %d image record(s), %d annotations",
             len(records), int(sum(len(r.annotations) for r in records)))
    return records


# --------------------------------------------------------------------------- #
class TileDataset:
    """Random tiles from full images with dense targets and stratified sampling."""

    def __init__(self, records: Sequence[ImageRecord], tile: int = 384, *,
                 samples_per_image: int = 40, aug: AugConfig | None = None,
                 targets: TargetConfig | None = None, train: bool = True,
                 norm: str = "per_image", seed: int = 0, positive_bias: float = 0.85,
                 prior_cfg: PriorConfig | None = None, use_prior: bool = True) -> None:
        self.records = list(records)
        if not self.records:
            raise ValueError("TileDataset received no image records")
        self.tile = int(tile)
        self.samples_per_image = int(samples_per_image)
        self.aug_cfg = aug or AugConfig(enabled=train)
        self.tcfg = targets or TargetConfig()
        self.train = train
        self.norm = norm
        self.seed = int(seed)
        self.positive_bias = float(positive_bias)
        self.prior_cfg = prior_cfg or PriorConfig()
        self.use_prior = use_prior
        # stratified sampling weights per annotation (from THIS split's labels)
        self._ann_weights = []
        for rec in self.records:
            w = rec.annotations["width_px"].to_numpy(np.float64) if len(rec.annotations) \
                else np.zeros(0)
            sw = strata_weight_map(w, self.tcfg) if w.size else np.zeros(0, np.float32)
            self._ann_weights.append(sw / sw.sum() if sw.sum() > 0 else None)

    def __len__(self) -> int:
        return len(self.records) * self.samples_per_image

    def _crop_box(self, k: int, rec: ImageRecord, rng: np.random.Generator) -> tuple[int, int]:
        h, w = rec.image().shape
        t = self.tile
        if h <= t or w <= t:
            return 0, 0
        pw = self._ann_weights[k]
        if pw is not None and self.train and rng.random() < self.positive_bias:
            i = int(rng.choice(len(pw), p=pw))
            r = rec.annotations.iloc[i]
            cx = int(np.clip(r["center_x_px"] + rng.integers(-t // 3, t // 3), t // 2, w - t // 2))
            cy = int(np.clip(r["center_y_px"] + rng.integers(-t // 3, t // 3), t // 2, h - t // 2))
            return cx - t // 2, cy - t // 2
        if not self.train:
            # deterministic grid coverage for validation
            n = self.samples_per_image
            j = rng.integers(0, n)
            gx, gy = max(1, int(np.ceil(np.sqrt(n)))), max(1, int(np.ceil(np.sqrt(n))))
            return (int((j % gx) * (w - t) / max(1, gx - 1)),
                    int((j // gx) * (h - t) / max(1, gy - 1)))
        return int(rng.integers(0, w - t)), int(rng.integers(0, h - t))

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        k = index % len(self.records)
        rec = self.records[k]
        rng = np.random.default_rng(self.seed * 1_000_003 + index)
        img = rec.image()
        x0, y0 = self._crop_box(k, rec, rng)
        t = self.tile
        sub = img[y0:y0 + t, x0:x0 + t]
        if sub.shape != (t, t):
            sub = np.pad(sub, ((0, t - sub.shape[0]), (0, t - sub.shape[1])), mode="reflect")
        sub_valid = np.ones(sub.shape, bool)
        ann = rec.annotations
        ann = ann[(ann.center_x_px.between(x0 - 8, x0 + t + 8))
                  & (ann.center_y_px.between(y0 - 8, y0 + t + 8))].copy()
        for c in ("center_x_px", "x1_px", "x2_px"):
            if c in ann.columns:
                ann[c] = ann[c] - x0
        for c in ("center_y_px", "y1_px", "y2_px"):
            if c in ann.columns:
                ann[c] = ann[c] - y0

        aug = None
        if self.train:
            aug = GeometricAug(self.aug_cfg, rng)
            sub, ann, sub_valid = aug(sub, ann, sub_valid)
            sub = PhotometricAug(self.aug_cfg, rng)(sub)

        prior = None
        if self.use_prior:
            full = rec.prior()
            base = full.crop(y0, x0, t, t)
            if base.mask.shape != (t, t):
                base = base.pad_to(t, t)
            if self.train and getattr(aug, "last_M", None) is not None:
                prior = base.warp_like(aug.last_M, sub.shape[:2])
            else:
                prior = base

        tgt = encode_targets(sub.shape, ann, self.tcfg, valid_mask=sub_valid, prior=prior)
        x = normalize(sub, self.norm)[None]
        out = {"image": torch.from_numpy(np.ascontiguousarray(x)).float(),
               "image_id": rec.image_id, "n_ann": int(len(ann))}
        for kk, v in tgt.items():
            out[kk] = torch.from_numpy(v[None]).float()
        return out
