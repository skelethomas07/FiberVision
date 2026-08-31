"""Datasets and leakage-safe splitting.

The single most important rule in this project lives here: **annotations from
one SEM image never appear in two different splits**.  Fiber networks are
locally self-similar, so two patches cut from the same field are far more alike
than two patches from different specimens; a random patch split would report
excellent metrics that say nothing about a new image.

Splitting is therefore always by ``image_id``, and images that are perceptually
near-duplicates (adjacent crops of the same field, re-saves of the same file)
are merged into one group before splitting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .augmentations import AugConfig, GeometricAug, PhotometricAug, normalize
from .fiber_prior import FiberPrior, PriorConfig, audit_prior
from .targets import TargetConfig, encode_targets, sample_negative_centers
from .utils import get_logger, read_gray

LOG = get_logger(__name__)


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def group_near_duplicates(images: dict[str, np.ndarray], *, hamming_max: int = 6
                          ) -> dict[str, str]:
    """Map ``image_id -> group_id``, merging perceptually near-identical fields."""
    from .image_registration import hamming, perceptual_hash

    ids = sorted(images)
    hashes = {i: perceptual_hash(images[i]) for i in ids}
    parent = {i: i for i in ids}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            d = hamming(hashes[a], hashes[b])
            if d <= hamming_max:
                LOG.warning("images %s and %s look like the same field "
                            "(hamming=%d); they will share a split", a, b, d)
                parent[find(b)] = find(a)
    return {i: find(i) for i in ids}


def grouped_split(image_ids: Sequence[str], *, val_frac: float = 0.2,
                  test_frac: float = 0.2, seed: int = 1337,
                  groups: dict[str, str] | None = None
                  ) -> dict[str, list[str]]:
    """Split image ids into train/val/test by group, never by annotation."""
    groups = groups or {i: i for i in image_ids}
    uniq = sorted(set(groups[i] for i in image_ids))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    n = len(uniq)
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))
    if n < 3:
        LOG.error("only %d independent image group(s): a held-out evaluation is "
                  "not possible. Everything will be placed in 'train' and any "
                  "metric computed on it is a sanity check, NOT evidence of "
                  "generalisation.", n)
        return {"train": list(image_ids), "val": [], "test": []}
    test_g = {uniq[i] for i in order[:n_test]}
    val_g = {uniq[i] for i in order[n_test:n_test + n_val]}
    out = {"train": [], "val": [], "test": []}
    for i in image_ids:
        g = groups[i]
        key = "test" if g in test_g else "val" if g in val_g else "train"
        out[key].append(i)
    LOG.info("split by image group -> train=%d val=%d test=%d images",
             len(out["train"]), len(out["val"]), len(out["test"]))
    return out


def specimen_groups(image_ids: Sequence[str]) -> dict[str, str]:
    """Group image ids by the specimen they came from, not by the image.

    ``group_near_duplicates`` only catches images that LOOK alike, so five
    fields imaged from one sample are treated as five independent observations.
    They are not: they share a specimen, a preparation and a coating, so a model
    that has trained on four of them has already seen most of what the fifth has
    to offer. Counting them as independent inflates the apparent sample size and
    makes every cross-validated error bar optimistic.

    The convention in this dataset is ``<specimen>-<field>`` -- ``40s_48-1`` and
    ``40s_48-3`` are two fields of one specimen, ``2-10`` and ``2-22`` two
    fields of another. That is a heuristic about filenames, so the grouping it
    produces is logged: if it merges fields that are genuinely separate samples
    the split becomes conservative rather than wrong, but it is worth a look.
    """
    groups = {i: str(i).rsplit("-", 1)[0] if "-" in str(i) else str(i)
              for i in image_ids}
    sizes: dict[str, int] = {}
    for g in groups.values():
        sizes[g] = sizes.get(g, 0) + 1
    multi = {g: n for g, n in sizes.items() if n > 1}
    LOG.info("specimen grouping: %d image(s) -> %d specimen(s)",
             len(groups), len(sizes))
    if multi:
        LOG.info("  fields sharing a specimen: %s",
                 ", ".join(f"{g} x{n}" for g, n in sorted(multi.items())))
        LOG.warning("effective sample size is %d specimens, not %d images -- "
                    "quote cross-validation spread against the former",
                    len(sizes), len(groups))
    return groups


def merge_groups(*maps: dict[str, str]) -> dict[str, str]:
    """Union of several groupings: ids linked by ANY of them share a group.

    Near-duplicate detection and specimen naming catch different things, and an
    image caught by either must not be split across train and test.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for m in maps:
        for i, g in m.items():
            union(f"id::{i}", f"grp::{g}")
    ids = {i for m in maps for i in m}
    return {i: find(f"id::{i}") for i in ids}


def stratified_grouped_split(image_ids: Sequence[str], strata: dict[str, str], *,
                             val_frac: float = 0.2, test_frac: float = 0.2,
                             seed: int = 1337,
                             groups: dict[str, str] | None = None
                             ) -> dict[str, list[str]]:
    """Grouped split that also balances a stratum across train/val/test.

    The first run of this project put both test images in the one calibration
    group whose pixel size never resolved, so the held-out set differed from
    the training set in magnification *and* in whether nanometre labels existed
    at all.  Any gap measured that way confounds generalisation with a covariate
    shift the split created.  Stratifying on the calibration group removes that.
    """
    groups = groups or {i: i for i in image_ids}
    rng = np.random.default_rng(seed)
    # one stratum label per group (majority vote inside the group)
    g_to_stratum: dict[str, str] = {}
    for i in image_ids:
        g_to_stratum.setdefault(groups[i], strata.get(i, "?"))
    by_stratum: dict[str, list[str]] = {}
    for g, st in g_to_stratum.items():
        by_stratum.setdefault(st, []).append(g)

    # Deal the groups round-robin across strata, then cut the global test and
    # val slices off the front.  Taking a fixed fraction *within* each stratum
    # fails on this dataset: several strata contain a single field, so every
    # per-stratum val/test slice rounds to zero and the held-out sets come back
    # empty.  Round-robin dealing spends the global budget while still touching
    # every stratum before it repeats one.
    order: list[str] = []
    queues = [[gs[i] for i in rng.permutation(len(gs))]
              for _st, gs in sorted(by_stratum.items())]
    while any(queues):
        for q in queues:
            if q:
                order.append(q.pop(0))

    n = len(order)
    n_test = min(n - 1, max(1, int(round(test_frac * n)))) if n >= 3 else 0
    n_val = min(n - n_test - 1, max(1, int(round(val_frac * n)))) if n >= 3 else 0
    if n < 3:
        LOG.error("only %d independent image group(s): a held-out evaluation is "
                  "not possible; use grouped_kfold instead", n)
    test_g = set(order[:n_test])
    val_g = set(order[n_test:n_test + n_val])

    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for i in image_ids:
        g = groups[i]
        key = "test" if g in test_g else "val" if g in val_g else "train"
        out[key].append(i)
    LOG.info("stratified split over %d stratum/strata -> train=%d val=%d test=%d",
             len(by_stratum), len(out["train"]), len(out["val"]), len(out["test"]))
    for st, gs in sorted(by_stratum.items()):
        LOG.info("  stratum %-14s %d group(s)", st, len(gs))
    return out


def calibration_strata(records: Sequence["ImageRecord"]) -> dict[str, str]:
    """Stratum label per image: the nm/px regime, or 'uncalibrated'.

    Fiber width in *pixels* is what the network regresses, so the scale factor
    between fields is the covariate that matters most for generalisation.
    """
    out: dict[str, str] = {}
    for r in records:
        nmpp = r.nm_per_pixel
        if nmpp is None or not np.isfinite(nmpp):
            out[r.image_id] = "uncalibrated"
        elif nmpp < 1.5:
            out[r.image_id] = "nmpp<1.5"
        elif nmpp < 3.0:
            out[r.image_id] = "nmpp1.5-3"
        else:
            out[r.image_id] = "nmpp>=3"
    return out


def grouped_kfold(image_ids: Sequence[str], n_splits: int = 5, *, seed: int = 1337,
                  groups: dict[str, str] | None = None
                  ) -> list[dict[str, list[str]]]:
    """Grouped K-fold, for when there are too few images for a fixed split."""
    groups = groups or {i: i for i in image_ids}
    uniq = sorted(set(groups.values()))
    if len(uniq) < n_splits:
        LOG.warning("%d groups < %d folds; reducing", len(uniq), n_splits)
        n_splits = max(2, len(uniq))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    folds = [set() for _ in range(n_splits)]
    for k, idx in enumerate(order):
        folds[k % n_splits].add(uniq[idx])
    out = []
    for k in range(n_splits):
        val_g = folds[k]
        out.append({
            "train": [i for i in image_ids if groups[i] not in val_g],
            "val": [i for i in image_ids if groups[i] in val_g],
            "test": [],
        })
    return out


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #
@dataclass
class ImageRecord:
    image_id: str
    path: Path
    annotations: Any
    nm_per_pixel: float | None = None
    valid_mask_path: Path | None = None
    negatives: Any = None
    prior_cfg: PriorConfig | None = None
    prior_cache_dir: Path | None = None
    _image: np.ndarray | None = field(default=None, repr=False)
    _valid: np.ndarray | None = field(default=None, repr=False)
    _prior: Any = field(default=None, repr=False)
    _prior_image_only: Any = field(default=None, repr=False)

    def image(self) -> np.ndarray:
        if self._image is None:
            from .calibration import strip_footer
            g = read_gray(self.path)
            g, _row = strip_footer(g)
            self._image = g
        return self._image

    def valid(self) -> np.ndarray:
        if self._valid is None:
            img = self.image()
            if self.valid_mask_path and Path(self.valid_mask_path).exists():
                m = read_gray(self.valid_mask_path)
                # stored mask marks PAINTED pixels; valid is the complement
                self._valid = (m[:img.shape[0], :img.shape[1]] < 128)
            else:
                self._valid = np.ones(img.shape, bool)
        return self._valid

    def prior(self, *, image_only: bool = False) -> FiberPrior:
        """Return the fiber prior, optionally without consulting annotations.

        Training may use annotations to settle an ambiguous bright/dark polarity.
        Evaluation metrics must not: doing so lets the answer key alter the mask
        used to score skeleton coverage.  ``image_only=True`` therefore selects
        polarity from image separability alone and uses a physically separate
        cache so an annotation-assisted prior can never be reused accidentally.
        """
        if image_only:
            if self._prior_image_only is None:
                from .fiber_prior import best_polarity
                prior, pol, auc = best_polarity(
                    self.image(), self.prior_cfg or PriorConfig(), ann=None)
                if prior is None:
                    raise RuntimeError(
                        f"{self.image_id}: image-only fiber prior failed at both polarities")
                self._prior_image_only = prior
                LOG.debug("%s: image-only prior polarity=%s AUC=%s",
                          self.image_id, pol, f"{auc:.3f}" if np.isfinite(auc) else "nan")
            return self._prior_image_only

        if self._prior is None:
            cache = None
            if self.prior_cache_dir is not None:
                cache = Path(self.prior_cache_dir) / f"{self.image_id}_prior"
            self._prior = FiberPrior.load_or_compute(
                self.image(), cache, self.prior_cfg or PriorConfig(),
                ann=self.annotations)
        return self._prior


def load_records(labels_csv: str | Path, image_dir: str | Path, *,
                 mask_dir: str | Path | None = None,
                 prior_cfg: PriorConfig | None = None,
                 prior_cache_dir: str | Path | None = None) -> list[ImageRecord]:
    """Join the consolidated labels table with the clean images on disk.

    Rows flagged ``is_negative`` are reviewer-rejected sites.  They are split
    off into :attr:`ImageRecord.negatives` rather than dropped, because a site a
    human inspected and refused is the most informative negative available --
    and because leaving them in ``annotations`` would paint peaks on them.
    """
    import pandas as pd

    from .utils import duplicate_image_ids, image_id_from_path, list_images

    df = pd.read_csv(labels_csv)
    paths = list_images(image_dir)
    # [v3] A dict comprehension over colliding ids keeps whichever file the
    # directory listing yielded last and says nothing. In the August run
    # 462_1..462_9 all resolved to "462" -- 36 micrographs silently reduced to
    # 4, with one file's measurements attached to another file's pixels.
    clashes = duplicate_image_ids(paths)
    if clashes:
        detail = "\n".join(
            f"  {k}: " + ", ".join(sorted(q.name for q in v))
            for k, v in sorted(clashes.items())[:10])
        raise ValueError(
            f"{len(clashes)} image id(s) match more than one file in "
            f"{image_dir}. Every table is keyed by this id, so one field's "
            f"labels would be attached to another field's pixels. Rename the "
            f"files so each has a distinct stem:\n{detail}")
    by_id = {image_id_from_path(p): p for p in paths}
    if "is_negative" not in df.columns:
        df["is_negative"] = False
    df["is_negative"] = df["is_negative"].fillna(False).astype(bool)

    records: list[ImageRecord] = []
    n_pos = n_neg = 0
    for image_id, sub in df.groupby("image_id"):
        path = by_id.get(str(image_id))
        if path is None:
            LOG.error("no image on disk for image_id=%s (looked in %s)",
                      image_id, image_dir)
            continue
        nmpp = sub["nm_per_pixel"].dropna()
        mask_path = None
        if mask_dir:
            cand = Path(mask_dir) / f"{image_id}_overlay_mask.png"
            mask_path = cand if cand.exists() else None
        pos = sub[~sub["is_negative"]].reset_index(drop=True)
        neg = sub[sub["is_negative"]].reset_index(drop=True)
        n_pos += len(pos)
        n_neg += len(neg)
        records.append(ImageRecord(str(image_id), path, pos,
                                   float(nmpp.iloc[0]) if len(nmpp) else None,
                                   mask_path, negatives=neg,
                                   prior_cfg=prior_cfg,
                                   prior_cache_dir=Path(prior_cache_dir)
                                   if prior_cache_dir else None))
    LOG.info("loaded %d image record(s): %d annotations, %d reviewer-rejected "
             "negatives", len(records), n_pos, n_neg)
    return records


# --------------------------------------------------------------------------- #
# torch datasets
# --------------------------------------------------------------------------- #
class TileDataset:
    """Random tiles from full images, with dense targets.

    Tiles (rather than whole images) keep memory bounded on a 4 GB GPU and give
    the sampler a chance to balance annotated against empty regions.
    """

    def __init__(self, records: Sequence[ImageRecord], tile: int = 384, *,
                 samples_per_image: int = 40, aug: AugConfig | None = None,
                 targets: TargetConfig | None = None, train: bool = True,
                 norm: str = "per_image", seed: int = 0,
                 positive_bias: float = 0.8,
                 prior_cfg: PriorConfig | None = None,
                 use_prior: bool = True) -> None:
        self.prior_cfg = prior_cfg or PriorConfig()
        self.use_prior = use_prior
        self.cache_and_warp = bool(getattr(prior_cfg, "cache_and_warp", True))
        self.records = list(records)
        self.tile = int(tile)
        self.samples_per_image = int(samples_per_image)
        self.aug_cfg = aug or AugConfig(enabled=train)
        self.tcfg = targets or TargetConfig()
        self.train = train
        self.norm = norm
        self.seed = seed
        self.positive_bias = positive_bias
        if not self.records:
            raise ValueError("TileDataset received no image records")

    def __len__(self) -> int:
        return len(self.records) * self.samples_per_image

    def _crop_box(self, rec: ImageRecord, rng: np.random.Generator
                  ) -> tuple[int, int]:
        img = rec.image()
        h, w = img.shape
        t = self.tile
        if h <= t or w <= t:
            return 0, 0
        if len(rec.annotations) and rng.random() < self.positive_bias:
            r = rec.annotations.iloc[int(rng.integers(len(rec.annotations)))]
            cx = int(np.clip(r["center_x_px"] + rng.integers(-t // 3, t // 3),
                             t // 2, w - t // 2))
            cy = int(np.clip(r["center_y_px"] + rng.integers(-t // 3, t // 3),
                             t // 2, h - t // 2))
            return cx - t // 2, cy - t // 2
        return int(rng.integers(0, w - t)), int(rng.integers(0, h - t))

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        rec = self.records[index % len(self.records)]
        rng = np.random.default_rng(self.seed * 1_000_003 + index)
        img = rec.image()
        valid = rec.valid()
        x0, y0 = self._crop_box(rec, rng)
        t = self.tile
        sub = img[y0:y0 + t, x0:x0 + t]
        sub_valid = valid[y0:y0 + t, x0:x0 + t]
        if sub.shape != (t, t):
            pad_y, pad_x = t - sub.shape[0], t - sub.shape[1]
            sub = np.pad(sub, ((0, pad_y), (0, pad_x)), mode="reflect")
            sub_valid = np.pad(sub_valid, ((0, pad_y), (0, pad_x)))

        ann = rec.annotations
        ann = ann[(ann.center_x_px.between(x0 - 8, x0 + t + 8))
                  & (ann.center_y_px.between(y0 - 8, y0 + t + 8))].copy()
        for c in ("center_x_px", "x1_px", "x2_px"):
            if c in ann.columns:
                ann[c] = ann[c] - x0
        for c in ("center_y_px", "y1_px", "y2_px"):
            if c in ann.columns:
                ann[c] = ann[c] - y0

        neg = rec.negatives
        if neg is not None and len(neg):
            neg = neg[(neg.center_x_px.between(x0 - 8, x0 + t + 8))
                      & (neg.center_y_px.between(y0 - 8, y0 + t + 8))].copy()
            neg["center_x_px"] = neg["center_x_px"] - x0
            neg["center_y_px"] = neg["center_y_px"] - y0

        aug = None
        if self.train:
            aug = GeometricAug(self.aug_cfg, rng)
            sub, ann, sub_valid = aug(sub, ann, sub_valid)
            if neg is not None and len(neg):
                _s, neg, _v = aug(sub, neg, sub_valid)
            sub = PhotometricAug(self.aug_cfg, rng)(sub)

        # The prior is recomputed on the final crop rather than warped from the
        # full-image prior.  Warping a binary mask and an angle field through a
        # rotation introduces interpolation error in exactly the quantity being
        # supervised; the ridge filter on a 384px tile is cheap by comparison,
        # and the photometric augmentation is applied first so the prior sees
        # the same pixels the network does.
        prior = None
        if self.use_prior:
            # [v4] The v3 comment is right that interpolating an angle field
            # corrupts the quantity being supervised -- so we do not interpolate
            # it.  The full-image prior is computed once per image (and cached
            # on disk across workers), cropped, then warped with the SAME matrix
            # the image got: positions resampled nearest-neighbour, angle VALUES
            # mapped analytically through the transform's linear part.  That is
            # exact for a similarity transform and turns a per-crop ridge filter
            # into two warps.  Set prior.cache_and_warp=false to restore v3.
            if getattr(self, "cache_and_warp", True):
                full = rec.prior()            # [1e] method, not a field
                base = full.crop(y0, x0, t, t)
                if base.mask.shape != (t, t):  # [1e] boundary / small image
                    base = base.pad_to(t, t)
                if self.train and getattr(aug, "last_M", None) is not None:
                    prior = base.warp_like(aug.last_M, sub.shape[:2])
                else:
                    prior = base
            else:
                prior = FiberPrior.compute(sub, self.prior_cfg, ann=ann)

        tgt = encode_targets(sub.shape, ann, self.tcfg, valid_mask=sub_valid,
                             prior=prior, negatives=neg)
        x = normalize(sub, self.norm)[None]
        out = {"image": torch.from_numpy(np.ascontiguousarray(x)).float(),
               "image_id": rec.image_id,
               "n_ann": int(len(ann))}
        for k, v in tgt.items():
            out[k] = torch.from_numpy(v[None]).float()
        return out


class PatchDataset:
    """Fixed-size patches centred on measurements and on hard negatives.

    This is the sanity-check baseline: if a small CNN cannot regress width from
    a patch centred on the measurement, nothing more elaborate will work either.
    """

    def __init__(self, records: Sequence[ImageRecord], patch: int = 64, *,
                 neg_per_pos: float = 1.0, aug: AugConfig | None = None,
                 train: bool = True, norm: str = "per_image", seed: int = 0,
                 log_width: bool = True, use_prior: bool = True,
                 hard_negative_weight: float = 3.0) -> None:
        self.records = list(records)
        self.patch = int(patch)
        self.train = train
        self.norm = norm
        self.aug_cfg = aug or AugConfig(enabled=train)
        self.log_width = log_width
        rng = np.random.default_rng(seed)
        self.hard_negative_weight = float(hard_negative_weight)
        self.items: list[dict[str, Any]] = []
        for ri, rec in enumerate(self.records):
            ann = rec.annotations
            for _, r in ann.iterrows():
                self.items.append({"rec": ri, "x": float(r["center_x_px"]),
                                   "y": float(r["center_y_px"]),
                                   "width": float(r["width_px"]),
                                   "angle": float(r["measurement_angle_deg"]),
                                   "fiber": float(r.get("local_fiber_angle_deg",
                                                        np.nan)),
                                   "valid": 1.0,
                                   "conf": float(r.get("annotation_confidence", 1.0)),
                                   "weight": 1.0})
            # reviewer-rejected sites first: these are the hard negatives
            hard = rec.negatives
            if hard is not None and len(hard):
                for _, r in hard.iterrows():
                    self.items.append({"rec": ri, "x": float(r["center_x_px"]),
                                       "y": float(r["center_y_px"]),
                                       "width": np.nan, "angle": np.nan,
                                       "fiber": np.nan, "valid": 0.0,
                                       "conf": 1.0,
                                       "weight": hard_negative_weight})
            n_neg = int(round(neg_per_pos * max(1, len(ann))))
            fmask = None
            if use_prior:
                try:
                    fmask = rec.prior().mask
                except Exception as exc:               # pragma: no cover
                    LOG.warning("prior unavailable for %s (%s); sampling "
                                "negatives without a fiber mask",
                                rec.image_id, exc)
            negs = sample_negative_centers(rec.image().shape, ann, n_neg, rng=rng,
                                           valid_mask=rec.valid(),
                                           fiber_mask=fmask)
            for (x, y) in negs:
                self.items.append({"rec": ri, "x": float(x), "y": float(y),
                                   "width": np.nan, "angle": np.nan,
                                   "fiber": np.nan, "valid": 0.0, "conf": 1.0,
                                   "weight": 1.0})
        LOG.info("patch dataset: %d items (%d positive, %d reviewer-rejected)",
                 len(self.items), sum(1 for i in self.items if i["valid"] > 0),
                 sum(1 for i in self.items if i.get("weight", 1.0) > 1.0))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        it = self.items[index]
        rec = self.records[it["rec"]]
        img = rec.image()
        p = self.patch
        h, w = img.shape
        x0 = int(round(it["x"])) - p // 2
        y0 = int(round(it["y"])) - p // 2
        x0 = int(np.clip(x0, 0, max(0, w - p)))
        y0 = int(np.clip(y0, 0, max(0, h - p)))
        sub = img[y0:y0 + p, x0:x0 + p]
        if sub.shape != (p, p):
            sub = np.pad(sub, ((0, p - sub.shape[0]), (0, p - sub.shape[1])),
                         mode="reflect")
        rng = np.random.default_rng(index)
        angle = it["angle"]
        fiber = it["fiber"]
        if self.train:
            if rng.random() < 0.5:
                sub = sub[:, ::-1].copy()
                angle = 180.0 - angle if np.isfinite(angle) else angle
                fiber = 180.0 - fiber if np.isfinite(fiber) else fiber
            if rng.random() < 0.5:
                sub = sub[::-1].copy()
                angle = -angle if np.isfinite(angle) else angle
                fiber = -fiber if np.isfinite(fiber) else fiber
            sub = PhotometricAug(self.aug_cfg, rng)(sub)

        width = it["width"]
        target_w = (np.log(width) if (self.log_width and np.isfinite(width) and width > 0)
                    else (width if np.isfinite(width) else 0.0))
        a = np.deg2rad(angle) if np.isfinite(angle) else 0.0
        f = np.deg2rad(fiber) if np.isfinite(fiber) else 0.0
        return {
            "image": torch.from_numpy(normalize(sub, self.norm)[None]).float(),
            "valid": torch.tensor([it["valid"]], dtype=torch.float32),
            "width": torch.tensor([target_w], dtype=torch.float32),
            "meas_vec": torch.tensor([np.cos(2 * a), np.sin(2 * a)], dtype=torch.float32),
            "fiber_vec": torch.tensor([np.cos(2 * f), np.sin(2 * f)],
                                      dtype=torch.float32),
            "has_angle": torch.tensor([1.0 if np.isfinite(angle) else 0.0],
                                      dtype=torch.float32),
            "conf": torch.tensor([it["conf"]], dtype=torch.float32),
            "weight": torch.tensor([it.get("weight", 1.0)], dtype=torch.float32),
        }
