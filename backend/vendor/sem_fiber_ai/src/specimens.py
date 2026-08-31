"""Specimen grouping, near-duplicate detection and SEALED splits (v7).

Rules enforced here:

* every split decision is made per independent *group*, never per image;
* a group is the union of (a) the specimen a field was imaged from and (b) any
  near-duplicate cluster it belongs to (re-saves, adjacent crops);
* test groups are chosen once, deterministically from the seed, and written to
  a manifest with a hash; :func:`assert_no_leakage` is called by the trainer and
  by the evaluator, so an overlapping split cannot be trained or scored;
* development-time model comparison uses leave-one-group-out folds over the
  non-test groups; the test groups never appear in any fold.

Specimen key.  Filenames on this dataset use ``<specimen>-<field>`` (``2-10``)
and ``<specimen>_<field>`` (``A_8``, ``48_3``, ``462_1``); ``40s_48-1`` is
field 1 of specimen ``40s_48``.  The default rule strips ONE trailing numeric
field token.  Because a filename rule is a heuristic, an explicit
``specimen_map.yaml`` (``image_id: specimen``) overrides it, and a coarser
``series`` level (leading token, e.g. ``40s``) is available for a more
conservative split.  Whatever was used is recorded in the manifest.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Sequence

import numpy as np

from .utils import LOG, load_json, save_json

_FIELD_TOKEN = re.compile(r"^(.*?)[-_](\d+)$")


def specimen_key(image_id: str, *, level: str = "specimen",
                 override: dict[str, str] | None = None) -> str:
    iid = str(image_id)
    if override and iid in override:
        return str(override[iid])
    if level == "image":
        return iid
    if level == "series":
        m = re.match(r"^([A-Za-z0-9]+)", iid)
        return m.group(1) if m else iid
    if level != "specimen":
        raise ValueError(f"unknown grouping level {level!r}")
    m = _FIELD_TOKEN.match(iid)
    return m.group(1) if m else iid


def load_specimen_map(path) -> dict[str, str]:
    from pathlib import Path

    p = Path(path) if path else None
    if p is None or not p.exists():
        return {}
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    else:
        data = load_json(p)
    return {str(k): str(v) for k, v in data.items() if v is not None}


def specimen_groups(image_ids: Sequence[str], *, level: str = "specimen",
                    override: dict[str, str] | None = None) -> dict[str, str]:
    groups = {i: specimen_key(i, level=level, override=override) for i in image_ids}
    sizes: dict[str, int] = {}
    for g in groups.values():
        sizes[g] = sizes.get(g, 0) + 1
    LOG.info("specimen grouping (%s): %d image(s) -> %d specimen(s)%s", level,
             len(groups), len(sizes),
             "; multi-field: " + ", ".join(f"{g} x{n}" for g, n in sorted(sizes.items())
                                          if n > 1) if any(n > 1 for n in sizes.values())
             else "")
    return groups


def near_duplicate_groups(images: dict[str, np.ndarray], *, hamming_max: int = 6
                          ) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Perceptual-hash clustering; returns (image_id -> cluster id, pairs found)."""
    from .image_registration import hamming, perceptual_hash

    ids = sorted(images)
    hashes = {i: perceptual_hash(images[i]) for i in ids}
    parent = {i: i for i in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs = []
    for k, a in enumerate(ids):
        for b in ids[k + 1:]:
            d = int(hamming(hashes[a], hashes[b]))
            if d <= hamming_max:
                pairs.append({"a": a, "b": b, "hamming": d})
                parent[find(b)] = find(a)
    if pairs:
        LOG.warning("near-duplicate fields (hamming <= %d): %s", hamming_max,
                    ", ".join(f"{p['a']}~{p['b']}({p['hamming']})" for p in pairs))
    return {i: find(i) for i in ids}, pairs


def merge_groups(*maps: dict[str, str]) -> dict[str, str]:
    """Union of groupings: ids linked by ANY of them share a group."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for m in maps:
        for i, g in m.items():
            ra, rb = find(f"id::{i}"), find(f"grp::{g}")
            if ra != rb:
                parent[ra] = rb
    ids = sorted({i for m in maps for i in m})
    # name each final group after the smallest specimen label inside it
    labels: dict[str, list[str]] = {}
    for i in ids:
        labels.setdefault(find(f"id::{i}"), []).append(i)
    name = {}
    for root, members in labels.items():
        spec = sorted({maps[0].get(m, m) for m in members}) if maps else sorted(members)
        name[root] = "+".join(spec)
    return {i: name[find(f"id::{i}")] for i in ids}


# --------------------------------------------------------------------------- #
def sealed_split(image_ids: Sequence[str], groups: dict[str, str], *,
                 strata: dict[str, str] | None = None, val_frac: float = 0.2,
                 test_frac: float = 0.2, seed: int = 1337,
                 min_test_groups: int = 1, min_val_groups: int = 1) -> dict[str, Any]:
    """Deterministic grouped split with round-robin stratification.

    Returns ``{"train": [...], "val": [...], "test": [...], "groups": {...},
    "test_groups": [...], "val_groups": [...]}``.  Raises when there are too
    few independent groups to hold anything out -- a held-out set that does not
    exist must not be silently replaced by training images.
    """
    strata = strata or {}
    g_of = {i: groups[i] for i in image_ids}
    uniq = sorted(set(g_of.values()))
    n = len(uniq)
    if n < 3:
        raise ValueError(f"only {n} independent group(s); a sealed train/val/test "
                         "split needs at least 3. Add specimens or use LOSO folds.")
    rng = np.random.default_rng(seed)
    g_stratum: dict[str, str] = {}
    for i in image_ids:
        g_stratum.setdefault(g_of[i], strata.get(i, "?"))
    by_stratum: dict[str, list[str]] = {}
    for g in uniq:
        by_stratum.setdefault(g_stratum[g], []).append(g)
    queues = [[gs[k] for k in rng.permutation(len(gs))] for _s, gs in sorted(by_stratum.items())]
    order: list[str] = []
    while any(queues):
        for q in queues:
            if q:
                order.append(q.pop(0))
    n_test = max(min_test_groups, int(round(test_frac * n)))
    n_val = max(min_val_groups, int(round(val_frac * n)))
    n_test = min(n_test, n - 2)
    n_val = min(n_val, n - n_test - 1)
    test_g = set(order[:n_test])
    val_g = set(order[n_test:n_test + n_val])
    out: dict[str, Any] = {"train": [], "val": [], "test": []}
    for i in image_ids:
        g = g_of[i]
        out["test" if g in test_g else "val" if g in val_g else "train"].append(str(i))
    out["groups"] = {str(i): str(g_of[i]) for i in image_ids}
    out["test_groups"] = sorted(test_g)
    out["val_groups"] = sorted(val_g)
    out["train_groups"] = sorted(set(uniq) - test_g - val_g)
    out["seed"] = int(seed)
    LOG.info("sealed split: train=%d val=%d test=%d images over %d groups "
             "(test groups %s)", len(out["train"]), len(out["val"]), len(out["test"]),
             n, out["test_groups"])
    return out


def loso_folds(image_ids: Sequence[str], groups: dict[str, str], *,
               exclude_groups: Sequence[str] = (), max_folds: int | None = None,
               seed: int = 1337) -> list[dict[str, list[str]]]:
    """Leave-one-group-out folds over the non-excluded (i.e. non-test) groups."""
    ex = set(exclude_groups)
    ids = [i for i in image_ids if groups[i] not in ex]
    uniq = sorted(set(groups[i] for i in ids))
    if max_folds and len(uniq) > max_folds:
        rng = np.random.default_rng(seed)
        uniq = sorted(uniq[k] for k in rng.permutation(len(uniq))[:max_folds])
    folds = []
    for g in uniq:
        folds.append({"train": [i for i in ids if groups[i] != g],
                      "val": [i for i in ids if groups[i] == g],
                      "test": [], "held_out_group": g})
    return folds


def assert_no_leakage(split: dict[str, Any], groups: dict[str, str] | None = None) -> None:
    """Raise if any image OR any group appears in more than one split."""
    groups = groups or split.get("groups") or {}
    seen: dict[str, str] = {}
    gseen: dict[str, str] = {}
    for part in ("train", "val", "test"):
        for i in split.get(part, []):
            if i in seen and seen[i] != part:
                raise RuntimeError(f"LEAKAGE: image {i} is in both {seen[i]} and {part}")
            seen[i] = part
            g = groups.get(i, i)
            if g in gseen and gseen[g] != part:
                raise RuntimeError(f"LEAKAGE: group {g} spans {gseen[g]} and {part} "
                                   f"(image {i})")
            gseen[g] = part


def split_digest(split: dict[str, Any]) -> str:
    core = {k: sorted(split.get(k, [])) for k in ("train", "val", "test")}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:16]


def write_split_manifest(split: dict[str, Any], path, *, extra: dict[str, Any] | None = None
                         ) -> dict[str, Any]:
    man = {**split, "digest": split_digest(split), **(extra or {})}
    save_json(man, path)
    return man


def calibration_strata(nm_per_px: dict[str, float | None]) -> dict[str, str]:
    out = {}
    for i, v in nm_per_px.items():
        if v is None or not np.isfinite(v):
            out[i] = "uncalibrated"
        elif v < 1.5:
            out[i] = "nmpp<1.5"
        elif v < 3.0:
            out[i] = "nmpp1.5-3"
        else:
            out[i] = "nmpp>=3"
    return out
