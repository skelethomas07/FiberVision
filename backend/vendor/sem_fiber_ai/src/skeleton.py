"""Medial axis, branches and junction geometry of a fibre mask (v7).

One implementation shared by target construction, post-processing and the
fibre roll-up, so "a branch" means the same thing everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BranchStructure:
    skeleton: np.ndarray        # bool
    labels: np.ndarray          # int32, 0 off-branch, 1..N on branch pixels
    junction: np.ndarray        # bool, dilated junction pixels
    junction_dist: np.ndarray   # float32, distance to nearest junction pixel
    edt: np.ndarray             # float32, distance-to-boundary inside the mask
    n_branches: int


def neighbour_count(skel: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi

    k = np.ones((3, 3), np.uint8)
    k[1, 1] = 0
    return ndi.convolve(skel.astype(np.uint8), k, mode="constant")


def branch_structure(mask: np.ndarray, *, min_branch_px: int = 8) -> BranchStructure:
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize

    m = np.asarray(mask, bool)
    h, w = m.shape
    if not m.any():
        z = np.zeros((h, w), np.float32)
        return BranchStructure(np.zeros((h, w), bool), np.zeros((h, w), np.int32),
                               np.zeros((h, w), bool), z + np.hypot(h, w), z, 0)
    edt = ndi.distance_transform_edt(m).astype(np.float32)
    skel = skeletonize(m)
    nb = neighbour_count(skel)
    junction = skel & (nb >= 3)
    junction = ndi.binary_dilation(junction, np.ones((3, 3), bool)) & skel
    branches = skel & ~junction
    lab, n = ndi.label(branches, structure=np.ones((3, 3), int))
    if n:
        sizes = np.bincount(lab.ravel())
        small = np.flatnonzero(sizes < min_branch_px)
        small = small[small > 0]
        if small.size:
            lab[np.isin(lab, small)] = 0
        used = np.unique(lab)
        used = used[used > 0]
        remap = np.zeros(int(lab.max()) + 1, np.int32)
        remap[used] = np.arange(1, used.size + 1, dtype=np.int32)
        lab = remap[lab]
        n = int(used.size)
    if junction.any():
        jd = ndi.distance_transform_edt(~junction).astype(np.float32)
    else:
        jd = np.full((h, w), float(np.hypot(h, w)), np.float32)
    return BranchStructure(skel, lab.astype(np.int32), junction, jd, edt, int(n))


def nearest_branch(bs: BranchStructure) -> tuple[np.ndarray, np.ndarray]:
    """For every pixel: label of the nearest branch pixel and the distance to it."""
    from scipy import ndimage as ndi

    on = bs.labels > 0
    if not on.any():
        h, w = bs.labels.shape
        return np.zeros((h, w), np.int32), np.full((h, w), np.inf, np.float32)
    dist, (iy, ix) = ndi.distance_transform_edt(~on, return_indices=True)
    return bs.labels[iy, ix], dist.astype(np.float32)


def spaced_sites(bs: BranchStructure, spacing_px: float, *, score: np.ndarray | None = None
                 ) -> list[tuple[int, int, int]]:
    """Greedy sites along every branch at >= ``spacing_px`` apart.

    Returns ``[(y, x, branch_label), ...]``.  Pixels are visited in descending
    ``score`` (default: distance-to-boundary, so the best-centred pixels win),
    and a pixel is kept only if no kept pixel of the SAME branch lies within the
    spacing.  Spacing along the arc is therefore controlled explicitly instead of
    being a by-product of the fibre width, which is what produced the 1/w
    chord weighting in earlier versions.
    """
    ys, xs = np.nonzero(bs.labels > 0)
    if ys.size == 0:
        return []
    sc = (bs.edt if score is None else score)[ys, xs]
    order = np.argsort(-sc, kind="stable")
    labs = bs.labels[ys, xs]
    kept: dict[int, list[tuple[float, float]]] = {}
    out = []
    s2 = float(spacing_px) ** 2
    for k in order:
        y, x, lb = int(ys[k]), int(xs[k]), int(labs[k])
        pts = kept.setdefault(lb, [])
        ok = True
        for (py, px) in pts:
            if (py - y) ** 2 + (px - x) ** 2 < s2:
                ok = False
                break
        if ok:
            pts.append((y, x))
            out.append((y, x, lb))
    return out
