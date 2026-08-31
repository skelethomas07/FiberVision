"""Roll measurement sites up to fibres/branches (v7).

Thousands of sites along the same fibre are correlated, not independent.  The
reported distributions are therefore built at three separated levels:

a. raw sites (``*_sites.csv``);
b. one record per fibre branch (``*_fibres.csv``): median width, circular-mean
   orientation, length, number of sites;
c. specimen/field summaries, NUMBER-weighted (one vote per fibre) and,
   separately, LENGTH-weighted.

Branches are cut at junctions and re-joined through them when two stubs
continue each other (same continuity rule a person uses when tracing).  Sites
that cannot be assigned to a branch are counted and reported, never dropped
silently.  Adapted from the v6.10 ``fiber_rollup`` module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .coords import angular_diff_180, circular_mean_180, order_parameter_2d
from .skeleton import BranchStructure, branch_structure, nearest_branch


@dataclass
class RollupConfig:
    min_branch_px: float = 12.0
    assign_scale: float = 1.2
    assign_min_px: float = 6.0
    max_angle_deg: float = 35.0
    tangent_window: int = 9
    min_sites: int = 1
    merge_gap_px: float = 14.0
    merge_max_angle_deg: float = 40.0
    merge_junctions: bool = True


def _endpoint_tangents(labels, lab, win):
    ys, xs = np.nonzero(labels == lab)
    if ys.size < 2:
        return []
    y0, x0 = ys.mean(), xs.mean()
    dy, dx = ys - y0, xs - x0
    cov = np.array([[np.dot(dx, dx), np.dot(dx, dy)], [np.dot(dx, dy), np.dot(dy, dy)]])
    w, v = np.linalg.eigh(cov)
    ax, ay = v[:, int(np.argmax(w))]
    order = np.argsort(dx * ax + dy * ay)
    ys, xs = ys[order], xs[order]
    out, k = [], int(min(win, ys.size))
    for end in (0, -1):
        if end == 0:
            py, px, qy, qx = ys[0], xs[0], ys[k - 1], xs[k - 1]
        else:
            py, px, qy, qx = ys[-1], xs[-1], ys[-k], xs[-k]
        uy, ux = float(py - qy), float(px - qx)
        n = np.hypot(uy, ux)
        if n > 1e-6:
            out.append((float(py), float(px), uy / n, ux / n))
    return out


def merge_across_junctions(labels: np.ndarray, cfg: RollupConfig) -> np.ndarray:
    n = int(labels.max())
    if n <= 1:
        return labels
    stubs = [(lab, *t) for lab in range(1, n + 1)
             for t in _endpoint_tangents(labels, lab, cfg.tangent_window)]
    if len(stubs) < 2:
        return labels
    lab_a = np.array([s[0] for s in stubs])
    py, px = np.array([s[1] for s in stubs]), np.array([s[2] for s in stubs])
    uy, ux = np.array([s[3] for s in stubs]), np.array([s[4] for s in stubs])
    d = np.hypot(py[:, None] - py[None, :], px[:, None] - px[None, :])
    dot = uy[:, None] * uy[None, :] + ux[:, None] * ux[None, :]
    vy, vx = py[None, :] - py[:, None], px[None, :] - px[:, None]
    fwd = (uy[:, None] * vy + ux[:, None] * vx) / (np.hypot(vy, vx) + 1e-9)
    ok = ((d <= cfg.merge_gap_px) & (d > 0) & (dot <= -np.cos(np.deg2rad(cfg.merge_max_angle_deg)))
          & (fwd > 0.3) & (lab_a[:, None] != lab_a[None, :]))
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ii, jj = np.nonzero(np.triu(ok, 1))
    if ii.size:
        score = -dot[ii, jj] - 0.01 * d[ii, jj]
        used = np.zeros(len(stubs), bool)
        for k in np.argsort(-score):
            i, j = int(ii[k]), int(jj[k])
            if used[i] or used[j]:
                continue
            ra, rb = find(lab_a[i]), find(lab_a[j])
            if ra == rb:
                continue
            parent[rb] = ra
            used[i] = used[j] = True
    remap, roots = np.zeros(n + 1, np.int32), {}
    for lab in range(1, n + 1):
        r = find(lab)
        roots.setdefault(r, len(roots) + 1)
        remap[lab] = roots[r]
    return remap[labels]


def branch_tangents(labels: np.ndarray) -> dict[int, float]:
    out: dict[int, float] = {}
    if labels.max() == 0:
        return out
    ys, xs = np.nonzero(labels)
    ls = labels[ys, xs]
    order = np.argsort(ls, kind="stable")
    ys, xs, ls = ys[order], xs[order], ls[order]
    bounds = np.append(np.searchsorted(ls, np.arange(1, labels.max() + 1), side="left"), len(ls))
    for lab in range(1, int(labels.max()) + 1):
        a, b = bounds[lab - 1], bounds[lab]
        if b - a < 2:
            continue
        y = ys[a:b].astype(float) - ys[a:b].mean()
        x = xs[a:b].astype(float) - xs[a:b].mean()
        cov = np.array([[np.dot(x, x), np.dot(x, y)], [np.dot(x, y), np.dot(y, y)]]) / (b - a)
        w, v = np.linalg.eigh(cov)
        vx, vy = v[:, int(np.argmax(w))]
        out[lab] = float(np.degrees(np.arctan2(vy, vx)))    # raster
    return out


def rollup(sites: "Any", fibre_mask: np.ndarray, cfg: RollupConfig | None = None,
           *, only_accepted: bool = True) -> tuple["Any", "Any", dict[str, Any]]:
    """Site table -> (fibres, sites with fibre_id, info).  fibre_id 0 = unassigned."""
    import pandas as pd

    cfg = cfg or RollupConfig()
    bs = branch_structure(fibre_mask, min_branch_px=int(cfg.min_branch_px))
    labels = merge_across_junctions(bs.labels, cfg) if cfg.merge_junctions else bs.labels
    tangents = branch_tangents(labels)
    s = sites.copy()
    if only_accepted and "rejected_reason" in s.columns:
        s = s[s["rejected_reason"].fillna("") == ""].copy()
    fid = np.zeros(len(s), np.int32)
    if len(s) and labels.max() > 0:
        nb_lab, nb_dist = nearest_branch(BranchStructure(bs.skeleton, labels, bs.junction,
                                                         bs.junction_dist, bs.edt, int(labels.max())))
        H, W = labels.shape
        cx = np.clip(np.rint(s["center_x_px"].to_numpy(float)), 0, W - 1).astype(int)
        cy = np.clip(np.rint(s["center_y_px"].to_numpy(float)), 0, H - 1).astype(int)
        wid = s["width_px"].to_numpy(float)
        ang = s["fiber_angle_raster_deg"].to_numpy(float)
        near, d = nb_lab[cy, cx], nb_dist[cy, cx]
        tol = np.maximum(cfg.assign_min_px, cfg.assign_scale * np.nan_to_num(wid))
        tan = np.array([tangents.get(int(l), np.nan) for l in near])
        ok = (near > 0) & (d <= tol)
        with np.errstate(invalid="ignore"):
            ok &= np.isfinite(tan) & (angular_diff_180(ang, tan) <= cfg.max_angle_deg)
        fid[ok] = near[ok]
    s["fiber_id"] = fid
    lengths = np.bincount(labels.ravel()) if labels.max() > 0 else np.zeros(1)
    rows = []
    for lab, grp in s[s["fiber_id"] > 0].groupby("fiber_id"):
        if len(grp) < cfg.min_sites:
            continue
        w = grp["width_px"].to_numpy(float)
        a = grp["fiber_angle_raster_deg"].to_numpy(float)
        mean_a = circular_mean_180(a)
        rows.append({
            "fiber_id": int(lab), "n_sites": int(len(grp)),
            "length_px": float(lengths[lab]) if lab < len(lengths) else np.nan,
            "width_px": float(np.median(w)),
            "width_iqr_px": float(np.subtract(*np.percentile(w, [75, 25]))) if len(w) > 1 else 0.0,
            "fiber_angle_raster_deg": mean_a,
            "angle_spread_deg": float(np.median(angular_diff_180(a, mean_a))) if len(a) > 1 else 0.0,
            "skeleton_angle_raster_deg": tangents.get(int(lab), np.nan),
            "confidence": float(np.median(grp["confidence"])) if "confidence" in grp else np.nan,
            "validity": float(np.median(grp["validity"])) if "validity" in grp else np.nan,
        })
    fibres = pd.DataFrame(rows, columns=[
        "fiber_id", "n_sites", "length_px", "width_px", "width_iqr_px", "fiber_angle_raster_deg",
        "angle_spread_deg", "skeleton_angle_raster_deg", "confidence", "validity"])
    if "nm_per_pixel" in s.columns and len(fibres):
        nmpp = float(np.nanmedian(s["nm_per_pixel"].to_numpy(float))) if len(s) else np.nan
        fibres["width_nm"] = fibres["width_px"] * nmpp if np.isfinite(nmpp) else np.nan
        fibres["length_nm"] = fibres["length_px"] * nmpp if np.isfinite(nmpp) else np.nan
    info = {"n_sites": int(len(s)), "n_assigned": int((fid > 0).sum()),
            "n_unassigned": int((fid == 0).sum()),
            "unassigned_fraction": float((fid == 0).mean()) if len(s) else np.nan,
            "n_fibres": int(len(fibres)), "n_branches": int(labels.max()),
            "sites_per_fibre_median": float(fibres["n_sites"].median()) if len(fibres) else np.nan}
    return fibres, s, info


def distribution_summary(values: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    v = v[ok]
    if v.size == 0:
        return {"n": 0}
    if weights is None:
        q = np.percentile(v, [5, 25, 50, 75, 90, 95])
        return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                "p5": float(q[0]), "p25": float(q[1]), "median": float(q[2]), "p75": float(q[3]),
                "p90": float(q[4]), "p95": float(q[5]), "iqr": float(q[3] - q[1])}
    w = np.asarray(weights, float)[ok]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w) / w.sum()

    def wq(p):
        return float(np.interp(p / 100.0, cw, v))

    mean = float(np.average(v, weights=w))
    sd = float(np.sqrt(np.average((v - mean) ** 2, weights=w)))
    return {"n": int(v.size), "mean": mean, "sd": sd, "p5": wq(5), "p25": wq(25),
            "median": wq(50), "p75": wq(75), "p90": wq(90), "p95": wq(95),
            "iqr": wq(75) - wq(25), "weighting": "length"}


def field_summary(fibres: "Any", sites: "Any", info: dict[str, Any], *, nm_valid: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"roll_up": info}
    if len(fibres):
        fw = fibres["width_px"].to_numpy(float)
        out["number_weighted_px"] = distribution_summary(fw)
        out["length_weighted_px"] = distribution_summary(fw, fibres["length_px"].to_numpy(float))
        out["order_parameter_S_fibre"] = order_parameter_2d(fibres["fiber_angle_raster_deg"].to_numpy(float))
        if nm_valid and "width_nm" in fibres.columns:
            out["number_weighted_nm"] = distribution_summary(fibres["width_nm"].to_numpy(float))
            out["length_weighted_nm"] = distribution_summary(fibres["width_nm"].to_numpy(float),
                                                              fibres["length_px"].to_numpy(float))
    if len(sites):
        acc = sites[sites["rejected_reason"].fillna("") == ""] if "rejected_reason" in sites else sites
        out["raw_sites_px"] = distribution_summary(acc["width_px"].to_numpy(float))
        out["n_raw_sites_accepted"] = int(len(acc))
    out["nm_valid"] = bool(nm_valid)
    out["note"] = ("number_weighted = one vote per fibre branch (the reported distribution); "
                   "raw_sites are correlated chords and are NOT independent fibres")
    return out
