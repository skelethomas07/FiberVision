"""Shared test helpers (oracle maps for synthetic fields)."""
from __future__ import annotations

import numpy as np


def oracle_maps(field, *, big: float = 12.0, exact_distance: bool = True) -> dict:
    """Maps a perfect model would produce for a synthetic field.

    ``exact_distance=True`` uses the continuous distance to the boundary from
    the fibre geometry; ``False`` uses the EDT of the discrete mask (which is
    biased high by ~0.5 px, the bias the learned target scaling absorbs).
    """
    from scipy import ndimage as ndi

    mask = field.mask
    H, W = mask.shape
    yy, xx = np.mgrid[0:H, 0:W]
    best_d = np.full((H, W), np.inf)
    ang = np.zeros((H, W))
    dist = np.zeros((H, W))
    for r in field.fibres.itertuples():
        d = np.abs(-(xx - r.cx) * r.uy + (yy - r.cy) * r.ux)
        t = (xx - r.cx) * r.ux + (yy - r.cy) * r.uy
        inside = (d <= r.width / 2.0) & (np.abs(t) <= r.L / 2.0)
        sel = inside & (d < best_d)
        best_d[sel] = d[sel]
        ang[sel] = r.angle
        dist = np.where(sel, np.maximum(dist, r.width / 2.0 - d), dist)
    if not exact_distance:
        dist = ndi.distance_transform_edt(mask)
    th = np.deg2rad(ang)
    return {"segment_logit": np.where(mask, big, -big).astype(np.float32),
            "center_logit": np.full((H, W), -big, np.float32),
            "validity_logit": np.full((H, W), big, np.float32),
            "dist": dist.astype(np.float32),
            "width": np.log(np.maximum(2 * dist, 1.0)).astype(np.float32),
            "logvar": np.full((H, W), -4.0, np.float32),
            "orient": np.stack([np.cos(2 * th), np.sin(2 * th)]).astype(np.float32)}
