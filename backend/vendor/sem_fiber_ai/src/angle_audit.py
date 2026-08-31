"""Decide, empirically, what the CSV angle column actually means.

The orientation error on the first evaluation was 19.9 deg median.  That is a
suspicious number: it is far too large for a model that fits width to within a
pixel, and it is the same magnitude as the chord-versus-ridge disagreement
already logged during annotation recovery.  Two very different things produce
it, and they call for opposite responses:

*a convention error* -- the angle column is the fiber direction rather than the
chord direction, or the export used y-up while the code assumes y-down.  These
show up as a large *systematic* offset near 90 deg or as a sign flip, and no
amount of training will fix them.

*annotator scatter* -- the chord is hand-drawn and only roughly perpendicular
to the fiber.  This shows up as a near-zero median with a wide spread, and the
right response is to stop supervising orientation from the chord at all, which
is what :mod:`targets` now does.

This module tests every hypothesis against the structure tensor, which is
independent of the CSV, and reports which one wins.  It answers the question
with the data instead of by reading the exporter's documentation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fiber_prior import FiberPrior, PriorConfig
from .utils import angular_diff_180, get_logger, wrap_deg_180

LOG = get_logger(__name__)

#: (name, y_sign, offset_deg) -- every reading of the angle column worth testing
HYPOTHESES: tuple[tuple[str, float, float], ...] = (
    ("chord, y down (current default)", 1.0, -90.0),
    ("chord, y up", -1.0, -90.0),
    ("fiber direction, y down", 1.0, 0.0),
    ("fiber direction, y up", -1.0, 0.0),
)


@dataclass
class AngleAudit:
    image_id: str
    n: int
    best: str
    results: list[dict[str, Any]]
    scatter_deg: float
    verdict: str


def audit_angles(gray: np.ndarray, ann: "Any", *,
                 image_id: str = "", cfg: PriorConfig | None = None,
                 coherency_min: float = 0.35) -> AngleAudit:
    """Score each interpretation of the angle column against the image."""
    cfg = cfg or PriorConfig()
    prior = FiberPrior.compute(gray, cfg, ann=ann)
    h, w = prior.shape

    xs = np.clip(ann["center_x_px"].to_numpy(float).round().astype(int), 0, w - 1)
    ys = np.clip(ann["center_y_px"].to_numpy(float).round().astype(int), 0, h - 1)
    csv_ang = ann["measurement_angle_deg"].to_numpy(float)
    ref = prior.angle_deg[ys, xs]
    coh = prior.coherency[ys, xs]

    ok = np.isfinite(csv_ang) & (coh >= coherency_min)
    results: list[dict[str, Any]] = []
    if ok.sum() < 10:
        LOG.warning("%s: only %d annotations sit on coherent fiber; the angle "
                    "audit is not conclusive", image_id, int(ok.sum()))

    for name, y_sign, offset in HYPOTHESES:
        # y_sign = -1 mirrors the angle about the x axis
        a = csv_ang[ok] * (1.0 if y_sign > 0 else -1.0) + offset
        d = np.asarray(angular_diff_180(wrap_deg_180(a), ref[ok]), float)
        results.append({
            "hypothesis": name, "y_sign": y_sign, "offset_deg": offset,
            "median_error_deg": float(np.median(d)) if d.size else float("nan"),
            "mean_error_deg": float(d.mean()) if d.size else float("nan"),
            "within_15deg": float((d <= 15).mean()) if d.size else float("nan"),
            "n": int(d.size),
        })

    results.sort(key=lambda r: (np.nan_to_num(r["median_error_deg"], nan=1e9)))
    best = results[0]
    scatter = float(best["median_error_deg"])

    runner_up = results[1]["median_error_deg"] if len(results) > 1 else np.inf
    margin = runner_up - best["median_error_deg"]
    if not np.isfinite(scatter):
        verdict = "inconclusive: no coherent annotated sites"
    elif margin < 5.0:
        verdict = ("inconclusive: two interpretations score within 5 deg of "
                   "each other, so the column does not determine the geometry")
    elif scatter <= 8.0:
        verdict = (f"convention confirmed ({best['hypothesis']}); residual "
                   f"{scatter:.1f} deg is ordinary annotator scatter")
    elif best["hypothesis"] != HYPOTHESES[0][0]:
        verdict = (f"CONVENTION ERROR: the data fits '{best['hypothesis']}', "
                   f"not the current default. Fix the reader before training.")
    else:
        verdict = (f"convention is right but scatter is large ({scatter:.1f} deg): "
                   "supervise orientation from the image, not the chord")
    LOG.info("%s: %s", image_id or "image", verdict)
    return AngleAudit(image_id, int(ok.sum()), best["hypothesis"], results,
                      scatter, verdict)


def audit_records(records, *, cfg: PriorConfig | None = None) -> "Any":
    """Run the audit over every labelled field and summarise."""
    import pandas as pd

    rows = []
    for rec in records:
        try:
            a = audit_angles(rec.image(), rec.annotations, image_id=rec.image_id,
                             cfg=cfg)
        except Exception as exc:                          # pragma: no cover
            LOG.error("angle audit failed for %s: %s", rec.image_id, exc)
            continue
        rows.append({"image_id": a.image_id, "n": a.n, "best": a.best,
                     "median_error_deg": round(a.scatter_deg, 2),
                     "verdict": a.verdict})
    df = pd.DataFrame(rows)
    if len(df) and df["best"].nunique() > 1:
        LOG.error("different fields prefer different angle conventions (%s). "
                  "That means at least one export differs from the others; "
                  "resolve it per source_csv before pooling them.",
                  ", ".join(sorted(df["best"].unique())))
    return df
