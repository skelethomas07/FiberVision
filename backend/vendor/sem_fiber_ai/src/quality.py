"""Field-level quality status: PASS / REVIEW / FAIL (v7).

Distinct quantities, never conflated:

* model confidence           (segmentation probability, centre score)
* aleatoric width uncertainty (``width_sigma_px`` from the logvar head)
* calibration validity        (nm/px status; affects nm outputs only)
* segmentation support        (mask area, intensity separability AUC)
* boundary agreement          (distance-head width vs width-head width)
* out-of-distribution warning (intensity statistics vs the training fields)

Publication-quality summaries use PASS fields only by default.  REVIEW fields
are kept in the tables with their reasons; FAIL fields are excluded from every
summary.  Confidence is never treated as accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

PASS, REVIEW, FAIL = "PASS", "REVIEW", "FAIL"


@dataclass
class QualityThresholds:
    seg_area_min: float = 0.05
    seg_area_max: float = 0.90
    auc_min: float = 0.70
    auc_review: float = 0.78
    coherent_fraction_min: float = 0.20
    boundary_agreement_min: float = 0.50      # fraction of sites within tol
    boundary_agreement_review: float = 0.70
    unassigned_fraction_max: float = 0.50
    unassigned_fraction_review: float = 0.30
    min_fibres_fail: int = 10
    min_fibres_review: int = 30
    ood_z_review: float = 3.0
    ood_z_fail: float = 5.0
    accepted_fraction_review: float = 0.4


@dataclass
class TrainingStats:
    """Intensity statistics of the training fields for OOD screening."""
    mean_mu: float = 128.0
    mean_sd: float = 30.0
    std_mu: float = 40.0
    std_sd: float = 15.0
    contrast_mu: float = 0.5
    contrast_sd: float = 0.2
    n_fields: int = 0
    nm_per_px_range: tuple[float, float] = (0.5, 6.0)

    def to_dict(self) -> dict[str, Any]:
        return {"mean_mu": self.mean_mu, "mean_sd": self.mean_sd, "std_mu": self.std_mu,
                "std_sd": self.std_sd, "contrast_mu": self.contrast_mu,
                "contrast_sd": self.contrast_sd, "n_fields": self.n_fields,
                "nm_per_px_range": list(self.nm_per_px_range)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingStats":
        d = dict(d)
        if "nm_per_px_range" in d:
            d["nm_per_px_range"] = tuple(d["nm_per_px_range"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def image_stats(gray: np.ndarray) -> dict[str, float]:
    g = np.asarray(gray, np.float64)
    lo, hi = np.percentile(g, (2, 98))
    return {"mean": float(g.mean()), "std": float(g.std()),
            "contrast": float((hi - lo) / 255.0)}


def training_stats(images: list[np.ndarray], nm_per_px: list[float | None]) -> TrainingStats:
    s = [image_stats(g) for g in images]
    m = np.array([x["mean"] for x in s])
    sd = np.array([x["std"] for x in s])
    c = np.array([x["contrast"] for x in s])
    nm = [v for v in nm_per_px if v is not None and np.isfinite(v)]
    return TrainingStats(float(m.mean()), float(max(m.std(), 5.0)), float(sd.mean()),
                         float(max(sd.std(), 3.0)), float(c.mean()), float(max(c.std(), 0.05)),
                         len(s), (float(min(nm)), float(max(nm))) if nm else (0.5, 6.0))


def ood_warnings(gray: np.ndarray, stats: TrainingStats | None, nm_per_px: float | None,
                 thr: QualityThresholds) -> tuple[list[dict[str, Any]], float]:
    if stats is None:
        return [], 0.0
    s = image_stats(gray)
    zs = {"mean": abs(s["mean"] - stats.mean_mu) / stats.mean_sd,
          "std": abs(s["std"] - stats.std_mu) / stats.std_sd,
          "contrast": abs(s["contrast"] - stats.contrast_mu) / stats.contrast_sd}
    warns = [{"code": f"ood_intensity_{k}", "z": float(z)} for k, z in zs.items()
             if z > thr.ood_z_review]
    if nm_per_px is not None and np.isfinite(nm_per_px):
        lo, hi = stats.nm_per_px_range
        if not (0.7 * lo <= nm_per_px <= 1.3 * hi):
            warns.append({"code": "ood_pixel_size", "nm_per_px": float(nm_per_px),
                          "training_range": [lo, hi]})
    return warns, float(max(zs.values()))


def field_status(*, seg_area: float, separability_auc: float | None, coherent_fraction: float,
                 boundary_agreement: float | None, unassigned_fraction: float | None,
                 n_fibres: int, accepted_fraction: float, ood_z: float, ood: list[dict[str, Any]],
                 calibration_valid: bool, calibration_reason: str = "",
                 thr: QualityThresholds | None = None) -> dict[str, Any]:
    thr = thr or QualityThresholds()
    fails, reviews = [], []
    if not (thr.seg_area_min <= seg_area <= thr.seg_area_max):
        fails.append(f"segmentation_area_{seg_area:.2f}_outside_[{thr.seg_area_min},{thr.seg_area_max}]")
    if separability_auc is not None and np.isfinite(separability_auc):
        if separability_auc < thr.auc_min:
            fails.append(f"segmentation_separability_auc_{separability_auc:.2f}<{thr.auc_min}")
        elif separability_auc < thr.auc_review:
            reviews.append(f"segmentation_separability_auc_{separability_auc:.2f}<{thr.auc_review}")
    if coherent_fraction < thr.coherent_fraction_min:
        reviews.append(f"low_coherent_fraction_{coherent_fraction:.2f}")
    if boundary_agreement is not None and np.isfinite(boundary_agreement):
        if boundary_agreement < thr.boundary_agreement_min:
            fails.append(f"boundary_agreement_{boundary_agreement:.2f}<{thr.boundary_agreement_min}")
        elif boundary_agreement < thr.boundary_agreement_review:
            reviews.append(f"boundary_agreement_{boundary_agreement:.2f}<{thr.boundary_agreement_review}")
    if unassigned_fraction is not None and np.isfinite(unassigned_fraction):
        if unassigned_fraction > thr.unassigned_fraction_max:
            fails.append(f"unassigned_fraction_{unassigned_fraction:.2f}>{thr.unassigned_fraction_max}")
        elif unassigned_fraction > thr.unassigned_fraction_review:
            reviews.append(f"unassigned_fraction_{unassigned_fraction:.2f}>{thr.unassigned_fraction_review}")
    if n_fibres < thr.min_fibres_fail:
        fails.append(f"too_few_fibres_{n_fibres}<{thr.min_fibres_fail}")
    elif n_fibres < thr.min_fibres_review:
        reviews.append(f"few_fibres_{n_fibres}<{thr.min_fibres_review}")
    if accepted_fraction < thr.accepted_fraction_review:
        reviews.append(f"low_accepted_site_fraction_{accepted_fraction:.2f}")
    if ood_z > thr.ood_z_fail:
        fails.append(f"out_of_distribution_z_{ood_z:.1f}")
    elif ood:
        reviews.append("out_of_distribution_warning")
    status = FAIL if fails else REVIEW if reviews else PASS
    return {"status": status, "fail_reasons": fails, "review_reasons": reviews,
            "ood_warnings": ood, "nm_status": "valid" if calibration_valid else "calibration_invalid",
            "nm_reason": calibration_reason,
            "note": "PASS fields only enter publication summaries; nm values exist only when "
                    "nm_status is valid; model confidence is not accuracy"}


def publication_set(statuses: dict[str, dict[str, Any]], *, include_review: bool = False) -> list[str]:
    ok = {PASS} | ({REVIEW} if include_review else set())
    return sorted(k for k, v in statuses.items() if v.get("status") in ok)
