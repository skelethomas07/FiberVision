"""Loss functions, one per head, all mask-aware.

The recurring hazard in this task is supervising where there is no label.  Only
a few hundred pixels in a million carry a width or an orientation, so every
regression loss is evaluated strictly inside its mask, and the ignore map
removes inpainted overlay, footer, ambiguous sites and -- since the encoder
rewrite -- unlabelled fiber from *all* terms.

Two things changed here alongside that rewrite.

Confirmed negatives are weighted
    Sites a reviewer inspected and rejected carry ``neg_boost`` > 1.  They are
    worth more than ordinary background because they are the cases the model
    actually gets wrong: a crossing that looks like a clean fiber, a fold, an
    artefact.  Ordinary background is trivially negative and there is a million
    pixels of it.

The positive term is normalised by *effective* positives
    With most of the image now ignored, ``n_pos`` and the surviving negative
    count both drop.  The focal loss is renormalised so the positive/negative
    balance does not silently shift when the ignore fraction changes between
    images -- otherwise a dense field and a sparse one would train at different
    effective learning rates on the same head.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    center: float = 1.0
    segment: float = 0.5
    width: float = 1.0
    orient: float = 0.5
    validity: float = 0.2
    uncertainty: float = 0.1


def focal_heatmap_loss(logit: torch.Tensor, target: torch.Tensor,
                       ignore: torch.Tensor | None = None,
                       neg_boost: torch.Tensor | None = None,
                       alpha: float = 2.0, beta: float = 4.0,
                       neg_norm: bool = True) -> torch.Tensor:
    """CornerNet penalty-reduced focal loss for Gaussian heatmaps.

    Pixels near a peak are down-weighted rather than treated as hard negatives,
    which matters here because the exact centre a human picked is arbitrary
    within a few pixels along the fiber.

    ``neg_boost`` multiplies the negative term at reviewer-rejected sites.
    ``neg_norm`` divides the negative term by its own pixel count instead of by
    the positive count, so that changing how much of the image is ignored does
    not rescale the negative pressure.
    """
    pred = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = target.ge(0.999).float()
    neg = 1.0 - pos
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, alpha) * pos
    neg_loss = (-torch.log(1 - pred) * torch.pow(pred, alpha)
                * torch.pow(1 - target, beta) * neg)
    if neg_boost is not None:
        neg_loss = neg_loss * (1.0 + neg_boost)
    keep = torch.ones_like(target)
    if ignore is not None:
        keep = 1.0 - ignore
        pos_loss, neg_loss = pos_loss * keep, neg_loss * keep
        pos = pos * keep
        neg = neg * keep
    n_pos = pos.sum().clamp_min(1.0)
    if neg_norm:
        n_neg = neg.sum().clamp_min(1.0)
        # keep the two terms on the historical scale (both were /n_pos) while
        # making the negative term invariant to the ignored fraction
        return pos_loss.sum() / n_pos + neg_loss.sum() / n_neg * 1.0
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def dice_bce_loss(logit: torch.Tensor, target: torch.Tensor,
                  ignore: torch.Tensor | None = None,
                  eps: float = 1.0) -> torch.Tensor:
    keep = (1.0 - ignore) if ignore is not None else torch.ones_like(target)
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    bce = (bce * keep).sum() / keep.sum().clamp_min(1.0)
    p = torch.sigmoid(logit) * keep
    t = target * keep
    inter = (p * t).sum()
    dice = 1.0 - (2 * inter + eps) / (p.sum() + t.sum() + eps)
    return bce + dice


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                 delta: float = 1.0) -> torch.Tensor:
    n = mask.sum().clamp_min(1.0)
    return (F.huber_loss(pred, target, reduction="none", delta=delta) * mask).sum() / n


def heteroscedastic_loss(pred: torch.Tensor, target: torch.Tensor,
                         logvar: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Gaussian NLL with a learned per-pixel variance.

    Lets the model say "this site is a crossing, my width estimate is unreliable"
    instead of being forced to commit.  The 0.5*logvar term is what stops it from
    declaring everything uncertain.

    ``logvar`` is clamped: with the width target now propagated along fibers,
    an unclamped variance can run away on the propagated pixels and switch the
    head off entirely.
    """
    n = mask.sum().clamp_min(1.0)
    logvar = logvar.clamp(-6.0, 6.0)
    inv = torch.exp(-logvar)
    nll = 0.5 * (inv * (pred - target) ** 2 + logvar)
    return (nll * mask).sum() / n


def orientation_loss(pred_vec: torch.Tensor, cos2t: torch.Tensor,
                     sin2t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cosine distance in (cos 2t, sin 2t) space.

    Working on the doubled angle makes the loss pi-periodic by construction: a
    fiber at +85 deg and one at -85 deg are 10 deg apart, not 170.
    """
    target = torch.cat([cos2t, sin2t], dim=1)
    tn = target.norm(dim=1, keepdim=True).clamp_min(1e-6)
    target = target / tn
    cos_sim = (pred_vec * target).sum(dim=1, keepdim=True)
    n = mask.sum().clamp_min(1.0)
    return ((1.0 - cos_sim) * mask).sum() / n


class MultiHeadLoss(nn.Module):
    """Weighted sum of the per-head losses; weights come from the YAML."""

    def __init__(self, weights: LossWeights | dict[str, float] | None = None,
                 *, use_uncertainty: bool = True,
                 use_neg_boost: bool = True) -> None:
        super().__init__()
        if isinstance(weights, dict):
            known = {f for f in LossWeights.__dataclass_fields__}
            weights = LossWeights(**{k: v for k, v in weights.items() if k in known})
        self.w = weights or LossWeights()
        self.use_uncertainty = use_uncertainty
        self.use_neg_boost = use_neg_boost

    def forward(self, out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
                ) -> tuple[torch.Tensor, dict[str, float]]:
        ignore = batch["ignore"]
        reg = batch["reg_mask"]
        boost = batch.get("neg_boost") if self.use_neg_boost else None

        l_center = focal_heatmap_loss(out["center_logit"], batch["center"],
                                      ignore, boost)
        l_seg = dice_bce_loss(out["segment_logit"], batch["segment"], ignore)
        if self.use_uncertainty:
            l_width = heteroscedastic_loss(out["width"], batch["width"],
                                           out["logvar"], reg)
        else:
            l_width = masked_huber(out["width"], batch["width"], reg)
        l_orient = orientation_loss(out["orient"], batch["cos2t"], batch["sin2t"],
                                    reg)

        keep = 1.0 - ignore
        val_bce = F.binary_cross_entropy_with_logits(
            out["validity_logit"], (batch["validity"] > 0.5).float(), reduction="none")
        if boost is not None:
            # a rejected site is a validity negative the reviewer confirmed
            keep = keep * (1.0 + boost)
        l_valid = (val_bce * keep).sum() / keep.sum().clamp_min(1.0)

        total = (self.w.center * l_center + self.w.segment * l_seg
                 + self.w.width * l_width + self.w.orient * l_orient
                 + self.w.validity * l_valid)

        parts = {"center": float(l_center.detach()), "segment": float(l_seg.detach()),
                 "width": float(l_width.detach()), "orient": float(l_orient.detach()),
                 "validity": float(l_valid.detach()), "total": float(total.detach())}
        return total, parts


class PatchLoss(nn.Module):
    """Baseline objective: validity BCE + masked width/orientation regression."""

    def __init__(self, weights: LossWeights | dict[str, float] | None = None,
                 *, use_uncertainty: bool = True) -> None:
        super().__init__()
        if isinstance(weights, dict):
            known = {f for f in LossWeights.__dataclass_fields__}
            weights = LossWeights(**{k: v for k, v in weights.items() if k in known})
        self.w = weights or LossWeights()
        self.use_uncertainty = use_uncertainty

    def forward(self, out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
                ) -> tuple[torch.Tensor, dict[str, float]]:
        valid = batch["valid"]
        weight = batch.get("weight")
        if weight is None:
            weight = torch.ones_like(valid)
        bce = F.binary_cross_entropy_with_logits(out["valid_logit"], valid,
                                                 reduction="none")
        l_valid = (bce * weight).sum() / weight.sum().clamp_min(1.0)
        if self.use_uncertainty:
            l_width = heteroscedastic_loss(out["width"], batch["width"],
                                           out["logvar"], valid)
        else:
            l_width = masked_huber(out["width"], batch["width"], valid)
        ang_mask = valid * batch["has_angle"]
        n = ang_mask.sum().clamp_min(1.0)
        l_meas = ((1 - (out["meas_vec"] * batch["meas_vec"]).sum(1, keepdim=True))
                  * ang_mask).sum() / n
        l_fiber = ((1 - (out["fiber_vec"] * batch["fiber_vec"]).sum(1, keepdim=True))
                   * ang_mask).sum() / n
        total = (self.w.validity * l_valid + self.w.width * l_width
                 + self.w.orient * (l_meas + l_fiber))
        return total, {"valid": float(l_valid.detach()), "width": float(l_width.detach()),
                       "meas": float(l_meas.detach()), "fiber": float(l_fiber.detach()),
                       "total": float(total.detach())}
