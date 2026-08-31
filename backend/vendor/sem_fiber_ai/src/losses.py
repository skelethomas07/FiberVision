"""Mask-aware losses, one per head (v7).

Every regression term is evaluated strictly inside its weight map, and the
``ignore`` map removes invalid pixels from all terms.  The distance loss uses
``dist_weight`` -- 1 near a manual measurement on the same branch, decaying to
``unverified_weight`` elsewhere on the mask, 0 off the mask -- multiplied by
the width-stratum weight so thick fibres are not drowned out.
"""
from __future__ import annotations

from dataclasses import dataclass
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
    validity: float = 0.3
    dist: float = 1.0


def focal_heatmap_loss(logit, target, ignore=None, alpha: float = 2.0, beta: float = 4.0):
    pred = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = target.ge(0.999).float()
    neg = 1.0 - pos
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, alpha) * pos
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, alpha) * torch.pow(1 - target, beta) * neg
    if ignore is not None:
        keep = 1.0 - ignore
        pos_loss, neg_loss, pos, neg = pos_loss * keep, neg_loss * keep, pos * keep, neg * keep
    n_pos = pos.sum().clamp_min(1.0)
    n_neg = neg.sum().clamp_min(1.0)
    return pos_loss.sum() / n_pos + neg_loss.sum() / n_neg


def dice_bce_loss(logit, target, ignore=None, eps: float = 1.0):
    keep = (1.0 - ignore) if ignore is not None else torch.ones_like(target)
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    bce = (bce * keep).sum() / keep.sum().clamp_min(1.0)
    p = torch.sigmoid(logit) * keep
    t = target * keep
    inter = (p * t).sum()
    dice = 1.0 - (2 * inter + eps) / (p.sum() + t.sum() + eps)
    return bce + dice


def heteroscedastic_loss(pred, target, logvar, weight):
    n = weight.sum().clamp_min(1.0)
    logvar = logvar.clamp(-6.0, 6.0)
    nll = 0.5 * (torch.exp(-logvar) * (pred - target) ** 2 + logvar)
    return (nll * weight).sum() / n


def orientation_loss(pred_vec, cos2t, sin2t, weight):
    target = torch.cat([cos2t, sin2t], dim=1)
    target = target / target.norm(dim=1, keepdim=True).clamp_min(1e-6)
    cos_sim = (pred_vec * target).sum(dim=1, keepdim=True)
    n = weight.sum().clamp_min(1.0)
    return ((1.0 - cos_sim) * weight).sum() / n


def distance_loss(pred, target, weight, *, log_space: bool = True):
    """Weighted L1 on the boundary distance; log1p space makes errors relative."""
    n = weight.sum().clamp_min(1.0)
    if log_space:
        diff = torch.log1p(pred.clamp_min(0)) - torch.log1p(target.clamp_min(0))
    else:
        diff = pred - target
    return (diff.abs() * weight).sum() / n


class MultiHeadLoss(nn.Module):
    def __init__(self, weights: LossWeights | dict[str, float] | None = None, *,
                 mode: str = "geometry", use_uncertainty: bool = True) -> None:
        super().__init__()
        if isinstance(weights, dict):
            known = set(LossWeights.__dataclass_fields__)
            weights = LossWeights(**{k: v for k, v in weights.items() if k in known})
        self.w = weights or LossWeights()
        self.mode = mode
        self.use_uncertainty = use_uncertainty

    def forward(self, out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
                ) -> tuple[torch.Tensor, dict[str, float]]:
        ignore = batch["ignore"]
        strata = batch.get("strata_weight", torch.ones_like(ignore))
        reg = batch["reg_mask"] * strata
        l_center = focal_heatmap_loss(out["center_logit"], batch["center"], ignore)
        l_seg = dice_bce_loss(out["segment_logit"], batch["segment"], ignore)
        if self.use_uncertainty:
            l_width = heteroscedastic_loss(out["width"], batch["width"], out["logvar"], reg)
        else:
            n = reg.sum().clamp_min(1.0)
            l_width = (F.huber_loss(out["width"], batch["width"], reduction="none") * reg).sum() / n
        l_orient = orientation_loss(out["orient"], batch["cos2t"], batch["sin2t"], batch["reg_mask"])
        vmask = batch.get("validity_mask", 1.0 - ignore) * (1.0 - ignore)
        val_bce = F.binary_cross_entropy_with_logits(
            out["validity_logit"], (batch["validity"] > 0.5).float(), reduction="none")
        l_valid = (val_bce * vmask).sum() / vmask.sum().clamp_min(1.0)
        parts = {"center": float(l_center.detach()), "segment": float(l_seg.detach()),
                 "width": float(l_width.detach()), "orient": float(l_orient.detach()),
                 "validity": float(l_valid.detach())}
        total = (self.w.center * l_center + self.w.segment * l_seg + self.w.width * l_width
                 + self.w.orient * l_orient + self.w.validity * l_valid)
        if self.mode == "geometry" and self.w.dist > 0:
            dw = batch["dist_weight"] * strata
            l_dist = distance_loss(out["dist"], batch["dist"], dw)
            total = total + self.w.dist * l_dist
            parts["dist"] = float(l_dist.detach())
        parts["total"] = float(total.detach())
        return total, parts
