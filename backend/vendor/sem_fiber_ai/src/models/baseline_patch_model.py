"""Patch-level baseline: the sanity check the main model must beat.

Given a patch centred on a candidate site, predict
    * whether a valid measurement can be made there,
    * the local fiber orientation and the measurement-line orientation,
    * the fiber width,
    * the aleatoric uncertainty of that width.

It cannot *find* measurement sites -- that is the whole point.  If this model
regresses width well but the full-image model does not, the problem is
detection; if this model also fails, the problem is the labels or the features,
and no architecture change will rescue it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class PatchNetConfig:
    in_channels: int = 1
    base: int = 24
    n_blocks: int = 4
    dropout: float = 0.1
    patch: int = 64


class PatchMeasurementNet(nn.Module):
    def __init__(self, cfg: PatchNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PatchNetConfig()
        c = self.cfg.base
        layers: list[nn.Module] = []
        cin = self.cfg.in_channels
        for i in range(self.cfg.n_blocks):
            cout = c * (2 ** i)
            layers += [nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                       nn.GroupNorm(min(8, cout), cout), nn.SiLU(inplace=True),
                       nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                       nn.GroupNorm(min(8, cout), cout), nn.SiLU(inplace=True),
                       nn.MaxPool2d(2)]
            cin = cout
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(self.cfg.dropout)
        self.fc = nn.Sequential(nn.Linear(cin, 128), nn.SiLU(inplace=True))
        self.head_valid = nn.Linear(128, 1)
        self.head_width = nn.Linear(128, 1)
        self.head_logvar = nn.Linear(128, 1)
        self.head_meas = nn.Linear(128, 2)
        self.head_fiber = nn.Linear(128, 2)
        nn.init.constant_(self.head_width.bias, 2.9)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.fc(self.drop(self.pool(self.body(x)).flatten(1)))
        meas = self.head_meas(z)
        fiber = self.head_fiber(z)
        return {
            "valid_logit": self.head_valid(z),
            "width": self.head_width(z),
            "logvar": self.head_logvar(z).clamp(-6.0, 6.0),
            "meas_vec": meas / meas.norm(dim=1, keepdim=True).clamp_min(1e-6),
            "fiber_vec": fiber / fiber.norm(dim=1, keepdim=True).clamp_min(1e-6),
        }


def build_baseline(cfg: dict[str, Any] | None = None) -> PatchMeasurementNet:
    if not cfg:
        return PatchMeasurementNet()
    known = {f for f in PatchNetConfig.__dataclass_fields__}
    return PatchMeasurementNet(PatchNetConfig(**{k: v for k, v in cfg.items()
                                                 if k in known}))
