"""Multi-head fully convolutional network for local fiber measurement.

Why dense local measurement instead of instance segmentation
------------------------------------------------------------
In an electrospun / phase-separated membrane the fibers form a deeply
overlapping 3-D network projected onto 2-D.  A single fiber is occluded and
re-emerges many times, and at a crossing there is genuinely no image evidence
for which strand passes in front.  Asking a model for *instances* therefore
poses a question the data cannot answer: instance identity is not observable,
so the labels would be arbitrary and the metric would measure annotator
convention rather than physics.

Thickness, by contrast, is a *local* property.  At a point on a fiber the
diameter is well defined from the local intensity ridge alone.  Predicting
dense per-pixel quantities (is this a fiber, which way does it run, how thick is
it, is this a place where a reliable measurement can be made) asks only
questions the pixels can answer, and it matches how the manual measurements were
made: a human picks a clean spot and draws one chord.  The centre heatmap then
learns the *human's site-selection policy*, which is exactly the behaviour we
want to reproduce.

Capacity
--------
Two presets, selected by ``NetConfig.preset`` or by naming an encoder directly:

``small``  plain U-Net, ~2 M parameters, trained from scratch.  Appropriate for
           a handful of labelled fields, where a large backbone memorises the
           training image before it learns anything transferable.
``large``  ImageNet-pretrained ResNet-34 encoder, residual + squeeze-excitation
           decoder, ~25 M parameters.  Appropriate once there are on the order
           of 30 labelled fields and several thousand annotations: the extra
           capacity buys a much better ridge/crossing discriminator, and the
           pretrained early filters (edges, bars, ridges) transfer well to SEM
           texture even though the domain is nothing like ImageNet.

Bigger is not automatically better here.  Check the baseline and the validation
curve before assuming the larger preset helps.

Heads
-----
``center``   1ch  logits, Gaussian peak at each measurable site
``segment``  1ch  logits, fiber presence
``orient``   2ch  (cos 2t, sin 2t), pi-periodic fiber direction
``width``    1ch  log fiber thickness in pixels
``validity`` 1ch  logits, is a reliable measurement possible here
``logvar``   1ch  aleatoric uncertainty for the width head
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
def _norm(c: int, kind: str = "group") -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(c)
    if kind == "none":
        return nn.Identity()
    return nn.GroupNorm(min(8, c), c)


class SEBlock(nn.Module):
    """Squeeze-and-excitation: cheap channel attention.

    Useful here because the decoder has to weigh 'is this a ridge' against 'is
    this the dark space between ridges' differently at different scales.
    """

    def __init__(self, c: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(4, c // reduction)
        self.fc = nn.Sequential(nn.Linear(c, hidden), nn.SiLU(inplace=True),
                                nn.Linear(hidden, c), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x.mean(dim=(2, 3)))
        return x * w[:, :, None, None]


class ConvBlock(nn.Module):
    """Two 3x3 convs, optionally residual, optionally SE-gated."""

    def __init__(self, cin: int, cout: int, norm: str = "group", *,
                 residual: bool = False, attention: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.n1 = _norm(cout, norm)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.n2 = _norm(cout, norm)
        self.act = nn.SiLU(inplace=True)
        self.se = SEBlock(cout) if attention else nn.Identity()
        self.residual = residual
        self.skip = (nn.Identity() if (residual and cin == cout)
                     else nn.Conv2d(cin, cout, 1, bias=False) if residual
                     else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.n1(self.conv1(x)))
        y = self.n2(self.conv2(y))
        y = self.se(y)
        if self.skip is not None:
            y = y + self.skip(x)
        return self.act(y)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass
class NetConfig:
    in_channels: int = 1
    base: int = 32
    depth: int = 4
    norm: str = "group"
    encoder: str = "unet"          # "unet" | "timm:<model_name>"
    pretrained: bool = False
    dropout: float = 0.0           # > 0 enables MC-dropout at inference
    residual: bool = False
    attention: bool = False
    decoder_channels: tuple[int, ...] | None = None
    preset: str | None = None      # "small" | "large" -> fills the fields above


PRESETS: dict[str, dict[str, Any]] = {
    "small": {"encoder": "unet", "base": 32, "depth": 4, "norm": "group",
              "pretrained": False, "residual": False, "attention": False},
    "large": {"encoder": "timm:resnet34", "pretrained": True, "norm": "batch",
              "residual": True, "attention": True, "base": 32,
              "decoder_channels": (256, 128, 64, 48, 32)},
}


def apply_preset(cfg: NetConfig) -> NetConfig:
    if not cfg.preset:
        return cfg
    if cfg.preset not in PRESETS:
        raise ValueError(f"unknown preset {cfg.preset!r}; choose from {list(PRESETS)}")
    merged = {**PRESETS[cfg.preset]}
    merged["preset"] = cfg.preset
    merged["in_channels"] = cfg.in_channels
    merged["dropout"] = cfg.dropout
    return NetConfig(**merged)


# --------------------------------------------------------------------------- #
# encoders
# --------------------------------------------------------------------------- #
class UNetEncoder(nn.Module):
    """Plain U-Net encoder; first feature map is at full resolution."""

    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        chans = [cfg.base * (2 ** i) for i in range(cfg.depth + 1)]
        self.stages = nn.ModuleList()
        cin = cfg.in_channels
        for c in chans:
            self.stages.append(ConvBlock(cin, c, cfg.norm, residual=cfg.residual,
                                         attention=cfg.attention))
            cin = c
        self.pool = nn.MaxPool2d(2)
        self.out_channels = list(chans)
        self.reductions = [2 ** i for i in range(cfg.depth + 1)]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.pool(x)
            x = stage(x)
            feats.append(x)
        return feats


class TimmEncoder(nn.Module):
    """Pretrained backbone via ``timm.create_model(features_only=True)``.

    ``in_chans=1`` makes timm fold the pretrained RGB stem weights into a single
    grayscale channel, which preserves the learned edge and bar detectors rather
    than discarding them -- the reason a pretrained encoder is worth anything on
    single-channel SEM data.

    Note these features start at stride 2, so the decoder adds a full-resolution
    stem skip to get back to input size.
    """

    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "encoder='timm:...' needs the timm package (pip install timm)"
            ) from exc
        name = cfg.encoder.split(":", 1)[1]
        self.body = timm.create_model(name, pretrained=cfg.pretrained,
                                      features_only=True,
                                      in_chans=cfg.in_channels)
        self.out_channels = list(self.body.feature_info.channels())
        self.reductions = list(self.body.feature_info.reduction())

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(self.body(x))


def build_encoder(cfg: NetConfig) -> nn.Module:
    if cfg.encoder.startswith("timm:"):
        return TimmEncoder(cfg)
    if cfg.encoder == "unet":
        return UNetEncoder(cfg)
    raise ValueError(f"unknown encoder {cfg.encoder!r}")


# --------------------------------------------------------------------------- #
# network
# --------------------------------------------------------------------------- #
class FiberMeasurementNet(nn.Module):
    """Encoder-decoder with six prediction heads at input resolution."""

    def __init__(self, cfg: NetConfig | None = None) -> None:
        super().__init__()
        cfg = apply_preset(cfg or NetConfig())
        self.cfg = cfg
        self.encoder = build_encoder(cfg)
        enc = list(self.encoder.out_channels)
        n_up = len(enc) - 1

        dec = list(cfg.decoder_channels) if cfg.decoder_channels else None
        if dec is None:
            dec = enc[:-1][::-1]                      # mirror the encoder
        if len(dec) < n_up:
            dec = list(dec) + [dec[-1]] * (n_up - len(dec))
        dec = dec[:n_up]

        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        cin = enc[-1]
        for i in range(n_up):
            skip_c = enc[-2 - i]
            out_c = dec[i]
            self.ups.append(nn.Conv2d(cin, out_c, 1, bias=False))
            self.dec.append(ConvBlock(out_c + skip_c, out_c, cfg.norm,
                                      residual=cfg.residual,
                                      attention=cfg.attention))
            cin = out_c

        # the encoder's shallowest feature may be at stride 2 (timm), so keep a
        # full-resolution stem to fuse back in and predict at input resolution
        self.needs_stem = getattr(self.encoder, "reductions", [1])[0] != 1
        head_in = cin
        if self.needs_stem:
            stem_c = max(16, cfg.base // 2)
            self.stem = ConvBlock(cfg.in_channels, stem_c, cfg.norm)
            self.fuse = ConvBlock(cin + stem_c, cfg.base, cfg.norm,
                                  residual=cfg.residual, attention=cfg.attention)
            head_in = cfg.base
        else:
            self.stem = None
            self.fuse = None

        self.drop = (nn.Dropout2d(cfg.dropout) if cfg.dropout > 0 else nn.Identity())
        self.head_center = nn.Conv2d(head_in, 1, 1)
        self.head_segment = nn.Conv2d(head_in, 1, 1)
        self.head_orient = nn.Conv2d(head_in, 2, 1)
        self.head_width = nn.Conv2d(head_in, 1, 1)
        self.head_validity = nn.Conv2d(head_in, 1, 1)
        self.head_logvar = nn.Conv2d(head_in, 1, 1)

        # start the sparse heatmap head strongly negative: with a few hundred
        # positive pixels in a million, a zero-init head spends its first epochs
        # unlearning a 0.5 prior instead of learning where fibers are
        nn.init.constant_(self.head_center.bias, -4.0)
        nn.init.constant_(self.head_validity.bias, -2.0)
        nn.init.constant_(self.head_width.bias, 2.9)      # exp(2.9) ~ 18 px

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.encoder(x)
        y = feats[-1]
        for i, (up, dec) in enumerate(zip(self.ups, self.dec)):
            skip = feats[-2 - i]
            y = up(y)
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
            y = dec(torch.cat([y, skip], dim=1))
        if self.needs_stem:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear",
                              align_corners=False)
            y = self.fuse(torch.cat([y, self.stem(x)], dim=1))
        elif y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear",
                              align_corners=False)
        y = self.drop(y)
        orient = self.head_orient(y)
        orient = orient / orient.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return {
            "center_logit": self.head_center(y),
            "segment_logit": self.head_segment(y),
            "orient": orient,
            "width": self.head_width(y),
            "validity_logit": self.head_validity(y),
            "logvar": self.head_logvar(y).clamp(-6.0, 6.0),
        }

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_tiled(self, image: torch.Tensor, *, tile: int = 512,
                      overlap: int = 64, tta: bool = False,
                      mc_samples: int = 0, pad_to: int = 32
                      ) -> dict[str, torch.Tensor]:
        """Memory-bounded inference on an arbitrarily large image.

        Tiles are blended with a Hann window so seams cannot create spurious
        heatmap peaks at tile borders, and each tile is padded up to a multiple
        of the encoder stride so a strided backbone gets a legal input size.
        """
        self.eval()
        b, c, h, w = image.shape
        assert b == 1, "predict_tiled expects a single image"
        tile = min(tile, max(h, w))
        step = max(1, tile - overlap)
        device = image.device
        acc: dict[str, torch.Tensor] | None = None
        weight = torch.zeros((1, 1, h, w), device=device)

        win1 = torch.hann_window(tile, periodic=False, device=device).clamp_min(1e-3)
        win = (win1[:, None] * win1[None, :])[None, None]

        ys = list(range(0, max(1, h - tile + 1), step))
        xs = list(range(0, max(1, w - tile + 1), step))
        if ys[-1] + tile < h:
            ys.append(max(0, h - tile))
        if xs[-1] + tile < w:
            xs.append(max(0, w - tile))

        for y0 in ys:
            for x0 in xs:
                patch = image[:, :, y0:y0 + tile, x0:x0 + tile]
                ph, pw = patch.shape[-2:]
                py = (-ph) % pad_to
                px = (-pw) % pad_to
                if py or px:
                    patch = F.pad(patch, (0, px, 0, py), mode="reflect")
                out = self._forward_ensemble(patch, tta=tta, mc_samples=mc_samples)
                out = {k: v[:, :, :ph, :pw] for k, v in out.items()}
                wgt = win[:, :, :ph, :pw]
                if acc is None:
                    acc = {k: torch.zeros((1, v.shape[1], h, w), device=device)
                           for k, v in out.items()}
                for k, v in out.items():
                    acc[k][:, :, y0:y0 + ph, x0:x0 + pw] += v * wgt
                weight[:, :, y0:y0 + ph, x0:x0 + pw] += wgt

        assert acc is not None
        weight = weight.clamp_min(1e-6)
        return {k: v / weight for k, v in acc.items()}

    def _forward_ensemble(self, patch: torch.Tensor, *, tta: bool,
                          mc_samples: int) -> dict[str, torch.Tensor]:
        """Average predictions over flips (TTA) and/or dropout samples."""
        variants: list[tuple[torch.Tensor, str | None]] = [(patch, None)]
        if tta:
            variants += [(torch.flip(patch, [-1]), "h"), (torch.flip(patch, [-2]), "v")]

        if mc_samples > 0:
            for m in self.modules():
                if isinstance(m, nn.Dropout2d):
                    m.train()
        n_draws = max(1, mc_samples)

        sums: dict[str, torch.Tensor] = {}
        count = 0
        for _ in range(n_draws):
            for x, flip in variants:
                out = self.forward(x)
                out = {k: self._unflip(k, v, flip) for k, v in out.items()}
                for k, v in out.items():
                    sums[k] = v if k not in sums else sums[k] + v
                count += 1
        return {k: v / count for k, v in sums.items()}

    @staticmethod
    def _unflip(key: str, v: torch.Tensor, flip: str | None) -> torch.Tensor:
        if flip is None:
            return v
        v = torch.flip(v, [-1] if flip == "h" else [-2])
        if key == "orient":
            # (cos2t, sin2t) under a mirror: t -> -t, so sin2t changes sign
            v = v.clone()
            v[:, 1] = -v[:, 1]
        return v


def build_model(cfg: dict[str, Any] | NetConfig | None = None) -> FiberMeasurementNet:
    if isinstance(cfg, NetConfig) or cfg is None:
        return FiberMeasurementNet(cfg)
    known = set(NetConfig.__dataclass_fields__)
    kwargs = {k: v for k, v in cfg.items() if k in known}
    if isinstance(kwargs.get("decoder_channels"), list):
        kwargs["decoder_channels"] = tuple(kwargs["decoder_channels"])
    return FiberMeasurementNet(NetConfig(**kwargs))
