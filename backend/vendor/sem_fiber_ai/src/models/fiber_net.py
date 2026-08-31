"""Multi-head fully convolutional fibre-measurement network (v7).

Heads (all at input resolution):

``center``   1ch logits  -- baseline: Gaussian peak at each annotated site;
                            geometry: the medial-axis ridge of the fibre mask
``segment``  1ch logits  -- fibre presence
``orient``   2ch         -- (cos 2t, sin 2t), pi-periodic RASTER fibre direction
``width``    1ch         -- log fibre width (px) at measurement sites
``validity`` 1ch logits  -- is a reliable measurement possible here
``logvar``   1ch         -- aleatoric (heteroscedastic) variance of ``width``
``dist``     1ch         -- distance to the nearest fibre boundary (px, >= 0);
                            width at a ridge pixel is 2 * dist   [geometry mode]

Normalisation is BatchNorm.  With running statistics at inference the output
at a pixel depends only on its receptive field, so whole-image and tiled
inference agree up to border effects, which the overlap-aware blending
removes.  (GroupNorm computes statistics over the whole tile and is NOT
tile-invariant; that is why it is no longer used.)

The ``dist`` head is present in both modes so a checkpoint's architecture does
not depend on the training mode; in baseline mode its loss weight is 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(c: int, kind: str = "batch") -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(c)
    if kind == "none":
        return nn.Identity()
    return nn.GroupNorm(min(8, c), c)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, norm: str = "batch", *, residual: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.n1 = _norm(cout, norm)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.n2 = _norm(cout, norm)
        self.act = nn.SiLU(inplace=True)
        self.residual = residual
        self.skip = (nn.Identity() if (residual and cin == cout)
                     else nn.Conv2d(cin, cout, 1, bias=False) if residual else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.n1(self.conv1(x)))
        y = self.n2(self.conv2(y))
        if self.skip is not None:
            y = y + self.skip(x)
        return self.act(y)


@dataclass
class NetConfig:
    in_channels: int = 1
    base: int = 32
    depth: int = 4
    norm: str = "batch"
    encoder: str = "unet"           # "unet" | "timm:<name>"
    pretrained: bool = False
    dropout: float = 0.0
    residual: bool = False
    decoder_channels: tuple[int, ...] | None = None
    preset: str | None = None


PRESETS: dict[str, dict[str, Any]] = {
    "tiny": {"encoder": "unet", "base": 8, "depth": 3, "norm": "batch", "residual": False},
    "small": {"encoder": "unet", "base": 32, "depth": 4, "norm": "batch", "residual": False},
    "medium": {"encoder": "unet", "base": 48, "depth": 5, "norm": "batch", "residual": True},
    "large": {"encoder": "timm:resnet34", "pretrained": True, "norm": "batch",
              "residual": True, "base": 32, "decoder_channels": (256, 128, 64, 48, 32)},
}


def apply_preset(cfg: NetConfig) -> NetConfig:
    if not cfg.preset:
        return cfg
    if cfg.preset not in PRESETS:
        raise ValueError(f"unknown preset {cfg.preset!r}; choose from {list(PRESETS)}")
    merged = {**PRESETS[cfg.preset], "preset": cfg.preset,
              "in_channels": cfg.in_channels, "dropout": cfg.dropout}
    return NetConfig(**merged)


class UNetEncoder(nn.Module):
    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        chans = [cfg.base * (2 ** i) for i in range(cfg.depth + 1)]
        self.stages = nn.ModuleList()
        cin = cfg.in_channels
        for c in chans:
            self.stages.append(ConvBlock(cin, c, cfg.norm, residual=cfg.residual))
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
    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        import timm

        name = cfg.encoder.split(":", 1)[1]
        self.body = timm.create_model(name, pretrained=cfg.pretrained, features_only=True,
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


HEAD_KEYS = ("center_logit", "segment_logit", "orient", "width", "validity_logit",
             "logvar", "dist")


class FiberNet(nn.Module):
    def __init__(self, cfg: NetConfig | None = None) -> None:
        super().__init__()
        cfg = apply_preset(cfg or NetConfig())
        self.cfg = cfg
        self.encoder = build_encoder(cfg)
        enc = list(self.encoder.out_channels)
        n_up = len(enc) - 1
        dec = list(cfg.decoder_channels) if cfg.decoder_channels else enc[:-1][::-1]
        if len(dec) < n_up:
            dec = list(dec) + [dec[-1]] * (n_up - len(dec))
        dec = dec[:n_up]
        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        cin = enc[-1]
        for i in range(n_up):
            skip_c = enc[-2 - i]
            self.ups.append(nn.Conv2d(cin, dec[i], 1, bias=False))
            self.dec.append(ConvBlock(dec[i] + skip_c, dec[i], cfg.norm, residual=cfg.residual))
            cin = dec[i]
        self.needs_stem = getattr(self.encoder, "reductions", [1])[0] != 1
        head_in = cin
        if self.needs_stem:
            stem_c = max(16, cfg.base // 2)
            self.stem = ConvBlock(cfg.in_channels, stem_c, cfg.norm)
            self.fuse = ConvBlock(cin + stem_c, cfg.base, cfg.norm, residual=cfg.residual)
            head_in = cfg.base
        else:
            self.stem = None
            self.fuse = None
        self.drop = nn.Dropout2d(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.head_center = nn.Conv2d(head_in, 1, 1)
        self.head_segment = nn.Conv2d(head_in, 1, 1)
        self.head_orient = nn.Conv2d(head_in, 2, 1)
        self.head_width = nn.Conv2d(head_in, 1, 1)
        self.head_validity = nn.Conv2d(head_in, 1, 1)
        self.head_logvar = nn.Conv2d(head_in, 1, 1)
        self.head_dist = nn.Conv2d(head_in, 1, 1)
        nn.init.constant_(self.head_center.bias, -3.0)
        nn.init.constant_(self.head_validity.bias, -1.0)
        nn.init.constant_(self.head_width.bias, 2.3)      # exp(2.3) ~ 10 px
        nn.init.constant_(self.head_dist.bias, 1.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.encoder(x)
        y = feats[-1]
        for i, (up, dec) in enumerate(zip(self.ups, self.dec)):
            skip = feats[-2 - i]
            y = up(y)
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = dec(torch.cat([y, skip], dim=1))
        if self.needs_stem:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
            y = self.fuse(torch.cat([y, self.stem(x)], dim=1))
        elif y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
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
            "dist": F.softplus(self.head_dist(y)),
        }

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def stride(self) -> int:
        return int(max(getattr(self.encoder, "reductions", [1])))

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_tiled(self, image: torch.Tensor, *, tile: int = 512, overlap: int = 96,
                      tile_batch: int = 4, tta: bool = False,
                      autocast_dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
        """Overlap-aware tiled inference with cosine-window blending.

        ``tile`` is the maximum tile size (chosen from VRAM); the tiling grid is
        built so every tile is a legal size for the encoder stride, and tiles are
        processed ``tile_batch`` at a time.  Only the tile is a hardware choice;
        the blending makes the result invariant to it within numerical
        tolerance (tested by ``tests/test_model.py``).
        """
        self.eval()
        b, c, h, w = image.shape
        assert b == 1, "predict_tiled expects a single image"
        stride = self.stride
        tile = int(tile)
        tile = max(stride * 4, min(tile, max(h, w)))
        tile = int((tile + stride - 1) // stride * stride)
        if tile >= h and tile >= w:
            th, tw = h, w
        else:
            th, tw = min(tile, h), min(tile, w)
        pad_h, pad_w = (-h) % stride, (-w) % stride
        img = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect") if (pad_h or pad_w) else image
        H, W = img.shape[-2:]
        th = min(th + (-th) % stride, H)
        tw = min(tw + (-tw) % stride, W)
        step_y = max(1, th - overlap)
        step_x = max(1, tw - overlap)
        ys = list(range(0, max(1, H - th + 1), step_y))
        xs = list(range(0, max(1, W - tw + 1), step_x))
        if ys[-1] + th < H:
            ys.append(H - th)
        if xs[-1] + tw < W:
            xs.append(W - tw)
        device = image.device
        wy = _blend_window(th, overlap, device)
        wx = _blend_window(tw, overlap, device)
        win = (wy[:, None] * wx[None, :])[None, None]
        acc: dict[str, torch.Tensor] | None = None
        weight = torch.zeros((1, 1, H, W), device=device, dtype=torch.float32)
        coords = [(y0, x0) for y0 in ys for x0 in xs]
        for i in range(0, len(coords), max(1, tile_batch)):
            chunk = coords[i:i + max(1, tile_batch)]
            patches = torch.cat([img[:, :, y0:y0 + th, x0:x0 + tw] for y0, x0 in chunk], 0)
            out = self._forward_ensemble(patches, tta=tta, autocast_dtype=autocast_dtype)
            if acc is None:
                acc = {k: torch.zeros((1, v.shape[1], H, W), device=device, dtype=torch.float32)
                       for k, v in out.items()}
            for j, (y0, x0) in enumerate(chunk):
                for k, v in out.items():
                    acc[k][:, :, y0:y0 + th, x0:x0 + tw] += v[j:j + 1].float() * win
                weight[:, :, y0:y0 + th, x0:x0 + tw] += win
        assert acc is not None
        weight = weight.clamp_min(1e-6)
        res = {k: (v / weight)[:, :, :h, :w] for k, v in acc.items()}
        # renormalise the orientation vector after blending
        res["orient"] = res["orient"] / res["orient"].norm(dim=1, keepdim=True).clamp_min(1e-6)
        return res

    def _forward_ensemble(self, patch: torch.Tensor, *, tta: bool,
                          autocast_dtype: torch.dtype | None) -> dict[str, torch.Tensor]:
        variants = [(patch, None)]
        if tta:
            variants += [(torch.flip(patch, [-1]), "h"), (torch.flip(patch, [-2]), "v")]
        sums: dict[str, torch.Tensor] = {}
        for x, flip in variants:
            if autocast_dtype is not None and x.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    out = self.forward(x)
            else:
                out = self.forward(x)
            out = {k: self._unflip(k, v.float(), flip) for k, v in out.items()}
            for k, v in out.items():
                sums[k] = v if k not in sums else sums[k] + v
        return {k: v / len(variants) for k, v in sums.items()}

    @staticmethod
    def _unflip(key: str, v: torch.Tensor, flip: str | None) -> torch.Tensor:
        if flip is None:
            return v
        v = torch.flip(v, [-1] if flip == "h" else [-2])
        if key == "orient":          # mirror: t -> -t, sin 2t changes sign
            v = v.clone()
            v[:, 1] = -v[:, 1]
        return v


def _blend_window(n: int, overlap: int, device) -> torch.Tensor:
    """Raised-cosine ramps over the overlap zone, flat 1 in the interior."""
    w = torch.ones(n, device=device, dtype=torch.float32)
    r = int(min(max(overlap, 0), n // 2))
    if r > 0:
        ramp = 0.5 - 0.5 * torch.cos(torch.linspace(0, torch.pi, r, device=device))
        w[:r] = torch.minimum(w[:r], ramp)
        w[n - r:] = torch.minimum(w[n - r:], ramp.flip(0))
    return w.clamp_min(1e-3)


def build_model(cfg: dict[str, Any] | NetConfig | None = None) -> FiberNet:
    if isinstance(cfg, NetConfig) or cfg is None:
        return FiberNet(cfg)
    known = set(NetConfig.__dataclass_fields__)
    kwargs = {k: v for k, v in cfg.items() if k in known}
    if isinstance(kwargs.get("decoder_channels"), list):
        kwargs["decoder_channels"] = tuple(kwargs["decoder_channels"])
    return FiberNet(NetConfig(**kwargs))
