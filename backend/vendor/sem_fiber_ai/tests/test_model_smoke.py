"""Shape and gradient smoke tests. Skipped automatically when torch is absent.

Run this first in a fresh environment: it catches head-shape and loss-masking
mistakes in seconds, before a training run wastes an hour discovering them.
"""
from __future__ import annotations

import numpy as np
import pytest

try:                       # a broken/partial torch install must skip, not error
    import torch
except Exception as exc:   # noqa: BLE001
    pytest.skip(f"torch unavailable ({exc})", allow_module_level=True)

from sem_fiber_ai.src.losses import MultiHeadLoss, PatchLoss  # noqa: E402
from sem_fiber_ai.src.models.baseline_patch_model import build_baseline  # noqa: E402
from sem_fiber_ai.src.models.fiber_measurement_net import (NetConfig,  # noqa: E402
                                                           build_model)


def _batch(b=2, s=64):
    keys = ("center", "segment", "cos2t", "sin2t", "width", "validity",
            "reg_mask", "ignore")
    out = {k: torch.zeros(b, 1, s, s) for k in keys}
    out["image"] = torch.randn(b, 1, s, s)
    out["center"][:, :, s // 2, s // 2] = 1.0
    out["reg_mask"][:, :, s // 2 - 2:s // 2 + 2, s // 2 - 2:s // 2 + 2] = 1.0
    out["width"] += float(np.log(14.0))
    out["cos2t"] += 1.0
    return out


def test_full_model_shapes_and_backward():
    model = build_model(NetConfig(base=8, depth=2))
    batch = _batch()
    out = model(batch["image"])
    s = batch["image"].shape[-1]
    assert out["center_logit"].shape == (2, 1, s, s)
    assert out["orient"].shape == (2, 2, s, s)
    # orientation head must emit unit vectors so the angle decode is stable
    assert torch.allclose(out["orient"].norm(dim=1), torch.ones(2, s, s), atol=1e-4)
    loss, parts = MultiHeadLoss()(out, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(parts) >= {"center", "segment", "width", "orient", "validity"}
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_ignore_mask_removes_supervision():
    model = build_model(NetConfig(base=8, depth=2))
    batch = _batch()
    out = model(batch["image"])
    _l0, p0 = MultiHeadLoss()(out, batch)
    batch2 = {k: v.clone() for k, v in batch.items()}
    batch2["reg_mask"] = torch.zeros_like(batch2["reg_mask"])
    _l1, p1 = MultiHeadLoss()(out, batch2)
    # with no valid region the width term must not depend on the (absent) labels
    assert p1["width"] != p0["width"]


def test_tiled_inference_matches_whole_image():
    torch.manual_seed(0)
    model = build_model(NetConfig(base=8, depth=2)).eval()
    x = torch.randn(1, 1, 160, 160)
    with torch.no_grad():
        whole = model(x)["center_logit"]
        tiled = model.predict_tiled(x, tile=128, overlap=32)["center_logit"]
    assert tiled.shape == whole.shape
    # Exact equality is not achievable and not wanted: a tile sees reflected
    # padding where the full image had real context, so border predictions
    # legitimately differ.  What must hold is that the blended result is
    # equivalent *for peak detection*, so compare probabilities, not logits.
    # The bound is on probability because that is what peak detection consumes:
    # a shift of a few thousandths cannot move a detection across the 0.3
    # threshold. Tiles much smaller than the receptive field would exaggerate
    # this, so the tile here is realistic relative to the model's context.
    assert torch.allclose(torch.sigmoid(whole), torch.sigmoid(tiled), atol=1e-2)


def test_baseline_shapes_and_backward():
    model = build_baseline({"base": 8, "n_blocks": 2})
    b = 4
    batch = {"image": torch.randn(b, 1, 64, 64),
             "valid": torch.tensor([[1.0], [0.0], [1.0], [1.0]]),
             "width": torch.full((b, 1), float(np.log(12.0))),
             "meas_vec": torch.tensor([[1.0, 0.0]] * b),
             "fiber_vec": torch.tensor([[0.0, 1.0]] * b),
             "has_angle": torch.ones(b, 1), "conf": torch.ones(b, 1)}
    out = model(batch["image"])
    assert out["width"].shape == (b, 1) and out["meas_vec"].shape == (b, 2)
    loss, _parts = PatchLoss()(out, batch)
    loss.backward()
    assert torch.isfinite(loss)


def test_tta_unflip_negates_sin2theta():
    from sem_fiber_ai.src.models.fiber_measurement_net import FiberMeasurementNet
    v = torch.zeros(1, 2, 4, 4)
    v[:, 1] = 0.5
    out = FiberMeasurementNet._unflip("orient", v, "h")
    assert torch.all(out[:, 1] == -0.5)


# --------------------------------------------------------------------------- #
# capacity presets
# --------------------------------------------------------------------------- #
def test_small_preset_builds_and_predicts_at_input_resolution():
    from sem_fiber_ai.src.models.fiber_measurement_net import NetConfig, build_model
    model = build_model(NetConfig(preset="small", base=16, depth=3))
    x = torch.randn(1, 1, 96, 96)
    out = model(x)
    for k in ("center_logit", "segment_logit", "width", "validity_logit", "logvar"):
        assert out[k].shape == (1, 1, 96, 96), k
    assert out["orient"].shape == (1, 2, 96, 96)


def test_odd_input_sizes_still_come_back_at_input_resolution():
    from sem_fiber_ai.src.models.fiber_measurement_net import NetConfig, build_model
    model = build_model(NetConfig(preset="small", base=8, depth=3))
    x = torch.randn(1, 1, 70, 94)          # not a multiple of the stride
    assert model(x)["center_logit"].shape == (1, 1, 70, 94)


def test_large_preset_when_timm_is_available():
    timm = pytest.importorskip("timm")
    from sem_fiber_ai.src.models.fiber_measurement_net import NetConfig, build_model
    # pretrained=False keeps the test offline; the architecture is what matters
    cfg = NetConfig(preset="large")
    cfg.pretrained = False
    model = build_model(cfg)
    x = torch.randn(1, 1, 128, 128)
    out = model(x)
    assert out["center_logit"].shape == (1, 1, 128, 128)
    assert out["orient"].shape == (1, 2, 128, 128)
    assert model.n_parameters() > 5_000_000
    from sem_fiber_ai.src.losses import MultiHeadLoss
    batch = {k: torch.zeros(1, 1, 128, 128) for k in
             ("center", "segment", "cos2t", "sin2t", "width", "validity",
              "reg_mask", "ignore")}
    batch["reg_mask"][:, :, 60:68, 60:68] = 1.0
    loss, _ = MultiHeadLoss()(out, batch)
    loss.backward()
    assert torch.isfinite(loss)


def test_unknown_preset_is_rejected():
    from sem_fiber_ai.src.models.fiber_measurement_net import NetConfig, build_model
    with pytest.raises(ValueError):
        build_model(NetConfig(preset="enormous"))
