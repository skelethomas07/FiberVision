"""Model heads and tile invariance, exact checkpoint/resume, hardware + protocol policy."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sem_fiber_ai.src import checkpoint as CK
from sem_fiber_ai.src import hardware as HW
from sem_fiber_ai.src import train as T
from sem_fiber_ai.src.models.fiber_net import PRESETS, build_model

PKG = Path(__file__).resolve().parents[1]


def _tiny():
    torch.manual_seed(0)
    return build_model({"preset": "tiny"}).eval()


def test_presets_build_and_heads_have_expected_shapes_and_ranges():
    for preset in ("tiny", "small"):
        m = build_model({"preset": preset}).eval()
        x = torch.rand(1, 1, 128, 160)
        with torch.no_grad():
            out = m(x)
        for k in ("center_logit", "segment_logit", "validity_logit", "width", "logvar", "dist"):
            assert out[k].shape == (1, 1, 128, 160), (preset, k, out[k].shape)
        assert out["orient"].shape == (1, 2, 128, 160)
        assert torch.all(out["dist"] >= 0), "distance head must be non-negative (softplus)"
    assert set(PRESETS) >= {"tiny", "small", "medium", "large"}
    # BatchNorm is required for tile invariance in eval mode
    assert all(PRESETS[p]["norm"] == "batch" for p in PRESETS)


def test_whole_image_and_tiled_inference_agree():
    m = _tiny()
    torch.manual_seed(1)
    img = torch.rand(1, 1, 256, 320)
    with torch.no_grad():
        whole = m(img)
        tiled_a = m.predict_tiled(img, tile=128, overlap=32, tile_batch=2)
        tiled_b = m.predict_tiled(img, tile=192, overlap=48, tile_batch=3)
    for k in ("segment_logit", "dist", "width", "orient"):
        a, b, w = tiled_a[k].float(), tiled_b[k].float(), whole[k].float()
        assert a.shape == w.shape
        d_ab = (a - b).abs().flatten()
        d_aw = (a - w).abs().flatten()
        scale = float(w.abs().mean()) + 1e-6
        assert float(torch.quantile(d_ab, 0.99)) < 0.05 * scale + 1e-3, (k, float(torch.quantile(d_ab, 0.99)))
        assert float(torch.quantile(d_aw, 0.99)) < 0.10 * scale + 1e-3, (k, float(torch.quantile(d_aw, 0.99)))


def test_tiled_inference_handles_images_smaller_than_tile_and_odd_sizes():
    m = _tiny()
    img = torch.rand(1, 1, 93, 141)
    with torch.no_grad():
        out = m.predict_tiled(img, tile=512, overlap=64)
    assert out["segment_logit"].shape == (1, 1, 93, 141)


def test_checkpoint_round_trip_restores_every_state(tmp_path):
    m = _tiny()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    x = torch.rand(2, 1, 64, 64)
    for _ in range(3):                      # give optimizer/scheduler a non-trivial state
        opt.zero_grad()
        m.train()(x)["segment_logit"].mean().backward()
        opt.step()
        sched.step()
    torch.manual_seed(123)
    np.random.seed(123)
    _ = torch.rand(3)                       # advance RNGs to a non-initial state
    p = CK.save_checkpoint(tmp_path / "last.pt", model=m, optimizer=opt, scheduler=sched,
                           scaler=scaler, epoch=3, best=0.5, best_epoch=2,
                           history={"train_loss": [1, 0.9, 0.8]}, config={"a": 1},
                           split_manifest={"train": ["x"]}, protocol_digest="abc")
    expected_next = torch.rand(4).clone()   # what the RNG produces right after saving
    assert not list(tmp_path.glob(".last.pt.*")), "temp file must not survive an atomic save"
    ck = CK.load_checkpoint(p)
    assert ck["epoch"] == 3 and ck["best_epoch"] == 2 and ck["protocol_digest"] == "abc"
    assert ck["split_manifest"] == {"train": ["x"]} and ck["config"] == {"a": 1}
    m2 = _tiny()
    m2.load_state_dict(ck["model"])
    for (n1, p1), (n2, p2) in zip(m.state_dict().items(), m2.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    opt2.load_state_dict(ck["optimizer"])
    assert opt2.state_dict()["param_groups"][0]["lr"] == opt.state_dict()["param_groups"][0]["lr"]
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=10)
    sched2.load_state_dict(ck["scheduler"])
    assert sched2.last_epoch == sched.last_epoch
    from sem_fiber_ai.src.utils import rng_state_restore

    rng_state_restore(ck["rng"])
    assert torch.equal(torch.rand(4), expected_next), "RNG state must resume exactly"
    assert CK.checkpoint_digest(p) == CK.checkpoint_digest(p) and len(CK.checkpoint_digest(p)) == 64


def test_v6_checkpoints_are_rejected(tmp_path):
    torch.save({"model": {}, "epoch": 3}, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="v7"):
        CK.load_checkpoint(tmp_path / "old.pt")


def test_claim_run_dir_refuses_foreign_owner(tmp_path):
    d = CK.claim_run_dir(tmp_path, "run1", user="jake")
    assert (d / "OWNER.json").exists()
    CK.claim_run_dir(tmp_path, "run1", user="jake")            # same owner is fine
    with pytest.raises(RuntimeError, match="owned by"):
        CK.claim_run_dir(tmp_path, "run1", user="someone_else")


def test_sync_tree_is_atomic_and_incremental(tmp_path):
    src, dst = tmp_path / "local", tmp_path / "drive"
    src.mkdir()
    (src / "a.json").write_text("1")
    (src / "sub").mkdir()
    (src / "sub" / "b.txt").write_text("2")
    rep = CK.sync_tree(src, dst)
    assert (dst / "a.json").read_text() == "1" and (dst / "sub" / "b.txt").read_text() == "2"
    rep2 = CK.sync_tree(src, dst)
    assert len(rep2) == 0 and sorted(rep) == ["a.json", "sub/b.txt"]
    assert not list(dst.rglob(".*.tmp*"))


def test_precision_policy_on_cpu_and_gpu_profiles():
    cpu = HW.Hardware(device="cpu")
    assert HW.choose_precision(cpu, "auto") == "fp32"
    assert HW.choose_precision(cpu, "bf16") == "fp32"
    a100 = HW.Hardware(device="cuda", gpu_name="A100", bf16_supported=True, vram_gb=40)
    t4 = HW.Hardware(device="cuda", gpu_name="Tesla T4", bf16_supported=False, vram_gb=15)
    assert HW.choose_precision(a100, "auto") == "bf16"
    assert HW.choose_precision(t4, "auto") == "fp16"
    assert HW.choose_precision(t4, "bf16") == "fp16"     # unsupported request degrades, never crashes
    assert HW.choose_precision(a100, "fp32") == "fp32"
    assert HW.autocast_dtype("fp32") is None and HW.autocast_dtype("bf16") is torch.bfloat16
    for hw in (a100, t4, cpu):
        prof = HW.profile_for(hw)
        assert "micro_batch_candidates" in prof and "infer_tile" in prof
    assert a100.profile == "a100" and t4.profile == "t4" and cpu.profile == "cpu"
    assert HW.profile_for(a100)["micro_batch_candidates"][0] > HW.profile_for(t4)["micro_batch_candidates"][0]


def test_oom_detection_matches_cuda_messages_only():
    assert HW.is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert HW.is_oom(RuntimeError("CUDNN_STATUS_ALLOC_FAILED"))
    assert not HW.is_oom(ValueError("shape mismatch"))
    assert not HW.is_oom(RuntimeError("index out of range"))


def test_try_compile_falls_back_to_eager_when_compiled_output_disagrees_or_fails():
    m = _tiny()
    x = torch.rand(1, 1, 64, 64)
    out, info = HW.try_compile(m, example=x)
    assert isinstance(info, str) and ("compile" in info)
    with torch.no_grad():
        ref = m(x)["segment_logit"]
        got = out(x)["segment_logit"]
    assert torch.allclose(ref, got, atol=1e-2)


def test_full_run_protocol_is_never_reduced_by_hardware():
    cfg = T.load_config(PKG / "config" / "default.yaml")
    full = T.resolve_protocol(cfg, "FULL_RUN")
    smoke = T.resolve_protocol(cfg, "FAST_SMOKE_TEST")
    assert full["epochs"] == cfg["protocol"]["epochs"] and full["epochs"] >= 40
    assert full["tile"] == cfg["protocol"]["tile"] and full["effective_batch"] == cfg["protocol"]["effective_batch"]
    assert smoke["epochs"] <= 3 and smoke["run_mode"] == "FAST_SMOKE_TEST"
    assert full["model"].get("preset") != smoke["model"].get("preset") or full["tile"] != smoke["tile"]
    # digest changes with the protocol and with the split
    assert T.protocol_digest(full, "s1") != T.protocol_digest(full, "s2")
    assert T.protocol_digest(full, "s1") != T.protocol_digest(smoke, "s1")
    # a mutated copy of the same protocol has a different digest
    mut = copy.deepcopy(full)
    mut["epochs"] -= 1
    assert T.protocol_digest(mut, "s1") != T.protocol_digest(full, "s1")


def test_full_run_refuses_to_start_without_cuda(tmp_path):
    cfg = T.load_config(PKG / "config" / "default.yaml")
    hw = HW.Hardware(device="cpu")
    with pytest.raises(RuntimeError, match="CUDA"):
        T.train(cfg, records=[], split={"train": [], "val": [], "test": []}, run_dir=tmp_path,
                run_mode="FULL_RUN", hw=hw)


def test_unknown_run_mode_is_rejected(tmp_path):
    cfg = T.load_config(PKG / "config" / "default.yaml")
    with pytest.raises(ValueError):
        T.train(cfg, records=[], split={}, run_dir=tmp_path, run_mode="QUICK")


def test_single_version_string_everywhere():
    from sem_fiber_ai.src import __version__
    from sem_fiber_ai.src.utils import package_version

    v = (PKG / "VERSION").read_text().strip()
    assert v == package_version() == __version__ == "7.0.0"
    txt = json.dumps({"v": v})
    assert "6." not in v and txt
