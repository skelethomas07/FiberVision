"""Training (v7): the scientific protocol is fixed; hardware only changes speed.

``RUN_MODE``:

``FULL_RUN``          requires CUDA.  Uses the protocol in ``cfg['protocol']``
                      unchanged.  If CUDA is missing it STOPS.
``FAST_SMOKE_TEST``   a separately declared, deliberately tiny protocol
                      (``cfg['smoke_protocol']``) that runs on CPU to prove the
                      code path works.  Its outputs are labelled ``smoke`` and are
                      never scientific results.

Everything hardware-dependent (micro-batch, accumulation, precision, workers)
is decided in :mod:`hardware` and recorded in the manifest; the effective batch
size, tile, epochs, patience, model, targets, loss and split never change.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .augmentations import AugConfig
from .checkpoint import (checkpoint_digest, load_checkpoint, save_checkpoint, sync_tree)
from .dataset import ImageRecord, TileDataset
from .fiber_prior import PriorConfig, audit_prior
from .hardware import (Hardware, autocast_dtype, choose_precision, clear_cuda,
                       cuda_memory_stats, detect, is_oom, probe_micro_batch, profile_for,
                       try_compile)
from .losses import MultiHeadLoss
from .models.fiber_net import build_model
from .specimens import assert_no_leakage
from .targets import TargetConfig, strata_weights_from_widths
from .utils import (LOG, ensure_dir, environment_report, now_iso, package_version,
                    pick_device, rng_state_restore, save_json, set_seed)

PROTOCOL_KEYS = ("tile", "effective_batch", "epochs", "patience", "lr", "weight_decay",
                 "samples_per_image", "val_samples_per_image", "monitor")


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _prior_cfg(cfg: dict[str, Any]) -> PriorConfig:
    raw = dict(cfg.get("prior") or {})
    known = {k: v for k, v in raw.items() if k in PriorConfig.__dataclass_fields__}
    if isinstance(known.get("sigmas"), list):
        known["sigmas"] = tuple(known["sigmas"])
    return PriorConfig(**known)


def _target_cfg(cfg: dict[str, Any], train_widths: np.ndarray | None = None) -> TargetConfig:
    raw = dict(cfg.get("targets") or {})
    known = {k: v for k, v in raw.items() if k in TargetConfig.__dataclass_fields__}
    for k in ("strata_edges", "strata_weights", "ratio_clip"):
        if isinstance(known.get(k), list):
            known[k] = tuple(known[k])
    t = TargetConfig(**known)
    if train_widths is not None and len(train_widths):
        t.strata_weights = strata_weights_from_widths(train_widths, t.strata_edges,
                                                     t.strata_weight_cap)
    return t


def _aug_cfg(cfg: dict[str, Any]) -> AugConfig:
    raw = dict(cfg.get("augment") or {})
    return AugConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items()
                        if k in AugConfig.__dataclass_fields__})


def resolve_protocol(cfg: dict[str, Any], run_mode: str) -> dict[str, Any]:
    key = "smoke_protocol" if run_mode == "FAST_SMOKE_TEST" else "protocol"
    proto = dict(cfg.get(key) or {})
    missing = [k for k in PROTOCOL_KEYS if k not in proto]
    if missing:
        raise ValueError(f"config['{key}'] is missing {missing}")
    proto["run_mode"] = run_mode
    proto["model"] = dict(cfg.get("model") or {})
    proto["targets"] = dict(cfg.get("targets") or {})
    proto["loss_weights"] = dict(cfg.get("loss_weights") or {})
    proto["augment"] = dict(cfg.get("augment") or {})
    proto["prior"] = dict(cfg.get("prior") or {})
    proto["seed"] = int(cfg.get("seed", 1337))
    if run_mode == "FAST_SMOKE_TEST":
        proto["model"] = dict(proto["model"], **(cfg.get("smoke_model") or {}))
    return proto


def protocol_digest(proto: dict[str, Any], split_digest: str = "") -> str:
    blob = json.dumps({"protocol": proto, "split": split_digest}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class _VirtualSampler:
    """Yields ``epoch * n + i`` so tile randomness varies per epoch while workers
    stay persistent; the order is a deterministic function of (seed, epoch)."""

    def __init__(self, n: int, seed: int, epoch: int) -> None:
        self.n, self.seed, self.epoch = int(n), int(seed), int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed * 100_003 + self.epoch)
        base = self.epoch * self.n
        return iter([int(base + i) for i in rng.permutation(self.n)])

    def __len__(self) -> int:
        return self.n


def prior_gate(records: Sequence[ImageRecord], pcfg: PriorConfig, out_dir: Path
               ) -> tuple[list[ImageRecord], dict[str, Any]]:
    """Audit every training/validation prior; exclude failures with a reason."""
    report, keep = {}, []
    for rec in records:
        try:
            rep = audit_prior(rec.prior(), rec.annotations, rec.image(), pcfg)
        except Exception as exc:                          # noqa: BLE001
            rep = {"ok": False, "failures": [f"audit raised {exc!r}"]}
        report[rec.image_id] = rep
        if rep.get("ok", False):
            keep.append(rec)
    save_json(report, out_dir / "prior_audit.json")
    dropped = sorted(set(r.image_id for r in records) - set(r.image_id for r in keep))
    if dropped:
        LOG.warning("prior audit excluded %d field(s) from training/validation: %s",
                    len(dropped), dropped)
    return keep, {"report": report, "excluded": dropped}


# --------------------------------------------------------------------------- #
def train(cfg: dict[str, Any], *, records: Sequence[ImageRecord], split: dict[str, Any],
          run_dir: str | Path, run_mode: str = "FULL_RUN", drive_dir: str | Path | None = None,
          resume: bool = True, hw: Hardware | None = None, user: str = "",
          max_seconds: float | None = None) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    if run_mode not in ("FULL_RUN", "FAST_SMOKE_TEST"):
        raise ValueError("run_mode must be FULL_RUN or FAST_SMOKE_TEST")
    hw = hw or detect()
    if run_mode == "FULL_RUN" and hw.device != "cuda":
        raise RuntimeError(
            "FULL_RUN requires a CUDA GPU. Runtime > Change runtime type > GPU, then re-run. "
            "The protocol is NOT reduced to fit a CPU; use RUN_MODE='FAST_SMOKE_TEST' "
            "to exercise the code path only.")
    device = pick_device("cuda" if hw.device == "cuda" else "cpu")
    proto = resolve_protocol(cfg, run_mode)
    seed = int(proto["seed"])
    set_seed(seed, deterministic=bool(cfg.get("deterministic", True)))
    assert_no_leakage(split)
    run_dir = ensure_dir(run_dir)
    pdig = protocol_digest(proto, split.get("digest", ""))
    prof = profile_for(hw)
    precision = choose_precision(hw, str(cfg.get("hardware", {}).get("precision", "auto")))
    ac_dtype = autocast_dtype(precision)

    by_id = {r.image_id: r for r in records}
    tr = [by_id[i] for i in split["train"] if i in by_id]
    va = [by_id[i] for i in split["val"] if i in by_id]
    if not tr:
        raise RuntimeError("no training records after applying the split")
    if not va:
        raise RuntimeError("validation split is empty; model selection needs held-out groups")
    pcfg = _prior_cfg(cfg)
    tr, gate_tr = prior_gate(tr, pcfg, run_dir)
    va, gate_va = prior_gate(va, pcfg, run_dir / "val_prior_audit_tmp")
    if not tr or not va:
        raise RuntimeError("prior audit left no usable training or validation field")
    train_widths = np.concatenate([r.annotations["width_px"].to_numpy(float) for r in tr])
    tcfg = _target_cfg({**cfg, "targets": proto["targets"]}, train_widths)
    aug = _aug_cfg({**cfg, "augment": proto["augment"]})

    tile = int(proto["tile"])
    eff_bs = int(proto["effective_batch"])
    epochs = int(proto["epochs"])
    patience = int(proto["patience"])
    model = build_model(proto["model"]).to(device)
    criterion = MultiHeadLoss(proto["loss_weights"], mode=tcfg.mode,
                              use_uncertainty=bool(cfg.get("loss", {}).get("uncertainty", True)))
    monitor_criterion = MultiHeadLoss(proto["loss_weights"], mode=tcfg.mode, use_uncertainty=False)

    ds_tr = TileDataset(tr, tile=tile, samples_per_image=int(proto["samples_per_image"]),
                        aug=aug, targets=tcfg, train=True, seed=seed, prior_cfg=pcfg)
    ds_va = TileDataset(va, tile=tile, samples_per_image=int(proto["val_samples_per_image"]),
                        targets=tcfg, train=False, seed=seed + 1, prior_cfg=pcfg)

    # ---- hardware-only decisions -------------------------------------- #
    def _fake_targets(mb, t, dev):
        return {"center": torch.zeros(mb, 1, t, t, device=dev),
                "segment": torch.zeros(mb, 1, t, t, device=dev),
                "cos2t": torch.ones(mb, 1, t, t, device=dev),
                "sin2t": torch.zeros(mb, 1, t, t, device=dev),
                "width": torch.zeros(mb, 1, t, t, device=dev),
                "validity": torch.zeros(mb, 1, t, t, device=dev),
                "validity_mask": torch.ones(mb, 1, t, t, device=dev),
                "reg_mask": torch.ones(mb, 1, t, t, device=dev),
                "ignore": torch.zeros(mb, 1, t, t, device=dev),
                "dist": torch.ones(mb, 1, t, t, device=dev),
                "dist_weight": torch.ones(mb, 1, t, t, device=dev),
                "strata_weight": torch.ones(mb, 1, t, t, device=dev)}

    if device.type == "cuda":
        micro = probe_micro_batch(model, tile=tile, candidates=prof["micro_batch_candidates"],
                                  device=device, precision=precision, effective_batch=eff_bs,
                                  loss_fn=criterion, target_maker=_fake_targets)
    else:
        micro = min(eff_bs, int(prof["micro_batch_candidates"][0]))
        while eff_bs % micro:
            micro -= 1
    accum = eff_bs // micro
    workers = int(prof["num_workers"])
    compile_note = "not requested"
    if bool(cfg.get("hardware", {}).get("compile", False)) and device.type == "cuda":
        model, compile_note = try_compile(model, example=torch.randn(1, 1, tile, tile, device=device))

    opt = torch.optim.AdamW(model.parameters(), lr=float(proto["lr"]),
                            weight_decay=float(proto["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16" and device.type == "cuda"))

    start_epoch, best, best_epoch = 0, float("inf"), -1
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_monitor": [],
                                       "epoch_seconds": [], "lr": []}
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    resumed_from = None
    if resume and last_path.exists():
        ck = load_checkpoint(last_path, map_location=device)
        if ck.get("protocol_digest") != pdig:
            raise RuntimeError(f"last.pt was trained under protocol {ck.get('protocol_digest')} "
                               f"but this run is {pdig}; refusing to resume across protocols. "
                               "Use a new RUN_TAG.")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        if ck.get("scaler") is not None and scaler.is_enabled():
            scaler.load_state_dict(ck["scaler"])
        start_epoch = int(ck["epoch"]) + 1
        best, best_epoch = float(ck["best"]), int(ck["best_epoch"])
        history = ck["history"]
        rng_state_restore(ck.get("rng"))
        resumed_from = {"path": str(last_path), "epoch": int(ck["epoch"]),
                        "digest": checkpoint_digest(last_path)}
        LOG.info("resumed from %s at epoch %d (best %.4f @ %d)", last_path, start_epoch,
                 best, best_epoch + 1)

    loader_kw: dict[str, Any] = {"num_workers": workers, "pin_memory": device.type == "cuda"}
    if workers > 0:
        loader_kw.update(persistent_workers=True, prefetch_factor=4)

    manifest: dict[str, Any] = {
        "package_version": package_version(), "run_mode": run_mode, "user": user,
        "protocol": proto, "protocol_digest": pdig, "split_digest": split.get("digest"),
        "split": {k: split.get(k) for k in ("train", "val", "test", "test_groups", "val_groups")},
        "prior_gate": {"train_excluded": gate_tr["excluded"], "val_excluded": gate_va["excluded"]},
        "strata_weights": list(tcfg.strata_weights), "strata_edges": list(tcfg.strata_edges),
        "hardware": hw.to_dict(), "precision": precision, "micro_batch": micro,
        "grad_accumulation": accum, "effective_batch": eff_bs, "num_workers": workers,
        "compile": compile_note, "environment": environment_report(),
        "started_at": now_iso(), "resumed_from": resumed_from,
        "n_train_images": len(tr), "n_val_images": len(va),
        "n_parameters": int(model.n_parameters() if hasattr(model, "n_parameters")
                            else sum(p.numel() for p in model.parameters())),
    }
    save_json(manifest, run_dir / "run_manifest.json")

    def _to_dev(batch):
        return {k: (v.to(device, non_blocking=True) if hasattr(v, "to") else v)
                for k, v in batch.items()}

    def run_epoch(epoch: int, training: bool, micro_bs: int, accum_steps: int):
        model.train(training)
        ds = ds_tr if training else ds_va
        if training:
            loader = DataLoader(ds, batch_size=micro_bs, sampler=_VirtualSampler(len(ds), seed, epoch),
                                drop_last=len(ds) >= micro_bs, **loader_kw)
        else:
            loader = DataLoader(ds, batch_size=micro_bs, shuffle=False, **loader_kw)
        total, mon_total, n, n_tiles = 0.0, 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        step_in_accum = 0
        for batch in loader:
            batch = _to_dev(batch)
            bs = batch["image"].shape[0]
            with torch.set_grad_enabled(training):
                with torch.autocast(device_type=device.type, dtype=ac_dtype,
                                    enabled=ac_dtype is not None and device.type == "cuda"):
                    out = model(batch["image"])
                loss, _parts = criterion({k: v.float() for k, v in out.items()}, batch)
                if not training:
                    with torch.no_grad():
                        mon, _ = monitor_criterion({k: v.float() for k, v in out.items()}, batch)
                    mon_total += float(mon.detach()) * bs
            if training:
                scaler.scale(loss / accum_steps).backward()
                step_in_accum += 1
                if step_in_accum == accum_steps:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    step_in_accum = 0
            total += float(loss.detach()) * bs
            n += bs
            n_tiles += bs
        if training and step_in_accum:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        return total / max(1, n), mon_total / max(1, n), n_tiles

    t_start = time.time()
    stop_reason = "completed"
    tiles_seen = 0
    oom_events = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        while True:
            try:
                tr_loss, _, n_t = run_epoch(epoch, True, micro, accum)
                break
            except Exception as exc:                        # noqa: BLE001
                if is_oom(exc) and micro > 1:
                    opt.zero_grad(set_to_none=True)
                    clear_cuda()
                    oom_events.append({"epoch": epoch, "micro_batch": micro})
                    micro = max(1, micro // 2)
                    accum = eff_bs // micro
                    LOG.warning("CUDA OOM at epoch %d: micro-batch -> %d, accumulation -> %d "
                                "(tile, model and effective batch unchanged)", epoch + 1, micro, accum)
                    continue
                raise
        va_loss, va_mon, _ = run_epoch(epoch, False, micro, accum)
        sched.step()
        dt = time.time() - t0
        tiles_seen += n_t
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_monitor"].append(va_mon)
        history["epoch_seconds"].append(dt)
        history["lr"].append(float(opt.param_groups[0]["lr"]))
        score = va_mon if proto["monitor"] == "val_monitor" else va_loss
        improved = score < best
        if improved:
            best, best_epoch = score, epoch
        LOG.info("epoch %3d/%d  train=%.4f  val=%.4f  monitor=%.4f  %s (%.1fs, %.1f tiles/s)",
                 epoch + 1, epochs, tr_loss, va_loss, va_mon, "*" if improved else " ",
                 dt, n_t / max(dt, 1e-6))
        ck_kw = dict(model=model, optimizer=opt, scheduler=sched, scaler=scaler, epoch=epoch,
                     best=best, best_epoch=best_epoch, history=history, config=cfg,
                     split_manifest=split, protocol_digest=pdig,
                     extra={"run_mode": run_mode, "micro_batch": micro, "precision": precision})
        save_checkpoint(last_path, **ck_kw)
        if improved:
            save_checkpoint(best_path, **ck_kw)
        if drive_dir is not None:
            try:
                sync_tree(run_dir, drive_dir)
            except Exception as exc:                        # noqa: BLE001
                LOG.warning("Drive sync failed (%s); continuing locally", exc)
        if max_seconds and (time.time() - t_start) > max_seconds:
            stop_reason = f"wall-clock limit {max_seconds/3600:.1f} h reached at epoch {epoch + 1}"
            LOG.warning("%s -- resume this run to finish the protocol", stop_reason)
            break
        if patience > 0 and (epoch - best_epoch) >= patience:
            stop_reason = f"early stop: no improvement for {patience} epochs"
            LOG.info("%s (best %.4f at epoch %d)", stop_reason, best, best_epoch + 1)
            break

    elapsed = time.time() - t_start
    manifest.update({
        "finished_at": now_iso(), "stop_reason": stop_reason, "best_epoch": best_epoch + 1,
        "best_monitor": best, "epochs_run": len(history["train_loss"]),
        "training_seconds": elapsed, "tiles_per_second": tiles_seen / max(elapsed, 1e-6),
        "micro_batch_final": micro, "grad_accumulation_final": accum,
        "oom_events": oom_events, "cuda_memory": cuda_memory_stats(),
        "best_checkpoint": str(best_path), "best_checkpoint_sha256":
            checkpoint_digest(best_path) if best_path.exists() else None,
        "last_checkpoint_sha256": checkpoint_digest(last_path) if last_path.exists() else None,
        "history": history,
        "protocol_complete": stop_reason in ("completed",) or stop_reason.startswith("early stop"),
    })
    save_json(manifest, run_dir / "run_manifest.json")
    if drive_dir is not None:
        try:
            sync_tree(run_dir, drive_dir)
        except Exception as exc:                            # noqa: BLE001
            LOG.warning("final Drive sync failed (%s)", exc)
    return {"best_monitor": best, "best_epoch": best_epoch + 1, "history": history,
            "stop_reason": stop_reason, "manifest": manifest, "best_path": str(best_path),
            "last_path": str(last_path), "target_cfg": tcfg}
