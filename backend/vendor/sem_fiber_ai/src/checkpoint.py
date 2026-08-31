"""Resumable checkpoints and atomic synchronisation to Google Drive (v7).

A checkpoint holds EVERYTHING needed to continue exactly: model, optimizer,
scheduler, GradScaler, epoch, best score/epoch, history, every RNG state, the
resolved config, the split manifest, the protocol digest and the package
version.  Files are written to a temporary name and renamed, so a Colab
disconnect can never leave a half-written ``last.pt``.

Two users sharing one Drive folder cannot overwrite each other: every run
lives under ``<drive_root>/runs/<run_id>/`` where ``run_id`` includes the user
tag, the timestamp and the protocol digest, and a lock file records who owns it.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .utils import (LOG, ensure_dir, now_iso, package_version, rng_state_snapshot,
                    save_json, sha256_file)


def save_checkpoint(path: str | Path, *, model, optimizer, scheduler, scaler, epoch: int,
                    best: float, best_epoch: int, history: dict[str, Any], config: dict[str, Any],
                    split_manifest: dict[str, Any], protocol_digest: str,
                    extra: dict[str, Any] | None = None) -> Path:
    import torch

    state = {
        "format": "sem_fiber_ai_checkpoint_v7",
        "package_version": package_version(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch), "best": float(best), "best_epoch": int(best_epoch),
        "history": history, "config": config, "split_manifest": split_manifest,
        "protocol_digest": protocol_digest, "rng": rng_state_snapshot(),
        "saved_at": now_iso(), **(extra or {}),
    }
    p = Path(path)
    ensure_dir(p.parent)
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", dir=str(p.parent))
    os.close(fd)
    try:
        torch.save(state, tmp)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return p


def load_checkpoint(path: str | Path, map_location="cpu") -> dict[str, Any]:
    import torch

    ck = torch.load(str(path), map_location=map_location, weights_only=False)
    if ck.get("format") != "sem_fiber_ai_checkpoint_v7":
        raise ValueError(f"{path} is not a v7 checkpoint (format={ck.get('format')!r}); "
                         "v6 checkpoints are not compatible with the repaired protocol")
    return ck


def checkpoint_digest(path: str | Path) -> str:
    return sha256_file(path)


def atomic_copy(src: str | Path, dst: str | Path) -> Path:
    src, dst = Path(src), Path(dst)
    ensure_dir(dst.parent)
    fd, tmp = tempfile.mkstemp(prefix=f".{dst.name}.", dir=str(dst.parent))
    os.close(fd)
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return dst


def sync_tree(local_dir: str | Path, remote_dir: str | Path, *, patterns=("*",),
              skip_dirs=("prior_cache", "_tmp")) -> list[str]:
    """Copy new/changed files from the local run dir to Drive, atomically each."""
    local_dir, remote_dir = Path(local_dir), Path(remote_dir)
    if local_dir.resolve() == remote_dir.resolve():
        return []
    copied = []
    for p in sorted(local_dir.rglob("*")):
        if not p.is_file() or any(s in p.relative_to(local_dir).parts for s in skip_dirs):
            continue
        if p.name.startswith("."):
            continue
        rel = p.relative_to(local_dir)
        dst = remote_dir / rel
        if dst.exists() and dst.stat().st_size == p.stat().st_size \
                and abs(dst.stat().st_mtime - p.stat().st_mtime) < 1.0:
            continue
        atomic_copy(p, dst)
        copied.append(str(rel))
    return copied


def claim_run_dir(root: str | Path, run_id: str, *, user: str) -> Path:
    """Create ``<root>/<run_id>`` and record the owner; refuse a foreign claim."""
    d = ensure_dir(Path(root) / run_id)
    lock = d / "OWNER.json"
    if lock.exists():
        import json

        owner = json.loads(lock.read_text(encoding="utf-8")).get("user")
        if owner and owner != user:
            raise RuntimeError(f"run dir {d} is owned by {owner!r}; choose another RUN_TAG "
                               "instead of overwriting their checkpoints")
    else:
        save_json({"user": user, "claimed_at": now_iso(), "run_id": run_id}, lock)
    return d
