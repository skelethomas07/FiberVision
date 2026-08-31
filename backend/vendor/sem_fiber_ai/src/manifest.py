"""Reproducibility manifest (v7).

One JSON that lets someone else re-run the experiment and check they got the
same thing: package tree hash, resolved config, protocol digest, split
membership and digest, per-field calibration table and hash of the label
table, image hashes, checkpoint hashes, hardware and versions, timestamps.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .utils import (environment_report, now_iso, package_version, save_json, sha256_file,
                    sha256_tree)


def dataset_manifest(records: Sequence[Any], labels_csv: str | Path | None = None) -> dict[str, Any]:
    images = {}
    for r in records:
        try:
            h = sha256_file(r.path)[:16]
        except Exception:                                  # noqa: BLE001
            h = None
        images[r.image_id] = {"path": str(r.path), "sha256_16": h, "n_annotations": int(len(r.annotations)),
                              "specimen": r.specimen, "calibration_status": r.calibration_status,
                              "calibration_valid": bool(r.calibration_valid),
                              "nm_per_pixel": r.nm_per_pixel, "resample_factor": r.resample_factor}
    out = {"n_images": len(images), "images": images}
    if labels_csv and Path(labels_csv).exists():
        out["labels_csv"] = str(labels_csv)
        out["labels_sha256"] = sha256_file(labels_csv)
    return out


def build_manifest(*, run_id: str, user: str, run_mode: str, cfg: dict[str, Any], protocol_digest: str,
                   split: dict[str, Any], records: Sequence[Any], package_dir: str | Path,
                   run_dir: str | Path, labels_csv: str | Path | None = None,
                   calibration_table: list[dict[str, Any]] | None = None,
                   selection: dict[str, Any] | None = None, train_result: dict[str, Any] | None = None,
                   evaluation: dict[str, Any] | None = None, notes: Sequence[str] = ()) -> dict[str, Any]:
    run_dir = Path(run_dir)
    ck = {}
    for name in ("best.pt", "last.pt"):
        p = run_dir / name
        if p.exists():
            ck[name] = sha256_file(p)
    m = {
        "format": "sem_fiber_ai_manifest_v7", "package_version": package_version(),
        "package_tree_sha256": sha256_tree(package_dir), "run_id": run_id, "user": user,
        "run_mode": run_mode, "created_at": now_iso(), "config": cfg,
        "protocol_digest": protocol_digest,
        "split": {k: split.get(k) for k in ("level", "train", "val", "test", "train_groups",
                                            "val_groups", "test_groups", "digest", "seed")},
        "dataset": dataset_manifest(records, labels_csv), "calibration_table": calibration_table,
        "checkpoints": ck, "environment": environment_report(),
        "selection": selection, "training": {k: v for k, v in (train_result or {}).items()
                                            if k not in ("history", "target_cfg")},
        "evaluation": evaluation, "notes": list(notes),
    }
    return m


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    return save_json(manifest, path)
