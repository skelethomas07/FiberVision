from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

from ..config import get_settings
from ..db import SessionLocal
from ..storage import build_storage
from .bundle import prepare_training_bundle
from .export import export_approved_dataset


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _copy_images(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
            shutil.copy2(path, destination / path.name)


def _prepare(output_dir: Path, base_labels: Path | None, base_images: Path | None):
    settings = get_settings()
    storage = build_storage(settings)
    reviewed_dir = output_dir / "reviewed"
    with SessionLocal() as session:
        reviewed = prepare_training_bundle(session, storage, reviewed_dir)

    if not base_labels and not base_images:
        return reviewed.labels_csv, reviewed.image_dir, reviewed.n_examples
    if not base_labels or not base_images:
        raise SystemExit("--base-labels-csv and --base-image-dir must be supplied together")

    combined = output_dir / "combined"
    combined_images = combined / "images"
    _copy_images(base_images, combined_images)
    _copy_images(reviewed.image_dir, combined_images)
    base = pd.read_csv(base_labels)
    new = pd.read_csv(reviewed.labels_csv)
    labels = pd.concat([base, new], ignore_index=True, sort=False)
    combined.mkdir(parents=True, exist_ok=True)
    labels_csv = combined / "labels.csv"
    labels.to_csv(labels_csv, index=False)
    return labels_csv, combined_images, len(new)


def command_export(args) -> None:
    with SessionLocal() as session:
        count = export_approved_dataset(session, args.output)
    print(json.dumps({"output": str(args.output), "training_examples": count}, ensure_ascii=False))


def command_prepare(args) -> None:
    labels, images, count = _prepare(args.output_dir, args.base_labels_csv, args.base_image_dir)
    print(json.dumps({"labels_csv": str(labels), "image_dir": str(images), "reviewed_examples": count}, ensure_ascii=False))


def command_retrain(args) -> None:
    labels, images, count = _prepare(args.dataset_dir, args.base_labels_csv, args.base_image_dir)
    backend_root = _backend_root()
    vendor_root = backend_root / "vendor"
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))

    default_config = backend_root / "vendor" / "sem_fiber_ai" / "config" / "default.yaml"
    cfg = yaml.safe_load(default_config.read_text())
    cfg["data"]["labels_csv"] = str(labels)
    cfg["data"]["image_dir"] = str(images)
    cfg["output"]["dir"] = str(args.output_dir)
    cfg["train"]["init_from"] = str(args.init_from) if args.init_from else None
    table = pd.read_csv(labels)
    cfg["loss"]["neg_boost"] = bool("is_negative" in table and table["is_negative"].fillna(False).astype(bool).any())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_config = args.output_dir / "training_config.yaml"
    generated_config.write_text(yaml.safe_dump(cfg, sort_keys=False))

    print(f"reviewed examples added: {count}")
    print(f"training config: {generated_config}")
    from sem_fiber_ai.src.train import train
    train(cfg, model_kind=args.model)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Human-reviewed SEM training dataset tools")
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="export immutable approved supervision as JSONL")
    export.add_argument("--output", type=Path, default=Path("training_exports/approved.jsonl"))
    export.set_defaults(func=command_export)

    prepare = sub.add_parser("prepare", help="materialize v6.11 labels.csv + images")
    prepare.add_argument("--output-dir", type=Path, default=Path("training_exports/bundle"))
    prepare.add_argument("--base-labels-csv", type=Path, default=None)
    prepare.add_argument("--base-image-dir", type=Path, default=None)
    prepare.set_defaults(func=command_prepare)

    retrain = sub.add_parser("retrain", help="prepare approved data and call vendored v6.11 training pipeline")
    retrain.add_argument("--dataset-dir", type=Path, default=Path("training_exports/retrain_dataset"))
    retrain.add_argument("--output-dir", type=Path, default=Path("training_runs/candidate"))
    retrain.add_argument("--base-labels-csv", type=Path, default=None)
    retrain.add_argument("--base-image-dir", type=Path, default=None)
    retrain.add_argument("--init-from", type=Path, default=None)
    retrain.add_argument("--model", choices=["full", "baseline"], default="full")
    retrain.set_defaults(func=command_retrain)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
