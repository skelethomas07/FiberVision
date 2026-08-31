from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

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



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Human-reviewed SEM training dataset tools")
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="export immutable approved supervision as JSONL")
    export.add_argument("--output", type=Path, default=Path("training_exports/approved.jsonl"))
    export.set_defaults(func=command_export)

    prepare = sub.add_parser("prepare", help="materialize approved review labels.csv + images")
    prepare.add_argument("--output-dir", type=Path, default=Path("training_exports/bundle"))
    prepare.add_argument("--base-labels-csv", type=Path, default=None)
    prepare.add_argument("--base-image-dir", type=Path, default=None)
    prepare.set_defaults(func=command_prepare)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
