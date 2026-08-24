"""Dataset audit: everything you should know before training anything.

Run this first.  It answers, from the files themselves rather than from
assumption: how many images and tables are there, do they pair up, what is the
pixel size and how was it established, what units is the measurement column in,
how wide are the fibers, are there duplicate fields, and which annotations look
unusable.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import (detect_footer_row, detect_scale_bar_px, read_footer_text,
                          resolve_calibration)
from .csv_parser import infer_length_quantum, parse_measurement_csv
from .image_registration import hamming, perceptual_hash
from .utils import (ensure_dir, environment_report, get_logger, image_id_from_path,
                    list_images, read_gray, read_rgb, save_json, wrap_deg_180)

LOG = get_logger(__name__)


def audit(data_dir: str | Path, *, original_sub: str = "original",
          annotated_sub: str = "annotated", csv_sub: str = "csv",
          output: str | Path | None = None) -> dict[str, Any]:
    data_dir = Path(data_dir)
    report: dict[str, Any] = {"data_dir": str(data_dir),
                              "environment": environment_report()}

    def _list(sub: str) -> list[Path]:
        d = data_dir / sub
        return list_images(d) if d.is_dir() else []

    originals = _list(original_sub)
    annotated = _list(annotated_sub)
    csv_dir = data_dir / csv_sub
    csvs = sorted(p for p in csv_dir.glob("*")
                  if p.suffix.lower() in (".csv", ".tsv", ".txt", ".xls", ".xlsx")
                  ) if csv_dir.is_dir() else []

    report["counts"] = {"original_images": len(originals),
                        "annotated_images": len(annotated),
                        "csv_files": len(csvs)}

    # ---- pairing --------------------------------------------------------
    ids = {"original": {image_id_from_path(p): p for p in originals},
           "annotated": {image_id_from_path(p): p for p in annotated},
           "csv": {image_id_from_path(p): p for p in csvs}}
    all_ids = sorted(set().union(*[set(v) for v in ids.values()]) if ids else [])
    pairing = []
    for i in all_ids:
        row = {"image_id": i,
               **{k: (str(v[i]) if i in v else None) for k, v in ids.items()}}
        row["complete_triplet"] = all(row[k] for k in ("original", "annotated", "csv"))
        pairing.append(row)
    report["pairing"] = pairing
    report["unmatched"] = [r["image_id"] for r in pairing if not r["complete_triplet"]]

    # ---- duplicate detection -------------------------------------------
    hashes: dict[str, np.ndarray] = {}
    for i, p in {**ids["annotated"], **ids["original"]}.items():
        try:
            hashes[i] = perceptual_hash(read_gray(p))
        except Exception as exc:  # noqa: BLE001
            LOG.warning("hash failed for %s: %s", p, exc)
    dups = []
    keys = sorted(hashes)
    for a_i, a in enumerate(keys):
        for b in keys[a_i + 1:]:
            d = hamming(hashes[a], hashes[b])
            if d <= 6:
                dups.append({"a": a, "b": b, "hamming": d})
    report["possible_duplicate_fields"] = dups

    # ---- per-image image properties ------------------------------------
    images_info = []
    magnifications = set()
    for i in all_ids:
        p = ids["original"].get(i) or ids["annotated"].get(i)
        if p is None:
            continue
        try:
            gray = read_gray(p)
        except Exception as exc:  # noqa: BLE001
            images_info.append({"image_id": i, "error": str(exc)})
            continue
        footer = detect_footer_row(gray)
        calib = resolve_calibration(p, gray, image_id=i)
        rgb = read_rgb(p)
        chroma = int(((np.abs(rgb[..., 0].astype(int) - rgb[..., 1]) > 25)
                      | (np.abs(rgb[..., 1].astype(int) - rgb[..., 2]) > 25)).sum())
        info = {
            "image_id": i, "path": str(p),
            "height": int(gray.shape[0]), "width": int(gray.shape[1]),
            "dtype": str(read_gray(p).dtype), "bit_depth_effective": 8,
            "intensity_min": float(gray.min()), "intensity_max": float(gray.max()),
            "intensity_mean": float(gray.mean()), "intensity_std": float(gray.std()),
            "footer_row": footer,
            "scale_bar_px": detect_scale_bar_px(gray, footer),
            "footer_text": read_footer_text(gray, footer).strip()[:200],
            "nm_per_pixel": calib.nm_per_pixel,
            "calibration_source": calib.source,
            "calibration_detail": calib.detail,
            "overlay_pixels": chroma,
            "overlay_fraction": chroma / float(gray.size),
        }
        if calib.nm_per_pixel:
            magnifications.add(round(calib.nm_per_pixel, 4))
        images_info.append(info)
    report["images"] = images_info
    report["distinct_pixel_sizes"] = sorted(magnifications)
    report["multiple_magnifications"] = len(magnifications) > 1

    # ---- CSV audit ------------------------------------------------------
    csv_info = []
    all_lengths: list[np.ndarray] = []
    all_angles: list[np.ndarray] = []
    for i, p in ids["csv"].items():
        try:
            parsed = parse_measurement_csv(p)
        except Exception as exc:  # noqa: BLE001
            csv_info.append({"image_id": i, "path": str(p), "error": str(exc)})
            continue
        lengths = parsed.frame["length"].to_numpy(float)
        angles = parsed.frame["angle"].to_numpy(float)
        lengths = lengths[np.isfinite(lengths)]
        if lengths.size == 0:
            csv_info.append({"image_id": i, "path": str(p),
                             "columns": parsed.raw_columns,
                             "n_measurements": 0,
                             "n_parse_errors": len(parsed.errors),
                             "has_coordinates": parsed.has_coordinates,
                             "has_endpoints": parsed.has_endpoints,
                             "error": "no usable Length values -- check the "
                                      "column mapping in column_map"})
            continue
        quantum = infer_length_quantum(lengths)
        img = next((x for x in images_info if x["image_id"] == i), None)
        oob = 0
        if parsed.has_coordinates and img:
            xs = parsed.frame["cx"].to_numpy(float)
            ys = parsed.frame["cy"].to_numpy(float)
            oob = int(((xs < 0) | (xs >= img["width"])
                       | (ys < 0) | (ys >= img["height"])).sum())
        entry = {
            "image_id": i, "path": str(p),
            "columns": parsed.raw_columns, "column_map": parsed.column_map,
            "n_measurements": int(parsed.n_rows), "n_dropped": int(parsed.n_dropped),
            "has_coordinates": parsed.has_coordinates,
            "has_endpoints": parsed.has_endpoints,
            "n_parse_errors": len(parsed.errors),
            "measurements_outside_image": oob,
            "length_percentiles": {f"p{q}": float(np.percentile(lengths, q))
                                   for q in (0, 5, 25, 50, 75, 95, 100)},
            "length_lattice": quantum["best"],
            "length_lattice_fraction": quantum["frac"],
            "angle_range": [float(np.nanmin(angles)), float(np.nanmax(angles))]
            if np.isfinite(angles).any() else None,
        }
        if quantum["best"] and img and img.get("nm_per_pixel"):
            q = quantum["best"]["quantum"]
            entry["units_note"] = (
                f"lengths sit on a lattice of {q:.4f}; if the measurement was made "
                f"in whole pixels this implies {q:.4f} units/px on the source image, "
                f"against {img['nm_per_pixel']:.4f} nm/px measured from this file")
        csv_info.append(entry)
        all_lengths.append(lengths[np.isfinite(lengths)])
        all_angles.append(angles[np.isfinite(angles)])
    report["csv_files"] = csv_info

    if all_lengths:
        L = np.concatenate(all_lengths)
        report["width_distribution"] = {
            "n": int(L.size), "mean": float(L.mean()), "median": float(np.median(L)),
            "std": float(L.std(ddof=1)) if L.size > 1 else 0.0,
            **{f"p{q}": float(np.percentile(L, q)) for q in (1, 5, 25, 50, 75, 95, 99)},
        }
    if all_angles:
        A = wrap_deg_180(np.concatenate(all_angles))
        hist, edges = np.histogram(A, bins=18, range=(-90, 90))
        report["angle_distribution"] = {
            "counts": hist.tolist(),
            "bin_edges_deg": edges.tolist(),
            "circular_uniformity_note": "counts flat => isotropic fiber network",
        }

    # ---- headline warnings ---------------------------------------------
    warnings: list[str] = []
    n_triplets = sum(1 for r in pairing if r["complete_triplet"])
    if n_triplets == 0:
        warnings.append("NO complete (original, annotated, csv) triplet exists. "
                        "Training data cannot be assembled without one.")
    if n_triplets < 3:
        warnings.append(f"only {n_triplets} complete triplet(s): a grouped "
                        "train/val/test split is not possible, so any result is a "
                        "proof of concept, not a validated model.")
    if any(c.get("has_coordinates") is False for c in csv_info):
        warnings.append("at least one CSV has no coordinate columns; positions must "
                        "be recovered from the annotated overlay, which is lossy. "
                        "Re-export with 'Bounding rectangle' and 'Centroid' enabled.")
    if any(i.get("nm_per_pixel") is None for i in images_info):
        warnings.append("at least one image has no determinable pixel size; results "
                        "for it will be in pixels only.")
    if dups:
        warnings.append(f"{len(dups)} image pair(s) look like the same field; they "
                        "must not be split across train and test.")
    report["warnings"] = warnings
    for w in warnings:
        LOG.warning(w)

    if output:
        save_json(report, output)
        LOG.info("audit written to %s", output)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Audit the SEM measurement dataset")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--original_sub", default="original")
    ap.add_argument("--annotated_sub", default="annotated")
    ap.add_argument("--csv_sub", default="csv")
    ap.add_argument("--output", default="outputs/dataset_audit.json")
    args = ap.parse_args(argv)
    ensure_dir(Path(args.output).parent)
    rep = audit(args.data_dir, original_sub=args.original_sub,
                annotated_sub=args.annotated_sub, csv_sub=args.csv_sub,
                output=args.output)
    print(f"images: {rep['counts']}   warnings: {len(rep['warnings'])}")
    for w in rep["warnings"]:
        print(f"  ! {w}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
