"""End-to-end orchestration (v7) shared by the Colab notebook and the self-test.

Steps, each writing its artefacts to the run directory:

1. ``extract_labels``     annotated overlays + tables (+ originals) -> labels.csv
2. ``audit_calibration``  physical route vs annotator scale -> calibration table + gate
3. ``make_split``         specimen-grouped, near-duplicate-merged, sealed split
4. ``plan_physical``      reference nm/px from TRAIN fields only; resample factors
5. ``run_training``       :func:`train.train` (protocol fixed; hardware adapts)
6. ``select_and_evaluate`` post-processing chosen on validation; test evaluated ONCE
7. ``write_run_manifest``  everything needed to reproduce
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .calib_audit import (apply_calibration_to_labels, audit_from_extraction_meta, audit_table,
                          calibration_gate)
from .dataset import ImageRecord, load_records
from .evaluate import evaluate_split, post_from_cfg, select_on_validation
from .physical import plan_resample, reference_nm_per_px
from .quality import training_stats
from .specimens import (assert_no_leakage, calibration_strata, load_specimen_map, merge_groups,
                        near_duplicate_groups, sealed_split, specimen_groups, split_digest,
                        write_split_manifest)
from .train import _prior_cfg, protocol_digest, resolve_protocol, train
from .utils import LOG, ensure_dir, package_version, read_gray, save_json


# --------------------------------------------------------------------------- #
def extract_labels(annotated_dir: str | Path, csv_dir: str | Path, original_dir: str | Path | None,
                   out_dir: str | Path, *, calib_table: dict[str, float] | None = None,
                   csv_angle_convention: str = "imagej_y_up", length_units: str = "auto",
                   limit: int | None = None) -> dict[str, Any]:
    import pandas as pd

    from .annotation_extraction import _pair_inputs, extract_one
    from .labels import LABEL_COLUMNS

    out_dir = ensure_dir(out_dir)
    pairs = _pair_inputs(Path(original_dir) if original_dir else None, Path(annotated_dir), Path(csv_dir))
    frames, metas, errors = [], [], []
    for k, (image_id, files) in enumerate(sorted(pairs.items())):
        if limit and k >= limit:
            break
        if files["annotated"] is None or files["csv"] is None:
            errors.append({"image_id": image_id, "reason": "incomplete_triplet",
                           "detail": f"annotated={files['annotated']}, csv={files['csv']}"})
            continue
        try:
            rec = extract_one(image_id, files["annotated"], files["csv"], files["original"],
                              overlays=files.get("overlays"), calib_table=calib_table or {},
                              length_units=length_units, csv_angle_convention=csv_angle_convention)
        except Exception as exc:                            # noqa: BLE001
            LOG.exception("%s failed", image_id)
            errors.append({"image_id": image_id, "reason": "extraction_exception", "detail": str(exc)})
            continue
        frames.append(rec["labels"])
        metas.append(rec["meta"])
        errors.extend(rec["errors"])
    labels = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=LABEL_COLUMNS)
    labels_csv = out_dir / "labels.csv"
    labels.to_csv(labels_csv, index=False)
    save_json(metas, out_dir / "labels_meta.json")
    pd.DataFrame(errors).to_csv(out_dir / "labels_errors.csv", index=False)
    LOG.info("extracted %d annotations from %d field(s); %d issue(s) logged", len(labels), len(frames), len(errors))
    return {"labels_csv": str(labels_csv), "meta": metas, "errors": errors, "n_fields": len(frames)}


def audit_calibration(metas: list[dict[str, Any]], labels_csv: str | Path, out_dir: str | Path, *,
                      manual_table: dict[str, float] | None = None, tolerance: float = 0.10) -> dict[str, Any]:
    import pandas as pd

    out_dir = ensure_dir(out_dir)
    audits = audit_from_extraction_meta(metas, manual_table=manual_table, tolerance=tolerance)
    table = audit_table(audits)
    table.to_csv(out_dir / "calibration_table.csv", index=False)
    df = pd.read_csv(labels_csv)
    df = apply_calibration_to_labels(df, audits)
    df.to_csv(labels_csv, index=False)
    gate = calibration_gate(audits)
    save_json(gate, out_dir / "calibration_gate.json")
    LOG.info("calibration audit: %d field(s), %d valid for nm, %d invalid", len(audits),
             sum(a.calibration_valid for a in audits), gate["n_invalid"])
    return {"audits": audits, "table": table, "gate": gate}


# --------------------------------------------------------------------------- #
def make_split(records: Sequence[ImageRecord], cfg: dict[str, Any], out_dir: str | Path, *,
               seed: int | None = None) -> dict[str, Any]:
    sc = dict(cfg.get("split") or {})
    level = str(sc.get("level", "specimen"))
    ids = [r.image_id for r in records]
    override = load_specimen_map(sc.get("specimen_map")) if sc.get("specimen_map") else {}
    g_spec = specimen_groups(ids, level=level, override=override)
    thumbs = {}
    for r in records:
        try:
            thumbs[r.image_id] = r.image()
        except Exception as exc:                            # noqa: BLE001
            LOG.warning("%s: could not load for duplicate check (%s)", r.image_id, exc)
    g_dup, dup_pairs = near_duplicate_groups(thumbs, hamming_max=int(sc.get("duplicate_hamming", 6)))
    groups = merge_groups(g_spec, g_dup)
    n_merged = len(set(g_spec.values())) - len(set(groups.values()))
    strata = calibration_strata({r.image_id: r.nm_per_pixel for r in records}) if sc.get("stratify", True) else None
    split = sealed_split(ids, groups, strata=strata, val_frac=float(sc.get("val_frac", 0.25)),
                         test_frac=float(sc.get("test_frac", 0.25)),
                         seed=int(seed if seed is not None else cfg.get("seed", 1337)))
    split["level"] = level
    split["specimen_groups"] = g_spec
    split["duplicate_pairs"] = dup_pairs
    split["n_groups_merged_by_duplicates"] = int(n_merged)
    assert_no_leakage(split, groups)
    man = write_split_manifest(split, Path(out_dir) / "split_manifest.json")
    for r in records:
        r.specimen = groups.get(r.image_id, r.image_id)
    return man


def plan_physical(records: Sequence[ImageRecord], split: dict[str, Any], cfg: dict[str, Any],
                  out_dir: str | Path) -> dict[str, Any]:
    pc = dict(cfg.get("physical") or {})
    mode = str(pc.get("resample_to_reference", "auto"))
    train_ids = set(split["train"])
    by_id = {r.image_id: r for r in records}
    train_nm = {i: (by_id[i].nm_per_pixel if by_id[i].calibration_valid else None) for i in train_ids if i in by_id}
    frac = float(np.mean([v is not None for v in train_nm.values()])) if train_nm else 0.0
    ref = reference_nm_per_px(train_nm)
    enabled = mode == "on" or (mode == "auto" and frac >= float(pc.get("min_calibrated_fraction", 0.7)))
    decisions, factors = {}, {}
    if enabled and ref:
        rng = tuple(pc.get("factor_range", (0.4, 2.5)))
        for r in records:
            d = plan_resample(r.image_id, r.image().shape, r.nm_per_pixel, ref,
                              calibration_valid=r.calibration_valid, factor_range=rng)
            decisions[r.image_id] = d.to_dict()
            if d.included and d.factor_applied:
                factors[r.image_id] = float(d.factor_applied)
    out = {"enabled": bool(enabled and ref), "mode": mode, "reference_nm_per_px": ref if enabled else None,
           "train_calibrated_fraction": frac, "decisions": decisions, "factors": factors,
           "note": "reference resolution computed from TRAINING fields only; uncalibrated fields "
                   "are used at native resolution (pixel widths only)"}
    save_json(out, Path(out_dir) / "physical_reference.json")
    return out


# --------------------------------------------------------------------------- #
def load_dataset(labels_csv: str | Path, image_dir: str | Path, cfg: dict[str, Any], *,
                 prior_cache_dir: str | Path | None = None, resample_factors: dict[str, float] | None = None,
                 image_ids: Sequence[str] | None = None) -> list[ImageRecord]:
    return load_records(labels_csv, image_dir, prior_cfg=_prior_cfg(cfg), prior_cache_dir=prior_cache_dir,
                        resample_factors=resample_factors, image_ids=image_ids)


def run_training(cfg: dict[str, Any], records: Sequence[ImageRecord], split: dict[str, Any],
                 run_dir: str | Path, *, run_mode: str, drive_dir: str | Path | None = None,
                 resume: bool = True, user: str = "", max_seconds: float | None = None) -> dict[str, Any]:
    run_dir = ensure_dir(run_dir)
    res = train(cfg, records=records, split=split, run_dir=run_dir, run_mode=run_mode,
                drive_dir=drive_dir, resume=resume, user=user, max_seconds=max_seconds)
    by_id = {r.image_id: r for r in records}
    tr = [by_id[i] for i in split["train"] if i in by_id]
    stats = training_stats([r.image() for r in tr], [r.nm_per_pixel if r.calibration_valid else None for r in tr])
    save_json(stats.to_dict(), run_dir / "training_stats.json")
    try:
        from .visualization import training_curves

        training_curves(res["history"], run_dir / "training_curves.png")
    except Exception as exc:                                # noqa: BLE001
        LOG.warning("training curve figure failed (%s)", exc)
    return res


def select_and_evaluate(cfg: dict[str, Any], records: Sequence[ImageRecord], split: dict[str, Any],
                        run_dir: str | Path, *, label: str = "", tta: bool = False,
                        evaluate_test: bool = True) -> dict[str, Any]:
    import torch

    from .infer import load_run

    run_dir = Path(run_dir)
    run = load_run(run_dir)
    model, device = run["model"], run["device"]
    by_id = {r.image_id: r for r in records}
    val = [by_id[i] for i in split["val"] if i in by_id]
    choice = select_on_validation(model, val, cfg, device, tta=tta, out_json=run_dir / "selection.json")
    post = post_from_cfg(cfg, spacing_px=choice["spacing_px"], min_validity=choice["min_validity"],
                         seg_threshold=choice["seg_threshold"])
    stats = run.get("train_stats")
    out: dict[str, Any] = {"selection": choice, "post": post}
    out["val"] = evaluate_split(model, records, split, "val", post, cfg, device, run_dir / "eval_val",
                                tta=tta, train_stats=stats, label=label)
    if evaluate_test and split.get("test"):
        test_marker = run_dir / "TEST_EVALUATED.json"
        if test_marker.exists():
            prev = json.loads(test_marker.read_text(encoding="utf-8"))
            LOG.warning("the sealed test set was already evaluated for this run (%s); re-running "
                        "it with different settings would not be a sealed evaluation", prev.get("at"))
        out["test"] = evaluate_split(model, records, split, "test", post, cfg, device, run_dir / "eval_test",
                                     tta=tta, train_stats=stats, label=label)
        from .utils import now_iso

        save_json({"at": now_iso(), "post": choice, "split_digest": split.get("digest")}, test_marker)
    return out


# --------------------------------------------------------------------------- #
def write_run_manifest(cfg: dict[str, Any], *, run_id: str, user: str, run_mode: str, split: dict[str, Any],
                       records: Sequence[ImageRecord], package_dir: str | Path, run_dir: str | Path,
                       labels_csv: str | Path | None, calibration: dict[str, Any] | None,
                       selection: dict[str, Any] | None, train_result: dict[str, Any] | None,
                       evaluation: dict[str, Any] | None, notes: Sequence[str] = ()) -> dict[str, Any]:
    from .manifest import build_manifest, write_manifest

    proto = resolve_protocol(cfg, run_mode)
    cal_rows = None
    if calibration is not None and "table" in calibration:
        cal_rows = calibration["table"].to_dict(orient="records")
    ev = None
    if evaluation:
        ev = {}
        for part in ("val", "test"):
            if part in evaluation:
                e = evaluation[part]
                ev[part] = {"n_fields": e["n_fields"], "n_specimens": e["n_specimens"],
                            "pass_fields": e["pass_fields"], "aggregate_all_fields": e["aggregate_all_fields"],
                            "aggregate_pass_only": e["aggregate_pass_only"]}
    man = build_manifest(run_id=run_id, user=user, run_mode=run_mode, cfg=cfg,
                         protocol_digest=protocol_digest(proto, split.get("digest", "")), split=split,
                         records=records, package_dir=package_dir, run_dir=run_dir, labels_csv=labels_csv,
                         calibration_table=cal_rows, selection=selection, train_result=train_result,
                         evaluation=ev, notes=notes)
    write_manifest(man, Path(run_dir) / "manifest.json")
    return man
