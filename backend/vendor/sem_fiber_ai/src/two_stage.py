"""Leakage-safe two-stage SEM fiber training.

Stage 1 learns generic fiber appearance from deterministic classical pseudo-labels.
Stage 2 learns the human measurement convention from manual annotations.

The publication path in this module is :func:`nested_loso_two_stage`: every outer
specimen is treated as a sealed test set.  No image, manual width, pseudo-label,
threshold decision, early-stopping decision, or calibration value from that
specimen can enter training.  An inner specimen is used for model/threshold
selection; optionally the model is then refit on every non-test specimen for a
fixed number of epochs before the outer specimen is evaluated exactly once.
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .dataset import (group_near_duplicates, grouped_kfold, load_records,
                      merge_groups, specimen_groups)
from .evaluate import evaluate, tune_threshold
from .fiber_prior import PriorConfig
from .postprocess import PostConfig
from .pseudolabel import (PseudoConfig, build_pseudo_labels, calibrate_widths,
                          validate_against_manual)
from .train import _prior_cfg, _sub, train
from .utils import ensure_dir, get_logger, save_json

LOG = get_logger(__name__)


def _pseudo_cfg(cfg: dict[str, Any]) -> PseudoConfig:
    raw = _sub(cfg, "pseudo", default={}) or {}
    known = {k: v for k, v in raw.items() if k in PseudoConfig.__dataclass_fields__}
    return PseudoConfig(**known)


def _positive_manual(csv_path: str | Path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "is_negative" in df.columns:
        df = df[~df["is_negative"].fillna(False).astype(bool)]
    df = df.copy()
    df["image_id"] = df["image_id"].astype(str)
    return df


def _count_fields(csv_path: str | Path) -> int:
    try:
        return int(_positive_manual(csv_path)["image_id"].nunique())
    except Exception:
        return 0


def _spec_map(ids: Iterable[str]) -> dict[str, str]:
    ids = [str(x) for x in ids]
    return specimen_groups(ids)


def _spec_set(ids: Iterable[str]) -> set[str]:
    return set(_spec_map(ids).values())


def _ids_in_specimens(ids: Iterable[str], specimens: set[str]) -> list[str]:
    m = _spec_map(ids)
    return [i for i in m if m[i] in specimens]


def _raw_pseudo_fingerprint(cfg: dict[str, Any]) -> dict[str, Any]:
    """Cheap deterministic fingerprint for the raw teacher cache."""
    from .utils import list_images
    from . import __version__

    image_dir = Path(cfg["data"]["image_dir"])
    files = []
    for p in list_images(image_dir):
        st = p.stat()
        files.append((p.name, int(st.st_size), int(st.st_mtime_ns)))
    payload = {
        "version": __version__,
        "image_dir": str(image_dir.resolve()),
        "images": files,
        "pseudo": asdict(_pseudo_cfg(cfg)),
        "prior": asdict(_prior_cfg(cfg)),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return {"sha256": hashlib.sha256(blob).hexdigest(),
            "version": __version__, "n_source_images": len(files)}


def prepare_raw_pseudo_labels(cfg: dict[str, Any], *,
                              out_csv: str | Path | None = None,
                              limit: int | None = None,
                              force: bool = False) -> dict[str, Any]:
    """Build the image-only FWHM teacher once, with *zero* manual-width input.

    The cache is reusable only when the source-image metadata, pseudo/prior
    configuration and package version fingerprint match.  This prevents a stale
    pseudo teacher from surviving an image replacement or algorithm edit.
    """
    import pandas as pd

    out_dir = ensure_dir(_sub(cfg, "output", "dir", default="outputs"))
    out_csv = Path(out_csv or Path(out_dir) / "pseudo_labels_raw.csv")
    manifest_path = out_csv.with_suffix(".manifest.json")
    fp = _raw_pseudo_fingerprint(cfg)
    if limit is not None:
        # A limited smoke-test cache must never masquerade as the full teacher.
        fp = dict(fp, limit=int(limit))

    if out_csv.exists() and manifest_path.exists() and not force:
        try:
            old = json.loads(manifest_path.read_text())
        except Exception:
            old = {}
        if old.get("fingerprint") == fp:
            df = pd.read_csv(out_csv)
            return {"path": str(out_csv), "n": int(len(df)),
                    "n_images": int(df["image_id"].nunique()) if len(df) else 0,
                    "reused": True, "fingerprint": fp}
        LOG.warning("raw pseudo cache fingerprint changed; rebuilding %s", out_csv)

    pseudo = build_pseudo_labels(
        cfg["data"]["image_dir"], out_csv,
        exclude_ids=(), calibration=None,
        cfg=_pseudo_cfg(cfg), prior_cfg=_prior_cfg(cfg), limit=limit)
    if not len(pseudo):
        raise RuntimeError("no raw pseudo-labels were produced; audit the image-only fiber prior")
    pseudo["image_id"] = pseudo["image_id"].astype(str)
    pseudo.to_csv(out_csv, index=False)
    info = {"path": str(out_csv), "n": int(len(pseudo)),
            "n_images": int(pseudo["image_id"].nunique()), "reused": False,
            "manual_widths_used": False, "fingerprint": fp}
    save_json({"fingerprint": fp, "n": info["n"], "n_images": info["n_images"]},
              manifest_path)
    save_json(info, Path(out_dir) / "pseudo_raw_report.json")
    return info


def prepare_pseudo_labels(cfg: dict[str, Any], *,
                          out_csv: str | Path | None = None,
                          raw_csv: str | Path | None = None,
                          calibration_image_ids: Iterable[str] | None = None,
                          allowed_specimens: Iterable[str] | None = None,
                          excluded_specimens: Iterable[str] = (),
                          limit: int | None = None,
                          strict: bool | None = None) -> dict[str, Any]:
    """Create a stage-1 CSV from raw image-only pseudo-labels.

    ``calibration_image_ids`` is the complete answer-key allow-list.  In nested
    validation pass only inner-training manual fields.  ``allowed_specimens`` and
    ``excluded_specimens`` gate the image appearances that may enter stage 1.
    """
    import pandas as pd

    out_dir = ensure_dir(_sub(cfg, "output", "dir", default="outputs"))
    out_csv = Path(out_csv or Path(out_dir) / "pseudo_labels.csv")
    raw_csv = Path(raw_csv or Path(out_dir) / "pseudo_labels_raw.csv")
    if not raw_csv.exists():
        prepare_raw_pseudo_labels(cfg, out_csv=raw_csv, limit=limit)

    raw = pd.read_csv(raw_csv)
    raw["image_id"] = raw["image_id"].astype(str)
    manual = _positive_manual(cfg["data"]["labels_csv"])
    manual_ids = set(manual["image_id"].unique())

    allowed = set(map(str, allowed_specimens)) if allowed_specimens is not None else None
    excluded = set(map(str, excluded_specimens))
    raw_spec = _spec_map(raw["image_id"].unique())
    keep_ids = {iid for iid, sp in raw_spec.items()
                if (allowed is None or sp in allowed) and sp not in excluded}
    working = raw[raw["image_id"].isin(keep_ids)].copy()

    if calibration_image_ids is None:
        cal_ids = set(manual["image_id"].unique())
        if allowed is not None or excluded:
            mspec = _spec_map(cal_ids)
            cal_ids = {i for i in cal_ids
                       if (allowed is None or mspec[i] in allowed) and mspec[i] not in excluded}
    else:
        cal_ids = set(map(str, calibration_image_ids))
    cal_manual = manual[manual["image_id"].isin(cal_ids)].copy()

    info: dict[str, Any] = {
        "raw_csv": str(raw_csv), "n_raw": int(len(working)),
        "n_raw_images": int(working["image_id"].nunique()) if len(working) else 0,
        "calibration_image_ids": sorted(cal_ids),
        "allowed_specimens": sorted(allowed) if allowed is not None else None,
        "excluded_specimens": sorted(excluded),
    }
    info["agreement_before"] = validate_against_manual(working, cal_manual)

    if bool(_sub(cfg, "pseudo", "calibrate_to_manual", default=True)):
        strict = bool(_sub(cfg, "pseudo", "calibration_strict", default=True) if strict is None
                      else strict)
        working, cal = calibrate_widths(
            working, cal_manual, strict=strict,
            max_relative_range=float(_sub(cfg, "pseudo", "max_ratio_range_fraction", default=0.25)),
            min_images=int(_sub(cfg, "pseudo", "min_calibration_images", default=2)))
        info["calibration"] = cal
        info["agreement_after"] = validate_against_manual(working, cal_manual)

    # Never pretrain on a pseudo version of a manually-labelled field.  Manual
    # fields are present in raw only to estimate the training-fold calibration.
    kept = working[~working["image_id"].isin(manual_ids)].copy()
    if not len(kept):
        raise RuntimeError("no unlabelled pseudo-labelled fields remain after specimen/leakage gates")
    kept.to_csv(out_csv, index=False)
    info.update({"n_kept": int(len(kept)),
                 "n_kept_images": int(kept["image_id"].nunique()),
                 "kept_image_ids": sorted(kept["image_id"].unique())})
    save_json(info, out_csv.with_suffix(".report.json"))
    LOG.info("prepared %d pseudo labels over %d stage-1 field(s) -> %s",
             len(kept), kept["image_id"].nunique(), out_csv)
    return info


def _run_two_stage(cfg: dict[str, Any], *, pseudo_csv: str | Path,
                   manual_splits: dict[str, list[str]], out_root: str | Path,
                   pretrain_epochs: int, finetune_lr_scale: float = 0.25,
                   finetune_epochs: int | None = None,
                   use_last_finetune: bool = False) -> dict[str, Any]:
    """Internal two-stage runner with explicit, externally audited splits."""
    import pandas as pd

    out_root = ensure_dir(out_root)
    pseudo_csv = Path(pseudo_csv)
    manual_csv = cfg["data"]["labels_csv"]
    pdf = pd.read_csv(pseudo_csv)
    pseudo_ids = sorted(pdf["image_id"].astype(str).unique())
    if not pseudo_ids:
        raise RuntimeError("pseudo CSV is empty")

    # Stage 1: fixed schedule; pseudo validation would only measure agreement
    # with the teacher, so the final epoch is intentionally the warm start.
    stage1 = json.loads(json.dumps(cfg))
    stage1["data"]["labels_csv"] = str(pseudo_csv)
    stage1["output"]["dir"] = str(Path(out_root) / "pretrain")
    stage1["train"]["epochs"] = int(pretrain_epochs)
    stage1["train"]["early_stop_patience"] = 0
    per_img = int(_sub(cfg, "train", "samples_per_image", default=40))
    n_fields = max(1, len(pseudo_ids))
    n_manual = max(1, len(manual_splits.get("train", [])))
    stage1["train"]["samples_per_image"] = max(4, int(round(per_img * n_manual / n_fields)))
    stage1.setdefault("prior", {})["cache_dir"] = str(Path(out_root) / "prior_cache")
    p_split = {"train": pseudo_ids, "val": [], "test": []}
    r1 = train(stage1, model_kind="full", _forced_splits=p_split)
    ck1 = Path(stage1["output"]["dir"]) / "last_full.pt"
    if not ck1.exists():
        raise RuntimeError(f"stage-1 checkpoint missing: {ck1}")

    # Stage 2: only the caller decides which manual specimen is validation/test.
    stage2 = json.loads(json.dumps(cfg))
    stage2["data"]["labels_csv"] = str(manual_csv)
    stage2["output"]["dir"] = str(Path(out_root) / "finetune")
    stage2["train"]["init_from"] = str(ck1)
    stage2["train"]["lr"] = float(_sub(cfg, "train", "lr", default=3e-4)) * finetune_lr_scale
    if finetune_epochs is not None:
        stage2["train"]["epochs"] = int(finetune_epochs)
        stage2["train"]["early_stop_patience"] = 0
    stage2.setdefault("prior", {})["cache_dir"] = str(Path(out_root) / "prior_cache")
    r2 = train(stage2, model_kind="full", _forced_splits=manual_splits)
    ck2 = Path(stage2["output"]["dir"]) / ("last_full.pt" if use_last_finetune else "best_full.pt")
    if not ck2.exists():
        ck2 = Path(stage2["output"]["dir"]) / "last_full.pt"
    if not ck2.exists():
        raise RuntimeError("stage-2 checkpoint missing")

    return {"stage1": r1, "stage2": r2, "pretrain_checkpoint": str(ck1),
            "finetune_checkpoint": str(ck2), "stage2_config": stage2}


def pretrain_then_finetune(cfg: dict[str, Any], *,
                           pseudo_csv: str | Path | None = None,
                           pretrain_epochs: int | None = None,
                           finetune_lr_scale: float = 0.25) -> dict[str, Any]:
    """Convenience two-stage run for development; nested LOSO is for claims."""
    from .dataset import build_splits

    out_root = ensure_dir(_sub(cfg, "output", "dir", default="outputs"))
    pseudo_csv = Path(pseudo_csv or Path(out_root) / "pseudo_labels.csv")
    if not pseudo_csv.exists():
        prepare_pseudo_labels(cfg, out_csv=pseudo_csv)
    records = load_records(cfg["data"]["labels_csv"], cfg["data"]["image_dir"],
                           prior_cfg=_prior_cfg(cfg), prior_cache_dir=Path(out_root)/"prior_cache")
    splits = build_splits(records, cfg)
    run = _run_two_stage(
        cfg, pseudo_csv=pseudo_csv, manual_splits=splits, out_root=out_root,
        pretrain_epochs=int(pretrain_epochs or max(8, int(_sub(cfg,"train","epochs",default=80))//4)),
        finetune_lr_scale=finetune_lr_scale)
    summary = {"pretrain": {"checkpoint": run["pretrain_checkpoint"],
                             "epochs": int(pretrain_epochs or max(8, int(_sub(cfg,"train","epochs",default=80))//4))},
               "finetune": {"checkpoint": run["finetune_checkpoint"],
                             "best_val_loss": run["stage2"]["best_val_loss"],
                             "best_epoch": run["stage2"].get("best_epoch"),
                             "splits": run["stage2"]["splits"]}}
    save_json(summary, Path(out_root) / "two_stage_summary.json")
    return summary


def _outer_folds(cfg: dict[str, Any], n_splits: int | None = None):
    out_root = ensure_dir(_sub(cfg, "output", "dir", default="outputs"))
    records = load_records(cfg["data"]["labels_csv"], cfg["data"]["image_dir"],
                           prior_cfg=_prior_cfg(cfg), prior_cache_dir=Path(out_root)/"prior_cache")
    ids = [r.image_id for r in records]
    images = {r.image_id: r.image() for r in records}
    groups = group_near_duplicates(images,
        hamming_max=int(_sub(cfg, "split", "duplicate_hamming", default=6)))
    groups = merge_groups(groups, specimen_groups(ids))
    n_groups = len(set(groups.values()))
    if n_groups < 3:
        raise RuntimeError(f"nested specimen validation needs >=3 independent groups; found {n_groups}")
    k = int(n_splits or n_groups)
    k = min(k, n_groups)
    return records, groups, grouped_kfold(ids, k, seed=int(_sub(cfg,"seed",default=1337)), groups=groups)


def _choose_inner_split(outer_train_ids: list[str], *, fold_index: int,
                        seed: int, groups: dict[str, str] | None = None) -> dict[str, list[str]]:
    # Use the same merged near-duplicate + specimen grouping as the outer CV.
    # Otherwise a perceptual duplicate with a different filename/specimen prefix
    # could sit in inner-train while its twin is used for early stopping.
    groups = groups or specimen_groups(outer_train_ids)
    uniq = sorted({groups[i] for i in outer_train_ids})
    if len(uniq) < 2:
        raise RuntimeError("outer-training set has fewer than two independent groups; no inner validation is possible")
    inner_group = uniq[(fold_index + seed) % len(uniq)]
    val = [i for i in outer_train_ids if groups[i] == inner_group]
    train_ids = [i for i in outer_train_ids if groups[i] != inner_group]
    return {"train": train_ids, "val": val, "test": []}


def _assert_no_specimen_leakage(*, train_ids: Iterable[str], val_ids: Iterable[str],
                                test_ids: Iterable[str], pseudo_ids: Iterable[str],
                                calibration_ids: Iterable[str]) -> dict[str, Any]:
    tr, va, te = _spec_set(train_ids), _spec_set(val_ids), _spec_set(test_ids)
    ps, cs = _spec_set(pseudo_ids), _spec_set(calibration_ids)
    checks = {
        "train_vs_test": sorted(tr & te),
        "inner_val_vs_test": sorted(va & te),
        "pseudo_vs_test": sorted(ps & te),
        "calibration_vs_test": sorted(cs & te),
        "pseudo_vs_inner_val": sorted(ps & va),
        "calibration_vs_inner_val": sorted(cs & va),
    }
    bad = {k: v for k, v in checks.items() if v}
    if bad:
        raise RuntimeError(f"specimen leakage detected: {bad}")
    return {"passed": True, "intersections": checks,
            "train_specimens": sorted(tr), "inner_val_specimens": sorted(va),
            "test_specimens": sorted(te), "pseudo_specimens": sorted(ps),
            "calibration_specimens": sorted(cs)}


def nested_loso_two_stage(cfg: dict[str, Any], *,
                          n_splits: int | None = None,
                          pretrain_epochs: int = 8,
                          finetune_lr_scale: float = 0.25,
                          refit_outer_train: bool = True,
                          raw_pseudo_csv: str | Path | None = None,
                          threshold_objective: str = "width_wasserstein",
                          save_figures: bool = False) -> dict[str, Any]:
    """Nested specimen-level CV for thesis-grade performance estimates.

    Outer test specimen
        sealed from raw-pseudo filtering, width calibration, model fitting,
        early stopping, and threshold selection.
    Inner validation specimen
        sealed from pseudo pretraining/calibration during model selection.
    Optional refit
        after epochs + postprocessing are chosen, retrain on *all* non-test
        specimens for that fixed schedule, then evaluate the outer test once.
    """
    import pandas as pd

    root = Path(ensure_dir(_sub(cfg, "output", "dir", default="outputs"))) / "nested_loso"
    root.mkdir(parents=True, exist_ok=True)
    raw_pseudo_csv = Path(raw_pseudo_csv or root / "pseudo_labels_raw.csv")
    prepare_raw_pseudo_labels(cfg, out_csv=raw_pseudo_csv)
    raw_df = pd.read_csv(raw_pseudo_csv)
    raw_ids_all = sorted(raw_df["image_id"].astype(str).unique())

    records, outer_groups, folds = _outer_folds(cfg, n_splits=n_splits)
    all_manual_ids = [r.image_id for r in records]
    seed = int(_sub(cfg, "seed", default=1337))
    fold_results: list[dict[str, Any]] = []

    for k, outer in enumerate(folds, 1):
        fold_dir = root / f"fold{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        outer_train = list(outer["train"])
        outer_test = list(outer["val"])
        inner = _choose_inner_split(outer_train, fold_index=k-1, seed=seed, groups=outer_groups)
        inner["test"] = outer_test
        train_specs = _spec_set(inner["train"])
        val_specs = _spec_set(inner["val"])
        test_specs = _spec_set(outer_test)

        # Development pseudo labels: neither outer test nor inner validation may
        # influence stage 1 or its width scale.
        dev_pseudo = fold_dir / "dev_pseudo.csv"
        dev_info = prepare_pseudo_labels(
            cfg, out_csv=dev_pseudo, raw_csv=raw_pseudo_csv,
            calibration_image_ids=inner["train"], allowed_specimens=train_specs,
            excluded_specimens=test_specs | val_specs, strict=True)
        dev_pseudo_ids = dev_info["kept_image_ids"]
        leakage = _assert_no_specimen_leakage(
            train_ids=inner["train"], val_ids=inner["val"], test_ids=outer_test,
            pseudo_ids=dev_pseudo_ids, calibration_ids=dev_info["calibration_image_ids"])
        save_json(leakage, fold_dir / "leakage_audit_dev.json")

        dev = _run_two_stage(
            cfg, pseudo_csv=dev_pseudo, manual_splits=inner,
            out_root=fold_dir / "development", pretrain_epochs=int(pretrain_epochs),
            finetune_lr_scale=finetune_lr_scale)
        dev_ck = dev["finetune_checkpoint"]
        tuned = tune_threshold(
            dev_ck, split="val", labels_csv=cfg["data"]["labels_csv"],
            image_dir=cfg["data"]["image_dir"],
            splits_json=Path(dev_ck).parent / "splits.json",
            objective=threshold_objective, device_pref=_sub(cfg,"device",default="auto"))
        best_post = tuned.attrs["best_post"]
        tuned.to_csv(fold_dir / "threshold_sweep.csv", index=False)
        selection = {
            "best_epoch": int(dev["stage2"].get("best_epoch") or 1),
            "peak_threshold": float(best_post.peak_threshold),
            "min_validity": float(best_post.min_validity),
            "inner_train_images": inner["train"], "inner_val_images": inner["val"],
            "outer_test_images": outer_test,
        }
        save_json(selection, fold_dir / "selection.json")

        if refit_outer_train:
            # Refit is allowed to use the former inner-val specimen because the
            # hyperparameters are now frozen; the outer test is still sealed.
            outer_train_specs = _spec_set(outer_train)
            refit_pseudo = fold_dir / "refit_pseudo.csv"
            refit_info = prepare_pseudo_labels(
                cfg, out_csv=refit_pseudo, raw_csv=raw_pseudo_csv,
                calibration_image_ids=outer_train,
                allowed_specimens=outer_train_specs,
                excluded_specimens=test_specs, strict=True)
            refit_split = {"train": outer_train, "val": [], "test": outer_test}
            # Only outer-test separation matters during refit.
            rps = _spec_set(refit_info["kept_image_ids"])
            rcs = _spec_set(refit_info["calibration_image_ids"])
            if (rps | rcs | _spec_set(outer_train)) & test_specs:
                raise RuntimeError("outer-test specimen leaked into refit")
            refit_audit = {"passed": True,
                           "outer_train_specimens": sorted(outer_train_specs),
                           "outer_test_specimens": sorted(test_specs),
                           "pseudo_specimens": sorted(rps),
                           "calibration_specimens": sorted(rcs)}
            save_json(refit_audit, fold_dir / "leakage_audit_refit.json")
            fit = _run_two_stage(
                cfg, pseudo_csv=refit_pseudo, manual_splits=refit_split,
                out_root=fold_dir / "refit", pretrain_epochs=int(pretrain_epochs),
                finetune_lr_scale=finetune_lr_scale,
                finetune_epochs=max(1, selection["best_epoch"]),
                use_last_finetune=True)
            checkpoint = fit["finetune_checkpoint"]
            split_json = Path(checkpoint).parent / "splits.json"
        else:
            checkpoint = dev_ck
            split_json = Path(checkpoint).parent / "splits.json"

        metrics = evaluate(
            checkpoint, split="test", labels_csv=cfg["data"]["labels_csv"],
            image_dir=cfg["data"]["image_dir"], splits_json=split_json,
            output_dir=fold_dir / "outer_test_eval", post=best_post,
            device_pref=_sub(cfg,"device",default="auto"), save_figures=save_figures)
        fold_result = {
            "fold": k, "outer_test_images": outer_test,
            "outer_test_specimens": sorted(test_specs),
            "inner_val_images": inner["val"], "selection": selection,
            "checkpoint": checkpoint, "headline": metrics.get("headline", {}),
            "per_image": metrics.get("per_image", {}),
            "leakage_audit": leakage,
        }
        fold_results.append(fold_result)
        save_json(fold_result, fold_dir / "fold_result.json")

    # Aggregate only out-of-fold outer-test numbers.  With four specimens the
    # individual fold values are the primary result; summary moments are included
    # as descriptive statistics, never a confidence interval.
    def collect(metric: str):
        vals=[]
        for fr in fold_results:
            node=fr.get("headline",{}).get(metric)
            if isinstance(node,dict) and np.isfinite(node.get("mean",np.nan)):
                vals.append(float(node["mean"]))
        return vals
    agg={}
    for metric in ("fiber_recall","skeleton_coverage","width_median_relative_error",
                   "width_wasserstein_px","chord_recall","chord_precision",
                   "orientation_median_error_deg"):
        vals=collect(metric)
        if vals:
            agg[metric]={"values":vals,"mean":float(np.mean(vals)),
                         "sd":float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0,
                         "median":float(np.median(vals)),"n_outer_folds":len(vals)}
    summary={"protocol":"nested specimen-level LOSO two-stage",
             "n_folds":len(fold_results),"folds":fold_results,
             "outer_fold_summary":agg,
             "warning":"With very few specimens, report fold values individually; mean±sd is descriptive, not a confidence interval."}
    save_json(summary, root / "nested_loso_summary.json")
    return summary
