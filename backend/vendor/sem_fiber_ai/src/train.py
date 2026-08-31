"""Training entry point for both the baseline and the full-image model.

Everything needed to reproduce a run is written next to the checkpoint: the
resolved config, the seed, package versions, and the exact split assignment.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .augmentations import AugConfig
from .dataset import (ImageRecord, PatchDataset, TileDataset, calibration_strata,
                      group_near_duplicates, grouped_kfold, grouped_split,
                      load_records, merge_groups, specimen_groups,
                      stratified_grouped_split)
from .fiber_prior import PriorConfig, audit_prior
from .losses import MultiHeadLoss, PatchLoss
from .models.baseline_patch_model import build_baseline
from .models.fiber_measurement_net import build_model
from .targets import TargetConfig
from .utils import (ensure_dir, environment_report, get_logger, pick_device,
                    save_json, set_seed)

LOG = get_logger(__name__)


# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """Fail loudly and early on a malformed config rather than mid-training."""
    required = {"data": ("labels_csv", "image_dir"),
                "train": ("epochs", "batch_size", "lr"),
                "model": (),
                "output": ("dir",)}
    for section, keys in required.items():
        if section not in cfg:
            raise ValueError(f"config is missing the '{section}' section")
        for k in keys:
            if k not in cfg[section]:
                raise ValueError(f"config['{section}'] is missing '{k}'")
    if cfg["train"]["epochs"] <= 0:
        raise ValueError("train.epochs must be positive")
    if not 0 < cfg["train"]["lr"] < 1:
        raise ValueError("train.lr looks implausible")


def _sub(cfg: dict[str, Any], *path: str, default: Any = None) -> Any:
    node: Any = cfg
    for p in path:
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def build_splits(records: list[ImageRecord], cfg: dict[str, Any]
                 ) -> dict[str, list[str]]:
    images = {r.image_id: r.image() for r in records}
    groups = group_near_duplicates(
        images, hamming_max=_sub(cfg, "split", "duplicate_hamming", default=6))
    ids = [r.image_id for r in records]
    if _sub(cfg, "split", "group_by_specimen", default=True):
        groups = merge_groups(groups, specimen_groups(ids))
    if _sub(cfg, "split", "stratify", default=True):
        return stratified_grouped_split(
            ids, calibration_strata(records),
            val_frac=_sub(cfg, "split", "val_frac", default=0.2),
            test_frac=_sub(cfg, "split", "test_frac", default=0.2),
            seed=_sub(cfg, "seed", default=1337), groups=groups)
    return grouped_split(ids,
                         val_frac=_sub(cfg, "split", "val_frac", default=0.2),
                         test_frac=_sub(cfg, "split", "test_frac", default=0.2),
                         seed=_sub(cfg, "seed", default=1337), groups=groups)


def _prior_cfg(cfg: dict[str, Any]) -> PriorConfig:
    raw = _sub(cfg, "prior", default={}) or {}
    known = {k: v for k, v in raw.items() if k in PriorConfig.__dataclass_fields__}
    if "sigmas" in known and isinstance(known["sigmas"], list):
        known["sigmas"] = tuple(known["sigmas"])
    return PriorConfig(**known)


class PriorAuditError(RuntimeError):
    """The fiber prior failed its own audit and training would be wasted."""


class AngleConventionError(RuntimeError):
    """Fields disagree on the angle convention and are about to be pooled."""


def check_priors(records: list[ImageRecord], out_dir: str | Path, *,
                 cfg: PriorConfig | None = None,
                 strict: bool = True) -> dict[str, Any]:
    """Audit the fiber prior on every field before spending a GPU hour on it.

    ``centre_coverage`` is the number to read: the fraction of manually measured
    centres that land inside the derived fiber mask.  The ignore map is built
    from that mask, so a mask that misses fibers silently reinstates the exact
    failure it exists to prevent.
    """
    report: dict[str, Any] = {}
    failed: dict[str, list[str]] = {}
    for rec in records:
        try:
            rep = audit_prior(rec.prior(), rec.annotations, rec.image(), cfg)
            report[rec.image_id] = rep
            if not rep.get("ok", True):
                failed[rec.image_id] = rep.get("failures", [])
        except Exception as exc:                          # pragma: no cover
            LOG.error("prior audit failed for %s: %s", rec.image_id, exc)
            failed[rec.image_id] = [f"audit raised {exc!r}"]
    save_json(report, Path(out_dir) / "prior_audit.json")

    if not failed:
        LOG.info("fiber prior passed on all %d field(s)", len(report))
        return report

    lines = [f"  {k}: " + "; ".join(v) for k, v in sorted(failed.items())]
    msg = ("the fiber prior failed its audit on %d of %d field(s):\n%s\n"
           "Run fiber_prior.tune_prior_config() on a few fields and copy the "
           "result into config['prior'], or set config['prior']['strict']=False "
           "to train anyway and record that the run is not evidence."
           % (len(failed), len(report), "\n".join(lines)))
    if strict:
        # [v3] v2 logged this and trained anyway -- twice, in the August run.
        raise PriorAuditError(msg)
    LOG.error("%s", msg)
    return report


def _check_angle_conventions(records: list[ImageRecord], out_dir: str | Path, *,
                             strict: bool = True) -> "Any":
    """[v3] Refuse to pool fields whose angle column means different things.

    Only reached when ``targets.orientation_source`` is ``chord``.  With the
    default (``image``) the convention is irrelevant to training, and the audit
    stays advisory -- but it is still worth running, because a field that
    prefers a different convention from its own specimen usually indicates a
    mis-registered export rather than a different pen.
    """
    from .angle_audit import audit_records

    df = audit_records(records)
    save_json({"conventions": df.to_dict(orient="records") if len(df) else []},
              Path(out_dir) / "angle_audit.json")
    if len(df) and df["best"].nunique() > 1:
        msg = ("orientation_source='chord' but the fields disagree on the "
               "angle convention (%s). Pooling them trains the model on the "
               "disagreement. Resolve it per source_csv, drop the odd fields, "
               "or use orientation_source='image'."
               % ", ".join(sorted(df["best"].unique())))
        if strict:
            raise AngleConventionError(msg)
        LOG.error("%s", msg)
    return df


# --------------------------------------------------------------------------- #
def train(cfg: dict[str, Any], *, model_kind: str = "full",
          resume: str | None = None,
          _forced_splits: dict[str, list[str]] | None = None) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    seed = int(_sub(cfg, "seed", default=1337))
    set_seed(seed, deterministic=bool(_sub(cfg, "deterministic", default=True)))
    device = pick_device(_sub(cfg, "device", default="auto"))
    out_dir = ensure_dir(_sub(cfg, "output", "dir", default="outputs"))
    LOG.info("device=%s output=%s", device, out_dir)

    pcfg = _prior_cfg(cfg)
    records = load_records(cfg["data"]["labels_csv"], cfg["data"]["image_dir"],
                           mask_dir=_sub(cfg, "data", "mask_dir"),
                           prior_cfg=pcfg,
                           prior_cache_dir=(_sub(cfg, "prior", "cache_dir")
                                            or Path(out_dir) / "prior_cache"))
    if not records:
        raise RuntimeError("no usable image records; check labels_csv and image_dir")
    splits = _forced_splits or build_splits(records, cfg)
    save_json(splits, Path(out_dir) / "splits.json")

    _spec = specimen_groups(splits["val"]) if splits["val"] else {}
    n_val_groups = len(set(_spec.values()))
    if 0 < n_val_groups < 2:
        LOG.warning("validation is %d field(s) from a single specimen. Treat this "
                    "as model selection only, not as a generalisation estimate; "
                    "a separate outer specimen is required for a quoted result.",
                    len(splits["val"]))

    proof_of_concept = len(splits["val"]) == 0
    if proof_of_concept:
        LOG.warning("this training run has no validation split. That is valid for a "
                    "fixed-schedule pretrain/refit stage only when a separate outer "
                    "test specimen remains sealed; do not quote this run's own loss "
                    "as a generalisation metric.")

    by_id = {r.image_id: r for r in records}
    tr = [by_id[i] for i in splits["train"]]
    va = [by_id[i] for i in splits["val"]] or tr

    if bool(_sub(cfg, "prior", "enabled", default=True)) and \
            bool(_sub(cfg, "prior", "audit", default=True)):
        check_priors(tr + va, out_dir, cfg=pcfg,
                     strict=bool(_sub(cfg, "prior", "strict", default=True)))

    # [v3] orientation supervised from the chord is only meaningful if every
    # export agrees on what the chord angle means. The August audit found three
    # different conventions across twelve fields and training pooled them.
    if str(_sub(cfg, "targets", "orientation_source", default="image")) != "image":
        _check_angle_conventions(tr + va, out_dir,
                                 strict=bool(_sub(cfg, "targets",
                                                  "strict_angles", default=True)))

    aug = AugConfig(**{k: tuple(v) if isinstance(v, list) else v
                       for k, v in (_sub(cfg, "augment", default={}) or {}).items()
                       if k in AugConfig.__dataclass_fields__})
    tcfg = TargetConfig(**{k: v for k, v in (_sub(cfg, "targets", default={}) or {}).items()
                           if k in TargetConfig.__dataclass_fields__})

    if model_kind == "baseline":
        ds_tr = PatchDataset(tr, patch=_sub(cfg, "baseline", "patch", default=64),
                             neg_per_pos=_sub(cfg, "baseline", "neg_per_pos", default=1.0),
                             aug=aug, train=True, seed=seed)
        ds_va = PatchDataset(va, patch=_sub(cfg, "baseline", "patch", default=64),
                             neg_per_pos=1.0, train=False, seed=seed + 1)
        model = build_baseline(_sub(cfg, "baseline", default={})).to(device)
        criterion = PatchLoss(_sub(cfg, "loss_weights", default={}),
                              use_uncertainty=_sub(cfg, "loss", "uncertainty",
                                                   default=True))
    else:
        tile = int(_sub(cfg, "train", "tile", default=384))
        use_prior = bool(_sub(cfg, "prior", "enabled", default=True))
        ds_tr = TileDataset(tr, tile=tile, aug=aug, targets=tcfg, train=True,
                            samples_per_image=_sub(cfg, "train", "samples_per_image",
                                                   default=40), seed=seed,
                            prior_cfg=pcfg, use_prior=use_prior)
        ds_va = TileDataset(va, tile=tile, targets=tcfg, train=False,
                            samples_per_image=max(4, _sub(cfg, "train",
                                                          "samples_per_image",
                                                          default=40) // 4),
                            seed=seed + 1, prior_cfg=pcfg, use_prior=use_prior)
        model = build_model(_sub(cfg, "model", default={})).to(device)
        criterion = MultiHeadLoss(_sub(cfg, "loss_weights", default={}),
                                  use_uncertainty=_sub(cfg, "loss", "uncertainty",
                                                       default=True))

    bs = int(cfg["train"]["batch_size"])
    workers = int(_sub(cfg, "train", "num_workers", default=2))
    # [v3] persistent workers. Without them the loader tears its workers down
    # after every epoch, and each new worker re-reads (or recomputes) the fiber
    # prior for every image it touches -- the in-memory cache on ImageRecord
    # only lives as long as the worker does. That is a large part of why stage 1
    # ran at 2248 s/epoch, which is 15.6 h over 25 epochs: longer than a Colab
    # session, so the run could never finish.
    loader_kw: dict[str, Any] = {}
    if workers > 0:
        loader_kw = {"persistent_workers": True, "prefetch_factor": 4}
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=workers,
                       drop_last=len(ds_tr) > bs,
                       pin_memory=(device.type == "cuda"), **loader_kw)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=workers,
                       **loader_kw)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]),
                            weight_decay=float(_sub(cfg, "train", "weight_decay",
                                                    default=1e-4)))
    epochs = int(cfg["train"]["epochs"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = bool(_sub(cfg, "train", "amp", default=True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    init_from = _sub(cfg, "train", "init_from", default=None)
    if init_from and Path(init_from).exists():
        ck0 = torch.load(init_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ck0["model"], strict=False)
        LOG.info("initialised from %s (%d missing, %d unexpected tensors)",
                 init_from, len(missing), len(unexpected))

    start_epoch, best, best_epoch = 0, float("inf"), -1
    patience = int(_sub(cfg, "train", "early_stop_patience", default=15))
    monitor_loss = str(_sub(cfg, "train", "monitor", default="val_monitor"))
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [],
                                       "val_monitor": []}
    if resume and Path(resume).exists():
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        best = float(ck.get("best", best))
        history = ck.get("history", history)
        LOG.info("resumed from %s at epoch %d", resume, start_epoch)

    # [v4] A second criterion with the uncertainty term swapped for a plain
    # Huber.  logvar is clamped to [-6, 6], so exp(-logvar) reaches ~403 and the
    # training objective's val number moves mostly with the width head's
    # CONFIDENCE.  Selecting on that picks the least confident epoch, not the
    # most accurate one.  This monitor can only move when the predictions move.
    monitor_criterion = type(criterion)(_sub(cfg, "loss_weights", default={}),
                                        use_uncertainty=False)

    def run_epoch(loader, training: bool):
        model.train(training)
        total, mon_total, n = 0.0, 0.0, 0
        for batch in loader:
            batch = {k: (v.to(device, non_blocking=True)
                         if hasattr(v, "to") else v) for k, v in batch.items()}
            with torch.set_grad_enabled(training):
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    out = model(batch["image"])
                    loss, parts = criterion(out, batch)
                    if not training:
                        with torch.no_grad():
                            mon, _ = monitor_criterion(out, batch)
                        mon_total += float(mon.detach()) * batch["image"].shape[0]
            if training:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
            total += float(loss.detach()) * batch["image"].shape[0]
            n += batch["image"].shape[0]
        return total / max(1, n), mon_total / max(1, n)

    manifest = {"config": cfg, "seed": seed, "environment": environment_report(),
                "splits": splits, "model_kind": model_kind,
                "proof_of_concept": proof_of_concept,
                "n_train_images": len(tr), "n_val_images": len(splits["val"])}
    save_json(manifest, Path(out_dir) / "run_manifest.json")

    # [v3] wall-clock budget. A hosted runtime is reclaimed on a timer, and a
    # run that is killed mid-epoch leaves a checkpoint nobody can interpret.
    # Stop cleanly instead, with the reason recorded in the manifest.
    budget_s = float(_sub(cfg, "train", "max_seconds", default=0.0) or 0.0)
    t_start = time.time()
    stop_reason = "completed"

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        tr_loss, _ = run_epoch(dl_tr, True)
        va_loss, va_monitor = run_epoch(dl_va, False)
        sched.step()
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history.setdefault("val_monitor", []).append(va_monitor)
        dt = time.time() - t0
        LOG.info("epoch %3d/%d  train=%.4f  val=%.4f  monitor=%.4f  (%.1fs)",
                 epoch + 1, epochs, tr_loss, va_loss, va_monitor, dt)
        if epoch == start_epoch:
            projected = dt * (epochs - start_epoch)
            LOG.info("projected total: %.1f h at this rate", projected / 3600.0)
            if budget_s and projected > budget_s:
                LOG.warning("that exceeds the %.1f h budget -- this run will "
                            "stop early rather than be killed mid-epoch. Lower "
                            "train.samples_per_image or train.epochs to finish "
                            "the schedule instead.", budget_s / 3600.0)

        ck = {"model": model.state_dict(), "optimizer": opt.state_dict(),
              "scheduler": sched.state_dict(), "epoch": epoch, "best": best,
              "history": history, "config": cfg, "model_kind": model_kind}
        torch.save(ck, Path(out_dir) / f"last_{model_kind}.pt")
        score = va_monitor if monitor_loss == "val_monitor" else va_loss
        if score < best:
            best, best_epoch = score, epoch
            ck["best"] = best
            torch.save(ck, Path(out_dir) / f"best_{model_kind}.pt")
        if budget_s and (time.time() - t_start) > budget_s:
            stop_reason = f"time budget of {budget_s / 3600.0:.1f} h reached"
            LOG.warning("stopping after epoch %d: %s", epoch + 1, stop_reason)
            break
        if best_epoch == epoch:
            pass                     # improved this epoch; patience clock reset
        elif patience > 0 and (epoch - best_epoch) >= patience:
            # The first run of this project trained for 10 epochs against a
            # config default of 60 and stopped while validation was still
            # falling.  Stopping is now decided by the curve, not by whoever
            # edited the cell.
            LOG.info("early stop: no val improvement for %d epochs "
                     "(best %.4f at epoch %d)", patience, best, best_epoch + 1)
            break

    from .visualization import training_curves
    training_curves(history, Path(out_dir) / f"training_curves_{model_kind}.png")
    save_json({"history": history, "best_val_loss": best,
               "best_epoch": best_epoch + 1},
              Path(out_dir) / f"history_{model_kind}.json")
    LOG.info("done. best val loss = %.4f at epoch %d", best, best_epoch + 1)
    if best_epoch + 1 >= epochs and epochs > 1:
        LOG.warning("the best epoch was the last one: validation was still "
                    "improving when the budget ran out. Raise train.epochs.")
    return {"best_val_loss": best, "history": history, "splits": splits,
            "best_epoch": best_epoch + 1, "stop_reason": stop_reason,
            "n_val_specimens": n_val_groups,
            "generalisation_evidence": bool(n_val_groups >= 2)}


def cross_validate(cfg: dict[str, Any], *, n_splits: int = 5,
                   model_kind: str = "full") -> dict[str, Any]:
    """Grouped K-fold over the labelled fields.

    With eleven fields a single held-out pair is not an evaluation, it is an
    anecdote: swap which two images are held out and the numbers move more than
    any change to the model does.  K-fold spends the same images k times and
    reports the spread, which is the only defensible way to quote a result at
    this sample size.  The cost is k training runs, which at a few minutes each
    is nothing.
    """
    out_root = ensure_dir(_sub(cfg, "output", "dir", default="outputs"))
    pcfg = _prior_cfg(cfg)
    records = load_records(cfg["data"]["labels_csv"], cfg["data"]["image_dir"],
                           mask_dir=_sub(cfg, "data", "mask_dir"),
                           prior_cfg=pcfg,
                           prior_cache_dir=Path(out_root) / "prior_cache")
    images = {r.image_id: r.image() for r in records}
    groups = group_near_duplicates(
        images, hamming_max=_sub(cfg, "split", "duplicate_hamming", default=6))
    if _sub(cfg, "split", "group_by_specimen", default=True):
        groups = merge_groups(groups, specimen_groups([r.image_id for r in records]))
    n_groups = len({groups.get(r.image_id, r.image_id) for r in records})
    if n_splits > n_groups:
        LOG.warning("asked for %d folds but there are only %d independent "
                    "group(s); using %d (leave-one-out). More folds than groups "
                    "would put the same specimen in train and validation.",
                    n_splits, n_groups, n_groups)
        n_splits = n_groups
    if n_groups < 4:
        LOG.error("only %d independent group(s): cross-validation here reports "
                  "the variance between %d samples, which is not a confidence "
                  "interval. Report the fold values individually.",
                  n_groups, n_groups)
    folds = grouped_kfold([r.image_id for r in records], n_splits,
                          seed=_sub(cfg, "seed", default=1337), groups=groups)

    results = []
    for k, fold in enumerate(folds):
        LOG.info("=== fold %d/%d: %d train, %d val ===", k + 1, len(folds),
                 len(fold["train"]), len(fold["val"]))
        sub = json.loads(json.dumps(cfg))          # deep copy, config is plain
        sub["output"]["dir"] = str(Path(out_root) / f"fold{k + 1}")
        sub["split"] = dict(sub.get("split", {}), preset=fold)
        res = train(sub, model_kind=model_kind, _forced_splits=fold)
        results.append({"fold": k + 1, "val_images": fold["val"],
                        "best_val_loss": res["best_val_loss"],
                        "best_epoch": res.get("best_epoch")})

    losses = [r["best_val_loss"] for r in results]
    summary = {"n_folds": len(results), "folds": results,
               "val_loss_mean": float(np.mean(losses)),
               "val_loss_sd": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0}
    save_json(summary, Path(out_root) / "cross_validation.json")
    LOG.info("cross-validation: val loss %.4f +/- %.4f over %d folds",
             summary["val_loss_mean"], summary["val_loss_sd"], len(results))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the SEM fiber measurement model")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", choices=("full", "baseline"), default="full")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--kfold", type=int, default=0,
                    help="run grouped K-fold cross-validation instead")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.kfold:
        cross_validate(cfg, n_splits=args.kfold, model_kind=args.model)
    else:
        train(cfg, model_kind=args.model, resume=args.resume)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
