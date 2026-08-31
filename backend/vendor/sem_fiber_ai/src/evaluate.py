"""Prediction, validation-only selection and sealed evaluation (v7).

* :func:`predict_maps` -- whole-image inference (tiled when needed; blending
  keeps it invariant to the tile choice).
* :func:`measure_field` -- maps -> site table -> fibre table on an IMAGE-ONLY
  fibre mask (annotations never shape the mask a field is scored on).
* :func:`evaluate_field` -- manual chords are rolled up with the SAME roll-up
  on the SAME mask, so both sides of every distribution comparison are
  number-weighted fibre records.
* :func:`select_on_validation` -- post-processing thresholds/spacing are chosen
  on validation fields ONLY, by a predefined objective with a recall floor; the
  full grid is written so the choice is auditable.
* :func:`evaluate_split` -- per-field metrics, per-specimen aggregation with
  bootstrap CIs, quality statuses, and leakage assertions.  Test fields are
  evaluated exactly once with the validation-selected settings.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .augmentations import normalize
from .dataset import ImageRecord
from .hardware import Hardware, autocast_dtype, choose_precision, detect, profile_for
from .metrics import (aggregate_by_specimen, distribution_metrics, fiber_level_recall,
                      match_sites, matched_site_metrics, skeleton_coverage)
from .orientation import (fibre_orientation_summary, head_vs_tensor, orientation_summary,
                          sites_orientation_error)
from .postprocess import PostConfig, decode_predictions, maps_from_torch, rejection_summary
from .quality import QualityThresholds, TrainingStats, field_status, ood_warnings
from .rollup import RollupConfig, field_summary, rollup
from .specimens import assert_no_leakage
from .utils import LOG, ensure_dir, save_json


def predict_maps(model, gray: np.ndarray, device, *, hw: Hardware | None = None,
                 tta: bool = False, tile: int | None = None, tile_batch: int | None = None
                 ) -> dict[str, np.ndarray]:
    import torch

    hw = hw or detect()
    prof = profile_for(hw)
    x = torch.from_numpy(normalize(gray, "per_image")[None, None]).float().to(device)
    dtype = autocast_dtype(choose_precision(hw)) if device.type == "cuda" else None
    with torch.no_grad():
        maps = model.predict_tiled(x, tile=int(tile or prof["infer_tile"]),
                                   tile_batch=int(tile_batch or prof["infer_tile_batch"]),
                                   tta=tta, autocast_dtype=dtype)
    return maps_from_torch(maps)


def post_from_cfg(cfg: dict[str, Any], **override) -> PostConfig:
    raw = dict(cfg.get("postprocess") or {})
    raw.update(override)
    return PostConfig(**{k: v for k, v in raw.items() if k in PostConfig.__dataclass_fields__})


def measure_field(model, rec: ImageRecord, post: PostConfig, device, *, hw=None,
                  tta: bool = False, maps: dict[str, np.ndarray] | None = None,
                  rollup_cfg: RollupConfig | None = None) -> dict[str, Any]:
    gray = rec.image()
    maps = maps if maps is not None else predict_maps(model, gray, device, hw=hw, tta=tta)
    prior = rec.prior(image_only=True)
    sites = decode_predictions(maps, image_id=rec.image_id, nm_per_pixel=rec.nm_per_pixel,
                               calibration_valid=rec.calibration_valid, cfg=post,
                               coherency=prior.coherency)
    seg_mask = (1.0 / (1.0 + np.exp(-maps["segment_logit"]))) >= post.seg_threshold
    fibres, sites, info = rollup(sites, seg_mask, rollup_cfg)
    return {"maps": maps, "sites": sites, "fibres": fibres, "rollup_info": info,
            "seg_mask": seg_mask, "prior": prior, "gray": gray}


def gt_fibres(rec: ImageRecord, seg_mask: np.ndarray, rollup_cfg: RollupConfig | None = None):
    """Manual chords rolled up on the same mask, same convention."""
    g = rec.annotations.copy()
    g["confidence"] = 1.0
    g["validity"] = 1.0
    g["rejected_reason"] = ""
    return rollup(g, seg_mask, rollup_cfg)


def evaluate_field(model, rec: ImageRecord, post: PostConfig, device, *, cfg: dict[str, Any],
                   hw=None, tta: bool = False, maps=None, train_stats: TrainingStats | None = None,
                   thresholds: QualityThresholds | None = None) -> dict[str, Any]:
    ev = dict(cfg.get("evaluate") or {})
    m = measure_field(model, rec, post, device, hw=hw, tta=tta, maps=maps)
    sites, fibres, info = m["sites"], m["fibres"], m["rollup_info"]
    acc = sites[sites["rejected_reason"] == ""]
    gt = rec.annotations
    gfib, _gs, ginfo = gt_fibres(rec, m["seg_mask"])
    res: dict[str, Any] = {"image_id": rec.image_id, "specimen": rec.specimen,
                           "n_manual": int(len(gt)), "n_sites": int(len(sites)),
                           "n_sites_accepted": int(len(acc)), "n_fibres": int(len(fibres)),
                           "n_gt_fibres": int(len(gfib)), "roll_up": info,
                           "gt_roll_up": ginfo, "rejections": rejection_summary(sites),
                           "calibration_valid": bool(rec.calibration_valid),
                           "calibration_status": rec.calibration_status,
                           "resample_factor": rec.resample_factor}
    if len(fibres) >= 3 and len(gfib) >= 3:
        res["distribution_fibre_px"] = distribution_metrics(gfib["width_px"], fibres["width_px"])
        if rec.calibration_valid and rec.nm_per_pixel:
            res["distribution_fibre_nm"] = distribution_metrics(
                gfib["width_px"] * rec.nm_per_pixel, fibres["width_px"] * rec.nm_per_pixel)
    if len(acc) >= 3 and len(gt) >= 3:
        res["distribution_sites_px"] = distribution_metrics(gt["width_px"], acc["width_px"])
    if len(acc) and len(gt):
        mt = match_sites(gt, acc, max_distance_scale=float(ev.get("match_distance_scale", 1.5)),
                         min_distance_px=float(ev.get("match_min_distance_px", 8.0)),
                         max_angle_deg=float(ev.get("match_max_angle_deg", 30.0)))
        res["matched_sites"] = matched_site_metrics(mt, len(gt), len(acc))
        res["fiber_level"] = fiber_level_recall(gt, acc)
        res["orientation_sites"] = sites_orientation_error(acc, gt)
    res["coverage"] = skeleton_coverage(acc, m["seg_mask"])
    # orientation deliverable: tensor field, predicted fibres, GT fibres
    tens = orientation_summary(m["gray"], m["seg_mask"])
    ori = {"S_tensor": tens["order_parameter_S"], "mean_angle_tensor_deg": tens["mean_angle_deg"],
           "coherent_fraction": tens["coherent_fraction"]}
    if len(fibres):
        f = fibre_orientation_summary(fibres)
        ori["S_pred_fibre"], ori["mean_angle_pred_fibre_deg"] = f["order_parameter_S"], f["mean_angle_deg"]
    if len(gfib):
        f = fibre_orientation_summary(gfib)
        ori["S_gt_fibre"], ori["mean_angle_gt_fibre_deg"] = f["order_parameter_S"], f["mean_angle_deg"]
    if len(acc):
        ori["head_vs_tensor"] = head_vs_tensor(acc, m["gray"], m["seg_mask"])
    res["orientation"] = ori
    # quality
    thr = thresholds or QualityThresholds()
    from .fiber_prior import mask_separability_auc

    auc = mask_separability_auc(m["gray"], m["seg_mask"], polarity=m["prior"].polarity)
    bd = sites["boundary_disagreement"].to_numpy(float) if len(sites) else np.array([])
    bd = bd[np.isfinite(bd)]
    warns, z = ood_warnings(m["gray"], train_stats, rec.nm_per_pixel, thr)
    res["quality"] = field_status(
        seg_area=float(m["seg_mask"].mean()), separability_auc=float(auc) if np.isfinite(auc) else None,
        coherent_fraction=float(tens["coherent_fraction"]),
        boundary_agreement=float((bd <= post.boundary_tol).mean()) if bd.size else None,
        unassigned_fraction=info.get("unassigned_fraction"), n_fibres=int(len(fibres)),
        accepted_fraction=float(len(acc) / max(len(sites), 1)), ood_z=z, ood=warns,
        calibration_valid=bool(rec.calibration_valid), calibration_reason=rec.calibration_status,
        thr=thr)
    res["field_summary"] = field_summary(fibres, sites, info, nm_valid=bool(rec.calibration_valid))
    return {"metrics": res, "sites": sites, "fibres": fibres, "gt_fibres": gfib, "maps": m["maps"],
            "seg_mask": m["seg_mask"], "gray": m["gray"]}


# --------------------------------------------------------------------------- #
def select_on_validation(model, val_records: Sequence[ImageRecord], cfg: dict[str, Any], device,
                         *, hw=None, tta: bool = False, out_json: str | Path | None = None
                         ) -> dict[str, Any]:
    """Grid over (spacing, min_validity, seg_threshold) on VALIDATION fields only."""
    sel = dict(cfg.get("selection") or {})
    objective = str(sel.get("objective", "fiber_wasserstein_relative"))
    floor_frac = float(sel.get("min_recall_frac", 0.85))
    cached = [(rec, predict_maps(model, rec.image(), device, hw=hw, tta=tta)) for rec in val_records]
    rows = []
    grid = list(itertools.product(sel.get("spacing_grid", [12.0]), sel.get("validity_grid", [0.3]),
                                  sel.get("seg_grid", [0.5])))
    for spacing, minval, segthr in grid:
        post = post_from_cfg(cfg, spacing_px=float(spacing), min_validity=float(minval),
                             seg_threshold=float(segthr))
        vals: dict[str, list[float]] = {}
        for rec, maps in cached:
            r = evaluate_field(model, rec, post, device, cfg=cfg, hw=hw, maps=maps)["metrics"]
            d = r.get("distribution_fibre_px", {})
            vals.setdefault("fiber_wasserstein_relative", []).append(d.get("wasserstein_relative", np.nan))
            vals.setdefault("fiber_sd_ratio", []).append(d.get("sd_ratio", np.nan))
            vals.setdefault("fiber_p90_ratio", []).append(d.get("p90_ratio", np.nan))
            vals.setdefault("fiber_recall", []).append(r.get("fiber_level", {}).get("fiber_recall", np.nan))
            vals.setdefault("n_fibres", []).append(r.get("n_fibres", 0))
            vals.setdefault("n_sites_accepted", []).append(r.get("n_sites_accepted", 0))
        row = {"spacing_px": float(spacing), "min_validity": float(minval), "seg_threshold": float(segthr)}
        for k, v in vals.items():
            a = np.asarray(v, float)
            row[k] = float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
        rows.append(row)
    import pandas as pd

    df = pd.DataFrame(rows)
    best_recall = float(np.nanmax(df["fiber_recall"])) if df["fiber_recall"].notna().any() else np.nan
    floor = floor_frac * best_recall if np.isfinite(best_recall) else 0.0
    ok = df[df["fiber_recall"] >= floor] if np.isfinite(best_recall) else df
    if not len(ok):
        ok = df
    col = objective if objective in ok.columns else "fiber_wasserstein_relative"
    if col in ("fiber_sd_ratio", "fiber_p90_ratio"):
        score = (ok[col] - 1.0).abs()
    else:
        score = ok[col]
    best = ok.loc[score.idxmin()] if score.notna().any() else ok.iloc[0]
    choice = {"spacing_px": float(best["spacing_px"]), "min_validity": float(best["min_validity"]),
              "seg_threshold": float(best["seg_threshold"]), "objective": objective,
              "objective_value": float(best[col]), "recall_floor": float(floor),
              "fiber_recall": float(best["fiber_recall"]),
              "selected_on_split": "val",
              "selected_on": [r.image_id for r in val_records],
              "n_val_specimens": len({r.specimen for r in val_records}),
              "grid": df.to_dict(orient="records")}
    at_edge = {k: float(best[k]) in (min(cfg["selection"][g]), max(cfg["selection"][g]))
               for k, g in (("spacing_px", "spacing_grid"), ("min_validity", "validity_grid"),
                            ("seg_threshold", "seg_grid"))}
    choice["at_grid_edge"] = at_edge
    LOG.info("validation selection (%s): spacing %.0f px, min_validity %.2f, seg %.2f -> %.4f "
             "(recall %.3f >= floor %.3f)", objective, choice["spacing_px"], choice["min_validity"],
             choice["seg_threshold"], choice["objective_value"], choice["fiber_recall"], floor)
    if out_json:
        save_json(choice, out_json)
    return choice


# --------------------------------------------------------------------------- #
def evaluate_split(model, records: Sequence[ImageRecord], split: dict[str, Any], part: str,
                   post: PostConfig, cfg: dict[str, Any], device, out_dir: str | Path, *,
                   hw=None, tta: bool = False, train_stats: TrainingStats | None = None,
                   save_figures: bool = True, label: str = "") -> dict[str, Any]:
    assert_no_leakage(split)
    out_dir = ensure_dir(out_dir)
    wanted = list(split.get(part, []))
    recs = [r for r in records if r.image_id in set(wanted)]
    if not recs:
        raise RuntimeError(f"split '{part}' has no records")
    if part == "test" and (set(wanted) & set(split.get("train", []))):
        raise RuntimeError("test/train overlap -- refusing to evaluate")
    per_field: dict[str, Any] = {}
    all_sites, all_fibres = [], []
    for rec in recs:
        r = evaluate_field(model, rec, post, device, cfg=cfg, hw=hw, tta=tta, train_stats=train_stats)
        per_field[rec.image_id] = r["metrics"]
        s = r["sites"].copy()
        s["split"] = part
        s["quality_status"] = r["metrics"]["quality"]["status"]
        f = r["fibres"].copy()
        f.insert(0, "image_id", rec.image_id)
        f["split"] = part
        f["quality_status"] = r["metrics"]["quality"]["status"]
        f["calibration_valid"] = bool(rec.calibration_valid)
        all_sites.append(s)
        all_fibres.append(f)
        s.to_csv(out_dir / f"{rec.image_id}_sites.csv", index=False)
        f.to_csv(out_dir / f"{rec.image_id}_fibres.csv", index=False)
        r["gt_fibres"].to_csv(out_dir / f"{rec.image_id}_gt_fibres.csv", index=False)
        if save_figures:
            try:
                from .visualization import overlay_figure, width_distribution_figure

                overlay_figure(r["gray"], rec.annotations, r["sites"], out_dir / f"{rec.image_id}_overlay.png",
                               title=f"{rec.image_id} [{part}] {r['metrics']['quality']['status']}")
                width_distribution_figure(r["gt_fibres"], r["fibres"], rec.nm_per_pixel if rec.calibration_valid else None,
                                          out_dir / f"{rec.image_id}_width_distribution.png",
                                          title=rec.image_id)
            except Exception as exc:                        # noqa: BLE001
                LOG.warning("%s: figure failed (%s)", rec.image_id, exc)
    import pandas as pd

    if all_sites:
        pd.concat(all_sites, ignore_index=True).to_csv(out_dir / f"{part}_all_sites.csv", index=False)
        pd.concat(all_fibres, ignore_index=True).to_csv(out_dir / f"{part}_all_fibres.csv", index=False)
    ev = dict(cfg.get("evaluate") or {})
    specimen_of = {r.image_id: r.specimen for r in recs}
    agg = aggregate_by_specimen(per_field, specimen_of, n_boot=int(ev.get("bootstrap", 1000)),
                                min_groups_for_ci=int(ev.get("min_groups_for_ci", 5)))
    statuses = {k: v["quality"] for k, v in per_field.items()}
    pass_fields = [k for k, v in statuses.items() if v["status"] == "PASS"]
    agg_pass = aggregate_by_specimen({k: per_field[k] for k in pass_fields}, specimen_of,
                                     n_boot=int(ev.get("bootstrap", 1000)),
                                     min_groups_for_ci=int(ev.get("min_groups_for_ci", 5))) \
        if pass_fields else None
    result = {"part": part, "label": label, "post": asdict(post), "per_field": per_field,
              "aggregate_all_fields": agg, "aggregate_pass_only": agg_pass,
              "quality": statuses, "pass_fields": pass_fields,
              "n_fields": len(recs), "n_specimens": len(set(specimen_of.values())),
              "split_digest": split.get("digest")}
    save_json(result, out_dir / f"metrics_{part}.json")
    return result
