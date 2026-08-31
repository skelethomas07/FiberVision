"""Measure new SEM images with a trained v7 model.

Policy
* The model checkpoint fixes the post-processing settings that were SELECTED
  ON VALIDATION (``selection.json`` in the run dir); nothing is re-tuned here.
* Pixel-unit results are always produced.  Nanometre results exist only for
  images whose calibration is ``calibration_valid`` (manual table entry, or a
  footer read that passed the audit) -- otherwise ``width_nm`` is NaN and the
  reason is recorded.
* Images that were part of the training/validation/test split are refused by
  default (measuring them again would be evaluated on training data).
* Every image gets a PASS/REVIEW/FAIL status; the batch summary uses PASS
  images only unless ``include_review=True``.
* Physical resampling: if the model was trained at a reference nm/px, the
  image is resampled to it BEFORE inference and results are mapped back.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .calibration import resolve_calibration, strip_footer
from .calib_audit import audit_field
from .checkpoint import load_checkpoint
from .dataset import ImageRecord
from .evaluate import measure_field, post_from_cfg, predict_maps
from .hardware import detect
from .metrics import skeleton_coverage
from .models.fiber_net import build_model
from .orientation import fibre_orientation_summary, orientation_summary
from .physical import plan_resample, unresample_predictions
from .postprocess import PostConfig, rejection_summary
from .quality import QualityThresholds, TrainingStats, field_status, ood_warnings
from .rollup import field_summary
from .utils import LOG, ensure_dir, image_id_from_path, list_images, pick_device, read_gray, save_json


def load_run(run_dir: str | Path, device=None) -> dict[str, Any]:
    """Load best.pt + the validation-selected post-processing + training stats."""
    run_dir = Path(run_dir)
    ck = load_checkpoint(run_dir / "best.pt", map_location="cpu")
    cfg = ck["config"]
    run_mode = ck.get("run_mode", "?")
    proto_key = "smoke_protocol" if run_mode == "FAST_SMOKE_TEST" else "protocol"
    model_cfg = dict(cfg.get("model") or {})
    if run_mode == "FAST_SMOKE_TEST":
        model_cfg.update(cfg.get("smoke_model") or {})
    model = build_model(model_cfg)
    model.load_state_dict(ck["model"])
    device = device or pick_device("cuda" if detect().device == "cuda" else "cpu")
    model.to(device).eval()
    sel_path = run_dir / "selection.json"
    selection = json.loads(sel_path.read_text(encoding="utf-8")) if sel_path.exists() else None
    if selection:
        post = post_from_cfg(cfg, spacing_px=selection["spacing_px"],
                             min_validity=selection["min_validity"],
                             seg_threshold=selection["seg_threshold"])
    else:
        LOG.warning("no selection.json in %s: using config defaults for post-processing "
                    "(they were NOT selected on validation)", run_dir)
        post = post_from_cfg(cfg)
    ts_path = run_dir / "training_stats.json"
    train_stats = TrainingStats.from_dict(json.loads(ts_path.read_text(encoding="utf-8"))) \
        if ts_path.exists() else None
    ref_path = run_dir / "physical_reference.json"
    reference = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else None
    split = ck.get("split_manifest") or {}
    members = set(split.get("train", [])) | set(split.get("val", [])) | set(split.get("test", []))
    return {"model": model, "cfg": cfg, "post": post, "device": device, "train_stats": train_stats,
            "reference_nm_per_px": (reference or {}).get("reference_nm_per_px"),
            "split_members": members, "checkpoint": {"epoch": ck.get("epoch"), "best": ck.get("best"),
                                                     "protocol_digest": ck.get("protocol_digest"),
                                                     "package_version": ck.get("package_version"),
                                                     "run_mode": run_mode, "protocol": proto_key},
            "selection": selection}


def calibrate_new_image(path: Path, gray_full: np.ndarray, *, manual_nm_per_px: float | None,
                        calib_table: dict[str, float] | None = None) -> dict[str, Any]:
    iid = image_id_from_path(path)
    table = dict(calib_table or {})
    if manual_nm_per_px is not None:
        table[iid] = float(manual_nm_per_px)
    cal = resolve_calibration(path, gray_full, image_id=iid, override=table.get(iid))
    audit = audit_field(iid, physical_nm_per_px=cal.nm_per_pixel if cal.known else None,
                        physical_source=cal.source, physical_detail=cal.detail,
                        manual_nm_per_px=table.get(iid))
    return {"calibration": cal.to_dict(), "audit": audit.to_dict(),
            "nm_per_px": audit.nm_per_px, "valid": bool(audit.calibration_valid),
            "reason": audit.reason}


def measure_image(run: dict[str, Any], path: str | Path, out_dir: str | Path, *,
                  manual_nm_per_px: float | None = None, calib_table: dict[str, float] | None = None,
                  include_split_members: bool = False, tta: bool = False,
                  thresholds: QualityThresholds | None = None, save_maps: bool = False,
                  thick: bool = False) -> dict[str, Any]:
    path = Path(path)
    out_dir = ensure_dir(out_dir)
    iid = image_id_from_path(path)
    if iid in run["split_members"] and not include_split_members:
        raise RuntimeError(f"{iid} belongs to the training/validation/test split of this model; "
                           "measuring it would not be an independent result. Pass "
                           "include_split_members=True only for diagnostics.")
    gray_full = read_gray(path)
    cal = calibrate_new_image(path, gray_full, manual_nm_per_px=manual_nm_per_px,
                              calib_table=calib_table)
    body, footer_row = strip_footer(gray_full)
    ref = run.get("reference_nm_per_px")
    plan = plan_resample(iid, body.shape, cal["nm_per_px"] if cal["valid"] else None, ref,
                         calibration_valid=cal["valid"]) if ref else None
    factor = float(plan.factor_applied) if (plan and plan.included and plan.factor_applied) else 1.0
    rec = ImageRecord(iid, path, _empty_labels(), cal["nm_per_px"] if cal["valid"] else None,
                      cal["valid"], cal["reason"], iid, factor)
    m = measure_field(run["model"], rec, run["post"], run["device"], tta=tta)
    sites, fibres, info = m["sites"], m["fibres"], m["rollup_info"]
    if abs(factor - 1.0) > 1e-9:
        sites = unresample_predictions(sites, factor)
        fibres = fibres.copy()
        fibres["width_px"] = fibres["width_px"] / factor
        fibres["length_px"] = fibres["length_px"] / factor
        if cal["valid"] and cal["nm_per_px"]:
            fibres["width_nm"] = fibres["width_px"] * cal["nm_per_px"]
            fibres["length_nm"] = fibres["length_px"] * cal["nm_per_px"]
    acc = sites[sites["rejected_reason"] == ""]
    thr = thresholds or QualityThresholds()
    from .fiber_prior import mask_separability_auc

    auc = mask_separability_auc(m["gray"], m["seg_mask"], polarity=m["prior"].polarity)
    bd = sites["boundary_disagreement"].to_numpy(float) if len(sites) else np.array([])
    bd = bd[np.isfinite(bd)]
    tens = orientation_summary(m["gray"], m["seg_mask"])
    warns, z = ood_warnings(m["gray"], run.get("train_stats"), cal["nm_per_px"], thr)
    status = field_status(seg_area=float(m["seg_mask"].mean()),
                          separability_auc=float(auc) if np.isfinite(auc) else None,
                          coherent_fraction=float(tens["coherent_fraction"]),
                          boundary_agreement=float((bd <= run["post"].boundary_tol).mean()) if bd.size else None,
                          unassigned_fraction=info.get("unassigned_fraction"), n_fibres=int(len(fibres)),
                          accepted_fraction=float(len(acc) / max(len(sites), 1)), ood_z=z, ood=warns,
                          calibration_valid=cal["valid"], calibration_reason=cal["reason"], thr=thr)
    summary = field_summary(fibres, sites, info, nm_valid=cal["valid"])
    summary["orientation_tensor"] = tens
    summary["orientation_fibres"] = fibre_orientation_summary(fibres) if len(fibres) else None
    summary["coverage"] = skeleton_coverage(acc, m["seg_mask"])
    result = {"image_id": iid, "path": str(path), "footer_row": footer_row,
              "calibration": cal, "resample": asdict(plan) if plan else None,
              "quality": status, "rejections": rejection_summary(sites), "summary": summary,
              "n_sites": int(len(sites)), "n_sites_accepted": int(len(acc)), "n_fibres": int(len(fibres)),
              "post": asdict(run["post"]), "checkpoint": run["checkpoint"]}
    sites.to_csv(out_dir / f"{iid}_sites.csv", index=False)
    fibres.to_csv(out_dir / f"{iid}_fibres.csv", index=False)
    save_json(result, out_dir / f"{iid}_summary.json")
    try:
        from .visualization import maps_panel, overlay_figure, width_distribution_figure

        overlay_figure(m["gray"], None, sites, out_dir / f"{iid}_overlay.png",
                       title=f"{iid} {status['status']} nm={status['nm_status']}")
        import pandas as pd

        width_distribution_figure(pd.DataFrame(columns=["width_px"]), fibres,
                                  cal["nm_per_px"] if cal["valid"] else None,
                                  out_dir / f"{iid}_width_distribution.png", title=iid)
        if save_maps:
            maps_panel(m["gray"], m["maps"], out_dir / f"{iid}_maps.png", title=iid)
    except Exception as exc:                                # noqa: BLE001
        LOG.warning("%s: figure failed (%s)", iid, exc)
    if thick:
        from .thick_experimental import measure_thick_from_maps

        t = measure_thick_from_maps(m["maps"], m["gray"], run["cfg"], image_id=iid,
                                    nm_per_px=cal["nm_per_px"] if cal["valid"] else None)
        t["table"].to_csv(out_dir / f"{iid}_thick_EXPERIMENTAL.csv", index=False)
        result["thick_experimental"] = t["summary"]
    return result


def _empty_labels():
    from .labels import empty_labels

    return empty_labels()


def measure_folder(run: dict[str, Any], image_dir: str | Path, out_dir: str | Path, *,
                   calib_table: dict[str, float] | None = None, include_review: bool = False,
                   **kw) -> dict[str, Any]:
    import pandas as pd

    out_dir = ensure_dir(out_dir)
    paths = list_images(image_dir)
    results, fibre_frames = [], []
    for p in paths:
        try:
            r = measure_image(run, p, out_dir, calib_table=calib_table, **kw)
        except RuntimeError as exc:
            LOG.warning("%s skipped: %s", p.name, exc)
            results.append({"image_id": image_id_from_path(p), "skipped": str(exc)})
            continue
        results.append(r)
        f = pd.read_csv(out_dir / f"{r['image_id']}_fibres.csv")
        f.insert(0, "image_id", r["image_id"])
        f["quality_status"] = r["quality"]["status"]
        f["calibration_valid"] = r["calibration"]["valid"]
        fibre_frames.append(f)
    batch: dict[str, Any] = {"n_images": len(paths), "results": results}
    if fibre_frames:
        allf = pd.concat(fibre_frames, ignore_index=True)
        allf.to_csv(out_dir / "all_fibres.csv", index=False)
        ok = {"PASS"} | ({"REVIEW"} if include_review else set())
        use = allf[allf["quality_status"].isin(ok)]
        from .rollup import distribution_summary

        batch["publication_set"] = sorted(use["image_id"].unique().tolist())
        batch["excluded"] = sorted(set(allf["image_id"]) - set(batch["publication_set"]))
        batch["number_weighted_px"] = distribution_summary(use["width_px"].to_numpy(float))
        nm_ok = use[use["calibration_valid"] & use.get("width_nm", pd.Series(dtype=float)).notna()] \
            if "width_nm" in use.columns else use.iloc[0:0]
        batch["number_weighted_nm"] = (distribution_summary(nm_ok["width_nm"].to_numpy(float))
                                       if len(nm_ok) else {"n": 0, "reason": "no calibration-valid PASS images"})
        batch["per_image_number_weighted_px"] = {
            iid: distribution_summary(g["width_px"].to_numpy(float)) for iid, g in use.groupby("image_id")}
        batch["note"] = ("pooled distributions treat every fibre from every PASS image equally; "
                         "per-image values are the unit for specimen comparisons")
    save_json(batch, out_dir / "batch_summary.json")
    return batch
