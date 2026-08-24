"""Inference on a single image or a folder of images.

Calibration policy is strict: nanometre outputs appear only when the pixel size
was established from the file or supplied by the user.  Otherwise ``width_nm``
is NaN and the summary says so, because a plausible-looking wrong scale is worse
than an honest missing one.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .augmentations import normalize
from .calibration import load_calibration_table, resolve_calibration, strip_footer
from .postprocess import PostConfig, decode_predictions
from .utils import (ensure_dir, get_logger, list_images, pick_device, read_gray,
                    save_json, set_seed)
from .visualization import (distribution_figure, draw_measurements,
                            prediction_panel, to_rgb)

LOG = get_logger(__name__)


def load_checkpoint(path: str | Path, device: Any):
    import torch

    from .models.fiber_measurement_net import build_model

    ck = torch.load(path, map_location=device)
    cfg = ck.get("config", {})
    model = build_model(cfg.get("model", {}))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, ck


def predict_image(model, gray: np.ndarray, device, *, tile: int = 512,
                  overlap: int = 64, tta: bool = False, mc_samples: int = 0,
                  norm: str = "per_image") -> dict[str, np.ndarray]:
    import torch

    x = torch.from_numpy(normalize(gray, norm)[None, None]).float().to(device)
    with torch.no_grad():
        maps = model.predict_tiled(x, tile=tile, overlap=overlap, tta=tta,
                                   mc_samples=mc_samples)
    return {k: v[0].cpu().numpy() if v.shape[1] > 1 else v[0, 0].cpu().numpy()
            for k, v in maps.items()}


def run_one(model, image_path: Path, out_dir: Path, device, *,
            nm_per_pixel: float | None, calib_table: dict[str, float],
            post: PostConfig, tile: int, overlap: int, tta: bool,
            mc_samples: int, save_maps: bool) -> "Any":
    import pandas as pd

    gray_full = read_gray(image_path)
    gray, footer_row = strip_footer(gray_full)
    calib = resolve_calibration(image_path, gray_full, override=nm_per_pixel,
                                table=calib_table)
    maps = predict_image(model, gray, device, tile=tile, overlap=overlap,
                         tta=tta, mc_samples=mc_samples)
    df = decode_predictions(maps, image_id=image_path.stem,
                            nm_per_pixel=calib.nm_per_pixel if calib.known else None,
                            cfg=post)

    ensure_dir(out_dir)
    stem = image_path.stem
    df.to_csv(out_dir / f"{stem}_predictions.csv", index=False)

    canvas = draw_measurements(to_rgb(gray), df, show_ids=True,
                               value_col="width_nm" if calib.known else "width_px")
    import cv2
    cv2.imwrite(str(out_dir / f"{stem}_annotated.png"),
                cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    if save_maps:
        prediction_panel(gray, {
            "centre heatmap": 1 / (1 + np.exp(-maps["center_logit"])),
            "fiber segmentation": 1 / (1 + np.exp(-maps["segment_logit"])),
            "width (px)": np.exp(maps["width"]),
            "orientation (deg)": np.rad2deg(0.5 * np.arctan2(maps["orient"][1],
                                                             maps["orient"][0])),
            "validity": 1 / (1 + np.exp(-maps["validity_logit"])),
            "uncertainty (sigma)": np.exp(0.5 * maps["logvar"]),
        }, out_dir / f"{stem}_maps.png", title=stem)

    unit = "nm" if calib.known else "px"
    col = "width_nm" if calib.known else "width_px"
    summary: dict[str, Any] = {
        "image": str(image_path), "n_predictions": int(len(df)),
        "calibration": calib.to_dict(), "footer_row": footer_row,
        "units": unit,
    }
    if len(df):
        w = df[col].to_numpy(float)
        w = w[np.isfinite(w)]
        if w.size:
            summary["thickness"] = {
                "mean": float(w.mean()), "median": float(np.median(w)),
                "std": float(w.std(ddof=1)) if w.size > 1 else 0.0,
                **{f"p{q}": float(np.percentile(w, q)) for q in (5, 25, 75, 95)},
            }
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=140)
            ax.hist(w, bins=40)
            ax.set_xlabel(f"predicted fiber thickness ({unit})")
            ax.set_ylabel("count")
            ax.set_title(f"{stem}: n={w.size}, median={np.median(w):.1f} {unit}",
                         fontsize=9)
            fig.tight_layout()
            fig.savefig(out_dir / f"{stem}_thickness_histogram.png")
            plt.close(fig)
    if not calib.known:
        summary["warning"] = ("pixel size unknown: thickness is reported in pixels "
                              "only and width_nm is NaN")
    save_json(summary, out_dir / f"{stem}_summary.json")
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Predict fiber measurements on SEM images")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--image_dir", default=None)
    ap.add_argument("--output_dir", default="predictions")
    ap.add_argument("--nm_per_pixel", type=float, default=None)
    ap.add_argument("--calibration_table", default=None)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--mc_samples", type=int, default=0)
    ap.add_argument("--peak_threshold", type=float, default=0.30)
    ap.add_argument("--min_validity", type=float, default=0.3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save_maps", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(argv)

    if not args.image and not args.image_dir:
        ap.error("give --image or --image_dir")
    set_seed(args.seed)
    device = pick_device(args.device)
    model, _ck = load_checkpoint(args.checkpoint, device)
    table = load_calibration_table(args.calibration_table)
    post = PostConfig(peak_threshold=args.peak_threshold,
                      min_validity=args.min_validity)
    out_dir = ensure_dir(args.output_dir)

    paths = [Path(args.image)] if args.image else list_images(args.image_dir)
    LOG.info("running inference on %d image(s) with device=%s", len(paths), device)
    frames = []
    for p in paths:
        try:
            frames.append(run_one(model, p, out_dir, device,
                                  nm_per_pixel=args.nm_per_pixel, calib_table=table,
                                  post=post, tile=args.tile, overlap=args.overlap,
                                  tta=args.tta, mc_samples=args.mc_samples,
                                  save_maps=args.save_maps))
        except Exception as exc:  # noqa: BLE001
            LOG.exception("failed on %s: %s", p, exc)
    if frames:
        import pandas as pd
        allp = pd.concat(frames, ignore_index=True)
        allp.to_csv(Path(out_dir) / "all_predictions.csv", index=False)
        LOG.info("wrote %d predictions across %d image(s)", len(allp), len(frames))
    return 0



def run_one(model, image_path: Path, out_dir: Path, device, *,
            nm_per_pixel: float | None, calib_table: dict, post: PostConfig,
            tile: int, overlap: int, tta: bool, mc_samples: int,
            save_maps: bool, width_calib=None, zoom_panels: int = 4,
            thick_cfg=None):
    """Run the learned detector, then optionally recover missed wide fibres.

    The learned network remains the primary detector.  ``thick_cfg`` activates a
    classical supplement that uses the model's dense segmentation only as weak
    support, while width itself comes from the SEM image (scale bank + medial EDT
    + cross-fibre intensity profile).
    """
    import cv2
    import pandas as pd

    from .visualization import measurement_zoom_panel

    gray_full = read_gray(image_path)
    gray, footer_row = strip_footer(gray_full)
    calib = resolve_calibration(image_path, gray_full, override=nm_per_pixel,
                                table=calib_table)
    maps = predict_image(model, gray, device, tile=tile, overlap=overlap,
                         tta=tta, mc_samples=mc_samples)

    # ----------------------------- learned branch ------------------------- #
    df = decode_predictions(
        maps, image_id=image_path.stem,
        nm_per_pixel=calib.nm_per_pixel if calib.known else None,
        cfg=post,
    )
    df["measurement_source"] = "ai"
    df["recovered_thick"] = False
    df["scale_sigma_px"] = np.nan
    df["edt_width_px"] = np.nan
    df["profile_width_px"] = np.nan
    df["profile_contrast"] = np.nan
    df["measurement_method"] = "network"

    if width_calib and width_calib.get("fitted") and len(df):
        from .width_calibration import apply_width_calibration
        df["width_px_raw"] = df["width_px"]
        df["width_px"] = apply_width_calibration(
            df["width_px"].to_numpy(float), width_calib)
        df["width_nm"] = (df["width_px"] * calib.nm_per_pixel
                          if calib.known else np.nan)
        df["width_calibrated"] = True
    else:
        df["width_calibrated"] = False

    # -------------------------- thick-fibre branch ------------------------ #
    thick_diag = None
    thick_stats = {
        "n_ai_input": int(len(df)),
        "n_ai_replaced": 0,
        "n_thick_candidates": 0,
        "n_thick_added": 0,
        "n_combined": int(len(df)),
    }
    if thick_cfg is not None and getattr(thick_cfg, "enabled", False):
        from .thick_fiber import recover_thick_measurements, merge_with_ai

        seg_prob = 1.0 / (1.0 + np.exp(-maps["segment_logit"]))
        thick_df, thick_diag = recover_thick_measurements(
            gray,
            image_id=image_path.stem,
            nm_per_pixel=calib.nm_per_pixel if calib.known else None,
            segment_prob=seg_prob,
            cfg=thick_cfg,
        )
        df, thick_stats = merge_with_ai(df, thick_df, thick_cfg)
        LOG.info(
            "%s thick recovery: %d classical candidates -> %d added; "
            "%d narrow same-fibre AI flank detections replaced",
            image_path.stem,
            thick_stats["n_thick_candidates"],
            thick_stats["n_thick_added"],
            thick_stats["n_ai_replaced"],
        )

    ensure_dir(out_dir)
    stem = image_path.stem
    df.to_csv(out_dir / f"{stem}_predictions.csv", index=False)

    # Amber = learned AI, red = classical wide-fibre recovery.
    source_colors = {
        "ai": (255, 210, 60),
        "thick_recovery": (255, 70, 70),
    }
    value_col = "width_nm" if calib.known else "width_px"
    canvas = draw_measurements(
        to_rgb(gray), df, show_ids=False,
        source_colors=source_colors,
    )
    cv2.imwrite(str(out_dir / f"{stem}_annotated.png"),
                cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    if zoom_panels:
        try:
            measurement_zoom_panel(
                gray, df, out_dir / f"{stem}_zoom.png",
                n_panels=int(zoom_panels), title=stem,
                value_col=value_col,
                source_colors=source_colors,
                prefer_source=("thick_recovery"
                               if thick_stats["n_thick_added"] else None),
            )
        except Exception as exc:  # noqa: BLE001
            LOG.warning("%s: zoom panel failed (%s)", stem, exc)

    if save_maps:
        panel_maps = {
            "centre heatmap": 1 / (1 + np.exp(-maps["center_logit"])),
            "fiber segmentation": 1 / (1 + np.exp(-maps["segment_logit"])),
            "AI width (px)": np.exp(maps["width"]),
            "orientation (deg)": np.rad2deg(
                0.5 * np.arctan2(maps["orient"][1], maps["orient"][0])),
            "validity": 1 / (1 + np.exp(-maps["validity_logit"])),
            "uncertainty (sigma)": np.exp(0.5 * maps["logvar"]),
        }
        if thick_diag is not None:
            panel_maps.update({
                "classical local scale sigma (px)": thick_diag["scale_sigma"],
                "wide candidate skeleton": thick_diag["candidate_mask"].astype(float),
                "EDT body width (px)": thick_diag["width_map"],
            })
        prediction_panel(gray, panel_maps,
                         out_dir / f"{stem}_maps.png", title=stem)

    # ------------------------------- summary ------------------------------ #
    unit = "nm" if calib.known else "px"
    col = "width_nm" if calib.known else "width_px"
    summary = {
        "image": str(image_path),
        "n_predictions": int(len(df)),
        "calibration": calib.to_dict(),
        "footer_row": footer_row,
        "units": unit,
        "width_calibration": width_calib if width_calib else None,
        "measurement_sources": (
            {str(k): int(v) for k, v in
             df["measurement_source"].value_counts().to_dict().items()}
            if len(df) and "measurement_source" in df.columns else {}
        ),
    }

    if thick_cfg is not None and getattr(thick_cfg, "enabled", False):
        cfg_dict = {
            "min_width_px": float(thick_cfg.min_width_px),
            "max_width_px": float(thick_cfg.max_width_px),
            "min_sigma": float(thick_cfg.min_sigma),
            "spacing_px": float(thick_cfg.spacing_px),
            "min_ridge_coherence": float(thick_cfg.min_ridge_coherence),
            "segment_support": float(thick_cfg.segment_support),
            "sigmas": [float(x) for x in thick_cfg.sigmas],
        }
        diag_scalars = {}
        if thick_diag is not None:
            for k in ("n_candidate_pixels", "n_sampled_sites", "n_accepted",
                      "n_profile_fallback", "n_profile_ratio_rejected",
                      "response_floor"):
                v = thick_diag.get(k)
                if isinstance(v, (np.integer, int)):
                    diag_scalars[k] = int(v)
                elif isinstance(v, (np.floating, float)):
                    diag_scalars[k] = float(v)
        summary["thick_recovery"] = {
            "enabled": True, "config": cfg_dict,
            **{k: int(v) for k, v in thick_stats.items()},
            **diag_scalars,
        }
    else:
        summary["thick_recovery"] = {"enabled": False}

    if len(df):
        w = df[col].to_numpy(float)
        w = w[np.isfinite(w)]
        if w.size:
            summary["thickness"] = {
                "mean": float(w.mean()),
                "median": float(np.median(w)),
                "std": float(w.std(ddof=1)) if w.size > 1 else 0.0,
                **{f"p{q}": float(np.percentile(w, q))
                   for q in (5, 25, 75, 95)},
            }

            # Combined distribution plus the recovered wide tail as a separate
            # outline.  Same bins = no visual trick from different bin edges.
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5.8, 3.5), dpi=140)
            finite_df = df[np.isfinite(df[col].to_numpy(float))].copy()
            vals = finite_df[col].to_numpy(float)
            bins = np.histogram_bin_edges(vals, bins=40)
            ai_vals = finite_df.loc[
                finite_df["measurement_source"].astype(str) == "ai", col
            ].to_numpy(float)
            tr_vals = finite_df.loc[
                finite_df["measurement_source"].astype(str) == "thick_recovery", col
            ].to_numpy(float)
            if ai_vals.size:
                ax.hist(ai_vals, bins=bins, alpha=0.65, label="AI retained")
            if tr_vals.size:
                ax.hist(tr_vals, bins=bins, histtype="step", linewidth=2.0,
                        label="recovered thick")
            if not ai_vals.size and vals.size:
                ax.hist(vals, bins=bins)
            ax.set_xlabel(f"fiber thickness ({unit})")
            ax.set_ylabel("count")
            title = f"{stem}: n={vals.size}, median={np.median(vals):.1f} {unit}"
            if tr_vals.size:
                title += f" | +{tr_vals.size} thick"
                ax.legend(fontsize=8)
            ax.set_title(title, fontsize=9)
            fig.tight_layout()
            fig.savefig(out_dir / f"{stem}_thickness_histogram.png")
            plt.close(fig)

    if not calib.known:
        summary["warning"] = (
            "pixel size unknown: thickness is in pixels only and width_nm is NaN")
    save_json(summary, out_dir / f"{stem}_summary.json")
    return df


# --------------------------------------------------------------------------- #
# v6.6 inference CLI: v6.5 + optional classical wide-fibre recovery.
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    import json as _json
    from pathlib import Path as _P

    ap = argparse.ArgumentParser(description="Predict fiber measurements on SEM images")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--image_dir", default=None)
    ap.add_argument("--output_dir", default="predictions")
    ap.add_argument("--nm_per_pixel", type=float, default=None)
    ap.add_argument("--calibration_table", default=None)
    ap.add_argument("--width_calibration", default=None,
                    help="JSON written by width_calibration.fit_on_split")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--mc_samples", type=int, default=0)
    ap.add_argument("--peak_threshold", type=float, default=0.30)
    ap.add_argument("--min_validity", type=float, default=0.3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save_maps", action="store_true")
    ap.add_argument("--zoom_panels", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)

    # The recovery is opt-in at the CLI so evaluation of an old pure-AI run is
    # never silently changed.  Notebook cell 10 turns it on explicitly.
    ap.add_argument("--thick_recovery", action="store_true",
                    help="supplement learned detections with wide-fibre recovery")
    ap.add_argument("--thick_min_width_px", type=float, default=26.0)
    ap.add_argument("--thick_max_width_px", type=float, default=160.0)
    ap.add_argument("--thick_min_sigma", type=float, default=8.0)
    ap.add_argument("--thick_spacing_px", type=float, default=28.0)
    ap.add_argument("--thick_min_coherence", type=float, default=0.22)
    ap.add_argument("--thick_segment_support", type=float, default=0.15)
    args = ap.parse_args(argv)

    if not args.image and not args.image_dir:
        ap.error("give --image or --image_dir")
    if args.image_dir and not _P(args.image_dir).is_dir():
        ap.error(f"--image_dir {args.image_dir!r} is not a directory. Create it "
                 f"and put the images in it, or pass --image for a single file.")

    set_seed(args.seed)
    device = pick_device(args.device)
    model, _ck = load_checkpoint(args.checkpoint, device)

    table_path = args.calibration_table
    if table_path is None and _P("calibration.yaml").exists():
        table_path = "calibration.yaml"
        LOG.info("using calibration.yaml for pixel sizes")
    table = load_calibration_table(table_path)

    wcal = None
    if args.width_calibration:
        from .width_calibration import load_width_calibration
        wcal = load_width_calibration(args.width_calibration)
        if wcal and wcal.get("fitted"):
            LOG.info("width calibration: scale %.3f shift %.3f (fitted on '%s')",
                     wcal["scale"], wcal["shift"], wcal.get("split", "?"))

    post = PostConfig(peak_threshold=args.peak_threshold,
                      min_validity=args.min_validity)
    LOG.info("post-processing: peak_threshold=%.2f min_validity=%.2f",
             post.peak_threshold, post.min_validity)

    thick_cfg = None
    if args.thick_recovery:
        from .thick_fiber import ThickRecoveryConfig
        thick_cfg = ThickRecoveryConfig(
            enabled=True,
            min_width_px=float(args.thick_min_width_px),
            max_width_px=float(args.thick_max_width_px),
            min_sigma=float(args.thick_min_sigma),
            spacing_px=float(args.thick_spacing_px),
            min_ridge_coherence=float(args.thick_min_coherence),
            segment_support=float(args.thick_segment_support),
        )
        LOG.info(
            "thick recovery ON: width>=%.1f px sigma>=%.1f spacing=%.1f "
            "coherence>=%.2f",
            thick_cfg.min_width_px, thick_cfg.min_sigma,
            thick_cfg.spacing_px, thick_cfg.min_ridge_coherence,
        )

    out_dir = ensure_dir(args.output_dir)
    paths = [Path(args.image)] if args.image else list_images(args.image_dir)
    if not paths:
        LOG.error("no images found in %s", args.image_dir)
        return 1

    LOG.info("running inference on %d image(s) with device=%s", len(paths), device)
    frames, uncalibrated = [], []
    for p in paths:
        try:
            df = run_one(
                model, p, out_dir, device,
                nm_per_pixel=args.nm_per_pixel,
                calib_table=table,
                post=post,
                tile=args.tile,
                overlap=args.overlap,
                tta=args.tta,
                mc_samples=args.mc_samples,
                save_maps=args.save_maps,
                width_calib=wcal,
                zoom_panels=args.zoom_panels,
                thick_cfg=thick_cfg,
            )
            frames.append(df)
            if len(df) and not np.isfinite(df["width_nm"].to_numpy(float)).any():
                uncalibrated.append(p.stem)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("failed on %s: %s", p, exc)

    if not frames:
        return 1

    import pandas as pd
    allp = pd.concat(frames, ignore_index=True, sort=False)
    allp.to_csv(_P(out_dir) / "all_predictions.csv", index=False)
    if uncalibrated:
        LOG.warning(
            "no pixel size for %d image(s): %s -- rows remain in "
            "all_predictions.csv with width_nm = NaN and are excluded from the "
            "pooled nanometre summary.",
            len(uncalibrated), ", ".join(uncalibrated),
        )

    nm = allp["width_nm"].to_numpy(float)
    nm = nm[np.isfinite(nm)]
    source_counts = (
        {str(k): int(v) for k, v in
         allp["measurement_source"].value_counts().to_dict().items()}
        if "measurement_source" in allp.columns else {}
    )
    summary = {
        "n_images": len(frames),
        "n_predictions": int(len(allp)),
        "measurement_sources": source_counts,
        "n_uncalibrated_images": len(uncalibrated),
        "uncalibrated": uncalibrated,
        "width_calibration": wcal if wcal else None,
        "thick_recovery_enabled": bool(args.thick_recovery),
    }
    if nm.size:
        summary["pooled_nm"] = {
            "n": int(nm.size),
            "median": float(np.median(nm)),
            "mean": float(nm.mean()),
            "sd": float(nm.std(ddof=1)) if nm.size > 1 else 0.0,
            **{f"p{q}": float(np.percentile(nm, q))
               for q in (5, 25, 75, 95)},
        }
        LOG.info("pooled over %d calibrated measurement(s): median %.1f nm",
                 nm.size, float(np.median(nm)))

    save_json(summary, _P(out_dir) / "batch_summary.json")
    LOG.info("wrote %d predictions across %d image(s)", len(allp), len(frames))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
