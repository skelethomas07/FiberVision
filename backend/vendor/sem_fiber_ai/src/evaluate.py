"""Evaluate a checkpoint on whole, unseen SEM images.

Evaluation is always at image level: the model sees a full field, produces its
own detections, and those are matched one-to-one against the manual
measurements.  Patch-level accuracy is deliberately not reported as headline
performance because it hides the detection problem entirely.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import load_records
from .infer import load_checkpoint, predict_image
from .fiber_metrics import (distribution_distance, fiber_level_recall, headline,
                            skeleton_coverage)
from .matching import MatchConfig, match_measurements
from .metrics import aggregate, evaluate_image
from .postprocess import PostConfig, decode_predictions, refine_width_from_image
from .utils import ensure_dir, get_logger, load_json, pick_device, save_json, set_seed
from .visualization import (distribution_figure, error_map_figure, match_figure,
                            prediction_panel)

LOG = get_logger(__name__)


def evaluate(checkpoint: str | Path, split: str = "test", *,
             labels_csv: str | Path | None = None,
             image_dir: str | Path | None = None,
             splits_json: str | Path | None = None,
             output_dir: str | Path = "outputs/eval",
             post: PostConfig | None = None,
             match_cfg: MatchConfig | None = None,
             device_pref: str = "auto", tta: bool = False,
             mc_samples: int = 0, save_figures: bool = True) -> dict[str, Any]:
    import pandas as pd

    device = pick_device(device_pref)
    model, ck = load_checkpoint(checkpoint, device)
    cfg = ck.get("config", {})
    labels_csv = labels_csv or cfg.get("data", {}).get("labels_csv")
    image_dir = image_dir or cfg.get("data", {}).get("image_dir")
    out_dir = ensure_dir(output_dir)

    splits_path = Path(splits_json or Path(checkpoint).parent / "splits.json")
    if splits_path.exists():
        splits = load_json(splits_path)
    else:
        LOG.warning("no splits.json next to the checkpoint; evaluating every image")
        splits = None

    records = load_records(labels_csv, image_dir,
                           mask_dir=cfg.get("data", {}).get("mask_dir"))
    if splits is not None:
        wanted = set(splits.get(split, []))
        if not wanted:
            LOG.error("split '%s' is EMPTY. With this dataset there is no held-out "
                      "image, so no generalisation claim can be made.", split)
            return {"error": f"split '{split}' is empty", "split": split}
        train_ids = set(splits.get("train", []))
        if wanted & train_ids:
            raise RuntimeError("split overlap detected -- refusing to evaluate on "
                               "images that were trained on")
        records = [r for r in records if r.image_id in wanted]

    if post is None:
        raw_post = cfg.get("postprocess", {}) or {}
        post = PostConfig(**{k: v for k, v in raw_post.items()
                             if k in PostConfig.__dataclass_fields__})
    if match_cfg is None:
        raw_match = cfg.get("match", {}) or {}
        match_cfg = MatchConfig(**{k: v for k, v in raw_match.items()
                                   if k in MatchConfig.__dataclass_fields__})
    eval_cfg = cfg.get("evaluate", {}) or {}
    per_image: dict[str, Any] = {}
    all_gt, all_pred = [], []

    for rec in records:
        gray = rec.image()
        maps = predict_image(model, gray, device, tta=tta, mc_samples=mc_samples)
        pred = decode_predictions(maps, image_id=rec.image_id,
                                  nm_per_pixel=rec.nm_per_pixel, cfg=post)
        gt = rec.annotations
        matches = match_measurements(gt, pred, match_cfg)
        res = evaluate_image(gt, pred, matches, nm_per_pixel=rec.nm_per_pixel)

        # Fiber-level numbers, which are the ones that answer the question the
        # thesis actually asks.  Chord-level precision/recall stays in the
        # report but is no longer the headline: the manual chords sit at
        # positions the annotator happened to pick, so demanding the model
        # reproduce those coordinates measures agreement with an arbitrary
        # sampling rather than measurement skill.
        res["fiber_level"] = fiber_level_recall(
            gt, pred,
            distance_scale=float(eval_cfg.get("fiber_distance_scale", 1.5)),
            min_distance_px=float(eval_cfg.get("fiber_min_distance_px", 8.0)),
            max_angle_deg=float(eval_cfg.get("fiber_max_angle_deg", 30.0)))
        try:
            res["coverage"] = skeleton_coverage(pred, rec.prior(image_only=True).mask, radius_px=float(eval_cfg.get("coverage_radius_px", 12.0)))
        except Exception as exc:                          # pragma: no cover
            LOG.warning("%s: coverage unavailable (%s)", rec.image_id, exc)
        # Independent width estimate straight off the intensity profile, with
        # no network involved.
        #
        # READ THIS AS A RELATIVE SIGNAL, NOT AN ABSOLUTE ONE.  A profile FWHM
        # carries a positive offset against a drawn width that grows with how
        # densely packed the fibers are -- measured on synthetic fields of known
        # width, +15% at 9% fiber area rising to +72% at 49%, because the scan
        # line starts running into the neighbouring fiber.  A separator image is
        # at the dense end, so a large offset here is expected and does NOT by
        # itself indicate a broken width head.
        #
        # What is interpretable: how this number MOVES.  Between two checkpoints
        # on the same images, or between images of similar packing, a change in
        # the offset points at the width head.  Its absolute value points at the
        # packing fraction.
        if len(pred):
            try:
                chk = refine_width_from_image(pred, gray)
                ok = chk["width_px_profile"].notna()
                # Restrict the check to sites on a clean single fiber.  The
                # profile FWHM is only meaningful where the scan crosses one
                # fiber: at a crossing the perpendicular really is wider, and on
                # a dense network the scan runs into the neighbour.  Measured on
                # synthetic fields of known width, the apparent error grows from
                # +15% at 9% fiber area to +72% at 49% purely from density, so
                # an ungated check on a separator image reports the packing
                # fraction rather than the width head.  Coherency from the
                # structure tensor is what separates the two.
                try:
                    pri = rec.prior(image_only=True)
                    h, w = pri.shape
                    xs = np.clip(chk["center_x_px"].to_numpy(float).round()
                                 .astype(int), 0, w - 1)
                    ys = np.clip(chk["center_y_px"].to_numpy(float).round()
                                 .astype(int), 0, h - 1)
                    ok &= pri.coherency[ys, xs] >= 0.5
                except Exception:
                    pass
                if ok.sum() >= 5:
                    d = (chk.loc[ok, "width_px_profile"]
                         - chk.loc[ok, "width_px"]).to_numpy(float)
                    ref = float(chk.loc[ok, "width_px"].median())
                    res["profile_crosscheck"] = {
                        "n": int(ok.sum()),
                        "median_difference_px": float(np.median(d)),
                        "median_relative_difference": float(np.median(d)) / max(ref, 1e-6),
                        "iqr_px": float(np.percentile(d, 75) - np.percentile(d, 25)),
                        "usable_fraction": float(ok.sum() / max(len(chk), 1)),
                    }
            except Exception as exc:                      # pragma: no cover
                LOG.warning("%s: profile cross-check failed (%s)",
                            rec.image_id, exc)

        if len(pred):
            res["distribution_px"] = distribution_distance(
                gt["width_px"].to_numpy(float), pred["width_px"].to_numpy(float))
            if rec.nm_per_pixel:
                res["distribution_nm"] = distribution_distance(
                    gt["width_px"].to_numpy(float) * rec.nm_per_pixel,
                    pred["width_px"].to_numpy(float) * rec.nm_per_pixel)
        per_image[rec.image_id] = res
        pred.to_csv(Path(out_dir) / f"{rec.image_id}_predictions.csv", index=False)
        matches.to_csv(Path(out_dir) / f"{rec.image_id}_matches.csv", index=False)
        all_gt.append(gt)
        all_pred.append(pred)

        if save_figures:
            match_figure(gray, gt, pred, matches,
                         Path(out_dir) / f"{rec.image_id}_matches.png",
                         title=f"{rec.image_id}: {len(matches)} matched of "
                               f"{len(gt)} manual / {len(pred)} predicted")
            error_map_figure(gray, matches,
                             Path(out_dir) / f"{rec.image_id}_error_map.png")
            prediction_panel(gray, {
                "centre heatmap": 1 / (1 + np.exp(-maps["center_logit"])),
                "fiber segmentation": 1 / (1 + np.exp(-maps["segment_logit"])),
                "width (px)": np.exp(maps["width"]),
                "orientation (deg)": np.rad2deg(0.5 * np.arctan2(
                    maps["orient"][1], maps["orient"][0])),
                "validity": 1 / (1 + np.exp(-maps["validity_logit"])),
                "uncertainty (sigma)": np.exp(0.5 * maps["logvar"]),
            }, Path(out_dir) / f"{rec.image_id}_maps.png", title=rec.image_id)
            if len(pred):
                distribution_figure(gt["width_px"].to_numpy(),
                                    pred["width_px"].to_numpy(),
                                    Path(out_dir) / f"{rec.image_id}_distribution.png")

    result = {"split": split, "checkpoint": str(checkpoint),
              "per_image": per_image, "pooled": aggregate(per_image),
              "headline": headline(per_image),
              "n_images": len(records)}
    save_json(result, Path(out_dir) / f"metrics_{split}.json")
    LOG.info("evaluation written to %s", out_dir)
    return result


def tune_threshold(checkpoint: str | Path, *, split: str = "val",
                   thresholds: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25,
                                                    0.30, 0.40, 0.50, 0.60, 0.70),
                   validities: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5),
                   labels_csv: str | Path | None = None,
                   image_dir: str | Path | None = None,
                   splits_json: str | Path | None = None,
                   objective: str = "width_wasserstein",
                   device_pref: str = "auto") -> "Any":
    """Sweep the peak threshold on the validation split and pick one.

    The threshold was previously chosen by eye from the heatmap maximum ("set
    it below the max, roughly half is a good start").  That is a guess about a
    quantity the validation set can simply measure, and it guessed badly: at
    0.30 against a heatmap that peaked at 0.56 the model returned three
    detections on a field containing 566 fibers.

    ``objective`` picks what the threshold is tuned *for*, and the choice is a
    scientific one rather than a technical one.  ``width_wasserstein`` optimises
    agreement of the width distribution, which is what a materials result
    reports.  ``fiber_f1`` optimises detection instead.  They do not generally
    agree, and the difference between them is worth looking at.
    """
    import pandas as pd

    device = pick_device(device_pref)
    model, ck = load_checkpoint(checkpoint, device)
    cfg = ck.get("config", {})
    labels_csv = labels_csv or cfg.get("data", {}).get("labels_csv")
    image_dir = image_dir or cfg.get("data", {}).get("image_dir")
    splits_path = Path(splits_json or Path(checkpoint).parent / "splits.json")
    splits = load_json(splits_path) if splits_path.exists() else None

    records = load_records(labels_csv, image_dir,
                           mask_dir=cfg.get("data", {}).get("mask_dir"))
    if splits is not None:
        wanted = set(splits.get(split, []))
        if not wanted:
            raise RuntimeError(f"split '{split}' is empty; cannot tune on it")
        if wanted & set(splits.get("train", [])):
            raise RuntimeError("refusing to tune the threshold on training images")
        records = [r for r in records if r.image_id in wanted]

    # [v3] The threshold and min_validity chosen here are hyper-parameters
    # selected on this split, so the split has to be able to carry them. One
    # field of one specimen -- what the August split produced -- cannot: the
    # numbers that come out are fitted to that field, and the test metrics
    # computed with them inherit the fit.
    from .dataset import specimen_groups
    n_spec = len(set(specimen_groups([r.image_id for r in records]).values()))
    if n_spec < 2:
        LOG.warning("tuning on %d field(s) from %d specimen(s). This is noisy as a "
                    "standalone validation estimate, but it is valid as an INNER "
                    "selection split when a separate outer specimen remains sealed.",
                    len(records), n_spec)

    raw_post = cfg.get("postprocess", {}) or {}
    base_post = {k: v for k, v in raw_post.items() if k in PostConfig.__dataclass_fields__}
    eval_cfg = cfg.get("evaluate", {}) or {}

    def _post(thr: float, minval: float) -> PostConfig:
        d = dict(base_post)
        d.update(peak_threshold=float(thr), min_validity=float(minval))
        return PostConfig(**d)

    # predict once per image; only the decoding depends on the threshold
    cached = []
    for rec in records:
        maps = predict_image(model, rec.image(), device)
        cached.append((rec, maps))
        heat = 1.0 / (1.0 + np.exp(-maps["center_logit"]))
        LOG.info("%s: heatmap max %.3f, p99.9 %.3f", rec.image_id,
                 float(heat.max()), float(np.percentile(heat, 99.9)))

    # Report where the peaks are actually going before tuning anything.  On the
    # first run, 40s_48-4 produced 32 peaks and only 13 predictions: 19 were
    # discarded by the post-processing filters, not missed by the detector.
    # Which filter did the discarding decides what to loosen, and guessing at it
    # is how min_validity ends up throttling a model that is working.
    reasons: dict[str, int] = {}
    probe = _post(float(min(thresholds)), 0.0)
    for rec, maps in cached:
        allp = decode_predictions(maps, image_id=rec.image_id,
                                  nm_per_pixel=rec.nm_per_pixel, cfg=probe,
                                  keep_rejected=True)
        if len(allp):
            for k, v in allp["rejected_reason"].fillna("").value_counts().items():
                reasons[str(k) or "kept"] = reasons.get(str(k) or "kept", 0) + int(v)
    LOG.info("peak disposition at the loosest setting: %s", reasons)

    def _safe_mean(values):
        a = np.asarray(values, float)
        a = a[np.isfinite(a)]
        return float(a.mean()) if a.size else float("nan")

    rows = []
    for thr in thresholds:
        for minval in validities:
            post = _post(float(thr), float(minval))
            w1, rec_, prec_, n_pred = [], [], [], []
            for rec, maps in cached:
                pred = decode_predictions(maps, image_id=rec.image_id,
                                          nm_per_pixel=rec.nm_per_pixel, cfg=post)
                gt = rec.annotations
                n_pred.append(len(pred))
                fl = fiber_level_recall(
                    gt, pred,
                    distance_scale=float(eval_cfg.get("fiber_distance_scale", 1.5)),
                    min_distance_px=float(eval_cfg.get("fiber_min_distance_px", 8.0)),
                    max_angle_deg=float(eval_cfg.get("fiber_max_angle_deg", 30.0)))
                rec_.append(fl.get("fiber_recall", float("nan")))
                if len(pred):
                    d = distribution_distance(gt["width_px"].to_numpy(float),
                                              pred["width_px"].to_numpy(float))
                    w1.append(d.get("wasserstein", float("nan")))
                    m = match_measurements(gt, pred, MatchConfig())
                    prec_.append(len(m) / len(pred))
                else:
                    w1.append(float("nan"))
                    prec_.append(0.0)
            rows.append({"peak_threshold": thr, "min_validity": minval,
                         "mean_predictions": _safe_mean(n_pred),
                         "fiber_recall": _safe_mean(rec_),
                         "chord_precision": _safe_mean(prec_),
                         "width_wasserstein": _safe_mean(w1)})

    df = pd.DataFrame(rows)
    df.attrs["peak_disposition"] = reasons
    if objective == "fiber_f1":
        f1 = 2 * df["fiber_recall"] * df["chord_precision"] / \
            (df["fiber_recall"] + df["chord_precision"]).replace(0, np.nan)
        best = df.loc[f1.idxmax()] if f1.notna().any() else df.iloc[0]
    else:
        best = df.loc[df["width_wasserstein"].idxmin()] \
            if df["width_wasserstein"].notna().any() else df.iloc[0]
    LOG.info("best setting on '%s' by %s: peak_threshold=%.2f min_validity=%.2f "
             "(%.0f predictions/image)", split, objective,
             float(best["peak_threshold"]), float(best["min_validity"]),
             float(best["mean_predictions"]))
    df.attrs["best_threshold"] = float(best["peak_threshold"])
    df.attrs["best_min_validity"] = float(best["min_validity"])
    df.attrs["best_post"] = _post(float(best["peak_threshold"]), float(best["min_validity"]))
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate on whole unseen SEM images")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--labels_csv", default=None)
    ap.add_argument("--image_dir", default=None)
    ap.add_argument("--splits_json", default=None)
    ap.add_argument("--output_dir", default="outputs/eval")
    ap.add_argument("--peak_threshold", type=float, default=0.30)
    ap.add_argument("--match_distance", type=float, default=12.0)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--mc_samples", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(argv)
    set_seed(args.seed)
    res = evaluate(args.checkpoint, args.split, labels_csv=args.labels_csv,
                   image_dir=args.image_dir, splits_json=args.splits_json,
                   output_dir=args.output_dir,
                   post=PostConfig(peak_threshold=args.peak_threshold),
                   match_cfg=MatchConfig(max_center_distance=args.match_distance),
                   device_pref=args.device, tta=args.tta, mc_samples=args.mc_samples)
    print(__import__("json").dumps(res.get("pooled", res), indent=2)[:4000])
    return 0



# --------------------------------------------------------------------------- #
# ---- v6.4 -----------------------------------------------------------------
#
# The v6.3 sweep chose peak_threshold=0.70, the top of its grid; the run before
# it chose 0.05, the bottom.  Both mean the optimum was outside the grid and
# nobody was told.  Worse, width_wasserstein improves monotonically as
# predictions are deleted -- across that sweep fibre recall fell 0.46 -> 0.35
# while the objective kept getting "better" -- so on its own the objective wants
# the model to predict nothing.
#
# select_threshold therefore (a) refuses settings that give up more recall than
# you allow, and (b) says so out loud when the winner sits on a grid edge.
# --------------------------------------------------------------------------- #
def select_threshold(df, *, objective: str = "width_wasserstein",
                     min_recall_frac: float = 0.90,
                     min_recall_abs: float = 0.0):
    """Pick a row of a tune_threshold table under a recall constraint.

    ``min_recall_frac`` is relative to the best fibre recall the sweep achieved,
    so it adapts to a model that is simply weak rather than demanding an
    absolute number the run cannot reach.
    """
    import numpy as _np

    d = df.copy()
    best_recall = float(_np.nanmax(d["fiber_recall"])) if len(d) else float("nan")
    floor = max(float(min_recall_abs), min_recall_frac * best_recall
                if _np.isfinite(best_recall) else 0.0)
    ok = d[d["fiber_recall"] >= floor]
    if not len(ok):
        LOG.warning("no setting reaches fibre recall %.3f; ignoring the floor", floor)
        ok = d
    if objective == "fiber_f1":
        f1 = 2 * ok["fiber_recall"] * ok["chord_precision"] / \
            (ok["fiber_recall"] + ok["chord_precision"]).replace(0, _np.nan)
        row = ok.loc[f1.idxmax()] if f1.notna().any() else ok.iloc[0]
    else:
        col = objective if objective in ok.columns else "width_wasserstein"
        row = ok.loc[ok[col].idxmin()] if ok[col].notna().any() else ok.iloc[0]

    thr = float(row["peak_threshold"])
    grid = sorted(set(float(v) for v in d["peak_threshold"]))
    at_edge = thr in (grid[0], grid[-1])
    LOG.info("selected peak_threshold=%.2f min_validity=%.2f "
             "(recall %.3f >= floor %.3f, %s %.4f, %.0f pred/image)",
             thr, float(row["min_validity"]), float(row["fiber_recall"]), floor,
             objective, float(row.get(objective, _np.nan)),
             float(row["mean_predictions"]))
    if at_edge:
        LOG.warning("the winner sits at the %s of the grid (%.2f). The optimum is "
                    "probably outside it -- widen the grid before quoting this.",
                    "bottom" if thr == grid[0] else "top", thr)
    return {"peak_threshold": thr, "min_validity": float(row["min_validity"]),
            "fiber_recall": float(row["fiber_recall"]),
            "recall_floor": floor, "at_grid_edge": bool(at_edge),
            "objective": objective,
            "objective_value": float(row.get(objective, _np.nan)),
            "mean_predictions": float(row["mean_predictions"])}


def _gt_counts(checkpoint, split, labels_csv=None, splits_json=None):
    """Manual measurements per field on a split -- the human's sampling density.

    Needed because "predict like the annotator would" is a statement about HOW
    MANY chords a person places per field, and the sweep table only knew how
    many the model placed.
    """
    import pandas as pd
    import torch

    if labels_csv is None or splits_json is None:
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {})
        labels_csv = labels_csv or cfg.get("data", {}).get("labels_csv")
        splits_json = splits_json or str(Path(checkpoint).parent / "splits.json")
    lab = pd.read_csv(labels_csv)
    ids = set(load_json(splits_json).get(split, []))
    sub = lab[lab.image_id.isin(ids)]
    if not len(sub):
        return float("nan")
    return float(sub.groupby("image_id").size().mean())


def add_sampling_columns(df, mean_gt):
    """Annotate a sweep table with how the model's sampling density compares.

    ``count_ratio`` is predictions per field over manual chords per field, and
    ``count_match`` is its absolute log -- 0 when the model places as many
    measurements as the person did, symmetric in over- and under-sampling.
    """
    import numpy as _np

    d = df.copy()
    d["count_ratio"] = d["mean_predictions"] / mean_gt
    d["count_match"] = _np.abs(_np.log(d["count_ratio"].replace(0, _np.nan)))
    return d


def tune_threshold_v64(checkpoint, *, split: str = "val",
                       thresholds=(0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                                   0.60, 0.70, 0.80, 0.90, 0.95),
                       validities=(0.0, 0.1, 0.2, 0.3, 0.5),
                       objective: str = "width_wasserstein",
                       min_recall_frac: float = 0.90,
                       labels_csv=None, splits_json=None, **kw):
    """tune_threshold over a wider grid, with sampling density and a recall floor.

    ``objective="count_match"`` is the one that matches the stated goal of this
    project: reproduce the way a person samples a field, not the coordinates
    they happened to click.  It picks the setting whose measurement count per
    field is closest to the annotator's.  ``width_wasserstein`` optimises the
    width distribution instead, and the two are worth reading side by side --
    if they disagree strongly, the model is getting the right distribution from
    the wrong number of measurements, which is worth knowing before you quote
    either.
    """
    df = tune_threshold(checkpoint, split=split, thresholds=tuple(thresholds),
                        validities=tuple(validities),
                        objective=("fiber_f1" if objective == "fiber_f1"
                                   else "width_wasserstein"),
                        labels_csv=labels_csv, splits_json=splits_json, **kw)
    try:
        mean_gt = _gt_counts(checkpoint, split, labels_csv, splits_json)
        df = add_sampling_columns(df, mean_gt)
        LOG.info("the annotator placed %.0f measurements per field on '%s'",
                 mean_gt, split)
    except Exception as exc:                                   # noqa: BLE001
        LOG.warning("could not compute the manual sampling density (%s); "
                    "objective='count_match' unavailable", exc)
    choice = select_threshold(df, objective=objective,
                              min_recall_frac=min_recall_frac)
    df.attrs.update({"best_threshold": choice["peak_threshold"],
                     "best_min_validity": choice["min_validity"],
                     "selection": choice})
    return df

# ---- v6.4i ----
_tune_threshold_v64_floored = tune_threshold_v64


def tune_threshold_v64(checkpoint, *, split="val",
                       thresholds=(0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                                   0.60, 0.70, 0.80, 0.90, 0.95),
                       validities=(0.0, 0.1, 0.2, 0.3, 0.5),
                       objective="width_wasserstein",
                       min_recall_frac=None, **kw):
    """[v6.4i] the recall floor no longer fights ``count_match``.

    In the v6.4 run the floor sat at 0.9 x the best fibre recall (0.403) and
    ``count_match`` was told to pick under it.  The setting it wanted --
    threshold 0.80, 1.68x the annotator's measurement count -- has recall 0.252,
    so it was blocked, and the winner became 0.70 at 3.09x.  The objective was
    overruled by a constraint that exists for a different objective entirely:
    ``width_wasserstein`` improves without limit as predictions are deleted and
    needs a floor to stop it, while ``count_match`` already punishes
    under-prediction symmetrically and needs none.

    So the default now depends on the objective.  Pass ``min_recall_frac``
    explicitly to override either way.
    """
    if min_recall_frac is None:
        min_recall_frac = 0.0 if objective == "count_match" else 0.90
    return _tune_threshold_v64_floored(
        checkpoint, split=split, thresholds=thresholds, validities=validities,
        objective=objective, min_recall_frac=min_recall_frac, **kw)

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
