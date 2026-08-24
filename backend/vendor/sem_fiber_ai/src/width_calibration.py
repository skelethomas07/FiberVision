"""[v6.4] Correct the shrinkage of the predicted width distribution.

Every test field in the v6.3 run showed the same thing: the predicted widths
were too narrow and slightly too large.  Manual standard deviation 4.71 px
against 2.30 predicted, 4.13 against 1.87, 4.30 against 2.93, 3.60 against 3.21,
with the median 4-34% high.  That is regression to the mean -- a squared-error
head trained on limited data hedges towards the centre -- and it matters more
here than the recall does, because the deliverable *is* a distribution.

The fix is a monotone two-parameter map fitted in log space on the VALIDATION
split and applied unchanged to test and to new images:

    w' = exp(shift + scale * log(w))

``scale`` is the ratio of interquartile ranges and ``shift`` aligns the medians.
Quartiles rather than moments because a handful of 60 px detections on a pore
would otherwise set the scale.

What this is and is not:

* It is a calibration, in the same sense as calibrating a thermometer against a
  reference: fitted on held-out data, applied blind, reported separately.
* It is NOT a fix for the model.  A corrected distribution can have the right
  spread with the wrong fibre-by-fibre assignment, so the per-chord agreement
  (Pearson r, MAE) must still be quoted from the RAW predictions.  Report both.
* If ``scale`` comes out far from 1 the model is badly under-dispersed and the
  honest move is to train longer, not to stretch harder.  ``fit_width_calibration``
  warns above ``max_scale``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .utils import get_logger

LOG = get_logger(__name__)


def _log_stats(w: np.ndarray) -> tuple[float, float, int]:
    w = np.asarray(w, dtype=float)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size < 8:
        return float("nan"), float("nan"), int(w.size)
    lw = np.log(w)
    q1, med, q3 = np.percentile(lw, [25, 50, 75])
    return float(med), float(q3 - q1), int(w.size)


def fit_width_calibration(pred_px, gt_px, *, max_scale: float = 3.0,
                          image_id: str = "") -> dict[str, Any]:
    """Fit ``w' = exp(shift + scale*log(w))`` matching median and IQR in log space."""
    p_med, p_iqr, n_p = _log_stats(pred_px)
    g_med, g_iqr, n_g = _log_stats(gt_px)
    if not (np.isfinite(p_med) and np.isfinite(g_med)):
        LOG.warning("width calibration: too few finite widths (pred %d, gt %d)",
                    n_p, n_g)
        return {"shift": 0.0, "scale": 1.0, "n_pred": n_p, "n_gt": n_g,
                "fitted": False, "reason": "too few widths"}
    if p_iqr <= 1e-6:
        LOG.warning("width calibration: predicted IQR is zero -- the head has "
                    "collapsed to a constant; not stretching it")
        return {"shift": float(g_med - p_med), "scale": 1.0,
                "n_pred": n_p, "n_gt": n_g, "fitted": True,
                "reason": "median shift only (degenerate spread)"}
    scale = float(g_iqr / p_iqr)
    if scale > max_scale:
        LOG.warning("%swidth calibration wants to stretch the spread by %.2fx, "
                    "above the %.1fx cap. Capping, and note that a model this "
                    "under-dispersed should be trained longer rather than "
                    "stretched.", f"{image_id}: " if image_id else "",
                    scale, max_scale)
        scale = max_scale
    shift = float(g_med - scale * p_med)
    return {"shift": shift, "scale": scale, "n_pred": n_p, "n_gt": n_g,
            "pred_log_median": p_med, "pred_log_iqr": p_iqr,
            "gt_log_median": g_med, "gt_log_iqr": g_iqr,
            "fitted": True, "reason": "median+IQR match in log space"}


def apply_width_calibration(width_px, calib: dict[str, Any] | None):
    """Apply a fitted map.  ``None`` or an unfitted dict is the identity."""
    w = np.asarray(width_px, dtype=float)
    if not calib or not calib.get("fitted"):
        return w
    out = np.full_like(w, np.nan)
    ok = np.isfinite(w) & (w > 0)
    out[ok] = np.exp(float(calib["shift"]) + float(calib["scale"]) * np.log(w[ok]))
    return out


def load_width_calibration(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        LOG.warning("width calibration %s not found -- predictions stay raw", p)
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def fit_on_split(checkpoint, *, split: str = "val", post=None,
                 labels_csv=None, image_dir=None, splits_json=None,
                 out_json: str | Path = "width_calibration.json",
                 device_pref: str = "auto", tta: bool = False) -> dict[str, Any]:
    """Predict the split, pool the widths, fit the map, report before/after.

    Pooled across the split rather than fitted per image on purpose: a per-image
    fit would absorb genuine between-field differences in fibre width, which is
    the thing the thesis is trying to measure.
    """
    from .dataset import load_records
    from .fiber_metrics import distribution_distance
    from .infer import load_checkpoint, predict_image
    from .postprocess import PostConfig, decode_predictions
    from .utils import load_json, pick_device

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
            raise RuntimeError(f"split '{split}' is empty; cannot calibrate on it")
        if wanted & set(splits.get("train", [])):
            raise RuntimeError("refusing to calibrate widths on training images")
        records = [r for r in records if r.image_id in wanted]

    if post is None:
        raw = cfg.get("postprocess", {}) or {}
        post = PostConfig(**{k: v for k, v in raw.items()
                             if k in PostConfig.__dataclass_fields__})

    pred_all, gt_all, per_image = [], [], {}
    for rec in records:
        maps = predict_image(model, rec.image(), device, tta=tta)
        pred = decode_predictions(maps, image_id=rec.image_id,
                                  nm_per_pixel=rec.nm_per_pixel, cfg=post)
        pw = pred["width_px"].to_numpy(float) if len(pred) else np.array([])
        gw = rec.annotations["width_px"].to_numpy(float)
        pred_all.append(pw)
        gt_all.append(gw)
        per_image[rec.image_id] = {
            "n_pred": int(pw.size), "n_gt": int(gw.size),
            "pred_median": float(np.median(pw)) if pw.size else float("nan"),
            "gt_median": float(np.median(gw)) if gw.size else float("nan"),
            "pred_sd": float(np.std(pw, ddof=1)) if pw.size > 1 else float("nan"),
            "gt_sd": float(np.std(gw, ddof=1)) if gw.size > 1 else float("nan"),
        }

    pred_px = np.concatenate(pred_all) if pred_all else np.array([])
    gt_px = np.concatenate(gt_all) if gt_all else np.array([])
    calib = fit_width_calibration(pred_px, gt_px)
    calib["split"] = split
    calib["images"] = sorted(per_image)
    calib["per_image_raw"] = per_image

    before = distribution_distance(gt_px, pred_px)
    after = distribution_distance(gt_px, apply_width_calibration(pred_px, calib))
    calib["wasserstein_before_px"] = float(before.get("wasserstein", float("nan")))
    calib["wasserstein_after_px"] = float(after.get("wasserstein", float("nan")))
    LOG.info("width calibration on '%s': scale %.3f shift %.3f | "
             "Wasserstein %.3f -> %.3f px", split, calib["scale"], calib["shift"],
             calib["wasserstein_before_px"], calib["wasserstein_after_px"])
    if calib["wasserstein_after_px"] > calib["wasserstein_before_px"]:
        LOG.warning("the calibration made the validation distribution WORSE. "
                    "Do not apply it; report the raw distribution instead.")
        calib["fitted"] = False
        calib["reason"] = "rejected: no improvement on validation"
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(calib, indent=2), encoding="utf-8")
    return calib

# ---- v6.4 ----

# ---- v6.4i ----
def fit_width_calibration_grouped(pairs, *, max_scale: float = 3.0):
    """Fit the log-space map from PER-FIELD spreads, not the pooled one.

    The pooled fit in v6.4 came out at scale 1.069 -- effectively the identity --
    on a validation set where every individual field was 20-40% too narrow.  The
    reason is that pooling mixes two different spreads: the width variation
    *inside* a field, which is what the model is getting wrong, and the variation
    *between* fields (val medians ran 7.5 to 28 px), which the model reproduces
    fine.  Both the predicted and the manual pool inherit the same between-field
    inflation, so it cancels and the ratio lands near 1.

    Fitting per field and taking the median across fields measures the thing that
    is actually shrunk, and the median makes one pathological field unable to set
    the answer -- which matters here, because 3-11 has a manual sd of 23 px
    against a median of 28 while its four neighbours sit at 2.8-3.9 px.

    ``pairs`` is a list of ``(pred_widths, gt_widths)``, one per field.
    """
    import numpy as _np

    def _ls(w):
        w = _np.asarray(w, float)
        w = w[_np.isfinite(w) & (w > 0)]
        if w.size < 8:
            return None
        lw = _np.log(w)
        q1, me, q3 = _np.percentile(lw, [25, 50, 75])
        return float(me), float(q3 - q1)

    scales, shifts, used = [], [], 0
    for pw, gw in pairs:
        a, b = _ls(pw), _ls(gw)
        if a is None or b is None or a[1] <= 1e-6:
            continue
        s = b[1] / a[1]
        scales.append(s)
        shifts.append(b[0] - s * a[0])
        used += 1
    if used < 2:
        LOG.warning("only %d usable field(s); falling back to the pooled fit", used)
        return None
    scale = float(_np.median(scales))
    if scale > max_scale:
        LOG.warning("per-field fit wants a %.2fx stretch, above the %.1fx cap. "
                    "Capping; a model this under-dispersed needs a longer run, "
                    "not a harder stretch.", scale, max_scale)
        scale = max_scale
    shift = float(_np.median(shifts))
    LOG.info("per-field width calibration: scale %.3f (field values %s), "
             "shift %+.3f", scale,
             _np.array2string(_np.round(_np.asarray(scales), 2)), shift)
    return {"shift": shift, "scale": scale, "fitted": True,
            "n_fields": used, "per_field_scale": [float(s) for s in scales],
            "reason": "median of per-field log-IQR ratios"}


_fit_on_split_pooled = fit_on_split


def fit_on_split(checkpoint, *, split: str = "val", post=None,
                 labels_csv=None, image_dir=None, splits_json=None,
                 out_json="width_calibration.json",
                 device_pref: str = "auto", tta: bool = False):
    """[v6.4i] as before, but the scale comes from the per-field fit."""
    import json as _json
    from pathlib import Path as _P

    import numpy as _np

    from .dataset import load_records
    from .fiber_metrics import distribution_distance
    from .infer import load_checkpoint, predict_image
    from .postprocess import PostConfig, decode_predictions
    from .utils import load_json, pick_device

    device = pick_device(device_pref)
    model, ck = load_checkpoint(checkpoint, device)
    cfg = ck.get("config", {})
    labels_csv = labels_csv or cfg.get("data", {}).get("labels_csv")
    image_dir = image_dir or cfg.get("data", {}).get("image_dir")
    splits_path = _P(splits_json or _P(checkpoint).parent / "splits.json")
    splits = load_json(splits_path) if splits_path.exists() else None

    records = load_records(labels_csv, image_dir,
                           mask_dir=cfg.get("data", {}).get("mask_dir"))
    if splits is not None:
        wanted = set(splits.get(split, []))
        if not wanted:
            raise RuntimeError(f"split '{split}' is empty; cannot calibrate on it")
        if wanted & set(splits.get("train", [])):
            raise RuntimeError("refusing to calibrate widths on training images")
        records = [r for r in records if r.image_id in wanted]

    if post is None:
        raw = cfg.get("postprocess", {}) or {}
        post = PostConfig(**{k: v for k, v in raw.items()
                             if k in PostConfig.__dataclass_fields__})

    pairs, per_image = [], {}
    for rec in records:
        maps = predict_image(model, rec.image(), device, tta=tta)
        pred = decode_predictions(maps, image_id=rec.image_id,
                                  nm_per_pixel=rec.nm_per_pixel, cfg=post)
        pw = pred["width_px"].to_numpy(float) if len(pred) else _np.array([])
        gw = rec.annotations["width_px"].to_numpy(float)
        pairs.append((pw, gw))
        per_image[rec.image_id] = {
            "n_pred": int(pw.size), "n_gt": int(gw.size),
            "pred_median": float(_np.median(pw)) if pw.size else float("nan"),
            "gt_median": float(_np.median(gw)) if gw.size else float("nan"),
            "pred_sd": float(_np.std(pw, ddof=1)) if pw.size > 1 else float("nan"),
            "gt_sd": float(_np.std(gw, ddof=1)) if gw.size > 1 else float("nan"),
        }

    pred_px = _np.concatenate([p for p, _ in pairs]) if pairs else _np.array([])
    gt_px = _np.concatenate([g for _, g in pairs]) if pairs else _np.array([])

    calib = fit_width_calibration_grouped(pairs)
    pooled = fit_width_calibration(pred_px, gt_px)
    if calib is None:
        calib = pooled
    calib["pooled_scale_for_reference"] = float(pooled.get("scale", float("nan")))
    calib["split"] = split
    calib["images"] = sorted(per_image)
    calib["per_image_raw"] = per_image
    calib["n_pred"] = int(pred_px.size)
    calib["n_gt"] = int(gt_px.size)

    # Judge it the way it will be used: per field, then take the median, so one
    # field cannot carry the verdict.
    before, after = [], []
    for pw, gw in pairs:
        if pw.size < 8 or gw.size < 8:
            continue
        before.append(distribution_distance(gw, pw).get("wasserstein", _np.nan))
        after.append(distribution_distance(
            gw, apply_width_calibration(pw, calib)).get("wasserstein", _np.nan))
    calib["wasserstein_before_px"] = float(_np.nanmedian(before)) if before else float("nan")
    calib["wasserstein_after_px"] = float(_np.nanmedian(after)) if after else float("nan")
    LOG.info("width calibration on '%s': scale %.3f shift %+.3f | median "
             "per-field Wasserstein %.3f -> %.3f px", split, calib["scale"],
             calib["shift"], calib["wasserstein_before_px"],
             calib["wasserstein_after_px"])
    if calib["wasserstein_after_px"] > calib["wasserstein_before_px"]:
        LOG.warning("the calibration made the validation distribution WORSE. "
                    "Not applying it; report the raw distribution instead.")
        calib["fitted"] = False
        calib["reason"] = "rejected: no improvement on validation"
    _P(out_json).parent.mkdir(parents=True, exist_ok=True)
    _P(out_json).write_text(_json.dumps(calib, indent=2), encoding="utf-8")
    return calib
