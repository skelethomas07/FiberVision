"""Rendering helpers for annotation verification and prediction review.

Design rules used throughout:

* the SEM image is the evidence, so overlays stay thin and labels stay small;
* colour always encodes something (ground truth vs prediction vs error), never
  decoration;
* every figure can be produced without a display (Agg backend), because most of
  this runs in Colab or on a headless box.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .utils import ensure_dir, get_logger

LOG = get_logger(__name__)

GT_COLOR = (255, 210, 60)        # amber  - manual measurement
PRED_COLOR = (60, 200, 255)      # cyan   - model prediction
MATCH_COLOR = (120, 255, 120)    # green  - matched pair
FP_COLOR = (255, 90, 90)         # red    - false positive
FN_COLOR = (255, 140, 255)       # violet - false negative


def to_rgb(gray: np.ndarray) -> np.ndarray:
    """Contrast-stretched grayscale -> uint8 RGB canvas."""
    g = np.asarray(gray, np.float32)
    lo, hi = np.percentile(g[np.isfinite(g)], (0.5, 99.5))
    if hi <= lo:
        lo, hi = float(g.min()), float(max(g.max(), g.min() + 1))
    g = np.clip((g - lo) / (hi - lo), 0, 1)
    return (np.stack([g] * 3, axis=-1) * 255).astype(np.uint8)


def draw_measurements(canvas: np.ndarray, df: "Any", *, color=GT_COLOR,
                      thickness: int = 1, show_ids: bool = True,
                      value_col: str | None = None, font_scale: float = 0.3,
                      max_labels: int = 400,
                      source_colors: dict[str, tuple[int, int, int]] | None = None
                      ) -> np.ndarray:
    """Draw measurement segments (and optionally small labels) on a canvas.

    ``source_colors`` is optional and backwards compatible.  When present, rows
    carrying ``measurement_source`` can be colour-coded, e.g. learned AI chords
    versus classical thick-fibre recovery.
    """
    import cv2

    out = canvas.copy()
    id_col = ("annotation_id" if "annotation_id" in df.columns
              else "prediction_id" if "prediction_id" in df.columns else None)
    has_ends = all(c in df.columns for c in ("x1_px", "y1_px", "x2_px", "y2_px"))
    for i, (_, r) in enumerate(df.iterrows()):
        row_color = color
        if source_colors and "measurement_source" in df.columns:
            row_color = source_colors.get(str(r.get("measurement_source", "")), color)
        if has_ends and np.isfinite(r["x1_px"]) and np.isfinite(r["x2_px"]):
            p1 = (int(round(r["x1_px"])), int(round(r["y1_px"])))
            p2 = (int(round(r["x2_px"])), int(round(r["y2_px"])))
        else:
            cx, cy = float(r["center_x_px"]), float(r["center_y_px"])
            half = float(r.get("width_px", 6.0)) / 2.0
            a = np.deg2rad(float(r.get("measurement_angle_deg", 0.0)))
            dx, dy = half * np.cos(a), half * np.sin(a)
            p1 = (int(round(cx - dx)), int(round(cy - dy)))
            p2 = (int(round(cx + dx)), int(round(cy + dy)))
        cv2.line(out, p1, p2, row_color, thickness, cv2.LINE_AA)
        cv2.circle(out, p1, 1, row_color, -1, cv2.LINE_AA)
        cv2.circle(out, p2, 1, row_color, -1, cv2.LINE_AA)
        if show_ids and i < max_labels:
            parts = []
            if id_col is not None:
                parts.append(str(int(r[id_col])))
            if value_col and value_col in df.columns and np.isfinite(r[value_col]):
                parts.append(f"{r[value_col]:.0f}")
            if parts:
                cv2.putText(out, "/".join(parts),
                            (p2[0] + 2, p2[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, row_color, 1, cv2.LINE_AA)
    return out


def annotation_verification_figure(gray: np.ndarray, df: "Any", out_path: str | Path,
                                   *, title: str = "", zoom_boxes: int = 3,
                                   zoom_size: int = 220) -> Path:
    """Full-frame overlay plus zoomed insets, for eyeballing the extraction.

    The insets matter more than the full frame: at 1280x960 with 500 annotations
    the only way to see whether a chord really spans its fiber is to magnify.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    canvas = draw_measurements(to_rgb(gray), df, show_ids=False)
    ensure_dir(Path(out_path).parent)
    fig = plt.figure(figsize=(14, 10), dpi=130)
    gs = fig.add_gridspec(2, zoom_boxes, height_ratios=[2.6, 1.0])
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(canvas)
    ax.set_title(title or f"{len(df)} recovered measurements", fontsize=10)
    ax.axis("off")

    rng = np.random.default_rng(0)
    if len(df):
        picks = rng.choice(len(df), size=min(zoom_boxes, len(df)), replace=False)
        for k, idx in enumerate(picks):
            r = df.iloc[int(idx)]
            cx, cy = int(r["center_x_px"]), int(r["center_y_px"])
            h = zoom_size // 2
            x0, y0 = max(0, cx - h), max(0, cy - h)
            x1, y1 = min(canvas.shape[1], cx + h), min(canvas.shape[0], cy + h)
            sub = df[(df.center_x_px.between(x0, x1)) & (df.center_y_px.between(y0, y1))]
            tile = draw_measurements(to_rgb(gray), sub, show_ids=True, font_scale=0.35)
            axz = fig.add_subplot(gs[1, k])
            axz.imshow(tile[y0:y1, x0:x1])
            axz.set_title(f"zoom @ ({cx}, {cy})", fontsize=8)
            axz.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", out_path)
    return Path(out_path)


def profile_figure(geom: dict[str, Any], out_path: str | Path) -> Path | None:
    """Plot the pooled intensity profile that justified the marker geometry."""
    if not geom or not geom.get("profile"):
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = np.asarray(geom["profile"], float)
    t = np.linspace(-1.5, 1.5, p.size)
    ensure_dir(Path(out_path).parent)
    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=140)
    ax.plot(t, p, lw=1.6)
    ax.axvspan(-0.5, 0.5, color="0.85", zorder=0)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xlabel("position along the measurement line (units of reported width)")
    ax.set_ylabel("mean grey level")
    ax.set_title(f"offset=({geom['dx']:+.0f}, {geom['dy']:+.0f})  "
                 f"y_sign={geom['y_sign']:+.0f}  "
                 f"contrast={geom['contrast']:.0f}  "
                 f"asymmetry={geom['asymmetry']:.1f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return Path(out_path)


def prediction_panel(gray: np.ndarray, maps: dict[str, np.ndarray],
                     out_path: str | Path, *, title: str = "") -> Path:
    """Grid of the network's dense outputs (heatmap, seg, width, angle, sigma)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(Path(out_path).parent)
    items = [(k, v) for k, v in maps.items() if v is not None]
    n = len(items) + 1
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), dpi=110)
    axes = np.atleast_1d(axes).ravel()
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("SEM (input)", fontsize=9)
    axes[0].axis("off")
    for ax, (name, m) in zip(axes[1:], items):
        cmap = "twilight" if "angle" in name or "orient" in name else "viridis"
        im = ax.imshow(m, cmap=cmap)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.04)
    for ax in axes[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


def match_figure(gray: np.ndarray, gt: "Any", pred: "Any", matches: "Any",
                 out_path: str | Path, *, title: str = "") -> Path:
    """Ground truth, predictions, matched pairs, FP and FN on one canvas."""
    import cv2

    canvas = to_rgb(gray)
    matched_gt = set(matches["gt_index"].tolist()) if len(matches) else set()
    matched_pr = set(matches["pred_index"].tolist()) if len(matches) else set()

    canvas = draw_measurements(canvas, gt, color=GT_COLOR, show_ids=False)
    canvas = draw_measurements(canvas, pred, color=PRED_COLOR, show_ids=False)
    if len(matches):
        for _, m in matches.iterrows():
            g = gt.iloc[int(m["gt_index"])]
            p = pred.iloc[int(m["pred_index"])]
            cv2.line(canvas,
                     (int(g["center_x_px"]), int(g["center_y_px"])),
                     (int(p["center_x_px"]), int(p["center_y_px"])),
                     MATCH_COLOR, 1, cv2.LINE_AA)
    for i, (_, p) in enumerate(pred.iterrows()):
        if i not in matched_pr:
            cv2.circle(canvas, (int(p["center_x_px"]), int(p["center_y_px"])),
                       4, FP_COLOR, 1, cv2.LINE_AA)
    for i, (_, g) in enumerate(gt.iterrows()):
        if i not in matched_gt:
            cv2.circle(canvas, (int(g["center_x_px"]), int(g["center_y_px"])),
                       4, FN_COLOR, 1, cv2.LINE_AA)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ensure_dir(Path(out_path).parent)
    fig, ax = plt.subplots(figsize=(13, 10), dpi=130)
    ax.imshow(canvas)
    ax.axis("off")
    ax.set_title(title or "amber = manual, cyan = predicted, green = matched, "
                          "red = false positive, violet = missed", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


def error_map_figure(gray: np.ndarray, matches: "Any", out_path: str | Path,
                     *, unit: str = "px") -> Path:
    """Colour-code each matched prediction by absolute thickness error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(Path(out_path).parent)
    fig, ax = plt.subplots(figsize=(12, 9), dpi=130)
    ax.imshow(gray, cmap="gray")
    if len(matches):
        err = np.abs(matches["pred_width"].to_numpy() - matches["gt_width"].to_numpy())
        sc = ax.scatter(matches["pred_x"], matches["pred_y"], c=err, s=14,
                        cmap="magma", vmin=0, vmax=np.nanpercentile(err, 95))
        fig.colorbar(sc, ax=ax, label=f"|error| ({unit})", fraction=0.03)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


def distribution_figure(gt_widths: np.ndarray, pred_widths: np.ndarray,
                        out_path: str | Path, *, unit: str = "px") -> Path:
    """Histogram + KDE + Bland-Altman comparison of the two distributions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    ensure_dir(Path(out_path).parent)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=130)
    bins = np.histogram_bin_edges(np.concatenate([gt_widths, pred_widths]), bins=40)
    axes[0].hist(gt_widths, bins=bins, alpha=0.55, label="manual", density=True)
    axes[0].hist(pred_widths, bins=bins, alpha=0.55, label="predicted", density=True)
    axes[0].set_xlabel(f"thickness ({unit})")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8)

    grid = np.linspace(bins[0], bins[-1], 300)
    for data, name in ((gt_widths, "manual"), (pred_widths, "predicted")):
        if data.size > 2 and np.ptp(data) > 0:
            axes[1].plot(grid, stats.gaussian_kde(data)(grid), label=name)
    axes[1].set_xlabel(f"thickness ({unit})")
    axes[1].set_title("kernel density", fontsize=9)
    axes[1].legend(fontsize=8)

    n = min(gt_widths.size, pred_widths.size)
    if n:
        mean = (gt_widths[:n] + pred_widths[:n]) / 2
        diff = pred_widths[:n] - gt_widths[:n]
        bias, sd = float(np.mean(diff)), float(np.std(diff, ddof=1)) if n > 1 else 0.0
        axes[2].scatter(mean, diff, s=8, alpha=0.5)
        for y, ls in ((bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")):
            axes[2].axhline(y, ls=ls, lw=1, color="crimson")
        axes[2].set_xlabel(f"mean of the pair ({unit})")
        axes[2].set_ylabel(f"predicted - manual ({unit})")
        axes[2].set_title(f"Bland-Altman: bias {bias:+.2f}, "
                          f"LoA [{bias - 1.96 * sd:+.2f}, {bias + 1.96 * sd:+.2f}]",
                          fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


def training_curves(history: dict[str, Sequence[float]], out_path: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(Path(out_path).parent)
    keys = [k for k, v in history.items() if len(v)]
    fig, axes = plt.subplots(1, max(1, len(keys)), figsize=(4 * max(1, len(keys)), 3.2),
                             dpi=130, squeeze=False)
    for ax, k in zip(axes[0], keys):
        ax.plot(history[k])
        ax.set_title(k, fontsize=9)
        ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


# --------------------------------------------------------------------------- #
# ---- v6.4 -----------------------------------------------------------------
# A full field carries 2000+ chords.  Drawn all at once with their id labels it
# is unreadable at any size a thesis page allows, so nobody can actually check
# whether the chords sit on fibres -- which is the only thing the figure is for.
# --------------------------------------------------------------------------- #
def measurement_zoom_panel(gray, df, out_path, *, n_panels: int = 4,
                           crop: int = 180, title: str = "",
                           value_col: str = "width_px", seed: int = 0,
                           color=(255, 210, 40),
                           source_colors: dict[str, tuple[int, int, int]] | None = None,
                           prefer_source: str | None = None):
    """Whole field with chords plus N zoomed crops with values.

    If ``prefer_source`` is present (for example ``"thick_recovery"``), the
    first zoom is forced onto the widest row from that source.  This makes the
    recovery auditable instead of letting four random/busy crops all show only
    the easy thin fibres.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as _np
    from pathlib import Path as _P

    rgb = to_rgb(gray)
    full = draw_measurements(rgb, df, show_ids=False, color=color, thickness=1,
                             source_colors=source_colors)

    h, w = gray.shape[:2]
    rng = _np.random.default_rng(seed)
    centres = []

    # First panel: deliberately show the new recovery branch when it exists.
    if (prefer_source and len(df) and "measurement_source" in df.columns
            and (df["measurement_source"].astype(str) == prefer_source).any()):
        pref = df[df["measurement_source"].astype(str) == prefer_source].copy()
        if "width_px" in pref.columns:
            pref = pref.sort_values("width_px", ascending=False)
        r0 = pref.iloc[0]
        centres.append((float(r0["center_x_px"]), float(r0["center_y_px"])))

    if len(df) and "center_x_px" in df.columns:
        xs = df["center_x_px"].to_numpy(float)
        ys = df["center_y_px"].to_numpy(float)
        ok = _np.isfinite(xs) & _np.isfinite(ys)
        xs, ys = xs[ok], ys[ok]
        gx = _np.clip((xs / max(w / 3, 1)).astype(int), 0, 2)
        gy = _np.clip((ys / max(h / 3, 1)).astype(int), 0, 2)
        cells = {}
        for cx, cy, x, y in zip(gx, gy, xs, ys):
            cells.setdefault((cx, cy), []).append((x, y))
        for _key, pts in sorted(cells.items(), key=lambda kv: -len(kv[1])):
            if len(centres) >= n_panels:
                break
            px, py = pts[rng.integers(len(pts))]
            # Avoid duplicating the forced thick-fibre panel.
            if any((px - qx) ** 2 + (py - qy) ** 2 < (0.6 * crop) ** 2
                   for qx, qy in centres):
                continue
            centres.append((float(px), float(py)))

    while len(centres) < n_panels:
        centres.append((rng.uniform(crop, max(w - crop, crop + 1)),
                        rng.uniform(crop, max(h - crop, crop + 1))))

    fig = plt.figure(figsize=(11, 7.4), dpi=170)
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(full)
    label = f"{title}  ({len(df)} measurements)"
    if source_colors and "measurement_source" in df.columns:
        n_rec = int((df["measurement_source"].astype(str) == "thick_recovery").sum())
        if n_rec:
            label += f" | {n_rec} recovered thick"
    ax.set_title(label, fontsize=8)
    ax.axis("off")

    for i, (cx, cy) in enumerate(centres[:n_panels]):
        x0 = int(_np.clip(cx - crop // 2, 0, max(w - crop, 0)))
        y0 = int(_np.clip(cy - crop // 2, 0, max(h - crop, 0)))
        ax.add_patch(plt.Rectangle((x0, y0), crop, crop, fill=False,
                                   edgecolor="red", linewidth=0.8))
        ax.text(x0 + 4, y0 + 14, str(i + 1), color="red", fontsize=7)
        sub = df.copy()
        if "center_x_px" in sub.columns:
            sub = sub[(sub.center_x_px >= x0) & (sub.center_x_px < x0 + crop)
                      & (sub.center_y_px >= y0) & (sub.center_y_px < y0 + crop)]
        panel = draw_measurements(to_rgb(gray[y0:y0 + crop, x0:x0 + crop]),
                                  _shift_df(sub, x0, y0), show_ids=True,
                                  value_col=value_col, color=color,
                                  source_colors=source_colors,
                                  font_scale=0.32, thickness=1,
                                  max_labels=18)
        axp = fig.add_subplot(2, 4, (3, 4, 7, 8)[i] if n_panels == 4 else 2 + i)
        axp.imshow(panel)
        ttl = f"{i + 1}: {len(sub)} chords"
        if "measurement_source" in sub.columns:
            nr = int((sub["measurement_source"].astype(str) == "thick_recovery").sum())
            if nr:
                ttl += f" ({nr} thick)"
        axp.set_title(ttl, fontsize=7)
        axp.axis("off")

    fig.tight_layout()
    _P(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return _P(out_path)


def _shift_df(df, x0, y0):
    """Copy of df with coordinates moved into a crop's frame."""
    out = df.copy()
    for c in ("center_x_px", "x1_px", "x2_px"):
        if c in out.columns:
            out[c] = out[c] - x0
    for c in ("center_y_px", "y1_px", "y2_px"):
        if c in out.columns:
            out[c] = out[c] - y0
    return out
