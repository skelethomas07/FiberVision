"""Figures (v7).  All matplotlib, Agg backend, no display needed."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def overlay_figure(gray: np.ndarray, gt: "Any" | None, sites: "Any", out_path, *, title: str = "",
                   max_sites: int = 4000) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 10 * gray.shape[0] / max(gray.shape[1], 1)), dpi=110)
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    if gt is not None and len(gt) and {"x1_px", "y1_px", "x2_px", "y2_px"} <= set(gt.columns):
        for r in gt.itertuples():
            ax.plot([r.x1_px, r.x2_px], [r.y1_px, r.y2_px], "-", color="#00e5ff", lw=1.0, alpha=0.9)
    if len(sites):
        acc = sites[sites["rejected_reason"].fillna("") == ""] if "rejected_reason" in sites else sites
        rej = sites[sites["rejected_reason"].fillna("") != ""] if "rejected_reason" in sites else sites.iloc[0:0]
        for r in rej.head(max_sites).itertuples():
            ax.plot([r.x1_px, r.x2_px], [r.y1_px, r.y2_px], "-", color="#ff3b30", lw=0.6, alpha=0.5)
        for r in acc.head(max_sites).itertuples():
            ax.plot([r.x1_px, r.x2_px], [r.y1_px, r.y2_px], "-", color="#ffd60a", lw=1.0, alpha=0.95)
    ax.set_title(f"{title}   cyan=manual  yellow=accepted  red=rejected", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return Path(out_path)


def width_distribution_figure(gt_fibres: "Any", pred_fibres: "Any", nm_per_px: float | None, out_path,
                              *, title: str = "") -> Path:
    plt = _plt()
    g = gt_fibres["width_px"].to_numpy(float) if len(gt_fibres) else np.zeros(0)
    p = pred_fibres["width_px"].to_numpy(float) if len(pred_fibres) else np.zeros(0)
    fig, ax = plt.subplots(1, 2 if nm_per_px else 1, figsize=(11 if nm_per_px else 6, 3.6), dpi=120,
                           squeeze=False)
    for k, scale in enumerate([1.0] + ([nm_per_px] if nm_per_px else [])):
        a = ax[0][k]
        allv = np.concatenate([g, p]) * scale
        bins = np.linspace(0, np.percentile(allv, 99.5) * 1.05 if allv.size else 1, 40)
        if g.size:
            a.hist(g * scale, bins=bins, alpha=0.55, label=f"manual fibres (n={g.size})", color="#1f77b4")
        if p.size:
            a.hist(p * scale, bins=bins, histtype="step", lw=2, label=f"predicted fibres (n={p.size})",
                   color="#d62728")
        a.set_xlabel("fibre width (px)" if k == 0 else "fibre width (nm)")
        a.set_ylabel("fibres (number-weighted)")
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return Path(out_path)


def orientation_figure(summaries: dict[str, dict[str, Any]], out_path, *, title: str = "") -> Path:
    plt = _plt()
    n = max(1, len(summaries))
    fig, ax = plt.subplots(1, n, figsize=(4.2 * n, 3.2), dpi=120, squeeze=False)
    for k, (name, s) in enumerate(summaries.items()):
        a = ax[0][k]
        e = np.asarray(s["hist_edges_deg"], float)
        c = 0.5 * (e[1:] + e[:-1])
        w = np.asarray(s.get("hist_weight", s.get("hist_count")), float)
        a.bar(c, w, width=180 / len(c) * 0.9, color="#3b6ea5")
        a.set_title(f"{name}  S={s.get('order_parameter_S', float('nan')):.2f}", fontsize=9)
        a.set_xlabel("fibre direction, raster deg (y down)")
        a.set_xlim(-90, 90)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return Path(out_path)


def maps_panel(gray: np.ndarray, maps: dict[str, np.ndarray], out_path, *, title: str = "") -> Path:
    plt = _plt()
    panels = {"image": gray,
              "segmentation": 1 / (1 + np.exp(-maps["segment_logit"])),
              "medial-axis (centre)": 1 / (1 + np.exp(-maps["center_logit"])),
              "distance to boundary (px)": maps["dist"],
              "width head (px)": np.exp(maps["width"]),
              "validity": 1 / (1 + np.exp(-maps["validity_logit"])),
              "orientation (raster deg)": np.degrees(0.5 * np.arctan2(maps["orient"][1], maps["orient"][0])),
              "width sigma (px)": np.exp(0.5 * maps["logvar"]) * np.exp(maps["width"])}
    n = len(panels)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(4 * cols, 3.4 * rows), dpi=100)
    ax = np.asarray(ax).ravel()
    for a in ax:
        a.axis("off")
    for k, (name, arr) in enumerate(panels.items()):
        im = ax[k].imshow(arr, cmap="gray" if name == "image" else "viridis")
        ax[k].set_title(name, fontsize=9)
        if name != "image":
            fig.colorbar(im, ax=ax[k], fraction=0.046)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return Path(out_path)


def training_curves(history: dict[str, Any], out_path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=120)
    for k in ("train_loss", "val_loss", "val_monitor"):
        if history.get(k):
            ax.plot(np.arange(1, len(history[k]) + 1), history[k], label=k)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return Path(out_path)
