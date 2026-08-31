"""Synthetic fibre fields with KNOWN widths and raster angles (v7).

Used by the automated tests and by the FAST_SMOKE_TEST mode so the whole
pipeline can be exercised without Drive data.  Synthetic success proves that
the code recovers what it was told; it is never evidence about real images.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .coords import RASTER, chord_endpoints, fiber_angle_from_measurement, wrap180


@dataclass
class SynthField:
    image: np.ndarray            # float32 [0, 255]
    mask: np.ndarray             # bool, true fibre pixels
    fibres: "Any"                # DataFrame: width, angle (raster), cx, cy, L
    annotations: "Any"           # DataFrame in the v7 label schema
    nm_per_px: float


def make_field(seed: int = 0, *, H: int = 512, W: int = 512, n_fibres: int = 30,
               width_median: float = 9.0, width_sigma: float = 0.5,
               width_min: float = 4.0, width_max: float = 40.0,
               mean_angle: float | None = None, angle_spread: float = 90.0,
               noise: float = 0.05, blur: float = 1.0, nm_per_px: float = 2.0,
               n_annotations: int = 120, image_id: str = "synth") -> SynthField:
    import pandas as pd

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W), np.float64)
    mask = np.zeros((H, W), bool)
    rows = []
    widths = np.clip(rng.lognormal(np.log(width_median), width_sigma, n_fibres),
                     width_min, width_max)
    for w in widths:
        a = (float(rng.uniform(-90, 90)) if mean_angle is None
             else float(wrap180(mean_angle + angle_spread * rng.standard_normal())))
        th = np.deg2rad(a)
        ux, uy = np.cos(th), np.sin(th)             # raster direction along the fibre
        cx, cy = rng.uniform(0.1 * W, 0.9 * W), rng.uniform(0.1 * H, 0.9 * H)
        L = rng.uniform(0.35, 0.8) * max(H, W)
        d = np.abs(-(xx - cx) * uy + (yy - cy) * ux)   # distance to the axis
        t = (xx - cx) * ux + (yy - cy) * uy
        band = (d <= w / 2.0) & (np.abs(t) <= L / 2.0)
        # cylinder-like shading: brighter along the centre line
        shade = np.clip(1.0 - (d / (w / 2.0 + 1e-6)) ** 2, 0.0, 1.0)
        img = np.where(band, np.maximum(img, 0.55 + 0.45 * shade), img)
        mask |= band
        rows.append(dict(width=float(w), angle=a, cx=cx, cy=cy, ux=ux, uy=uy, L=L))
    from scipy import ndimage as ndi

    if blur > 0:
        img = ndi.gaussian_filter(img, blur)
    img = np.clip(img + noise * rng.standard_normal((H, W)), 0, 1) * 255.0
    fibres = pd.DataFrame(rows)

    ann = []
    k = 0
    while len(ann) < n_annotations and k < 20 * n_annotations:
        k += 1
        f = fibres.iloc[int(rng.integers(len(fibres)))]
        t = rng.uniform(-0.4, 0.4) * f.L
        cx, cy = f.cx + f.ux * t, f.cy + f.uy * t
        if not (8 <= cx < W - 8 and 8 <= cy < H - 8):
            continue
        # skip crossings: the true mask must be a single fibre here
        wsum = 0
        for _, g in fibres.iterrows():
            dd = abs(-(cx - g.cx) * g.uy + (cy - g.cy) * g.ux)
            tt = (cx - g.cx) * g.ux + (cy - g.cy) * g.uy
            if dd <= g.width / 2 + 1 and abs(tt) <= g.L / 2:
                wsum += 1
        if wsum != 1:
            continue
        meas = float(wrap180(f.angle + 90.0))
        x1, y1, x2, y2 = chord_endpoints(cx, cy, meas, f.width)
        ann.append({
            "image_id": image_id, "annotation_id": len(ann) + 1,
            "center_x_px": float(cx), "center_y_px": float(cy),
            "x1_px": float(x1), "y1_px": float(y1), "x2_px": float(x2), "y2_px": float(y2),
            "measurement_angle_raster_deg": meas,
            "fiber_angle_raster_deg": float(fiber_angle_from_measurement(meas)),
            "width_px": float(f.width), "width_nm": float(f.width) * nm_per_px,
            "nm_per_pixel": nm_per_px, "calibration_status": "manual",
            "calibration_valid": True,
            "source_angle_deg": float(wrap180(-meas)),
            "angle_source_convention": RASTER, "imagej_angle_deg": np.nan,
            "annotation_confidence": 1.0, "ambiguous_crossing": False,
            "is_negative": False, "extraction_path": "synthetic", "source_csv": "",
        })
    from .labels import ensure_schema

    return SynthField(img.astype(np.float32), mask, fibres,
                      ensure_schema(pd.DataFrame(ann)), nm_per_px)


def write_synthetic_dataset(root: str | Path, *, n_specimens: int = 5,
                            fields_per_specimen: int = 2, seed: int = 7,
                            H: int = 384, W: int = 384, n_annotations: int = 60,
                            with_footer: bool = False) -> dict[str, Any]:
    """Write ``<root>/original/*.png``, ``<root>/processed/labels.csv`` and a
    manual calibration table, for the CPU smoke test.

    Specimens differ in width distribution and nm/px so the grouped split has
    real structure to protect.
    """
    import cv2
    import pandas as pd

    root = Path(root)
    (root / "original").mkdir(parents=True, exist_ok=True)
    (root / "processed").mkdir(parents=True, exist_ok=True)
    frames, cal, truth = [], {}, {}
    rng = np.random.default_rng(seed)
    for s in range(n_specimens):
        wmed = float(rng.uniform(6.0, 14.0))
        nmpp = float(rng.choice([1.0, 1.5, 2.0, 2.5, 3.0]))
        for f in range(fields_per_specimen):
            iid = f"S{s + 1}-{f + 1}"
            fld = make_field(seed * 1000 + s * 10 + f, H=H, W=W, n_fibres=22,
                             width_median=wmed, nm_per_px=nmpp,
                             n_annotations=n_annotations, image_id=iid)
            img = fld.image
            if with_footer:
                footer = np.zeros((40, W), np.float32)
                img = np.vstack([img, footer])
            cv2.imwrite(str(root / "original" / f"{iid}.png"),
                        np.clip(img, 0, 255).astype(np.uint8))
            frames.append(fld.annotations)
            cal[iid] = nmpp
            truth[iid] = fld.fibres
    labels = pd.concat(frames, ignore_index=True)
    labels.to_csv(root / "processed" / "labels.csv", index=False)
    import yaml

    (root / "calibration.yaml").write_text(yaml.safe_dump(cal), encoding="utf-8")
    return {"labels_csv": str(root / "processed" / "labels.csv"),
            "image_dir": str(root / "original"),
            "calibration_yaml": str(root / "calibration.yaml"),
            "calibration": dict(cal),
            "truth": truth, "n_images": len(cal)}
