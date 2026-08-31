from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import AnalysisResult, MeasurementPrediction


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None




def _apply_fibervision_review_profile(run: dict[str, Any]) -> bool:
    """Use a recall-first post profile when validation selection is unavailable.

    FiberVision is a human-review workflow: candidates can be filtered by model
    confidence and then corrected/removed by the reviewer.  The v7 scientific
    defaults are intentionally conservative and, without selection.json, can
    reject nearly every site on dense unseen fields.  Only the unselected case
    is relaxed; a future validation-selected profile remains authoritative.
    """
    if run.get("selection"):
        return False
    post = run.get("post")
    if post is None:
        return False
    run["post"] = replace(
        post,
        seg_threshold=0.4,
        min_validity=0.0,
        min_seg_confidence=0.0,
        junction_clear_scale=0.0,
        boundary_tol=0.9,
        spacing_px=12.0,
    )
    return True

def _accepted_sites(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    if "rejected_reason" not in frame.columns:
        return frame.copy()
    reasons = frame["rejected_reason"].fillna("").astype(str).str.strip()
    return frame.loc[reasons.eq("")].copy()


def map_predictions(frame: pd.DataFrame) -> list[MeasurementPrediction]:
    """Map accepted sem_fiber_ai v7 measurement sites to FiberVision rows."""
    mapped: list[MeasurementPrediction] = []
    for row in _accepted_sites(frame).to_dict(orient="records"):
        width_nm = _finite_or_none(row.get("width_nm"))
        metadata = {
            key: value
            for key, value in {
                "validity": _finite_or_none(row.get("validity")),
                "uncertainty_px": _finite_or_none(row.get("width_sigma_px")),
                "fiber_angle_deg": (-_finite_or_none(row.get("fiber_angle_raster_deg")) if _finite_or_none(row.get("fiber_angle_raster_deg")) is not None else None),
                "boundary_disagreement": _finite_or_none(row.get("boundary_disagreement")),
                "junction_distance_px": _finite_or_none(row.get("junction_distance_px")),
                "coherence": _finite_or_none(row.get("coherence")),
                "branch_id": int(row["branch_id"]) if _finite_or_none(row.get("branch_id")) is not None else None,
            }.items()
            if value is not None
        }
        external = row.get("site_id")
        mapped.append(
            MeasurementPrediction(
                external_id=str(external) if external is not None else None,
                x1=float(row["x1_px"]),
                y1=float(row["y1_px"]),
                x2=float(row["x2_px"]),
                y2=float(row["y2_px"]),
                width_px=float(row["width_px"]),
                width_nm=width_nm,
                angle_deg=-float(row.get("measurement_angle_raster_deg", 0.0)),
                confidence=float(row.get("confidence", 0.0)),
                source=str(row.get("measurement_source", "model_geometry")),
                metadata=metadata,
            )
        )
    return mapped


class SemFiberEngine:
    """FiberVision adapter for the embedded sem_fiber_ai v7.0.0 inference package."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.run_dir = self.checkpoint_path.parent
        self.device_name = device
        self._run: dict[str, Any] | None = None

    @staticmethod
    def _ensure_vendor_path() -> None:
        backend_root = Path(__file__).resolve().parents[2]
        vendor = backend_root / "vendor"
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))

    def _load(self) -> dict[str, Any]:
        if self._run is not None:
            return self._run
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"SEM v7 checkpoint not found: {self.checkpoint_path}. "
                "Place best.pt in the configured v7 run directory."
            )
        self._ensure_vendor_path()
        from sem_fiber_ai.src.infer import load_run
        from sem_fiber_ai.src.utils import pick_device

        device = pick_device(self.device_name)
        self._run = load_run(self.run_dir, device=device)
        review_profile = _apply_fibervision_review_profile(self._run)
        self._run["fibervision_review_profile"] = review_profile
        package_version = str(self._run.get("checkpoint", {}).get("package_version") or "")
        if package_version and package_version != "7.0.0":
            raise ValueError(f"expected sem_fiber_ai 7.0.0 checkpoint, got {package_version}")
        return self._run

    def analyze(
        self,
        image_path: Path,
        output_dir: Path,
        nm_per_pixel: float | None = None,
    ) -> AnalysisResult:
        run = self._load()
        self._ensure_vendor_path()
        from sem_fiber_ai.src.infer import measure_image

        output_dir.mkdir(parents=True, exist_ok=True)
        summary = measure_image(
            run,
            image_path,
            output_dir,
            manual_nm_per_px=nm_per_pixel,
            include_split_members=True,
            tta=False,
            save_maps=False,
            thick=False,
        )
        summary["fibervision_review_profile"] = bool(run.get("fibervision_review_profile"))
        image_id = str(summary["image_id"])
        sites_path = output_dir / f"{image_id}_sites.csv"
        frame = pd.read_csv(sites_path) if sites_path.is_file() else pd.DataFrame()
        artifacts = {
            name: path
            for name, path in {
                "sites_csv": sites_path,
                "fibres_csv": output_dir / f"{image_id}_fibres.csv",
                "overlay_png": output_dir / f"{image_id}_overlay.png",
                "width_distribution_png": output_dir / f"{image_id}_width_distribution.png",
                "summary_json": output_dir / f"{image_id}_summary.json",
            }.items()
            if path.is_file()
        }
        return AnalysisResult(
            measurements=map_predictions(frame),
            summary=summary,
            artifacts=artifacts,
        )
