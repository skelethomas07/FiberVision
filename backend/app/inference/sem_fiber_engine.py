from __future__ import annotations

import json
import math
import sys
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


def _accepted_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    if "rejected_reason" not in frame.columns:
        return frame.copy()
    reasons = frame["rejected_reason"].fillna("").astype(str).str.strip()
    return frame.loc[reasons.eq("")].copy()


def map_predictions(frame: pd.DataFrame) -> list[MeasurementPrediction]:
    """Map sem_fiber_ai v6.12 combined AI/thick-recovery rows to FiberVision."""
    mapped: list[MeasurementPrediction] = []
    for row in _accepted_rows(frame).to_dict(orient="records"):
        width_nm = _finite_or_none(row.get("width_nm"))
        fiber_angle = _finite_or_none(row.get("local_fiber_angle_deg"))
        metadata_values = {
            "validity": _finite_or_none(row.get("validity")),
            "uncertainty_px": _finite_or_none(row.get("width_sigma_px")),
            "fiber_angle_deg": -fiber_angle if fiber_angle is not None else None,
            "measurement_method": row.get("measurement_method"),
            "recovered_thick": bool(row.get("recovered_thick")) if row.get("recovered_thick") is not None else None,
            "scale_sigma_px": _finite_or_none(row.get("scale_sigma_px")),
            "edt_width_px": _finite_or_none(row.get("edt_width_px")),
            "profile_width_px": _finite_or_none(row.get("profile_width_px")),
            "profile_contrast": _finite_or_none(row.get("profile_contrast")),
            "width_calibrated": bool(row.get("width_calibrated")) if row.get("width_calibrated") is not None else None,
        }
        metadata = {key: value for key, value in metadata_values.items() if value is not None}
        external = row.get("prediction_id")
        mapped.append(
            MeasurementPrediction(
                external_id=str(external) if external is not None else None,
                x1=float(row["x1_px"]),
                y1=float(row["y1_px"]),
                x2=float(row["x2_px"]),
                y2=float(row["y2_px"]),
                width_px=float(row["width_px"]),
                width_nm=width_nm,
                angle_deg=-float(row.get("measurement_angle_deg", 0.0)),
                confidence=float(row.get("confidence", 0.0)),
                source=str(row.get("measurement_source", "ai")),
                metadata=metadata,
            )
        )
    return mapped


class SemFiberEngine:
    """FiberVision adapter for the v6.12 notebook's full-model inference path."""

    # Validation-selected settings from notebook cell 8.
    peak_threshold = 0.4
    min_validity = 0.5

    # Cell 10 uses TTA and the hybrid thick-fibre supplement for new images.
    tta = True
    recover_thick = True
    thick_min_width_px = 18.0
    thick_max_width_px = 160.0
    thick_min_sigma = 8.0
    thick_spacing_px = 14.0
    thick_min_coherence = 0.45
    thick_segment_support = 0.15

    def __init__(self, checkpoint_path: str | Path, *, device: str = "auto") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device_name = device
        self._model: Any | None = None
        self._checkpoint: dict[str, Any] | None = None
        self._device: Any | None = None

    @staticmethod
    def _ensure_vendor_path() -> None:
        backend_root = Path(__file__).resolve().parents[2]
        vendor = backend_root / "vendor"
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))

    def _load(self) -> tuple[Any, dict[str, Any], Any]:
        if self._model is not None and self._checkpoint is not None and self._device is not None:
            return self._model, self._checkpoint, self._device
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"SEM v6.12 checkpoint not found: {self.checkpoint_path}")
        self._ensure_vendor_path()
        from sem_fiber_ai.src.infer import load_checkpoint
        from sem_fiber_ai.src.utils import pick_device

        device = pick_device(self.device_name)
        model, checkpoint = load_checkpoint(self.checkpoint_path, device)
        model_kind = str(checkpoint.get("model_kind") or "full")
        if model_kind != "full":
            raise ValueError(f"expected v6.12 full checkpoint, got model_kind={model_kind!r}")
        self._model = model
        self._checkpoint = checkpoint
        self._device = device
        return model, checkpoint, device

    def analyze(
        self,
        image_path: Path,
        output_dir: Path,
        nm_per_pixel: float | None = None,
    ) -> AnalysisResult:
        model, checkpoint, device = self._load()
        self._ensure_vendor_path()
        from sem_fiber_ai.src.infer import run_one
        from sem_fiber_ai.src.postprocess import PostConfig
        from sem_fiber_ai.src.thick_fiber import ThickRecoveryConfig

        output_dir.mkdir(parents=True, exist_ok=True)
        post = PostConfig(
            peak_threshold=self.peak_threshold,
            min_validity=self.min_validity,
        )
        thick_cfg = ThickRecoveryConfig(
            enabled=self.recover_thick,
            min_width_px=self.thick_min_width_px,
            max_width_px=self.thick_max_width_px,
            min_sigma=self.thick_min_sigma,
            spacing_px=self.thick_spacing_px,
            min_ridge_coherence=self.thick_min_coherence,
            segment_support=self.thick_segment_support,
        )
        frame = run_one(
            model,
            Path(image_path),
            Path(output_dir),
            device,
            nm_per_pixel=nm_per_pixel,
            calib_table={},
            post=post,
            tile=512,
            overlap=64,
            tta=self.tta,
            mc_samples=0,
            save_maps=False,
            width_calib=None,
            zoom_panels=0,
            thick_cfg=thick_cfg,
        )

        stem = Path(image_path).stem
        summary_path = Path(output_dir) / f"{stem}_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        summary["model_version"] = "v6.12"
        summary["checkpoint"] = {
            "epoch": checkpoint.get("epoch"),
            "best": checkpoint.get("best"),
            "model_kind": checkpoint.get("model_kind"),
        }
        summary["fibervision_inference"] = {
            "peak_threshold": self.peak_threshold,
            "min_validity": self.min_validity,
            "tta": self.tta,
            "thick_recovery": self.recover_thick,
        }

        artifacts = {
            name: path
            for name, path in {
                "predictions_csv": Path(output_dir) / f"{stem}_predictions.csv",
                "annotated_png": Path(output_dir) / f"{stem}_annotated.png",
                "thickness_histogram_png": Path(output_dir) / f"{stem}_thickness_histogram.png",
                "summary_json": summary_path,
            }.items()
            if path.is_file()
        }
        return AnalysisResult(
            measurements=map_predictions(frame),
            summary=summary,
            artifacts=artifacts,
        )
