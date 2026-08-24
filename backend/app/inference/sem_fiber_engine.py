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


def map_predictions(frame: pd.DataFrame) -> list[MeasurementPrediction]:
    mapped: list[MeasurementPrediction] = []
    for row in frame.to_dict(orient="records"):
        width_nm = _finite_or_none(row.get("width_nm"))
        metadata = {
            key: value
            for key, value in {
                "validity": _finite_or_none(row.get("validity")),
                "uncertainty_px": _finite_or_none(row.get("width_sigma_px")),
                "measurement_method": row.get("measurement_method"),
                "local_fiber_angle_deg": _finite_or_none(row.get("local_fiber_angle_deg")),
                "recovered_thick": bool(row.get("recovered_thick", False)),
            }.items()
            if value is not None
        }
        mapped.append(
            MeasurementPrediction(
                external_id=(str(row["prediction_id"]) if row.get("prediction_id") is not None else None),
                x1=float(row["x1_px"]),
                y1=float(row["y1_px"]),
                x2=float(row["x2_px"]),
                y2=float(row["y2_px"]),
                width_px=float(row["width_px"]),
                width_nm=width_nm,
                angle_deg=float(row.get("measurement_angle_deg", 0.0)),
                confidence=float(row.get("confidence", 0.0)),
                source=str(row.get("measurement_source", "ai")),
                metadata=metadata,
            )
        )
    return mapped


class SemFiberEngine:
    """Lazy wrapper around the v6.11 notebook's vendored sem_fiber_ai package."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        peak_threshold: float = 0.30,
        min_validity: float = 0.30,
        width_calibration_path: str | Path | None = None,
        calibration_table_path: str | Path | None = None,
        thick_recovery: bool = True,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device_name = device
        self.peak_threshold = float(peak_threshold)
        self.min_validity = float(min_validity)
        self.width_calibration_path = Path(width_calibration_path) if width_calibration_path else None
        self.calibration_table_path = Path(calibration_table_path) if calibration_table_path else None
        self.thick_recovery = thick_recovery
        self._model = None
        self._device = None
        self._checkpoint_meta: dict[str, Any] | None = None

    @staticmethod
    def _ensure_vendor_path() -> None:
        backend_root = Path(__file__).resolve().parents[2]
        vendor = backend_root / "vendor"
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"SEM model checkpoint not found: {self.checkpoint_path}. "
                "Mount the trained v6.11 checkpoint and set MODEL_CHECKPOINT."
            )
        self._ensure_vendor_path()
        from sem_fiber_ai.src.infer import load_checkpoint
        from sem_fiber_ai.src.utils import pick_device

        self._device = pick_device(self.device_name)
        self._model, self._checkpoint_meta = load_checkpoint(self.checkpoint_path, self._device)

    def analyze(
        self,
        image_path: Path,
        output_dir: Path,
        nm_per_pixel: float | None = None,
    ) -> AnalysisResult:
        self._load()
        self._ensure_vendor_path()
        from sem_fiber_ai.src.calibration import load_calibration_table
        from sem_fiber_ai.src.infer import run_one
        from sem_fiber_ai.src.postprocess import PostConfig

        width_calib = None
        if self.width_calibration_path and self.width_calibration_path.is_file():
            from sem_fiber_ai.src.width_calibration import load_width_calibration
            width_calib = load_width_calibration(self.width_calibration_path)

        table = {}
        if self.calibration_table_path and self.calibration_table_path.is_file():
            table = load_calibration_table(self.calibration_table_path)
        thick_cfg = None
        if self.thick_recovery:
            from sem_fiber_ai.src.thick_fiber import ThickRecoveryConfig
            thick_cfg = ThickRecoveryConfig(enabled=True)

        output_dir.mkdir(parents=True, exist_ok=True)
        frame = run_one(
            self._model,
            image_path,
            output_dir,
            self._device,
            nm_per_pixel=nm_per_pixel,
            calib_table=table,
            post=PostConfig(
                peak_threshold=self.peak_threshold,
                min_validity=self.min_validity,
            ),
            tile=512,
            overlap=64,
            tta=False,
            mc_samples=0,
            save_maps=False,
            width_calib=width_calib,
            zoom_panels=0,
            thick_cfg=thick_cfg,
        )

        summary_path = output_dir / f"{image_path.stem}_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        artifacts = {
            name: path
            for name, path in {
                "predictions_csv": output_dir / f"{image_path.stem}_predictions.csv",
                "annotated_png": output_dir / f"{image_path.stem}_annotated.png",
                "summary_json": summary_path,
            }.items()
            if path.is_file()
        }
        return AnalysisResult(
            measurements=map_predictions(frame),
            summary=summary,
            artifacts=artifacts,
        )
