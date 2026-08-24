from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class MeasurementPrediction:
    external_id: str | None
    x1: float
    y1: float
    x2: float
    y2: float
    width_px: float
    width_nm: float | None
    angle_deg: float
    confidence: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisResult:
    measurements: list[MeasurementPrediction]
    summary: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)


class InferenceEngine(Protocol):
    def analyze(
        self,
        image_path: Path,
        output_dir: Path,
        nm_per_pixel: float | None = None,
    ) -> AnalysisResult:
        ...
