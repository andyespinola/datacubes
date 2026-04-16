from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class LabelConfig:
    fine_grid_size: int = 256
    faceon_padding_factor: float = 1.2
    projection_smoothing_sigma_px: float = 1.0
    arm_residual_sigma_px: float = 2.0
    arm_residual_threshold: float = 0.15
    bulge_width_fraction: float = 0.35
    other_width_fraction: float = 0.20
    bar_radius_fraction: float = 1.0
    bar_soft_edge_fraction: float = 0.15
    bar_min_strength: float = 0.10
    hard_label_min_prob: float = 0.50
    hard_label_thresholds: list[float] = field(default_factory=lambda: [0.50, 0.55, 0.60])
    hard_label_margin: float = 0.15
    valid_smoothing_sigma_px: float = 1.5
    valid_flux_percentile: float = 35.0
    valid_peak_fraction: float = 0.02
    valid_min_component_pixels: int = 24
    valid_closing_iterations: int = 0
    hard_spatial_sigma_px: float = 1.0
    hard_spatial_blend: float = 0.45
    hard_min_component_pixels: int = 5
    central_radius_scale: float = 1.0
    central_bulge_boost: float = 0.65
    central_disk_suppression: float = 0.30
    max_calibration_iters: int = 25
    family_scaling_iters: int = 20
    gas_arm_boost: float = 0.25

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "LabelConfig":
        data: dict[str, Any] = json.loads(Path(path).read_text())
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    def resolved_hard_label_thresholds(self) -> list[float]:
        thresholds = list(self.hard_label_thresholds)
        thresholds.append(float(self.hard_label_min_prob))
        return sorted({round(float(value), 4) for value in thresholds})
