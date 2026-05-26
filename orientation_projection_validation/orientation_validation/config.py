from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any

from .paths import ensure_structural_labeling_on_path

ensure_structural_labeling_on_path()

from labeling.config import LabelConfig  # noqa: E402


@dataclass(slots=True)
class ProjectionConfig:
    grid_size: int = 69
    pixel_scale_arcsec: float = 0.5
    psf_fwhm_arcsec: float = 1.43
    primary_rcov_reff: float = 1.5
    secondary_rcov_reff: float = 2.5
    n_eff_min: float = 30.0
    entropy_max: float = 0.75
    max_prob_min: float = 0.50
    valid_min_component_pixels: int = 5
    fvalid_min: float = 0.60
    cglobal_min: float = 0.60
    bulge_disk_min: float = 0.70
    main_metric_variant: str = "Y_lum_psf"
    orientation_degrees: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["orientation_degrees"] = list(self.orientation_degrees)
        return data


def load_configs(path: str | Path | None = None) -> tuple[ProjectionConfig, LabelConfig]:
    projection = ProjectionConfig()
    label = LabelConfig()
    if not path:
        return projection, label

    data = json.loads(Path(path).read_text())
    projection_keys = {field.name for field in fields(ProjectionConfig)}
    label_keys = {field.name for field in fields(LabelConfig)}

    projection_payload = {key: value for key, value in data.items() if key in projection_keys}
    if "orientation_degrees" in projection_payload:
        projection_payload["orientation_degrees"] = tuple(float(v) for v in projection_payload["orientation_degrees"])
    projection = ProjectionConfig(**{**projection.to_dict(), **projection_payload})

    label_payload: dict[str, Any] = {}
    if isinstance(data.get("label_config"), dict):
        label_payload.update(data["label_config"])
    label_payload.update({key: value for key, value in data.items() if key in label_keys})
    if label_payload:
        label = LabelConfig(**{**label.to_dict(), **label_payload})
    return projection, label

