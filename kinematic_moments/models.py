from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


DEFAULT_EMISSION_LINES = (
    3727.0,
    4861.0,
    4959.0,
    5007.0,
    6300.0,
    6548.0,
    6563.0,
    6583.0,
    6717.0,
    6731.0,
)


@dataclass(slots=True)
class KinematicMomentsConfig:
    template_path: Path | None = None
    moments: int = 4
    degree: int = 4
    mdegree: int = 0
    bias: float | None = None
    velscale_ratio: int = 2
    wave_range_fit: tuple[float, float] = (3700.0, 7400.0)
    snr_window: tuple[float, float] = (5000.0, 7000.0)
    snr_min: float = 5.0
    start_velocity: float = 0.0
    start_sigma: float = 150.0
    emission_lines_mask: tuple[float, ...] = DEFAULT_EMISSION_LINES
    emission_mask_width_kms: float = 800.0
    min_goodpixels: int = 50
    min_valid_fraction: float = 0.8
    quiet: bool = True
    max_templates: int | None = None
    template_norm_range: tuple[float, float] = (5070.0, 5950.0)


@dataclass(slots=True)
class OfficialCube:
    path: Path
    galaxy_id: str
    flux: np.ndarray
    error: np.ndarray
    valid_cube: np.ndarray
    wave: np.ndarray
    redshift: float
    header: fits.Header

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return int(self.flux.shape[1]), int(self.flux.shape[2])


@dataclass(slots=True)
class TemplateLibrary:
    path: Path
    templates: np.ndarray
    lam_temp: np.ndarray
    velscale: float
    n_templates: int


@dataclass(slots=True)
class KinematicMaps:
    galaxy_id: str
    cube_path: Path
    h3: np.ndarray
    h4: np.ndarray
    v_ppxf: np.ndarray
    sigma_ppxf: np.ndarray
    h3_err: np.ndarray
    h4_err: np.ndarray
    quality_mask: np.ndarray
    coverage_mask: np.ndarray
    snr_map: np.ndarray
    chi2_map: np.ndarray
    n_spaxels_fitted: int
    n_quality_ok: int
    config_summary: dict[str, Any] = field(default_factory=dict)
    message: str = ""
