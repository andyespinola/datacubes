from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(slots=True)
class ManifestRow:
    canonical_id: str
    rss_path: str
    cube_path: str
    pipe3d_path: str
    snapshot: int
    subhalo_id: int
    view: int
    re_kpc: float
    ifu_design: int
    repeat_count: int
    n_star_part: int
    n_gas_cell: int

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "canonical_id": self.canonical_id,
            "rss_path": self.rss_path,
            "cube_path": self.cube_path,
            "pipe3d_path": self.pipe3d_path,
            "snapshot": self.snapshot,
            "subhalo_id": self.subhalo_id,
            "view": self.view,
            "re_kpc": self.re_kpc,
            "ifu_design": self.ifu_design,
            "repeat_count": self.repeat_count,
            "n_star_part": self.n_star_part,
            "n_gas_cell": self.n_gas_cell,
        }


@dataclass(slots=True)
class CubeGeometry:
    shape: tuple[int, int]
    valid_mask: np.ndarray
    base_valid_mask: np.ndarray
    signal_map: np.ndarray
    valid_threshold: float
    pixel_scale_arcsec: float
    kpc_per_arcsec: float
    psf_fwhm_arcsec: float
    header_summary: dict[str, float | int | str]


@dataclass(slots=True)
class TNGTruth:
    stellar_pos: np.ndarray
    stellar_vel: np.ndarray
    stellar_mass: np.ndarray
    stellar_age_gyr: np.ndarray
    stellar_metallicity: np.ndarray
    gas_pos: np.ndarray | None
    gas_vel: np.ndarray | None
    gas_mass: np.ndarray | None
    gas_sfr: np.ndarray | None
    gas_metallicity: np.ndarray | None
    gas_density: np.ndarray | None
    subhalo_pos: np.ndarray
    subhalo_vel: np.ndarray
    stellar_halfmass_rad: float


@dataclass(slots=True)
class MorphologyTargets:
    thin_disk: float
    thick_disk: float
    pseudo_bulge: float
    bulge: float
    halo: float
    unbound: float
    barred: bool
    bar_size_kpc: float
    bar_size_alt_kpc: float
    bar_strength: float
    bar_strength_alt: float
    quality_krot: float
    quality_sigma_ratio: float
    quality_b1b2: float

    @property
    def disk_family(self) -> float:
        return max(0.0, self.thin_disk + self.thick_disk)

    @property
    def bulge_family(self) -> float:
        return max(0.0, self.pseudo_bulge + self.bulge)

    @property
    def other_family(self) -> float:
        return float(max(0.0, self.halo + self.unbound))


@dataclass(slots=True)
class LabelProducts:
    soft_mass: np.ndarray
    soft_light: np.ndarray
    hard_mass: np.ndarray
    hard_light: np.ndarray
    hard_mass_variants: dict[str, np.ndarray]
    hard_light_variants: dict[str, np.ndarray]
    confidence_mass: np.ndarray
    confidence_light: np.ndarray
    valid_mask: np.ndarray
    qa_maps: dict[str, np.ndarray]
    bar_metadata: dict[str, float | int | bool]
    global_fraction_targets: dict[str, float | int | bool | str]
    global_fraction_recovered: dict[str, float]
    hard_variant_summary: dict[str, dict[str, dict[str, int]]]
