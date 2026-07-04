"""Contratos pydantic del pipeline v2 (docs/04_contratos.md).

Los dataclasses del v1 migran aquí como modelos pydantic con
`arbitrary_types_allowed` para los arrays numpy.
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


class _ArrayModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class TNGTruth(_ArrayModel):
    """Partículas crudas del cutout (estrellas + gas opcional + DM opcional)."""

    stellar_pos: np.ndarray
    stellar_vel: np.ndarray
    stellar_mass: np.ndarray
    stellar_formation_scale: np.ndarray  # factor de escala de formación (crudo)
    stellar_age_gyr: Optional[np.ndarray] = None  # poblado tras convert_truth_units
    stellar_metallicity: np.ndarray
    stellar_potential: Optional[np.ndarray] = None  # (km/s)^2 físicas si disponible
    gas_pos: Optional[np.ndarray] = None
    gas_vel: Optional[np.ndarray] = None
    gas_mass: Optional[np.ndarray] = None
    gas_sfr: Optional[np.ndarray] = None
    dm_pos: Optional[np.ndarray] = None
    dm_mass: Optional[np.ndarray] = None
    subhalo_pos: np.ndarray
    subhalo_vel: np.ndarray
    stellar_halfmass_rad: float
    snapshot: int
    subhalo_id: int
    scale_factor: float  # a del snapshot (del Header del cutout si existe)
    redshift: float


class SubhaloMeta(_ArrayModel):
    subhalo_id: int
    snapshot: int
    center_pos: tuple[float, float, float]  # ckpc/h
    bulk_vel: tuple[float, float, float]  # km/s (peculiar; ver conversión)
    r_eff_kpc: float
    m_star: float  # M_sun


class MorphologyTargets(BaseModel):
    """Fila MORDOR (morphs_kinematic_bars.hdf5) para un subhalo."""

    thin_disk: float
    thick_disk: float
    pseudo_bulge: float
    bulge: float
    halo: float
    unbound: float
    barred: bool
    bar_size_kpc: float  # R_Phi
    bar_size_alt_kpc: float  # R_peak
    bar_strength: float  # A2_max
    bar_strength_alt: float  # A2(<R_peak)
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


class CatalogPriors(BaseModel):
    """Prior de fracciones para el Classifier (spec 11)."""

    source: Literal["mordor", "rodriguez_gomez", "none"]
    bulge_frac: float
    disk_frac: float
    other_frac: float
    confidence: float = 0.5  # α; nunca 1.0


class BarMeta(BaseModel):
    """Metadatos de barra para BarDetector (spec 12), derivados de MORDOR."""

    has_bar: bool
    bar_size_kpc: Optional[float] = None
    bar_strength: Optional[float] = None
    bar_angle_deg: Optional[float] = None


class CubeGeometry(_ArrayModel):
    """Geometría observacional del cubo MaNGIA (header + máscara base)."""

    shape: tuple[int, int]
    base_valid_mask: np.ndarray
    signal_map: np.ndarray
    pixel_scale_arcsec: float
    kpc_per_arcsec: float
    psf_fwhm_arcsec: float
    redshift: float
    fov_arcsec: float
    wavelength_start: float  # CRVAL3 [Å]
    wavelength_step: float  # CDELT3 [Å]
    header_summary: dict

    @property
    def pixel_scale_kpc(self) -> float:
        return self.pixel_scale_arcsec * self.kpc_per_arcsec

    @property
    def psf_sigma_pixels(self) -> float:
        return self.psf_fwhm_arcsec / 2.355 / max(self.pixel_scale_arcsec, 1e-6)


class ViewDefinition(_ArrayModel):
    """Definición geométrica de una vista MaNGIA (spec 20, convención resuelta)."""

    view_id: int
    view_vector: tuple[float, float, float]  # línea de visión, marco simulación
    grid_shape: tuple[int, int]
    spaxel_scale_arcsec: float
    kpc_per_arcsec: float
    fwhm_psf_arcsec: float

    @property
    def spaxel_scale_kpc(self) -> float:
        return self.spaxel_scale_arcsec * self.kpc_per_arcsec


class ManifestRow(BaseModel):
    canonical_id: str
    cutout_path: str
    cutout_phase2_path: str
    subhalo_json_path: str
    cube_path: str
    pipe3d_maps_path: str
    snapshot: int
    subhalo_id: int
    view: int
    re_kpc: float
    ifu_design: int
    repeat_count: int


class ParticleFeaturesMeta(BaseModel):
    """Metadata/quality del producto particle_features.h5 (spec 10)."""

    galaxy_id: str
    snapshot: int
    subhalo_id: int
    n_particles: int
    n_central: int
    r_eff_kpc: float
    l_total: tuple[float, float, float]
    epsilon_mean: float
    epsilon_std: float
    epsilon_p7_fraction: float
    epsilon_n3_fraction: float
    potential_method_used: str
    compute_time_sec: float

    @field_validator("n_particles")
    @classmethod
    def _enough_particles(cls, v: int) -> int:
        if v < 100:
            raise ValueError(f"Resolución insuficiente: {v} partículas (<100)")
        return v
