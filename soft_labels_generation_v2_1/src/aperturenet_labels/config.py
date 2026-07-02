from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    return project_root().parent


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (project_root() / path).resolve()


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(slots=True)
class DataConfig:
    data_dir: Path = field(default_factory=lambda: workspace_root() / "data")
    morphology_catalog: Path = field(default_factory=lambda: workspace_root() / "data" / "morphs_kinematic_bars.hdf5")
    stellar_circularities: Path = field(default_factory=lambda: workspace_root() / "data" / "stellar_circs.hdf5")
    potential_cache_dir: Path = field(default_factory=lambda: workspace_root() / "data" / "potential_cache")
    ssp_template: Path = field(default_factory=lambda: workspace_root() / "kinematic_moments" / "templates" / "MaStar_CB19.slog_1_5.fits.gz")
    output_dir: Path = field(default_factory=lambda: project_root() / "outputs")
    local_galaxy_ids: tuple[str, ...] = ("TNG50-87-155298", "TNG50-87-192324")
    file_ifu_design: int = 127


@dataclass(slots=True)
class ExtractorConfig:
    align_radius_factor: float = 2.0
    max_particles: int = 250_000
    random_seed: int = 42
    use_potential_cache: bool = True
    require_potential_cache: bool = False
    energy_bins: int = 64
    jc_percentile: float = 95.0


@dataclass(slots=True)
class ClassifierConfig:
    disk_epsilon_midpoint: float = 0.25
    disk_epsilon_scale: float = 5.0
    z_scale_reff: float = 0.75
    bulge_radius_scale_reff: float = 0.60
    prior_strength: float = 0.55


@dataclass(slots=True)
class BarDetectorConfig:
    epsilon_min: float = 0.15
    epsilon_max: float = 0.75
    z_max_reff: float = 0.25
    a2_threshold: float = 0.15
    phi_tolerance_rad: float = 0.7853981633974483
    catalog_strength_min: float = 0.05
    max_transfer_fraction: float = 0.65
    radial_inner_fraction: float = 0.15


@dataclass(slots=True)
class ArmDetectorConfig:
    min_disk_prob: float = 0.30
    grid_size: int = 128
    residual_threshold: float = 0.35
    min_island_area: int = 12
    smooth_sigma_pixels: float = 1.25
    radial_profile_percentile: float = 55.0
    min_radius_reff: float = 0.50
    max_radius_reff: float = 4.00
    max_transfer_fraction: float = 0.50


@dataclass(slots=True)
class ProjectionConfig:
    output_shape: tuple[int, int] = (69, 69)
    psf_enabled: bool = True
    align_to_pipe3d: bool = True
    alignment_reference_map: str = "stellar_mass_density_log10"


@dataclass(slots=True)
class MaskConfig:
    min_particles_per_spaxel: int = 5
    snr_window_angstrom: tuple[float, float] = (5000.0, 5500.0)
    min_snr: float = 1.0
    min_island_area: int = 10
    closing_iterations: int = 1


@dataclass(slots=True)
class PackerConfig:
    pad_to_shape: tuple[int, int] = (74, 74)
    compression: str = "lzf"
    include_cube: bool = True
    include_pipe3d: bool = True


@dataclass(slots=True)
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    bar_detector: BarDetectorConfig = field(default_factory=BarDetectorConfig)
    arm_detector: ArmDetectorConfig = field(default_factory=ArmDetectorConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    packer: PackerConfig = field(default_factory=PackerConfig)

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "PipelineConfig":
        base_path = project_root() / "configs" / "default.yaml"
        payload: dict[str, Any] = {}
        if base_path.exists():
            payload = yaml.safe_load(base_path.read_text()) or {}
        if path:
            override_path = resolve_project_path(path)
            payload = _deep_update(payload, yaml.safe_load(override_path.read_text()) or {})
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PipelineConfig":
        data = payload.get("data", {})
        extractor = payload.get("extractor", {})
        classifier = payload.get("classifier", {})
        bar = payload.get("bar_detector", {})
        arm = payload.get("arm_detector", {})
        projection = payload.get("projection", {})
        mask = payload.get("mask", {})
        packer = payload.get("packer", {})
        return cls(
            data=DataConfig(
                data_dir=resolve_project_path(data.get("data_dir", "../data")),
                morphology_catalog=resolve_project_path(data.get("morphology_catalog", "../data/morphs_kinematic_bars.hdf5")),
                stellar_circularities=resolve_project_path(data.get("stellar_circularities", "../data/stellar_circs.hdf5")),
                potential_cache_dir=resolve_project_path(data.get("potential_cache_dir", "../data/potential_cache")),
                ssp_template=resolve_project_path(data.get("ssp_template", "../kinematic_moments/templates/MaStar_CB19.slog_1_5.fits.gz")),
                output_dir=resolve_project_path(data.get("output_dir", "outputs")),
                local_galaxy_ids=tuple(data.get("local_galaxy_ids", ["TNG50-87-155298", "TNG50-87-192324"])),
                file_ifu_design=int(data.get("file_ifu_design", 127)),
            ),
            extractor=ExtractorConfig(**extractor),
            classifier=ClassifierConfig(**classifier),
            bar_detector=BarDetectorConfig(**bar),
            arm_detector=ArmDetectorConfig(**arm),
            projection=ProjectionConfig(
                output_shape=tuple(projection.get("output_shape", (69, 69))),
                psf_enabled=bool(projection.get("psf_enabled", True)),
                align_to_pipe3d=bool(projection.get("align_to_pipe3d", True)),
                alignment_reference_map=str(projection.get("alignment_reference_map", "stellar_mass_density_log10")),
            ),
            mask=MaskConfig(
                min_particles_per_spaxel=int(mask.get("min_particles_per_spaxel", 5)),
                snr_window_angstrom=tuple(mask.get("snr_window_angstrom", (5000.0, 5500.0))),
                min_snr=float(mask.get("min_snr", 1.0)),
                min_island_area=int(mask.get("min_island_area", 10)),
                closing_iterations=int(mask.get("closing_iterations", 1)),
            ),
            packer=PackerConfig(
                pad_to_shape=tuple(packer.get("pad_to_shape", (74, 74))),
                compression=str(packer.get("compression", "lzf")),
                include_cube=bool(packer.get("include_cube", True)),
                include_pipe3d=bool(packer.get("include_pipe3d", True)),
            ),
        )
