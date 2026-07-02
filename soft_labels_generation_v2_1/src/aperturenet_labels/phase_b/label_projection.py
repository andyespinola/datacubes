from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from scipy.ndimage import gaussian_filter

from aperturenet_labels.config import ProjectionConfig
from aperturenet_labels.core.constants import CLASS_NAMES
from aperturenet_labels.core.geometry import deposit_to_grid, project_positions, view_vector_from_index
from aperturenet_labels.io.assets import LocalGalaxyAssets
from aperturenet_labels.io.cube_reader import CubeGeometry, read_pipe3d_maps
from aperturenet_labels.phase_a.classifier import ParticleLabels
from aperturenet_labels.phase_a.extractor import ParticleFeatures
from aperturenet_labels.phase_b.alignment import GridAlignment, apply_grid_alignment, estimate_d4_alignment


@dataclass(slots=True)
class ProjectedLabels:
    galaxy_id: str
    view: int
    y_mass_raw: np.ndarray
    y_mass_psf: np.ndarray
    y_light_raw: np.ndarray
    y_light_psf: np.ndarray
    raw_mass_per_class: np.ndarray
    raw_light_per_class: np.ndarray
    total_mass_per_spaxel: np.ndarray
    total_light_per_spaxel: np.ndarray
    n_eff: np.ndarray
    particle_count: np.ndarray
    metadata: dict[str, float | int | str | bool]


def _normalize(raw: np.ndarray) -> np.ndarray:
    denom = np.sum(raw, axis=2)
    out = np.zeros_like(raw, dtype=np.float32)
    positive = denom > 0.0
    out[positive] = raw[positive] / denom[positive, None]
    return out


def _psf_normalized(raw: np.ndarray, sigma_pixels: float) -> np.ndarray:
    if sigma_pixels <= 0.0:
        return _normalize(raw)
    smoothed = np.zeros_like(raw, dtype=np.float32)
    for idx in range(raw.shape[2]):
        smoothed[:, :, idx] = gaussian_filter(raw[:, :, idx], sigma=sigma_pixels, mode="constant")
    return _normalize(smoothed)


def project_labels(
    assets: LocalGalaxyAssets,
    features: ParticleFeatures,
    labels: ParticleLabels,
    cube_geometry: CubeGeometry,
    config: ProjectionConfig,
) -> ProjectedLabels:
    shape = tuple(config.output_shape or cube_geometry.shape)
    view_vector = view_vector_from_index(assets.view, repeat_count=1)
    x, y, _los = project_positions(features.pos_sim_centered, view_vector)
    pixel_scale_kpc = cube_geometry.pixel_scale_kpc
    sigma_pixels = cube_geometry.psf_fwhm_arcsec / 2.355 / max(cube_geometry.pixel_scale_arcsec, 1.0e-6)
    if not config.psf_enabled:
        sigma_pixels = 0.0

    n_classes = len(CLASS_NAMES)
    raw_mass = np.zeros((*shape, n_classes), dtype=np.float32)
    raw_light = np.zeros_like(raw_mass)
    for class_idx in range(n_classes):
        raw_mass[:, :, class_idx] = deposit_to_grid(
            x,
            y,
            features.mass * labels.p_class[:, class_idx],
            shape,
            pixel_scale_kpc,
            sigma_pixels=0.0,
        )
        raw_light[:, :, class_idx] = deposit_to_grid(
            x,
            y,
            features.light_g * labels.p_class[:, class_idx],
            shape,
            pixel_scale_kpc,
            sigma_pixels=0.0,
        )

    total_mass = np.sum(raw_mass, axis=2)
    total_light = np.sum(raw_light, axis=2)
    sum_w2 = deposit_to_grid(x, y, features.mass**2, shape, pixel_scale_kpc, sigma_pixels=0.0)
    particle_count = deposit_to_grid(x, y, np.ones(features.mass.size, dtype=np.float64), shape, pixel_scale_kpc, sigma_pixels=0.0)

    alignment = GridAlignment(enabled=False)
    if config.align_to_pipe3d:
        reference_maps = read_pipe3d_maps(assets.maps_path)
        reference_map = reference_maps.get(config.alignment_reference_map)
        if reference_map is None:
            raise KeyError(f"Missing Pipe3D map for alignment: {config.alignment_reference_map}")
        alignment = estimate_d4_alignment(total_mass, reference_map, config.alignment_reference_map)
        raw_mass = apply_grid_alignment(raw_mass, alignment)
        raw_light = apply_grid_alignment(raw_light, alignment)
        sum_w2 = apply_grid_alignment(sum_w2, alignment)
        particle_count = apply_grid_alignment(particle_count, alignment)
        total_mass = np.sum(raw_mass, axis=2)
        total_light = np.sum(raw_light, axis=2)

    n_eff = np.divide(total_mass**2, np.clip(sum_w2, 1.0e-12, None), out=np.zeros_like(total_mass), where=sum_w2 > 0)
    y_mass_raw = _normalize(raw_mass)
    y_light_raw = _normalize(raw_light)
    y_mass_psf = _psf_normalized(raw_mass, sigma_pixels)
    y_light_psf = _psf_normalized(raw_light, sigma_pixels)
    metadata: dict[str, float | int | str | bool] = {
        "galaxy_id": assets.galaxy_id,
        "view": int(assets.view),
        "file_ifu_design": int(assets.file_ifu_design),
        "catalog_ifu_design": int(assets.catalog_ifu_design),
        "grid_height": int(shape[0]),
        "grid_width": int(shape[1]),
        "pixel_scale_kpc": float(pixel_scale_kpc),
        "psf_sigma_pixels": float(sigma_pixels),
        "n_particles_projected": int(features.mass.size),
        **alignment.as_metadata(),
    }
    return ProjectedLabels(
        galaxy_id=assets.galaxy_id,
        view=assets.view,
        y_mass_raw=y_mass_raw,
        y_mass_psf=y_mass_psf,
        y_light_raw=y_light_raw,
        y_light_psf=y_light_psf,
        raw_mass_per_class=raw_mass,
        raw_light_per_class=raw_light,
        total_mass_per_spaxel=total_mass.astype(np.float32),
        total_light_per_spaxel=total_light.astype(np.float32),
        n_eff=n_eff.astype(np.float32),
        particle_count=particle_count.astype(np.float32),
        metadata=metadata,
    )


def write_projected_labels(path: str | Path, projected: ProjectedLabels) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        Y_mass_raw=projected.y_mass_raw.astype("f4"),
        Y_mass_psf=projected.y_mass_psf.astype("f4"),
        Y_light_raw=projected.y_light_raw.astype("f4"),
        Y_light_psf=projected.y_light_psf.astype("f4"),
        raw_mass_per_class=projected.raw_mass_per_class.astype("f4"),
        raw_light_per_class=projected.raw_light_per_class.astype("f4"),
        total_mass_per_spaxel=projected.total_mass_per_spaxel.astype("f4"),
        total_light_per_spaxel=projected.total_light_per_spaxel.astype("f4"),
        n_eff=projected.n_eff.astype("f4"),
        particle_count=projected.particle_count.astype("f4"),
        class_names=np.asarray(CLASS_NAMES),
        metadata_json=np.asarray(json.dumps(projected.metadata, sort_keys=True)),
    )
    return path
