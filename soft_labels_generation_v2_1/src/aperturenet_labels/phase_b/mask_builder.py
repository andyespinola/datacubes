from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from scipy.ndimage import binary_closing, label

from aperturenet_labels.config import MaskConfig
from aperturenet_labels.io.cube_reader import CubeGeometry, read_cube_flux, read_cube_signal_mask
from .label_projection import ProjectedLabels


@dataclass(slots=True)
class ValidMask:
    galaxy_id: str
    m_valid: np.ndarray
    m_particles: np.ndarray
    m_snr: np.ndarray
    m_connected: np.ndarray
    snr_map: np.ndarray
    diagnostics: dict[str, float | int]


def _largest_component(mask: np.ndarray, min_area: int) -> np.ndarray:
    labeled, n_labels = label(mask)
    if n_labels == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest = int(np.argmax(counts))
    if counts[largest] < min_area:
        return np.zeros_like(mask, dtype=bool)
    return labeled == largest


def build_valid_mask(
    galaxy_id: str,
    projected: ProjectedLabels,
    cube_path: str | Path,
    cube_geometry: CubeGeometry,
    config: MaskConfig,
) -> ValidMask:
    m_particles = projected.particle_count >= float(config.min_particles_per_spaxel)
    cube = read_cube_flux(cube_path)
    low, high = config.snr_window_angstrom
    wave_mask = (cube_geometry.wavelength >= low) & (cube_geometry.wavelength <= high)
    if not np.any(wave_mask):
        wave_mask = np.ones(cube.shape[0], dtype=bool)
    window = cube[wave_mask]
    signal = np.nanmean(np.abs(window), axis=0)
    noise = np.nanstd(window, axis=0)
    snr = np.divide(signal, np.clip(noise, 1.0e-8, None), out=np.zeros_like(signal, dtype=np.float32), where=noise > 0)
    m_snr = snr >= float(config.min_snr)
    footprint = read_cube_signal_mask(cube_path)
    combined = m_particles & m_snr & footprint
    if config.closing_iterations > 0:
        combined = binary_closing(combined, structure=np.ones((3, 3), dtype=bool), iterations=config.closing_iterations)
        combined &= footprint
    m_connected = _largest_component(combined, config.min_island_area)
    diagnostics = {
        "n_valid": int(np.count_nonzero(m_connected)),
        "fraction_valid": float(np.mean(m_connected)),
        "n_particles_valid": int(np.count_nonzero(m_particles)),
        "n_snr_valid": int(np.count_nonzero(m_snr)),
        "n_footprint_valid": int(np.count_nonzero(footprint)),
    }
    return ValidMask(galaxy_id, m_connected.astype(bool), m_particles.astype(bool), m_snr.astype(bool), m_connected.astype(bool), snr.astype(np.float32), diagnostics)


def write_valid_mask(path: str | Path, valid_mask: ValidMask) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        M_valid=valid_mask.m_valid,
        M_criterion_particles=valid_mask.m_particles,
        M_criterion_snr=valid_mask.m_snr,
        M_criterion_connected=valid_mask.m_connected,
        snr_map=valid_mask.snr_map.astype("f4"),
        diagnostics_json=np.asarray(json.dumps(valid_mask.diagnostics, sort_keys=True)),
    )
    return path
