"""MaskBuilder (spec 22): máscara de validez M_valid = A (conteo) ∧ B (S/N) ∧ C (conectividad)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import structlog
from pydantic import BaseModel
from scipy.ndimage import binary_closing, label as ndi_label

from ..core.geometry import largest_connected_component
from ..io.mangia_reader import load_cube_flux

log = structlog.get_logger(__name__)


class MaskBuilderConfig(BaseModel):
    min_particles_per_spaxel: int = 30
    snr_window_angstrom: tuple[float, float] = (5000.0, 5500.0)
    min_snr: float = 3.0
    min_island_area: int = 10
    closing_radius: int = 1


def run_mask_builder(
    n_particles_map: np.ndarray,
    cube_path: str | Path,
    galaxy_id: str,
    view_id: int,
    output_path: str | Path,
    config: MaskBuilderConfig | None = None,
) -> Path:
    config = config or MaskBuilderConfig()
    t0 = time.time()

    # Criterio A — conteo de partículas
    m_a = n_particles_map >= config.min_particles_per_spaxel

    # Criterio B — S/N observacional en ventana espectral
    flux, error, wave = load_cube_flux(cube_path)
    lo, hi = config.snr_window_angstrom
    wsel = (wave >= lo) & (wave <= hi)
    window = flux[wsel]
    signal = np.nanmean(window, axis=0)
    err_win = error[wsel]
    with np.errstate(invalid="ignore", divide="ignore"):
        noise_obs = np.nanmedian(np.abs(err_win), axis=0)
        noise_std = np.nanstd(window, axis=0)
    noise = np.where(noise_obs > 0, noise_obs, noise_std)
    snr = np.divide(signal, noise, out=np.zeros_like(signal), where=noise > 0)
    m_b = snr >= config.min_snr

    if m_a.shape != m_b.shape:
        raise ValueError(f"shapes A {m_a.shape} vs B {m_b.shape} no coinciden")

    # Criterio C — conectividad espacial
    m_ab = m_a & m_b
    m_c = largest_connected_component(m_ab, config.min_island_area)
    if config.closing_radius > 0:
        structure = np.ones((3, 3), dtype=bool)
        m_c = binary_closing(m_c, structure=structure, iterations=int(config.closing_radius))

    m_valid = m_c & (n_particles_map > 0)  # el closing no debe inventar spaxels sin partículas

    _, n_islands = ndi_label(m_ab)
    diagnostics = {
        "n_valid_total": int(m_valid.sum()),
        "n_only_A_invalid": int((~m_a & m_b).sum()),
        "n_only_B_invalid": int((m_a & ~m_b).sum()),
        "n_dropped_by_C": int((m_ab & ~m_valid).sum()),
        "n_islands_AB": int(n_islands),
        "fraction_valid": float(m_valid.mean()),
        "snr_median_valid": float(np.median(snr[m_valid])) if m_valid.any() else 0.0,
        "compute_time_sec": float(time.time() - t0),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        M_valid=m_valid,
        M_criterion_A=m_a,
        M_criterion_B=m_b,
        M_criterion_C=m_c,
        snr_map=snr.astype(np.float32),
        diagnostics=np.array([(k, str(v)) for k, v in diagnostics.items()]),
        galaxy_id=galaxy_id,
        view_id=view_id,
    )
    log.info("mask_builder.done", galaxy_id=galaxy_id, **{k: v for k, v in diagnostics.items() if k != "compute_time_sec"})
    return output_path


def load_mask(path: str | Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}
