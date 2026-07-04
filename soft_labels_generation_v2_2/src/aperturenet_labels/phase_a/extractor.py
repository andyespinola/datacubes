"""Extractor (spec 10): features físicas por partícula estelar. No clasifica.

Produce particle_features.h5 con ε = j_z/j_c(E) canónica, R, z, E, j_z, j_c,
j_total, frame face-on, masa/edad/Z/luz y métricas de calidad.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import structlog
from pydantic import BaseModel
from scipy.interpolate import UnivariateSpline

from ..core.geometry import center_and_rotate_faceon, weighted_quantile
from ..io.ssp_grid import SSPGrid, particle_light_weights
from ..schemas.models import TNGTruth
from .potential import compute_potential_octree, compute_potential_spherical

log = structlog.get_logger(__name__)


class ExtractorConfig(BaseModel):
    align_radius_factor: float = 2.0
    potential_method: Literal["snapshot", "octree", "spherical", "auto"] = "auto"
    n_jc_bins: int = 200
    octree_leafsize: int = 32
    octree_theta: float = 0.6
    softening_kpc: float = 0.288  # softening estelar TNG50-1 a z<1
    seed: int = 42


def _jc_envelope(E: np.ndarray, j_z: np.ndarray, n_bins: int) -> np.ndarray:
    """j_c(E) por envolvente: máximo de |j_z| en bins de E, suavizado y monótono."""
    order = np.argsort(E)
    E_sorted = E[order]
    edges = np.quantile(E_sorted, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    bin_idx = np.clip(np.searchsorted(edges, E, side="right") - 1, 0, len(edges) - 2)
    j_max = np.zeros(len(edges) - 1)
    np.maximum.at(j_max, bin_idx, np.abs(j_z))
    centers = 0.5 * (edges[:-1] + edges[1:])
    valid = j_max > 0
    if valid.sum() < 8:
        return np.interp(E, centers[valid], j_max[valid]) if valid.any() else np.ones_like(E)
    # envolvente creciente con E (más energía → órbita circular más grande)
    j_env = np.maximum.accumulate(j_max)
    try:
        spline = UnivariateSpline(centers[valid], j_env[valid], k=3, s=len(centers) * 0.5)
        jc_at = spline(E)
        # el spline puede ondular; asegurar positividad
        jc_at = np.clip(jc_at, np.percentile(j_env[valid], 1) * 0.1, None)
    except Exception:
        jc_at = np.interp(E, centers[valid], j_env[valid])
    return jc_at


def run_extractor(
    truth: TNGTruth,
    galaxy_id: str,
    output_path: str | Path,
    ssp_grid: SSPGrid,
    config: ExtractorConfig | None = None,
) -> Path:
    """truth DEBE venir ya en unidades físicas (io/units.convert_truth_units)."""
    config = config or ExtractorConfig()
    t0 = time.time()
    n = len(truth.stellar_mass)
    log.info("extractor.start", galaxy_id=galaxy_id, n_particles=n)

    # --- R_eff y máscara central para alinear ---
    centered = truth.stellar_pos - truth.subhalo_pos[None, :]
    r_sph = np.linalg.norm(centered, axis=1)
    r_eff_3d = truth.stellar_halfmass_rad or weighted_quantile(r_sph, truth.stellar_mass, 0.5)
    align_mask = r_sph < config.align_radius_factor * r_eff_3d

    pos_f, vel_f, rot = center_and_rotate_faceon(
        truth.stellar_pos,
        truth.stellar_vel,
        truth.stellar_mass,
        truth.subhalo_pos,
        truth.subhalo_vel,
        align_mask=align_mask,
    )
    l_total = np.sum(
        np.cross(centered[align_mask], (truth.stellar_vel - truth.subhalo_vel)[align_mask])
        * truth.stellar_mass[align_mask, None],
        axis=0,
    )

    R = np.hypot(pos_f[:, 0], pos_f[:, 1])
    z = pos_f[:, 2]
    j_z = pos_f[:, 0] * vel_f[:, 1] - pos_f[:, 1] * vel_f[:, 0]
    j_vec = np.cross(pos_f, vel_f)
    j_total = np.linalg.norm(j_vec, axis=1)

    # --- Potencial ---
    method = config.potential_method
    if method == "auto":
        method = "snapshot" if truth.stellar_potential is not None else "octree"
    if method == "snapshot":
        if truth.stellar_potential is None:
            raise ValueError("potential_method=snapshot pero truth.stellar_potential es None")
        phi = truth.stellar_potential
    else:
        sources_pos = [centered]
        sources_mass = [truth.stellar_mass]
        if truth.gas_pos is not None and truth.gas_mass is not None:
            sources_pos.append(truth.gas_pos - truth.subhalo_pos[None, :])
            sources_mass.append(truth.gas_mass)
        if truth.dm_pos is not None and truth.dm_mass is not None:
            sources_pos.append(truth.dm_pos - truth.subhalo_pos[None, :])
            sources_mass.append(truth.dm_mass)
        src_pos = np.vstack(sources_pos)
        src_mass = np.concatenate(sources_mass)
        log.info("extractor.potential", method=method, n_sources=len(src_mass))
        if method == "octree":
            phi = compute_potential_octree(
                centered,
                src_pos,
                src_mass,
                theta=config.octree_theta,
                softening=config.softening_kpc,
                leafsize=config.octree_leafsize,
            )
        else:
            phi = compute_potential_spherical(centered, src_pos, src_mass)

    v2 = np.sum((truth.stellar_vel - truth.subhalo_vel) ** 2, axis=1)
    E = 0.5 * v2 + phi

    j_c = _jc_envelope(E, j_z, config.n_jc_bins)
    epsilon = np.clip(
        np.divide(j_z, j_c, out=np.zeros_like(j_z), where=j_c > 0), -1.0, 1.0
    )

    # --- edad ya viene en truth.stellar_age_gyr (convert_truth_units) ---
    age = truth.stellar_age_gyr
    light_g, ml = particle_light_weights(ssp_grid, truth.stellar_mass, age, truth.stellar_metallicity)

    # --- R_eff proyectado recomputado (50% de la masa en R cilíndrico face-on) ---
    r_eff_kpc = weighted_quantile(R, truth.stellar_mass, 0.5)

    quality = {
        "n_particles": n,
        "n_central": int(align_mask.sum()),
        "L_total_magnitude": float(np.linalg.norm(l_total)),
        "epsilon_mean": float(epsilon.mean()),
        "epsilon_std": float(epsilon.std()),
        "epsilon_p7_fraction": float((epsilon > 0.7).mean()),
        "epsilon_n3_fraction": float((epsilon < -0.3).mean()),
        "potential_method_used": method,
        "compute_time_sec": float(time.time() - t0),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["schema_version"] = "1.0"
        f.attrs["source_module"] = "extractor"
        meta = f.create_group("metadata")
        meta.attrs["galaxy_id"] = galaxy_id
        meta.attrs["snapshot"] = truth.snapshot
        meta.attrs["subhalo_id"] = truth.subhalo_id
        meta.attrs["n_particles"] = n
        meta.attrs["extracted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        kin = f.create_group("kinematic")
        for name, arr, units_ in [
            ("epsilon", epsilon, "dimensionless"),
            ("R", R, "kpc"),
            ("z", z, "kpc"),
            ("E", E, "(km/s)^2"),
            ("j_z", j_z, "kpc·km/s"),
            ("j_c", j_c, "kpc·km/s"),
            ("j_total", j_total, "kpc·km/s"),
        ]:
            d = kin.create_dataset(name, data=arr.astype(np.float32), compression="lzf")
            d.attrs["units"] = units_
        phys = f.create_group("physical")
        for name, arr, units_ in [
            ("pos_aligned", pos_f, "kpc"),
            ("vel_aligned", vel_f, "km/s"),
            ("pos_centered", centered, "kpc"),  # marco simulación (para Fase B)
            ("vel_centered", truth.stellar_vel - truth.subhalo_vel, "km/s"),
            ("mass", truth.stellar_mass, "M_sun"),
            ("age", age, "Gyr"),
            ("metallicity", truth.stellar_metallicity, "Z"),
            ("light_g", light_g, "L_sun"),
        ]:
            d = phys.create_dataset(name, data=np.asarray(arr, dtype=np.float32), compression="lzf")
            d.attrs["units"] = units_
        qual = f.create_group("quality")
        for k, v in quality.items():
            qual.attrs[k] = v
        meta.attrs["R_eff_kpc"] = r_eff_kpc
        meta.attrs["L_total"] = l_total.astype(np.float64)
        meta.attrs["faceon_rotation"] = rot.astype(np.float64)

    log.info("extractor.done", galaxy_id=galaxy_id, **quality)
    return output_path


def load_particle_features(path: str | Path) -> dict:
    """Lee particle_features.h5 a un dict plano (validación ligera)."""
    out: dict = {}
    with h5py.File(path, "r") as f:
        out["galaxy_id"] = f["metadata"].attrs["galaxy_id"]
        out["n_particles"] = int(f["metadata"].attrs["n_particles"])
        out["R_eff_kpc"] = float(f["metadata"].attrs["R_eff_kpc"])
        out["faceon_rotation"] = np.asarray(f["metadata"].attrs["faceon_rotation"])
        for g in ("kinematic", "physical"):
            for k in f[g]:
                out[k] = f[g][k][:]
        out["quality"] = dict(f["quality"].attrs)
    eps = out["epsilon"]
    if eps.min() < -1.01 or eps.max() > 1.01:
        raise ValueError("epsilon fuera de rango [-1, 1]")
    return out
