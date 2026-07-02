from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import h5py
import numpy as np

from aperturenet_labels.config import ExtractorConfig
from aperturenet_labels.core.geometry import center_and_rotate_faceon
from aperturenet_labels.core.units import (
    comoving_ckpc_h_to_physical_kpc,
    formation_scale_to_age_gyr,
    snapshot_scale_factor,
    tng_mass_to_msun,
    tng_velocity_to_kms,
)
from aperturenet_labels.io.assets import LocalGalaxyAssets
from aperturenet_labels.io.ssp import SSPGrid, particle_light_weights
from aperturenet_labels.io.tng_reader import load_stellar_particles, load_subhalo_metadata
from aperturenet_labels.io.tng_potential import TNGPotentialError, load_stellar_potential_cache


@dataclass(slots=True)
class ParticleFeatures:
    galaxy_id: str
    snapshot: int
    subhalo_id: int
    selected_indices: np.ndarray
    pos_sim_centered: np.ndarray
    vel_sim_centered: np.ndarray
    pos_aligned: np.ndarray
    vel_aligned: np.ndarray
    mass: np.ndarray
    age_gyr: np.ndarray
    metallicity: np.ndarray
    light_g: np.ndarray
    potential: np.ndarray | None
    epsilon: np.ndarray
    r_cyl: np.ndarray
    z_abs: np.ndarray
    energy_proxy: np.ndarray
    j_c: np.ndarray
    j_z: np.ndarray
    j_total: np.ndarray
    r_eff_kpc: float
    l_total: np.ndarray
    quality: dict[str, float | int | str]


def extract_particle_features(assets: LocalGalaxyAssets, config: ExtractorConfig, ssp_grid: SSPGrid) -> ParticleFeatures:
    start = time.monotonic()
    metadata = load_subhalo_metadata(assets.metadata_path)
    particles = load_stellar_particles(assets.cutout_path, max_particles=config.max_particles, seed=config.random_seed)

    pos = comoving_ckpc_h_to_physical_kpc(particles.coordinates, assets.snapshot)
    vel = tng_velocity_to_kms(particles.velocities, assets.snapshot)
    mass = tng_mass_to_msun(particles.masses)
    center = comoving_ckpc_h_to_physical_kpc(metadata.pos, assets.snapshot)
    systemic = tng_velocity_to_kms(metadata.vel, assets.snapshot)
    centered_pos = pos - center[None, :]
    centered_vel = vel - systemic[None, :]
    age = formation_scale_to_age_gyr(particles.formation_scale)

    r_eff = float(assets.re_kpc or comoving_ckpc_h_to_physical_kpc(np.asarray([metadata.halfmassrad_stars]), assets.snapshot)[0])
    spherical_r = np.linalg.norm(centered_pos, axis=1)
    central_mask = spherical_r <= max(r_eff * config.align_radius_factor, 1.0e-3)
    pos_aligned, vel_aligned, _rotation, l_total = center_and_rotate_faceon(centered_pos, centered_vel, mass, central_mask)

    j_vec = np.cross(pos_aligned, vel_aligned)
    j_z = j_vec[:, 2]
    j_total = np.linalg.norm(j_vec, axis=1)
    speed2 = np.sum(vel_aligned**2, axis=1)
    potential = None
    potential_cache_path = ""
    potential_status = "disabled"
    energy_definition = "negative_kinetic_only"
    epsilon_definition = "j_z_over_j_total"
    if config.use_potential_cache:
        try:
            cache = load_stellar_potential_cache(
                assets.potential_cache_dir,
                assets.galaxy_id,
                particles.selected_indices,
                particles.particle_ids,
            )
            if cache is not None:
                potential = cache.potential_raw * snapshot_scale_factor(assets.snapshot)
                potential_cache_path = str(cache.path)
                potential_status = "loaded"
            else:
                potential_status = "missing"
        except TNGPotentialError:
            if config.require_potential_cache:
                raise
            potential_status = "invalid"
    if potential is not None:
        energy_proxy = 0.5 * speed2 + potential
        j_c = _estimate_j_c_from_energy(energy_proxy, j_total, config.energy_bins, config.jc_percentile)
        epsilon = np.clip(j_z / np.clip(j_c, 1.0e-8, None), -1.5, 1.5)
        energy_definition = "kinetic_plus_tng_potential_scaled_by_a"
        epsilon_definition = "j_z_over_jc_energy_quantile"
    else:
        if config.require_potential_cache:
            raise FileNotFoundError(f"Missing required potential cache for {assets.galaxy_id}: {assets.potential_cache_dir}")
        # Skeleton fallback: negative kinetic energy. The potential-cache path
        # replaces this with 0.5*v^2 + Phi and a sampled j_c(E).
        energy_proxy = -0.5 * speed2
        j_c = np.clip(j_total, 1.0e-8, None)
        epsilon = np.clip(j_z / j_c, -1.0, 1.0)
    light_g, _ml = particle_light_weights(ssp_grid, mass, age, particles.metallicity)
    r_cyl = np.sqrt(pos_aligned[:, 0] ** 2 + pos_aligned[:, 1] ** 2)
    z_abs = np.abs(pos_aligned[:, 2])

    quality: dict[str, float | int | str] = {
        "n_particles_raw": int(particles.n_star_raw),
        "n_particles_used": int(mass.size),
        "n_gas_raw": int(particles.n_gas_raw),
        "max_particles_config": int(config.max_particles),
        "epsilon_mean": float(np.nanmean(epsilon)),
        "epsilon_std": float(np.nanstd(epsilon)),
        "epsilon_p70_fraction": float(np.mean(epsilon > 0.70)),
        "epsilon_definition": epsilon_definition,
        "epsilon_proxy": epsilon_definition,
        "energy_definition": energy_definition,
        "energy_proxy": energy_definition,
        "potential_status": potential_status,
        "potential_cache_path": potential_cache_path,
        "compute_time_sec": float(time.monotonic() - start),
    }
    return ParticleFeatures(
        galaxy_id=assets.galaxy_id,
        snapshot=assets.snapshot,
        subhalo_id=assets.subhalo_id,
        selected_indices=particles.selected_indices,
        pos_sim_centered=centered_pos,
        vel_sim_centered=centered_vel,
        pos_aligned=pos_aligned,
        vel_aligned=vel_aligned,
        mass=mass,
        age_gyr=age,
        metallicity=particles.metallicity,
        light_g=light_g,
        potential=potential,
        epsilon=epsilon,
        r_cyl=r_cyl,
        z_abs=z_abs,
        energy_proxy=energy_proxy,
        j_c=j_c,
        j_z=j_z,
        j_total=j_total,
        r_eff_kpc=r_eff,
        l_total=l_total,
        quality=quality,
    )


def _estimate_j_c_from_energy(energy: np.ndarray, j_total: np.ndarray, n_bins: int, percentile: float) -> np.ndarray:
    energy = np.asarray(energy, dtype=np.float64)
    j_total = np.asarray(j_total, dtype=np.float64)
    valid = np.isfinite(energy) & np.isfinite(j_total) & (j_total > 0.0)
    if np.count_nonzero(valid) < 32:
        return np.clip(j_total, 1.0e-8, None)
    n_bins = max(8, int(n_bins))
    percentile = float(np.clip(percentile, 50.0, 99.9))
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.nanquantile(energy[valid], quantiles))
    if edges.size < 4:
        return np.clip(j_total, 1.0e-8, None)
    centers = []
    jc_values = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = valid & (energy >= left) & (energy <= right)
        if np.count_nonzero(mask) < 16:
            continue
        centers.append(float(np.nanmedian(energy[mask])))
        jc_values.append(float(np.nanpercentile(j_total[mask], percentile)))
    if len(centers) < 2:
        return np.clip(j_total, 1.0e-8, None)
    centers_arr = np.asarray(centers, dtype=np.float64)
    jc_arr = np.maximum.accumulate(np.asarray(jc_values, dtype=np.float64))
    order = np.argsort(centers_arr)
    estimated = np.interp(energy, centers_arr[order], jc_arr[order], left=jc_arr[order][0], right=jc_arr[order][-1])
    return np.clip(estimated, 1.0e-8, None)


def write_particle_features(path: str | Path, features: ParticleFeatures) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "1.0-skeleton"
        handle.attrs["source_module"] = "phase_a.extractor"
        meta = handle.create_group("metadata")
        meta.attrs["galaxy_id"] = features.galaxy_id
        meta.attrs["snapshot"] = int(features.snapshot)
        meta.attrs["subhalo_id"] = int(features.subhalo_id)
        meta.attrs["n_particles"] = int(features.mass.size)
        meta.attrs["r_eff_kpc"] = float(features.r_eff_kpc)
        physical = handle.create_group("physical")
        kinematic = handle.create_group("kinematic")
        quality = handle.create_group("quality")

        datasets = {
            "physical/selected_indices": features.selected_indices,
            "physical/pos_sim_centered": features.pos_sim_centered.astype("f4"),
            "physical/vel_sim_centered": features.vel_sim_centered.astype("f4"),
            "physical/pos_aligned": features.pos_aligned.astype("f4"),
            "physical/vel_aligned": features.vel_aligned.astype("f4"),
            "physical/mass": features.mass.astype("f4"),
            "physical/age_gyr": features.age_gyr.astype("f4"),
            "physical/metallicity": features.metallicity.astype("f4"),
            "physical/light_g": features.light_g.astype("f4"),
            "physical/potential": features.potential.astype("f4") if features.potential is not None else np.zeros(0, dtype=np.float32),
            "kinematic/epsilon": features.epsilon.astype("f4"),
            "kinematic/R": features.r_cyl.astype("f4"),
            "kinematic/z_abs": features.z_abs.astype("f4"),
            "kinematic/E_proxy": features.energy_proxy.astype("f4"),
            "kinematic/j_c": features.j_c.astype("f4"),
            "kinematic/j_z": features.j_z.astype("f4"),
            "kinematic/j_total": features.j_total.astype("f4"),
            "kinematic/L_total": features.l_total.astype("f4"),
        }
        for name, value in datasets.items():
            handle.create_dataset(name, data=value, compression="lzf")
        for key, value in features.quality.items():
            quality.attrs[key] = value
        _ = physical, kinematic
    return path
