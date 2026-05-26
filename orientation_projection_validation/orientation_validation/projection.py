from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, label

from .config import ProjectionConfig
from .manifest import ProjectionManifestRow
from .paths import ensure_structural_labeling_on_path

ensure_structural_labeling_on_path()

from labeling.config import LabelConfig  # noqa: E402
from labeling.constants import CLASS_INDEX, CLASS_NAMES, PHYSICAL_CLASS_INDICES, PHYSICAL_CLASS_NAMES  # noqa: E402
from labeling.decomposition import derive_family_probabilities, derive_substructure_probabilities  # noqa: E402
from labeling.geometry import center_and_rotate_faceon, deposit_to_grid, sample_grid_at_points, weighted_quantile  # noqa: E402
from labeling.models import MorphologyTargets, TNGTruth  # noqa: E402
from labeling.pipeline import HUBBLE_PARAM, SNAP_A  # noqa: E402
from labeling.ssp import SSPGrid, particle_light_weights, scale_factor_to_age_gyr  # noqa: E402


@dataclass(slots=True)
class ParticleLabelState:
    faceon_pos: np.ndarray
    stellar_mass: np.ndarray
    light_weights: np.ndarray
    particle_probs: np.ndarray
    inside_rcov: np.ndarray
    metadata: dict[str, float | int | bool]


def rotation_matrix_z(angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(float(angle_degrees))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def convert_truth_units(truth: TNGTruth, snapshot: int) -> TNGTruth:
    a = SNAP_A.get(snapshot, 1.0)
    return TNGTruth(
        stellar_pos=truth.stellar_pos * a / HUBBLE_PARAM,
        stellar_vel=truth.stellar_vel * np.sqrt(a),
        stellar_mass=truth.stellar_mass * 1e10 / HUBBLE_PARAM,
        stellar_age_gyr=scale_factor_to_age_gyr(truth.stellar_age_gyr),
        stellar_metallicity=truth.stellar_metallicity,
        gas_pos=truth.gas_pos * a / HUBBLE_PARAM if truth.gas_pos is not None else None,
        gas_vel=truth.gas_vel * np.sqrt(a) if truth.gas_vel is not None else None,
        gas_mass=truth.gas_mass * 1e10 / HUBBLE_PARAM if truth.gas_mass is not None else None,
        gas_sfr=truth.gas_sfr,
        gas_metallicity=truth.gas_metallicity,
        gas_density=truth.gas_density,
        subhalo_pos=truth.subhalo_pos * a / HUBBLE_PARAM,
        subhalo_vel=truth.subhalo_vel * np.sqrt(a),
        stellar_halfmass_rad=truth.stellar_halfmass_rad * a / HUBBLE_PARAM if truth.stellar_halfmass_rad else 0.0,
    )


def build_gas_arm_boost(
    faceon_rotation: np.ndarray,
    truth: TNGTruth,
    config: LabelConfig,
    extent_kpc: float,
    stellar_faceon_x: np.ndarray,
    stellar_faceon_y: np.ndarray,
) -> np.ndarray | None:
    if truth.gas_pos is None or truth.gas_sfr is None or truth.gas_pos.size == 0:
        return None
    gas_faceon = (faceon_rotation @ (truth.gas_pos - truth.subhalo_pos[None, :]).T).T
    pixel_scale = (2.0 * extent_kpc) / config.fine_grid_size
    gas_map = deposit_to_grid(
        gas_faceon[:, 0],
        gas_faceon[:, 1],
        np.clip(truth.gas_sfr, 0.0, None),
        shape=(config.fine_grid_size, config.fine_grid_size),
        pixel_scale_kpc=pixel_scale,
        sigma_pixels=config.projection_smoothing_sigma_px,
    )
    max_val = float(np.nanmax(gas_map)) if gas_map.size else 0.0
    if max_val <= 0:
        return np.zeros(stellar_faceon_x.size, dtype=np.float64)
    return sample_grid_at_points(gas_map / max_val, stellar_faceon_x, stellar_faceon_y, pixel_scale)


def build_particle_label_state(
    row: ProjectionManifestRow,
    truth: TNGTruth,
    targets: MorphologyTargets,
    ssp_grid: SSPGrid,
    label_config: LabelConfig,
    projection_config: ProjectionConfig,
) -> ParticleLabelState:
    truth = convert_truth_units(truth, row.snapshot)
    faceon_pos, faceon_vel, faceon_rotation = center_and_rotate_faceon(
        truth.stellar_pos,
        truth.stellar_vel,
        truth.stellar_mass,
        truth.subhalo_pos,
        truth.subhalo_vel,
    )
    faceon_x = faceon_pos[:, 0]
    faceon_y = faceon_pos[:, 1]
    radius_kpc = np.sqrt(faceon_x**2 + faceon_y**2)
    phi = np.arctan2(faceon_y, faceon_x)
    inside_rcov = radius_kpc <= float(row.rcov_kpc)
    if np.count_nonzero(inside_rcov) < 10:
        raise ValueError(f"{row.galaxy_id} tiene muy pocas partículas dentro de Rcov={row.rcov_kpc:.3f} kpc")

    light_weights, particle_ml = particle_light_weights(
        ssp_grid,
        truth.stellar_mass,
        truth.stellar_age_gyr,
        truth.stellar_metallicity,
    )

    family_probs, family_meta = derive_family_probabilities(radius_kpc, truth.stellar_mass, targets, label_config)
    particle_extent = float(np.nanmax(np.abs(np.concatenate((faceon_x[inside_rcov], faceon_y[inside_rcov])))))
    fine_extent = max(float(row.rcov_kpc), particle_extent, 1e-3) * label_config.faceon_padding_factor
    gas_boost = build_gas_arm_boost(
        faceon_rotation,
        truth,
        label_config,
        fine_extent,
        faceon_x,
        faceon_y,
    )
    sub_probs, sub_meta, _ = derive_substructure_probabilities(
        faceon_x,
        faceon_y,
        radius_kpc,
        phi,
        truth.stellar_mass,
        family_probs,
        targets,
        label_config,
        fine_extent,
        gas_boost=gas_boost,
    )

    particle_probs = np.zeros((truth.stellar_mass.size, len(CLASS_NAMES)), dtype=np.float64)
    particle_probs[:, CLASS_INDEX["bulbo"]] = family_probs[:, 0]
    particle_probs[:, CLASS_INDEX["other"]] = family_probs[:, 2]
    particle_probs[:, CLASS_INDEX["barra"]] = family_probs[:, 1] * sub_probs[:, 1]
    particle_probs[:, CLASS_INDEX["brazos"]] = family_probs[:, 1] * sub_probs[:, 2]
    particle_probs[:, CLASS_INDEX["disco"]] = family_probs[:, 1] * sub_probs[:, 0]
    denom = np.sum(particle_probs[:, PHYSICAL_CLASS_INDICES], axis=1, keepdims=True)
    particle_probs[:, PHYSICAL_CLASS_INDICES] /= np.clip(denom, 1e-8, None)

    bar_mass_weights = truth.stellar_mass * particle_probs[:, CLASS_INDEX["barra"]]
    bar_recovered_radius = (
        weighted_quantile(radius_kpc, np.clip(bar_mass_weights, 0.0, None) + 1e-8, 0.90)
        if np.any(bar_mass_weights > 0)
        else 0.0
    )
    metadata: dict[str, float | int | bool] = {
        **family_meta,
        **sub_meta,
        "n_particles_total": int(truth.stellar_mass.size),
        "n_particles_rcov": int(np.count_nonzero(inside_rcov)),
        "rcov_kpc": float(row.rcov_kpc),
        "barred_target": bool(targets.barred),
        "bar_radius_recovered_kpc": float(bar_recovered_radius),
        "median_mass_to_light": float(np.nanmedian(particle_ml)),
    }
    return ParticleLabelState(
        faceon_pos=faceon_pos,
        stellar_mass=truth.stellar_mass,
        light_weights=light_weights,
        particle_probs=particle_probs,
        inside_rcov=inside_rcov,
        metadata=metadata,
    )


def normalize_contributions(contrib: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    soft = np.zeros_like(contrib, dtype=np.float32)
    denom = np.sum(contrib[list(PHYSICAL_CLASS_INDICES)], axis=0)
    positive = footprint & (denom > 0)
    for idx in PHYSICAL_CLASS_INDICES:
        soft[idx, positive] = contrib[idx, positive] / denom[positive]
    soft[CLASS_INDEX["no_valido"], ~positive] = 1.0
    return soft


def effective_particle_count(x: np.ndarray, y: np.ndarray, weights: np.ndarray, shape: tuple[int, int], pixel_scale_kpc: float) -> np.ndarray:
    sum_w = deposit_to_grid(x, y, weights, shape=shape, pixel_scale_kpc=pixel_scale_kpc, sigma_pixels=0.0)
    sum_w2 = deposit_to_grid(x, y, weights**2, shape=shape, pixel_scale_kpc=pixel_scale_kpc, sigma_pixels=0.0)
    return np.divide(sum_w**2, np.clip(sum_w2, 1e-12, None), out=np.zeros_like(sum_w), where=sum_w2 > 0).astype(np.float32)


def normalized_entropy(soft: np.ndarray) -> np.ndarray:
    probs = np.clip(soft[list(PHYSICAL_CLASS_INDICES)], 1e-8, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=0)
    entropy /= np.log(len(PHYSICAL_CLASS_INDICES))
    physical_sum = np.sum(soft[list(PHYSICAL_CLASS_INDICES)], axis=0)
    entropy[physical_sum <= 0] = 1.0
    return entropy.astype(np.float32)


def connected_valid_mask(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    if min_pixels <= 1:
        return mask.astype(bool)
    labeled, n_labels = label(mask)
    if n_labels == 0:
        return np.zeros_like(mask, dtype=bool)
    keep = np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labeled.ravel())
    for label_id in range(1, n_labels + 1):
        if counts[label_id] >= min_pixels:
            keep |= labeled == label_id
    return keep


def build_mval(
    soft: np.ndarray,
    n_eff: np.ndarray,
    footprint: np.ndarray,
    projection_config: ProjectionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    entropy = normalized_entropy(soft)
    max_prob = np.max(soft[list(PHYSICAL_CLASS_INDICES)], axis=0)
    has_signal = np.sum(soft[list(PHYSICAL_CLASS_INDICES)], axis=0) > 0
    valid = (
        footprint
        & has_signal
        & (n_eff >= projection_config.n_eff_min)
        & (entropy <= projection_config.entropy_max)
        & (max_prob >= projection_config.max_prob_min)
    )
    valid = connected_valid_mask(valid, projection_config.valid_min_component_pixels)
    mval = np.zeros(footprint.shape, dtype=np.uint8)
    mval[footprint & has_signal] = 2
    mval[valid] = 1
    return mval, entropy


def project_orientation(
    state: ParticleLabelState,
    angle_degrees: float,
    row: ProjectionManifestRow,
    projection_config: ProjectionConfig,
) -> dict[str, np.ndarray]:
    size = int(projection_config.grid_size)
    shape = (size, size)
    pixel_scale_kpc = (2.0 * float(row.rcov_kpc)) / size
    sigma_psf_px = projection_config.psf_fwhm_arcsec / 2.355 / max(projection_config.pixel_scale_arcsec, 1e-6)

    rotation = rotation_matrix_z(angle_degrees)
    selected = state.inside_rcov
    coords = (rotation @ state.faceon_pos[selected].T).T
    x = coords[:, 0]
    y = coords[:, 1]

    yy, xx = np.indices(shape)
    center = (size - 1) / 2.0
    footprint = np.sqrt((xx - center) ** 2 + (yy - center) ** 2) <= (size / 2.0 - 0.5)
    n_classes = len(CLASS_NAMES)
    tensors: dict[str, np.ndarray] = {}
    for weight_name, weights in (
        ("mass", state.stellar_mass[selected]),
        ("lum", state.light_weights[selected]),
    ):
        raw_contrib = np.zeros((n_classes, *shape), dtype=np.float32)
        psf_contrib = np.zeros_like(raw_contrib)
        for class_name in PHYSICAL_CLASS_NAMES:
            idx = CLASS_INDEX[class_name]
            class_weights = weights * state.particle_probs[selected, idx]
            raw_contrib[idx] = deposit_to_grid(
                x,
                y,
                class_weights,
                shape=shape,
                pixel_scale_kpc=pixel_scale_kpc,
                sigma_pixels=0.0,
            )
            psf_contrib[idx] = gaussian_filter(raw_contrib[idx], sigma=sigma_psf_px, mode="constant")
        tensors[f"Y_{weight_name}_raw"] = normalize_contributions(raw_contrib, footprint)
        tensors[f"Y_{weight_name}_psf"] = normalize_contributions(psf_contrib, footprint)

    n_eff = effective_particle_count(x, y, state.stellar_mass[selected], shape, pixel_scale_kpc)
    mval, entropy = build_mval(tensors["Y_lum_psf"], n_eff, footprint, projection_config)
    tensors["Mval"] = mval
    tensors["n_eff"] = n_eff
    tensors["entropy"] = entropy
    tensors["valid_mask"] = (mval == 1).astype(np.uint8)
    tensors["orientation_matrix"] = rotation.astype(np.float32)
    return tensors


def build_projection_product(
    row: ProjectionManifestRow,
    truth: TNGTruth,
    targets: MorphologyTargets,
    ssp_grid: SSPGrid,
    label_config: LabelConfig,
    projection_config: ProjectionConfig,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float | int | bool]]:
    state = build_particle_label_state(row, truth, targets, ssp_grid, label_config, projection_config)
    products: dict[str, dict[str, np.ndarray]] = {}
    for angle in projection_config.orientation_degrees:
        key = f"q{int(round(angle)):03d}"
        products[key] = project_orientation(state, angle, row, projection_config)
    return products, state.metadata


def save_projection_product(
    output_path: str | Path,
    row: ProjectionManifestRow,
    products: dict[str, dict[str, np.ndarray]],
    metadata: dict[str, float | int | bool],
    projection_config: ProjectionConfig,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.attrs["galaxy_id"] = row.galaxy_id
        handle.attrs["snapshot"] = row.snapshot
        handle.attrs["subhalo_id"] = row.subhalo_id
        handle.attrs["class_names"] = json.dumps(list(CLASS_NAMES))
        handle.attrs["projection_config"] = json.dumps(projection_config.to_dict(), sort_keys=True)
        meta_group = handle.create_group("metadata")
        for key, value in metadata.items():
            meta_group.attrs[key] = value
        for orientation, payload in products.items():
            group = handle.create_group(orientation)
            for name, array in payload.items():
                compression = "gzip" if array.ndim >= 2 else None
                group.create_dataset(name, data=array, compression=compression)

