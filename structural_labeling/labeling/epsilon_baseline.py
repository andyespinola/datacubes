from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .config import LabelConfig
from .constants import CLASS_INDEX, CLASS_NAMES, PHYSICAL_CLASS_INDICES, PHYSICAL_CLASS_NAMES
from .models import LabelProducts, ManifestRow, TNGTruth

if TYPE_CHECKING:
    from .ssp import SSPGrid


SNAP_A = {
    85: 0.8459,
    86: 0.8564,
    87: 0.8671,
    88: 0.8778,
    89: 0.8885,
    90: 0.8993,
    91: 0.9091,
    92: 0.9212,
    93: 0.9322,
    94: 0.9433,
    95: 0.9545,
    96: 0.9657,
    97: 0.9771,
    98: 0.9885,
    99: 1.0,
}
HUBBLE_PARAM = 0.6774


def scale_factor_to_age_gyr(scale_factors: np.ndarray) -> np.ndarray:
    clipped = np.clip(scale_factors, 1e-4, 1.0)
    return np.maximum(1e-3, 13.8 * (1.0 - clipped))


@dataclass(frozen=True, slots=True)
class EpsilonBaselineConfig:
    disk_threshold: float = 0.70
    circularity_definition: str = "vphi_over_vtotal"
    counterrotating_as_other: bool = False
    counterrotating_threshold: float = -0.70


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


def circularity_proxy(
    faceon_pos: np.ndarray,
    faceon_vel: np.ndarray,
    definition: str = "vphi_over_vtotal",
) -> np.ndarray:
    x = np.asarray(faceon_pos[:, 0], dtype=np.float64)
    y = np.asarray(faceon_pos[:, 1], dtype=np.float64)
    vx = np.asarray(faceon_vel[:, 0], dtype=np.float64)
    vy = np.asarray(faceon_vel[:, 1], dtype=np.float64)
    r_xy = np.hypot(x, y)
    j_z = x * vy - y * vx

    if definition == "vphi_over_vtotal":
        v_phi = np.divide(j_z, r_xy, out=np.zeros_like(j_z), where=r_xy > 0)
        v_total = np.linalg.norm(faceon_vel, axis=1)
        epsilon = np.divide(v_phi, v_total, out=np.zeros_like(v_phi), where=v_total > 0)
    elif definition == "jz_over_jnorm":
        angular_momentum = np.cross(faceon_pos, faceon_vel)
        j_norm = np.linalg.norm(angular_momentum, axis=1)
        epsilon = np.divide(j_z, j_norm, out=np.zeros_like(j_z), where=j_norm > 0)
    else:
        raise ValueError(f"Unsupported circularity definition: {definition}")

    return np.clip(np.where(np.isfinite(epsilon), epsilon, 0.0), -1.0, 1.0)


def particle_probabilities_from_epsilon(
    epsilon: np.ndarray,
    config: EpsilonBaselineConfig,
) -> np.ndarray:
    epsilon = np.asarray(epsilon, dtype=np.float64)
    probabilities = np.zeros((epsilon.size, len(CLASS_NAMES)), dtype=np.float64)
    disk = epsilon >= float(config.disk_threshold)
    if config.counterrotating_as_other:
        other = epsilon <= float(config.counterrotating_threshold)
    else:
        other = np.zeros_like(disk, dtype=bool)
    bulge = ~(disk | other)
    probabilities[disk, CLASS_INDEX["disco"]] = 1.0
    probabilities[bulge, CLASS_INDEX["bulbo"]] = 1.0
    probabilities[other, CLASS_INDEX["other"]] = 1.0
    return probabilities


def normalize_contributions(contrib: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    soft = contrib.copy()
    denom = np.sum(soft[list(PHYSICAL_CLASS_INDICES)], axis=0)
    positive = valid_mask & (denom > 0)
    for idx in PHYSICAL_CLASS_INDICES:
        soft[idx, positive] /= denom[positive]
        soft[idx, ~positive] = 0.0
    soft[CLASS_INDEX["no_valido"], ~positive] = 1.0
    soft[CLASS_INDEX["no_valido"], positive] = 0.0
    soft[CLASS_INDEX["incierto"], :, :] = 0.0
    return soft.astype(np.float32)


def _build_hard_variants(
    soft: np.ndarray,
    valid_mask: np.ndarray,
    label_config: LabelConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, dict[str, dict[str, int]]]:
    from .pipeline import format_threshold_key, harden_labels, summarize_hard_labels

    variants: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, int]] = {}
    first_confidence: np.ndarray | None = None
    for threshold in label_config.resolved_hard_label_thresholds():
        key = format_threshold_key(threshold)
        hard, confidence = harden_labels(soft, valid_mask, label_config, min_prob=threshold)
        variants[key] = hard
        summaries[key] = summarize_hard_labels(hard)
        if first_confidence is None:
            first_confidence = confidence
    default_key = format_threshold_key(label_config.hard_label_min_prob)
    if first_confidence is None:
        first_confidence = np.zeros(valid_mask.shape, dtype=np.float32)
    return variants[default_key], variants, first_confidence, summaries


def build_epsilon_baseline_products(
    row: ManifestRow,
    truth: TNGTruth,
    ssp_grid: "SSPGrid",
    label_config: LabelConfig,
    baseline_config: EpsilonBaselineConfig | None = None,
) -> LabelProducts:
    from .geometry import center_and_rotate_faceon, deposit_to_grid, load_cube_geometry, project_positions, view_vector_from_index
    from .pipeline import recover_global_fractions
    from .ssp import particle_light_weights

    baseline_config = baseline_config or EpsilonBaselineConfig()
    if not row.cube_path:
        raise ValueError("El baseline epsilon necesita cube_path para remuestrear a la grilla MaNGIA.")

    cube_geometry = load_cube_geometry(row.cube_path, label_config)
    truth = convert_truth_units(truth, row.snapshot)
    faceon_pos, faceon_vel, _ = center_and_rotate_faceon(
        truth.stellar_pos,
        truth.stellar_vel,
        truth.stellar_mass,
        truth.subhalo_pos,
        truth.subhalo_vel,
    )
    epsilon = circularity_proxy(faceon_pos, faceon_vel, baseline_config.circularity_definition)
    particle_probs = particle_probabilities_from_epsilon(epsilon, baseline_config)
    light_weights, _ = particle_light_weights(
        ssp_grid,
        truth.stellar_mass,
        truth.stellar_age_gyr,
        truth.stellar_metallicity,
    )

    view_vector = view_vector_from_index(row.view, row.repeat_count)
    obs_x, obs_y, _ = project_positions(truth.stellar_pos - truth.subhalo_pos[None, :], view_vector)
    pixel_scale_kpc = cube_geometry.pixel_scale_arcsec * cube_geometry.kpc_per_arcsec
    sigma_pixels = cube_geometry.psf_fwhm_arcsec / 2.355 / max(cube_geometry.pixel_scale_arcsec, 1e-6)

    mass_contrib = np.zeros((len(CLASS_NAMES), *cube_geometry.shape), dtype=np.float32)
    light_contrib = np.zeros_like(mass_contrib)
    for class_name in PHYSICAL_CLASS_NAMES:
        idx = CLASS_INDEX[class_name]
        class_mass = truth.stellar_mass * particle_probs[:, idx]
        class_light = light_weights * particle_probs[:, idx]
        mass_contrib[idx] = deposit_to_grid(
            obs_x,
            obs_y,
            class_mass,
            shape=cube_geometry.shape,
            pixel_scale_kpc=pixel_scale_kpc,
            sigma_pixels=sigma_pixels,
        )
        light_contrib[idx] = deposit_to_grid(
            obs_x,
            obs_y,
            class_light,
            shape=cube_geometry.shape,
            pixel_scale_kpc=pixel_scale_kpc,
            sigma_pixels=sigma_pixels,
        )

    valid_mask = cube_geometry.valid_mask
    soft_mass = normalize_contributions(mass_contrib, valid_mask)
    soft_light = normalize_contributions(light_contrib, valid_mask)
    hard_mass, hard_mass_variants, confidence_mass, hard_mass_summary = _build_hard_variants(
        soft_mass,
        valid_mask,
        label_config,
    )
    hard_light, hard_light_variants, confidence_light, hard_light_summary = _build_hard_variants(
        soft_light,
        valid_mask,
        label_config,
    )

    faceon_x = faceon_pos[:, 0]
    faceon_y = faceon_pos[:, 1]
    extent = max(float(np.nanmax(np.abs(np.concatenate((faceon_x, faceon_y))))) if faceon_x.size else 1.0, 1e-3)
    faceon_pixel_scale = (2.0 * extent * label_config.faceon_padding_factor) / label_config.fine_grid_size
    epsilon_mass_map = deposit_to_grid(
        faceon_x,
        faceon_y,
        truth.stellar_mass * epsilon,
        shape=(label_config.fine_grid_size, label_config.fine_grid_size),
        pixel_scale_kpc=faceon_pixel_scale,
        sigma_pixels=label_config.projection_smoothing_sigma_px,
    )
    epsilon_weight_map = deposit_to_grid(
        faceon_x,
        faceon_y,
        truth.stellar_mass,
        shape=(label_config.fine_grid_size, label_config.fine_grid_size),
        pixel_scale_kpc=faceon_pixel_scale,
        sigma_pixels=label_config.projection_smoothing_sigma_px,
    )
    epsilon_map = np.divide(
        epsilon_mass_map,
        np.clip(epsilon_weight_map, 1e-8, None),
        out=np.zeros_like(epsilon_mass_map),
        where=epsilon_weight_map > 0,
    )

    hard_variant_summary = {
        key: {
            "mass": hard_mass_summary.get(key, {}),
            "light": hard_light_summary.get(key, {}),
        }
        for key in sorted(set(hard_mass_summary) | set(hard_light_summary))
    }
    return LabelProducts(
        soft_mass=soft_mass,
        soft_light=soft_light,
        hard_mass=hard_mass,
        hard_light=hard_light,
        hard_mass_variants=hard_mass_variants,
        hard_light_variants=hard_light_variants,
        confidence_mass=confidence_mass,
        confidence_light=confidence_light,
        valid_mask=valid_mask,
        qa_maps={
            "epsilon_face_on": epsilon_map.astype(np.float32),
            "observed_mass_components": soft_mass.astype(np.float32),
            "observed_light_components": soft_light.astype(np.float32),
            "valid_mask": valid_mask.astype(np.uint8),
            "base_valid_mask": cube_geometry.base_valid_mask.astype(np.uint8),
            "valid_signal_map": cube_geometry.signal_map.astype(np.float32),
        },
        bar_metadata={
            "barred_target": False,
            "bar_radius_input_kpc": 0.0,
            "bar_radius_recovered_kpc": 0.0,
            "bar_fraction_recovered": 0.0,
        },
        global_fraction_targets={
            "baseline": "epsilon_threshold",
            "disk_threshold": float(baseline_config.disk_threshold),
            "circularity_definition": baseline_config.circularity_definition,
            "counterrotating_as_other": bool(baseline_config.counterrotating_as_other),
            "counterrotating_threshold": float(baseline_config.counterrotating_threshold),
        },
        global_fraction_recovered=recover_global_fractions(soft_mass, valid_mask),
        hard_variant_summary=hard_variant_summary,
    )
