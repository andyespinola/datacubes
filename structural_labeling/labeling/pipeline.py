from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, label

from .config import LabelConfig
from .constants import CLASS_INDEX, CLASS_NAMES, PHYSICAL_CLASS_INDICES, PHYSICAL_CLASS_NAMES
from .decomposition import derive_family_probabilities, derive_substructure_probabilities
from .geometry import (
    center_and_rotate_faceon,
    deposit_to_grid,
    load_cube_geometry,
    project_positions,
    view_vector_from_index,
    weighted_quantile,
)
from .models import CubeGeometry, LabelProducts, ManifestRow, MorphologyTargets, TNGTruth
from .ssp import SSPGrid, particle_light_weights, scale_factor_to_age_gyr


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
PHYSICAL_INDEX_LIST = list(PHYSICAL_CLASS_INDICES)


class LabelingPipeline:
    def __init__(self, config: LabelConfig, ssp_grid: SSPGrid):
        self.config = config
        self.ssp_grid = ssp_grid

    def _convert_truth_units(self, truth: TNGTruth, snapshot: int) -> TNGTruth:
        a = SNAP_A.get(snapshot, 1.0)
        pos = truth.stellar_pos * a / HUBBLE_PARAM
        vel = truth.stellar_vel * np.sqrt(a)
        mass = truth.stellar_mass * 1e10 / HUBBLE_PARAM
        ages = scale_factor_to_age_gyr(truth.stellar_age_gyr)
        gas_pos = truth.gas_pos * a / HUBBLE_PARAM if truth.gas_pos is not None else None
        gas_vel = truth.gas_vel * np.sqrt(a) if truth.gas_vel is not None else None
        gas_mass = truth.gas_mass * 1e10 / HUBBLE_PARAM if truth.gas_mass is not None else None
        center = truth.subhalo_pos * a / HUBBLE_PARAM
        systemic_velocity = truth.subhalo_vel * np.sqrt(a)
        return TNGTruth(
            stellar_pos=pos,
            stellar_vel=vel,
            stellar_mass=mass,
            stellar_age_gyr=ages,
            stellar_metallicity=truth.stellar_metallicity,
            gas_pos=gas_pos,
            gas_vel=gas_vel,
            gas_mass=gas_mass,
            gas_sfr=truth.gas_sfr,
            gas_metallicity=truth.gas_metallicity,
            gas_density=truth.gas_density,
            subhalo_pos=center,
            subhalo_vel=systemic_velocity,
            stellar_halfmass_rad=truth.stellar_halfmass_rad * a / HUBBLE_PARAM if truth.stellar_halfmass_rad else 0.0,
        )

    def _build_gas_arm_boost(
        self,
        faceon_rotation: np.ndarray,
        truth: TNGTruth,
        extent_kpc: float,
        n_stars: int,
        stellar_faceon_x: np.ndarray,
        stellar_faceon_y: np.ndarray,
    ) -> np.ndarray | None:
        if truth.gas_pos is None or truth.gas_sfr is None or truth.gas_pos.size == 0:
            return None
        gas_faceon = (faceon_rotation @ (truth.gas_pos - truth.subhalo_pos[None, :]).T).T
        gas_x = gas_faceon[:, 0]
        gas_y = gas_faceon[:, 1]
        pixel_scale = (2.0 * extent_kpc) / self.config.fine_grid_size
        gas_map = deposit_to_grid(
            gas_x,
            gas_y,
            np.clip(truth.gas_sfr, 0.0, None),
            shape=(self.config.fine_grid_size, self.config.fine_grid_size),
            pixel_scale_kpc=pixel_scale,
            sigma_pixels=self.config.projection_smoothing_sigma_px,
        )
        max_val = float(np.nanmax(gas_map))
        if max_val <= 0:
            return np.zeros(n_stars, dtype=np.float64)
        from .geometry import sample_grid_at_points

        return sample_grid_at_points(gas_map / max_val, stellar_faceon_x, stellar_faceon_y, pixel_scale)

    def run(
        self,
        row: ManifestRow,
        truth: TNGTruth,
        targets: MorphologyTargets,
    ) -> LabelProducts:
        if not row.cube_path:
            raise ValueError("Este pipeline necesita cube_path en el manifiesto para remuestrear a la grilla MaNGIA final.")
        cube_geometry = load_cube_geometry(row.cube_path, self.config)
        truth = self._convert_truth_units(truth, row.snapshot)

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

        light_weights, particle_ml = particle_light_weights(
            self.ssp_grid,
            truth.stellar_mass,
            truth.stellar_age_gyr,
            truth.stellar_metallicity,
        )

        family_probs, family_meta = derive_family_probabilities(radius_kpc, truth.stellar_mass, targets, self.config)

        cube_half_extent_kpc = 0.5 * max(cube_geometry.shape) * cube_geometry.pixel_scale_arcsec * cube_geometry.kpc_per_arcsec
        particle_extent_kpc = float(np.nanmax(np.abs(np.concatenate((faceon_x, faceon_y))))) if faceon_x.size else cube_half_extent_kpc
        fine_extent_kpc = max(cube_half_extent_kpc, particle_extent_kpc) * self.config.faceon_padding_factor

        gas_arm_boost = self._build_gas_arm_boost(
            faceon_rotation,
            truth,
            fine_extent_kpc,
            len(faceon_x),
            faceon_x,
            faceon_y,
        )
        sub_probs, sub_meta, qa_faceon_aux = derive_substructure_probabilities(
            faceon_x,
            faceon_y,
            radius_kpc,
            phi,
            truth.stellar_mass,
            family_probs,
            targets,
            self.config,
            fine_extent_kpc,
            gas_boost=gas_arm_boost,
        )

        particle_probs = np.zeros((truth.stellar_mass.size, len(CLASS_NAMES)), dtype=np.float64)
        particle_probs[:, CLASS_INDEX["bulbo"]] = family_probs[:, 0]
        particle_probs[:, CLASS_INDEX["other"]] = family_probs[:, 2]
        particle_probs[:, CLASS_INDEX["barra"]] = family_probs[:, 1] * sub_probs[:, 1]
        particle_probs[:, CLASS_INDEX["brazos"]] = family_probs[:, 1] * sub_probs[:, 2]
        particle_probs[:, CLASS_INDEX["disco"]] = family_probs[:, 1] * sub_probs[:, 0]
        particle_probs[:, CLASS_INDEX["incierto"]] = 0.0
        particle_probs /= np.clip(np.sum(particle_probs[:, PHYSICAL_INDEX_LIST], axis=1, keepdims=True), 1e-8, None)

        view_vector = view_vector_from_index(row.view, row.repeat_count)
        obs_x, obs_y, _ = project_positions(truth.stellar_pos - truth.subhalo_pos[None, :], view_vector)
        pixel_scale_kpc = cube_geometry.pixel_scale_arcsec * cube_geometry.kpc_per_arcsec
        sigma_pixels = cube_geometry.psf_fwhm_arcsec / 2.355 / max(cube_geometry.pixel_scale_arcsec, 1e-6)

        mass_contrib = np.zeros((len(CLASS_NAMES), *cube_geometry.shape), dtype=np.float32)
        light_contrib = np.zeros((len(CLASS_NAMES), *cube_geometry.shape), dtype=np.float32)

        faceon_components_mass = np.zeros((len(CLASS_NAMES), self.config.fine_grid_size, self.config.fine_grid_size), dtype=np.float32)
        faceon_components_light = np.zeros_like(faceon_components_mass)
        faceon_pixel_scale = (2.0 * fine_extent_kpc) / self.config.fine_grid_size

        for class_name in PHYSICAL_CLASS_NAMES:
            idx = CLASS_INDEX[class_name]
            class_mass_weights = truth.stellar_mass * particle_probs[:, idx]
            class_light_weights = light_weights * particle_probs[:, idx]

            observed_mass_map = deposit_to_grid(
                obs_x,
                obs_y,
                class_mass_weights,
                shape=cube_geometry.shape,
                pixel_scale_kpc=pixel_scale_kpc,
                sigma_pixels=sigma_pixels,
            )
            observed_light_map = deposit_to_grid(
                obs_x,
                obs_y,
                class_light_weights,
                shape=cube_geometry.shape,
                pixel_scale_kpc=pixel_scale_kpc,
                sigma_pixels=sigma_pixels,
            )

            if observed_mass_map.sum() > 0 and class_mass_weights.sum() > 0:
                observed_mass_map *= float(class_mass_weights.sum() / observed_mass_map.sum())
            if observed_light_map.sum() > 0 and class_light_weights.sum() > 0:
                observed_light_map *= float(class_light_weights.sum() / observed_light_map.sum())

            mass_contrib[idx] = observed_mass_map
            light_contrib[idx] = observed_light_map

            faceon_mass_map = deposit_to_grid(
                faceon_x,
                faceon_y,
                class_mass_weights,
                shape=(self.config.fine_grid_size, self.config.fine_grid_size),
                pixel_scale_kpc=faceon_pixel_scale,
                sigma_pixels=self.config.projection_smoothing_sigma_px,
            )
            faceon_light_map = deposit_to_grid(
                faceon_x,
                faceon_y,
                class_light_weights,
                shape=(self.config.fine_grid_size, self.config.fine_grid_size),
                pixel_scale_kpc=faceon_pixel_scale,
                sigma_pixels=self.config.projection_smoothing_sigma_px,
            )
            faceon_components_mass[idx] = faceon_mass_map
            faceon_components_light[idx] = faceon_light_map

        recalibrate_central_bulge_disk(
            mass_contrib,
            light_contrib,
            cube_geometry,
            family_meta,
            self.config,
        )
        rescale_families_to_targets(mass_contrib, light_contrib, cube_geometry.valid_mask, targets)
        recovered = recover_global_fractions(mass_contrib, cube_geometry.valid_mask)

        soft_mass = mass_contrib.copy()
        soft_light = light_contrib.copy()
        valid_mask = cube_geometry.valid_mask
        for tensor in (soft_mass, soft_light):
            tensor[CLASS_INDEX["incierto"], :, :] = 0.0
            denom = np.sum(tensor[PHYSICAL_INDEX_LIST], axis=0)
            positive = valid_mask & (denom > 0)
            for idx in PHYSICAL_INDEX_LIST:
                tensor[idx, positive] /= denom[positive]
            tensor[CLASS_INDEX["no_valido"], ~positive] = 1.0
            for idx in PHYSICAL_INDEX_LIST:
                tensor[idx, ~positive] = 0.0
            tensor[CLASS_INDEX["incierto"], :, :] = 0.0

        threshold_values = self.config.resolved_hard_label_thresholds()
        hard_mass_variants: dict[str, np.ndarray] = {}
        hard_light_variants: dict[str, np.ndarray] = {}
        hard_variant_summary: dict[str, dict[str, dict[str, int]]] = {}
        confidence_mass = None
        confidence_light = None
        for threshold in threshold_values:
            threshold_key = format_threshold_key(threshold)
            hard_mass_variant, confidence_mass_variant = harden_labels(
                soft_mass,
                valid_mask,
                self.config,
                min_prob=threshold,
            )
            hard_light_variant, confidence_light_variant = harden_labels(
                soft_light,
                valid_mask,
                self.config,
                min_prob=threshold,
            )
            hard_mass_variants[threshold_key] = hard_mass_variant
            hard_light_variants[threshold_key] = hard_light_variant
            if confidence_mass is None:
                confidence_mass = confidence_mass_variant
            if confidence_light is None:
                confidence_light = confidence_light_variant
            hard_variant_summary[threshold_key] = {
                "mass": summarize_hard_labels(hard_mass_variant),
                "light": summarize_hard_labels(hard_light_variant),
            }

        default_threshold_key = format_threshold_key(self.config.hard_label_min_prob)
        hard_mass = hard_mass_variants[default_threshold_key]
        hard_light = hard_light_variants[default_threshold_key]
        targets_summary = {
            "bulge_family": float(targets.bulge_family),
            "disk_family": float(targets.disk_family),
            "other_family": float(targets.other_family),
            "barred": bool(targets.barred),
            "bar_size_kpc": float(max(targets.bar_size_kpc, targets.bar_size_alt_kpc)),
            "bar_strength": float(max(targets.bar_strength, targets.bar_strength_alt)),
        }

        bar_mass_weights = truth.stellar_mass * particle_probs[:, CLASS_INDEX["barra"]]
        bar_recovered_radius = weighted_quantile(radius_kpc, np.clip(bar_mass_weights, 0.0, None) + 1e-8, 0.90) if np.any(bar_mass_weights > 0) else 0.0
        bar_metadata = {
            **sub_meta,
            "barred_target": bool(targets.barred),
            "bar_radius_recovered_kpc": float(bar_recovered_radius),
            "bar_fraction_recovered": float(recovered["barra"]),
        }

        qa_maps = {
            "face_on_mass_total": np.sum(faceon_components_mass[1:], axis=0).astype(np.float32),
            "face_on_light_total": np.sum(faceon_components_light[1:], axis=0).astype(np.float32),
            "face_on_mass_components": faceon_components_mass.astype(np.float32),
            "face_on_light_components": faceon_components_light.astype(np.float32),
            "observed_mass_components": soft_mass.astype(np.float32),
            "observed_light_components": soft_light.astype(np.float32),
            "observed_mass_contributions": mass_contrib.astype(np.float32),
            "observed_light_contributions": light_contrib.astype(np.float32),
            "valid_mask": valid_mask.astype(np.uint8),
            "base_valid_mask": cube_geometry.base_valid_mask.astype(np.uint8),
            "valid_signal_map": cube_geometry.signal_map.astype(np.float32),
            **qa_faceon_aux,
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
            qa_maps=qa_maps,
            bar_metadata=bar_metadata,
            global_fraction_targets=targets_summary,
            global_fraction_recovered=recovered,
            hard_variant_summary=hard_variant_summary,
        )


def central_weight_map(cube_geometry: CubeGeometry, scale_kpc: float) -> np.ndarray:
    ny, nx = cube_geometry.shape
    yy, xx = np.indices((ny, nx))
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0
    pixel_scale_kpc = cube_geometry.pixel_scale_arcsec * cube_geometry.kpc_per_arcsec
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) * pixel_scale_kpc
    scale = max(scale_kpc, pixel_scale_kpc, 1e-3)
    return np.exp(-0.5 * (radius / scale) ** 2).astype(np.float32)


def recalibrate_central_bulge_disk(
    mass_contrib: np.ndarray,
    light_contrib: np.ndarray,
    cube_geometry: CubeGeometry,
    family_meta: dict[str, float],
    config: LabelConfig,
) -> None:
    bulge_scale = max(
        float(family_meta.get("r_bulge_kpc", 0.0)) * config.central_radius_scale,
        cube_geometry.psf_fwhm_arcsec * cube_geometry.kpc_per_arcsec * 0.5,
    )
    weight = central_weight_map(cube_geometry, bulge_scale)
    bulge_factor = 1.0 + config.central_bulge_boost * weight
    disk_factor = np.clip(1.0 - config.central_disk_suppression * weight, 0.25, 1.0)

    for tensor in (mass_contrib, light_contrib):
        tensor[CLASS_INDEX["bulbo"]] *= bulge_factor
        for idx in (
            CLASS_INDEX["disco"],
            CLASS_INDEX["barra"],
            CLASS_INDEX["brazos"],
        ):
            tensor[idx] *= disk_factor


def smooth_probabilities_spatially(
    probs: np.ndarray,
    valid_mask: np.ndarray,
    config: LabelConfig,
) -> np.ndarray:
    sigma = max(float(config.hard_spatial_sigma_px), 0.0)
    blend = np.clip(float(config.hard_spatial_blend), 0.0, 1.0)
    if sigma <= 0 or blend <= 0:
        return probs

    valid = valid_mask.astype(np.float32)
    norm = gaussian_filter(valid, sigma=sigma, mode="nearest")
    smoothed = np.zeros_like(probs, dtype=np.float32)
    for idx in range(probs.shape[0]):
        filtered = gaussian_filter(probs[idx] * valid, sigma=sigma, mode="nearest")
        smoothed[idx] = np.divide(
            filtered,
            np.clip(norm, 1e-6, None),
            out=np.zeros_like(filtered, dtype=np.float32),
            where=norm > 0,
        )

    blended = (1.0 - blend) * probs + blend * smoothed
    denom = np.sum(blended, axis=0)
    positive = valid_mask & (denom > 0)
    for idx in range(blended.shape[0]):
        blended[idx, positive] /= denom[positive]
    blended[:, ~positive] = 0.0
    return blended.astype(np.float32)


def remove_small_hard_components(
    hard: np.ndarray,
    valid_mask: np.ndarray,
    min_pixels: int,
) -> np.ndarray:
    if min_pixels <= 1:
        return hard
    cleaned = hard.copy()
    for idx in PHYSICAL_INDEX_LIST:
        labeled, n_labels = label(cleaned == idx)
        if n_labels == 0:
            continue
        counts = np.bincount(labeled.ravel())
        for label_id in range(1, n_labels + 1):
            if counts[label_id] < min_pixels:
                cleaned[labeled == label_id] = CLASS_INDEX["incierto"]
    cleaned[~valid_mask] = CLASS_INDEX["no_valido"]
    return cleaned


def harden_labels(
    soft_labels: np.ndarray,
    valid_mask: np.ndarray,
    config: LabelConfig,
    min_prob: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    classes = smooth_probabilities_spatially(soft_labels[PHYSICAL_INDEX_LIST, :, :], valid_mask, config)
    top_idx = np.argmax(classes, axis=0)
    sorted_probs = np.sort(classes, axis=0)
    p_max = sorted_probs[-1]
    p_second = sorted_probs[-2]
    confidence = p_max - p_second
    hard = np.full(valid_mask.shape, CLASS_INDEX["no_valido"], dtype=np.int16)
    threshold = float(config.hard_label_min_prob if min_prob is None else min_prob)
    confident = valid_mask & (p_max >= threshold) & (confidence >= config.hard_label_margin)
    hard[valid_mask] = CLASS_INDEX["incierto"]
    mapped_indices = np.asarray(PHYSICAL_INDEX_LIST, dtype=np.int16)
    hard[confident] = mapped_indices[top_idx[confident]]
    hard = remove_small_hard_components(hard, valid_mask, int(config.hard_min_component_pixels))
    return hard, confidence.astype(np.float32)


def format_threshold_key(threshold: float) -> str:
    return f"{int(round(threshold * 100)):03d}"


def summarize_hard_labels(hard: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(np.asarray(hard, dtype=np.int16), return_counts=True)
    return {
        CLASS_NAMES[int(value)]: int(count)
        for value, count in zip(values.tolist(), counts.tolist(), strict=True)
    }


def recover_global_fractions(soft_mass: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    valid = valid_mask
    fractions = {}
    total = np.sum(soft_mass[PHYSICAL_INDEX_LIST, :, :][:, valid])
    if total <= 0:
        total = 1.0
    for name in PHYSICAL_CLASS_NAMES:
        idx = CLASS_INDEX[name]
        fractions[name] = float(np.sum(soft_mass[idx, valid]) / total)
    fractions["disk_family_total"] = float(
        fractions["disco"] + fractions["barra"] + fractions["brazos"]
    )
    return fractions


def rescale_families_to_targets(
    mass_contrib: np.ndarray,
    light_contrib: np.ndarray,
    valid_mask: np.ndarray,
    targets: MorphologyTargets,
) -> None:
    family_defs = {
        "bulge": ([CLASS_INDEX["bulbo"]], float(targets.bulge_family)),
        "disk": ([CLASS_INDEX["disco"], CLASS_INDEX["barra"], CLASS_INDEX["brazos"]], float(targets.disk_family)),
        "other": ([CLASS_INDEX["other"]], float(targets.other_family)),
    }
    total_valid = float(np.sum(mass_contrib[1:, valid_mask]))
    if total_valid <= 0:
        return
    for _, (indices, target_fraction) in family_defs.items():
        current = float(np.sum(mass_contrib[indices, :, :][:, valid_mask]))
        if current <= 0:
            continue
        desired = max(target_fraction, 1e-6) * total_valid
        scale = desired / current
        for idx in indices:
            mass_contrib[idx] *= scale
            light_contrib[idx] *= scale


def save_products(base_path: str | Path, products: LabelProducts) -> None:
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    labels_payload = {
        "soft_mass": products.soft_mass,
        "soft_light": products.soft_light,
        "hard_mass": products.hard_mass,
        "hard_light": products.hard_light,
        "confidence_mass": products.confidence_mass,
        "confidence_light": products.confidence_light,
        "valid_mask": products.valid_mask,
        "class_names": np.array(CLASS_NAMES),
        "hard_threshold_keys": np.array(sorted(products.hard_mass_variants.keys())),
    }
    for key, value in products.hard_mass_variants.items():
        labels_payload[f"hard_mass_{key}"] = value
    for key, value in products.hard_light_variants.items():
        labels_payload[f"hard_light_{key}"] = value

    np.savez_compressed(base.with_suffix(".labels.npz"), **labels_payload)
    np.savez_compressed(base.with_suffix(".qa.npz"), **products.qa_maps)
    summary = {
        "bar_metadata": products.bar_metadata,
        "global_fraction_targets": products.global_fraction_targets,
        "global_fraction_recovered": products.global_fraction_recovered,
        "hard_variant_summary": products.hard_variant_summary,
    }
    base.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
