from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import LabelConfig
from .models import MorphologyTargets
from .geometry import deposit_to_grid, sample_grid_at_points, weighted_quantile


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def circular_mean_angle(phi: np.ndarray, weights: np.ndarray) -> float:
    coeff = np.sum(weights * np.exp(2j * phi))
    if np.abs(coeff) == 0:
        return 0.0
    return float(0.5 * np.angle(coeff))


def iterative_scale_rows(
    raw_scores: np.ndarray,
    weights: np.ndarray,
    targets: np.ndarray,
    iterations: int,
) -> np.ndarray:
    eps = 1e-8
    raw = np.clip(raw_scores, eps, None)
    scales = np.ones(raw.shape[1], dtype=np.float64)
    target_sum = np.sum(targets)
    if target_sum <= 0:
        targets = np.array([1.0 / raw.shape[1]] * raw.shape[1], dtype=np.float64)
    else:
        targets = targets / target_sum
    for _ in range(iterations):
        probs = raw * scales[None, :]
        probs /= np.sum(probs, axis=1, keepdims=True)
        recovered = np.sum(probs * weights[:, None], axis=0)
        recovered /= max(np.sum(weights), eps)
        scales *= targets / np.clip(recovered, eps, None)
    probs = raw * scales[None, :]
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs


def derive_family_probabilities(
    radius_kpc: np.ndarray,
    mass_weights: np.ndarray,
    targets: MorphologyTargets,
    config: LabelConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    total_mass = max(float(np.sum(mass_weights)), 1e-8)
    bulge_target = np.clip(targets.bulge_family, 0.0, 0.95)
    other_target = np.clip(targets.other_family, 0.0, 0.95)
    disk_target = max(1e-3, 1.0 - bulge_target - other_target)

    r_bulge = weighted_quantile(radius_kpc, mass_weights, min(max(bulge_target, 0.05), 0.90))
    r_other = weighted_quantile(radius_kpc, mass_weights, max(1.0 - other_target, 0.50))
    bulge_width = max(config.bulge_width_fraction * max(r_bulge, 0.5), 0.25)
    other_width = max(config.other_width_fraction * max(r_other, 0.5), 0.25)

    bulge_raw = np.exp(-0.5 * (radius_kpc / max(r_bulge, 0.25)) ** 2)
    other_raw = sigmoid((radius_kpc - r_other) / other_width)
    disk_raw = np.ones_like(radius_kpc)

    family_probs = iterative_scale_rows(
        np.column_stack((bulge_raw, disk_raw, other_raw)),
        mass_weights,
        np.array([bulge_target, disk_target, other_target], dtype=np.float64),
        iterations=config.family_scaling_iters,
    )
    metadata = {
        "r_bulge_kpc": float(r_bulge),
        "r_other_kpc": float(r_other),
        "bulge_target": float(bulge_target),
        "disk_target": float(disk_target),
        "other_target": float(other_target),
        "total_mass": total_mass,
    }
    return family_probs, metadata


def _radial_profile(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    yy, xx = np.indices((h, w))
    rr = np.sqrt((xx - w / 2.0) ** 2 + (yy - h / 2.0) ** 2)
    bins = np.floor(rr).astype(int)
    max_bin = int(bins.max()) + 1
    profile = np.zeros(max_bin, dtype=np.float64)
    counts = np.zeros(max_bin, dtype=np.float64)
    for idx in range(max_bin):
        mask = bins == idx
        counts[idx] = np.count_nonzero(mask)
        if counts[idx] > 0:
            profile[idx] = np.nanmean(grid[mask])
    return profile


def derive_substructure_probabilities(
    faceon_x: np.ndarray,
    faceon_y: np.ndarray,
    radius_kpc: np.ndarray,
    phi: np.ndarray,
    mass_weights: np.ndarray,
    family_probs: np.ndarray,
    targets: MorphologyTargets,
    config: LabelConfig,
    fine_extent_kpc: float,
    gas_boost: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
    disk_family_prob = family_probs[:, 1]
    disk_mass_weights = mass_weights * disk_family_prob

    bar_radius = max(targets.bar_size_kpc, targets.bar_size_alt_kpc, 0.0)
    if not targets.barred:
        bar_radius = 0.0
    if bar_radius <= 0:
        bar_radius = weighted_quantile(radius_kpc, disk_mass_weights + 1e-8, 0.30)

    inner = radius_kpc <= max(bar_radius * config.bar_radius_fraction, 0.5)
    phi_bar = circular_mean_angle(phi[inner], disk_mass_weights[inner] + 1e-8) if np.any(inner) else 0.0
    strength = max(targets.bar_strength, targets.bar_strength_alt, 0.0)
    strength_norm = np.clip(strength / 0.4, 0.0, 2.0)

    bar_angular = np.clip(np.cos(2.0 * (phi - phi_bar)), 0.0, 1.0)
    bar_soft_edge = max(config.bar_soft_edge_fraction * max(bar_radius, 0.5), 0.2)
    bar_radial = sigmoid((bar_radius - radius_kpc) / bar_soft_edge)
    bar_raw = bar_angular * bar_radial * strength_norm
    if not targets.barred or strength < config.bar_min_strength:
        bar_raw[:] = 0.0

    pixel_scale = (2.0 * fine_extent_kpc) / config.fine_grid_size
    disk_map = deposit_to_grid(
        faceon_x,
        faceon_y,
        disk_mass_weights,
        shape=(config.fine_grid_size, config.fine_grid_size),
        pixel_scale_kpc=pixel_scale,
        sigma_pixels=config.projection_smoothing_sigma_px,
    )
    smooth_map = gaussian_filter(disk_map, sigma=config.arm_residual_sigma_px, mode="nearest")
    profile = _radial_profile(smooth_map)
    yy, xx = np.indices(disk_map.shape)
    rr = np.sqrt((xx - disk_map.shape[1] / 2.0) ** 2 + (yy - disk_map.shape[0] / 2.0) ** 2).astype(int)
    rr = np.clip(rr, 0, len(profile) - 1)
    axisymmetric = profile[rr]
    residual = disk_map - axisymmetric
    arm_threshold = config.arm_residual_threshold * np.maximum(axisymmetric, np.nanmedian(axisymmetric[axisymmetric > 0]) if np.any(axisymmetric > 0) else 0.0)
    arms_grid = np.where(residual > arm_threshold, residual, 0.0)

    if gas_boost is not None and gas_boost.size == faceon_x.size:
        gas_map = deposit_to_grid(
            faceon_x,
            faceon_y,
            gas_boost,
            shape=(config.fine_grid_size, config.fine_grid_size),
            pixel_scale_kpc=pixel_scale,
            sigma_pixels=config.projection_smoothing_sigma_px,
        )
        gas_map /= max(float(np.nanmax(gas_map)), 1e-8)
        arms_grid *= 1.0 + config.gas_arm_boost * gas_map

    arm_raw = sample_grid_at_points(arms_grid, faceon_x, faceon_y, pixel_scale)
    arm_raw *= (radius_kpc > max(bar_radius, 0.5))
    arm_raw = np.clip(arm_raw, 0.0, None)

    sub_raw = np.column_stack(
        (
            np.ones_like(radius_kpc),
            bar_raw,
            arm_raw,
        )
    )
    sub_probs = sub_raw / np.clip(np.sum(sub_raw, axis=1, keepdims=True), 1e-8, None)
    metadata = {
        "bar_radius_input_kpc": float(bar_radius),
        "bar_angle_rad": float(phi_bar),
        "bar_strength_input": float(strength),
    }
    qa = {
        "disk_faceon_map": disk_map.astype(np.float32),
        "disk_axisymmetric_map": axisymmetric.astype(np.float32),
        "disk_residual_map": residual.astype(np.float32),
        "arm_candidate_map": arms_grid.astype(np.float32),
    }
    return sub_probs, metadata, qa
