from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, label

from aperturenet_labels.config import ArmDetectorConfig
from aperturenet_labels.io.morphology import MorphologyTargets
from .classifier import ParticleLabels
from .extractor import ParticleFeatures


def _remove_small_islands(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, int]:
    labeled, n_labels = label(mask)
    if n_labels == 0:
        return np.zeros_like(mask, dtype=bool), 0
    counts = np.bincount(labeled.ravel())
    keep = np.zeros(n_labels + 1, dtype=bool)
    for label_id in range(1, n_labels + 1):
        keep[label_id] = counts[label_id] >= min_area
    cleaned = keep[labeled]
    return cleaned, int(np.count_nonzero(keep[1:]))


def add_arm_component(
    features: ParticleFeatures,
    labels: ParticleLabels,
    targets: MorphologyTargets,
    config: ArmDetectorConfig,
) -> ParticleLabels:
    p = labels.p_class.astype(np.float64).copy()
    disk_dominated = p[:, 1] > config.min_disk_prob
    diagnostics = dict(labels.diagnostics)
    if np.count_nonzero(disk_dominated) < 100:
        diagnostics.update({"n_arm_crests": 0, "arm_mass_fraction": 0.0, "arm_reason": "too_few_disk_particles"})
        return ParticleLabels(labels.galaxy_id, labels.class_names, p.astype(np.float32), diagnostics)

    x = features.pos_aligned[:, 0]
    y = features.pos_aligned[:, 1]
    extent = max(float(np.nanpercentile(np.sqrt(x**2 + y**2), 98.0)), features.r_eff_kpc * 2.0, 1.0)
    bins = int(config.grid_size)
    edges = np.linspace(-extent, extent, bins + 1)
    weights = features.mass * p[:, 1]
    disk_map, _, _ = np.histogram2d(y, x, bins=(edges, edges), weights=weights)
    if config.smooth_sigma_pixels > 0.0:
        disk_map = gaussian_filter(disk_map, sigma=float(config.smooth_sigma_pixels), mode="constant")
    yy, xx = np.indices((bins, bins))
    grid_x = (xx + 0.5 - bins / 2.0) * (2.0 * extent / bins)
    grid_y = (yy + 0.5 - bins / 2.0) * (2.0 * extent / bins)
    grid_r = np.sqrt(grid_x**2 + grid_y**2)
    radial_bins = np.linspace(0.0, extent, max(12, bins // 2))
    radial_index = np.digitize(grid_r.ravel(), radial_bins) - 1
    axisym = np.zeros_like(disk_map.ravel(), dtype=np.float64)
    flat = disk_map.ravel()
    for idx in range(len(radial_bins) - 1):
        mask = radial_index == idx
        if np.any(mask):
            positive = flat[mask]
            positive = positive[np.isfinite(positive) & (positive > 0.0)]
            axisym[mask] = float(np.nanpercentile(positive, config.radial_profile_percentile)) if positive.size else 0.0
    axisym = axisym.reshape(disk_map.shape)
    residual = (disk_map - axisym) / np.clip(axisym, 1.0e-8, None)
    radial_allowed = (grid_r >= config.min_radius_reff * features.r_eff_kpc) & (grid_r <= config.max_radius_reff * features.r_eff_kpc)
    spiral_mask = (residual > config.residual_threshold) & radial_allowed
    if targets.barred:
        spiral_mask &= grid_r > max(targets.bar_size_kpc, targets.bar_size_alt_kpc, 0.0)
    spiral_mask, n_crests = _remove_small_islands(spiral_mask, config.min_island_area)

    ix = np.floor((x + extent) / (2.0 * extent) * bins).astype(int)
    iy = np.floor((y + extent) / (2.0 * extent) * bins).astype(int)
    valid = (ix >= 0) & (ix < bins) & (iy >= 0) & (iy < bins)
    in_arm = np.zeros(features.mass.size, dtype=bool)
    arm_strength = np.zeros(features.mass.size, dtype=np.float64)
    in_arm[valid] = spiral_mask[iy[valid], ix[valid]]
    sampled_residual = np.zeros(features.mass.size, dtype=np.float64)
    sampled_residual[valid] = residual[iy[valid], ix[valid]]
    arm_strength[valid] = np.clip(
        (sampled_residual[valid] - config.residual_threshold) / max(1.0 - config.residual_threshold, 1.0e-6),
        0.0,
        1.0,
    )
    in_arm &= disk_dominated
    arm_strength[~in_arm] = 0.0
    p_arm = p[:, 1] * np.clip(arm_strength, 0.0, config.max_transfer_fraction)
    p[:, 1] -= p_arm
    p[:, 3] += p_arm
    p /= np.clip(p.sum(axis=1, keepdims=True), 1.0e-8, None)
    arm_mass_fraction = float(np.sum(features.mass * p[:, 3]) / max(float(np.sum(features.mass)), 1.0e-8))
    diagnostics.update(
        {
            "n_arm_crests": int(n_crests),
            "arm_mass_fraction": arm_mass_fraction,
            "n_arm_particles": int(np.count_nonzero(in_arm)),
            "arm_map_extent_kpc": float(extent),
            "arm_residual_threshold": float(config.residual_threshold),
            "arm_smooth_sigma_pixels": float(config.smooth_sigma_pixels),
        }
    )
    return ParticleLabels(labels.galaxy_id, labels.class_names, p.astype(np.float32), diagnostics)
