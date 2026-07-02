from __future__ import annotations

import numpy as np

from aperturenet_labels.config import BarDetectorConfig
from aperturenet_labels.io.morphology import MorphologyTargets
from .classifier import ParticleLabels
from .extractor import ParticleFeatures


def add_bar_component(
    features: ParticleFeatures,
    labels: ParticleLabels,
    targets: MorphologyTargets,
    config: BarDetectorConfig,
) -> ParticleLabels:
    p = labels.p_class.astype(np.float64).copy()
    diagnostics = dict(labels.diagnostics)
    if not targets.barred:
        diagnostics.update({"bar_detected": False, "bar_reason": "catalog_unbarred", "bar_a2": 0.0})
        return ParticleLabels(labels.galaxy_id, labels.class_names, p.astype(np.float32), diagnostics)

    r_bar = max(float(targets.bar_size_kpc), float(targets.bar_size_alt_kpc), 0.0)
    if r_bar <= 0.0:
        diagnostics.update({"bar_detected": False, "bar_reason": "missing_bar_radius", "bar_a2": 0.0})
        return ParticleLabels(labels.galaxy_id, labels.class_names, p.astype(np.float32), diagnostics)

    x = features.pos_aligned[:, 0]
    y = features.pos_aligned[:, 1]
    phi = np.arctan2(y, x)
    r = features.r_cyl
    inner = max(float(config.radial_inner_fraction) * r_bar, 0.0)
    inside = (r >= inner) & (r < r_bar)
    if np.count_nonzero(inside) < 100:
        diagnostics.update({"bar_detected": False, "bar_reason": "too_few_particles", "bar_a2": 0.0})
        return ParticleLabels(labels.galaxy_id, labels.class_names, p.astype(np.float32), diagnostics)

    weights = features.mass[inside] * p[inside, 1]
    c2 = np.sum(weights * np.exp(2.0j * phi[inside])) / max(float(np.sum(weights)), 1.0e-12)
    a2 = float(np.abs(c2))
    phi_bar = float(np.angle(c2) / 2.0)
    catalog_strength = max(float(targets.bar_strength), float(targets.bar_strength_alt), 0.0)
    catalog_prior = bool(targets.barred and catalog_strength >= config.catalog_strength_min)
    if a2 < config.a2_threshold and not catalog_prior:
        diagnostics.update({"bar_detected": False, "bar_reason": "a2_below_threshold", "bar_a2": a2, "bar_phi_rad": phi_bar})
        return ParticleLabels(labels.galaxy_id, labels.class_names, p.astype(np.float32), diagnostics)

    rel_phi = ((phi - phi_bar + np.pi / 2.0) % np.pi) - np.pi / 2.0
    z_limit = config.z_max_reff * max(features.r_eff_kpc, 1.0e-6)
    base_candidate = (
        (features.epsilon >= config.epsilon_min)
        & (features.epsilon <= config.epsilon_max)
        & (features.z_abs <= z_limit)
        & (r < r_bar)
        & (r >= inner)
    )
    angular_weight = np.exp(-0.5 * (rel_phi / max(config.phi_tolerance_rad, 1.0e-3)) ** 2)
    radial_center = 0.55 * r_bar
    radial_width = max(0.35 * r_bar, 1.0e-6)
    radial_weight = np.exp(-0.5 * ((r - radial_center) / radial_width) ** 2)
    transfer_strength = float(np.clip(max(a2, catalog_strength) / max(config.a2_threshold, config.catalog_strength_min, 1.0e-6), 0.0, 1.0))
    transfer = np.clip(angular_weight * radial_weight * transfer_strength, 0.0, config.max_transfer_fraction)
    transfer[~base_candidate] = 0.0
    p_bar = p[:, 1] * transfer
    p[:, 1] -= p_bar
    p[:, 2] += p_bar
    p /= np.clip(p.sum(axis=1, keepdims=True), 1.0e-8, None)
    bar_mass_fraction = float(np.sum(features.mass * p[:, 2]) / max(float(np.sum(features.mass)), 1.0e-8))
    diagnostics.update(
        {
            "bar_detected": True,
            "bar_detection_mode": "a2" if a2 >= config.a2_threshold else "catalog_prior",
            "bar_a2": a2,
            "bar_phi_rad": phi_bar,
            "bar_radius_kpc": r_bar,
            "bar_mass_fraction": bar_mass_fraction,
            "bar_catalog_strength": catalog_strength,
            "n_bar_candidate_particles": int(np.count_nonzero(transfer > 0.0)),
        }
    )
    return ParticleLabels(labels.galaxy_id, labels.class_names, p.astype(np.float32), diagnostics)
