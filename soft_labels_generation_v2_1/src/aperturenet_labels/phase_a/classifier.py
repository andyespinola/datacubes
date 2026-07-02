from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from aperturenet_labels.config import ClassifierConfig
from aperturenet_labels.core.constants import CLASS_NAMES
from aperturenet_labels.io.morphology import MorphologyTargets
from .extractor import ParticleFeatures


@dataclass(slots=True)
class ParticleLabels:
    galaxy_id: str
    class_names: tuple[str, ...]
    p_class: np.ndarray
    diagnostics: dict[str, float | int | str | bool]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def classify_primary_components(
    features: ParticleFeatures,
    targets: MorphologyTargets,
    config: ClassifierConfig,
) -> ParticleLabels:
    r_norm = features.r_cyl / max(features.r_eff_kpc, 1.0e-6)
    z_norm = features.z_abs / max(features.r_eff_kpc, 1.0e-6)
    eps = features.epsilon
    priors = targets.priors()
    alpha = float(np.clip(config.prior_strength, 0.0, 0.95))

    disk_score = _sigmoid((eps - config.disk_epsilon_midpoint) * config.disk_epsilon_scale)
    disk_score *= np.exp(-z_norm / max(config.z_scale_reff, 1.0e-3))
    bulge_score = np.exp(-r_norm / max(config.bulge_radius_scale_reff, 1.0e-3)) * np.exp(-np.abs(eps) / 0.7)
    halo_score = (1.0 - disk_score) * (1.0 - np.exp(-r_norm / 0.9) + 0.35 * np.clip(z_norm, 0.0, 3.0))

    scores = np.column_stack((bulge_score, disk_score, halo_score))
    scores = np.clip(scores, 1.0e-8, None)
    data_probs = scores / scores.sum(axis=1, keepdims=True)
    prior_probs = np.broadcast_to(priors[None, :], data_probs.shape)
    primary = (1.0 - alpha) * data_probs + alpha * prior_probs
    primary /= np.clip(primary.sum(axis=1, keepdims=True), 1.0e-8, None)

    p_class = np.zeros((features.mass.size, len(CLASS_NAMES)), dtype=np.float32)
    p_class[:, 0] = primary[:, 0]
    p_class[:, 1] = primary[:, 1]
    p_class[:, 4] = primary[:, 2]
    p_class /= np.clip(p_class.sum(axis=1, keepdims=True), 1.0e-8, None)

    mass_total = float(np.sum(features.mass))
    fractions = (features.mass[:, None] * p_class).sum(axis=0) / max(mass_total, 1.0e-8)
    diagnostics: dict[str, float | int | str | bool] = {
        "method": "heuristic_proxy_with_morphology_prior",
        "epsilon_definition": str(features.quality.get("epsilon_definition", "j_z_over_j_total")),
        "energy_definition": str(features.quality.get("energy_definition", "negative_kinetic_only")),
        "potential_status": str(features.quality.get("potential_status", "unknown")),
        "prior_strength": alpha,
        "catalog_bulge_fraction": float(priors[0]),
        "catalog_disk_fraction": float(priors[1]),
        "catalog_halo_fraction": float(priors[2]),
        "fraction_bulge": float(fractions[0]),
        "fraction_disk": float(fractions[1]),
        "fraction_bar": float(fractions[2]),
        "fraction_arm": float(fractions[3]),
        "fraction_halo": float(fractions[4]),
    }
    return ParticleLabels(features.galaxy_id, CLASS_NAMES, p_class.astype(np.float32), diagnostics)


def write_particle_labels(path: str | Path, labels: ParticleLabels, source_module: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "1.0-skeleton"
        handle.attrs["source_module"] = source_module
        meta = handle.create_group("metadata")
        meta.attrs["galaxy_id"] = labels.galaxy_id
        meta.attrs["n_particles"] = int(labels.p_class.shape[0])
        handle.create_dataset("P_class", data=labels.p_class.astype("f4"), compression="lzf")
        handle.create_dataset("class_names", data=np.asarray(labels.class_names, dtype=object), dtype=string_dtype)
        diag = handle.create_group("diagnostics")
        for key, value in labels.diagnostics.items():
            diag.attrs[key] = value
    return path
