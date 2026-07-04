"""Baseline ε-threshold del artículo (portado de epsilon_baseline.py v1).

OJO (MIGRATION.md): `circularity_proxy` (v_phi/v_total) NO es la ε canónica
j_z/j_c(E) del Extractor. Se mantiene solo como baseline etiquetado como proxy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EpsilonBaselineConfig:
    disk_threshold: float = 0.70
    circularity_definition: str = "vphi_over_vtotal"
    counterrotating_as_other: bool = False
    counterrotating_threshold: float = -0.70


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


def hard_threshold_classification(
    epsilon: np.ndarray,
    config: EpsilonBaselineConfig | None = None,
) -> np.ndarray:
    """P_class (N, 3) en columnas [bulge, disk, halo] por umbral duro sobre ε.

    Usado como fallback del Classifier (spec 11) cuando el GMM no converge.
    Con la ε canónica del Extractor, no con el proxy.
    """
    config = config or EpsilonBaselineConfig()
    epsilon = np.asarray(epsilon, dtype=np.float64)
    p = np.zeros((epsilon.size, 3), dtype=np.float64)
    disk = epsilon >= float(config.disk_threshold)
    if config.counterrotating_as_other:
        other = epsilon <= float(config.counterrotating_threshold)
    else:
        other = np.zeros_like(disk, dtype=bool)
    bulge = ~(disk | other)
    p[bulge, 0] = 1.0
    p[disk, 1] = 1.0
    p[other, 2] = 1.0
    return p
