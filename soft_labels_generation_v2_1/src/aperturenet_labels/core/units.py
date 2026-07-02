from __future__ import annotations

import numpy as np

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


def snapshot_scale_factor(snapshot: int) -> float:
    return float(SNAP_A.get(int(snapshot), 1.0))


def comoving_ckpc_h_to_physical_kpc(values: np.ndarray, snapshot: int) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) * snapshot_scale_factor(snapshot) / HUBBLE_PARAM


def tng_mass_to_msun(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) * 1.0e10 / HUBBLE_PARAM


def tng_velocity_to_kms(values: np.ndarray, snapshot: int) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) * np.sqrt(snapshot_scale_factor(snapshot))


def formation_scale_to_age_gyr(scale_factors: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(scale_factors, dtype=np.float64), 1.0e-4, 1.0)
    return np.maximum(1.0e-3, 13.8 * (1.0 - clipped))
