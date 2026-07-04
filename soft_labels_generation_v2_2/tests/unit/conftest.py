"""Fixtures sintéticas compartidas (galaxias de juguete)."""
from __future__ import annotations

import numpy as np
import pytest


def make_disk(n: int = 20000, r_max: float = 10.0, v_circ: float = 200.0, seed: int = 1):
    """Disco frío en órbitas circulares planas (ε ≈ 1)."""
    rng = np.random.default_rng(seed)
    r = r_max * np.sqrt(rng.uniform(0.01, 1, n))
    phi = rng.uniform(0, 2 * np.pi, n)
    x, y = r * np.cos(phi), r * np.sin(phi)
    z = rng.normal(0, 0.05, n)
    vx = -v_circ * np.sin(phi)
    vy = v_circ * np.cos(phi)
    vz = rng.normal(0, 5.0, n)
    pos = np.column_stack([x, y, z])
    vel = np.column_stack([vx, vy, vz])
    mass = np.full(n, 1e6)
    return pos, vel, mass


def make_bulge(n: int = 20000, r_scale: float = 1.5, sigma: float = 150.0, seed: int = 2):
    """Esferoide isotrópico sin rotación neta (ε ~ 0)."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(0, r_scale, (n, 3))
    vel = rng.normal(0, sigma, (n, 3))
    mass = np.full(n, 1e6)
    return pos, vel, mass


def make_halo(n: int = 20000, r_scale: float = 30.0, sigma: float = 120.0, seed: int = 3):
    """Esferoide difuso poco ligado (ε ~ 0, R grande)."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(0, r_scale, (n, 3))
    vel = rng.normal(0, sigma, (n, 3))
    mass = np.full(n, 1e6)
    return pos, vel, mass


@pytest.fixture()
def disk_galaxy():
    return make_disk()


@pytest.fixture()
def bulge_galaxy():
    return make_bulge()


def synthetic_features(pos, vel, mass, r_eff=3.0):
    """Dict tipo particle_features para tests de Classifier/detectores."""
    R = np.hypot(pos[:, 0], pos[:, 1])
    j_z = pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]
    j_vec = np.cross(pos, vel)
    j_total = np.linalg.norm(j_vec, axis=1)
    r = np.linalg.norm(pos, axis=1)
    v2 = np.sum(vel**2, axis=1)
    E = 0.5 * v2 - 1e5 / np.clip(r, 0.1, None)  # potencial kepleriano de juguete
    j_c = np.clip(j_total.max() * (E - E.min()) / (E.max() - E.min() + 1e-12), 1e-3, None)
    # ε de juguete: usar j_z normalizado por percentil alto local
    eps = np.clip(j_z / np.percentile(np.abs(j_z), 99), -1, 1)
    return {
        "galaxy_id": "TOY",
        "epsilon": eps.astype(np.float32),
        "R": R.astype(np.float32),
        "z": pos[:, 2].astype(np.float32),
        "E": E.astype(np.float32),
        "j_z": j_z.astype(np.float32),
        "j_c": j_c.astype(np.float32),
        "j_total": j_total.astype(np.float32),
        "pos_aligned": pos.astype(np.float32),
        "vel_aligned": vel.astype(np.float32),
        "pos_centered": pos.astype(np.float32),
        "mass": mass.astype(np.float32),
        "age": np.full(len(mass), 5.0, np.float32),
        "metallicity": np.full(len(mass), 0.02, np.float32),
        "light_g": mass.astype(np.float32),
        "R_eff_kpc": r_eff,
        "quality": {},
    }
