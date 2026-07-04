"""Tests de BarDetector (spec 12) y ArmDetector (spec 13)."""
from __future__ import annotations

import numpy as np

from aperturenet_labels.phase_a.bar_detector import BarDetectorConfig, run_bar_detector
from aperturenet_labels.phase_a.arm_detector import ArmDetectorConfig, run_arm_detector
from aperturenet_labels.phase_a.classifier import load_labels
from aperturenet_labels.schemas.models import BarMeta

from conftest import make_disk, synthetic_features


def _initial_labels_disk(n):
    P = np.zeros((n, 3))
    P[:, 1] = 1.0  # todo disco
    return {"P_class": P, "galaxy_id": "TOY"}


def make_barred_disk(n=30000, r_bar=3.0, bar_frac=0.3, seed=5):
    """Disco con sobredensa barra m=2 a lo largo del eje x."""
    rng = np.random.default_rng(seed)
    n_bar = int(n * bar_frac)
    pos, vel, mass = make_disk(n - n_bar, seed=seed)
    # partículas de barra: elipse alargada en x, ε intermedio, z fino
    xb = rng.normal(0, r_bar * 0.5, n_bar)
    yb = rng.normal(0, r_bar * 0.12, n_bar)
    zb = rng.normal(0, 0.1, n_bar)
    r = np.hypot(xb, yb)
    phi = np.arctan2(yb, xb)
    v = 100.0  # rotación lenta → ε intermedio
    vxb = -v * np.sin(phi) + rng.normal(0, 30, n_bar)
    vyb = v * np.cos(phi) + rng.normal(0, 30, n_bar)
    pos_b = np.column_stack([xb, yb, zb])
    vel_b = np.column_stack([vxb, vyb, rng.normal(0, 10, n_bar)])
    pos_all = np.vstack([pos, pos_b])
    vel_all = np.vstack([vel, vel_b])
    mass_all = np.concatenate([mass, np.full(n_bar, 1e6)])
    return pos_all, vel_all, mass_all


def test_no_bar_in_catalog(tmp_path):
    pos, vel, mass = make_disk(10000)
    feats = synthetic_features(pos, vel, mass)
    out = run_bar_detector(
        feats, _initial_labels_disk(len(mass)), BarMeta(has_bar=False), tmp_path / "b.h5"
    )
    labels = load_labels(out)
    assert labels["P_class"][:, 2].sum() == 0.0


def test_synthetic_bar_detected(tmp_path):
    pos, vel, mass = make_barred_disk()
    feats = synthetic_features(pos, vel, mass)
    bar_meta = BarMeta(has_bar=True, bar_size_kpc=3.0)
    out = run_bar_detector(
        feats, _initial_labels_disk(len(mass)), bar_meta, tmp_path / "b.h5",
        BarDetectorConfig(epsilon_min=0.1, epsilon_max=0.8, z_max_kpc=0.5),
    )
    labels = load_labels(out)
    import h5py

    with h5py.File(out) as f:
        a2 = float(f["bar_diagnostics"].attrs["a2"])
    assert a2 > 0.3
    frac = labels["P_class"][:, 2].sum() / len(mass)
    assert 0.02 < frac < 0.4


def test_bar_probability_conserved(tmp_path):
    pos, vel, mass = make_barred_disk()
    feats = synthetic_features(pos, vel, mass)
    init = _initial_labels_disk(len(mass))
    init["P_class"][:, 0] = 0.2
    init["P_class"][:, 1] = 0.7
    init["P_class"][:, 2] = 0.1
    out = run_bar_detector(
        feats, init, BarMeta(has_bar=True, bar_size_kpc=3.0), tmp_path / "b.h5"
    )
    labels = load_labels(out)
    P = labels["P_class"]
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(P[:, 0], 0.2, atol=1e-6)  # bulge intacto
    np.testing.assert_allclose(P[:, 3], 0.1, atol=1e-6)  # halo intacto


def make_spiral_disk(n=40000, n_arms=2, amp=2.0, seed=7):
    """Disco exponencial con 2 brazos logarítmicos sobredensos."""
    rng = np.random.default_rng(seed)
    r = rng.exponential(4.0, n * 3)
    r = r[(r > 0.5) & (r < 15)][:n]
    n = len(r)
    phi = rng.uniform(0, 2 * np.pi, n)
    # sobredensidad espiral: aceptar partículas cerca de la cresta con prob alta
    pitch = 0.3
    phase = (phi - np.log(r) / pitch) * n_arms
    w = 1.0 + amp * np.exp(-((np.mod(phase, 2 * np.pi) - np.pi) ** 2) / 0.5)
    keep = rng.uniform(0, 1 + amp, n) < w
    r, phi = r[keep], phi[keep]
    n = len(r)
    x, y = r * np.cos(phi), r * np.sin(phi)
    z = rng.normal(0, 0.1, n)
    v = 200.0
    vx, vy = -v * np.sin(phi), v * np.cos(phi)
    pos = np.column_stack([x, y, z])
    vel = np.column_stack([vx, vy, rng.normal(0, 5, n)])
    return pos, vel, np.full(n, 1e6)


def _with_bar_labels_disk(n):
    P = np.zeros((n, 4))
    P[:, 1] = 1.0
    return {"P_class": P, "galaxy_id": "TOY"}


def test_axisymmetric_no_arms(tmp_path):
    pos, vel, mass = make_disk(40000)
    feats = synthetic_features(pos, vel, mass)
    out = run_arm_detector(
        feats, _with_bar_labels_disk(len(mass)), BarMeta(has_bar=False), tmp_path / "a.h5",
        ArmDetectorConfig(map_extent_kpc=12.0),
    )
    labels = load_labels(out)
    arm_frac = labels["P_class"][:, 3].sum() / len(mass)
    assert arm_frac < 0.05


def test_spiral_arms_detected(tmp_path):
    pos, vel, mass = make_spiral_disk()
    feats = synthetic_features(pos, vel, mass)
    out = run_arm_detector(
        feats, _with_bar_labels_disk(len(mass)), BarMeta(has_bar=False), tmp_path / "a.h5",
        ArmDetectorConfig(map_extent_kpc=16.0, fine_grid_size=128, min_island_area=10),
    )
    labels = load_labels(out)
    arm_frac = (labels["P_class"][:, 3] > 0).mean()
    import h5py

    with h5py.File(out) as f:
        n_crests = int(f["arm_diagnostics"].attrs["n_crests"])
    assert n_crests >= 1
    assert 0.03 < arm_frac < 0.6


def test_arm_excludes_bar_region(tmp_path):
    pos, vel, mass = make_spiral_disk()
    feats = synthetic_features(pos, vel, mass)
    bar_meta = BarMeta(has_bar=True, bar_size_kpc=4.0)
    out = run_arm_detector(
        feats, _with_bar_labels_disk(len(mass)), bar_meta, tmp_path / "a.h5",
        ArmDetectorConfig(map_extent_kpc=16.0, fine_grid_size=128, min_island_area=10),
    )
    labels = load_labels(out)
    R = feats["R"]
    in_bar = R < 4.0
    assert labels["P_class"][in_bar, 3].sum() == 0.0


def test_arm_probability_conserved(tmp_path):
    pos, vel, mass = make_spiral_disk()
    feats = synthetic_features(pos, vel, mass)
    out = run_arm_detector(
        feats, _with_bar_labels_disk(len(mass)), BarMeta(has_bar=False), tmp_path / "a.h5"
    )
    labels = load_labels(out)
    np.testing.assert_allclose(labels["P_class"].sum(axis=1), 1.0, atol=1e-6)
