"""Tests del Extractor (spec 10): ε en casos sintéticos, conservación."""
from __future__ import annotations

import numpy as np
import pytest

from aperturenet_labels.phase_a.extractor import ExtractorConfig, run_extractor, load_particle_features
from aperturenet_labels.phase_a.potential import compute_potential_spherical
from aperturenet_labels.io.ssp_grid import SSPGrid
from aperturenet_labels.schemas.models import TNGTruth
from scipy.spatial import cKDTree

from conftest import make_bulge, make_disk


def _toy_ssp() -> SSPGrid:
    ages = np.array([0.1, 1.0, 5.0, 13.0])
    mets = np.array([0.004, 0.02, 0.004, 0.02])
    ml = np.ones(4)
    feats = np.column_stack((np.log10(ages), mets))
    return SSPGrid(ages_gyr=ages, metallicities=mets, mass_to_light=ml, tree=cKDTree(feats))


def _make_truth(pos, vel, mass) -> TNGTruth:
    n = len(mass)
    return TNGTruth(
        stellar_pos=pos,
        stellar_vel=vel,
        stellar_mass=mass,
        stellar_formation_scale=np.full(n, 0.5),
        stellar_age_gyr=np.full(n, 5.0),
        stellar_metallicity=np.full(n, 0.02),
        subhalo_pos=np.zeros(3),
        subhalo_vel=np.zeros(3),
        stellar_halfmass_rad=3.0,
        snapshot=87,
        subhalo_id=0,
        scale_factor=1.0,
        redshift=0.0,
    )


def test_disk_high_epsilon(tmp_path):
    pos, vel, mass = make_disk()
    truth = _make_truth(pos, vel, mass)
    out = run_extractor(
        truth, "toy-disk", tmp_path / "f.h5", _toy_ssp(),
        ExtractorConfig(potential_method="spherical"),
    )
    feats = load_particle_features(out)
    assert np.median(feats["epsilon"]) > 0.8


def test_bulge_epsilon_centered_zero(tmp_path):
    pos, vel, mass = make_bulge()
    truth = _make_truth(pos, vel, mass)
    out = run_extractor(
        truth, "toy-bulge", tmp_path / "f.h5", _toy_ssp(),
        ExtractorConfig(potential_method="spherical"),
    )
    feats = load_particle_features(out)
    assert abs(np.mean(feats["epsilon"])) < 0.15
    assert 0.1 < np.std(feats["epsilon"]) < 0.6


def test_mixed_bimodal(tmp_path):
    p1, v1, m1 = make_disk(10000)
    p2, v2, m2 = make_bulge(10000)
    truth = _make_truth(np.vstack([p1, p2]), np.vstack([v1, v2]), np.concatenate([m1, m2]))
    out = run_extractor(
        truth, "toy-mixed", tmp_path / "f.h5", _toy_ssp(),
        ExtractorConfig(potential_method="spherical"),
    )
    feats = load_particle_features(out)
    eps = feats["epsilon"]
    frac_high = (eps > 0.7).mean()
    frac_low = (np.abs(eps) < 0.3).mean()
    assert frac_high > 0.25 and frac_low > 0.25  # bimodal


def test_mass_conserved(tmp_path):
    pos, vel, mass = make_disk(5000)
    truth = _make_truth(pos, vel, mass)
    out = run_extractor(
        truth, "toy", tmp_path / "f.h5", _toy_ssp(),
        ExtractorConfig(potential_method="spherical"),
    )
    feats = load_particle_features(out)
    assert np.isclose(feats["mass"].sum(), mass.sum(), rtol=1e-5)


def test_spherical_potential_monotone():
    rng = np.random.default_rng(0)
    pos = rng.normal(0, 5, (5000, 3))
    mass = np.full(5000, 1e6)
    phi = compute_potential_spherical(pos, pos, mass)
    r = np.linalg.norm(pos, axis=1)
    # potencial más profundo en el centro
    assert phi[r < 2].mean() < phi[r > 10].mean()
    assert (phi < 0).all()


def test_insufficient_resolution(tmp_path):
    pos, vel, mass = make_disk(150)
    truth = _make_truth(pos, vel, mass)
    # <100 partículas lo bloquea el reader; el extractor acepta 150
    out = run_extractor(
        truth, "toy-small", tmp_path / "f.h5", _toy_ssp(),
        ExtractorConfig(potential_method="spherical"),
    )
    assert out.exists()
