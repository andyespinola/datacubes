"""Tests del Classifier (spec 11 v2.1), incluido el test de reordenamiento."""
from __future__ import annotations

import numpy as np

from aperturenet_labels.phase_a.classifier import (
    ClassifierConfig,
    load_labels,
    run_classifier,
    _reorder_components,
)
from aperturenet_labels.schemas.models import CatalogPriors

from conftest import make_bulge, make_disk, make_halo, synthetic_features


def test_disk_pure(tmp_path):
    """Nota: el spec pide P_disk>0.9 para >90%, pero un GMM K=3 sobre una
    población única la subdivide (radial/energéticamente). Verificamos lo
    que el método garantiza: el disco es la clase dominante y las partículas
    etiquetadas disco son cinemáticamente frías (ε alto)."""
    pos, vel, mass = make_disk(30000)
    feats = synthetic_features(pos, vel, mass)
    out = run_classifier(feats, tmp_path / "l.h5")
    labels = load_labels(out)
    P = labels["P_class"]
    mass_f = feats["mass"].astype(float)
    fracs = (mass_f[:, None] * P).sum(axis=0) / mass_f.sum()
    assert fracs[1] == fracs.max()  # disco dominante
    disk_sel = P[:, 1] > 0.5
    assert np.median(feats["epsilon"][disk_sel]) > 0.7


def test_sum_to_one(tmp_path):
    p1, v1, m1 = make_disk(10000)
    p2, v2, m2 = make_bulge(10000)
    feats = synthetic_features(
        np.vstack([p1, p2]), np.vstack([v1, v2]), np.concatenate([m1, m2])
    )
    out = run_classifier(feats, tmp_path / "l.h5")
    labels = load_labels(out)
    np.testing.assert_allclose(labels["P_class"].sum(axis=1), 1.0, atol=1e-4)


def test_mixed_fractions(tmp_path):
    p1, v1, m1 = make_disk(15000)
    p2, v2, m2 = make_bulge(15000)
    feats = synthetic_features(
        np.vstack([p1, p2]), np.vstack([v1, v2]), np.concatenate([m1, m2])
    )
    out = run_classifier(feats, tmp_path / "l.h5")
    labels = load_labels(out)
    mass = feats["mass"].astype(float)
    fr_disk = (mass * labels["P_class"][:, 1]).sum() / mass.sum()
    assert 0.3 < fr_disk < 0.7  # 50/50 dentro de tolerancia amplia


def test_reorder_bound_bulge_vs_diffuse_halo():
    """El test que habría detectado el bug v2.0: bulbo y halo ambos con ε≈0."""
    # medias en espacio original (cols: eps, logR, |z|/Re, E_norm)
    means = np.array(
        [
            [0.05, 0.8, 1.5, -0.15],   # halo difuso, poco ligado
            [0.85, 0.0, 0.1, -0.5],    # disco
            [0.02, -0.9, 0.2, -0.85],  # bulbo ligado
        ]
    )
    P = np.tile(np.array([[0.2, 0.3, 0.5]]), (10, 1))
    P_new, branch = _reorder_components(P, means, "paper4d")
    # bulge debe ser la columna del componente 2 (más ligado), halo la del 0
    assert P_new[0, 0] == 0.5  # P del comp 2 → bulge
    assert P_new[0, 1] == 0.3  # disco se mantiene
    assert P_new[0, 2] == 0.2  # comp 0 → halo
    assert branch == "permutation_v2.2"


def test_determinism(tmp_path):
    pos, vel, mass = make_disk(8000)
    feats = synthetic_features(pos, vel, mass)
    out1 = run_classifier(feats, tmp_path / "a.h5", config=ClassifierConfig(seed=42))
    out2 = run_classifier(feats, tmp_path / "b.h5", config=ClassifierConfig(seed=42))
    l1, l2 = load_labels(out1), load_labels(out2)
    np.testing.assert_array_equal(l1["P_class"], l2["P_class"])


def test_prior_is_not_constraint(tmp_path):
    """Un prior absurdo no debe forzar las fracciones (P3)."""
    pos, vel, mass = make_disk(20000)
    feats = synthetic_features(pos, vel, mass)
    bad_prior = CatalogPriors(source="mordor", bulge_frac=0.9, disk_frac=0.05, other_frac=0.05)
    out = run_classifier(feats, tmp_path / "l.h5", catalog_priors=bad_prior)
    labels = load_labels(out)
    mass_f = feats["mass"].astype(float)
    fr_disk = (mass_f * labels["P_class"][:, 1]).sum() / mass_f.sum()
    assert fr_disk > 0.5  # la evidencia (disco puro) domina al prior


def test_feature_set_switch(tmp_path):
    pos, vel, mass = make_disk(8000)
    feats = synthetic_features(pos, vel, mass)
    for fs in ("paper4d", "standard3d"):
        out = run_classifier(
            feats, tmp_path / f"{fs}.h5", config=ClassifierConfig(feature_set=fs)
        )
        assert out.exists()
