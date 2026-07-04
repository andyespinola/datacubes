"""Tests de LabelProjection (spec 20): conservación, normalización, Kish, cobertura."""
from __future__ import annotations

import numpy as np
import pytest

from aperturenet_labels.phase_b.label_projection import (
    LabelProjectionConfig,
    load_projection,
    run_label_projection,
    _bin_indices_cic,
    _aggregate,
)
from aperturenet_labels.schemas.models import ViewDefinition

from conftest import make_disk


def _view(grid=32, scale_arcsec=0.5, kpc_per_arcsec=1.0, psf=1.0, vec=(0.0, 0.0, 1.0)):
    return ViewDefinition(
        view_id=0,
        view_vector=vec,
        grid_shape=(grid, grid),
        spaxel_scale_arcsec=scale_arcsec,
        kpc_per_arcsec=kpc_per_arcsec,
        fwhm_psf_arcsec=psf,
    )


def _run(tmp_path, pos, mass, p_class, view, r_eff=5.0, **cfg):
    return load_projection(
        run_label_projection(
            positions_centered=pos,
            mass=mass,
            light=mass.copy(),
            p_class=p_class,
            view=view,
            galaxy_id="TOY",
            output_path=tmp_path / "p.npz",
            r_eff_kpc=r_eff,
            config=LabelProjectionConfig(**cfg) if cfg else None,
        )
    )


def _uniform_pclass(n, c=5):
    p = np.zeros((n, c))
    p[:, 1] = 1.0
    return p


def test_mass_conservation(tmp_path):
    pos, vel, mass = make_disk(20000, r_max=6.0)
    view = _view(grid=64)
    proj = _run(tmp_path, pos, mass, _uniform_pclass(len(mass)), view)
    r_cov = float(proj["metadata"]["r_cov_kpc"])
    within = np.linalg.norm(pos, axis=1) <= r_cov
    expected = mass[within].sum()
    total = proj["total_mass_per_spaxel"].sum()
    assert abs(total - expected) / expected < 1e-3


def test_normalization(tmp_path):
    pos, vel, mass = make_disk(20000, r_max=6.0)
    view = _view(grid=64)
    proj = _run(tmp_path, pos, mass, _uniform_pclass(len(mass)), view)
    for key in ("Y_mass_raw", "Y_mass_psf", "Y_lum_raw", "Y_lum_psf"):
        y = proj[key]
        tot = y.sum(axis=0)
        occupied = proj["total_mass_per_spaxel"] > 0
        np.testing.assert_allclose(tot[occupied], 1.0, atol=1e-3)


def test_kish_uniform_weights(tmp_path):
    """Con n partículas de peso igual en un spaxel, N_eff = n."""
    n = 100
    pos = np.zeros((n, 3))  # todas en el spaxel central
    mass = np.full(n, 2.0)
    view = _view(grid=8, vec=(0.0, 0.0, 1.0))
    proj = _run(tmp_path, pos, mass, _uniform_pclass(n), view, use_r_cov=False)
    n_eff_max = proj["n_eff"].max()
    assert abs(n_eff_max - n) / n < 0.05


def test_kish_unequal_weights_lower(tmp_path):
    n = 100
    pos = np.zeros((n, 3))
    mass = np.ones(n)
    mass[0] = 1e4  # un peso domina
    view = _view(grid=8)
    proj = _run(tmp_path, pos, mass, _uniform_pclass(n), view, use_r_cov=False)
    assert proj["n_eff"].max() < 10


def test_coverage_excludes_outside_rcov(tmp_path):
    pos = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    mass = np.array([1.0, 1.0])
    view = _view(grid=16, scale_arcsec=1.0)
    proj = _run(tmp_path, pos, mass, _uniform_pclass(2), view, r_eff=1.0)
    # la partícula a 100 kpc queda fuera de R_cov y fuera de la grilla
    assert proj["total_mass_per_spaxel"].sum() <= 1.0 + 1e-9


def test_cic_weights_sum_one_interior():
    u = np.array([0.3, -2.7])
    v = np.array([1.1, 0.0])
    idx, wts, inside = _bin_indices_cic(u, v, (16, 16), 1.0)
    np.testing.assert_allclose(wts.sum(axis=0), 1.0, atol=1e-12)


def test_class_invariance_under_rotation(tmp_path):
    """Las fracciones globales de clase no dependen de la vista."""
    pos, vel, mass = make_disk(30000, r_max=6.0)
    rng = np.random.default_rng(0)
    p_class = rng.dirichlet(np.ones(5), len(mass))
    fracs = []
    for vec in [(0, 0, 1.0), (1.0, 0, 0), (0.577, 0.577, 0.577)]:
        view = _view(grid=64, vec=vec)
        proj = _run(tmp_path, pos, mass, p_class, view, use_r_cov=False)
        raw = proj["raw_mass_per_class"]
        fracs.append(raw.sum(axis=(1, 2)) / raw.sum())
    fracs = np.array(fracs)
    assert np.abs(fracs - fracs[0]).max() < 0.05


def test_no_nan(tmp_path):
    pos, vel, mass = make_disk(5000)
    view = _view(grid=32)
    proj = _run(tmp_path, pos, mass, _uniform_pclass(len(mass)), view)
    for key in ("Y_mass_raw", "Y_mass_psf", "Y_lum_raw", "Y_lum_psf", "n_eff"):
        assert not np.isnan(proj[key]).any()
