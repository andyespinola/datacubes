"""Tests de MaskBuilder (spec 22) y Packer (spec 30) con FITS/HDF5 sintéticos."""
from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from aperturenet_labels.phase_b.mask_builder import MaskBuilderConfig, load_mask, run_mask_builder
from aperturenet_labels.phase_c.packer import pad_to


def _toy_cube(tmp_path, shape=(200, 16, 16), signal_radius=5, noise=0.001, signal=0.1):
    nw, h, w = shape
    rng = np.random.default_rng(0)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    r = np.hypot(yy - h / 2 + 0.5, xx - w / 2 + 0.5)
    base = np.where(r < signal_radius, signal, 0.0)
    flux = base[None] + rng.normal(0, noise, shape)
    error = np.full(shape, noise)
    primary = fits.PrimaryHDU(flux.astype(np.float32))
    primary.header["CRVAL3"] = 5000.0
    primary.header["CDELT3"] = 2.0
    err_hdu = fits.ImageHDU(error.astype(np.float32), name="ERROR")
    path = tmp_path / "cube.fits"
    fits.HDUList([primary, err_hdu]).writeto(path)
    return path, r


def test_criteria_and_connectivity(tmp_path):
    cube_path, r = _toy_cube(tmp_path)
    n_map = np.full((16, 16), 100.0)
    out = run_mask_builder(
        n_map, cube_path, "TOY", 0, tmp_path / "m.npz",
        MaskBuilderConfig(snr_window_angstrom=(5000.0, 5400.0), min_island_area=4),
    )
    mask = load_mask(out)
    assert mask["M_criterion_A"].all()  # conteo masivo
    assert mask["M_valid"][8, 8]  # centro válido
    assert not mask["M_valid"][0, 0]  # esquina sin señal
    # la región válida ≈ disco central conexo
    assert mask["M_valid"].sum() >= 20


def test_low_particle_count_blocks(tmp_path):
    cube_path, r = _toy_cube(tmp_path)
    n_map = np.zeros((16, 16))
    out = run_mask_builder(n_map, cube_path, "TOY", 0, tmp_path / "m.npz")
    mask = load_mask(out)
    assert mask["M_valid"].sum() == 0


def test_pad_to_centered():
    arr = np.arange(69 * 69, dtype=float).reshape(69, 69)
    padded = pad_to(arr, 74)
    assert padded.shape == (74, 74)
    # contenido centrado: offset (74-69)//2 = 2
    np.testing.assert_array_equal(padded[2:71, 2:71], arr)
    assert padded[0].sum() == 0 and padded[-1].sum() == 0


def test_pad_to_3d():
    cube = np.ones((10, 69, 69))
    padded = pad_to(cube, 74)
    assert padded.shape == (10, 74, 74)
    assert padded.sum() == cube.sum()


def test_pad_to_rejects_larger():
    with pytest.raises(ValueError):
        pad_to(np.ones((80, 80)), 74)
