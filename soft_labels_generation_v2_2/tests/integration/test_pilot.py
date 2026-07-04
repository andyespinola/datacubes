"""Integración con las galaxias piloto (requiere data/ con cutouts y cubos).

Incluye el test de alineación obligatorio del spec 20 / MIGRATION.md.
Se salta si los datos del piloto no están disponibles.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "MaNGIA_catalog.fits").exists(), reason="data/ del piloto no disponible"
)


@pytest.fixture(scope="module")
def pilot_rows():
    from aperturenet_labels.io import manifest

    return manifest.pilot_manifest(DATA_DIR)


def test_manifest_finds_both_pilots(pilot_rows):
    assert {r.subhalo_id for r in pilot_rows} == {155298, 192324}


def test_alignment_against_pipe3d(pilot_rows):
    """Gate obligatorio: proyección de masa vs mapa de masa pyPipe3D.

    Umbrales: centroide < 1 spaxel SIEMPRE; Spearman > 0.9 para el mejor
    piloto y > 0.8 para ambos (el mapa pyPipe3D de 192324 es plano y
    degrada la correlación de rangos; la inspección visual confirma el
    registro espacial — ver notebook de validación).
    """
    from scipy.stats import spearmanr
    from scipy.ndimage import center_of_mass
    from aperturenet_labels.io import tng_reader, units, mangia_reader
    from aperturenet_labels.core.geometry import view_vector_from_index, deposit_to_grid
    from aperturenet_labels.phase_b.label_projection import mangia_raster_coords

    rhos = []
    for r in pilot_rows:
        truth = tng_reader.load_cutout_truth(
            r.cutout_path, r.subhalo_json_path, r.cutout_phase2_path
        )
        truth = units.convert_truth_units(truth)
        geom = mangia_reader.load_cube_geometry(r.cube_path)
        maps = mangia_reader.load_pipe3d_maps(r.pipe3d_maps_path)
        mass_p3d = maps["mass_density"]

        centered = truth.stellar_pos - truth.subhalo_pos[None, :]
        vec = view_vector_from_index(r.view, r.repeat_count)
        u, v, _ = mangia_raster_coords(centered, vec)
        grid = deposit_to_grid(
            v, u, truth.stellar_mass, geom.shape, geom.pixel_scale_kpc,
            sigma_pixels=geom.psf_sigma_pixels,
        )
        good = np.isfinite(mass_p3d) & (mass_p3d != 0)
        thr = np.percentile(mass_p3d[good], 30)
        m = good & (mass_p3d > thr) & (grid > 0)
        rho = spearmanr(np.log10(grid[m]), mass_p3d[m]).statistic
        rhos.append(rho)

        p3d_lin = np.where(good, 10 ** np.clip(mass_p3d, -5, 15), 0.0)
        c_ours = center_of_mass(grid / grid.sum())
        c_p3d = center_of_mass(p3d_lin)
        shift = float(np.hypot(c_ours[0] - c_p3d[0], c_ours[1] - c_p3d[1]))
        assert shift < 1.0, f"{r.canonical_id}: centroide {shift:.2f} px"
        assert rho > 0.8, f"{r.canonical_id}: Spearman {rho:.3f}"
    assert max(rhos) > 0.9


def test_dataset_entries_roundtrip(pilot_rows):
    from aperturenet_labels.phase_c.packer import validate_dataset_entry
    from aperturenet_labels.io import manifest as mmod

    for r in pilot_rows:
        gal = mmod.galaxy_id(r.snapshot, r.subhalo_id)
        entry = DATA_DIR / "output" / "dataset_entries" / f"{gal}_v{r.view}.h5"
        if not entry.exists():
            pytest.skip("dataset entries aún no generados (correr run --pilot)")
        report = validate_dataset_entry(entry)
        assert report["galaxy_id"] == gal
        assert report["n_valid"] > 100


def test_phase_a_products_exist_and_valid(pilot_rows):
    from aperturenet_labels.phase_a.classifier import load_labels
    from aperturenet_labels.io import manifest as mmod

    for r in pilot_rows:
        gal = mmod.galaxy_id(r.snapshot, r.subhalo_id)
        final = DATA_DIR / "intermediate" / "phase_a" / gal / "particle_labels_final.h5"
        if not final.exists():
            pytest.skip("fase A aún no corrida")
        labels = load_labels(final)
        P = labels["P_class"]
        assert P.shape[1] == 5
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-3)
