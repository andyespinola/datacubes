from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from scipy.spatial import cKDTree

from orientation_validation.config import ProjectionConfig
from orientation_validation.manifest import ProjectionManifestRow
from orientation_validation.metrics import compute_interorientation_metrics
from orientation_validation.paths import ensure_structural_labeling_on_path
from orientation_validation.projection import build_projection_product, save_projection_product

ensure_structural_labeling_on_path()

from labeling.config import LabelConfig  # noqa: E402
from labeling.models import MorphologyTargets, TNGTruth  # noqa: E402
from labeling.ssp import SSPGrid  # noqa: E402


class ProjectionSmokeTest(unittest.TestCase):
    def test_synthetic_projection_product(self) -> None:
        rng = np.random.default_rng(42)
        n = 400
        coords = rng.normal(0, 2.0, size=(n, 3))
        coords[:, 2] *= 0.2
        vel = np.column_stack((-coords[:, 1], coords[:, 0], rng.normal(0, 0.1, n))) * 40.0
        truth = TNGTruth(
            stellar_pos=coords,
            stellar_vel=vel,
            stellar_mass=np.ones(n),
            stellar_age_gyr=np.clip(rng.normal(0.7, 0.1, n), 0.2, 0.95),
            stellar_metallicity=np.full(n, 0.02),
            gas_pos=rng.normal(0, 2.5, size=(80, 3)),
            gas_vel=rng.normal(0, 20.0, size=(80, 3)),
            gas_mass=np.ones(80),
            gas_sfr=np.ones(80),
            gas_metallicity=np.full(80, 0.02),
            gas_density=np.ones(80),
            subhalo_pos=np.zeros(3),
            subhalo_vel=np.zeros(3),
            stellar_halfmass_rad=4.0,
        )
        targets = MorphologyTargets(
            thin_disk=0.45,
            thick_disk=0.20,
            pseudo_bulge=0.10,
            bulge=0.15,
            halo=0.10,
            unbound=0.0,
            barred=True,
            bar_size_kpc=3.0,
            bar_size_alt_kpc=2.5,
            bar_strength=0.2,
            bar_strength_alt=0.18,
            quality_krot=0.6,
            quality_sigma_ratio=0.8,
            quality_b1b2=1.0,
        )
        features = np.column_stack((np.log10(np.array([1.0, 5.0, 10.0])), np.array([0.01, 0.02, 0.03])))
        ssp_grid = SSPGrid(
            ages_gyr=np.array([1.0, 5.0, 10.0]),
            metallicities=np.array([0.01, 0.02, 0.03]),
            mass_to_light=np.array([0.5, 1.0, 2.0]),
            tree=cKDTree(features),
        )
        row = ProjectionManifestRow(
            galaxy_id="TNG50-87-999",
            snapshot=87,
            subhalo_id=999,
            re_kpc=4.0,
            sample_manga=2,
            ifu_design=127,
            n_star_part=n,
            n_gas_cell=80,
            source_rows=1,
            views="0",
            rcov_kpc=10.0,
            estimated_raw_mb=1.0,
        )
        projection_config = ProjectionConfig(grid_size=31, n_eff_min=1.0)
        products, metadata = build_projection_product(row, truth, targets, ssp_grid, LabelConfig(), projection_config)
        self.assertEqual(sorted(products), ["q000", "q045", "q090", "q135"])
        self.assertEqual(products["q000"]["Y_lum_psf"].shape, (7, 31, 31))
        metrics = compute_interorientation_metrics(products, projection_config)
        self.assertIn("Cglobal", metrics)
        self.assertTrue(np.isfinite(metrics["Cglobal"]))

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "projections.h5"
            save_projection_product(out, row, products, metadata, projection_config)
            with h5py.File(out, "r") as handle:
                self.assertIn("q000", handle)
                self.assertIn("Y_mass_raw", handle["q045"])


if __name__ == "__main__":
    unittest.main()

