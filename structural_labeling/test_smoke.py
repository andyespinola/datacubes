from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from astropy.io import fits
import h5py
import numpy as np

from labeling.config import LabelConfig
from labeling.manifest import read_manifest
from labeling.pipeline import LabelingPipeline
from labeling.ssp import load_ssp_grid
from labeling.tng import load_cutout_truth, load_morphology_targets


class StructuralLabelingSmokeTest(unittest.TestCase):
    def test_pipeline_runs_on_synthetic_data(self) -> None:
        work = Path(tempfile.mkdtemp(prefix="structural-labeling-"))

        ssp = np.ones((3, 20), dtype=np.float32)
        hdu = fits.PrimaryHDU(ssp)
        header = hdu.header
        header["CRPIX1"] = 1
        header["CDELT1"] = 1.0
        header["CRVAL1"] = 4000.0
        header["NAME0"] = "spec_ssp_0.5Gyr_z0.01.spec"
        header["NAME1"] = "spec_ssp_2.0Gyr_z0.02.spec"
        header["NAME2"] = "spec_ssp_8.0Gyr_z0.03.spec"
        header["NORM0"] = 1.0
        header["NORM1"] = 2.0
        header["NORM2"] = 4.0
        ssp_path = work / "ssp.fits"
        hdu.writeto(ssp_path)

        cube = np.ones((10, 20, 20), dtype=np.float32)
        mask = np.zeros_like(cube, dtype=np.int16)
        primary = fits.PrimaryHDU(cube)
        primary.header["KPCSEC"] = 1.2
        primary.header["PSF"] = 1.43
        primary.header["CD1_1"] = -0.5 / 3600.0
        primary.header["CD2_2"] = 0.5 / 3600.0
        cube_path = work / "TNG50-87-141934-0-127.cube.fits.gz"
        fits.HDUList(
            [
                primary,
                fits.ImageHDU(np.ones_like(cube), name="ERROR"),
                fits.ImageHDU(mask, name="MASK"),
            ]
        ).writeto(cube_path)

        manifest_path = work / "manifest.csv"
        with manifest_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "canonical_id",
                    "rss_path",
                    "cube_path",
                    "pipe3d_path",
                    "snapshot",
                    "subhalo_id",
                    "view",
                    "re_kpc",
                    "ifu_design",
                    "repeat_count",
                    "n_star_part",
                    "n_gas_cell",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "canonical_id": "TNG50-87-141934-0-127",
                    "rss_path": "",
                    "cube_path": str(cube_path),
                    "pipe3d_path": "",
                    "snapshot": 87,
                    "subhalo_id": 141934,
                    "view": 0,
                    "re_kpc": 7.5,
                    "ifu_design": 127,
                    "repeat_count": 1,
                    "n_star_part": 500,
                    "n_gas_cell": 200,
                }
            )

        cutout_path = work / "cutout.hdf5"
        with h5py.File(cutout_path, "w") as handle:
            stars = handle.create_group("PartType4")
            n_stars = 500
            coords = np.random.normal(0, 3, size=(n_stars, 3)).astype("f8")
            coords[:150, 0] *= 0.3
            coords[:150, 1] *= 0.3
            stars["Coordinates"] = coords
            stars["Velocities"] = np.random.normal(0, 50, size=(n_stars, 3)).astype("f8")
            stars["Masses"] = np.abs(np.random.normal(1.0, 0.3, size=n_stars)).astype("f8")
            stars["GFM_StellarFormationTime"] = np.clip(np.random.normal(0.7, 0.15, size=n_stars), 0.1, 0.99).astype("f8")
            stars["GFM_Metallicity"] = np.clip(np.random.normal(0.02, 0.005, size=n_stars), 0.005, 0.04).astype("f8")

            gas = handle.create_group("PartType0")
            n_gas = 200
            gas["Coordinates"] = np.random.normal(0, 4, size=(n_gas, 3)).astype("f8")
            gas["Velocities"] = np.random.normal(0, 40, size=(n_gas, 3)).astype("f8")
            gas["Masses"] = np.abs(np.random.normal(1.0, 0.3, size=n_gas)).astype("f8")
            gas["StarFormationRate"] = np.abs(np.random.normal(0.3, 0.2, size=n_gas)).astype("f8")
            gas["Density"] = np.abs(np.random.normal(1.0, 0.1, size=n_gas)).astype("f8")
            gas["InternalEnergy"] = np.abs(np.random.normal(1.0, 0.1, size=n_gas)).astype("f8")
            gas["ElectronAbundance"] = np.abs(np.random.normal(1.0, 0.1, size=n_gas)).astype("f8")
            gas["GFM_Metallicity"] = np.clip(np.random.normal(0.02, 0.005, size=n_gas), 0.005, 0.04).astype("f8")

        metadata_path = work / "subhalo.json"
        metadata_path.write_text(json.dumps({"pos": [0, 0, 0], "vel": [0, 0, 0], "halfmassrad_stars": 5.0}))

        morph_path = work / "morph.hdf5"
        with h5py.File(morph_path, "w") as handle:
            group = handle.create_group("Snapshot_87")
            group["SubhaloID"] = np.array([141934], dtype="i8")
            group["ThinDisc"] = np.array([[0.45], [0.0], [0.0]], dtype="f8")
            group["ThickDisc"] = np.array([[0.20], [0.0], [0.0]], dtype="f8")
            group["PseudoBulge"] = np.array([[0.10], [0.0], [0.0]], dtype="f8")
            group["Bulge"] = np.array([[0.15], [0.0], [0.0]], dtype="f8")
            group["Halo"] = np.array([[0.10], [0.0], [0.0]], dtype="f8")
            group["UnboundMass"] = np.array([0.0], dtype="f8")
            group["Barred"] = np.array([1], dtype="i4")
            group["BarSize"] = np.array([[3.0], [2.5]], dtype="f8")
            group["BarStrength"] = np.array([[0.2], [0.18]], dtype="f8")
            group["QualityFlags"] = np.array([[0.6], [0.8], [1.0]], dtype="f8")

        row = read_manifest(manifest_path)[0]
        ssp_grid = load_ssp_grid(ssp_path)
        truth = load_cutout_truth(cutout_path, metadata_path)
        targets = load_morphology_targets(morph_path, 87, 141934)

        pipeline = LabelingPipeline(LabelConfig(), ssp_grid)
        products = pipeline.run(row, truth, targets)

        self.assertEqual(products.soft_mass.shape, (7, 20, 20))
        self.assertEqual(products.soft_light.shape, (7, 20, 20))
        sums = np.sum(products.soft_mass[:, products.valid_mask], axis=0)
        self.assertTrue(np.allclose(sums, 1.0, atol=1e-4))
        self.assertTrue(np.all(products.hard_mass[products.valid_mask] >= 1))
        self.assertIn("050", products.hard_mass_variants)
        self.assertIn("055", products.hard_mass_variants)
        self.assertIn("060", products.hard_mass_variants)
        self.assertIn("mass", products.hard_variant_summary["050"])
        self.assertIn("light", products.hard_variant_summary["060"])
        self.assertIn("disk_family_total", products.global_fraction_recovered)
        self.assertIn("other", products.global_fraction_recovered)


if __name__ == "__main__":
    unittest.main()
