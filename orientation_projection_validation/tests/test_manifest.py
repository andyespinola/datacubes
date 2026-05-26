from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from astropy.io import fits
import numpy as np

from orientation_validation.manifest import build_projection_manifest, read_manifest, select_pilot_rows, write_manifest


class ManifestTest(unittest.TestCase):
    def test_build_manifest_deduplicates_galaxies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.fits"
            table = fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="snapshot", array=np.array([87, 87, 88]), format="K"),
                    fits.Column(name="subhalo_id", array=np.array([10, 10, 20]), format="K"),
                    fits.Column(name="view", array=np.array([0, 1, 0]), format="K"),
                    fits.Column(name="stellar_mass", array=np.array([10.0, 10.0, 9.5]), format="E"),
                    fits.Column(name="re_kpc", array=np.array([4.0, 4.2, 5.0]), format="E"),
                    fits.Column(name="sample_manga", array=np.array([2, 2, 1]), format="K"),
                    fits.Column(name="manga_ifu_dsn", array=np.array([127, 127, 91]), format="K"),
                    fits.Column(name="distances_selec", array=np.array([0.1, 0.1, 0.2]), format="E"),
                    fits.Column(name="n_star_part", array=np.array([100, 110, 200]), format="K"),
                    fits.Column(name="n_gas_cell", array=np.array([30, 35, 40]), format="K"),
                ]
            )
            fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)

            rows = build_projection_manifest(path)
            self.assertEqual(len(rows), 2)
            first = rows[0]
            self.assertEqual(first.galaxy_id, "TNG50-87-10")
            self.assertEqual(first.views, "0;1")
            self.assertEqual(first.n_star_part, 110)
            self.assertAlmostEqual(first.rcov_kpc, 10.25, places=5)

            out = Path(tmp) / "manifest.csv"
            write_manifest(out, rows)
            loaded = read_manifest(out)
            self.assertEqual([row.galaxy_id for row in loaded], [row.galaxy_id for row in rows])

    def test_select_pilot_rows_is_bounded(self) -> None:
        rows = build_projection_manifest("/home/andy/pythonprojects/datacubes/MaNGIA_catalog.fits")
        selected = select_pilot_rows(rows, 5)
        self.assertEqual(len(selected), 5)
        self.assertEqual(len({row.galaxy_id for row in selected}), 5)


if __name__ == "__main__":
    unittest.main()
