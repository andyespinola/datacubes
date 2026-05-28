from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from astropy.io import fits
except Exception as exc:  # pragma: no cover - environment dependent
    fits = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from orientation_validation.manifest import (
    build_projection_manifest,
    build_projection_manifest_from_matched,
    read_manifest,
    select_pilot_rows,
    write_manifest,
)


class ManifestTest(unittest.TestCase):
    def test_build_manifest_deduplicates_galaxies(self) -> None:
        if IMPORT_ERROR is not None:
            self.skipTest(f"missing optional dependency: {IMPORT_ERROR}")
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
        if IMPORT_ERROR is not None:
            self.skipTest(f"missing optional dependency: {IMPORT_ERROR}")
        rows = build_projection_manifest("/home/andy/pythonprojects/datacubes/MaNGIA_catalog.fits")
        selected = select_pilot_rows(rows, 5)
        self.assertEqual(len(selected), 5)
        self.assertEqual(len({row.galaxy_id for row in selected}), 5)

    def test_build_manifest_from_matched_units_deduplicates_selected_galaxies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matched_units.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "unit_id",
                        "galaxy_id",
                        "canonical_id",
                        "snapshot",
                        "subhalo_id",
                        "view",
                        "ifu_design_catalog",
                        "re_kpc",
                        "sample_manga",
                        "n_star_part",
                        "n_gas_cell",
                        "selection_rank",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "unit_id": "TNG50-87-10-1",
                        "galaxy_id": "TNG50-87-10",
                        "canonical_id": "TNG50-87-10-1-37",
                        "snapshot": 87,
                        "subhalo_id": 10,
                        "view": 1,
                        "ifu_design_catalog": 37,
                        "re_kpc": 4.0,
                        "sample_manga": 2,
                        "n_star_part": 100,
                        "n_gas_cell": 20,
                        "selection_rank": 1,
                    }
                )
                writer.writerow(
                    {
                        "unit_id": "TNG50-87-10-0",
                        "galaxy_id": "TNG50-87-10",
                        "canonical_id": "TNG50-87-10-0-127",
                        "snapshot": 87,
                        "subhalo_id": 10,
                        "view": 0,
                        "ifu_design_catalog": 127,
                        "re_kpc": 5.0,
                        "sample_manga": 2,
                        "n_star_part": 150,
                        "n_gas_cell": 25,
                        "selection_rank": 2,
                    }
                )
                writer.writerow(
                    {
                        "unit_id": "TNG50-88-20-0",
                        "galaxy_id": "TNG50-88-20",
                        "canonical_id": "TNG50-88-20-0-91",
                        "snapshot": 88,
                        "subhalo_id": 20,
                        "view": 0,
                        "ifu_design_catalog": 91,
                        "re_kpc": 3.0,
                        "sample_manga": 1,
                        "n_star_part": 200,
                        "n_gas_cell": 40,
                        "selection_rank": 3,
                    }
                )

            rows = build_projection_manifest_from_matched(path)

        self.assertEqual([row.galaxy_id for row in rows], ["TNG50-87-10", "TNG50-88-20"])
        self.assertEqual(rows[0].views, "0;1")
        self.assertEqual(rows[0].source_rows, 2)
        self.assertEqual(rows[0].ifu_design, 127)
        self.assertEqual(rows[0].n_star_part, 150)
        self.assertEqual(rows[0].n_gas_cell, 25)
        self.assertAlmostEqual(rows[0].re_kpc, 4.5)


if __name__ == "__main__":
    unittest.main()
