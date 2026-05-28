from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from run_matched_labeling import build_structural_manifest


class MatchedLabelingManifestTests(unittest.TestCase):
    def test_builds_structural_manifest_from_matched_units_without_catalog(self) -> None:
        with tempfile.TemporaryDirectory(prefix="matched-labeling-") as tmp:
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
                        "cube_ifu_file",
                        "cube_path",
                        "cutout_path",
                        "metadata_path",
                        "morphology_catalog_path",
                        "maps2d_path",
                        "re_kpc",
                        "n_star_part",
                        "n_gas_cell",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "unit_id": "TNG50-87-141934-0",
                        "galaxy_id": "TNG50-87-141934",
                        "canonical_id": "TNG50-87-141934-0-37",
                        "snapshot": 87,
                        "subhalo_id": 141934,
                        "view": 0,
                        "ifu_design_catalog": 37,
                        "cube_ifu_file": 127,
                        "cube_path": "/data/TNG50-87-141934-0-127.cube.fits.gz",
                        "cutout_path": "/cache/TNG50-87-141934.cutout.hdf5",
                        "metadata_path": "/cache/TNG50-87-141934.subhalo.json",
                        "morphology_catalog_path": "/cache/morphology/morphs_kinematic_bars.hdf5",
                        "maps2d_path": "/maps/TNG50-87-141934-0-127.cube_maps.fits",
                        "re_kpc": 7.5,
                        "n_star_part": 100,
                        "n_gas_cell": 20,
                    }
                )

            rows, matched = build_structural_manifest(path, catalog_path=None)

        self.assertEqual(len(rows), 1)
        self.assertEqual(len(matched), 1)
        self.assertEqual(rows[0].canonical_id, "TNG50-87-141934-0-37")
        self.assertEqual(rows[0].ifu_design, 37)
        self.assertEqual(rows[0].cube_path, "/data/TNG50-87-141934-0-127.cube.fits.gz")
        self.assertEqual(rows[0].pipe3d_path, "/maps/TNG50-87-141934-0-127.cube_maps.fits")


if __name__ == "__main__":
    unittest.main()
