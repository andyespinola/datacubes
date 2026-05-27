from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mangia_asset_matcher.ids import UnitKey, parse_unit_from_text
from mangia_asset_matcher.match_assets import main
from mangia_asset_matcher.matcher import MatchConfig, build_matches, write_outputs
from mangia_asset_matcher.scanners import _pipe3d_stack_keys


def write_catalog(path: Path, n_units: int) -> None:
    fieldnames = [
        "snapshot",
        "subhalo_id",
        "view",
        "manga_ifu_dsn",
        "re_kpc",
        "sample_manga",
        "n_star_part",
        "n_gas_cell",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(n_units):
            writer.writerow(
                {
                    "snapshot": 87,
                    "subhalo_id": 1000 + idx,
                    "view": idx % 2,
                    "manga_ifu_dsn": 37,
                    "re_kpc": 2.0 + idx,
                    "sample_manga": 1,
                    "n_star_part": 100 + idx,
                    "n_gas_cell": 10,
                }
            )


def unit_name(idx: int) -> str:
    return f"TNG50-87-{1000 + idx}-{idx % 2}-37"


def create_assets(
    root: Path,
    n_units: int,
    *,
    missing_cube: set[int] | None = None,
    missing_cutout: set[int] | None = None,
    missing_metadata: set[int] | None = None,
    missing_v: set[int] | None = None,
    missing_sigma: set[int] | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    missing_cube = missing_cube or set()
    missing_cutout = missing_cutout or set()
    missing_metadata = missing_metadata or set()
    missing_v = missing_v or set()
    missing_sigma = missing_sigma or set()
    catalog = root / "catalog.csv"
    cubes = root / "cubes"
    tng = root / "tng"
    maps = root / "maps"
    outdir = root / "out"
    for path in (cubes, tng / "cutouts", tng / "metadata", tng / "morphology", maps):
        path.mkdir(parents=True, exist_ok=True)
    (tng / "morphology" / "morphs_kinematic_bars.hdf5").touch()
    write_catalog(catalog, n_units)

    for idx in range(n_units):
        name = unit_name(idx)
        if idx not in missing_cube:
            (cubes / f"{name}.cube.fits.gz").touch()
            (cubes / f"{name}.cube_val.fits.gz").touch()
        galaxy_name = f"TNG50-87-{1000 + idx}"
        if idx not in missing_cutout:
            (tng / "cutouts" / f"{galaxy_name}.cutout.hdf5").touch()
        if idx not in missing_metadata:
            (tng / "metadata" / f"{galaxy_name}.subhalo.json").write_text("{}", encoding="utf-8")
        payload = {}
        if idx not in missing_v:
            payload["V"] = np.ones((4, 4), dtype=np.float32)
        if idx not in missing_sigma:
            payload["SIGMA"] = np.ones((4, 4), dtype=np.float32)
        np.savez_compressed(maps / f"{name}.maps.npz", **payload)
    return catalog, cubes, tng, maps, outdir


def config_for(catalog: Path, cubes: Path, tng: Path, maps: Path, limit: int = 0, require_count: int = 0) -> MatchConfig:
    return MatchConfig(
        catalog=catalog,
        cube_roots=(cubes,),
        tng_cache=tng,
        maps2d_roots=(maps,),
        limit=limit,
        require_count=require_count,
    )


class AssetMatcherTests(unittest.TestCase):
    def test_parses_units_with_and_without_ifu(self) -> None:
        with_ifu = parse_unit_from_text("TNG50-87-141934-0-127.cube.fits.gz")
        without_ifu = parse_unit_from_text("maps/TNG50-87-141934-0.maps.npz")

        self.assertEqual(with_ifu.key, UnitKey(87, 141934, 0))
        self.assertEqual(with_ifu.ifu_design, 127)
        self.assertEqual(without_ifu.key, UnitKey(87, 141934, 0))
        self.assertIsNone(without_ifu.ifu_design)

    def test_builds_strict_matches_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="asset-matcher-") as tmp:
            root = Path(tmp)
            catalog, cubes, tng, maps, outdir = create_assets(root, 3)
            result = build_matches(config_for(catalog, cubes, tng, maps, limit=2))
            write_outputs(result, outdir)

            self.assertEqual(len(result.matched_all), 3)
            self.assertEqual(len(result.selected), 2)
            self.assertEqual(result.selected[0]["selection_rank"], 1)
            self.assertTrue((outdir / "matched_units.csv").exists())
            self.assertTrue((outdir / "matched_units_all.csv").exists())
            self.assertTrue((outdir / "asset_inventory.csv").exists())
            self.assertTrue((outdir / "unmatched_report.json").exists())
            self.assertTrue((outdir / "unmatched_report.md").exists())

    def test_excludes_missing_cube_cutout_metadata_v_and_sigma(self) -> None:
        with tempfile.TemporaryDirectory(prefix="asset-matcher-missing-") as tmp:
            root = Path(tmp)
            catalog, cubes, tng, maps, _outdir = create_assets(
                root,
                5,
                missing_cube={0},
                missing_cutout={1},
                missing_metadata={2},
                missing_v={3},
                missing_sigma={4},
            )
            result = build_matches(config_for(catalog, cubes, tng, maps))
            reasons = {
                int(row["subhalo_id"]): row["exclusion_reasons"]
                for row in result.inventory
            }

            self.assertEqual(len(result.matched_all), 0)
            self.assertIn("missing_cube", reasons[1000])
            self.assertIn("missing_cutout", reasons[1001])
            self.assertIn("missing_metadata", reasons[1002])
            self.assertIn("missing_v_map", reasons[1003])
            self.assertIn("missing_sigma_map", reasons[1004])

    def test_limit_zero_keeps_all_and_limit_ten_selects_ten(self) -> None:
        with tempfile.TemporaryDirectory(prefix="asset-matcher-limit-") as tmp:
            root = Path(tmp)
            catalog, cubes, tng, maps, _outdir = create_assets(root, 12)

            all_result = build_matches(config_for(catalog, cubes, tng, maps, limit=0))
            ten_result = build_matches(config_for(catalog, cubes, tng, maps, limit=10))

            self.assertEqual(len(all_result.selected), 12)
            self.assertEqual(len(ten_result.selected), 10)

    def test_require_count_failure_from_cli_still_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory(prefix="asset-matcher-cli-") as tmp:
            root = Path(tmp)
            catalog, cubes, tng, maps, outdir = create_assets(root, 3)

            code = main(
                [
                    "--catalog",
                    str(catalog),
                    "--cube-root",
                    str(cubes),
                    "--tng-cache",
                    str(tng),
                    "--maps2d-root",
                    str(maps),
                    "--limit",
                    "0",
                    "--require-count",
                    "4",
                    "--outdir",
                    str(outdir),
                ]
            )

            self.assertEqual(code, 2)
            self.assertTrue((outdir / "matched_units.csv").exists())
            with (outdir / "matched_units.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)

    def test_pipe3d_stack_keys_use_documented_ssp_channels(self) -> None:
        v_key, sigma_key, shape = _pipe3d_stack_keys("SSP_pyPipe3D_REC", (20, 69, 69))

        self.assertEqual(v_key, "SSP_pyPipe3D_REC[13]")
        self.assertEqual(sigma_key, "SSP_pyPipe3D_REC[15]")
        self.assertEqual(shape, "69x69")

    def test_detects_pipe3d_fits_stack_channels_when_astropy_is_available(self) -> None:
        try:
            from astropy.io import fits
        except Exception:
            self.skipTest("astropy is not installed")

        with tempfile.TemporaryDirectory(prefix="asset-matcher-fits-") as tmp:
            root = Path(tmp)
            catalog, cubes, tng, maps, _outdir = create_assets(root, 1, missing_v={0}, missing_sigma={0})
            map_path = maps / f"{unit_name(0)}.cube_maps.fits"
            payload = np.zeros((20, 4, 4), dtype=np.float32)
            hdu = fits.ImageHDU(payload, name="SSP_pyPipe3D_REC")
            hdu.header["DESC_13"] = "Vlos"
            hdu.header["DESC_15"] = "sigma"
            fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(map_path)
            (maps / f"{unit_name(0)}.maps.npz").unlink()

            result = build_matches(config_for(catalog, cubes, tng, maps))

        self.assertEqual(len(result.matched_all), 1)
        self.assertEqual(result.matched_all[0]["v_map_key"], "SSP_pyPipe3D_REC[13]")
        self.assertEqual(result.matched_all[0]["sigma_map_key"], "SSP_pyPipe3D_REC[15]")
        self.assertEqual(result.matched_all[0]["maps2d_shape"], "4x4")


if __name__ == "__main__":
    unittest.main()
