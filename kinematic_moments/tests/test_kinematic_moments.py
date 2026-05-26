from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from kinematic_moments.io import (
    output_paths,
    read_mangia_official_cube,
    write_fits,
    write_npz,
)
from kinematic_moments.models import KinematicMaps, KinematicMomentsConfig
from kinematic_moments.pipeline import collect_cube_paths
from kinematic_moments.ppxf_fit import build_fit_grid, build_goodpixels, load_mastar_templates


class KinematicMomentsTests(unittest.TestCase):
    def test_reads_official_mangia_cube_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kinematics-test-") as tmp:
            path = Path(tmp) / "TNG50-1-2-0-127.cube.fits.gz"
            flux = np.ones((12, 5, 4), dtype=np.float32)
            error = np.full_like(flux, 0.2)
            mask = np.zeros_like(flux, dtype=np.int16)
            mask[:, 1:4, 1:3] = 1
            primary = fits.PrimaryHDU(flux)
            primary.header["CRVAL3"] = 3749.0
            primary.header["CRPIX3"] = 1.0
            primary.header["CDELT3"] = 1.0
            primary.header["REDSHIFT"] = 0.15
            fits.HDUList(
                [
                    primary,
                    fits.ImageHDU(error, name="ERROR"),
                    fits.ImageHDU(mask, name="MASK"),
                ]
            ).writeto(path)

            cube = read_mangia_official_cube(path)

            self.assertEqual(cube.galaxy_id, "TNG50-1-2-0-127")
            self.assertEqual(cube.flux.shape, (12, 5, 4))
            self.assertAlmostEqual(cube.redshift, 0.15)
            self.assertAlmostEqual(float(cube.wave[0]), 3749.0)
            self.assertTrue(cube.valid_cube[:, 2, 2].all())
            self.assertFalse(cube.valid_cube[:, 0, 0].any())

    def test_goodpixels_masks_emission_lines(self) -> None:
        config = KinematicMomentsConfig(emission_mask_width_kms=800.0)
        lam = np.array([4700.0, 4861.0, 4875.0, 5007.0, 5200.0], dtype=np.float64)
        goodpixels = build_goodpixels(lam, np.ones(lam.shape, dtype=bool), config)

        self.assertIn(0, goodpixels)
        self.assertIn(4, goodpixels)
        self.assertNotIn(1, goodpixels)
        self.assertNotIn(3, goodpixels)

    def test_mastar_template_loader_log_rebins(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kinematics-template-") as tmp:
            template_path = Path(tmp) / "MaStar_CB19.slog_1_5.fits.gz"
            wave = np.linspace(3200.0, 8200.0, 600)
            spectra = []
            for scale in (0.8, 1.0, 1.2):
                spec = scale * np.ones_like(wave)
                spec -= 0.1 * np.exp(-0.5 * ((wave - 5175.0) / 3.0) ** 2)
                spectra.append(spec)
            hdu = fits.PrimaryHDU(np.asarray(spectra, dtype=np.float32))
            hdu.header["CRVAL1"] = float(wave[0])
            hdu.header["CRPIX1"] = 1.0
            hdu.header["CDELT1"] = float(wave[1] - wave[0])
            hdu.writeto(template_path)

            galaxy_wave = np.linspace(3700.0, 7400.0, 500)
            config = KinematicMomentsConfig(max_templates=2)
            grid = build_fit_grid(galaxy_wave, 0.0, config)
            library = load_mastar_templates(template_path, grid.velscale, config)

            self.assertEqual(library.templates.shape[1], 2)
            self.assertEqual(library.templates.shape[0], library.lam_temp.size)
            self.assertTrue(np.all(np.isfinite(library.templates)))
            self.assertLess(library.velscale, grid.velscale)

    def test_ppxf_recovers_synthetic_gauss_hermite_moments(self) -> None:
        from ppxf.ppxf import ppxf
        from ppxf import ppxf_util

        wave = np.linspace(3800.0, 7300.0, 1400)
        template = np.ones_like(wave)
        for center, amplitude, sigma in (
            (4300.0, 0.20, 2.0),
            (5175.0, 0.28, 2.5),
            (5890.0, 0.18, 2.0),
            (6560.0, 0.10, 2.2),
        ):
            template -= amplitude * np.exp(-0.5 * ((wave - center) / sigma) ** 2)

        template_log, _, velscale = ppxf_util.log_rebin(wave, template)
        templates = template_log[:, None]
        truth = np.array([40.0, 120.0, 0.06, -0.04])
        galaxy = ppxf_util.convolve_gauss_hermite(templates, velscale, truth, template_log.size).ravel()
        noise = np.full_like(galaxy, 0.002)

        fit = ppxf(
            templates,
            galaxy,
            noise,
            velscale,
            [0.0, 100.0],
            moments=4,
            degree=0,
            mdegree=0,
            quiet=True,
        )

        self.assertLess(abs(fit.sol[0] - truth[0]), 2.0)
        self.assertLess(abs(fit.sol[1] - truth[1]), 2.0)
        self.assertLess(abs(fit.sol[2] - truth[2]), 0.01)
        self.assertLess(abs(fit.sol[3] - truth[3]), 0.01)

    def test_writes_npz_and_fits_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kinematics-write-") as tmp:
            outdir = Path(tmp)
            shape = (4, 5)
            maps = KinematicMaps(
                galaxy_id="TNG50-test",
                cube_path=Path("TNG50-test.cube.fits.gz"),
                h3=np.zeros(shape, dtype=np.float32),
                h4=np.ones(shape, dtype=np.float32),
                v_ppxf=np.full(shape, 10.0, dtype=np.float32),
                sigma_ppxf=np.full(shape, 150.0, dtype=np.float32),
                h3_err=np.full(shape, 0.01, dtype=np.float32),
                h4_err=np.full(shape, 0.02, dtype=np.float32),
                quality_mask=np.ones(shape, dtype=np.uint8),
                coverage_mask=np.ones(shape, dtype=np.uint8),
                snr_map=np.full(shape, 20.0, dtype=np.float32),
                chi2_map=np.full(shape, 1.1, dtype=np.float32),
                n_spaxels_fitted=20,
                n_quality_ok=20,
            )
            header = fits.Header()
            header["REDSHIFT"] = 0.15
            header["CRPIX1"] = 2.5
            header["CRPIX2"] = 2.0

            npz_path, fits_path = output_paths("TNG50-test.cube.fits.gz", outdir)
            write_npz(maps, npz_path)
            write_fits(maps, header, fits_path)

            with np.load(npz_path) as data:
                self.assertEqual(data["h3"].shape, shape)
                self.assertEqual(int(data["n_quality_ok"]), 20)
            with fits.open(fits_path, memmap=False) as hdul:
                self.assertIn("H3", hdul)
                self.assertIn("H4", hdul)
                self.assertEqual(hdul["H4"].data.shape, shape)
                self.assertEqual(hdul[0].header["NQUAL"], 20)

    def test_collect_cube_paths_applies_deterministic_limit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kinematics-limit-") as tmp:
            root = Path(tmp)
            for name in (
                "TNG50-2-2-0-127.cube.fits.gz",
                "TNG50-1-1-0-127.cube.fits.gz",
                "TNG50-3-3-0-127.cube.fits.gz",
            ):
                (root / name).touch()

            paths = collect_cube_paths(cube_glob=str(root / "*.cube.fits.gz"), limit=2)

            self.assertEqual(len(paths), 2)
            self.assertEqual(paths[0].name, "TNG50-1-1-0-127.cube.fits.gz")
            self.assertEqual(paths[1].name, "TNG50-2-2-0-127.cube.fits.gz")


if __name__ == "__main__":
    unittest.main()
