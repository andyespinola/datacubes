from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from ImagesMangGenerator.phase_input.image_provider import (
    ImageProvider,
    ImageProviderConfig,
    ImageProviderInput,
    center_pad_or_crop,
    output_path_for,
    save_provided_image,
)


pytest.importorskip("speclite")


def write_synthetic_cube(path: Path, shape: tuple[int, int, int] = (128, 8, 8)) -> None:
    n_wave, height, width = shape
    wave0 = 3600.0
    delta = 45.0
    wave = wave0 + np.arange(n_wave, dtype=np.float32) * delta
    y, x = np.mgrid[:height, :width]
    profile = np.exp(-((x - width / 2) ** 2 + (y - height / 2) ** 2) / 12.0).astype(np.float32)
    spectrum = (1.0 + 0.1 * np.sin(wave / 500.0)).astype(np.float32)
    flux = spectrum[:, None, None] * profile[None, :, :] * 10.0

    header = fits.Header()
    header["UNITS"] = "1E-17 erg/s/cm^2/Angstrom/spaxel"
    header["CRVAL3"] = wave0
    header["CRPIX3"] = 1
    header["CDELT3"] = delta
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRPIX1"] = width / 2
    header["CRPIX2"] = height / 2
    header["CRVAL1"] = 150.0
    header["CRVAL2"] = 2.0
    header["CD1_1"] = -0.5 / 3600.0
    header["CD2_2"] = 0.5 / 3600.0
    fits.PrimaryHDU(flux.astype(np.float32), header=header).writeto(path)


def test_center_pad_or_crop() -> None:
    image = np.ones((3, 4, 6), dtype=np.float32)
    assert center_pad_or_crop(image, (6, 4)).shape == (3, 6, 4)
    assert center_pad_or_crop(image, (2, 2)).shape == (3, 2, 2)


def test_mangia_provider_writes_npz(tmp_path: Path) -> None:
    cube_path = tmp_path / "synthetic.cube.fits"
    write_synthetic_cube(cube_path)
    config = ImageProviderConfig(output_shape=(10, 10))
    provided = ImageProvider().provide(
        ImageProviderInput(
            mode="mangia",
            galaxy_id="synthetic",
            cube_path=cube_path,
            view_id=0,
            config=config,
        )
    )
    assert provided.image.shape == (3, 10, 10)
    assert provided.image.dtype == np.float32
    assert np.isfinite(provided.image).all()
    assert provided.image.sum() > 0

    output = save_provided_image(provided, output_path_for(tmp_path, provided))
    with np.load(output) as payload:
        assert payload["image"].shape == (3, 10, 10)
        metadata = json.loads(payload["metadata"].item())
    assert metadata["source"] == "synthesized"
