from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from astropy.io import fits
import numpy as np


@dataclass(slots=True)
class CubeGeometry:
    shape: tuple[int, int]
    n_wave: int
    pixel_scale_arcsec: float
    kpc_per_arcsec: float
    psf_fwhm_arcsec: float
    wavelength: np.ndarray

    @property
    def pixel_scale_kpc(self) -> float:
        return self.pixel_scale_arcsec * self.kpc_per_arcsec

    @property
    def psf_fwhm_kpc(self) -> float:
        return self.psf_fwhm_arcsec * self.kpc_per_arcsec


def read_cube_geometry(path: str | Path) -> CubeGeometry:
    with fits.open(path, memmap=True, lazy_load_hdus=True, do_not_scale_image_data=True) as hdul:
        hdu = hdul[0] if hdul[0].data is not None else hdul["FLUX"]
        header = hdu.header
        shape = hdu.shape
        if len(shape) != 3:
            raise ValueError(f"Expected 3D cube in {path}, got shape={shape}")
        n_wave, height, width = shape
        pixel_scale_arcsec = 0.5
        if "CD1_1" in header:
            pixel_scale_arcsec = abs(float(header["CD1_1"])) * 3600.0
        elif "CDELT1" in header:
            pixel_scale_arcsec = abs(float(header["CDELT1"])) * 3600.0
        crval = float(header.get("CRVAL3", 0.0))
        cdelt = float(header.get("CDELT3", 1.0))
        crpix = float(header.get("CRPIX3", 1.0))
        wave = crval + (np.arange(n_wave, dtype=np.float64) + 1.0 - crpix) * cdelt
        return CubeGeometry(
            shape=(int(height), int(width)),
            n_wave=int(n_wave),
            pixel_scale_arcsec=float(pixel_scale_arcsec),
            kpc_per_arcsec=float(header.get("KPCSEC", 1.0)),
            psf_fwhm_arcsec=float(header.get("PSF", 1.43)),
            wavelength=wave,
        )


def read_cube_flux(path: str | Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        hdu = hdul[0] if hdul[0].data is not None else hdul["FLUX"]
        return np.asarray(hdu.data, dtype=np.float32)


def read_cube_signal_mask(path: str | Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        flux_hdu = hdul[0] if hdul[0].data is not None else hdul["FLUX"]
        cube = np.asarray(flux_hdu.data, dtype=np.float32)
        if "MASK" in hdul:
            mask = np.asarray(hdul["MASK"].data)
            spatial_mask = np.isfinite(mask).any(axis=0) & (np.nanmin(mask, axis=0) == 0)
        else:
            spatial_mask = np.isfinite(cube).any(axis=0)
        signal_mask = np.nanmax(np.abs(cube), axis=0) > 0.0
        return np.asarray(spatial_mask & signal_mask, dtype=bool)


def read_pipe3d_maps(path: str | Path) -> dict[str, np.ndarray]:
    maps: dict[str, np.ndarray] = {}
    with fits.open(path, memmap=False) as hdul:
        ssp = np.asarray(hdul["SSP_pyPipe3D_REC"].data, dtype=np.float32)
        maps["v_star"] = ssp[13]
        maps["sigma_star"] = ssp[15]
        maps["age_lw_log10"] = ssp[5]
        maps["age_mw_log10"] = ssp[6]
        maps["metallicity_lw_log10"] = ssp[8]
        maps["metallicity_mw_log10"] = ssp[9]
        maps["av"] = ssp[11]
        maps["stellar_mass_density_log10"] = ssp[18]
    return maps
