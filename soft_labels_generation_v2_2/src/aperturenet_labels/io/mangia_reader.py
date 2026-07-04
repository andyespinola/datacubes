"""Lectura del cubo MaNGIA y de los mapas pyPipe3D (cube_maps.fits).

`load_cube_geometry` se porta de geometry.py v1 sin la lógica de máscara
estricta (eso ahora es responsabilidad del MaskBuilder, spec 22).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from ..core.geometry import view_vector_from_index
from ..schemas.models import CubeGeometry, ViewDefinition

# Canales del extension SSP_pyPipe3D_REC (ver DESC_* del header)
PIPE3D_SSP_CHANNELS = {
    "v_star": 13,
    "sigma_star": 15,
    "age_lw": 5,
    "age_mw": 6,
    "metallicity_lw": 8,
    "metallicity_mw": 9,
    "av": 11,
    "ml": 17,
    "mass_density": 18,
    "mass_density_dust_corr": 19,
}


def load_cube_geometry(cube_path: str | Path) -> CubeGeometry:
    with fits.open(cube_path) as hdul:
        cube = np.asarray(hdul[0].data, dtype=np.float32)
        header = hdul[0].header
        if cube.ndim != 3:
            raise ValueError(f"Esperaba un cubo 3D en {cube_path}, shape={cube.shape}")
        if "MASK" in hdul:
            mask_data = np.asarray(hdul["MASK"].data)
            spatial_mask = np.isfinite(mask_data).any(axis=0) & (
                np.nanmin(mask_data, axis=0) == 0
            )
        else:
            spatial_mask = np.isfinite(cube).any(axis=0)
        signal_mask = np.nanmax(np.abs(cube), axis=0) > 0
        base_valid_mask = spatial_mask & signal_mask
        signal_map = np.nanmean(np.abs(cube), axis=0).astype(np.float32)

        pixel_scale_arcsec = 0.5
        if "CD1_1" in header:
            pixel_scale_arcsec = abs(float(header["CD1_1"])) * 3600.0
        elif "CDELT1" in header:
            pixel_scale_arcsec = abs(float(header["CDELT1"])) * 3600.0
        kpc_per_arcsec = float(header.get("KPCSEC", 1.0))
        psf = float(header.get("PSF", 1.43))

        return CubeGeometry(
            shape=(cube.shape[1], cube.shape[2]),
            base_valid_mask=np.asarray(base_valid_mask, dtype=bool),
            signal_map=signal_map,
            pixel_scale_arcsec=pixel_scale_arcsec,
            kpc_per_arcsec=kpc_per_arcsec,
            psf_fwhm_arcsec=psf,
            redshift=float(header.get("REDSHIFT", 0.0)),
            fov_arcsec=float(header.get("FOV", 0.0)),
            wavelength_start=float(header.get("CRVAL3", 0.0)),
            wavelength_step=float(header.get("CDELT3", 1.0)),
            header_summary={
                "shape": f"{cube.shape}",
                "psf": psf,
                "pixel_scale_arcsec": pixel_scale_arcsec,
                "kpc_per_arcsec": kpc_per_arcsec,
                "ifucon": str(header.get("IFUCON", "")).strip(),
                "cam": [float(header.get(k, 0.0)) for k in ("CAMX", "CAMY", "CAMZ")],
                "valid_pixels": int(np.count_nonzero(base_valid_mask)),
            },
        )


def load_cube_flux(cube_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (flux, error, wavelengths)."""
    with fits.open(cube_path) as hdul:
        flux = np.asarray(hdul[0].data, dtype=np.float32)
        error = (
            np.asarray(hdul["ERROR"].data, dtype=np.float32)
            if "ERROR" in hdul
            else np.zeros_like(flux)
        )
        header = hdul[0].header
        n_wave = flux.shape[0]
        wave = float(header.get("CRVAL3", 0.0)) + np.arange(n_wave) * float(
            header.get("CDELT3", 1.0)
        )
    return flux, error, wave


def view_definition_from_cube(
    cube_path: str | Path, view: int, repeat_count: int
) -> ViewDefinition:
    geom = load_cube_geometry(cube_path)
    vec = view_vector_from_index(view, repeat_count)
    return ViewDefinition(
        view_id=view,
        view_vector=(float(vec[0]), float(vec[1]), float(vec[2])),
        grid_shape=geom.shape,
        spaxel_scale_arcsec=geom.pixel_scale_arcsec,
        kpc_per_arcsec=geom.kpc_per_arcsec,
        fwhm_psf_arcsec=geom.psf_fwhm_arcsec,
    )


def load_pipe3d_maps(maps_path: str | Path) -> dict[str, np.ndarray]:
    """Extrae los mapas 2D de pyPipe3D usados por el Packer y el QA."""
    with fits.open(maps_path) as hdul:
        ssp = np.asarray(hdul["SSP_pyPipe3D_REC"].data, dtype=np.float64)
        if ssp.ndim != 3:
            raise ValueError(f"SSP_pyPipe3D_REC con shape inesperado {ssp.shape}")
        out = {name: ssp[idx].astype(np.float32) for name, idx in PIPE3D_SSP_CHANNELS.items()}
        if "INTRINSIC_ASSIGNED" in hdul:
            out["intrinsic_assigned"] = np.asarray(hdul["INTRINSIC_ASSIGNED"].data, dtype=np.float32)
    return out
