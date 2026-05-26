from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from .models import KinematicMomentsConfig, TemplateLibrary


C_KMS = 299792.458


@dataclass(slots=True)
class FitGrid:
    fit_index: np.ndarray
    wave_rest_fit: np.ndarray
    lam_gal: np.ndarray
    velscale: float


def log_rebin_compat(
    lam: np.ndarray,
    spec: np.ndarray,
    velscale: float | None = None,
    oversample: int = 1,
    flux: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compatibility wrapper for pPXF log_rebin with NumPy 2.4 scalar rules."""
    from ppxf import ppxf_util

    if velscale is None:
        rebinned, ln_lam, out_velscale = ppxf_util.log_rebin(
            lam,
            spec,
            velscale=None,
            oversample=oversample,
            flux=flux,
        )
        return rebinned, ln_lam, float(out_velscale)

    lam = np.asarray(lam, dtype=float)
    spec = np.asarray(spec, dtype=float)
    if not np.all(np.diff(lam) > 0):
        raise ValueError("lam must be monotonically increasing")
    n_pix = len(spec)
    if lam.size not in (2, n_pix):
        raise ValueError("lam must be a 2-element range or match the spectral axis")

    if lam.size == 2:
        dlam = np.diff(lam) / (n_pix - 1)
        lim = lam + np.array([-0.5, 0.5]) * dlam
        borders = np.linspace(float(lim[0]), float(lim[1]), n_pix + 1)
    else:
        lim = 1.5 * lam[[0, -1]] - 0.5 * lam[[1, -2]]
        borders = np.hstack([lim[0], (lam[1:] + lam[:-1]) / 2.0, lim[1]])
        dlam = np.diff(borders)

    ln_lim = np.log(lim)
    ln_scale = float(velscale) / C_KMS
    n_out = int(float(np.diff(ln_lim).item()) / ln_scale)
    new_borders = np.exp(float(ln_lim[0]) + ln_scale * np.arange(n_out + 1))

    if lam.size == 2:
        k = ((new_borders - lim[0]) / dlam).clip(0, n_pix - 1).astype(int)
    else:
        k = (np.searchsorted(borders, new_borders) - 1).clip(0, n_pix - 1)

    spec_new = np.add.reduceat((spec.T * dlam).T, k)[:-1]
    spec_new.T[...] *= np.diff(k) > 0
    spec_new.T[...] += np.diff(((new_borders - borders[k])) * spec[k].T)
    if not flux:
        spec_new.T[...] /= np.diff(new_borders)

    ln_lam = 0.5 * np.log(new_borders[1:] * new_borders[:-1])
    return spec_new, ln_lam, float(velscale)


def build_fit_grid(wave_observed: np.ndarray, redshift: float, config: KinematicMomentsConfig) -> FitGrid:
    from ppxf import ppxf_util

    wave_rest = np.asarray(wave_observed, dtype=np.float64) / (1.0 + float(redshift))
    fit_index = np.flatnonzero(
        (wave_rest >= config.wave_range_fit[0]) & (wave_rest <= config.wave_range_fit[1])
    )
    if fit_index.size < config.min_goodpixels:
        observed_min = float(np.nanmin(wave_observed)) if np.size(wave_observed) else float("nan")
        observed_max = float(np.nanmax(wave_observed)) if np.size(wave_observed) else float("nan")
        rest_min = float(np.nanmin(wave_rest)) if np.size(wave_rest) else float("nan")
        rest_max = float(np.nanmax(wave_rest)) if np.size(wave_rest) else float("nan")
        raise ValueError(
            f"Fit range {config.wave_range_fit} leaves only {fit_index.size} wavelength pixels "
            f"(n_wave={len(wave_rest)}, redshift={float(redshift)}, "
            f"observed_range=({observed_min}, {observed_max}), "
            f"rest_range=({rest_min}, {rest_max}))"
        )

    wave_rest_fit = wave_rest[fit_index]
    _, ln_lam_gal, velscale = ppxf_util.log_rebin(wave_rest_fit, np.ones_like(wave_rest_fit))
    return FitGrid(
        fit_index=fit_index,
        wave_rest_fit=wave_rest_fit,
        lam_gal=np.exp(ln_lam_gal),
        velscale=float(velscale),
    )


def _wave_from_template_header(header: fits.Header, n_wave: int) -> np.ndarray:
    crval = float(header["CRVAL1"])
    cdelt = float(header["CDELT1"])
    crpix = float(header.get("CRPIX1", 1.0))
    pixels = np.arange(n_wave, dtype=np.float64) + 1.0
    return crval + (pixels - crpix) * cdelt


def load_mastar_templates(
    template_path: str | Path,
    galaxy_velscale: float,
    config: KinematicMomentsConfig,
) -> TemplateLibrary:
    template_path = Path(template_path).resolve()
    with fits.open(template_path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header.copy()

    if data.ndim != 2:
        raise ValueError(f"Expected 2D MaStar template grid, found shape={data.shape}")
    n_templates, n_wave = data.shape
    wave = _wave_from_template_header(header, n_wave)

    margin = 500.0
    use_wave = (
        (wave >= config.wave_range_fit[0] - margin)
        & (wave <= config.wave_range_fit[1] + margin)
        & np.isfinite(wave)
    )
    if np.count_nonzero(use_wave) < config.min_goodpixels:
        raise ValueError(f"Template wavelength range does not cover fit range {config.wave_range_fit}")

    spectra = data[:, use_wave].T
    wave_use = wave[use_wave]
    if config.max_templates is not None:
        spectra = spectra[:, : int(config.max_templates)]
        n_templates = spectra.shape[1]

    norm_use = (
        (wave_use >= config.template_norm_range[0])
        & (wave_use <= config.template_norm_range[1])
    )
    if not np.any(norm_use):
        norm_use = np.ones_like(wave_use, dtype=bool)

    norm = np.nanmedian(spectra[norm_use], axis=0)
    fallback = np.nanmedian(np.abs(spectra), axis=0)
    norm = np.where(np.isfinite(norm) & (np.abs(norm) > 0), norm, fallback)
    keep = np.isfinite(norm) & (np.abs(norm) > 0)
    if not np.any(keep):
        raise ValueError(f"No usable templates after normalization: {template_path}")
    spectra = spectra[:, keep] / norm[keep]

    template_velscale = float(galaxy_velscale) / int(config.velscale_ratio)
    templates_log, ln_lam_temp, velscale_temp = log_rebin_compat(
        wave_use,
        spectra,
        velscale=template_velscale,
    )
    finite = np.all(np.isfinite(templates_log), axis=0)
    templates_log = templates_log[:, finite]
    if templates_log.shape[1] == 0:
        raise ValueError(f"No finite log-rebinned templates: {template_path}")

    return TemplateLibrary(
        path=template_path,
        templates=np.asarray(templates_log, dtype=np.float64),
        lam_temp=np.exp(ln_lam_temp),
        velscale=float(velscale_temp),
        n_templates=int(templates_log.shape[1]),
    )


def compute_snr(
    flux: np.ndarray,
    error: np.ndarray,
    wave_rest: np.ndarray,
    valid: np.ndarray,
    config: KinematicMomentsConfig,
) -> float:
    use = (
        (wave_rest >= config.snr_window[0])
        & (wave_rest <= config.snr_window[1])
        & valid
        & np.isfinite(flux)
        & np.isfinite(error)
        & (error > 0)
    )
    if not np.any(use):
        return float("nan")
    ratio = flux[use] / error[use]
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        return float("nan")
    return float(np.nanmedian(ratio))


def build_goodpixels(
    lam_gal: np.ndarray,
    valid_log: np.ndarray,
    config: KinematicMomentsConfig,
) -> np.ndarray:
    good = np.asarray(valid_log, dtype=bool).copy()
    good &= np.isfinite(lam_gal)
    for line in config.emission_lines_mask:
        velocity_distance = np.abs(C_KMS * np.log(lam_gal / float(line)))
        good &= velocity_distance > float(config.emission_mask_width_kms)
    return np.flatnonzero(good)


def fit_spaxel(
    flux: np.ndarray,
    error: np.ndarray,
    valid: np.ndarray,
    grid: FitGrid,
    templates: TemplateLibrary,
    config: KinematicMomentsConfig,
) -> tuple[float, float, float, float, float, float, float]:
    from ppxf.ppxf import ppxf

    flux_fit = np.asarray(flux[grid.fit_index], dtype=np.float64)
    err_fit = np.asarray(error[grid.fit_index], dtype=np.float64)
    valid_fit = np.asarray(valid[grid.fit_index], dtype=bool)
    finite = np.isfinite(flux_fit) & np.isfinite(err_fit) & (err_fit > 0)
    valid_fit &= finite

    if np.mean(valid_fit) < config.min_valid_fraction:
        raise ValueError("too few valid wavelength pixels")

    if np.any(valid_fit):
        fill_flux = float(np.nanmedian(flux_fit[valid_fit]))
        fill_err = float(np.nanmedian(err_fit[valid_fit]))
    else:
        raise ValueError("no valid wavelength pixels")
    if not np.isfinite(fill_err) or fill_err <= 0:
        fill_err = 1.0

    flux_clean = np.where(valid_fit, flux_fit, fill_flux)
    err_clean = np.where(valid_fit, err_fit, fill_err * 1.0e6)
    galaxy_log, _, _ = log_rebin_compat(grid.wave_rest_fit, flux_clean, velscale=grid.velscale)
    noise_log, _, _ = log_rebin_compat(grid.wave_rest_fit, err_clean, velscale=grid.velscale)
    valid_log = np.interp(grid.lam_gal, grid.wave_rest_fit, valid_fit.astype(float)) > 0.5

    goodpixels = build_goodpixels(grid.lam_gal, valid_log, config)
    if goodpixels.size < config.min_goodpixels:
        raise ValueError(f"too few goodpixels after masks: {goodpixels.size}")

    norm = float(np.nanmedian(galaxy_log[goodpixels]))
    if not np.isfinite(norm) or abs(norm) <= 0:
        norm = float(np.nanmedian(np.abs(galaxy_log[goodpixels])))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("non-positive spectral normalization")

    galaxy = galaxy_log / norm
    noise = np.abs(noise_log / norm)
    good_noise = np.isfinite(noise[goodpixels]) & (noise[goodpixels] > 0)
    if not np.any(good_noise):
        raise ValueError("non-positive noise vector")
    noise_floor = float(np.nanmedian(noise[goodpixels][good_noise])) * 1.0e-6
    noise = np.where(np.isfinite(noise) & (noise > 0), noise, max(noise_floor, 1.0e-8))

    pp = ppxf(
        templates.templates,
        galaxy,
        noise,
        grid.velscale,
        [config.start_velocity, config.start_sigma],
        moments=config.moments,
        degree=config.degree,
        mdegree=config.mdegree,
        bias=config.bias,
        goodpixels=goodpixels,
        lam=grid.lam_gal,
        lam_temp=templates.lam_temp,
        velscale_ratio=config.velscale_ratio,
        quiet=config.quiet,
        plot=False,
    )
    sol = np.asarray(pp.sol, dtype=np.float64)
    error_vec = np.asarray(getattr(pp, "error", np.full(4, np.nan)), dtype=np.float64)
    if sol.size < 4:
        raise ValueError(f"pPXF returned only {sol.size} moments")
    if error_vec.size < 4:
        error_vec = np.pad(error_vec, (0, 4 - error_vec.size), constant_values=np.nan)
    return (
        float(sol[0]),
        float(sol[1]),
        float(sol[2]),
        float(sol[3]),
        float(error_vec[2]),
        float(error_vec[3]),
        float(pp.chi2),
    )
