from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal
from urllib.request import urlopen

import numpy as np
from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - pydantic v1 compatibility
    ConfigDict = None

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy import units as u
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter


BAND_NAMES = ("g", "r", "i")
C_ANGSTROM_PER_S = 2.99792458e18
FNU_AB_CGS = 3631.0e-23
SDSS_PIXEL_SCALE_ARCSEC = 0.396
IFU_DIAMETER_ARCSEC = {
    19: 12.0,
    37: 17.0,
    61: 22.0,
    91: 27.0,
    127: 32.0,
}


class ProviderModel(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(arbitrary_types_allowed=True)
    else:  # pragma: no cover - pydantic v1 compatibility
        class Config:
            arbitrary_types_allowed = True


class ImageProviderConfig(ProviderModel):
    filter_set: tuple[str, ...] = ("sdss2010-g", "sdss2010-r", "sdss2010-i")
    output_shape: tuple[int, int] | None = (69, 69)
    output_dtype: str = "float32"
    output_unit: Literal["nanomaggie", "ab_flux", "ab_mag_arcsec2"] = "nanomaggie"
    add_synthetic_noise: bool = False
    noise_sigma_relative: float = 0.0
    noise_seed: int | None = None
    sdss_cutout_size_arcsec: float = 80.0
    sdss_source: Literal["sas", "skyserver", "astroquery"] = "astroquery"
    sdss_data_release: str = "DR17"
    on_missing_band: Literal["skip", "interpolate", "fail"] = "fail"
    on_wcs_failure: Literal["fallback_synthesis", "fail"] = "fallback_synthesis"
    match_psf_to_manga: bool = False
    manga_psf_fwhm_arcsec: float = 2.5


class ImageProviderInput(ProviderModel):
    mode: Literal["mangia", "manga"]
    galaxy_id: str
    cube_path: Path
    drpall_row: dict[str, Any] | None = None
    cache_dir: Path | None = None
    view_id: int | None = None
    config: ImageProviderConfig = Field(default_factory=ImageProviderConfig)


class ProvidedImage(ProviderModel):
    galaxy_id: str
    view_id: int | None = None
    image: np.ndarray
    band_names: list[str]
    unit: str
    source: Literal["synthesized", "sdss_real"]
    wcs_aligned: bool
    fwhm_psf_arcsec: float | None = None
    n_bands_imputed: int = 0

    def metadata(self) -> dict[str, Any]:
        return {
            "galaxy_id": self.galaxy_id,
            "view_id": self.view_id,
            "band_names": self.band_names,
            "unit": self.unit,
            "source": self.source,
            "wcs_aligned": self.wcs_aligned,
            "fwhm_psf_arcsec": self.fwhm_psf_arcsec,
            "n_bands_imputed": self.n_bands_imputed,
        }


@dataclass(slots=True)
class CatalogResult:
    galaxy_id: str
    view_id: int | None
    mode: str
    status: str
    source: str
    fwhm_psf: float | None
    n_bands_imputed: int
    wcs_aligned: bool
    output_path: str
    message: str = ""

    def as_dict(self) -> dict[str, str | int | float | bool | None]:
        return {
            "galaxy_id": self.galaxy_id,
            "view_id": "" if self.view_id is None else self.view_id,
            "mode": self.mode,
            "status": self.status,
            "source": self.source,
            "fwhm_psf": "" if self.fwhm_psf is None else self.fwhm_psf,
            "n_bands_imputed": self.n_bands_imputed,
            "wcs_aligned": self.wcs_aligned,
            "output_path": self.output_path,
            "message": self.message,
        }


@dataclass(slots=True)
class CatalogReport:
    rows: list[CatalogResult]
    manifest_path: Path | None = None

    @property
    def n_ok(self) -> int:
        return sum(row.status == "ok" for row in self.rows)

    @property
    def n_fallback(self) -> int:
        return sum(row.status == "fallback" for row in self.rows)

    @property
    def n_failed(self) -> int:
        return sum(row.status == "failed" for row in self.rows)

    @property
    def n_skipped(self) -> int:
        return sum(row.status == "skipped" for row in self.rows)

    def summary(self) -> dict[str, int | str | None]:
        return {
            "total": len(self.rows),
            "ok": self.n_ok,
            "fallback": self.n_fallback,
            "failed": self.n_failed,
            "skipped": self.n_skipped,
            "manifest_path": None if self.manifest_path is None else str(self.manifest_path),
        }


class ImageProviderSkip(RuntimeError):
    """Raised when config asks to skip an entry."""


def get_named_hdu(hdul: fits.HDUList, name: str) -> fits.ImageHDU | fits.PrimaryHDU | None:
    upper = name.upper()
    for hdu in hdul:
        if hdu.name.upper() == upper:
            return hdu
    return None


def infer_wave_from_header(header: fits.Header, n_wave: int, axis: int = 3) -> np.ndarray | None:
    crval = header.get(f"CRVAL{axis}")
    crpix = float(header.get(f"CRPIX{axis}", 1.0))
    cdelt = header.get(f"CDELT{axis}", header.get(f"CD{axis}_{axis}"))
    if crval is None or cdelt is None:
        return None
    pixels = np.arange(n_wave, dtype=np.float64) + 1.0
    values = float(crval) + (pixels - crpix) * float(cdelt)
    ctype = str(header.get(f"CTYPE{axis}", "")).upper()
    if "LOG" in ctype:
        return np.power(10.0, values)
    return values


def read_ifu_cube(cube_path: str | Path) -> tuple[np.ndarray, np.ndarray, fits.Header]:
    with fits.open(cube_path, memmap=False) as hdul:
        flux_hdu = get_named_hdu(hdul, "FLUX")
        if flux_hdu is None:
            flux_hdu = hdul[0]
        if flux_hdu.data is None:
            raise ValueError(f"No flux data found in {cube_path}")

        flux = np.asarray(flux_hdu.data, dtype=np.float32)
        if flux.ndim != 3:
            raise ValueError(f"Expected 3D IFU cube in {cube_path}, found shape={flux.shape}")

        wave_hdu = get_named_hdu(hdul, "WAVE")
        if wave_hdu is not None and wave_hdu.data is not None:
            wave = np.asarray(wave_hdu.data, dtype=np.float64).reshape(-1)
        else:
            wave = infer_wave_from_header(flux_hdu.header, flux.shape[0], axis=3)
        if wave is None:
            raise ValueError(f"Could not infer wavelength axis for {cube_path}")
        if len(wave) != flux.shape[0]:
            raise ValueError(f"Wavelength axis length {len(wave)} does not match cube {flux.shape[0]}")

        return flux, wave, flux_hdu.header.copy()


def read_ifu_spatial_contract(cube_path: str | Path) -> tuple[tuple[int, int], fits.Header]:
    with fits.open(cube_path, memmap=False) as hdul:
        flux_hdu = get_named_hdu(hdul, "FLUX")
        if flux_hdu is None:
            flux_hdu = hdul[0]
        header = flux_hdu.header.copy()
        naxis1 = header.get("NAXIS1")
        naxis2 = header.get("NAXIS2")
        if naxis1 is None or naxis2 is None:
            if flux_hdu.data is None:
                raise ValueError(f"No spatial contract found in {cube_path}")
            shape = tuple(flux_hdu.data.shape[1:])
        else:
            shape = (int(naxis2), int(naxis1))
        return shape, header


def infer_flux_cgs_scale(header: fits.Header) -> float:
    unit_text = f"{header.get('BUNIT', '')} {header.get('UNITS', '')}".upper()
    if "1E-16" in unit_text or "10^-16" in unit_text:
        return 1e-16
    if "1E-17" in unit_text or "10^-17" in unit_text:
        return 1e-17
    return 1e-17


def pixel_area_arcsec2(header: fits.Header) -> float:
    scale_x = abs(float(header.get("CD1_1", header.get("CDELT1", 0.5 / 3600.0)))) * 3600.0
    scale_y = abs(float(header.get("CD2_2", header.get("CDELT2", 0.5 / 3600.0)))) * 3600.0
    return max(scale_x * scale_y, 1e-12)


def sanitize_galaxy_id(galaxy_id: str) -> str:
    return galaxy_id.replace("/", "_").replace("\\", "_").replace(" ", "_")


def strip_cube_suffix(path: Path) -> str:
    name = path.name
    for suffix in (".cube.fits.gz", ".cube.fits", ".fits.gz", ".fits"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def infer_view_id(galaxy_id: str) -> int | None:
    parts = galaxy_id.split("-")
    if len(parts) >= 5:
        try:
            return int(parts[-2])
        except ValueError:
            return None
    return None


def center_pad_or_crop(image: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = output_shape
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid output_shape={output_shape}")

    result = image
    _, h, w = result.shape
    if h > target_h:
        start = (h - target_h) // 2
        result = result[:, start : start + target_h, :]
        h = target_h
    if w > target_w:
        start = (w - target_w) // 2
        result = result[:, :, start : start + target_w]
        w = target_w

    pad_h = target_h - h
    pad_w = target_w - w
    if pad_h > 0 or pad_w > 0:
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        result = np.pad(result, ((0, 0), (top, bottom), (left, right)), mode="constant")
    return result


def resolve_output_shape(image: np.ndarray, output_shape: tuple[int, int] | None) -> np.ndarray:
    if output_shape is None:
        return image
    return center_pad_or_crop(image, output_shape)


def output_path_for(output_dir: str | Path, provided: ProvidedImage) -> Path:
    output_dir = Path(output_dir)
    safe_id = sanitize_galaxy_id(provided.galaxy_id)
    if provided.view_id is None:
        filename = f"{safe_id}.npz"
    else:
        filename = f"{safe_id}_v{provided.view_id}.npz"
    return output_dir / filename


def save_provided_image(provided: ProvidedImage, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        image=provided.image,
        band_names=np.asarray(provided.band_names),
        metadata=json.dumps(provided.metadata(), sort_keys=True),
    )
    return output_path


def _quantity_values(values: Any) -> np.ndarray:
    if hasattr(values, "to_value"):
        return np.asarray(values.to_value(u.AA), dtype=np.float64)
    if hasattr(values, "value"):
        return np.asarray(values.value, dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def _load_speclite_filters(filter_set: tuple[str, ...]) -> list[Any]:
    try:
        from speclite.filters import load_filters
    except ImportError as exc:  # pragma: no cover - exercised in unprepared envs
        raise RuntimeError(
            "speclite is required for synthetic photometry. Install requirements.txt first."
        ) from exc
    return list(load_filters(*filter_set))


def _filter_band_name(filter_name: str) -> str:
    tail = filter_name.split("-")[-1]
    return tail[-1].lower()


def _nanomaggies_to_unit(image_nanomaggies: np.ndarray, unit: str, header: fits.Header | None = None) -> np.ndarray:
    if unit == "nanomaggie":
        return image_nanomaggies
    if unit == "ab_flux":
        return image_nanomaggies / 1e9
    if unit == "ab_mag_arcsec2":
        area = 1.0 if header is None else pixel_area_arcsec2(header)
        maggies_per_arcsec2 = np.clip(image_nanomaggies / 1e9 / area, 1e-30, None)
        return -2.5 * np.log10(maggies_per_arcsec2)
    raise ValueError(f"Unsupported output_unit={unit}")


def _trapz(values: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, x, axis=axis)
    return np.trapz(values, x, axis=axis)


class ImageProvider:
    def provide(self, provider_input: ImageProviderInput) -> ProvidedImage:
        if provider_input.mode == "mangia":
            return self.provide_mangia(provider_input)
        if provider_input.mode == "manga":
            return self.provide_manga(provider_input)
        raise ValueError(f"Unsupported mode={provider_input.mode}")

    def provide_mangia(self, provider_input: ImageProviderInput, *, wcs_aligned: bool = True) -> ProvidedImage:
        config = provider_input.config
        flux, wave, header = read_ifu_cube(provider_input.cube_path)
        image = self.synthesize_from_cube(flux, wave, header, config)
        image = resolve_output_shape(image, config.output_shape).astype(config.output_dtype, copy=False)
        return ProvidedImage(
            galaxy_id=provider_input.galaxy_id,
            view_id=provider_input.view_id,
            image=image,
            band_names=list(BAND_NAMES),
            unit=config.output_unit,
            source="synthesized",
            wcs_aligned=wcs_aligned,
            fwhm_psf_arcsec=None,
            n_bands_imputed=0,
        )

    def synthesize_from_cube(
        self,
        flux: np.ndarray,
        wave: np.ndarray,
        header: fits.Header,
        config: ImageProviderConfig,
    ) -> np.ndarray:
        filters = _load_speclite_filters(config.filter_set)
        flux_cgs = np.nan_to_num(np.asarray(flux, dtype=np.float64), nan=0.0) * infer_flux_cgs_scale(header)
        wave = np.asarray(wave, dtype=np.float64)
        if not np.all(np.diff(wave) > 0):
            order = np.argsort(wave)
            wave = wave[order]
            flux_cgs = flux_cgs[order]

        bands: list[np.ndarray] = []
        for filt in filters:
            filt_wave = _quantity_values(getattr(filt, "wavelength"))
            response = np.asarray(getattr(filt, "response"), dtype=np.float64)
            response_on_cube = np.interp(wave, filt_wave, response, left=0.0, right=0.0)
            overlap = response_on_cube > 0
            if np.count_nonzero(overlap) < 2:
                raise ValueError(f"Filter {getattr(filt, 'name', filt)} does not overlap cube wavelength range")

            denominator = C_ANGSTROM_PER_S * FNU_AB_CGS * _trapz(response_on_cube / wave, wave)
            if denominator <= 0:
                raise ValueError(f"Invalid denominator for filter {getattr(filt, 'name', filt)}")
            weighted_flux = flux_cgs * (response_on_cube * wave)[:, None, None]
            maggies = _trapz(weighted_flux, wave, axis=0) / denominator
            nanomaggies = maggies * 1e9
            bands.append(nanomaggies.astype(np.float32))

        image_nanomaggies = np.stack(bands, axis=0)
        image_nanomaggies = np.where(np.isfinite(image_nanomaggies), image_nanomaggies, 0.0)
        image_nanomaggies = np.clip(image_nanomaggies, 0.0, None)

        if config.add_synthetic_noise and config.noise_sigma_relative > 0:
            rng = np.random.default_rng(config.noise_seed)
            sigma = float(config.noise_sigma_relative) * image_nanomaggies
            image_nanomaggies = np.clip(image_nanomaggies + rng.normal(0.0, sigma), 0.0, None)

        return _nanomaggies_to_unit(image_nanomaggies, config.output_unit, header)

    def provide_manga(self, provider_input: ImageProviderInput) -> ProvidedImage:
        config = provider_input.config
        if provider_input.drpall_row is None:
            raise ValueError("drpall_row is required in manga mode")
        try:
            return self._provide_manga_real(provider_input)
        except ImageProviderSkip:
            raise
        except Exception:
            if config.on_wcs_failure != "fallback_synthesis":
                raise
            return self.provide_mangia(provider_input, wcs_aligned=False)

    def _provide_manga_real(self, provider_input: ImageProviderInput) -> ProvidedImage:
        config = provider_input.config
        row = provider_input.drpall_row or {}
        plateifu = str(_row_get(row, "plateifu", "PLATEIFU", default=provider_input.galaxy_id))
        ra = float(_row_get(row, "objra", "ra", "OBJRA", "RA"))
        dec = float(_row_get(row, "objdec", "dec", "OBJDEC", "DEC"))
        ifusize = _infer_ifusize(row, plateifu)
        cutout_size = max(float(config.sdss_cutout_size_arcsec), IFU_DIAMETER_ARCSEC.get(ifusize, 32.0) * 2.5)

        target_shape, cube_header = read_ifu_spatial_contract(provider_input.cube_path)
        target_wcs = WCS(cube_header).celestial

        cache_dir = provider_input.cache_dir or Path("cache/sdss")
        band_images: list[np.ndarray | None] = []
        fwhm_values: list[float] = []
        for filter_name in config.filter_set:
            band = _filter_band_name(filter_name)
            try:
                frame_path = self._get_sdss_band_frame(
                    plateifu=plateifu,
                    ra=ra,
                    dec=dec,
                    band=band,
                    cutout_size_arcsec=cutout_size,
                    cache_dir=cache_dir,
                    source=config.sdss_source,
                    data_release=config.sdss_data_release,
                )
                aligned_cache = _aligned_cache_path(frame_path, target_wcs, target_shape)
                if aligned_cache.exists():
                    projected, fwhm = _read_aligned_cache(aligned_cache)
                else:
                    data, frame_wcs, fwhm = self._read_sdss_frame(frame_path)
                    projected, _footprint = _reproject_to_target(data, frame_wcs, target_wcs, target_shape)
                    _write_aligned_cache(aligned_cache, projected, fwhm)
                band_images.append(np.nan_to_num(projected, nan=0.0).astype(np.float32))
                if fwhm is not None:
                    fwhm_values.append(float(fwhm))
            except Exception:
                band_images.append(None)

        imputed = self._fill_missing_bands(band_images, config)
        if any(band is None for band in band_images):
            raise RuntimeError("Missing SDSS band after imputation")

        image_nanomaggies = np.stack([band for band in band_images if band is not None], axis=0)
        image_nanomaggies = np.clip(np.nan_to_num(image_nanomaggies, nan=0.0), 0.0, None)
        if config.match_psf_to_manga and fwhm_values:
            sdss_fwhm = float(np.nanmedian(fwhm_values))
            sigma_arcsec = math.sqrt(max(config.manga_psf_fwhm_arcsec**2 - sdss_fwhm**2, 0.0)) / 2.355
            sigma_pixels = sigma_arcsec / max(_pixel_scale_from_wcs(target_wcs), 1e-6)
            if sigma_pixels > 0:
                image_nanomaggies = gaussian_filter(image_nanomaggies, sigma=(0.0, sigma_pixels, sigma_pixels))

        image = _nanomaggies_to_unit(image_nanomaggies, config.output_unit, cube_header)
        image = resolve_output_shape(image, config.output_shape).astype(config.output_dtype, copy=False)
        fwhm = float(np.nanmedian(fwhm_values)) if fwhm_values else None
        return ProvidedImage(
            galaxy_id=provider_input.galaxy_id,
            view_id=None,
            image=image,
            band_names=list(BAND_NAMES),
            unit=config.output_unit,
            source="sdss_real",
            wcs_aligned=True,
            fwhm_psf_arcsec=fwhm,
            n_bands_imputed=imputed,
        )

    def _fill_missing_bands(self, band_images: list[np.ndarray | None], config: ImageProviderConfig) -> int:
        missing = [idx for idx, band in enumerate(band_images) if band is None]
        if not missing:
            return 0
        if config.on_missing_band == "fail":
            raise RuntimeError(f"Missing SDSS bands: {[BAND_NAMES[idx] for idx in missing]}")
        if config.on_missing_band == "skip":
            raise ImageProviderSkip(f"Missing SDSS bands: {[BAND_NAMES[idx] for idx in missing]}")
        for idx in missing:
            left = next((band_images[j] for j in range(idx - 1, -1, -1) if band_images[j] is not None), None)
            right = next((band_images[j] for j in range(idx + 1, len(band_images)) if band_images[j] is not None), None)
            if left is not None and right is not None:
                band_images[idx] = 0.5 * (left + right)
            elif left is not None:
                band_images[idx] = left.copy()
            elif right is not None:
                band_images[idx] = right.copy()
            else:
                raise RuntimeError("Cannot interpolate SDSS bands when all bands are missing")
        return len(missing)

    def _get_sdss_band_frame(
        self,
        *,
        plateifu: str,
        ra: float,
        dec: float,
        band: str,
        cutout_size_arcsec: float,
        cache_dir: str | Path,
        source: str,
        data_release: str,
    ) -> Path:
        plate = plateifu.split("-", 1)[0]
        destination = Path(cache_dir) / plate / f"{plateifu}_{band}.fits"
        if _valid_fits(destination):
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()

        if source == "astroquery":
            self._download_with_astroquery(destination, ra, dec, band, cutout_size_arcsec, data_release)
        elif source == "skyserver":
            self._download_with_skyserver(destination, ra, dec, band, cutout_size_arcsec, data_release)
        elif source == "sas":
            self._download_with_astroquery(destination, ra, dec, band, cutout_size_arcsec, data_release)
        else:
            raise ValueError(f"Unsupported SDSS source={source}")
        return destination

    def _download_with_astroquery(
        self,
        destination: Path,
        ra: float,
        dec: float,
        band: str,
        cutout_size_arcsec: float,
        data_release: str,
    ) -> None:
        try:
            from astroquery.sdss import SDSS
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("astroquery is required for SDSS downloads") from exc

        release = _release_number(data_release)
        pos = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        radius = (cutout_size_arcsec / 2.0) * u.arcsec
        matches = SDSS.query_region(pos, radius=radius, data_release=release)
        if matches is None or len(matches) == 0:
            raise RuntimeError(f"No SDSS frame found around ra={ra}, dec={dec}")
        frames = SDSS.get_images(matches=matches[:1], band=band, data_release=release)
        if not frames:
            raise RuntimeError(f"No SDSS image returned for band={band}")

        frame_hdul = frames[0]
        try:
            data = np.asarray(frame_hdul[0].data, dtype=np.float32)
            wcs = WCS(frame_hdul[0].header).celestial
            cutout = Cutout2D(
                data,
                position=pos,
                size=(cutout_size_arcsec * u.arcsec, cutout_size_arcsec * u.arcsec),
                wcs=wcs,
                mode="partial",
                fill_value=np.nan,
            )
            header = cutout.wcs.to_header()
            for key in ("SEEING", "PSF_FWHM", "PSFFWHM"):
                if key in frame_hdul[0].header:
                    header[key] = frame_hdul[0].header[key]
            fits.PrimaryHDU(np.asarray(cutout.data, dtype=np.float32), header=header).writeto(destination)
        finally:
            frame_hdul.close()

    def _download_with_skyserver(
        self,
        destination: Path,
        ra: float,
        dec: float,
        band: str,
        cutout_size_arcsec: float,
        data_release: str,
    ) -> None:
        release = data_release.lower()
        size_pix = max(int(round(cutout_size_arcsec / SDSS_PIXEL_SCALE_ARCSEC)), 32)
        url = (
            f"https://skyserver.sdss.org/{release}/SkyServerWS/ImgCutout/getfits"
            f"?ra={ra}&dec={dec}&scale={SDSS_PIXEL_SCALE_ARCSEC}"
            f"&width={size_pix}&height={size_pix}&filter={band}"
        )
        with urlopen(url, timeout=120) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)

    def _read_sdss_frame(self, frame_path: str | Path) -> tuple[np.ndarray, WCS, float | None]:
        with fits.open(frame_path, memmap=False) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float32)
            header = hdul[0].header.copy()
        wcs = WCS(header).celestial
        fwhm = None
        for key in ("SEEING", "PSF_FWHM", "PSFFWHM"):
            if key in header:
                try:
                    fwhm = float(header[key])
                    break
                except (TypeError, ValueError):
                    pass
        return data, wcs, fwhm


class CatalogImageBuilder:
    def __init__(self, provider: ImageProvider | None = None) -> None:
        self.provider = provider or ImageProvider()

    def build_mangia_catalog(
        self,
        mangia_root: str | Path,
        output_dir: str | Path,
        config: ImageProviderConfig,
        n_workers: int = 1,
        manifest_path: str | Path | None = None,
        pattern: str = "*.cube.fits.gz",
        recursive: bool = False,
        limit: int | None = None,
        skip_existing: bool = False,
    ) -> CatalogReport:
        mangia_root = Path(mangia_root)
        output_dir = Path(output_dir)
        entries = discover_mangia_cubes(mangia_root, pattern=pattern, recursive=recursive)
        if limit is not None:
            entries = entries[:limit]
        rows = self._run_parallel(
            [
                lambda entry=entry: self._process_mangia_one(
                    entry,
                    output_dir,
                    config,
                    skip_existing=skip_existing,
                )
                for entry in entries
            ],
            n_workers=n_workers,
        )
        manifest = Path(manifest_path) if manifest_path else output_dir / "manifest.csv"
        write_manifest(manifest, rows)
        return CatalogReport(rows=rows, manifest_path=manifest)

    def build_manga_catalog(
        self,
        drpall_path: str | Path,
        cubes_dir: str | Path,
        output_dir: str | Path,
        config: ImageProviderConfig,
        n_workers: int = 1,
        cache_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        limit: int | None = None,
    ) -> CatalogReport:
        drpall = Table.read(drpall_path)
        rows_to_process = list(drpall)
        if limit is not None:
            rows_to_process = rows_to_process[:limit]
        output_dir = Path(output_dir)
        cubes_dir = Path(cubes_dir)
        cache = Path(cache_dir) if cache_dir is not None else output_dir / "sdss_cache"
        jobs = [
            lambda row=row: self._process_manga_one(row, cubes_dir, output_dir, cache, config)
            for row in rows_to_process
        ]
        rows = self._run_parallel(jobs, n_workers=n_workers)
        manifest = Path(manifest_path) if manifest_path else output_dir / "manifest.csv"
        write_manifest(manifest, rows)
        return CatalogReport(rows=rows, manifest_path=manifest)

    def _run_parallel(self, jobs: list[Any], n_workers: int) -> list[CatalogResult]:
        if n_workers <= 1:
            return [job() for job in jobs]
        results: list[CatalogResult | None] = [None] * len(jobs)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(job): index for index, job in enumerate(jobs)}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return [result for result in results if result is not None]

    def _process_mangia_one(
        self,
        cube_path: Path,
        output_dir: Path,
        config: ImageProviderConfig,
        skip_existing: bool = False,
    ) -> CatalogResult:
        galaxy_id = strip_cube_suffix(cube_path)
        view_id = infer_view_id(galaxy_id)
        safe_id = sanitize_galaxy_id(galaxy_id)
        expected_path = output_dir / (f"{safe_id}_v{view_id}.npz" if view_id is not None else f"{safe_id}.npz")
        if skip_existing and expected_path.exists():
            return CatalogResult(
                galaxy_id,
                view_id,
                "mangia",
                "skipped",
                "existing",
                None,
                0,
                True,
                str(expected_path),
                "Output already exists",
            )
        try:
            provided = self.provider.provide(
                ImageProviderInput(
                    mode="mangia",
                    galaxy_id=galaxy_id,
                    view_id=view_id,
                    cube_path=cube_path,
                    config=config,
                )
            )
            out_path = save_provided_image(provided, output_path_for(output_dir, provided))
            return _result_from_provided(provided, "mangia", "ok", out_path)
        except Exception as exc:
            return CatalogResult(galaxy_id, view_id, "mangia", "failed", "", None, 0, False, "", str(exc))

    def _process_manga_one(
        self,
        row: Any,
        cubes_dir: Path,
        output_dir: Path,
        cache_dir: Path,
        config: ImageProviderConfig,
    ) -> CatalogResult:
        row_dict = _row_to_dict(row)
        plateifu = str(_row_get(row_dict, "plateifu", "PLATEIFU"))
        cube_path = find_manga_cube(cubes_dir, plateifu)
        if cube_path is None:
            return CatalogResult(plateifu, None, "manga", "failed", "", None, 0, False, "", "Cube not found")
        try:
            provided = self.provider.provide(
                ImageProviderInput(
                    mode="manga",
                    galaxy_id=plateifu,
                    cube_path=cube_path,
                    drpall_row=row_dict,
                    cache_dir=cache_dir,
                    config=config,
                )
            )
            status = "fallback" if provided.source == "synthesized" else "ok"
            out_path = save_provided_image(provided, output_path_for(output_dir, provided))
            return _result_from_provided(provided, "manga", status, out_path)
        except ImageProviderSkip as exc:
            return CatalogResult(plateifu, None, "manga", "skipped", "", None, 0, False, "", str(exc))
        except Exception as exc:
            return CatalogResult(plateifu, None, "manga", "failed", "", None, 0, False, "", str(exc))


def _result_from_provided(provided: ProvidedImage, mode: str, status: str, output_path: Path) -> CatalogResult:
    return CatalogResult(
        galaxy_id=provided.galaxy_id,
        view_id=provided.view_id,
        mode=mode,
        status=status,
        source=provided.source,
        fwhm_psf=provided.fwhm_psf_arcsec,
        n_bands_imputed=provided.n_bands_imputed,
        wcs_aligned=provided.wcs_aligned,
        output_path=str(output_path),
    )


def write_manifest(path: str | Path, rows: Iterable[CatalogResult]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "galaxy_id",
        "view_id",
        "mode",
        "status",
        "source",
        "fwhm_psf",
        "n_bands_imputed",
        "wcs_aligned",
        "output_path",
        "message",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
    return path


def find_manga_cube(cubes_dir: str | Path, plateifu: str) -> Path | None:
    cubes_dir = Path(cubes_dir)
    patterns = [
        f"manga-{plateifu}-LOGCUBE.fits.gz",
        f"manga-{plateifu}-LINCUBE.fits.gz",
        f"*{plateifu}*LOGCUBE*.fits.gz",
        f"*{plateifu}*.fits.gz",
    ]
    for pattern in patterns:
        matches = sorted(cubes_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def discover_mangia_cubes(
    mangia_root: str | Path,
    pattern: str = "*.cube.fits.gz",
    recursive: bool = False,
) -> list[Path]:
    mangia_root = Path(mangia_root)
    if mangia_root.is_file():
        candidates = [mangia_root]
    else:
        globber = mangia_root.rglob if recursive else mangia_root.glob
        candidates = sorted(globber(pattern))
    return [
        path
        for path in candidates
        if path.is_file()
        and "cube_val" not in path.name
        and not path.name.endswith(".npz")
        and (path.name.endswith(".fits") or path.name.endswith(".fits.gz"))
    ]


def bootstrap_pilot_data(
    source_root: str | Path = "/home/andy/pythonprojects/cubes",
    destination_root: str | Path = "ImagesMangGenerator/data/pilot",
) -> list[Path]:
    source_root = Path(source_root)
    destination_root = Path(destination_root)
    copies = [
        (
            source_root / "TNG50-87-141934-0-127.cube.fits.gz",
            destination_root / "mangia" / "TNG50-87-141934-0-127.cube.fits.gz",
        ),
        (
            source_root / "TNG50-87-141934-0-127.manga_logcube_74x74.fits.gz",
            destination_root / "mangia" / "TNG50-87-141934-0-127.manga_logcube_74x74.fits.gz",
        ),
        (
            source_root / "manga_compare_project/data/manga-7443-12703-LOGCUBE.fits.gz",
            destination_root / "manga" / "manga-7443-12703-LOGCUBE.fits.gz",
        ),
    ]
    written: list[Path] = []
    for src, dst in copies:
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        written.append(dst)
    return written


def _row_to_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    if hasattr(row, "colnames"):
        return {name: row[name] for name in row.colnames}
    if hasattr(row, "dtype") and getattr(row.dtype, "names", None):
        return {name: row[name] for name in row.dtype.names}
    return dict(row)


def _row_get(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    lower = {key.lower(): key for key in row.keys()}
    for name in names:
        if name in row:
            value = row[name]
        elif name.lower() in lower:
            value = row[lower[name.lower()]]
        else:
            continue
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return value
    if default is not None:
        return default
    raise KeyError(f"Missing row field. Tried: {names}")


def _infer_ifusize(row: dict[str, Any], plateifu: str) -> int:
    for name in ("ifudesignsize", "IFUDESIGNSIZE", "ifudsgn", "IFUDSGN", "ifusize"):
        try:
            value = int(_row_get(row, name))
            if value in IFU_DIAMETER_ARCSEC:
                return value
        except (KeyError, TypeError, ValueError):
            pass
    if "-" in plateifu:
        ifu = plateifu.split("-", 1)[1]
        for size in sorted(IFU_DIAMETER_ARCSEC, reverse=True):
            if ifu.startswith(str(size)):
                return size
    return 127


def _release_number(data_release: str) -> int:
    digits = "".join(ch for ch in data_release if ch.isdigit())
    return int(digits or 17)


def _valid_fits(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with fits.open(path, memmap=False) as hdul:
            return hdul[0].data is not None
    except Exception:
        return False


def _aligned_cache_path(frame_path: Path, target_wcs: WCS, target_shape: tuple[int, int]) -> Path:
    header = target_wcs.to_header(relax=True)
    fingerprint = hashlib.sha1(
        (repr(tuple(target_shape)) + "\n" + header.tostring(sep="\n", endcard=False)).encode("utf-8")
    ).hexdigest()[:16]
    return frame_path.with_name(f"{frame_path.stem}_aligned_{fingerprint}.npz")


def _read_aligned_cache(path: Path) -> tuple[np.ndarray, float | None]:
    with np.load(path) as payload:
        image = np.asarray(payload["image"], dtype=np.float32)
        fwhm_raw = float(payload["fwhm_psf_arcsec"])
    fwhm = None if not np.isfinite(fwhm_raw) else fwhm_raw
    return image, fwhm


def _write_aligned_cache(path: Path, image: np.ndarray, fwhm_psf_arcsec: float | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        image=np.asarray(image, dtype=np.float32),
        fwhm_psf_arcsec=np.nan if fwhm_psf_arcsec is None else float(fwhm_psf_arcsec),
    )


def _reproject_to_target(
    data: np.ndarray,
    source_wcs: WCS,
    target_wcs: WCS,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from reproject import reproject_interp
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("reproject is required for MaNGA image alignment") from exc
    return reproject_interp((data, source_wcs), target_wcs, shape_out=target_shape)


def _pixel_scale_from_wcs(wcs: WCS) -> float:
    try:
        proj = np.asarray(wcs.proj_plane_pixel_scales()) * 3600.0
        return float(np.nanmean(np.abs(proj)))
    except Exception:
        return 0.5
