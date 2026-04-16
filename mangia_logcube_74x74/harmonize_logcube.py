import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates


def load_config(config_path):
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_named_hdu(hdul, name):
    upper = name.upper()
    for hdu in hdul:
        if hdu.name.upper() == upper:
            return hdu
    return None


def infer_wave_from_header(header, n_wave, axis=3):
    crval = header.get(f"CRVAL{axis}")
    crpix = float(header.get(f"CRPIX{axis}", 1.0))
    cdelt = header.get(f"CDELT{axis}", header.get(f"CD{axis}_{axis}"))
    if crval is None or cdelt is None:
        return None

    pixels = np.arange(n_wave, dtype=np.float64) + 1.0
    ctype = str(header.get(f"CTYPE{axis}", "")).upper()
    if "LOG" in ctype:
        return np.power(10.0, float(crval) + (pixels - crpix) * float(cdelt))
    return float(crval) + (pixels - crpix) * float(cdelt)


def build_selected_header(template, axis=3, extname=None, bunit=None):
    header = fits.Header()
    keep = [
        "BSCALE",
        "BZERO",
        "BUNIT",
        "CTYPE1",
        "CTYPE2",
        "CTYPE3",
        "CUNIT1",
        "CUNIT2",
        "CUNIT3",
        "CRPIX1",
        "CRPIX2",
        "CRPIX3",
        "CRVAL1",
        "CRVAL2",
        "CRVAL3",
        "CD1_1",
        "CD1_2",
        "CD2_1",
        "CD2_2",
        "CD3_3",
        "CDELT1",
        "CDELT2",
        "CDELT3",
        "LONPOLE",
        "LATPOLE",
        "EQUINOX",
    ]
    for key in keep:
        if key in template:
            header[key] = template[key]
    if extname is not None:
        header["EXTNAME"] = extname
    if bunit is not None:
        header["BUNIT"] = bunit
    return header


def read_reference_contract(reference_logcube):
    reference_logcube = Path(reference_logcube).resolve()
    with fits.open(reference_logcube, memmap=False) as hdul:
        flux_hdu = hdul["FLUX"]
        wave_hdu = get_named_hdu(hdul, "WAVE")
        if wave_hdu is not None and wave_hdu.data is not None:
            wave = np.asarray(wave_hdu.data, dtype=np.float64).reshape(-1)
            wave_header = wave_hdu.header.copy()
        else:
            wave = infer_wave_from_header(flux_hdu.header, flux_hdu.data.shape[0], axis=3)
            wave_header = fits.Header()
            wave_header["EXTNAME"] = "WAVE"
            wave_header["BUNIT"] = "Angstrom"

        return {
            "path": reference_logcube,
            "primary_header": hdul[0].header.copy(),
            "flux_header": flux_hdu.header.copy(),
            "ivar_header": hdul["IVAR"].header.copy(),
            "mask_header": hdul["MASK"].header.copy(),
            "wave_header": wave_header,
            "wave": wave,
            "shape": tuple(flux_hdu.data.shape),
            "bunit": str(flux_hdu.header.get("BUNIT", "")),
        }


def read_official_cube(official_cube):
    official_cube = Path(official_cube).resolve()
    with fits.open(official_cube, memmap=False) as hdul:
        flux = np.asarray(hdul[0].data, dtype=np.float32)
        source_header = hdul[0].header.copy()
        error_hdu = get_named_hdu(hdul, "ERROR")
        mask_hdu = get_named_hdu(hdul, "MASK")
        gas_hdu = get_named_hdu(hdul, "GAS")
        error = None if error_hdu is None else np.asarray(error_hdu.data, dtype=np.float32)
        mask = None if mask_hdu is None else np.asarray(mask_hdu.data)
        gas = None if gas_hdu is None else np.asarray(gas_hdu.data, dtype=np.float32)
        wave = infer_wave_from_header(source_header, flux.shape[0], axis=3)
        if wave is None:
            raise ValueError("No pude reconstruir el eje espectral del cubo oficial")

        return {
            "path": official_cube,
            "flux": flux,
            "error": error,
            "mask": mask,
            "gas": gas,
            "wave": wave,
            "header": source_header,
            "units": str(source_header.get("UNITS", "")),
        }


def spectral_resample_cube(data, source_wave, target_wave, fill_value=np.nan, kind="linear"):
    if data is None:
        return None
    source = np.asarray(data, dtype=np.float32)
    original_shape = source.shape
    flattened = source.reshape(original_shape[0], -1)
    interpolator = interp1d(
        source_wave,
        flattened,
        axis=0,
        kind=kind,
        bounds_error=False,
        fill_value=fill_value,
        assume_sorted=True,
        copy=False,
    )
    resampled = interpolator(target_wave)
    return np.asarray(resampled.reshape((len(target_wave),) + original_shape[1:]), dtype=np.float32)


def build_spatial_coordinates(source_header, target_header, flip_x=True):
    cd1_source = abs(float(source_header.get("CD1_1", source_header.get("CDELT1")))) * 3600.0
    cd2_source = abs(float(source_header.get("CD2_2", source_header.get("CDELT2")))) * 3600.0
    cd1_target = abs(float(target_header.get("CD1_1", target_header.get("CDELT1")))) * 3600.0
    cd2_target = abs(float(target_header.get("CD2_2", target_header.get("CDELT2")))) * 3600.0
    crpix1_source = float(source_header["CRPIX1"])
    crpix2_source = float(source_header["CRPIX2"])
    crpix1_target = float(target_header["CRPIX1"])
    crpix2_target = float(target_header["CRPIX2"])
    nx_target = int(target_header["NAXIS1"])
    ny_target = int(target_header["NAXIS2"])

    x_target = (np.arange(nx_target, dtype=np.float64) + 1.0 - crpix1_target) * cd1_target
    if flip_x:
        x_target = -x_target
    y_target = (np.arange(ny_target, dtype=np.float64) + 1.0 - crpix2_target) * cd2_target
    grid_x, grid_y = np.meshgrid(x_target, y_target)

    x_source = crpix1_source - 1.0
    y_source = crpix2_source - 1.0
    if flip_x:
        x_indices = x_source - grid_x / cd1_source
    else:
        x_indices = x_source + grid_x / cd1_source
    y_indices = y_source + grid_y / cd2_source

    return np.stack([y_indices, x_indices])


def spatial_resample_cube(data, coordinates, order, cval):
    if data is None:
        return None
    source = np.asarray(data)
    output = np.empty((source.shape[0],) + coordinates.shape[1:], dtype=np.float32)
    for index in range(source.shape[0]):
        output[index] = map_coordinates(
            source[index],
            coordinates,
            order=order,
            mode="constant",
            cval=cval,
            prefilter=(order > 1),
        )
    return output


def build_primary_header(reference_primary, official_header, official_cube, reference_logcube):
    header = fits.Header()
    for key in ("SIMPLE", "BITPIX", "EXTEND"):
        if key in reference_primary:
            header[key] = reference_primary[key]
    header["ORIGIN"] = "mangia_logcube_74x74"
    header["MNGSRC"] = Path(official_cube).name
    header["MNGTEMP"] = Path(reference_logcube).name
    if "IFUCON" in official_header:
        header["IFUCON"] = official_header["IFUCON"]
    if "PSF" in official_header:
        header["PSF"] = official_header["PSF"]
    if "FOV" in official_header:
        header["FOV"] = official_header["FOV"]
    if "WGAS" in official_header:
        header["WGAS"] = bool(official_header["WGAS"])
    header["COMMENT"] = "MaNGIA official cube harmonized to a MaNGA LOGCUBE-like grid."
    return header


def build_wave_hdu(target_wave, template_wave_header):
    header = fits.Header()
    if template_wave_header is not None:
        for key in ("EXTNAME", "BUNIT"):
            if key in template_wave_header:
                header[key] = template_wave_header[key]
    header["EXTNAME"] = "WAVE"
    header["BUNIT"] = "Angstrom"
    return fits.ImageHDU(np.asarray(target_wave, dtype=np.float64), header=header, name="WAVE")


def sanitize_mask(mask):
    mask_array = np.asarray(mask, dtype=np.float32)
    return np.where(mask_array > 0, 1.0, 0.0)


def compute_delta_wave(wave):
    wave = np.asarray(wave, dtype=np.float64)
    delta = np.gradient(wave)
    return delta


def harmonize_official_cube(
    official_cube,
    reference_logcube,
    output_path,
    config,
):
    reference = read_reference_contract(reference_logcube)
    official = read_official_cube(official_cube)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_wave = reference["wave"]
    spectral_kind = config.get("spectral_interpolation_kind", "linear")
    flux_unit_scale = float(config.get("flux_unit_scale", 10.0))

    flux_spec = spectral_resample_cube(
        official["flux"],
        official["wave"],
        target_wave,
        fill_value=np.nan,
        kind=spectral_kind,
    )
    error_spec = spectral_resample_cube(
        official["error"],
        official["wave"],
        target_wave,
        fill_value=np.nan,
        kind=spectral_kind,
    )
    gas_spec = spectral_resample_cube(
        official["gas"],
        official["wave"],
        target_wave,
        fill_value=np.nan,
        kind=spectral_kind,
    )

    if official["mask"] is None:
        mask_source = np.zeros_like(official["flux"], dtype=np.float32)
    else:
        mask_source = sanitize_mask(official["mask"])
    mask_spec = spectral_resample_cube(
        mask_source,
        official["wave"],
        target_wave,
        fill_value=1.0,
        kind="linear",
    )
    mask_spec = mask_spec > float(config.get("mask_spectral_threshold", 1e-6))

    coordinates = build_spatial_coordinates(
        official["header"],
        reference["flux_header"],
        flip_x=bool(config.get("flip_x_to_match_manga", True)),
    )
    flux_spatial = spatial_resample_cube(
        flux_spec,
        coordinates,
        order=int(config.get("spatial_interpolation_order", 1)),
        cval=np.nan,
    )
    error_spatial = spatial_resample_cube(
        error_spec,
        coordinates,
        order=int(config.get("spatial_interpolation_order", 1)),
        cval=np.nan,
    )
    gas_spatial = spatial_resample_cube(
        gas_spec,
        coordinates,
        order=int(config.get("spatial_interpolation_order", 1)),
        cval=np.nan,
    )
    mask_spatial = spatial_resample_cube(
        mask_spec.astype(np.float32),
        coordinates,
        order=int(config.get("mask_interpolation_order", 0)),
        cval=1.0,
    )

    final_mask = mask_spatial > 0.5
    final_mask |= ~np.isfinite(flux_spatial)
    if error_spatial is not None:
        final_mask |= ~np.isfinite(error_spatial)
        final_mask |= error_spatial <= 0

    flux_final = np.nan_to_num(flux_spatial * flux_unit_scale, nan=0.0).astype(np.float32)
    error_final = None
    if error_spatial is not None:
        error_final = np.nan_to_num(error_spatial * flux_unit_scale, nan=np.inf).astype(np.float32)
    gas_final = None
    if gas_spatial is not None and bool(config.get("propagate_gas", True)):
        gas_final = np.nan_to_num(gas_spatial * flux_unit_scale, nan=0.0).astype(np.float32)

    ivar_final = np.zeros_like(flux_final, dtype=np.float32)
    if error_final is not None:
        good = np.isfinite(error_final) & (error_final > 0) & (~final_mask)
        ivar_final[good] = 1.0 / np.square(error_final[good])

    mask_final = final_mask.astype(np.uint32)

    primary_hdu = fits.PrimaryHDU(
        header=build_primary_header(
            reference["primary_header"],
            official["header"],
            official["path"],
            reference["path"],
        )
    )
    flux_hdu = fits.ImageHDU(
        flux_final,
        header=build_selected_header(
            reference["flux_header"],
            extname="FLUX",
            bunit="1E-17 erg/s/cm^2/Angstrom/spaxel",
        ),
        name="FLUX",
    )
    ivar_hdu = fits.ImageHDU(
        ivar_final,
        header=build_selected_header(reference["ivar_header"], extname="IVAR"),
        name="IVAR",
    )
    mask_hdu = fits.ImageHDU(
        mask_final,
        header=build_selected_header(reference["mask_header"], extname="MASK"),
        name="MASK",
    )
    wave_hdu = build_wave_hdu(target_wave, reference["wave_header"])
    hdus = [primary_hdu, flux_hdu, ivar_hdu, mask_hdu, wave_hdu]
    if gas_final is not None:
        gas_header = build_selected_header(
            reference["flux_header"],
            extname="GAS",
            bunit="1E-17 erg/s/cm^2/Angstrom/spaxel",
        )
        hdus.append(fits.ImageHDU(gas_final, header=gas_header, name="GAS"))

    fits.HDUList(hdus).writeto(output_path, overwrite=True)

    delta_source = compute_delta_wave(official["wave"])
    delta_target = compute_delta_wave(target_wave)
    source_integrated_flux = float(np.nansum(np.asarray(official["flux"], dtype=np.float64) * delta_source[:, None, None]) * flux_unit_scale)
    target_integrated_flux = float(np.nansum(np.asarray(flux_final, dtype=np.float64) * delta_target[:, None, None]))

    return {
        "output_path": str(output_path),
        "source_shape": list(official["flux"].shape),
        "target_shape": list(flux_final.shape),
        "source_wave_len": int(len(official["wave"])),
        "target_wave_len": int(len(target_wave)),
        "source_units": official["units"],
        "target_bunit": "1E-17 erg/s/cm^2/Angstrom/spaxel",
        "source_integrated_flux_scaled": source_integrated_flux,
        "target_integrated_flux": target_integrated_flux,
        "mask_fraction": float(mask_final.mean()),
        "gas_included": gas_final is not None,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Armoniza un cubo oficial MaNGIA a un LOGCUBE-like 74x74")
    parser.add_argument("official_cube", help="Ruta al cubo oficial *.cube.fits.gz")
    parser.add_argument("--reference-logcube", required=True, help="LOGCUBE real usado como template")
    parser.add_argument("--output", required=True, help="Ruta del FITS final armonizado")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "default_config.json"),
        help="Archivo JSON de configuración",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    result = harmonize_official_cube(
        official_cube=args.official_cube,
        reference_logcube=args.reference_logcube,
        output_path=args.output,
        config=config,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
