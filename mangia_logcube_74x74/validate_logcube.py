import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from harmonize_logcube import get_named_hdu, infer_wave_from_header, load_config


def load_flux_bundle(path):
    path = Path(path).resolve()
    with fits.open(path, memmap=False) as hdul:
        flux_hdu = get_named_hdu(hdul, "FLUX")
        if flux_hdu is None:
            raise ValueError(f"No encontré la extensión FLUX en {path}")
        ivar_hdu = get_named_hdu(hdul, "IVAR")
        mask_hdu = get_named_hdu(hdul, "MASK")
        wave_hdu = get_named_hdu(hdul, "WAVE")
        wave = None
        if wave_hdu is not None and wave_hdu.data is not None:
            wave = np.asarray(wave_hdu.data, dtype=np.float64).reshape(-1)
        else:
            wave = infer_wave_from_header(flux_hdu.header, flux_hdu.data.shape[0], axis=3)
        return {
            "path": path,
            "primary_header": hdul[0].header.copy(),
            "flux_header": flux_hdu.header.copy(),
            "flux": np.asarray(flux_hdu.data),
            "ivar": None if ivar_hdu is None else np.asarray(ivar_hdu.data),
            "mask": None if mask_hdu is None else np.asarray(mask_hdu.data),
            "wave": wave,
            "extnames": [h.name for h in hdul],
        }


def load_official_flux(path):
    path = Path(path).resolve()
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header.copy()
        wave = infer_wave_from_header(header, hdul[0].data.shape[0], axis=3)
        flux = np.asarray(hdul[0].data, dtype=np.float64)
        return {"path": path, "flux": flux, "wave": wave}


def compute_delta_wave(wave):
    return np.gradient(np.asarray(wave, dtype=np.float64))


def validate_product(product_path, reference_logcube, official_cube, config):
    product = load_flux_bundle(product_path)
    reference = load_flux_bundle(reference_logcube)
    official = load_official_flux(official_cube)

    wcs_keys = (
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
        "CTYPE1",
        "CTYPE2",
        "CTYPE3",
    )
    wcs_tolerance = float(config.get("wcs_tolerance", 1e-9))
    wcs_ok = True
    wcs_deltas = {}
    for key in wcs_keys:
        if key not in reference["flux_header"]:
            continue
        ref_value = reference["flux_header"][key]
        prod_value = product["flux_header"].get(key)
        if isinstance(ref_value, str):
            ok = prod_value == ref_value
            wcs_deltas[key] = {"reference": ref_value, "product": prod_value, "ok": ok}
        else:
            diff = abs(float(prod_value) - float(ref_value))
            ok = diff <= wcs_tolerance
            wcs_deltas[key] = {"reference": float(ref_value), "product": float(prod_value), "abs_diff": diff, "ok": ok}
        wcs_ok &= bool(wcs_deltas[key]["ok"])

    delta_source = compute_delta_wave(official["wave"])
    delta_target = compute_delta_wave(product["wave"])
    source_integrated_flux = float(np.nansum(official["flux"] * delta_source[:, None, None]) * float(config.get("flux_unit_scale", 10.0)))
    target_integrated_flux = float(np.nansum(np.asarray(product["flux"], dtype=np.float64) * delta_target[:, None, None]))
    if source_integrated_flux == 0:
        relative_flux_error = 0.0
    else:
        relative_flux_error = abs(target_integrated_flux - source_integrated_flux) / abs(source_integrated_flux)

    mask = np.asarray(product["mask"]) if product["mask"] is not None else np.zeros_like(product["flux"], dtype=np.uint32)
    flux = np.asarray(product["flux"], dtype=np.float64)
    ivar = np.asarray(product["ivar"], dtype=np.float64) if product["ivar"] is not None else None
    valid_voxels = mask == 0

    result = {
        "product_path": str(Path(product_path).resolve()),
        "reference_logcube": str(Path(reference_logcube).resolve()),
        "official_cube": str(Path(official_cube).resolve()),
        "shape_matches": tuple(product["flux"].shape) == tuple(reference["flux"].shape),
        "wave_matches": np.allclose(product["wave"], reference["wave"], atol=1e-8, rtol=1e-10),
        "wcs_matches": wcs_ok,
        "wcs_deltas": wcs_deltas,
        "extnames": product["extnames"],
        "required_extnames_present": {"FLUX", "IVAR", "MASK", "WAVE"}.issubset(set(product["extnames"])),
        "bunit_matches": str(product["flux_header"].get("BUNIT", "")) == str(reference["flux_header"].get("BUNIT", "")),
        "mask_shape_matches": product["mask"] is not None and tuple(product["mask"].shape) == tuple(reference["flux"].shape),
        "ivar_shape_matches": product["ivar"] is not None and tuple(product["ivar"].shape) == tuple(reference["flux"].shape),
        "ivar_non_negative": ivar is not None and bool(np.all(ivar >= 0)),
        "flux_has_no_nan_on_valid": bool(np.all(np.isfinite(flux[valid_voxels]))),
        "source_integrated_flux_scaled": source_integrated_flux,
        "target_integrated_flux": target_integrated_flux,
        "relative_flux_error": relative_flux_error,
        "flux_conservation_ok": relative_flux_error <= float(config.get("flux_conservation_tolerance", 0.2)),
    }
    result["all_checks_passed"] = all(
        [
            result["shape_matches"],
            result["wave_matches"],
            result["wcs_matches"],
            result["required_extnames_present"],
            result["bunit_matches"],
            result["mask_shape_matches"],
            result["ivar_shape_matches"],
            result["ivar_non_negative"],
            result["flux_has_no_nan_on_valid"],
            result["flux_conservation_ok"],
        ]
    )
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="Valida un cubo LOGCUBE-like contra el template MaNGA")
    parser.add_argument("product_path", help="Ruta al FITS armonizado")
    parser.add_argument("--reference-logcube", required=True, help="LOGCUBE real usado como template")
    parser.add_argument("--official-cube", required=True, help="Cubo oficial previo a la armonización")
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
    result = validate_product(
        product_path=args.product_path,
        reference_logcube=args.reference_logcube,
        official_cube=args.official_cube,
        config=config,
    )
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["all_checks_passed"] else 1)


if __name__ == "__main__":
    main()
