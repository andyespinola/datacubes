from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.io import fits

from .models import KinematicMaps, OfficialCube


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE_NAME = "MaStar_CB19.slog_1_5.fits.gz"


def cube_id_from_path(path: str | Path) -> str:
    name = Path(path).name
    for suffix in (".cube.fits.gz", ".cube.fits", ".fits.gz", ".fits"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def infer_wave_from_header(header: fits.Header, n_wave: int, axis: int = 3) -> np.ndarray:
    crval = header.get(f"CRVAL{axis}")
    cdelt = header.get(f"CDELT{axis}", header.get(f"CD{axis}_{axis}"))
    crpix = float(header.get(f"CRPIX{axis}", 1.0))
    if crval is None or cdelt is None:
        raise ValueError(f"Missing wavelength WCS keywords CRVAL{axis}/CDELT{axis}")

    pixels = np.arange(n_wave, dtype=np.float64) + 1.0
    return float(crval) + (pixels - crpix) * float(cdelt)


def _numpy_axis_from_fits_axis(axis: int) -> int:
    return 3 - int(axis)


def infer_spectral_axis(header: fits.Header, shape: tuple[int, int, int]) -> int:
    scored: list[tuple[int, int]] = []
    for axis in (1, 2, 3):
        crval = header.get(f"CRVAL{axis}")
        cdelt = header.get(f"CDELT{axis}", header.get(f"CD{axis}_{axis}"))
        numpy_axis = _numpy_axis_from_fits_axis(axis)
        axis_len = int(shape[numpy_axis])
        if crval is None or cdelt is None:
            continue
        ctype = str(header.get(f"CTYPE{axis}", "")).upper()
        cunit = str(header.get(f"CUNIT{axis}", "")).upper()
        score = 0
        if any(token in ctype or token in cunit for token in ("WAVE", "LAMBDA", "ANGSTROM")):
            score += 100
        if axis_len == max(shape):
            score += 10
        if axis_len > 100:
            score += 5
        if "RA" in ctype or "DEC" in ctype:
            score -= 100
        scored.append((score, axis))
    if not scored:
        raise ValueError("Could not infer spectral axis from FITS header")
    scored.sort(reverse=True)
    return scored[0][1]


def orient_spectral_first(data: np.ndarray, spectral_fits_axis: int) -> np.ndarray:
    numpy_axis = _numpy_axis_from_fits_axis(spectral_fits_axis)
    if numpy_axis == 0:
        return data
    return np.moveaxis(data, numpy_axis, 0)


def read_mangia_official_cube(path: str | Path) -> OfficialCube:
    path = Path(path).resolve()
    with fits.open(path, memmap=False) as hdul:
        if hdul[0].data is None:
            raise ValueError(f"Expected flux in PRIMARY for official MaNGIA cube: {path}")
        flux = np.asarray(hdul[0].data, dtype=np.float32)
        if flux.ndim != 3:
            raise ValueError(f"Expected 3D flux cube in {path}, found shape={flux.shape}")

        try:
            error = np.asarray(hdul["ERROR"].data, dtype=np.float32)
            mask = np.asarray(hdul["MASK"].data)
        except KeyError as exc:
            raise ValueError(f"Official MaNGIA cube must contain ERROR and MASK HDUs: {path}") from exc

        if error.shape != flux.shape or mask.shape != flux.shape:
            raise ValueError(
                f"ERROR/MASK shapes must match flux shape {flux.shape}; "
                f"found ERROR={error.shape}, MASK={mask.shape}"
            )

        header = hdul[0].header.copy()
        if "REDSHIFT" not in header:
            raise ValueError(f"Official MaNGIA cube is missing REDSHIFT: {path}")

        spectral_axis = infer_spectral_axis(header, tuple(flux.shape))
        flux = orient_spectral_first(flux, spectral_axis)
        error = orient_spectral_first(error, spectral_axis)
        mask = orient_spectral_first(mask, spectral_axis)
        wave = infer_wave_from_header(header, flux.shape[0], axis=spectral_axis)
        valid_cube = np.asarray(mask > 0, dtype=bool)
        return OfficialCube(
            path=path,
            galaxy_id=cube_id_from_path(path),
            flux=flux,
            error=error,
            valid_cube=valid_cube,
            wave=wave,
            redshift=float(header["REDSHIFT"]),
            header=header,
        )


def _template_candidates() -> list[Path]:
    return [
        PROJECT_ROOT / "kinematic_moments" / "templates" / DEFAULT_TEMPLATE_NAME,
        PROJECT_ROOT / "official_mangia" / "libs" / DEFAULT_TEMPLATE_NAME,
        PROJECT_ROOT / "deploy_mangia_10k" / "official_mangia" / "libs" / DEFAULT_TEMPLATE_NAME,
        PROJECT_ROOT / DEFAULT_TEMPLATE_NAME,
        Path.cwd() / DEFAULT_TEMPLATE_NAME,
    ]


def resolve_template_path(provided: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if provided is not None:
        candidates.append(Path(provided).expanduser())
    env_path = os.environ.get("KINEMATICS_TEMPLATE_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(_template_candidates())

    checked: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        checked.append(str(resolved))
        if resolved.exists():
            return resolved

    joined = "\n  - ".join(checked)
    raise FileNotFoundError(
        "Could not find MaStar template MaStar_CB19.slog_1_5.fits.gz. "
        "Pass --template-path or set KINEMATICS_TEMPLATE_PATH. Checked:\n  - "
        f"{joined}"
    )


def output_paths(cube_path: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir).resolve()
    base = cube_id_from_path(cube_path)
    return (
        output_dir / f"{base}.kinematics_ppxf.npz",
        output_dir / f"{base}.kinematics_ppxf.fits.gz",
    )


def write_npz(maps: KinematicMaps, path: str | Path) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        galaxy_id=np.asarray(maps.galaxy_id),
        cube_path=np.asarray(str(maps.cube_path)),
        h3=maps.h3.astype(np.float32),
        h4=maps.h4.astype(np.float32),
        v_ppxf=maps.v_ppxf.astype(np.float32),
        sigma_ppxf=maps.sigma_ppxf.astype(np.float32),
        h3_err=maps.h3_err.astype(np.float32),
        h4_err=maps.h4_err.astype(np.float32),
        quality_mask=maps.quality_mask.astype(np.uint8),
        coverage_mask=maps.coverage_mask.astype(np.uint8),
        snr_map=maps.snr_map.astype(np.float32),
        chi2_map=maps.chi2_map.astype(np.float32),
        n_spaxels_fitted=np.asarray(maps.n_spaxels_fitted, dtype=np.int32),
        n_quality_ok=np.asarray(maps.n_quality_ok, dtype=np.int32),
        message=np.asarray(maps.message),
    )
    return path


def _spatial_header(source: fits.Header, extname: str, bunit: str | None = None) -> fits.Header:
    header = fits.Header()
    for key in (
        "CRVAL1",
        "CRPIX1",
        "CD1_1",
        "CD1_2",
        "CDELT1",
        "CTYPE1",
        "CUNIT1",
        "CRVAL2",
        "CRPIX2",
        "CD2_1",
        "CD2_2",
        "CDELT2",
        "CTYPE2",
        "CUNIT2",
        "RADECSYS",
        "SYSTEM",
        "EQUINOX",
        "PSF",
        "FOV",
        "KPCSEC",
        "REDSHIFT",
        "IFUCON",
    ):
        if key in source:
            header[key] = source[key]
    header["EXTNAME"] = extname
    if bunit is not None:
        header["BUNIT"] = bunit
    return header


def write_fits(maps: KinematicMaps, source_header: fits.Header, path: str | Path) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    primary = fits.Header()
    primary["ORIGIN"] = "kinematic_moments"
    primary["MNGSRC"] = Path(maps.cube_path).name
    primary["GALID"] = maps.galaxy_id
    primary["NFIT"] = int(maps.n_spaxels_fitted)
    primary["NQUAL"] = int(maps.n_quality_ok)
    if "REDSHIFT" in source_header:
        primary["REDSHIFT"] = source_header["REDSHIFT"]
    if maps.message:
        primary["COMMENT"] = maps.message[:68]

    hdus = [fits.PrimaryHDU(header=primary)]
    arrays = [
        ("H3", maps.h3, "dimensionless"),
        ("H4", maps.h4, "dimensionless"),
        ("V_PPXF", maps.v_ppxf, "km/s"),
        ("SIGMA_PPXF", maps.sigma_ppxf, "km/s"),
        ("H3_ERR", maps.h3_err, "dimensionless"),
        ("H4_ERR", maps.h4_err, "dimensionless"),
        ("QUALITY", maps.quality_mask.astype(np.uint8), None),
        ("COVERAGE", maps.coverage_mask.astype(np.uint8), None),
        ("SNR", maps.snr_map, None),
        ("CHI2", maps.chi2_map, None),
    ]
    for name, array, unit in arrays:
        hdus.append(
            fits.ImageHDU(
                np.asarray(array),
                header=_spatial_header(source_header, name, unit),
                name=name,
            )
        )
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def read_manifest_cube_paths(path: str | Path) -> list[Path]:
    rows: list[Path] = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "cube_path" not in (reader.fieldnames or []):
            raise ValueError(f"Manifest must contain a cube_path column: {path}")
        for raw in reader:
            value = (raw.get("cube_path") or "").strip()
            if value:
                rows.append(Path(value))
    return rows


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out
