import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import warnings

import numpy as np
from astropy.io import fits
from flask import Flask, abort, jsonify, request, send_from_directory


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "cube_viewer_static"


@dataclass
class CubeBundle:
    path: Path
    flux: np.ndarray
    wave: np.ndarray
    error: np.ndarray | None
    mask: np.ndarray | None
    gas: np.ndarray | None
    x_grid: np.ndarray | None
    y_grid: np.ndarray | None
    overview_map: np.ndarray
    extnames: list[str]
    header_summary: dict

    @property
    def n_wave(self):
        return int(self.flux.shape[0])

    @property
    def ny(self):
        return int(self.flux.shape[1])

    @property
    def nx(self):
        return int(self.flux.shape[2])


def infer_wave_from_header(header, n_wave):
    crval = header.get("CRVAL3")
    crpix = float(header.get("CRPIX3", 1.0))
    cdelt = header.get("CDELT3", header.get("CD3_3"))
    if crval is None or cdelt is None:
        return None

    pix = np.arange(n_wave, dtype=np.float64) + 1.0
    return float(crval) + (pix - crpix) * float(cdelt)


def get_named_hdu(hdul, name):
    upper = name.upper()
    for hdu in hdul:
        if hdu.name.upper() == upper:
            return hdu
    return None


def is_spectral_cube(path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with fits.open(path, memmap=True) as hdul:
                if hdul[0].data is None or np.ndim(hdul[0].data) != 3:
                    return False

                if get_named_hdu(hdul, "WAVE") is not None:
                    return True

                header = hdul[0].header
                ctype3 = str(header.get("CTYPE3", "")).upper()
                cunit3 = str(header.get("CUNIT3", "")).upper()
                return "WAVE" in ctype3 or "LAMBDA" in ctype3 or "WAVELENGTH" in cunit3
    except Exception:
        return False


def discover_cube_files():
    cubes = []
    for path in sorted(BASE_DIR.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or STATIC_DIR in path.parents:
            continue
        if not (path.name.endswith(".fits") or path.name.endswith(".fits.gz")):
            continue
        if not is_spectral_cube(path):
            continue

        rel = path.relative_to(BASE_DIR).as_posix()
        cubes.append(
            {
                "path": rel,
                "label": rel,
            }
        )
    return cubes


def resolve_cube_path(relative_path):
    if not relative_path:
        abort(400, "Falta el parámetro 'path'")

    resolved = (BASE_DIR / relative_path).resolve()
    try:
        resolved.relative_to(BASE_DIR)
    except ValueError as exc:
        raise ValueError("La ruta solicitada queda fuera del workspace") from exc

    if not resolved.exists():
        raise FileNotFoundError(f"No existe el archivo: {relative_path}")

    return resolved


def sanitize_image(image):
    array = np.asarray(image, dtype=np.float32)
    array = np.where(np.isfinite(array), array, np.nan)
    if not np.isfinite(array).any():
        return np.zeros_like(array, dtype=np.float32)
    return array


def serialize_image(image):
    clean = sanitize_image(image)
    finite = clean[np.isfinite(clean)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.nanpercentile(finite, 2))
        vmax = float(np.nanpercentile(finite, 98))
        if vmax <= vmin:
            vmax = vmin + 1.0

    return {
        "data": np.nan_to_num(clean, nan=0.0).tolist(),
        "vmin": vmin,
        "vmax": vmax,
    }


def header_summary(header):
    keys = (
        "OBJECT",
        "REDSHIFT",
        "PSF",
        "FOV",
        "IFUCON",
        "BUNIT",
        "UNITS",
        "R",
        "WGAS",
    )
    summary = {}
    for key in keys:
        if key in header:
            value = header[key]
            if isinstance(value, np.generic):
                value = value.item()
            summary[key] = value
    return summary


@lru_cache(maxsize=4)
def load_cube_bundle(relative_path):
    path = resolve_cube_path(relative_path)
    with fits.open(path, memmap=False) as hdul:
        flux = np.asarray(hdul[0].data, dtype=np.float32)
        if flux.ndim != 3:
            raise ValueError(f"El cubo debe ser 3D y tiene forma {flux.shape}")

        wave_hdu = get_named_hdu(hdul, "WAVE")
        if wave_hdu is not None:
            wave = np.asarray(wave_hdu.data, dtype=np.float64).reshape(-1)
        else:
            wave = infer_wave_from_header(hdul[0].header, flux.shape[0])
            if wave is None:
                wave = np.arange(flux.shape[0], dtype=np.float64)

        x_hdu = get_named_hdu(hdul, "XGRID")
        y_hdu = get_named_hdu(hdul, "YGRID")
        x_grid = None if x_hdu is None else np.asarray(x_hdu.data, dtype=np.float64).reshape(-1)
        y_grid = None if y_hdu is None else np.asarray(y_hdu.data, dtype=np.float64).reshape(-1)

        error_hdu = get_named_hdu(hdul, "ERROR")
        mask_hdu = get_named_hdu(hdul, "MASK")
        gas_hdu = get_named_hdu(hdul, "GAS")

        error = None if error_hdu is None else np.asarray(error_hdu.data, dtype=np.float32)
        mask = None if mask_hdu is None else np.asarray(mask_hdu.data)
        gas = None if gas_hdu is None else np.asarray(gas_hdu.data, dtype=np.float32)

        finite_mask = np.isfinite(flux)
        summed = np.sum(np.where(finite_mask, flux, 0.0), axis=0, dtype=np.float64)
        counts = np.sum(finite_mask, axis=0)
        overview = np.divide(
            summed,
            counts,
            out=np.zeros_like(summed, dtype=np.float32),
            where=counts > 0,
        )
        extnames = [hdu.name for hdu in hdul]
        summary = header_summary(hdul[0].header)

    return CubeBundle(
        path=path,
        flux=flux,
        wave=wave,
        error=error,
        mask=mask,
        gas=gas,
        x_grid=x_grid,
        y_grid=y_grid,
        overview_map=np.asarray(overview, dtype=np.float32),
        extnames=extnames,
        header_summary=summary,
    )


def create_app():
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/api/files")
    def api_files():
        return jsonify({"files": discover_cube_files()})

    @app.get("/api/cube")
    def api_cube():
        relative_path = request.args.get("path")
        bundle = load_cube_bundle(relative_path)

        return jsonify(
            {
                "path": bundle.path.relative_to(BASE_DIR).as_posix(),
                "shape": {
                    "n_wave": bundle.n_wave,
                    "ny": bundle.ny,
                    "nx": bundle.nx,
                },
                "wave": {
                    "min": float(bundle.wave[0]),
                    "max": float(bundle.wave[-1]),
                    "count": bundle.wave.size,
                },
                "has_error": bundle.error is not None,
                "has_mask": bundle.mask is not None,
                "has_gas": bundle.gas is not None,
                "overview_map": serialize_image(bundle.overview_map),
                "default_slice_index": bundle.n_wave // 2,
                "extnames": bundle.extnames,
                "header_summary": bundle.header_summary,
                "x_grid": None if bundle.x_grid is None else bundle.x_grid.tolist(),
                "y_grid": None if bundle.y_grid is None else bundle.y_grid.tolist(),
            }
        )

    @app.get("/api/slice")
    def api_slice():
        relative_path = request.args.get("path")
        bundle = load_cube_bundle(relative_path)

        index = int(request.args.get("index", bundle.n_wave // 2))
        index = max(0, min(index, bundle.n_wave - 1))
        plane = bundle.flux[index]

        return jsonify(
            {
                "index": index,
                "wavelength": float(bundle.wave[index]),
                "image": serialize_image(plane),
            }
        )

    @app.get("/api/spaxel")
    def api_spaxel():
        relative_path = request.args.get("path")
        bundle = load_cube_bundle(relative_path)

        x = int(request.args.get("x"))
        y = int(request.args.get("y"))

        if not (0 <= x < bundle.nx and 0 <= y < bundle.ny):
            abort(400, "El spaxel pedido está fuera del rango del cubo")

        flux = bundle.flux[:, y, x]
        error = None if bundle.error is None else bundle.error[:, y, x]
        gas = None if bundle.gas is None else bundle.gas[:, y, x]
        mask = None if bundle.mask is None else bundle.mask[:, y, x]

        valid = np.isfinite(flux)
        finite_flux = flux[valid]
        stats = {
            "min": None if finite_flux.size == 0 else float(np.nanmin(finite_flux)),
            "max": None if finite_flux.size == 0 else float(np.nanmax(finite_flux)),
            "mean": None if finite_flux.size == 0 else float(np.nanmean(finite_flux)),
            "sum": None if finite_flux.size == 0 else float(np.nansum(finite_flux)),
            "masked_count": 0 if mask is None else int(np.count_nonzero(mask)),
        }

        response = {
            "x": x,
            "y": y,
            "wave": bundle.wave.tolist(),
            "flux": np.nan_to_num(flux, nan=0.0).tolist(),
            "error": None if error is None else np.nan_to_num(error, nan=0.0).tolist(),
            "gas": None if gas is None else np.nan_to_num(gas, nan=0.0).tolist(),
            "mask": None if mask is None else np.asarray(mask).tolist(),
            "stats": stats,
            "coords": {
                "x_grid": None if bundle.x_grid is None else float(bundle.x_grid[x]),
                "y_grid": None if bundle.y_grid is None else float(bundle.y_grid[y]),
            },
        }
        return jsonify(response)

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Visor web local para cubos FITS reconstruidos"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host del servidor")
    parser.add_argument("--port", type=int, default=8000, help="Puerto del servidor")
    parser.add_argument("--debug", action="store_true", help="Activa modo debug")
    args = parser.parse_args()

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
