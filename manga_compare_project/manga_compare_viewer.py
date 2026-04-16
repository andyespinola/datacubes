import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import warnings

import numpy as np
from astropy.io import fits
from flask import Flask, abort, jsonify, request, send_from_directory


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "viewer_static"
DEFAULT_DATA_DIR = BASE_DIR / "data"


@dataclass
class ObservationBundle:
    path: Path
    mode: str
    source: str
    flux: np.ndarray
    wave: np.ndarray
    error: np.ndarray | None
    mask: np.ndarray | None
    gas: np.ndarray | None
    x_grid: np.ndarray | None
    y_grid: np.ndarray | None
    xpos: np.ndarray | None
    ypos: np.ndarray | None
    overview_map: np.ndarray | None
    overview_values: np.ndarray | None
    rss_geometry: str | None
    extnames: list[str]
    header_summary: dict

    @property
    def n_wave(self):
        if self.flux.ndim == 3:
            return int(self.flux.shape[0])
        return int(self.flux.shape[1])

    @property
    def ny(self):
        if self.flux.ndim == 3:
            return int(self.flux.shape[1])
        return None

    @property
    def nx(self):
        if self.flux.ndim == 3:
            return int(self.flux.shape[2])
        return None

    @property
    def n_sample(self):
        if self.flux.ndim == 2:
            return int(self.flux.shape[0])
        return None


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

    pix = np.arange(n_wave, dtype=np.float64) + 1.0
    return float(crval) + (pix - crpix) * float(cdelt)


def infer_regular_spatial_axis(header, axis, n_pix):
    crpix = float(header.get(f"CRPIX{axis}", (n_pix + 1) / 2.0))
    cdelt = header.get(f"CDELT{axis}")
    if cdelt is None:
        if axis == 1:
            cdelt = header.get("CD1_1")
        elif axis == 2:
            cdelt = header.get("CD2_2")
    if cdelt is None:
        return np.arange(n_pix, dtype=np.float64)

    scale_arcsec = float(cdelt) * 3600.0
    pix = np.arange(n_pix, dtype=np.float64) + 1.0
    return (pix - crpix) * scale_arcsec


def make_sigma_from_ivar(ivar):
    sigma = np.zeros_like(ivar, dtype=np.float32)
    good = np.isfinite(ivar) & (ivar > 0)
    sigma[good] = 1.0 / np.sqrt(ivar[good])
    sigma[~good] = np.nan
    return sigma


def sanitize_image(image):
    array = np.asarray(image, dtype=np.float32)
    array = np.where(np.isfinite(array), array, np.nan)
    if not np.isfinite(array).any():
        return np.zeros_like(array, dtype=np.float32)
    return array


def percentile_range(values, low=2, high=98):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(finite, low))
    vmax = float(np.nanpercentile(finite, high))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def serialize_image(image):
    clean = sanitize_image(image)
    vmin, vmax = percentile_range(clean)
    return {
        "kind": "image",
        "data": np.nan_to_num(clean, nan=0.0).tolist(),
        "vmin": vmin,
        "vmax": vmax,
    }


def normalize_rss_orientation(array, wave_len):
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"Se esperaba un arreglo 2D para RSS y llego {arr.shape}")

    if wave_len is not None:
        if arr.shape[1] == wave_len:
            return arr
        if arr.shape[0] == wave_len:
            return arr.T

    if arr.shape[0] <= arr.shape[1]:
        return arr
    return arr.T


def orient_position_array(array, n_sample, n_wave):
    if array is None:
        return None
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size == n_sample:
            return np.repeat(arr[:, None], n_wave, axis=1)
        raise ValueError(f"Posiciones 1D incompatibles con RSS: {arr.shape}")
    if arr.ndim != 2:
        raise ValueError(f"Posiciones no compatibles con RSS: {arr.shape}")
    if arr.shape == (n_sample, n_wave):
        return arr
    if arr.shape == (n_wave, n_sample):
        return arr.T
    if arr.shape[0] == n_sample:
        return arr
    if arr.shape[1] == n_sample:
        return arr.T
    raise ValueError(f"No pude orientar posiciones RSS con shape {arr.shape}")


def header_summary(header):
    keys = (
        "TELESCOP",
        "INSTRUME",
        "PLATEIFU",
        "MANGAID",
        "OBJRA",
        "OBJDEC",
        "IFURA",
        "IFUDEC",
        "REDSHIFT",
        "BUNIT",
        "UNITS",
        "MASKNAME",
        "DRP3QUAL",
        "VERSDRP3",
        "DAPFRMT",
        "PSF",
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


def merged_header(*headers):
    combined = fits.Header()
    for header in headers:
        if header is None:
            continue
        for card in header.cards:
            key = card.keyword
            if key in ("", "COMMENT", "HISTORY", "END"):
                continue
            combined[key] = card.value
    return combined


def is_supported_observation(path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with fits.open(path, memmap=True) as hdul:
                primary = hdul[0].data
                flux_hdu = get_named_hdu(hdul, "FLUX")
                if primary is not None and np.ndim(primary) == 3:
                    return True
                if flux_hdu is not None and np.ndim(flux_hdu.data) in (2, 3):
                    return True
                return False
    except Exception:
        return False


def discover_files(data_dir):
    files = []
    data_dir = Path(data_dir).resolve()
    if not data_dir.exists():
        return files

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if not (path.name.endswith(".fits") or path.name.endswith(".fits.gz")):
            continue
        if not is_supported_observation(path):
            continue
        files.append(
            {
                "path": path.relative_to(data_dir).as_posix(),
                "label": path.relative_to(data_dir).as_posix(),
            }
        )
    return files


def resolve_data_path(data_dir, relative_path):
    if not relative_path:
        abort(400, "Falta el parametro 'path'")
    root = Path(data_dir).resolve()
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("La ruta solicitada queda fuera del directorio de datos") from exc
    if not resolved.exists():
        raise FileNotFoundError(f"No existe el archivo: {relative_path}")
    return resolved


def load_cube_from_primary(path, hdul):
    flux = np.asarray(hdul[0].data, dtype=np.float32)
    wave = get_named_hdu(hdul, "WAVE")
    if wave is not None:
        wave_data = np.asarray(wave.data, dtype=np.float64).reshape(-1)
    else:
        wave_data = infer_wave_from_header(hdul[0].header, flux.shape[0], axis=3)
        if wave_data is None:
            wave_data = np.arange(flux.shape[0], dtype=np.float64)

    error_hdu = get_named_hdu(hdul, "ERROR")
    gas_hdu = get_named_hdu(hdul, "GAS")
    mask_hdu = get_named_hdu(hdul, "MASK")
    x_hdu = get_named_hdu(hdul, "XGRID")
    y_hdu = get_named_hdu(hdul, "YGRID")

    error = None if error_hdu is None else np.asarray(error_hdu.data, dtype=np.float32)
    gas = None if gas_hdu is None else np.asarray(gas_hdu.data, dtype=np.float32)
    mask = None if mask_hdu is None else np.asarray(mask_hdu.data)

    if x_hdu is not None and y_hdu is not None:
        x_grid = np.asarray(x_hdu.data, dtype=np.float64).reshape(-1)
        y_grid = np.asarray(y_hdu.data, dtype=np.float64).reshape(-1)
    else:
        x_grid = infer_regular_spatial_axis(hdul[0].header, axis=1, n_pix=flux.shape[2])
        y_grid = infer_regular_spatial_axis(hdul[0].header, axis=2, n_pix=flux.shape[1])

    overview = np.nanmean(flux, axis=0)
    return ObservationBundle(
        path=path,
        mode="cube",
        source="primary_cube",
        flux=flux,
        wave=wave_data,
        error=error,
        mask=mask,
        gas=gas,
        x_grid=x_grid,
        y_grid=y_grid,
        xpos=None,
        ypos=None,
        overview_map=np.asarray(overview, dtype=np.float32),
        overview_values=None,
        rss_geometry=None,
        extnames=[hdu.name for hdu in hdul],
        header_summary=header_summary(hdul[0].header),
    )


def load_cube_from_flux(path, hdul):
    flux_hdu = get_named_hdu(hdul, "FLUX")
    flux = np.asarray(flux_hdu.data, dtype=np.float32)
    wave_hdu = get_named_hdu(hdul, "WAVE")
    if wave_hdu is not None:
        wave = np.asarray(wave_hdu.data, dtype=np.float64).reshape(-1)
    else:
        wave = infer_wave_from_header(flux_hdu.header, flux.shape[0], axis=3)
        if wave is None:
            wave = np.arange(flux.shape[0], dtype=np.float64)

    ivar_hdu = get_named_hdu(hdul, "IVAR")
    mask_hdu = get_named_hdu(hdul, "MASK")
    error = None if ivar_hdu is None else make_sigma_from_ivar(np.asarray(ivar_hdu.data, dtype=np.float32))
    mask = None if mask_hdu is None else np.asarray(mask_hdu.data)

    x_grid = infer_regular_spatial_axis(flux_hdu.header, axis=1, n_pix=flux.shape[2])
    y_grid = infer_regular_spatial_axis(flux_hdu.header, axis=2, n_pix=flux.shape[1])
    overview = np.nanmean(flux, axis=0)

    return ObservationBundle(
        path=path,
        mode="cube",
        source="manga_cube",
        flux=flux,
        wave=wave,
        error=error,
        mask=mask,
        gas=None,
        x_grid=x_grid,
        y_grid=y_grid,
        xpos=None,
        ypos=None,
        overview_map=np.asarray(overview, dtype=np.float32),
        overview_values=None,
        rss_geometry=None,
        extnames=[hdu.name for hdu in hdul],
        header_summary=header_summary(merged_header(hdul[0].header, flux_hdu.header)),
    )


def load_rss_from_flux(path, hdul):
    flux_hdu = get_named_hdu(hdul, "FLUX")
    wave_hdu = get_named_hdu(hdul, "WAVE")
    wave = None
    if wave_hdu is not None:
        wave = np.asarray(wave_hdu.data, dtype=np.float64).reshape(-1)

    raw_flux = np.asarray(flux_hdu.data, dtype=np.float32)
    flux = normalize_rss_orientation(raw_flux, None if wave is None else wave.size)
    n_sample, n_wave = flux.shape

    if wave is None:
        wave = infer_wave_from_header(flux_hdu.header, n_wave, axis=1)
        if wave is None:
            wave = np.arange(n_wave, dtype=np.float64)

    ivar_hdu = get_named_hdu(hdul, "IVAR")
    mask_hdu = get_named_hdu(hdul, "MASK")
    xpos_hdu = get_named_hdu(hdul, "XPOS")
    ypos_hdu = get_named_hdu(hdul, "YPOS")

    error = None
    if ivar_hdu is not None:
        ivar = normalize_rss_orientation(np.asarray(ivar_hdu.data, dtype=np.float32), wave.size)
        error = make_sigma_from_ivar(ivar)

    mask = None
    if mask_hdu is not None:
        mask = normalize_rss_orientation(np.asarray(mask_hdu.data), wave.size)

    xpos = orient_position_array(None if xpos_hdu is None else xpos_hdu.data, n_sample, n_wave)
    ypos = orient_position_array(None if ypos_hdu is None else ypos_hdu.data, n_sample, n_wave)
    rss_geometry = "ifu_positions"

    if xpos is None or ypos is None:
        side = int(np.ceil(np.sqrt(n_sample)))
        indices = np.arange(n_sample, dtype=np.float32)
        pseudo_x = (indices % side) - (side - 1) / 2.0
        pseudo_y = (indices // side) - (side - 1) / 2.0
        xpos = np.repeat(pseudo_x[:, None], n_wave, axis=1)
        ypos = np.repeat(pseudo_y[:, None], n_wave, axis=1)
        rss_geometry = "pseudo_grid_by_fiber_index"

    overview_values = np.nanmean(flux, axis=1)
    summary = header_summary(merged_header(hdul[0].header, flux_hdu.header))
    summary["RSS_LAYOUT"] = rss_geometry
    return ObservationBundle(
        path=path,
        mode="rss",
        source="manga_rss",
        flux=flux,
        wave=wave,
        error=error,
        mask=mask,
        gas=None,
        x_grid=None,
        y_grid=None,
        xpos=xpos,
        ypos=ypos,
        overview_map=None,
        overview_values=np.asarray(overview_values, dtype=np.float32),
        rss_geometry=rss_geometry,
        extnames=[hdu.name for hdu in hdul],
        header_summary=summary,
    )


@lru_cache(maxsize=8)
def load_observation(data_dir, relative_path):
    path = resolve_data_path(data_dir, relative_path)
    with fits.open(path, memmap=False) as hdul:
        flux_hdu = get_named_hdu(hdul, "FLUX")
        if flux_hdu is not None and np.ndim(flux_hdu.data) == 3:
            return load_cube_from_flux(path, hdul)
        if flux_hdu is not None and np.ndim(flux_hdu.data) == 2:
            return load_rss_from_flux(path, hdul)
        if hdul[0].data is not None and np.ndim(hdul[0].data) == 3:
            return load_cube_from_primary(path, hdul)
        raise ValueError(f"No reconoci un cubo o RSS soportado en {path.name}")


def serialize_scatter(xpos, ypos, values):
    values = np.asarray(values, dtype=np.float32)
    xpos = np.asarray(xpos, dtype=np.float32)
    ypos = np.asarray(ypos, dtype=np.float32)
    good = np.isfinite(values) & np.isfinite(xpos) & np.isfinite(ypos)

    if not np.any(good):
        return {
            "kind": "scatter",
            "points": [],
            "vmin": 0.0,
            "vmax": 1.0,
            "bounds": {"xmin": -1.0, "xmax": 1.0, "ymin": -1.0, "ymax": 1.0},
        }

    x = xpos[good]
    y = ypos[good]
    v = values[good]
    vmin, vmax = percentile_range(v)
    points = [
        {"index": int(idx), "x": float(xx), "y": float(yy), "value": float(vv)}
        for idx, (xx, yy, vv) in enumerate(zip(xpos, ypos, values))
        if np.isfinite(xx) and np.isfinite(yy) and np.isfinite(vv)
    ]
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    pad_x = 0.05 * (xmax - xmin if xmax > xmin else 1.0)
    pad_y = 0.05 * (ymax - ymin if ymax > ymin else 1.0)

    return {
        "kind": "scatter",
        "points": points,
        "vmin": vmin,
        "vmax": vmax,
        "bounds": {
            "xmin": xmin - pad_x,
            "xmax": xmax + pad_x,
            "ymin": ymin - pad_y,
            "ymax": ymax + pad_y,
        },
    }


def rss_map_payload(bundle, wave_index=None):
    if wave_index is None:
        values = bundle.overview_values
        xpos = np.nanmedian(bundle.xpos, axis=1)
        ypos = np.nanmedian(bundle.ypos, axis=1)
        payload = serialize_scatter(xpos, ypos, values)
        if bundle.rss_geometry == "pseudo_grid_by_fiber_index":
            payload["label"] = "Mapa medio sobre fibras (pseudo-grilla por indice)"
        else:
            payload["label"] = "Mapa medio sobre fibras"
        return payload

    index = max(0, min(int(wave_index), bundle.n_wave - 1))
    values = bundle.flux[:, index]
    xpos = bundle.xpos[:, index]
    ypos = bundle.ypos[:, index]
    payload = serialize_scatter(xpos, ypos, values)
    payload["index"] = index
    payload["wavelength"] = float(bundle.wave[index])
    if bundle.rss_geometry == "pseudo_grid_by_fiber_index":
        payload["label"] = f"Mapa RSS en {bundle.wave[index]:.1f} A (pseudo-grilla por indice)"
    else:
        payload["label"] = f"Mapa RSS en {bundle.wave[index]:.1f} A"
    return payload


def create_app(data_dir):
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/api/files")
    def api_files():
        return jsonify({"files": discover_files(data_dir)})

    @app.get("/api/observation")
    def api_observation():
        relative_path = request.args.get("path")
        bundle = load_observation(data_dir, relative_path)

        if bundle.mode == "cube":
            map_payload = serialize_image(bundle.overview_map)
            default_slice_index = bundle.n_wave // 2
        else:
            map_payload = rss_map_payload(bundle, wave_index=None)
            default_slice_index = bundle.n_wave // 2

        shape = {
            "mode": bundle.mode,
            "n_wave": bundle.n_wave,
            "n_sample": bundle.n_sample,
            "ny": bundle.ny,
            "nx": bundle.nx,
        }
        return jsonify(
            {
                "path": bundle.path.relative_to(Path(data_dir).resolve()).as_posix(),
                "source": bundle.source,
                "shape": shape,
                "wave": {
                    "min": float(bundle.wave[0]),
                    "max": float(bundle.wave[-1]),
                    "count": bundle.wave.size,
                },
                "has_error": bundle.error is not None,
                "has_mask": bundle.mask is not None,
                "has_gas": bundle.gas is not None,
                "map_payload": map_payload,
                "default_slice_index": default_slice_index,
                "extnames": bundle.extnames,
                "header_summary": bundle.header_summary,
            }
        )

    @app.get("/api/map")
    def api_map():
        relative_path = request.args.get("path")
        bundle = load_observation(data_dir, relative_path)
        mode = request.args.get("mode", "overview")

        if bundle.mode == "cube":
            if mode == "slice":
                index = int(request.args.get("index", bundle.n_wave // 2))
                index = max(0, min(index, bundle.n_wave - 1))
                payload = serialize_image(bundle.flux[index])
                payload["index"] = index
                payload["wavelength"] = float(bundle.wave[index])
                payload["label"] = f"Slice en {bundle.wave[index]:.1f} A"
                return jsonify(payload)
            payload = serialize_image(bundle.overview_map)
            payload["label"] = "Mapa medio del cubo"
            return jsonify(payload)

        if mode == "slice":
            index = int(request.args.get("index", bundle.n_wave // 2))
            return jsonify(rss_map_payload(bundle, wave_index=index))
        return jsonify(rss_map_payload(bundle, wave_index=None))

    @app.get("/api/spectrum")
    def api_spectrum():
        relative_path = request.args.get("path")
        bundle = load_observation(data_dir, relative_path)

        if bundle.mode == "cube":
            x = int(request.args.get("x"))
            y = int(request.args.get("y"))
            if not (0 <= x < bundle.nx and 0 <= y < bundle.ny):
                abort(400, "El spaxel pedido esta fuera del rango del cubo")
            flux = bundle.flux[:, y, x]
            error = None if bundle.error is None else bundle.error[:, y, x]
            gas = None if bundle.gas is None else bundle.gas[:, y, x]
            mask = None if bundle.mask is None else bundle.mask[:, y, x]
            coords = {
                "x_grid": None if bundle.x_grid is None else float(bundle.x_grid[x]),
                "y_grid": None if bundle.y_grid is None else float(bundle.y_grid[y]),
            }
            label = f"Spaxel ({x}, {y})"
            selection = {"kind": "spaxel", "x": x, "y": y}
        else:
            index = int(request.args.get("index"))
            if not (0 <= index < bundle.n_sample):
                abort(400, "La fibra pedida esta fuera del rango del RSS")
            flux = bundle.flux[index]
            error = None if bundle.error is None else bundle.error[index]
            gas = None
            mask = None if bundle.mask is None else bundle.mask[index]
            coords = {
                "x_ifu": float(np.nanmedian(bundle.xpos[index])),
                "y_ifu": float(np.nanmedian(bundle.ypos[index])),
            }
            label = f"Fibra {index}"
            selection = {"kind": "fiber", "index": index}

        valid = np.isfinite(flux)
        finite_flux = flux[valid]
        stats = {
            "min": None if finite_flux.size == 0 else float(np.nanmin(finite_flux)),
            "max": None if finite_flux.size == 0 else float(np.nanmax(finite_flux)),
            "mean": None if finite_flux.size == 0 else float(np.nanmean(finite_flux)),
            "sum": None if finite_flux.size == 0 else float(np.nansum(finite_flux)),
            "masked_count": 0 if mask is None else int(np.count_nonzero(mask)),
        }

        return jsonify(
            {
                "mode": bundle.mode,
                "label": label,
                "selection": selection,
                "wave": bundle.wave.tolist(),
                "flux": np.nan_to_num(flux, nan=0.0).tolist(),
                "error": None if error is None else np.nan_to_num(error, nan=0.0).tolist(),
                "gas": None if gas is None else np.nan_to_num(gas, nan=0.0).tolist(),
                "mask": None if mask is None else np.asarray(mask).tolist(),
                "coords": coords,
                "stats": stats,
            }
        )

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Visor local para comparar cubos MaNGIA mock con datos reales de MaNGA"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host del servidor")
    parser.add_argument("--port", type=int, default=8010, help="Puerto del servidor")
    parser.add_argument("--debug", action="store_true", help="Activa modo debug")
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directorio donde buscar cubos y RSS para comparar",
    )
    args = parser.parse_args()

    app = create_app(args.data_dir)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
