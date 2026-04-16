import argparse
from dataclasses import dataclass
from functools import lru_cache
import json
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


@dataclass
class LabelBundle:
    labels_path: Path
    summary_path: Path | None
    class_names: list[str]
    soft_mass: np.ndarray
    soft_light: np.ndarray
    hard_mass: np.ndarray
    hard_light: np.ndarray
    hard_mass_variants: dict[str, np.ndarray]
    hard_light_variants: dict[str, np.ndarray]
    hard_threshold_keys: list[str]
    confidence_mass: np.ndarray | None
    confidence_light: np.ndarray | None
    valid_mask: np.ndarray
    summary: dict | None

    def matches_cube(self, cube: CubeBundle):
        return self.valid_mask.shape == (cube.ny, cube.nx)


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


def sigma_from_ivar(ivar):
    ivar = np.asarray(ivar, dtype=np.float32)
    sigma = np.full_like(ivar, np.nan, dtype=np.float32)
    good = np.isfinite(ivar) & (ivar > 0)
    sigma[good] = 1.0 / np.sqrt(ivar[good])
    return sigma


def is_spectral_cube(path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with fits.open(path, memmap=True) as hdul:
                flux_hdu = None
                if hdul[0].data is not None and np.ndim(hdul[0].data) == 3:
                    flux_hdu = hdul[0]
                else:
                    candidate = get_named_hdu(hdul, "FLUX")
                    if candidate is not None and candidate.data is not None and np.ndim(candidate.data) == 3:
                        flux_hdu = candidate

                if flux_hdu is None:
                    return False

                if get_named_hdu(hdul, "WAVE") is not None:
                    return True

                header = flux_hdu.header
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


def extract_canonical_id(path: Path):
    name = path.name
    for suffix in (".cube.fits.gz", ".cube.fits", ".fits.gz", ".fits"):
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            if ".cube_RSS" in stem:
                stem = stem.split(".cube_RSS", 1)[0]
            return stem
    return path.stem


def find_first_existing(paths: list[Path]):
    for path in paths:
        if path.exists():
            return path
    return None


@lru_cache(maxsize=64)
def find_label_artifacts(relative_path):
    cube_path = resolve_cube_path(relative_path)
    canonical_id = extract_canonical_id(cube_path)

    direct_labels = [
        BASE_DIR / "structural_labeling" / "outputs" / f"{canonical_id}.labels.npz",
        BASE_DIR / f"{canonical_id}.labels.npz",
        cube_path.with_name(f"{canonical_id}.labels.npz"),
    ]
    labels_path = find_first_existing(direct_labels)
    if labels_path is None:
        matches = sorted(
            path
            for path in BASE_DIR.rglob(f"{canonical_id}.labels.npz")
            if "__pycache__" not in path.parts and STATIC_DIR not in path.parents
        )
        labels_path = matches[0] if matches else None

    summary_path = None
    if labels_path is not None:
        candidate = labels_path.with_name(
            labels_path.name.replace(".labels.npz", ".summary.json")
        )
        if candidate.exists():
            summary_path = candidate

    return {
        "canonical_id": canonical_id,
        "labels_path": labels_path,
        "summary_path": summary_path,
    }


def serialize_label_summary(summary):
    if summary is None:
        return None
    return json.loads(json.dumps(summary))


def named_probabilities(class_names, values):
    return {
        str(name): float(value)
        for name, value in zip(class_names, np.asarray(values, dtype=np.float32), strict=True)
    }


def threshold_label(key):
    try:
        return f"{int(key) / 100:.2f}"
    except (TypeError, ValueError):
        return str(key)


def hard_variant_modes(bundle: LabelBundle):
    modes = []
    for key in bundle.hard_threshold_keys:
        if key in bundle.hard_mass_variants:
            modes.append({"key": f"hard_mass_{key}", "label": f"Hard mass {threshold_label(key)}"})
        if key in bundle.hard_light_variants:
            modes.append({"key": f"hard_light_{key}", "label": f"Hard light {threshold_label(key)}"})
    return modes


def hard_plane_for_mode(bundle: LabelBundle, mode: str):
    if mode == "hard_mass":
        return bundle.hard_mass
    if mode == "hard_light":
        return bundle.hard_light
    if mode.startswith("hard_mass_"):
        key = mode.removeprefix("hard_mass_")
        if key in bundle.hard_mass_variants:
            return bundle.hard_mass_variants[key]
    if mode.startswith("hard_light_"):
        key = mode.removeprefix("hard_light_")
        if key in bundle.hard_light_variants:
            return bundle.hard_light_variants[key]
    return None


def named_hard_variants(class_names, variants, y, x):
    named = {}
    for key, labels in variants.items():
        index = int(labels[y, x])
        named[key] = {
            "index": index,
            "name": class_names[index],
            "threshold": threshold_label(key),
        }
    return named


@lru_cache(maxsize=16)
def load_label_bundle(relative_path):
    artifacts = find_label_artifacts(relative_path)
    labels_path = artifacts["labels_path"]
    if labels_path is None:
        return None

    with np.load(labels_path, allow_pickle=False) as data:
        class_names = [str(name) for name in data["class_names"].tolist()]
        confidence_mass = (
            None if "confidence_mass" not in data else np.asarray(data["confidence_mass"], dtype=np.float32)
        )
        confidence_light = (
            None if "confidence_light" not in data else np.asarray(data["confidence_light"], dtype=np.float32)
        )
        if "hard_threshold_keys" in data:
            hard_threshold_keys = [str(key) for key in data["hard_threshold_keys"].tolist()]
        else:
            hard_threshold_keys = []
        hard_mass_variants = {
            key: np.asarray(data[f"hard_mass_{key}"], dtype=np.int16)
            for key in hard_threshold_keys
            if f"hard_mass_{key}" in data
        }
        hard_light_variants = {
            key: np.asarray(data[f"hard_light_{key}"], dtype=np.int16)
            for key in hard_threshold_keys
            if f"hard_light_{key}" in data
        }
        bundle = LabelBundle(
            labels_path=labels_path,
            summary_path=artifacts["summary_path"],
            class_names=class_names,
            soft_mass=np.asarray(data["soft_mass"], dtype=np.float32),
            soft_light=np.asarray(data["soft_light"], dtype=np.float32),
            hard_mass=np.asarray(data["hard_mass"], dtype=np.int16),
            hard_light=np.asarray(data["hard_light"], dtype=np.int16),
            hard_mass_variants=hard_mass_variants,
            hard_light_variants=hard_light_variants,
            hard_threshold_keys=hard_threshold_keys,
            confidence_mass=confidence_mass,
            confidence_light=confidence_light,
            valid_mask=np.asarray(data["valid_mask"], dtype=bool),
            summary=None,
        )

    if bundle.summary_path is not None:
        bundle.summary = json.loads(bundle.summary_path.read_text())

    return bundle


def label_color(class_name):
    return {
        "no_valido": [120, 132, 146],
        "bulbo": [244, 162, 97],
        "disco": [90, 157, 255],
        "barra": [231, 111, 81],
        "brazos": [42, 157, 143],
        "other": [204, 185, 116],
        "incierto": [181, 131, 204],
        "incierto_otro": [181, 131, 204],
    }.get(class_name, [240, 240, 240])


@lru_cache(maxsize=4)
def load_cube_bundle(relative_path):
    path = resolve_cube_path(relative_path)
    with fits.open(path, memmap=False) as hdul:
        flux_hdu = hdul[0]
        if flux_hdu.data is None or np.ndim(flux_hdu.data) != 3:
            named_flux = get_named_hdu(hdul, "FLUX")
            if named_flux is None or named_flux.data is None or np.ndim(named_flux.data) != 3:
                raise ValueError("No encontré una extensión 3D de flujo compatible")
            flux_hdu = named_flux

        flux = np.asarray(flux_hdu.data, dtype=np.float32)
        if flux.ndim != 3:
            raise ValueError(f"El cubo debe ser 3D y tiene forma {flux.shape}")

        wave_hdu = get_named_hdu(hdul, "WAVE")
        if wave_hdu is not None:
            wave = np.asarray(wave_hdu.data, dtype=np.float64).reshape(-1)
        else:
            wave = infer_wave_from_header(flux_hdu.header, flux.shape[0])
            if wave is None:
                wave = np.arange(flux.shape[0], dtype=np.float64)

        x_hdu = get_named_hdu(hdul, "XGRID")
        y_hdu = get_named_hdu(hdul, "YGRID")
        x_grid = None if x_hdu is None else np.asarray(x_hdu.data, dtype=np.float64).reshape(-1)
        y_grid = None if y_hdu is None else np.asarray(y_hdu.data, dtype=np.float64).reshape(-1)

        error_hdu = get_named_hdu(hdul, "ERROR")
        ivar_hdu = get_named_hdu(hdul, "IVAR")
        mask_hdu = get_named_hdu(hdul, "MASK")
        gas_hdu = get_named_hdu(hdul, "GAS")

        if error_hdu is not None:
            error = np.asarray(error_hdu.data, dtype=np.float32)
        elif ivar_hdu is not None:
            error = sigma_from_ivar(ivar_hdu.data)
        else:
            error = None
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
        summary = header_summary(flux_hdu.header)
        primary_summary = header_summary(hdul[0].header)
        for key, value in primary_summary.items():
            summary.setdefault(key, value)

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
        artifacts = find_label_artifacts(relative_path)

        return jsonify(
            {
                "path": bundle.path.relative_to(BASE_DIR).as_posix(),
                "canonical_id": artifacts["canonical_id"],
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

    @app.get("/api/labels")
    def api_labels():
        relative_path = request.args.get("path")
        cube = load_cube_bundle(relative_path)
        artifacts = find_label_artifacts(relative_path)
        bundle = load_label_bundle(relative_path)

        if bundle is None:
            return jsonify(
                {
                    "available": False,
                    "canonical_id": artifacts["canonical_id"],
                    "reason": "No se encontraron artefactos de etiquetas para este cubo",
                }
            )

        if not bundle.matches_cube(cube):
            return jsonify(
                {
                    "available": False,
                    "canonical_id": artifacts["canonical_id"],
                    "reason": (
                        "Las etiquetas existen, pero su forma "
                        f"{bundle.valid_mask.shape} no coincide con el cubo {(cube.ny, cube.nx)}"
                    ),
                }
            )

        try:
            default_class_index = bundle.class_names.index("disco")
        except ValueError:
            default_class_index = 1 if len(bundle.class_names) > 1 else 0

        return jsonify(
            {
                "available": True,
                "canonical_id": artifacts["canonical_id"],
                "labels_path": bundle.labels_path.relative_to(BASE_DIR).as_posix(),
                "summary_path": (
                    None
                    if bundle.summary_path is None
                    else bundle.summary_path.relative_to(BASE_DIR).as_posix()
                ),
                "class_names": bundle.class_names,
                "modes": [
                    {"key": "off", "label": "Sin overlay"},
                    {"key": "soft_mass", "label": "Soft mass"},
                    {"key": "soft_light", "label": "Soft light"},
                    {"key": "hard_mass", "label": "Hard mass default"},
                    {"key": "hard_light", "label": "Hard light default"},
                ]
                + hard_variant_modes(bundle),
                "default_mode": "soft_mass",
                "default_class_index": default_class_index,
                "hard_threshold_keys": bundle.hard_threshold_keys,
                "valid_fraction": float(np.mean(bundle.valid_mask)),
                "summary": serialize_label_summary(bundle.summary),
            }
        )

    @app.get("/api/label-map")
    def api_label_map():
        relative_path = request.args.get("path")
        cube = load_cube_bundle(relative_path)
        bundle = load_label_bundle(relative_path)

        if bundle is None or not bundle.matches_cube(cube):
            abort(404, "No hay etiquetas compatibles para este cubo")

        mode = request.args.get("mode", "soft_mass")
        class_index = int(request.args.get("class_index", 0))
        if not (0 <= class_index < len(bundle.class_names)):
            abort(400, "class_index fuera de rango")

        if mode == "soft_mass":
            plane = np.asarray(bundle.soft_mass[class_index], dtype=np.float32)
        elif mode == "soft_light":
            plane = np.asarray(bundle.soft_light[class_index], dtype=np.float32)
        else:
            hard_labels = hard_plane_for_mode(bundle, mode)
            if hard_labels is None:
                abort(400, "Modo de etiquetas no soportado")
            plane = (hard_labels == class_index).astype(np.float32)

        if class_index != 0:
            plane = np.where(bundle.valid_mask, plane, 0.0)

        finite = plane[np.isfinite(plane)]
        stats = {
            "mean": 0.0 if finite.size == 0 else float(np.mean(finite)),
            "max": 0.0 if finite.size == 0 else float(np.max(finite)),
            "coverage": float(np.mean(plane > 0.01)),
        }

        return jsonify(
            {
                "mode": mode,
                "class_index": class_index,
                "class_name": bundle.class_names[class_index],
                "color": label_color(bundle.class_names[class_index]),
                "image": {
                    "data": np.nan_to_num(plane, nan=0.0).tolist(),
                    "vmin": 0.0,
                    "vmax": 1.0,
                },
                "stats": stats,
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
        labels = load_label_bundle(relative_path)

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
            "labels": {"available": False},
        }

        if labels is not None and labels.matches_cube(bundle):
            hard_mass_index = int(labels.hard_mass[y, x])
            hard_light_index = int(labels.hard_light[y, x])
            response["labels"] = {
                "available": True,
                "valid": bool(labels.valid_mask[y, x]),
                "class_names": labels.class_names,
                "soft_mass": named_probabilities(labels.class_names, labels.soft_mass[:, y, x]),
                "soft_light": named_probabilities(labels.class_names, labels.soft_light[:, y, x]),
                "hard_mass": {
                    "index": hard_mass_index,
                    "name": labels.class_names[hard_mass_index],
                },
                "hard_light": {
                    "index": hard_light_index,
                    "name": labels.class_names[hard_light_index],
                },
                "hard_mass_variants": named_hard_variants(
                    labels.class_names,
                    labels.hard_mass_variants,
                    y,
                    x,
                ),
                "hard_light_variants": named_hard_variants(
                    labels.class_names,
                    labels.hard_light_variants,
                    y,
                    x,
                ),
                "confidence_mass": (
                    None
                    if labels.confidence_mass is None
                    else float(labels.confidence_mass[y, x])
                ),
                "confidence_light": (
                    None
                    if labels.confidence_light is None
                    else float(labels.confidence_light[y, x])
                ),
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
