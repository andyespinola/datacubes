"""Visor web de la segmentación estructural (pipeline v2).

Sirve los `dataset_entry_*.h5` finales: mapa de segmentación (clase dominante),
probabilidades por clase en sus 4 variantes, N_eff, M_valid, mapas pyPipe3D y
el espectro del cubo IFU por spaxel con su desglose de probabilidades.

Basado en la arquitectura de manga_compare_project/manga_compare_viewer.py.

Uso:
    python viewer/segmentation_viewer.py [--data-dir ../data/output/dataset_entries] [--port 8020]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from flask import Flask, abort, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DEFAULT_DATA_DIR = BASE_DIR.parent.parent / "data" / "output" / "dataset_entries"

CLASS_COLORS = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
LABEL_VARIANTS = {
    "mass": "Y_int_mass",
    "light": "Y_int_light",
    "mass_psf": "Y_int_mass_psf",
    "light_psf": "Y_int_light_psf",
}
PIPE3D_LABELS = {
    "v_star": "v★ [km/s]",
    "sigma_star": "σ★ [km/s]",
    "age_lw": "edad LW [log Gyr]",
    "metallicity_lw": "[Z/H] LW [dex]",
    "mass_density": "log Σ★ [M☉/spx]",
    "av": "Av [mag]",
}


@dataclass
class EntryBundle:
    path: Path
    galaxy_id: str
    view_id: int
    class_names: list[str]
    labels: dict[str, np.ndarray]  # variant -> (H, W, C)
    n_eff: np.ndarray
    m_valid: np.ndarray
    cube: np.ndarray  # (n_wave, H, W)
    wave: np.ndarray
    pipe3d: dict[str, np.ndarray]
    qa: dict
    meta: dict


def sanitize_image(image):
    array = np.asarray(image, dtype=np.float32)
    return np.where(np.isfinite(array), array, np.nan)


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


def serialize_image(image, vmin=None, vmax=None):
    clean = sanitize_image(image)
    if vmin is None or vmax is None:
        vmin, vmax = percentile_range(clean)
    return {
        "kind": "image",
        "data": np.nan_to_num(clean, nan=0.0).tolist(),
        "nan_mask": (~np.isfinite(sanitize_image(image))).tolist(),
        "vmin": vmin,
        "vmax": vmax,
    }


def serialize_segmentation(labels_hwc, m_valid, class_names):
    """Mapa categórico: índice de clase dominante + alpha = P máxima."""
    total = labels_hwc.sum(axis=-1)
    has_signal = total > 0
    argmax = labels_hwc.argmax(axis=-1)
    maxprob = labels_hwc.max(axis=-1)
    class_idx = np.where(has_signal, argmax, -1)
    return {
        "kind": "categorical",
        "class_index": class_idx.tolist(),
        "max_prob": np.nan_to_num(maxprob, nan=0.0).tolist(),
        "valid": m_valid.astype(bool).tolist(),
        "class_names": class_names,
        "colors": CLASS_COLORS,
    }


def discover_files(data_dir):
    data_dir = Path(data_dir).resolve()
    files = []
    if not data_dir.exists():
        return files
    for path in sorted(data_dir.glob("*.h5")):
        files.append({"path": path.name, "label": path.stem})
    return files


def resolve_data_path(data_dir, relative_path):
    if not relative_path:
        abort(400, "Falta el parametro 'path'")
    root = Path(data_dir).resolve()
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        abort(400, "La ruta queda fuera del directorio de datos")
    if not resolved.exists():
        abort(404, f"No existe: {relative_path}")
    return resolved


@lru_cache(maxsize=4)
def load_entry(data_dir: str, relative_path: str) -> EntryBundle:
    path = resolve_data_path(data_dir, relative_path)
    with h5py.File(path, "r") as f:
        class_names = [c.decode() if isinstance(c, bytes) else str(c) for c in f["labels/class_names"][:]]
        labels = {}
        for variant, dset in LABEL_VARIANTS.items():
            if f"labels/{dset}" in f:
                labels[variant] = f[f"labels/{dset}"][:]
        n_eff = f["labels/n_eff"][:] if "labels/n_eff" in f else np.zeros(labels["mass"].shape[:2])
        m_valid = f["masks/M_valid"][:]
        cube = f["inputs/cube_ifu"][:]
        wave = (
            f["inputs/wavelength"][:]
            if "inputs/wavelength" in f
            else np.arange(cube.shape[0], dtype=np.float64)
        )
        pipe3d = {}
        if "inputs/pipe3d_maps" in f:
            for name in f["inputs/pipe3d_maps"]:
                pipe3d[name] = f[f"inputs/pipe3d_maps/{name}"][:]
        qa = {}
        if "qa" in f:
            for k, v in f["qa"].attrs.items():
                try:
                    qa[k] = json.loads(v) if isinstance(v, str) and v.startswith(("{", "[")) else v
                except (json.JSONDecodeError, TypeError):
                    qa[k] = str(v)
        meta = {
            k: (v.tolist() if isinstance(v, np.ndarray) else (v.item() if isinstance(v, np.generic) else v))
            for k, v in f["metadata"].attrs.items()
        }
    return EntryBundle(
        path=path,
        galaxy_id=str(meta.get("galaxy_id", path.stem)),
        view_id=int(meta.get("view_id", 0)),
        class_names=class_names,
        labels=labels,
        n_eff=n_eff,
        m_valid=m_valid,
        cube=cube,
        wave=np.asarray(wave, dtype=np.float64),
        pipe3d=pipe3d,
        qa=qa,
        meta=meta,
    )


def _qa_native(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def create_app(data_dir):
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
    data_dir = str(Path(data_dir).resolve())

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/api/files")
    def api_files():
        return jsonify({"files": discover_files(data_dir)})

    @app.get("/api/entry")
    def api_entry():
        bundle = load_entry(data_dir, request.args.get("path"))
        h, w, n_classes = bundle.labels["mass"].shape
        return jsonify(
            {
                "path": request.args.get("path"),
                "galaxy_id": bundle.galaxy_id,
                "view_id": bundle.view_id,
                "class_names": bundle.class_names,
                "class_colors": CLASS_COLORS,
                "variants": list(bundle.labels.keys()),
                "pipe3d_maps": [
                    {"name": k, "label": PIPE3D_LABELS.get(k, k)} for k in bundle.pipe3d
                ],
                "shape": {"ny": h, "nx": w, "n_classes": n_classes, "n_wave": int(bundle.cube.shape[0])},
                "wave": {"min": float(bundle.wave[0]), "max": float(bundle.wave[-1])},
                "n_valid": int(bundle.m_valid.sum()),
                "qa": {k: _qa_native(v) for k, v in bundle.qa.items()},
                "meta": bundle.meta,
            }
        )

    @app.get("/api/map")
    def api_map():
        bundle = load_entry(data_dir, request.args.get("path"))
        layer = request.args.get("layer", "segmentation")
        variant = request.args.get("variant", "mass")
        if variant not in bundle.labels:
            abort(400, f"Variante desconocida: {variant}")
        labels = bundle.labels[variant]

        if layer == "segmentation":
            payload = serialize_segmentation(labels, bundle.m_valid, bundle.class_names)
            payload["label"] = f"Clase dominante · variante {variant}"
        elif layer == "class_prob":
            c = int(request.args.get("class_index", 0))
            if not (0 <= c < labels.shape[-1]):
                abort(400, "class_index fuera de rango")
            payload = serialize_image(labels[:, :, c], vmin=0.0, vmax=1.0)
            payload["label"] = f"P({bundle.class_names[c]}) · variante {variant}"
        elif layer == "n_eff":
            payload = serialize_image(np.log10(np.clip(bundle.n_eff, 1e-3, None)))
            payload["label"] = "log10 N_eff (Kish)"
        elif layer == "m_valid":
            payload = serialize_image(bundle.m_valid.astype(np.float32), vmin=0.0, vmax=1.0)
            payload["label"] = "M_valid"
        elif layer == "cube_overview":
            payload = serialize_image(np.nanmean(bundle.cube, axis=0))
            payload["label"] = "Cubo IFU · flujo medio"
        elif layer == "cube_slice":
            index = int(request.args.get("index", bundle.cube.shape[0] // 2))
            index = max(0, min(index, bundle.cube.shape[0] - 1))
            payload = serialize_image(bundle.cube[index])
            payload["index"] = index
            payload["label"] = f"Cubo IFU · {bundle.wave[index]:.1f} Å"
        elif layer.startswith("pipe3d:"):
            name = layer.split(":", 1)[1]
            if name not in bundle.pipe3d:
                abort(404, f"Mapa pyPipe3D desconocido: {name}")
            img = bundle.pipe3d[name]
            img = np.where(img == 0, np.nan, img)  # 0 = sin dato en pyPipe3D
            extra = ""
            if name == "v_star":
                # quitar la velocidad sistémica (cz) para ver el campo de rotación
                img = img - np.nanmedian(img)
                extra = " (− mediana)"
            payload = serialize_image(img)
            payload["label"] = f"pyPipe3D · {PIPE3D_LABELS.get(name, name)}{extra}"
        else:
            abort(400, f"Capa desconocida: {layer}")
        return jsonify(payload)

    @app.get("/api/spaxel")
    def api_spaxel():
        bundle = load_entry(data_dir, request.args.get("path"))
        h, w, _ = bundle.labels["mass"].shape
        x = int(request.args.get("x"))
        y = int(request.args.get("y"))
        if not (0 <= x < w and 0 <= y < h):
            abort(400, "Spaxel fuera de rango")

        probs = {
            variant: [float(v) for v in arr[y, x]] for variant, arr in bundle.labels.items()
        }
        flux = bundle.cube[:, y, x]
        pipe3d_values = {
            name: (None if not np.isfinite(arr[y, x]) or arr[y, x] == 0 else float(arr[y, x]))
            for name, arr in bundle.pipe3d.items()
        }
        return jsonify(
            {
                "x": x,
                "y": y,
                "valid": bool(bundle.m_valid[y, x]),
                "n_eff": float(bundle.n_eff[y, x]),
                "class_names": bundle.class_names,
                "probs": probs,
                "pipe3d": pipe3d_values,
                "wave": bundle.wave.tolist(),
                "flux": np.nan_to_num(flux, nan=0.0).tolist(),
                "flux_units": "1e-16 erg/s/cm²/Å",
            }
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="Visor web de segmentación estructural v2")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directorio con los dataset_entry_*.h5",
    )
    args = parser.parse_args()
    app = create_app(args.data_dir)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
