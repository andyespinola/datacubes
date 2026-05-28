from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

import numpy as np

from .ids import UnitKey, parse_galaxy_from_text, parse_unit_from_text
from .models import CubeAsset, MapAsset, TngAssets


VELOCITY_CANDIDATES = {
    "V",
    "VEL",
    "VELOCITY",
    "VSTAR",
    "VSTARS",
    "VSTELLAR",
    "STELLARV",
    "STELLARVELOCITY",
}
SIGMA_CANDIDATES = {
    "SIGMA",
    "SIGMASTAR",
    "SIGMASTARS",
    "SIGMASTELLAR",
    "STELLARSIGMA",
    "DISP",
    "DISPERSION",
}
MAP_SUFFIXES = (".npz", ".fits", ".fits.gz", ".h5", ".hdf5")
MORPHOLOGY_SUFFIXES = (".hdf5", ".h5", ".fits", ".fits.gz")
PIPE3D_STACK_EXTNAME = "SSP_pyPipe3D_REC"
PIPE3D_VLOS_INDEX = 13
PIPE3D_SIGMA_INDEX = 15


def shape_string(shape: tuple[int, ...] | list[int] | None) -> str:
    if not shape:
        return ""
    return "x".join(str(int(value)) for value in shape)


def _normalized_name(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def _best_key(names: Iterable[str], candidates: set[str]) -> str:
    original = list(names)
    normalized = {name: _normalized_name(name) for name in original}
    for name in original:
        if normalized[name] in candidates:
            return name
    for name in original:
        norm = normalized[name]
        if any(norm.endswith(candidate) for candidate in candidates):
            return name
    return ""


def detect_kinematic_keys(names: Iterable[str]) -> tuple[str, str]:
    names = list(names)
    return _best_key(names, VELOCITY_CANDIDATES), _best_key(names, SIGMA_CANDIDATES)


def _stack_key(dataset: str, index: int) -> str:
    return f"{dataset}[{int(index)}]"


def _pipe3d_stack_keys(dataset: str, shape: tuple[int, ...] | None) -> tuple[str, str, str]:
    if shape is None or len(shape) < 3 or int(shape[0]) <= PIPE3D_SIGMA_INDEX:
        return "", "", ""
    return (
        _stack_key(dataset, PIPE3D_VLOS_INDEX),
        _stack_key(dataset, PIPE3D_SIGMA_INDEX),
        shape_string(tuple(shape[1:])),
    )


def _is_cube_path(path: Path) -> bool:
    name = path.name
    return (name.endswith(".cube.fits") or name.endswith(".cube.fits.gz")) and ".cube_val." not in name


def _read_fits_shape(path: Path) -> str:
    try:
        from astropy.io import fits
    except Exception:
        return ""
    try:
        with fits.open(path, memmap=False) as hdul:
            data_shape = None if hdul[0].data is None else tuple(int(value) for value in hdul[0].data.shape)
            if data_shape is None:
                return ""
            if len(data_shape) >= 2:
                return shape_string(data_shape[-2:])
            return shape_string(data_shape)
    except Exception:
        return ""


def scan_cubes(roots: Iterable[str | Path]) -> dict[UnitKey, CubeAsset]:
    assets: dict[UnitKey, CubeAsset] = {}
    for root in roots:
        root_path = Path(root).expanduser()
        if not root_path.exists():
            continue
        paths = [root_path] if root_path.is_file() else sorted(root_path.rglob("*.cube.fits*"))
        for path in paths:
            if not _is_cube_path(path):
                continue
            parsed = parse_unit_from_text(path.name)
            if parsed is None:
                continue
            asset = CubeAsset(
                key=parsed.key,
                path=path.resolve(),
                ifu_design=parsed.ifu_design,
                shape=_read_fits_shape(path),
            )
            existing = assets.get(parsed.key)
            if existing is None or str(asset.path) < str(existing.path):
                assets[parsed.key] = asset
    return assets


def _register_tng_path(
    path: Path,
    by_unit: dict[UnitKey, Path],
    by_galaxy: dict[tuple[int, int], Path],
) -> None:
    parsed = parse_unit_from_text(path.name)
    resolved = path.resolve()
    if parsed is not None:
        existing = by_unit.get(parsed.key)
        if existing is None or str(resolved) < str(existing):
            by_unit[parsed.key] = resolved
        return
    galaxy = parse_galaxy_from_text(path.name)
    if galaxy is None:
        return
    key = (galaxy.snapshot, galaxy.subhalo_id)
    existing = by_galaxy.get(key)
    if existing is None or str(resolved) < str(existing):
        by_galaxy[key] = resolved


def scan_tng_cache(tng_cache: str | Path) -> TngAssets:
    root = Path(tng_cache).expanduser()
    cutouts_by_unit: dict[UnitKey, Path] = {}
    metadata_by_unit: dict[UnitKey, Path] = {}
    cutouts_by_galaxy: dict[tuple[int, int], Path] = {}
    metadata_by_galaxy: dict[tuple[int, int], Path] = {}

    cutout_dir = root / "cutouts"
    if cutout_dir.exists():
        for path in sorted(cutout_dir.rglob("*.cutout.hdf5")) + sorted(cutout_dir.rglob("*.cutout.h5")):
            _register_tng_path(path, cutouts_by_unit, cutouts_by_galaxy)

    metadata_dir = root / "metadata"
    if metadata_dir.exists():
        for path in sorted(metadata_dir.rglob("*.subhalo.json")) + sorted(metadata_dir.rglob("*.json")):
            _register_tng_path(path, metadata_by_unit, metadata_by_galaxy)

    morphology_catalog_path: Path | None = None
    morphology_dir = root / "morphology"
    if morphology_dir.exists():
        candidates = [
            path.resolve()
            for path in sorted(morphology_dir.rglob("*"))
            if path.is_file() and any(path.name.endswith(suffix) for suffix in MORPHOLOGY_SUFFIXES)
        ]
        morphology_catalog_path = candidates[0] if candidates else None

    return TngAssets(
        cutouts_by_unit=cutouts_by_unit,
        metadata_by_unit=metadata_by_unit,
        cutouts_by_galaxy=cutouts_by_galaxy,
        metadata_by_galaxy=metadata_by_galaxy,
        morphology_catalog_path=morphology_catalog_path,
    )


def _npz_map_asset(path: Path) -> MapAsset:
    parsed = parse_unit_from_text(path.name)
    ifu_design = parsed.ifu_design if parsed else None
    try:
        with np.load(path, allow_pickle=False) as data:
            names = list(data.files)
            v_key, sigma_key = detect_kinematic_keys(names)
            shape = shape_string(tuple(data[v_key].shape)) if v_key else ""
    except Exception as exc:
        return MapAsset(
            parsed.key if parsed else None,
            path.resolve(),
            "npz",
            message=f"{type(exc).__name__}: {exc}",
            ifu_design=ifu_design,
        )
    return MapAsset(
        parsed.key if parsed else None,
        path.resolve(),
        "npz",
        v_key,
        sigma_key,
        shape,
        ifu_design=ifu_design,
    )


def _fits_map_asset(path: Path) -> MapAsset:
    parsed = parse_unit_from_text(path.name)
    ifu_design = parsed.ifu_design if parsed else None
    try:
        from astropy.io import fits
    except Exception as exc:
        return MapAsset(
            parsed.key if parsed else None,
            path.resolve(),
            "fits",
            message=f"astropy unavailable: {exc}",
            ifu_design=ifu_design,
        )
    try:
        with fits.open(path, memmap=False) as hdul:
            names = [hdu.name for hdu in hdul if getattr(hdu, "data", None) is not None]
            v_key, sigma_key = detect_kinematic_keys(names)
            shape = ""
            for hdu in hdul:
                if hdu.name == v_key and getattr(hdu, "data", None) is not None:
                    shape = shape_string(tuple(hdu.data.shape))
                    break
            if not (v_key and sigma_key):
                for hdu in hdul:
                    data = getattr(hdu, "data", None)
                    if data is None:
                        continue
                    extname = str(hdu.header.get("EXTNAME", hdu.name)).strip() or str(hdu.name)
                    data_shape = tuple(int(value) for value in data.shape)
                    if _normalized_name(extname) == _normalized_name(PIPE3D_STACK_EXTNAME):
                        v_key, sigma_key, shape = _pipe3d_stack_keys(extname, data_shape)
                        if v_key and sigma_key:
                            break
    except Exception as exc:
        return MapAsset(
            parsed.key if parsed else None,
            path.resolve(),
            "fits",
            message=f"{type(exc).__name__}: {exc}",
            ifu_design=ifu_design,
        )
    return MapAsset(
        parsed.key if parsed else None,
        path.resolve(),
        "fits",
        v_key,
        sigma_key,
        shape,
        ifu_design=ifu_design,
    )


def _hdf5_map_asset(path: Path) -> MapAsset:
    parsed = parse_unit_from_text(path.name)
    ifu_design = parsed.ifu_design if parsed else None
    try:
        import h5py
    except Exception as exc:
        return MapAsset(
            parsed.key if parsed else None,
            path.resolve(),
            "hdf5",
            message=f"h5py unavailable: {exc}",
            ifu_design=ifu_design,
        )
    names: list[str] = []
    shapes: dict[str, tuple[int, ...]] = {}
    try:
        with h5py.File(path, "r") as handle:
            def visitor(name: str, obj) -> None:
                if hasattr(obj, "shape"):
                    names.append(name)
                    shapes[name] = tuple(int(value) for value in obj.shape)

            handle.visititems(visitor)
        v_key, sigma_key = detect_kinematic_keys(names)
        shape = shape_string(shapes.get(v_key)) if v_key else ""
        if not (v_key and sigma_key):
            for name, data_shape in shapes.items():
                if _normalized_name(Path(name).name) == _normalized_name(PIPE3D_STACK_EXTNAME):
                    v_key, sigma_key, shape = _pipe3d_stack_keys(name, data_shape)
                    if v_key and sigma_key:
                        break
    except Exception as exc:
        return MapAsset(
            parsed.key if parsed else None,
            path.resolve(),
            "hdf5",
            message=f"{type(exc).__name__}: {exc}",
            ifu_design=ifu_design,
        )
    return MapAsset(
        parsed.key if parsed else None,
        path.resolve(),
        "hdf5",
        v_key,
        sigma_key,
        shape,
        ifu_design=ifu_design,
    )


def _inspect_map_file(path: Path) -> MapAsset:
    if path.name.endswith(".npz"):
        return _npz_map_asset(path)
    if path.name.endswith(".fits") or path.name.endswith(".fits.gz"):
        return _fits_map_asset(path)
    if path.name.endswith(".h5") or path.name.endswith(".hdf5"):
        return _hdf5_map_asset(path)
    parsed = parse_unit_from_text(path.name)
    return MapAsset(
        parsed.key if parsed else None,
        path.resolve(),
        "unknown",
        ifu_design=parsed.ifu_design if parsed else None,
    )


def _path_from_manifest_row(row: dict[str, str], manifest_path: Path) -> Path | None:
    for column in ("maps2d_path", "map_path", "path", "filename", "file"):
        value = row.get(column, "").strip()
        if value:
            path = Path(value).expanduser()
            return path if path.is_absolute() else (manifest_path.parent / path)
    return None


def _unit_from_manifest_row(row: dict[str, str], fallback_text: str) -> UnitKey | None:
    try:
        if row.get("snapshot") and row.get("subhalo_id") and row.get("view"):
            return UnitKey(int(row["snapshot"]), int(row["subhalo_id"]), int(row["view"]))
    except ValueError:
        pass
    parsed = parse_unit_from_text(fallback_text)
    return parsed.key if parsed else None


def _map_asset_from_manifest_row(row: dict[str, str], manifest_path: Path) -> MapAsset | None:
    path = _path_from_manifest_row(row, manifest_path)
    if path is None:
        return None
    parsed_path = parse_unit_from_text(str(path))
    ifu_design = parsed_path.ifu_design if parsed_path else None
    key = _unit_from_manifest_row(row, str(path))
    v_key = row.get("v_map_key", row.get("v_key", row.get("velocity_key", ""))).strip()
    sigma_key = row.get("sigma_map_key", row.get("sigma_key", row.get("dispersion_key", ""))).strip()
    if v_key and sigma_key:
        fmt = row.get("maps2d_format", row.get("format", "")).strip() or _format_from_path(path)
        shape = row.get("maps2d_shape", row.get("shape", "")).strip()
        return MapAsset(key, path.resolve(), fmt, v_key, sigma_key, shape, ifu_design=ifu_design)
    inspected = (
        _inspect_map_file(path)
        if path.exists()
        else MapAsset(key, path.resolve(), _format_from_path(path), message="path missing", ifu_design=ifu_design)
    )
    return MapAsset(
        key or inspected.key,
        inspected.path,
        inspected.format,
        inspected.v_map_key,
        inspected.sigma_map_key,
        inspected.shape,
        message=inspected.message,
        ifu_design=inspected.ifu_design if inspected.ifu_design is not None else ifu_design,
    )


def _format_from_path(path: Path) -> str:
    if path.name.endswith(".fits") or path.name.endswith(".fits.gz"):
        return "fits"
    if path.name.endswith(".h5") or path.name.endswith(".hdf5"):
        return "hdf5"
    if path.name.endswith(".npz"):
        return "npz"
    return path.suffix.lstrip(".")


def _manifest_assets(path: Path) -> list[MapAsset]:
    assets: list[MapAsset] = []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                asset = _map_asset_from_manifest_row(row, path)
                if asset is not None:
                    assets.append(asset)
    except UnicodeDecodeError:
        return []
    except Exception:
        return []
    return assets


def _asset_score(asset: MapAsset) -> tuple[int, str]:
    score = int(asset.key is not None) + int(asset.has_v) + int(asset.has_sigma)
    return score, str(asset.path)


def scan_maps(roots: Iterable[str | Path]) -> tuple[dict[UnitKey, MapAsset], dict[str, int]]:
    by_key: dict[UnitKey, MapAsset] = {}
    counters = {
        "map_files_total": 0,
        "map_files_id_unknown": 0,
        "map_files_without_v": 0,
        "map_files_without_sigma": 0,
    }
    for root in roots:
        root_path = Path(root).expanduser()
        if not root_path.exists():
            continue
        candidates = [root_path] if root_path.is_file() else sorted(root_path.rglob("*"))
        for path in candidates:
            if not path.is_file():
                continue
            assets: list[MapAsset]
            if path.name.endswith(".csv"):
                assets = _manifest_assets(path)
            elif any(path.name.endswith(suffix) for suffix in MAP_SUFFIXES):
                assets = [_inspect_map_file(path)]
            else:
                continue
            for asset in assets:
                counters["map_files_total"] += 1
                if asset.key is None:
                    counters["map_files_id_unknown"] += 1
                    continue
                if not asset.has_v:
                    counters["map_files_without_v"] += 1
                if not asset.has_sigma:
                    counters["map_files_without_sigma"] += 1
                existing = by_key.get(asset.key)
                if existing is None or _asset_score(asset) > _asset_score(existing):
                    by_key[asset.key] = asset
    return by_key, counters
