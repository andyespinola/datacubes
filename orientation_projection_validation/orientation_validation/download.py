from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import time
from typing import Any

import h5py
import requests

from .manifest import ProjectionManifestRow


STAR_FIELDS = "Coordinates,Velocities,Masses,GFM_StellarFormationTime,GFM_Metallicity"
GAS_FIELDS = "Coordinates,Velocities,Masses,StarFormationRate,Density,InternalEnergy,ElectronAbundance,GFM_Metallicity"


class DownloadError(RuntimeError):
    pass


@dataclass(slots=True)
class DownloadRecord:
    galaxy_id: str
    asset: str
    path: str
    status: str
    message: str = ""


def api_headers(api_key: str) -> dict[str, str]:
    return {"API-Key": api_key}


def cutout_url(snapshot: int, subhalo_id: int, include_gas: bool = True, simulation: str = "TNG50-1") -> str:
    query = f"stars={STAR_FIELDS}"
    if include_gas:
        query += f"&gas={GAS_FIELDS}"
    return (
        f"https://www.tng-project.org/api/{simulation}/snapshots/{snapshot}/subhalos/"
        f"{subhalo_id}/cutout.hdf5?{query}"
    )


def metadata_url(snapshot: int, subhalo_id: int, simulation: str = "TNG50-1") -> str:
    return f"https://www.tng-project.org/api/{simulation}/snapshots/{snapshot}/subhalos/{subhalo_id}"


def log_download_state(cache_dir: str | Path, record: DownloadRecord) -> None:
    path = Path(cache_dir) / "download_state.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def download_stream_atomic(
    url: str,
    output_path: str | Path,
    api_key: str,
    force: bool = False,
    timeout: int = 120,
    retries: int = 3,
    backoff_seconds: float = 5.0,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    if output_path.exists() and not force:
        return output_path
    if force and output_path.exists():
        output_path.unlink()
    if part_path.exists():
        part_path.unlink()

    response = None
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=api_headers(api_key), timeout=timeout, stream=True)
            if response.status_code < 500:
                break
            last_error = DownloadError(f"HTTP {response.status_code}: {response.text[:200]}")
        except requests.RequestException as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(backoff_seconds * attempt)
    if response is None:
        raise DownloadError(str(last_error))
    if response.status_code >= 400:
        raise DownloadError(f"HTTP {response.status_code}: {response.text[:200]}")
    with part_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    part_path.replace(output_path)
    return output_path


def download_json_atomic(
    url: str,
    output_path: str | Path,
    api_key: str,
    force: bool = False,
    timeout: int = 120,
    retries: int = 3,
    backoff_seconds: float = 5.0,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    if output_path.exists() and not force:
        return output_path
    if force and output_path.exists():
        output_path.unlink()
    if part_path.exists():
        part_path.unlink()

    response = None
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=api_headers(api_key), timeout=timeout)
            if response.status_code < 500:
                break
            last_error = DownloadError(f"HTTP {response.status_code}: {response.text[:200]}")
        except requests.RequestException as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(backoff_seconds * attempt)
    if response is None:
        raise DownloadError(str(last_error))
    if response.status_code >= 400:
        raise DownloadError(f"HTTP {response.status_code}: {response.text[:200]}")
    payload: Any = response.json()
    part_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    part_path.replace(output_path)
    return output_path


def validate_cutout(path: str | Path, include_gas: bool = True) -> None:
    required_star = {
        "Coordinates",
        "Velocities",
        "Masses",
        "GFM_StellarFormationTime",
        "GFM_Metallicity",
    }
    required_gas = {
        "Coordinates",
        "Velocities",
        "Masses",
        "StarFormationRate",
        "Density",
        "InternalEnergy",
        "ElectronAbundance",
        "GFM_Metallicity",
    }
    with h5py.File(path, "r") as handle:
        if "PartType4" not in handle:
            raise DownloadError(f"Cutout inválido sin PartType4: {path}")
        missing_star = sorted(required_star - set(handle["PartType4"].keys()))
        if missing_star:
            raise DownloadError(f"Cutout inválido sin campos estelares {missing_star}: {path}")
        if include_gas:
            if "PartType0" not in handle:
                raise DownloadError(f"Cutout inválido sin PartType0: {path}")
            missing_gas = sorted(required_gas - set(handle["PartType0"].keys()))
            if missing_gas:
                raise DownloadError(f"Cutout inválido sin campos de gas {missing_gas}: {path}")


def validate_metadata(path: str | Path) -> None:
    payload = json.loads(Path(path).read_text())
    for key in ("pos_x", "pos_y", "pos_z"):
        if key in payload:
            return
    if "pos" in payload or "SubhaloPos" in payload:
        return
    raise DownloadError(f"Metadatos inválidos sin posición del subhalo: {path}")


def validate_morphology_catalog(path: str | Path) -> None:
    with h5py.File(path, "r") as handle:
        if not any(name.startswith("Snapshot_") for name in handle.keys()):
            raise DownloadError(f"Catálogo morfológico inválido sin grupos Snapshot_*: {path}")


def cutout_path(cache_dir: str | Path, row: ProjectionManifestRow) -> Path:
    return Path(cache_dir) / "cutouts" / f"{row.galaxy_id}.cutout.hdf5"


def metadata_path(cache_dir: str | Path, row: ProjectionManifestRow) -> Path:
    return Path(cache_dir) / "metadata" / f"{row.galaxy_id}.subhalo.json"


def download_row_assets(
    row: ProjectionManifestRow,
    cache_dir: str | Path,
    api_key: str,
    include_gas: bool = True,
    force: bool = False,
    simulation: str = "TNG50-1",
) -> None:
    cache_dir = Path(cache_dir)
    c_path = cutout_path(cache_dir, row)
    m_path = metadata_path(cache_dir, row)
    try:
        download_stream_atomic(
            cutout_url(row.snapshot, row.subhalo_id, include_gas=include_gas, simulation=simulation),
            c_path,
            api_key,
            force=force,
        )
        validate_cutout(c_path, include_gas=include_gas)
        log_download_state(cache_dir, DownloadRecord(row.galaxy_id, "cutout", str(c_path), "ok"))
    except Exception as exc:
        log_download_state(cache_dir, DownloadRecord(row.galaxy_id, "cutout", str(c_path), "error", str(exc)))
        raise

    try:
        download_json_atomic(metadata_url(row.snapshot, row.subhalo_id, simulation=simulation), m_path, api_key, force=force)
        validate_metadata(m_path)
        log_download_state(cache_dir, DownloadRecord(row.galaxy_id, "metadata", str(m_path), "ok"))
    except Exception as exc:
        log_download_state(cache_dir, DownloadRecord(row.galaxy_id, "metadata", str(m_path), "error", str(exc)))
        raise


def ensure_morphology_catalog(
    cache_dir: str | Path,
    api_key: str,
    morphology_url: str = "",
    morphology_path: str | Path | None = None,
    force: bool = False,
) -> Path | None:
    cache_dir = Path(cache_dir)
    out_path = cache_dir / "morphology" / "morphs_kinematic_bars.hdf5"
    if morphology_path:
        source = Path(morphology_path)
        if not source.exists():
            raise DownloadError(f"No existe TNG_MORPHOLOGY_CATALOG_PATH={source}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if force or not out_path.exists():
            shutil.copy2(source, out_path)
        validate_morphology_catalog(out_path)
        return out_path
    if morphology_url:
        download_stream_atomic(morphology_url, out_path, api_key, force=force, timeout=300, retries=3, backoff_seconds=10.0)
        validate_morphology_catalog(out_path)
        return out_path
    return None
