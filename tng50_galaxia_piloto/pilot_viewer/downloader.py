from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import requests

from .constants import (
    DEFAULT_CUTOUT_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_MORPHOLOGY_PATH,
    PILOT_SIMULATION,
    PILOT_SNAPSHOT,
    PILOT_SUBHALO_ID,
    RAW_DATA_DIR,
    SOURCE_CUTOUT_CANDIDATES,
    SOURCE_METADATA_CANDIDATES,
    SOURCE_MORPHOLOGY_CANDIDATES,
)


class DownloadError(RuntimeError):
    pass


def _first_existing(paths: tuple[Path, ...]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _download_stream(url: str, target: Path, api_key: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, headers={"API-Key": api_key}, timeout=120, stream=True)
    if response.status_code >= 400:
        raise DownloadError(f"Error descargando recurso TNG: {response.status_code} {response.text[:200]}")
    with target.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)


def _download_json(url: str, target: Path, api_key: str) -> None:
    response = requests.get(url, headers={"API-Key": api_key}, timeout=120)
    if response.status_code >= 400:
        raise DownloadError(f"Error descargando metadatos TNG: {response.status_code} {response.text[:200]}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(response.json(), indent=2, sort_keys=True))


def bootstrap_pilot_data(force_download: bool = False, api_key: str = "") -> dict[str, str]:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    api_key = api_key or os.environ.get("TNG_API_KEY", "")

    cutout_status = "existing"
    metadata_status = "existing"
    morphology_status = "existing"

    if force_download:
        if DEFAULT_CUTOUT_PATH.exists():
            DEFAULT_CUTOUT_PATH.unlink()
        if DEFAULT_METADATA_PATH.exists():
            DEFAULT_METADATA_PATH.unlink()

    if not DEFAULT_CUTOUT_PATH.exists():
        cached = None if force_download else _first_existing(SOURCE_CUTOUT_CANDIDATES)
        if cached is not None:
            _copy_file(cached, DEFAULT_CUTOUT_PATH)
            cutout_status = "copied_from_repo_cache"
        else:
            if not api_key:
                raise DownloadError("Falta TNG_API_KEY y tampoco existe el cutout piloto en el cache local")
            url = (
                f"https://www.tng-project.org/api/{PILOT_SIMULATION}/snapshots/{PILOT_SNAPSHOT}/subhalos/"
                f"{PILOT_SUBHALO_ID}/cutout.hdf5"
                "?stars=Coordinates,Velocities,Masses,GFM_StellarFormationTime,GFM_Metallicity"
                "&gas=Coordinates,Velocities,Masses,StarFormationRate,Density,InternalEnergy,ElectronAbundance,GFM_Metallicity"
            )
            _download_stream(url, DEFAULT_CUTOUT_PATH, api_key)
            cutout_status = "downloaded_from_tng"

    if not DEFAULT_METADATA_PATH.exists():
        cached = None if force_download else _first_existing(SOURCE_METADATA_CANDIDATES)
        if cached is not None:
            _copy_file(cached, DEFAULT_METADATA_PATH)
            metadata_status = "copied_from_repo_cache"
        else:
            if not api_key:
                raise DownloadError("Falta TNG_API_KEY y tampoco existe el metadata piloto en el cache local")
            url = (
                f"https://www.tng-project.org/api/{PILOT_SIMULATION}/snapshots/{PILOT_SNAPSHOT}/subhalos/{PILOT_SUBHALO_ID}"
            )
            _download_json(url, DEFAULT_METADATA_PATH, api_key)
            metadata_status = "downloaded_from_tng"

    if not DEFAULT_MORPHOLOGY_PATH.exists():
        cached = _first_existing(SOURCE_MORPHOLOGY_CANDIDATES)
        if cached is not None:
            _copy_file(cached, DEFAULT_MORPHOLOGY_PATH)
            morphology_status = "copied_from_repo_cache"
        else:
            morphology_status = "missing_optional_file"

    return {
        "cutout": str(DEFAULT_CUTOUT_PATH),
        "cutout_status": cutout_status,
        "metadata": str(DEFAULT_METADATA_PATH),
        "metadata_status": metadata_status,
        "morphology_catalog": str(DEFAULT_MORPHOLOGY_PATH),
        "morphology_status": morphology_status,
    }
