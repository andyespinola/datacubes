"""Lectura de cutouts TNG (portado de labeling/tng.py v1) + descarga API.

Extensiones v2:
- lee el Header del cutout (scale factor, redshift) para la conversión de unidades
- combina el cutout principal (gas+estrellas) con el cutout fase 2 (DM)
- soporta el campo Potential per-partícula si está presente o se descarga
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

from ..core.constants import DM_MASS_CODE_UNITS, TNG_SIMULATION
from ..schemas.models import TNGTruth


class TNGAPIError(RuntimeError):
    pass


class InsufficientResolutionError(RuntimeError):
    pass


def _api_headers(api_key: str) -> dict[str, str]:
    return {"API-Key": api_key}


def download_cutout_fields(
    snapshot: int,
    subhalo_id: int,
    output_path: str | Path,
    api_key: str,
    query: str,
    simulation: str = TNG_SIMULATION,
) -> Path:
    """Descarga un cutout con campos arbitrarios, e.g. `stars=Coordinates,Potential`."""
    import requests

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://www.tng-project.org/api/{simulation}/snapshots/{snapshot}/subhalos/"
        f"{subhalo_id}/cutout.hdf5?{query}"
    )
    response = requests.get(url, headers=_api_headers(api_key), timeout=600, stream=True)
    if response.status_code >= 400:
        raise TNGAPIError(
            f"Error descargando cutout TNG: {response.status_code} {response.text[:200]}"
        )
    with output_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    return output_path


def download_potential_cutout(
    snapshot: int,
    subhalo_id: int,
    output_path: str | Path,
    api_key: str,
    simulation: str = TNG_SIMULATION,
) -> Path:
    """Descarga Coordinates+Potential de estrellas (para ε canónica sin octree)."""
    return download_cutout_fields(
        snapshot,
        subhalo_id,
        output_path,
        api_key,
        query="stars=Coordinates,ParticleIDs,Potential",
        simulation=simulation,
    )


def _extract_json_vector(metadata: dict[str, Any], prefix: str) -> np.ndarray:
    if all(f"{prefix}_{axis}" in metadata for axis in ("x", "y", "z")):
        return np.asarray(
            [metadata[f"{prefix}_x"], metadata[f"{prefix}_y"], metadata[f"{prefix}_z"]],
            dtype=np.float64,
        )
    if prefix in metadata:
        return np.asarray(metadata[prefix], dtype=np.float64)
    raise KeyError(f"No encontré el vector {prefix} en los metadatos TNG")


def load_cutout_truth(
    cutout_path: str | Path,
    metadata_path: str | Path,
    phase2_path: str | Path | None = None,
    potential_path: str | Path | None = None,
) -> TNGTruth:
    """Carga partículas crudas (unidades de código). Conversión: io/units.py."""
    cutout_path = Path(cutout_path)
    metadata = json.loads(Path(metadata_path).read_text())

    with h5py.File(cutout_path, "r") as handle:
        header = dict(handle["Header"].attrs) if "Header" in handle else {}
        stars = handle["PartType4"]
        gas = handle["PartType0"] if "PartType0" in handle else None
        formation_scale = np.asarray(stars["GFM_StellarFormationTime"], dtype=np.float64)
        # GFM_StellarFormationTime < 0 marca celdas de viento, no estrellas reales.
        valid = formation_scale >= 0
        if int(valid.sum()) < 100:
            raise InsufficientResolutionError(
                f"{cutout_path.name}: {int(valid.sum())} partículas estelares (<100)"
            )

        stellar_potential = None
        if "Potential" in stars:
            stellar_potential = np.asarray(stars["Potential"], dtype=np.float64)[valid]

        star_ids = (
            np.asarray(stars["ParticleIDs"], dtype=np.uint64)[valid]
            if "ParticleIDs" in stars
            else None
        )

        truth_kwargs: dict[str, Any] = dict(
            stellar_pos=np.asarray(stars["Coordinates"], dtype=np.float64)[valid],
            stellar_vel=np.asarray(stars["Velocities"], dtype=np.float64)[valid],
            stellar_mass=np.asarray(stars["Masses"], dtype=np.float64)[valid],
            stellar_formation_scale=formation_scale[valid],
            stellar_metallicity=np.asarray(stars["GFM_Metallicity"], dtype=np.float64)[valid],
            stellar_potential=stellar_potential,
            gas_pos=np.asarray(gas["Coordinates"], dtype=np.float64) if gas is not None else None,
            gas_vel=np.asarray(gas["Velocities"], dtype=np.float64) if gas is not None else None,
            gas_mass=np.asarray(gas["Masses"], dtype=np.float64) if gas is not None else None,
            gas_sfr=np.asarray(gas["StarFormationRate"], dtype=np.float64)
            if gas is not None and "StarFormationRate" in gas
            else None,
        )

    if phase2_path is not None and Path(phase2_path).exists():
        with h5py.File(phase2_path, "r") as handle:
            if "PartType1" in handle:
                dm_pos = np.asarray(handle["PartType1"]["Coordinates"], dtype=np.float64)
                truth_kwargs["dm_pos"] = dm_pos
                truth_kwargs["dm_mass"] = np.full(len(dm_pos), DM_MASS_CODE_UNITS)

    if potential_path is not None and Path(potential_path).exists():
        truth_kwargs["stellar_potential"] = _match_potential(
            potential_path, star_ids, truth_kwargs["stellar_pos"]
        )

    return TNGTruth(
        subhalo_pos=_extract_json_vector(metadata, "pos"),
        subhalo_vel=_extract_json_vector(metadata, "vel"),
        stellar_halfmass_rad=float(
            metadata.get("halfmassrad_stars") or metadata.get("halfmassrad") or 0.0
        ),
        snapshot=int(header.get("SnapshotNumber", metadata.get("snap", 0))),
        subhalo_id=int(header.get("CutoutID", metadata.get("id", 0))),
        scale_factor=float(header.get("Time", 0.0)),
        redshift=float(header.get("Redshift", 0.0)),
        **truth_kwargs,
    )


def _match_potential(
    potential_path: str | Path,
    star_ids: Optional[np.ndarray],
    stellar_pos: np.ndarray,
) -> np.ndarray:
    """Asocia el Potential descargado a las partículas del cutout principal.

    El cutout principal no trae ParticleIDs; se asume el mismo orden de
    partículas que el cutout de Potential (la API devuelve orden estable
    para el mismo subhalo). Verifica por coordenadas.
    """
    with h5py.File(potential_path, "r") as handle:
        stars = handle["PartType4"]
        pot = np.asarray(stars["Potential"], dtype=np.float64)
        pos = np.asarray(stars["Coordinates"], dtype=np.float64)
    if len(pot) < len(stellar_pos):
        raise ValueError(
            f"Potential cutout tiene {len(pot)} partículas, esperaba >= {len(stellar_pos)}"
        )
    # El cutout principal filtró vientos (formation_time<0); el de Potential no.
    # Match posicional: ambas descargas comparten orden de partículas del snapshot.
    if len(pot) == len(stellar_pos):
        if not np.allclose(pos[: min(1000, len(pos))], stellar_pos[: min(1000, len(pos))]):
            raise ValueError("Potential cutout no alinea posicionalmente con el principal")
        return pot
    # Resolver por coincidencia exacta de coordenadas (float64 idénticos).
    index = {tuple(p): i for i, p in enumerate(np.round(pos, 6))}
    matched = np.empty(len(stellar_pos), dtype=np.float64)
    for j, p in enumerate(np.round(stellar_pos, 6)):
        i = index.get(tuple(p))
        if i is None:
            raise ValueError(f"Partícula {j} sin match en Potential cutout")
        matched[j] = pot[i]
    return matched
