from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import requests

from .constants import TNG_SIMULATION
from .models import MorphologyTargets, TNGTruth


class TNGAPIError(RuntimeError):
    pass


def _api_headers(api_key: str) -> dict[str, str]:
    return {"API-Key": api_key}


def download_url(url: str, output_path: str | Path, api_key: str | None = None) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = _api_headers(api_key) if api_key else {}
    response = requests.get(url, headers=headers, timeout=120, stream=True)
    if response.status_code >= 400:
        raise TNGAPIError(f"Error descargando recurso TNG: {response.status_code} {response.text[:200]}")
    with output_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    return output_path


def download_cutout(
    snapshot: int,
    subhalo_id: int,
    output_path: str | Path,
    api_key: str,
    include_gas: bool = True,
    simulation: str = TNG_SIMULATION,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    star_fields = "Coordinates,Velocities,Masses,GFM_StellarFormationTime,GFM_Metallicity"
    gas_fields = "Coordinates,Velocities,Masses,StarFormationRate,Density,InternalEnergy,ElectronAbundance,GFM_Metallicity"
    query = f"stars={star_fields}"
    if include_gas:
        query += f"&gas={gas_fields}"
    url = (
        f"https://www.tng-project.org/api/{simulation}/snapshots/{snapshot}/subhalos/"
        f"{subhalo_id}/cutout.hdf5?{query}"
    )
    response = requests.get(url, headers=_api_headers(api_key), timeout=120, stream=True)
    if response.status_code >= 400:
        raise TNGAPIError(f"Error descargando cutout TNG: {response.status_code} {response.text[:200]}")
    with output_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    return output_path


def download_subhalo_metadata(
    snapshot: int,
    subhalo_id: int,
    output_path: str | Path,
    api_key: str,
    simulation: str = TNG_SIMULATION,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://www.tng-project.org/api/{simulation}/snapshots/{snapshot}/subhalos/{subhalo_id}"
    response = requests.get(url, headers=_api_headers(api_key), timeout=120)
    if response.status_code >= 400:
        raise TNGAPIError(f"Error descargando metadatos TNG: {response.status_code} {response.text[:200]}")
    output_path.write_text(json.dumps(response.json(), indent=2, sort_keys=True))
    return output_path


def _extract_json_vector(metadata: dict[str, Any], prefix: str) -> np.ndarray:
    if all(f"{prefix}_{axis}" in metadata for axis in ("x", "y", "z")):
        return np.asarray([metadata[f"{prefix}_x"], metadata[f"{prefix}_y"], metadata[f"{prefix}_z"]], dtype=np.float64)
    if prefix in metadata:
        return np.asarray(metadata[prefix], dtype=np.float64)
    candidates = [
        f"Subhalo{prefix.capitalize()}",
        f"subhalo_{prefix}",
        f"Subhalo{prefix}",
    ]
    for candidate in candidates:
        if candidate in metadata:
            return np.asarray(metadata[candidate], dtype=np.float64)
    raise KeyError(f"No encontré el vector {prefix} en los metadatos TNG")


def load_cutout_truth(
    cutout_path: str | Path,
    metadata_path: str | Path,
) -> TNGTruth:
    cutout_path = Path(cutout_path)
    metadata = json.loads(Path(metadata_path).read_text())
    with h5py.File(cutout_path, "r") as handle:
        stars = handle["PartType4"]
        gas = handle["PartType0"] if "PartType0" in handle else None
        formation_scale = np.asarray(stars["GFM_StellarFormationTime"], dtype=np.float64)
        valid = formation_scale >= 0
        gas_pos = np.asarray(gas["Coordinates"], dtype=np.float64) if gas is not None else None
        gas_vel = np.asarray(gas["Velocities"], dtype=np.float64) if gas is not None else None
        gas_mass = np.asarray(gas["Masses"], dtype=np.float64) if gas is not None else None
        gas_sfr = np.asarray(gas["StarFormationRate"], dtype=np.float64) if gas is not None else None
        gas_met = np.asarray(gas["GFM_Metallicity"], dtype=np.float64) if gas is not None else None
        gas_density = np.asarray(gas["Density"], dtype=np.float64) if gas is not None else None

        return TNGTruth(
            stellar_pos=np.asarray(stars["Coordinates"], dtype=np.float64)[valid],
            stellar_vel=np.asarray(stars["Velocities"], dtype=np.float64)[valid],
            stellar_mass=np.asarray(stars["Masses"], dtype=np.float64)[valid],
            stellar_age_gyr=formation_scale[valid],
            stellar_metallicity=np.asarray(stars["GFM_Metallicity"], dtype=np.float64)[valid],
            gas_pos=gas_pos,
            gas_vel=gas_vel,
            gas_mass=gas_mass,
            gas_sfr=gas_sfr,
            gas_metallicity=gas_met,
            gas_density=gas_density,
            subhalo_pos=_extract_json_vector(metadata, "pos"),
            subhalo_vel=_extract_json_vector(metadata, "vel"),
            stellar_halfmass_rad=float(
                metadata.get("halfmassrad_stars")
                or metadata.get("halfmassrad")
                or metadata.get("stellarhalfmassrad")
                or 0.0
            ),
        )


def load_morphology_targets(
    path: str | Path,
    snapshot: int,
    subhalo_id: int,
) -> MorphologyTargets:
    with h5py.File(path, "r") as handle:
        group = handle[f"Snapshot_{snapshot}"]
        ids = np.asarray(group["SubhaloID"], dtype=np.int64)
        indices = np.where(ids == subhalo_id)[0]
        if indices.size == 0:
            raise KeyError(f"No encontré SubhaloID={subhalo_id} en Snapshot_{snapshot} del catálogo morfológico")
        idx = int(indices[0])

        def scalar_from(dataset_name: str, row: int | None = None, default: float = 0.0) -> float:
            if dataset_name not in group:
                return default
            data = np.asarray(group[dataset_name])
            if row is None:
                value = data[idx]
            else:
                value = data[row, idx]
            return float(value)

        unbound = scalar_from("UnboundMass", default=0.0)
        return MorphologyTargets(
            thin_disk=scalar_from("ThinDisc", row=0),
            thick_disk=scalar_from("ThickDisc", row=0),
            pseudo_bulge=scalar_from("PseudoBulge", row=0),
            bulge=scalar_from("Bulge", row=0),
            halo=scalar_from("Halo", row=0),
            unbound=unbound,
            barred=bool(scalar_from("Barred") > 0),
            bar_size_kpc=max(0.0, scalar_from("BarSize", row=0, default=-1.0)),
            bar_size_alt_kpc=max(0.0, scalar_from("BarSize", row=1, default=-1.0)),
            bar_strength=max(0.0, scalar_from("BarStrength", row=0, default=-1.0)),
            bar_strength_alt=max(0.0, scalar_from("BarStrength", row=1, default=-1.0)),
            quality_krot=scalar_from("QualityFlags", row=0, default=0.0),
            quality_sigma_ratio=scalar_from("QualityFlags", row=1, default=0.0),
            quality_b1b2=scalar_from("QualityFlags", row=2, default=0.0),
        )


def morphology_targets_to_dict(targets: MorphologyTargets) -> dict[str, float | bool]:
    return asdict(targets)
