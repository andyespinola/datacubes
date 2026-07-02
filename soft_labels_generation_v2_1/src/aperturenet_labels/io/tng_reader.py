from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import h5py
import numpy as np


REQUIRED_STAR_FIELDS = {
    "Coordinates",
    "Velocities",
    "Masses",
    "GFM_StellarFormationTime",
    "GFM_Metallicity",
}

REQUIRED_GAS_FIELDS = {
    "Coordinates",
    "Velocities",
    "Masses",
    "StarFormationRate",
    "Density",
    "InternalEnergy",
    "ElectronAbundance",
    "GFM_Metallicity",
}


@dataclass(slots=True)
class SubhaloMetadata:
    snapshot: int
    subhalo_id: int
    pos: np.ndarray
    vel: np.ndarray
    halfmassrad_stars: float
    mass_stars: float


@dataclass(slots=True)
class TNGParticles:
    coordinates: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    formation_scale: np.ndarray
    metallicity: np.ndarray
    particle_ids: np.ndarray | None
    selected_indices: np.ndarray
    n_star_raw: int
    n_gas_raw: int
    has_gas: bool


def load_subhalo_metadata(path: str | Path) -> SubhaloMetadata:
    payload = json.loads(Path(path).read_text())

    def vector(prefix: str) -> np.ndarray:
        if all(f"{prefix}_{axis}" in payload for axis in ("x", "y", "z")):
            return np.asarray([payload[f"{prefix}_x"], payload[f"{prefix}_y"], payload[f"{prefix}_z"]], dtype=np.float64)
        if prefix in payload:
            return np.asarray(payload[prefix], dtype=np.float64)
        raise KeyError(f"Missing {prefix} vector in {path}")

    return SubhaloMetadata(
        snapshot=int(payload.get("snap", payload.get("snapshot", 0))),
        subhalo_id=int(payload.get("id", payload.get("subhalo_id", 0))),
        pos=vector("pos"),
        vel=vector("vel"),
        halfmassrad_stars=float(payload.get("halfmassrad_stars") or payload.get("halfmassrad") or 0.0),
        mass_stars=float(payload.get("mass_stars") or 0.0),
    )


def validate_cutout(path: str | Path) -> dict[str, int | bool | list[str]]:
    with h5py.File(path, "r") as handle:
        if "PartType4" not in handle:
            raise ValueError(f"Cutout without PartType4: {path}")
        missing_star = sorted(REQUIRED_STAR_FIELDS - set(handle["PartType4"].keys()))
        missing_gas: list[str] = []
        has_gas = "PartType0" in handle
        if has_gas:
            missing_gas = sorted(REQUIRED_GAS_FIELDS - set(handle["PartType0"].keys()))
        return {
            "n_star_raw": int(handle["PartType4/Coordinates"].shape[0]),
            "n_gas_raw": int(handle["PartType0/Coordinates"].shape[0]) if has_gas else 0,
            "has_gas": bool(has_gas),
            "missing_star": missing_star,
            "missing_gas": missing_gas,
        }


def load_stellar_particles(path: str | Path, max_particles: int = 0, seed: int = 42) -> TNGParticles:
    with h5py.File(path, "r") as handle:
        stars = handle["PartType4"]
        missing = sorted(REQUIRED_STAR_FIELDS - set(stars.keys()))
        if missing:
            raise ValueError(f"Cutout {path} missing stellar fields: {missing}")
        formation = np.asarray(stars["GFM_StellarFormationTime"], dtype=np.float64)
        valid = np.where(formation >= 0.0)[0]
        n_raw = int(stars["Coordinates"].shape[0])
        if max_particles and max_particles > 0 and valid.size > max_particles:
            rng = np.random.default_rng(seed)
            selected = np.sort(rng.choice(valid, size=int(max_particles), replace=False))
        else:
            selected = valid
        has_gas = "PartType0" in handle
        return TNGParticles(
            coordinates=np.asarray(stars["Coordinates"][selected], dtype=np.float64),
            velocities=np.asarray(stars["Velocities"][selected], dtype=np.float64),
            masses=np.asarray(stars["Masses"][selected], dtype=np.float64),
            formation_scale=formation[selected],
            metallicity=np.asarray(stars["GFM_Metallicity"][selected], dtype=np.float64),
            particle_ids=np.asarray(stars["ParticleIDs"][selected], dtype=np.uint64) if "ParticleIDs" in stars else None,
            selected_indices=np.asarray(selected, dtype=np.int64),
            n_star_raw=n_raw,
            n_gas_raw=int(handle["PartType0/Coordinates"].shape[0]) if has_gas else 0,
            has_gas=has_gas,
        )
