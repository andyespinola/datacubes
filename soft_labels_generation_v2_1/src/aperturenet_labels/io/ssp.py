from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from astropy.io import fits
import numpy as np
from scipy.spatial import cKDTree


@dataclass(slots=True)
class SSPGrid:
    ages_gyr: np.ndarray
    metallicities: np.ndarray
    mass_to_light: np.ndarray
    tree: cKDTree


def parse_age_token(token: str) -> float:
    cleaned = token.strip()
    if cleaned.endswith("Myr"):
        return float(cleaned[:-3]) / 1000.0
    if cleaned.endswith("Gyr"):
        return float(cleaned[:-3])
    return float(cleaned)


def parse_metallicity_token(token: str) -> float:
    cleaned = token.strip()
    if cleaned.startswith("z"):
        cleaned = cleaned[1:]
        if "." not in cleaned:
            cleaned = f"0.{cleaned}"
    if cleaned.startswith("."):
        cleaned = f"0{cleaned}"
    return float(cleaned)


def load_ssp_grid(template_path: str | Path) -> SSPGrid:
    flux, header = fits.getdata(template_path, 0, header=True)
    ages: list[float] = []
    metallicities: list[float] = []
    mass_to_light: list[float] = []
    for idx in range(flux.shape[0]):
        name = header[f"NAME{idx}"].replace("spec_ssp_", "").replace(".spec", "").replace(".dat", "")
        age_token, metallicity_token = name.split("_")[:2]
        norm = float(header[f"NORM{idx}"])
        ages.append(parse_age_token(age_token))
        metallicities.append(parse_metallicity_token(metallicity_token))
        mass_to_light.append(1.0 / norm if norm != 0.0 else 1.0)
    ages_arr = np.asarray(ages, dtype=np.float64)
    metals_arr = np.asarray(metallicities, dtype=np.float64)
    ml_arr = np.asarray(mass_to_light, dtype=np.float64)
    features = np.column_stack((np.log10(np.clip(ages_arr, 1.0e-4, None)), metals_arr))
    return SSPGrid(ages_arr, metals_arr, ml_arr, cKDTree(features))


def particle_light_weights(grid: SSPGrid, mass: np.ndarray, age_gyr: np.ndarray, metallicity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.column_stack((np.log10(np.clip(age_gyr, 1.0e-4, None)), metallicity))
    _, idx = grid.tree.query(features, k=1)
    ml = np.clip(grid.mass_to_light[idx], 1.0e-6, None)
    return np.asarray(mass / ml, dtype=np.float64), np.asarray(ml, dtype=np.float64)
