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


def load_ssp_grid(template_path: str | Path) -> SSPGrid:
    flux, header = fits.getdata(template_path, 0, header=True)
    ages = []
    metallicities = []
    mass_to_light = []
    for idx in range(flux.shape[0]):
        name = header[f"NAME{idx}"]
        name = name.replace("spec_ssp_", "").replace(".spec", "").replace(".dat", "")
        age_token, metallicity_token = name.split("_")[:2]
        age_gyr = parse_age_token(age_token)
        metallicity = parse_metallicity_token(metallicity_token)
        norm = float(header[f"NORM{idx}"])
        ml = 1.0 / norm if norm != 0 else 1.0
        ages.append(age_gyr)
        metallicities.append(metallicity)
        mass_to_light.append(ml)

    ages_arr = np.asarray(ages, dtype=np.float64)
    met_arr = np.asarray(metallicities, dtype=np.float64)
    ml_arr = np.asarray(mass_to_light, dtype=np.float64)
    features = np.column_stack((np.log10(np.clip(ages_arr, 1e-4, None)), met_arr))
    tree = cKDTree(features)
    return SSPGrid(
        ages_gyr=ages_arr,
        metallicities=met_arr,
        mass_to_light=ml_arr,
        tree=tree,
    )


def parse_metallicity_token(token: str) -> float:
    cleaned = token.strip()
    if cleaned.startswith("z"):
        stripped = cleaned[1:]
        if "." not in stripped:
            cleaned = f"0.{stripped}"
        else:
            cleaned = stripped
    if cleaned.startswith("."):
        cleaned = f"0{cleaned}"
    return float(cleaned)


def parse_age_token(token: str) -> float:
    cleaned = token.strip()
    if cleaned.endswith("Myr"):
        return float(cleaned[:-3]) / 1000.0
    if cleaned.endswith("Gyr"):
        return float(cleaned[:-3])
    return float(cleaned)


def scale_factor_to_age_gyr(scale_factors: np.ndarray) -> np.ndarray:
    # The MaNGIA local code approximates stellar age using the observation
    # snapshot age and the stellar formation scale factor. Here we use a simple
    # monotonic proxy that is stable for labeling and M/L assignment.
    clipped = np.clip(scale_factors, 1e-4, 1.0)
    return np.maximum(1e-3, 13.8 * (1.0 - clipped))


def particle_light_weights(
    ssp_grid: SSPGrid,
    stellar_mass: np.ndarray,
    stellar_age_proxy: np.ndarray,
    stellar_metallicity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.column_stack(
        (
            np.log10(np.clip(stellar_age_proxy, 1e-4, None)),
            stellar_metallicity,
        )
    )
    _, indices = ssp_grid.tree.query(features, k=1)
    ml = np.clip(ssp_grid.mass_to_light[indices], 1e-6, None)
    return stellar_mass / ml, ml
