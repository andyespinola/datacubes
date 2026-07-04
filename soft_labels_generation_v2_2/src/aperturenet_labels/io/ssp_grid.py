"""Rejilla SSP (edad, Z) → M/L para pesos de luminosidad (portado de ssp.py v1)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree


@dataclass(slots=True)
class SSPGrid:
    ages_gyr: np.ndarray
    metallicities: np.ndarray
    mass_to_light: np.ndarray
    tree: cKDTree


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


def load_ssp_grid(template_path: str | Path) -> SSPGrid:
    flux, header = fits.getdata(template_path, 0, header=True)
    ages = []
    metallicities = []
    mass_to_light = []
    for idx in range(flux.shape[0]):
        name = header[f"NAME{idx}"]
        name = name.replace("spec_ssp_", "").replace(".spec", "").replace(".dat", "")
        age_token, metallicity_token = name.split("_")[:2]
        ages.append(parse_age_token(age_token))
        metallicities.append(parse_metallicity_token(metallicity_token))
        norm = float(header[f"NORM{idx}"])
        mass_to_light.append(1.0 / norm if norm != 0 else 1.0)

    ages_arr = np.asarray(ages, dtype=np.float64)
    met_arr = np.asarray(metallicities, dtype=np.float64)
    ml_arr = np.asarray(mass_to_light, dtype=np.float64)
    features = np.column_stack((np.log10(np.clip(ages_arr, 1e-4, None)), met_arr))
    return SSPGrid(
        ages_gyr=ages_arr,
        metallicities=met_arr,
        mass_to_light=ml_arr,
        tree=cKDTree(features),
    )


def particle_light_weights(
    ssp_grid: SSPGrid,
    stellar_mass: np.ndarray,
    stellar_age_gyr: np.ndarray,
    stellar_metallicity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """L_i = m_i / (M/L)(edad_i, Z_i), vecino más cercano en la rejilla SSP."""
    features = np.column_stack(
        (
            np.log10(np.clip(stellar_age_gyr, 1e-4, None)),
            stellar_metallicity,
        )
    )
    _, indices = ssp_grid.tree.query(features, k=1)
    ml = np.clip(ssp_grid.mass_to_light[indices], 1e-6, None)
    return stellar_mass / ml, ml
