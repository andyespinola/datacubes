from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from .constants import TNG_HUBBLE, TNG_OMEGA_B, TNG_OMEGA_M


COSMOLOGY = FlatLambdaCDM(H0=TNG_HUBBLE * 100.0, Om0=TNG_OMEGA_M, Ob0=TNG_OMEGA_B)


@dataclass(frozen=True)
class AxisBundle:
    x: np.ndarray
    y: np.ndarray
    label_x: str
    label_y: str


def universe_age_gyr(redshift: float) -> float:
    return float(COSMOLOGY.age(redshift).value)


def stellar_age_gyr_from_scale_factor(scale_factor: np.ndarray, snapshot_redshift: float) -> np.ndarray:
    age_now = universe_age_gyr(snapshot_redshift)
    grid = np.linspace(0.02, 1.0, 4096, dtype=np.float64)
    birth_redshift = (1.0 / grid) - 1.0
    birth_age = COSMOLOGY.age(birth_redshift).value
    particle_birth_age = np.interp(scale_factor, grid, birth_age)
    ages = age_now - particle_birth_age
    return np.clip(ages, a_min=0.0, a_max=None).astype(np.float32)


def safe_unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("No puedo normalizar un vector nulo")
    return vector / norm


def weighted_covariance(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    centered = points - np.average(points, axis=0, weights=weights)
    weighted = centered * weights[:, None]
    return centered.T @ weighted / np.sum(weights)


def build_rotation_matrix(positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
    angular_momentum = np.sum(np.cross(positions, velocities) * masses[:, None], axis=0)
    z_axis = safe_unit_vector(angular_momentum)

    reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(reference, z_axis)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    provisional_x = safe_unit_vector(reference - np.dot(reference, z_axis) * z_axis)
    projected = positions - np.outer(positions @ z_axis, z_axis)
    good = np.linalg.norm(projected, axis=1) > 0
    if np.count_nonzero(good) >= 8:
        cov = weighted_covariance(projected[good], masses[good])
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal = eigenvectors[:, np.argmax(eigenvalues)]
        principal = principal - np.dot(principal, z_axis) * z_axis
        if np.linalg.norm(principal) > 0:
            provisional_x = safe_unit_vector(principal)

    y_axis = safe_unit_vector(np.cross(z_axis, provisional_x))
    x_axis = safe_unit_vector(np.cross(y_axis, z_axis))
    return np.vstack([x_axis, y_axis, z_axis])


def rotate_positions(positions: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return positions @ rotation.T


def axes_for_view(rotated_positions: np.ndarray, view: str) -> AxisBundle:
    view = view.lower()
    if view == "faceon":
        return AxisBundle(
            x=rotated_positions[:, 0],
            y=rotated_positions[:, 1],
            label_x="x [kpc]",
            label_y="y [kpc]",
        )
    if view == "edgeon":
        return AxisBundle(
            x=rotated_positions[:, 0],
            y=rotated_positions[:, 2],
            label_x="x [kpc]",
            label_y="z [kpc]",
        )
    raise KeyError(f"Vista no soportada: {view}")


def histogram_surface_density(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    radius_kpc: float,
    bins: int,
) -> np.ndarray:
    hist, _, _ = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[[-radius_kpc, radius_kpc], [-radius_kpc, radius_kpc]],
        weights=weights,
    )
    pixel_size = (2.0 * radius_kpc) / float(bins)
    area = pixel_size * pixel_size
    density = hist.T / area
    density[density <= 0] = np.nan
    return density


def histogram_weighted_mean(
    x: np.ndarray,
    y: np.ndarray,
    value: np.ndarray,
    weights: np.ndarray,
    radius_kpc: float,
    bins: int,
) -> np.ndarray:
    weighted_sum, _, _ = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[[-radius_kpc, radius_kpc], [-radius_kpc, radius_kpc]],
        weights=value * weights,
    )
    total_weight, _, _ = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[[-radius_kpc, radius_kpc], [-radius_kpc, radius_kpc]],
        weights=weights,
    )
    mean = np.divide(
        weighted_sum.T,
        total_weight.T,
        out=np.full_like(weighted_sum.T, np.nan, dtype=np.float64),
        where=total_weight.T > 0,
    )
    return mean


def radial_surface_density_profile(radius: np.ndarray, weights: np.ndarray, max_radius: float, bins: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, max_radius, bins + 1, dtype=np.float64)
    totals, _ = np.histogram(radius, bins=edges, weights=weights)
    areas = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    values = totals / areas
    values[values <= 0] = np.nan
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, values


def radial_weighted_mean_profile(
    radius: np.ndarray,
    value: np.ndarray,
    weights: np.ndarray,
    max_radius: float,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, max_radius, bins + 1, dtype=np.float64)
    weighted_sum, _ = np.histogram(radius, bins=edges, weights=value * weights)
    total_weight, _ = np.histogram(radius, bins=edges, weights=weights)
    means = np.divide(
        weighted_sum,
        total_weight,
        out=np.full_like(weighted_sum, np.nan, dtype=np.float64),
        where=total_weight > 0,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, means


def finite_percentile_range(values: np.ndarray, low: float = 2.0, high: float = 98.0) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(finite, low))
    vmax = float(np.nanpercentile(finite, high))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def serializable_grid(values: np.ndarray) -> list[list[float | None]]:
    rows: list[list[float | None]] = []
    for row in np.asarray(values):
        rows.append([None if not np.isfinite(cell) else float(cell) for cell in row])
    return rows


def serializable_vector(values: np.ndarray) -> list[float | None]:
    return [None if not np.isfinite(value) else float(value) for value in np.asarray(values)]
