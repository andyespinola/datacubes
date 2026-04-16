from __future__ import annotations

from pathlib import Path

from astropy.io import fits
import numpy as np
from scipy.ndimage import binary_closing, gaussian_filter, label

from .constants import AXIS_VIEWS
from .config import LabelConfig
from .models import CubeGeometry


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("No puedo normalizar un vector nulo")
    return vector / norm


def rotation_matrix_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = normalize(source)
    target = normalize(target)
    v = np.cross(source, target)
    c = float(np.dot(source, target))
    if np.isclose(c, 1.0):
        return np.eye(3)
    if np.isclose(c, -1.0):
        axis = np.array([1.0, 0.0, 0.0])
        if np.allclose(source, axis):
            axis = np.array([0.0, 1.0, 0.0])
        v = normalize(np.cross(source, axis))
        vx = skew(v)
        return np.eye(3) + 2.0 * vx @ vx
    vx = skew(v)
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def icosahedron_positive_x_views() -> np.ndarray:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array(
        [
            [0.0, 1.0, phi],
            [0.0, 1.0, -phi],
            [0.0, -1.0, phi],
            [0.0, -1.0, -phi],
            [1.0, phi, 0.0],
            [1.0, -phi, 0.0],
            [-1.0, phi, 0.0],
            [-1.0, -phi, 0.0],
            [phi, 0.0, 1.0],
            [phi, 0.0, -1.0],
            [-phi, 0.0, 1.0],
            [-phi, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    vertices = np.array([normalize(v) for v in vertices])
    seed = vertices[np.argmax(vertices[:, 0])]
    rot = rotation_matrix_from_vectors(seed, np.array([1.0, 0.0, 0.0]))
    rotated = (rot @ vertices.T).T
    selected = rotated[rotated[:, 0] >= -1e-8]
    selected = np.array(sorted(selected, key=lambda row: (-row[0], -row[1], -row[2])))
    selected = np.array([normalize(v) for v in selected[:6]])
    return selected


def view_vector_from_index(view: int, repeat_count: int) -> np.ndarray:
    if repeat_count <= 3:
        if view >= len(AXIS_VIEWS):
            raise ValueError(f"Vista {view} inválida para repeat_count={repeat_count}")
        return np.asarray(AXIS_VIEWS[view], dtype=np.float64)
    views = icosahedron_positive_x_views()
    if view >= len(views):
        raise ValueError(f"Vista {view} inválida para el conjunto isotrópico de 6 vistas")
    return views[view]


def center_and_rotate_faceon(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    center: np.ndarray,
    systemic_velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered_pos = positions - center[None, :]
    centered_vel = velocities - systemic_velocity[None, :]
    ang_momentum = np.sum(np.cross(centered_pos, centered_vel) * masses[:, None], axis=0)
    if np.linalg.norm(ang_momentum) == 0:
        rotation = np.eye(3)
    else:
        rotation = rotation_matrix_from_vectors(ang_momentum, np.array([0.0, 0.0, 1.0]))
    return (rotation @ centered_pos.T).T, (rotation @ centered_vel.T).T, rotation


def project_positions(positions: np.ndarray, view_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    los = normalize(view_vector)
    trial = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if np.allclose(np.abs(np.dot(los, trial)), 1.0):
        trial = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = normalize(np.cross(trial, los))
    y_axis = normalize(np.cross(los, x_axis))
    x = positions @ x_axis
    y = positions @ y_axis
    z = positions @ los
    return x, y, z


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    if cdf[-1] <= 0:
        return float(np.nanmedian(values))
    threshold = quantile * cdf[-1]
    idx = np.searchsorted(cdf, threshold, side="left")
    idx = min(max(idx, 0), len(values) - 1)
    return float(values[idx])


def largest_connected_component(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    labeled, n_labels = label(mask)
    if n_labels == 0:
        return np.zeros_like(mask, dtype=bool)

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = np.zeros_like(mask, dtype=bool)
    for label_id in np.argsort(counts)[::-1]:
        if label_id == 0 or counts[label_id] < min_pixels:
            continue
        component = labeled == label_id
        keep |= component
        if keep.any():
            break
    if not keep.any():
        label_id = int(np.argmax(counts))
        keep = labeled == label_id
    return keep


def build_strict_valid_mask(
    cube: np.ndarray,
    base_valid_mask: np.ndarray,
    config: LabelConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    signal_map = np.nanmean(np.abs(cube), axis=0).astype(np.float32)
    signal_map = np.where(np.isfinite(signal_map), signal_map, 0.0)
    smooth_map = gaussian_filter(signal_map, sigma=config.valid_smoothing_sigma_px, mode="nearest")

    candidate = smooth_map[base_valid_mask]
    candidate = candidate[np.isfinite(candidate)]
    if candidate.size == 0:
        return base_valid_mask.copy(), smooth_map.astype(np.float32), 0.0

    percentile_floor = float(np.nanpercentile(candidate, config.valid_flux_percentile))
    peak_floor = float(np.nanmax(candidate)) * config.valid_peak_fraction
    threshold = max(percentile_floor, peak_floor, 0.0)

    strict_mask = base_valid_mask & (smooth_map >= threshold)
    if not np.any(strict_mask):
        return base_valid_mask.copy(), smooth_map.astype(np.float32), threshold

    structure = np.ones((3, 3), dtype=bool)
    if int(config.valid_closing_iterations) > 0:
        strict_mask = binary_closing(
            strict_mask,
            structure=structure,
            iterations=int(config.valid_closing_iterations),
        )
    strict_mask &= base_valid_mask
    strict_mask = largest_connected_component(strict_mask, max(int(config.valid_min_component_pixels), 1))
    if not np.any(strict_mask):
        strict_mask = base_valid_mask.copy()
    return strict_mask.astype(bool), smooth_map.astype(np.float32), threshold


def load_cube_geometry(cube_path: str | Path, config: LabelConfig | None = None) -> CubeGeometry:
    with fits.open(cube_path) as hdul:
        cube = np.asarray(hdul[0].data, dtype=np.float32)
        header = hdul[0].header
        if cube.ndim != 3:
            raise ValueError(f"Esperaba un cubo 3D en {cube_path}, pero encontré shape={cube.shape}")
        if "MASK" in hdul:
            mask_data = np.asarray(hdul["MASK"].data)
            spatial_mask = np.isfinite(mask_data).any(axis=0) & (np.nanmin(mask_data, axis=0) == 0)
        else:
            spatial_mask = np.isfinite(cube).any(axis=0)
        signal_mask = np.nanmax(np.abs(cube), axis=0) > 0
        base_valid_mask = spatial_mask & signal_mask
        if config is None:
            valid_mask = base_valid_mask.copy()
            signal_map = np.nanmean(np.abs(cube), axis=0).astype(np.float32)
            valid_threshold = 0.0
        else:
            valid_mask, signal_map, valid_threshold = build_strict_valid_mask(
                cube,
                base_valid_mask,
                config,
            )
        pixel_scale_arcsec = 0.5
        if "CD1_1" in header:
            pixel_scale_arcsec = abs(float(header["CD1_1"])) * 3600.0
        elif "CDELT1" in header:
            pixel_scale_arcsec = abs(float(header["CDELT1"])) * 3600.0
        kpc_per_arcsec = float(header.get("KPCSEC", 1.0))
        psf = float(header.get("PSF", 1.43))
        return CubeGeometry(
            shape=(cube.shape[1], cube.shape[2]),
            valid_mask=np.asarray(valid_mask, dtype=bool),
            base_valid_mask=np.asarray(base_valid_mask, dtype=bool),
            signal_map=np.asarray(signal_map, dtype=np.float32),
            valid_threshold=float(valid_threshold),
            pixel_scale_arcsec=pixel_scale_arcsec,
            kpc_per_arcsec=kpc_per_arcsec,
            psf_fwhm_arcsec=psf,
            header_summary={
                "shape": f"{cube.shape}",
                "psf": psf,
                "pixel_scale_arcsec": pixel_scale_arcsec,
                "kpc_per_arcsec": kpc_per_arcsec,
                "valid_pixels": int(np.count_nonzero(valid_mask)),
                "base_valid_pixels": int(np.count_nonzero(base_valid_mask)),
                "valid_threshold": float(valid_threshold),
            },
        )


def deposit_to_grid(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    shape: tuple[int, int],
    pixel_scale_kpc: float,
    sigma_pixels: float = 1.0,
) -> np.ndarray:
    h, w = shape
    x_edges = (np.arange(w + 1) - w / 2.0) * pixel_scale_kpc
    y_edges = (np.arange(h + 1) - h / 2.0) * pixel_scale_kpc
    hist, _, _ = np.histogram2d(y, x, bins=(y_edges, x_edges), weights=weights)
    if sigma_pixels > 0:
        hist = gaussian_filter(hist, sigma=sigma_pixels, mode="constant")
    return np.asarray(hist, dtype=np.float32)


def sample_grid_at_points(
    grid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    pixel_scale_kpc: float,
) -> np.ndarray:
    h, w = grid.shape
    ix = np.floor(x / pixel_scale_kpc + w / 2.0).astype(int)
    iy = np.floor(y / pixel_scale_kpc + h / 2.0).astype(int)
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    values = np.zeros_like(x, dtype=np.float64)
    values[valid] = grid[iy[valid], ix[valid]]
    return values
