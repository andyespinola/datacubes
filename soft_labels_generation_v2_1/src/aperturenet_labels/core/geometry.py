from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .constants import AXIS_VIEWS


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero vector")
    return np.asarray(vector, dtype=np.float64) / norm


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def rotation_matrix_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = normalize(source)
    target = normalize(target)
    cross = np.cross(source, target)
    dot = float(np.dot(source, target))
    if np.isclose(dot, 1.0):
        return np.eye(3, dtype=np.float64)
    if np.isclose(dot, -1.0):
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if np.allclose(source, axis):
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        v = normalize(np.cross(source, axis))
        vx = skew(v)
        return np.eye(3, dtype=np.float64) + 2.0 * vx @ vx
    vx = skew(cross)
    return np.eye(3, dtype=np.float64) + vx + vx @ vx * (1.0 / (1.0 + dot))


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
    vertices = np.array([normalize(row) for row in vertices])
    seed = vertices[np.argmax(vertices[:, 0])]
    rot = rotation_matrix_from_vectors(seed, np.array([1.0, 0.0, 0.0]))
    rotated = (rot @ vertices.T).T
    selected = rotated[rotated[:, 0] >= -1.0e-8]
    selected = np.array(sorted(selected, key=lambda row: (-row[0], -row[1], -row[2])))
    return np.array([normalize(row) for row in selected[:6]], dtype=np.float64)


def view_vector_from_index(view: int, repeat_count: int) -> np.ndarray:
    if repeat_count <= 3:
        if view >= len(AXIS_VIEWS):
            raise ValueError(f"Invalid view={view} for repeat_count={repeat_count}")
        return np.asarray(AXIS_VIEWS[view], dtype=np.float64)
    views = icosahedron_positive_x_views()
    if view >= len(views):
        raise ValueError(f"Invalid view={view} for isotropic 6-view set")
    return views[view]


def center_and_rotate_faceon(
    centered_positions: np.ndarray,
    centered_velocities: np.ndarray,
    masses: np.ndarray,
    central_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if central_mask is None or not np.any(central_mask):
        central_mask = np.ones(centered_positions.shape[0], dtype=bool)
    angular_momentum = np.sum(
        np.cross(centered_positions[central_mask], centered_velocities[central_mask]) * masses[central_mask, None],
        axis=0,
    )
    if np.linalg.norm(angular_momentum) == 0.0:
        rotation = np.eye(3, dtype=np.float64)
    else:
        rotation = rotation_matrix_from_vectors(angular_momentum, np.array([0.0, 0.0, 1.0]))
    return (
        (rotation @ centered_positions.T).T,
        (rotation @ centered_velocities.T).T,
        rotation,
        angular_momentum,
    )


def project_positions(positions: np.ndarray, view_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    los = normalize(view_vector)
    trial = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if np.allclose(abs(float(np.dot(los, trial))), 1.0):
        trial = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = normalize(np.cross(trial, los))
    y_axis = normalize(np.cross(los, x_axis))
    return positions @ x_axis, positions @ y_axis, positions @ los


def deposit_to_grid(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    shape: tuple[int, int],
    pixel_scale_kpc: float,
    sigma_pixels: float = 0.0,
) -> np.ndarray:
    h, w = shape
    x_edges = (np.arange(w + 1, dtype=np.float64) - w / 2.0) * pixel_scale_kpc
    y_edges = (np.arange(h + 1, dtype=np.float64) - h / 2.0) * pixel_scale_kpc
    hist, _, _ = np.histogram2d(y, x, bins=(y_edges, x_edges), weights=weights)
    if sigma_pixels > 0.0:
        hist = gaussian_filter(hist, sigma=sigma_pixels, mode="constant")
    return np.asarray(hist, dtype=np.float32)


def sample_grid_at_points(grid: np.ndarray, x: np.ndarray, y: np.ndarray, pixel_scale_kpc: float) -> np.ndarray:
    h, w = grid.shape
    ix = np.floor(x / pixel_scale_kpc + w / 2.0).astype(int)
    iy = np.floor(y / pixel_scale_kpc + h / 2.0).astype(int)
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    values = np.zeros_like(x, dtype=np.float64)
    values[valid] = grid[iy[valid], ix[valid]]
    return values


def pad_center(array: np.ndarray, target_shape: tuple[int, int], fill_value: float = 0.0) -> np.ndarray:
    source_shape = array.shape[-2:]
    if source_shape == target_shape:
        return array
    if source_shape[0] > target_shape[0] or source_shape[1] > target_shape[1]:
        raise ValueError(f"Cannot pad source_shape={source_shape} to smaller target_shape={target_shape}")
    output_shape = (*array.shape[:-2], *target_shape)
    out = np.full(output_shape, fill_value, dtype=array.dtype)
    y0 = (target_shape[0] - source_shape[0]) // 2
    x0 = (target_shape[1] - source_shape[1]) // 2
    out[..., y0 : y0 + source_shape[0], x0 : x0 + source_shape[1]] = array
    return out
