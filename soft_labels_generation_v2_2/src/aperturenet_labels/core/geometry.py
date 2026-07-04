"""Geometría portada ÍNTEGRA del v1 (structural_labeling/labeling/geometry.py).

Fija la convención de orientación del spec 20 (vector de línea de visión por
índice de vista). No modificar la convención sin re-pasar el test de
alineación contra pyPipe3D.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, label

from .constants import AXIS_VIEWS


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("No puedo normalizar un vector nulo")
    return vector / norm


def skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


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
    align_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centra, sustrae velocidad sistémica y rota L_total → +z.

    `align_mask` (extensión v2, spec 10): subconjunto de partículas usado
    para computar L_total (e.g. R < 2×R_eff). Si es None, usa todas (v1).
    """
    centered_pos = positions - center[None, :]
    centered_vel = velocities - systemic_velocity[None, :]
    if align_mask is None:
        sel_pos, sel_vel, sel_mass = centered_pos, centered_vel, masses
    else:
        sel_pos, sel_vel, sel_mass = centered_pos[align_mask], centered_vel[align_mask], masses[align_mask]
    ang_momentum = np.sum(np.cross(sel_pos, sel_vel) * sel_mass[:, None], axis=0)
    if np.linalg.norm(ang_momentum) == 0:
        rotation = np.eye(3)
    else:
        rotation = rotation_matrix_from_vectors(ang_momentum, np.array([0.0, 0.0, 1.0]))
    return (rotation @ centered_pos.T).T, (rotation @ centered_vel.T).T, rotation


def project_positions(
    positions: np.ndarray, view_vector: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
