"""ArmDetector (spec 13): brazos espirales por residuales δΣ. disk → arm."""
from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np
import structlog
from pydantic import BaseModel
from scipy.ndimage import label as ndi_label

from ..schemas.models import BarMeta

log = structlog.get_logger(__name__)


class ArmDetectorConfig(BaseModel):
    min_disk_prob: float = 0.3
    fine_grid_size: int = 256
    map_extent_kpc: float = 30.0
    residual_threshold: float = 0.3
    min_island_area: int = 20
    min_azimuthal_extent_deg: float = 30.0
    min_disk_particles: int = 100


def remove_small_islands(mask: np.ndarray, min_area: int) -> np.ndarray:
    labeled, n = ndi_label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(labeled.flat)
    sizes[0] = 0
    keep = sizes >= min_area
    return keep[labeled]


def require_azimuthal_extent(mask: np.ndarray, min_extent_deg: float) -> np.ndarray:
    labeled, n = ndi_label(mask)
    if n == 0:
        return mask
    keep = np.zeros(n + 1, dtype=bool)
    h, w = mask.shape
    cx, cy = w // 2, h // 2
    for k in range(1, n + 1):
        ys, xs = np.where(labeled == k)
        phis = np.arctan2(ys - cy, xs - cx)
        # extensión azimutal robusta a wrap-around: rango circular
        phis_sorted = np.sort(phis)
        gaps = np.diff(np.concatenate([phis_sorted, [phis_sorted[0] + 2 * np.pi]]))
        extent = 2 * np.pi - gaps.max()
        if np.degrees(extent) >= min_extent_deg:
            keep[k] = True
    return keep[labeled]


def run_arm_detector(
    feats: dict,
    with_bar_labels: dict,
    bar_meta: BarMeta,
    output_path: str | Path,
    config: ArmDetectorConfig | None = None,
) -> Path:
    config = config or ArmDetectorConfig()
    t0 = time.time()
    galaxy_id = str(feats["galaxy_id"])
    P = with_bar_labels["P_class"].astype(np.float64)  # (N,4) [bulge, disk, bar, halo]
    n = len(P)
    P_bulge, P_disk, P_bar, P_halo = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
    P_arm = np.zeros(n)

    diagnostics = {"n_crests": 0, "total_arm_area": 0, "arm_mass_fraction": 0.0, "skip_reason": ""}

    x = feats["pos_aligned"][:, 0].astype(np.float64)
    y = feats["pos_aligned"][:, 1].astype(np.float64)
    mass = feats["mass"].astype(np.float64)
    disk_dominated = P_disk > config.min_disk_prob

    if int(disk_dominated.sum()) < config.min_disk_particles:
        diagnostics["skip_reason"] = "too_few_disk_particles"
    else:
        g = config.fine_grid_size
        ext = config.map_extent_kpc
        edges = np.linspace(-ext, ext, g + 1)
        w_disk = mass * P_disk
        sigma_disk, _, _ = np.histogram2d(y, x, bins=(edges, edges), weights=w_disk)

        yy, xx = np.meshgrid(
            0.5 * (edges[:-1] + edges[1:]), 0.5 * (edges[:-1] + edges[1:]), indexing="ij"
        )
        r_grid = np.hypot(xx, yy)
        n_rbins = 64
        r_edges = np.linspace(0, ext * np.sqrt(2), n_rbins + 1)
        r_idx = np.clip(np.digitize(r_grid, r_edges) - 1, 0, n_rbins - 1)
        sums = np.bincount(r_idx.ravel(), weights=sigma_disk.ravel(), minlength=n_rbins)
        counts = np.bincount(r_idx.ravel(), minlength=n_rbins)
        prof = np.divide(sums, counts, out=np.zeros(n_rbins), where=counts > 0)
        sigma_axisym = prof[r_idx]

        eps_floor = max(float(np.percentile(sigma_axisym[sigma_axisym > 0], 5)), 1e-12) \
            if (sigma_axisym > 0).any() else 1e-12
        delta = (sigma_disk - sigma_axisym) / np.maximum(sigma_axisym, eps_floor)

        spiral_mask = delta > config.residual_threshold
        # solo donde hay disco real (evita detectar ruido del fondo)
        spiral_mask &= sigma_axisym > 0
        spiral_mask = remove_small_islands(spiral_mask, config.min_island_area)
        spiral_mask = require_azimuthal_extent(spiral_mask, config.min_azimuthal_extent_deg)

        if bar_meta.has_bar and bar_meta.bar_size_kpc:
            spiral_mask &= r_grid >= float(bar_meta.bar_size_kpc)

        _, n_crests = ndi_label(spiral_mask)
        diagnostics["n_crests"] = int(n_crests)
        diagnostics["total_arm_area"] = int(spiral_mask.sum())

        ix = np.clip(np.digitize(x, edges) - 1, 0, g - 1)
        iy = np.clip(np.digitize(y, edges) - 1, 0, g - 1)
        inside = (np.abs(x) < ext) & (np.abs(y) < ext)
        in_arm = np.zeros(n, dtype=bool)
        in_arm[inside] = spiral_mask[iy[inside], ix[inside]]
        if bar_meta.has_bar and bar_meta.bar_size_kpc:
            # exclusión también a nivel de partícula (test 4 del spec):
            # las celdas de borde pueden contener partículas con R < R_bar
            in_arm &= np.hypot(x, y) >= float(bar_meta.bar_size_kpc)
        P_arm = (in_arm & disk_dominated).astype(np.float64) * P_disk
        diagnostics["arm_mass_fraction"] = float((mass * P_arm).sum() / mass.sum())

    P_disk_new = P_disk - P_arm
    P_new = np.stack([P_bulge, P_disk_new, P_bar, P_arm, P_halo], axis=1)
    assert np.allclose(P_new.sum(axis=1), P.sum(axis=1), atol=1e-9)
    diagnostics["compute_time_sec"] = float(time.time() - t0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["schema_version"] = "1.0"
        f.attrs["source_module"] = "arm_detector"
        meta = f.create_group("metadata")
        meta.attrs["galaxy_id"] = galaxy_id
        meta.attrs["n_particles"] = n
        meta.attrs["phase_a_complete"] = True
        meta.attrs["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if "has_bar" in with_bar_labels:
            meta.attrs["has_bar"] = bool(with_bar_labels["has_bar"])
        d = f.create_dataset("P_class", data=P_new.astype(np.float32), compression="lzf")
        d.attrs["column_names"] = ["bulge", "disk", "bar", "arm", "halo"]
        diag = f.create_group("arm_diagnostics")
        for k, v in diagnostics.items():
            diag.attrs[k] = v
        full = f.create_group("full_pipeline_diagnostics")
        for src in ("quality", "bar_diagnostics"):
            if src in with_bar_labels:
                for k, v in with_bar_labels[src].items():
                    full.attrs[f"{src}.{k}"] = v

    log.info(
        "arm_detector.done",
        galaxy_id=galaxy_id,
        n_crests=diagnostics["n_crests"],
        arm_mass_fraction=round(diagnostics["arm_mass_fraction"], 4),
        skip=diagnostics["skip_reason"] or None,
    )
    return output_path
