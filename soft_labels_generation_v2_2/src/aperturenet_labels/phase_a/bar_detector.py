"""BarDetector (spec 12): Fourier m=2 + criterio cinemático. disk → bar."""
from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np
import structlog
from pydantic import BaseModel

from ..schemas.models import BarMeta

log = structlog.get_logger(__name__)


class BarDetectorConfig(BaseModel):
    epsilon_min: float = 0.3
    epsilon_max: float = 0.6
    z_max_kpc: float = 0.5
    a2_threshold: float = 0.3
    phi_tolerance_rad: float = 0.785  # ±π/4
    min_particles_in_bar_region: int = 100


def run_bar_detector(
    feats: dict,
    initial_labels: dict,
    bar_meta: BarMeta,
    output_path: str | Path,
    config: BarDetectorConfig | None = None,
) -> Path:
    config = config or BarDetectorConfig()
    t0 = time.time()
    galaxy_id = str(feats["galaxy_id"])
    P = initial_labels["P_class"].astype(np.float64)  # (N,3) [bulge, disk, halo]
    n = len(P)
    P_bulge, P_disk, P_halo = P[:, 0], P[:, 1], P[:, 2]
    P_bar = np.zeros(n)

    diagnostics = {
        "a2": 0.0,
        "phi_bar_rad": 0.0,
        "n_bar_particles": 0,
        "bar_mass_fraction": 0.0,
        "skip_reason": "",
    }

    x = feats["pos_aligned"][:, 0].astype(np.float64)
    y = feats["pos_aligned"][:, 1].astype(np.float64)
    z = feats["z"].astype(np.float64)
    R = feats["R"].astype(np.float64)
    eps = feats["epsilon"].astype(np.float64)
    mass = feats["mass"].astype(np.float64)

    if not bar_meta.has_bar or not bar_meta.bar_size_kpc:
        diagnostics["skip_reason"] = "catalog_no_bar"
    else:
        r_bar = float(bar_meta.bar_size_kpc)
        in_region = R < r_bar
        if int(in_region.sum()) < config.min_particles_in_bar_region:
            log.warning("bar_detector.pocas_particulas", n=int(in_region.sum()))
            diagnostics["skip_reason"] = "too_few_particles"
        else:
            phi_p = np.arctan2(y, x)
            m_in = mass[in_region]
            c2 = np.sum(m_in * np.exp(2j * phi_p[in_region])) / m_in.sum()
            a2 = float(np.abs(c2))
            phi_bar = float(np.angle(c2) / 2.0)
            diagnostics["a2"] = a2
            diagnostics["phi_bar_rad"] = phi_bar
            if a2 < config.a2_threshold:
                diagnostics["skip_reason"] = "a2_below_threshold"
            else:
                is_kin = (eps > config.epsilon_min) & (eps < config.epsilon_max) & (
                    np.abs(z) < config.z_max_kpc
                )
                phi_rel = ((phi_p - phi_bar + np.pi / 2) % np.pi) - np.pi / 2
                is_morph = (R < r_bar) & (np.abs(phi_rel) < config.phi_tolerance_rad)
                in_bar = is_kin & is_morph
                P_bar = in_bar.astype(np.float64) * P_disk
                diagnostics["n_bar_particles"] = int(in_bar.sum())
                diagnostics["bar_mass_fraction"] = float(
                    (mass * P_bar).sum() / (mass.sum())
                )

    P_disk_new = P_disk - P_bar
    P_new = np.stack([P_bulge, P_disk_new, P_bar, P_halo], axis=1)
    assert np.allclose(P_new.sum(axis=1), P.sum(axis=1), atol=1e-9)

    diagnostics["compute_time_sec"] = float(time.time() - t0)
    has_bar_detected = diagnostics["n_bar_particles"] > 0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["schema_version"] = "1.0"
        f.attrs["source_module"] = "bar_detector"
        meta = f.create_group("metadata")
        meta.attrs["galaxy_id"] = galaxy_id
        meta.attrs["n_particles"] = n
        meta.attrs["has_bar"] = has_bar_detected
        meta.attrs["bar_size_kpc"] = float(bar_meta.bar_size_kpc or 0.0)
        d = f.create_dataset("P_class", data=P_new.astype(np.float32), compression="lzf")
        d.attrs["column_names"] = ["bulge", "disk", "bar", "halo"]
        diag = f.create_group("bar_diagnostics")
        for k, v in diagnostics.items():
            diag.attrs[k] = v
        if "quality" in initial_labels:
            qual = f.create_group("quality")
            for k, v in initial_labels["quality"].items():
                qual.attrs[k] = v

    log.info(
        "bar_detector.done",
        galaxy_id=galaxy_id,
        a2=round(diagnostics["a2"], 3),
        n_bar=diagnostics["n_bar_particles"],
        bar_mass_fraction=round(diagnostics["bar_mass_fraction"], 4),
        skip=diagnostics["skip_reason"] or None,
        catalog_a2=bar_meta.bar_strength,
    )
    return output_path
