"""Estadisticas de normalizacion. Spec: specs/10_dataset.md - Hito 1.

Un solo JSON versionado con mean/std por modalidad + SIGMA_REF_* / SNR_REF
(medianas del split de train) + p99(N_eff) para el cap del ancla (specs/50).
`scripts/compute_norm_stats.py` es el CLI; el dataset puede recomputar en
memoria (fixture/CI) con `compute_norm_stats` si el JSON no existe.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

STATS_VERSION = 1


def symlog(x: np.ndarray) -> np.ndarray:
    """log1p con signo: igual a log1p para x>=0, definido para flujo negativo
    (spaxels con sustraccion de cielo). Monotona y estable."""
    return np.sign(x) * np.log1p(np.abs(x))


def compute_norm_stats(files: list[Path | str]) -> dict:
    """Acumula sobre los entries de train (un pase, acumuladores en float64).

    Devuelve el dict listo para serializar:
      cube_mean/std (L,), image_mean/std (3,), maps_mean/std (8,),
      sigma_ref_phys (8,), sigma_ref_spat (3,) o null, snr_ref (escalar),
      n_eff_cap {raw_mass, raw_lum, psf_mass, psf_lum} (p99).
    """
    import h5py

    n_files = 0
    cube_sum = cube_sq = cube_n = None
    img_sum = img_sq = img_n = None
    map_sum = map_sq = map_n = None
    err_phys: list[np.ndarray] = []
    err_spat: list[np.ndarray] = []
    snr_all: list[np.ndarray] = []
    neff: dict[str, list[np.ndarray]] = {k: [] for k in
                                         ("raw_mass", "raw_lum", "psf_mass", "psf_lum")}

    for path in files:
        with h5py.File(path, "r") as f:
            M = f["masks/M_valid"][()].astype(bool)
            if not M.any():
                continue
            n_files += 1

            cube = symlog(f["inputs/cube_ifu"][()].astype(np.float64))[:, M]
            if cube_sum is None:
                L = cube.shape[0]
                cube_sum, cube_sq, cube_n = np.zeros(L), np.zeros(L), 0
            cube_sum += np.nansum(cube, axis=1)
            cube_sq += np.nansum(cube ** 2, axis=1)
            cube_n += M.sum()

            img = f["inputs/image"][()].astype(np.float64)[:, M]
            if img_sum is None:
                img_sum, img_sq, img_n = np.zeros(3), np.zeros(3), 0
            img_sum += np.nansum(img, axis=1)
            img_sq += np.nansum(img ** 2, axis=1)
            img_n += M.sum()

            maps = f["inputs/pipe3d_maps"][()].astype(np.float64)[:, M]
            if map_sum is None:
                C = maps.shape[0]
                map_sum, map_sq, map_n = np.zeros(C), np.zeros(C), np.zeros(C)
            map_sum += np.nansum(maps, axis=1)
            map_sq += np.nansum(maps ** 2, axis=1)
            map_n += np.isfinite(maps).sum(axis=1)

            err_phys.append(f["inputs/pipe3d_err"][()][:, M])
            if "image_err" in f["inputs"]:
                err_spat.append(f["inputs/image_err"][()][:, M])
            snr_all.append(f["inputs/snr_spec"][()][M])
            for key in neff:
                variant, weight = key.split("_")
                neff[key].append(f[f"labels/N_eff_{variant}_{weight}"][()][M])

    if n_files == 0:
        raise ValueError("compute_norm_stats: ningun entry con spaxels validos")

    def _mean_std(s, sq, n):
        n = np.maximum(np.asarray(n, dtype=np.float64), 1.0)
        mean = s / n
        var = np.maximum(sq / n - mean ** 2, 0.0)
        return mean, np.sqrt(var)

    cube_mean, cube_std = _mean_std(cube_sum, cube_sq, cube_n)
    img_mean, img_std = _mean_std(img_sum, img_sq, img_n)
    map_mean, map_std = _mean_std(map_sum, map_sq, map_n)

    ephys = np.concatenate(err_phys, axis=1)
    sigma_ref_phys = np.nanmedian(ephys, axis=1)
    sigma_ref_spat = (np.nanmedian(np.concatenate(err_spat, axis=1), axis=1)
                      if err_spat else None)
    snr_ref = float(np.nanmedian(np.concatenate(snr_all)))
    n_eff_cap = {k: float(np.nanpercentile(np.concatenate(v), 99.0))
                 for k, v in neff.items()}

    return {
        "version": STATS_VERSION,
        "n_files": n_files,
        "cube_mean": cube_mean.tolist(), "cube_std": cube_std.tolist(),
        "image_mean": img_mean.tolist(), "image_std": img_std.tolist(),
        "maps_mean": map_mean.tolist(), "maps_std": map_std.tolist(),
        "sigma_ref_phys": sigma_ref_phys.tolist(),
        "sigma_ref_spat": (sigma_ref_spat.tolist()
                           if sigma_ref_spat is not None else None),
        "snr_ref": snr_ref,
        "n_eff_cap": n_eff_cap,
    }


def load_or_compute_stats(root: Path, files: list[Path | str]) -> dict:
    """Lee `norm_stats.json` de `root`; si no existe, recomputa en memoria
    (valido para la fixture sintetica/CI; en produccion el JSON es obligatorio
    y lo emite scripts/compute_norm_stats.py sobre el split de train)."""
    path = Path(root) / "norm_stats.json"
    if path.exists():
        with open(path) as fh:
            stats = json.load(fh)
        if stats.get("version") != STATS_VERSION:
            raise ValueError(f"norm_stats.json version {stats.get('version')} "
                             f"!= {STATS_VERSION}; regenerar con el script")
        return stats
    return compute_norm_stats(files)
