"""LabelProjection (spec 20): proyección 3D→2D + agregación + N_eff + 4 variantes.

Convención de orientación (RESUELTA, validada contra pyPipe3D en los pilotos):
- vector de línea de visión `view_vector_from_index` (v1, portado);
- base del plano del cielo de `project_positions` (v1, portado);
- raster del cubo MaNGIA: el spaxel (fila, col) corresponde a (x_proy, −y_proy),
  i.e. se deposita con (u, v) = (−y_proy, x_proy). Validado con Spearman 0.98
  (155298) y 0.87 (192324, mapa pyPipe3D plano) y centroide < 1 spaxel.

Las posiciones proyectadas son las del marco de la simulación centradas en el
subhalo (pos_centered del Extractor), NO las del marco face-on.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import structlog
from pydantic import BaseModel
from scipy.ndimage import gaussian_filter

from ..core.constants import CLASS_NAMES
from ..core.geometry import project_positions
from ..schemas.models import ViewDefinition

log = structlog.get_logger(__name__)


class LabelProjectionConfig(BaseModel):
    binning: str = "cic"  # "cic" | "nearest"
    epsilon: float = 1e-8
    r_cov_factor: float = 1.5  # R_cov = min(r_cov_factor × R_eff, R_ifu)
    use_r_cov: bool = True


def mangia_raster_coords(
    positions_centered: np.ndarray, view_vector: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(u, v, l): u = coordenada de filas, v = de columnas del raster MaNGIA."""
    x, y, z = project_positions(positions_centered, view_vector)
    return x, -y, z  # filas ← x_proy, columnas ← −y_proy


def _bin_indices_cic(
    u: np.ndarray, v: np.ndarray, shape: tuple[int, int], pixel_scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Índices y pesos CIC: cada partícula reparte en 4 spaxels vecinos.

    Devuelve (idx_flat (4,N), weights (4,N), inside (N,)).
    Coordenada de pixel continua: p = coord/scale + dim/2 − 0.5 (centro de
    pixel en el centro de la celda, consistente con histogram2d del v1).
    """
    h, w = shape
    pu = u / pixel_scale + h / 2.0 - 0.5
    pv = v / pixel_scale + w / 2.0 - 0.5
    iu0 = np.floor(pu).astype(np.int64)
    iv0 = np.floor(pv).astype(np.int64)
    fu = pu - iu0
    fv = pv - iv0
    idx = np.empty((4, len(u)), dtype=np.int64)
    wts = np.empty((4, len(u)), dtype=np.float64)
    k = 0
    for du, wu in ((0, 1 - fu), (1, fu)):
        for dv, wv in ((0, 1 - fv), (1, fv)):
            ii = iu0 + du
            jj = iv0 + dv
            ok = (ii >= 0) & (ii < h) & (jj >= 0) & (jj < w)
            flat = np.where(ok, ii * w + jj, 0)
            idx[k] = flat
            wts[k] = np.where(ok, wu * wv, 0.0)
            k += 1
    inside = wts.sum(axis=0) > 0
    return idx, wts, inside


def _bin_indices_nearest(
    u: np.ndarray, v: np.ndarray, shape: tuple[int, int], pixel_scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    iu = np.floor(u / pixel_scale + h / 2.0).astype(np.int64)
    iv = np.floor(v / pixel_scale + w / 2.0).astype(np.int64)
    ok = (iu >= 0) & (iu < h) & (iv >= 0) & (iv < w)
    idx = np.where(ok, iu * w + iv, 0)[None, :]
    wts = ok.astype(np.float64)[None, :]
    return idx, wts, ok


def _aggregate(
    idx: np.ndarray,
    wts: np.ndarray,
    weights_particle: np.ndarray,
    p_class: np.ndarray,
    shape: tuple[int, int],
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Promedio ponderado por spaxel. Devuelve (Y (C,H,W), W_tot (H,W), W2_tot)."""
    h, w = shape
    n_cells = h * w
    n_classes = p_class.shape[1]
    raw = np.zeros((n_classes, n_cells))
    w_tot = np.zeros(n_cells)
    w2_tot = np.zeros(n_cells)
    for k in range(idx.shape[0]):
        wk = wts[k] * weights_particle
        np.add.at(w_tot, idx[k], wk)
        np.add.at(w2_tot, idx[k], wk**2)
        for c in range(n_classes):
            np.add.at(raw[c], idx[k], wk * p_class[:, c])
    Y = raw / (w_tot[None, :] + epsilon)
    return Y.reshape(n_classes, h, w), w_tot.reshape(h, w), w2_tot.reshape(h, w)


def _psf_convolve_renormalize(
    Y_raw: np.ndarray, w_tot: np.ndarray, sigma_px: float, epsilon: float
) -> np.ndarray:
    """Convolución por canal de los flujos de clase (Y·W) y renormalización.

    Convolucionar el numerador (masa/luz por clase) y el denominador (total)
    conserva la física de la PSF: mezcla flujo, no probabilidades.
    """
    if sigma_px <= 0:
        return Y_raw.copy()
    num = np.stack([gaussian_filter(Y_raw[c] * w_tot, sigma_px, mode="constant") for c in range(Y_raw.shape[0])])
    den = gaussian_filter(w_tot, sigma_px, mode="constant")
    Y_psf = num / (den[None, :, :] + epsilon)
    total = Y_psf.sum(axis=0)
    Y_psf = np.divide(
        Y_psf, total[None, :, :], out=np.zeros_like(Y_psf), where=total[None, :, :] > 0
    )
    return Y_psf


def run_label_projection(
    positions_centered: np.ndarray,
    mass: np.ndarray,
    light: np.ndarray,
    p_class: np.ndarray,
    view: ViewDefinition,
    galaxy_id: str,
    output_path: str | Path,
    r_eff_kpc: float,
    config: LabelProjectionConfig | None = None,
) -> Path:
    config = config or LabelProjectionConfig()
    t0 = time.time()
    h, w = view.grid_shape
    pixel_scale = view.spaxel_scale_kpc

    # normalizar p_class si hace falta (edge case del spec)
    p_sum = p_class.sum(axis=1)
    if not np.allclose(p_sum, 1.0, atol=1e-3):
        log.warning("label_projection.pclass_renormalizada", galaxy_id=galaxy_id)
        p_class = p_class / np.clip(p_sum[:, None], 1e-12, None)

    # --- Paso 1: cobertura ---
    r_i = np.linalg.norm(positions_centered, axis=1)
    r_ifu = 0.5 * max(h, w) * pixel_scale
    if config.use_r_cov:
        r_cov = min(config.r_cov_factor * r_eff_kpc, r_ifu) if r_eff_kpc > 0 else r_ifu
        # nunca menor que el FoV: el cubo observa todo el FoV
        r_cov = max(r_cov, r_ifu)
    else:
        r_cov = np.inf
    within = r_i <= r_cov
    frac_clipped = float(1.0 - mass[within].sum() / mass.sum())

    # --- Paso 3: rotación a la vista (convención raster validada) ---
    u, v, _ = mangia_raster_coords(positions_centered, np.asarray(view.view_vector))

    u = u[within]
    v = v[within]
    m_w = mass[within].astype(np.float64)
    l_w = light[within].astype(np.float64)
    pc = p_class[within].astype(np.float64)

    # --- Paso 4: binning + agregación ---
    if config.binning == "cic":
        idx, wts, _ = _bin_indices_cic(u, v, (h, w), pixel_scale)
    else:
        idx, wts, _ = _bin_indices_nearest(u, v, (h, w), pixel_scale)

    Y_mass_raw, w_mass_tot, w_mass2_tot = _aggregate(idx, wts, m_w, pc, (h, w), config.epsilon)
    Y_lum_raw, w_lum_tot, _ = _aggregate(idx, wts, l_w, pc, (h, w), config.epsilon)

    # normalizar a suma 1 en spaxels con peso
    for Y, wt in ((Y_mass_raw, w_mass_tot), (Y_lum_raw, w_lum_tot)):
        tot = Y.sum(axis=0)
        np.divide(Y, tot[None, :, :], out=Y, where=tot[None, :, :] > 0)

    # --- Paso 5: N_eff (Kish, con pesos de masa) y conteo bruto ---
    n_eff = w_mass_tot**2 / (w_mass2_tot + config.epsilon)
    # conteo bruto: partícula → spaxel más cercano (independiente del binning)
    idx_n, wts_n, _ = _bin_indices_nearest(u, v, (h, w), pixel_scale)
    n_particles_map = np.zeros(h * w)
    np.add.at(n_particles_map, idx_n[0], wts_n[0])
    n_particles_map = n_particles_map.reshape(h, w)

    # --- Paso 6: PSF ---
    sigma_px = view.fwhm_psf_arcsec / 2.355 / max(view.spaxel_scale_arcsec, 1e-6)
    Y_mass_psf = _psf_convolve_renormalize(Y_mass_raw, w_mass_tot, sigma_px, config.epsilon)
    Y_lum_psf = _psf_convolve_renormalize(Y_lum_raw, w_lum_tot, sigma_px, config.epsilon)

    # --- Paso 7: empaquetado ---
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inclination = _inclination_estimate(positions_centered, mass, np.asarray(view.view_vector))
    np.savez_compressed(
        output_path,
        Y_mass_raw=Y_mass_raw.astype(np.float32),
        Y_mass_psf=Y_mass_psf.astype(np.float32),
        Y_lum_raw=Y_lum_raw.astype(np.float32),
        Y_lum_psf=Y_lum_psf.astype(np.float32),
        raw_mass_per_class=(Y_mass_raw * w_mass_tot[None]).astype(np.float32),
        raw_light_per_class=(Y_lum_raw * w_lum_tot[None]).astype(np.float32),
        total_mass_per_spaxel=w_mass_tot.astype(np.float32),
        total_light_per_spaxel=w_lum_tot.astype(np.float32),
        n_eff=n_eff.astype(np.float32),
        n_particles_map=n_particles_map.astype(np.float32),
        class_names=np.array(CLASS_NAMES),
        metadata=np.array(
            [
                ("galaxy_id", galaxy_id),
                ("view_id", str(view.view_id)),
                ("view_vector", str(view.view_vector)),
                ("binning", config.binning),
                ("r_cov_kpc", f"{r_cov:.3f}"),
                ("fraction_clipped_by_rcov", f"{frac_clipped:.5f}"),
                ("n_particles_within_rcov", str(int(within.sum()))),
                ("inclination_deg", f"{inclination:.2f}"),
                ("fwhm_psf_arcsec", f"{view.fwhm_psf_arcsec:.3f}"),
                ("compute_time_sec", f"{time.time() - t0:.2f}"),
            ]
        ),
    )
    log.info(
        "label_projection.done",
        galaxy_id=galaxy_id,
        view_id=view.view_id,
        frac_clipped=round(frac_clipped, 4),
        inclination_deg=round(inclination, 1),
        t=round(time.time() - t0, 1),
    )
    return output_path


def _inclination_estimate(
    positions: np.ndarray, mass: np.ndarray, view_vector: np.ndarray
) -> float:
    """Ángulo entre el eje de simetría y la línea de visión (0° = face-on).

    Proxy por tensor de inercia central (no requiere velocidades).
    """
    r = np.linalg.norm(positions, axis=1)
    sel = r < np.percentile(r, 50)
    pos_c = positions[sel]
    m_c = mass[sel]
    inertia = (pos_c * m_c[:, None]).T @ pos_c
    _, vecs = np.linalg.eigh(inertia)
    minor_axis = vecs[:, 0]
    cosi = abs(float(np.dot(minor_axis, view_vector / np.linalg.norm(view_vector))))
    return float(np.degrees(np.arccos(np.clip(cosi, 0, 1))))


def load_projection(path: str | Path) -> dict:
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files if k != "metadata"}
    out["metadata"] = dict(data["metadata"])
    for key in ("Y_mass_raw", "Y_mass_psf", "Y_lum_raw", "Y_lum_psf"):
        if np.isnan(out[key]).any():
            raise ValueError(f"{key} contiene NaN")
    return out
