"""Classifier (spec 11 v2.1): GMM sobre el vector 4D del artículo IBERAMIA.

P(bulge|p), P(disk|p), P(halo|p) por partícula. Prior MORDOR solo inicializa
(P3: prior, no constraint). Reordenamiento por energía (nunca solo por ε).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Literal, Optional

import h5py
import numpy as np
import structlog
from pydantic import BaseModel
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ..baselines.epsilon_threshold import hard_threshold_classification
from ..schemas.models import CatalogPriors

log = structlog.get_logger(__name__)


class ClassifierConfig(BaseModel):
    method: Literal["gmm", "hard_thresholds"] = "gmm"
    n_components: int = 3
    feature_set: Literal["paper4d", "standard3d"] = "paper4d"
    log_radius_delta: float = 0.05
    epsilon_init_thresholds: dict = {
        "disk_min": 0.5,
        "bulge_max": 0.3,
        "halo_max": -0.3,
    }
    bulge_radial_max_reff: float = 1.0
    max_iter: int = 200
    tol: float = 1e-4
    reg_covar: float = 1e-6
    seed: int = 42


def build_features(feats: dict, config: ClassifierConfig) -> np.ndarray:
    eps = feats["epsilon"].astype(np.float64)
    e_norm = feats["E"].astype(np.float64)
    e_norm = e_norm / np.abs(e_norm).max()
    r_eff = float(feats["R_eff_kpc"])
    if config.feature_set == "paper4d":
        delta = config.log_radius_delta
        X = np.stack(
            [
                eps,
                np.log10(feats["R"].astype(np.float64) / r_eff + delta),
                np.abs(feats["z"].astype(np.float64)) / r_eff,
                e_norm,
            ],
            axis=1,
        )
    else:  # standard3d
        j_total = feats["j_total"].astype(np.float64)
        j_z = feats["j_z"].astype(np.float64)
        j_c = np.clip(feats["j_c"].astype(np.float64), 1e-8, None)
        j_p = np.sqrt(np.maximum(j_total**2 - j_z**2, 0.0))
        X = np.stack([eps, j_p / j_c, e_norm], axis=1)
    return X


def _nearest_class_by_distance(
    X_un: np.ndarray, X_as: np.ndarray, labels_as: np.ndarray, max_ref: int = 50000, seed: int = 42
) -> np.ndarray:
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    if len(X_as) > max_ref:
        sel = rng.choice(len(X_as), max_ref, replace=False)
        X_as, labels_as = X_as[sel], labels_as[sel]
    tree = cKDTree(X_as)
    _, idx = tree.query(X_un, k=1)
    return labels_as[idx]


def _initial_labels(feats: dict, X_scaled: np.ndarray, config: ClassifierConfig) -> np.ndarray:
    eps = feats["epsilon"]
    R = feats["R"]
    r_eff = float(feats["R_eff_kpc"])
    thr = config.epsilon_init_thresholds
    init = np.full(len(eps), -1, dtype=np.int64)
    init[eps > thr["disk_min"]] = 1
    init[(eps < thr["bulge_max"]) & (R < config.bulge_radial_max_reff * r_eff)] = 0
    init[eps < thr["halo_max"]] = 2
    unassigned = init == -1
    # garantizar las 3 clases pobladas; si alguna queda vacía, sembrar por cuantiles de ε
    for k, sel in [(0, np.argsort(np.abs(eps))[:100]), (1, np.argsort(-eps)[:100]), (2, np.argsort(eps)[:100])]:
        if not (init == k).any():
            init[sel] = k
            unassigned = init == -1
    if unassigned.any():
        init[unassigned] = _nearest_class_by_distance(
            X_scaled[unassigned], X_scaled[~unassigned], init[~unassigned], seed=config.seed
        )
    return init


def _reorder_components(
    P: np.ndarray, means_orig: np.ndarray, feature_set: str
) -> tuple[np.ndarray, str]:
    """Regla v2.2: asignacion CONJUNTA de roles por permutacion (3! = 6).

    Sustituye a la regla secuencial v2.1 (disco = argmax eps; bulbo = mas
    ligado del resto), que invertia bulbo<->disco en el 23% de la muestra
    (22/94): cuando el componente compacto central tiene eps medio
    comparable o mayor que el del disco extendido (empates resueltos por
    orden arbitrario, o pseudo-bulbos rotantes), coronaba al componente
    CENTRAL como disco y el disco real caia en bulbo. Ver
    reports/investigacion_inversion_bulbo_disco.md.

    Puntuaciones por rol sobre las medias GMM estandarizadas ENTRE los 3
    componentes (paper4d: [eps, log10(R/Reff), |z|/Reff, E_norm]):
      s_bulge = -R_hat - E_hat        compacto y ligado; SIN eps: el
                                      pseudo-bulbo rotante es bulbo
                                      (semantica observacional / MORDOR)
      s_disk  = eps_hat + R_hat - z_hat   rotante, extendido, delgado
      s_halo  = z_hat - eps_hat           grueso, no rotante
    Se elige la permutacion (b, d, h) que maximiza la suma. Sigue
    prohibido distinguir bulbo de halo solo por eps.
    """
    from itertools import permutations

    eps_col = 0
    e_col = means_orig.shape[1] - 1
    if feature_set == "paper4d":
        r_col, z_col = 1, 2
        m = (means_orig - means_orig.mean(axis=0)) / np.maximum(
            means_orig.std(axis=0), 1e-9
        )
        s_bulge = -m[:, r_col] - m[:, e_col]
        s_disk = m[:, eps_col] + m[:, r_col] - m[:, z_col]
        s_halo = m[:, z_col] - m[:, eps_col]
        best_total, best = -np.inf, (0, 1, 2)
        for b, d, h in permutations(range(3)):
            total = s_bulge[b] + s_disk[d] + s_halo[h]
            if total > best_total:
                best_total, best = total, (b, d, h)
        bulge_k, disk_k, halo_k = best
        branch = "permutation_v2.2"
    else:  # standard3d: sin columnas R/z, se conserva la regla v2.1
        disk_k = int(np.argmax(means_orig[:, eps_col]))
        rest = [k for k in range(3) if k != disk_k]
        bulge_k = rest[
            int(np.argmin([means_orig[rest[0], e_col], means_orig[rest[1], e_col]]))
        ]
        halo_k = [k for k in rest if k != bulge_k][0]
        branch = "energy"
    return P[:, [bulge_k, disk_k, halo_k]], branch


def run_classifier(
    feats: dict,
    output_path: str | Path,
    catalog_priors: Optional[CatalogPriors] = None,
    config: ClassifierConfig | None = None,
) -> Path:
    config = config or ClassifierConfig()
    t0 = time.time()
    n = len(feats["epsilon"])
    galaxy_id = str(feats["galaxy_id"])
    log.info("classifier.start", galaxy_id=galaxy_id, n=n, feature_set=config.feature_set)

    X = build_features(feats, config)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    if config.method == "hard_thresholds":
        P_class = hard_threshold_classification(feats["epsilon"])
        method_used = "hard_thresholds"
        gmm = None
        means_orig = np.full((3, X.shape[1]), np.nan)
        branch = "n/a"
    else:
        init_labels = _initial_labels(feats, X_scaled, config)
        means_init = np.array([X_scaled[init_labels == k].mean(axis=0) for k in range(3)])
        weights_data = np.bincount(init_labels, minlength=3).astype(np.float64) / n
        if catalog_priors is not None and catalog_priors.source != "none":
            alpha = catalog_priors.confidence
            weights_init = alpha * np.array(
                [catalog_priors.bulge_frac, catalog_priors.disk_frac, catalog_priors.other_frac]
            ) + (1 - alpha) * weights_data
            weights_init /= weights_init.sum()
        else:
            weights_init = weights_data
        gmm = GaussianMixture(
            n_components=config.n_components,
            covariance_type="full",
            weights_init=weights_init,
            means_init=means_init,
            max_iter=config.max_iter,
            tol=config.tol,
            reg_covar=config.reg_covar,
            random_state=config.seed,
        )
        gmm.fit(X_scaled)
        if not gmm.converged_:
            log.warning("classifier.gmm_no_convergio", galaxy_id=galaxy_id)
            P_class = hard_threshold_classification(feats["epsilon"])
            method_used = "hard_thresholds_fallback"
            means_orig = scaler.inverse_transform(gmm.means_)
            branch = "n/a"
        else:
            P_raw = gmm.predict_proba(X_scaled)
            means_orig = scaler.inverse_transform(gmm.means_)
            P_class, branch = _reorder_components(P_raw, means_orig, config.feature_set)
            method_used = "gmm"

    # --- métricas ---
    mass = feats["mass"].astype(np.float64)
    m_tot = mass.sum()
    fractions_recovered = (mass[:, None] * P_class).sum(axis=0) / m_tot
    if catalog_priors is not None and catalog_priors.source != "none":
        fr_cat = np.array(
            [catalog_priors.bulge_frac, catalog_priors.disk_frac, catalog_priors.other_frac]
        )
    else:
        fr_cat = np.full(3, np.nan)
    hard = hard_threshold_classification(feats["epsilon"])
    agreement = float((np.argmax(P_class, axis=1) == np.argmax(hard, axis=1)).mean())
    max_p = P_class.max(axis=1)
    # gate de sanidad radial (investigacion inversion bulbo/disco): el bulbo
    # debe ser mas interno que el disco; si no, la asignacion de roles fallo
    R_part = feats["R"].astype(np.float64)
    r_mean = [
        float((R_part * P_class[:, k]).sum() / max(P_class[:, k].sum(), 1e-9))
        for k in range(3)
    ]
    radial_inversion = bool(r_mean[1] < 0.9 * r_mean[0])
    if radial_inversion:
        log.warning(
            "classifier.inversion_radial_bulbo_disco",
            galaxy_id=galaxy_id,
            r_mean_bulge=round(r_mean[0], 2),
            r_mean_disk=round(r_mean[1], 2),
        )
    quality = {
        "r_mean_bulge_kpc": r_mean[0],
        "r_mean_disk_kpc": r_mean[1],
        "r_mean_halo_kpc": r_mean[2],
        "radial_inversion_flag": radial_inversion,
        "method_used": method_used,
        "feature_set": config.feature_set,
        "bic": float(gmm.bic(X_scaled)) if gmm is not None else np.nan,
        "aic": float(gmm.aic(X_scaled)) if gmm is not None else np.nan,
        "n_iter": int(gmm.n_iter_) if gmm is not None else 0,
        "converged": bool(gmm.converged_) if gmm is not None else True,
        "agreement_with_hard_thresholds": agreement,
        "uncertainty_mean": float((1 - max_p).mean()),
        "uncertainty_p95": float(np.percentile(1 - max_p, 95)),
        "reorder_rule_branch": branch,
        "compute_time_sec": float(time.time() - t0),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["schema_version"] = "1.0"
        f.attrs["source_module"] = "classifier"
        meta = f.create_group("metadata")
        meta.attrs["galaxy_id"] = galaxy_id
        meta.attrs["n_particles"] = n
        meta.attrs["method_used"] = method_used
        d = f.create_dataset("P_class", data=P_class.astype(np.float32), compression="lzf")
        d.attrs["column_names"] = ["bulge", "disk", "halo"]
        if gmm is not None:
            g = f.create_group("gmm_params")
            g.create_dataset("means", data=gmm.means_.astype(np.float32))
            g.create_dataset("covariances", data=gmm.covariances_.astype(np.float32))
            g.create_dataset("weights", data=gmm.weights_.astype(np.float32))
            g.create_dataset("means_original_space", data=means_orig.astype(np.float32))
        qual = f.create_group("quality")
        for k, v in quality.items():
            qual.attrs[k] = v
        qual.attrs["fractions_recovered"] = fractions_recovered.astype(np.float64)
        qual.attrs["fractions_catalog"] = fr_cat.astype(np.float64)

    log.info(
        "classifier.done",
        galaxy_id=galaxy_id,
        method=method_used,
        fractions_recovered=np.round(fractions_recovered, 3).tolist(),
        fractions_catalog=np.round(fr_cat, 3).tolist(),
        agreement=round(agreement, 3),
        branch=branch,
    )
    return output_path


def load_labels(path: str | Path) -> dict:
    with h5py.File(path, "r") as f:
        out = {
            "galaxy_id": str(f["metadata"].attrs["galaxy_id"]),
            "P_class": f["P_class"][:],
            "column_names": [str(c) for c in f["P_class"].attrs["column_names"]],
            "quality": dict(f["quality"].attrs) if "quality" in f else {},
        }
        for extra in ("bar_diagnostics", "arm_diagnostics", "full_pipeline_diagnostics"):
            if extra in f:
                out[extra] = dict(f[extra].attrs)
        if "metadata" in f and "has_bar" in f["metadata"].attrs:
            out["has_bar"] = bool(f["metadata"].attrs["has_bar"])
    p_sum = out["P_class"].sum(axis=1)
    if not np.allclose(p_sum[p_sum > 0], 1.0, atol=1e-3):
        raise ValueError("P_class no normalizada")
    return out
