"""Prueba en frio del fix propuesto para _reorder_components (classifier.py),
SIN tocar el codigo real ni reajustar el GMM: reconstruye las probabilidades
por particula desde los parametros del GMM ya guardados en
particle_labels_initial.h5, aplica la regla nueva, y reproyecta a spaxels
(Fase B) para comparar el mapa espacial resultante contra el actual.

No corre bar_detector/arm_detector (no afectan bulbo/disco/halo, solo
subdividen disco -> bar/arm) asi que el resultado es bulbo/disco/halo puro.

Uso:
    python scripts/test_reorder_fix.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import multivariate_normal

from aperturenet_labels.io import catalogs, mangia_reader
from aperturenet_labels.phase_a import extractor
from aperturenet_labels.phase_a.classifier import ClassifierConfig, build_features
from aperturenet_labels.phase_b import label_projection

WAVE1_INTER = Path("/media/andy/Data/tng/mangia_flat/intermediate/phase_a")
PILOT_INTER = Path("/home/andy/pythonProjects/datacubes/data/intermediate/phase_a")
WAVE1_FLAT = Path("/media/andy/Data/tng/mangia_flat")
PILOT_FLAT = Path("/home/andy/pythonProjects/datacubes/data")
MORDOR = Path("/home/andy/pythonProjects/datacubes/data/morphs_kinematic_bars.hdf5")

STRATA_CSV = Path(
    "/home/andy/pythonProjects/datacubes/orientation_projection_validation/data/wave1_strata.csv"
)

CENTER_R_PX = 3.0
DISK_EPS_GATE = -1.0  # desactivado: solo probar el chequeo de extension radial


def inter_dir(gal: str) -> Path:
    p = WAVE1_INTER / gal
    return p if p.exists() else PILOT_INTER / gal


def flat_dir(gal: str) -> Path:
    return WAVE1_FLAT if (WAVE1_FLAT / f"{gal}-0-127.cube.fits.gz").exists() else PILOT_FLAT


def reconstruct_praw(feats: dict, gmm_h5: Path, config: ClassifierConfig):
    """Recalcula P_raw (N,3) desde means/covariances/weights guardados, sin reajustar EM."""
    X = build_features(feats, config)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    with h5py.File(gmm_h5, "r") as f:
        means = f["gmm_params/means"][:].astype(np.float64)
        covs = f["gmm_params/covariances"][:].astype(np.float64)
        weights = f["gmm_params/weights"][:].astype(np.float64)
        means_orig = f["gmm_params/means_original_space"][:].astype(np.float64)
        stored_P_class = f["P_class"][:].astype(np.float64)

    dens = np.stack(
        [weights[k] * multivariate_normal(mean=means[k], cov=covs[k], allow_singular=True).pdf(X_scaled)
         for k in range(3)],
        axis=1,
    )
    P_raw = dens / np.clip(dens.sum(axis=1, keepdims=True), 1e-300, None)
    return P_raw, means_orig, stored_P_class


def reorder_old(P_raw, means_orig):
    eps_col, e_col, r_col = 0, means_orig.shape[1] - 1, 1
    disk_k = int(np.argmax(means_orig[:, eps_col]))
    rest = [k for k in range(3) if k != disk_k]
    e0, e1 = means_orig[rest[0], e_col], means_orig[rest[1], e_col]
    if abs(e0 - e1) >= 0.05:
        bulge_k = rest[0] if e0 < e1 else rest[1]
    else:
        bulge_k = rest[int(np.argmin([means_orig[rest[0], r_col], means_orig[rest[1], r_col]]))]
    halo_k = [k for k in rest if k != bulge_k][0]
    return P_raw[:, [bulge_k, disk_k, halo_k]]


def _mirror_energy_decomposition(feats: dict, seed: int = 42) -> np.ndarray:
    """Descomposicion cinematica clasica (Abadi 2003 / familia MORDOR):

    - disco = exceso progrado sobre el espejo de las retrogradas (por bin de eps)
    - esferoide = el resto; se parte bulbo/halo con GMM 1D en energia de ligadura
      (componente mas ligado = bulbo).

    Devuelve (N,3) [bulge, disk, halo]. No usa el catalogo: solo cinematica.
    """
    from sklearn.mixture import GaussianMixture

    eps = feats["epsilon"].astype(np.float64)
    mass = feats["mass"].astype(np.float64)
    e_norm = feats["E"].astype(np.float64)
    e_norm = e_norm / np.abs(e_norm).max()

    # --- espejo de contra-rotacion: w_disk por bin de eps ---
    n_bins = 80
    bins = np.linspace(-1.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(eps, bins) - 1, 0, n_bins - 1)
    m_hist = np.bincount(idx, weights=mass, minlength=n_bins)
    w_disk_bin = np.zeros(n_bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    for b in range(n_bins):
        if centers[b] <= 0 or m_hist[b] <= 0:
            continue
        mb = n_bins - 1 - b  # bin espejado (-eps)
        w_disk_bin[b] = max(0.0, 1.0 - m_hist[mb] / m_hist[b])
    w_disk = w_disk_bin[idx]
    w_sph = 1.0 - w_disk

    # --- bulbo vs halo dentro del esferoide: GMM 1D en energia ---
    rng = np.random.default_rng(seed)
    p_sel = mass * w_sph
    if p_sel.sum() <= 0:
        return np.stack([np.zeros_like(w_disk), w_disk, w_sph], axis=1)
    p_sel = p_sel / p_sel.sum()
    n_sub = min(200_000, len(eps))
    sub = rng.choice(len(eps), size=n_sub, replace=False, p=p_sel)
    g = GaussianMixture(n_components=2, random_state=seed).fit(e_norm[sub, None])
    bound_k = int(np.argmin(g.means_.ravel()))  # E mas negativa = mas ligado = bulbo
    post_bound = g.predict_proba(e_norm[:, None])[:, bound_k]

    P_bulge = w_sph * post_bound
    P_halo = w_sph * (1.0 - post_bound)
    return np.stack([P_bulge, w_disk, P_halo], axis=1)


def reorder_new(P_raw, means_orig, feats):
    """Chequeo de extension radial; si el candidato a disco no es extendido,
    cae a la descomposicion cinematica espejo+energia (ignora el GMM 4D)."""
    eps_col, e_col, r_col = 0, means_orig.shape[1] - 1, 1
    disk_candidate = int(np.argmax(means_orig[:, eps_col]))

    rest = [k for k in range(3) if k != disk_candidate]
    e0, e1 = means_orig[rest[0], e_col], means_orig[rest[1], e_col]
    if abs(e0 - e1) >= 0.05:
        bulge_k = rest[0] if e0 < e1 else rest[1]
    else:
        bulge_k = rest[int(np.argmin([means_orig[rest[0], r_col], means_orig[rest[1], r_col]]))]
    halo_k = [k for k in rest if k != bulge_k][0]

    # chequeo de extension: el candidato a disco debe ser MAS extendido que el bulbo
    if means_orig[disk_candidate, r_col] <= means_orig[bulge_k, r_col]:
        return _mirror_energy_decomposition(feats)

    return P_raw[:, [bulge_k, disk_candidate, halo_k]]


def project_and_check(feats, gal, cube_path, view_idx, repeat_count, P_class3, config):
    view = mangia_reader.view_definition_from_cube(cube_path, view_idx, repeat_count)
    P5 = np.zeros((len(P_class3), 5))
    P5[:, 0] = P_class3[:, 0]  # bulge
    P5[:, 1] = P_class3[:, 1]  # disk
    P5[:, 4] = P_class3[:, 2]  # halo

    out_path = Path(f"/tmp/claude_reorder_test_{gal}.npz")
    label_projection.run_label_projection(
        positions_centered=feats["pos_centered"].astype(np.float64),
        mass=feats["mass"].astype(np.float64),
        light=feats["light_g"].astype(np.float64),
        p_class=P5,
        view=view,
        galaxy_id=gal,
        output_path=out_path,
        r_eff_kpc=float(feats["R_eff_kpc"]),
    )
    d = np.load(out_path, allow_pickle=True)
    Y = np.moveaxis(d["Y_mass_raw"], 0, -1)  # (H,W,5)
    out_path.unlink(missing_ok=True)

    h, w, _ = Y.shape
    cy, cx = h / 2 - 0.5, w / 2 - 0.5
    yy, xx = np.indices((h, w))
    r_px = np.hypot(yy - cy, xx - cx)
    has_signal = Y.sum(-1) > 0
    am = np.where(has_signal, Y.argmax(-1), -1)

    r_bulge_dom = r_px[has_signal & (am == 0)]
    r_disk_dom = r_px[has_signal & (am == 1)]
    r_b = float(r_bulge_dom.mean()) if len(r_bulge_dom) else np.nan
    r_d = float(r_disk_dom.mean()) if len(r_disk_dom) else np.nan
    inverted = (r_b > r_d) if not (np.isnan(r_b) or np.isnan(r_d)) else None

    total_mass = feats["mass"].sum()
    frac_bulge = float((P_class3[:, 0] * feats["mass"]).sum() / total_mass)
    frac_disk = float((P_class3[:, 1] * feats["mass"]).sum() / total_mass)
    frac_halo = float((P_class3[:, 2] * feats["mass"]).sum() / total_mass)

    return {
        "r_bulge_dom": round(r_b, 2) if not np.isnan(r_b) else None,
        "r_disk_dom": round(r_d, 2) if not np.isnan(r_d) else None,
        "invertido": inverted,
        "frac_bulge": round(frac_bulge, 3),
        "frac_disk": round(frac_disk, 3),
        "frac_halo": round(frac_halo, 3),
    }


def main() -> None:
    strata = {}
    with STRATA_CSV.open() as f:
        for row in csv.DictReader(f):
            strata[row["galaxy_id"]] = row["stratum"]
    strata["TNG50-87-155298"] = "halo"
    strata["TNG50-87-192324"] = "bulge"

    config = ClassifierConfig()
    results = []
    for gal in sorted(strata):
        idir = inter_dir(gal)
        fdir = flat_dir(gal)
        feats = extractor.load_particle_features(idir / "particle_features.h5")
        _, snap_str, sub_str = gal.rsplit("-", 2)
        mordor = catalogs.load_morphology_targets(MORDOR, int(snap_str), int(sub_str))
        priors = catalogs.priors_from_mordor(mordor)
        cat_fracs = {"bulge": priors.bulge_frac, "disk": priors.disk_frac, "other": priors.other_frac}

        P_raw, means_orig, stored_P_class = reconstruct_praw(feats, idir / "particle_labels_initial.h5", config)

        P_old = reorder_old(P_raw, means_orig)
        sanity = float(np.abs(P_old - stored_P_class).max())

        P_new = reorder_new(P_raw, means_orig, feats)

        cube_path = fdir / f"{gal}-0-127.cube.fits.gz"
        r_old = project_and_check(feats, gal + "_old", cube_path, 0, 1, P_old, config)
        r_new = project_and_check(feats, gal + "_new", cube_path, 0, 1, P_new, config)

        results.append({
            "galaxy_id": gal, "stratum": strata[gal], "sanity_max_diff": round(sanity, 4),
            "catalogo": {k: round(v, 3) for k, v in cat_fracs.items()},
            "antes": r_old, "despues": r_new,
        })
        print(f"{gal:<18} [{strata[gal]:<6}] sanity_diff={sanity:.4f}  "
              f"invertido antes={r_old['invertido']}  despues={r_new['invertido']}  "
              f"frac_halo cat={cat_fracs['other']:.3f} antes={r_old['frac_halo']:.3f} despues={r_new['frac_halo']:.3f}")

    out = Path("/media/andy/Data/tng/mangia_flat/reorder_fix_test.json")
    out.write_text(json.dumps(results, indent=2))

    print("\n=== resumen ===")
    n_inv_before = sum(1 for r in results if r["antes"]["invertido"])
    n_inv_after = sum(1 for r in results if r["despues"]["invertido"])
    print(f"Invertidas ANTES: {n_inv_before}/{len(results)}")
    print(f"Invertidas DESPUES: {n_inv_after}/{len(results)}")

    mad_before = np.mean([abs(r["antes"]["frac_halo"] - r["catalogo"]["other"]) for r in results])
    mad_after = np.mean([abs(r["despues"]["frac_halo"] - r["catalogo"]["other"]) for r in results])
    print(f"Error medio abs en fraccion halo vs catalogo — ANTES: {mad_before:.3f}  DESPUES: {mad_after:.3f}")
    print(f"\nDetalle: {out}")


if __name__ == "__main__":
    main()
