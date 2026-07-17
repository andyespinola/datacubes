"""Validacion congelada del fix (chequeo de extension + espejo + gate hibrido
s_min=2.5, calibrado SOLO con las 6 galaxias rechazadas de la oleada 1) sobre
un conjunto arbitrario de galaxias ya procesadas por el pipeline v2.

Ningun parametro se reajusta aqui -- es la prueba de generalizacion.
Reporta solo metricas agregadas + detalle de las rechazadas.

Asume la convencion de layout del propio pipeline (la que usa
aperturenet_labels.cli.main): para cada "--data-dir BASE" tiene que existir
  BASE/output/dataset_entries/*_v0.h5
  BASE/intermediate/phase_a/{galaxy_id}/particle_features.h5
  BASE/intermediate/phase_a/{galaxy_id}/particle_labels_initial.h5
  BASE/{galaxy_id}-0-127.cube.fits.gz

Uso (ejemplo en el equipo remoto):
    python scripts/validate_frozen_fix.py \
        --data-dir /mnt/nuevo/labels_out \
        --mordor /ruta/a/morphs_kinematic_bars.hdf5 \
        --output-json /mnt/nuevo/labels_out/frozen_validation_result.json

Se puede pasar --data-dir mas de una vez si las galaxias estan repartidas
en varias carpetas (ej. pilotos en un lado, oleadas en otro).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.mixture import GaussianMixture

from aperturenet_labels.io import catalogs
from aperturenet_labels.phase_a import extractor
from aperturenet_labels.phase_a.classifier import ClassifierConfig, build_features
from aperturenet_labels.phase_b import label_projection
from aperturenet_labels.io import mangia_reader

S_MIN = 2.5  # congelado de la calibracion anterior; NO tocar aca

# se completan en main() a partir de los argumentos de linea de comandos
DATA_DIRS: list[Path] = []
MORDOR: Path | None = None
TMP_DIR = Path("/tmp")


def all_processed_galaxies() -> list[str]:
    gals = set()
    for base in DATA_DIRS:
        entries_dir = base / "output" / "dataset_entries"
        if entries_dir.exists():
            for p in entries_dir.glob("*_v0.h5"):
                gals.add(p.name.replace("_v0.h5", ""))
    return sorted(gals)


def inter_dir(gal: str) -> Path:
    for base in DATA_DIRS:
        p = base / "intermediate" / "phase_a" / gal
        if p.exists():
            return p
    raise FileNotFoundError(f"No encontre intermediate/phase_a/{gal} en ninguno de {DATA_DIRS}")


def flat_dir(gal: str) -> Path:
    for base in DATA_DIRS:
        if (base / f"{gal}-0-127.cube.fits.gz").exists():
            return base
    raise FileNotFoundError(f"No encontre {gal}-0-127.cube.fits.gz en ninguno de {DATA_DIRS}")


def reconstruct_praw(feats, gmm_h5, config):
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import multivariate_normal

    X = build_features(feats, config)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    with h5py.File(gmm_h5, "r") as f:
        means = f["gmm_params/means"][:].astype(np.float64)
        covs = f["gmm_params/covariances"][:].astype(np.float64)
        weights = f["gmm_params/weights"][:].astype(np.float64)
        means_orig = f["gmm_params/means_original_space"][:].astype(np.float64)
        stored = f["P_class"][:].astype(np.float64)
    dens = np.stack(
        [weights[k] * multivariate_normal(mean=means[k], cov=covs[k], allow_singular=True).pdf(Xs)
         for k in range(3)], axis=1)
    P_raw = dens / np.clip(dens.sum(axis=1, keepdims=True), 1e-300, None)
    return P_raw, means_orig, stored


def reorder_old(P_raw, mo):
    eps_c, e_c, r_c = 0, mo.shape[1] - 1, 1
    dk = int(np.argmax(mo[:, eps_c]))
    rest = [k for k in range(3) if k != dk]
    e0, e1 = mo[rest[0], e_c], mo[rest[1], e_c]
    bk = rest[0] if (abs(e0 - e1) >= 0.05 and e0 < e1) else (rest[1] if abs(e0 - e1) >= 0.05 else rest[int(np.argmin([mo[rest[0], r_c], mo[rest[1], r_c]]))])
    hk = [k for k in rest if k != bk][0]
    return P_raw[:, [bk, dk, hk]], mo[dk, r_c] <= mo[bk, r_c]


def mirror_disk_weights(eps, mass, n_bins=80):
    bins = np.linspace(-1.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(eps, bins) - 1, 0, n_bins - 1)
    m_hist = np.bincount(idx, weights=mass, minlength=n_bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    w_disk_bin = np.zeros(n_bins)
    for b in range(n_bins):
        if centers[b] <= 0 or m_hist[b] <= 0:
            continue
        w_disk_bin[b] = max(0.0, 1.0 - m_hist[n_bins - 1 - b] / m_hist[b])
    return w_disk_bin[idx]


def reorder_hybrid(feats, P_raw, mo):
    _, is_inverted = reorder_old(P_raw, mo)
    if not is_inverted:
        return None  # sin cambios: usar el resultado original tal cual
    eps = feats["epsilon"].astype(np.float64)
    mass = feats["mass"].astype(np.float64)
    E = feats["E"].astype(np.float64)
    e_norm = E / np.abs(E).max()
    w_disk = mirror_disk_weights(eps, mass)
    w_sph = 1.0 - w_disk

    rng = np.random.default_rng(42)
    p_sel = mass * w_sph
    p_sel = p_sel / p_sel.sum()
    n_sub = min(200_000, len(eps))
    sub = rng.choice(len(eps), size=n_sub, replace=False, p=p_sel)
    g = GaussianMixture(n_components=2, random_state=42).fit(e_norm[sub, None])
    mu = g.means_.ravel()
    var = g.covariances_.ravel()
    sep = float(abs(mu[1] - mu[0]) / np.sqrt(var.mean()))

    if sep >= S_MIN:
        bk = int(np.argmin(mu))
        post_bulge = g.predict_proba(e_norm[:, None])[:, bk]
    else:
        post_bulge = np.ones_like(eps)

    P_bulge = mass * w_sph * post_bulge
    P_halo = mass * w_sph * (1 - post_bulge)
    P_disk = mass * w_disk
    # normalizar a probabilidad por particula (dividir por mass ya que P_class es prob, no masa)
    P3 = np.stack([P_bulge, P_disk, P_halo], axis=1) / np.clip(mass[:, None], 1e-30, None)
    P3 = P3 / np.clip(P3.sum(axis=1, keepdims=True), 1e-12, None)
    return P3, sep


def project_fracs(feats, gal, cube_path, P_class3):
    view = mangia_reader.view_definition_from_cube(cube_path, 0, 1)
    P5 = np.zeros((len(P_class3), 5))
    P5[:, 0], P5[:, 1], P5[:, 4] = P_class3[:, 0], P_class3[:, 1], P_class3[:, 2]
    out_path = TMP_DIR / f"claude_val_{gal}.npz"
    label_projection.run_label_projection(
        positions_centered=feats["pos_centered"].astype(np.float64),
        mass=feats["mass"].astype(np.float64),
        light=feats["light_g"].astype(np.float64),
        p_class=P5, view=view, galaxy_id=gal, output_path=out_path,
        r_eff_kpc=float(feats["R_eff_kpc"]),
    )
    d = np.load(out_path, allow_pickle=True)
    Y = np.moveaxis(d["Y_mass_raw"], 0, -1)
    out_path.unlink(missing_ok=True)
    h, w, _ = Y.shape
    cy, cx = h / 2 - 0.5, w / 2 - 0.5
    yy, xx = np.indices((h, w))
    r_px = np.hypot(yy - cy, xx - cx)
    has_signal = Y.sum(-1) > 0
    am = np.where(has_signal, Y.argmax(-1), -1)
    rb = r_px[has_signal & (am == 0)]
    rd = r_px[has_signal & (am == 1)]
    r_b = float(rb.mean()) if len(rb) else np.nan
    r_d = float(rd.mean()) if len(rd) else np.nan
    inverted = bool(r_b > r_d) if not (np.isnan(r_b) or np.isnan(r_d)) else None

    total = feats["mass"].sum()
    return {
        "frac_bulge": float((P_class3[:, 0] * feats["mass"]).sum() / total),
        "frac_disk": float((P_class3[:, 1] * feats["mass"]).sum() / total),
        "frac_halo": float((P_class3[:, 2] * feats["mass"]).sum() / total),
        "invertido": inverted,
    }


def catalog_fracs(gal: str) -> dict:
    if gal == "TNG50-87-155298":
        return {"bulge": 0.1914, "disk": 0.2456, "halo": 0.5630}
    if gal == "TNG50-87-192324":
        return {"bulge": 0.6762, "disk": 0.3238, "halo": 0.0000}
    _, snap_str, sub_str = gal.rsplit("-", 2)
    mordor = catalogs.load_morphology_targets(MORDOR, int(snap_str), int(sub_str))
    priors = catalogs.priors_from_mordor(mordor)
    return {"bulge": priors.bulge_frac, "disk": priors.disk_frac, "halo": priors.other_frac}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir", action="append", required=True,
        help="Carpeta base del pipeline (contiene output/, intermediate/, y los .cube.fits.gz planos). "
        "Repetible si las galaxias estan en mas de una carpeta.",
    )
    parser.add_argument(
        "--mordor", required=True,
        help="Ruta a morphs_kinematic_bars.hdf5 (catalogo MORDOR) en el equipo donde se corre esto.",
    )
    parser.add_argument(
        "--output-json", default="",
        help="Donde guardar el resultado. Default: <primer --data-dir>/frozen_validation_result.json",
    )
    parser.add_argument(
        "--tmp-dir", default="/tmp",
        help="Carpeta temporal para los npz intermedios de la reproyeccion (se borran solos).",
    )
    return parser.parse_args()


def main() -> None:
    global DATA_DIRS, MORDOR, TMP_DIR
    args = parse_args()
    DATA_DIRS = [Path(d) for d in args.data_dir]
    MORDOR = Path(args.mordor)
    TMP_DIR = Path(args.tmp_dir)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    output_json = Path(args.output_json) if args.output_json else DATA_DIRS[0] / "frozen_validation_result.json"

    for d in DATA_DIRS:
        if not d.exists():
            raise SystemExit(f"--data-dir no existe: {d}")
    if not MORDOR.exists():
        raise SystemExit(f"--mordor no existe: {MORDOR}")

    config = ClassifierConfig()
    gals = all_processed_galaxies()
    print(f"Total galaxias con dataset_entry: {len(gals)}", flush=True)

    n_rejected = 0
    errs_old_all, errs_new_all, was_rejected = [], [], []
    errs_old_rej, errs_new_rej = [], []
    n_inv_before = n_inv_after = 0
    detail_rejected = []

    n_skipped = 0
    for i, gal in enumerate(gals, 1):
        try:
            idir = inter_dir(gal)
            feats_path = idir / "particle_features.h5"
            gmm_path = idir / "particle_labels_initial.h5"
            if not feats_path.exists() or not gmm_path.exists():
                print(f"[{i}/{len(gals)}] {gal}: SALTEADA (faltan intermedios de Fase A)", flush=True)
                n_skipped += 1
                continue
            fdir = flat_dir(gal)
            cube_path = fdir / f"{gal}-0-127.cube.fits.gz"
        except FileNotFoundError as exc:
            print(f"[{i}/{len(gals)}] {gal}: SALTEADA ({exc})", flush=True)
            n_skipped += 1
            continue

        feats = extractor.load_particle_features(feats_path)
        cat = catalog_fracs(gal)

        P_raw, mo, stored = reconstruct_praw(feats, gmm_path, config)
        P_old, is_inv = reorder_old(P_raw, mo)

        r_old = project_fracs(feats, gal + "_o", cube_path, P_old)
        err_old = (abs(r_old["frac_bulge"] - cat["bulge"]) + abs(r_old["frac_disk"] - cat["disk"]) + abs(r_old["frac_halo"] - cat["halo"])) / 3
        errs_old_all.append(err_old)
        if r_old["invertido"]:
            n_inv_before += 1

        hybrid = reorder_hybrid(feats, P_raw, mo)
        if hybrid is None:
            errs_new_all.append(err_old)
            was_rejected.append(False)
            r_new = r_old
        else:
            P_new, sep = hybrid
            n_rejected += 1
            was_rejected.append(True)
            r_new = project_fracs(feats, gal + "_n", cube_path, P_new)
            err_new = (abs(r_new["frac_bulge"] - cat["bulge"]) + abs(r_new["frac_disk"] - cat["disk"]) + abs(r_new["frac_halo"] - cat["halo"])) / 3
            errs_new_all.append(err_new)
            errs_old_rej.append(err_old)
            errs_new_rej.append(err_new)
            detail_rejected.append({"galaxy_id": gal, "sep": sep, "err_old": err_old, "err_new": err_new,
                                     "cat": cat, "old": r_old, "new": r_new})
        if r_new["invertido"]:
            n_inv_after += 1

        print(f"[{i}/{len(gals)}] {gal}: rechazada={hybrid is not None}", flush=True)

    print("\n=== RESUMEN AGREGADO (validacion congelada, sin recalibrar) ===")
    print(f"Total galaxias evaluadas: {len(errs_old_all)}  (salteadas por datos faltantes: {n_skipped})")
    print(f"Rechazadas por chequeo de extension (candidatas al fix): {n_rejected} ({100*n_rejected/len(errs_old_all):.0f}%)")
    print(f"Invertidas ANTES: {n_inv_before}   DESPUES: {n_inv_after}")
    print(f"\nError medio 3-familias, TODAS las galaxias:")
    print(f"  actual (sin fix):  {np.mean(errs_old_all):.4f}")
    print(f"  con fix hibrido:   {np.mean(errs_new_all):.4f}")
    print(f"\nError medio 3-familias, SOLO las rechazadas (n={n_rejected}):")
    if n_rejected:
        print(f"  actual (sin fix):  {np.mean(errs_old_rej):.4f}")
        print(f"  con fix hibrido:   {np.mean(errs_new_rej):.4f}")
        n_mejoraron = sum(1 for a, b in zip(errs_old_rej, errs_new_rej) if b < a)
        n_empeoraron = sum(1 for a, b in zip(errs_old_rej, errs_new_rej) if b > a)
        print(f"  mejoraron: {n_mejoraron}/{n_rejected}   empeoraron: {n_empeoraron}/{n_rejected}")

    n_regresion = sum(
        1 for a, b, rej in zip(errs_old_all, errs_new_all, was_rejected)
        if (not rej) and b > a + 1e-6
    )
    print(f"\nRegresiones en galaxias NO rechazadas (deberia ser 0): {n_regresion}")

    out = {
        "n_total": len(errs_old_all), "n_skipped": n_skipped, "n_rejected": n_rejected,
        "n_inv_before": n_inv_before, "n_inv_after": n_inv_after,
        "mean_err_old_all": float(np.mean(errs_old_all)), "mean_err_new_all": float(np.mean(errs_new_all)),
        "mean_err_old_rejected": float(np.mean(errs_old_rej)) if n_rejected else None,
        "mean_err_new_rejected": float(np.mean(errs_new_rej)) if n_rejected else None,
        "detail_rejected": detail_rejected,
    }
    output_json.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nGuardado: {output_json}")


if __name__ == "__main__":
    main()
