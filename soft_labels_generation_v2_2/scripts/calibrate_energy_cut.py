"""Calibracion del ancla del corte de energia bulbo/halo dentro del esferoide,
para la rama 'disco rechazado' de la descomposicion espejo+energia.

Contexto: el GMM 1D K=2 en energia fabrica un halo inexistente cuando la
distribucion es unimodal (340908, 346164). Se reemplaza por un corte con UN
hiperparametro global, calibrado contra MORDOR en las galaxias rechazadas de
la oleada 1 (set de calibracion). La validacion congelada se hara con la
oleada 2 (80 galaxias, en descarga).

Parametrizaciones escaneadas:
  B) corte fijo en energia normalizada e_norm = E/|E|_max  (c in [-0.85,-0.10])
  C) corte en la energia mediana de un cascaron a k*R_eff  (k in [0.5..4])

Objetivo: minimizar el error medio |frac - catalogo| promediado sobre las
3 familias (bulge, disk, halo) en las galaxias rechazadas.

Uso:
    python scripts/calibrate_energy_cut.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np

from aperturenet_labels.phase_a import extractor

WAVE1_INTER = Path("/media/andy/Data/tng/mangia_flat/intermediate/phase_a")
PILOT_INTER = Path("/home/andy/pythonProjects/datacubes/data/intermediate/phase_a")
STRATA_CSV = Path(
    "/home/andy/pythonProjects/datacubes/orientation_projection_validation/data/wave1_strata.csv"
)

# fracciones MORDOR de los pilotos (convencion bulge+pseudo / thin+thick / halo,
# tomadas de fractions_catalog de sus qa_report)
PILOT_FRACS = {
    "TNG50-87-155298": {"bulge": 0.1914, "disk": 0.2456, "halo": 0.5630},
    "TNG50-87-192324": {"bulge": 0.6762, "disk": 0.3238, "halo": 0.0000},
}

B_CUTS = np.round(np.arange(-0.85, -0.09, 0.05), 2)
C_KS = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]


def inter_dir(gal: str) -> Path:
    p = WAVE1_INTER / gal
    return p if p.exists() else PILOT_INTER / gal


def extent_rejected(idir: Path) -> bool:
    with h5py.File(idir / "particle_labels_initial.h5") as f:
        mo = f["gmm_params/means_original_space"][:].astype(np.float64)
    eps_col, e_col, r_col = 0, mo.shape[1] - 1, 1
    dk = int(np.argmax(mo[:, eps_col]))
    rest = [k for k in range(3) if k != dk]
    e0, e1 = mo[rest[0], e_col], mo[rest[1], e_col]
    if abs(e0 - e1) >= 0.05:
        bk = rest[0] if e0 < e1 else rest[1]
    else:
        bk = rest[int(np.argmin([mo[rest[0], r_col], mo[rest[1], r_col]]))]
    return mo[dk, r_col] <= mo[bk, r_col]


def mirror_disk_weights(eps: np.ndarray, mass: np.ndarray, n_bins: int = 80) -> np.ndarray:
    bins = np.linspace(-1.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(eps, bins) - 1, 0, n_bins - 1)
    m_hist = np.bincount(idx, weights=mass, minlength=n_bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    w_disk_bin = np.zeros(n_bins)
    for b in range(n_bins):
        if centers[b] <= 0 or m_hist[b] <= 0:
            continue
        mb = n_bins - 1 - b
        w_disk_bin[b] = max(0.0, 1.0 - m_hist[mb] / m_hist[b])
    return w_disk_bin[idx]


def main() -> None:
    cat = {}
    with STRATA_CSV.open() as f:
        for row in csv.DictReader(f):
            cat[row["galaxy_id"]] = {
                "bulge": float(row["bulge_frac"]),
                "disk": float(row["disk_frac"]),
                "halo": float(row["halo_frac"]),
                "stratum": row["stratum"],
            }
    for g, fr in PILOT_FRACS.items():
        cat[g] = {**fr, "stratum": "halo" if g.endswith("155298") else "bulge"}

    rejected = []
    for gal in sorted(cat):
        if extent_rejected(inter_dir(gal)):
            rejected.append(gal)
    print(f"Galaxias rechazadas por el chequeo de extension (set de calibracion): {rejected}\n")

    # matrices de error: por galaxia x parametro
    errB = np.zeros((len(rejected), len(B_CUTS)))
    errC = np.full((len(rejected), len(C_KS)), np.nan)
    fracsB = {}  # (gal, c) -> dict
    fracsC = {}

    for gi, gal in enumerate(rejected):
        feats = extractor.load_particle_features(inter_dir(gal) / "particle_features.h5")
        eps = feats["epsilon"].astype(np.float64)
        mass = feats["mass"].astype(np.float64)
        E = feats["E"].astype(np.float64)
        e_norm = E / np.abs(E).max()
        r_sph = np.hypot(feats["R"].astype(np.float64), feats["z"].astype(np.float64))
        r_eff = float(feats["R_eff_kpc"])
        m_tot = mass.sum()

        w_disk = mirror_disk_weights(eps, mass)
        w_sph = 1.0 - w_disk
        f_disk = float((mass * w_disk).sum() / m_tot)

        target = cat[gal]

        for ci, c in enumerate(B_CUTS):
            bound = e_norm < c
            f_b = float((mass * w_sph * bound).sum() / m_tot)
            f_h = float((mass * w_sph * ~bound).sum() / m_tot)
            err = (abs(f_b - target["bulge"]) + abs(f_disk - target["disk"]) + abs(f_h - target["halo"])) / 3
            errB[gi, ci] = err
            fracsB[(gal, c)] = {"bulge": f_b, "disk": f_disk, "halo": f_h}

        for ki, k in enumerate(C_KS):
            r_anchor = k * r_eff
            shell = np.abs(r_sph - r_anchor) < 0.15 * r_anchor
            if shell.sum() < 100:
                continue
            # mediana ponderada por masa de E en el cascaron
            e_shell = E[shell]
            m_shell = mass[shell]
            order = np.argsort(e_shell)
            cw = np.cumsum(m_shell[order])
            e_cut = float(e_shell[order][np.searchsorted(cw, 0.5 * cw[-1])])
            bound = E < e_cut
            f_b = float((mass * w_sph * bound).sum() / m_tot)
            f_h = float((mass * w_sph * ~bound).sum() / m_tot)
            err = (abs(f_b - target["bulge"]) + abs(f_disk - target["disk"]) + abs(f_h - target["halo"])) / 3
            errC[gi, ki] = err
            fracsC[(gal, k)] = {"bulge": f_b, "disk": f_disk, "halo": f_h}

        print(f"{gal} [{target['stratum']}] procesada: f_disk(espejo)={f_disk:.3f} "
              f"(cat={target['disk']:.3f})", flush=True)

    # --- resultados ---
    meanB = errB.mean(axis=0)
    print("\n=== Parametrizacion B: corte fijo en e_norm ===")
    for ci, c in enumerate(B_CUTS):
        print(f"  c={c:+.2f}  error_medio={meanB[ci]:.4f}")
    bestB_i = int(np.argmin(meanB))
    bestB = B_CUTS[bestB_i]

    meanC = np.nanmean(errC, axis=0)
    print("\n=== Parametrizacion C: corte en E(k*R_eff) ===")
    for ki, k in enumerate(C_KS):
        print(f"  k={k:.2f}  error_medio={meanC[ki]:.4f}")
    bestC_i = int(np.nanargmin(meanC))
    bestC = C_KS[bestC_i]

    print(f"\nMejor B: c={bestB:+.2f} (err={meanB[bestB_i]:.4f})")
    print(f"Mejor C: k={bestC:.2f} (err={meanC[bestC_i]:.4f})")

    winner = ("B", bestB) if meanB[bestB_i] <= meanC[bestC_i] else ("C", bestC)
    print(f"\nGANADOR: {winner[0]} con parametro {winner[1]}")

    print(f"\n=== detalle por galaxia con el ganador ===")
    print(f"{'galaxia':<18} {'estrato':<7} {'bulbo':>15} {'disco':>15} {'halo':>15}")
    for gal in rejected:
        target = cat[gal]
        fr = fracsB[(gal, winner[1])] if winner[0] == "B" else fracsC.get((gal, winner[1]))
        if fr is None:
            continue
        print(f"{gal:<18} {target['stratum']:<7} "
              f"{fr['bulge']:.3f} (cat {target['bulge']:.3f}) "
              f"{fr['disk']:.3f} (cat {target['disk']:.3f}) "
              f"{fr['halo']:.3f} (cat {target['halo']:.3f})")

    out = {
        "rejected_galaxies": rejected,
        "B_cuts": B_CUTS.tolist(), "B_mean_err": meanB.tolist(),
        "C_ks": C_KS, "C_mean_err": [None if np.isnan(v) else float(v) for v in meanC],
        "best_B": float(bestB), "best_C": float(bestC),
        "winner": {"param": winner[0], "value": float(winner[1])},
        "detail_winner": {
            gal: (fracsB[(gal, winner[1])] if winner[0] == "B" else fracsC.get((gal, winner[1])))
            for gal in rejected
        },
    }
    out_path = Path("/media/andy/Data/tng/mangia_flat/energy_cut_calibration.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
