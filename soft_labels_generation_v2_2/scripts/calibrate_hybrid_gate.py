"""Variante hibrida para el corte bulbo/halo del esferoide (rama rechazada):

  - si la distribucion de energia del esferoide es BIMODAL (separacion entre
    los dos componentes del GMM 1D >= s_min) -> usar posteriors del GMM 1D
    (funciono muy bien en las halo-dominadas)
  - si es UNIMODAL -> no fabricar halo: todo el esferoide a bulbo
    (esperado mejor en bulbo-puras 340908/346164)

Escanea s_min y compara contra los baselines (original 0.187, v3 0.159,
corte fijo c=-0.40 0.141) con el mismo objetivo de error 3-familias.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.mixture import GaussianMixture

from aperturenet_labels.phase_a import extractor

WAVE1_INTER = Path("/media/andy/Data/tng/mangia_flat/intermediate/phase_a")
PILOT_INTER = Path("/home/andy/pythonProjects/datacubes/data/intermediate/phase_a")
STRATA_CSV = Path(
    "/home/andy/pythonProjects/datacubes/orientation_projection_validation/data/wave1_strata.csv"
)
PILOT_FRACS = {
    "TNG50-87-155298": {"bulge": 0.1914, "disk": 0.2456, "halo": 0.5630, "stratum": "halo"},
    "TNG50-87-192324": {"bulge": 0.6762, "disk": 0.3238, "halo": 0.0000, "stratum": "bulge"},
}
REJECTED = [
    "TNG50-87-155298", "TNG50-87-340908", "TNG50-88-312423",
    "TNG50-89-346164", "TNG50-89-372192", "TNG50-89-444555",
]
S_MINS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]


def inter_dir(gal: str) -> Path:
    p = WAVE1_INTER / gal
    return p if p.exists() else PILOT_INTER / gal


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


def main() -> None:
    cat = {}
    with STRATA_CSV.open() as f:
        for row in csv.DictReader(f):
            cat[row["galaxy_id"]] = {
                "bulge": float(row["bulge_frac"]), "disk": float(row["disk_frac"]),
                "halo": float(row["halo_frac"]), "stratum": row["stratum"],
            }
    cat.update(PILOT_FRACS)

    per_gal = {}
    for gal in REJECTED:
        feats = extractor.load_particle_features(inter_dir(gal) / "particle_features.h5")
        eps = feats["epsilon"].astype(np.float64)
        mass = feats["mass"].astype(np.float64)
        E = feats["E"].astype(np.float64)
        e_norm = E / np.abs(E).max()
        m_tot = mass.sum()

        w_disk = mirror_disk_weights(eps, mass)
        w_sph = 1.0 - w_disk
        f_disk = float((mass * w_disk).sum() / m_tot)

        rng = np.random.default_rng(42)
        p_sel = mass * w_sph
        p_sel = p_sel / p_sel.sum()
        sub = rng.choice(len(eps), size=min(200_000, len(eps)), replace=False, p=p_sel)
        g = GaussianMixture(n_components=2, random_state=42).fit(e_norm[sub, None])
        mu = g.means_.ravel()
        var = g.covariances_.ravel()
        sep = float(abs(mu[1] - mu[0]) / np.sqrt(var.mean()))
        bound_k = int(np.argmin(mu))
        post_bound = g.predict_proba(e_norm[:, None])[:, bound_k]

        f_b_gmm = float((mass * w_sph * post_bound).sum() / m_tot)
        f_h_gmm = float((mass * w_sph * (1 - post_bound)).sum() / m_tot)
        f_b_all = float((mass * w_sph).sum() / m_tot)  # todo-bulbo

        target = cat[gal]
        per_gal[gal] = {
            "sep": sep, "f_disk": f_disk,
            "gmm": {"bulge": f_b_gmm, "disk": f_disk, "halo": f_h_gmm},
            "allbulge": {"bulge": f_b_all, "disk": f_disk, "halo": 0.0},
            "target": target,
        }
        print(f"{gal} [{target['stratum']}] separacion_bimodal={sep:.2f}  "
              f"halo_gmm={f_h_gmm:.3f}  halo_cat={target['halo']:.3f}", flush=True)

    def family_err(fr, t):
        return (abs(fr["bulge"] - t["bulge"]) + abs(fr["disk"] - t["disk"]) + abs(fr["halo"] - t["halo"])) / 3

    print("\n=== escaneo del gate s_min ===")
    best = None
    for s_min in S_MINS:
        errs = []
        for gal, d in per_gal.items():
            fr = d["gmm"] if d["sep"] >= s_min else d["allbulge"]
            errs.append(family_err(fr, d["target"]))
        e = float(np.mean(errs))
        used_gmm = [g for g, d in per_gal.items() if d["sep"] >= s_min]
        print(f"  s_min={s_min:.1f}  error_medio={e:.4f}  (GMM en {len(used_gmm)}/6)")
        if best is None or e < best[1]:
            best = (s_min, e)

    print(f"\nMejor hibrido: s_min={best[0]:.1f} -> error={best[1]:.4f}")
    print("Baselines: original=0.1873, v3(GMM siempre)=0.1586, corte fijo c=-0.40=0.1413")

    s_min = best[0]
    print(f"\n=== detalle con s_min={s_min:.1f} ===")
    for gal, d in per_gal.items():
        rule = "GMM" if d["sep"] >= s_min else "todo-bulbo"
        fr = d["gmm"] if d["sep"] >= s_min else d["allbulge"]
        t = d["target"]
        print(f"{gal:<18} [{t['stratum']:<6}] sep={d['sep']:.2f} regla={rule:<10} "
              f"b={fr['bulge']:.3f}({t['bulge']:.3f}) d={fr['disk']:.3f}({t['disk']:.3f}) "
              f"h={fr['halo']:.3f}({t['halo']:.3f}) err={family_err(fr, t):.4f}")

    out = {"per_galaxy": {g: {k: (v if not isinstance(v, dict) else v) for k, v in d.items()} for g, d in per_gal.items()},
           "best_s_min": best[0], "best_err": best[1]}
    Path("/media/andy/Data/tng/mangia_flat/hybrid_gate_calibration.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nGuardado: /media/andy/Data/tng/mangia_flat/hybrid_gate_calibration.json")


if __name__ == "__main__":
    main()
