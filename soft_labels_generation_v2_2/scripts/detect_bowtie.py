"""Detecta el patron 'pajarita' (bulbo-disco-bulbo en X): proyeccion de un
disco inclinado y GRUESO donde el bulbo redondo domina el eje menor
escorzado -> cuñas de bulbo confiado a gran radio alternadas con banda de
disco. NO es el bug de inversion; es un efecto de proyeccion.

Firma proyectada (2D, entry):
  - anillo exterior (0.4-0.9 r90) partido AZIMUTALMENTE: unos sectores
    bulbo, otros disco -> contraste azimutal alto (la X)
  - coexisten bulbo Y disco confiados en el exterior (excluye elipticas puras)
Firma 3D (particulas, causa raiz):
  - grueso (z/xy p90 alto) + caliente (poca fraccion de disco frio eps>0.7)
"""
import csv
from pathlib import Path

import h5py
import numpy as np
import os as _os

ENT = Path(_os.environ.get("GALSTRUCT_ENTRIES", "/media/andy/Data/tng/mangia_flat/output/dataset_entries"))
PA = ENT.parent.parent / "intermediate" / "phase_a"
OUT = ENT.parent / "bowtie_sweep.csv"
N_SECT = 16


def projected(gid):
    with h5py.File(ENT / f"{gid}_v0.h5") as f:
        if "labels/Y_int_mass" not in f:
            return None
        Y = f["labels/Y_int_mass"][()]
        M = f["masks/M_valid"][()]
    lab = Y.argmax(-1)
    conf = Y.max(-1)
    yy, xx = np.mgrid[0:M.shape[0], 0:M.shape[1]]
    cy, cx = yy[M].mean(), xx[M].mean()
    r = np.hypot(yy - cy, xx - cx)
    th = np.arctan2(yy - cy, xx - cx)
    r90 = np.percentile(r[M], 90)
    # inclinacion: axis ratio de la mascara via momentos de 2o orden
    yv, xv = yy[M] - cy, xx[M] - cx
    cov = np.cov(np.stack([xv, yv]))
    ev = np.linalg.eigvalsh(cov)
    axis_ratio = float(np.sqrt(ev[0] / max(ev[1], 1e-9)))   # b/a (1=cara, ~0=canto)
    ring = M & (r > 0.4 * r90) & (r < 0.9 * r90) & (conf > 0.6)
    if ring.sum() < 30:
        return None
    _axis_ratio = axis_ratio
    # fraccion de bulbo por sector azimutal
    sect = ((th[ring] + np.pi) / (2 * np.pi) * N_SECT).astype(int) % N_SECT
    isb = (lab[ring] == 0)
    isd = (lab[ring] == 1)
    fb = np.array([isb[sect == s].mean() if (sect == s).sum() else np.nan
                   for s in range(N_SECT)])
    fb = fb[~np.isnan(fb)]
    return {
        "outer_bulge_frac": float(isb.mean()),
        "outer_disk_frac": float(isd.mean()),
        "azimuth_contrast": float(np.nanstd(fb)),   # alto = pajarita
        "axis_ratio": _axis_ratio,                  # b/a (bajo = inclinado/canto)
        "n_ring": int(ring.sum()),
    }


def structure3d(gid):
    fp = PA / gid / "particle_features.h5"
    if not fp.exists():
        return None
    with h5py.File(fp) as f:
        eps = f["kinematic/epsilon"][()]
        pos = f["physical/pos_aligned"][()]
    xy = np.percentile(np.hypot(pos[:, 0], pos[:, 1]), 90)
    zz = np.percentile(np.abs(pos[:, 2]), 90)
    return {"cold_disk_frac": float((eps > 0.7).mean()),
            "thick_ratio": float(zz / max(xy, 1e-6))}


rows = []
for p in sorted(ENT.glob("*_v0.h5")):
    gid = p.name.replace("_v0.h5", "")
    pr = projected(gid)
    if pr is None:
        rows.append({"galaxy": gid, "status": "sin_datos"})
        continue
    s3 = structure3d(gid) or {"cold_disk_frac": np.nan, "thick_ratio": np.nan}
    # pajarita: contraste azimutal alto + ambos presentes en el exterior +
    # sistema grueso y caliente en 3D
    # disco inclinado con bulbo proyectado fuera del plano: alternancia
    # azimutal (bulbo Y disco confiados en el exterior) en galaxia inclinada
    # firma unificada: bulbo confiado a gran radio CONVIVIENDO con disco
    # confiado (alternancia azimutal) -> el mapa 2D no refleja bulbo+disco
    # limpio. Sub-mecanismos (descriptores, no filtros): disco inclinado/canto
    # (axis_ratio bajo) o sistema grueso/caliente (thick alto, cold bajo).
    bowtie = (pr["azimuth_contrast"] > 0.30
              and pr["outer_bulge_frac"] > 0.15
              and pr["outer_disk_frac"] > 0.15)
    rows.append({"galaxy": gid, "status": "ok", "bowtie_flag": bowtie,
                 **{k: round(v, 3) for k, v in pr.items() if k != "n_ring"},
                 "n_ring": pr["n_ring"],
                 "cold_disk_frac": round(s3["cold_disk_frac"], 3),
                 "thick_ratio": round(s3["thick_ratio"], 3)})

keys = ["galaxy", "status", "bowtie_flag", "azimuth_contrast", "axis_ratio",
        "outer_bulge_frac", "outer_disk_frac", "cold_disk_frac",
        "thick_ratio", "n_ring"]
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in keys})

ok = [r for r in rows if r["status"] == "ok"]
flagged = sorted([r for r in ok if r["bowtie_flag"]],
                 key=lambda r: -r["azimuth_contrast"])
print(f"galaxias: {len(ok)}   FLAG pajarita (disco inclinado grueso): "
      f"{len(flagged)} ({100*len(flagged)/len(ok):.0f}%)")
print(f"\n{'galaxia':22s} {'azim':>6s} {'b_ext':>6s} {'d_ext':>6s} "
      f"{'cold':>6s} {'thick':>6s}")
for r in flagged:
    print(f"{r['galaxy']:22s} {r['azimuth_contrast']:>6.2f} "
          f"{r['axis_ratio']:>6.2f} {r['outer_bulge_frac']:>6.2f} "
          f"{r['outer_disk_frac']:>6.2f} {r['cold_disk_frac']:>6.2f} {r['thick_ratio']:>6.2f}")
# control
print("\ncontrol:")
for g, exp in [("TNG50-87-323359", "PAJARITA"), ("TNG50-88-312423", "limpia"),
               ("TNG50-87-340908", "limpia"), ("TNG50-89-464534", "limpia")]:
    r = next((x for x in ok if x["galaxy"] == g), None)
    got = "FLAG" if (r and r["bowtie_flag"]) else "ok"
    ac = r["azimuth_contrast"] if r else "?"
    print(f"  {g:22s} esperado={exp:9s} detector={got:5s} (azim={ac})")
print(f"\nCSV: {OUT}")
import json
print("FLAGGED=" + json.dumps([r["galaxy"] for r in flagged]))
