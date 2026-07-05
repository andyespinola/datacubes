# Cuantifica la inversion "disco en el centro rodeado de bulbo" en las 94
# entries v2_2. Por galaxia (target masa, raw):
#   - centroide de M_valid y radio caracteristico R_m (percentil 90 de r)
#   - perfil radial de fraccion de clase (10 bins hasta R_m)
#   - r_medio ponderado por prob de bulge y de disk (en unidades R_m)
#   - clase dominante en el nucleo (r < 0.15 R_m)
#   - flag inversion: dominante nuclear == disk Y fraccion de bulge en
#     0.2-0.6 R_m mayor que en el nucleo (anillo de bulbo alrededor)
import csv
import os as _os
from pathlib import Path

import h5py
import numpy as np

V22 = Path(_os.environ.get("GALSTRUCT_ENTRIES", "/media/andy/Data/tng/mangia_flat/output/dataset_entries"))
OUT = V22.parent / "inversion_bulge_disk.csv"
CLASSES = ["bulge", "disk", "bar", "arm", "halo"]

rows = []
profiles = {}
for p in sorted(V22.glob("*_v0.h5")):
    gid = p.name.replace("_v0.h5", "")
    with h5py.File(p) as f:
        Y = f["labels/Y_int_mass"][()]          # (74,74,5)
        M = f["masks/M_valid"][()]
    if M.sum() < 50:
        rows.append({"galaxy": gid, "status": "mascara_chica"})
        continue
    yy, xx = np.mgrid[0:M.shape[0], 0:M.shape[1]]
    cy, cx = yy[M].mean(), xx[M].mean()
    r = np.hypot(yy - cy, xx - cx)
    R_m = np.percentile(r[M], 90)
    rn = r / max(R_m, 1e-6)

    pb = Y[..., 0]
    pd = Y[..., 1]
    core = M & (rn < 0.15)
    ring = M & (rn > 0.2) & (rn < 0.6)
    if core.sum() < 5:
        rows.append({"galaxy": gid, "status": "sin_nucleo"})
        continue

    frac_core = Y[core].mean(0)
    dom_core = CLASSES[int(np.argmax(frac_core))]
    fb_core, fd_core = float(frac_core[0]), float(frac_core[1])
    fb_ring = float(pb[ring].mean()) if ring.sum() else np.nan

    # radios medios ponderados por probabilidad de clase
    wsum_b = pb[M].sum()
    wsum_d = pd[M].sum()
    r_b = float((rn[M] * pb[M]).sum() / wsum_b) if wsum_b > 1 else np.nan
    r_d = float((rn[M] * pd[M]).sum() / wsum_d) if wsum_d > 1 else np.nan

    # perfil radial (para inspeccion visual posterior)
    bins = np.linspace(0, 1.0, 11)
    prof_b, prof_d = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = M & (rn >= lo) & (rn < hi)
        prof_b.append(float(pb[sel].mean()) if sel.sum() > 3 else np.nan)
        prof_d.append(float(pd[sel].mean()) if sel.sum() > 3 else np.nan)
    profiles[gid] = (prof_b, prof_d)

    frac_b_tot = float(pb[M].mean())
    frac_d_tot = float(pd[M].mean())
    invertida = (dom_core == "disk" and fb_ring > fb_core + 0.03
                 and frac_b_tot > 0.02)
    r_inverso = (not np.isnan(r_b) and not np.isnan(r_d) and r_d < r_b
                 and frac_b_tot > 0.02 and frac_d_tot > 0.02)
    rows.append({
        "galaxy": gid, "status": "ok",
        "dom_core": dom_core,
        "fb_core": round(fb_core, 3), "fd_core": round(fd_core, 3),
        "fb_ring": round(fb_ring, 3),
        "r_medio_bulge": round(r_b, 3), "r_medio_disk": round(r_d, 3),
        "frac_bulge_tot": round(frac_b_tot, 3),
        "frac_disk_tot": round(frac_d_tot, 3),
        "inversion_anillo": invertida,
        "r_disk_lt_r_bulge": r_inverso,
    })

keys = list({k for r in rows for k in r})
order = ["galaxy", "status", "dom_core", "fb_core", "fd_core", "fb_ring",
         "r_medio_bulge", "r_medio_disk", "frac_bulge_tot", "frac_disk_tot",
         "inversion_anillo", "r_disk_lt_r_bulge"]
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=order)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in order})

ok = [r for r in rows if r["status"] == "ok"]
inv_ring = [r for r in ok if r["inversion_anillo"]]
inv_r = [r for r in ok if r["r_disk_lt_r_bulge"]]
core_disk = [r for r in ok if r["dom_core"] == "disk"]
print(f"galaxias analizadas: {len(ok)} / {len(rows)}")
print(f"nucleo dominado por DISK:        {len(core_disk)} ({100*len(core_disk)/len(ok):.0f}%)")
print(f"inversion anillo (disk centro + anillo bulge): {len(inv_ring)} ({100*len(inv_ring)/len(ok):.0f}%)")
print(f"r_medio(disk) < r_medio(bulge):  {len(inv_r)} ({100*len(inv_r)/len(ok):.0f}%)")
print("\ndominante nuclear (conteo):")
from collections import Counter
print(" ", Counter(r["dom_core"] for r in ok))
print("\npeores 10 por inversion (fb_ring - fb_core):")
for r in sorted(ok, key=lambda r: -(r["fb_ring"] - r["fb_core"]))[:10]:
    print(f"  {r['galaxy']:22s} core: b={r['fb_core']:.2f} d={r['fd_core']:.2f} "
          f"anillo b={r['fb_ring']:.2f}  dom={r['dom_core']}  "
          f"r_b={r['r_medio_bulge']} r_d={r['r_medio_disk']}")
np.save("/media/andy/Data/tng/mangia_flat/output/inversion_profiles.npy",
        profiles, allow_pickle=True)
print(f"\nCSV: {OUT}")
