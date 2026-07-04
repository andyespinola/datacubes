# Comparacion pre-fix (backup) vs post-fix (reproceso v2.2) de las 94
# galaxias con entry en ambos. Metricas por galaxia (target masa, raw):
#   - inversion radial 2D antes/despues (r_medio disco < 0.9*bulbo)
#   - fraccion de spaxels con argmax cambiado (sobre M_valid comun)
#   - swap bulbo<->disco: fraccion de spaxels que pasan de bulge a disk y vv.
#   - aparicion de barra/brazo: delta de spaxels con argmax==bar y ==arm
#   - fracciones globales por clase antes/despues
import csv
from pathlib import Path

import h5py
import numpy as np

PRE = Path("/media/andy/Data/tng/mangia_flat_pre_fix_20260703/output/dataset_entries")
POST = Path("/media/andy/Data/tng/mangia_flat/output/dataset_entries")
OUT = Path("/media/andy/Data/tng/mangia_flat/output/comparacion_pre_post_fix.csv")
CLASSES = ["bulge", "disk", "bar", "arm", "halo"]


def rmean(argmax_ok, M, cls):
    yy, xx = np.mgrid[0:M.shape[0], 0:M.shape[1]]
    cy, cx = yy[M].mean(), xx[M].mean()
    r = np.hypot(yy - cy, xx - cx)
    sel = M & (argmax_ok == cls)
    return float(r[sel].mean()) if sel.sum() else np.nan


rows = []
gids = sorted(p.name.replace("_v0.h5", "") for p in PRE.glob("*_v0.h5"))
for gid in gids:
    p_pre, p_post = PRE / f"{gid}_v0.h5", POST / f"{gid}_v0.h5"
    if not p_post.exists():
        rows.append({"galaxy": gid, "status": "sin_post"})
        continue
    with h5py.File(p_pre) as f1, h5py.File(p_post) as f2:
        Y1, M1 = f1["labels/Y_int_mass"][()], f1["masks/M_valid"][()]
        Y2, M2 = f2["labels/Y_int_mass"][()], f2["masks/M_valid"][()]
    M = M1 & M2
    a1, a2 = Y1.argmax(-1), Y2.argmax(-1)
    n = max(M.sum(), 1)
    changed = float(((a1 != a2) & M).sum() / n)
    # swaps concretos
    b2d = float(((a1 == 0) & (a2 == 1) & M).sum() / n)   # bulge->disk
    d2b = float(((a1 == 1) & (a2 == 0) & M).sum() / n)   # disk->bulge
    # aparicion bar/arm
    bar1 = float(((a1 == 2) & M).sum() / n)
    bar2 = float(((a2 == 2) & M).sum() / n)
    arm1 = float(((a1 == 3) & M).sum() / n)
    arm2 = float(((a2 == 3) & M).sum() / n)
    rb1, rd1 = rmean(a1, M, 0), rmean(a1, M, 1)
    rb2, rd2 = rmean(a2, M, 0), rmean(a2, M, 1)
    inv1 = bool(not np.isnan(rd1) and not np.isnan(rb1) and rd1 < 0.9 * rb1)
    inv2 = bool(not np.isnan(rd2) and not np.isnan(rb2) and rd2 < 0.9 * rb2)
    fr1 = Y1[M1].mean(0)
    fr2 = Y2[M2].mean(0)
    rows.append({
        "galaxy": gid, "status": "ok",
        "inv_pre": inv1, "inv_post": inv2,
        "spaxels_cambiados": round(changed, 4),
        "bulge_to_disk": round(b2d, 4), "disk_to_bulge": round(d2b, 4),
        "bar_pre": round(bar1, 4), "bar_post": round(bar2, 4),
        "arm_pre": round(arm1, 4), "arm_post": round(arm2, 4),
        "frac_pre": np.round(fr1, 3).tolist(),
        "frac_post": np.round(fr2, 3).tolist(),
    })

keys = ["galaxy", "status", "inv_pre", "inv_post", "spaxels_cambiados",
        "bulge_to_disk", "disk_to_bulge", "bar_pre", "bar_post",
        "arm_pre", "arm_post", "frac_pre", "frac_post"]
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in keys})

ok = [r for r in rows if r["status"] == "ok"]
inv_pre = [r for r in ok if r["inv_pre"]]
inv_post = [r for r in ok if r["inv_post"]]
corregidas = [r for r in ok if r["inv_pre"] and not r["inv_post"]]
rotas = [r for r in ok if r["inv_post"] and not r["inv_pre"]]
sin_cambio = [r for r in ok if r["spaxels_cambiados"] < 1e-6]
bar_nuevas = [r for r in corregidas if r["bar_post"] > r["bar_pre"] + 0.005]
arm_nuevas = [r for r in corregidas if r["arm_post"] > r["arm_pre"] + 0.005]

print(f"galaxias comparadas: {len(ok)} / {len(rows)}")
print(f"inversion radial   PRE: {len(inv_pre)}   POST: {len(inv_post)}")
print(f"corregidas: {len(corregidas)}   nuevas inversiones: {len(rotas)}")
print(f"sin ningun cambio de argmax: {len(sin_cambio)}")
print(f"de las corregidas, ganan BARRA: {len(bar_nuevas)}   ganan BRAZO: {len(arm_nuevas)}")
chg = [r["spaxels_cambiados"] for r in ok]
print(f"spaxels cambiados: mediana={np.median(chg):.3f} max={max(chg):.3f}")
print("\ncorregidas (galaxia | spaxels cambiados | bar pre->post | arm pre->post):")
for r in sorted(corregidas, key=lambda r: -r["spaxels_cambiados"]):
    print(f"  {r['galaxy']:22s} {r['spaxels_cambiados']:.2f}  "
          f"bar {r['bar_pre']:.3f}->{r['bar_post']:.3f}  "
          f"arm {r['arm_pre']:.3f}->{r['arm_post']:.3f}")
print(f"\nCSV: {OUT}")
