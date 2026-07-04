"""Barrido de sistemas en fusion / con compañera en las 94 galaxias.

Detecta DOBLES NUCLEOS por picos multiples en Sigma* (densidad de masa), con
verificacion cruzada en sigma* (dispersion): una compañera real tiene su
propio nucleo cinematicamente caliente; un brazo espiral es solo una
sobredensidad fria. Criterios:
  - pico secundario >= 15% del pico primario (en masa LINEAL)
  - separado >= 5 px del primario
  - con sigma* localmente realzada (nucleo distinto), no solo densidad
"""
import csv
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

ENT = Path("/media/andy/Data/tng/mangia_flat/output/dataset_entries")
OUT = Path("/media/andy/Data/tng/mangia_flat/output/merger_sweep.csv")

SEC_FRAC = 0.15      # secundario >= 15% del primario (masa lineal)
MIN_SEP = 5          # px minimos entre nucleos
FOOTPRINT = 5        # ventana de maximo local


def peaks(gid):
    with h5py.File(ENT / f"{gid}_v0.h5") as f:
        if "inputs/pipe3d_maps/mass_density" not in f:
            return "sin_mapas"
        md = f["inputs/pipe3d_maps/mass_density"][()]      # log10
        sig = f["inputs/pipe3d_maps/sigma_star"][()]
        M = f["masks/M_valid"][()]
    lin = np.where(M, 10.0 ** md, 0.0)                     # masa lineal
    sm = gaussian_filter(lin, 1.0)
    sm = np.where(M, sm, 0.0)
    mx = maximum_filter(sm, size=FOOTPRINT)
    is_peak = (sm == mx) & (sm > 0)
    ys, xs = np.where(is_peak)
    if len(ys) == 0:
        return None
    vals = sm[ys, xs]
    order = np.argsort(-vals)
    ys, xs, vals = ys[order], xs[order], vals[order]
    # primario
    py, px, pv = ys[0], xs[0], vals[0]
    # sigma* de fondo para el cross-check
    sig_v = sig[M]
    sig_med, sig_std = np.median(sig_v), sig_v.std()
    sec = []
    for y, x, v in zip(ys[1:], xs[1:], vals[1:]):
        if v < SEC_FRAC * pv:
            break
        d = np.hypot(y - py, x - px)
        if d < MIN_SEP:
            continue
        hot = sig[y, x] > sig_med + 0.5 * sig_std          # nucleo caliente
        sec.append((float(v / pv), float(d), bool(hot), int(y), int(x)))
    return {"py": int(py), "px": int(px), "n_valid": int(M.sum()),
            "secondaries": sec}


rows = []
for p in sorted(ENT.glob("*_v0.h5")):
    gid = p.name.replace("_v0.h5", "")
    r = peaks(gid)
    if r is None or r == "sin_mapas":
        rows.append({"galaxy": gid, "status": r or "sin_pico"})
        continue
    hot_sec = [s for s in r["secondaries"] if s[2]]        # con sigma* caliente
    flag = len(hot_sec) >= 1
    top = max(r["secondaries"], key=lambda s: s[0]) if r["secondaries"] else None
    rows.append({
        "galaxy": gid, "status": "ok",
        "n_secundarios": len(r["secondaries"]),
        "n_sec_calientes": len(hot_sec),
        "merger_flag": flag,
        "ratio_sec": round(top[0], 3) if top else 0.0,
        "sep_px": round(top[1], 1) if top else 0.0,
        "sec_caliente": bool(top[2]) if top else False,
    })

keys = ["galaxy", "status", "merger_flag", "n_secundarios", "n_sec_calientes",
        "ratio_sec", "sep_px", "sec_caliente"]
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in keys})

ok = [r for r in rows if r["status"] == "ok"]
flagged = sorted([r for r in ok if r["merger_flag"]],
                 key=lambda r: -r["ratio_sec"])
print(f"galaxias: {len(ok)}   FLAG merger/companera: {len(flagged)} "
      f"({100*len(flagged)/len(ok):.0f}%)")
print(f"\n{'galaxia':22s} {'ratio':>6s} {'sep_px':>7s} {'n_sec':>6s}")
for r in flagged:
    print(f"{r['galaxy']:22s} {r['ratio_sec']:>6.2f} {r['sep_px']:>7.1f} "
          f"{r['n_secundarios']:>6d}")
# control: las que inspeccione (deben coincidir)
print("\ncontrol (inspeccion visual previa):")
for g, exp in [("TNG50-89-372192", "doble"), ("TNG50-88-382174", "doble"),
               ("TNG50-88-312423", "single"), ("TNG50-87-340908", "single"),
               ("TNG50-89-346164", "single"), ("TNG50-89-464534", "single")]:
    r = next((x for x in ok if x["galaxy"] == g), None)
    got = "FLAG" if (r and r["merger_flag"]) else "ok-single"
    print(f"  {g:22s} esperado={exp:7s} detector={got}")
print(f"\nCSV: {OUT}")
import json
print("FLAGGED_LIST=" + json.dumps([r["galaxy"] for r in flagged]))
