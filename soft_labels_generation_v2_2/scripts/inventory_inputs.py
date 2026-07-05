#!/usr/bin/env python3
"""Inventario de inputs para el etiquetado masivo.

Recorre --input-dir y reporta, por galaxia, qué archivos necesarios existen:
cubo, cube_maps (pyPipe3D), cutout (partículas TNG), subhalo.json, y la
MATERIA OSCURA (crítica para el potencial): puede venir dentro del cutout
principal (PartType1) o en un archivo phase2 aparte.

Uso:
    python scripts/inventory_inputs.py \
        --input-dir "/run/media/aespinola/ADATA HM800/datacubes" \
        [--catalog aux/MaNGIA_catalog.fits] [--out inventory.csv]

Imprime un resumen y (si se pide) escribe un CSV por galaxia. NO descarga ni
procesa nada — solo diagnostica el estado del disco.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import h5py

_ID_RE = re.compile(r"TNG50-(?P<snap>\d+)-(?P<sub>\d+)-(?P<view>\d+)-"
                    r"(?P<ifu>\d+)\.cube\.fits\.gz$")


def _cutout_has_dm(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as f:
            return "PartType1" in f
    except Exception:  # noqa: BLE001 - cutout corrupto = sin DM utilizable
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--catalog", type=Path, default=None,
                    help="MaNGIA_catalog.fits (opcional; si falta se omite el "
                         "chequeo de cobertura del catálogo)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--check-dm-in-cutout", action="store_true",
                    help="Abre cada cutout para ver si trae PartType1 (más "
                         "lento; hazlo si sospechas cutouts 'completos').")
    args = ap.parse_args()

    # el catálogo vive en la USB junto a los cubos; auto-detectar si no se pasa
    if args.catalog is None:
        cand = args.input_dir / "MaNGIA_catalog.fits"
        args.catalog = cand if cand.exists() else None

    catalog_ids = None
    if args.catalog and args.catalog.exists():
        from astropy.io import fits
        import numpy as np
        t = fits.getdata(args.catalog, 1)
        catalog_ids = {(int(s), int(sub)) for s, sub in
                       zip(np.array(t["snapshot"]), np.array(t["subhalo_id"]))}

    cubes = sorted(args.input_dir.glob("TNG50-*.cube.fits.gz"))
    rows = []
    cnt = Counter()
    for cube in cubes:
        m = _ID_RE.search(cube.name)
        if not m:
            cnt["nombre_no_canonico"] += 1
            continue
        snap, sub = int(m["snap"]), int(m["sub"])
        base = f"TNG50-{snap}-{sub}"
        cutout = args.input_dir / f"{base}.cutout.hdf5"
        phase2 = args.input_dir / f"{base}.cutout_phase2.hdf5"
        subj = args.input_dir / f"{base}.subhalo.json"
        maps = args.input_dir / cube.name.replace(".cube.fits.gz",
                                                  ".cube_maps.fits")
        has_cut = cutout.exists()
        dm_in_cut = (has_cut and args.check_dm_in_cutout
                     and _cutout_has_dm(cutout))
        has_dm = phase2.exists() or dm_in_cut
        r = {
            "galaxy": base, "cube": True,
            "cube_maps": maps.exists(),
            "cutout": has_cut,
            "subhalo_json": subj.exists(),
            "phase2_dm": phase2.exists(),
            "dm_in_cutout": dm_in_cut,
            "has_dm": has_dm,
            "in_catalog": (catalog_ids is None or (snap, sub) in catalog_ids),
        }
        # clasificación del estado
        if not has_cut:
            r["estado"] = "FALTA_cutout"
        elif not subj.exists():
            r["estado"] = "FALTA_subhalo_json"
        elif not r["in_catalog"]:
            r["estado"] = "FALTA_en_catalogo"
        elif not has_dm:
            r["estado"] = "SIN_DM_potencial_sesgado"
        elif not maps.exists():
            r["estado"] = "ok_sin_maps_faseB_limitada"
        else:
            r["estado"] = "COMPLETO"
        cnt[r["estado"]] += 1
        rows.append(r)

    print(f"cubos encontrados: {len(cubes)}")
    print("estado por galaxia:")
    for k, v in cnt.most_common():
        print(f"  {k:32s} {v:6d}")
    n_dm = sum(1 for r in rows if r["has_dm"])
    print(f"\ncon materia oscura (phase2 o en cutout): {n_dm}/{len(rows)}")
    if cnt.get("SIN_DM_potencial_sesgado"):
        print(f"  ⚠ {cnt['SIN_DM_potencial_sesgado']} galaxias SIN DM: descargar "
              f"phase2 (scripts/download_tng_inputs.py) antes de etiquetar, o el "
              f"potencial saldrá sesgado.")

    if args.out:
        keys = ["galaxy", "estado", "cube", "cube_maps", "cutout",
                "subhalo_json", "phase2_dm", "dm_in_cutout", "has_dm",
                "in_catalog"]
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nCSV: {args.out}")


if __name__ == "__main__":
    main()
