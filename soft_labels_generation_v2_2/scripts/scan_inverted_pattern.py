"""Escaneo liviano del patron invertido (disco compacto adentro, bulbo
dominando afuera) sobre un conjunto arbitrariamente grande de dataset_entry.h5
YA EMPAQUETADOS -- no necesita intermediate/phase_a ni los cutouts TNG crudos,
solo lee Y_int_mass y M_valid del HDF5 final.

Sirve como paso 1 (barato) antes de correr validate_frozen_fix.py (paso 2,
necesita intermediate/phase_a) solo sobre el subconjunto de galaxias que este
script marque como candidatas.

Uso:
    python scripts/scan_inverted_pattern.py \
        --entries-dir /mnt/nuevo/labels_out/output/dataset_entries \
        --output-json /mnt/nuevo/labels_out/inverted_scan_result.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def check_galaxy(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        Y = f["labels/Y_int_mass"][:]
        M = f["masks/M_valid"][:]
        class_names = [c.decode() if isinstance(c, bytes) else str(c) for c in f["labels/class_names"][:]]

    bulge_idx = class_names.index("bulge")
    disk_idx = class_names.index("disk")

    h, w, _ = Y.shape
    cy, cx = h / 2 - 0.5, w / 2 - 0.5
    yy, xx = np.indices((h, w))
    r_px = np.hypot(yy - cy, xx - cx)

    valid = M & (Y.sum(-1) > 0)
    am = np.where(valid, Y.argmax(-1), -1)

    r_bulge = r_px[valid & (am == bulge_idx)]
    r_disk = r_px[valid & (am == disk_idx)]
    n_bulge_dom = int(len(r_bulge))
    n_disk_dom = int(len(r_disk))
    r_bulge_mean = float(r_bulge.mean()) if n_bulge_dom else None
    r_disk_mean = float(r_disk.mean()) if n_disk_dom else None

    invertido = (
        r_bulge_mean is not None and r_disk_mean is not None and r_bulge_mean > r_disk_mean
    )

    return {
        "galaxy_id": path.name.replace("_v0.h5", ""),
        "n_valid": int(valid.sum()),
        "n_bulge_dom": n_bulge_dom, "n_disk_dom": n_disk_dom,
        "r_bulge_dom": r_bulge_mean, "r_disk_dom": r_disk_mean,
        "invertido": invertido,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entries-dir", required=True, action="append",
                         help="Carpeta con *_v0.h5 empaquetados. Repetible.")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    paths = []
    for d in args.entries_dir:
        d = Path(d)
        if not d.exists():
            raise SystemExit(f"--entries-dir no existe: {d}")
        paths.extend(sorted(d.glob("*_v0.h5")))
    print(f"Total dataset_entry.h5 encontrados: {len(paths)}", flush=True)

    results = []
    n_errors = 0
    for i, p in enumerate(paths, 1):
        try:
            r = check_galaxy(p)
        except Exception as exc:
            print(f"[{i}/{len(paths)}] {p.name}: ERROR ({exc})", flush=True)
            n_errors += 1
            continue
        results.append(r)
        if i % 50 == 0 or i == len(paths):
            print(f"[{i}/{len(paths)}] procesadas ({sum(1 for x in results if x['invertido'])} invertidas hasta ahora)", flush=True)

    n_inv = sum(1 for r in results if r["invertido"])
    print("\n=== RESUMEN ===")
    print(f"Total evaluadas: {len(results)}  (errores de lectura: {n_errors})")
    print(f"Invertidas (candidatas al fix): {n_inv} ({100*n_inv/max(len(results),1):.0f}%)")

    out = {
        "n_total": len(results), "n_errors": n_errors, "n_invertidas": n_inv,
        "invertidas": [r["galaxy_id"] for r in results if r["invertido"]],
        "detail": results,
    }
    out_path = Path(args.output_json)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nGuardado: {out_path}")
    print(f"\nLista de IDs invertidas (para pasarle a validate_frozen_fix.py despues de "
          f"regenerar/restaurar sus intermediate/phase_a):")
    for gid in out["invertidas"]:
        print(f"  {gid}")


if __name__ == "__main__":
    main()
