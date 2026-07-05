#!/usr/bin/env python3
"""Worker: etiqueta UNA galaxia (Fase A+B+C) en un subproceso aislado.

Lee su fila del manifest.jsonl (rutas absolutas a los inputs, típicamente en
la unidad USB) y escribe los productos bajo --output-dir (disco local rápido).
El aislamiento en subproceso garantiza que la memoria de una galaxia grande
(hasta ~23 GB en el octree) no se acumule entre galaxias.

Uso (lo invoca run_batch.py; también corre suelto para depurar una galaxia):
    python scripts/label_one.py \
        --manifest OUT/manifest.jsonl --galaxy-id TNG50-91-571097 \
        --output-dir OUT --ssp <ssp.fits.gz> [--cleanup]

Salida: una línea JSON a stdout con {galaxy_id, status, entry|error, seconds}.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--galaxy-id", required=True)
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--ssp", required=True, type=Path)
    ap.add_argument("--cleanup", action="store_true",
                    help="Borra intermedios phase_a/phase_b tras un entry OK "
                         "(ahorra ~0.5 GB/galaxia; el resume sigue funcionando "
                         "a nivel de galaxia porque comprueba el entry final).")
    ap.add_argument("--copy-cube", action="store_true",
                    help="Embebe el cubo IFU en el entry (~85 MB c/u). Por "
                         "defecto NO se copia: entry solo-etiquetas (~0.6 MB), "
                         "el cubo queda como referencia (producto independiente).")
    args = ap.parse_args()

    from aperturenet_labels.cli.main import (run_phase_a, run_phase_b,
                                             run_phase_c)
    from aperturenet_labels.phase_c import packer
    from aperturenet_labels.schemas.models import ManifestRow

    gal = args.galaxy_id
    t0 = time.time()
    try:
        row = None
        with open(args.manifest) as fh:
            for line in fh:
                d = json.loads(line)
                if (f"TNG50-{d['snapshot']}-{d['subhalo_id']}" == gal
                        and d["view"] == args.view):
                    row = ManifestRow(**d)
                    break
        if row is None:
            raise KeyError(f"{gal} v{args.view} no está en {args.manifest}")

        ctx_a = run_phase_a(row, args.output_dir, args.ssp)
        ctx_b = run_phase_b(row, args.output_dir, ctx_a)
        entry = run_phase_c(row, args.output_dir, ctx_a, ctx_b,
                            copy_cube=args.copy_cube)
        report = packer.validate_dataset_entry(entry)

        if args.cleanup:
            for sub in ("intermediate/phase_a", "intermediate/phase_b"):
                d = args.output_dir / sub / gal
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)

        print(json.dumps({"galaxy_id": gal, "view": args.view, "status": "ok",
                          "entry": str(entry), "n_valid": report.get("n_valid"),
                          "seconds": round(time.time() - t0, 1)}))
        return 0
    except Exception as exc:  # noqa: BLE001 - el orquestador clasifica el fallo
        print(json.dumps({"galaxy_id": gal, "view": args.view,
                          "status": "error", "error": f"{type(exc).__name__}: {exc}",
                          "seconds": round(time.time() - t0, 1)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
