#!/usr/bin/env python3
"""Orquestador del etiquetado masivo (10k+ galaxias MaNGIA).

Inputs desde una unidad externa (p. ej. USB), outputs a disco local rápido.
Cada galaxia corre en un SUBPROCESO aislado (label_one.py) — la memoria del
octree de una galaxia grande (~23 GB) no se acumula. Pool de N workers en
paralelo. Reanudable (salta galaxias con entry ya escrito). Robusto a fallos
(continue-on-error + timeout por galaxia). Al terminar, corre los barridos de
QA (inversión bulbo/disco residual, fusiones, pajarita).

Uso típico (ver docs/RUN_BATCH_REMOTE.md):
    python scripts/run_batch.py \
        --input-dir "/run/media/aespinola/ADATA HM800/datacubes" \
        --output-dir /datos/labels_out \
        --ssp        aux/MaStar_CB19.slog_1_5.fits.gz \
        --catalog    aux/MaNGIA_catalog.fits \
        --mordor     aux/morphs_kinematic_bars.hdf5 \
        --workers 4 --timeout-sec 3600 --cleanup

Memoria: cada worker puede pedir hasta ~23 GB (galaxia mayor). Con 128 GB,
--workers 4 deja holgura (4×23 = 92 GB). Súbelo solo si la muestra es de
galaxias pequeñas.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_ID_RE = re.compile(r"TNG50-(?P<snap>\d+)-(?P<sub>\d+)-(?P<view>\d+)-"
                    r"(?P<ifu>\d+)\.cube\.fits\.gz$")


# --------------------------------------------------------------------------
# Descubrimiento de inputs (manifest) decouplado del catálogo
# --------------------------------------------------------------------------

def build_manifest(input_dir: Path, catalog_path: Path) -> list[dict]:
    """Recorre input_dir buscando cubos con su set completo de inputs.

    Requisitos por galaxia (mismo directorio):
      TNG50-{s}-{sub}-{view}-{ifu}.cube.fits.gz   (cubo IFU)
      TNG50-{s}-{sub}.cutout.hdf5                 (partículas TNG) OBLIGATORIO
      TNG50-{s}-{sub}.subhalo.json                (metadatos)      OBLIGATORIO
      TNG50-{s}-{sub}.cutout_phase2.hdf5          (DM)      recomendado (octree)
      TNG50-{s}-{sub}-{view}-{ifu}.cube_maps.fits (pyPipe3D) recomendado (Fase B)
    El catálogo MaNGIA (re_kpc, repeat_count) se lee de --catalog.
    """
    from aperturenet_labels.io.manifest import load_mangia_catalog_row

    rows: list[dict] = []
    skipped: dict[str, int] = {"sin_cutout": 0, "sin_subhalo": 0,
                               "sin_catalogo": 0}
    for cube in sorted(input_dir.glob("TNG50-*.cube.fits.gz")):
        m = _ID_RE.search(cube.name)
        if not m:
            continue
        snap, sub = int(m["snap"]), int(m["sub"])
        view, ifu = int(m["view"]), int(m["ifu"])
        base = f"TNG50-{snap}-{sub}"
        cutout = input_dir / f"{base}.cutout.hdf5"
        phase2 = input_dir / f"{base}.cutout_phase2.hdf5"
        subj = input_dir / f"{base}.subhalo.json"
        maps = input_dir / cube.name.replace(".cube.fits.gz", ".cube_maps.fits")
        if not cutout.exists():
            skipped["sin_cutout"] += 1
            continue
        if not subj.exists():
            skipped["sin_subhalo"] += 1
            continue
        try:
            cat = load_mangia_catalog_row(catalog_path, snap, sub, view)
        except KeyError:
            skipped["sin_catalogo"] += 1
            continue
        rows.append({
            "canonical_id": f"TNG50-{snap}-{sub}-{view}-{ifu}",
            "cutout_path": str(cutout),
            "cutout_phase2_path": str(phase2) if phase2.exists() else "",
            "subhalo_json_path": str(subj),
            "cube_path": str(cube),
            "pipe3d_maps_path": str(maps) if maps.exists() else "",
            "snapshot": snap, "subhalo_id": sub, "view": view,
            "re_kpc": float(cat["re_kpc"]), "ifu_design": ifu,
            "repeat_count": int(cat["repeat_count"]),
        })
    return rows, skipped


# --------------------------------------------------------------------------
# Ejecución
# --------------------------------------------------------------------------

def entry_path(output_dir: Path, gal: str, view: int) -> Path:
    return output_dir / "output" / "dataset_entries" / f"{gal}_v{view}.h5"


def run_one(args_tuple) -> dict:
    (gal, view, manifest_path, output_dir, ssp, cleanup, timeout) = args_tuple
    cmd = [sys.executable, str(REPO / "scripts" / "label_one.py"),
           "--manifest", str(manifest_path), "--galaxy-id", gal,
           "--view", str(view), "--output-dir", str(output_dir),
           "--ssp", str(ssp)]
    if cleanup:
        cmd.append("--cleanup")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"galaxy_id": gal, "view": view, "status": "timeout",
                "seconds": round(time.time() - t0, 1)}
    if proc.returncode == 0 and proc.stdout.strip():
        try:
            return json.loads(proc.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError:
            pass
    # OOM / crash: returncode negativo (señal) o stderr
    err = (proc.stdout.strip().splitlines()[-1] if proc.stdout.strip()
           else proc.stderr.strip()[-300:])
    return {"galaxy_id": gal, "view": view, "status": "error",
            "returncode": proc.returncode, "error": err,
            "seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, type=Path,
                    help="Directorio (USB) con los cubos + cutouts + metadatos")
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Disco local rápido para intermedios + entries")
    ap.add_argument("--ssp", type=Path, default=REPO / "aux" /
                    "MaStar_CB19.slog_1_5.fits.gz")
    ap.add_argument("--catalog", type=Path,
                    default=REPO / "aux" / "MaNGIA_catalog.fits")
    ap.add_argument("--mordor", type=Path,
                    default=REPO / "aux" / "morphs_kinematic_bars.hdf5")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout-sec", type=int, default=3600)
    ap.add_argument("--cleanup", action="store_true",
                    help="Borra intermedios tras cada entry (ahorra ~0.5 GB/gal)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Procesa solo las primeras N (prueba); 0 = todas")
    ap.add_argument("--no-qa", action="store_true",
                    help="No correr los barridos de QA al final")
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "output" / "dataset_entries").mkdir(parents=True, exist_ok=True)

    # dependencias que la Fase A busca EN el output-dir
    for aux in (args.mordor,):
        dst = out / aux.name
        if not dst.exists():
            if not aux.exists():
                sys.exit(f"FALTA dependencia: {aux} (usar --mordor). Necesaria "
                         f"para los priors MORDOR del clasificador.")
            shutil.copy(aux, dst)
    for aux in (args.ssp, args.catalog):
        if not aux.exists():
            sys.exit(f"FALTA dependencia: {aux}")

    print(f"[{time.strftime('%H:%M:%S')}] construyendo manifest desde "
          f"{args.input_dir} ...", flush=True)
    rows, skipped = build_manifest(args.input_dir, args.catalog)
    manifest_path = out / "manifest.jsonl"
    with open(manifest_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"  galaxias con inputs completos: {len(rows)}   "
          f"descartadas: {skipped}", flush=True)
    if not rows:
        sys.exit("No hay galaxias con el set de inputs completo. Revisa que la "
                 "USB tenga cutouts+subhalo.json además de los cubos "
                 "(ver docs/RUN_BATCH_REMOTE.md §Inputs).")

    # resume: salta las que ya tienen entry
    pending = [(f"TNG50-{r['snapshot']}-{r['subhalo_id']}", r["view"])
               for r in rows]
    pending = [(g, v) for g, v in pending
               if not entry_path(out, g, v).exists()]
    resume_skipped = len(rows) - len(pending)
    if args.limit:
        pending = pending[:args.limit]
    print(f"  ya hechas (resume, con entry): {resume_skipped}   "
          f"pendientes a procesar: {len(pending)}", flush=True)

    prog = open(out / "batch_progress.jsonl", "a")
    results = []
    tasks = [(g, v, manifest_path, out, args.ssp, args.cleanup,
              args.timeout_sec) for g, v in pending]
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            prog.write(json.dumps(r) + "\n")
            prog.flush()
            if i % 20 == 0 or r["status"] != "ok":
                el = time.time() - t_start
                rate = i / max(el, 1e-9)
                eta = (len(tasks) - i) / max(rate, 1e-9) / 3600
                print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(tasks)}  "
                      f"{r['status']:8s} {r['galaxy_id']}  "
                      f"({rate*3600:.0f}/h, ETA {eta:.1f}h)", flush=True)
    prog.close()

    from collections import Counter
    summary = {"total_inputs": len(rows), "processed": len(tasks),
               "resume_skipped": resume_skipped,
               "status": dict(Counter(r["status"] for r in results)),
               "skipped_inputs": skipped,
               "elapsed_h": round((time.time() - t_start) / 3600, 2)}
    (out / "batch_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n=== RESUMEN ===\n{json.dumps(summary, indent=1)}", flush=True)

    if not args.no_qa:
        print("\n[QA] corriendo barridos (inversión, fusiones, pajarita)...",
              flush=True)
        for script in ("quantify_inversion.py", "detect_mergers.py",
                       "detect_bowtie.py"):
            sp = REPO / "scripts" / script
            if sp.exists():
                subprocess.run([sys.executable, str(sp)],
                               env={**_env_with_entries(out)}, check=False)
        print("[QA] hecho. Ver output/*.csv y *_flagged.txt", flush=True)


def _env_with_entries(out: Path) -> dict:
    """Los scripts de QA apuntan por defecto al disco de desarrollo; se les
    pasa el dir de entries de esta corrida por variable de entorno."""
    import os
    e = dict(os.environ)
    e["GALSTRUCT_ENTRIES"] = str(out / "output" / "dataset_entries")
    return e


if __name__ == "__main__":
    main()
