"""Corre el pipeline completo (Fase A+B+C) sobre las galaxias de la oleada 1
en /media/andy/Data/tng/mangia_flat, una por una en un SUBPROCESO aislado
(para que la memoria de una galaxia grande no se acumule ni tumbe al resto
del lote -- la galaxia TNG50-88-312423, 23.6M fuentes en el octree, mato al
proceso largo original por presion de memoria).

Uso:
    python scripts/run_wave1.py [--timeout-sec 900]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from aperturenet_labels.io import manifest as manifest_mod

DATA_DIR = Path("/media/andy/Data/tng/mangia_flat")
SUMMARY_PATH = DATA_DIR / "wave1_run_summary.json"


def already_done(gal: str, view: int) -> bool:
    entry = DATA_DIR / "output" / "dataset_entries" / f"{gal}_v{view}.h5"
    return entry.exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-sec", type=int, default=900)
    args = parser.parse_args()

    rows = manifest_mod.build_manifest_from_dir(DATA_DIR)
    print(f"Galaxias resueltas desde {DATA_DIR}: {len(rows)}", flush=True)

    results: list[dict] = []
    if SUMMARY_PATH.exists():
        try:
            results = json.loads(SUMMARY_PATH.read_text())
        except Exception:
            results = []
    done_ids = {r["galaxy_id"] for r in results if r.get("status") == "ok"}

    for row in rows:
        gal = manifest_mod.galaxy_id(row.snapshot, row.subhalo_id)
        if gal in done_ids or already_done(gal, row.view):
            print(f"SKIP {gal} (ya tiene dataset_entry)", flush=True)
            if gal not in done_ids:
                results.append({"galaxy_id": gal, "status": "ok", "note": "ya existia"})
            continue

        print(f"\n=== {gal} (subproceso, timeout={args.timeout_sec}s) ===", flush=True)
        t0 = time.time()
        cmd = [
            sys.executable, "-m", "aperturenet_labels.cli.main", "run",
            "--galaxy-id", gal, "--view", str(row.view), "--data-dir", str(DATA_DIR),
        ]
        try:
            proc = subprocess.run(cmd, timeout=args.timeout_sec, capture_output=True, text=True)
            dt = time.time() - t0
            if proc.returncode == 0 and already_done(gal, row.view):
                print(f"OK {gal} en {dt:.1f}s", flush=True)
                results.append({"galaxy_id": gal, "status": "ok", "seconds": round(dt, 1)})
            else:
                tail_err = "\n".join(proc.stderr.splitlines()[-20:])
                print(f"FALLO {gal} (returncode={proc.returncode}) tras {dt:.1f}s:\n{tail_err}", flush=True)
                results.append(
                    {"galaxy_id": gal, "status": "error", "returncode": proc.returncode,
                     "stderr_tail": tail_err, "seconds": round(dt, 1)}
                )
        except subprocess.TimeoutExpired:
            dt = time.time() - t0
            print(f"TIMEOUT {gal} tras {dt:.1f}s (probable galaxia muy grande / limite de memoria)", flush=True)
            results.append({"galaxy_id": gal, "status": "timeout", "seconds": round(dt, 1)})

        SUMMARY_PATH.write_text(json.dumps(results, indent=2))

    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nResumen: {n_ok}/{len(results)} ok. Detalle en {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
