"""Descarga el cutout 'fase 2' (posiciones de materia oscura, PartType1) que
aperturenet_labels.io.tng_reader.load_cutout_truth espera como `cutout_phase2_path`
(solo usa PartType1/Coordinates; no hace falta nada de PartType4 aqui porque
Masses de estrellas ya viene en el cutout principal).

No lo cubre orientation_projection_validation/download_tng_assets.py (ese solo
pide stars+gas). Este script es un complemento minimo: una sola query
`dm=Coordinates` por galaxia, con reintentos y escritura atomica igual que
orientation_validation/download.py.

Uso:
    python download_phase2_dm.py \
        --manifest data/wave1_manifest.csv \
        --out-cache /media/andy/Data/tng \
        --env-file /home/andy/pythonProjects/datacubes/data/.env \
        --continue-on-error
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from orientation_validation.download import download_stream_atomic  # noqa: E402
from orientation_validation.env import load_env_file  # noqa: E402
from orientation_validation.manifest import read_manifest  # noqa: E402


def phase2_url(snapshot: int, subhalo_id: int, simulation: str = "TNG50-1") -> str:
    return (
        f"https://www.tng-project.org/api/{simulation}/snapshots/{snapshot}/subhalos/"
        f"{subhalo_id}/cutout.hdf5?dm=Coordinates"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-cache", required=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--simulation", default="TNG50-1")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    load_env_file(args.env_file)
    api_key = args.api_key or os.environ.get("TNG_API_KEY", "")
    if not api_key:
        raise SystemExit("Necesito TNG_API_KEY en .env o --api-key")

    rows = read_manifest(args.manifest)
    out_dir = Path(args.out_cache) / "phase2"
    failures = []
    for i, row in enumerate(rows, start=1):
        out_path = out_dir / f"{row.galaxy_id}.cutout_phase2.hdf5"
        print(f"[{i}/{len(rows)}] fase2 {row.galaxy_id} -> {out_path}", flush=True)
        try:
            download_stream_atomic(
                phase2_url(row.snapshot, row.subhalo_id, args.simulation),
                out_path,
                api_key,
                force=args.force_download,
                timeout=300,
                retries=3,
                backoff_seconds=5.0,
            )
        except Exception as exc:
            print(f"  fallo {row.galaxy_id}: {exc}", flush=True)
            failures.append(row.galaxy_id)
            if not args.continue_on_error:
                raise

    if failures:
        print(f"Fase2 terminada con {len(failures)} fallos: {failures}")
    else:
        print("Fase2 terminada sin fallos pendientes")


if __name__ == "__main__":
    main()
