from __future__ import annotations

import argparse
import os
from pathlib import Path

from labeling.env import load_env_file
from labeling.manifest import read_manifest
from labeling.tng import download_cutout, download_subhalo_metadata, download_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga cutouts y metadatos TNG para las galaxias del manifiesto.")
    parser.add_argument("--manifest", required=True, help="CSV de entrada")
    parser.add_argument("--outdir", required=True, help="Directorio cache de verdad TNG")
    parser.add_argument("--env-file", default="", help="Archivo .env opcional")
    parser.add_argument("--api-key", default="", help="TNG API key. Si se omite, usa TNG_API_KEY del entorno")
    parser.add_argument("--canonical-id", default="", help="Procesa una sola galaxia")
    parser.add_argument("--include-gas", action="store_true", help="Incluye gas en el cutout")
    parser.add_argument("--morphology-url", default="", help="URL directa del catálogo morfológico TNG")
    parser.add_argument("--morphology-path", default="", help="Ruta local de destino para el catálogo morfológico")
    args = parser.parse_args()

    load_env_file(args.env_file)
    api_key = args.api_key or os.environ.get("TNG_API_KEY", "")
    if not api_key:
        raise SystemExit("Necesito TNG_API_KEY para descargar datos desde el portal TNG")

    outdir = Path(args.outdir)
    cutout_dir = outdir / "cutouts"
    metadata_dir = outdir / "metadata"
    rows = read_manifest(args.manifest)
    if args.canonical_id:
        rows = [row for row in rows if row.canonical_id == args.canonical_id]
    if not rows:
        raise SystemExit("No encontré filas del manifiesto para descargar")

    for row in rows:
        cutout_path = cutout_dir / f"{row.canonical_id}.cutout.hdf5"
        metadata_path = metadata_dir / f"{row.canonical_id}.subhalo.json"
        if not cutout_path.exists():
            download_cutout(row.snapshot, row.subhalo_id, cutout_path, api_key, include_gas=args.include_gas)
            print(f"Cutout descargado: {cutout_path}")
        if not metadata_path.exists():
            download_subhalo_metadata(row.snapshot, row.subhalo_id, metadata_path, api_key)
            print(f"Metadatos descargados: {metadata_path}")

    if args.morphology_url and args.morphology_path:
        morphology_path = Path(args.morphology_path)
        if not morphology_path.exists():
            download_url(args.morphology_url, morphology_path, api_key=api_key)
            print(f"Catálogo morfológico descargado: {morphology_path}")


if __name__ == "__main__":
    main()
