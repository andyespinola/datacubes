from __future__ import annotations

import argparse
import os
import time

from orientation_validation.download import download_row_assets, ensure_morphology_catalog
from orientation_validation.env import load_env_file
from orientation_validation.manifest import ProjectionManifestRow, read_manifest


def download_rows(
    rows: list[ProjectionManifestRow],
    out_cache: str,
    api_key: str,
    include_gas: bool,
    force_download: bool,
    simulation: str,
    continue_on_error: bool,
) -> list[ProjectionManifestRow]:
    failures: list[ProjectionManifestRow] = []
    for index, row in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] descargando {row.galaxy_id}", flush=True)
        try:
            download_row_assets(
                row,
                out_cache,
                api_key,
                include_gas=include_gas,
                force=force_download,
                simulation=simulation,
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            failures.append(row)
            print(f"  fallo {row.galaxy_id}: {exc}", flush=True)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga cutouts, metadatos y catálogo morfológico TNG para validación.")
    parser.add_argument("--manifest", required=True, help="CSV de manifiesto")
    parser.add_argument("--out-cache", default="cache/pilot_fresh", help="Cache local del proyecto")
    parser.add_argument("--env-file", default=".env", help="Archivo .env")
    parser.add_argument("--api-key", default="", help="TNG API key; si se omite usa TNG_API_KEY")
    parser.add_argument("--include-gas", action="store_true", help="Descarga gas PartType0 además de estrellas")
    parser.add_argument("--force-download", action="store_true", help="Vuelve a descargar aunque existan archivos")
    parser.add_argument("--max-galaxies", type=int, default=0, help="Máximo de galaxias a descargar")
    parser.add_argument("--galaxy-id", action="append", default=[], help="Filtra por galaxy_id; puede repetirse")
    parser.add_argument("--simulation", default="TNG50-1", help="Nombre de simulación TNG")
    parser.add_argument("--morphology-url", default="", help="URL directa del catálogo morfológico")
    parser.add_argument("--morphology-path", default="", help="Ruta local del catálogo morfológico")
    parser.add_argument("--continue-on-error", action="store_true", help="Continúa descargando aunque una galaxia falle")
    parser.add_argument("--retry-failures", type=int, default=0, help="Rondas extra para reintentar solo galaxias fallidas")
    parser.add_argument("--retry-delay-seconds", type=float, default=60.0, help="Espera entre rondas de reintento")
    args = parser.parse_args()

    load_env_file(args.env_file)
    api_key = args.api_key or os.environ.get("TNG_API_KEY", "")
    if not api_key:
        raise SystemExit("Necesito TNG_API_KEY en .env o --api-key")

    rows = read_manifest(args.manifest)
    if args.galaxy_id:
        wanted = set(args.galaxy_id)
        rows = [row for row in rows if row.galaxy_id in wanted]
    if args.max_galaxies > 0:
        rows = rows[: args.max_galaxies]
    if not rows:
        raise SystemExit("No hay filas para descargar")

    morphology_url = args.morphology_url or os.environ.get("TNG_MORPHOLOGY_CATALOG_URL", "")
    morphology_path = args.morphology_path or os.environ.get("TNG_MORPHOLOGY_CATALOG_PATH", "")
    morphology = ensure_morphology_catalog(
        args.out_cache,
        api_key,
        morphology_url=morphology_url,
        morphology_path=morphology_path,
        force=args.force_download,
    )
    if morphology:
        print(f"Catálogo morfológico listo: {morphology}")

    failures = download_rows(
        rows,
        args.out_cache,
        api_key,
        args.include_gas,
        args.force_download,
        args.simulation,
        args.continue_on_error,
    )
    for round_number in range(1, args.retry_failures + 1):
        if not failures:
            break
        print(f"Ronda de reintento {round_number}/{args.retry_failures}: {len(failures)} galaxias", flush=True)
        if args.retry_delay_seconds > 0:
            time.sleep(args.retry_delay_seconds)
        failures = download_rows(
            failures,
            args.out_cache,
            api_key,
            args.include_gas,
            False,
            args.simulation,
            True,
        )
    if failures:
        print("Descarga terminada con fallos pendientes:", flush=True)
        for row in failures:
            print(f"  {row.galaxy_id}", flush=True)
    else:
        print("Descarga terminada sin fallos pendientes", flush=True)
    print(f"Descarga terminada en {args.out_cache}")


if __name__ == "__main__":
    main()
