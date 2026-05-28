from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from orientation_validation.config import load_configs
from orientation_validation.download import DownloadError, download_row_assets, ensure_morphology_catalog
from orientation_validation.env import load_env_file
from orientation_validation.manifest import build_projection_manifest, select_pilot_rows, write_manifest
from orientation_validation.paths import default_catalog_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara un piloto fresco: manifiesto deduplicado + descarga TNG.")
    parser.add_argument("--catalog", default=str(default_catalog_path()), help="Ruta a MaNGIA_catalog.fits")
    parser.add_argument("--manifest-out", default="data/projection_manifest.csv", help="Manifest completo de salida")
    parser.add_argument("--pilot-manifest-out", default="data/pilot_manifest.csv", help="Manifest piloto de salida")
    parser.add_argument("--max-galaxies", type=int, default=10, help="Tamaño del piloto")
    parser.add_argument("--include-gas", action="store_true", help="Descarga gas PartType0")
    parser.add_argument("--force-download", action="store_true", help="Vuelve a descargar todo en la cache piloto")
    parser.add_argument("--env-file", default=".env", help="Archivo .env")
    parser.add_argument("--out-cache", default="cache/pilot_fresh", help="Cache fresca del piloto")
    parser.add_argument("--api-key", default="", help="TNG API key; si se omite usa TNG_API_KEY")
    parser.add_argument("--config", default="", help="Config JSON opcional")
    parser.add_argument("--simulation", default="TNG50-1", help="Nombre de simulación TNG")
    parser.add_argument("--continue-on-error", action="store_true", help="Continúa si una galaxia falla durante descarga")
    args = parser.parse_args()

    load_env_file(args.env_file)
    api_key = args.api_key or os.environ.get("TNG_API_KEY", "")
    if not api_key:
        raise SystemExit("Necesito TNG_API_KEY en .env o --api-key")

    projection_config, _ = load_configs(args.config)
    rows = build_projection_manifest(
        args.catalog,
        primary_factor=projection_config.primary_rcov_reff,
        secondary_factor=projection_config.secondary_rcov_reff,
    )
    write_manifest(args.manifest_out, rows)
    pilot_rows = select_pilot_rows(rows, args.max_galaxies)
    write_manifest(args.pilot_manifest_out, pilot_rows)
    print(f"Manifest completo: {args.manifest_out} ({len(rows)} galaxias)")
    print(f"Manifest piloto: {args.pilot_manifest_out} ({len(pilot_rows)} galaxias)")

    morphology_url = os.environ.get("TNG_MORPHOLOGY_CATALOG_URL", "")
    morphology_path = os.environ.get("TNG_MORPHOLOGY_CATALOG_PATH", "")
    try:
        morphology = ensure_morphology_catalog(
            args.out_cache,
            api_key,
            morphology_url=morphology_url,
            morphology_path=morphology_path,
            force=args.force_download,
        )
    except DownloadError as exc:
        morphology = None
        print(f"Advertencia: no pude descargar el catálogo morfológico ahora: {exc}")
        print("Continúo con cutouts/metadatos; run_projection_validation.py necesitará el catálogo más tarde.")
    if morphology:
        print(f"Catálogo morfológico listo: {morphology}")
    else:
        print("Catálogo morfológico pendiente; usa TNG_MORPHOLOGY_CATALOG_PATH o reintenta la URL oficial.")

    Path(args.out_cache).mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, str]] = []
    for index, row in enumerate(pilot_rows, start=1):
        print(f"[{index}/{len(pilot_rows)}] descargando {row.galaxy_id}")
        try:
            download_row_assets(
                row,
                args.out_cache,
                api_key,
                include_gas=args.include_gas,
                force=args.force_download,
                simulation=args.simulation,
            )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failures.append((row.galaxy_id, str(exc)))
            print(f"  fallo {row.galaxy_id}: {exc}")
            continue
    if failures:
        print("Fallos durante el bootstrap:")
        for galaxy_id, message in failures:
            print(f"  {galaxy_id}: {message}")
    print(f"Bootstrap piloto terminado en {args.out_cache}")


if __name__ == "__main__":
    main()
