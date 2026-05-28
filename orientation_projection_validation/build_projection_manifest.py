from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from orientation_validation.config import ProjectionConfig
from orientation_validation.manifest import build_projection_manifest, build_projection_manifest_from_matched, write_manifest
from orientation_validation.paths import default_catalog_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye un manifiesto deduplicado por galaxia TNG50.")
    parser.add_argument("--catalog", default=str(default_catalog_path()), help="Ruta a MaNGIA_catalog.fits")
    parser.add_argument("--matched-units", default="", help="matched_units.csv para generar solo galaxias únicas macheadas")
    parser.add_argument("--out", default="data/projection_manifest.csv", help="CSV de salida")
    parser.add_argument("--config", default="", help="Config JSON opcional")
    args = parser.parse_args()

    projection_config, _ = ProjectionConfig(), None
    if args.config:
        from orientation_validation.config import load_configs

        projection_config, _ = load_configs(args.config)

    if args.matched_units:
        rows = build_projection_manifest_from_matched(
            args.matched_units,
            primary_factor=projection_config.primary_rcov_reff,
            secondary_factor=projection_config.secondary_rcov_reff,
        )
        source = args.matched_units
    else:
        rows = build_projection_manifest(
            args.catalog,
            primary_factor=projection_config.primary_rcov_reff,
            secondary_factor=projection_config.secondary_rcov_reff,
        )
        source = args.catalog
    write_manifest(args.out, rows)
    total_mb = sum(row.estimated_raw_mb for row in rows)
    print(f"Manifest escrito en {args.out}")
    print(f"Fuente: {source}")
    print(f"Galaxias únicas: {len(rows)}")
    print(f"Estimación cruda stars+gas: {total_mb / 1024:.2f} GiB")


if __name__ == "__main__":
    main()
