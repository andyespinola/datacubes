from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

from labeling.manifest import build_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye un manifiesto CSV para el pipeline de etiquetas estructurales.")
    parser.add_argument("--catalog", required=True, help="Ruta a MaNGIA_catalog.fits")
    parser.add_argument("--rss-glob", required=True, help="Glob para los RSS MaNGIA")
    parser.add_argument("--cube-glob", required=True, help="Glob para los cubos MaNGIA")
    parser.add_argument("--pipe3d-glob", default="", help="Glob opcional para productos pyPipe3D")
    parser.add_argument("--out", required=True, help="CSV de salida")
    args = parser.parse_args()

    rss_paths = [Path(path) for path in sorted(glob(args.rss_glob))]
    cube_paths = [Path(path) for path in sorted(glob(args.cube_glob))]
    pipe3d_paths = [Path(path) for path in sorted(glob(args.pipe3d_glob))] if args.pipe3d_glob else []
    rows = build_manifest(args.catalog, rss_paths, cube_paths, pipe3d_paths)
    write_manifest(args.out, rows)
    print(f"Manifest escrito en {args.out} con {len(rows)} filas")


if __name__ == "__main__":
    main()
