from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kinematic_moments.models import KinematicMomentsConfig
from kinematic_moments.pipeline import collect_cube_paths, process_catalog, write_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract pPXF h3/h4 maps from official MaNGIA *.cube.fits.gz products."
    )
    parser.add_argument("--cube", help="Single official MaNGIA cube path.")
    parser.add_argument("--cube-glob", help='Glob for batch mode, e.g. "data/*.cube.fits.gz".')
    parser.add_argument("--manifest", help="CSV manifest with a cube_path column.")
    parser.add_argument("--outdir", default="kinematic_moments/output", help="Output directory.")
    parser.add_argument("--template-path", help="Path to MaStar_CB19.slog_1_5.fits.gz.")
    parser.add_argument("--n-workers", type=int, default=1, help="Number of worker processes for cube-level batch parallelism.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N galaxies after input ordering.")
    parser.add_argument("--progress-every", type=int, default=10, help="Print batch progress every N completed galaxies.")
    parser.add_argument("--max-spaxels", type=int, default=None, help="Limit fitted spaxels for smoke tests.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing NPZ/FITS outputs.")
    parser.add_argument("--snr-min", type=float, default=5.0, help="Minimum continuum S/N for quality=1.")
    parser.add_argument("--degree", type=int, default=4, help="Additive Legendre polynomial degree.")
    parser.add_argument("--mdegree", type=int, default=0, help="Multiplicative Legendre polynomial degree.")
    parser.add_argument("--bias", type=float, default=None, help="pPXF bias. Omit for pPXF default/None.")
    parser.add_argument("--max-templates", type=int, default=None, help="Use only the first N templates.")
    parser.add_argument("--verbose-ppxf", action="store_true", help="Show per-spaxel pPXF output.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = KinematicMomentsConfig(
        template_path=Path(args.template_path).expanduser() if args.template_path else None,
        snr_min=args.snr_min,
        degree=args.degree,
        mdegree=args.mdegree,
        bias=args.bias,
        max_templates=args.max_templates,
        quiet=not args.verbose_ppxf,
    )
    cube_paths = collect_cube_paths(args.cube, args.cube_glob, args.manifest, limit=args.limit)
    rows = process_catalog(
        cube_paths,
        args.outdir,
        config,
        n_workers=args.n_workers,
        max_spaxels=args.max_spaxels,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
    )
    manifest_path = write_manifest(rows, args.outdir)
    summary = {
        "n_cubes": len(rows),
        "n_ok": sum(row["status"] == "ok" for row in rows),
        "n_failed": sum(row["status"] == "failed" for row in rows),
        "n_skipped": sum(row["status"] == "skipped" for row in rows),
        "manifest_path": str(manifest_path),
        "n_workers": args.n_workers,
        "limit": args.limit,
        "progress_every": args.progress_every,
        "config": asdict(config),
    }
    if summary["config"].get("template_path") is not None:
        summary["config"]["template_path"] = str(summary["config"]["template_path"])
    print(json.dumps(summary, indent=2))
    return 0 if summary["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
