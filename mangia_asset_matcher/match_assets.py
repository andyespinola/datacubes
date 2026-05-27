from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .matcher import MatchConfig, build_matches, write_outputs


DEFAULT_CUBE_ROOT = "/media/nuevo/output_cubos"
DEFAULT_TNG_CACHE = "/media/nuevo/tng_cutouts"
DEFAULT_MAPS2D_ROOT = "/home/aespinola/Documents/pythonprojects/datacubes/maps2D"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Match reconstructed MaNGIA cubes, TNG cutouts, and MaNGIA 2D maps."
    )
    parser.add_argument("--catalog", default="MaNGIA_catalog.fits", help="MaNGIA catalog FITS/CSV path.")
    parser.add_argument(
        "--cube-root",
        action="append",
        default=[],
        help=f"Root with reconstructed *.cube.fits(.gz) files. Repeatable. Default: {DEFAULT_CUBE_ROOT}",
    )
    parser.add_argument("--tng-cache", default=DEFAULT_TNG_CACHE, help="TNG cache root with cutouts/metadata/morphology.")
    parser.add_argument(
        "--maps2d-root",
        action="append",
        default=[],
        help=f"Root with MaNGIA 2D map files. Repeatable. Default: {DEFAULT_MAPS2D_ROOT}",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum selected units. 0 means all strict matches.")
    parser.add_argument("--require-count", type=int, default=0, help="Fail if fewer than N selected strict matches are available.")
    parser.add_argument(
        "--selection-order",
        choices=("estimated_raw_mb", "catalog_order", "random"),
        default="estimated_raw_mb",
        help="How to order strict matches before applying --limit.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used only with --selection-order=random.")
    parser.add_argument("--outdir", default="matched_assets", help="Output directory for manifests and reports.")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing output files.")
    return parser


def _paths(values: list[str], default: str) -> tuple[Path, ...]:
    selected = values or [default]
    return tuple(Path(value).expanduser() for value in selected)


def run(args: argparse.Namespace) -> int:
    config = MatchConfig(
        catalog=Path(args.catalog).expanduser(),
        cube_roots=_paths(args.cube_root, DEFAULT_CUBE_ROOT),
        tng_cache=Path(args.tng_cache).expanduser(),
        maps2d_roots=_paths(args.maps2d_root, DEFAULT_MAPS2D_ROOT),
        limit=int(args.limit),
        require_count=int(args.require_count),
        selection_order=args.selection_order,
        seed=int(args.seed),
    )
    result = build_matches(config)
    report = result.report
    print(
        "MaNGIA asset matcher: "
        f"catalog_units={report['n_catalog_units']} "
        f"strict_matches={report['n_strict_matches']} "
        f"selected={report['n_selected']} "
        f"limit={report['limit']} "
        f"require_count={report['require_count']}"
    )
    if not args.dry_run:
        write_outputs(result, args.outdir)
        print(f"Outputs written to {Path(args.outdir).resolve()}")

    if int(args.require_count) > 0 and int(report["n_selected"]) < int(args.require_count):
        print(
            f"ERROR: selected {report['n_selected']} strict matches, "
            f"but --require-count={args.require_count}",
            file=sys.stderr,
        )
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
