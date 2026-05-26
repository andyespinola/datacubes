from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kinematic_moments.io import resolve_template_path
from kinematic_moments.models import KinematicMomentsConfig
from kinematic_moments.pipeline import append_run_log, collect_cube_paths, process_catalog, write_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract pPXF h3/h4 maps from official MaNGIA *.cube.fits.gz products."
    )
    parser.add_argument("--cube", help="Single official MaNGIA cube path.")
    parser.add_argument("--cube-glob", help='Glob for batch mode, e.g. "data/*.cube.fits.gz".')
    parser.add_argument("--manifest", help="CSV manifest with a cube_path column.")
    parser.add_argument("--outdir", default="kinematic_moments/output", help="Output directory.")
    parser.add_argument("--log-file", help="Run log path. Defaults to <outdir>/kinematics_run.log.")
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
    output_dir = Path(args.outdir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file).expanduser().resolve() if args.log_file else output_dir / "kinematics_run.log"
    append_run_log(log_path, "=" * 80)
    append_run_log(log_path, f"RUN START argv={' '.join(sys.argv)}")

    try:
        config = KinematicMomentsConfig(
            template_path=Path(args.template_path).expanduser() if args.template_path else None,
            snr_min=args.snr_min,
            degree=args.degree,
            mdegree=args.mdegree,
            bias=args.bias,
            max_templates=args.max_templates,
            quiet=not args.verbose_ppxf,
        )
        template_path = resolve_template_path(config.template_path)
        config = KinematicMomentsConfig(**{**asdict(config), "template_path": template_path})
        append_run_log(log_path, f"TEMPLATE path={template_path}")

        cube_paths = collect_cube_paths(args.cube, args.cube_glob, args.manifest, limit=args.limit)
        append_run_log(
            log_path,
            f"INPUT n_cubes={len(cube_paths)} cube={args.cube!r} cube_glob={args.cube_glob!r} "
            f"manifest={args.manifest!r} limit={args.limit}",
        )
        if cube_paths:
            append_run_log(log_path, f"INPUT first_cube={cube_paths[0]} last_cube={cube_paths[-1]}")

        rows = process_catalog(
            cube_paths,
            output_dir,
            config,
            n_workers=args.n_workers,
            max_spaxels=args.max_spaxels,
            overwrite=args.overwrite,
            progress_every=args.progress_every,
            log_path=log_path,
        )
        manifest_path = write_manifest(rows, output_dir, log_path=log_path)
        summary = {
            "n_cubes": len(rows),
            "n_ok": sum(row["status"] == "ok" for row in rows),
            "n_failed": sum(row["status"] == "failed" for row in rows),
            "n_skipped": sum(row["status"] == "skipped" for row in rows),
            "manifest_path": str(manifest_path),
            "log_path": str(log_path),
            "n_workers": args.n_workers,
            "limit": args.limit,
            "progress_every": args.progress_every,
            "config": asdict(config),
        }
        if summary["config"].get("template_path") is not None:
            summary["config"]["template_path"] = str(summary["config"]["template_path"])
        append_run_log(log_path, "RUN END " + json.dumps(summary, default=str, sort_keys=True))
        print(json.dumps(summary, indent=2))
        return 0 if summary["n_failed"] == 0 else 1
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}"
        append_run_log(log_path, f"RUN FAIL message={message}")
        print(
            json.dumps(
                {
                    "status": "failed",
                    "log_path": str(log_path),
                    "message": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
