from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from aperturenet_labels.config import PipelineConfig, resolve_project_path
from aperturenet_labels.pipeline import run_selected, validate_local_data


def _load_config(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig.from_yaml(args.config)
    if getattr(args, "data_dir", ""):
        config.data.data_dir = resolve_project_path(args.data_dir)
    if getattr(args, "outdir", ""):
        config.data.output_dir = resolve_project_path(args.outdir)
    if getattr(args, "max_particles", None) is not None:
        config.extractor.max_particles = int(args.max_particles)
    if getattr(args, "no_copy_cube", False):
        config.packer.include_cube = False
    return config


def cmd_validate_data(args: argparse.Namespace) -> int:
    config = _load_config(args)
    rows = validate_local_data(config)
    print(json.dumps(rows, indent=2, sort_keys=True))
    failures = [row for row in rows if row["missing"]]
    return 1 if failures else 0


def cmd_run(args: argparse.Namespace) -> int:
    config = _load_config(args)
    outputs = run_selected(config, args.galaxy_id, args.all_local, overwrite=args.overwrite)
    print(json.dumps([output.as_dict() for output in outputs], indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aperturenet-labels", description="Soft Labels Generation v2.1")
    parser.add_argument("--config", default="", help="YAML config path, relative to soft_labels_generation_v2_1/")
    sub = parser.add_subparsers(dest="command", required=True)

    validate_parser = sub.add_parser("validate-data", help="Validate local data assets")
    validate_parser.add_argument("--data-dir", default="", help="Override data directory")
    validate_parser.add_argument("--outdir", default="", help=argparse.SUPPRESS)
    validate_parser.add_argument("--max-particles", type=int, default=None, help=argparse.SUPPRESS)
    validate_parser.add_argument("--no-copy-cube", action="store_true", help=argparse.SUPPRESS)
    validate_parser.set_defaults(func=cmd_validate_data)

    run_parser = sub.add_parser("run", help="Run the local pipeline")
    run_parser.add_argument("--galaxy-id", action="append", default=[], help="Galaxy id, e.g. TNG50-87-155298; can repeat")
    run_parser.add_argument("--all-local", action="store_true", help="Run all configured local galaxies")
    run_parser.add_argument("--data-dir", default="", help="Override data directory")
    run_parser.add_argument("--outdir", default="", help="Override output directory")
    run_parser.add_argument("--max-particles", type=int, default=None, help="Override extractor.max_particles; 0 means all valid stars")
    run_parser.add_argument("--no-copy-cube", action="store_true", help="Write a lightweight dataset_entry without copying the full IFU cube")
    run_parser.add_argument("--overwrite", action="store_true", help="Recompute even if dataset_entry already exists")
    run_parser.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
