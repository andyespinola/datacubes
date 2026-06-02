from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATCHED_UNITS = Path("/home/aespinola/Documents/pythonprojects/datacubes/matched_assets/matched_units.csv")
DEFAULT_EPSILON_LABELS = Path("/media/nuevo/epsilon_labels")
DEFAULT_KINEMATIC_OUTDIR = Path("/media/nuevo/structural_validations/kinematic_epsilon_a10_b10")
DEFAULT_PROJECTION_OUTDIR = Path("/media/nuevo/orientation_projection_validation/outputs_matched_epsilon")
DEFAULT_ORIENTATION_CSV = Path("/media/nuevo/orientation_projection_validation/catalog_interorientation_summary_epsilon.csv")
DEFAULT_ORIENTATION_MD = Path("/media/nuevo/orientation_projection_validation/catalog_interorientation_summary_epsilon.md")
DEFAULT_COMPARISON_MD = Path("/media/nuevo/structural_validations/gmm_vs_epsilon_baseline.md")
DEFAULT_STATE = Path("/media/nuevo/structural_validations/epsilon_followup_pipeline_state.json")
DEFAULT_LOG = Path("/media/nuevo/structural_validations/epsilon_followup_pipeline.log")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_matched_units(path: Path, limit: int = 0) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit > 0 else rows


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message)
        if not message.endswith("\n"):
            handle.write("\n")


def _labels_status(rows: list[dict[str, str]], labels_dir: Path) -> tuple[int, int, list[str]]:
    missing: list[str] = []
    for row in rows:
        canonical_id = (row.get("canonical_id") or "").strip()
        if not canonical_id:
            missing.append("<missing canonical_id>")
            continue
        base = labels_dir / canonical_id
        if not base.with_suffix(".labels.npz").exists() or not base.with_suffix(".summary.json").exists():
            missing.append(canonical_id)
    return len(rows) - len(missing), len(rows), missing


def _wait_for_labels(args: argparse.Namespace, state: dict) -> None:
    rows = _read_matched_units(Path(args.matched_units).expanduser(), args.limit)
    required = int(args.require_label_count or len(rows))
    if required > len(rows):
        raise SystemExit(f"--require-label-count={required} excede filas disponibles={len(rows)}")

    start = time.monotonic()
    while True:
        complete, total, missing = _labels_status(rows, Path(args.epsilon_labels).expanduser())
        state["wait_for_labels"] = {
            "status": "complete" if complete >= required else "waiting",
            "complete": complete,
            "total": total,
            "required": required,
            "missing_preview": missing[:20],
            "updated_at": _now(),
        }
        _write_state(Path(args.state), state)
        print(f"[wait-labels] complete={complete}/{total} required={required}", flush=True)
        if complete >= required:
            return
        if args.label_timeout_seconds > 0 and time.monotonic() - start >= args.label_timeout_seconds:
            raise SystemExit(f"Timeout esperando etiquetas epsilon: complete={complete}/{total}")
        time.sleep(max(float(args.poll_seconds), 1.0))


def _run_stage(args: argparse.Namespace, state: dict, name: str, command: list[str]) -> None:
    state[name] = {"status": "running", "command": command, "started_at": _now()}
    _write_state(Path(args.state), state)
    _append_log(Path(args.log), f"\n[{_now()}] START {name}\n$ {' '.join(command)}\n")
    print(f"\n[{name}] {' '.join(command)}", flush=True)

    process = subprocess.Popen(
        command,
        cwd=str(Path(args.repo_root).expanduser()),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        _append_log(Path(args.log), line.rstrip("\n"))
    return_code = process.wait()
    state[name].update({"status": "complete" if return_code == 0 else "failed", "return_code": return_code, "finished_at": _now()})
    _write_state(Path(args.state), state)
    _append_log(Path(args.log), f"[{_now()}] END {name} return_code={return_code}\n")
    if return_code != 0:
        raise SystemExit(f"Stage {name} failed with return_code={return_code}")


def _default_path(repo_root: Path, relative: str) -> Path:
    return repo_root / relative


def run(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).expanduser()
    args.repo_root = str(repo_root)
    args.state = str(Path(args.state).expanduser())
    args.log = str(Path(args.log).expanduser())

    state: dict = {
        "pipeline": "epsilon_followup",
        "started_at": _now(),
        "repo_root": str(repo_root),
        "matched_units": str(Path(args.matched_units).expanduser()),
        "epsilon_labels": str(Path(args.epsilon_labels).expanduser()),
    }
    _write_state(Path(args.state), state)
    _append_log(Path(args.log), f"[{_now()}] epsilon follow-up pipeline started")

    if not args.no_wait_for_labels:
        _wait_for_labels(args, state)

    python = sys.executable
    projection_manifest = Path(args.projection_manifest).expanduser()
    if args.force_build_projection_manifest or not projection_manifest.exists():
        build_manifest_cmd = [
            python,
            "-m",
            "orientation_projection_validation.build_projection_manifest",
            "--matched-units",
            str(Path(args.matched_units).expanduser()),
            "--out",
            str(projection_manifest),
            "--config",
            str(Path(args.projection_config).expanduser()),
        ]
        if args.catalog:
            build_manifest_cmd.extend(["--catalog", args.catalog])
        _run_stage(args, state, "build_projection_manifest", build_manifest_cmd)

    kinematic_cmd = [
        python,
        "-m",
        "validation.run_kinematic_validation",
        "--matched-units",
        str(Path(args.matched_units).expanduser()),
        "--labels-dir",
        str(Path(args.epsilon_labels).expanduser()),
        "--outdir",
        str(Path(args.kinematic_outdir).expanduser()),
        "--label-mode",
        args.label_mode,
        "--dominant-class-threshold",
        str(args.dominant_class_threshold),
        "--min-spaxels-for-test",
        str(args.min_spaxels_for_test),
        "--min-spaxels-test-b",
        str(args.min_spaxels_test_b),
        "--continue-on-error",
    ]
    if args.kinematics_dir:
        kinematic_cmd.extend(["--kinematics-dir", str(Path(args.kinematics_dir).expanduser())])
    if args.limit > 0:
        kinematic_cmd.extend(["--limit", str(args.limit)])
    _run_stage(args, state, "kinematic_validation", kinematic_cmd)

    projection_cmd = [
        python,
        "-m",
        "orientation_projection_validation.run_projection_validation",
        "--manifest",
        str(projection_manifest),
        "--cache",
        str(Path(args.tng_cache).expanduser()),
        "--morphology-catalog",
        str(Path(args.morphology_catalog).expanduser()),
        "--config",
        str(Path(args.projection_config).expanduser()),
        "--outdir",
        str(Path(args.projection_outdir).expanduser()),
        "--label-model",
        "epsilon",
        "--epsilon-threshold",
        str(args.epsilon_threshold),
        "--continue-on-error",
    ]
    if args.max_projection_galaxies > 0:
        projection_cmd.extend(["--max-galaxies", str(args.max_projection_galaxies)])
    _run_stage(args, state, "projection_validation", projection_cmd)

    summarize_cmd = [
        python,
        str(repo_root / "orientation_projection_validation" / "summarize_metrics.py"),
        "--metrics-glob",
        str(Path(args.projection_outdir).expanduser() / "*" / "metrics.json"),
        "--out",
        str(Path(args.orientation_csv).expanduser()),
        "--report",
        str(Path(args.orientation_md).expanduser()),
    ]
    _run_stage(args, state, "summarize_orientation", summarize_cmd)

    compare_cmd = [
        python,
        "-m",
        "validation.compare_baseline_reports",
        "--gmm-kinematic",
        str(Path(args.gmm_kinematic_report).expanduser()),
        "--epsilon-kinematic",
        str(Path(args.kinematic_outdir).expanduser() / "kinematic_validation_report.json"),
        "--gmm-orientation",
        str(Path(args.gmm_orientation_csv).expanduser()),
        "--epsilon-orientation",
        str(Path(args.orientation_csv).expanduser()),
        "--out",
        str(Path(args.comparison_md).expanduser()),
    ]
    _run_stage(args, state, "compare_reports", compare_cmd)

    state["finished_at"] = _now()
    state["status"] = "complete"
    state["outputs"] = {
        "kinematic_report": str(Path(args.kinematic_outdir).expanduser() / "kinematic_validation_report.md"),
        "orientation_report": str(Path(args.orientation_md).expanduser()),
        "comparison_report": str(Path(args.comparison_md).expanduser()),
        "state": str(Path(args.state).expanduser()),
        "log": str(Path(args.log).expanduser()),
    }
    _write_state(Path(args.state), state)
    print(json.dumps(state["outputs"], indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    repo_root = DEFAULT_REPO_ROOT
    parser = argparse.ArgumentParser(description="Run all post-epsilon-label baseline validation steps.")
    parser.add_argument("--repo-root", default=str(repo_root))
    parser.add_argument("--matched-units", default=str(DEFAULT_MATCHED_UNITS))
    parser.add_argument("--catalog", default="MaNGIA_catalog.fits")
    parser.add_argument("--epsilon-labels", default=str(DEFAULT_EPSILON_LABELS))
    parser.add_argument("--no-wait-for-labels", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--label-timeout-seconds", type=float, default=0.0, help="0 means wait indefinitely")
    parser.add_argument("--require-label-count", type=int, default=0, help="0 means all rows in matched_units")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for labels/kinematic validation")
    parser.add_argument("--kinematic-outdir", default=str(DEFAULT_KINEMATIC_OUTDIR))
    parser.add_argument("--label-mode", choices=("soft_mass", "soft_light"), default="soft_mass")
    parser.add_argument("--dominant-class-threshold", type=float, default=0.50)
    parser.add_argument("--min-spaxels-for-test", type=int, default=10)
    parser.add_argument("--min-spaxels-test-b", type=int, default=10)
    parser.add_argument("--kinematics-dir", default="")
    parser.add_argument("--projection-manifest", default=str(_default_path(repo_root, "orientation_projection_validation/data/projection_manifest_matched.csv")))
    parser.add_argument("--force-build-projection-manifest", action="store_true")
    parser.add_argument("--tng-cache", default="/media/nuevo/tng_cutouts")
    parser.add_argument("--morphology-catalog", default="/media/nuevo/tng_cutouts/morphology/morphs_kinematic_bars.hdf5")
    parser.add_argument("--projection-config", default=str(_default_path(repo_root, "orientation_projection_validation/default_config.json")))
    parser.add_argument("--projection-outdir", default=str(DEFAULT_PROJECTION_OUTDIR))
    parser.add_argument("--epsilon-threshold", type=float, default=0.70)
    parser.add_argument("--max-projection-galaxies", type=int, default=0)
    parser.add_argument("--orientation-csv", default=str(DEFAULT_ORIENTATION_CSV))
    parser.add_argument("--orientation-md", default=str(DEFAULT_ORIENTATION_MD))
    parser.add_argument("--gmm-kinematic-report", default="/media/nuevo/structural_validations/kinematic_central_a10_b10/kinematic_validation_report.json")
    parser.add_argument("--gmm-orientation-csv", default="/media/nuevo/orientation_projection_validation/catalog_interorientation_summary_matched.csv")
    parser.add_argument("--comparison-md", default=str(DEFAULT_COMPARISON_MD))
    parser.add_argument("--state", default=str(DEFAULT_STATE))
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
