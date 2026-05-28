from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from validation.run_kinematic_validation import main as run_kinematic_validation


def _parse_thresholds(value: str) -> list[float]:
    thresholds = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        threshold = float(item)
        if threshold <= 0.0:
            raise argparse.ArgumentTypeError("thresholds must be positive")
        thresholds.append(threshold)
    if not thresholds:
        raise argparse.ArgumentTypeError("at least one threshold is required")
    return thresholds


def _threshold_label(threshold: float) -> str:
    return f"ratio_{threshold:.2f}".replace(".", "p")


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if result == result else None


def _diagnostic_summary(path: Path) -> dict[str, object]:
    summary: dict[str, object] = {
        "n_pass_test_a": 0,
        "n_fail_test_a": 0,
        "n_na_test_a": 0,
        "too_few_disk_spaxels": 0,
        "too_few_reference_spaxels": 0,
        "low_disk_vsigma_contrast": 0,
        "median_vsigma_ratio_applicable": None,
        "median_global_vsigma_applicable": None,
    }
    ratios = []
    global_vsigma = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            status = (row.get("test_a_rotation") or "").strip() or "N/A"
            reason = (row.get("test_a_failure_mode") or "").strip()
            if status == "PASS":
                summary["n_pass_test_a"] = int(summary["n_pass_test_a"]) + 1
            elif status == "FAIL":
                summary["n_fail_test_a"] = int(summary["n_fail_test_a"]) + 1
            else:
                summary["n_na_test_a"] = int(summary["n_na_test_a"]) + 1
            if reason in ("too_few_disk_spaxels", "too_few_reference_spaxels", "low_disk_vsigma_contrast"):
                summary[reason] = int(summary[reason]) + 1
            if status in ("PASS", "FAIL"):
                ratio = _float_or_none(row.get("v_over_sigma_ratio"))
                global_value = _float_or_none(row.get("v_over_sigma_global_median"))
                if ratio is not None:
                    ratios.append(ratio)
                if global_value is not None:
                    global_vsigma.append(global_value)
    if ratios:
        summary["median_vsigma_ratio_applicable"] = float(statistics.median(ratios))
    if global_vsigma:
        summary["median_global_vsigma_applicable"] = float(statistics.median(global_vsigma))
    return summary


def _validation_argv(args: argparse.Namespace, threshold: float, outdir: Path) -> list[str]:
    argv = [
        "--matched-units",
        args.matched_units,
        "--labels-dir",
        args.labels_dir,
        "--outdir",
        str(outdir),
        "--label-mode",
        args.label_mode,
        "--dominant-class-threshold",
        str(args.dominant_class_threshold),
        "--min-spaxels-for-test",
        str(args.min_spaxels_for_test),
        "--min-spaxels-test-b",
        str(args.min_spaxels_test_b),
        "--rotation-test",
        "contrast",
        "--test-a-reference",
        args.test_a_reference,
        "--central-reference-radius-fraction",
        str(args.central_reference_radius_fraction),
        "--disk-vsigma-ratio-min",
        f"{threshold:.6g}",
        "--min-sigma-star",
        str(args.min_sigma_star),
        "--sigma-ratio-min",
        str(args.sigma_ratio_min),
        "--bar-tolerance",
        str(args.bar_tolerance),
        "--rho-h3v-min",
        str(args.rho_h3v_min),
    ]
    if args.kinematics_dir:
        argv.extend(["--kinematics-dir", args.kinematics_dir])
    if args.limit > 0:
        argv.extend(["--limit", str(args.limit)])
    if args.continue_on_error:
        argv.append("--continue-on-error")
    if args.no_center_velocity:
        argv.append("--no-center-velocity")
    return argv


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "threshold",
        "run_dir",
        "exit_code",
        "n_units_total",
        "n_units_skipped",
        "reference_mode",
        "central_reference_radius_fraction",
        "n_applicable_test_a",
        "n_pass_test_a",
        "n_fail_test_a",
        "n_na_test_a",
        "success_rate_test_a",
        "success_rate_overall",
        "n_applicable_test_b",
        "success_rate_test_b",
        "min_spaxels_for_test",
        "min_spaxels_test_b",
        "median_vsigma_ratio_applicable",
        "median_global_vsigma_applicable",
        "too_few_disk_spaxels",
        "too_few_reference_spaxels",
        "low_disk_vsigma_contrast",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def _fmt(value: object, digits: int = 1) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_summary_markdown(path: Path, rows: list[dict[str, object]]) -> Path:
    lines = [
        "# Test A sensitivity",
        "",
        "| threshold | applicable | pass | fail | N/A | success A | overall | median ratio | low contrast | few disk | few ref |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{_fmt(row.get('threshold'), 2)} | "
            f"{row.get('n_applicable_test_a', '')} | "
            f"{row.get('n_pass_test_a', '')} | "
            f"{row.get('n_fail_test_a', '')} | "
            f"{row.get('n_na_test_a', '')} | "
            f"{_fmt(row.get('success_rate_test_a'))} | "
            f"{_fmt(row.get('success_rate_overall'))} | "
            f"{_fmt(row.get('median_vsigma_ratio_applicable'), 3)} | "
            f"{row.get('low_disk_vsigma_contrast', '')} | "
            f"{row.get('too_few_disk_spaxels', '')} | "
            f"{row.get('too_few_reference_spaxels', '')} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    max_exit_code = 0

    for threshold in args.thresholds:
        run_dir = outdir / _threshold_label(threshold)
        report_path = run_dir / "kinematic_validation_report.json"
        diagnostics_path = run_dir / "test_a_diagnostics.csv"
        if args.skip_existing and report_path.exists() and diagnostics_path.exists():
            exit_code = 0
            print(f"[skip] threshold={threshold:.2f} using {run_dir}", flush=True)
        else:
            print(f"[run] threshold={threshold:.2f} outdir={run_dir}", flush=True)
            exit_code = run_kinematic_validation(_validation_argv(args, threshold, run_dir))
            max_exit_code = max(max_exit_code, int(exit_code))
        if not report_path.exists() or not diagnostics_path.exists():
            raise SystemExit(f"Missing validation outputs for threshold={threshold:.2f}: {run_dir}")

        report = json.loads(report_path.read_text(encoding="utf-8"))
        diagnostics = _diagnostic_summary(diagnostics_path)
        row = {
            "threshold": threshold,
            "run_dir": str(run_dir),
            "exit_code": exit_code,
            "n_units_total": report.get("n_units_total"),
            "n_units_skipped": report.get("n_units_skipped"),
            "reference_mode": report.get("rotation_reference_mode"),
            "central_reference_radius_fraction": report.get("central_reference_radius_fraction"),
            "n_applicable_test_a": report.get("n_applicable_test_a"),
            "success_rate_test_a": report.get("success_rate_test_a"),
            "success_rate_overall": report.get("success_rate_overall"),
            "n_applicable_test_b": report.get("n_applicable_test_b"),
            "success_rate_test_b": report.get("success_rate_test_b"),
            "min_spaxels_for_test": report.get("min_spaxels_for_test"),
            "min_spaxels_test_b": report.get("min_spaxels_test_b"),
            **diagnostics,
        }
        rows.append(row)

    summary_csv = _write_summary_csv(outdir / "test_a_sensitivity_summary.csv", rows)
    summary_md = _write_summary_markdown(outdir / "test_a_sensitivity_summary.md", rows)
    print(json.dumps({"summary_csv": str(summary_csv), "summary_md": str(summary_md), "runs": rows}, indent=2, sort_keys=True))
    return int(max_exit_code)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Test A central-reference sensitivity over multiple V/sigma thresholds.")
    parser.add_argument("--matched-units", default="/home/aespinola/Documents/pythonprojects/datacubes/matched_assets/matched_units.csv")
    parser.add_argument("--labels-dir", default="/media/nuevo/structural_labels")
    parser.add_argument("--outdir", default="/media/nuevo/structural_validations/kinematic_central_sensitivity")
    parser.add_argument("--thresholds", type=_parse_thresholds, default=[1.00, 1.10, 1.20], help="Comma-separated ratios, e.g. 1.00,1.10,1.20")
    parser.add_argument("--label-mode", choices=("soft_mass", "soft_light"), default="soft_mass")
    parser.add_argument("--test-a-reference", choices=("bulge_other", "bulge", "central"), default="central")
    parser.add_argument("--central-reference-radius-fraction", type=float, default=0.25)
    parser.add_argument("--dominant-class-threshold", type=float, default=0.50)
    parser.add_argument("--min-spaxels-for-test", type=int, default=30)
    parser.add_argument("--min-spaxels-test-b", type=int, default=10)
    parser.add_argument("--min-sigma-star", type=float, default=1.0)
    parser.add_argument("--sigma-ratio-min", type=float, default=1.10)
    parser.add_argument("--bar-tolerance", type=float, default=0.05)
    parser.add_argument("--rho-h3v-min", type=float, default=0.20)
    parser.add_argument("--kinematics-dir", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-center-velocity", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse completed threshold subruns")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
