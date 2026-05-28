from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .kinematic import (
    KinematicChecks,
    KinematicValidationConfig,
    KinematicValidationInput,
    build_success_report,
    load_kinematic_moments,
    load_label_tensor,
    load_map_pair,
    r_bar_from_summary,
    validate_kinematic_unit,
    write_report_json,
    write_report_markdown,
    write_score_histogram,
    write_test_a_diagnostics_csv,
    write_test_a_extreme_pass_fail_markdown,
    write_test_a_summary_by_global_vsigma_csv,
    write_test_a_summary_by_sample_csv,
    write_test_a_summary_by_view_csv,
    write_unit_results_csv,
)


def _labels_base(labels_dir: Path, canonical_id: str) -> Path:
    return labels_dir / canonical_id


def _kinematics_path(kinematics_dir: Path | None, row: dict[str, str]) -> Path | None:
    if kinematics_dir is None:
        return None
    cube_path = Path(row.get("cube_path", ""))
    candidates = []
    if cube_path.name:
        name = cube_path.name
        for suffix in (".cube.fits.gz", ".cube.fits"):
            if name.endswith(suffix):
                candidates.append(kinematics_dir / f"{name.removesuffix(suffix)}.kinematics_ppxf.npz")
    canonical = row.get("canonical_id", "")
    if canonical:
        candidates.append(kinematics_dir / f"{canonical}.kinematics_ppxf.npz")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _optional_int(row: dict[str, str], name: str) -> int | None:
    value = (row.get(name) or "").strip()
    return int(float(value)) if value else None


def _error_result(row: dict[str, str], labels_dir: Path, status: str, error: str) -> KinematicChecks:
    canonical_id = row.get("canonical_id", "")
    unit_id = row.get("unit_id", "")
    return KinematicChecks(
        unit_id=unit_id,
        galaxy_id=row.get("galaxy_id", ""),
        canonical_id=canonical_id,
        view_id=int(float(row.get("view") or 0)),
        test_a_rotation="N/A",
        test_b_dispersion="N/A",
        test_c_bar_sigma="N/A",
        test_d_h3_signature="N/A",
        rotation_test_mode="",
        rho_disk=None,
        v_over_sigma_disk_median=None,
        v_over_sigma_reference_median=None,
        v_over_sigma_ratio=None,
        sigma_ratio=None,
        sigma_bulge_median=None,
        sigma_disk_median=None,
        sigma_bar_median=None,
        rho_h3v=None,
        n_tests_applicable=0,
        n_tests_passed=0,
        coherence_score=float("nan"),
        passes=False,
        h3h4_used=False,
        sample_manga=_optional_int(row, "sample_manga"),
        label_path=str(_labels_base(labels_dir, canonical_id).with_suffix(".labels.npz")),
        maps2d_path=row.get("maps2d_path", ""),
        status=status,
        error=error,
    )


def _validate_row(
    row: dict[str, str],
    labels_dir: Path,
    label_mode: str,
    kinematics_dir: Path | None,
    config: KinematicValidationConfig,
) -> KinematicChecks:
    canonical_id = row.get("canonical_id", "")
    label_path = _labels_base(labels_dir, canonical_id).with_suffix(".labels.npz")
    summary_path = _labels_base(labels_dir, canonical_id).with_suffix(".summary.json")
    if not label_path.exists():
        raise FileNotFoundError(f"missing labels: {label_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary: {summary_path}")
    y_int, valid_mask = load_label_tensor(label_path, label_mode)
    v_star, sigma_star = load_map_pair(
        row["maps2d_path"],
        row["v_map_key"],
        row["sigma_map_key"],
        row.get("maps2d_format", ""),
    )
    moments = load_kinematic_moments(_kinematics_path(kinematics_dir, row))
    unit = KinematicValidationInput(
        unit_id=row.get("unit_id", ""),
        galaxy_id=row.get("galaxy_id", ""),
        canonical_id=canonical_id,
        view_id=int(float(row.get("view") or 0)),
        y_int=y_int,
        m_val=valid_mask,
        v_star=v_star,
        sigma_star=sigma_star,
        r_bar=r_bar_from_summary(summary_path),
        kinematic_moments=moments,
        label_path=str(label_path),
        maps2d_path=row.get("maps2d_path", ""),
        sample_manga=_optional_int(row, "sample_manga"),
    )
    return validate_kinematic_unit(unit, config)


def run(args: argparse.Namespace) -> int:
    matched_units = Path(args.matched_units).expanduser()
    labels_dir = Path(args.labels_dir).expanduser()
    outdir = Path(args.outdir).expanduser()
    kinematics_dir = Path(args.kinematics_dir).expanduser() if args.kinematics_dir else None
    outdir.mkdir(parents=True, exist_ok=True)

    with matched_units.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No rows to validate")

    config = KinematicValidationConfig(
        dominant_class_threshold=args.dominant_class_threshold,
        rotation_test_mode=args.rotation_test,
        rho_disk_min=args.rho_disk_min,
        disk_vsigma_ratio_min=args.disk_vsigma_ratio_min,
        sigma_ratio_min=args.sigma_ratio_min,
        bar_tolerance=args.bar_tolerance,
        rho_h3v_min=args.rho_h3v_min,
        min_spaxels_for_test=args.min_spaxels_for_test,
    )
    results: list[KinematicChecks] = []
    for index, row in enumerate(rows, start=1):
        try:
            print(f"[{index}/{len(rows)}] validating {row.get('canonical_id')}", flush=True)
            result = _validate_row(row, labels_dir, args.label_mode, kinematics_dir, config)
            results.append(result)
            print(f"  score={result.coherence_score:.3f} passes={result.passes}", flush=True)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            results.append(_error_result(row, labels_dir, "skipped", str(exc)))
            print(f"  skipped: {exc}", flush=True)

    skipped = sum(result.status != "ok" for result in results)
    report = build_success_report(results, n_units_skipped=skipped, config=config)
    unit_csv = write_unit_results_csv(outdir / "kinematic_validation_units.csv", results)
    report_json = write_report_json(outdir / "kinematic_validation_report.json", report)
    report_md = write_report_markdown(outdir / "kinematic_validation_report.md", report)
    histogram = write_score_histogram(outdir / "coherence_score_histogram.png", results)
    test_a_diagnostics = write_test_a_diagnostics_csv(outdir / "test_a_diagnostics.csv", results)
    test_a_by_view = write_test_a_summary_by_view_csv(outdir / "test_a_summary_by_view.csv", results)
    test_a_by_sample = write_test_a_summary_by_sample_csv(outdir / "test_a_summary_by_sample.csv", results)
    test_a_by_global_vsigma = write_test_a_summary_by_global_vsigma_csv(outdir / "test_a_summary_by_global_vsigma.csv", results)
    test_a_extremes = write_test_a_extreme_pass_fail_markdown(
        outdir / "test_a_extreme_pass_fail.md",
        results,
        ratio_threshold=args.disk_vsigma_ratio_min,
    )
    summary = {
        "unit_csv": str(unit_csv),
        "report_json": str(report_json),
        "report_md": str(report_md),
        "histogram": str(histogram),
        "test_a_diagnostics": str(test_a_diagnostics),
        "test_a_summary_by_view": str(test_a_by_view),
        "test_a_summary_by_sample": str(test_a_by_sample),
        "test_a_summary_by_global_vsigma": str(test_a_by_global_vsigma),
        "test_a_extreme_pass_fail": str(test_a_extremes),
        "report": json.loads(report_json.read_text(encoding="utf-8")),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if skipped == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate structural pseudo-labels against Pipe3D stellar kinematics.")
    parser.add_argument("--matched-units", required=True, help="matched_units.csv usado para etiquetar")
    parser.add_argument("--labels-dir", required=True, help="Directorio con *.labels.npz y *.summary.json")
    parser.add_argument("--outdir", required=True, help="Directorio de reportes de validacion")
    parser.add_argument("--label-mode", choices=("soft_mass", "soft_light"), default="soft_mass")
    parser.add_argument("--kinematics-dir", default="", help="Directorio opcional de *.kinematics_ppxf.npz con h3/h4")
    parser.add_argument("--limit", type=int, default=0, help="Maximo de filas a validar; 0=todas")
    parser.add_argument("--continue-on-error", action="store_true", help="Registra errores y continua")
    parser.add_argument("--dominant-class-threshold", type=float, default=0.70)
    parser.add_argument("--rotation-test", choices=("contrast", "spearman"), default="contrast")
    parser.add_argument("--rho-disk-min", type=float, default=0.20)
    parser.add_argument("--disk-vsigma-ratio-min", type=float, default=1.10)
    parser.add_argument("--sigma-ratio-min", type=float, default=1.10)
    parser.add_argument("--bar-tolerance", type=float, default=0.05)
    parser.add_argument("--rho-h3v-min", type=float, default=0.20)
    parser.add_argument("--min-spaxels-for-test", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
