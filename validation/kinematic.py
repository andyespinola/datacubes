from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from scipy.stats import spearmanr


Status = Literal["PASS", "FAIL", "N/A"]
RotationTestMode = Literal["contrast", "spearman"]
RotationReferenceMode = Literal["bulge_other", "bulge", "central"]

LABEL_CLASS_NAMES = ("bulge", "disk", "bar", "arm", "other")
LABEL_CLASS_ALIASES = {
    "bulge": ("bulge", "bulbo"),
    "disk": ("disk", "disco"),
    "bar": ("bar", "barra"),
    "arm": ("arm", "arms", "brazos"),
    "other": ("other",),
}


@dataclass(frozen=True, slots=True)
class KinematicMomentMaps:
    h3: np.ndarray
    h4: np.ndarray
    quality_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class KinematicValidationConfig:
    dominant_class_threshold: float = 0.70
    epsilon: float = 1e-8
    center_velocity: bool = True
    min_sigma_star: float = 1.0
    rotation_test_mode: RotationTestMode = "contrast"
    rotation_reference_mode: RotationReferenceMode = "central"
    central_reference_radius_fraction: float = 0.25
    rho_disk_min: float = 0.20
    disk_vsigma_ratio_min: float = 1.10
    sigma_ratio_min: float = 1.10
    bar_tolerance: float = 0.05
    rho_h3v_min: float = 0.20
    min_spaxels_for_test: int = 20


@dataclass(frozen=True, slots=True)
class KinematicValidationInput:
    unit_id: str
    galaxy_id: str
    canonical_id: str
    view_id: int
    y_int: np.ndarray
    m_val: np.ndarray
    v_star: np.ndarray
    sigma_star: np.ndarray
    r_bar: float | None
    kinematic_moments: KinematicMomentMaps | None = None
    label_path: str = ""
    maps2d_path: str = ""
    sample_manga: int | None = None


@dataclass(frozen=True, slots=True)
class KinematicChecks:
    unit_id: str
    galaxy_id: str
    canonical_id: str
    view_id: int
    test_a_rotation: Status
    test_b_dispersion: Status
    test_c_bar_sigma: Status
    test_d_h3_signature: Status
    rotation_test_mode: str
    rotation_reference_mode: str
    rho_disk: float | None
    v_over_sigma_disk_median: float | None
    v_over_sigma_reference_median: float | None
    v_over_sigma_ratio: float | None
    sigma_ratio: float | None
    sigma_bulge_median: float | None
    sigma_disk_median: float | None
    sigma_bar_median: float | None
    rho_h3v: float | None
    n_tests_applicable: int
    n_tests_passed: int
    coherence_score: float
    passes: bool
    h3h4_used: bool
    sample_manga: int | None = None
    velocity_center_median: float | None = None
    v_over_sigma_global_median: float | None = None
    n_valid_spaxels: int = 0
    n_disk_spaxels: int = 0
    n_reference_spaxels: int = 0
    n_bulge_spaxels: int = 0
    n_other_spaxels: int = 0
    test_a_failure_mode: str = ""
    label_path: str = ""
    maps2d_path: str = ""
    status: str = "ok"
    error: str = ""


@dataclass(frozen=True, slots=True)
class KinematicSuccessReport:
    n_units_total: int
    success_rate_test_a: float
    success_rate_test_b: float
    success_rate_test_c: float
    success_rate_test_d: float
    n_applicable_test_a: int
    n_applicable_test_b: int
    n_applicable_test_c: int
    n_applicable_test_d: int
    success_rate_overall: float
    n_units_with_h3h4: int
    n_units_without_h3h4: int
    coherence_score_percentiles: dict[str, float]
    n_units_skipped: int = 0
    rotation_test_mode: str = "contrast"
    rotation_reference_mode: str = "central"
    central_reference_radius_fraction: float = 0.25
    disk_vsigma_ratio_min: float = 1.10
    center_velocity: bool = True
    min_sigma_star: float = 1.0


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    result = spearmanr(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), nan_policy="omit")
    value = float(result.statistic)
    return 0.0 if not np.isfinite(value) else value


def _dominant_mask(y_int: np.ndarray, class_index: int, valid: np.ndarray, threshold: float) -> np.ndarray:
    return valid & (np.asarray(y_int[class_index], dtype=np.float64) > float(threshold))


def _median_or_none(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.nanmedian(values))


def _ratio_or_none(numerator: float | None, denominator: float | None, epsilon: float) -> float | None:
    if numerator is None or denominator is None:
        return None
    return float(numerator / (denominator + epsilon))


def _status(pass_value: bool) -> Status:
    return "PASS" if pass_value else "FAIL"


def _test_a_failure_mode(status: Status, mode: str, disk_count: int, reference_count: int, min_spaxels: int) -> str:
    if status == "PASS":
        return ""
    if mode == "contrast":
        if disk_count < min_spaxels:
            return "too_few_disk_spaxels"
        if reference_count < min_spaxels:
            return "too_few_reference_spaxels"
        return "low_disk_vsigma_contrast" if status == "FAIL" else "not_applicable"
    if mode == "spearman":
        return "low_disk_vsigma_spearman" if status == "FAIL" else "not_applicable"
    return "unknown"


def _central_reference_mask(valid: np.ndarray, radius_fraction: float) -> np.ndarray:
    valid = np.asarray(valid).astype(bool)
    if not np.any(valid):
        return np.zeros_like(valid, dtype=bool)
    yy, xx = np.indices(valid.shape)
    y_valid = yy[valid].astype(np.float64)
    x_valid = xx[valid].astype(np.float64)
    y0 = float(np.nanmedian(y_valid))
    x0 = float(np.nanmedian(x_valid))
    radius = np.sqrt((xx.astype(np.float64) - x0) ** 2 + (yy.astype(np.float64) - y0) ** 2)
    max_radius = float(np.nanmax(radius[valid]))
    if max_radius <= 0.0:
        return valid.copy()
    fraction = min(max(float(radius_fraction), 0.0), 1.0)
    return valid & (radius <= fraction * max_radius)


def _rotation_reference_mask(
    mode: RotationReferenceMode,
    valid: np.ndarray,
    bulge_mask: np.ndarray,
    other_mask: np.ndarray,
    central_reference_radius_fraction: float,
) -> np.ndarray:
    if mode == "bulge_other":
        return bulge_mask | other_mask
    if mode == "bulge":
        return bulge_mask
    if mode == "central":
        return _central_reference_mask(valid, central_reference_radius_fraction)
    raise ValueError(f"Invalid rotation_reference_mode={mode!r}")


def _score(statuses: Iterable[Status]) -> tuple[int, int, float, bool]:
    applicable = [status for status in statuses if status != "N/A"]
    n_applicable = len(applicable)
    n_passed = sum(status == "PASS" for status in applicable)
    if n_applicable == 0:
        return 0, 0, float("nan"), False
    score = float(n_passed / n_applicable)
    return n_applicable, n_passed, score, n_passed == n_applicable


def validate_kinematic_unit(
    unit: KinematicValidationInput,
    config: KinematicValidationConfig | None = None,
) -> KinematicChecks:
    config = config or KinematicValidationConfig()
    y_int = np.asarray(unit.y_int, dtype=np.float32)
    m_val = np.asarray(unit.m_val).astype(bool)
    v_star = np.asarray(unit.v_star, dtype=np.float32)
    sigma_star = np.asarray(unit.sigma_star, dtype=np.float32)
    if y_int.shape[0] != len(LABEL_CLASS_NAMES):
        raise ValueError(f"Expected y_int shape=(5,H,W), got {y_int.shape}")
    if y_int.shape[1:] != m_val.shape or m_val.shape != v_star.shape or v_star.shape != sigma_star.shape:
        raise ValueError(
            "Shape mismatch: "
            f"y_int={y_int.shape}, m_val={m_val.shape}, v_star={v_star.shape}, sigma_star={sigma_star.shape}"
        )

    finite = np.isfinite(v_star) & np.isfinite(sigma_star) & (np.abs(sigma_star) >= float(config.min_sigma_star))
    valid = m_val & finite
    min_spaxels = int(config.min_spaxels_for_test)
    velocity_center = _median_or_none(v_star[valid]) if config.center_velocity else 0.0
    velocity_center = 0.0 if velocity_center is None else velocity_center
    v_for_rotation = v_star - float(velocity_center)
    v_over_sigma = np.abs(v_for_rotation) / (np.abs(sigma_star) + float(config.epsilon))
    bulge_mask = _dominant_mask(y_int, 0, valid, config.dominant_class_threshold)
    disk_mask = _dominant_mask(y_int, 1, valid, config.dominant_class_threshold)
    bar_mask = _dominant_mask(y_int, 2, valid, config.dominant_class_threshold)
    other_mask = _dominant_mask(y_int, 4, valid, config.dominant_class_threshold)
    hot_reference_mask = _rotation_reference_mask(
        config.rotation_reference_mode,
        valid,
        bulge_mask,
        other_mask,
        config.central_reference_radius_fraction,
    )
    n_valid_spaxels = int(np.count_nonzero(valid))
    n_disk_spaxels = int(np.count_nonzero(disk_mask))
    n_reference_spaxels = int(np.count_nonzero(hot_reference_mask))
    n_bulge_spaxels = int(np.count_nonzero(bulge_mask))
    n_other_spaxels = int(np.count_nonzero(other_mask))

    rho_disk = None
    if n_valid_spaxels >= min_spaxels:
        rho_disk = _spearman(y_int[1][valid], v_over_sigma[valid])

    v_over_sigma_global = _median_or_none(v_over_sigma[valid])
    v_over_sigma_disk = _median_or_none(v_over_sigma[disk_mask])
    v_over_sigma_reference = _median_or_none(v_over_sigma[hot_reference_mask])
    v_over_sigma_ratio = _ratio_or_none(v_over_sigma_disk, v_over_sigma_reference, config.epsilon)

    if config.rotation_test_mode == "spearman":
        if rho_disk is None:
            test_a = "N/A"
        else:
            test_a = _status(rho_disk >= config.rho_disk_min)
    elif config.rotation_test_mode == "contrast":
        if np.count_nonzero(disk_mask) >= min_spaxels and np.count_nonzero(hot_reference_mask) >= min_spaxels:
            test_a = _status(float(v_over_sigma_ratio) >= config.disk_vsigma_ratio_min)
        else:
            test_a = "N/A"
    else:
        raise ValueError(f"Invalid rotation_test_mode={config.rotation_test_mode!r}")
    test_a_failure_mode = _test_a_failure_mode(
        test_a,
        config.rotation_test_mode,
        n_disk_spaxels,
        n_reference_spaxels,
        min_spaxels,
    )

    sigma_bulge = _median_or_none(sigma_star[bulge_mask])
    sigma_disk = _median_or_none(sigma_star[disk_mask])
    sigma_bar = _median_or_none(sigma_star[bar_mask])

    sigma_ratio = None
    if np.count_nonzero(bulge_mask) >= min_spaxels and np.count_nonzero(disk_mask) >= min_spaxels:
        sigma_ratio = float(sigma_bulge / (sigma_disk + config.epsilon))
        test_b = _status(sigma_ratio >= config.sigma_ratio_min)
    else:
        test_b = "N/A"

    if unit.r_bar is None:
        test_c = "N/A"
    elif (
        np.count_nonzero(bar_mask) >= min_spaxels
        and np.count_nonzero(bulge_mask) >= min_spaxels
        and np.count_nonzero(disk_mask) >= min_spaxels
    ):
        lower = float(sigma_disk) * (1.0 - config.bar_tolerance)
        upper = float(sigma_bulge) * (1.0 + config.bar_tolerance)
        test_c = _status(lower <= float(sigma_bar) <= upper)
    else:
        test_c = "N/A"

    rho_h3v = None
    h3h4_used = False
    if unit.r_bar is None or unit.kinematic_moments is None:
        test_d = "N/A"
    else:
        moments = unit.kinematic_moments
        quality = np.asarray(moments.quality_mask).astype(bool)
        d_mask = bar_mask & quality & np.isfinite(moments.h3) & np.isfinite(v_star)
        if np.count_nonzero(d_mask) >= min_spaxels:
            rho_h3v = _spearman(np.asarray(moments.h3)[d_mask], v_star[d_mask])
            test_d = _status(rho_h3v <= -config.rho_h3v_min)
            h3h4_used = True
        else:
            test_d = "N/A"

    n_applicable, n_passed, coherence_score, passes = _score((test_a, test_b, test_c, test_d))
    return KinematicChecks(
        unit_id=unit.unit_id,
        galaxy_id=unit.galaxy_id,
        canonical_id=unit.canonical_id,
        view_id=int(unit.view_id),
        test_a_rotation=test_a,
        test_b_dispersion=test_b,
        test_c_bar_sigma=test_c,
        test_d_h3_signature=test_d,
        rotation_test_mode=config.rotation_test_mode,
        rotation_reference_mode=config.rotation_reference_mode,
        rho_disk=rho_disk,
        v_over_sigma_disk_median=v_over_sigma_disk,
        v_over_sigma_reference_median=v_over_sigma_reference,
        v_over_sigma_ratio=v_over_sigma_ratio,
        sigma_ratio=sigma_ratio,
        sigma_bulge_median=sigma_bulge,
        sigma_disk_median=sigma_disk,
        sigma_bar_median=sigma_bar,
        rho_h3v=rho_h3v,
        n_tests_applicable=n_applicable,
        n_tests_passed=n_passed,
        coherence_score=coherence_score,
        passes=passes,
        h3h4_used=h3h4_used,
        sample_manga=unit.sample_manga,
        velocity_center_median=float(velocity_center),
        v_over_sigma_global_median=v_over_sigma_global,
        n_valid_spaxels=n_valid_spaxels,
        n_disk_spaxels=n_disk_spaxels,
        n_reference_spaxels=n_reference_spaxels,
        n_bulge_spaxels=n_bulge_spaxels,
        n_other_spaxels=n_other_spaxels,
        test_a_failure_mode=test_a_failure_mode,
        label_path=unit.label_path,
        maps2d_path=unit.maps2d_path,
    )


def _rate(results: list[KinematicChecks], attr: str) -> tuple[float, int]:
    values = [getattr(result, attr) for result in results if getattr(result, attr) in ("PASS", "FAIL")]
    if not values:
        return float("nan"), 0
    return float(100.0 * sum(value == "PASS" for value in values) / len(values)), len(values)


def build_success_report(
    results: list[KinematicChecks],
    n_units_skipped: int = 0,
    config: KinematicValidationConfig | None = None,
) -> KinematicSuccessReport:
    ok_results = [result for result in results if result.status == "ok"]
    rate_a, n_a = _rate(ok_results, "test_a_rotation")
    rate_b, n_b = _rate(ok_results, "test_b_dispersion")
    rate_c, n_c = _rate(ok_results, "test_c_bar_sigma")
    rate_d, n_d = _rate(ok_results, "test_d_h3_signature")
    passable = [result for result in ok_results if result.n_tests_applicable > 0]
    overall = float("nan") if not passable else 100.0 * sum(result.passes for result in passable) / len(passable)
    scores = np.asarray([result.coherence_score for result in passable if np.isfinite(result.coherence_score)], dtype=np.float64)
    if scores.size:
        percentiles = {
            f"p{p}": float(np.nanpercentile(scores, p))
            for p in (10, 25, 50, 75, 90)
        }
    else:
        percentiles = {f"p{p}": float("nan") for p in (10, 25, 50, 75, 90)}
    return KinematicSuccessReport(
        n_units_total=len(ok_results),
        success_rate_test_a=rate_a,
        success_rate_test_b=rate_b,
        success_rate_test_c=rate_c,
        success_rate_test_d=rate_d,
        n_applicable_test_a=n_a,
        n_applicable_test_b=n_b,
        n_applicable_test_c=n_c,
        n_applicable_test_d=n_d,
        success_rate_overall=float(overall),
        n_units_with_h3h4=sum(result.h3h4_used for result in ok_results),
        n_units_without_h3h4=sum(not result.h3h4_used for result in ok_results),
        coherence_score_percentiles=percentiles,
        n_units_skipped=int(n_units_skipped),
        rotation_test_mode=(config.rotation_test_mode if config else (ok_results[0].rotation_test_mode if ok_results else "contrast")),
        rotation_reference_mode=(
            config.rotation_reference_mode if config else (ok_results[0].rotation_reference_mode if ok_results else "central")
        ),
        central_reference_radius_fraction=(config.central_reference_radius_fraction if config else 0.25),
        disk_vsigma_ratio_min=(config.disk_vsigma_ratio_min if config else 1.10),
        center_velocity=(config.center_velocity if config else True),
        min_sigma_star=(config.min_sigma_star if config else 1.0),
    )


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def write_unit_results_csv(path: str | Path, results: list[KinematicChecks]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(results[0]).keys()) if results else [field.name for field in KinematicChecks.__dataclass_fields__.values()]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = _json_safe(asdict(result))
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def write_report_json(path: str | Path, report: KinematicSuccessReport) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(asdict(report)), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _fmt_rate(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.1f} %"


def write_report_markdown(path: str | Path, report: KinematicSuccessReport) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = report.coherence_score_percentiles
    test_a_description = (
        f"Contraste mediano disco vs {report.rotation_reference_mode} en V/sigma"
        if report.rotation_test_mode == "contrast"
        else "Correlacion disco-rotacion"
    )
    lines = [
        "# Reporte de validacion cinematica",
        "",
        (
            f"Unidades evaluadas: {report.n_units_total} "
            f"({report.n_units_with_h3h4} con h3/h4, {report.n_units_without_h3h4} sin h3/h4)"
        ),
        f"Unidades omitidas: {report.n_units_skipped}",
        f"Modo Test A: {report.rotation_test_mode}",
        f"Referencia Test A: {report.rotation_reference_mode}",
        f"Radio central Test A: {report.central_reference_radius_fraction}",
        f"Velocidad centrada: {report.center_velocity}",
        f"Sigma minima para V/sigma: {report.min_sigma_star}",
        "",
        "## Porcentajes de exito por test",
        "",
        "| Test | Descripcion | Aplicable a | Exito |",
        "|---|---|---:|---:|",
        f"| A | {test_a_description} | {report.n_applicable_test_a} unidades | {_fmt_rate(report.success_rate_test_a)} |",
        f"| B | Dispersion bulbo > disco | {report.n_applicable_test_b} unidades | {_fmt_rate(report.success_rate_test_b)} |",
        f"| C | Dispersion intermedia barra | {report.n_applicable_test_c} barradas | {_fmt_rate(report.success_rate_test_c)} |",
        f"| D | Firma h3 en barra | {report.n_applicable_test_d} barradas con h3/h4 | {_fmt_rate(report.success_rate_test_d)} |",
        "",
        "## Exito global",
        "",
        f"{_fmt_rate(report.success_rate_overall)} de las unidades superan todos sus tests aplicables.",
        "",
        "## Distribucion del coherence_score",
        "",
        f"p10={p['p10']:.3f}  p25={p['p25']:.3f}  p50={p['p50']:.3f}  p75={p['p75']:.3f}  p90={p['p90']:.3f}",
        "",
        "## Nota sobre h3/h4",
        "",
        (
            f"El Test D se ejecuto sobre {report.n_applicable_test_d} unidades. "
            "En las unidades sin mapas h3/h4 el Test D se omitio y no penaliza el exito global."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_score_histogram(path: str | Path, results: list[KinematicChecks]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scores = [result.coherence_score for result in results if result.status == "ok" and np.isfinite(result.coherence_score)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=np.linspace(0.0, 1.0, 11), color="#3b82f6", edgecolor="white")
    ax.set_xlabel("coherence_score")
    ax.set_ylabel("unidades")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Distribucion de validacion cinematica")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


TEST_A_DIAGNOSTIC_FIELDNAMES = [
    "unit_id",
    "galaxy_id",
    "canonical_id",
    "view_id",
    "sample_manga",
    "test_a_rotation",
    "test_a_failure_mode",
    "rotation_test_mode",
    "rotation_reference_mode",
    "velocity_center_median",
    "v_over_sigma_ratio",
    "v_over_sigma_disk_median",
    "v_over_sigma_reference_median",
    "v_over_sigma_global_median",
    "rho_disk",
    "n_valid_spaxels",
    "n_disk_spaxels",
    "n_reference_spaxels",
    "n_bulge_spaxels",
    "n_other_spaxels",
    "label_path",
    "maps2d_path",
]


def _test_a_diagnostic_row(result: KinematicChecks) -> dict[str, object]:
    return {field: getattr(result, field) for field in TEST_A_DIAGNOSTIC_FIELDNAMES}


def write_test_a_diagnostics_csv(path: str | Path, results: list[KinematicChecks]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TEST_A_DIAGNOSTIC_FIELDNAMES)
        writer.writeheader()
        for result in results:
            if result.status != "ok":
                continue
            writer.writerow(_json_safe(_test_a_diagnostic_row(result)))
    return path


def _finite_values(results: list[KinematicChecks], attr: str) -> list[float]:
    values = []
    for result in results:
        value = getattr(result, attr)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return values


def _median_value(results: list[KinematicChecks], attr: str) -> float | None:
    values = _finite_values(results, attr)
    return float(np.nanmedian(values)) if values else None


def _success_rate(results: list[KinematicChecks]) -> tuple[float | None, int, int]:
    applicable = [result for result in results if result.test_a_rotation in ("PASS", "FAIL")]
    if not applicable:
        return None, 0, 0
    passed = sum(result.test_a_rotation == "PASS" for result in applicable)
    return float(100.0 * passed / len(applicable)), len(applicable), passed


def _summary_row(group_name: str, results: list[KinematicChecks]) -> dict[str, object]:
    rate, applicable, passed = _success_rate(results)
    return {
        "group": group_name,
        "n_units": len(results),
        "n_applicable": applicable,
        "n_pass": passed,
        "success_rate": rate,
        "n_fail": sum(result.test_a_rotation == "FAIL" for result in results),
        "n_na": sum(result.test_a_rotation == "N/A" for result in results),
        "median_vsigma_ratio": _median_value(results, "v_over_sigma_ratio"),
        "median_global_vsigma": _median_value(results, "v_over_sigma_global_median"),
        "median_disk_spaxels": _median_value(results, "n_disk_spaxels"),
        "median_reference_spaxels": _median_value(results, "n_reference_spaxels"),
        "too_few_disk_spaxels": sum(result.test_a_failure_mode == "too_few_disk_spaxels" for result in results),
        "too_few_reference_spaxels": sum(result.test_a_failure_mode == "too_few_reference_spaxels" for result in results),
        "low_disk_vsigma_contrast": sum(result.test_a_failure_mode == "low_disk_vsigma_contrast" for result in results),
    }


def _write_summary_csv(path: str | Path, rows: list[dict[str, object]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "n_units",
        "n_applicable",
        "n_pass",
        "success_rate",
        "n_fail",
        "n_na",
        "median_vsigma_ratio",
        "median_global_vsigma",
        "median_disk_spaxels",
        "median_reference_spaxels",
        "too_few_disk_spaxels",
        "too_few_reference_spaxels",
        "low_disk_vsigma_contrast",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _json_safe(row.get(field, "")) for field in fieldnames})
    return path


def write_test_a_summary_by_view_csv(path: str | Path, results: list[KinematicChecks]) -> Path:
    ok_results = [result for result in results if result.status == "ok"]
    views = sorted({int(result.view_id) for result in ok_results})
    rows = [_summary_row(str(view), [result for result in ok_results if int(result.view_id) == view]) for view in views]
    return _write_summary_csv(path, rows)


def write_test_a_summary_by_sample_csv(path: str | Path, results: list[KinematicChecks]) -> Path:
    ok_results = [result for result in results if result.status == "ok"]
    samples = sorted({result.sample_manga for result in ok_results if result.sample_manga is not None})
    rows = [_summary_row(str(sample), [result for result in ok_results if result.sample_manga == sample]) for sample in samples]
    return _write_summary_csv(path, rows)


def _vsigma_bin(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "missing"
    bins = [
        (0.25, "0.00-0.25"),
        (0.50, "0.25-0.50"),
        (0.75, "0.50-0.75"),
        (1.00, "0.75-1.00"),
        (1.50, "1.00-1.50"),
        (2.00, "1.50-2.00"),
    ]
    for limit, label in bins:
        if value < limit:
            return label
    return "2.00+"


def write_test_a_summary_by_global_vsigma_csv(path: str | Path, results: list[KinematicChecks]) -> Path:
    ok_results = [result for result in results if result.status == "ok"]
    labels = ["missing", "0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00", "1.00-1.50", "1.50-2.00", "2.00+"]
    rows = []
    for label in labels:
        subset = [result for result in ok_results if _vsigma_bin(result.v_over_sigma_global_median) == label]
        if subset:
            rows.append(_summary_row(label, subset))
    return _write_summary_csv(path, rows)


def _format_result_line(result: KinematicChecks) -> str:
    return (
        f"| {result.canonical_id} | {result.view_id} | {result.sample_manga if result.sample_manga is not None else ''} "
        f"| {result.test_a_rotation} | {result.rotation_reference_mode} "
        f"| {result.v_over_sigma_ratio if result.v_over_sigma_ratio is not None else ''} "
        f"| {result.v_over_sigma_global_median if result.v_over_sigma_global_median is not None else ''} "
        f"| {result.n_disk_spaxels} | {result.n_reference_spaxels} | {result.test_a_failure_mode} |"
    )


def write_test_a_extreme_pass_fail_markdown(
    path: str | Path,
    results: list[KinematicChecks],
    limit: int = 15,
    ratio_threshold: float = 1.10,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok_results = [result for result in results if result.status == "ok"]
    finite_ratio = [result for result in ok_results if result.v_over_sigma_ratio is not None and np.isfinite(result.v_over_sigma_ratio)]
    severe_fail = sorted(
        [result for result in finite_ratio if result.test_a_rotation == "FAIL"],
        key=lambda result: float(result.v_over_sigma_ratio),
    )[:limit]
    strong_pass = sorted(
        [result for result in finite_ratio if result.test_a_rotation == "PASS"],
        key=lambda result: float(result.v_over_sigma_ratio),
        reverse=True,
    )[:limit]
    near_threshold = sorted(
        [result for result in finite_ratio if result.test_a_rotation == "FAIL"],
        key=lambda result: abs(float(result.v_over_sigma_ratio) - float(ratio_threshold)),
    )[:limit]
    na_reasons: dict[str, int] = {}
    for result in ok_results:
        if result.test_a_rotation == "N/A":
            key = result.test_a_failure_mode or "not_applicable"
            na_reasons[key] = na_reasons.get(key, 0) + 1

    lines = [
        "# Test A diagnostics",
        "",
        "Columns: canonical_id, view, sample_manga, Test A, reference, V/sigma ratio, global V/sigma, disk spaxels, reference spaxels, reason.",
        "",
        "## Severe FAIL",
        "",
        "| canonical_id | view | sample | test_a | reference | ratio | global_vsigma | disk_px | ref_px | reason |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---|",
    ]
    lines.extend(_format_result_line(result) for result in severe_fail)
    lines.extend(
        [
            "",
            "## Strong PASS",
            "",
            "| canonical_id | view | sample | test_a | reference | ratio | global_vsigma | disk_px | ref_px | reason |",
            "|---|---:|---:|---|---|---:|---:|---:|---:|---|",
        ]
    )
    lines.extend(_format_result_line(result) for result in strong_pass)
    lines.extend(
        [
            "",
            "## FAIL near threshold",
            "",
            "| canonical_id | view | sample | test_a | reference | ratio | global_vsigma | disk_px | ref_px | reason |",
            "|---|---:|---:|---|---|---:|---:|---:|---:|---|",
        ]
    )
    lines.extend(_format_result_line(result) for result in near_threshold)
    lines.extend(["", "## N/A reasons", ""])
    if na_reasons:
        for reason, count in sorted(na_reasons.items()):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _decode_class_names(raw: np.ndarray) -> list[str]:
    values = []
    for value in raw.tolist():
        if isinstance(value, bytes):
            values.append(value.decode("utf-8"))
        else:
            values.append(str(value))
    return values


def load_label_tensor(path: str | Path, mode: str = "soft_mass") -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        source = np.asarray(data[mode], dtype=np.float32)
        valid_mask = np.asarray(data["valid_mask"]).astype(bool)
        class_names = _decode_class_names(data["class_names"])
    indices = []
    lower = [name.lower() for name in class_names]
    for target in LABEL_CLASS_NAMES:
        aliases = LABEL_CLASS_ALIASES[target]
        matches = [lower.index(alias) for alias in aliases if alias in lower]
        if not matches:
            raise KeyError(f"Class {target} not found in {path}: {class_names}")
        indices.append(matches[0])
    return source[indices], valid_mask


def _read_key_from_fits(path: Path, key: str) -> np.ndarray:
    from astropy.io import fits

    match = re.fullmatch(r"(.+)\[(\d+)\]", key)
    with fits.open(path, memmap=False) as hdul:
        if match:
            extname, index = match.group(1), int(match.group(2))
            for hdu in hdul:
                name = str(hdu.header.get("EXTNAME", hdu.name)).strip()
                if name == extname:
                    return np.asarray(hdu.data[index], dtype=np.float32)
            raise KeyError(f"{key} not found in {path}")
        for hdu in hdul:
            name = str(hdu.header.get("EXTNAME", hdu.name)).strip()
            if name == key:
                return np.asarray(hdu.data, dtype=np.float32)
    raise KeyError(f"{key} not found in {path}")


def _read_key_from_hdf5(path: Path, key: str) -> np.ndarray:
    import h5py

    match = re.fullmatch(r"(.+)\[(\d+)\]", key)
    with h5py.File(path, "r") as handle:
        if match:
            dataset, index = match.group(1), int(match.group(2))
            return np.asarray(handle[dataset][index], dtype=np.float32)
        return np.asarray(handle[key], dtype=np.float32)


def load_map_pair(path: str | Path, v_key: str, sigma_key: str, fmt: str = "") -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".")).lower()
    if path.name.endswith(".npz") or fmt == "npz":
        with np.load(path, allow_pickle=False) as data:
            return np.asarray(data[v_key], dtype=np.float32), np.asarray(data[sigma_key], dtype=np.float32)
    if path.name.endswith(".fits") or path.name.endswith(".fits.gz") or fmt == "fits":
        return _read_key_from_fits(path, v_key), _read_key_from_fits(path, sigma_key)
    if path.name.endswith(".h5") or path.name.endswith(".hdf5") or fmt in ("h5", "hdf5"):
        return _read_key_from_hdf5(path, v_key), _read_key_from_hdf5(path, sigma_key)
    raise ValueError(f"Unsupported maps2d format for {path}")


def r_bar_from_summary(path: str | Path) -> float | None:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    bar = raw.get("bar_metadata", {})
    barred = bool(bar.get("barred_target", False))
    if not barred:
        return None
    value = bar.get("bar_radius_input_kpc", bar.get("bar_radius_recovered_kpc", 0.0))
    value = float(value or 0.0)
    return value if value > 0 else None


def load_kinematic_moments(path: str | Path | None) -> KinematicMomentMaps | None:
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return KinematicMomentMaps(
            h3=np.asarray(data["h3"], dtype=np.float32),
            h4=np.asarray(data["h4"], dtype=np.float32),
            quality_mask=np.asarray(data["quality_mask"]).astype(bool),
        )
