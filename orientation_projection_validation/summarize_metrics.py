from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from statistics import mean, median


def _metric_paths(pattern: str) -> list[Path]:
    return [Path(path) for path in sorted(glob.glob(pattern))]


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _stats(values: list[float]) -> dict[str, float | None]:
    finite_values = [_finite_float(value) for value in values]
    values = [float(value) for value in finite_values if value is not None]
    if not values:
        return {
            "mean": None,
            "median": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "p10": _percentile(values, 10),
        "p25": _percentile(values, 25),
        "p75": _percentile(values, 75),
        "p90": _percentile(values, 90),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _fmt(value: object, digits: int = 3) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "N/A"
        return f"{value:.{digits}f}"
    return str(value)


def _write_markdown_report(
    path: Path,
    rows: list[dict[str, object]],
    class_names: list[str],
    metrics_glob: str,
    csv_path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_total = len(rows)
    accepted = sum(_as_bool(row.get("accepted")) for row in rows)
    rejected = n_total - accepted
    accepted_rate = 100.0 * accepted / n_total if n_total else float("nan")
    runtime_errors = sum("runtime_error" in str(row.get("failure_reasons", "")) for row in rows)

    failure_counts: dict[str, int] = {}
    for row in rows:
        reasons = str(row.get("failure_reasons", "") or "").split(";")
        for reason in [item.strip() for item in reasons if item.strip()]:
            failure_counts[reason] = failure_counts.get(reason, 0) + 1

    cglobal_values = [_finite_float(row.get("Cglobal")) for row in rows]
    cglobal_values = [value for value in cglobal_values if value is not None]
    fvalid_values = [_finite_float(row.get("fvalid")) for row in rows]
    fvalid_values = [value for value in fvalid_values if value is not None]

    lines = [
        "# Reporte de validación inter-orientación",
        "",
        "## Resumen",
        "",
        f"- Archivos `metrics.json` leídos: {n_total}",
        f"- Galaxias aceptadas: {accepted}",
        f"- Galaxias rechazadas: {rejected}",
        f"- Tasa de aceptación: {_fmt(accepted_rate, 1)} %",
        f"- Errores de ejecución registrados: {runtime_errors}",
        f"- CSV asociado: `{csv_path}`",
        f"- Patrón de entrada: `{metrics_glob}`",
        "",
        "## Métricas globales",
        "",
        "| Métrica | media | mediana | p10 | p25 | p75 | p90 | min | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in (("Cglobal", cglobal_values), ("fvalid", fvalid_values)):
        stats = _stats(values)
        lines.append(
            f"| {name} | {_fmt(stats['mean'])} | {_fmt(stats['median'])} | {_fmt(stats['p10'])} | "
            f"{_fmt(stats['p25'])} | {_fmt(stats['p75'])} | {_fmt(stats['p90'])} | {_fmt(stats['min'])} | {_fmt(stats['max'])} |"
        )

    lines.extend(
        [
            "",
            "## Consistencia por clase",
            "",
            "| Clase | media | mediana | p10 | p25 | p75 | p90 | min | max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for class_name in class_names:
        key = f"C_{class_name}"
        values = [_finite_float(row.get(key)) for row in rows]
        stats = _stats([value for value in values if value is not None])
        lines.append(
            f"| {class_name} | {_fmt(stats['mean'])} | {_fmt(stats['median'])} | {_fmt(stats['p10'])} | "
            f"{_fmt(stats['p25'])} | {_fmt(stats['p75'])} | {_fmt(stats['p90'])} | {_fmt(stats['min'])} | {_fmt(stats['max'])} |"
        )

    lines.extend(["", "## Razones de rechazo", "", "| Razón | Galaxias |", "|---|---:|"])
    if failure_counts:
        for reason, count in sorted(failure_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"| `{reason}` | {count} |")
    else:
        lines.append("| ninguna | 0 |")

    rejected_rows = [row for row in rows if not _as_bool(row.get("accepted"))]
    rejected_rows.sort(
        key=lambda row: (
            _finite_float(row.get("Cglobal")) is None,
            _finite_float(row.get("Cglobal")) or 0.0,
            str(row.get("galaxy_id", "")),
        )
    )
    lines.extend(
        [
            "",
            "## Galaxias rechazadas con menor Cglobal",
            "",
            "| galaxy_id | Cglobal | fvalid | razones |",
            "|---|---:|---:|---|",
        ]
    )
    for row in rejected_rows[:30]:
        lines.append(
            f"| `{row.get('galaxy_id', '')}` | {_fmt(_finite_float(row.get('Cglobal')))} | "
            f"{_fmt(_finite_float(row.get('fvalid')))} | {row.get('failure_reasons', '') or ''} |"
        )
    if not rejected_rows:
        lines.append("| ninguna | N/A | N/A | |")

    lines.extend(
        [
            "",
            "## Notas metodológicas",
            "",
            "- `Cglobal` es el promedio de IoU probabilística entre orientaciones, tras compensar la rotación de cada mapa.",
            "- `fvalid` es la fracción media de spaxels válidos por orientación.",
            "- Una galaxia se marca como aceptada si no activa ninguna razón de rechazo en `metrics.json`.",
            "- Las columnas `C_<clase>` reportan la consistencia media por clase física.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Resume metrics.json de validación inter-orientación.")
    parser.add_argument("--metrics-glob", default="outputs/*/metrics.json", help="Glob de metrics.json")
    parser.add_argument("--out", default="outputs/catalog_interorientation_summary.csv", help="CSV de salida")
    parser.add_argument("--report", default="", help="Markdown de salida; default: mismo nombre que --out con .md")
    parser.add_argument("--no-report", action="store_true", help="No escribe reporte Markdown")
    args = parser.parse_args(argv)

    paths = _metric_paths(args.metrics_glob)
    if not paths:
        raise SystemExit(f"No encontré métricas con glob {args.metrics_glob}")

    rows = []
    class_names = set()
    for path in paths:
        payload = json.loads(path.read_text())
        class_names.update((payload.get("classes") or {}).keys())
        rows.append((path, payload))
    class_columns = [f"C_{name}" for name in sorted(class_names)]
    fieldnames = [
        "galaxy_id",
        "snapshot",
        "subhalo_id",
        "accepted",
        "fvalid",
        "Cglobal",
        "variant",
        "failure_reasons",
        "metrics_path",
        *class_columns,
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        output_rows = []
        for path, payload in rows:
            classes = payload.get("classes") or {}
            row = {
                "galaxy_id": payload.get("galaxy_id", ""),
                "snapshot": payload.get("snapshot", ""),
                "subhalo_id": payload.get("subhalo_id", ""),
                "accepted": payload.get("accepted", False),
                "fvalid": payload.get("fvalid", ""),
                "Cglobal": payload.get("Cglobal", ""),
                "variant": payload.get("variant", ""),
                "failure_reasons": ";".join(payload.get("failure_reasons", [])),
                "metrics_path": str(path),
            }
            for name in sorted(class_names):
                row[f"C_{name}"] = classes.get(name, "")
            writer.writerow(row)
            output_rows.append(row)
    print(f"Resumen escrito en {out_path} con {len(rows)} filas")
    if not args.no_report:
        report_path = Path(args.report) if args.report else out_path.with_suffix(".md")
        _write_markdown_report(report_path, output_rows, sorted(class_names), args.metrics_glob, out_path)
        print(f"Reporte escrito en {report_path}")


if __name__ == "__main__":
    main()
