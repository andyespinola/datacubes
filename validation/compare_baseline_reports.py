from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fmt(value: Any, digits: int = 1) -> str:
    number = _finite_float(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


def _load_kinematic(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _load_orientation(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    rows: list[dict[str, str]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"n_orientation": 0}

    accepted = sum(_as_bool(row.get("accepted")) for row in rows)

    def values(key: str) -> list[float]:
        return [value for value in (_finite_float(row.get(key)) for row in rows) if value is not None]

    restricted = []
    for row in rows:
        subset = [value for value in (_finite_float(row.get(key)) for key in ("C_bulbo", "C_disco", "C_other")) if value is not None]
        if subset:
            restricted.append(float(sum(subset) / len(subset)))

    return {
        "n_orientation": len(rows),
        "orientation_acceptance_rate": 100.0 * accepted / len(rows),
        "Cglobal_median": median(values("Cglobal")) if values("Cglobal") else None,
        "Cglobal_bulge_disk_other_median": median(restricted) if restricted else None,
        "C_bulbo_median": median(values("C_bulbo")) if values("C_bulbo") else None,
        "C_disco_median": median(values("C_disco")) if values("C_disco") else None,
        "C_other_median": median(values("C_other")) if values("C_other") else None,
        "fvalid_median": median(values("fvalid")) if values("fvalid") else None,
    }


def _method_row(name: str, kinematic: dict[str, Any], orientation: dict[str, Any]) -> dict[str, Any]:
    row = {
        "method": name,
        "n_units": kinematic.get("n_units_total"),
        "kinematic_overall": kinematic.get("success_rate_overall"),
        "kinematic_test_a": kinematic.get("success_rate_test_a"),
        "test_a_applicable": kinematic.get("n_applicable_test_a"),
        "kinematic_test_b": kinematic.get("success_rate_test_b"),
        "test_b_applicable": kinematic.get("n_applicable_test_b"),
        "coherence_p50": (kinematic.get("coherence_score_percentiles") or {}).get("p50"),
        "coherence_p10": (kinematic.get("coherence_score_percentiles") or {}).get("p10"),
    }
    row.update(orientation)
    return row


def _delta_row(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    row = {"method": f"delta {left['method']} - {right['method']}"}
    keys = set(left) | set(right)
    for key in keys:
        if key == "method":
            continue
        left_value = _finite_float(left.get(key))
        right_value = _finite_float(right.get(key))
        row[key] = left_value - right_value if left_value is not None and right_value is not None else ""
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def _write_markdown(path: Path, rows: list[dict[str, Any]], csv_path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Comparación GMM-4D contra baseline de umbral duro en epsilon",
        "",
        "## Tabla principal",
        "",
        "| Método | N | Cinemática global | Test A | Test A N | Test B | Test B N | p50 coherencia | IoU aceptadas | Cglobal mediana | Cglobal B/D/O mediana | C_disco mediana | fvalid mediana |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if str(row["method"]).startswith("delta"):
            continue
        lines.append(
            f"| {row['method']} | {_fmt(row.get('n_units'), 0)} | {_fmt(row.get('kinematic_overall'))} % | "
            f"{_fmt(row.get('kinematic_test_a'))} % | {_fmt(row.get('test_a_applicable'), 0)} | "
            f"{_fmt(row.get('kinematic_test_b'))} % | {_fmt(row.get('test_b_applicable'), 0)} | "
            f"{_fmt(row.get('coherence_p50'), 2)} | {_fmt(row.get('orientation_acceptance_rate'))} % | "
            f"{_fmt(row.get('Cglobal_median'), 3)} | {_fmt(row.get('Cglobal_bulge_disk_other_median'), 3)} | "
            f"{_fmt(row.get('C_disco_median'), 3)} | {_fmt(row.get('fvalid_median'), 3)} |"
        )

    delta = next((row for row in rows if str(row["method"]).startswith("delta")), None)
    if delta:
        lines.extend(
            [
                "",
                "## Diferencia directa",
                "",
                f"- Delta cinemática global: {_fmt(delta.get('kinematic_overall'))} puntos porcentuales.",
                f"- Delta Test A: {_fmt(delta.get('kinematic_test_a'))} puntos porcentuales.",
                f"- Delta Test B: {_fmt(delta.get('kinematic_test_b'))} puntos porcentuales.",
                f"- Delta aceptación IoU: {_fmt(delta.get('orientation_acceptance_rate'))} puntos porcentuales.",
                f"- Delta Cglobal B/D/O mediana: {_fmt(delta.get('Cglobal_bulge_disk_other_median'), 3)}.",
            ]
        )

    lines.extend(
        [
            "",
            "## Notas metodológicas",
            "",
            "- El baseline `epsilon` etiqueta `disco` con un umbral duro de circularidad y asigna el resto a la componente esferoidal.",
            "- Las métricas cinemáticas se calculan con el mismo `matched_units.csv`, los mismos mapas Pipe3D y los mismos umbrales de validación.",
            "- `Cglobal B/D/O` promedia solo `bulbo`, `disco` y `other`; evita que clases no modeladas por el baseline simple, como barra o brazos, inflen el promedio de IoU.",
            f"- CSV asociado: `{csv_path}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare GMM-4D labels against the hard-epsilon baseline.")
    parser.add_argument("--gmm-kinematic", required=True, help="kinematic_validation_report.json del método GMM-4D")
    parser.add_argument("--epsilon-kinematic", required=True, help="kinematic_validation_report.json del baseline epsilon")
    parser.add_argument("--gmm-orientation", default="", help="catalog_interorientation_summary*.csv del método GMM-4D")
    parser.add_argument("--epsilon-orientation", default="", help="catalog_interorientation_summary*.csv del baseline epsilon")
    parser.add_argument("--out", required=True, help="Markdown de salida")
    parser.add_argument("--csv-out", default="", help="CSV de salida; default: mismo stem que --out")
    parser.add_argument("--gmm-name", default="GMM-4D")
    parser.add_argument("--epsilon-name", default="epsilon-threshold")
    args = parser.parse_args(argv)

    gmm = _method_row(args.gmm_name, _load_kinematic(args.gmm_kinematic), _load_orientation(args.gmm_orientation))
    epsilon = _method_row(args.epsilon_name, _load_kinematic(args.epsilon_kinematic), _load_orientation(args.epsilon_orientation))
    delta = _delta_row(gmm, epsilon)
    rows = [gmm, epsilon, delta]
    out_path = Path(args.out).expanduser()
    csv_path = Path(args.csv_out).expanduser() if args.csv_out else out_path.with_suffix(".csv")
    _write_csv(csv_path, rows)
    _write_markdown(out_path, rows, csv_path)
    print(json.dumps({"markdown": str(out_path), "csv": str(csv_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
