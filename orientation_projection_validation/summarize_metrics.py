from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume metrics.json de validación inter-orientación.")
    parser.add_argument("--metrics-glob", default="outputs/*/metrics.json", help="Glob de metrics.json")
    parser.add_argument("--out", default="outputs/catalog_interorientation_summary.csv", help="CSV de salida")
    args = parser.parse_args()

    paths = sorted(Path().glob(args.metrics_glob))
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
    print(f"Resumen escrito en {out_path} con {len(rows)} filas")


if __name__ == "__main__":
    main()

