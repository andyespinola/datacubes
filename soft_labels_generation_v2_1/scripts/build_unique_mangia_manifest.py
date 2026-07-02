from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from astropy.io import fits
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent


def build_unique_manifest(catalog_path: Path) -> tuple[list[dict], dict]:
    with fits.open(catalog_path, memmap=True) as hdul:
        data = hdul[1].data
        grouped: dict[tuple[int, int], dict] = {}
        for row in data:
            snapshot = int(row["snapshot"])
            subhalo_id = int(row["subhalo_id"])
            key = (snapshot, subhalo_id)
            item = grouped.setdefault(
                key,
                {
                    "galaxy_id": f"TNG50-{snapshot}-{subhalo_id}",
                    "snapshot": snapshot,
                    "subhalo_id": subhalo_id,
                    "n_rows": 0,
                    "views": set(),
                    "ifu_designs": set(),
                    "n_star_part": int(row["n_star_part"]) if "n_star_part" in data.names else 0,
                    "n_gas_cell": int(row["n_gas_cell"]) if "n_gas_cell" in data.names else 0,
                    "re_kpc": float(row["re_kpc"]) if "re_kpc" in data.names else 0.0,
                },
            )
            item["n_rows"] += 1
            if "view" in data.names:
                item["views"].add(int(row["view"]))
            if "manga_ifu_dsn" in data.names:
                item["ifu_designs"].add(int(row["manga_ifu_dsn"]))
        rows = []
        for item in grouped.values():
            rows.append(
                {
                    **{key: value for key, value in item.items() if key not in {"views", "ifu_designs"}},
                    "views": ",".join(str(value) for value in sorted(item["views"])),
                    "ifu_designs": ",".join(str(value) for value in sorted(item["ifu_designs"])),
                }
            )
    rows = sorted(rows, key=lambda row: (row["snapshot"], row["subhalo_id"]))
    snapshots = {}
    for row in rows:
        snap = int(row["snapshot"])
        snapshots.setdefault(snap, {"n_galaxies": 0, "n_rows": 0, "n_star_part": 0})
        snapshots[snap]["n_galaxies"] += 1
        snapshots[snap]["n_rows"] += int(row["n_rows"])
        snapshots[snap]["n_star_part"] += int(row["n_star_part"])
    summary = {
        "catalog_path": str(catalog_path),
        "n_unique_galaxies": len(rows),
        "n_catalog_rows": int(sum(int(row["n_rows"]) for row in rows)),
        "snapshots": {str(key): value for key, value in sorted(snapshots.items())},
        "total_star_particles_catalog": int(np.sum([int(row["n_star_part"]) for row in rows])),
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a unique TNG galaxy manifest from MaNGIA_catalog.fits.")
    parser.add_argument("--catalog", default=str(WORKSPACE_DIR / "MaNGIA_catalog.fits"))
    parser.add_argument("--out-csv", default=str(PROJECT_DIR / "outputs_catalog" / "mangia_unique_galaxies.csv"))
    parser.add_argument("--out-json", default=str(PROJECT_DIR / "outputs_catalog" / "mangia_unique_galaxies_summary.json"))
    args = parser.parse_args()

    rows, summary = build_unique_manifest(Path(args.catalog))
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["galaxy_id", "snapshot", "subhalo_id", "n_rows", "views", "ifu_designs", "n_star_part", "n_gas_cell", "re_kpc"]
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({"csv": str(out_csv), "summary": str(out_json), **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
