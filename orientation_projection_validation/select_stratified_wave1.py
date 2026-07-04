"""Selecciona una muestra estratificada por morfología MORDOR desde el catálogo
MaNGIA completo (10k filas), para extender el piloto de soft_labels_generation_v2_2
mas alla de las 2 galaxias iniciales (155298, 192324).

Estratos (mismo criterio bulge=Bulge+PseudoBulge, disk=ThinDisc+ThickDisc,
other=Halo que usa aperturenet_labels/io/catalogs.py):
  - halo:  fraccion halo   > 0.5  (estrato de riesgo: anomalia GMM conocida)
  - disk:  fraccion disco  > 0.5  (control limpio)
  - bulge: fraccion bulbo  > 0.5  (control bulbo/pseudo-bulbo)
  - mixed: ninguna familia > 0.5  (control ambiguo)

Uso:
    python select_stratified_wave1.py \
        --catalog /home/andy/pythonProjects/datacubes/MaNGIA_catalog.fits \
        --mordor /home/andy/pythonProjects/datacubes/data/morphs_kinematic_bars.hdf5 \
        --out-manifest data/wave1_manifest.csv \
        --out-report data/wave1_strata.csv \
        --n-halo 8 --n-disk 4 --n-bulge 4 --n-mixed 4 \
        --seed 42
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from orientation_validation.manifest import (  # noqa: E402
    ProjectionManifestRow,
    build_projection_manifest,
    read_manifest,
    write_manifest,
)

# Ya procesadas como pilotos en soft_labels_generation_v2_2; no re-seleccionar.
EXCLUDED_SUBHALOS = {(87, 155298), (87, 192324), (87, 141934)}

STRATA = ("halo", "disk", "bulge", "mixed")


def load_mordor_fractions(mordor_path: str | Path, snapshot: int, subhalo_id: int) -> dict | None:
    with h5py.File(mordor_path, "r") as f:
        key = f"Snapshot_{snapshot}"
        if key not in f:
            return None
        g = f[key]
        ids = np.asarray(g["SubhaloID"])
        idx = np.where(ids == subhalo_id)[0]
        if idx.size == 0:
            return None
        i = int(idx[0])
        bulge = float(g["Bulge"][0, i]) + float(g["PseudoBulge"][0, i])
        disk = float(g["ThinDisc"][0, i]) + float(g["ThickDisc"][0, i])
        halo = float(g["Halo"][0, i])
        total = bulge + disk + halo
        if total <= 0:
            return None
        barred = bool(g["Barred"][i]) if "Barred" in g else False
        return {
            "bulge_frac": bulge / total,
            "disk_frac": disk / total,
            "halo_frac": halo / total,
            "barred": barred,
        }


def classify(fracs: dict) -> str:
    if fracs["halo_frac"] > 0.5:
        return "halo"
    if fracs["disk_frac"] > 0.5:
        return "disk"
    if fracs["bulge_frac"] > 0.5:
        return "bulge"
    return "mixed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, help="MaNGIA_catalog.fits")
    parser.add_argument("--mordor", required=True, help="morphs_kinematic_bars.hdf5 (MORDOR)")
    parser.add_argument("--out-manifest", required=True, help="CSV manifiesto (formato ProjectionManifestRow)")
    parser.add_argument("--out-report", required=True, help="CSV con fracciones/estrato por galaxia")
    parser.add_argument("--n-halo", type=int, default=8)
    parser.add_argument("--n-disk", type=int, default=4)
    parser.add_argument("--n-bulge", type=int, default=4)
    parser.add_argument("--n-mixed", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--available-ids-file",
        default="",
        help="Restringe el pool a (snapshot, subhalo_id) listados como 'snapshot-subhalo_id' por linea "
        "(ej. cubos MaNGIA ya reconstruidos en disco)",
    )
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        default=[],
        help="CSV manifiesto (formato ProjectionManifestRow) cuyas galaxias se excluyen del pool "
        "(ej. una oleada anterior ya seleccionada); puede repetirse",
    )
    args = parser.parse_args()

    n_target = {"halo": args.n_halo, "disk": args.n_disk, "bulge": args.n_bulge, "mixed": args.n_mixed}

    available_ids: set[tuple[int, int]] | None = None
    if args.available_ids_file:
        available_ids = set()
        for line in Path(args.available_ids_file).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            snap_str, sub_str = line.split("-", 1)
            available_ids.add((int(snap_str), int(sub_str)))
        print(f"Restringiendo pool a {len(available_ids)} galaxias disponibles en disco", flush=True)

    excluded_extra: set[tuple[int, int]] = set()
    for path in args.exclude_manifest:
        for row in read_manifest(path):
            excluded_extra.add((row.snapshot, row.subhalo_id))
    if excluded_extra:
        print(f"Excluyendo {len(excluded_extra)} galaxias de oleadas previas", flush=True)

    rows = build_projection_manifest(args.catalog)
    print(f"Galaxias unicas en catalogo MaNGIA: {len(rows)}", flush=True)

    strata: dict[str, list[tuple[ProjectionManifestRow, dict]]] = {k: [] for k in STRATA}
    missing_mordor = 0
    for row in rows:
        if (row.snapshot, row.subhalo_id) in EXCLUDED_SUBHALOS:
            continue
        if (row.snapshot, row.subhalo_id) in excluded_extra:
            continue
        if available_ids is not None and (row.snapshot, row.subhalo_id) not in available_ids:
            continue
        fracs = load_mordor_fractions(args.mordor, row.snapshot, row.subhalo_id)
        if fracs is None:
            missing_mordor += 1
            continue
        strata[classify(fracs)].append((row, fracs))

    print(
        f"Disponibles por estrato: "
        + ", ".join(f"{k}={len(v)}" for k, v in strata.items())
        + f" | sin match MORDOR: {missing_mordor}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    selected_rows: list[ProjectionManifestRow] = []
    report_lines: list[dict] = []
    for stratum in STRATA:
        pool = strata[stratum]
        n = min(n_target[stratum], len(pool))
        if n < n_target[stratum]:
            print(f"AVISO: estrato {stratum} solo tiene {len(pool)} candidatos (< {n_target[stratum]} pedidos)", flush=True)
        idx = rng.choice(len(pool), size=n, replace=False) if pool else np.array([], dtype=int)
        for i in idx:
            row, fracs = pool[int(i)]
            selected_rows.append(row)
            report_lines.append(
                {
                    "galaxy_id": row.galaxy_id,
                    "snapshot": row.snapshot,
                    "subhalo_id": row.subhalo_id,
                    "stratum": stratum,
                    "bulge_frac": round(fracs["bulge_frac"], 4),
                    "disk_frac": round(fracs["disk_frac"], 4),
                    "halo_frac": round(fracs["halo_frac"], 4),
                    "barred": fracs["barred"],
                    "estimated_raw_mb": round(row.estimated_raw_mb, 1),
                }
            )

    out_manifest = Path(args.out_manifest)
    write_manifest(out_manifest, selected_rows)
    print(f"Manifiesto escrito: {out_manifest} ({len(selected_rows)} galaxias)", flush=True)

    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with out_report.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_lines[0].keys()) if report_lines else [])
        writer.writeheader()
        writer.writerows(report_lines)
    print(f"Reporte de estratos escrito: {out_report}", flush=True)

    total_mb = sum(row.estimated_raw_mb for row in selected_rows)
    print(f"Tamano estimado total (solo estrellas+gas, sin DM): {total_mb:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
