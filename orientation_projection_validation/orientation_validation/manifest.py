from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from pathlib import Path
from statistics import median
from typing import Iterable

from astropy.io import fits
import numpy as np


@dataclass(slots=True)
class ProjectionManifestRow:
    galaxy_id: str
    snapshot: int
    subhalo_id: int
    re_kpc: float
    sample_manga: int
    ifu_design: int
    n_star_part: int
    n_gas_cell: int
    source_rows: int
    views: str
    rcov_kpc: float
    estimated_raw_mb: float

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "galaxy_id": self.galaxy_id,
            "snapshot": self.snapshot,
            "subhalo_id": self.subhalo_id,
            "re_kpc": self.re_kpc,
            "sample_manga": self.sample_manga,
            "ifu_design": self.ifu_design,
            "n_star_part": self.n_star_part,
            "n_gas_cell": self.n_gas_cell,
            "source_rows": self.source_rows,
            "views": self.views,
            "rcov_kpc": self.rcov_kpc,
            "estimated_raw_mb": self.estimated_raw_mb,
        }


def galaxy_id(snapshot: int, subhalo_id: int) -> str:
    return f"TNG50-{snapshot}-{subhalo_id}"


def _mode_int(values: Iterable[int]) -> int:
    counts: dict[int, int] = {}
    for value in values:
        counts[int(value)] = counts.get(int(value), 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def rcov_from_sample(re_kpc: float, sample_manga: int, primary_factor: float = 1.5, secondary_factor: float = 2.5) -> float:
    factor = secondary_factor if int(sample_manga) == 2 else primary_factor
    return float(max(re_kpc, 1e-6) * factor)


def estimated_cutout_mb(n_star_part: int, n_gas_cell: int) -> float:
    # Requested fields: stars have 9 float64 scalars, gas has 12 float64 scalars.
    return float((int(n_star_part) * 9 * 8 + int(n_gas_cell) * 12 * 8) / 1_000_000.0)


def build_projection_manifest(
    catalog_path: str | Path,
    primary_factor: float = 1.5,
    secondary_factor: float = 2.5,
) -> list[ProjectionManifestRow]:
    data = fits.getdata(catalog_path, 1)
    groups: dict[tuple[int, int], list] = {}
    for row in data:
        key = (int(row["snapshot"]), int(row["subhalo_id"]))
        groups.setdefault(key, []).append(row)

    rows: list[ProjectionManifestRow] = []
    for (snapshot, subhalo_id), items in sorted(groups.items()):
        re_values = [float(item["re_kpc"]) for item in items]
        sample_values = [int(item["sample_manga"]) for item in items]
        ifu_values = [int(item["manga_ifu_dsn"]) for item in items]
        views = sorted({int(item["view"]) for item in items})
        n_star = max(int(item["n_star_part"]) for item in items)
        n_gas = max(int(item["n_gas_cell"]) for item in items)
        re_kpc = float(median(re_values))
        sample = _mode_int(sample_values)
        rcov = rcov_from_sample(re_kpc, sample, primary_factor, secondary_factor)
        rows.append(
            ProjectionManifestRow(
                galaxy_id=galaxy_id(snapshot, subhalo_id),
                snapshot=snapshot,
                subhalo_id=subhalo_id,
                re_kpc=re_kpc,
                sample_manga=sample,
                ifu_design=max(ifu_values),
                n_star_part=n_star,
                n_gas_cell=n_gas,
                source_rows=len(items),
                views=";".join(str(view) for view in views),
                rcov_kpc=rcov,
                estimated_raw_mb=estimated_cutout_mb(n_star, n_gas),
            )
        )
    return rows


def write_manifest(path: str | Path, rows: list[ProjectionManifestRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(ProjectionManifestRow)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def read_manifest(path: str | Path) -> list[ProjectionManifestRow]:
    rows: list[ProjectionManifestRow] = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                ProjectionManifestRow(
                    galaxy_id=raw["galaxy_id"],
                    snapshot=int(raw["snapshot"]),
                    subhalo_id=int(raw["subhalo_id"]),
                    re_kpc=float(raw["re_kpc"]),
                    sample_manga=int(raw["sample_manga"]),
                    ifu_design=int(raw["ifu_design"]),
                    n_star_part=int(raw["n_star_part"]),
                    n_gas_cell=int(raw["n_gas_cell"]),
                    source_rows=int(raw["source_rows"]),
                    views=raw["views"],
                    rcov_kpc=float(raw["rcov_kpc"]),
                    estimated_raw_mb=float(raw["estimated_raw_mb"]),
                )
            )
    return rows


def select_pilot_rows(rows: list[ProjectionManifestRow], max_galaxies: int) -> list[ProjectionManifestRow]:
    if max_galaxies <= 0 or len(rows) <= max_galaxies:
        return list(rows)

    size_values = np.asarray([row.estimated_raw_mb for row in rows], dtype=np.float64)
    q1, q2 = np.nanpercentile(size_values, [33.3, 66.6])

    def size_bin(row: ProjectionManifestRow) -> int:
        if row.estimated_raw_mb <= q1:
            return 0
        if row.estimated_raw_mb <= q2:
            return 1
        return 2

    strata: dict[tuple[int, int, int], list[ProjectionManifestRow]] = {}
    for row in sorted(rows, key=lambda item: (item.snapshot, item.subhalo_id)):
        key = (row.sample_manga, row.ifu_design, size_bin(row))
        strata.setdefault(key, []).append(row)

    selected: list[ProjectionManifestRow] = []
    for key in sorted(strata):
        selected.append(strata[key][0])
        if len(selected) >= max_galaxies:
            return selected

    chosen = {row.galaxy_id for row in selected}
    for row in sorted(rows, key=lambda item: (item.estimated_raw_mb, item.snapshot, item.subhalo_id)):
        if row.galaxy_id not in chosen:
            selected.append(row)
            chosen.add(row.galaxy_id)
        if len(selected) >= max_galaxies:
            break
    return selected

