from __future__ import annotations

import csv
from dataclasses import fields
from pathlib import Path
import re

from astropy.io import fits

from .models import ManifestRow


RSS_RE = re.compile(
    r"^TNG50-(?P<snapshot>\d+)-(?P<subhalo_id>\d+)-(?P<view>\d+)-(?P<ifu>\d+)(?:\.[^.]+)?\.cube_RSS(?:\.fits(?:\.gz)?)?$"
)
CUBE_RE = re.compile(
    r"^TNG50-(?P<snapshot>\d+)-(?P<subhalo_id>\d+)-(?P<view>\d+)-(?P<ifu>\d+)\.cube(?:_val)?\.fits(?:\.gz)?$"
)


def parse_canonical_id(name: str) -> dict[str, int]:
    match = RSS_RE.match(name) or CUBE_RE.match(name)
    if not match:
        raise ValueError(f"No pude extraer snapshot/subhalo_id/view/ifu desde {name}")
    return {
        "snapshot": int(match.group("snapshot")),
        "subhalo_id": int(match.group("subhalo_id")),
        "view": int(match.group("view")),
        "ifu_design": int(match.group("ifu")),
    }


def canonical_id_from_parts(snapshot: int, subhalo_id: int, view: int, ifu_design: int) -> str:
    return f"TNG50-{snapshot}-{subhalo_id}-{view}-{ifu_design}"


def _load_catalog_rows(catalog_path: str | Path) -> dict[tuple[int, int, int], dict[str, float | int]]:
    data = fits.getdata(catalog_path, 1)
    rows: dict[tuple[int, int, int], dict[str, float | int]] = {}
    duplicate_counter: dict[tuple[int, int], int] = {}

    for row in data:
        snap = int(row["snapshot"])
        sub = int(row["subhalo_id"])
        view = int(row["view"])
        key = (snap, sub, view)
        rows[key] = {
            "re_kpc": float(row["re_kpc"]),
            "ifu_design": int(row["manga_ifu_dsn"]),
            "n_star_part": int(row["n_star_part"]),
            "n_gas_cell": int(row["n_gas_cell"]),
        }
        duplicate_counter[(snap, sub)] = duplicate_counter.get((snap, sub), 0) + 1

    for (snap, sub, view), values in rows.items():
        values["repeat_count"] = duplicate_counter[(snap, sub)]
    return rows


def build_manifest(
    catalog_path: str | Path,
    rss_paths: list[Path],
    cube_paths: list[Path],
    pipe3d_paths: list[Path],
) -> list[ManifestRow]:
    catalog_rows = _load_catalog_rows(catalog_path)
    rss_index = {}
    cube_index = {}
    pipe3d_index = {}

    for path in rss_paths:
        parts = parse_canonical_id(path.name)
        canonical = canonical_id_from_parts(**parts)
        rss_index[canonical] = str(path.resolve())

    for path in cube_paths:
        if path.name.endswith(".cube_val.fits.gz") or path.name.endswith(".cube_val.fits"):
            continue
        parts = parse_canonical_id(path.name)
        canonical = canonical_id_from_parts(**parts)
        cube_index[canonical] = str(path.resolve())

    for path in pipe3d_paths:
        pipe3d_index[path.stem] = str(path.resolve())

    rows: list[ManifestRow] = []
    canonical_ids = sorted(set(rss_index) | set(cube_index))
    for canonical in canonical_ids:
        reference_path = rss_index.get(canonical) or cube_index.get(canonical)
        parts = parse_canonical_id(Path(reference_path).name)
        key = (parts["snapshot"], parts["subhalo_id"], parts["view"])
        if key not in catalog_rows:
            continue
        catalog = catalog_rows[key]
        pipe3d_matches = [value for key_name, value in pipe3d_index.items() if canonical in key_name]
        rows.append(
            ManifestRow(
                canonical_id=canonical,
                rss_path=rss_index.get(canonical, ""),
                cube_path=cube_index.get(canonical, ""),
                pipe3d_path=pipe3d_matches[0] if pipe3d_matches else "",
                snapshot=parts["snapshot"],
                subhalo_id=parts["subhalo_id"],
                view=parts["view"],
                re_kpc=float(catalog["re_kpc"]),
                ifu_design=int(parts["ifu_design"]),
                repeat_count=int(catalog["repeat_count"]),
                n_star_part=int(catalog["n_star_part"]),
                n_gas_cell=int(catalog["n_gas_cell"]),
            )
        )
    return rows


def write_manifest(path: str | Path, rows: list[ManifestRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(ManifestRow)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def read_manifest(path: str | Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with Path(path).open() as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                ManifestRow(
                    canonical_id=raw["canonical_id"],
                    rss_path=raw["rss_path"],
                    cube_path=raw["cube_path"],
                    pipe3d_path=raw["pipe3d_path"],
                    snapshot=int(raw["snapshot"]),
                    subhalo_id=int(raw["subhalo_id"]),
                    view=int(raw["view"]),
                    re_kpc=float(raw["re_kpc"]),
                    ifu_design=int(raw["ifu_design"]),
                    repeat_count=int(raw["repeat_count"]),
                    n_star_part=int(raw["n_star_part"]),
                    n_gas_cell=int(raw["n_gas_cell"]),
                )
            )
    return rows
