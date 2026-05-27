from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterable, Literal

from .ids import UnitKey
from .models import AssetScan, CatalogUnit
from .scanners import scan_cubes, scan_maps, scan_tng_cache


SelectionOrder = Literal["estimated_raw_mb", "catalog_order", "random"]


@dataclass(frozen=True, slots=True)
class MatchConfig:
    catalog: Path
    cube_roots: tuple[Path, ...]
    tng_cache: Path
    maps2d_roots: tuple[Path, ...]
    limit: int = 0
    require_count: int = 0
    selection_order: SelectionOrder = "estimated_raw_mb"
    seed: int = 42


@dataclass(frozen=True, slots=True)
class MatchResult:
    matched_all: list[dict[str, object]]
    selected: list[dict[str, object]]
    inventory: list[dict[str, object]]
    report: dict[str, object]


MATCHED_FIELDNAMES = [
    "unit_id",
    "galaxy_id",
    "canonical_id",
    "snapshot",
    "subhalo_id",
    "view",
    "ifu_design_catalog",
    "cube_path",
    "cutout_path",
    "metadata_path",
    "morphology_catalog_path",
    "maps2d_path",
    "maps2d_format",
    "v_map_key",
    "sigma_map_key",
    "cube_shape",
    "maps2d_shape",
    "re_kpc",
    "sample_manga",
    "n_star_part",
    "n_gas_cell",
    "estimated_raw_mb",
    "selection_rank",
]

INVENTORY_FIELDNAMES = [
    *MATCHED_FIELDNAMES,
    "has_cube",
    "has_cutout",
    "has_metadata",
    "has_morphology_catalog",
    "has_maps2d",
    "has_v",
    "has_sigma",
    "is_strict_match",
    "exclusion_reasons",
]


def estimated_cutout_mb(n_star_part: int, n_gas_cell: int) -> float:
    return float((int(n_star_part) * 9 * 8 + int(n_gas_cell) * 12 * 8) / 1_000_000.0)


def read_catalog(path: str | Path) -> list[CatalogUnit]:
    path = Path(path).expanduser()
    if path.name.endswith(".csv"):
        return _read_catalog_csv(path)
    return _read_catalog_fits(path)


def _read_catalog_csv(path: Path) -> list[CatalogUnit]:
    rows: list[CatalogUnit] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for order, raw in enumerate(reader):
            ifu = raw.get("manga_ifu_dsn", raw.get("ifu_design", raw.get("ifu_design_catalog", "0")))
            n_star = int(float(raw.get("n_star_part", 0)))
            n_gas = int(float(raw.get("n_gas_cell", 0)))
            rows.append(
                CatalogUnit(
                    key=UnitKey(
                        int(float(raw["snapshot"])),
                        int(float(raw["subhalo_id"])),
                        int(float(raw["view"])),
                    ),
                    ifu_design=int(float(ifu)),
                    re_kpc=float(raw.get("re_kpc", 0.0)),
                    sample_manga=int(float(raw.get("sample_manga", 0))),
                    n_star_part=n_star,
                    n_gas_cell=n_gas,
                    estimated_raw_mb=estimated_cutout_mb(n_star, n_gas),
                    catalog_order=order,
                )
            )
    return rows


def _read_catalog_fits(path: Path) -> list[CatalogUnit]:
    try:
        from astropy.io import fits
    except Exception as exc:
        raise RuntimeError("Reading FITS catalogs requires astropy. Install astropy or pass a CSV catalog.") from exc
    rows: list[CatalogUnit] = []
    data = fits.getdata(path, 1)
    for order, raw in enumerate(data):
        n_star = int(raw["n_star_part"])
        n_gas = int(raw["n_gas_cell"])
        rows.append(
            CatalogUnit(
                key=UnitKey(int(raw["snapshot"]), int(raw["subhalo_id"]), int(raw["view"])),
                ifu_design=int(raw["manga_ifu_dsn"]),
                re_kpc=float(raw["re_kpc"]),
                sample_manga=int(raw["sample_manga"]),
                n_star_part=n_star,
                n_gas_cell=n_gas,
                estimated_raw_mb=estimated_cutout_mb(n_star, n_gas),
                catalog_order=order,
            )
        )
    return rows


def scan_assets(config: MatchConfig) -> AssetScan:
    maps, counters = scan_maps(config.maps2d_roots)
    return AssetScan(
        cubes=scan_cubes(config.cube_roots),
        tng=scan_tng_cache(config.tng_cache),
        maps=maps,
        map_files_total=counters["map_files_total"],
        map_files_id_unknown=counters["map_files_id_unknown"],
        map_files_without_v=counters["map_files_without_v"],
        map_files_without_sigma=counters["map_files_without_sigma"],
    )


def _base_row(unit: CatalogUnit, morphology_catalog_path: Path | None) -> dict[str, object]:
    key = unit.key
    return {
        "unit_id": key.unit_id,
        "galaxy_id": key.galaxy_id,
        "canonical_id": key.canonical_id(unit.ifu_design),
        "snapshot": key.snapshot,
        "subhalo_id": key.subhalo_id,
        "view": key.view,
        "ifu_design_catalog": unit.ifu_design,
        "cube_path": "",
        "cutout_path": "",
        "metadata_path": "",
        "morphology_catalog_path": str(morphology_catalog_path) if morphology_catalog_path else "",
        "maps2d_path": "",
        "maps2d_format": "",
        "v_map_key": "",
        "sigma_map_key": "",
        "cube_shape": "",
        "maps2d_shape": "",
        "re_kpc": unit.re_kpc,
        "sample_manga": unit.sample_manga,
        "n_star_part": unit.n_star_part,
        "n_gas_cell": unit.n_gas_cell,
        "estimated_raw_mb": unit.estimated_raw_mb,
        "selection_rank": "",
    }


def _inventory_row(unit: CatalogUnit, assets: AssetScan) -> dict[str, object]:
    key = unit.key
    cube = assets.cubes.get(key)
    cutout = assets.tng.cutout_for(key)
    metadata = assets.tng.metadata_for(key)
    maps = assets.maps.get(key)
    morphology = assets.tng.morphology_catalog_path
    row = _base_row(unit, morphology)
    if cube is not None:
        row["cube_path"] = str(cube.path)
        row["cube_shape"] = cube.shape
    if cutout is not None:
        row["cutout_path"] = str(cutout)
    if metadata is not None:
        row["metadata_path"] = str(metadata)
    if maps is not None:
        row["maps2d_path"] = str(maps.path)
        row["maps2d_format"] = maps.format
        row["v_map_key"] = maps.v_map_key
        row["sigma_map_key"] = maps.sigma_map_key
        row["maps2d_shape"] = maps.shape

    has_cube = cube is not None
    has_cutout = cutout is not None
    has_metadata = metadata is not None
    has_morphology = morphology is not None
    has_maps = maps is not None
    has_v = bool(maps and maps.has_v)
    has_sigma = bool(maps and maps.has_sigma)
    reasons = []
    if not has_cube:
        reasons.append("missing_cube")
    if not has_cutout:
        reasons.append("missing_cutout")
    if not has_metadata:
        reasons.append("missing_metadata")
    if not has_morphology:
        reasons.append("missing_morphology_catalog")
    if not has_maps:
        reasons.append("missing_maps2d")
    else:
        if not has_v:
            reasons.append("missing_v_map")
        if not has_sigma:
            reasons.append("missing_sigma_map")

    strict = not reasons
    row.update(
        {
            "has_cube": has_cube,
            "has_cutout": has_cutout,
            "has_metadata": has_metadata,
            "has_morphology_catalog": has_morphology,
            "has_maps2d": has_maps,
            "has_v": has_v,
            "has_sigma": has_sigma,
            "is_strict_match": strict,
            "exclusion_reasons": ";".join(reasons),
        }
    )
    return row


def _matched_row(inventory_row: dict[str, object]) -> dict[str, object]:
    return {field: inventory_row.get(field, "") for field in MATCHED_FIELDNAMES}


def _sort_matches(rows: list[dict[str, object]], units_by_key: dict[tuple[int, int, int], CatalogUnit], order: SelectionOrder, seed: int) -> list[dict[str, object]]:
    if order == "estimated_raw_mb":
        return sorted(
            rows,
            key=lambda row: (
                float(row["estimated_raw_mb"]),
                int(row["snapshot"]),
                int(row["subhalo_id"]),
                int(row["view"]),
            ),
        )
    if order == "catalog_order":
        return sorted(
            rows,
            key=lambda row: units_by_key[(int(row["snapshot"]), int(row["subhalo_id"]), int(row["view"]))].catalog_order,
        )
    if order == "random":
        shuffled = sorted(rows, key=lambda row: (int(row["snapshot"]), int(row["subhalo_id"]), int(row["view"])))
        rng = random.Random(int(seed))
        rng.shuffle(shuffled)
        return shuffled
    raise ValueError(f"Invalid selection order: {order}")


def _apply_selection_rank(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ranked = []
    for index, row in enumerate(rows, start=1):
        updated = dict(row)
        updated["selection_rank"] = index
        ranked.append(updated)
    return ranked


def _build_report(catalog: list[CatalogUnit], assets: AssetScan, inventory: list[dict[str, object]], selected: list[dict[str, object]], config: MatchConfig) -> dict[str, object]:
    reason_counts: dict[str, int] = {}
    for row in inventory:
        reasons = str(row.get("exclusion_reasons", ""))
        if not reasons:
            continue
        for reason in reasons.split(";"):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    strict_count = sum(bool(row["is_strict_match"]) for row in inventory)
    return {
        "n_catalog_units": len(catalog),
        "n_cubes": len(assets.cubes),
        "n_map_units": len(assets.maps),
        "n_map_files_total": assets.map_files_total,
        "n_map_files_id_unknown": assets.map_files_id_unknown,
        "n_map_files_without_v": assets.map_files_without_v,
        "n_map_files_without_sigma": assets.map_files_without_sigma,
        "n_cutouts_unit": len(assets.tng.cutouts_by_unit),
        "n_cutouts_galaxy": len(assets.tng.cutouts_by_galaxy),
        "n_metadata_unit": len(assets.tng.metadata_by_unit),
        "n_metadata_galaxy": len(assets.tng.metadata_by_galaxy),
        "morphology_catalog_path": str(assets.tng.morphology_catalog_path or ""),
        "n_strict_matches": strict_count,
        "n_selected": len(selected),
        "limit": config.limit,
        "require_count": config.require_count,
        "selection_order": config.selection_order,
        "seed": config.seed,
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
    }


def build_matches(config: MatchConfig) -> MatchResult:
    catalog = read_catalog(config.catalog)
    assets = scan_assets(config)
    inventory = [_inventory_row(unit, assets) for unit in catalog]
    all_matches = [_matched_row(row) for row in inventory if bool(row["is_strict_match"])]
    units_by_key = {
        (unit.key.snapshot, unit.key.subhalo_id, unit.key.view): unit
        for unit in catalog
    }
    sorted_matches = _sort_matches(all_matches, units_by_key, config.selection_order, config.seed)
    selected_without_rank = sorted_matches if int(config.limit) <= 0 else sorted_matches[: int(config.limit)]
    selected = _apply_selection_rank(selected_without_rank)
    all_ranked = _apply_selection_rank(sorted_matches)
    report = _build_report(catalog, assets, inventory, selected, config)
    return MatchResult(matched_all=all_ranked, selected=selected, inventory=inventory, report=report)


def write_csv(path: str | Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def write_report_json(path: str | Path, report: dict[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_report_markdown(path: str | Path, report: dict[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MaNGIA Asset Matcher Report",
        "",
        f"Catalog units: {report['n_catalog_units']}",
        f"Strict matches: {report['n_strict_matches']}",
        f"Selected units: {report['n_selected']}",
        f"Limit: {report['limit']}",
        f"Required count: {report['require_count']}",
        f"Selection order: {report['selection_order']}",
        "",
        "## Inventario",
        "",
        f"- Cubes: {report['n_cubes']}",
        f"- Map units: {report['n_map_units']}",
        f"- Map files scanned: {report['n_map_files_total']}",
        f"- Map files without parseable ID: {report['n_map_files_id_unknown']}",
        f"- Map files without V: {report['n_map_files_without_v']}",
        f"- Map files without sigma: {report['n_map_files_without_sigma']}",
        f"- Cutouts by unit: {report['n_cutouts_unit']}",
        f"- Cutouts by galaxy: {report['n_cutouts_galaxy']}",
        f"- Metadata by unit: {report['n_metadata_unit']}",
        f"- Metadata by galaxy: {report['n_metadata_galaxy']}",
        f"- Morphology catalog: {report['morphology_catalog_path'] or 'MISSING'}",
        "",
        "## Exclusiones",
        "",
    ]
    counts = report.get("exclusion_reason_counts", {})
    if counts:
        for reason, count in counts.items():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- ninguna")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_outputs(result: MatchResult, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    write_csv(output_dir / "matched_units.csv", result.selected, MATCHED_FIELDNAMES)
    write_csv(output_dir / "matched_units_all.csv", result.matched_all, MATCHED_FIELDNAMES)
    write_csv(output_dir / "asset_inventory.csv", result.inventory, INVENTORY_FIELDNAMES)
    write_report_json(output_dir / "unmatched_report.json", result.report)
    write_report_markdown(output_dir / "unmatched_report.md", result.report)
