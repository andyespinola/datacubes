from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

STRUCTURAL_DIR = Path(__file__).resolve().parent
if str(STRUCTURAL_DIR) not in sys.path:
    sys.path.insert(0, str(STRUCTURAL_DIR))

from labeling.models import ManifestRow


RUN_FIELDNAMES = [
    "canonical_id",
    "unit_id",
    "galaxy_id",
    "snapshot",
    "subhalo_id",
    "view",
    "status",
    "label_path",
    "qa_path",
    "summary_path",
    "cube_path",
    "cutout_path",
    "metadata_path",
    "maps2d_path",
    "elapsed_seconds",
    "error",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_ssp_template_path() -> Path:
    return repo_root() / "kinematic_moments" / "templates" / "MaStar_CB19.slog_1_5.fits.gz"


def default_config_path() -> Path:
    return Path(__file__).resolve().with_name("default_config.json")


def _int_value(row: dict[str, str], name: str, default: int = 0) -> int:
    value = (row.get(name) or "").strip()
    return int(float(value)) if value else default


def _float_value(row: dict[str, str], name: str, default: float = 0.0) -> float:
    value = (row.get(name) or "").strip()
    return float(value) if value else default


def _load_repeat_counts(catalog_path: str | Path) -> dict[tuple[int, int], int]:
    from astropy.io import fits

    counts: Counter[tuple[int, int]] = Counter()
    data = fits.getdata(Path(catalog_path), 1)
    for raw in data:
        counts[(int(raw["snapshot"]), int(raw["subhalo_id"]))] += 1
    return dict(counts)


def _fallback_repeat_counts(rows: Iterable[dict[str, str]]) -> dict[tuple[int, int], int]:
    counts: Counter[tuple[int, int]] = Counter()
    for row in rows:
        counts[(_int_value(row, "snapshot"), _int_value(row, "subhalo_id"))] += 1
    return dict(counts)


def _matched_row_to_manifest(
    row: dict[str, str],
    repeat_counts: dict[tuple[int, int], int],
) -> ManifestRow:
    snapshot = _int_value(row, "snapshot")
    subhalo_id = _int_value(row, "subhalo_id")
    view = _int_value(row, "view")
    ifu_design = _int_value(row, "ifu_design_catalog", _int_value(row, "cube_ifu_file", 0))
    unit_id = row.get("unit_id") or f"TNG50-{snapshot}-{subhalo_id}-{view}"
    canonical_id = row.get("canonical_id") or f"{unit_id}-{ifu_design}"
    return ManifestRow(
        canonical_id=canonical_id,
        rss_path="",
        cube_path=(row.get("cube_path") or "").strip(),
        pipe3d_path=(row.get("maps2d_path") or "").strip(),
        snapshot=snapshot,
        subhalo_id=subhalo_id,
        view=view,
        re_kpc=_float_value(row, "re_kpc"),
        ifu_design=ifu_design,
        repeat_count=max(1, int(repeat_counts.get((snapshot, subhalo_id), 1))),
        n_star_part=_int_value(row, "n_star_part"),
        n_gas_cell=_int_value(row, "n_gas_cell"),
    )


def read_matched_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_structural_manifest(
    matched_units_path: str | Path,
    catalog_path: str | Path | None = None,
) -> tuple[list[ManifestRow], list[dict[str, str]]]:
    matched_rows = read_matched_rows(matched_units_path)
    if catalog_path:
        repeat_counts = _load_repeat_counts(catalog_path)
    else:
        repeat_counts = _fallback_repeat_counts(matched_rows)
    return [_matched_row_to_manifest(row, repeat_counts) for row in matched_rows], matched_rows


def _row_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    indexed = {}
    for row in rows:
        canonical_id = (row.get("canonical_id") or "").strip()
        if canonical_id:
            indexed[canonical_id] = row
    return indexed


def _existing_outputs(outdir: Path, canonical_id: str) -> tuple[Path, Path, Path]:
    base = outdir / canonical_id
    return (
        base.with_suffix(".labels.npz"),
        base.with_suffix(".qa.npz"),
        base.with_suffix(".summary.json"),
    )


def _write_run_row(path: Path, row: dict[str, object], wrote_header: bool) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDNAMES)
        if not wrote_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RUN_FIELDNAMES})
    return True


def write_structural_manifest(path: str | Path, rows: list[ManifestRow]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].as_dict().keys()) if rows else list(ManifestRow.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
    return path


def _status_row(
    manifest_row: ManifestRow,
    matched_row: dict[str, str],
    status: str,
    outdir: Path,
    elapsed_seconds: float = 0.0,
    error: str = "",
) -> dict[str, object]:
    label_path, qa_path, summary_path = _existing_outputs(outdir, manifest_row.canonical_id)
    return {
        "canonical_id": manifest_row.canonical_id,
        "unit_id": matched_row.get("unit_id", ""),
        "galaxy_id": matched_row.get("galaxy_id", ""),
        "snapshot": manifest_row.snapshot,
        "subhalo_id": manifest_row.subhalo_id,
        "view": manifest_row.view,
        "status": status,
        "label_path": str(label_path),
        "qa_path": str(qa_path),
        "summary_path": str(summary_path),
        "cube_path": manifest_row.cube_path,
        "cutout_path": matched_row.get("cutout_path", ""),
        "metadata_path": matched_row.get("metadata_path", ""),
        "maps2d_path": matched_row.get("maps2d_path", ""),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "error": error,
    }


def _resolve_morphology_path(args: argparse.Namespace, matched_rows: list[dict[str, str]]) -> Path:
    if args.morphology_catalog:
        path = Path(args.morphology_catalog).expanduser()
    else:
        values = [(row.get("morphology_catalog_path") or "").strip() for row in matched_rows]
        values = [value for value in values if value]
        if not values:
            raise SystemExit("Necesito --morphology-catalog o una columna morphology_catalog_path en matched_units.csv")
        path = Path(values[0]).expanduser()
    if not path.exists():
        raise SystemExit(f"No existe el catalogo morfologico: {path}")
    return path


def _selected_rows(
    structural_rows: list[ManifestRow],
    matched_rows: list[dict[str, str]],
    canonical_ids: list[str],
    limit: int,
) -> tuple[list[ManifestRow], list[dict[str, str]]]:
    pairs = list(zip(structural_rows, matched_rows, strict=True))
    if canonical_ids:
        wanted = set(canonical_ids)
        pairs = [(row, matched) for row, matched in pairs if row.canonical_id in wanted]
    if limit > 0:
        pairs = pairs[:limit]
    if not pairs:
        raise SystemExit("No hay filas para procesar despues de aplicar filtros")
    return [row for row, _ in pairs], [matched for _, matched in pairs]


def run(args: argparse.Namespace) -> int:
    from labeling.config import LabelConfig
    from labeling.env import load_env_file
    from labeling.pipeline import LabelingPipeline, save_products
    from labeling.ssp import load_ssp_grid
    from labeling.tng import load_cutout_truth, load_morphology_targets

    load_env_file(args.env_file)
    structural_rows, matched_rows = build_structural_manifest(args.matched_units, args.catalog)
    structural_rows, matched_rows = _selected_rows(structural_rows, matched_rows, args.canonical_id, args.limit)
    matched_by_canonical = _row_index(matched_rows)

    outdir = Path(args.outdir).expanduser()
    manifest_out = Path(args.structural_manifest_out).expanduser() if args.structural_manifest_out else outdir / "structural_manifest.csv"
    run_manifest = Path(args.run_manifest).expanduser() if args.run_manifest else outdir / "labeling_run_manifest.csv"
    summary_path = outdir / "labeling_run_summary.json"
    morphology_catalog = _resolve_morphology_path(args, matched_rows)

    outdir.mkdir(parents=True, exist_ok=True)
    write_structural_manifest(manifest_out, structural_rows)

    config = LabelConfig.from_json(args.config) if args.config else LabelConfig()
    ssp_grid = load_ssp_grid(args.ssp_template)
    pipeline = LabelingPipeline(config, ssp_grid)

    counts: Counter[str] = Counter()
    wrote_header = run_manifest.exists() and run_manifest.stat().st_size > 0
    start_all = time.monotonic()
    for index, row in enumerate(structural_rows, start=1):
        matched_row = matched_by_canonical.get(row.canonical_id, {})
        label_path, qa_path, summary_product_path = _existing_outputs(outdir, row.canonical_id)
        if label_path.exists() and qa_path.exists() and summary_product_path.exists() and not args.overwrite:
            counts["skipped_existing"] += 1
            wrote_header = _write_run_row(
                run_manifest,
                _status_row(row, matched_row, "skipped_existing", outdir),
                wrote_header,
            )
            print(f"[{index}/{len(structural_rows)}] skip existing {row.canonical_id}", flush=True)
            continue

        t0 = time.monotonic()
        try:
            cutout_path = Path(matched_row.get("cutout_path", "")).expanduser()
            metadata_path = Path(matched_row.get("metadata_path", "")).expanduser()
            for required in (Path(row.cube_path), cutout_path, metadata_path):
                if not required.exists():
                    raise FileNotFoundError(str(required))

            print(f"[{index}/{len(structural_rows)}] labeling {row.canonical_id}", flush=True)
            truth = load_cutout_truth(cutout_path, metadata_path)
            targets = load_morphology_targets(morphology_catalog, row.snapshot, row.subhalo_id)
            products = pipeline.run(row, truth, targets)
            save_products(outdir / row.canonical_id, products)
            elapsed = time.monotonic() - t0
            counts["ok"] += 1
            wrote_header = _write_run_row(
                run_manifest,
                _status_row(row, matched_row, "ok", outdir, elapsed),
                wrote_header,
            )
            print(f"  ok elapsed={elapsed:.1f}s", flush=True)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            counts["error"] += 1
            wrote_header = _write_run_row(
                run_manifest,
                _status_row(row, matched_row, "error", outdir, elapsed, str(exc)),
                wrote_header,
            )
            print(f"  error: {exc}", flush=True)
            if not args.continue_on_error:
                summary_path.write_text(
                    json.dumps({"counts": dict(counts), "failed_at": row.canonical_id}, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                return 1

    summary = {
        "matched_units": str(Path(args.matched_units).expanduser()),
        "structural_manifest": str(manifest_out),
        "run_manifest": str(run_manifest),
        "outdir": str(outdir),
        "morphology_catalog": str(morphology_catalog),
        "n_units_requested": len(structural_rows),
        "elapsed_seconds": round(time.monotonic() - start_all, 3),
        "counts": dict(sorted(counts.items())),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if counts.get("error", 0) == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run structural labeling over mangia_asset_matcher matched units.")
    parser.add_argument("--matched-units", required=True, help="matched_units.csv generado por mangia_asset_matcher")
    parser.add_argument("--catalog", default="MaNGIA_catalog.fits", help="MaNGIA_catalog.fits para repeat_count correcto")
    parser.add_argument("--morphology-catalog", default="", help="Override del catalogo morfologico HDF5")
    parser.add_argument(
        "--ssp-template",
        default=str(default_ssp_template_path()),
        help="Plantilla SSP usada para pesos por luz",
    )
    parser.add_argument("--config", default=str(default_config_path()), help="Config JSON de etiquetado")
    parser.add_argument("--outdir", required=True, help="Directorio donde escribir *.labels.npz, *.qa.npz y summaries")
    parser.add_argument("--structural-manifest-out", default="", help="CSV derivado para auditoria")
    parser.add_argument("--run-manifest", default="", help="CSV incremental con status por unidad")
    parser.add_argument("--canonical-id", action="append", default=[], help="Procesa solo este canonical_id; puede repetirse")
    parser.add_argument("--limit", type=int, default=0, help="Maximo de filas a procesar; 0=todas")
    parser.add_argument("--overwrite", action="store_true", help="Recalcula aunque existan productos")
    parser.add_argument("--continue-on-error", action="store_true", help="Registra errores y continua")
    parser.add_argument("--env-file", default="", help="Archivo .env opcional")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
