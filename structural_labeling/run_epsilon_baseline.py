from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

STRUCTURAL_DIR = Path(__file__).resolve().parent
if str(STRUCTURAL_DIR) not in sys.path:
    sys.path.insert(0, str(STRUCTURAL_DIR))

from run_matched_labeling import (  # noqa: E402
    _existing_outputs,
    _row_index,
    _selected_rows,
    _status_row,
    _write_run_row,
    build_structural_manifest,
    default_config_path,
    default_ssp_template_path,
    write_structural_manifest,
)


def run(args: argparse.Namespace) -> int:
    from labeling.config import LabelConfig
    from labeling.env import load_env_file
    from labeling.epsilon_baseline import EpsilonBaselineConfig, build_epsilon_baseline_products
    from labeling.pipeline import save_products
    from labeling.ssp import load_ssp_grid
    from labeling.tng import load_cutout_truth

    load_env_file(args.env_file)
    structural_rows, matched_rows = build_structural_manifest(args.matched_units, args.catalog)
    structural_rows, matched_rows = _selected_rows(structural_rows, matched_rows, args.canonical_id, args.limit)
    matched_by_canonical = _row_index(matched_rows)

    outdir = Path(args.outdir).expanduser()
    manifest_out = Path(args.structural_manifest_out).expanduser() if args.structural_manifest_out else outdir / "epsilon_structural_manifest.csv"
    run_manifest = Path(args.run_manifest).expanduser() if args.run_manifest else outdir / "epsilon_baseline_run_manifest.csv"
    summary_path = outdir / "epsilon_baseline_run_summary.json"

    outdir.mkdir(parents=True, exist_ok=True)
    write_structural_manifest(manifest_out, structural_rows)

    label_config = LabelConfig.from_json(args.config) if args.config else LabelConfig()
    baseline_config = EpsilonBaselineConfig(
        disk_threshold=args.disk_threshold,
        circularity_definition=args.circularity_definition,
        counterrotating_as_other=args.counterrotating_as_other,
        counterrotating_threshold=args.counterrotating_threshold,
    )
    ssp_grid = load_ssp_grid(args.ssp_template)

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
            print(f"[{index}/{len(structural_rows)}] skip existing epsilon {row.canonical_id}", flush=True)
            continue

        t0 = time.monotonic()
        try:
            cutout_path = Path(matched_row.get("cutout_path", "")).expanduser()
            metadata_path = Path(matched_row.get("metadata_path", "")).expanduser()
            for required in (Path(row.cube_path), cutout_path, metadata_path):
                if not required.exists():
                    raise FileNotFoundError(str(required))

            print(f"[{index}/{len(structural_rows)}] epsilon baseline {row.canonical_id}", flush=True)
            truth = load_cutout_truth(cutout_path, metadata_path)
            products = build_epsilon_baseline_products(row, truth, ssp_grid, label_config, baseline_config)
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
        "n_units_requested": len(structural_rows),
        "elapsed_seconds": round(time.monotonic() - start_all, 3),
        "baseline": {
            "name": "epsilon_threshold",
            "disk_threshold": float(args.disk_threshold),
            "circularity_definition": args.circularity_definition,
            "counterrotating_as_other": bool(args.counterrotating_as_other),
            "counterrotating_threshold": float(args.counterrotating_threshold),
        },
        "counts": dict(sorted(counts.items())),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if counts.get("error", 0) == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a simple hard-epsilon baseline over matched MaNGIA units.")
    parser.add_argument("--matched-units", required=True, help="matched_units.csv generado por mangia_asset_matcher")
    parser.add_argument("--catalog", default="MaNGIA_catalog.fits", help="MaNGIA_catalog.fits para repeat_count correcto")
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
    parser.add_argument("--disk-threshold", type=float, default=0.70, help="Etiqueta disco si epsilon >= este valor")
    parser.add_argument(
        "--circularity-definition",
        choices=("vphi_over_vtotal", "jz_over_jnorm"),
        default="vphi_over_vtotal",
        help="Proxy de circularidad orbital usado para el umbral duro",
    )
    parser.add_argument("--counterrotating-as-other", action="store_true", help="Manda estrellas muy contrarrotantes a other")
    parser.add_argument("--counterrotating-threshold", type=float, default=-0.70)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
