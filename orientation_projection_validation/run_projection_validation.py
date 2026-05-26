from __future__ import annotations

import argparse
import json
from pathlib import Path

from orientation_validation.config import load_configs
from orientation_validation.download import cutout_path, metadata_path
from orientation_validation.manifest import read_manifest
from orientation_validation.metrics import compute_interorientation_metrics, write_metrics
from orientation_validation.paths import default_ssp_template_path, ensure_structural_labeling_on_path
from orientation_validation.projection import build_projection_product, save_projection_product
from orientation_validation.qa import write_qa_mosaic

ensure_structural_labeling_on_path()

from labeling.ssp import load_ssp_grid  # noqa: E402
from labeling.tng import load_cutout_truth, load_morphology_targets  # noqa: E402


def resolve_morphology_catalog(cache_dir: str | Path, explicit_path: str) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise SystemExit(f"No existe --morphology-catalog={path}")
        return path
    path = Path(cache_dir) / "morphology" / "morphs_kinematic_bars.hdf5"
    if path.exists():
        return path
    raise SystemExit("Necesito --morphology-catalog o cache/morphology/morphs_kinematic_bars.hdf5")


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera cuatro proyecciones y métricas de validación 8.1.")
    parser.add_argument("--manifest", required=True, help="CSV de manifiesto")
    parser.add_argument("--cache", default="cache/pilot_fresh", help="Cache con cutouts/metadatos")
    parser.add_argument("--morphology-catalog", default="", help="Catálogo morfológico TNG HDF5")
    parser.add_argument("--ssp-template", default=str(default_ssp_template_path()), help="Template SSP MaNGIA/MaStar")
    parser.add_argument("--config", default="", help="Config JSON opcional")
    parser.add_argument("--outdir", default="outputs", help="Directorio de salida")
    parser.add_argument("--max-galaxies", type=int, default=0, help="Máximo de galaxias")
    parser.add_argument("--galaxy-id", action="append", default=[], help="Filtra por galaxy_id; puede repetirse")
    parser.add_argument("--continue-on-error", action="store_true", help="Registra errores y continúa con la siguiente galaxia")
    args = parser.parse_args()

    projection_config, label_config = load_configs(args.config)
    morphology_catalog = resolve_morphology_catalog(args.cache, args.morphology_catalog)
    ssp_grid = load_ssp_grid(args.ssp_template)

    rows = read_manifest(args.manifest)
    if args.galaxy_id:
        wanted = set(args.galaxy_id)
        rows = [row for row in rows if row.galaxy_id in wanted]
    if args.max_galaxies > 0:
        rows = rows[: args.max_galaxies]
    if not rows:
        raise SystemExit("No hay galaxias para procesar")

    for index, row in enumerate(rows, start=1):
        galaxy_outdir = Path(args.outdir) / row.galaxy_id
        metrics_path = galaxy_outdir / "metrics.json"
        try:
            print(f"[{index}/{len(rows)}] proyectando {row.galaxy_id}")
            c_path = cutout_path(args.cache, row)
            m_path = metadata_path(args.cache, row)
            if not c_path.exists() or not m_path.exists():
                raise FileNotFoundError(f"Faltan assets para {row.galaxy_id}: {c_path}, {m_path}")

            truth = load_cutout_truth(c_path, m_path)
            targets = load_morphology_targets(morphology_catalog, row.snapshot, row.subhalo_id)
            products, metadata = build_projection_product(
                row,
                truth,
                targets,
                ssp_grid,
                label_config,
                projection_config,
            )
            projection_path = galaxy_outdir / "projections.h5"
            save_projection_product(projection_path, row, products, metadata, projection_config)
            metrics = compute_interorientation_metrics(products, projection_config)
            metrics = {
                "galaxy_id": row.galaxy_id,
                "snapshot": row.snapshot,
                "subhalo_id": row.subhalo_id,
                "projection_file": str(projection_path),
                **metrics,
            }
            write_metrics(metrics_path, metrics)
            write_qa_mosaic(galaxy_outdir / "qa_mosaic.png", products, variant=projection_config.main_metric_variant)
            print(f"  Cglobal={metrics['Cglobal']:.3f} accepted={metrics['accepted']}")
        except Exception as exc:
            if not args.continue_on_error:
                raise
            galaxy_outdir.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                json.dumps(
                    {
                        "galaxy_id": row.galaxy_id,
                        "snapshot": row.snapshot,
                        "subhalo_id": row.subhalo_id,
                        "accepted": False,
                        "failure_reasons": ["runtime_error"],
                        "error": str(exc),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            print(f"  error: {exc}")


if __name__ == "__main__":
    main()

