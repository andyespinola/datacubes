from __future__ import annotations

import argparse
import os
from pathlib import Path

from labeling.config import LabelConfig
from labeling.env import load_env_file
from labeling.manifest import read_manifest
from labeling.pipeline import LabelingPipeline, save_products
from labeling.ssp import load_ssp_grid
from labeling.tng import load_cutout_truth, load_morphology_targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera etiquetas estructurales bulbo/disco/barra/brazos para MaNGIA.")
    parser.add_argument("--manifest", required=True, help="CSV con el manifiesto")
    parser.add_argument("--canonical-id", required=True, help="ID canónico de la galaxia a procesar")
    parser.add_argument("--cutout", required=True, help="Cutout HDF5 local del subhalo")
    parser.add_argument("--metadata", required=True, help="JSON local con metadatos del subhalo")
    parser.add_argument("--morphology-catalog", required=True, help="Catálogo morfológico TNG (t)")
    parser.add_argument("--ssp-template", default="/home/andy/pythonprojects/cubes/official_mangia/libs/MaStar_CB19.slog_1_5.fits.gz", help="Plantilla SSP usada para pesos por luz")
    parser.add_argument("--config", default="", help="JSON opcional con hiperparámetros")
    parser.add_argument("--outdir", required=True, help="Directorio de salida")
    parser.add_argument("--env-file", default="", help="Archivo .env opcional")
    args = parser.parse_args()

    load_env_file(args.env_file)
    rows = read_manifest(args.manifest)
    selected = [row for row in rows if row.canonical_id == args.canonical_id]
    if not selected:
        raise SystemExit(f"No encontré {args.canonical_id} en el manifiesto")
    row = selected[0]

    config = LabelConfig.from_json(args.config) if args.config else LabelConfig()
    ssp_grid = load_ssp_grid(args.ssp_template)
    truth = load_cutout_truth(args.cutout, args.metadata)
    targets = load_morphology_targets(args.morphology_catalog, row.snapshot, row.subhalo_id)

    pipeline = LabelingPipeline(config, ssp_grid)
    products = pipeline.run(row, truth, targets)
    outbase = Path(args.outdir) / row.canonical_id
    save_products(outbase, products)
    print(f"Etiquetas guardadas en {outbase}.labels.npz")
    print(f"QA guardado en {outbase}.qa.npz")
    print(f"Resumen guardado en {outbase}.summary.json")


if __name__ == "__main__":
    main()
