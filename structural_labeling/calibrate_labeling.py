from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np

from labeling.config import LabelConfig
from labeling.manifest import read_manifest
from labeling.pipeline import LabelingPipeline
from labeling.ssp import load_ssp_grid
from labeling.tng import load_cutout_truth, load_morphology_targets


def calibration_score(targets: dict[str, float | bool], recovered: dict[str, float]) -> float:
    loss = 0.0
    loss += abs(recovered["bulbo"] - float(targets["bulge_family"]))
    loss += abs(recovered["disk_family_total"] - float(targets["disk_family"]))
    loss += abs(recovered["other"] - float(targets["other_family"]))
    if bool(targets["barred"]):
        loss += max(0.0, 0.02 - recovered["barra"]) * 5.0
    else:
        loss += recovered["barra"] * 5.0
    return float(loss)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibra hiperparámetros globales del etiquetado estructural.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--truth-dir", required=True, help="Directorio con cutouts/ y metadata/")
    parser.add_argument("--morphology-catalog", required=True)
    parser.add_argument("--ssp-template", default="/home/andy/pythonprojects/cubes/official_mangia/libs/MaStar_CB19.slog_1_5.fits.gz")
    parser.add_argument("--out-config", required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)[: args.max_samples]
    if not rows:
        raise SystemExit("El manifiesto está vacío")

    ssp_grid = load_ssp_grid(args.ssp_template)
    truth_dir = Path(args.truth_dir)

    best_score = np.inf
    best_config = None
    for bulge_width, other_width, arm_thresh in product((0.25, 0.35, 0.45), (0.15, 0.20, 0.30), (0.10, 0.15, 0.20)):
        config = LabelConfig(
            bulge_width_fraction=bulge_width,
            other_width_fraction=other_width,
            arm_residual_threshold=arm_thresh,
        )
        pipeline = LabelingPipeline(config, ssp_grid)
        losses = []
        for row in rows:
            cutout_path = truth_dir / "cutouts" / f"{row.canonical_id}.cutout.hdf5"
            metadata_path = truth_dir / "metadata" / f"{row.canonical_id}.subhalo.json"
            if not cutout_path.exists() or not metadata_path.exists():
                continue
            truth = load_cutout_truth(cutout_path, metadata_path)
            targets = load_morphology_targets(args.morphology_catalog, row.snapshot, row.subhalo_id)
            products = pipeline.run(row, truth, targets)
            losses.append(calibration_score(products.global_fraction_targets, products.global_fraction_recovered))
        if not losses:
            continue
        score = float(np.mean(losses))
        if score < best_score:
            best_score = score
            best_config = config

    if best_config is None:
        raise SystemExit("No pude calibrar: faltan archivos de verdad TNG o el conjunto piloto quedó vacío")
    best_config.to_json(args.out_config)
    print(f"Config calibrada en {args.out_config} con score={best_score:.4f}")


if __name__ == "__main__":
    main()
