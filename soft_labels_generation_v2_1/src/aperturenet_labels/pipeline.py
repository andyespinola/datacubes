from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

from aperturenet_labels.config import PipelineConfig
from aperturenet_labels.io.assets import LocalGalaxyAssets, assets_for_galaxy, discover_local_assets, validate_assets
from aperturenet_labels.io.circularity import load_stellar_circularity_summary
from aperturenet_labels.io.cube_reader import read_cube_geometry
from aperturenet_labels.io.morphology import load_morphology_targets
from aperturenet_labels.io.ssp import load_ssp_grid
from aperturenet_labels.io.tng_reader import validate_cutout
from aperturenet_labels.phase_a.arm_detector import add_arm_component
from aperturenet_labels.phase_a.bar_detector import add_bar_component
from aperturenet_labels.phase_a.classifier import classify_primary_components, write_particle_labels
from aperturenet_labels.phase_a.extractor import extract_particle_features, write_particle_features
from aperturenet_labels.phase_b.label_projection import project_labels, write_projected_labels
from aperturenet_labels.phase_b.mask_builder import build_valid_mask, write_valid_mask
from aperturenet_labels.phase_c.packer import write_dataset_entry
from aperturenet_labels.phase_c.quality_check import build_quality_report, write_quality_report


@dataclass(slots=True)
class PipelineOutputs:
    galaxy_id: str
    output_dir: Path
    particle_features: Path
    particle_labels_initial: Path
    particle_labels_with_bar: Path
    particle_labels_final: Path
    projected_labels: Path
    valid_mask: Path
    qa_report: Path
    dataset_entry: Path
    elapsed_seconds: float

    def as_dict(self) -> dict[str, str | float]:
        return {
            "galaxy_id": self.galaxy_id,
            "output_dir": str(self.output_dir),
            "particle_features": str(self.particle_features),
            "particle_labels_initial": str(self.particle_labels_initial),
            "particle_labels_with_bar": str(self.particle_labels_with_bar),
            "particle_labels_final": str(self.particle_labels_final),
            "projected_labels": str(self.projected_labels),
            "valid_mask": str(self.valid_mask),
            "qa_report": str(self.qa_report),
            "dataset_entry": str(self.dataset_entry),
            "elapsed_seconds": self.elapsed_seconds,
        }


def validate_local_data(config: PipelineConfig) -> list[dict]:
    rows = []
    for assets in discover_local_assets(config.data):
        missing = validate_assets(assets)
        cutout_summary = None
        if not missing and assets.cutout_path.exists():
            cutout_summary = validate_cutout(assets.cutout_path)
        rows.append(
            {
                "galaxy_id": assets.galaxy_id,
                "snapshot": assets.snapshot,
                "subhalo_id": assets.subhalo_id,
                "file_ifu_design": assets.file_ifu_design,
                "catalog_ifu_design": assets.catalog_ifu_design,
                "re_kpc": assets.re_kpc,
                "missing": missing,
                "cutout": cutout_summary,
            }
        )
    return rows


def run_one_galaxy(assets: LocalGalaxyAssets, config: PipelineConfig, overwrite: bool = False) -> PipelineOutputs:
    start = time.monotonic()
    missing = validate_assets(assets)
    if missing:
        raise FileNotFoundError(f"Missing assets for {assets.galaxy_id}: {missing}")
    outdir = config.data.output_dir / assets.galaxy_id
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "features": outdir / "particle_features.h5",
        "initial": outdir / "particle_labels_initial.h5",
        "with_bar": outdir / "particle_labels_with_bar.h5",
        "final": outdir / "particle_labels_final.h5",
        "projected": outdir / "labels2d_v0.npz",
        "mask": outdir / "M_valid_v0.npz",
        "qa": outdir / "qa_report_v0.json",
        "entry": outdir / f"dataset_entry_{assets.galaxy_id}_v0.h5",
    }
    if paths["entry"].exists() and not overwrite:
        return PipelineOutputs(
            assets.galaxy_id,
            outdir,
            paths["features"],
            paths["initial"],
            paths["with_bar"],
            paths["final"],
            paths["projected"],
            paths["mask"],
            paths["qa"],
            paths["entry"],
            0.0,
        )

    ssp_grid = load_ssp_grid(assets.ssp_template_path)
    targets = load_morphology_targets(assets.morphology_catalog_path, assets.snapshot, assets.subhalo_id)
    circularity = load_stellar_circularity_summary(assets.stellar_circularities_path, assets.snapshot, assets.subhalo_id)
    cube_geometry = read_cube_geometry(assets.cube_path)

    features = extract_particle_features(assets, config.extractor, ssp_grid)
    write_particle_features(paths["features"], features)

    initial = classify_primary_components(features, targets, config.classifier)
    write_particle_labels(paths["initial"], initial, "phase_a.classifier")
    with_bar = add_bar_component(features, initial, targets, config.bar_detector)
    write_particle_labels(paths["with_bar"], with_bar, "phase_a.bar_detector")
    final = add_arm_component(features, with_bar, targets, config.arm_detector)
    write_particle_labels(paths["final"], final, "phase_a.arm_detector")

    projected = project_labels(assets, features, final, cube_geometry, config.projection)
    write_projected_labels(paths["projected"], projected)
    valid_mask = build_valid_mask(assets.galaxy_id, projected, assets.cube_path, cube_geometry, config.mask)
    write_valid_mask(paths["mask"], valid_mask)
    qa = build_quality_report(features, final, projected, valid_mask, targets, circularity)
    write_quality_report(paths["qa"], qa)
    write_dataset_entry(paths["entry"], assets, projected, valid_mask, qa, config.packer)
    output = PipelineOutputs(
        assets.galaxy_id,
        outdir,
        paths["features"],
        paths["initial"],
        paths["with_bar"],
        paths["final"],
        paths["projected"],
        paths["mask"],
        paths["qa"],
        paths["entry"],
        float(time.monotonic() - start),
    )
    (outdir / "run_summary.json").write_text(json.dumps(output.as_dict(), indent=2, sort_keys=True))
    return output


def run_selected(config: PipelineConfig, galaxy_ids: list[str], all_local: bool, overwrite: bool = False) -> list[PipelineOutputs]:
    if all_local:
        assets_list = discover_local_assets(config.data)
    else:
        if not galaxy_ids:
            raise ValueError("Provide --galaxy-id or --all-local")
        assets_list = [assets_for_galaxy(galaxy_id, config.data) for galaxy_id in galaxy_ids]
    outputs = []
    for assets in assets_list:
        outputs.append(run_one_galaxy(assets, config, overwrite=overwrite))
    return outputs
