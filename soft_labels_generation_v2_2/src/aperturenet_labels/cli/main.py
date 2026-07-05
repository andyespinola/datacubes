"""CLI del pipeline v2: aperturenet-labels {run,phase-a,phase-b} [--pilot]."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import structlog
import typer

from ..core.geometry import view_vector_from_index
from ..io import catalogs, manifest as manifest_mod, mangia_reader, ssp_grid, tng_reader, units
from ..phase_a import arm_detector, bar_detector, classifier, extractor
from ..phase_b import label_projection, mask_builder
from ..phase_c import packer, quality_check
from ..schemas.models import ManifestRow

log = structlog.get_logger(__name__)
app = typer.Typer(help="Pipeline v2 de etiquetas estructurales ApertureNet-S3")

_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # .../datacubes
DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"
DEFAULT_SSP = _PROJECT_ROOT / "kinematic_moments/templates/MaStar_CB19.slog_1_5.fits.gz"


def _paths(data_dir: Path, row: ManifestRow) -> dict[str, Path]:
    gal = manifest_mod.galaxy_id(row.snapshot, row.subhalo_id)
    inter = data_dir / "intermediate"
    out = data_dir / "output"
    pa = inter / "phase_a" / gal
    pb = inter / "phase_b" / gal
    return {
        "features": pa / "particle_features.h5",
        "labels_initial": pa / "particle_labels_initial.h5",
        "labels_with_bar": pa / "particle_labels_with_bar.h5",
        "labels_final": pa / "particle_labels_final.h5",
        "projection": pb / f"labels2d_v{row.view}.npz",
        "mask": pb / f"M_valid_v{row.view}.npz",
        "qa": out / "qa_reports" / f"{gal}_v{row.view}.json",
        "entry": out / "dataset_entries" / f"{gal}_v{row.view}.h5",
    }


def run_phase_a(row: ManifestRow, data_dir: Path, ssp_path: Path, force: bool = False) -> dict:
    paths = _paths(data_dir, row)
    gal = manifest_mod.galaxy_id(row.snapshot, row.subhalo_id)
    grid = ssp_grid.load_ssp_grid(ssp_path)

    if force or not paths["features"].exists():
        truth = tng_reader.load_cutout_truth(
            row.cutout_path, row.subhalo_json_path, row.cutout_phase2_path or None
        )
        truth = units.convert_truth_units(truth)
        extractor.run_extractor(truth, gal, paths["features"], grid)
    feats = extractor.load_particle_features(paths["features"])

    mordor = catalogs.load_morphology_targets(
        data_dir / "morphs_kinematic_bars.hdf5", row.snapshot, row.subhalo_id
    )
    priors = catalogs.priors_from_mordor(mordor)
    bar_meta = catalogs.bar_meta_from_mordor(mordor)

    if force or not paths["labels_initial"].exists():
        classifier.run_classifier(feats, paths["labels_initial"], priors)
    initial = classifier.load_labels(paths["labels_initial"])

    if force or not paths["labels_with_bar"].exists():
        bar_detector.run_bar_detector(feats, initial, bar_meta, paths["labels_with_bar"])
    with_bar = classifier.load_labels(paths["labels_with_bar"])

    if force or not paths["labels_final"].exists():
        arm_detector.run_arm_detector(feats, with_bar, bar_meta, paths["labels_final"])
    final = classifier.load_labels(paths["labels_final"])
    return {"feats": feats, "final": final, "priors": priors, "bar_meta": bar_meta, "paths": paths}


def run_phase_b(row: ManifestRow, data_dir: Path, ctx: dict, force: bool = False) -> dict:
    paths = ctx["paths"]
    feats = ctx["feats"]
    gal = manifest_mod.galaxy_id(row.snapshot, row.subhalo_id)
    view = mangia_reader.view_definition_from_cube(row.cube_path, row.view, row.repeat_count)

    if force or not paths["projection"].exists():
        label_projection.run_label_projection(
            positions_centered=feats["pos_centered"].astype(np.float64),
            mass=feats["mass"].astype(np.float64),
            light=feats["light_g"].astype(np.float64),
            p_class=ctx["final"]["P_class"].astype(np.float64),
            view=view,
            galaxy_id=gal,
            output_path=paths["projection"],
            r_eff_kpc=float(feats["R_eff_kpc"]),
        )
    projection = label_projection.load_projection(paths["projection"])

    if force or not paths["mask"].exists():
        mask_builder.run_mask_builder(
            projection["n_particles_map"], row.cube_path, gal, row.view, paths["mask"]
        )
    mask = mask_builder.load_mask(paths["mask"])
    return {"view": view, "projection": projection, "mask": mask}


def run_phase_c(row: ManifestRow, data_dir: Path, ctx_a: dict, ctx_b: dict,
                force: bool = False, copy_cube: bool = True) -> Path:
    paths = ctx_a["paths"]
    gal = manifest_mod.galaxy_id(row.snapshot, row.subhalo_id)
    priors = ctx_a["priors"]
    catalog_fractions = {
        "bulge": priors.bulge_frac,
        "disk": priors.disk_frac,
        "other": priors.other_frac,
    }
    final = dict(ctx_a["final"])
    # reconstruir bar_diagnostics desde full_pipeline_diagnostics si hace falta
    if "bar_diagnostics" not in final and "full_pipeline_diagnostics" in final:
        fp = final["full_pipeline_diagnostics"]
        final["bar_diagnostics"] = {
            k.split(".", 1)[1]: v for k, v in fp.items() if k.startswith("bar_diagnostics.")
        }
    quality_check.run_quality_check(
        galaxy_id=gal,
        view_id=row.view,
        feats=ctx_a["feats"],
        final_labels=final,
        projection=ctx_b["projection"],
        mask=ctx_b["mask"],
        catalog_fractions=catalog_fractions,
        output_path=paths["qa"],
    )
    return packer.run_packer(
        galaxy_id=gal,
        view_id=row.view,
        snapshot=row.snapshot,
        subhalo_id=row.subhalo_id,
        view_vector=tuple(view_vector_from_index(row.view, row.repeat_count)),
        cube_path=row.cube_path,
        pipe3d_maps_path=row.pipe3d_maps_path or None,
        projection=ctx_b["projection"],
        mask=ctx_b["mask"],
        qa_report_path=paths["qa"],
        output_path=paths["entry"],
        config=packer.PackerConfig(copy_cube=copy_cube),
    )


@app.command()
def run(
    pilot: bool = typer.Option(False, "--pilot", help="Correr sobre las galaxias piloto"),
    galaxy_id: Optional[str] = typer.Option(None, help="ej. TNG50-87-155298"),
    view: int = typer.Option(0),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR),
    ssp_path: Path = typer.Option(DEFAULT_SSP),
    force: bool = typer.Option(False, help="Recomputar aunque existan productos"),
) -> None:
    """Pipeline completo (Fases A+B+C)."""
    rows = (
        manifest_mod.pilot_manifest(data_dir)
        if pilot
        else [
            r
            for r in manifest_mod.build_manifest_from_dir(data_dir)
            if galaxy_id and manifest_mod.galaxy_id(r.snapshot, r.subhalo_id) == galaxy_id and r.view == view
        ]
    )
    if not rows:
        raise typer.BadParameter("No hay galaxias que procesar (usa --pilot o --galaxy-id)")
    for row in rows:
        t0 = time.time()
        gal = manifest_mod.galaxy_id(row.snapshot, row.subhalo_id)
        log.info("pipeline.start", galaxy=gal, view=row.view)
        ctx_a = run_phase_a(row, data_dir, ssp_path, force)
        ctx_b = run_phase_b(row, data_dir, ctx_a, force)
        entry = run_phase_c(row, data_dir, ctx_a, ctx_b, force)
        report = packer.validate_dataset_entry(entry)
        log.info("pipeline.done", **report, total_sec=round(time.time() - t0, 1))


@app.command("phase-a")
def phase_a_cmd(
    galaxy_id: str = typer.Option(...),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR),
    ssp_path: Path = typer.Option(DEFAULT_SSP),
    force: bool = typer.Option(False),
) -> None:
    rows = [
        r
        for r in manifest_mod.build_manifest_from_dir(data_dir)
        if manifest_mod.galaxy_id(r.snapshot, r.subhalo_id) == galaxy_id
    ]
    if not rows:
        raise typer.BadParameter(f"{galaxy_id} no encontrado en {data_dir}")
    run_phase_a(rows[0], data_dir, ssp_path, force)


@app.command("phase-b")
def phase_b_cmd(
    galaxy_id: str = typer.Option(...),
    view: int = typer.Option(0),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR),
    ssp_path: Path = typer.Option(DEFAULT_SSP),
    force: bool = typer.Option(False),
) -> None:
    rows = [
        r
        for r in manifest_mod.build_manifest_from_dir(data_dir)
        if manifest_mod.galaxy_id(r.snapshot, r.subhalo_id) == galaxy_id and r.view == view
    ]
    if not rows:
        raise typer.BadParameter(f"{galaxy_id} vista {view} no encontrado")
    ctx_a = run_phase_a(rows[0], data_dir, ssp_path, force=False)
    run_phase_b(rows[0], data_dir, ctx_a, force=True)


if __name__ == "__main__":
    app()
