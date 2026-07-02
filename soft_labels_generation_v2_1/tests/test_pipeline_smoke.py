from __future__ import annotations

import h5py
import json
import numpy as np

from aperturenet_labels.config import PipelineConfig
from aperturenet_labels.io.assets import assets_for_galaxy
from aperturenet_labels.pipeline import run_one_galaxy


def test_lightweight_pipeline_writes_dataset_entry(tmp_path) -> None:
    config = PipelineConfig.from_yaml()
    config.data.output_dir = tmp_path
    config.extractor.max_particles = 5_000
    config.mask.min_particles_per_spaxel = 1
    config.packer.include_cube = False
    config.packer.include_pipe3d = True
    assets = assets_for_galaxy("TNG50-87-155298", config.data)

    outputs = run_one_galaxy(assets, config, overwrite=True)

    assert outputs.dataset_entry.exists()
    assert outputs.particle_features.exists()
    assert outputs.particle_labels_final.exists()
    assert outputs.projected_labels.exists()
    assert outputs.valid_mask.exists()
    assert outputs.qa_report.exists()

    with h5py.File(outputs.dataset_entry, "r") as handle:
        for group in ("metadata", "inputs", "labels", "masks", "qa"):
            assert group in handle
        assert handle["labels/Y_mass_psf"].shape == (74, 74, 5)
        assert handle["labels/Y_light_psf"].shape == (74, 74, 5)
        assert handle["masks/M_valid"].shape == (74, 74)
        cube = handle["inputs/cube_ifu"]
        assert cube.attrs["skipped"]
        labels = handle["labels/Y_mass_psf"][:]
        sums = labels.sum(axis=2)
        positive = sums > 0
        assert np.all(np.isfinite(labels))
        assert np.allclose(sums[positive], 1.0, atol=1.0e-5)
        qa = json.loads(handle["qa"].attrs["json"])
        assert "circularity_catalog" in qa
        assert "CircAbove07Frac" in qa["circularity_catalog"]
        assert qa["phase_a_diagnostics"]["potential_status"] in {"missing", "loaded", "invalid", "disabled"}

    projected = np.load(outputs.projected_labels, allow_pickle=True)
    metadata = json.loads(str(projected["metadata_json"]))
    assert metadata["sky_alignment_enabled"]
    assert metadata["sky_alignment_reference"] == "stellar_mass_density_log10"
    assert np.isfinite(float(metadata["sky_alignment_score"]))
