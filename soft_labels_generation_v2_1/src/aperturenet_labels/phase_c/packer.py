from __future__ import annotations

from pathlib import Path
import json

import h5py
import numpy as np

from aperturenet_labels.config import PackerConfig
from aperturenet_labels.core.constants import CLASS_NAMES
from aperturenet_labels.core.geometry import pad_center
from aperturenet_labels.io.assets import LocalGalaxyAssets
from aperturenet_labels.io.cube_reader import read_cube_flux, read_pipe3d_maps
from aperturenet_labels.phase_b.label_projection import ProjectedLabels
from aperturenet_labels.phase_b.mask_builder import ValidMask
from aperturenet_labels.phase_c.quality_check import QualityReport


def _write_dataset(group: h5py.Group, name: str, data: np.ndarray, compression: str) -> None:
    kwargs = {"compression": compression} if data.size > 0 and compression else {}
    group.create_dataset(name, data=data, **kwargs)


def write_dataset_entry(
    path: str | Path,
    assets: LocalGalaxyAssets,
    projected: ProjectedLabels,
    valid_mask: ValidMask,
    qa: QualityReport,
    config: PackerConfig,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    target_shape = tuple(config.pad_to_shape)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "1.0-skeleton"
        handle.attrs["pipeline_version"] = "soft_labels_generation_v2_1"
        meta = handle.create_group("metadata")
        meta.attrs["galaxy_id"] = assets.galaxy_id
        meta.attrs["snapshot"] = int(assets.snapshot)
        meta.attrs["subhalo_id"] = int(assets.subhalo_id)
        meta.attrs["view"] = int(assets.view)
        meta.attrs["file_ifu_design"] = int(assets.file_ifu_design)
        meta.attrs["catalog_ifu_design"] = int(assets.catalog_ifu_design)
        meta.attrs["original_shape"] = json.dumps(list(projected.y_mass_psf.shape[:2]))
        meta.attrs["padded_shape"] = json.dumps(list(target_shape))

        inputs = handle.create_group("inputs")
        if config.include_cube:
            cube = pad_center(read_cube_flux(assets.cube_path).astype("f4"), target_shape, fill_value=0.0)
            _write_dataset(inputs, "cube_ifu", cube, config.compression)
        else:
            inputs.create_dataset("cube_ifu", data=np.zeros((0, 0, 0), dtype=np.float32))
            inputs["cube_ifu"].attrs["skipped"] = True

        if config.include_pipe3d:
            maps_group = inputs.create_group("pipe3d_maps")
            for name, value in read_pipe3d_maps(assets.maps_path).items():
                _write_dataset(maps_group, name, pad_center(value.astype("f4"), target_shape, fill_value=np.nan), config.compression)

        labels = handle.create_group("labels")
        for name, value in {
            "Y_mass_raw": projected.y_mass_raw,
            "Y_mass_psf": projected.y_mass_psf,
            "Y_light_raw": projected.y_light_raw,
            "Y_light_psf": projected.y_light_psf,
            "raw_mass_per_class": projected.raw_mass_per_class,
            "raw_light_per_class": projected.raw_light_per_class,
        }.items():
            padded = pad_center(np.moveaxis(value, 2, 0), target_shape, fill_value=0.0)
            padded = np.moveaxis(padded, 0, 2)
            _write_dataset(labels, name, padded.astype("f4"), config.compression)
        labels.create_dataset("class_names", data=np.asarray(CLASS_NAMES, dtype=object), dtype=string_dtype)

        masks = handle.create_group("masks")
        _write_dataset(masks, "M_valid", pad_center(valid_mask.m_valid.astype(np.uint8), target_shape, fill_value=0).astype(bool), config.compression)

        qa_group = handle.create_group("qa")
        qa_group.attrs["json"] = json.dumps(qa.payload, sort_keys=True)
        for key, value in qa.payload.items():
            if isinstance(value, (str, int, float, bool)):
                qa_group.attrs[key] = value

    with h5py.File(path, "r") as handle:
        for required in ("metadata", "inputs", "labels", "masks", "qa"):
            if required not in handle:
                raise RuntimeError(f"Missing group {required} after writing {path}")
    return path
