"""Packer (spec 30): HDF5 final por galaxia × orientación para el dataloader.

Padding 69→74 centrado (sección 8 de NOTA_DIRECTOR_MANGIA_MANGA_SEGMENTACION):
se aplica por igual a cubo, mapas pyPipe3D, etiquetas y máscara para mantener
el registro pixel-a-pixel.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np
import structlog
from pydantic import BaseModel

from ..core.constants import CLASS_NAMES
from ..io.mangia_reader import load_cube_flux, load_pipe3d_maps

log = structlog.get_logger(__name__)

PACKED_PIPE3D = {
    "v_star": "km/s",
    "sigma_star": "km/s",
    "age_lw": "log(Gyr)",
    "metallicity_lw": "dex",
    "mass_density": "log(M_sun/spaxel)",
    "av": "mag",
}


class PackerConfig(BaseModel):
    include_qa: bool = True
    include_pipe3d: bool = True
    compression: str = "lzf"
    target_size: int = 74
    pad_value: float = 0.0
    copy_cube: bool = True   # False = entry SOLO etiquetas (~0.6 MB); el cubo
    #                          queda como referencia (metadata.cube_file). Para
    #                          10k: ~6 GB en vez de ~850 GB. El cubo es un
    #                          producto independiente (no se transporta con las
    #                          etiquetas).


def pad_to(arr: np.ndarray, target: int, pad_value: float = 0.0) -> np.ndarray:
    """Padding centrado de los DOS últimos ejes espaciales a target×target."""
    h, w = arr.shape[-2], arr.shape[-1]
    if h == target and w == target:
        return arr
    if h > target or w > target:
        raise ValueError(f"shape {arr.shape} mayor que target {target}")
    pad_h = target - h
    pad_w = target - w
    top, left = pad_h // 2, pad_w // 2
    pads = [(0, 0)] * (arr.ndim - 2) + [(top, pad_h - top), (left, pad_w - left)]
    return np.pad(arr, pads, mode="constant", constant_values=pad_value)


def run_packer(
    galaxy_id: str,
    view_id: int,
    snapshot: int,
    subhalo_id: int,
    view_vector: tuple[float, float, float],
    cube_path: str | Path,
    pipe3d_maps_path: str | Path | None,
    projection: dict,
    mask: dict,
    qa_report_path: str | Path | None,
    output_path: str | Path,
    config: PackerConfig | None = None,
) -> Path:
    config = config or PackerConfig()
    t0 = time.time()
    tgt = config.target_size

    if config.copy_cube:
        flux, error, wave = load_cube_flux(cube_path)
        cube74 = pad_to(flux, tgt)
        native_hw = tuple(flux.shape[1:])
    else:
        # producto solo-etiquetas: no se carga el cubo (grid nativo desde la
        # máscara, que ya está a resolución nativa antes del padding)
        cube74 = wave = None
        native_hw = tuple(mask["M_valid"].shape)
    y_mass = pad_to(projection["Y_mass_raw"], tgt)
    y_mass_psf = pad_to(projection["Y_mass_psf"], tgt)
    y_lum = pad_to(projection["Y_lum_raw"], tgt)
    y_lum_psf = pad_to(projection["Y_lum_psf"], tgt)
    m_valid = pad_to(mask["M_valid"].astype(np.uint8), tgt).astype(bool)
    n_eff = pad_to(projection["n_eff"], tgt)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["schema_version"] = "1.0"
        f.attrs["pipeline_version"] = "v2"
        meta = f.create_group("metadata")
        meta.attrs["galaxy_id"] = galaxy_id
        meta.attrs["view_id"] = view_id
        meta.attrs["snapshot"] = snapshot
        meta.attrs["subhalo_id"] = subhalo_id
        meta.attrs["view_vector"] = np.asarray(view_vector, dtype=np.float64)
        meta.attrs["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        meta.attrs["original_grid"] = np.asarray(native_hw, dtype=np.int64)
        meta.attrs["padded_grid"] = np.asarray([tgt, tgt], dtype=np.int64)
        # referencia al cubo (producto independiente); el consumidor lo carga
        # por su cuenta desde el directorio de cubos usando este nombre.
        meta.attrs["cube_file"] = Path(cube_path).name
        meta.attrs["cube_embedded"] = bool(config.copy_cube)

        inputs = f.create_group("inputs")
        if config.copy_cube:
            d = inputs.create_dataset(
                "cube_ifu", data=cube74.astype(np.float32),
                compression=config.compression
            )
            d.attrs["units"] = "1e-16 erg/s/cm^2/Å"
            inputs.create_dataset("wavelength", data=wave.astype(np.float32))
        if config.include_pipe3d and pipe3d_maps_path:
            p3d = load_pipe3d_maps(pipe3d_maps_path)
            g = inputs.create_group("pipe3d_maps")
            for name, units_ in PACKED_PIPE3D.items():
                d = g.create_dataset(
                    name, data=pad_to(p3d[name], tgt).astype(np.float32),
                    compression=config.compression,
                )
                d.attrs["units"] = units_

        labels = f.create_group("labels")
        for name, arr in [
            ("Y_int_mass", y_mass),
            ("Y_int_light", y_lum),
            ("Y_int_mass_psf", y_mass_psf),
            ("Y_int_light_psf", y_lum_psf),
        ]:
            labels.create_dataset(
                name,
                data=np.moveaxis(arr, 0, -1).astype(np.float32),  # (H, W, 5)
                compression=config.compression,
            )
        labels.create_dataset("class_names", data=np.array(CLASS_NAMES, dtype="S"))
        labels.create_dataset("n_eff", data=n_eff.astype(np.float32), compression=config.compression)

        masks = f.create_group("masks")
        masks.create_dataset("M_valid", data=m_valid)

        if config.include_qa and qa_report_path and Path(qa_report_path).exists():
            qa = f.create_group("qa")
            report = json.loads(Path(qa_report_path).read_text())
            for k, v in report.items():
                if isinstance(v, (str, int, float, bool)):
                    qa.attrs[k] = v
                else:
                    qa.attrs[k] = json.dumps(v, default=str)

    size_mb = output_path.stat().st_size / 1e6
    log.info(
        "packer.done",
        galaxy_id=galaxy_id,
        view_id=view_id,
        size_mb=round(size_mb, 1),
        t=round(time.time() - t0, 1),
    )
    if size_mb > 100:
        log.warning("packer.tamano_excesivo", size_mb=round(size_mb, 1))
    return output_path


def validate_dataset_entry(path: str | Path) -> dict:
    """Roundtrip de validación: estructura, shapes, dtypes, sin NaN en labels."""
    with h5py.File(path, "r") as f:
        assert f.attrs["pipeline_version"] == "v2"
        y_mass = f["labels/Y_int_mass"][:]
        y_light = f["labels/Y_int_light"][:]
        m_valid = f["masks/M_valid"][:]
        tgt = f["metadata"].attrs["padded_grid"]
        if "inputs/cube_ifu" in f:   # solo si el cubo va embebido (copy_cube)
            cube = f["inputs/cube_ifu"]
            assert cube.shape[1] == tgt[0] and cube.shape[2] == tgt[1], cube.shape
        assert y_mass.shape == (tgt[0], tgt[1], len(CLASS_NAMES)), y_mass.shape
        assert not np.isnan(y_mass).any() and not np.isnan(y_light).any()
        sums = y_mass.sum(axis=-1)
        ok = m_valid & (sums > 0)
        assert np.allclose(sums[ok], 1.0, atol=1e-3), "Y_int_mass no suma 1 en spaxels válidos"
        out = {
            "galaxy_id": str(f["metadata"].attrs["galaxy_id"]),
            "view_id": int(f["metadata"].attrs["view_id"]),
            "qa_status": str(f["qa"].attrs.get("status", "n/a")) if "qa" in f else "n/a",
            "n_valid": int(m_valid.sum()),
        }
    return out
