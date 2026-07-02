from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from astropy.io import fits

from aperturenet_labels.config import DataConfig


GALAXY_RE = re.compile(r"^TNG50-(?P<snapshot>\d+)-(?P<subhalo_id>\d+)$")


@dataclass(slots=True)
class LocalGalaxyAssets:
    galaxy_id: str
    snapshot: int
    subhalo_id: int
    view: int
    file_ifu_design: int
    catalog_ifu_design: int
    re_kpc: float
    sample_manga: int
    n_star_part_catalog: int
    n_gas_cell_catalog: int
    cutout_path: Path
    metadata_path: Path
    cube_path: Path
    cube_val_path: Path
    maps_path: Path
    morphology_catalog_path: Path
    stellar_circularities_path: Path
    potential_cache_dir: Path
    ssp_template_path: Path


def parse_galaxy_id(galaxy_id: str) -> tuple[int, int]:
    match = GALAXY_RE.match(galaxy_id)
    if not match:
        raise ValueError(f"Invalid galaxy_id={galaxy_id!r}; expected TNG50-<snapshot>-<subhalo_id>")
    return int(match.group("snapshot")), int(match.group("subhalo_id"))


def _catalog_row(galaxy_id: str) -> dict[str, int | float]:
    snapshot, subhalo_id = parse_galaxy_id(galaxy_id)
    catalog_path = Path(__file__).resolve().parents[4] / "MaNGIA_catalog.fits"
    defaults = {
        "catalog_ifu_design": 0,
        "re_kpc": 0.0,
        "sample_manga": 0,
        "n_star_part_catalog": 0,
        "n_gas_cell_catalog": 0,
    }
    if not catalog_path.exists():
        return defaults
    data = fits.getdata(catalog_path, 1)
    mask = (data["snapshot"] == snapshot) & (data["subhalo_id"] == subhalo_id)
    if not mask.any():
        return defaults
    row = data[mask][0]
    return {
        "catalog_ifu_design": int(row["manga_ifu_dsn"]),
        "re_kpc": float(row["re_kpc"]),
        "sample_manga": int(row["sample_manga"]),
        "n_star_part_catalog": int(row["n_star_part"]),
        "n_gas_cell_catalog": int(row["n_gas_cell"]),
    }


def assets_for_galaxy(galaxy_id: str, data_config: DataConfig) -> LocalGalaxyAssets:
    snapshot, subhalo_id = parse_galaxy_id(galaxy_id)
    stem = f"{galaxy_id}-0-{data_config.file_ifu_design}"
    catalog = _catalog_row(galaxy_id)
    return LocalGalaxyAssets(
        galaxy_id=galaxy_id,
        snapshot=snapshot,
        subhalo_id=subhalo_id,
        view=0,
        file_ifu_design=data_config.file_ifu_design,
        catalog_ifu_design=int(catalog["catalog_ifu_design"]),
        re_kpc=float(catalog["re_kpc"]),
        sample_manga=int(catalog["sample_manga"]),
        n_star_part_catalog=int(catalog["n_star_part_catalog"]),
        n_gas_cell_catalog=int(catalog["n_gas_cell_catalog"]),
        cutout_path=data_config.data_dir / f"{galaxy_id}.cutout.hdf5",
        metadata_path=data_config.data_dir / f"{galaxy_id}.subhalo.json",
        cube_path=data_config.data_dir / f"{stem}.cube.fits.gz",
        cube_val_path=data_config.data_dir / f"{stem}.cube_val.fits.gz",
        maps_path=data_config.data_dir / f"{stem}.cube_maps.fits",
        morphology_catalog_path=data_config.morphology_catalog,
        stellar_circularities_path=data_config.stellar_circularities,
        potential_cache_dir=data_config.potential_cache_dir,
        ssp_template_path=data_config.ssp_template,
    )


def validate_assets(assets: LocalGalaxyAssets) -> list[str]:
    missing = []
    for path in (
        assets.cutout_path,
        assets.metadata_path,
        assets.cube_path,
        assets.cube_val_path,
        assets.maps_path,
        assets.morphology_catalog_path,
        assets.stellar_circularities_path,
        assets.ssp_template_path,
    ):
        if not path.exists():
            missing.append(str(path))
    return missing


def discover_local_assets(data_config: DataConfig) -> list[LocalGalaxyAssets]:
    return [assets_for_galaxy(galaxy_id, data_config) for galaxy_id in data_config.local_galaxy_ids]
