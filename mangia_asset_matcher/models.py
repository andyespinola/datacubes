from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .ids import UnitKey


@dataclass(frozen=True, slots=True)
class CatalogUnit:
    key: UnitKey
    ifu_design: int
    re_kpc: float
    sample_manga: int
    n_star_part: int
    n_gas_cell: int
    estimated_raw_mb: float
    catalog_order: int


@dataclass(frozen=True, slots=True)
class CubeAsset:
    key: UnitKey
    path: Path
    ifu_design: int | None
    shape: str = ""


@dataclass(frozen=True, slots=True)
class MapAsset:
    key: UnitKey | None
    path: Path
    format: str
    v_map_key: str = ""
    sigma_map_key: str = ""
    shape: str = ""
    message: str = ""

    @property
    def has_v(self) -> bool:
        return bool(self.v_map_key)

    @property
    def has_sigma(self) -> bool:
        return bool(self.sigma_map_key)

    @property
    def is_usable(self) -> bool:
        return self.key is not None and self.has_v and self.has_sigma


@dataclass(frozen=True, slots=True)
class TngAssets:
    cutouts_by_unit: dict[UnitKey, Path]
    metadata_by_unit: dict[UnitKey, Path]
    cutouts_by_galaxy: dict[tuple[int, int], Path]
    metadata_by_galaxy: dict[tuple[int, int], Path]
    morphology_catalog_path: Path | None

    def cutout_for(self, key: UnitKey) -> Path | None:
        return self.cutouts_by_unit.get(key) or self.cutouts_by_galaxy.get((key.snapshot, key.subhalo_id))

    def metadata_for(self, key: UnitKey) -> Path | None:
        return self.metadata_by_unit.get(key) or self.metadata_by_galaxy.get((key.snapshot, key.subhalo_id))


@dataclass(frozen=True, slots=True)
class AssetScan:
    cubes: dict[UnitKey, CubeAsset]
    tng: TngAssets
    maps: dict[UnitKey, MapAsset]
    map_files_total: int
    map_files_id_unknown: int
    map_files_without_v: int
    map_files_without_sigma: int
