from __future__ import annotations

from aperturenet_labels.config import PipelineConfig
from aperturenet_labels.io.assets import discover_local_assets, validate_assets
from aperturenet_labels.io.circularity import load_stellar_circularity_summary
from aperturenet_labels.io.cube_reader import read_cube_geometry, read_pipe3d_maps
from aperturenet_labels.io.morphology import load_morphology_targets
from aperturenet_labels.io.tng_reader import validate_cutout


def test_local_data_assets_are_complete() -> None:
    config = PipelineConfig.from_yaml()
    assets_list = discover_local_assets(config.data)
    assert [asset.galaxy_id for asset in assets_list] == ["TNG50-87-155298", "TNG50-87-192324"]
    for assets in assets_list:
        assert validate_assets(assets) == []
        cutout = validate_cutout(assets.cutout_path)
        assert cutout["n_star_raw"] > 1_000_000
        assert cutout["n_gas_raw"] > 100_000
        assert cutout["missing_star"] == []
        assert cutout["missing_gas"] == []


def test_cube_maps_and_morphology_cover_both_galaxies() -> None:
    config = PipelineConfig.from_yaml()
    for assets in discover_local_assets(config.data):
        geometry = read_cube_geometry(assets.cube_path)
        assert geometry.shape == (69, 69)
        assert geometry.n_wave == 6603
        maps = read_pipe3d_maps(assets.maps_path)
        assert {"v_star", "sigma_star", "stellar_mass_density_log10"}.issubset(maps)
        assert maps["v_star"].shape == (69, 69)
        targets = load_morphology_targets(assets.morphology_catalog_path, assets.snapshot, assets.subhalo_id)
        assert targets.disk_fraction + targets.bulge_fraction + targets.halo_fraction > 0.99
        circularity = load_stellar_circularity_summary(assets.stellar_circularities_path, assets.snapshot, assets.subhalo_id)
        assert circularity is not None
        assert "CircAbove07Frac" in circularity
