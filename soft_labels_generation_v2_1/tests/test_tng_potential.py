from __future__ import annotations

import numpy as np
import h5py
import pytest

from aperturenet_labels.io.tng_potential import (
    TNGPotentialError,
    chunk_ranges_for_global_range,
    download_stream_atomic,
    load_stellar_potential_cache,
    match_potential_to_local_ids,
)


def test_chunk_ranges_for_global_range_spans_snapshot_files() -> None:
    offsets = np.asarray([0, 10, 25, 40], dtype=np.int64)
    ranges = chunk_ranges_for_global_range(offsets, global_start=8, count=20)

    assert [(item.file_number, item.local_start, item.count, item.global_start) for item in ranges] == [
        (0, 8, 2, 8),
        (1, 0, 15, 10),
        (2, 0, 3, 25),
    ]


def test_match_potential_to_local_ids_reorders_snapshot_values() -> None:
    local_ids = np.asarray([44, 11, 33], dtype=np.uint64)
    snapshot_ids = np.asarray([11, 33, 44], dtype=np.uint64)
    potential = np.asarray([-1.1, -3.3, -4.4], dtype=np.float32)

    ordered = match_potential_to_local_ids(local_ids, snapshot_ids, potential)

    assert np.allclose(ordered, [-4.4, -1.1, -3.3])


def test_load_stellar_potential_cache_selects_and_validates_ids(tmp_path) -> None:
    cache_dir = tmp_path / "potential_cache"
    cache_dir.mkdir()
    path = cache_dir / "TNG50-87-1.stellar_potential.hdf5"
    with h5py.File(path, "w") as handle:
        handle.attrs["potential_units_raw"] = "(km/s)^2 / a"
        handle.create_dataset("ParticleIDs", data=np.asarray([10, 11, 12, 13], dtype=np.uint64))
        handle.create_dataset("Potential", data=np.asarray([-10.0, -11.0, -12.0, -13.0], dtype=np.float32))

    cache = load_stellar_potential_cache(cache_dir, "TNG50-87-1", np.asarray([3, 1]), np.asarray([13, 11], dtype=np.uint64))

    assert cache is not None
    assert cache.path == path
    assert np.array_equal(cache.particle_ids, [13, 11])
    assert np.allclose(cache.potential_raw, [-13.0, -11.0])
    with pytest.raises(TNGPotentialError):
        load_stellar_potential_cache(cache_dir, "TNG50-87-1", np.asarray([3, 1]), np.asarray([13, 12], dtype=np.uint64))


def test_download_stream_atomic_keeps_incomplete_partials(tmp_path, monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        headers = {"content-length": "10"}
        text = ""

        def iter_content(self, chunk_size):
            _ = chunk_size
            yield b"12345"

        def close(self):
            return None

    monkeypatch.setattr("aperturenet_labels.io.tng_potential.requests.get", lambda *args, **kwargs: FakeResponse())
    output_path = tmp_path / "offsets.87.hdf5"

    with pytest.raises(TNGPotentialError, match="incomplete download"):
        download_stream_atomic(
            "https://example.invalid/offsets.87.hdf5",
            output_path,
            api_key="dummy",
            retries=1,
            resume=False,
        )

    assert not output_path.exists()
    assert output_path.with_suffix(".hdf5.part").exists()
