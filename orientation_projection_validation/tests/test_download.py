from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from orientation_validation.download import cutout_url, download_stream_atomic, validate_cutout


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, payload: bytes):
        self.payload = payload

    def iter_content(self, chunk_size: int):
        yield self.payload


class DownloadTest(unittest.TestCase):
    def test_cutout_url_includes_stars_and_gas(self) -> None:
        url = cutout_url(87, 141934, include_gas=True)
        self.assertIn("stars=Coordinates", url)
        self.assertIn("gas=Coordinates", url)

    def test_atomic_hdf5_download_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.hdf5"
            with h5py.File(source, "w") as handle:
                stars = handle.create_group("PartType4")
                gas = handle.create_group("PartType0")
                n = 2
                for group in (stars, gas):
                    group["Coordinates"] = np.zeros((n, 3))
                    group["Velocities"] = np.zeros((n, 3))
                    group["Masses"] = np.ones(n)
                    group["GFM_Metallicity"] = np.ones(n)
                stars["GFM_StellarFormationTime"] = np.ones(n)
                gas["StarFormationRate"] = np.ones(n)
                gas["Density"] = np.ones(n)
                gas["InternalEnergy"] = np.ones(n)
                gas["ElectronAbundance"] = np.ones(n)

            payload = source.read_bytes()
            target = Path(tmp) / "target.hdf5"
            with patch("orientation_validation.download.requests.get", return_value=FakeResponse(payload)):
                download_stream_atomic("https://example.invalid/file.hdf5", target, "token", force=True)
            self.assertTrue(target.exists())
            self.assertFalse((Path(tmp) / "target.hdf5.part").exists())
            validate_cutout(target, include_gas=True)


if __name__ == "__main__":
    unittest.main()

