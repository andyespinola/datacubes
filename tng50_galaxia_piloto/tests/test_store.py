from __future__ import annotations

import json
import tempfile
import unittest
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pilot_viewer.store import GalaxyStore


class StoreTests(unittest.TestCase):
    def test_store_loads_small_cutout_and_morphology(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cutout_path = root / "pilot.cutout.hdf5"
            metadata_path = root / "pilot.subhalo.json"
            morph_path = root / "morph.hdf5"

            with h5py.File(cutout_path, "w") as handle:
                stars = handle.create_group("PartType4")
                stars.create_dataset(
                    "Coordinates",
                    data=np.array(
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [-1.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0],
                            [0.3, 0.3, 0.0],
                            [0.4, -0.1, 0.0],
                            [0.1, 0.4, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        dtype=np.float64,
                    ),
                )
                stars.create_dataset(
                    "Velocities",
                    data=np.array(
                        [
                            [0.0, 1.0, 0.0],
                            [-1.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [-0.3, 0.3, 0.0],
                            [0.1, 0.4, 0.0],
                            [-0.4, 0.1, 0.0],
                            [0.0, 0.0, 0.0],
                        ],
                        dtype=np.float32,
                    ),
                )
                stars.create_dataset("Masses", data=np.full(8, 1.0e-3, dtype=np.float32))
                stars.create_dataset(
                    "GFM_StellarFormationTime",
                    data=np.array([0.5, 0.55, 0.6, 0.7, 0.5, 0.8, 0.75, -1.0], dtype=np.float32),
                )
                stars.create_dataset("GFM_Metallicity", data=np.linspace(0.01, 0.08, 8, dtype=np.float32))

                gas = handle.create_group("PartType0")
                gas.create_dataset(
                    "Coordinates",
                    data=np.array([[0.2, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=np.float64),
                )
                gas.create_dataset("Velocities", data=np.zeros((2, 3), dtype=np.float32))
                gas.create_dataset("Masses", data=np.array([1.0e-3, 2.0e-3], dtype=np.float32))
                gas.create_dataset("StarFormationRate", data=np.array([0.2, 0.1], dtype=np.float32))
                gas.create_dataset("Density", data=np.array([1.0, 2.0], dtype=np.float32))
                gas.create_dataset("InternalEnergy", data=np.array([1.0, 1.0], dtype=np.float32))
                gas.create_dataset("ElectronAbundance", data=np.array([0.5, 0.5], dtype=np.float32))
                gas.create_dataset("GFM_Metallicity", data=np.array([0.02, 0.03], dtype=np.float32))

            metadata_path.write_text(
                json.dumps(
                    {
                        "id": 141934,
                        "snap": 87,
                        "pos_x": 0.0,
                        "pos_y": 0.0,
                        "pos_z": 0.0,
                        "vel_x": 0.0,
                        "vel_y": 0.0,
                        "vel_z": 0.0,
                        "halfmassrad_stars": 5.0,
                        "mass_log_msun": 10.8,
                        "cutouts": {"subhalo": "https://example.invalid/cutout"},
                        "related": {"url": "https://example.invalid/subhalo"},
                    }
                )
            )

            with h5py.File(morph_path, "w") as handle:
                group = handle.create_group("Snapshot_87")
                group.create_dataset("SubhaloID", data=np.array([141934], dtype=np.int64))
                for name, value in {
                    "ThinDisc": 0.5,
                    "ThickDisc": 0.1,
                    "PseudoBulge": 0.05,
                    "Bulge": 0.2,
                    "Halo": 0.15,
                }.items():
                    group.create_dataset(name, data=np.array([[value]], dtype=np.float32))
                group.create_dataset("UnboundMass", data=np.array([0.0], dtype=np.float32))
                group.create_dataset("Barred", data=np.array([1.0], dtype=np.float32))
                group.create_dataset("BarSize", data=np.array([[3.2]], dtype=np.float32))
                group.create_dataset("BarStrength", data=np.array([[0.22]], dtype=np.float32))

            store = GalaxyStore.from_files(cutout_path, metadata_path, morph_path)
            summary = store.summary

            self.assertEqual(summary["snapshot"], 87)
            self.assertEqual(summary["subhalo_id"], 141934)
            self.assertEqual(summary["n_stellar_particles"], 7)
            self.assertEqual(summary["n_gas_cells"], 2)
            self.assertTrue(summary["morphology"]["barred"])

            map_payload = store.get_map("stars", "mass", "faceon", radius_kpc=10.0, bins=64)
            profile_payload = store.get_profile("gas", "sfr", radius_kpc=10.0, bins=20)
            cloud_payload = store.get_particle_cloud(
                radius_kpc=10.0,
                max_stars=10,
                max_gas=10,
                star_quantity="metallicity",
                gas_quantity="metallicity",
            )
            star_selection = store.get_particle_selection("stars", radius_kpc=10.0, max_points=10, quantity="metallicity")
            gas_selection = store.get_particle_selection("gas", radius_kpc=10.0, max_points=10, quantity="metallicity")

            self.assertEqual(map_payload["component"], "stars")
            self.assertEqual(profile_payload["component"], "gas")
            self.assertTrue(len(profile_payload["x"]) == 20)
            self.assertIn("stars", cloud_payload)
            self.assertLessEqual(cloud_payload["stars"]["sampled_count"], 10)
            self.assertLessEqual(cloud_payload["gas"]["sampled_count"], 10)
            self.assertEqual(star_selection.packed.shape[1], 4)
            self.assertEqual(gas_selection.packed.shape[1], 4)
            self.assertEqual(cloud_payload["stars"]["quantity"], "metallicity")
            self.assertEqual(star_selection.quantity, "metallicity")
            self.assertEqual(star_selection.sampled_count, cloud_payload["stars"]["sampled_count"])
            self.assertEqual(gas_selection.sampled_count, cloud_payload["gas"]["sampled_count"])


if __name__ == "__main__":
    unittest.main()
