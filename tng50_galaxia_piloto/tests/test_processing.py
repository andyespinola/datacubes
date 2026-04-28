from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pilot_viewer.processing import (
    build_rotation_matrix,
    histogram_surface_density,
    radial_weighted_mean_profile,
    rotate_positions,
)


class ProcessingTests(unittest.TestCase):
    def test_rotation_matrix_aligns_disk_angular_momentum(self):
        theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        positions = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
        velocities = np.column_stack([-np.sin(theta), np.cos(theta), np.zeros_like(theta)])
        masses = np.ones(theta.shape[0], dtype=np.float64)

        rotation = build_rotation_matrix(positions, velocities, masses)
        rotated = rotate_positions(positions, rotation)

        self.assertEqual(rotation.shape, (3, 3))
        self.assertLess(np.max(np.abs(rotated[:, 2])), 1.0e-5)

    def test_surface_density_map_has_finite_signal(self):
        x = np.array([0.0, 0.2, -0.2, 0.0], dtype=np.float64)
        y = np.array([0.0, 0.2, -0.2, 0.0], dtype=np.float64)
        weights = np.array([2.0, 1.0, 1.0, 3.0], dtype=np.float64)
        image = histogram_surface_density(x, y, weights, radius_kpc=1.0, bins=16)

        self.assertEqual(image.shape, (16, 16))
        self.assertTrue(np.isfinite(image).any())

    def test_radial_weighted_mean_profile_returns_expected_bins(self):
        radius = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64)
        value = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

        x, y = radial_weighted_mean_profile(radius, value, weights, max_radius=1.0, bins=5)
        self.assertEqual(len(x), 5)
        self.assertEqual(len(y), 5)
        self.assertTrue(np.isfinite(y[:4]).any())


if __name__ == "__main__":
    unittest.main()
