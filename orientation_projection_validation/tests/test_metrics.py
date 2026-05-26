from __future__ import annotations

import unittest

import numpy as np

from orientation_validation.config import ProjectionConfig
from orientation_validation.metrics import compute_interorientation_metrics, probabilistic_iou


class MetricsTest(unittest.TestCase):
    def test_probabilistic_iou(self) -> None:
        a = np.array([[0.0, 0.5], [1.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0], [0.5, 0.0]], dtype=np.float32)
        self.assertAlmostEqual(probabilistic_iou(a, b), 1.0 / 2.0)

    def test_compute_metrics_on_symmetric_maps(self) -> None:
        size = 25
        yy, xx = np.indices((size, size))
        rr = np.sqrt((xx - 12) ** 2 + (yy - 12) ** 2)
        disk = (rr <= 8).astype(np.float32)
        soft = np.zeros((7, size, size), dtype=np.float32)
        soft[2] = disk
        soft[0] = 1.0 - disk
        products = {}
        for key in ("q000", "q045", "q090", "q135"):
            products[key] = {"Y_lum_psf": soft.copy(), "Mval": (disk > 0).astype(np.uint8)}
        metrics = compute_interorientation_metrics(products, ProjectionConfig())
        self.assertEqual(metrics["n_orientations"], 4)
        self.assertGreater(metrics["classes"]["disco"], 0.8)
        self.assertIn("q000_q045", metrics["pairwise_iou"])


if __name__ == "__main__":
    unittest.main()

