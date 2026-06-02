from __future__ import annotations

import unittest

import numpy as np

from labeling.constants import CLASS_INDEX
from labeling.epsilon_baseline import EpsilonBaselineConfig, circularity_proxy, particle_probabilities_from_epsilon


class EpsilonBaselineTest(unittest.TestCase):
    def test_circularity_and_threshold_labels(self) -> None:
        faceon_pos = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        faceon_vel = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
        epsilon = circularity_proxy(faceon_pos, faceon_vel)
        self.assertTrue(np.allclose(epsilon, [1.0, 0.0, -1.0]))

        probs = particle_probabilities_from_epsilon(epsilon, EpsilonBaselineConfig(disk_threshold=0.70))
        self.assertEqual(int(np.argmax(probs[0])), CLASS_INDEX["disco"])
        self.assertEqual(int(np.argmax(probs[1])), CLASS_INDEX["bulbo"])
        self.assertEqual(int(np.argmax(probs[2])), CLASS_INDEX["bulbo"])

    def test_counterrotating_optional_other_class(self) -> None:
        epsilon = np.array([0.9, 0.1, -0.9])
        config = EpsilonBaselineConfig(disk_threshold=0.70, counterrotating_as_other=True)
        probs = particle_probabilities_from_epsilon(epsilon, config)
        self.assertEqual(int(np.argmax(probs[0])), CLASS_INDEX["disco"])
        self.assertEqual(int(np.argmax(probs[1])), CLASS_INDEX["bulbo"])
        self.assertEqual(int(np.argmax(probs[2])), CLASS_INDEX["other"])


if __name__ == "__main__":
    unittest.main()
