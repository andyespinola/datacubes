from __future__ import annotations

import numpy as np

from aperturenet_labels.phase_b.alignment import apply_grid_alignment, estimate_d4_alignment


def test_estimate_d4_alignment_recovers_quarter_turn() -> None:
    source = np.zeros((9, 9), dtype=np.float32)
    source[1:4, 2:4] = 2.0
    source[5:8, 6:8] = 5.0
    reference = np.rot90(np.log10(np.clip(source, 1.0, None)), k=3)

    alignment = estimate_d4_alignment(source, reference, "synthetic")
    recovered = apply_grid_alignment(source, alignment)

    assert alignment.rot90_k == 3
    assert not alignment.flip_x
    assert not alignment.flip_y
    assert alignment.score > 0.99
    assert np.array_equal(recovered, np.rot90(source, k=3))
