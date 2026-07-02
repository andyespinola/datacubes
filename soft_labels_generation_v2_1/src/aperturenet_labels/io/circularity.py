from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


def load_stellar_circularity_summary(path: str | Path, snapshot: int, subhalo_id: int) -> dict[str, float] | None:
    path = Path(path)
    if not path.exists():
        return None
    group_name = f"Snapshot_{int(snapshot)}"
    with h5py.File(path, "r") as handle:
        if group_name not in handle:
            return None
        group = handle[group_name]
        if "SubfindID" not in group:
            return None
        ids = np.asarray(group["SubfindID"], dtype=np.int64)
        matches = np.flatnonzero(ids == int(subhalo_id))
        if matches.size == 0:
            return None
        idx = int(matches[0])
        out: dict[str, float] = {}
        for name in (
            "CircAbove07Frac",
            "CircAbove07MinusBelowNeg07Frac",
            "CircTwiceBelow0Frac",
            "SpecificAngMom",
            "CircAbove07Frac_allstars",
            "CircAbove07MinusBelowNeg07Frac_allstars",
            "CircTwiceBelow0Frac_allstars",
            "SpecificAngMom_allstars",
        ):
            if name in group:
                value: Any = group[name][idx]
                if np.ndim(value) == 0 and np.isfinite(value):
                    out[name] = float(value)
        return out
