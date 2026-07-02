from __future__ import annotations

CLASS_NAMES = ("bulge", "disk", "bar", "arm", "halo")
CLASS_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

AXIS_VIEWS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

TNG_SIMULATION = "TNG50-1"
