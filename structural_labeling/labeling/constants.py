from __future__ import annotations

CLASS_NAMES = (
    "no_valido",
    "bulbo",
    "disco",
    "barra",
    "brazos",
    "other",
    "incierto",
)

CLASS_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
PHYSICAL_CLASS_NAMES = CLASS_NAMES[1:-1]
PHYSICAL_CLASS_INDICES = tuple(CLASS_INDEX[name] for name in PHYSICAL_CLASS_NAMES)

TNG_SIMULATION = "TNG50-1"

# Views in the local MaNGIA naming are zero-indexed. For galaxies repeated up to
# three times we follow the axis-aligned definitions described in the paper.
AXIS_VIEWS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
