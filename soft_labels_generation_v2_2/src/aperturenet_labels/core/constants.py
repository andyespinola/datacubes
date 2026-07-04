"""Constantes del pipeline v2.

Esquema de clases v2: 5 clases físicas; la validez vive en M_valid (spec 22)
y la incertidumbre en las propias probabilidades (principio 19.5 del
diagnóstico). No existe el eje "incierto" del v1.
"""
from __future__ import annotations

CLASS_NAMES: tuple[str, ...] = ("bulge", "disk", "bar", "arm", "halo")
CLASS_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
N_CLASSES: int = len(CLASS_NAMES)

TNG_SIMULATION = "TNG50-1"

# Vistas axiales para galaxias repetidas hasta 3 veces (convención v1 validada,
# marco de coordenadas de la simulación — NO el marco face-on).
AXIS_VIEWS: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

# Factor de escala por snapshot TNG50 (portado del v1).
SNAP_A: dict[int, float] = {
    85: 0.8459,
    86: 0.8564,
    87: 0.8671,
    88: 0.8778,
    89: 0.8885,
    90: 0.8993,
    91: 0.9091,
    92: 0.9212,
    93: 0.9322,
    94: 0.9433,
    95: 0.9545,
    96: 0.9657,
    97: 0.9771,
    98: 0.9885,
    99: 1.0,
}
HUBBLE_PARAM = 0.6774
OMEGA_M = 0.3089
OMEGA_L = 0.6911

# Constante gravitacional en kpc·(km/s)^2 / M_sun
G_KPC_KMS2_MSUN = 4.30091e-6

# Masa de partícula DM en unidades de código (MassTable[1] de TNG50-1);
# los cutouts de DM no traen dataset Masses.
DM_MASS_CODE_UNITS = 3.07367709e-5
