"""Conversión de unidades comoving→físicas (portado de pipeline.py v1).

TNG: posiciones en ckpc/h, masas en 1e10 M_sun/h, velocidades peculiares
v = a·(dx/dt) en km/s — para subhalos la API ya reporta km/s físicos; las
velocidades de partícula del snapshot llevan el factor sqrt(a) (convención
GADGET). Documentado en cada función.
"""
from __future__ import annotations

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from ..core.constants import HUBBLE_PARAM, OMEGA_M, SNAP_A
from ..schemas.models import TNGTruth

COSMO = FlatLambdaCDM(H0=100.0 * HUBBLE_PARAM, Om0=OMEGA_M)


def formation_scale_to_age_gyr(formation_scale: np.ndarray, snapshot_scale: float) -> np.ndarray:
    """Edad estelar cosmológica: t(a_snap) − t(a_form), en Gyr.

    Reemplaza el proxy lineal 13.8·(1−a) del v1 (spec 10 paso 16).
    """
    a_form = np.clip(np.asarray(formation_scale, dtype=np.float64), 1e-4, 1.0)
    z_form = 1.0 / a_form - 1.0
    z_snap = 1.0 / float(snapshot_scale) - 1.0
    t_snap = COSMO.age(z_snap).value  # Gyr
    # age() es caro por partícula; interpolar sobre una rejilla de z
    # (lineal en z bajos, log en z altos).
    z_grid = np.concatenate([np.linspace(0.0, 20.0, 2048), np.geomspace(20.0, 1e4, 1024)])
    t_grid = COSMO.age(z_grid).value
    t_form = np.interp(z_form, z_grid, t_grid)
    ages = t_snap - t_form
    return np.maximum(ages, 1e-3)


def convert_truth_units(truth: TNGTruth) -> TNGTruth:
    """ckpc/h → kpc, 1e10 M_sun/h → M_sun, vel·sqrt(a) → km/s físicas.

    Único bloque rescatado de pipeline.py v1 (`_convert_truth_units`),
    con el factor de escala tomado del Header del cutout (más preciso que
    la tabla SNAP_A).
    """
    a = truth.scale_factor or SNAP_A.get(truth.snapshot, 1.0)
    h = HUBBLE_PARAM
    return truth.model_copy(
        update={
            "stellar_pos": truth.stellar_pos * a / h,
            "stellar_vel": truth.stellar_vel * np.sqrt(a),
            "stellar_mass": truth.stellar_mass * 1e10 / h,
            "stellar_age_gyr": formation_scale_to_age_gyr(truth.stellar_formation_scale, a),
            "gas_pos": truth.gas_pos * a / h if truth.gas_pos is not None else None,
            "gas_vel": truth.gas_vel * np.sqrt(a) if truth.gas_vel is not None else None,
            "gas_mass": truth.gas_mass * 1e10 / h if truth.gas_mass is not None else None,
            "dm_pos": truth.dm_pos * a / h if truth.dm_pos is not None else None,
            "dm_mass": truth.dm_mass * 1e10 / h if truth.dm_mass is not None else None,
            # La API de subhalos reporta pos en ckpc/h y vel en km/s físicos.
            "subhalo_pos": truth.subhalo_pos * a / h,
            "subhalo_vel": truth.subhalo_vel.copy(),
            "stellar_halfmass_rad": truth.stellar_halfmass_rad * a / h
            if truth.stellar_halfmass_rad
            else 0.0,
            # Potential del snapshot viene en (km/s)^2/a → físicas
            "stellar_potential": truth.stellar_potential / a
            if truth.stellar_potential is not None
            else None,
        }
    )
