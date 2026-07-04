"""Readers de catálogos externos: MORDOR, circularidades TNG, catálogo MaNGIA.

MORDOR (Zana et al. 2022) vive en data/morphs_kinematic_bars.hdf5:
  Snapshot_{snap}/{Bulge,PseudoBulge,ThinDisc,ThickDisc,Halo} con shape (3, N)
  donde fila 0 = fracción de masa. Barred (N,), BarSize (2,N) = [R_Phi, R_peak],
  BarStrength (2,N) = [A2_max, A2(<R_peak)].
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

from ..schemas.models import BarMeta, CatalogPriors, MorphologyTargets


def load_morphology_targets(
    path: str | Path, snapshot: int, subhalo_id: int
) -> MorphologyTargets:
    """Portado de labeling/tng.py v1 (load_morphology_targets)."""
    with h5py.File(path, "r") as handle:
        group = handle[f"Snapshot_{snapshot}"]
        ids = np.asarray(group["SubhaloID"], dtype=np.int64)
        indices = np.where(ids == subhalo_id)[0]
        if indices.size == 0:
            raise KeyError(
                f"No encontré SubhaloID={subhalo_id} en Snapshot_{snapshot} del catálogo MORDOR"
            )
        idx = int(indices[0])

        def scalar_from(dataset_name: str, row: int | None = None, default: float = 0.0) -> float:
            if dataset_name not in group:
                return default
            data = np.asarray(group[dataset_name])
            value = data[idx] if row is None else data[row, idx]
            if not np.isfinite(value):
                return default
            return float(value)

        return MorphologyTargets(
            thin_disk=scalar_from("ThinDisc", row=0),
            thick_disk=scalar_from("ThickDisc", row=0),
            pseudo_bulge=scalar_from("PseudoBulge", row=0),
            bulge=scalar_from("Bulge", row=0),
            halo=scalar_from("Halo", row=0),
            unbound=scalar_from("UnboundMass", default=0.0),
            barred=bool(scalar_from("Barred") > 0),
            bar_size_kpc=max(0.0, scalar_from("BarSize", row=0, default=-1.0)),
            bar_size_alt_kpc=max(0.0, scalar_from("BarSize", row=1, default=-1.0)),
            bar_strength=max(0.0, scalar_from("BarStrength", row=0, default=-1.0)),
            bar_strength_alt=max(0.0, scalar_from("BarStrength", row=1, default=-1.0)),
            quality_krot=scalar_from("QualityFlags", row=0),
            quality_sigma_ratio=scalar_from("QualityFlags", row=1),
            quality_b1b2=scalar_from("QualityFlags", row=2),
        )


def priors_from_mordor(targets: MorphologyTargets, confidence: float = 0.5) -> CatalogPriors:
    """Mapeo MORDOR → prior K=3 del Classifier (spec 11 paso 4).

    bulge = bulge + pseudo-bulge; disk = thin + thick; other = halo (+unbound).
    Renormaliza por si las fracciones no suman 1 (e.g. NaN→0 en ThickDisc).
    """
    fracs = np.array(
        [targets.bulge_family, targets.disk_family, targets.other_family], dtype=np.float64
    )
    total = fracs.sum()
    if total <= 0:
        return CatalogPriors(
            source="none", bulge_frac=1 / 3, disk_frac=1 / 3, other_frac=1 / 3,
            confidence=0.0,
        )
    fracs /= total
    return CatalogPriors(
        source="mordor",
        bulge_frac=float(fracs[0]),
        disk_frac=float(fracs[1]),
        other_frac=float(fracs[2]),
        confidence=confidence,
    )


def bar_meta_from_mordor(targets: MorphologyTargets) -> BarMeta:
    return BarMeta(
        has_bar=targets.barred,
        # R_peak (fila 1) es el tamaño de barra que el spec 12 v2.1 usa para validar.
        bar_size_kpc=targets.bar_size_alt_kpc or targets.bar_size_kpc or None,
        bar_strength=targets.bar_strength_alt or targets.bar_strength or None,
        bar_angle_deg=None,  # MORDOR no reporta ángulo; se deriva del Fourier m=2
    )


def load_stellar_circs(
    path: str | Path, snapshot: int, subhalo_id: int
) -> dict[str, float]:
    """Catálogo de circularidades TNG (subhalo-level) para validar ε (spec 10)."""
    with h5py.File(path, "r") as handle:
        group = handle[f"Snapshot_{snapshot}"]
        ids = np.asarray(group["SubfindID"], dtype=np.int64)
        indices = np.where(ids == subhalo_id)[0]
        if indices.size == 0:
            raise KeyError(f"SubfindID={subhalo_id} no está en Snapshot_{snapshot} de stellar_circs")
        idx = int(indices[0])
        out: dict[str, float] = {}
        for key in (
            "CircAbove07Frac",
            "CircAbove07Frac_allstars",
            "CircAbove07MinusBelowNeg07Frac",
            "CircTwiceBelow0Frac",
        ):
            if key in group:
                out[key] = float(np.asarray(group[key])[idx])
        return out


def load_mangia_catalog_row(
    path: str | Path, snapshot: int, subhalo_id: int, view: int
) -> dict[str, float | int]:
    data = fits.getdata(path, 1)
    mask = (
        (data["snapshot"] == snapshot)
        & (data["subhalo_id"] == subhalo_id)
        & (data["view"] == view)
    )
    rows = data[mask]
    if len(rows) == 0:
        raise KeyError(f"({snapshot},{subhalo_id},{view}) no está en el catálogo MaNGIA")
    row = rows[0]
    repeat_count = int(((data["snapshot"] == snapshot) & (data["subhalo_id"] == subhalo_id)).sum())
    return {
        "re_kpc": float(row["re_kpc"]),
        "stellar_mass_log": float(row["stellar_mass"]),
        "ifu_design": int(row["manga_ifu_dsn"]),
        "n_star_part": int(row["n_star_part"]),
        "n_gas_cell": int(row["n_gas_cell"]),
        "repeat_count": repeat_count,
    }
