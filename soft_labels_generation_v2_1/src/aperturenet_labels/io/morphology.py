from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(slots=True)
class MorphologyTargets:
    thin_disk: float
    thick_disk: float
    pseudo_bulge: float
    bulge: float
    halo: float
    unbound: float
    barred: bool
    bar_size_kpc: float
    bar_size_alt_kpc: float
    bar_strength: float
    bar_strength_alt: float

    @property
    def disk_fraction(self) -> float:
        return max(0.0, self.thin_disk + self.thick_disk)

    @property
    def bulge_fraction(self) -> float:
        return max(0.0, self.pseudo_bulge + self.bulge)

    @property
    def halo_fraction(self) -> float:
        return max(0.0, self.halo + self.unbound)

    def priors(self) -> np.ndarray:
        values = np.asarray([self.bulge_fraction, self.disk_fraction, self.halo_fraction], dtype=np.float64)
        values = np.clip(values, 1.0e-4, None)
        return values / values.sum()

    def as_catalog_fractions(self) -> dict[str, float | bool]:
        return {
            "bulge": float(self.bulge_fraction),
            "disk": float(self.disk_fraction),
            "halo": float(self.halo_fraction),
            "barred": bool(self.barred),
            "bar_size_kpc": float(max(self.bar_size_kpc, self.bar_size_alt_kpc)),
            "bar_strength": float(max(self.bar_strength, self.bar_strength_alt)),
        }


def load_morphology_targets(path: str | Path, snapshot: int, subhalo_id: int) -> MorphologyTargets:
    with h5py.File(path, "r") as handle:
        group = handle[f"Snapshot_{snapshot}"]
        ids = np.asarray(group["SubhaloID"], dtype=np.int64)
        matches = np.where(ids == int(subhalo_id))[0]
        if matches.size == 0:
            raise KeyError(f"SubhaloID={subhalo_id} not found in Snapshot_{snapshot} of {path}")
        idx = int(matches[0])

        def scalar(name: str, row: int | None = None, default: float = 0.0) -> float:
            if name not in group:
                return default
            data = np.asarray(group[name])
            value = data[row, idx] if row is not None and data.ndim == 2 else data[idx]
            if np.isnan(value):
                return default
            return float(value)

        return MorphologyTargets(
            thin_disk=scalar("ThinDisc", row=0),
            thick_disk=scalar("ThickDisc", row=0),
            pseudo_bulge=scalar("PseudoBulge", row=0),
            bulge=scalar("Bulge", row=0),
            halo=scalar("Halo", row=0),
            unbound=scalar("UnboundMass"),
            barred=bool(scalar("Barred") > 0.0),
            bar_size_kpc=max(0.0, scalar("BarSize", row=0, default=0.0)),
            bar_size_alt_kpc=max(0.0, scalar("BarSize", row=1, default=0.0)),
            bar_strength=max(0.0, scalar("BarStrength", row=0, default=0.0)),
            bar_strength_alt=max(0.0, scalar("BarStrength", row=1, default=0.0)),
        )
