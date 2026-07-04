"""Vinculación cubo ↔ subhalo ↔ vista ↔ pyPipe3D (adaptado del v1 al layout plano data/)."""
from __future__ import annotations

import re
from pathlib import Path

from ..schemas.models import ManifestRow
from .catalogs import load_mangia_catalog_row

CUBE_RE = re.compile(
    r"^TNG50-(?P<snapshot>\d+)-(?P<subhalo_id>\d+)-(?P<view>\d+)-(?P<ifu>\d+)\.cube\.fits(?:\.gz)?$"
)

PILOT_SUBHALOS = (155298, 192324)
PILOT_SNAPSHOT = 87


def parse_canonical_id(name: str) -> dict[str, int]:
    match = CUBE_RE.match(name)
    if not match:
        raise ValueError(f"No pude extraer snapshot/subhalo_id/view/ifu desde {name}")
    return {
        "snapshot": int(match.group("snapshot")),
        "subhalo_id": int(match.group("subhalo_id")),
        "view": int(match.group("view")),
        "ifu_design": int(match.group("ifu")),
    }


def canonical_id(snapshot: int, subhalo_id: int, view: int, ifu_design: int) -> str:
    return f"TNG50-{snapshot}-{subhalo_id}-{view}-{ifu_design}"


def galaxy_id(snapshot: int, subhalo_id: int) -> str:
    return f"TNG50-{snapshot}-{subhalo_id}"


def build_manifest_from_dir(data_dir: str | Path, catalog_name: str = "MaNGIA_catalog.fits") -> list[ManifestRow]:
    """Descubre entradas en el layout plano de data/ (caso piloto)."""
    data_dir = Path(data_dir)
    catalog_path = data_dir / catalog_name
    rows: list[ManifestRow] = []
    for cube_path in sorted(data_dir.glob("TNG50-*.cube.fits.gz")):
        parts = parse_canonical_id(cube_path.name)
        snap, sub, view = parts["snapshot"], parts["subhalo_id"], parts["view"]
        base = f"TNG50-{snap}-{sub}"
        cutout = data_dir / f"{base}.cutout.hdf5"
        phase2 = data_dir / f"{base}.cutout_phase2.hdf5"
        subhalo_json = data_dir / f"{base}.subhalo.json"
        maps_path = data_dir / cube_path.name.replace(".cube.fits.gz", ".cube_maps.fits")
        if not (cutout.exists() and subhalo_json.exists()):
            continue
        try:
            cat = load_mangia_catalog_row(catalog_path, snap, sub, view)
        except KeyError:
            continue
        rows.append(
            ManifestRow(
                canonical_id=canonical_id(snap, sub, view, parts["ifu_design"]),
                cutout_path=str(cutout),
                cutout_phase2_path=str(phase2) if phase2.exists() else "",
                subhalo_json_path=str(subhalo_json),
                cube_path=str(cube_path),
                pipe3d_maps_path=str(maps_path) if maps_path.exists() else "",
                snapshot=snap,
                subhalo_id=sub,
                view=view,
                re_kpc=float(cat["re_kpc"]),
                ifu_design=parts["ifu_design"],
                repeat_count=int(cat["repeat_count"]),
            )
        )
    return rows


def pilot_manifest(data_dir: str | Path) -> list[ManifestRow]:
    rows = build_manifest_from_dir(data_dir)
    pilots = [r for r in rows if r.subhalo_id in PILOT_SUBHALOS and r.snapshot == PILOT_SNAPSHOT]
    if not pilots:
        raise FileNotFoundError(f"No encontré galaxias piloto {PILOT_SUBHALOS} en {data_dir}")
    return pilots
