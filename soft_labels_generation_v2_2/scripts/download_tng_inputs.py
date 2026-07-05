#!/usr/bin/env python3
"""Descarga de TNG los inputs que falten para etiquetar (cutout, subhalo.json,
materia oscura). NO baja cubos ni cube_maps: esos son productos de la
reconstrucción MaNGIA local, no de la API de TNG.

Por cada galaxia con un cubo presente, comprueba y descarga lo que falte:
  - `TNG50-{s}-{sub}.cutout.hdf5`         (estrellas + gas) si no existe
  - `TNG50-{s}-{sub}.subhalo.json`        (metadatos)       si no existe
  - materia oscura: si el cutout no trae PartType1 y no hay phase2, baja
    `TNG50-{s}-{sub}.cutout_phase2.hdf5`  (query mínima `dm=Coordinates`)

Requiere API-Key de TNG (línea `TNG_API_KEY=...` en --env-file o env var).
Idempotente y reanudable: salta lo que ya está. Escribe en --out-dir (por
defecto, junto a los cubos; usa otro dir si la USB es de solo-lectura).

Uso:
    python scripts/download_tng_inputs.py \
        --input-dir "/run/media/aespinola/ADATA HM800/datacubes" \
        --env-file /ruta/.env \
        [--what all|cutout|subhalo|dm] [--out-dir <dir>] [--workers 4] [--limit N]

Nota de volumen: el DM domina el peso (decenas–cientos de MB/galaxia). Bajar
cutouts+DM de 10k galaxias puede ser cientos de GB — planifica el ancho de
banda y el espacio.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import requests

REPO = Path(__file__).resolve().parents[1]
SIM = "TNG50-1"
STAR_FIELDS = "Coordinates,Velocities,Masses,GFM_StellarFormationTime,GFM_Metallicity"
GAS_FIELDS = ("Coordinates,Velocities,Masses,StarFormationRate,Density,"
              "InternalEnergy,ElectronAbundance,GFM_Metallicity")
_ID_RE = re.compile(r"TNG50-(?P<snap>\d+)-(?P<sub>\d+)-\d+-\d+\.cube\.fits\.gz$")


def _api_key(env_file: Path | None) -> str:
    if os.environ.get("TNG_API_KEY"):
        return os.environ["TNG_API_KEY"]
    if env_file and env_file.exists():
        for ln in env_file.read_text().splitlines():
            if ln.startswith("TNG_API_KEY"):
                return ln.split("=", 1)[1].strip()
    sys.exit("Falta TNG_API_KEY (usa --env-file o exporta la variable).")


def _cutout_has_dm(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as f:
            return "PartType1" in f
    except Exception:  # noqa: BLE001
        return False


def _get(url: str, key: str, timeout: int = 600):
    r = requests.get(url, headers={"API-Key": key}, timeout=timeout, stream=True)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:150]}")
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--env-file", type=Path, default=None)
    ap.add_argument("--what", choices=["all", "cutout", "subhalo", "dm"],
                    default="all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "src"))
    from aperturenet_labels.io.tng_reader import download_cutout_fields

    key = _api_key(args.env_file)
    out = args.out_dir or args.input_dir
    out.mkdir(parents=True, exist_ok=True)

    # galaxias únicas (varias vistas comparten cutout/subhalo/DM)
    seen: set[tuple[int, int]] = set()
    gals = []
    for cube in sorted(args.input_dir.glob("TNG50-*.cube.fits.gz")):
        m = _ID_RE.search(cube.name)
        if not m:
            continue
        snap, sub = int(m["snap"]), int(m["sub"])
        if (snap, sub) in seen:
            continue
        seen.add((snap, sub))
        gals.append((snap, sub))
    if args.limit:
        gals = gals[:args.limit]
    print(f"galaxias únicas con cubo: {len(gals)}  (what={args.what})")

    def work(item):
        snap, sub = item
        base = f"TNG50-{snap}-{sub}"
        cutout = out / f"{base}.cutout.hdf5"
        cutout_in = args.input_dir / f"{base}.cutout.hdf5"
        subj = out / f"{base}.subhalo.json"
        phase2 = out / f"{base}.cutout_phase2.hdf5"
        done = []
        try:
            # 1. cutout (estrellas+gas)
            if args.what in ("all", "cutout") and not (cutout.exists()
                                                       or cutout_in.exists()):
                download_cutout_fields(snap, sub, cutout, key,
                                       query=f"stars={STAR_FIELDS}&gas={GAS_FIELDS}")
                done.append("cutout")
            # 2. subhalo.json
            if args.what in ("all", "subhalo") and not subj.exists():
                url = (f"https://www.tng-project.org/api/{SIM}/snapshots/"
                       f"{snap}/subhalos/{sub}/")
                subj.write_text(json.dumps(_get(url, key, 120).json(), indent=2))
                done.append("subhalo")
            # 3. materia oscura (si el cutout final no la trae y no hay phase2)
            if args.what in ("all", "dm"):
                cut = cutout if cutout.exists() else cutout_in
                if not phase2.exists() and not (cut.exists()
                                                and _cutout_has_dm(cut)):
                    download_cutout_fields(snap, sub, phase2, key,
                                           query="dm=Coordinates")
                    done.append("dm")
            return (base, "ok" if done else "ya_completo", done)
        except Exception as exc:  # noqa: BLE001
            return (base, f"error: {type(exc).__name__}: {exc}", done)

    t0 = time.time()
    ok = err = nada = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, g) for g in gals]
        for i, fut in enumerate(as_completed(futs), 1):
            base, status, done = fut.result()
            if status.startswith("error"):
                err += 1
                print(f"  {base}: {status}")
            elif status == "ok":
                ok += 1
            else:
                nada += 1
            if i % 50 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(gals)}  (descargó={ok} ya={nada} err={err}, "
                      f"{i/max(el,1e-9)*3600:.0f}/h)", flush=True)
    print(f"\nhecho: {ok} con descargas, {nada} ya completas, {err} error "
          f"de {len(gals)} ({(time.time()-t0)/3600:.1f} h)")


if __name__ == "__main__":
    main()
