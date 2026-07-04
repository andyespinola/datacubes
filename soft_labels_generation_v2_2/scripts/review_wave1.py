"""Revision de consistencia espacial (perfil radial P(bulbo) vs P(disco), y
sobre-asignacion bulbo-vs-halo) sobre las 20 galaxias de la oleada 1 mas los
2 pilotos originales, agrupado por estrato MORDOR (halo/disk/bulge/mixed).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np

WAVE1_DIR = Path("/media/andy/Data/tng/mangia_flat/output")
PILOT_DIR = Path("/home/andy/pythonProjects/datacubes/data/output")
STRATA_CSV = Path(
    "/home/andy/pythonProjects/datacubes/orientation_projection_validation/data/wave1_strata.csv"
)
CENTER_R_PX = 3.0
CLASSES = ["bulge", "disk", "bar", "arm", "halo"]


def load_strata() -> dict[str, str]:
    strata = {}
    with STRATA_CSV.open() as f:
        for row in csv.DictReader(f):
            strata[row["galaxy_id"]] = row["stratum"]
    # pilotos: clasificados con la misma convencion (bulge=Bulge+PseudoBulge, etc.)
    strata["TNG50-87-155298"] = "halo"   # halo_frac=0.563 (ver ADR-001)
    strata["TNG50-87-192324"] = "bulge"  # bulge+pseudo=0.676
    return strata


def analyze_galaxy(entry_path: Path, qa_path: Path) -> dict:
    with h5py.File(entry_path, "r") as f:
        Y = f["labels/Y_int_mass"][:]
        M = f["masks/M_valid"][:]
    h, w, _ = Y.shape
    cy, cx = h / 2 - 0.5, w / 2 - 0.5
    yy, xx = np.indices((h, w))
    r_px = np.hypot(yy - cy, xx - cx)

    valid = M & (Y.sum(-1) > 0)
    am = np.where(valid, Y.argmax(-1), -1)
    central = valid & (r_px < CENTER_R_PX)
    n_central = int(central.sum())

    p_bulge, p_disk = Y[:, :, 0], Y[:, :, 1]
    n_disk_beats_bulge = int((p_disk[central] > p_bulge[central]).sum()) if n_central else 0
    n_argmax_disk = int((am[central] == 1).sum()) if n_central else 0

    qa = json.loads(qa_path.read_text())
    dev = qa.get("fraction_deviations", {})
    rec = qa.get("fractions_recovered", {})
    cat = qa.get("fractions_catalog", {})

    return {
        "n_central": n_central,
        "frac_disk_beats_bulge_central": n_disk_beats_bulge / n_central if n_central else None,
        "frac_argmax_disk_central": n_argmax_disk / n_central if n_central else None,
        "bulge_recovered": rec.get("bulge"),
        "bulge_catalog": cat.get("bulge"),
        "other_recovered": rec.get("halo", rec.get("other")),
        "other_catalog": cat.get("other"),
        "dev_other": dev.get("other"),
        "n_valid": int(valid.sum()),
        "flags": qa.get("flags", []),
        "status": qa.get("status"),
    }


def main() -> None:
    strata = load_strata()
    rows = []

    for gal in sorted(strata):
        if gal in ("TNG50-87-155298", "TNG50-87-192324"):
            entry = PILOT_DIR / "dataset_entries" / f"{gal}_v0.h5"
            qa = PILOT_DIR / "qa_reports" / f"{gal}_v0.json"
        else:
            entry = WAVE1_DIR / "dataset_entries" / f"{gal}_v0.h5"
            qa = WAVE1_DIR / "qa_reports" / f"{gal}_v0.json"
        if not entry.exists() or not qa.exists():
            print(f"SKIP {gal}: falta {entry if not entry.exists() else qa}")
            continue
        r = analyze_galaxy(entry, qa)
        r["galaxy_id"] = gal
        r["stratum"] = strata[gal]
        rows.append(r)

    print(
        f"{'galaxia':<18} {'estrato':<7} {'n_c':>4} {'%disco>bulbo_c':>15} "
        f"{'bulbo_rec':>9} {'bulbo_cat':>9} {'other_rec':>9} {'other_cat':>9} {'dev_other':>9}"
    )
    for r in sorted(rows, key=lambda x: (x["stratum"], x["galaxy_id"])):
        pct = f"{100*r['frac_disk_beats_bulge_central']:.0f}%" if r["frac_disk_beats_bulge_central"] is not None else "n/a"
        print(
            f"{r['galaxy_id']:<18} {r['stratum']:<7} {r['n_central']:>4} {pct:>15} "
            f"{r['bulge_recovered']:>9.3f} {r['bulge_catalog']:>9.3f} "
            f"{r['other_recovered']:>9.3f} {r['other_catalog']:>9.3f} {r['dev_other']:>9.3f}"
        )

    print("\n=== resumen por estrato ===")
    for stratum in ["halo", "disk", "bulge", "mixed"]:
        sub = [r for r in rows if r["stratum"] == stratum]
        if not sub:
            continue
        central_anomaly = [r for r in sub if r["frac_disk_beats_bulge_central"] and r["frac_disk_beats_bulge_central"] > 0.3]
        halo_swallowed = [r for r in sub if r["dev_other"] is not None and r["dev_other"] > 0.25 and r["other_recovered"] < r["other_catalog"]]
        print(
            f"{stratum:<7} n={len(sub):2d}  "
            f"flip_central(>30% disco>bulbo en r<{CENTER_R_PX:.0f}px)={len(central_anomaly)}/{len(sub)}  "
            f"bulbo_devora_halo(dev_other>0.25)={len(halo_swallowed)}/{len(sub)}"
        )

    out_path = Path("/media/andy/Data/tng/mangia_flat/wave1_review_summary.json")
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nDetalle completo en {out_path}")


if __name__ == "__main__":
    main()
