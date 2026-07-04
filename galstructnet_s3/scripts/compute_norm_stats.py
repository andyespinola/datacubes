#!/usr/bin/env python3
"""Precalcula mean/std + SIGMA_REF_* / SNR_REF + p99(N_eff) del split train.

Emite UN JSON versionado (specs/10 'Normalizacion', specs/50 'Cap'):

    python scripts/compute_norm_stats.py data/dataset_entries \
        [--split train] [--out data/dataset_entries/norm_stats.json]

Si `root/splits/{split}.txt` no existe, usa todos los entries (fixture/CI).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from galstructnet_s3.data.stats import compute_norm_stats

_ENTRY_RE = re.compile(r"dataset_entry_(?P<gid>.+)_v(?P<view>\d+)\.h5$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    files = sorted(args.root.glob("dataset_entry_*.h5"))
    split_file = args.root / "splits" / f"{args.split}.txt"
    if split_file.exists():
        gids = {ln.strip() for ln in split_file.read_text().splitlines()
                if ln.strip()}
        files = [p for p in files
                 if (m := _ENTRY_RE.search(p.name)) and m.group("gid") in gids]
    if not files:
        raise SystemExit(f"sin entries para split '{args.split}' en {args.root}")

    stats = compute_norm_stats(files)
    out = args.out or args.root / "norm_stats.json"
    out.write_text(json.dumps(stats, indent=1))
    caps = {k: round(v, 1) for k, v in stats["n_eff_cap"].items()}
    print(f"{out}: {stats['n_files']} entries, "
          f"snr_ref={stats['snr_ref']:.2f}, n_eff_cap={caps}")


if __name__ == "__main__":
    main()
