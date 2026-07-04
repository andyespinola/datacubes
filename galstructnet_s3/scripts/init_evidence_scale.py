#!/usr/bin/env python3
"""Calibra el init de `b` en EvidenceHead: S_init ~ K + kappa*mediana(N_eff).

Spec: specs/40 'Init de b' / specs/45 P4. Con c ~ 1 y e_net ~ softplus(0)
~ 0.693 por clase, S = K + K*0.693*softplus(b); se resuelve b para que
S_init caiga en el orden de kappa*mediana(N_eff_train):

    softplus(b) = (kappa*med(N_eff)) / (K * 0.693)
    b = log(exp(x) - 1)   con x = softplus(b)

Lee la mediana de N_eff del norm_stats.json (o la computa) y emite el b
recomendado para pasar a EvidenceHead(b_init=...) / config del modelo.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def solve_b(kappa: float, n_eff_median: float, n_classes: int = 5) -> float:
    e_net0 = math.log(2.0)                       # softplus(0)
    target = max(kappa * n_eff_median / (n_classes * e_net0), 1e-3)
    # inversa de softplus
    return float(np.log(np.expm1(target)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="dir con dataset_entry_*.h5")
    ap.add_argument("--kappa", type=float, default=0.5)
    ap.add_argument("--weighting", default="raw_lum",
                    choices=["raw_mass", "raw_lum", "psf_mass", "psf_lum"])
    args = ap.parse_args()

    stats_path = args.root / "norm_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        med = _median_from_entries(args.root, args.weighting)
        cap = stats["n_eff_cap"][args.weighting]
    else:
        med = _median_from_entries(args.root, args.weighting)
        cap = float("nan")
    b = solve_b(args.kappa, med)
    print(f"mediana N_eff_{args.weighting} = {med:.1f} (cap p99 = {cap:.1f})")
    print(f"b_init recomendado (kappa={args.kappa}): {b:.3f}  "
          f"-> S_init ~ {5 + args.kappa * med:.1f}")


def _median_from_entries(root: Path, weighting: str) -> float:
    import h5py
    variant, weight = weighting.split("_")
    vals = []
    for p in sorted(root.glob("dataset_entry_*.h5")):
        with h5py.File(p, "r") as f:
            M = f["masks/M_valid"][()].astype(bool)
            vals.append(f[f"labels/N_eff_{variant}_{weight}"][()][M])
    return float(np.median(np.concatenate(vals)))


if __name__ == "__main__":
    main()
