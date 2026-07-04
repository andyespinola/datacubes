"""Renderiza paneles de inspeccion visual por galaxia: segmentacion (masa y
luz) + probabilidades por clase + evidencia fisica (v*, sigma*, masa, edad)
+ N_eff y M_valid. Un PNG por galaxia para inspeccion directa."""
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

ENT = Path("/media/andy/Data/tng/mangia_flat/output/dataset_entries")
OUT = Path("/tmp/claude-1000/-home-andy-pythonProjects-datacubes-galstructnet-s3/"
           "bd62c8da-817a-4742-ae97-d71e19831170/scratchpad/inspect")
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = ["bulge", "disk", "bar", "arm", "halo"]
CMAP = ListedColormap(["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"])


def crop(M):
    ys, xs = np.where(M)
    if len(ys) == 0:
        return slice(None), slice(None)
    pad = 3
    return (slice(max(ys.min() - pad, 0), ys.max() + pad + 1),
            slice(max(xs.min() - pad, 0), xs.max() + pad + 1))


def render(gid, tag):
    with h5py.File(ENT / f"{gid}_v0.h5") as f:
        Ym = f["labels/Y_int_mass"][()]
        Yl = f["labels/Y_int_light"][()]
        M = f["masks/M_valid"][()]
        neff = f["labels/n_eff"][()]
        maps = {k: f[f"inputs/pipe3d_maps/{k}"][()] for k in
                ("v_star", "sigma_star", "mass_density", "age_lw")}
    sy, sx = crop(M)
    Mc = M[sy, sx]
    am = np.where(Mc, Ym[sy, sx].argmax(-1), np.nan)
    al = np.where(Mc, Yl[sy, sx].argmax(-1), np.nan)

    fig, ax = plt.subplots(2, 6, figsize=(17, 6))
    fig.suptitle(f"{gid}   [{tag}]", fontsize=13, weight="bold")

    def show(a, img, title, cmap="viridis", vmin=None, vmax=None, mask=True):
        d = np.where(Mc, img, np.nan) if mask else img
        im = a.imshow(d, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                      interpolation="nearest")
        a.set_title(title, fontsize=9)
        a.set_xticks([]); a.set_yticks([])
        return im

    # fila 1: segmentaciones + probabilidades de masa
    a = ax[0, 0]
    a.imshow(am, origin="lower", cmap=CMAP, vmin=0, vmax=4, interpolation="nearest")
    a.set_title("SEG masa (argmax)", fontsize=9, weight="bold")
    a.set_xticks([]); a.set_yticks([])
    a = ax[0, 1]
    a.imshow(al, origin="lower", cmap=CMAP, vmin=0, vmax=4, interpolation="nearest")
    a.set_title("SEG luz (argmax)", fontsize=9, weight="bold")
    a.set_xticks([]); a.set_yticks([])
    for i, cls in enumerate([0, 1, 2, 3]):  # bulge, disk, bar, arm
        show(ax[0, 2 + i], Ym[sy, sx][..., cls], f"P({CLASSES[cls]}) masa",
             cmap="magma", vmin=0, vmax=1)

    # fila 2: evidencia fisica
    v = maps["v_star"][sy, sx]
    vlim = np.nanpercentile(np.abs(v[Mc]), 95) if Mc.any() else 1
    show(ax[1, 0], v - np.nanmedian(v[Mc]), "v* (cinemática)", cmap="RdBu_r",
         vmin=-vlim, vmax=vlim)
    show(ax[1, 1], maps["sigma_star"][sy, sx], "σ* (dispersión)", cmap="inferno")
    show(ax[1, 2], maps["mass_density"][sy, sx], "Σ* densidad masa", cmap="cividis")
    show(ax[1, 3], maps["age_lw"][sy, sx], "edad", cmap="YlOrBr")
    show(ax[1, 4], neff[sy, sx], "N_eff (Kish)", cmap="viridis")
    a = ax[1, 5]
    a.imshow(Mc, origin="lower", cmap="gray", interpolation="nearest")
    a.set_title("M_valid", fontsize=9); a.set_xticks([]); a.set_yticks([])

    # leyenda de clases
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=CMAP(i), label=c) for i, c in enumerate(CLASSES)],
               loc="lower center", ncol=5, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    out = OUT / f"{gid}.png"
    fig.savefig(out, dpi=85, bbox_inches="tight")
    plt.close(fig)
    return out


GALS = [("TNG50-88-312423", "corregida"), ("TNG50-87-340908", "corregida"),
        ("TNG50-89-346164", "corregida"), ("TNG50-89-372192", "corregida + ganó brazo"),
        ("TNG50-88-382174", "corregida, con brazo"), ("TNG50-89-464534", "sana (referencia)")]
for gid, tag in GALS:
    print(render(gid, tag))
