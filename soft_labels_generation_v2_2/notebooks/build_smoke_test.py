"""Construye y ejecuta notebooks/00_smoke_test.ipynb sobre los productos del piloto.

Uso: python notebooks/build_smoke_test.py
Requiere que `aperturenet-labels run --pilot` ya haya corrido.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        "# 00 — Smoke test del pipeline v2 sobre las galaxias piloto\n\n"
        "Verificación rápida de los productos generados para "
        "`TNG50-87-155298-0-127` y `TNG50-87-192324-0-127`.\n\n"
        "Secciones: (1) features de Fase A y ε vs catálogo, (2) alineación "
        "3D→2D vs pyPipe3D, (3) tensores `Y_int` y máscara, (4) QA reports.",
    ),
    (
        "code",
        "from pathlib import Path\n"
        "import json\n"
        "import numpy as np\n"
        "import h5py\n"
        "import matplotlib.pyplot as plt\n"
        "from aperturenet_labels.io import manifest, catalogs, mangia_reader\n"
        "from aperturenet_labels.phase_a import extractor, classifier\n"
        "from aperturenet_labels.phase_c.packer import validate_dataset_entry\n"
        "\n"
        "DATA = Path('../../data').resolve()\n"
        "rows = manifest.pilot_manifest(DATA)\n"
        "CLASSES = ['bulge', 'disk', 'bar', 'arm', 'halo']\n"
        "print([r.canonical_id for r in rows])",
    ),
    ("markdown", "## 1. Fase A — distribución de ε y comparación con catálogos"),
    (
        "code",
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "for ax, r in zip(axes, rows):\n"
        "    gal = manifest.galaxy_id(r.snapshot, r.subhalo_id)\n"
        "    feats = extractor.load_particle_features(\n"
        "        DATA / 'intermediate/phase_a' / gal / 'particle_features.h5')\n"
        "    circs = catalogs.load_stellar_circs(DATA / 'stellar_circs.hdf5', r.snapshot, r.subhalo_id)\n"
        "    mordor = catalogs.load_morphology_targets(\n"
        "        DATA / 'morphs_kinematic_bars.hdf5', r.snapshot, r.subhalo_id)\n"
        "    ax.hist(feats['epsilon'], bins=120, density=True, alpha=0.8)\n"
        "    p7 = feats['quality']['epsilon_p7_fraction']\n"
        "    ax.set_title(f\"{gal}\\n\"\n"
        "                 f\"eps_p7={p7:.3f} vs catálogo {circs['CircAbove07Frac']:.3f} | \"\n"
        "                 f\"MORDOR barred={mordor.barred}\")\n"
        "    ax.set_xlabel('ε = j_z / j_c(E)')\n"
        "plt.tight_layout()",
    ),
    ("markdown", "## 2. Alineación 3D→2D contra el mapa de masa de pyPipe3D (gate del spec 20)"),
    (
        "code",
        "from scipy.stats import spearmanr\n"
        "from scipy.ndimage import center_of_mass\n"
        "from aperturenet_labels.io import tng_reader, units\n"
        "from aperturenet_labels.core.geometry import view_vector_from_index, deposit_to_grid\n"
        "from aperturenet_labels.phase_b.label_projection import mangia_raster_coords\n"
        "\n"
        "fig, axes = plt.subplots(2, 3, figsize=(13, 8))\n"
        "for i, r in enumerate(rows):\n"
        "    truth = units.convert_truth_units(tng_reader.load_cutout_truth(\n"
        "        r.cutout_path, r.subhalo_json_path, r.cutout_phase2_path))\n"
        "    geom = mangia_reader.load_cube_geometry(r.cube_path)\n"
        "    p3d = mangia_reader.load_pipe3d_maps(r.pipe3d_maps_path)['mass_density']\n"
        "    centered = truth.stellar_pos - truth.subhalo_pos[None, :]\n"
        "    u, v, _ = mangia_raster_coords(centered, view_vector_from_index(r.view, r.repeat_count))\n"
        "    grid = deposit_to_grid(v, u, truth.stellar_mass, geom.shape,\n"
        "                           geom.pixel_scale_kpc, sigma_pixels=geom.psf_sigma_pixels)\n"
        "    good = np.isfinite(p3d) & (p3d != 0)\n"
        "    m = good & (p3d > np.percentile(p3d[good], 30)) & (grid > 0)\n"
        "    rho = spearmanr(np.log10(grid[m]), p3d[m]).statistic\n"
        "    c1 = center_of_mass(grid / grid.sum())\n"
        "    c2 = center_of_mass(np.where(good, 10 ** np.clip(p3d, -5, 15), 0.0))\n"
        "    shift = float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))\n"
        "    axes[i, 0].imshow(np.log10(np.clip(grid, 1, None)), origin='lower', cmap='inferno')\n"
        "    axes[i, 0].set_title(f'{r.subhalo_id} masa proyectada')\n"
        "    axes[i, 1].imshow(np.where(good, p3d, np.nan), origin='lower', cmap='inferno')\n"
        "    axes[i, 1].set_title('pyPipe3D masa')\n"
        "    axes[i, 2].axis('off')\n"
        "    axes[i, 2].text(0.1, 0.5, f'Spearman = {rho:.3f}\\ncentroide = {shift:.2f} px',\n"
        "                    fontsize=14)\n"
        "    assert shift < 1.0 and rho > 0.8\n"
        "for ax in axes[:, :2].flat:\n"
        "    ax.axis('off')\n"
        "plt.tight_layout()",
    ),
    ("markdown", "## 3. Tensores de pseudo-etiquetas `Y_int` y máscara `M_valid`"),
    (
        "code",
        "from matplotlib.colors import ListedColormap\n"
        "import matplotlib.patches as mpatches\n"
        "colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']\n"
        "fig, axes = plt.subplots(2, 6, figsize=(21, 7.5))\n"
        "for i, r in enumerate(rows):\n"
        "    gal = manifest.galaxy_id(r.snapshot, r.subhalo_id)\n"
        "    with h5py.File(DATA / 'output/dataset_entries' / f'{gal}_v0.h5') as f:\n"
        "        Y = f['labels/Y_int_mass'][:]\n"
        "        M = f['masks/M_valid'][:]\n"
        "    am = np.where(M & (Y.sum(-1) > 0), Y.argmax(-1), -1)\n"
        "    axes[i, 0].imshow(am, origin='lower', cmap=ListedColormap(['black'] + colors),\n"
        "                      vmin=-1, vmax=4)\n"
        "    axes[i, 0].set_title(f'{gal}\\nclase dominante')\n"
        "    for c in range(5):\n"
        "        axes[i, 1 + c].imshow(np.where(M, Y[:, :, c], np.nan), origin='lower',\n"
        "                              vmin=0, vmax=1)\n"
        "        axes[i, 1 + c].set_title(f'P({CLASSES[c]})')\n"
        "for ax in axes.flat:\n"
        "    ax.axis('off')\n"
        "fig.legend(handles=[mpatches.Patch(color=colors[c], label=CLASSES[c]) for c in range(5)],\n"
        "           loc='lower center', ncol=5)\n"
        "plt.tight_layout(rect=[0, 0.04, 1, 1])",
    ),
    ("markdown", "## 4. QA reports y roundtrip del HDF5 final"),
    (
        "code",
        "for r in rows:\n"
        "    gal = manifest.galaxy_id(r.snapshot, r.subhalo_id)\n"
        "    qa = json.loads((DATA / 'output/qa_reports' / f'{gal}_v0.json').read_text())\n"
        "    entry = validate_dataset_entry(DATA / 'output/dataset_entries' / f'{gal}_v0.h5')\n"
        "    print(gal)\n"
        "    print('  status:', qa['status'], '| flags:', qa['flags'])\n"
        "    print('  conservación de masa:', f\"{qa['mass_conservation_error']:.2e}\")\n"
        "    print('  fracciones (en M_valid):',\n"
        "          {k: round(v, 3) for k, v in qa['fractions_recovered'].items()})\n"
        "    print('  roundtrip OK, n_valid =', entry['n_valid'])",
    ),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src)
        for kind, src in CELLS
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "language": "python", "display_name": "Python 3"}
    client = NotebookClient(nb, timeout=600, resources={"metadata": {"path": str(HERE)}})
    client.execute()
    out = HERE / "00_smoke_test.ipynb"
    nbf.write(nb, out)
    print(f"notebook ejecutado y guardado en {out}")


if __name__ == "__main__":
    main()
