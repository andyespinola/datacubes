from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aperturenet_labels.config import PipelineConfig  # noqa: E402
from aperturenet_labels.io.assets import assets_for_galaxy  # noqa: E402
from aperturenet_labels.io.cube_reader import read_pipe3d_maps  # noqa: E402


CLASS_COLORS = {
    "bulge": "#d95f02",
    "disk": "#1b9e77",
    "bar": "#e6ab02",
    "arm": "#e7298a",
    "halo": "#7570b3",
}


def _log_image(value: np.ndarray) -> np.ndarray:
    positive = value[np.isfinite(value) & (value > 0)]
    if positive.size == 0:
        return np.zeros_like(value, dtype=np.float32)
    floor = np.nanpercentile(positive, 2.0)
    return np.log10(np.clip(value, floor, None))


def _show(ax: plt.Axes, image: np.ndarray, title: str, cmap: str = "magma", vmin: float | None = None, vmax: float | None = None) -> None:
    ax.imshow(image, origin="lower", cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _visualize_galaxy(galaxy_id: str, outputs_dir: Path, figure_dir: Path, config: PipelineConfig) -> dict:
    galaxy_dir = outputs_dir / galaxy_id
    labels_path = galaxy_dir / "labels2d_v0.npz"
    mask_path = galaxy_dir / "M_valid_v0.npz"
    qa_path = galaxy_dir / "qa_report_v0.json"
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)
    if not qa_path.exists():
        raise FileNotFoundError(qa_path)

    labels = np.load(labels_path, allow_pickle=True)
    mask = np.load(mask_path)["M_valid"].astype(bool)
    qa = json.loads(qa_path.read_text())
    metadata = json.loads(str(labels["metadata_json"]))
    class_names = [str(name) for name in labels["class_names"]]
    y = labels["Y_mass_psf"]
    total_mass = labels["total_mass_per_spaxel"]
    n_eff = labels["n_eff"]
    confidence = np.max(y, axis=2)
    winner = np.argmax(y, axis=2).astype(float)
    winner[~mask] = np.nan

    assets = assets_for_galaxy(galaxy_id, config.data)
    pipe3d_maps = read_pipe3d_maps(assets.maps_path)
    pipe3d_mass = pipe3d_maps["stellar_mass_density_log10"]

    colors = [CLASS_COLORS.get(name, "#999999") for name in class_names]
    class_cmap = ListedColormap(colors)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6.8), constrained_layout=True)
    fig.suptitle(
        (
            f"{galaxy_id} | status={qa['status']} | valid={qa['fraction_spaxels_valid']:.2f} | "
            f"mean max P={qa['mean_max_probability']:.2f} | particles={qa['n_particles_used']:,} | "
            f"align k={metadata.get('sky_alignment_rot90_k', 'off')} score={metadata.get('sky_alignment_score', 0.0):.2f}"
        ),
        fontsize=12,
    )

    _show(axes[0, 0], pipe3d_mass, "pyPipe3D log stellar mass", cmap="viridis")
    _show(axes[0, 1], _log_image(total_mass), "projected log mass", cmap="magma")
    _show(axes[0, 2], mask.astype(float), "valid mask", cmap="gray", vmin=0, vmax=1)
    im = axes[0, 3].imshow(winner, origin="lower", cmap=class_cmap, interpolation="nearest", vmin=-0.5, vmax=len(class_names) - 0.5)
    axes[0, 3].set_title("dominant class", fontsize=9)
    axes[0, 3].set_xticks([])
    axes[0, 3].set_yticks([])
    cbar = fig.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04, ticks=range(len(class_names)))
    cbar.ax.set_yticklabels(class_names, fontsize=8)
    _show(axes[0, 4], np.where(mask, confidence, np.nan), "max probability", cmap="cividis", vmin=0, vmax=1)

    for idx, name in enumerate(class_names):
        _show(axes[1, idx], np.where(mask, y[:, :, idx], np.nan), f"P({name})", cmap="magma", vmin=0, vmax=1)

    out_path = figure_dir / f"{galaxy_id}_visual_validation.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

    return {
        "galaxy_id": galaxy_id,
        "figure": str(out_path),
        "fractions_recovered_valid_mass": qa["fractions_recovered_valid_mass"],
        "fractions_catalog": qa["fractions_catalog"],
        "n_eff_median_valid": float(np.nanmedian(n_eff[mask])) if np.any(mask) else 0.0,
        "sky_alignment": {key: metadata[key] for key in sorted(metadata) if key.startswith("sky_alignment")},
    }


def _write_fraction_summary(rows: list[dict], figure_dir: Path) -> Path:
    class_names = ["bulge", "disk", "bar", "arm", "halo"]
    fig, axes = plt.subplots(1, len(rows), figsize=(6.2 * len(rows), 4.0), sharey=True, constrained_layout=True)
    if len(rows) == 1:
        axes = [axes]
    x = np.arange(len(class_names))
    width = 0.38
    for ax, row in zip(axes, rows):
        recovered = [row["fractions_recovered_valid_mass"].get(name, 0.0) for name in class_names]
        catalog = [
            row["fractions_catalog"].get("bulge", 0.0),
            row["fractions_catalog"].get("disk", 0.0),
            0.0,
            0.0,
            row["fractions_catalog"].get("halo", 0.0),
        ]
        ax.bar(x - width / 2, catalog, width, label="catalog", color="#8da0cb")
        ax.bar(x + width / 2, recovered, width, label="projected valid", color="#fc8d62")
        ax.set_title(row["galaxy_id"], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("mass fraction")
    axes[0].legend(frameon=False)
    out_path = figure_dir / "fraction_summary.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def _write_visual_summary_csv(rows: list[dict], figure_dir: Path) -> Path:
    out_path = figure_dir / "visual_qa_summary.csv"
    fieldnames = [
        "galaxy_id",
        "figure",
        "n_eff_median_valid",
        "align_rot90_k",
        "align_score",
        "catalog_bulge",
        "catalog_disk",
        "catalog_halo",
        "recovered_bulge",
        "recovered_disk",
        "recovered_bar",
        "recovered_arm",
        "recovered_halo",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            catalog = row["fractions_catalog"]
            recovered = row["fractions_recovered_valid_mass"]
            alignment = row.get("sky_alignment", {})
            writer.writerow(
                {
                    "galaxy_id": row["galaxy_id"],
                    "figure": row["figure"],
                    "n_eff_median_valid": row["n_eff_median_valid"],
                    "align_rot90_k": alignment.get("sky_alignment_rot90_k", ""),
                    "align_score": alignment.get("sky_alignment_score", ""),
                    "catalog_bulge": catalog.get("bulge", ""),
                    "catalog_disk": catalog.get("disk", ""),
                    "catalog_halo": catalog.get("halo", ""),
                    "recovered_bulge": recovered.get("bulge", ""),
                    "recovered_disk": recovered.get("disk", ""),
                    "recovered_bar": recovered.get("bar", ""),
                    "recovered_arm": recovered.get("arm", ""),
                    "recovered_halo": recovered.get("halo", ""),
                }
            )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create visual QA figures for soft_labels_generation_v2_1 outputs.")
    parser.add_argument("--outputs-dir", default="outputs_visual", help="Pipeline output directory relative to project root.")
    parser.add_argument("--figure-dir", default="", help="Figure directory. Defaults to <outputs-dir>/figures.")
    parser.add_argument("--galaxy-id", action="append", default=[], help="Galaxy id to plot; can repeat. Defaults to all local galaxies.")
    args = parser.parse_args()

    config = PipelineConfig.from_yaml()
    outputs_dir = (PROJECT_DIR / args.outputs_dir).resolve() if not Path(args.outputs_dir).is_absolute() else Path(args.outputs_dir)
    figure_dir = Path(args.figure_dir) if args.figure_dir else outputs_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    galaxy_ids = args.galaxy_id or list(config.data.local_galaxy_ids)

    rows = [_visualize_galaxy(galaxy_id, outputs_dir, figure_dir, config) for galaxy_id in galaxy_ids]
    summary_path = _write_fraction_summary(rows, figure_dir)
    summary_csv = _write_visual_summary_csv(rows, figure_dir)
    manifest = {
        "outputs_dir": str(outputs_dir),
        "figure_dir": str(figure_dir),
        "summary_figure": str(summary_path),
        "summary_csv": str(summary_csv),
        "galaxies": rows,
    }
    manifest_path = figure_dir / "visual_validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
