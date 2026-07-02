from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb
import numpy as np
from scipy.ndimage import label as connected_label


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


def _load_outputs(galaxy_id: str, outputs_dir: Path) -> tuple[dict, dict, np.lib.npyio.NpzFile, np.ndarray]:
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
    return qa, metadata, labels, mask


def _log_image(value: np.ndarray) -> np.ndarray:
    positive = value[np.isfinite(value) & (value > 0)]
    if positive.size == 0:
        return np.zeros_like(value, dtype=np.float32)
    floor = np.nanpercentile(positive, 2.0)
    return np.log10(np.clip(value, floor, None))


def _normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1.0e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=2)
    return entropy / np.log(probabilities.shape[2])


def _top_margin(probabilities: np.ndarray) -> np.ndarray:
    sorted_prob = np.sort(probabilities, axis=2)
    return sorted_prob[:, :, -1] - sorted_prob[:, :, -2]


def _probability_rgb(probabilities: np.ndarray, mask: np.ndarray, class_names: list[str]) -> np.ndarray:
    colors = np.asarray([to_rgb(CLASS_COLORS.get(name, "#999999")) for name in class_names], dtype=np.float32)
    rgb = np.tensordot(probabilities, colors, axes=([2], [0]))
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb[~mask] = 1.0
    return rgb


def _component_stats(winner: np.ndarray, mask: np.ndarray, class_names: list[str]) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    for idx, name in enumerate(class_names):
        class_mask = mask & (winner == idx)
        labeled, n_labels = connected_label(class_mask)
        if n_labels == 0:
            stats[name] = {
                "n_spaxels": 0,
                "n_components": 0,
                "largest_component_spaxels": 0,
                "largest_component_fraction": 0.0,
            }
            continue
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        largest = int(np.max(counts))
        n_spaxels = int(np.count_nonzero(class_mask))
        stats[name] = {
            "n_spaxels": n_spaxels,
            "n_components": int(n_labels),
            "largest_component_spaxels": largest,
            "largest_component_fraction": float(largest / max(n_spaxels, 1)),
        }
    return stats


def _show(
    ax: plt.Axes,
    image: np.ndarray,
    title: str,
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
):
    im = ax.imshow(image, origin="lower", cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _validate_galaxy(
    galaxy_id: str,
    outputs_dir: Path,
    figure_dir: Path,
    config: PipelineConfig,
    confidence_threshold: float,
    margin_threshold: float,
) -> dict:
    qa, metadata, labels, mask = _load_outputs(galaxy_id, outputs_dir)
    class_names = [str(name) for name in labels["class_names"]]
    y = labels["Y_mass_psf"]
    total_mass = labels["total_mass_per_spaxel"]
    n_eff = labels["n_eff"]
    confidence = np.max(y, axis=2)
    winner = np.argmax(y, axis=2)
    margin = _top_margin(y)
    entropy = _normalized_entropy(y)
    ambiguous = mask & (margin < margin_threshold)
    low_confidence = mask & (confidence < confidence_threshold)

    assets = assets_for_galaxy(galaxy_id, config.data)
    pipe3d_mass = read_pipe3d_maps(assets.maps_path)["stellar_mass_density_log10"]

    colors = [CLASS_COLORS.get(name, "#999999") for name in class_names]
    class_cmap = ListedColormap(colors)
    winner_float = winner.astype(float)
    winner_float[~mask] = np.nan

    fig, axes = plt.subplots(3, 4, figsize=(14.5, 10.0), constrained_layout=True)
    fig.suptitle(
        (
            f"{galaxy_id} segmentation | status={qa['status']} | valid={np.mean(mask):.2f} | "
            f"mean max P={np.nanmean(confidence[mask]):.2f} | "
            f"low-conf={np.mean(low_confidence[mask]):.2f} | ambiguous={np.mean(ambiguous[mask]):.2f}"
        ),
        fontsize=12,
    )

    _show(axes[0, 0], pipe3d_mass, "pyPipe3D log mass", cmap="viridis")
    _show(axes[0, 1], _log_image(total_mass), "projected log mass", cmap="magma")
    im_seg = axes[0, 2].imshow(
        winner_float,
        origin="lower",
        cmap=class_cmap,
        interpolation="nearest",
        vmin=-0.5,
        vmax=len(class_names) - 0.5,
    )
    axes[0, 2].set_title("hard segmentation", fontsize=9)
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])
    cbar = fig.colorbar(im_seg, ax=axes[0, 2], fraction=0.046, pad=0.04, ticks=range(len(class_names)))
    cbar.ax.set_yticklabels(class_names, fontsize=8)
    axes[0, 3].imshow(_probability_rgb(y, mask, class_names), origin="lower", interpolation="nearest")
    axes[0, 3].set_title("soft class blend", fontsize=9)
    axes[0, 3].set_xticks([])
    axes[0, 3].set_yticks([])

    im_conf = _show(axes[1, 0], np.where(mask, confidence, np.nan), "max probability", cmap="cividis", vmin=0, vmax=1)
    fig.colorbar(im_conf, ax=axes[1, 0], fraction=0.046, pad=0.04)
    im_margin = _show(axes[1, 1], np.where(mask, margin, np.nan), "top1 - top2 margin", cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im_margin, ax=axes[1, 1], fraction=0.046, pad=0.04)
    im_entropy = _show(axes[1, 2], np.where(mask, entropy, np.nan), "normalized entropy", cmap="plasma", vmin=0, vmax=1)
    fig.colorbar(im_entropy, ax=axes[1, 2], fraction=0.046, pad=0.04)
    diagnostic_mask = np.zeros((*mask.shape, 3), dtype=np.float32)
    diagnostic_mask[mask] = (0.88, 0.88, 0.88)
    diagnostic_mask[low_confidence] = (0.86, 0.24, 0.12)
    diagnostic_mask[ambiguous] = (0.26, 0.45, 0.86)
    diagnostic_mask[low_confidence & ambiguous] = (0.55, 0.20, 0.70)
    diagnostic_mask[~mask] = (1.0, 1.0, 1.0)
    axes[1, 3].imshow(diagnostic_mask, origin="lower", interpolation="nearest")
    axes[1, 3].set_title("low confidence / ambiguous", fontsize=9)
    axes[1, 3].set_xticks([])
    axes[1, 3].set_yticks([])

    for ax, idx in zip(axes[2], [0, 1, 2, 4]):
        name = class_names[idx]
        im = _show(ax, np.where(mask, y[:, :, idx], np.nan), f"P({name})", cmap="magma", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_path = figure_dir / f"{galaxy_id}_segmentation_validation.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

    valid_confidence = confidence[mask]
    valid_margin = margin[mask]
    valid_entropy = entropy[mask]
    components = _component_stats(winner, mask, class_names)
    return {
        "galaxy_id": galaxy_id,
        "figure": str(out_path),
        "qa_status": qa["status"],
        "qa_flags": qa["flags"],
        "n_valid_spaxels": int(np.count_nonzero(mask)),
        "fraction_spaxels_valid": float(np.mean(mask)),
        "mean_confidence": float(np.nanmean(valid_confidence)),
        "median_confidence": float(np.nanmedian(valid_confidence)),
        "p10_confidence": float(np.nanpercentile(valid_confidence, 10.0)),
        "mean_margin": float(np.nanmean(valid_margin)),
        "median_margin": float(np.nanmedian(valid_margin)),
        "mean_entropy": float(np.nanmean(valid_entropy)),
        "low_confidence_threshold": float(confidence_threshold),
        "low_confidence_fraction": float(np.mean(low_confidence[mask])),
        "ambiguous_margin_threshold": float(margin_threshold),
        "ambiguous_fraction": float(np.mean(ambiguous[mask])),
        "n_eff_median_valid": float(np.nanmedian(n_eff[mask])),
        "fractions_recovered_valid_mass": qa["fractions_recovered_valid_mass"],
        "fractions_catalog": qa["fractions_catalog"],
        "components": components,
        "potential_status": qa["phase_a_diagnostics"].get("potential_status", ""),
        "epsilon_definition": qa["phase_a_diagnostics"].get("epsilon_definition", ""),
        "sky_alignment_rot90_k": metadata.get("sky_alignment_rot90_k", ""),
        "sky_alignment_score": metadata.get("sky_alignment_score", ""),
    }


def _write_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = [
        "galaxy_id",
        "qa_status",
        "n_valid_spaxels",
        "fraction_spaxels_valid",
        "mean_confidence",
        "median_confidence",
        "p10_confidence",
        "mean_margin",
        "median_margin",
        "mean_entropy",
        "low_confidence_fraction",
        "ambiguous_fraction",
        "n_eff_median_valid",
        "potential_status",
        "epsilon_definition",
        "sky_alignment_rot90_k",
        "sky_alignment_score",
        "recovered_bulge",
        "recovered_disk",
        "recovered_bar",
        "recovered_arm",
        "recovered_halo",
        "figure",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            recovered = row["fractions_recovered_valid_mass"]
            writer.writerow(
                {
                    **{key: row.get(key, "") for key in fieldnames if key not in {"recovered_bulge", "recovered_disk", "recovered_bar", "recovered_arm", "recovered_halo"}},
                    "recovered_bulge": recovered.get("bulge", 0.0),
                    "recovered_disk": recovered.get("disk", 0.0),
                    "recovered_bar": recovered.get("bar", 0.0),
                    "recovered_arm": recovered.get("arm", 0.0),
                    "recovered_halo": recovered.get("halo", 0.0),
                }
            )


def _write_markdown(rows: list[dict], output_path: Path) -> None:
    lines = [
        "# Segmentation visual validation",
        "",
        "This report summarizes hard segmentation diagnostics derived from the mass-PSF soft-label tensor.",
        "",
        "| galaxy_id | status | valid spaxels | mean max P | low-conf frac | ambiguous frac | median N_eff | notes |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        notes = []
        if row["potential_status"] != "loaded":
            notes.append("potential fallback")
        if row["ambiguous_fraction"] > 0.50:
            notes.append("high soft mixing")
        dominant_hard = max(row["components"].items(), key=lambda item: item[1]["n_spaxels"])
        dominant_fraction = dominant_hard[1]["n_spaxels"] / max(row["n_valid_spaxels"], 1)
        if dominant_fraction > 0.95:
            notes.append(f"hard argmax dominated by {dominant_hard[0]}")
        if row["fractions_recovered_valid_mass"].get("bar", 0.0) > 0:
            notes.append("bar recovered")
        if not notes:
            notes.append("stable visual QA")
        lines.append(
            "| {galaxy_id} | {qa_status} | {n_valid_spaxels} | {mean_confidence:.3f} | "
            "{low_confidence_fraction:.3f} | {ambiguous_fraction:.3f} | "
            "{n_eff_median_valid:.2f} | {notes} |".format(
                **row,
                notes=", ".join(notes),
            )
        )
    lines.extend(
        [
            "",
            "Thresholds:",
            "",
            "- low confidence: `max(P_class) < low_confidence_threshold`",
            "- ambiguous: `top1_probability - top2_probability < ambiguous_margin_threshold`",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create segmentation-specific visual QA for soft labels.")
    parser.add_argument("--outputs-dir", default="outputs_visual", help="Pipeline output directory relative to project root.")
    parser.add_argument("--figure-dir", default="", help="Defaults to <outputs-dir>/figures/segmentation.")
    parser.add_argument("--galaxy-id", action="append", default=[], help="Galaxy id to plot; can repeat. Defaults to all local galaxies.")
    parser.add_argument("--confidence-threshold", type=float, default=0.45, help="Threshold for low-confidence valid spaxels.")
    parser.add_argument("--margin-threshold", type=float, default=0.15, help="Threshold for top1-top2 ambiguity.")
    args = parser.parse_args()

    config = PipelineConfig.from_yaml()
    outputs_dir = (PROJECT_DIR / args.outputs_dir).resolve() if not Path(args.outputs_dir).is_absolute() else Path(args.outputs_dir)
    figure_dir = Path(args.figure_dir) if args.figure_dir else outputs_dir / "figures" / "segmentation"
    figure_dir.mkdir(parents=True, exist_ok=True)
    galaxy_ids = args.galaxy_id or list(config.data.local_galaxy_ids)

    rows = [
        _validate_galaxy(
            galaxy_id,
            outputs_dir,
            figure_dir,
            config,
            args.confidence_threshold,
            args.margin_threshold,
        )
        for galaxy_id in galaxy_ids
    ]
    csv_path = figure_dir / "segmentation_validation_summary.csv"
    json_path = figure_dir / "segmentation_validation_summary.json"
    md_path = figure_dir / "segmentation_validation_report.md"
    _write_csv(rows, csv_path)
    json_path.write_text(json.dumps({"outputs_dir": str(outputs_dir), "figure_dir": str(figure_dir), "galaxies": rows}, indent=2, sort_keys=True))
    _write_markdown(rows, md_path)
    print(json.dumps({"summary_csv": str(csv_path), "summary_json": str(json_path), "report": str(md_path), "galaxies": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
