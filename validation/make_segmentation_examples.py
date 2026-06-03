from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CLASS_ALIASES = {
    "bulge": ("bulge", "bulbo"),
    "disk": ("disk", "disco"),
    "bar": ("bar", "barra"),
    "arms": ("arm", "arms", "brazos"),
    "other": ("other",),
    "invalid": ("invalid", "no_valido", "no_valido"),
    "uncertain": ("uncertain", "incierto"),
}
DISPLAY_NAMES = {
    "bulge": "Bulbo",
    "disk": "Disco",
    "bar": "Barra",
    "arms": "Brazos",
    "other": "Other",
}
CLASS_COLORS = {
    "bulge": np.array([0.89, 0.29, 0.20]),
    "disk": np.array([0.12, 0.47, 0.71]),
    "bar": np.array([0.83, 0.18, 0.58]),
    "arms": np.array([0.10, 0.66, 0.45]),
    "other": np.array([0.50, 0.50, 0.56]),
    "uncertain": np.array([0.88, 0.88, 0.88]),
    "invalid": np.array([1.0, 1.0, 1.0]),
}
PHYSICAL_CLASSES = ("bulge", "disk", "bar", "arms", "other")


@dataclass(frozen=True, slots=True)
class LabelMaps:
    soft: np.ndarray
    valid_mask: np.ndarray
    class_names: list[str]
    indices: dict[str, int]


@dataclass(frozen=True, slots=True)
class Candidate:
    canonical_id: str
    galaxy_id: str
    unit_id: str
    view: str
    label_path: Path
    score: float
    class_fractions: dict[str, float]
    class_pixels: dict[str, int]
    median_pmax: float
    coherence_score: float | None
    v_over_sigma_ratio: float | None
    sigma_ratio: float | None
    test_a: str
    test_b: str
    passes: bool


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _decode_class_names(raw: np.ndarray) -> list[str]:
    names: list[str] = []
    for value in raw.tolist():
        if isinstance(value, bytes):
            names.append(value.decode("utf-8"))
        else:
            names.append(str(value))
    return names


def _class_indices(class_names: list[str]) -> dict[str, int]:
    lower = [name.lower() for name in class_names]
    indices: dict[str, int] = {}
    for canonical, aliases in CLASS_ALIASES.items():
        matches = [lower.index(alias) for alias in aliases if alias in lower]
        if matches:
            indices[canonical] = matches[0]
    missing = [name for name in PHYSICAL_CLASSES if name not in indices]
    if missing:
        raise KeyError(f"Faltan clases {missing} en class_names={class_names}")
    return indices


def load_label_maps(path: str | Path, mode: str) -> LabelMaps:
    with np.load(path, allow_pickle=False) as data:
        class_names = _decode_class_names(data["class_names"])
        if mode not in data:
            raise KeyError(f"{mode} no existe en {path}")
        soft = np.asarray(data[mode], dtype=np.float32)
        valid_mask = np.asarray(data["valid_mask"]).astype(bool)
    return LabelMaps(soft=soft, valid_mask=valid_mask, class_names=class_names, indices=_class_indices(class_names))


def _segmentation_stats(labels: LabelMaps, threshold: float) -> tuple[dict[str, float], dict[str, int], float]:
    physical_indices = [labels.indices[name] for name in PHYSICAL_CLASSES]
    physical = labels.soft[physical_indices]
    pmax = np.max(physical, axis=0)
    argmax = np.argmax(physical, axis=0)
    confident = labels.valid_mask & (pmax >= threshold)
    valid_total = max(int(np.count_nonzero(labels.valid_mask)), 1)
    fractions: dict[str, float] = {}
    pixels: dict[str, int] = {}
    for idx, name in enumerate(PHYSICAL_CLASSES):
        count = int(np.count_nonzero(confident & (argmax == idx)))
        pixels[name] = count
        fractions[name] = count / valid_total
    median_pmax = float(np.nanmedian(pmax[labels.valid_mask])) if np.any(labels.valid_mask) else 0.0
    return fractions, pixels, median_pmax


def _label_path(labels_dir: Path, canonical_id: str) -> Path:
    return (labels_dir / canonical_id).with_suffix(".labels.npz")


def _kinematic_index(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    rows = _read_csv(path)
    return {(row.get("canonical_id") or ""): row for row in rows if row.get("canonical_id")}


def _candidate_score(row: dict[str, str], fractions: dict[str, float], pixels: dict[str, int], median_pmax: float) -> float:
    coherence = _finite_float(row.get("coherence_score")) or 0.0
    vratio = _finite_float(row.get("v_over_sigma_ratio")) or 0.0
    sigma_ratio = _finite_float(row.get("sigma_ratio")) or 0.0
    passes = 1.0 if _as_bool(row.get("passes")) else 0.0
    test_a = 1.0 if row.get("test_a_rotation") == "PASS" else 0.0
    test_b = 1.0 if row.get("test_b_dispersion") == "PASS" else 0.0
    class_balance = min(fractions.get("bulge", 0.0), 0.35) + min(fractions.get("disk", 0.0), 0.45)
    substructure = 0.5 * min(fractions.get("bar", 0.0) + fractions.get("arms", 0.0), 0.20)
    size_bonus = min(math.log10(max(pixels.get("bulge", 0) + pixels.get("disk", 0), 1)) / 3.0, 1.0)
    return (
        3.0 * passes
        + 2.0 * coherence
        + 1.5 * test_a
        + 1.5 * test_b
        + 2.0 * class_balance
        + substructure
        + 1.5 * median_pmax
        + 0.4 * min(vratio, 2.0)
        + 0.4 * min(sigma_ratio, 2.0)
        + 0.5 * size_bonus
    )


def select_candidates(
    matched_units: Path,
    labels_dir: Path,
    kinematic_units: Path | None,
    mode: str,
    threshold: float,
    min_bulge_pixels: int,
    min_disk_pixels: int,
    max_examples: int,
) -> list[Candidate]:
    matched_rows = _read_csv(matched_units)
    kinematic_by_id = _kinematic_index(kinematic_units)
    candidates: list[Candidate] = []
    for row in matched_rows:
        canonical_id = (row.get("canonical_id") or "").strip()
        if not canonical_id:
            continue
        path = _label_path(labels_dir, canonical_id)
        if not path.exists():
            continue
        try:
            labels = load_label_maps(path, mode)
            fractions, pixels, median_pmax = _segmentation_stats(labels, threshold)
        except Exception:
            continue
        if pixels.get("bulge", 0) < min_bulge_pixels or pixels.get("disk", 0) < min_disk_pixels:
            continue
        kin = kinematic_by_id.get(canonical_id, {})
        score = _candidate_score(kin, fractions, pixels, median_pmax)
        candidates.append(
            Candidate(
                canonical_id=canonical_id,
                galaxy_id=row.get("galaxy_id", ""),
                unit_id=row.get("unit_id", ""),
                view=row.get("view", ""),
                label_path=path,
                score=score,
                class_fractions=fractions,
                class_pixels=pixels,
                median_pmax=median_pmax,
                coherence_score=_finite_float(kin.get("coherence_score")),
                v_over_sigma_ratio=_finite_float(kin.get("v_over_sigma_ratio")),
                sigma_ratio=_finite_float(kin.get("sigma_ratio")),
                test_a=kin.get("test_a_rotation", ""),
                test_b=kin.get("test_b_dispersion", ""),
                passes=_as_bool(kin.get("passes")),
            )
        )
    candidates.sort(key=lambda item: (-item.score, item.galaxy_id, item.canonical_id))
    selected: list[Candidate] = []
    used_galaxies: set[str] = set()
    for candidate in candidates:
        key = candidate.galaxy_id or candidate.canonical_id
        if key in used_galaxies:
            continue
        selected.append(candidate)
        used_galaxies.add(key)
        if len(selected) >= max_examples:
            return selected
    return selected


def _probability_rgb(labels: LabelMaps) -> np.ndarray:
    h, w = labels.valid_mask.shape
    rgb = np.ones((h, w, 3), dtype=np.float32)
    mix = np.zeros_like(rgb)
    physical = np.stack([labels.soft[labels.indices[name]] for name in PHYSICAL_CLASSES], axis=0)
    pmax = np.max(physical, axis=0)
    for name in PHYSICAL_CLASSES:
        mix += labels.soft[labels.indices[name], :, :, None] * CLASS_COLORS[name][None, None, :]
    alpha = np.clip(0.25 + 0.75 * pmax, 0.0, 1.0)
    alpha_valid = alpha[labels.valid_mask][:, None]
    rgb[labels.valid_mask] = (1.0 - alpha_valid) + alpha_valid * mix[labels.valid_mask]
    return np.clip(rgb, 0.0, 1.0)


def _hard_rgb(labels: LabelMaps, threshold: float) -> np.ndarray:
    h, w = labels.valid_mask.shape
    rgb = np.ones((h, w, 3), dtype=np.float32)
    physical = np.stack([labels.soft[labels.indices[name]] for name in PHYSICAL_CLASSES], axis=0)
    pmax = np.max(physical, axis=0)
    argmax = np.argmax(physical, axis=0)
    uncertain = labels.valid_mask & (pmax < threshold)
    rgb[uncertain] = CLASS_COLORS["uncertain"]
    for idx, name in enumerate(PHYSICAL_CLASSES):
        mask = labels.valid_mask & (pmax >= threshold) & (argmax == idx)
        rgb[mask] = CLASS_COLORS[name]
    return rgb


def _probability_map(labels: LabelMaps, class_name: str) -> np.ndarray:
    return np.asarray(labels.soft[labels.indices[class_name]], dtype=np.float32)


def _legend_handles() -> list[Any]:
    import matplotlib.patches as mpatches

    handles = []
    for name in PHYSICAL_CLASSES:
        handles.append(mpatches.Patch(color=CLASS_COLORS[name], label=DISPLAY_NAMES[name]))
    handles.append(mpatches.Patch(color=CLASS_COLORS["uncertain"], label="Incierto"))
    return handles


def _candidate_title(candidate: Candidate) -> str:
    return (
        f"{candidate.canonical_id} | score={candidate.score:.2f} | "
        f"A={candidate.test_a or 'NA'} B={candidate.test_b or 'NA'}"
    )


def render_candidate(candidate: Candidate, outdir: Path, mode: str, threshold: float) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = load_label_maps(candidate.label_path, mode)
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / f"{candidate.canonical_id}.segmentation.png"
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), constrained_layout=True)
    axes[0].imshow(_probability_rgb(labels), origin="lower")
    axes[0].set_title("Componentes suaves")
    axes[1].imshow(_hard_rgb(labels, threshold), origin="lower")
    axes[1].set_title(f"Dominante p>={threshold:.2f}")
    im2 = axes[2].imshow(_probability_map(labels, "bulge"), origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title("P(bulbo)")
    im3 = axes[3].imshow(_probability_map(labels, "disk"), origin="lower", cmap="Blues", vmin=0.0, vmax=1.0)
    axes[3].set_title("P(disco)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(_candidate_title(candidate), fontsize=10)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02)
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.02)
    fig.legend(handles=_legend_handles(), loc="lower center", ncols=6, frameon=False, fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def render_montage(candidates: list[Candidate], outdir: Path, mode: str, threshold: float) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / "segmentation_examples_montage.png"
    rows = max(len(candidates), 1)
    fig, axes = plt.subplots(rows, 2, figsize=(7.2, 3.1 * rows), squeeze=False, constrained_layout=True)
    for row_idx, candidate in enumerate(candidates):
        labels = load_label_maps(candidate.label_path, mode)
        axes[row_idx, 0].imshow(_probability_rgb(labels), origin="lower")
        axes[row_idx, 0].set_title(f"{candidate.canonical_id}\ncomponentes suaves", fontsize=9)
        axes[row_idx, 1].imshow(_hard_rgb(labels, threshold), origin="lower")
        axes[row_idx, 1].set_title("segmentación dominante", fontsize=9)
        for ax in axes[row_idx]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.legend(handles=_legend_handles(), loc="lower center", ncols=6, frameon=False, fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _write_selected_csv(path: Path, candidates: list[Candidate]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "canonical_id",
        "galaxy_id",
        "unit_id",
        "view",
        "score",
        "median_pmax",
        "coherence_score",
        "v_over_sigma_ratio",
        "sigma_ratio",
        "test_a",
        "test_b",
        "passes",
        "label_path",
        *[f"frac_{name}" for name in PHYSICAL_CLASSES],
        *[f"pixels_{name}" for name in PHYSICAL_CLASSES],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            row: dict[str, Any] = {
                "canonical_id": candidate.canonical_id,
                "galaxy_id": candidate.galaxy_id,
                "unit_id": candidate.unit_id,
                "view": candidate.view,
                "score": candidate.score,
                "median_pmax": candidate.median_pmax,
                "coherence_score": candidate.coherence_score,
                "v_over_sigma_ratio": candidate.v_over_sigma_ratio,
                "sigma_ratio": candidate.sigma_ratio,
                "test_a": candidate.test_a,
                "test_b": candidate.test_b,
                "passes": candidate.passes,
                "label_path": str(candidate.label_path),
            }
            for name in PHYSICAL_CLASSES:
                row[f"frac_{name}"] = candidate.class_fractions.get(name, 0.0)
                row[f"pixels_{name}"] = candidate.class_pixels.get(name, 0)
            writer.writerow(row)
    return path


def _fmt(value: Any, digits: int = 3) -> str:
    number = _finite_float(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


def _write_markdown(path: Path, candidates: list[Candidate], image_paths: list[Path], montage_path: Path, selected_csv: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Galaxias representativas segmentadas",
        "",
        "Este reporte selecciona ejemplos de segmentación estructural con criterios reproducibles: alta coherencia cinemática, Test A/B favorables cuando están disponibles, suficiente número de spaxels dominados por bulbo y disco, y alta probabilidad dominante media.",
        "",
        f"![Montaje]({montage_path.name})",
        "",
        "## Resumen de selección",
        "",
        "| canonical_id | score | A | B | coherence | V/sigma ratio | sigma ratio | bulge frac | disk frac | bar frac | arms frac | other frac |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in candidates:
        lines.append(
            f"| `{candidate.canonical_id}` | {_fmt(candidate.score, 2)} | {candidate.test_a or 'N/A'} | {candidate.test_b or 'N/A'} | "
            f"{_fmt(candidate.coherence_score, 2)} | {_fmt(candidate.v_over_sigma_ratio, 2)} | {_fmt(candidate.sigma_ratio, 2)} | "
            f"{_fmt(candidate.class_fractions.get('bulge'))} | {_fmt(candidate.class_fractions.get('disk'))} | "
            f"{_fmt(candidate.class_fractions.get('bar'))} | {_fmt(candidate.class_fractions.get('arms'))} | "
            f"{_fmt(candidate.class_fractions.get('other'))} |"
        )
    lines.extend(["", "## Figuras individuales", ""])
    for candidate, image_path in zip(candidates, image_paths, strict=True):
        lines.extend(
            [
                f"### {candidate.canonical_id}",
                "",
                f"- `galaxy_id`: `{candidate.galaxy_id}`",
                f"- `unit_id`: `{candidate.unit_id}`",
                f"- `label_path`: `{candidate.label_path}`",
                "",
                f"![{candidate.canonical_id}]({image_path.name})",
                "",
            ]
        )
    lines.extend(
        [
            "## Archivos",
            "",
            f"- CSV de selección: `{selected_csv}`",
            f"- Montaje: `{montage_path}`",
            "",
            "## Nota para el manuscrito",
            "",
            "Estas figuras no deben presentarse como casos escogidos manualmente. La selección queda trazada por `selected_segmentation_examples.csv`, donde se reportan los criterios usados para priorizar ejemplos con segmentación limpia y validación cinemática favorable.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir).expanduser()
    kinematic_units = Path(args.kinematic_units).expanduser() if args.kinematic_units else None
    candidates = select_candidates(
        matched_units=Path(args.matched_units).expanduser(),
        labels_dir=Path(args.labels_dir).expanduser(),
        kinematic_units=kinematic_units,
        mode=args.label_mode,
        threshold=args.dominant_threshold,
        min_bulge_pixels=args.min_bulge_pixels,
        min_disk_pixels=args.min_disk_pixels,
        max_examples=args.n_examples,
    )
    if not candidates:
        raise SystemExit("No encontré candidatos que cumplan los criterios de selección")

    image_paths = [render_candidate(candidate, outdir, args.label_mode, args.dominant_threshold) for candidate in candidates]
    montage_path = render_montage(candidates, outdir, args.label_mode, args.dominant_threshold)
    selected_csv = _write_selected_csv(outdir / "selected_segmentation_examples.csv", candidates)
    report_path = _write_markdown(outdir / "segmentation_examples_report.md", candidates, image_paths, montage_path, selected_csv)
    summary = {
        "report": str(report_path),
        "montage": str(montage_path),
        "selected_csv": str(selected_csv),
        "n_examples": len(candidates),
        "images": [str(path) for path in image_paths],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select and render representative colored structural segmentation examples.")
    parser.add_argument("--matched-units", default="/home/aespinola/Documents/pythonprojects/datacubes/matched_assets/matched_units.csv")
    parser.add_argument("--labels-dir", default="/media/nuevo/structural_labels")
    parser.add_argument("--kinematic-units", default="/media/nuevo/structural_validations/kinematic_central_a10_b10/kinematic_validation_units.csv")
    parser.add_argument("--outdir", default="/media/nuevo/structural_validations/segmentation_examples")
    parser.add_argument("--label-mode", choices=("soft_mass", "soft_light"), default="soft_mass")
    parser.add_argument("--n-examples", type=int, default=4)
    parser.add_argument("--dominant-threshold", type=float, default=0.50)
    parser.add_argument("--min-bulge-pixels", type=int, default=10)
    parser.add_argument("--min-disk-pixels", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
