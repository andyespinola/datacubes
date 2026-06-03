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
    component_fraction: float
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


def _central_component_mask(mask: np.ndarray) -> tuple[np.ndarray, float]:
    from scipy.ndimage import label

    mask = np.asarray(mask).astype(bool)
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool), 0.0
    labeled, n_labels = label(mask)
    if n_labels <= 1:
        return mask.copy(), 1.0

    h, w = mask.shape
    center_y = (h - 1) / 2.0
    center_x = (w - 1) / 2.0
    center_label = int(labeled[int(round(center_y)), int(round(center_x))])
    if center_label > 0:
        selected = labeled == center_label
        return selected, float(np.count_nonzero(selected) / np.count_nonzero(mask))

    best_label = 1
    best_distance = float("inf")
    for label_id in range(1, n_labels + 1):
        yy, xx = np.nonzero(labeled == label_id)
        if yy.size == 0:
            continue
        distance = float((np.median(yy) - center_y) ** 2 + (np.median(xx) - center_x) ** 2)
        if distance < best_distance:
            best_distance = distance
            best_label = label_id
    selected = labeled == best_label
    return selected, float(np.count_nonzero(selected) / np.count_nonzero(mask))


def _segmentation_stats(labels: LabelMaps, threshold: float, analysis_mask: np.ndarray) -> tuple[dict[str, float], dict[str, int], float]:
    physical_indices = [labels.indices[name] for name in PHYSICAL_CLASSES]
    physical = labels.soft[physical_indices]
    pmax = np.max(physical, axis=0)
    argmax = np.argmax(physical, axis=0)
    analysis_mask = np.asarray(analysis_mask).astype(bool)
    confident = analysis_mask & (pmax >= threshold)
    valid_total = max(int(np.count_nonzero(analysis_mask)), 1)
    fractions: dict[str, float] = {}
    pixels: dict[str, int] = {}
    for idx, name in enumerate(PHYSICAL_CLASSES):
        count = int(np.count_nonzero(confident & (argmax == idx)))
        pixels[name] = count
        fractions[name] = count / valid_total
    median_pmax = float(np.nanmedian(pmax[analysis_mask])) if np.any(analysis_mask) else 0.0
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
    min_component_fraction: float,
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
            analysis_mask, component_fraction = _central_component_mask(labels.valid_mask)
            if component_fraction < min_component_fraction:
                continue
            fractions, pixels, median_pmax = _segmentation_stats(labels, threshold, analysis_mask)
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
                component_fraction=component_fraction,
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


def _probability_rgb(labels: LabelMaps, display_mask: np.ndarray) -> np.ndarray:
    h, w = labels.valid_mask.shape
    rgb = np.ones((h, w, 3), dtype=np.float32)
    mix = np.zeros_like(rgb)
    physical = np.stack([labels.soft[labels.indices[name]] for name in PHYSICAL_CLASSES], axis=0)
    pmax = np.max(physical, axis=0)
    for name in PHYSICAL_CLASSES:
        mix += labels.soft[labels.indices[name], :, :, None] * CLASS_COLORS[name][None, None, :]
    alpha = np.clip(0.25 + 0.75 * pmax, 0.0, 1.0)
    display_mask = np.asarray(display_mask).astype(bool)
    alpha_valid = alpha[display_mask][:, None]
    rgb[display_mask] = (1.0 - alpha_valid) + alpha_valid * mix[display_mask]
    return np.clip(rgb, 0.0, 1.0)


def _hard_rgb(labels: LabelMaps, threshold: float, display_mask: np.ndarray) -> np.ndarray:
    h, w = labels.valid_mask.shape
    rgb = np.ones((h, w, 3), dtype=np.float32)
    display_mask = np.asarray(display_mask).astype(bool)
    physical = np.stack([labels.soft[labels.indices[name]] for name in PHYSICAL_CLASSES], axis=0)
    pmax = np.max(physical, axis=0)
    argmax = np.argmax(physical, axis=0)
    uncertain = display_mask & (pmax < threshold)
    rgb[uncertain] = CLASS_COLORS["uncertain"]
    for idx, name in enumerate(PHYSICAL_CLASSES):
        mask = display_mask & (pmax >= threshold) & (argmax == idx)
        rgb[mask] = CLASS_COLORS[name]
    return rgb


def _qa_path(label_path: Path) -> Path:
    return label_path.with_name(label_path.name.replace(".labels.npz", ".qa.npz"))


def _unsegmented_image(label_path: Path, labels: LabelMaps, image_weight: str) -> np.ndarray:
    qa_path = _qa_path(label_path)
    key_candidates = (
        ("observed_light_contributions", "observed_light_components")
        if image_weight == "light"
        else ("observed_mass_contributions", "observed_mass_components")
    )
    if qa_path.exists():
        with np.load(qa_path, allow_pickle=False) as data:
            for key in key_candidates:
                if key in data:
                    cube = np.asarray(data[key], dtype=np.float32)
                    if cube.ndim == 3 and cube.shape[0] >= len(labels.class_names):
                        physical_indices = [labels.indices[name] for name in PHYSICAL_CLASSES]
                        image = np.sum(cube[physical_indices], axis=0)
                        if np.nanmax(image) > np.nanmin(image):
                            return image.astype(np.float32)
    physical_indices = [labels.indices[name] for name in PHYSICAL_CLASSES]
    return np.sum(labels.soft[physical_indices], axis=0).astype(np.float32)


def _crop_slices(mask: np.ndarray, pad: int = 3) -> tuple[slice, slice]:
    yy, xx = np.nonzero(mask)
    if yy.size == 0:
        return slice(None), slice(None)
    y0 = max(int(yy.min()) - pad, 0)
    y1 = min(int(yy.max()) + pad + 1, mask.shape[0])
    x0 = max(int(xx.min()) - pad, 0)
    x1 = min(int(xx.max()) + pad + 1, mask.shape[1])
    return slice(y0, y1), slice(x0, x1)


def _image_limits(image: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    values = np.asarray(image, dtype=np.float32)[mask & np.isfinite(image)]
    values = values[values > 0]
    if values.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(values, 2))
    vmax = float(np.nanpercentile(values, 99))
    if vmax <= vmin:
        vmax = float(np.nanmax(values))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


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


def render_candidate(candidate: Candidate, outdir: Path, mode: str, threshold: float, image_weight: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = load_label_maps(candidate.label_path, mode)
    display_mask, _ = _central_component_mask(labels.valid_mask)
    crop = _crop_slices(display_mask)
    image = _unsegmented_image(candidate.label_path, labels, image_weight)
    image = np.where(display_mask, image, np.nan)
    vmin, vmax = _image_limits(image, display_mask)
    segmented = _hard_rgb(labels, threshold, display_mask)
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / f"{candidate.canonical_id}.segmentation.png"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), constrained_layout=True)
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("white")
    axes[0].imshow(image[crop], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("Imagen sin segmentar")
    axes[1].imshow(segmented[crop], origin="lower")
    axes[1].set_title(f"Segmentación p>={threshold:.2f}")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(_candidate_title(candidate), fontsize=10)
    fig.legend(handles=_legend_handles(), loc="lower center", ncols=6, frameon=False, fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def render_montage(candidates: list[Candidate], outdir: Path, mode: str, threshold: float, image_weight: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / "segmentation_examples_montage.png"
    rows = max(len(candidates), 1)
    fig, axes = plt.subplots(rows, 2, figsize=(7.2, 3.1 * rows), squeeze=False, constrained_layout=True)
    for row_idx, candidate in enumerate(candidates):
        labels = load_label_maps(candidate.label_path, mode)
        display_mask, _ = _central_component_mask(labels.valid_mask)
        crop = _crop_slices(display_mask)
        image = np.where(display_mask, _unsegmented_image(candidate.label_path, labels, image_weight), np.nan)
        vmin, vmax = _image_limits(image, display_mask)
        cmap = plt.get_cmap("gray").copy()
        cmap.set_bad("white")
        axes[row_idx, 0].imshow(image[crop], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row_idx, 0].set_title(f"{candidate.canonical_id}\nimagen sin segmentar", fontsize=9)
        axes[row_idx, 1].imshow(_hard_rgb(labels, threshold, display_mask)[crop], origin="lower")
        axes[row_idx, 1].set_title("segmentación", fontsize=9)
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
        "component_fraction",
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
                "component_fraction": candidate.component_fraction,
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
        "| canonical_id | score | A | B | coherence | V/sigma ratio | sigma ratio | component frac | bulge frac | disk frac | bar frac | arms frac | other frac |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in candidates:
        lines.append(
            f"| `{candidate.canonical_id}` | {_fmt(candidate.score, 2)} | {candidate.test_a or 'N/A'} | {candidate.test_b or 'N/A'} | "
            f"{_fmt(candidate.coherence_score, 2)} | {_fmt(candidate.v_over_sigma_ratio, 2)} | {_fmt(candidate.sigma_ratio, 2)} | "
            f"{_fmt(candidate.component_fraction)} | "
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
            "Estas figuras no deben presentarse como casos escogidos manualmente. La selección queda trazada por `selected_segmentation_examples.csv`, donde se reportan los criterios usados para priorizar ejemplos con segmentación limpia, componente central conectada y validación cinemática favorable.",
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
        min_component_fraction=args.min_component_fraction,
        max_examples=args.n_examples,
    )
    if not candidates:
        raise SystemExit("No encontré candidatos que cumplan los criterios de selección")

    image_paths = [
        render_candidate(candidate, outdir, args.label_mode, args.dominant_threshold, args.image_weight)
        for candidate in candidates
    ]
    montage_path = render_montage(candidates, outdir, args.label_mode, args.dominant_threshold, args.image_weight)
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
    parser.add_argument("--min-component-fraction", type=float, default=0.80)
    parser.add_argument("--image-weight", choices=("light", "mass"), default="light")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
