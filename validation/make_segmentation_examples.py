from __future__ import annotations

import argparse
import csv
import json
import math
import re
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
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".npz"}


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
    image_path: Path | None
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


def _id_tokens(row: dict[str, str]) -> list[str]:
    tokens = []
    for key in ("canonical_id", "unit_id", "galaxy_id"):
        value = (row.get(key) or "").strip()
        if value:
            tokens.append(value)
    snapshot = (row.get("snapshot") or "").strip()
    subhalo_id = (row.get("subhalo_id") or "").strip()
    view = (row.get("view") or "").strip()
    if snapshot and subhalo_id:
        tokens.append(f"TNG50-{int(float(snapshot))}-{int(float(subhalo_id))}")
        if view:
            tokens.append(f"TNG50-{int(float(snapshot))}-{int(float(subhalo_id))}-{int(float(view))}")
    seen = set()
    unique = []
    for token in tokens:
        if token not in seen:
            unique.append(token)
            seen.add(token)
    return unique


def _image_match_tokens(path: Path) -> list[str]:
    text = path.as_posix()
    matches = re.findall(r"TNG50-\d+-\d+(?:-\d+)?(?:-\d+)?", text)
    return sorted(set(matches), key=lambda token: -len(token))


def build_image_index(root: Path | None) -> dict[str, Path]:
    if root is None or not root.exists():
        return {}
    index: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        for token in _image_match_tokens(path):
            current = index.get(token)
            if current is None or _image_rank(path) < _image_rank(current):
                index[token] = path
    return index


def _image_rank(path: Path) -> tuple[int, int, str]:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg"}:
        suffix_rank = 0
    elif suffix == ".npz":
        suffix_rank = 1
    else:
        suffix_rank = 2
    return suffix_rank, len(path.name), path.as_posix()


def match_image_path(row: dict[str, str], image_index: dict[str, Path]) -> Path | None:
    for token in sorted(_id_tokens(row), key=lambda item: -len(item)):
        if token in image_index:
            return image_index[token]
        # ImagesMangGenerator often writes canonical IDs as TNG...-ifu_v{view}.
        matches = [path for key, path in image_index.items() if key.startswith(token)]
        if matches:
            return sorted(matches, key=_image_rank)[0]
    return None


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
    image_index: dict[str, Path],
    require_image: bool,
    max_examples: int,
) -> list[Candidate]:
    matched_rows = _read_csv(matched_units)
    kinematic_by_id = _kinematic_index(kinematic_units)
    candidates: list[Candidate] = []
    for row in matched_rows:
        canonical_id = (row.get("canonical_id") or "").strip()
        if not canonical_id:
            continue
        image_path = match_image_path(row, image_index)
        if require_image and image_path is None:
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
                image_path=image_path,
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


def _stretch_rgb(rgb: np.ndarray, percentile: float = 99.5, stretch: str = "asinh") -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    low = float(np.nanpercentile(rgb, 1.0))
    high = float(np.nanpercentile(rgb, percentile))
    if high <= low:
        high = float(np.nanmax(rgb))
    if high <= low:
        return np.zeros_like(rgb, dtype=np.float32)
    scaled = np.clip((rgb - low) / (high - low), 0.0, None)
    if stretch == "sqrt":
        scaled = np.sqrt(scaled)
    elif stretch == "linear":
        pass
    else:
        scaled = np.arcsinh(10.0 * scaled) / np.arcsinh(10.0)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def _load_npz_rgb(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if "rgb" in payload:
            image = np.asarray(payload["rgb"], dtype=np.float32)
        elif "preview" in payload:
            image = np.asarray(payload["preview"], dtype=np.float32)
        elif "image" in payload:
            image = np.asarray(payload["image"], dtype=np.float32)
        else:
            raise KeyError(f"No encontré image/rgb/preview en {path}")
    if image.ndim == 3 and image.shape[0] == 3:
        # ImagesMangGenerator stores bands as g,r,i; use i,r,g for display.
        image = np.stack((image[2], image[1], image[0]), axis=-1)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Imagen NPZ inválida en {path}: shape={image.shape}")
    return _stretch_rgb(image[:, :, :3])


def _load_external_rgb(path: Path) -> np.ndarray:
    import matplotlib.image as mpimg

    if path.suffix.lower() == ".npz":
        return _load_npz_rgb(path)
    raw = mpimg.imread(path)
    image = np.asarray(raw)
    if image.dtype.kind in {"u", "i"}:
        image = image.astype(np.float32) / np.iinfo(image.dtype).max
    else:
        image = image.astype(np.float32)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    if image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Imagen inválida en {path}: shape={image.shape}")
    return np.clip(image[:, :, :3], 0.0, 1.0).astype(np.float32)


def _resize_rgb_to_shape(rgb: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if rgb.shape[:2] == shape:
        return rgb
    from scipy.ndimage import zoom

    zoom_y = shape[0] / rgb.shape[0]
    zoom_x = shape[1] / rgb.shape[1]
    resized = zoom(rgb, (zoom_y, zoom_x, 1.0), order=1)
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def _unsegmented_image(label_path: Path, labels: LabelMaps, image_weight: str, external_image: Path | None) -> np.ndarray:
    if external_image is not None:
        try:
            return _resize_rgb_to_shape(_load_external_rgb(external_image), labels.valid_mask.shape)
        except Exception as exc:
            raise RuntimeError(f"No pude cargar la imagen emparejada {external_image}") from exc

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
    if image.ndim == 3:
        return 0.0, 1.0
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


def _masked_image_for_display(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        output = np.ones_like(image, dtype=np.float32)
        output[mask] = image[mask]
        return output
    return np.where(mask, image, np.nan)


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
    image = _masked_image_for_display(
        _unsegmented_image(candidate.label_path, labels, image_weight, candidate.image_path),
        display_mask,
    )
    vmin, vmax = _image_limits(image, display_mask)
    segmented = _hard_rgb(labels, threshold, display_mask)
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / f"{candidate.canonical_id}.segmentation.png"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), constrained_layout=True)
    if image.ndim == 3:
        axes[0].imshow(image[crop], origin="lower")
    else:
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
        image = _masked_image_for_display(
            _unsegmented_image(candidate.label_path, labels, image_weight, candidate.image_path),
            display_mask,
        )
        vmin, vmax = _image_limits(image, display_mask)
        if image.ndim == 3:
            axes[row_idx, 0].imshow(image[crop], origin="lower")
        else:
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
        "image_path",
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
                "image_path": str(candidate.image_path or ""),
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
                f"- `image_path`: `{candidate.image_path or ''}`",
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
    image_root = Path(args.image_root).expanduser() if args.image_root else None
    image_index = build_image_index(image_root)
    if args.require_image and not image_index:
        raise SystemExit(f"No encontré imágenes en --image-root={image_root}")
    candidates = select_candidates(
        matched_units=Path(args.matched_units).expanduser(),
        labels_dir=Path(args.labels_dir).expanduser(),
        kinematic_units=kinematic_units,
        mode=args.label_mode,
        threshold=args.dominant_threshold,
        min_bulge_pixels=args.min_bulge_pixels,
        min_disk_pixels=args.min_disk_pixels,
        min_component_fraction=args.min_component_fraction,
        image_index=image_index,
        require_image=args.require_image,
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
        "image_root": str(image_root or ""),
        "n_images_indexed": len(image_index),
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
    parser.add_argument("--image-root", default="/media/nuevo/output_imagenes")
    parser.add_argument("--require-image", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--label-mode", choices=("soft_mass", "soft_light"), default="soft_mass")
    parser.add_argument("--n-examples", type=int, default=40)
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
