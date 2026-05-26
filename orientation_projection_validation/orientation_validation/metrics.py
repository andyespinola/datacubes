from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import rotate

from .config import ProjectionConfig
from .paths import ensure_structural_labeling_on_path

ensure_structural_labeling_on_path()

from labeling.constants import CLASS_INDEX, PHYSICAL_CLASS_INDICES, PHYSICAL_CLASS_NAMES  # noqa: E402


def rotate_tensor_to_reference(tensor: np.ndarray, angle_degrees: float) -> np.ndarray:
    compensated = np.zeros_like(tensor, dtype=np.float32)
    for idx in range(tensor.shape[0]):
        compensated[idx] = rotate(
            tensor[idx],
            angle=-float(angle_degrees),
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
        )
    return compensated


def probabilistic_iou(a: np.ndarray, b: np.ndarray) -> float:
    numerator = float(np.sum(np.minimum(a, b)))
    denominator = float(np.sum(np.maximum(a, b)))
    if denominator <= 0:
        return 1.0
    return numerator / denominator


def orientation_angle_from_key(key: str) -> float:
    if not key.startswith("q"):
        raise ValueError(f"Clave de orientación inválida: {key}")
    return float(int(key[1:]))


def compute_interorientation_metrics(
    products: dict[str, dict[str, np.ndarray]],
    projection_config: ProjectionConfig,
    variant: str | None = None,
) -> dict:
    variant = variant or projection_config.main_metric_variant
    orientation_keys = sorted(products)
    compensated = {
        key: rotate_tensor_to_reference(products[key][variant], orientation_angle_from_key(key))
        for key in orientation_keys
    }

    pairwise: dict[str, dict[str, float]] = {}
    per_class_values: dict[str, list[float]] = {name: [] for name in PHYSICAL_CLASS_NAMES}
    for left, right in combinations(orientation_keys, 2):
        pair_key = f"{left}_{right}"
        pairwise[pair_key] = {}
        for class_name in PHYSICAL_CLASS_NAMES:
            idx = CLASS_INDEX[class_name]
            value = probabilistic_iou(compensated[left][idx], compensated[right][idx])
            pairwise[pair_key][class_name] = float(value)
            per_class_values[class_name].append(float(value))

    class_consistency = {
        class_name: float(np.mean(values)) if values else 0.0
        for class_name, values in per_class_values.items()
    }
    cglobal = float(np.mean([class_consistency[name] for name in PHYSICAL_CLASS_NAMES]))
    fvalid_by_orientation = {
        key: float(np.count_nonzero(products[key]["Mval"] == 1) / max(np.count_nonzero(products[key]["Mval"] > 0), 1))
        for key in orientation_keys
    }
    fvalid = float(np.mean(list(fvalid_by_orientation.values()))) if fvalid_by_orientation else 0.0

    failure_reasons: list[str] = []
    if fvalid < projection_config.fvalid_min:
        failure_reasons.append(f"fvalid<{projection_config.fvalid_min:.2f}")
    if cglobal < projection_config.cglobal_min:
        failure_reasons.append(f"Cglobal<{projection_config.cglobal_min:.2f}")
    if class_consistency.get("bulbo", 0.0) < projection_config.bulge_disk_min:
        failure_reasons.append(f"C_bulbo<{projection_config.bulge_disk_min:.2f}")
    if class_consistency.get("disco", 0.0) < projection_config.bulge_disk_min:
        failure_reasons.append(f"C_disco<{projection_config.bulge_disk_min:.2f}")

    return {
        "n_orientations": len(orientation_keys),
        "variant": variant,
        "fvalid": fvalid,
        "fvalid_by_orientation": fvalid_by_orientation,
        "Cglobal": cglobal,
        "classes": class_consistency,
        "pairwise_iou": pairwise,
        "accepted": not failure_reasons,
        "failure_reasons": failure_reasons,
    }


def write_metrics(path: str | Path, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True))


def load_projection_product(path: str | Path, variant_names: tuple[str, ...] = ("Y_lum_psf",)) -> dict[str, dict[str, np.ndarray]]:
    products: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(path, "r") as handle:
        for key in sorted(name for name in handle.keys() if name.startswith("q")):
            products[key] = {
                name: np.asarray(handle[key][name])
                for name in set(variant_names) | {"Mval"}
            }
    return products

