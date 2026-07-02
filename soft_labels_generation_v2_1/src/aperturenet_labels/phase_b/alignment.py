from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class GridAlignment:
    enabled: bool
    rot90_k: int = 0
    flip_x: bool = False
    flip_y: bool = False
    score: float = 0.0
    reference: str = ""

    def as_metadata(self) -> dict[str, float | int | str | bool]:
        return {
            "sky_alignment_enabled": bool(self.enabled),
            "sky_alignment_rot90_k": int(self.rot90_k),
            "sky_alignment_degrees_ccw": float(90 * int(self.rot90_k)),
            "sky_alignment_flip_x": bool(self.flip_x),
            "sky_alignment_flip_y": bool(self.flip_y),
            "sky_alignment_score": float(self.score),
            "sky_alignment_reference": self.reference,
        }


def apply_grid_alignment(array: np.ndarray, alignment: GridAlignment) -> np.ndarray:
    if not alignment.enabled:
        return array
    out = np.asarray(array)
    if alignment.flip_x:
        out = np.flip(out, axis=1)
    if alignment.flip_y:
        out = np.flip(out, axis=0)
    if alignment.rot90_k % 4:
        out = np.rot90(out, k=alignment.rot90_k % 4, axes=(0, 1))
    return np.ascontiguousarray(out)


def _source_contrast(map_2d: np.ndarray) -> np.ndarray:
    source = np.asarray(map_2d, dtype=np.float64)
    out = np.zeros_like(source, dtype=np.float64)
    positive = np.isfinite(source) & (source > 0.0)
    if not np.any(positive):
        return out
    floor = float(np.nanpercentile(source[positive], 2.0))
    out[positive] = np.log10(np.clip(source[positive], floor, None))
    return out


def _reference_contrast(map_2d: np.ndarray) -> np.ndarray:
    reference = np.asarray(map_2d, dtype=np.float64)
    out = np.zeros_like(reference, dtype=np.float64)
    finite_positive = np.isfinite(reference) & (reference > 0.0)
    out[finite_positive] = reference[finite_positive]
    return out


def _normalized_cross_correlation(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(np.count_nonzero(mask)) < 16:
        return -1.0
    left_values = left[mask].astype(np.float64, copy=False)
    right_values = right[mask].astype(np.float64, copy=False)
    left_values = left_values - float(np.mean(left_values))
    right_values = right_values - float(np.mean(right_values))
    denom = float(np.sqrt(np.sum(left_values**2) * np.sum(right_values**2)))
    if denom <= 0.0:
        return -1.0
    return float(np.sum(left_values * right_values) / denom)


def estimate_d4_alignment(
    source_mass_map: np.ndarray,
    reference_map: np.ndarray,
    reference_name: str,
) -> GridAlignment:
    source = _source_contrast(source_mass_map)
    reference = _reference_contrast(reference_map)
    best = GridAlignment(enabled=True, score=-1.0, reference=reference_name)
    for flip_x in (False, True):
        for flip_y in (False, True):
            candidate = source
            if flip_x:
                candidate = np.flip(candidate, axis=1)
            if flip_y:
                candidate = np.flip(candidate, axis=0)
            for rot90_k in range(4):
                transformed = np.rot90(candidate, k=rot90_k, axes=(0, 1)) if rot90_k else candidate
                score = _normalized_cross_correlation(transformed, reference)
                if score > best.score:
                    best = GridAlignment(
                        enabled=True,
                        rot90_k=rot90_k,
                        flip_x=flip_x,
                        flip_y=flip_y,
                        score=score,
                        reference=reference_name,
                    )
    return best
