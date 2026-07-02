from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime, timezone

import numpy as np

from aperturenet_labels.core.constants import CLASS_NAMES
from aperturenet_labels.io.morphology import MorphologyTargets
from aperturenet_labels.phase_a.classifier import ParticleLabels
from aperturenet_labels.phase_a.extractor import ParticleFeatures
from aperturenet_labels.phase_b.label_projection import ProjectedLabels
from aperturenet_labels.phase_b.mask_builder import ValidMask


@dataclass(slots=True)
class QualityReport:
    payload: dict


def build_quality_report(
    features: ParticleFeatures,
    labels: ParticleLabels,
    projected: ProjectedLabels,
    valid_mask: ValidMask,
    targets: MorphologyTargets,
    circularity_summary: dict[str, float] | None = None,
) -> QualityReport:
    raw_total = float(np.sum(projected.raw_mass_per_class))
    feature_total = float(np.sum(features.mass))
    mass_error = abs(raw_total - feature_total) / max(feature_total, 1.0e-8)
    valid = valid_mask.m_valid
    valid_mass = projected.raw_mass_per_class[valid]
    class_mass = valid_mass.sum(axis=0) if valid_mass.size else np.zeros(len(CLASS_NAMES), dtype=np.float64)
    class_frac = class_mass / max(float(class_mass.sum()), 1.0e-8)
    max_prob = np.max(projected.y_mass_psf, axis=2)
    uncertainty = 1.0 - max_prob[valid] if np.any(valid) else np.asarray([1.0])
    flags: list[str] = []
    if mass_error > 0.15:
        flags.append("mass_projection_loss_gt_15pct")
    if valid_mask.diagnostics["fraction_valid"] < 0.10:
        flags.append("low_valid_fraction")
    status = "pass" if not flags else "warning"
    payload = {
        "galaxy_id": features.galaxy_id,
        "snapshot": int(features.snapshot),
        "subhalo_id": int(features.subhalo_id),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "flags": flags,
        "mass_conservation_error": float(mass_error),
        "n_particles_used": int(features.mass.size),
        "n_spaxels_valid": int(valid_mask.diagnostics["n_valid"]),
        "fraction_spaxels_valid": float(valid_mask.diagnostics["fraction_valid"]),
        "mean_max_probability": float(np.nanmean(max_prob[valid])) if np.any(valid) else 0.0,
        "p95_uncertainty": float(np.nanpercentile(uncertainty, 95.0)),
        "fractions_recovered_valid_mass": {name: float(class_frac[idx]) for idx, name in enumerate(CLASS_NAMES)},
        "fractions_catalog": targets.as_catalog_fractions(),
        "circularity_catalog": circularity_summary or {},
        "phase_a_diagnostics": labels.diagnostics,
        "mask_diagnostics": valid_mask.diagnostics,
        "projection_metadata": projected.metadata,
    }
    if circularity_summary:
        disk_fraction = float(class_frac[1])
        circ_disk = float(circularity_summary.get("CircAbove07Frac", np.nan))
        circ_net = float(circularity_summary.get("CircAbove07MinusBelowNeg07Frac", np.nan))
        if np.isfinite(circ_disk):
            payload["delta_disk_minus_circ_above07"] = float(disk_fraction - circ_disk)
        if np.isfinite(circ_net):
            payload["delta_disk_minus_circ_net"] = float(disk_fraction - circ_net)
    return QualityReport(payload)


def write_quality_report(path: str | Path, report: QualityReport) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.payload, indent=2, sort_keys=True))
    return path
