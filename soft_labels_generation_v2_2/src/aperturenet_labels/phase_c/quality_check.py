"""QualityCheck (spec 30): reporte de calidad por galaxia × orientación."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import structlog
from pydantic import BaseModel
from scipy.ndimage import label as ndi_label

from ..core.constants import CLASS_NAMES

log = structlog.get_logger(__name__)


class QualityCheckConfig(BaseModel):
    fraction_tolerance: float = 0.10
    min_valid_fraction: float = 0.30
    max_uncertainty_p95: float = 0.5
    max_mass_conservation_error: float = 0.05


def run_quality_check(
    galaxy_id: str,
    view_id: int,
    feats: dict,
    final_labels: dict,
    projection: dict,
    mask: dict,
    catalog_fractions: dict | None,
    output_path: str | Path,
    config: QualityCheckConfig | None = None,
    extra_metrics: dict | None = None,
) -> Path:
    config = config or QualityCheckConfig()
    flags: list[str] = []

    # --- conservación de masa (sobre R_cov) ---
    meta = projection["metadata"]
    frac_clipped = float(meta.get("fraction_clipped_by_rcov", 0.0))
    m_features = float(feats["mass"].sum()) * (1.0 - frac_clipped)
    m_aggregated = float(projection["total_mass_per_spaxel"].sum())
    mass_err = abs(m_aggregated - m_features) / max(m_features, 1e-12)
    l_features = float(feats["light_g"].sum())
    l_aggregated = float(projection["total_light_per_spaxel"].sum())
    light_err = abs(l_aggregated - l_features) / max(l_features, 1e-12)
    if mass_err > config.max_mass_conservation_error:
        flags.append("mass_conservation")

    # --- fracciones recuperadas (masa, dentro de M_valid) ---
    m_valid = mask["M_valid"]
    raw_mass = projection["raw_mass_per_class"]  # (5, H, W)
    total = float((raw_mass.sum(axis=0) * m_valid).sum())
    fractions_recovered = {
        c: float((raw_mass[i] * m_valid).sum() / total) if total > 0 else 0.0
        for i, c in enumerate(CLASS_NAMES)
    }
    fraction_deviations: dict[str, float] = {}
    if catalog_fractions:
        # catálogo K=3: comparar bulge, disk(=disk+bar+arm), other(=halo)
        recovered3 = {
            "bulge": fractions_recovered["bulge"],
            "disk": fractions_recovered["disk"]
            + fractions_recovered["bar"]
            + fractions_recovered["arm"],
            "other": fractions_recovered["halo"],
        }
        for k, target in catalog_fractions.items():
            dev = abs(recovered3.get(k, 0.0) - target)
            fraction_deviations[k] = dev
            if dev > config.fraction_tolerance:
                flags.append(f"fraction_dev_{k}")

    # --- validez espacial ---
    n_valid = int(m_valid.sum())
    frac_valid = float(m_valid.mean())
    if frac_valid < config.min_valid_fraction:
        flags.append("low_validity")
    labeled, n_cc = ndi_label(m_valid)
    if n_cc > 0:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        lcc_frac = float(sizes.max() / max(n_valid, 1))
    else:
        lcc_frac = 0.0
    if lcc_frac < 0.95:
        flags.append("fragmented_mask")

    # --- incertidumbre probabilística (por spaxel, variante mass_raw) ---
    y = projection["Y_mass_raw"]
    max_p = y.max(axis=0)[m_valid] if m_valid.any() else np.array([0.0])
    p95_unc = float(np.percentile(1.0 - max_p, 95))
    if p95_unc > config.max_uncertainty_p95:
        flags.append("high_uncertainty")

    critical = {"mass_conservation"}
    status = (
        "fail"
        if any(f in critical for f in flags)
        else ("warning" if flags else "pass")
    )

    report = {
        "galaxy_id": galaxy_id,
        "view_id": view_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": status,
        "flags": flags,
        "mass_conservation_error": mass_err,
        "light_conservation_error": light_err,
        "fraction_clipped_by_rcov": frac_clipped,
        "fractions_recovered": fractions_recovered,
        "fractions_catalog": catalog_fractions or {},
        "fraction_deviations": fraction_deviations,
        "n_spaxels_valid": n_valid,
        "fraction_spaxels_valid": frac_valid,
        "largest_connected_component_fraction": lcc_frac,
        "mean_max_probability": float(max_p.mean()),
        "p95_uncertainty": p95_unc,
        "bar_detected": bool(final_labels.get("bar_diagnostics", {}).get("n_bar_particles", 0) > 0),
        "bar_a2": float(final_labels.get("bar_diagnostics", {}).get("a2", 0.0))
        if "bar_diagnostics" in final_labels
        else None,
        "n_arm_crests": int(final_labels.get("arm_diagnostics", {}).get("n_crests", 0)),
        "extra": extra_metrics or {},
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    log.info(
        "quality_check.done",
        galaxy_id=galaxy_id,
        view_id=view_id,
        status=status,
        flags=flags,
        mass_err=round(mass_err, 5),
    )
    return output_path
