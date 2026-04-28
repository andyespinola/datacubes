from __future__ import annotations

from collections import OrderedDict
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .constants import (
    DEFAULT_MAP_BINS,
    DEFAULT_PROFILE_BINS,
    DEFAULT_RADIUS_KPC,
    DEFAULT_CUTOUT_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_MORPHOLOGY_PATH,
    MAX_RADIUS_KPC,
    MIN_RADIUS_KPC,
    PILOT_CANONICAL_ID,
    PILOT_IFU,
    PILOT_REDSHIFT,
    PILOT_SIMULATION,
    PILOT_SNAPSHOT,
    PILOT_SUBHALO_ID,
    PILOT_VIEW,
    TNG_HUBBLE,
)
from .processing import (
    axes_for_view,
    build_rotation_matrix,
    finite_percentile_range,
    histogram_surface_density,
    histogram_weighted_mean,
    radial_surface_density_profile,
    radial_weighted_mean_profile,
    rotate_positions,
    serializable_grid,
    serializable_vector,
    stellar_age_gyr_from_scale_factor,
)


STAR_QUANTITIES: dict[str, dict[str, Any]] = {
    "mass": {
        "label": "Densidad superficial estelar",
        "unit": "log10(Msun / kpc^2)",
        "kind": "surface_density",
        "weight_field": "stellar_mass",
        "profile_label": "Perfil radial estelar",
    },
    "age": {
        "label": "Edad estelar media",
        "unit": "Gyr",
        "kind": "weighted_mean",
        "value_field": "stellar_age",
        "weight_field": "stellar_mass",
        "profile_label": "Edad estelar media",
    },
    "metallicity": {
        "label": "Metallicidad estelar media",
        "unit": "Z",
        "kind": "weighted_mean",
        "value_field": "stellar_metallicity",
        "weight_field": "stellar_mass",
        "profile_label": "Metallicidad estelar media",
    },
}

GAS_QUANTITIES: dict[str, dict[str, Any]] = {
    "mass": {
        "label": "Densidad superficial de gas",
        "unit": "log10(Msun / kpc^2)",
        "kind": "surface_density",
        "weight_field": "gas_mass",
        "profile_label": "Perfil radial de gas",
    },
    "sfr": {
        "label": "Densidad superficial de SFR",
        "unit": "log10(Msun / yr / kpc^2)",
        "kind": "surface_density",
        "weight_field": "gas_sfr",
        "profile_label": "Perfil radial de SFR",
    },
    "metallicity": {
        "label": "Metallicidad media del gas",
        "unit": "Z",
        "kind": "weighted_mean",
        "value_field": "gas_metallicity",
        "weight_field": "gas_mass",
        "profile_label": "Metallicidad media del gas",
    },
}


@dataclass
class GalaxyData:
    stellar_positions: np.ndarray
    stellar_mass: np.ndarray
    stellar_age: np.ndarray
    stellar_metallicity: np.ndarray
    gas_positions: np.ndarray
    gas_mass: np.ndarray
    gas_sfr: np.ndarray
    gas_metallicity: np.ndarray
    rotation: np.ndarray
    summary: dict[str, Any]
    morphology: dict[str, Any] | None


@dataclass
class ParticleSelection:
    label: str
    quantity: str
    quantity_label: str
    selected_count: int
    sampled_count: int
    packed: np.ndarray


class GalaxyStore:
    def __init__(self, data: GalaxyData):
        self.data = data
        self._map_cache: dict[tuple[str, str, str, float, int], dict[str, Any]] = {}
        self._profile_cache: dict[tuple[str, str, float, int], dict[str, Any]] = {}
        self._cloud_manifest_cache: OrderedDict[tuple[float, int, int], dict[str, Any]] = OrderedDict()
        self._particle_cache: OrderedDict[tuple[str, float, int], ParticleSelection] = OrderedDict()

    @classmethod
    def from_default_files(cls) -> "GalaxyStore":
        return cls.from_files(DEFAULT_CUTOUT_PATH, DEFAULT_METADATA_PATH, DEFAULT_MORPHOLOGY_PATH)

    @classmethod
    def from_files(
        cls,
        cutout_path: str | Path,
        metadata_path: str | Path,
        morphology_path: str | Path | None = None,
    ) -> "GalaxyStore":
        cutout_path = Path(cutout_path)
        metadata_path = Path(metadata_path)
        metadata = json.loads(metadata_path.read_text())
        redshift = float(metadata.get("redshift") or PILOT_REDSHIFT)
        scale_factor = 1.0 / (1.0 + redshift)
        sqrt_scale_factor = np.sqrt(scale_factor)

        center_pos = np.array([metadata["pos_x"], metadata["pos_y"], metadata["pos_z"]], dtype=np.float64)
        center_vel = np.array([metadata["vel_x"], metadata["vel_y"], metadata["vel_z"]], dtype=np.float64)

        with h5py.File(cutout_path, "r") as handle:
            stars = handle["PartType4"]
            form = np.asarray(stars["GFM_StellarFormationTime"], dtype=np.float64)
            valid = form > 0

            stellar_positions = np.asarray(stars["Coordinates"], dtype=np.float64)[valid]
            stellar_positions = ((stellar_positions - center_pos) * scale_factor / TNG_HUBBLE).astype(np.float32)

            stellar_velocities = np.asarray(stars["Velocities"], dtype=np.float64)[valid]
            stellar_velocities = (stellar_velocities * sqrt_scale_factor - center_vel).astype(np.float32)

            stellar_mass = np.asarray(stars["Masses"], dtype=np.float64)[valid]
            stellar_mass = (stellar_mass * 1.0e10 / TNG_HUBBLE).astype(np.float32)

            stellar_age = stellar_age_gyr_from_scale_factor(form[valid], redshift)
            stellar_metallicity = np.asarray(stars["GFM_Metallicity"], dtype=np.float32)[valid]

            if "PartType0" in handle:
                gas = handle["PartType0"]
                gas_positions = np.asarray(gas["Coordinates"], dtype=np.float64)
                gas_positions = ((gas_positions - center_pos) * scale_factor / TNG_HUBBLE).astype(np.float32)
                gas_mass = np.asarray(gas["Masses"], dtype=np.float64)
                gas_mass = (gas_mass * 1.0e10 / TNG_HUBBLE).astype(np.float32)
                gas_sfr = np.asarray(gas["StarFormationRate"], dtype=np.float32)
                gas_metallicity = np.asarray(gas["GFM_Metallicity"], dtype=np.float32)
            else:
                gas_positions = np.empty((0, 3), dtype=np.float32)
                gas_mass = np.empty((0,), dtype=np.float32)
                gas_sfr = np.empty((0,), dtype=np.float32)
                gas_metallicity = np.empty((0,), dtype=np.float32)

        stellar_radius = np.linalg.norm(stellar_positions, axis=1)
        halfmass_radius_kpc = float(metadata.get("halfmassrad_stars", 0.0) * scale_factor / TNG_HUBBLE)
        orientation_mask = stellar_radius <= max(2.5 * halfmass_radius_kpc, 12.0)
        if np.count_nonzero(orientation_mask) < 1024:
            orientation_mask = np.ones_like(stellar_radius, dtype=bool)

        orientation_slice = np.flatnonzero(orientation_mask)
        if orientation_slice.size > 200000:
            orientation_slice = orientation_slice[:200000]

        rotation = build_rotation_matrix(
            stellar_positions[orientation_slice].astype(np.float64),
            stellar_velocities[orientation_slice].astype(np.float64),
            stellar_mass[orientation_slice].astype(np.float64),
        ).astype(np.float32)

        stellar_positions = rotate_positions(stellar_positions.astype(np.float64), rotation.astype(np.float64)).astype(np.float32)
        gas_positions = rotate_positions(gas_positions.astype(np.float64), rotation.astype(np.float64)).astype(np.float32)

        morphology = load_morphology_summary(morphology_path, int(metadata.get("snap") or PILOT_SNAPSHOT), int(metadata.get("id") or PILOT_SUBHALO_ID))
        summary = build_summary(
            metadata=metadata,
            redshift=redshift,
            scale_factor=scale_factor,
            halfmass_radius_kpc=halfmass_radius_kpc,
            stellar_mass=stellar_mass,
            gas_mass=gas_mass,
            gas_sfr=gas_sfr,
            n_stars=int(stellar_mass.size),
            n_gas=int(gas_mass.size),
            morphology=morphology,
            cutout_path=cutout_path,
            metadata_path=metadata_path,
        )
        data = GalaxyData(
            stellar_positions=stellar_positions,
            stellar_mass=stellar_mass,
            stellar_age=stellar_age,
            stellar_metallicity=stellar_metallicity,
            gas_positions=gas_positions,
            gas_mass=gas_mass,
            gas_sfr=gas_sfr,
            gas_metallicity=gas_metallicity,
            rotation=rotation,
            summary=summary,
            morphology=morphology,
        )
        return cls(data)

    def get_config(self) -> dict[str, Any]:
        return {
            "summary": self.data.summary,
            "quantities": {
                "stars": list(STAR_QUANTITIES.keys()),
                "gas": list(GAS_QUANTITIES.keys()),
            },
            "quantity_meta": {
                "stars": {key: {"label": value["label"], "unit": value["unit"]} for key, value in STAR_QUANTITIES.items()},
                "gas": {key: {"label": value["label"], "unit": value["unit"]} for key, value in GAS_QUANTITIES.items()},
            },
            "views": {
                "faceon": "Face-on",
                "edgeon": "Edge-on",
            },
            "defaults": {
                "component": "stars",
                "quantity": "mass",
                "view": "faceon",
                "radius_kpc": DEFAULT_RADIUS_KPC,
                "bins": DEFAULT_MAP_BINS,
                "profile_bins": DEFAULT_PROFILE_BINS,
                "cloud_detail": "equilibrada",
            },
            "limits": {
                "radius_kpc": {"min": MIN_RADIUS_KPC, "max": MAX_RADIUS_KPC},
            },
            "cloud_presets": {
                "rapida": {"label": "Rapida", "max_stars": 120000, "max_gas": 25000},
                "equilibrada": {"label": "Equilibrada", "max_stars": 450000, "max_gas": 60000},
                "completa": {"label": "Completa", "max_stars": 0, "max_gas": 0},
            },
        }

    def get_map(self, component: str, quantity: str, view: str, radius_kpc: float, bins: int) -> dict[str, Any]:
        component = component.lower()
        quantity = quantity.lower()
        view = view.lower()
        radius_kpc = clip_radius(radius_kpc)
        bins = int(np.clip(bins, 64, 320))
        cache_key = (component, quantity, view, radius_kpc, bins)
        if cache_key in self._map_cache:
            return self._map_cache[cache_key]

        axes, spec = self._axes_and_spec(component, quantity, view)
        if spec["kind"] == "surface_density":
            weights = self._field(component, spec["weight_field"])
            image = histogram_surface_density(axes.x, axes.y, weights, radius_kpc, bins)
            image = np.log10(image)
        else:
            values = self._field(component, spec["value_field"])
            weights = self._field(component, spec["weight_field"])
            image = histogram_weighted_mean(axes.x, axes.y, values, weights, radius_kpc, bins)

        vmin, vmax = finite_percentile_range(image)
        payload = {
            "component": component,
            "quantity": quantity,
            "view": view,
            "radius_kpc": radius_kpc,
            "bins": bins,
            "label": spec["label"],
            "unit": spec["unit"],
            "axis_labels": {"x": axes.label_x, "y": axes.label_y},
            "extent": {
                "xmin": -radius_kpc,
                "xmax": radius_kpc,
                "ymin": -radius_kpc,
                "ymax": radius_kpc,
            },
            "vmin": vmin,
            "vmax": vmax,
            "data": serializable_grid(image),
        }
        self._map_cache[cache_key] = payload
        return payload

    def get_profile(self, component: str, quantity: str, radius_kpc: float, bins: int) -> dict[str, Any]:
        component = component.lower()
        quantity = quantity.lower()
        radius_kpc = clip_radius(radius_kpc)
        bins = int(np.clip(bins, 16, 128))
        cache_key = (component, quantity, radius_kpc, bins)
        if cache_key in self._profile_cache:
            return self._profile_cache[cache_key]

        spec = self._quantity_spec(component, quantity)
        positions = self._positions(component)
        radius = np.hypot(positions[:, 0], positions[:, 1])
        if spec["kind"] == "surface_density":
            weights = self._field(component, spec["weight_field"])
            x, y = radial_surface_density_profile(radius, weights, radius_kpc, bins)
            y = np.log10(y)
        else:
            values = self._field(component, spec["value_field"])
            weights = self._field(component, spec["weight_field"])
            x, y = radial_weighted_mean_profile(radius, values, weights, radius_kpc, bins)

        payload = {
            "component": component,
            "quantity": quantity,
            "label": spec["profile_label"],
            "unit": spec["unit"],
            "radius_kpc": radius_kpc,
            "x": serializable_vector(x),
            "y": serializable_vector(y),
        }
        self._profile_cache[cache_key] = payload
        return payload

    def get_particle_cloud(
        self,
        radius_kpc: float,
        max_stars: int = 12000,
        max_gas: int = 6000,
        star_quantity: str = "mass",
        gas_quantity: str = "mass",
    ) -> dict[str, Any]:
        radius_kpc = clip_radius(radius_kpc)
        max_stars = sanitize_max_points(max_stars, upper=3_000_000)
        max_gas = sanitize_max_points(max_gas, upper=500_000)
        star_quantity = star_quantity.lower()
        gas_quantity = gas_quantity.lower()
        cache_key = (radius_kpc, max_stars, max_gas, star_quantity, gas_quantity)
        if cache_key in self._cloud_manifest_cache:
            manifest = self._cloud_manifest_cache.pop(cache_key)
            self._cloud_manifest_cache[cache_key] = manifest
            return manifest

        stars = self.get_particle_selection("stars", radius_kpc=radius_kpc, max_points=max_stars, quantity=star_quantity)
        gas = self.get_particle_selection("gas", radius_kpc=radius_kpc, max_points=max_gas, quantity=gas_quantity)

        payload = {
            "radius_kpc": radius_kpc,
            "halfmass_radius_kpc": float(self.data.summary.get("stellar_halfmass_radius_kpc", 0.0)),
            "render_backend": "webgl",
            "stars": particle_selection_summary(stars),
            "gas": particle_selection_summary(gas),
        }
        remember_cache_entry(self._cloud_manifest_cache, cache_key, payload)
        return payload

    def get_particle_selection(self, component: str, radius_kpc: float, max_points: int, quantity: str = "mass") -> ParticleSelection:
        component = component.lower()
        quantity = quantity.lower()
        radius_kpc = clip_radius(radius_kpc)
        max_points = sanitize_max_points(max_points, upper=3_000_000 if component == "stars" else 500_000)
        cache_key = (component, quantity, radius_kpc, max_points)
        if cache_key in self._particle_cache:
            selection = self._particle_cache.pop(cache_key)
            self._particle_cache[cache_key] = selection
            return selection

        spec = self._quantity_spec(component, quantity)
        intensity = self._particle_intensity_field(component, spec)
        if component == "stars":
            positions = self.data.stellar_positions
            label = "Estrellas"
            seed = 17
        elif component == "gas":
            positions = self.data.gas_positions
            label = "Gas"
            seed = 29
        else:
            raise KeyError(f"Componente no soportado para nube 3D: {component}")

        selection = build_particle_component_selection(
            positions=positions,
            intensity=intensity,
            radius_kpc=radius_kpc,
            max_points=max_points,
            seed=seed,
            label=label,
            quantity=quantity,
            quantity_label=spec["label"],
        )
        remember_cache_entry(self._particle_cache, cache_key, selection)
        return selection

    def _axes_and_spec(self, component: str, quantity: str, view: str) -> tuple[Any, dict[str, Any]]:
        positions = self._positions(component)
        axes = axes_for_view(positions, view)
        spec = self._quantity_spec(component, quantity)
        return axes, spec

    def _positions(self, component: str) -> np.ndarray:
        if component == "stars":
            return self.data.stellar_positions
        if component == "gas":
            return self.data.gas_positions
        raise KeyError(f"Componente no soportado: {component}")

    def _field(self, component: str, field_name: str) -> np.ndarray:
        attr = f"{component}_{field_name.split('_', 1)[1]}" if field_name.startswith(component + "_") else field_name
        if hasattr(self.data, attr):
            return getattr(self.data, attr)
        raise KeyError(f"No encontré el campo {field_name} para {component}")

    def _quantity_spec(self, component: str, quantity: str) -> dict[str, Any]:
        if component == "stars":
            spec = STAR_QUANTITIES.get(quantity)
        elif component == "gas":
            spec = GAS_QUANTITIES.get(quantity)
        else:
            spec = None
        if spec is None:
            raise KeyError(f"Cantidad no soportada: {component}/{quantity}")
        return spec

    def _particle_intensity_field(self, component: str, spec: dict[str, Any]) -> np.ndarray:
        if spec["kind"] == "surface_density":
            values = self._field(component, spec["weight_field"])
            return np.asarray(values, dtype=np.float32)
        values = self._field(component, spec["value_field"])
        return np.asarray(values, dtype=np.float32)

    @cached_property
    def summary(self) -> dict[str, Any]:
        return self.data.summary


def clip_radius(radius_kpc: float) -> float:
    return float(np.clip(radius_kpc, MIN_RADIUS_KPC, MAX_RADIUS_KPC))


def sanitize_max_points(max_points: int, upper: int) -> int:
    max_points = int(max_points)
    if max_points <= 0:
        return 0
    return int(np.clip(max_points, 1_000, upper))


def remember_cache_entry(cache: OrderedDict, key: Any, value: Any, max_entries: int = 6) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_entries:
        cache.popitem(last=False)


def build_particle_component_selection(
    positions: np.ndarray,
    intensity: np.ndarray,
    radius_kpc: float,
    max_points: int,
    seed: int,
    label: str,
    quantity: str,
    quantity_label: str,
) -> ParticleSelection:
    if positions.size == 0:
        return ParticleSelection(
            label=label,
            quantity=quantity,
            quantity_label=quantity_label,
            selected_count=0,
            sampled_count=0,
            packed=np.empty((0, 4), dtype=np.float32),
        )

    radius = np.linalg.norm(positions, axis=1)
    selected = np.flatnonzero(radius <= radius_kpc)
    selected_count = int(selected.size)
    if selected_count == 0:
        return ParticleSelection(
            label=label,
            quantity=quantity,
            quantity_label=quantity_label,
            selected_count=0,
            sampled_count=0,
            packed=np.empty((0, 4), dtype=np.float32),
        )

    if max_points > 0 and selected_count > max_points:
        rng = np.random.default_rng(seed)
        sample_indices = np.sort(rng.choice(selected, size=max_points, replace=False))
    else:
        sample_indices = selected

    sample_positions = np.asarray(positions[sample_indices], dtype=np.float32)
    sample_values = np.asarray(intensity[sample_indices], dtype=np.float64)
    sample_intensity = normalize_point_intensity(sample_values)

    packed = np.column_stack([sample_positions, sample_intensity]).astype(np.float32)
    packed = np.round(packed, 4)
    return ParticleSelection(
        label=label,
        quantity=quantity,
        quantity_label=quantity_label,
        selected_count=selected_count,
        sampled_count=int(sample_positions.shape[0]),
        packed=packed,
    )


def particle_selection_summary(selection: ParticleSelection) -> dict[str, Any]:
    return {
        "label": selection.label,
        "quantity": selection.quantity,
        "quantity_label": selection.quantity_label,
        "selected_count": selection.selected_count,
        "sampled_count": selection.sampled_count,
    }


def normalize_point_intensity(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    good = np.isfinite(values) & (values > 0)
    if not np.any(good):
        return np.full(values.shape, 0.3, dtype=np.float32)
    positive = values[good]
    ref = float(np.nanpercentile(positive, 95))
    if not np.isfinite(ref) or ref <= 0:
        ref = float(np.nanmax(positive))
    if not np.isfinite(ref) or ref <= 0:
        return np.full(values.shape, 0.3, dtype=np.float32)
    scaled = np.log1p(np.clip(values, a_min=0.0, a_max=None) / ref) / np.log(2.0)
    scaled = np.clip(scaled, 0.08, 1.0)
    return scaled.astype(np.float32)


def load_morphology_summary(path: str | Path | None, snapshot: int, subhalo_id: int) -> dict[str, Any] | None:
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    with h5py.File(path, "r") as handle:
        group_name = f"Snapshot_{snapshot}"
        if group_name not in handle:
            return None
        group = handle[group_name]
        ids = np.asarray(group["SubhaloID"], dtype=np.int64)
        indices = np.where(ids == subhalo_id)[0]
        if indices.size == 0:
            return None
        idx = int(indices[0])

        def scalar(name: str, row: int | None = None, default: float = 0.0) -> float:
            if name not in group:
                return default
            data = np.asarray(group[name])
            value = data[row, idx] if row is not None else data[idx]
            return float(value)

        return {
            "thin_disk": scalar("ThinDisc", row=0),
            "thick_disk": scalar("ThickDisc", row=0),
            "pseudo_bulge": scalar("PseudoBulge", row=0),
            "bulge": scalar("Bulge", row=0),
            "halo": scalar("Halo", row=0),
            "unbound": scalar("UnboundMass"),
            "barred": bool(scalar("Barred") > 0),
            "bar_size_kpc": max(0.0, scalar("BarSize", row=0, default=-1.0)),
            "bar_strength": max(0.0, scalar("BarStrength", row=0, default=-1.0)),
        }


def build_summary(
    metadata: dict[str, Any],
    redshift: float,
    scale_factor: float,
    halfmass_radius_kpc: float,
    stellar_mass: np.ndarray,
    gas_mass: np.ndarray,
    gas_sfr: np.ndarray,
    n_stars: int,
    n_gas: int,
    morphology: dict[str, Any] | None,
    cutout_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    summary = {
        "canonical_id": PILOT_CANONICAL_ID,
        "simulation": PILOT_SIMULATION,
        "snapshot": int(metadata.get("snap") or PILOT_SNAPSHOT),
        "subhalo_id": int(metadata.get("id") or PILOT_SUBHALO_ID),
        "view": PILOT_VIEW,
        "ifu": PILOT_IFU,
        "redshift": redshift,
        "scale_factor": scale_factor,
        "hubble": TNG_HUBBLE,
        "stellar_mass_msun": float(np.sum(stellar_mass, dtype=np.float64)),
        "gas_mass_msun": float(np.sum(gas_mass, dtype=np.float64)),
        "sfr_msun_per_yr": float(np.sum(gas_sfr, dtype=np.float64)),
        "stellar_halfmass_radius_kpc": halfmass_radius_kpc,
        "n_stellar_particles": n_stars,
        "n_gas_cells": n_gas,
        "mass_log_msun": float(metadata.get("mass_log_msun", 0.0)),
        "cutout_path": str(cutout_path),
        "metadata_path": str(metadata_path),
        "tng_urls": {
            "subhalo": metadata.get("related", {}).get("url"),
            "cutout": metadata.get("cutouts", {}).get("subhalo"),
            "info": metadata.get("trees", {}).get("sublinkgal_simple") or metadata.get("meta", {}).get("info"),
        },
    }
    if morphology is not None:
        summary["morphology"] = morphology
    return summary
