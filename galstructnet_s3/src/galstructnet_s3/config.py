"""Carga de configs YAML con herencia. Spec: specs/60 'Escalera' - Hito 4/5.

Formato (configs/ablation_epn/*.yaml):
    _base_: ../base.yaml          # ruta relativa al archivo actual
    model.physical: normconv      # claves con punto = override anidado
    loss.weights.psf: 0.4

La base se carga primero (recursivo) y los overrides se aplican encima.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise TypeError(f"'{dotted}': '{k}' no es un mapeo")
    node[keys[-1]] = value


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    """Carga un YAML resolviendo `_base_` (recursivo) y claves con punto."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}

    base: dict = {}
    if "_base_" in raw:
        base = load_config((path.parent / raw.pop("_base_")).resolve())

    flat: dict = {}
    for k, v in raw.items():
        if "." in k:
            _set_dotted(flat, k, v)
        else:
            flat[k] = v

    return _merge(base, flat)
