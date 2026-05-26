from __future__ import annotations

from pathlib import Path
import sys


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    return project_root().parent


def structural_labeling_root() -> Path:
    return workspace_root() / "structural_labeling"


def ensure_structural_labeling_on_path() -> None:
    root = structural_labeling_root()
    if not root.exists():
        raise RuntimeError(f"No existe structural_labeling en {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def default_catalog_path() -> Path:
    return workspace_root() / "MaNGIA_catalog.fits"


def default_ssp_template_path() -> Path:
    return workspace_root() / "kinematic_moments" / "templates" / "MaStar_CB19.slog_1_5.fits.gz"
