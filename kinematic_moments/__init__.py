"""Extract Gauss-Hermite kinematic moments from official MaNGIA cubes."""

from .io import read_mangia_official_cube, resolve_template_path
from .models import KinematicMaps, KinematicMomentsConfig, OfficialCube
from .pipeline import extract_kinematics, process_cube

__all__ = [
    "KinematicMaps",
    "KinematicMomentsConfig",
    "OfficialCube",
    "extract_kinematics",
    "process_cube",
    "read_mangia_official_cube",
    "resolve_template_path",
]
