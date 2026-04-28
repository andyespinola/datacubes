from __future__ import annotations

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
STATIC_DIR = PROJECT_DIR / "static"
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DERIVED_DATA_DIR = DATA_DIR / "derived"

PILOT_CANONICAL_ID = "TNG50-87-141934-0-127"
PILOT_SIMULATION = "TNG50-1"
PILOT_SNAPSHOT = 87
PILOT_SUBHALO_ID = 141934
PILOT_VIEW = 0
PILOT_IFU = 127

PILOT_REDSHIFT = 0.15
TNG_HUBBLE = 0.6774
TNG_OMEGA_M = 0.3089
TNG_OMEGA_B = 0.0486

DEFAULT_MAP_BINS = 220
DEFAULT_PROFILE_BINS = 48
DEFAULT_RADIUS_KPC = 35.0
MIN_RADIUS_KPC = 8.0
MAX_RADIUS_KPC = 80.0

DEFAULT_CUTOUT_PATH = RAW_DATA_DIR / f"{PILOT_CANONICAL_ID}.cutout.hdf5"
DEFAULT_METADATA_PATH = RAW_DATA_DIR / f"{PILOT_CANONICAL_ID}.subhalo.json"
DEFAULT_MORPHOLOGY_PATH = RAW_DATA_DIR / "morphs_kinematic_bars.hdf5"

SOURCE_CUTOUT_CANDIDATES = (
    REPO_DIR / "structural_labeling" / "cache" / "cutouts" / f"{PILOT_CANONICAL_ID}.cutout.hdf5",
)
SOURCE_METADATA_CANDIDATES = (
    REPO_DIR / "structural_labeling" / "cache" / "metadata" / f"{PILOT_CANONICAL_ID}.subhalo.json",
)
SOURCE_MORPHOLOGY_CANDIDATES = (
    REPO_DIR / "structural_labeling" / "cache" / "morphs_kinematic_bars.hdf5",
)

TNG_DOWNLOAD_URL = (
    f"https://www.tng-project.org/api/{PILOT_SIMULATION}/snapshots/{PILOT_SNAPSHOT}/subhalos/{PILOT_SUBHALO_ID}"
)
