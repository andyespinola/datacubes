"""Independent asset matcher for MaNGIA cube, TNG, and 2D map products."""

from .matcher import (
    MatchConfig,
    MatchResult,
    build_matches,
    read_catalog,
    scan_assets,
)

__all__ = [
    "MatchConfig",
    "MatchResult",
    "build_matches",
    "read_catalog",
    "scan_assets",
]
