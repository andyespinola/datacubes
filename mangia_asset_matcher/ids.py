from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


UNIT_RE = re.compile(
    r"TNG50-(?P<snapshot>\d+)-(?P<subhalo_id>\d+)-(?P<view>\d+)(?:-(?P<ifu_design>\d+))?"
)
GALAXY_RE = re.compile(r"TNG50-(?P<snapshot>\d+)-(?P<subhalo_id>\d+)")


@dataclass(frozen=True, slots=True, order=True)
class UnitKey:
    snapshot: int
    subhalo_id: int
    view: int

    @property
    def unit_id(self) -> str:
        return f"TNG50-{self.snapshot}-{self.subhalo_id}-{self.view}"

    @property
    def galaxy_id(self) -> str:
        return f"TNG50-{self.snapshot}-{self.subhalo_id}"

    def canonical_id(self, ifu_design: int) -> str:
        return f"{self.unit_id}-{int(ifu_design)}"


@dataclass(frozen=True, slots=True, order=True)
class GalaxyKey:
    snapshot: int
    subhalo_id: int

    @property
    def galaxy_id(self) -> str:
        return f"TNG50-{self.snapshot}-{self.subhalo_id}"


@dataclass(frozen=True, slots=True)
class ParsedUnit:
    key: UnitKey
    ifu_design: int | None = None


def parse_unit_from_text(text: str) -> ParsedUnit | None:
    match = UNIT_RE.search(Path(text).name)
    if match is None:
        match = UNIT_RE.search(str(text))
    if match is None:
        return None
    ifu = match.group("ifu_design")
    return ParsedUnit(
        key=UnitKey(
            snapshot=int(match.group("snapshot")),
            subhalo_id=int(match.group("subhalo_id")),
            view=int(match.group("view")),
        ),
        ifu_design=int(ifu) if ifu is not None else None,
    )


def parse_galaxy_from_text(text: str) -> GalaxyKey | None:
    match = GALAXY_RE.search(Path(text).name)
    if match is None:
        match = GALAXY_RE.search(str(text))
    if match is None:
        return None
    return GalaxyKey(snapshot=int(match.group("snapshot")), subhalo_id=int(match.group("subhalo_id")))
