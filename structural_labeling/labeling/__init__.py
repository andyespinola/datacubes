"""Utilities for building structural labels from TNG truth and MaNGIA cubes."""

from .constants import CLASS_INDEX, CLASS_NAMES
from .pipeline import LabelingPipeline

__all__ = ["CLASS_INDEX", "CLASS_NAMES", "LabelingPipeline"]
