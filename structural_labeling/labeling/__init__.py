"""Utilities for building structural labels from TNG truth and MaNGIA cubes."""

from .constants import CLASS_INDEX, CLASS_NAMES

__all__ = ["CLASS_INDEX", "CLASS_NAMES", "LabelingPipeline"]


def __getattr__(name: str):
    if name == "LabelingPipeline":
        from .pipeline import LabelingPipeline

        return LabelingPipeline
    raise AttributeError(name)
