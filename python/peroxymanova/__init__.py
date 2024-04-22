import importlib.metadata
from ._oxide import permanova
from ._peroxymanova import (
    calculate_distances,
    permanova_pipeline,
    ordinal_encoding,
    PermanovaResults,
)

__all__ = [
    "permanova",
    "calculate_distances",
    "permanova_pipeline",
    "ordinal_encoding",
    "PermanovaResults",
]

if __package__ is not None:
    __version__ = importlib.metadata.version(__package__)
