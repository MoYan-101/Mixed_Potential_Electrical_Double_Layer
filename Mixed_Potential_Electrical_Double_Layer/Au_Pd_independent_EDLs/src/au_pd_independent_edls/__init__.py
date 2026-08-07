"""Independent planar Au/Pd electrical-double-layer model."""

from .model import (
    IndependentPlanarEDLModel,
    default_params,
    load_params,
    solve_comparison,
)

__all__ = [
    "IndependentPlanarEDLModel",
    "default_params",
    "load_params",
    "solve_comparison",
]

__version__ = "0.1.0"
