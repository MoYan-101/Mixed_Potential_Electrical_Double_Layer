"""Facing Au/Pd electrolyte-slit mixed-potential model."""

from .model import FacingElectrolyteSlitModel, default_params, load_params, solve_cases
from .nonlinear_model import (
    NonlinearFacingElectrolyteSlitModel,
    solve_cases_nonlinear,
    solve_planar_gouy_chapman_stern_limit,
)

__all__ = [
    "FacingElectrolyteSlitModel",
    "NonlinearFacingElectrolyteSlitModel",
    "default_params",
    "load_params",
    "solve_cases",
    "solve_cases_nonlinear",
    "solve_planar_gouy_chapman_stern_limit",
]

__version__ = "0.2.0"
