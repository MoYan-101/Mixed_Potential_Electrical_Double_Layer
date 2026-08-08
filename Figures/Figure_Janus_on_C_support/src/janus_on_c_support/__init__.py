"""C-supported Au-Pd Janus mirror-cell linear-PB model."""

from .electrostatics import LinearPBModel, Segment
from .kinetics import au_local_current_density, pd_local_current_density
from .parameters import (
    ELECTROSTATIC_BACKEND,
    GEOMETRY_NAME,
    MODEL_NAME,
    PACKAGE_VERSION,
    RESULT_SCHEMA_VERSION,
    apply_param_overrides,
    compute_derived_params,
    default_params,
    validate_params,
)
from .pipeline import build_result_bundle
from .solver import solve_case

__all__ = [
    "ELECTROSTATIC_BACKEND",
    "GEOMETRY_NAME",
    "LinearPBModel",
    "MODEL_NAME",
    "PACKAGE_VERSION",
    "RESULT_SCHEMA_VERSION",
    "Segment",
    "apply_param_overrides",
    "au_local_current_density",
    "build_result_bundle",
    "compute_derived_params",
    "default_params",
    "pd_local_current_density",
    "solve_case",
    "validate_params",
]

__version__ = PACKAGE_VERSION
