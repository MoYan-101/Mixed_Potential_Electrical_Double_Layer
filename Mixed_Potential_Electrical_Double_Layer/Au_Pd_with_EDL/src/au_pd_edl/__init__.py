"""Linear-PB model for flush Au/Pd on an insulating substrate."""

from .electrostatics import ElectrostaticSystem, LinearPBModel
from .kinetics import (
    au_local_current_density,
    effective_reaction_params,
    emix_closed_form_no_edl,
    kinetics_context,
    local_current_densities,
    no_edl_absolute_currents,
    pd_local_current_density,
)
from .parameters import (
    ELECTROSTATIC_BACKEND,
    ELECTROLYTE_DOMAIN_DESCRIPTION,
    ELECTROLYTE_DOMAIN_ID,
    FORBIDDEN_LEGACY_KEYS,
    GEOMETRY_NAME,
    MAX_SEPARATION_M,
    MODEL_NAME,
    PACKAGE_VERSION,
    RESULT_SCHEMA_VERSION,
    apply_param_overrides,
    compute_derived_params,
    default_params,
    load_params,
    validate_params,
)
from .io import save_gouy_chapman_outputs, save_run
from .scan import (
    build_gouy_chapman_analysis,
    default_separations,
    run_separation_scan,
)
from .solver import (
    compute_polarization_curve,
    current_balance_at_potential,
    run_edl_comparison_pair,
    solve_case,
    top_surface_profiles,
)
from .surface_charge import compute_gouy_chapman_diagnostics
from .figure3 import generate_figure3_comparison_panels, load_figure3_saved_data
from .validation import run_numerical_validation

__all__ = [
    "FORBIDDEN_LEGACY_KEYS",
    "ELECTROSTATIC_BACKEND",
    "ELECTROLYTE_DOMAIN_DESCRIPTION",
    "ELECTROLYTE_DOMAIN_ID",
    "ElectrostaticSystem",
    "GEOMETRY_NAME",
    "MAX_SEPARATION_M",
    "MODEL_NAME",
    "LinearPBModel",
    "PACKAGE_VERSION",
    "RESULT_SCHEMA_VERSION",
    "apply_param_overrides",
    "au_local_current_density",
    "build_gouy_chapman_analysis",
    "compute_derived_params",
    "compute_gouy_chapman_diagnostics",
    "compute_polarization_curve",
    "current_balance_at_potential",
    "default_params",
    "default_separations",
    "effective_reaction_params",
    "emix_closed_form_no_edl",
    "kinetics_context",
    "generate_figure3_comparison_panels",
    "load_params",
    "load_figure3_saved_data",
    "local_current_densities",
    "no_edl_absolute_currents",
    "pd_local_current_density",
    "run_edl_comparison_pair",
    "run_numerical_validation",
    "run_separation_scan",
    "save_gouy_chapman_outputs",
    "save_run",
    "solve_case",
    "top_surface_profiles",
    "validate_params",
]

__version__ = PACKAGE_VERSION
