"""Numerical validation for the semi-infinite cosine-spectral PB model."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from .electrostatics import LinearPBModel
from .parameters import (
    ELECTROSTATIC_BACKEND,
    GEOMETRY_NAME,
    MODEL_NAME,
    RESULT_SCHEMA_VERSION,
    apply_param_overrides,
    compute_derived_params,
)
from .solver import solve_case


LEGACY_D0_E_MIX_V = 0.6006949941170235
LEGACY_D0_I_MIX_AVG_A_PER_M2 = 0.0815880733807085
FINITE_H10_Q1_D0_E_MIX_V = 0.6006820511512232
FINITE_H10_Q1_D0_I_MIX_AVG_A_PER_M2 = 0.08160527647541833

E_TOL_V = 0.5e-3
I_REL_TOL = 0.01
HOMOGENEOUS_SURFACE_ABS_TOL = 2.0e-10
HOMOGENEOUS_SURFACE_REL_TOL = 2.0e-10
AFFINE_FIELD_ABS_TOL = 1.0e-10
AFFINE_FIELD_REL_TOL = 1.0e-11


def _relative_difference(value: float, reference: float) -> float:
    denominator = abs(float(reference))
    difference = abs(float(value) - float(reference))
    if denominator == 0.0:
        return 0.0 if difference == 0.0 else math.inf
    return float(difference / denominator)


def _result_summary(result: Mapping[str, Any]) -> dict[str, float]:
    return {
        "d_Au_Pd_nm": float(result["d_Au_Pd_nm"]),
        "E_mix_V": float(result["E_mix_V"]),
        "i_mix_avg_A_per_m2": float(result["i_mix_avg_A_per_m2"]),
        "i_mix_abs_A": float(result["i_mix_abs_A"]),
        "relative_balance_residual": float(result["relative_balance_residual"]),
    }


def _compare_e_and_i(
    candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    delta_E = abs(float(candidate["E_mix_V"]) - float(reference["E_mix_V"]))
    relative_i = _relative_difference(
        float(candidate["i_mix_avg_A_per_m2"]),
        float(reference["i_mix_avg_A_per_m2"]),
    )
    return {
        "delta_E_V": float(delta_E),
        "relative_delta_i_mix_avg": float(relative_i),
        "E_tolerance_V": E_TOL_V,
        "i_relative_tolerance": I_REL_TOL,
        "passed": bool(delta_E <= E_TOL_V and relative_i <= I_REL_TOL),
    }


def _homogeneous_robin_validation(params: Mapping[str, Any]) -> dict[str, Any]:
    derived = compute_derived_params(params)
    common_g = float(derived["g_Au"])
    common_pzc = float(params["pzc_Au"])
    homogeneous = apply_param_overrides(
        params,
        {
            "d_Au_Pd": 0.0,
            "g_Au": common_g,
            "g_Pd": common_g,
            "pzc_Au": common_pzc,
            "pzc_Pd": common_pzc,
            "dh_violation_action": "ignore",
        },
    )
    model = LinearPBModel(homogeneous)
    test_potential = 0.60
    beta = float(model.derived["beta"])
    driving = beta * (test_potential - common_pzc)
    # u(y)=u(0) exp(-y), and -u_y(0)+g u(0)=g U.
    analytic_surface = common_g * driving / (1.0 + common_g)
    spectral_surface = np.concatenate(
        [
            np.ravel(model.boundary_field("Au", test_potential)),
            np.ravel(model.boundary_field("Pd", test_potential)),
        ]
    )
    max_abs_error = float(np.max(np.abs(spectral_surface - analytic_surface)))
    max_rel_error = float(
        max_abs_error / max(abs(analytic_surface), np.finfo(float).tiny)
    )

    # Quantify the old H=10 truncation for this analytic homogeneous limit.
    H_reference = 10.0
    finite_H10_surface = common_g * driving / (
        1.0 / math.tanh(H_reference) + common_g
    )
    finite_H10_error = abs(finite_H10_surface - analytic_surface)
    return {
        "description": "semi-infinite homogeneous Robin surface solution",
        "test_potential_V": test_potential,
        "common_g": common_g,
        "common_pzc_V": common_pzc,
        "analytic_phi_tilde_surface": float(analytic_surface),
        "spectral_phi_tilde_surface_min": float(np.min(spectral_surface)),
        "spectral_phi_tilde_surface_max": float(np.max(spectral_surface)),
        "spectral_surface_spread": float(
            np.max(spectral_surface) - np.min(spectral_surface)
        ),
        "max_abs_error_phi_tilde": max_abs_error,
        "max_relative_error": max_rel_error,
        "old_finite_H10_reference": {
            "H_over_lambda_D": H_reference,
            "phi_tilde_surface": float(finite_H10_surface),
            "absolute_truncation_error_vs_infinite": float(finite_H10_error),
            "interpretation": (
                "Analytic homogeneous estimate only; the production spectral "
                "model imposes y_tilde -> infinity exactly."
            ),
        },
        "absolute_tolerance_phi_tilde": HOMOGENEOUS_SURFACE_ABS_TOL,
        "relative_tolerance": HOMOGENEOUS_SURFACE_REL_TOL,
        "passed": bool(
            max_abs_error <= HOMOGENEOUS_SURFACE_ABS_TOL
            and max_rel_error <= HOMOGENEOUS_SURFACE_REL_TOL
        ),
    }


def _affine_direct_validation(
    result: Mapping[str, Any], model: LinearPBModel
) -> dict[str, Any]:
    potential = float(result["E_mix_V"])
    phi_metal_tilde = float(model.derived["beta"]) * potential
    rhs = phi_metal_tilde * model.rhs_m - model.rhs_pzc
    direct = np.linalg.solve(model.matrix, rhs)
    affine = model.coefficient_field(potential)
    difference = affine - direct
    max_abs_error = float(np.max(np.abs(difference)))
    relative_l2 = float(
        np.linalg.norm(difference)
        / max(float(np.linalg.norm(direct)), np.finfo(float).tiny)
    )
    direct_residual = model.matrix @ direct - rhs
    direct_relative_residual = float(
        np.linalg.norm(direct_residual)
        / max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
    )
    return {
        "description": "affine spectral coefficients versus direct dense solve",
        "test_potential_V": potential,
        "max_abs_error_phi_tilde_coefficient": max_abs_error,
        "relative_l2_error": relative_l2,
        "direct_relative_residual": direct_relative_residual,
        "absolute_tolerance": AFFINE_FIELD_ABS_TOL,
        "relative_l2_tolerance": AFFINE_FIELD_REL_TOL,
        "passed": bool(
            max_abs_error <= AFFINE_FIELD_ABS_TOL
            and relative_l2 <= AFFINE_FIELD_REL_TOL
        ),
    }


def run_numerical_validation(params: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the semi-infinite spectral solve and its numerical truncations."""

    started = time.perf_counter()
    canonical = apply_param_overrides(params, {"dh_violation_action": "ignore"})
    derived = compute_derived_params(canonical)
    lambda_D = float(derived["lambda_D"])
    mode_levels = (240, 480, 960)
    production_modes = 960
    production_nx = 5000
    coarse_quadrature_nx = 2501

    cache: dict[tuple[float, int, int], tuple[dict[str, Any], LinearPBModel]] = {}

    def solve_cached(
        separation_m: float,
        n_modes: int,
        n_x: int,
    ) -> tuple[dict[str, Any], LinearPBModel]:
        key = (float(separation_m), int(n_modes), int(n_x))
        if key not in cache:
            case_params = apply_param_overrides(
                canonical,
                {
                    "d_Au_Pd": float(separation_m),
                    "N_modes": int(n_modes),
                    "Nx": int(n_x),
                    "dh_violation_action": "ignore",
                },
            )
            solved, model = solve_case(case_params, use_edl=True, return_model=True)
            if model is None:
                raise RuntimeError("with-EDL validation solve returned no spectral model")
            cache[key] = (solved, model)
        return cache[key]

    d0_result, d0_model = solve_cached(0.0, production_modes, production_nx)
    reference_delta_E = abs(float(d0_result["E_mix_V"]) - LEGACY_D0_E_MIX_V)
    reference_relative_i = _relative_difference(
        float(d0_result["i_mix_avg_A_per_m2"]),
        LEGACY_D0_I_MIX_AVG_A_PER_M2,
    )
    adjacent_reference = {
        "description": (
            "d=0 semi-infinite spectral solve versus legacy "
            "N_modes=960/Nx=5000 spectral reference"
        ),
        "reference": {
            "E_mix_V": LEGACY_D0_E_MIX_V,
            "i_mix_avg_A_per_m2": LEGACY_D0_I_MIX_AVG_A_PER_M2,
        },
        "computed": _result_summary(d0_result),
        "delta_E_V": float(reference_delta_E),
        "relative_delta_i_mix_avg": float(reference_relative_i),
        "E_tolerance_V": E_TOL_V,
        "i_relative_tolerance": I_REL_TOL,
        "passed": bool(
            reference_delta_E <= E_TOL_V and reference_relative_i <= I_REL_TOL
        ),
    }

    old_finite_H10_reference = {
        "description": (
            "Archived finite-height Q1 H=10 lambda_D d=0 regression reference; retained "
            "only to quantify the previous finite-domain approximation"
        ),
        "reference": {
            "E_mix_V": FINITE_H10_Q1_D0_E_MIX_V,
            "i_mix_avg_A_per_m2": FINITE_H10_Q1_D0_I_MIX_AVG_A_PER_M2,
        },
        "computed": _result_summary(d0_result),
        "delta_E_V": float(
            float(d0_result["E_mix_V"]) - FINITE_H10_Q1_D0_E_MIX_V
        ),
        "relative_delta_i_mix_avg": float(
            (
                float(d0_result["i_mix_avg_A_per_m2"])
                - FINITE_H10_Q1_D0_I_MIX_AVG_A_PER_M2
            )
            / FINITE_H10_Q1_D0_I_MIX_AVG_A_PER_M2
        ),
        "not_a_pass_fail_check": True,
    }

    homogeneous_robin = _homogeneous_robin_validation(canonical)
    affine_direct = _affine_direct_validation(d0_result, d0_model)

    separations = [
        ("d0", 0.0),
        ("one_lambda_D", lambda_D),
        ("d100nm", 100.0e-9),
    ]
    mode_cases: list[dict[str, Any]] = []
    quadrature_cases: list[dict[str, Any]] = []
    for label, separation in separations:
        by_modes = {
            n_modes: solve_cached(separation, n_modes, production_nx)[0]
            for n_modes in mode_levels
        }
        mode_comparison = _compare_e_and_i(by_modes[480], by_modes[960])
        mode_cases.append(
            {
                "label": label,
                "d_Au_Pd_m": float(separation),
                "d_Au_Pd_nm": float(separation * 1.0e9),
                "results": {
                    f"N{n_modes}": _result_summary(by_modes[n_modes])
                    for n_modes in mode_levels
                },
                "comparison_N240_vs_N960": _compare_e_and_i(
                    by_modes[240], by_modes[960]
                ),
                "comparison_N480_vs_N960": mode_comparison,
                "passed": bool(mode_comparison["passed"]),
            }
        )

        coarse_quadrature, _ = solve_cached(
            separation, production_modes, coarse_quadrature_nx
        )
        production_result = by_modes[production_modes]
        quadrature_comparison = _compare_e_and_i(
            coarse_quadrature, production_result
        )
        quadrature_cases.append(
            {
                "label": label,
                "d_Au_Pd_m": float(separation),
                "d_Au_Pd_nm": float(separation * 1.0e9),
                "coarse": _result_summary(coarse_quadrature),
                "production": _result_summary(production_result),
                "comparison_Nx2501_vs_Nx5000": quadrature_comparison,
                "passed": bool(quadrature_comparison["passed"]),
            }
        )

    spectral_mode_convergence = {
        "description": "cosine truncation convergence at N_modes=240/480/960",
        "N_modes": list(mode_levels),
        "pass_comparison": "N480 versus N960",
        "cases": mode_cases,
        "passed": bool(all(case["passed"] for case in mode_cases)),
    }
    surface_quadrature_convergence = {
        "description": "surface trapezoidal convergence at Nx=2501/5000",
        "coarse_Nx": coarse_quadrature_nx,
        "production_Nx": production_nx,
        "cases": quadrature_cases,
        "passed": bool(all(case["passed"] for case in quadrature_cases)),
    }
    semi_infinite_farfield = {
        "description": (
            "No finite top boundary: every mode contains exp(-gamma_n y_tilde), "
            "so phi_tilde -> 0 is imposed analytically as y_tilde -> infinity."
        ),
        "finite_H_control_applicable": False,
        "passed": True,
    }

    sections = [
        adjacent_reference,
        homogeneous_robin,
        affine_direct,
        spectral_mode_convergence,
        surface_quadrature_convergence,
        semi_infinite_farfield,
    ]
    elapsed = time.perf_counter() - started
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "electrostatic_backend": ELECTROSTATIC_BACKEND,
        "model": MODEL_NAME,
        "geometry": GEOMETRY_NAME,
        "validation": "semi-infinite linear-PB cosine/Fourier spectral validation",
        "debye_huckel_violation_action": "ignore",
        "thresholds": {
            "E_mix_absolute_V": E_TOL_V,
            "i_mix_relative": I_REL_TOL,
            "homogeneous_surface_absolute_phi_tilde": HOMOGENEOUS_SURFACE_ABS_TOL,
            "homogeneous_surface_relative": HOMOGENEOUS_SURFACE_REL_TOL,
            "affine_coefficient_absolute": AFFINE_FIELD_ABS_TOL,
            "affine_coefficient_relative_l2": AFFINE_FIELD_REL_TOL,
        },
        "adjacent_legacy_reference": adjacent_reference,
        "archived_finite_H10_Q1_reference": old_finite_H10_reference,
        "homogeneous_robin": homogeneous_robin,
        "affine_direct_solve": affine_direct,
        "semi_infinite_farfield": semi_infinite_farfield,
        "spectral_mode_convergence": spectral_mode_convergence,
        "surface_quadrature_convergence": surface_quadrature_convergence,
        "cache": {
            "unique_mixed_potential_spectral_solves": int(len(cache)),
            "homogeneous_electrostatic_solves": 1,
        },
        "runtime_seconds": float(elapsed),
        "passed": bool(all(section["passed"] for section in sections)),
    }
