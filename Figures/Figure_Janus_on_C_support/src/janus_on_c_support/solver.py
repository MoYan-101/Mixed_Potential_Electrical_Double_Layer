"""Gauss--Legendre absolute-current mixed-potential solver."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

import numpy as np
from scipy.optimize import root_scalar

from .electrostatics import LinearPBModel
from .kinetics import effective_reaction_params, safe_exp
from .parameters import (
    ELECTROSTATIC_BACKEND,
    GEOMETRY_NAME,
    MODEL_NAME,
    RESULT_SCHEMA_VERSION,
    TOPOLOGY_ID,
    canonical_params,
    compute_derived_params,
)


CURRENT_BALANCE_RELATIVE_TOLERANCE = 1.0e-10


def _mapped_legendre_interval(
    a: float, b: float, order: int
) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    half_width = 0.5 * (b - a)
    midpoint = 0.5 * (a + b)
    return midpoint + half_width * nodes, half_width * weights


def _prepare_quadrature(model: LinearPBModel) -> dict[str, np.ndarray]:
    order = int(model.params["gl_order"])
    lambda_D = float(model.derived["lambda_D_m"])
    au = next(segment for segment in model.segments if segment.name == "Au")
    pd = next(segment for segment in model.segments if segment.name == "Pd")
    x_au, w_au = _mapped_legendre_interval(
        au.x_start_m / lambda_D, au.x_end_m / lambda_D, order
    )
    x_pd, w_pd = _mapped_legendre_interval(
        pd.x_start_m / lambda_D, pd.x_end_m / lambda_D, order
    )
    combined_m = np.concatenate((x_au, x_pd)) * lambda_D
    phi_m, phi_pzc = model.affine_surface_components(combined_m)
    split = x_au.size
    return {
        "x_Au_tilde": x_au,
        "w_Au_tilde": w_au,
        "phi_M_Au": phi_m[:split],
        "phi_pzc_Au": phi_pzc[:split],
        "x_Pd_tilde": x_pd,
        "w_Pd_tilde": w_pd,
        "phi_M_Pd": phi_m[split:],
        "phi_pzc_Pd": phi_pzc[split:],
    }


def _kinetic_constants(
    params: Mapping[str, Any], derived: Mapping[str, Any]
) -> dict[str, float]:
    reaction = effective_reaction_params(params)
    alpha1 = float(params["alpha1"])
    alpha2 = float(params["alpha2"])
    return {
        **reaction,
        "beta_per_V": float(derived["beta_per_V"]),
        "alpha1": alpha1,
        "alpha2": alpha2,
        "Gamma1": (1.0 - alpha1) + float(params["z_R1"]),
        "Gamma2": alpha2 - float(params["z_O2"]),
    }


def _integrated_currents(
    E_V: float,
    *,
    params: Mapping[str, Any],
    derived: Mapping[str, Any],
    constants: Mapping[str, float],
    use_edl: bool,
    quadrature: Mapping[str, np.ndarray] | None,
) -> dict[str, float]:
    beta = float(constants["beta_per_V"])
    if use_edl:
        if quadrature is None:
            raise ValueError("with-EDL currents require prepared quadrature")
        beta_e = beta * float(E_V)
        phi_au = quadrature["phi_M_Au"] * beta_e - quadrature["phi_pzc_Au"]
        phi_pd = quadrature["phi_M_Pd"] * beta_e - quadrature["phi_pzc_Pd"]
        integral_au_tilde = float(
            np.dot(
                quadrature["w_Au_tilde"],
                safe_exp(-float(constants["Gamma1"]) * phi_au),
            )
        )
        integral_pd_tilde = float(
            np.dot(
                quadrature["w_Pd_tilde"],
                safe_exp(float(constants["Gamma2"]) * phi_pd),
            )
        )
    else:
        lambda_D = float(derived["lambda_D_m"])
        integral_au_tilde = float(params["L_Au"]) / lambda_D
        integral_pd_tilde = float(params["L_Pd"]) / lambda_D

    prefactor_au = float(constants["it0_1_eff_A_per_m2"]) * float(
        safe_exp(
            (1.0 - float(constants["alpha1"]))
            * beta
            * (float(E_V) - float(constants["E1_eq_eff_V"]))
        )
    )
    prefactor_pd = -float(constants["it0_2_eff_A_per_m2"]) * float(
        safe_exp(
            -float(constants["alpha2"])
            * beta
            * (float(E_V) - float(constants["E2_eq_eff_V"]))
        )
    )
    current_scale = float(derived["lambda_D_m"]) * float(
        params["out_of_plane_width"]
    )
    current_au = prefactor_au * integral_au_tilde * current_scale
    current_pd = prefactor_pd * integral_pd_tilde * current_scale
    residual = current_au + current_pd
    denominator = abs(current_au) + abs(current_pd)
    return {
        "I_Au_A": float(current_au),
        "I_Pd_A": float(current_pd),
        "I_C_A": 0.0,
        "residual_A": float(residual),
        "relative_balance_residual": float(
            abs(residual) / denominator if denominator else 0.0
        ),
    }


def integrated_currents_A(
    E_V: float,
    params: Mapping[str, Any] | None = None,
    *,
    use_edl: bool = True,
    model: LinearPBModel | None = None,
) -> dict[str, float]:
    """Public absolute Au/Pd current components at a specified potential."""

    if model is not None and params is not None:
        raise ValueError("Pass either params or model, not both")
    if model is None:
        canonical = canonical_params(params)
        derived = compute_derived_params(canonical)
        model = LinearPBModel(canonical) if use_edl else None
    else:
        canonical = model.params
        derived = model.derived
    constants = _kinetic_constants(canonical, derived)
    quadrature = _prepare_quadrature(model) if use_edl and model is not None else None
    return _integrated_currents(
        float(E_V),
        params=canonical,
        derived=derived,
        constants=constants,
        use_edl=bool(use_edl),
        quadrature=quadrature,
    )


def _solve_root(
    *,
    params: Mapping[str, Any],
    derived: Mapping[str, Any],
    constants: Mapping[str, float],
    use_edl: bool,
    quadrature: Mapping[str, np.ndarray] | None,
) -> tuple[float, dict[str, Any]]:
    def residual(E_V: float) -> float:
        return _integrated_currents(
            E_V,
            params=params,
            derived=derived,
            constants=constants,
            use_edl=use_edl,
            quadrature=quadrature,
        )["residual_A"]

    lower_eq = min(
        float(constants["E1_eq_eff_V"]), float(constants["E2_eq_eff_V"])
    )
    upper_eq = max(
        float(constants["E1_eq_eff_V"]), float(constants["E2_eq_eff_V"])
    )
    padding = float(params["root_bracket_padding_V"])
    lower = lower_eq - padding
    upper = upper_eq + padding
    f_lower = residual(lower)
    f_upper = residual(upper)
    expansions = 0
    while (
        np.sign(f_lower) == np.sign(f_upper)
        and expansions < int(params["max_bracket_expands"])
    ):
        padding *= 2.0
        lower = lower_eq - padding
        upper = upper_eq + padding
        f_lower = residual(lower)
        f_upper = residual(upper)
        expansions += 1
    if not math.isfinite(f_lower) or not math.isfinite(f_upper):
        raise RuntimeError("Mixed-potential bracket produced non-finite current")
    if np.sign(f_lower) == np.sign(f_upper):
        raise RuntimeError("Could not bracket I_Au + I_Pd = 0")

    root = root_scalar(
        residual,
        bracket=(lower, upper),
        method="brentq",
        xtol=float(params["root_xtol_V"]),
        rtol=float(params["root_rtol"]),
        maxiter=int(params["root_maxiter"]),
    )
    if not root.converged:
        raise RuntimeError(f"Mixed-potential root solve failed: {root.flag}")
    return float(root.root), {
        "method": "brentq",
        "converged": bool(root.converged),
        "iterations": int(root.iterations),
        "function_calls": int(root.function_calls),
        "bracket_V": [float(lower), float(upper)],
        "f_lower_A": float(f_lower),
        "f_upper_A": float(f_upper),
        "bracket_expansions": int(expansions),
        "xtol_V": float(params["root_xtol_V"]),
        "rtol": float(params["root_rtol"]),
    }


def _json_roundtrip(value: Mapping[str, Any]) -> dict[str, Any]:
    """Enforce that public results are strictly JSON serializable."""

    return json.loads(
        json.dumps(value, allow_nan=False, sort_keys=True, ensure_ascii=False)
    )


def solve_case(
    params: Mapping[str, Any] | None = None,
    use_edl: bool = True,
    *,
    model: LinearPBModel | None = None,
) -> dict[str, Any]:
    """Solve one with-EDL or w/o-EDL absolute-current balance.

    ``I_Au_A`` and ``I_Pd_A`` are currents in the 18 nm mirror half-cell.
    The 36 nm full-period mixed current is exactly twice the half-cell value.
    ``i_mix_avg_A_per_m2`` is normalized only by the Au+Pd reactive area and
    therefore is independent of choosing the half-cell or full period.
    """

    if model is not None and params is not None:
        raise ValueError("Pass either params or model, not both")
    if model is not None:
        canonical = model.params
        derived = model.derived
    else:
        canonical = canonical_params(params)
        derived = compute_derived_params(canonical)
        if use_edl:
            model = LinearPBModel(canonical)

    constants = _kinetic_constants(canonical, derived)
    quadrature = _prepare_quadrature(model) if use_edl and model is not None else None
    E_mix, root_info = _solve_root(
        params=canonical,
        derived=derived,
        constants=constants,
        use_edl=bool(use_edl),
        quadrature=quadrature,
    )
    currents = _integrated_currents(
        E_mix,
        params=canonical,
        derived=derived,
        constants=constants,
        use_edl=bool(use_edl),
        quadrature=quadrature,
    )
    relative_balance = float(currents["relative_balance_residual"])
    if relative_balance >= CURRENT_BALANCE_RELATIVE_TOLERANCE:
        raise RuntimeError(
            "Absolute-current balance failed: "
            f"relative residual={relative_balance:.6g}"
        )

    halfcell_current = abs(float(currents["I_Au_A"]))
    full_period_current = 2.0 * halfcell_current
    reactive_area = float(derived["reactive_area_halfcell_m2"])
    if use_edl:
        if model is None:
            raise AssertionError("with-EDL solve lost its electrostatic model")
        phi_surface = model.phi_tilde_surface(E_mix, model.surface_grid_m())
        max_abs_phi = float(np.max(np.abs(phi_surface)))
    else:
        max_abs_phi = 0.0

    threshold = float(canonical["dh_warn_threshold"])
    result = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "model_name": MODEL_NAME,
        "geometry_name": GEOMETRY_NAME,
        "topology_id": TOPOLOGY_ID,
        "electrostatic_backend": ELECTROSTATIC_BACKEND if use_edl else "disabled",
        "condition": "with EDL" if use_edl else "w/o EDL",
        "use_edl": bool(use_edl),
        "electronic_constraint": "shared E_mix; I_Au + I_Pd = 0",
        "E_mix_V": float(E_mix),
        **currents,
        # Concise aliases retained for downstream validation code.
        "residual": float(currents["residual_A"]),
        "relative_balance": relative_balance,
        "i_mix_abs_halfcell_A": float(halfcell_current),
        "i_mix_abs_full_period_A": float(full_period_current),
        "i_mix_avg_A_per_m2": float(halfcell_current / reactive_area),
        "max_abs_phi_tilde": float(max_abs_phi),
        "debye_huckel_validity": {
            "max_abs_phi_tilde": float(max_abs_phi),
            "threshold": threshold,
            "threshold_exceeded": bool(max_abs_phi > threshold),
        },
        "E1_eq_eff_V": float(constants["E1_eq_eff_V"]),
        "E2_eq_eff_V": float(constants["E2_eq_eff_V"]),
        "it0_1_eff_A_per_m2": float(constants["it0_1_eff_A_per_m2"]),
        "it0_2_eff_A_per_m2": float(constants["it0_2_eff_A_per_m2"]),
        "quadrature": {
            "kind": "Gauss-Legendre" if use_edl else "uniform analytic limit",
            "order": int(canonical["gl_order"]),
            "integrated_segments": ["Au", "Pd"],
            "C_is_non_faradaic": True,
        },
        "root": root_info,
        "params": canonical,
        "derived": derived,
    }
    return _json_roundtrip(result)


def solve_comparison(params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Solve both EDL conditions and return one JSON-safe comparison."""

    canonical = canonical_params(params)
    model = LinearPBModel(canonical)
    with_edl = solve_case(use_edl=True, model=model)
    without_edl = solve_case(canonical, use_edl=False)
    result = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "model_name": MODEL_NAME,
        "geometry_name": GEOMETRY_NAME,
        "topology_id": TOPOLOGY_ID,
        "params": canonical,
        "derived": model.derived,
        "with_edl": with_edl,
        "without_edl": without_edl,
        "delta_E_with_minus_without_V": float(
            with_edl["E_mix_V"] - without_edl["E_mix_V"]
        ),
        "delta_i_with_minus_without_A_per_m2": float(
            with_edl["i_mix_avg_A_per_m2"]
            - without_edl["i_mix_avg_A_per_m2"]
        ),
    }
    return _json_roundtrip(result)

