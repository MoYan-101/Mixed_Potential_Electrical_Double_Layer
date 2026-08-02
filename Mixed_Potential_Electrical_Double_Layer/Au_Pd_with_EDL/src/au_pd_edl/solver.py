"""Mixed-potential solution for flush Au/Pd on an insulating substrate."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from typing import Any, overload

import numpy as np
from scipy.optimize import root_scalar

from .electrostatics import LinearPBModel
from .kinetics import (
    au_local_current_density,
    effective_reaction_params,
    emix_closed_form_no_edl,
    kinetics_context,
    no_edl_absolute_currents,
    pd_local_current_density,
)
from .parameters import (
    GEOMETRY_NAME,
    MODEL_NAME,
    apply_param_overrides,
    compute_derived_params,
)
from .surface_charge import compute_gouy_chapman_diagnostics


def _with_separation(
    params: Mapping[str, Any], separation_m: float | None
) -> dict[str, Any]:
    overrides = {} if separation_m is None else {"d_Au_Pd": float(separation_m)}
    return apply_param_overrides(params, overrides)


def _current_components_with_edl(
    E: float,
    params: Mapping[str, Any],
    model: LinearPBModel,
) -> dict[str, float]:
    """Integrate signed absolute currents over the two horizontal interfaces."""

    ctx = kinetics_context(float(E), params)
    derived = model.derived
    au_data = model.boundary_quadrature["Au"]
    pd_data = model.boundary_quadrature["Pd"]
    phi_au = model.boundary_field("Au", float(E))
    phi_pd = model.boundary_field("Pd", float(E))
    au_field_integral = model.integrate_boundary_exponential(
        "Au", float(E), -float(ctx["Gamma1"])
    )
    pd_field_integral = model.integrate_boundary_exponential(
        "Pd", float(E), float(ctx["Gamma2"])
    )
    au_base = (
        (1.0 - float(ctx["alpha1"]))
        * float(ctx["beta"])
        * float(ctx["eta1"])
    )
    pd_base = (
        -float(ctx["alpha2"])
        * float(ctx["beta"])
        * float(ctx["eta2"])
    )
    # Clip the complete local exponent, not separate products, so extreme but
    # finite inputs cannot overflow by multiplying two independently clipped
    # exponentials.
    au_integral = float(
        np.sum(
            np.exp(
                np.clip(
                    au_base - float(ctx["Gamma1"]) * phi_au,
                    -700.0,
                    700.0,
                )
            )
            * au_data.weights_tilde
        )
    )
    pd_integral = float(
        np.sum(
            np.exp(
                np.clip(
                    pd_base + float(ctx["Gamma2"]) * phi_pd,
                    -700.0,
                    700.0,
                )
            )
            * pd_data.weights_tilde
        )
    )
    dimensional_factor = (
        float(derived["lambda_D"]) * float(derived["out_of_plane_width"])
    )
    I_au = dimensional_factor * float(ctx["it0_1"]) * au_integral
    I_pd = -dimensional_factor * float(ctx["it0_2"]) * pd_integral
    denominator = abs(I_au) + abs(I_pd)
    residual = I_au + I_pd
    return {
        "I_Au_A": float(I_au),
        "I_Pd_A": float(I_pd),
        "residual_A": float(residual),
        "relative_balance_residual": (
            float(abs(residual) / denominator) if denominator else 0.0
        ),
        "i_mix_abs_A": float(abs(I_au)),
        "i_mix_avg_A_per_m2": float(
            abs(I_au) / float(derived["reactive_area_m2"])
        ),
        "Au_dimensionless_field_factor_integral": float(au_field_integral),
        "Pd_dimensionless_field_factor_integral": float(pd_field_integral),
    }


def current_balance_at_potential(
    E: float,
    params: Mapping[str, Any],
    *,
    use_edl: bool,
    model: LinearPBModel | None = None,
) -> dict[str, float]:
    """Return signed Au/Pd absolute currents and their balance at ``E``."""

    canonical = apply_param_overrides(params)
    if not use_edl:
        return no_edl_absolute_currents(float(E), canonical)
    active_model = model if model is not None else LinearPBModel(canonical)
    if active_model.params != canonical:
        raise ValueError("The supplied LinearPBModel does not match params")
    return _current_components_with_edl(float(E), canonical, active_model)


def _mixed_potential_root(
    params: Mapping[str, Any], model: LinearPBModel
) -> tuple[float, dict[str, float], dict[str, Any]]:
    rxn = effective_reaction_params(params)
    padding = float(params["root_bracket_padding_V"])
    lower = min(float(rxn["E1_eq_eff"]), float(rxn["E2_eq_eff"])) - padding
    upper = max(float(rxn["E1_eq_eff"]), float(rxn["E2_eq_eff"])) + padding

    def residual(E: float) -> float:
        return _current_components_with_edl(E, params, model)["residual_A"]

    f_lower = residual(lower)
    f_upper = residual(upper)
    expands = 0
    while f_lower * f_upper > 0.0 and expands < int(params["max_bracket_expands"]):
        padding *= 2.0
        lower = min(float(rxn["E1_eq_eff"]), float(rxn["E2_eq_eff"])) - padding
        upper = max(float(rxn["E1_eq_eff"]), float(rxn["E2_eq_eff"])) + padding
        f_lower = residual(lower)
        f_upper = residual(upper)
        expands += 1
    if f_lower * f_upper > 0.0:
        raise RuntimeError(
            "Could not bracket I_Au + I_Pd = 0 after "
            f"{expands} expansions: f({lower})={f_lower}, f({upper})={f_upper}"
        )

    root = root_scalar(
        residual,
        bracket=(lower, upper),
        method="brentq",
        xtol=float(params["root_xtol_V"]),
        rtol=float(params["root_rtol"]),
        maxiter=int(params["root_maxiter"]),
    )
    if not root.converged:
        raise RuntimeError(f"Mixed-potential root solve did not converge: {root.flag}")
    E_mix = float(root.root)
    currents = _current_components_with_edl(E_mix, params, model)
    metadata = {
        "method": "brentq",
        "converged": bool(root.converged),
        "iterations": int(root.iterations),
        "function_calls": int(root.function_calls),
        "bracket_V": [float(lower), float(upper)],
        "bracket_expansions": int(expands),
        "flag": str(root.flag),
    }
    return E_mix, currents, metadata


def _dh_status(
    params: Mapping[str, Any], model: LinearPBModel, E_mix: float
) -> dict[str, Any]:
    field = model.surface_field(E_mix)
    maximum = float(np.max(np.abs(field)))
    threshold = float(params["dh_warn_threshold"])
    exceeded = maximum > threshold
    message = (
        "Linearized-PB validity threshold exceeded; treat this run as an "
        "internal geometry-sensitivity result, not an absolute quantitative prediction."
    )
    if exceeded:
        action = str(params["dh_violation_action"]).lower()
        if action == "raise":
            raise RuntimeError(f"{message} max|phi_tilde|={maximum:.6g}")
        if action == "warn":
            warnings.warn(message, RuntimeWarning, stacklevel=3)
    return {
        "max_abs_phi_tilde": maximum,
        "threshold": threshold,
        "threshold_exceeded": bool(exceeded),
        "interpretation": "linearized-PB internal geometry sensitivity" if exceeded else "within configured threshold",
    }


@overload
def solve_case(
    params: Mapping[str, Any],
    separation_m: float | None = None,
    *,
    use_edl: bool = True,
    return_model: bool = False,
) -> dict[str, Any]: ...


@overload
def solve_case(
    params: Mapping[str, Any],
    separation_m: float | None = None,
    *,
    use_edl: bool = True,
    return_model: bool,
) -> tuple[dict[str, Any], LinearPBModel | None]: ...


def solve_case(
    params: Mapping[str, Any],
    separation_m: float | None = None,
    *,
    use_edl: bool = True,
    return_model: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], LinearPBModel | None]:
    """Solve one flush Au|insulating-substrate-gap|Pd case.

    The two metals share one electronic potential through the ideal external
    wire.  The root is always the absolute-current condition
    ``I_Au + I_Pd = 0``.
    """

    canonical = _with_separation(params, separation_m)
    derived = compute_derived_params(canonical)
    rxn = effective_reaction_params(canonical)
    model: LinearPBModel | None = None
    if use_edl:
        model = LinearPBModel(canonical)
        E_mix, currents, root_metadata = _mixed_potential_root(canonical, model)
        dh = _dh_status(canonical, model, E_mix)
        affine_residuals = model.affine_residuals()
        spectral_metadata = model.spectral_metadata()
    else:
        E_mix = emix_closed_form_no_edl(canonical, derived)
        currents = no_edl_absolute_currents(E_mix, canonical, derived)
        root_metadata = {
            "method": "closed_form",
            "converged": True,
            "iterations": 0,
            "function_calls": 0,
            "bracket_V": None,
            "bracket_expansions": 0,
            "flag": "exact",
        }
        dh = {
            "max_abs_phi_tilde": 0.0,
            "threshold": float(canonical["dh_warn_threshold"]),
            "threshold_exceeded": False,
            "interpretation": "w/o EDL reference",
        }
        affine_residuals = None
        spectral_metadata = None

    result: dict[str, Any] = {
        "model": MODEL_NAME,
        "geometry": GEOMETRY_NAME,
        "electronic_coupling": "ideal external wire; shared E_mix",
        "pb_model": (
            "linearized Poisson-Boltzmann / Debye-Huckel; "
            "semi-infinite cosine/Fourier spectral solution"
            if use_edl
            else "disabled"
        ),
        "use_edl": bool(use_edl),
        "condition_label": "with EDL" if use_edl else "w/o EDL",
        "d_Au_Pd_m": float(derived["d_Au_Pd"]),
        "d_Au_Pd_nm": float(derived["d_Au_Pd"] * 1.0e9),
        "E_mix_V": float(E_mix),
        **currents,
        "root": root_metadata,
        "debye_huckel_validity": dh,
        "affine_residuals": affine_residuals,
        "electrostatics": spectral_metadata,
        "effective_reaction": rxn,
        "params": canonical,
        "derived": derived,
    }
    result["gouy_chapman"] = compute_gouy_chapman_diagnostics(result, model)
    if return_model:
        return result, model
    return result


def run_edl_comparison_pair(
    params: Mapping[str, Any],
    separation_m: float | None = None,
    *,
    return_model: bool = False,
) -> dict[str, Any]:
    """Solve with-EDL and separation-invariant w/o-EDL references."""

    with_result, model = solve_case(
        params, separation_m, use_edl=True, return_model=True
    )
    without_result = solve_case(params, separation_m, use_edl=False)
    comparison = {
        "d_Au_Pd_m": with_result["d_Au_Pd_m"],
        "d_Au_Pd_nm": with_result["d_Au_Pd_nm"],
        "with_edl": with_result,
        "without_edl": without_result,
        "delta_E_with_minus_without_V": (
            float(with_result["E_mix_V"]) - float(without_result["E_mix_V"])
        ),
        "delta_i_with_minus_without_A_per_m2": (
            float(with_result["i_mix_avg_A_per_m2"])
            - float(without_result["i_mix_avg_A_per_m2"])
        ),
    }
    if return_model:
        comparison["model_instance"] = model
    return comparison


def compute_polarization_curve(
    params: Mapping[str, Any],
    potentials_V: Iterable[float],
    separation_m: float | None = None,
    *,
    use_edl: bool = True,
) -> dict[str, np.ndarray]:
    """Compute signed absolute Au/Pd half-reaction currents versus potential."""

    canonical = _with_separation(params, separation_m)
    potentials = np.asarray(list(potentials_V), dtype=float)
    if potentials.ndim != 1 or potentials.size == 0 or not np.all(np.isfinite(potentials)):
        raise ValueError("potentials_V must be a non-empty finite 1D sequence")
    model = LinearPBModel(canonical) if use_edl else None
    au = np.empty_like(potentials)
    pd = np.empty_like(potentials)
    for index, potential in enumerate(potentials):
        if use_edl:
            values = _current_components_with_edl(float(potential), canonical, model)  # type: ignore[arg-type]
        else:
            values = no_edl_absolute_currents(float(potential), canonical)
        au[index] = values["I_Au_A"]
        pd[index] = values["I_Pd_A"]
    return {
        "potential_V": potentials,
        "I_Au_A": au,
        "I_Pd_A": pd,
        "I_total_A": au + pd,
    }


def top_surface_profiles(
    result: Mapping[str, Any],
    model: LinearPBModel,
    *,
    n_x: int = 2001,
) -> dict[str, np.ndarray]:
    """Return RP potential, concentrations, and local currents on a regular x grid."""

    E_mix = float(result["E_mix_V"])
    params = result["params"]
    derived = result["derived"]
    profile = model.top_profile(E_mix, n_x=n_x)
    x_tilde = profile["x_tilde"]
    x_m = x_tilde * float(derived["lambda_D"])
    phi = profile["phi_tilde"]
    c_red = np.exp(np.clip(-float(params["z_R1"]) * phi, -700.0, 700.0))
    c_ox = np.exp(np.clip(-float(params["z_O2"]) * phi, -700.0, 700.0))
    j_au = np.full(phi.shape, np.nan, dtype=float)
    j_pd = np.full(phi.shape, np.nan, dtype=float)
    au_mask = x_m <= float(derived["x_Au_end"]) + 1.0e-15
    pd_mask = x_m >= float(derived["x_Pd_start"]) - 1.0e-15
    j_au[au_mask] = np.asarray(
        au_local_current_density(E_mix, phi[au_mask], params), dtype=float
    )
    j_pd[pd_mask] = np.asarray(
        pd_local_current_density(E_mix, phi[pd_mask], params), dtype=float
    )
    return {
        "x_m": x_m,
        "x_nm": x_m * 1.0e9,
        "phi_tilde": phi,
        "phi_s_V": phi * float(derived["thermal_voltage_V"]),
        "c_Red1_over_c_bulk": c_red,
        "c_Ox2_over_c_bulk": c_ox,
        "j_Au_A_per_m2": j_au,
        "j_Pd_A_per_m2": j_pd,
    }
