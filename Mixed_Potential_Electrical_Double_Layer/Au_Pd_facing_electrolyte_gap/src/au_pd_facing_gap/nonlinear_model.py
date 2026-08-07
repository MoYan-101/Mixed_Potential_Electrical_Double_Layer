"""Full nonlinear Poisson--Boltzmann model for the facing Au/Pd slit.

The finite slit is solved from ``psi'' = sinh(psi)`` with the same
Stern--Robin boundary conditions as the linear model.  The familiar
Gouy--Chapman ``asinh`` relation is used only for the isolated planar limit,
where a bulk plateau exists.  It is not imposed independently at the two
walls of an overlapping slit.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from scipy.integrate import simpson, solve_bvp
from scipy.optimize import root_scalar

from .model import (
    RESULT_SCHEMA_VERSION,
    TOPOLOGY_ID,
    FacingElectrolyteSlitModel,
    canonical_params,
    compute_derived,
    effective_reaction_params,
)


NONLINEAR_MODEL_ID = "facing_au_electrolyte_pd_shared_mixed_potential_nonlinear_pb"
NONLINEAR_ELECTROSTATIC_BACKEND = "nonlinear_pb_finite_slit_with_gc_planar_asymptote"


def gouy_chapman_stern_surface_phi_tilde(
    E_V: float,
    pzc_V: float,
    g: float,
    beta_per_V: float,
) -> float:
    """Return the isolated-planar nonlinear GC--Stern surface potential.

    For a symmetric 1:1 electrolyte, the decaying PB branch gives

        2 sinh(psi_s / 2) = g [beta (E - PZC) - psi_s].

    Equivalently, with dimensionless diffuse charge ``q_D``, the
    Gouy--Chapman relation is ``psi_s = 2 asinh(q_D / 2)``.
    """

    u = float(beta_per_V) * (float(E_V) - float(pzc_V))
    g_value = float(g)
    if g_value == 0.0:
        return 0.0

    def residual(psi: float) -> float:
        return 2.0 * math.sinh(0.5 * psi) + g_value * (psi - u)

    lower, upper = min(0.0, u), max(0.0, u)
    if lower == upper:
        return lower
    root = root_scalar(
        residual,
        bracket=(lower, upper),
        method="brentq",
        xtol=1.0e-13,
        rtol=1.0e-14,
        maxiter=200,
    )
    if not root.converged:
        raise RuntimeError("Isolated Gouy--Chapman--Stern surface solve failed")
    return float(root.root)


def _log_current_magnitudes(
    params: Mapping[str, Any],
    derived: Mapping[str, float],
    reaction: Mapping[str, float],
    E_V: float,
    phi_au: float,
    phi_pd: float,
) -> tuple[float, float]:
    beta = float(derived["beta_per_V"])
    gamma_au = (1.0 - float(params["alpha1"])) + float(params["z_R1"])
    gamma_pd = float(params["alpha2"]) - float(params["z_O2"])
    log_au = (
        math.log(float(derived["area_Au_m2"]) * float(reaction["it0_1_eff"]))
        + (1.0 - float(params["alpha1"]))
        * beta
        * (float(E_V) - float(reaction["E1_eq_eff"]))
        - gamma_au * float(phi_au)
    )
    log_pd = (
        math.log(float(derived["area_Pd_m2"]) * float(reaction["it0_2_eff"]))
        - float(params["alpha2"])
        * beta
        * (float(E_V) - float(reaction["E2_eq_eff"]))
        + gamma_pd * float(phi_pd)
    )
    return log_au, log_pd


def solve_planar_gouy_chapman_stern_limit(
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Solve the independent-planar nonlinear GC--Stern mixed-potential limit."""

    p = canonical_params(params)
    derived = compute_derived(p)
    reaction = effective_reaction_params(p)

    def surfaces(E_V: float) -> tuple[float, float]:
        return (
            gouy_chapman_stern_surface_phi_tilde(
                E_V,
                float(p["pzc_Au"]),
                derived["g_Au"],
                derived["beta_per_V"],
            ),
            gouy_chapman_stern_surface_phi_tilde(
                E_V,
                float(p["pzc_Pd"]),
                derived["g_Pd"],
                derived["beta_per_V"],
            ),
        )

    def balance(E_V: float) -> float:
        phi_au, phi_pd = surfaces(E_V)
        log_au, log_pd = _log_current_magnitudes(
            p, derived, reaction, E_V, phi_au, phi_pd
        )
        return log_au - log_pd

    padding = float(p["root_bracket_padding_V"])
    lower = min(reaction["E1_eq_eff"], reaction["E2_eq_eff"]) - padding
    upper = max(reaction["E1_eq_eff"], reaction["E2_eq_eff"]) + padding
    f_lower, f_upper = balance(lower), balance(upper)
    expansions = 0
    while f_lower * f_upper > 0.0 and expansions < int(p["max_bracket_expands"]):
        padding *= 2.0
        lower = min(reaction["E1_eq_eff"], reaction["E2_eq_eff"]) - padding
        upper = max(reaction["E1_eq_eff"], reaction["E2_eq_eff"]) + padding
        f_lower, f_upper = balance(lower), balance(upper)
        expansions += 1
    if f_lower * f_upper > 0.0:
        raise RuntimeError("Could not bracket planar nonlinear GC--Stern mixed potential")
    root = root_scalar(
        balance,
        bracket=(lower, upper),
        method="brentq",
        xtol=float(p["root_xtol_V"]),
        rtol=float(p["root_rtol"]),
        maxiter=int(p["root_maxiter"]),
    )
    if not root.converged:
        raise RuntimeError("Planar nonlinear GC--Stern mixed-potential solve failed")
    E_mix = float(root.root)
    phi_au, phi_pd = surfaces(E_mix)
    log_au, log_pd = _log_current_magnitudes(
        p, derived, reaction, E_mix, phi_au, phi_pd
    )
    I_au = math.exp(log_au)
    I_pd = -math.exp(log_pd)
    thermal = derived["thermal_voltage_V"]
    sigma_au = float(p["C_H_Au"]) * (
        E_mix - float(p["pzc_Au"]) - phi_au * thermal
    )
    sigma_pd = float(p["C_H_Pd"]) * (
        E_mix - float(p["pzc_Pd"]) - phi_pd * thermal
    )
    return {
        "solver": "isolated_planar_gouy_chapman_asinh_stern",
        "E_mix_V": E_mix,
        "phi_RP_Au_tilde": phi_au,
        "phi_RP_Pd_tilde": phi_pd,
        "phi_RP_Au_V": phi_au * thermal,
        "phi_RP_Pd_V": phi_pd * thermal,
        "sigma_Au_C_per_m2": sigma_au,
        "sigma_Pd_C_per_m2": sigma_pd,
        "I_Au_A": I_au,
        "I_Pd_A": I_pd,
        "j_Au_A_per_m2": I_au / derived["area_Au_m2"],
        "j_Pd_A_per_m2": I_pd / derived["area_Pd_m2"],
        "i_mix_abs_A": I_au,
        "i_mix_avg_A_per_m2": I_au / derived["reactive_area_m2"],
        "log_current_balance_residual": log_au - log_pd,
        "relative_balance_residual": abs(I_au + I_pd) / (abs(I_au) + abs(I_pd)),
        "root": {
            "method": "brentq_on_log_current_magnitudes",
            "iterations": int(root.iterations),
            "function_calls": int(root.function_calls),
            "bracket_V": [float(lower), float(upper)],
            "bracket_expansions": expansions,
        },
    }


def _gc_profile(surface_phi: float, distance_tilde: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return isolated GC profile and derivative with respect to distance."""

    amplitude = math.tanh(0.25 * float(surface_phi))
    q = amplitude * np.exp(-np.asarray(distance_tilde, dtype=float))
    q = np.clip(q, -1.0 + 1.0e-15, 1.0 - 1.0e-15)
    value = 4.0 * np.arctanh(q)
    derivative = -4.0 * q / (1.0 - q * q)
    return value, derivative


class NonlinearFacingElectrolyteSlitModel(FacingElectrolyteSlitModel):
    """Finite nonlinear-PB slit with a machine-precision planar asymptote."""

    electrostatic_backend = NONLINEAR_ELECTROSTATIC_BACKEND
    model_id = NONLINEAR_MODEL_ID

    def __init__(self, params: Mapping[str, Any] | None, gap_nm: float):
        super().__init__(params, gap_nm)
        if int(self.params["nonlinear_pb_max_nodes"]) <= int(
            self.params["nonlinear_pb_initial_points_small"]
        ):
            raise ValueError(
                "nonlinear_pb_max_nodes must exceed nonlinear_pb_initial_points_small"
            )
        self._cached_E_V: float | None = None
        self._cached_solution: Any | None = None
        self._planar_reference: dict[str, Any] | None = None

    @property
    def uses_planar_asymptote(self) -> bool:
        return self.D_tilde >= float(
            self.params["nonlinear_pb_planar_switch_lambda_D"]
        )

    def _isolated_surfaces(self, E_V: float) -> tuple[float, float]:
        return (
            gouy_chapman_stern_surface_phi_tilde(
                E_V,
                float(self.params["pzc_Au"]),
                self.derived["g_Au"],
                self.derived["beta_per_V"],
            ),
            gouy_chapman_stern_surface_phi_tilde(
                E_V,
                float(self.params["pzc_Pd"]),
                self.derived["g_Pd"],
                self.derived["beta_per_V"],
            ),
        )

    def _initial_mesh(self) -> np.ndarray:
        if self.D_tilde < float(
            self.params["nonlinear_pb_large_gap_edge_lambda_D"]
        ) * 2.0:
            return np.linspace(
                0.0,
                self.D_tilde,
                int(self.params["nonlinear_pb_initial_points_small"]),
            )
        edge = float(self.params["nonlinear_pb_large_gap_edge_lambda_D"])
        n_edge = int(self.params["nonlinear_pb_edge_points_large"])
        left = np.linspace(0.0, edge, n_edge)
        middle = np.linspace(edge, self.D_tilde - edge, 31)
        right = np.linspace(self.D_tilde - edge, self.D_tilde, n_edge)
        return np.unique(np.concatenate((left, middle, right)))

    def _initial_guess(self, E_V: float, xi: np.ndarray) -> np.ndarray:
        phi_au, phi_pd = self._isolated_surfaces(E_V)
        left, dleft = _gc_profile(phi_au, xi)
        right, dright_distance = _gc_profile(phi_pd, self.D_tilde - xi)
        return np.vstack((left + right, dleft - dright_distance))

    def _fixed_E_solution(self, E_V: float) -> Any:
        E_value = float(E_V)
        if self.uses_planar_asymptote:
            return None
        if (
            self._cached_solution is not None
            and self._cached_E_V is not None
            and math.isclose(E_value, self._cached_E_V, rel_tol=0.0, abs_tol=1.0e-14)
        ):
            return self._cached_solution

        xi = self._initial_mesh()
        guess = self._initial_guess(E_value, xi)
        g_au, g_pd = self.derived["g_Au"], self.derived["g_Pd"]
        beta = self.derived["beta_per_V"]
        u_au = beta * (E_value - float(self.params["pzc_Au"]))
        u_pd = beta * (E_value - float(self.params["pzc_Pd"]))

        def ode(_xi: np.ndarray, state: np.ndarray) -> np.ndarray:
            return np.vstack((state[1], np.sinh(state[0])))

        def boundary(left: np.ndarray, right: np.ndarray) -> np.ndarray:
            return np.asarray(
                [
                    -left[1] + g_au * left[0] - g_au * u_au,
                    right[1] + g_pd * right[0] - g_pd * u_pd,
                ]
            )

        solution = solve_bvp(
            ode,
            boundary,
            xi,
            guess,
            tol=float(self.params["nonlinear_pb_tol"]),
            max_nodes=int(self.params["nonlinear_pb_max_nodes"]),
            verbose=0,
        )
        if not solution.success:
            raise RuntimeError(
                "Nonlinear PB slit BVP failed at "
                f"d={self.gap_nm:g} nm, E={E_value:.9g} V: {solution.message}"
            )
        self._cached_E_V = E_value
        self._cached_solution = solution
        return solution

    def phi_tilde(self, E_V: float, x_m: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x_m, dtype=float)
        if (
            not np.all(np.isfinite(x))
            or np.any(x < -1.0e-18)
            or np.any(x > self.gap_m + 1.0e-18)
        ):
            raise ValueError("x_m must lie inside the finite electrolyte slit")
        xi = x / self.derived["lambda_D"]
        if self.uses_planar_asymptote:
            phi_au, phi_pd = self._isolated_surfaces(float(E_V))
            left, _ = _gc_profile(phi_au, xi)
            right, _ = _gc_profile(phi_pd, self.D_tilde - xi)
            return left + right
        solution = self._fixed_E_solution(float(E_V))
        return np.asarray(solution.sol(xi)[0], dtype=float)

    def _surface_state(self, E_V: float) -> tuple[float, float, float, float]:
        if self.uses_planar_asymptote:
            phi_au, phi_pd = self._isolated_surfaces(float(E_V))
            left_at_au, dleft_at_au = _gc_profile(phi_au, np.asarray([0.0]))
            right_at_au, dright_at_au = _gc_profile(
                phi_pd, np.asarray([self.D_tilde])
            )
            left_at_pd, dleft_at_pd = _gc_profile(
                phi_au, np.asarray([self.D_tilde])
            )
            right_at_pd, dright_at_pd = _gc_profile(phi_pd, np.asarray([0.0]))
            psi_au = float(left_at_au[0] + right_at_au[0])
            psi_pd = float(left_at_pd[0] + right_at_pd[0])
            derivative_au = float(dleft_at_au[0] - dright_at_au[0])
            derivative_pd = float(dleft_at_pd[0] - dright_at_pd[0])
            return psi_au, psi_pd, derivative_au, derivative_pd
        solution = self._fixed_E_solution(float(E_V))
        boundary = solution.sol(np.asarray([0.0, self.D_tilde]))
        return (
            float(boundary[0, 0]),
            float(boundary[0, 1]),
            float(boundary[1, 0]),
            float(boundary[1, 1]),
        )

    def surface_phi_tilde(self, E_V: float) -> tuple[float, float]:
        phi_au, phi_pd, _, _ = self._surface_state(E_V)
        return phi_au, phi_pd

    def robin_residuals(self, E_V: float) -> dict[str, float]:
        phi_au, phi_pd, derivative_au, derivative_pd = self._surface_state(E_V)
        beta = self.derived["beta_per_V"]
        g_au, g_pd = self.derived["g_Au"], self.derived["g_Pd"]
        return {
            "Au": float(
                -derivative_au
                + g_au * phi_au
                - g_au * beta * (float(E_V) - float(self.params["pzc_Au"]))
            ),
            "Pd": float(
                derivative_pd
                + g_pd * phi_pd
                - g_pd * beta * (float(E_V) - float(self.params["pzc_Pd"]))
            ),
        }

    def _log_current_balance(self, E_V: float) -> float:
        phi_au, phi_pd = self.surface_phi_tilde(E_V)
        log_au, log_pd = _log_current_magnitudes(
            self.params,
            self.derived,
            self.reaction,
            E_V,
            phi_au,
            phi_pd,
        )
        return log_au - log_pd

    def brent_emix(self) -> tuple[float, dict[str, Any]]:
        if self.uses_planar_asymptote:
            if self._planar_reference is None:
                self._planar_reference = solve_planar_gouy_chapman_stern_limit(
                    self.params
                )
            root = self._planar_reference["root"]
            return float(self._planar_reference["E_mix_V"]), dict(root)

        planar = solve_planar_gouy_chapman_stern_limit(self.params)
        center = float(planar["E_mix_V"])
        padding = float(self.params["root_bracket_padding_V"])
        lower, upper = center - padding, center + padding
        f_lower, f_upper = self._log_current_balance(lower), self._log_current_balance(upper)
        expansions = 0
        while f_lower * f_upper > 0.0 and expansions < int(
            self.params["max_bracket_expands"]
        ):
            padding *= 1.5
            lower, upper = center - padding, center + padding
            f_lower = self._log_current_balance(lower)
            f_upper = self._log_current_balance(upper)
            expansions += 1
        if f_lower * f_upper > 0.0:
            raise RuntimeError("Could not bracket nonlinear-PB mixed potential")
        root = root_scalar(
            self._log_current_balance,
            bracket=(lower, upper),
            method="brentq",
            xtol=float(self.params["root_xtol_V"]),
            rtol=float(self.params["root_rtol"]),
            maxiter=int(self.params["root_maxiter"]),
        )
        if not root.converged:
            raise RuntimeError("Nonlinear-PB mixed-potential root solve failed")
        return float(root.root), {
            "method": "brentq_on_log_current_magnitudes_with_nested_nonlinear_bvp",
            "iterations": int(root.iterations),
            "function_calls": int(root.function_calls),
            "bracket_V": [float(lower), float(upper)],
            "bracket_expansions": expansions,
        }

    def solve(self) -> dict[str, Any]:
        E_mix, root = self.brent_emix()
        currents = self.current_components(E_mix)
        phi_au, phi_pd, derivative_au, derivative_pd = self._surface_state(E_mix)
        thermal = self.derived["thermal_voltage_V"]
        sigma_au = float(self.params["C_H_Au"]) * (
            E_mix - float(self.params["pzc_Au"]) - phi_au * thermal
        )
        sigma_pd = float(self.params["C_H_Pd"]) * (
            E_mix - float(self.params["pzc_Pd"]) - phi_pd * thermal
        )

        if self.uses_planar_asymptote:
            bvp_diagnostics = {
                "solver_method": "analytic_independent_gouy_chapman_asinh_stern_limit",
                "planar_switch_lambda_D": float(
                    self.params["nonlinear_pb_planar_switch_lambda_D"]
                ),
                "cross_gap_tail_bound": 4.0 * self.q,
                "max_rms_residual": 0.0,
                "first_integral_span": 0.0,
                "integrated_charge_residual": 0.0,
                "midplane_phi_tilde": float(
                    self.phi_tilde(E_mix, np.asarray([0.5 * self.gap_m]))[0]
                ),
                "mesh_nodes": 0,
            }
        else:
            solution = self._fixed_E_solution(E_mix)
            diagnostic_x = np.linspace(0.0, self.D_tilde, 4001)
            state = solution.sol(diagnostic_x)
            first_integral = 0.5 * state[1] ** 2 - np.cosh(state[0])
            integrated_charge_residual = (
                state[1, -1]
                - state[1, 0]
                - simpson(np.sinh(state[0]), x=diagnostic_x)
            )
            bvp_diagnostics = {
                "solver_method": "scipy_solve_bvp_full_nonlinear_poisson_boltzmann",
                "status": int(solution.status),
                "message": str(solution.message),
                "mesh_nodes": int(solution.x.size),
                "max_rms_residual": float(np.max(solution.rms_residuals)),
                "first_integral_min": float(np.min(first_integral)),
                "first_integral_max": float(np.max(first_integral)),
                "first_integral_span": float(np.ptp(first_integral)),
                "integrated_charge_residual": float(integrated_charge_residual),
                "midplane_phi_tilde": float(
                    solution.sol(np.asarray([0.5 * self.D_tilde]))[0, 0]
                ),
            }

        return {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "model_id": NONLINEAR_MODEL_ID,
            "topology_id": TOPOLOGY_ID,
            "electrostatic_backend": NONLINEAR_ELECTROSTATIC_BACKEND,
            "electronic_constraint": "shared E_mix; I_Au + I_Pd = 0",
            "electrolyte_ensemble": "grand-canonical bulk chemical-potential reference",
            "gap_nm": self.gap_nm,
            "gap_over_lambda_D": self.D_tilde,
            "overlap_factor_exp_minus_d_over_lambda": self.q,
            "E_mix_V": E_mix,
            "phi_RP_Au_tilde": phi_au,
            "phi_RP_Pd_tilde": phi_pd,
            "phi_RP_Au_V": phi_au * thermal,
            "phi_RP_Pd_V": phi_pd * thermal,
            "surface_derivative_Au_dpsi_dxi": derivative_au,
            "surface_derivative_Pd_dpsi_dxi": derivative_pd,
            "sigma_Au_C_per_m2": sigma_au,
            "sigma_Pd_C_per_m2": sigma_pd,
            **currents,
            "root": root,
            "log_current_balance_residual": self._log_current_balance(E_mix),
            "robin_residuals": self.robin_residuals(E_mix),
            "nonlinear_pb_diagnostics": bvp_diagnostics,
            "linearization_diagnostic": {
                "max_abs_phi_tilde": max(abs(phi_au), abs(phi_pd)),
                "threshold": float(self.params["dh_warn_threshold"]),
                "linearized_pb_would_exceed_threshold": max(
                    abs(phi_au), abs(phi_pd)
                )
                > float(self.params["dh_warn_threshold"]),
            },
        }


def solve_cases_nonlinear(
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    canonical = canonical_params(params)
    derived = compute_derived(canonical)
    cases = [
        NonlinearFacingElectrolyteSlitModel(canonical, gap_nm).solve()
        for gap_nm in canonical["gap_distances_nm"]
    ]
    planar = solve_planar_gouy_chapman_stern_limit(canonical)
    large = max(cases, key=lambda case: float(case["gap_nm"]))
    comparison = {
        "gap_nm": float(large["gap_nm"]),
        "delta_E_mix_V": float(large["E_mix_V"]) - float(planar["E_mix_V"]),
        "delta_phi_RP_Au_tilde": float(large["phi_RP_Au_tilde"])
        - float(planar["phi_RP_Au_tilde"]),
        "delta_phi_RP_Pd_tilde": float(large["phi_RP_Pd_tilde"])
        - float(planar["phi_RP_Pd_tilde"]),
        "relative_i_mix_difference": (
            float(large["i_mix_avg_A_per_m2"])
            / float(planar["i_mix_avg_A_per_m2"])
            - 1.0
        ),
    }
    refined_params = dict(canonical)
    refined_params.update(
        {
            "nonlinear_pb_tol": float(canonical["nonlinear_pb_tol"]) / 5.0,
            "nonlinear_pb_initial_points_small": 2
            * int(canonical["nonlinear_pb_initial_points_small"])
            - 1,
            "nonlinear_pb_edge_points_large": 2
            * int(canonical["nonlinear_pb_edge_points_large"])
            - 1,
            "root_xtol_V": min(float(canonical["root_xtol_V"]), 2.0e-13),
            "root_rtol": min(float(canonical["root_rtol"]), 2.0e-13),
        }
    )
    refined_params = canonical_params(refined_params)
    refined_cases = [
        NonlinearFacingElectrolyteSlitModel(refined_params, gap_nm).solve()
        for gap_nm in canonical["gap_distances_nm"]
    ]
    convergence_rows = []
    for base, refined in zip(cases, refined_cases, strict=True):
        convergence_rows.append(
            {
                "gap_nm": float(base["gap_nm"]),
                "base_E_mix_V": float(base["E_mix_V"]),
                "refined_E_mix_V": float(refined["E_mix_V"]),
                "delta_E_mix_V": float(refined["E_mix_V"])
                - float(base["E_mix_V"]),
                "delta_phi_RP_Au_tilde": float(refined["phi_RP_Au_tilde"])
                - float(base["phi_RP_Au_tilde"]),
                "delta_phi_RP_Pd_tilde": float(refined["phi_RP_Pd_tilde"])
                - float(base["phi_RP_Pd_tilde"]),
                "relative_i_mix_difference": float(
                    refined["i_mix_avg_A_per_m2"]
                )
                / float(base["i_mix_avg_A_per_m2"])
                - 1.0,
                "base_mesh_nodes": int(
                    base["nonlinear_pb_diagnostics"]["mesh_nodes"]
                ),
                "refined_mesh_nodes": int(
                    refined["nonlinear_pb_diagnostics"]["mesh_nodes"]
                ),
            }
        )
    convergence = {
        "base_settings": {
            "nonlinear_pb_tol": float(canonical["nonlinear_pb_tol"]),
            "nonlinear_pb_initial_points_small": int(
                canonical["nonlinear_pb_initial_points_small"]
            ),
            "nonlinear_pb_edge_points_large": int(
                canonical["nonlinear_pb_edge_points_large"]
            ),
        },
        "refined_settings": {
            "nonlinear_pb_tol": float(refined_params["nonlinear_pb_tol"]),
            "nonlinear_pb_initial_points_small": int(
                refined_params["nonlinear_pb_initial_points_small"]
            ),
            "nonlinear_pb_edge_points_large": int(
                refined_params["nonlinear_pb_edge_points_large"]
            ),
        },
        "max_abs_delta_E_mix_V": max(
            abs(float(row["delta_E_mix_V"])) for row in convergence_rows
        ),
        "max_abs_delta_phi_RP_tilde": max(
            max(
                abs(float(row["delta_phi_RP_Au_tilde"])),
                abs(float(row["delta_phi_RP_Pd_tilde"])),
            )
            for row in convergence_rows
        ),
        "max_abs_relative_i_mix_difference": max(
            abs(float(row["relative_i_mix_difference"]))
            for row in convergence_rows
        ),
        "rows": convergence_rows,
    }
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "model_id": NONLINEAR_MODEL_ID,
        "topology_id": TOPOLOGY_ID,
        "electrostatic_backend": NONLINEAR_ELECTROSTATIC_BACKEND,
        "params": canonical,
        "derived": derived,
        "effective_reaction": effective_reaction_params(canonical),
        "large_gap_gouy_chapman_stern_reference": planar,
        "large_gap_comparison": comparison,
        "numerical_convergence": convergence,
        "cases": cases,
    }
