"""Analytic finite-slit linearized-PB model for facing Au/Pd electrodes."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import root_scalar


MODEL_ID = "facing_au_electrolyte_pd_shared_mixed_potential"
TOPOLOGY_ID = "parallel_facing_Au_electrolyte_Pd_slit"
ELECTROSTATIC_BACKEND = "analytic_linear_pb_finite_slit"
RESULT_SCHEMA_VERSION = 2


def default_params() -> dict[str, Any]:
    return {
        "R": 8.314,
        "F": 96485.0,
        "T": 298.0,
        "epsilon0": 8.8541878128e-12,
        "epsilon_r": 78.5,
        "epsilon_s": None,
        "C_tot": 10.0,
        "lambda_D": None,
        "L_face": 25.0e-9,
        "out_of_plane_width": 0.01,
        "gap_distances_nm": [3.0, 10.0, 1000.0],
        "C_H_Au": 0.20,
        "C_H_Pd": 0.40,
        "g_Au": None,
        "g_Pd": None,
        "pzc_Au": 0.93,
        "pzc_Pd": 0.78,
        "it0_1": 1.852573885166257e-4,
        "it0_2": 1.852573885166257e-4,
        "alpha1": 0.5,
        "alpha2": 0.5,
        "z_R1": -1.0,
        "z_O2": 1.0,
        "E1_eq": 0.10,
        "E2_eq": 0.834,
        "pH": 7.0,
        "pH_ref": 7.0,
        "E1_eq_pH_slope_V_per_pH": 0.0,
        "E2_eq_pH_slope_V_per_pH": -0.059126500015747985,
        "it0_1_pH_order": 0.0,
        "it0_2_pH_order": 1.0,
        "root_xtol_V": 1.0e-10,
        "root_rtol": 1.0e-12,
        "root_maxiter": 200,
        "root_bracket_padding_V": 0.25,
        "max_bracket_expands": 12,
        "dh_warn_threshold": 1.0,
        "profile_points_small": 1201,
        "profile_points_large": 4001,
        "plot_face_points": 161,
        "large_gap_window_lambda_D": 5.0,
        "nonlinear_pb_tol": 1.0e-7,
        "nonlinear_pb_max_nodes": 50000,
        "nonlinear_pb_initial_points_small": 401,
        "nonlinear_pb_edge_points_large": 301,
        "nonlinear_pb_large_gap_edge_lambda_D": 15.0,
        "nonlinear_pb_planar_switch_lambda_D": 50.0,
    }


def _finite(values: Mapping[str, Any], name: str) -> float:
    try:
        value = float(values[name])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def canonical_params(values: Mapping[str, Any] | None = None) -> dict[str, Any]:
    params = default_params()
    supplied = dict(values or {})
    unknown = sorted(set(supplied).difference(params))
    if unknown:
        raise KeyError(f"Unknown parameter(s): {', '.join(unknown)}")
    params.update(supplied)
    for name in (
        "R",
        "F",
        "T",
        "epsilon0",
        "epsilon_r",
        "C_tot",
        "L_face",
        "out_of_plane_width",
        "it0_1",
        "it0_2",
        "root_xtol_V",
        "root_rtol",
        "root_bracket_padding_V",
        "dh_warn_threshold",
        "large_gap_window_lambda_D",
        "nonlinear_pb_tol",
        "nonlinear_pb_large_gap_edge_lambda_D",
        "nonlinear_pb_planar_switch_lambda_D",
    ):
        if _finite(params, name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name in ("C_H_Au", "C_H_Pd"):
        if _finite(params, name) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    for name in ("alpha1", "alpha2"):
        if not 0.0 <= _finite(params, name) <= 1.0:
            raise ValueError(f"{name} must be within [0, 1]")
    for name in (
        "pzc_Au",
        "pzc_Pd",
        "z_R1",
        "z_O2",
        "E1_eq",
        "E2_eq",
        "pH",
        "pH_ref",
        "E1_eq_pH_slope_V_per_pH",
        "E2_eq_pH_slope_V_per_pH",
        "it0_1_pH_order",
        "it0_2_pH_order",
    ):
        _finite(params, name)
    for name in ("epsilon_s", "lambda_D"):
        if params[name] is not None and _finite(params, name) <= 0.0:
            raise ValueError(f"{name} must be positive when provided")
    for name in ("g_Au", "g_Pd"):
        if params[name] is not None and _finite(params, name) < 0.0:
            raise ValueError(f"{name} must be non-negative when provided")
    for name in (
        "root_maxiter",
        "max_bracket_expands",
        "profile_points_small",
        "profile_points_large",
        "plot_face_points",
        "nonlinear_pb_max_nodes",
        "nonlinear_pb_initial_points_small",
        "nonlinear_pb_edge_points_large",
    ):
        value = _finite(params, name)
        minimum = 0 if name == "max_bracket_expands" else (3 if "points" in name else 1)
        if not value.is_integer() or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        params[name] = int(value)
    gaps = params["gap_distances_nm"]
    if not isinstance(gaps, Sequence) or isinstance(gaps, (str, bytes)):
        raise TypeError("gap_distances_nm must be a sequence")
    canonical_gaps = [float(value) for value in gaps]
    if not canonical_gaps or not all(math.isfinite(value) and 0.0 < value <= 1000.0 for value in canonical_gaps):
        raise ValueError("gap_distances_nm must contain finite values in (0, 1000]")
    if len(set(canonical_gaps)) != len(canonical_gaps):
        raise ValueError("gap_distances_nm must not contain duplicates")
    params["gap_distances_nm"] = canonical_gaps
    return params


def load_params(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Parameter file must contain a JSON object")
    return canonical_params(value)


def compute_derived(params: Mapping[str, Any]) -> dict[str, float]:
    p = canonical_params(params)
    epsilon = (
        float(p["epsilon_s"])
        if p["epsilon_s"] is not None
        else float(p["epsilon0"]) * float(p["epsilon_r"])
    )
    lambda_D = (
        float(p["lambda_D"])
        if p["lambda_D"] is not None
        else math.sqrt(
            epsilon * float(p["R"]) * float(p["T"])
            / (2.0 * float(p["F"]) ** 2 * float(p["C_tot"]))
        )
    )
    g_au = float(p["g_Au"]) if p["g_Au"] is not None else lambda_D * float(p["C_H_Au"]) / epsilon
    g_pd = float(p["g_Pd"]) if p["g_Pd"] is not None else lambda_D * float(p["C_H_Pd"]) / epsilon
    area = float(p["L_face"]) * float(p["out_of_plane_width"])
    return {
        "epsilon_s": epsilon,
        "lambda_D": lambda_D,
        "thermal_voltage_V": float(p["R"]) * float(p["T"]) / float(p["F"]),
        "beta_per_V": float(p["F"]) / (float(p["R"]) * float(p["T"])),
        "g_Au": g_au,
        "g_Pd": g_pd,
        "area_Au_m2": area,
        "area_Pd_m2": area,
        "reactive_area_m2": 2.0 * area,
    }


def effective_reaction_params(params: Mapping[str, Any]) -> dict[str, float]:
    p = canonical_params(params)
    delta_pH = float(p["pH"]) - float(p["pH_ref"])
    return {
        "E1_eq_eff": float(p["E1_eq"]) + float(p["E1_eq_pH_slope_V_per_pH"]) * delta_pH,
        "E2_eq_eff": float(p["E2_eq"]) + float(p["E2_eq_pH_slope_V_per_pH"]) * delta_pH,
        "it0_1_eff": float(p["it0_1"]) * math.exp(-math.log(10.0) * float(p["it0_1_pH_order"]) * delta_pH),
        "it0_2_eff": float(p["it0_2"]) * math.exp(-math.log(10.0) * float(p["it0_2_pH_order"]) * delta_pH),
    }


class FacingElectrolyteSlitModel:
    """Finite positive-width slit with Au and Pd on opposing boundaries."""

    def __init__(self, params: Mapping[str, Any] | None, gap_nm: float):
        self.params = canonical_params(params)
        self.derived = compute_derived(self.params)
        self.reaction = effective_reaction_params(self.params)
        self.gap_nm = float(gap_nm)
        if not math.isfinite(self.gap_nm) or self.gap_nm <= 0.0:
            raise ValueError("Facing electrolyte gap must be strictly positive")
        if self.gap_nm > 1000.0 * (1.0 + 1.0e-12):
            raise ValueError("Facing electrolyte gap must not exceed 1000 nm")
        self.gap_m = self.gap_nm * 1.0e-9
        self.D_tilde = self.gap_m / self.derived["lambda_D"]
        self.q = math.exp(-self.D_tilde)

    def coefficients(self, E_V: float) -> tuple[float, float]:
        g_au, g_pd, q = self.derived["g_Au"], self.derived["g_Pd"], self.q
        beta = self.derived["beta_per_V"]
        u_au = beta * (float(E_V) - float(self.params["pzc_Au"]))
        u_pd = beta * (float(E_V) - float(self.params["pzc_Pd"]))
        matrix = np.asarray(
            [
                [1.0 + g_au, (g_au - 1.0) * q],
                [(g_pd - 1.0) * q, 1.0 + g_pd],
            ],
            dtype=float,
        )
        rhs = np.asarray([g_au * u_au, g_pd * u_pd], dtype=float)
        a, b = np.linalg.solve(matrix, rhs)
        return float(a), float(b)

    def phi_tilde(self, E_V: float, x_m: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x_m, dtype=float)
        if not np.all(np.isfinite(x)) or np.any(x < -1e-18) or np.any(x > self.gap_m + 1e-18):
            raise ValueError("x_m must lie inside the finite electrolyte slit")
        a, b = self.coefficients(E_V)
        xi = x / self.derived["lambda_D"]
        return a * np.exp(-xi) + b * np.exp(-(self.D_tilde - xi))

    def surface_phi_tilde(self, E_V: float) -> tuple[float, float]:
        a, b = self.coefficients(E_V)
        return float(a + b * self.q), float(a * self.q + b)

    def robin_residuals(self, E_V: float) -> dict[str, float]:
        a, b = self.coefficients(E_V)
        phi_au, phi_pd = self.surface_phi_tilde(E_V)
        derivative_au = -a + b * self.q
        derivative_pd = -a * self.q + b
        beta = self.derived["beta_per_V"]
        g_au, g_pd = self.derived["g_Au"], self.derived["g_Pd"]
        residual_au = -derivative_au + g_au * phi_au - g_au * beta * (float(E_V) - float(self.params["pzc_Au"]))
        residual_pd = derivative_pd + g_pd * phi_pd - g_pd * beta * (float(E_V) - float(self.params["pzc_Pd"]))
        return {"Au": float(residual_au), "Pd": float(residual_pd)}

    def local_current_density(self, E_V: float, material: str) -> float:
        p, r, beta = self.params, self.reaction, self.derived["beta_per_V"]
        phi_au, phi_pd = self.surface_phi_tilde(E_V)
        if material == "Au":
            gamma = (1.0 - float(p["alpha1"])) + float(p["z_R1"])
            exponent = (1.0 - float(p["alpha1"])) * beta * (float(E_V) - r["E1_eq_eff"]) - gamma * phi_au
            return float(r["it0_1_eff"] * math.exp(np.clip(exponent, -700.0, 700.0)))
        if material == "Pd":
            gamma = float(p["alpha2"]) - float(p["z_O2"])
            exponent = -float(p["alpha2"]) * beta * (float(E_V) - r["E2_eq_eff"]) + gamma * phi_pd
            return float(-r["it0_2_eff"] * math.exp(np.clip(exponent, -700.0, 700.0)))
        raise ValueError("material must be 'Au' or 'Pd'")

    def current_components(self, E_V: float) -> dict[str, float]:
        j_au = self.local_current_density(E_V, "Au")
        j_pd = self.local_current_density(E_V, "Pd")
        I_au = j_au * self.derived["area_Au_m2"]
        I_pd = j_pd * self.derived["area_Pd_m2"]
        residual = I_au + I_pd
        denominator = abs(I_au) + abs(I_pd)
        return {
            "j_Au_A_per_m2": j_au,
            "j_Pd_A_per_m2": j_pd,
            "I_Au_A": I_au,
            "I_Pd_A": I_pd,
            "residual_A": residual,
            "relative_balance_residual": abs(residual) / denominator if denominator else 0.0,
            "i_mix_abs_A": abs(I_au),
            "i_mix_avg_A_per_m2": abs(I_au) / self.derived["reactive_area_m2"],
        }

    def closed_form_emix(self) -> float:
        p, r, beta = self.params, self.reaction, self.derived["beta_per_V"]
        phi0 = self.surface_phi_tilde(0.0)
        phi1_at_one = self.surface_phi_tilde(1.0)
        phi_slope = (phi1_at_one[0] - phi0[0], phi1_at_one[1] - phi0[1])
        gamma1 = (1.0 - float(p["alpha1"])) + float(p["z_R1"])
        gamma2 = float(p["alpha2"]) - float(p["z_O2"])
        slope_au = (1.0 - float(p["alpha1"])) * beta - gamma1 * phi_slope[0]
        intercept_au = math.log(self.derived["area_Au_m2"] * r["it0_1_eff"]) - (1.0 - float(p["alpha1"])) * beta * r["E1_eq_eff"] - gamma1 * phi0[0]
        slope_pd = -float(p["alpha2"]) * beta + gamma2 * phi_slope[1]
        intercept_pd = math.log(self.derived["area_Pd_m2"] * r["it0_2_eff"]) + float(p["alpha2"]) * beta * r["E2_eq_eff"] + gamma2 * phi0[1]
        denominator = slope_au - slope_pd
        if abs(denominator) < 1e-15:
            raise RuntimeError("Current-balance exponent slopes are degenerate")
        return float((intercept_pd - intercept_au) / denominator)

    def brent_emix(self) -> tuple[float, dict[str, Any]]:
        padding = float(self.params["root_bracket_padding_V"])

        def residual(E: float) -> float:
            return self.current_components(E)["residual_A"]

        lower = min(self.reaction["E1_eq_eff"], self.reaction["E2_eq_eff"]) - padding
        upper = max(self.reaction["E1_eq_eff"], self.reaction["E2_eq_eff"]) + padding
        f_lower, f_upper = residual(lower), residual(upper)
        expands = 0
        while f_lower * f_upper > 0.0 and expands < int(self.params["max_bracket_expands"]):
            padding *= 2.0
            lower = min(self.reaction["E1_eq_eff"], self.reaction["E2_eq_eff"]) - padding
            upper = max(self.reaction["E1_eq_eff"], self.reaction["E2_eq_eff"]) + padding
            f_lower, f_upper = residual(lower), residual(upper)
            expands += 1
        if f_lower * f_upper > 0.0:
            raise RuntimeError("Could not bracket I_Au + I_Pd = 0")
        root = root_scalar(
            residual,
            bracket=(lower, upper),
            method="brentq",
            xtol=float(self.params["root_xtol_V"]),
            rtol=float(self.params["root_rtol"]),
            maxiter=int(self.params["root_maxiter"]),
        )
        if not root.converged:
            raise RuntimeError(f"Mixed-potential root solve failed: {root.flag}")
        return float(root.root), {
            "method": "brentq",
            "iterations": int(root.iterations),
            "function_calls": int(root.function_calls),
            "bracket_V": [float(lower), float(upper)],
            "bracket_expansions": expands,
        }

    def solve(self) -> dict[str, Any]:
        E_closed = self.closed_form_emix()
        E_brent, root = self.brent_emix()
        if not math.isclose(E_closed, E_brent, rel_tol=0.0, abs_tol=5e-11):
            raise RuntimeError("Closed-form and Brent mixed potentials disagree")
        currents = self.current_components(E_closed)
        phi_au, phi_pd = self.surface_phi_tilde(E_closed)
        maximum = max(abs(phi_au), abs(phi_pd))
        return {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "model_id": MODEL_ID,
            "topology_id": TOPOLOGY_ID,
            "electrostatic_backend": ELECTROSTATIC_BACKEND,
            "electronic_constraint": "shared E_mix; I_Au + I_Pd = 0",
            "gap_nm": self.gap_nm,
            "gap_over_lambda_D": self.D_tilde,
            "overlap_factor_exp_minus_d_over_lambda": self.q,
            "E_mix_V": E_closed,
            "phi_RP_Au_tilde": phi_au,
            "phi_RP_Pd_tilde": phi_pd,
            "phi_RP_Au_V": phi_au * self.derived["thermal_voltage_V"],
            "phi_RP_Pd_V": phi_pd * self.derived["thermal_voltage_V"],
            **currents,
            "root": root,
            "closed_form_minus_brent_V": E_closed - E_brent,
            "robin_residuals": self.robin_residuals(E_closed),
            "debye_huckel_validity": {
                "max_abs_phi_tilde": maximum,
                "threshold": float(self.params["dh_warn_threshold"]),
                "threshold_exceeded": maximum > float(self.params["dh_warn_threshold"]),
            },
        }

    def profile_grid_m(self) -> np.ndarray:
        if self.gap_nm < 100.0:
            return np.linspace(0.0, self.gap_m, int(self.params["profile_points_small"]))
        n = int(self.params["profile_points_large"])
        edge = min(20.0 * self.derived["lambda_D"], 0.49 * self.gap_m)
        left = np.linspace(0.0, edge, max(3, n // 3))
        middle = np.linspace(edge, self.gap_m - edge, max(3, n // 3))
        right = np.linspace(self.gap_m - edge, self.gap_m, max(3, n // 3))
        return np.unique(np.concatenate((left, middle, right)))


def solve_cases(params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    canonical = canonical_params(params)
    derived = compute_derived(canonical)
    cases = []
    for gap_nm in canonical["gap_distances_nm"]:
        model = FacingElectrolyteSlitModel(canonical, gap_nm)
        cases.append(model.solve())
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "topology_id": TOPOLOGY_ID,
        "electrostatic_backend": ELECTROSTATIC_BACKEND,
        "params": canonical,
        "derived": derived,
        "effective_reaction": effective_reaction_params(canonical),
        "cases": cases,
    }
