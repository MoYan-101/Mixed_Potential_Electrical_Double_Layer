"""Analytic independent-planar-EDL electrostatics and mixed-potential solver."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.optimize import root_scalar


MODEL_ID = "independent_planar_edls_shared_mixed_potential"
TOPOLOGY_ID = "two_independent_planar_half_spaces"
ELECTROSTATIC_BACKEND = "analytic_independent_planar_linear_pb"
RESULT_SCHEMA_VERSION = 1


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
        "L_Au": 25.0e-9,
        "L_Pd": 25.0e-9,
        "out_of_plane_width": 0.01,
        "active_faces_Au": 1,
        "active_faces_Pd": 1,
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
        "L_Au",
        "L_Pd",
        "out_of_plane_width",
        "it0_1",
        "it0_2",
        "root_xtol_V",
        "root_rtol",
        "root_bracket_padding_V",
        "dh_warn_threshold",
    ):
        if _finite(params, name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name in ("C_H_Au", "C_H_Pd"):
        if _finite(params, name) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    for name in ("active_faces_Au", "active_faces_Pd"):
        value = _finite(params, name)
        if value not in (1.0, 2.0):
            raise ValueError(f"{name} must be 1 or 2")
        params[name] = int(value)
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
    for name in ("root_maxiter", "max_bracket_expands"):
        value = _finite(params, name)
        minimum = 1 if name == "root_maxiter" else 0
        if not value.is_integer() or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        params[name] = int(value)
    return params


def load_params(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
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
            epsilon
            * float(p["R"])
            * float(p["T"])
            / (2.0 * float(p["F"]) ** 2 * float(p["C_tot"]))
        )
    )
    g_au = (
        float(p["g_Au"])
        if p["g_Au"] is not None
        else lambda_D * float(p["C_H_Au"]) / epsilon
    )
    g_pd = (
        float(p["g_Pd"])
        if p["g_Pd"] is not None
        else lambda_D * float(p["C_H_Pd"]) / epsilon
    )
    area_au = (
        int(p["active_faces_Au"])
        * float(p["L_Au"])
        * float(p["out_of_plane_width"])
    )
    area_pd = (
        int(p["active_faces_Pd"])
        * float(p["L_Pd"])
        * float(p["out_of_plane_width"])
    )
    return {
        "epsilon_s": epsilon,
        "lambda_D": lambda_D,
        "thermal_voltage_V": float(p["R"]) * float(p["T"]) / float(p["F"]),
        "beta_per_V": float(p["F"]) / (float(p["R"]) * float(p["T"])),
        "g_Au": g_au,
        "g_Pd": g_pd,
        "q_Au": g_au / (1.0 + g_au),
        "q_Pd": g_pd / (1.0 + g_pd),
        "area_Au_m2": area_au,
        "area_Pd_m2": area_pd,
        "reactive_area_m2": area_au + area_pd,
    }


def effective_reaction_params(params: Mapping[str, Any]) -> dict[str, float]:
    p = canonical_params(params)
    delta_pH = float(p["pH"]) - float(p["pH_ref"])
    return {
        "E1_eq_eff": float(p["E1_eq"])
        + float(p["E1_eq_pH_slope_V_per_pH"]) * delta_pH,
        "E2_eq_eff": float(p["E2_eq"])
        + float(p["E2_eq_pH_slope_V_per_pH"]) * delta_pH,
        "it0_1_eff": float(p["it0_1"])
        * math.exp(-math.log(10.0) * float(p["it0_1_pH_order"]) * delta_pH),
        "it0_2_eff": float(p["it0_2"])
        * math.exp(-math.log(10.0) * float(p["it0_2_pH_order"]) * delta_pH),
    }


class IndependentPlanarEDLModel:
    """Two analytic half-space EDLs coupled only by mixed-potential balance."""

    def __init__(self, params: Mapping[str, Any] | None = None):
        self.params = canonical_params(params)
        self.derived = compute_derived(self.params)
        self.reaction = effective_reaction_params(self.params)

    def surface_phi_tilde(self, E_V: float, material: str, *, use_edl: bool = True) -> float:
        if not use_edl:
            return 0.0
        if material == "Au":
            q = self.derived["q_Au"]
            pzc = float(self.params["pzc_Au"])
        elif material == "Pd":
            q = self.derived["q_Pd"]
            pzc = float(self.params["pzc_Pd"])
        else:
            raise ValueError("material must be 'Au' or 'Pd'")
        return float(q * self.derived["beta_per_V"] * (float(E_V) - pzc))

    def phi_tilde_profile(
        self,
        E_V: float,
        material: str,
        y_m: np.ndarray | float,
    ) -> np.ndarray:
        y = np.asarray(y_m, dtype=float)
        if np.any(y < 0.0) or not np.all(np.isfinite(y)):
            raise ValueError("y_m must be finite and non-negative")
        return self.surface_phi_tilde(E_V, material) * np.exp(
            -y / self.derived["lambda_D"]
        )

    def local_current_density(
        self,
        E_V: float,
        material: str,
        *,
        use_edl: bool = True,
    ) -> float:
        p = self.params
        r = self.reaction
        beta = self.derived["beta_per_V"]
        phi = self.surface_phi_tilde(E_V, material, use_edl=use_edl)
        if material == "Au":
            gamma = (1.0 - float(p["alpha1"])) + float(p["z_R1"])
            exponent = (
                (1.0 - float(p["alpha1"]))
                * beta
                * (float(E_V) - r["E1_eq_eff"])
                - gamma * phi
            )
            return float(r["it0_1_eff"] * math.exp(np.clip(exponent, -700.0, 700.0)))
        if material == "Pd":
            gamma = float(p["alpha2"]) - float(p["z_O2"])
            exponent = (
                -float(p["alpha2"])
                * beta
                * (float(E_V) - r["E2_eq_eff"])
                + gamma * phi
            )
            return float(-r["it0_2_eff"] * math.exp(np.clip(exponent, -700.0, 700.0)))
        raise ValueError("material must be 'Au' or 'Pd'")

    def current_components(self, E_V: float, *, use_edl: bool = True) -> dict[str, float]:
        j_au = self.local_current_density(E_V, "Au", use_edl=use_edl)
        j_pd = self.local_current_density(E_V, "Pd", use_edl=use_edl)
        I_au = j_au * self.derived["area_Au_m2"]
        I_pd = j_pd * self.derived["area_Pd_m2"]
        residual = I_au + I_pd
        denom = abs(I_au) + abs(I_pd)
        return {
            "j_Au_A_per_m2": j_au,
            "j_Pd_A_per_m2": j_pd,
            "I_Au_A": I_au,
            "I_Pd_A": I_pd,
            "residual_A": residual,
            "relative_balance_residual": abs(residual) / denom if denom else 0.0,
            "i_mix_abs_A": abs(I_au),
            "i_mix_avg_A_per_m2": abs(I_au) / self.derived["reactive_area_m2"],
        }

    def closed_form_emix(self, *, use_edl: bool = True) -> float:
        p = self.params
        r = self.reaction
        beta = self.derived["beta_per_V"]
        q_au = self.derived["q_Au"] if use_edl else 0.0
        q_pd = self.derived["q_Pd"] if use_edl else 0.0
        gamma1 = (1.0 - float(p["alpha1"])) + float(p["z_R1"])
        gamma2 = float(p["alpha2"]) - float(p["z_O2"])
        slope_au = beta * ((1.0 - float(p["alpha1"])) - gamma1 * q_au)
        intercept_au = (
            math.log(self.derived["area_Au_m2"] * r["it0_1_eff"])
            + beta
            * (
                -(1.0 - float(p["alpha1"])) * r["E1_eq_eff"]
                + gamma1 * q_au * float(p["pzc_Au"])
            )
        )
        slope_pd = beta * (-float(p["alpha2"]) + gamma2 * q_pd)
        intercept_pd = (
            math.log(self.derived["area_Pd_m2"] * r["it0_2_eff"])
            + beta
            * (
                float(p["alpha2"]) * r["E2_eq_eff"]
                - gamma2 * q_pd * float(p["pzc_Pd"])
            )
        )
        denominator = slope_au - slope_pd
        if abs(denominator) < 1.0e-15:
            raise RuntimeError("Current-balance exponent slopes are degenerate")
        return float((intercept_pd - intercept_au) / denominator)

    def brent_emix(self, *, use_edl: bool = True) -> tuple[float, dict[str, Any]]:
        r = self.reaction
        padding = float(self.params["root_bracket_padding_V"])

        def residual(E: float) -> float:
            return self.current_components(E, use_edl=use_edl)["residual_A"]

        lower = min(r["E1_eq_eff"], r["E2_eq_eff"]) - padding
        upper = max(r["E1_eq_eff"], r["E2_eq_eff"]) + padding
        f_lower, f_upper = residual(lower), residual(upper)
        expands = 0
        while f_lower * f_upper > 0.0 and expands < int(self.params["max_bracket_expands"]):
            padding *= 2.0
            lower = min(r["E1_eq_eff"], r["E2_eq_eff"]) - padding
            upper = max(r["E1_eq_eff"], r["E2_eq_eff"]) + padding
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

    def solve(self, *, use_edl: bool = True) -> dict[str, Any]:
        E_closed = self.closed_form_emix(use_edl=use_edl)
        E_brent, root = self.brent_emix(use_edl=use_edl)
        if not math.isclose(E_closed, E_brent, rel_tol=0.0, abs_tol=5.0e-11):
            raise RuntimeError("Closed-form and Brent mixed potentials disagree")
        currents = self.current_components(E_closed, use_edl=use_edl)
        phi_au = self.surface_phi_tilde(E_closed, "Au", use_edl=use_edl)
        phi_pd = self.surface_phi_tilde(E_closed, "Pd", use_edl=use_edl)
        maximum = max(abs(phi_au), abs(phi_pd))
        return {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "model_id": MODEL_ID,
            "topology_id": TOPOLOGY_ID,
            "electrostatic_backend": ELECTROSTATIC_BACKEND if use_edl else "disabled",
            "condition": "with EDL" if use_edl else "w/o EDL",
            "electronic_constraint": "shared E_mix; I_Au + I_Pd = 0",
            "E_mix_V": E_closed,
            "phi_RP_Au_tilde": phi_au,
            "phi_RP_Pd_tilde": phi_pd,
            "phi_RP_Au_V": phi_au * self.derived["thermal_voltage_V"],
            "phi_RP_Pd_V": phi_pd * self.derived["thermal_voltage_V"],
            **currents,
            "root": root,
            "closed_form_minus_brent_V": E_closed - E_brent,
            "debye_huckel_validity": {
                "max_abs_phi_tilde": maximum,
                "threshold": float(self.params["dh_warn_threshold"]),
                "threshold_exceeded": maximum > float(self.params["dh_warn_threshold"]),
            },
        }


def solve_comparison(params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    model = IndependentPlanarEDLModel(params)
    with_edl = model.solve(use_edl=True)
    without_edl = model.solve(use_edl=False)
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "topology_id": TOPOLOGY_ID,
        "electrostatic_backend": ELECTROSTATIC_BACKEND,
        "params": model.params,
        "derived": model.derived,
        "effective_reaction": model.reaction,
        "with_edl": with_edl,
        "without_edl": without_edl,
        "delta_E_with_minus_without_V": with_edl["E_mix_V"] - without_edl["E_mix_V"],
        "delta_i_with_minus_without_A_per_m2": (
            with_edl["i_mix_avg_A_per_m2"]
            - without_edl["i_mix_avg_A_per_m2"]
        ),
    }
