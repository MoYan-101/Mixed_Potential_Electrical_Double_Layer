"""Local Frumkin-corrected kinetics for reactive Au and Pd segments."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .parameters import canonical_params, compute_derived_params


def _scalar_or_array(value: np.ndarray, original: Any) -> float | np.ndarray:
    if np.asarray(original).ndim == 0:
        return float(np.asarray(value).reshape(()))
    return np.asarray(value, dtype=float)


def safe_exp(value: np.ndarray | float) -> np.ndarray:
    return np.exp(np.clip(np.asarray(value, dtype=float), -700.0, 700.0))


def effective_reaction_params(params: Mapping[str, Any]) -> dict[str, float]:
    """Apply the baseline pH corrections used by the existing Janus model."""

    p = canonical_params(params)
    delta_pH = float(p["pH"]) - float(p["pH_ref"])
    return {
        "pH": float(p["pH"]),
        "pH_ref": float(p["pH_ref"]),
        "delta_pH": float(delta_pH),
        "E1_eq_eff_V": float(p["E1_eq"])
        + float(p["E1_eq_pH_slope_V_per_pH"]) * delta_pH,
        "E2_eq_eff_V": float(p["E2_eq"])
        + float(p["E2_eq_pH_slope_V_per_pH"]) * delta_pH,
        "it0_1_eff_A_per_m2": float(p["it0_1"])
        * math.exp(-math.log(10.0) * float(p["it0_1_pH_order"]) * delta_pH),
        "it0_2_eff_A_per_m2": float(p["it0_2"])
        * math.exp(-math.log(10.0) * float(p["it0_2_pH_order"]) * delta_pH),
    }


def kinetics_context(E_V: float, params: Mapping[str, Any]) -> dict[str, float]:
    """Return scalar factors shared by local-current evaluations."""

    p = canonical_params(params)
    derived = compute_derived_params(p)
    reaction = effective_reaction_params(p)
    alpha1 = float(p["alpha1"])
    alpha2 = float(p["alpha2"])
    return {
        "beta_per_V": float(derived["beta_per_V"]),
        "thermal_voltage_V": float(derived["thermal_voltage_V"]),
        "alpha1": alpha1,
        "alpha2": alpha2,
        "Gamma1": (1.0 - alpha1) + float(p["z_R1"]),
        "Gamma2": alpha2 - float(p["z_O2"]),
        "eta1_bulk_V": float(E_V) - reaction["E1_eq_eff_V"],
        "eta2_bulk_V": float(E_V) - reaction["E2_eq_eff_V"],
        **reaction,
    }


def au_local_current_density(
    E_V: float,
    phi_tilde: np.ndarray | float,
    params: Mapping[str, Any],
) -> float | np.ndarray:
    """Positive local Au oxidation current density in A/m^2."""

    context = kinetics_context(float(E_V), params)
    phi = np.asarray(phi_tilde, dtype=float)
    exponent = (
        (1.0 - context["alpha1"])
        * context["beta_per_V"]
        * context["eta1_bulk_V"]
        - context["Gamma1"] * phi
    )
    value = context["it0_1_eff_A_per_m2"] * safe_exp(exponent)
    return _scalar_or_array(value, phi_tilde)


def pd_local_current_density(
    E_V: float,
    phi_tilde: np.ndarray | float,
    params: Mapping[str, Any],
) -> float | np.ndarray:
    """Negative local Pd reduction current density in A/m^2."""

    context = kinetics_context(float(E_V), params)
    phi = np.asarray(phi_tilde, dtype=float)
    exponent = (
        -context["alpha2"]
        * context["beta_per_V"]
        * context["eta2_bulk_V"]
        + context["Gamma2"] * phi
    )
    value = -context["it0_2_eff_A_per_m2"] * safe_exp(exponent)
    return _scalar_or_array(value, phi_tilde)


def local_current_density_A_per_m2(
    E_V: float,
    phi_tilde: np.ndarray | float,
    material: str,
    params: Mapping[str, Any],
) -> float | np.ndarray:
    """Dispatch a local Faradaic current; C is represented by ``NaN``."""

    if material == "Au":
        return au_local_current_density(E_V, phi_tilde, params)
    if material == "Pd":
        return pd_local_current_density(E_V, phi_tilde, params)
    if material == "C":
        values = np.full_like(np.asarray(phi_tilde, dtype=float), np.nan, dtype=float)
        return _scalar_or_array(values, phi_tilde)
    raise ValueError("material must be 'Au', 'Pd', or 'C'")


def local_overpotential_V(
    E_V: float,
    phi_tilde: np.ndarray | float,
    material: str,
    params: Mapping[str, Any],
) -> float | np.ndarray:
    """Return ``E-E_eq-phi_RP``; non-reactive C is represented by ``NaN``."""

    p = canonical_params(params)
    reaction = effective_reaction_params(p)
    thermal = float(compute_derived_params(p)["thermal_voltage_V"])
    phi = np.asarray(phi_tilde, dtype=float)
    if material == "Au":
        equilibrium = reaction["E1_eq_eff_V"]
    elif material == "Pd":
        equilibrium = reaction["E2_eq_eff_V"]
    elif material == "C":
        values = np.full_like(phi, np.nan, dtype=float)
        return _scalar_or_array(values, phi_tilde)
    else:
        raise ValueError("material must be 'Au', 'Pd', or 'C'")
    values = float(E_V) - equilibrium - thermal * phi
    return _scalar_or_array(values, phi_tilde)


def normalized_reactant_concentrations(
    phi_tilde: np.ndarray | float,
    params: Mapping[str, Any],
) -> dict[str, float | np.ndarray]:
    """Return Boltzmann ratios ``c_R1/c_bulk`` and ``c_O2/c_bulk``."""

    p = canonical_params(params)
    phi = np.asarray(phi_tilde, dtype=float)
    c_r1 = safe_exp(-float(p["z_R1"]) * phi)
    c_o2 = safe_exp(-float(p["z_O2"]) * phi)
    return {
        "c_R1_over_c_bulk": _scalar_or_array(c_r1, phi_tilde),
        "c_O2_over_c_bulk": _scalar_or_array(c_o2, phi_tilde),
    }

