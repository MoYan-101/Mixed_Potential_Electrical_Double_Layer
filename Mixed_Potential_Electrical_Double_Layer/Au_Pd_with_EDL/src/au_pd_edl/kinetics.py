"""Frumkin-corrected irreversible Butler-Volmer kinetics.

The sign and electrostatic conventions intentionally match
``Solve_Emix_updating.py``:

* reaction 1 on Au is anodic and positive;
* reaction 2 on Pd is cathodic and negative;
* ``phi_tilde = F * phi_s / (R*T)`` is dimensionless.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .parameters import compute_derived_params, validate_params


def _safe_exp(value: ArrayLike, clip: float = 700.0) -> NDArray[np.float64]:
    values = np.asarray(value, dtype=float)
    return np.exp(np.clip(values, -clip, clip))


def _scalar_or_array(reference: ArrayLike, value: NDArray[np.float64]) -> float | NDArray[np.float64]:
    if np.asarray(reference).ndim == 0:
        return float(value)
    return value


def effective_reaction_params(params: Mapping[str, Any]) -> dict[str, float]:
    """Return pH-adjusted equilibrium potentials and exchange currents."""

    pH = float(params["pH"])
    pH_ref = float(params["pH_ref"])
    delta_pH = pH - pH_ref
    E1_slope = float(params["E1_eq_pH_slope_V_per_pH"])
    E2_slope = float(params["E2_eq_pH_slope_V_per_pH"])
    order1 = float(params["it0_1_pH_order"])
    order2 = float(params["it0_2_pH_order"])

    # Match the legacy solver's safe-exp policy instead of allowing an
    # otherwise valid finite pH input to raise OverflowError.
    exponent1 = float(np.clip(-math.log(10.0) * order1 * delta_pH, -700.0, 700.0))
    exponent2 = float(np.clip(-math.log(10.0) * order2 * delta_pH, -700.0, 700.0))
    it0_1_eff = float(params["it0_1"]) * math.exp(exponent1)
    it0_2_eff = float(params["it0_2"]) * math.exp(exponent2)
    if not math.isfinite(it0_1_eff) or not math.isfinite(it0_2_eff):
        raise ValueError("pH-adjusted exchange current overflowed or is non-finite")
    if it0_1_eff <= 0.0 or it0_2_eff <= 0.0:
        raise ValueError("pH-adjusted exchange currents must remain positive")

    return {
        "pH": pH,
        "pH_ref": pH_ref,
        "delta_pH": delta_pH,
        "E1_eq_eff": float(params["E1_eq"]) + E1_slope * delta_pH,
        "E2_eq_eff": float(params["E2_eq"]) + E2_slope * delta_pH,
        "it0_1_eff": it0_1_eff,
        "it0_2_eff": it0_2_eff,
        "E1_eq_pH_slope_V_per_pH": E1_slope,
        "E2_eq_pH_slope_V_per_pH": E2_slope,
        "it0_1_pH_order": order1,
        "it0_2_pH_order": order2,
    }


def kinetics_context(E: float, params: Mapping[str, Any]) -> dict[str, float]:
    """Return the shared scalar factors for local current evaluation."""

    if not math.isfinite(float(E)):
        raise ValueError("E must be finite")
    rxn = effective_reaction_params(params)
    beta = float(params["F"]) / (float(params["R"]) * float(params["T"]))
    alpha1 = float(params["alpha1"])
    alpha2 = float(params["alpha2"])
    z_R1 = float(params["z_R1"])
    z_O2 = float(params["z_O2"])
    return {
        "beta": beta,
        "it0_1": rxn["it0_1_eff"],
        "it0_2": rxn["it0_2_eff"],
        "alpha1": alpha1,
        "alpha2": alpha2,
        "eta1": float(E) - rxn["E1_eq_eff"],
        "eta2": float(E) - rxn["E2_eq_eff"],
        "E1_eq_eff": rxn["E1_eq_eff"],
        "E2_eq_eff": rxn["E2_eq_eff"],
        "pH": rxn["pH"],
        "pH_ref": rxn["pH_ref"],
        "Gamma1": (1.0 - alpha1) + z_R1,
        "Gamma2": alpha2 - z_O2,
    }


def au_local_current_density(
    E: float,
    phi_tilde: ArrayLike,
    params: Mapping[str, Any],
) -> float | NDArray[np.float64]:
    """Positive Au oxidation current density in A/m2.

    ``phi_tilde`` is the dimensionless solution potential at the reaction
    plane.  The Frumkin/Boltzmann exponent is ``-Gamma1 * phi_tilde``.
    """

    ctx = kinetics_context(E, params)
    exponent = (
        (1.0 - ctx["alpha1"]) * ctx["beta"] * ctx["eta1"]
        - ctx["Gamma1"] * np.asarray(phi_tilde, dtype=float)
    )
    current = ctx["it0_1"] * _safe_exp(exponent)
    return _scalar_or_array(phi_tilde, current)


def pd_local_current_density(
    E: float,
    phi_tilde: ArrayLike,
    params: Mapping[str, Any],
) -> float | NDArray[np.float64]:
    """Negative Pd reduction current density in A/m2.

    ``phi_tilde`` is the dimensionless solution potential at the reaction
    plane.  The Frumkin/Boltzmann exponent is ``+Gamma2 * phi_tilde``.
    """

    ctx = kinetics_context(E, params)
    exponent = (
        -ctx["alpha2"] * ctx["beta"] * ctx["eta2"]
        + ctx["Gamma2"] * np.asarray(phi_tilde, dtype=float)
    )
    current = -ctx["it0_2"] * _safe_exp(exponent)
    return _scalar_or_array(phi_tilde, current)


def local_current_densities(
    E: float,
    phi_tilde_Au: ArrayLike,
    phi_tilde_Pd: ArrayLike,
    params: Mapping[str, Any],
) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64]]:
    """Return signed local Au and Pd current densities."""

    return (
        au_local_current_density(E, phi_tilde_Au, params),
        pd_local_current_density(E, phi_tilde_Pd, params),
    )


def emix_closed_form_no_edl(
    params: Mapping[str, Any],
    derived: Mapping[str, Any] | None = None,
) -> float:
    """Exact mixed potential for ``phi_tilde=0`` on both metals.

    The balance is based on absolute currents, hence the logarithm contains
    the Au/Pd reactive lengths.  The exposed substrate width does not enter
    the expression and therefore cannot affect the w/o-EDL reference.
    """

    validate_params(params)
    d = compute_derived_params(params) if derived is None else derived
    rxn = effective_reaction_params(params)
    alpha1 = float(params["alpha1"])
    alpha2 = float(params["alpha2"])
    kappa = 1.0 - alpha1 + alpha2
    if kappa <= 0.0:
        raise ValueError("1 - alpha1 + alpha2 must be positive")

    E_mix = (
        (1.0 - alpha1) * rxn["E1_eq_eff"]
        + alpha2 * rxn["E2_eq_eff"]
    ) / kappa
    E_mix += (
        float(params["R"])
        * float(params["T"])
        / (float(params["F"]) * kappa)
        * math.log(
            (float(d["L_Pd"]) * rxn["it0_2_eff"])
            / (float(d["L_Au"]) * rxn["it0_1_eff"])
        )
    )
    return float(E_mix)


def no_edl_absolute_currents(
    E: float,
    params: Mapping[str, Any],
    derived: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Return uniform-surface w/o-EDL absolute currents and their balance."""

    validate_params(params)
    d = compute_derived_params(params) if derived is None else derived
    j_Au = float(au_local_current_density(E, 0.0, params))
    j_Pd = float(pd_local_current_density(E, 0.0, params))
    width = float(d["out_of_plane_width"])
    I_Au = j_Au * float(d["L_Au"]) * width
    I_Pd = j_Pd * float(d["L_Pd"]) * width
    i_mix_avg = abs(I_Au) / float(d["reactive_area_m2"])
    denominator = abs(I_Au) + abs(I_Pd)
    return {
        "j_Au_A_per_m2": j_Au,
        "j_Pd_A_per_m2": j_Pd,
        "I_Au_A": I_Au,
        "I_Pd_A": I_Pd,
        "residual_A": I_Au + I_Pd,
        "relative_balance_residual": abs(I_Au + I_Pd) / denominator if denominator else 0.0,
        "i_mix_abs_A": abs(I_Au),
        "i_mix_avg_A_per_m2": i_mix_avg,
    }
