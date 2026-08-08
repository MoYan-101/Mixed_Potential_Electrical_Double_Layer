"""Surface-charge and Gouy--Chapman-length diagnostics.

The substrate exposed between the flush electrodes is ideal and uncharged.
The quantities in this module are therefore inferred only from the two metal
Stern boundaries; they are not an additional length scale in the
linearized-PB field equation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .electrostatics import LinearPBModel


_MATERIAL_DATA = {
    "Au": ("au_top", "g_Au", "pzc_Au"),
    "Pd": ("pd_top", "g_Pd", "pzc_Pd"),
}


def _not_applicable_payload() -> dict[str, Any]:
    return {
        "applicable": False,
        "status": "not_applicable_without_edl",
        "formula": "l_GC = 2 epsilon_s R T / (F |sigma|)",
        "paper_formula": "l_GC = 2 epsilon_s k_B T / (e |sigma|)",
        "mean_definition": (
            "l_GC,mean = 2 epsilon_s R T / "
            "(F <|sigma|>_boundary)"
        ),
        "surface_charge_sign_convention": (
            "positive sigma denotes positive charge on the metal side"
        ),
        "interpretation": (
            "Not applicable to the w/o EDL reference; no Stern-boundary "
            "surface charge is inferred."
        ),
        "metals": None,
    }


def _charge_sign(values: np.ndarray, zero_tolerance: float) -> str:
    positive = values > zero_tolerance
    negative = values < -zero_tolerance
    if np.any(positive) and np.any(negative):
        return "mixed"
    if np.any(positive):
        return "positive"
    if np.any(negative):
        return "negative"
    return "zero"


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _summarize_material(
    *,
    sigma_quadrature: np.ndarray,
    weights_tilde: np.ndarray,
    sigma_samples: np.ndarray,
    effective_C_H: float,
    gc_numerator_C_per_m: float,
    lambda_D: float,
    out_of_plane_width: float,
) -> dict[str, Any]:
    sigma_q = np.asarray(sigma_quadrature, dtype=float).ravel()
    weights = np.asarray(weights_tilde, dtype=float).ravel()
    sigma_s = np.asarray(sigma_samples, dtype=float).ravel()
    if sigma_q.size == 0 or sigma_q.size != weights.size:
        raise ValueError("Boundary quadrature values and weights must be non-empty")
    if sigma_s.size == 0:
        raise ValueError("A reactive boundary must contain at least one surface sample")
    if not (
        np.all(np.isfinite(sigma_q))
        and np.all(np.isfinite(weights))
        and np.all(np.isfinite(sigma_s))
    ):
        raise ValueError("Surface-charge diagnostic inputs must be finite")
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("Reactive-boundary quadrature measure must be positive")

    mean_signed = float(np.sum(sigma_q * weights) / weight_sum)
    mean_abs = float(np.sum(np.abs(sigma_q) * weights) / weight_sum)
    maximum_scale = max(float(np.max(np.abs(sigma_s))), mean_abs)
    zero_tolerance = max(
        np.finfo(float).tiny,
        128.0 * np.finfo(float).eps * maximum_scale,
    )
    zero_at_sample = bool(np.any(np.abs(sigma_s) <= zero_tolerance))
    ordered_signs = np.sign(
        np.where(np.abs(sigma_s) <= zero_tolerance, 0.0, sigma_s)
    )
    has_sign_change = bool(
        np.any(ordered_signs > 0.0) and np.any(ordered_signs < 0.0)
    )
    # The spectral trace is continuous, so opposite-signed ordered samples (or
    # an explicit zero sample) imply an exact zero between/at those samples.
    has_zero_crossing = bool(zero_at_sample or has_sign_change)

    mean_length_infinite = bool(mean_abs <= zero_tolerance)
    mean_length = (
        None
        if mean_length_infinite
        else _finite_or_none(gc_numerator_C_per_m / mean_abs)
    )

    nonzero_abs = np.abs(sigma_s[np.abs(sigma_s) > zero_tolerance])
    if nonzero_abs.size:
        local_min_length = _finite_or_none(
            gc_numerator_C_per_m / float(np.max(nonzero_abs))
        )
        local_max_length = (
            None
            if has_zero_crossing
            else _finite_or_none(
                gc_numerator_C_per_m / float(np.min(nonzero_abs))
            )
        )
    else:
        local_min_length = None
        local_max_length = None

    return {
        "effective_C_H_F_per_m2": float(effective_C_H),
        "boundary_length_m": float(weight_sum * lambda_D),
        "boundary_area_m2": float(
            weight_sum * lambda_D * out_of_plane_width
        ),
        "n_boundary_quadrature_points": int(sigma_q.size),
        "n_boundary_samples": int(sigma_s.size),
        "mean_signed_sigma_C_per_m2": mean_signed,
        "mean_abs_sigma_C_per_m2": mean_abs,
        "local_min_sigma_C_per_m2": float(np.min(sigma_s)),
        "local_max_sigma_C_per_m2": float(np.max(sigma_s)),
        "charge_sign": _charge_sign(sigma_s, zero_tolerance),
        "has_sign_change": has_sign_change,
        "has_zero_charge": zero_at_sample,
        "has_zero_crossing": has_zero_crossing,
        "mean_gouy_chapman_length_m": mean_length,
        "mean_gouy_chapman_length_nm": (
            None if mean_length is None else float(mean_length * 1.0e9)
        ),
        "mean_length_infinite": mean_length_infinite,
        "local_min_gouy_chapman_length_m": local_min_length,
        "local_min_gouy_chapman_length_nm": (
            None
            if local_min_length is None
            else float(local_min_length * 1.0e9)
        ),
        "local_max_gouy_chapman_length_m": local_max_length,
        "local_max_gouy_chapman_length_nm": (
            None
            if local_max_length is None
            else float(local_max_length * 1.0e9)
        ),
        "local_max_length_infinite": has_zero_crossing,
        "zero_tolerance_C_per_m2": float(zero_tolerance),
    }


def compute_gouy_chapman_diagnostics(
    result: Mapping[str, Any],
    model: LinearPBModel | None,
) -> dict[str, Any]:
    """Compute charge-derived Gouy--Chapman diagnostics for Au and Pd.

    Surface charge follows the same Robin/Stern convention as the spectral model,

    ``sigma = C_H,eff * (E_mix - PZC - phi_s)``.

    The representative value uses the boundary-quadrature average of
    ``|sigma|``.  Local extrema and zero crossings use the ordered spectral
    surface samples on each metal segment.
    """

    use_edl = bool(result.get("use_edl", False))
    if not use_edl:
        return _not_applicable_payload()
    if model is None:
        raise ValueError("A LinearPBModel is required for with-EDL diagnostics")

    params = result.get("params")
    derived = result.get("derived")
    if not isinstance(params, Mapping) or not isinstance(derived, Mapping):
        raise ValueError("result must contain params and derived mappings")
    if dict(model.params) != dict(params):
        raise ValueError("The supplied LinearPBModel does not match result params")

    E_mix = float(result["E_mix_V"])
    epsilon_s = float(derived["epsilon_s"])
    lambda_D = float(derived["lambda_D"])
    R = float(derived["R"])
    T = float(derived["T"])
    F = float(derived["F"])
    thermal_voltage = float(derived["thermal_voltage_V"])
    out_of_plane_width = float(derived["out_of_plane_width"])
    gc_numerator = 2.0 * epsilon_s * R * T / F
    metals: dict[str, Any] = {}
    for material, (_boundary_name, g_name, pzc_name) in _MATERIAL_DATA.items():
        effective_C_H = float(derived[g_name]) * epsilon_s / lambda_D
        pzc = float(derived[pzc_name])
        quadrature = model.boundary_quadrature[material]
        phi_s_quadrature = thermal_voltage * np.asarray(
            model.boundary_field(material, E_mix), dtype=float
        )
        sigma_quadrature = effective_C_H * (
            E_mix - pzc - phi_s_quadrature
        )

        phi_s_samples = thermal_voltage * np.asarray(
            model.boundary_sample_field(material, E_mix), dtype=float
        )
        sigma_samples = effective_C_H * (
            E_mix - pzc - phi_s_samples
        )

        metals[material] = _summarize_material(
            sigma_quadrature=sigma_quadrature,
            weights_tilde=quadrature.weights_tilde,
            sigma_samples=sigma_samples,
            effective_C_H=effective_C_H,
            gc_numerator_C_per_m=gc_numerator,
            lambda_D=lambda_D,
            out_of_plane_width=out_of_plane_width,
        )

    return {
        "applicable": True,
        "status": "computed_from_stern_surface_charge",
        "formula": "l_GC = 2 epsilon_s R T / (F |sigma|)",
        "paper_formula": "l_GC = 2 epsilon_s k_B T / (e |sigma|)",
        "mean_definition": (
            "l_GC,mean = 2 epsilon_s R T / "
            "(F <|sigma|>_boundary)"
        ),
        "surface_charge_definition": (
            "sigma_M = C_H,M,eff (E_mix - PZC_M - phi_s)"
        ),
        "surface_charge_sign_convention": (
            "positive sigma denotes positive charge on the metal side"
        ),
        "interpretation": (
            "Charge-derived diagnostic on the flush Au/Pd Stern boundaries "
            "evaluated from the cosine-spectral surface field; "
            "not an independent screening length in the linearized-PB "
            "equation and not a substrate-gap charge length."
        ),
        "gouy_chapman_numerator_C_per_m": float(gc_numerator),
        "metals": metals,
    }
