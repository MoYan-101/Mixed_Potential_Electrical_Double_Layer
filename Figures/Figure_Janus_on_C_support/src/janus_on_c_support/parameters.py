"""Parameters and derived quantities for the C-supported Janus half-cell.

All public parameter dictionaries contain only JSON-native values.  Lengths
are in metres, potentials in volts versus RHE, concentrations in mol/m^3,
capacitances in F/m^2, and current densities in A/m^2.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping


PACKAGE_VERSION = "0.1.0"
RESULT_SCHEMA_VERSION = 1
MODEL_NAME = "C_supported_Janus_linear_PB"
GEOMETRY_NAME = "C_left_5nm_Au_4nm_Pd_4nm_C_right_5nm_mirror_halfcell"
TOPOLOGY_ID = "flush_coplanar_C_Au_Pd_C_mirror_neumann"
ELECTROSTATIC_BACKEND = "cosine_galerkin_linear_pb_piecewise_stern_robin"


def default_params() -> dict[str, Any]:
    """Return the canonical ``C(5)|Au(4)|Pd(4)|C(5)`` parameter set."""

    return {
        # Physical constants.
        "R": 8.314,
        "F": 96485.0,
        "T": 298.0,
        "epsilon0": 8.8541878128e-12,
        "epsilon_r": 78.5,
        "epsilon_s": None,
        # Symmetric 1:1 electrolyte.  10 mol/m^3 is 10 mM.
        "C_tot": 10.0,
        "lambda_D": None,
        # Mirror-half-cell geometry.
        "L_C_left": 5.0e-9,
        "L_Au": 4.0e-9,
        "L_Pd": 4.0e-9,
        "L_C_right": 5.0e-9,
        "out_of_plane_width": 0.01,
        # Stern/Helmholtz capacitances and optional dimensionless overrides.
        "C_H_Au": 0.50,
        "C_H_C": 0.20,
        "C_H_Pd": 0.50,
        "g_Au": None,
        "g_C": None,
        "g_Pd": None,
        # Potentials of zero charge (V vs. RHE).
        "pzc_Au": 0.93,
        "pzc_C": 0.50,
        "pzc_Pd": 0.78,
        # Frumkin-corrected irreversible Butler--Volmer kinetics.
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
        "E2_eq_pH_slope_V_per_pH": -(
            math.log(10.0) * 8.314 * 298.0 / 96485.0
        ),
        "it0_1_pH_order": 0.0,
        "it0_2_pH_order": 1.0,
        # Spectral, quadrature, profile, and root-solve settings.
        "N_modes": 960,
        "Nx": 5000,
        "gl_order": 128,
        "Ny_2d": 320,
        "y_max_lambda_D": 5.0,
        "root_xtol_V": 1.0e-13,
        "root_rtol": 1.0e-14,
        "root_maxiter": 200,
        "root_bracket_padding_V": 0.50,
        "max_bracket_expands": 12,
        "dh_warn_threshold": 1.0,
        "evaluation_block_size": 384,
        "mode_block_size": 256,
    }


_LAMBDA_D_DEPENDENCIES = {
    "R",
    "F",
    "T",
    "epsilon0",
    "epsilon_r",
    "epsilon_s",
    "C_tot",
}


def _synchronize_capacitances_with_g(params: dict[str, Any]) -> None:
    """Make an explicit ``g_*`` override the corresponding saved ``C_H_*``.

    The Robin boundary is parameterized by ``g=lambda_D*C_H/epsilon_s``.
    Keeping an unrelated input ``C_H`` after a direct ``g`` override would
    make Stern surface charge inconsistent with the electrostatic solution.
    Therefore an explicit dimensionless charging parameter is the source of
    truth and its physically equivalent capacitance is stored in ``params``.
    Invalid values are left for :func:`validate_params` to report.
    """

    if not any(params.get(f"g_{material}") is not None for material in ("Au", "C", "Pd")):
        return
    try:
        epsilon_s = (
            float(params["epsilon_s"])
            if params.get("epsilon_s") is not None
            else float(params["epsilon0"]) * float(params["epsilon_r"])
        )
        lambda_D = (
            float(params["lambda_D"])
            if params.get("lambda_D") is not None
            else math.sqrt(
                epsilon_s
                * float(params["R"])
                * float(params["T"])
                / (2.0 * float(params["F"]) ** 2 * float(params["C_tot"]))
            )
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return
    if not math.isfinite(epsilon_s) or not math.isfinite(lambda_D):
        return
    if epsilon_s <= 0.0 or lambda_D <= 0.0:
        return
    for material in ("Au", "C", "Pd"):
        override = params.get(f"g_{material}")
        if override is None:
            continue
        try:
            g_value = float(override)
        except (TypeError, ValueError):
            continue
        if math.isfinite(g_value) and g_value >= 0.0:
            params[f"C_H_{material}"] = g_value * epsilon_s / lambda_D


def apply_param_overrides(
    base_params: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
    *,
    reset_lambda_D: bool = True,
) -> dict[str, Any]:
    """Return a copied parameter dictionary with validated-key overrides.

    Changing an electrolyte input invalidates an auto-computed Debye length.
    Accordingly, ``lambda_D`` is reset to ``None`` unless the caller supplies
    an explicit ``lambda_D`` in the same override mapping.
    """

    base = copy.deepcopy(dict(base_params))
    required = set(default_params())
    missing = sorted(required.difference(base))
    unknown_base = sorted(set(base).difference(required))
    if missing:
        raise KeyError(f"Missing base parameter(s): {', '.join(missing)}")
    if unknown_base:
        raise KeyError(f"Unknown base parameter(s): {', '.join(unknown_base)}")
    if overrides is None:
        _synchronize_capacitances_with_g(base)
        return base

    supplied = dict(overrides)
    unknown = sorted(set(supplied).difference(required))
    if unknown:
        raise KeyError(f"Unknown parameter override(s): {', '.join(unknown)}")
    base.update(copy.deepcopy(supplied))
    if (
        reset_lambda_D
        and "lambda_D" not in supplied
        and _LAMBDA_D_DEPENDENCIES.intersection(supplied)
    ):
        base["lambda_D"] = None
    _synchronize_capacitances_with_g(base)
    return base


def canonical_params(values: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Merge a partial mapping into defaults and validate the result."""

    params = apply_param_overrides(default_params(), values or {})
    validate_params(params)
    return params


def _finite(params: Mapping[str, Any], name: str) -> float:
    try:
        value = float(params[name])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _integer(params: Mapping[str, Any], name: str, minimum: int) -> int:
    value = _finite(params, name)
    if not value.is_integer() or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def validate_params(params: Mapping[str, Any]) -> None:
    """Validate completeness, units-independent ranges, and geometry."""

    required = set(default_params())
    supplied = set(params)
    missing = sorted(required.difference(supplied))
    unknown = sorted(supplied.difference(required))
    if missing:
        raise KeyError(f"Missing required parameter(s): {', '.join(missing)}")
    if unknown:
        raise KeyError(f"Unknown parameter(s): {', '.join(unknown)}")

    for name in ("R", "F", "T", "epsilon0", "epsilon_r"):
        if _finite(params, name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    if params["epsilon_s"] is not None and _finite(params, "epsilon_s") <= 0.0:
        raise ValueError("epsilon_s must be positive when provided")
    if _finite(params, "C_tot") <= 0.0:
        raise ValueError("C_tot must be positive")
    if params["lambda_D"] is not None and _finite(params, "lambda_D") <= 0.0:
        raise ValueError("lambda_D must be positive when provided")

    for name in ("L_C_left", "L_C_right"):
        if _finite(params, name) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    for name in ("L_Au", "L_Pd", "out_of_plane_width"):
        if _finite(params, name) <= 0.0:
            raise ValueError(f"{name} must be positive")

    for name in ("C_H_Au", "C_H_C", "C_H_Pd"):
        if _finite(params, name) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    for name in ("g_Au", "g_C", "g_Pd"):
        if params[name] is not None and _finite(params, name) < 0.0:
            raise ValueError(f"{name} must be non-negative when provided")

    for name in ("it0_1", "it0_2"):
        if _finite(params, name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name in ("alpha1", "alpha2"):
        if not 0.0 <= _finite(params, name) <= 1.0:
            raise ValueError(f"{name} must be within [0, 1]")
    for name in (
        "pzc_Au",
        "pzc_C",
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

    _integer(params, "N_modes", 1)
    nx = _integer(params, "Nx", 2)
    edge_values = [0.0]
    for name in ("L_C_left", "L_Au", "L_Pd", "L_C_right"):
        edge_values.append(edge_values[-1] + float(params[name]))
    unique_edge_count = len(set(edge_values))
    if nx < unique_edge_count:
        raise ValueError(
            f"Nx must be >= {unique_edge_count} to retain every distinct segment edge"
        )
    _integer(params, "gl_order", 2)
    _integer(params, "Ny_2d", 2)
    _integer(params, "root_maxiter", 1)
    _integer(params, "max_bracket_expands", 0)
    _integer(params, "evaluation_block_size", 1)
    _integer(params, "mode_block_size", 1)
    for name in (
        "y_max_lambda_D",
        "root_xtol_V",
        "root_rtol",
        "root_bracket_padding_V",
        "dh_warn_threshold",
    ):
        if _finite(params, name) <= 0.0:
            raise ValueError(f"{name} must be positive")


def compute_derived_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return JSON-safe physical and geometrical derived quantities."""

    p = canonical_params(params)
    epsilon_s = (
        float(p["epsilon_s"])
        if p["epsilon_s"] is not None
        else float(p["epsilon0"]) * float(p["epsilon_r"])
    )
    lambda_D = (
        float(p["lambda_D"])
        if p["lambda_D"] is not None
        else math.sqrt(
            epsilon_s
            * float(p["R"])
            * float(p["T"])
            / (2.0 * float(p["F"]) ** 2 * float(p["C_tot"]))
        )
    )
    thermal_voltage = float(p["R"]) * float(p["T"]) / float(p["F"])
    beta = 1.0 / thermal_voltage

    lengths = [
        float(p["L_C_left"]),
        float(p["L_Au"]),
        float(p["L_Pd"]),
        float(p["L_C_right"]),
    ]
    edges = [0.0]
    for length in lengths:
        edges.append(edges[-1] + length)
    total = edges[-1]
    reactive_length = float(p["L_Au"]) + float(p["L_Pd"])
    width = float(p["out_of_plane_width"])

    def charging_parameter(material: str) -> float:
        override = p[f"g_{material}"]
        if override is not None:
            return float(override)
        return lambda_D * float(p[f"C_H_{material}"]) / epsilon_s

    g_au = charging_parameter("Au")
    g_c = charging_parameter("C")
    g_pd = charging_parameter("Pd")
    names = ("C_left", "Au", "Pd", "C_right")
    materials = ("C", "Au", "Pd", "C")
    segment_descriptors = [
        {
            "name": name,
            "material": material,
            "x_start_m": float(edges[index]),
            "x_end_m": float(edges[index + 1]),
            "length_m": float(lengths[index]),
            "faradaic": material in {"Au", "Pd"},
        }
        for index, (name, material) in enumerate(zip(names, materials, strict=True))
    ]

    return {
        "epsilon_s_F_per_m": float(epsilon_s),
        "epsilon_s": float(epsilon_s),
        "lambda_D_m": float(lambda_D),
        "lambda_D": float(lambda_D),
        "lambda_D_nm": float(lambda_D * 1.0e9),
        "thermal_voltage_V": float(thermal_voltage),
        "beta_per_V": float(beta),
        "L_total_m": float(total),
        "L_total_nm": float(total * 1.0e9),
        "L_halfcell_m": float(total),
        "L_full_period_m": float(2.0 * total),
        "L_full_period_nm": float(2.0e9 * total),
        "L_tilde": float(total / lambda_D),
        "edges_m": [float(value) for value in edges],
        "edges_nm": [float(value * 1.0e9) for value in edges],
        "reactive_length_halfcell_m": float(reactive_length),
        "reactive_length_full_period_m": float(2.0 * reactive_length),
        "reactive_area_halfcell_m2": float(reactive_length * width),
        "reactive_area_full_period_m2": float(2.0 * reactive_length * width),
        "g_Au": float(g_au),
        "g_C": float(g_c),
        "g_Pd": float(g_pd),
        "C_H_Au_effective_F_per_m2": float(g_au * epsilon_s / lambda_D),
        "C_H_C_effective_F_per_m2": float(g_c * epsilon_s / lambda_D),
        "C_H_Pd_effective_F_per_m2": float(g_pd * epsilon_s / lambda_D),
        "pzc_Au_tilde": float(beta * float(p["pzc_Au"])),
        "pzc_C_tilde": float(beta * float(p["pzc_C"])),
        "pzc_Pd_tilde": float(beta * float(p["pzc_Pd"])),
        "segments": segment_descriptors,
    }
