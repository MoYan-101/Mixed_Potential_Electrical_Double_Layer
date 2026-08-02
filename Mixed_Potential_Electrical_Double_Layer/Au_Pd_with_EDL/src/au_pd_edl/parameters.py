"""Parameters for flush Au/Pd electrodes on an insulating substrate.

All dimensional values use SI units.  In particular, ``C_tot`` is the
concentration (mol m-3) of *each* ion in a symmetric 1:1 electrolyte.  The
span between Au and Pd exposes an ideal, uncharged insulating substrate to
the electrolyte.  It has no Stern or Faradaic parameters, so legacy carbon
or charged-support parameters are rejected even when their value is zero.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping


# The standard overlap scan remains limited to 0--100 nm in ``scan.py``.
# Standalone cases may use a wider gap for explicit far-separation checks and
# polarization schematics, provided their spectral resolution is validated.
MAX_SEPARATION_M = 1000.0e-9
PACKAGE_VERSION = "0.3.0"
RESULT_SCHEMA_VERSION = 2
ELECTROSTATIC_BACKEND = "semi_infinite_cosine_fourier"
ELECTROLYTE_DOMAIN_ID = "semi_infinite_upper_half_strip_y_ge_0"
ELECTROLYTE_DOMAIN_DESCRIPTION = (
    "semi-infinite strip 0 <= x_tilde <= L_tilde, y_tilde >= 0"
)
MODEL_NAME = "flush Au|uncharged insulating substrate|Pd"
GEOMETRY_NAME = "flush_coplanar_on_uncharged_insulating_substrate"

# These keys have a superficially plausible meaning in the old Au|C|Pd model,
# so give them a targeted error instead of treating them as ordinary typos.
FORBIDDEN_LEGACY_KEYS = frozenset(
    {
        "L_gap",
        "Cdl_C",
        "C_H_C",
        "g_C",
        "pzc_C",
    }
)


def default_params() -> dict[str, Any]:
    """Return a fresh flat-SI baseline parameter dictionary.

    The physical baseline is the project's equal-length, equal-``i0``,
    ``alpha=0.5`` case.  ``d_Au_Pd`` is the width of exposed ideal insulating
    substrate between the flush electrodes and defaults to the exact adjacent
    limit.
    """

    params: dict[str, Any] = {
        # Physical constants
        "R": 8.314,
        "F": 96485.0,
        "T": 298.0,
        # Solution permittivity
        "epsilon0": 8.8541878128e-12,
        "epsilon_r": 78.5,
        "epsilon_s": None,
        # Symmetric 1:1 electrolyte; 10 mol/m3 = 10 mM
        "C_tot": 10.0,
        "lambda_D": None,
        # Geometry
        "L_Au": 25.0e-9,
        "d_Au_Pd": 0.0,
        "L_Pd": 25.0e-9,
        "out_of_plane_width": 0.01,
        # Flush horizontal metal-solution interfaces at y=0
        "C_H_Au": 0.20,
        "C_H_Pd": 0.40,
        "g_Au": None,
        "g_Pd": None,
        "pzc_Au": 0.93,
        "pzc_Pd": 0.78,
        # Irreversible Frumkin-corrected Butler-Volmer kinetics
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
        # Preserve the exact source-baseline value for traceable d=0 regression.
        "E2_eq_pH_slope_V_per_pH": -0.059126500015747985,
        "it0_1_pH_order": 0.0,
        "it0_2_pH_order": 1.0,
        # Semi-infinite cosine/Fourier electrostatic solver controls.  The
        # coefficient vector contains modes n=0,...,N_modes.  Nx controls the
        # surface quadrature/profile resolution and mirrors the legacy solver.
        "N_modes": 960,
        "Nx": 5000,
        "root_xtol_V": 1.0e-10,
        "root_rtol": 1.0e-12,
        "root_maxiter": 200,
        "root_bracket_padding_V": 0.25,
        "max_bracket_expands": 12,
        # Linearized-PB applicability reporting
        "dh_warn_threshold": 1.0,
        "dh_violation_action": "warn",
    }
    return copy.deepcopy(params)


def _reject_forbidden_keys(values: Mapping[str, Any]) -> None:
    present = sorted(FORBIDDEN_LEGACY_KEYS.intersection(values))
    if present:
        names = ", ".join(present)
        raise ValueError(
            f"Forbidden Au|C|Pd parameter(s): {names}.  The Au-Pd separation "
            "exposes an ideal uncharged insulating substrate; this model has "
            "no carbon/support Stern-charging parameters."
        )


def _check_known_keys(values: Mapping[str, Any]) -> None:
    _reject_forbidden_keys(values)
    unknown = sorted(set(values).difference(default_params()))
    if unknown:
        raise KeyError(f"Unknown Au|Pd parameter(s): {', '.join(unknown)}")


def apply_param_overrides(
    params: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
    **keyword_overrides: Any,
) -> dict[str, Any]:
    """Return validated parameters after applying non-mutating overrides.

    ``params`` may be a full parameter set or a partial mapping.  Missing keys
    are filled from :func:`default_params`.  ``keyword_overrides`` are applied
    last.  Unknown and legacy support/carbon keys are never silently retained.
    """

    base = default_params()
    for values in (params, overrides or {}, keyword_overrides):
        _check_known_keys(values)
        base.update(copy.deepcopy(dict(values)))
    validate_params(base)
    return base


def load_params(
    path: str | Path,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a JSON object, merge it over the baseline, and validate it."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"Parameter file must contain a JSON object: {source}")
    return apply_param_overrides(raw, overrides)


def _finite(params: Mapping[str, Any], name: str) -> float:
    try:
        value = float(params[name])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _optional_positive(params: Mapping[str, Any], name: str) -> None:
    if params[name] is None:
        return
    if _finite(params, name) <= 0.0:
        raise ValueError(f"{name} must be positive when provided")


def _optional_nonnegative(params: Mapping[str, Any], name: str) -> None:
    if params[name] is None:
        return
    if _finite(params, name) < 0.0:
        raise ValueError(f"{name} must be non-negative when provided")


def validate_params(params: Mapping[str, Any]) -> None:
    """Validate completeness, topology, and basic physical/numerical ranges."""

    _check_known_keys(params)
    missing = sorted(set(default_params()).difference(params))
    if missing:
        raise KeyError(f"Missing required parameter(s): {', '.join(missing)}")

    for name in ("R", "F", "T", "epsilon0", "epsilon_r"):
        if _finite(params, name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    _optional_positive(params, "epsilon_s")
    _optional_positive(params, "lambda_D")
    if params["lambda_D"] is None and _finite(params, "C_tot") <= 0.0:
        raise ValueError("C_tot must be positive when lambda_D is auto-calculated")
    if params["lambda_D"] is not None:
        # C_tot remains traceability metadata but must still be finite/positive.
        if _finite(params, "C_tot") <= 0.0:
            raise ValueError("C_tot must be positive")

    if _finite(params, "L_Au") <= 0.0 or _finite(params, "L_Pd") <= 0.0:
        raise ValueError("L_Au and L_Pd must be positive")
    separation = _finite(params, "d_Au_Pd")
    if separation < 0.0 or separation > MAX_SEPARATION_M * (1.0 + 1.0e-12):
        max_separation_nm = MAX_SEPARATION_M * 1.0e9
        raise ValueError(
            f"d_Au_Pd must be within 0-{max_separation_nm:g} nm"
        )
    if _finite(params, "out_of_plane_width") <= 0.0:
        raise ValueError("out_of_plane_width must be positive")

    for name in ("C_H_Au", "C_H_Pd"):
        if _finite(params, name) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    for name in ("g_Au", "g_Pd"):
        _optional_nonnegative(params, name)

    for name in ("it0_1", "it0_2"):
        if _finite(params, name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name in ("alpha1", "alpha2"):
        alpha = _finite(params, name)
        if not 0.0 <= alpha <= 1.0:
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

    for name in (
        "root_xtol_V",
        "root_rtol",
        "root_bracket_padding_V",
        "dh_warn_threshold",
    ):
        if _finite(params, name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name, minimum in (
        ("N_modes", 1),
        ("Nx", 3),
        ("root_maxiter", 1),
        ("max_bracket_expands", 0),
    ):
        raw = params[name]
        if isinstance(raw, bool):
            raise ValueError(f"{name} must be an integer >= {minimum}")
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer >= {minimum}") from exc
        if value != raw or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    action = str(params["dh_violation_action"]).lower()
    if action not in {"ignore", "warn", "raise"}:
        raise ValueError("dh_violation_action must be one of: ignore, warn, raise")


def compute_derived_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Compute electrostatic, geometric, and spectral quantities without mutation."""

    validate_params(params)
    R = float(params["R"])
    F = float(params["F"])
    T = float(params["T"])
    beta = F / (R * T)
    epsilon_s = (
        float(params["epsilon_s"])
        if params["epsilon_s"] is not None
        else float(params["epsilon0"]) * float(params["epsilon_r"])
    )
    lambda_D = (
        float(params["lambda_D"])
        if params["lambda_D"] is not None
        else math.sqrt(epsilon_s * R * T / (2.0 * F**2 * float(params["C_tot"])))
    )

    C_D = epsilon_s / lambda_D
    g_Au = (
        float(params["g_Au"])
        if params["g_Au"] is not None
        else float(params["C_H_Au"]) / C_D
    )
    g_Pd = (
        float(params["g_Pd"])
        if params["g_Pd"] is not None
        else float(params["C_H_Pd"]) / C_D
    )

    L_Au = float(params["L_Au"])
    d_Au_Pd = float(params["d_Au_Pd"])
    L_Pd = float(params["L_Pd"])
    L_total = L_Au + d_Au_Pd + L_Pd
    reactive_length = L_Au + L_Pd
    width = float(params["out_of_plane_width"])

    return {
        "R": R,
        "F": F,
        "T": T,
        "beta": beta,
        "thermal_voltage_V": 1.0 / beta,
        "epsilon_s": epsilon_s,
        "C_tot": float(params["C_tot"]),
        "lambda_D": lambda_D,
        "inverse_Debye_length_per_m": 1.0 / lambda_D,
        "C_D": C_D,
        "g_Au": g_Au,
        "g_Pd": g_Pd,
        "pzc_Au": float(params["pzc_Au"]),
        "pzc_Pd": float(params["pzc_Pd"]),
        "pzc_Au_tilde": beta * float(params["pzc_Au"]),
        "pzc_Pd_tilde": beta * float(params["pzc_Pd"]),
        "L_Au": L_Au,
        "d_Au_Pd": d_Au_Pd,
        "L_Pd": L_Pd,
        "L_total": L_total,
        "reactive_length": reactive_length,
        "x_Au_end": L_Au,
        "x_Pd_start": L_Au + d_Au_Pd,
        "L_Au_tilde": L_Au / lambda_D,
        "d_Au_Pd_tilde": d_Au_Pd / lambda_D,
        "L_Pd_tilde": L_Pd / lambda_D,
        "L_total_tilde": L_total / lambda_D,
        "out_of_plane_width": width,
        "reactive_area_m2": width * reactive_length,
        "N_modes": int(params["N_modes"]),
        "Nx": int(params["Nx"]),
        "electrostatic_backend": ELECTROSTATIC_BACKEND,
        "gap_exists": d_Au_Pd > 0.0,
        "substrate_gap_exists": d_Au_Pd > 0.0,
        "geometry": GEOMETRY_NAME,
    }
