"""Separation scans and overlap-effect bookkeeping."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from .parameters import (
    ELECTROSTATIC_BACKEND,
    GEOMETRY_NAME,
    MAX_SEPARATION_M,
    MODEL_NAME,
    RESULT_SCHEMA_VERSION,
    apply_param_overrides,
    compute_derived_params,
)
from .solver import solve_case


FIGURE_SEPARATIONS_NM = (0.0, 2.0, 3.0, 10.0)
GC_METALS = ("Au", "Pd")


def _gouy_chapman_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten one with-EDL surface-charge diagnostic into two metal rows."""

    diagnostic = result.get("gouy_chapman")
    if not isinstance(diagnostic, Mapping):
        raise KeyError("with-EDL result is missing gouy_chapman diagnostics")
    if not bool(diagnostic.get("applicable", False)):
        raise ValueError("Gouy-Chapman diagnostics must apply to with-EDL cases")
    metals = diagnostic.get("metals")
    if not isinstance(metals, Mapping):
        raise ValueError("gouy_chapman.metals must be a mapping")

    common = {
        "d_Au_Pd_m": float(result["d_Au_Pd_m"]),
        "d_Au_Pd_nm": float(result["d_Au_Pd_nm"]),
        "d_over_lambda_D": float(
            float(result["d_Au_Pd_m"])
            / float(result["derived"]["lambda_D"])
        ),
        "applicable": True,
        "diagnostic_status": str(diagnostic["status"]),
        "formula": str(diagnostic["formula"]),
        "mean_definition": str(diagnostic["mean_definition"]),
        "surface_charge_definition": str(
            diagnostic["surface_charge_definition"]
        ),
        "surface_charge_sign_convention": str(
            diagnostic["surface_charge_sign_convention"]
        ),
        "interpretation": str(diagnostic["interpretation"]),
        "gouy_chapman_numerator_C_per_m": float(
            diagnostic["gouy_chapman_numerator_C_per_m"]
        ),
    }
    rows: list[dict[str, Any]] = []
    for metal in GC_METALS:
        material = metals.get(metal)
        if not isinstance(material, Mapping):
            raise ValueError(f"gouy_chapman.metals.{metal} must be a mapping")
        rows.append({**common, "metal": metal, **dict(material)})
    return rows


def _zero_offset_exponential_fit(
    d_nm: np.ndarray,
    response: np.ndarray,
    *,
    amplitude_key: str,
) -> dict[str, Any]:
    """Fit ``response = A exp(-d / length)`` without a constant offset."""

    x = np.asarray(d_nm, dtype=float)
    y = np.asarray(response, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (x <= 20.0)
    x = x[valid]
    y = y[valid]
    base = {
        "fit_equation": "response = amplitude * exp(-d_nm / decay_length_nm)",
        "offset_fixed_to_zero": True,
        "fit_range_nm": [0.0, 20.0],
        "n_points": int(x.size),
    }
    if x.size < 3:
        return {
            **base,
            "status": "not_fitted",
            "reason": "at least three finite 0-20 nm points are required",
            amplitude_key: None,
            "decay_length_nm": None,
            "decay_length_m": None,
            "r_squared": None,
        }
    if np.any(y <= 0.0):
        return {
            **base,
            "status": "not_fitted",
            "reason": "the requested positively oriented response is not strictly positive",
            amplitude_key: None,
            "decay_length_nm": None,
            "decay_length_m": None,
            "r_squared": None,
        }

    def exponential(
        distance_nm: np.ndarray, amplitude: float, length_nm: float
    ) -> np.ndarray:
        return amplitude * np.exp(-distance_nm / length_nm)

    try:
        optimum, _ = curve_fit(
            exponential,
            x,
            y,
            p0=(float(np.max(y)), 3.0),
            bounds=([0.0, np.finfo(float).eps], [np.inf, np.inf]),
            ftol=1.0e-13,
            xtol=1.0e-13,
            gtol=1.0e-13,
            maxfev=20_000,
        )
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        return {
            **base,
            "status": "not_fitted",
            "reason": str(exc),
            amplitude_key: None,
            "decay_length_nm": None,
            "decay_length_m": None,
            "r_squared": None,
        }

    amplitude, length_nm = (float(optimum[0]), float(optimum[1]))
    fitted = exponential(x, amplitude, length_nm)
    residual_sum = float(np.sum((y - fitted) ** 2))
    total_sum = float(np.sum((y - float(np.mean(y))) ** 2))
    r_squared = (
        None
        if total_sum <= np.finfo(float).tiny
        else 1.0 - residual_sum / total_sum
    )
    if not (
        np.isfinite(amplitude)
        and np.isfinite(length_nm)
        and (r_squared is None or np.isfinite(r_squared))
    ):
        raise RuntimeError("Non-finite zero-offset exponential-fit output")
    return {
        **base,
        "status": "fitted",
        amplitude_key: amplitude,
        "decay_length_nm": length_nm,
        "decay_length_m": length_nm * 1.0e-9,
        "r_squared": None if r_squared is None else float(r_squared),
    }


def build_gouy_chapman_analysis(
    scan_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize charge-derived GC lengths and 0-20 nm overlap decay fits."""

    rows = list(scan_result.get("rows", []))
    gc_rows = list(scan_result.get("gouy_chapman_rows", []))
    if not rows:
        raise ValueError("scan_result rows must not be empty")
    if not gc_rows:
        raise ValueError("scan_result gouy_chapman_rows must not be empty")

    d_nm = np.asarray([float(row["d_Au_Pd_nm"]) for row in rows], dtype=float)
    delta_E = np.asarray(
        [float(row["delta_E_overlap_vs_100nm_V"]) for row in rows], dtype=float
    )
    positive_delta_i = np.asarray(
        [-float(row["delta_i_overlap_vs_100nm_A_per_m2"]) for row in rows],
        dtype=float,
    )
    fit_E = _zero_offset_exponential_fit(
        d_nm,
        delta_E,
        amplitude_key="amplitude_V",
    )
    if fit_E["amplitude_V"] is not None:
        fit_E["amplitude_mV"] = float(fit_E["amplitude_V"]) * 1.0e3
    else:
        fit_E["amplitude_mV"] = None
    fit_E["response_definition"] = "E_mix(d) - E_mix(100 nm)"

    fit_i = _zero_offset_exponential_fit(
        d_nm,
        positive_delta_i,
        amplitude_key="amplitude_A_per_m2",
    )
    fit_i["response_definition"] = "i_mix(100 nm) - i_mix(d)"

    reference = scan_result["no_overlap_100nm_reference"]
    derived = reference["derived"]
    epsilon_s = float(derived["epsilon_s"])
    R = float(derived["R"])
    temperature = float(derived["T"])
    faraday = float(derived["F"])
    lambda_D = float(derived["lambda_D"])
    benchmark_sigma_uC_per_cm2 = 10.0
    benchmark_sigma_C_per_m2 = benchmark_sigma_uC_per_cm2 * 0.01
    benchmark_length_m = (
        2.0 * epsilon_s * R * temperature
        / (faraday * benchmark_sigma_C_per_m2)
    )

    by_metal: dict[str, Any] = {}
    for metal in GC_METALS:
        selected = [row for row in gc_rows if str(row["metal"]) == metal]
        if not selected:
            raise ValueError(f"No Gouy-Chapman scan rows were found for {metal}")

        def finite_values(key: str) -> list[float]:
            values: list[float] = []
            for row in selected:
                value = row.get(key)
                if value is None:
                    continue
                number = float(value)
                if not np.isfinite(number):
                    raise ValueError(f"Non-finite {metal} diagnostic value: {key}")
                values.append(number)
            return values

        mean_lengths = finite_values("mean_gouy_chapman_length_nm")
        local_min_lengths = finite_values("local_min_gouy_chapman_length_nm")
        local_max_lengths = finite_values("local_max_gouy_chapman_length_nm")
        mean_abs_sigma = finite_values("mean_abs_sigma_C_per_m2")
        if not mean_lengths or not mean_abs_sigma:
            raise ValueError(
                f"Finite mean Gouy-Chapman diagnostics are required for {metal}"
            )
        by_metal[metal] = {
            "n_separations": len(selected),
            "mean_gouy_chapman_length_nm_range": [
                min(mean_lengths),
                max(mean_lengths),
            ],
            "local_min_gouy_chapman_length_nm_range": (
                [min(local_min_lengths), max(local_min_lengths)]
                if local_min_lengths
                else [None, None]
            ),
            "local_max_gouy_chapman_length_nm_range": (
                [min(local_max_lengths), max(local_max_lengths)]
                if local_max_lengths
                else [None, None]
            ),
            "mean_abs_sigma_C_per_m2_range": [
                min(mean_abs_sigma),
                max(mean_abs_sigma),
            ],
            "any_mean_length_infinite": any(
                bool(row["mean_length_infinite"]) for row in selected
            ),
            "any_local_max_length_infinite": any(
                bool(row["local_max_length_infinite"]) for row in selected
            ),
            "charge_signs": sorted({str(row["charge_sign"]) for row in selected}),
        }

    max_abs_phi = max(float(row["max_abs_phi_tilde"]) for row in rows)
    threshold = float(reference["params"]["dh_warn_threshold"])
    analysis = {
        "schema_version": 1,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "electrostatic_backend": ELECTROSTATIC_BACKEND,
        "model": MODEL_NAME,
        "geometry": GEOMETRY_NAME,
        "definition": {
            "formula": "l_GC = 2 epsilon_s R T / (F |sigma|)",
            "paper_formula": "l_GC = 2 epsilon_s k_B T / (e |sigma|)",
            "surface_charge_definition": (
                "sigma_M = C_H,M,eff (E_mix - PZC_M - phi_s)"
            ),
            "representative_length_definition": (
                "l_GC,M,mean = 2 epsilon_s R T / "
                "(F <|sigma_M|>_boundary)"
            ),
            "surface_charge_sign_convention": (
                "positive sigma denotes positive charge on the metal side"
            ),
        },
        "paper_benchmark": {
            "source_pdf": "MS/cs5c06754.pdf",
            "sigma_uC_per_cm2": benchmark_sigma_uC_per_cm2,
            "sigma_C_per_m2": benchmark_sigma_C_per_m2,
            "epsilon_s_F_per_m": epsilon_s,
            "temperature_K": temperature,
            "gouy_chapman_length_m": benchmark_length_m,
            "gouy_chapman_length_nm": benchmark_length_m * 1.0e9,
        },
        "model_length_scales": {
            "debye_length_m": lambda_D,
            "debye_length_nm": lambda_D * 1.0e9,
            "gouy_chapman_by_metal": by_metal,
        },
        "overlap_exponential_fits": {
            "plateau_reference_separation_nm": 100.0,
            "potential": fit_E,
            "current_density": fit_i,
        },
        "debye_huckel_validity": {
            "max_abs_phi_tilde_over_scan": max_abs_phi,
            "threshold": threshold,
            "threshold_exceeded": bool(max_abs_phi > threshold),
            "interpretation": (
                "When max_abs_phi_tilde > 1, results are linearized-PB "
                "internal geometry sensitivity and are not claimed as "
                "absolute quantitative predictions."
            ),
        },
        "interpretation": {
            "role_in_this_model": (
                "The Au/Pd Gouy-Chapman lengths are charge-derived "
                "diagnostics, not independent screening lengths in the "
                "linearized-PB equation."
            ),
            "gap_topology": (
                "The Au-Pd gap exposes a flush ideal uncharged insulating "
                "substrate with a homogeneous-Neumann electrostatic boundary, "
                "unlike the charged-support context of the paper."
            ),
            "comparison_rule": (
                "Compare the empirical overlap decay lengths with both the "
                "Debye length and the Au/Pd charge-derived Gouy-Chapman ranges; "
                "do not attribute the overlap range to one GC scalar alone."
            ),
        },
    }
    return analysis


def default_separations(params: Mapping[str, Any]) -> np.ndarray:
    """Return the required 0-100 nm grid, including exact Debye multiples."""

    canonical = apply_param_overrides(params)
    lambda_nm = float(compute_derived_params(canonical)["lambda_D"]) * 1.0e9
    values_nm = np.concatenate(
        [
            np.arange(0.0, 10.0 + 0.25, 0.5),
            np.asarray([15.0, 20.0, 30.0, 50.0, 75.0, 100.0]),
            lambda_nm * np.asarray([1.0, 2.0, 3.0, 5.0]),
        ]
    )
    values_nm = values_nm[(values_nm >= 0.0) & (values_nm <= 100.0)]
    # Picometre-scale rounding only removes floating duplicate representations;
    # the lambda_D multiples remain exact to well beyond solver precision.
    return np.unique(np.round(values_nm, decimals=12)) * 1.0e-9


def _validated_separations(
    params: Mapping[str, Any], separations_m: Iterable[float] | None
) -> np.ndarray:
    values = (
        default_separations(params)
        if separations_m is None
        else np.asarray(list(separations_m), dtype=float)
    )
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("separations_m must be a non-empty finite 1D sequence")
    if np.any(values < 0.0) or np.any(
        values > MAX_SEPARATION_M * (1.0 + 1.0e-12)
    ):
        maximum_nm = MAX_SEPARATION_M * 1.0e9
        raise ValueError(f"Every separation must be within 0-{maximum_nm:g} nm")
    return np.unique(values)


def run_separation_scan(
    params: Mapping[str, Any],
    separations_m: Iterable[float] | None = None,
    *,
    retain_models_at_nm: Iterable[float] = FIGURE_SEPARATIONS_NM,
) -> dict[str, Any]:
    """Solve the required separation series and decompose total/overlap effects."""

    canonical = apply_param_overrides(params)
    values = _validated_separations(canonical, separations_m)
    retain = tuple(float(value) for value in retain_models_at_nm)
    no_edl = solve_case(canonical, 0.0, use_edl=False)
    rows: list[dict[str, Any]] = []
    gouy_chapman_rows: list[dict[str, Any]] = []
    cases: dict[float, dict[str, Any]] = {}
    result_100: dict[str, Any] | None = None

    for separation in values:
        separation_nm = float(separation * 1.0e9)
        keep_model = any(abs(separation_nm - target) <= 1.0e-9 for target in retain)
        if keep_model:
            with_edl, model = solve_case(
                canonical, float(separation), use_edl=True, return_model=True
            )
            cases[min(retain, key=lambda target: abs(separation_nm - target))] = {
                "result": with_edl,
                "model": model,
            }
        else:
            with_edl = solve_case(canonical, float(separation), use_edl=True)
        if abs(separation_nm - 100.0) <= 1.0e-9:
            result_100 = with_edl
        gouy_chapman_rows.extend(_gouy_chapman_rows(with_edl))
        rows.append(
            {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "electrostatic_backend": ELECTROSTATIC_BACKEND,
                "d_Au_Pd_m": float(separation),
                "d_Au_Pd_nm": separation_nm,
                "d_over_lambda_D": float(
                    separation / float(with_edl["derived"]["lambda_D"])
                ),
                "E_mix_with_EDL_V": float(with_edl["E_mix_V"]),
                "E_mix_without_EDL_V": float(no_edl["E_mix_V"]),
                "delta_E_total_EDL_V": float(
                    with_edl["E_mix_V"] - no_edl["E_mix_V"]
                ),
                "i_mix_avg_with_EDL_A_per_m2": float(
                    with_edl["i_mix_avg_A_per_m2"]
                ),
                "i_mix_avg_without_EDL_A_per_m2": float(
                    no_edl["i_mix_avg_A_per_m2"]
                ),
                "delta_i_total_EDL_A_per_m2": float(
                    with_edl["i_mix_avg_A_per_m2"]
                    - no_edl["i_mix_avg_A_per_m2"]
                ),
                "I_Au_with_EDL_A": float(with_edl["I_Au_A"]),
                "I_Pd_with_EDL_A": float(with_edl["I_Pd_A"]),
                "relative_balance_residual": float(
                    with_edl["relative_balance_residual"]
                ),
                "max_abs_phi_tilde": float(
                    with_edl["debye_huckel_validity"]["max_abs_phi_tilde"]
                ),
                "N_modes": int(with_edl["electrostatics"]["n_modes"]),
                "n_coefficients": int(
                    with_edl["electrostatics"]["n_coefficients"]
                ),
                "Nx": int(
                    with_edl["electrostatics"][
                        "surface_quadrature_target_points"
                    ]
                ),
            }
        )

    # A custom scan may omit 100 nm; solve it once so overlap always has the
    # same, explicit reference rather than silently using the largest input.
    if result_100 is None:
        result_100 = solve_case(canonical, 100.0e-9, use_edl=True)

    E_reference = float(result_100["E_mix_V"])
    i_reference = float(result_100["i_mix_avg_A_per_m2"])
    for row in rows:
        row["E_mix_no_overlap_100nm_V"] = E_reference
        row["i_mix_avg_no_overlap_100nm_A_per_m2"] = i_reference
        row["delta_E_overlap_vs_100nm_V"] = (
            float(row["E_mix_with_EDL_V"]) - E_reference
        )
        row["delta_i_overlap_vs_100nm_A_per_m2"] = (
            float(row["i_mix_avg_with_EDL_A_per_m2"]) - i_reference
        )

    by_nm = {round(float(row["d_Au_Pd_nm"]), 9): row for row in rows}
    if 75.0 in by_nm and 100.0 in by_nm:
        delta_E_75_100 = abs(
            float(by_nm[75.0]["E_mix_with_EDL_V"])
            - float(by_nm[100.0]["E_mix_with_EDL_V"])
        )
        delta_i_rel_75_100 = abs(
            float(by_nm[75.0]["i_mix_avg_with_EDL_A_per_m2"])
            - float(by_nm[100.0]["i_mix_avg_with_EDL_A_per_m2"])
        ) / max(abs(float(by_nm[100.0]["i_mix_avg_with_EDL_A_per_m2"])), 1.0e-300)
        plateau = {
            "checked": True,
            "delta_E_75_vs_100_V": delta_E_75_100,
            "delta_i_relative_75_vs_100": delta_i_rel_75_100,
            "E_tolerance_V": 0.1e-3,
            "i_relative_tolerance": 0.005,
            "passed": bool(
                delta_E_75_100 <= 0.1e-3 and delta_i_rel_75_100 <= 0.005
            ),
        }
    else:
        plateau = {
            "checked": False,
            "passed": False,
            "reason": "75 and/or 100 nm missing from the requested rows",
        }

    reference_100 = dict(result_100)
    reference_100["no_overlap_plateau_validated"] = bool(plateau["passed"])
    reference_100["reference_label"] = (
        "validated no-overlap plateau (100 nm)"
        if plateau["passed"]
        else "100 nm reference; no-overlap plateau not validated"
    )
    for row in rows:
        row["reference_100nm_plateau_validated"] = bool(plateau["passed"])

    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "electrostatic_backend": ELECTROSTATIC_BACKEND,
        "rows": rows,
        "gouy_chapman_rows": gouy_chapman_rows,
        "cases": cases,
        "without_edl_reference": no_edl,
        "no_overlap_100nm_reference": reference_100,
        "plateau_check": plateau,
        "separations_m": values,
    }
