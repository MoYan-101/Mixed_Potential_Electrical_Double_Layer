"""Support-length OFAT figures for the Au=Pd=2 nm Au|C|Pd model.

The scan uses the same physical parameter set as the formal six legacy cases,
but is deliberately evaluated through ``build_scan_case`` so intermediate
support lengths do not become publication cases.  Mixed potentials and
currents use the absolute-current balance with 128-point Gauss--Legendre
quadrature on each active material.  No 2-D fields are built for this OFAT.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import legacy_au_c_pd_engine as engine  # noqa: E402


HERE = MODULE_DIR
OUTPUT_TAG = f"au2_pd2_{engine.BASE_RESULT_ID}"
OUT_DIR = HERE / "Au_C_Pd" / "Figure_L_support"
FIGURES_DIR = OUT_DIR / "OFAT"
CSV_DIR = OUT_DIR / "csv"
INPUTS_DIR = OUT_DIR / "inputs"

MAIN_SUPPORT_NM = (
    0.0,
    0.25,
    0.50,
    0.75,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
)
OVERLAP_EXTRA_SUPPORT_NM = (11.0, 12.0, 15.0)
OVERLAP_SUPPORT_NM = MAIN_SUPPORT_NM + OVERLAP_EXTRA_SUPPORT_NM
REFERENCE_SUPPORT_NM = 1000.0

PLATEAU_FIT_MIN_NM = 5.0
PLATEAU_TOLERANCE_V = 0.1e-3
OVERLAP_FRACTION_TOLERANCE = 0.05
OVERLAP_E_TOLERANCE_MV = 0.1
OVERLAP_PHI_TOLERANCE_MV = 0.1
OVERLAP_I_TOLERANCE_PERCENT = 0.1

WITH_COLOR = "#F26B38"
WITHOUT_COLOR = "#12355B"
DARK = "#111827"
GRAY = "#8C8C8C"
LIGHT_GRAY = "#D1D5DB"
LAMBDA_COLOR = "#5A90C8"
PLATEAU_COLOR = "#D83A2E"
OVERLAP_I_COLOR = "#4D8061"


def configure_style() -> None:
    """Apply the Figure 3 Helvetica-first publication style."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica",
                "Nimbus Sans",
                "Arial",
                "DejaVu Sans",
                "sans-serif",
            ],
            "font.size": 9.8,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10.4,
            "axes.linewidth": 1.0,
            "axes.edgecolor": DARK,
            "axes.labelcolor": DARK,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "legend.frameon": False,
            "savefig.dpi": 600,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "mathtext.fontset": "stixsans",
        }
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    return value


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            if list(row.keys()) != columns:
                raise ValueError(f"Inconsistent CSV columns for {path}")
            writer.writerow(_jsonable(row))
    return path


def _close(value: float, target: float, atol: float = 1e-9) -> bool:
    return math.isclose(float(value), float(target), rel_tol=0.0, abs_tol=atol)


def _case_for_value(cases: Iterable[Any], value_nm: float) -> Any | None:
    for case in cases:
        if _close(float(case.value_nm), value_nm):
            return case
    return None


def _validate_case(case: Any, value_nm: float) -> None:
    expected_modes = (
        engine.LONG_N_MODES if _close(value_nm, REFERENCE_SUPPORT_NM) else engine.SHORT_N_MODES
    )
    expected = {
        "L_Au": engine.L_AU_M,
        "L_gap": value_nm * 1.0e-9,
        "L_Pd_len": engine.L_PD_M,
        "Cdl_Au": engine.C_H_AU_F_PER_M2,
        "Cdl_C": engine.C_H_SUPPORT_F_PER_M2,
        "Cdl_Pd": engine.C_H_PD_F_PER_M2,
        "it0_1": engine.I0_EQUAL_A_PER_M2,
        "it0_2": engine.I0_EQUAL_A_PER_M2,
        "alpha1": engine.ALPHA_EQUAL,
        "alpha2": engine.ALPHA_EQUAL,
        "out_of_plane_width": engine.OUT_OF_PLANE_WIDTH_M,
    }
    for key, target in expected.items():
        actual = float(case.params[key])
        atol = max(1e-15, abs(target) * 1e-12)
        if not math.isclose(actual, target, rel_tol=0.0, abs_tol=atol):
            raise ValueError(
                f"L_support={value_nm:g} nm has {key}={actual:.15g}; expected {target:.15g}"
            )
    if int(case.params["N_modes"]) != expected_modes:
        raise ValueError(
            f"L_support={value_nm:g} nm has N_modes={case.params['N_modes']}; "
            f"expected {expected_modes}"
        )
    if int(case.gl_order) != engine.DEFAULT_GL_ORDER:
        raise ValueError(
            f"L_support={value_nm:g} nm has GL order {case.gl_order}; "
            f"expected {engine.DEFAULT_GL_ORDER}"
        )
    for result_name, result in (("with EDL", case.res_edl), ("w/o EDL", case.res_no)):
        residual = float(result["relative_balance_residual"])
        if not math.isfinite(residual) or residual >= engine.RELATIVE_BALANCE_TOL:
            raise ValueError(
                f"{result_name} balance failed at L_support={value_nm:g} nm: {residual:.3g}"
            )


def _get_or_build_cases(base_cases: Sequence[Any] | None) -> list[Any]:
    reusable = tuple(base_cases or ())
    values = OVERLAP_SUPPORT_NM + (REFERENCE_SUPPORT_NM,)
    cases: list[Any] = []
    for value_nm in values:
        case = _case_for_value(reusable, value_nm)
        if case is None:
            case = engine.build_scan_case(
                value_nm,
                n_modes=(
                    engine.LONG_N_MODES
                    if _close(value_nm, REFERENCE_SUPPORT_NM)
                    else engine.SHORT_N_MODES
                ),
                gl_order=engine.DEFAULT_GL_ORDER,
                include_2d=False,
            )
        _validate_case(case, value_nm)
        cases.append(case)
    return cases


def _support_charge(case: Any) -> dict[str, Any]:
    if float(case.value_nm) <= 1e-12:
        return {
            "support_charge_status": "not_applicable",
            "sigma_C_signed_mean_C_per_m2": None,
            "sigma_C_mean_abs_C_per_m2": None,
            "sigma_C_midpoint_C_per_m2": None,
            "sigma_C_min_C_per_m2": None,
            "sigma_C_max_C_per_m2": None,
            "sigma_C_sign_change": None,
            "L_GC_C_nm": None,
            "L_GC_C_midpoint_nm": None,
            "L_support_over_L_GC_C": None,
        }
    segment = next(
        (item for item in case.sigma_segments if str(item.material).lower() == "support"),
        None,
    )
    if segment is None:
        raise ValueError(f"Missing support charge segment at {case.value_nm:g} nm")
    x_nm = np.asarray(segment.x_nm, dtype=float)
    sigma = np.asarray(segment.sigma_C_per_m2, dtype=float)
    if x_nm.size < 2 or sigma.shape != x_nm.shape:
        raise ValueError(f"Invalid support charge profile at {case.value_nm:g} nm")
    width_nm = float(case.value_nm)
    signed_mean = float(np.trapezoid(sigma, x_nm) / width_nm)
    mean_abs = float(np.trapezoid(np.abs(sigma), x_nm) / width_nm)
    midpoint = float(np.interp(float(case.L_Au_nm) + 0.5 * width_nm, x_nm, sigma))
    if not math.isfinite(mean_abs) or mean_abs <= 0.0:
        raise ValueError(f"Invalid mean support charge at {case.value_nm:g} nm")
    prefactor = (
        2.0
        * float(case.derived["epsilon_s"])
        * float(case.params["R"])
        * float(case.params["T"])
        / float(case.params["F"])
    )
    lgc_nm = prefactor / mean_abs * 1.0e9
    lgc_mid_nm = prefactor / abs(midpoint) * 1.0e9 if midpoint != 0.0 else None
    sigma_min = float(np.min(sigma))
    sigma_max = float(np.max(sigma))
    return {
        "support_charge_status": "defined",
        "sigma_C_signed_mean_C_per_m2": signed_mean,
        "sigma_C_mean_abs_C_per_m2": mean_abs,
        "sigma_C_midpoint_C_per_m2": midpoint,
        "sigma_C_min_C_per_m2": sigma_min,
        "sigma_C_max_C_per_m2": sigma_max,
        "sigma_C_sign_change": bool(sigma_min < 0.0 < sigma_max),
        "L_GC_C_nm": float(lgc_nm),
        "L_GC_C_midpoint_nm": None if lgc_mid_nm is None else float(lgc_mid_nm),
        "L_support_over_L_GC_C": width_nm / float(lgc_nm),
    }


def _raw_row(case: Any, reference_case: Any) -> dict[str, Any]:
    value_nm = float(case.value_nm)
    result = case.res_edl
    result_no = case.res_no
    reference = reference_case.res_edl
    charge = _support_charge(case)
    delta_e_v = float(result["E_mix"]) - float(reference["E_mix"])
    delta_i = float(result["i_mix_avg_A_per_m2"]) - float(
        reference["i_mix_avg_A_per_m2"]
    )
    delta_i_percent = 100.0 * delta_i / float(reference["i_mix_avg_A_per_m2"])
    phi_au_mv = 1000.0 * float(result["phi2_1_meanV"])
    phi_pd_mv = 1000.0 * float(result["phi2_2_meanV"])
    delta_phi_au = phi_au_mv - 1000.0 * float(reference["phi2_1_meanV"])
    delta_phi_pd = phi_pd_mv - 1000.0 * float(reference["phi2_2_meanV"])
    l_au = float(case.derived["L_Au"])
    l_pd = float(case.derived["L_Pd_len"])
    delta_phi_rms = math.sqrt(
        (l_au * delta_phi_au**2 + l_pd * delta_phi_pd**2) / (l_au + l_pd)
    )
    lambda_nm = float(case.derived["lambda_D"]) * 1.0e9
    return {
        "L_support_nm": value_nm,
        "L_gap_m": float(case.params["L_gap"]),
        "N_modes": int(case.params["N_modes"]),
        "gl_order": int(case.gl_order),
        "include_2d": False,
        "lambda_D_nm": lambda_nm,
        "L_support_over_lambda_D": value_nm / lambda_nm,
        "E_mix_with_V": float(result["E_mix"]),
        "E_mix_no_V": float(result_no["E_mix"]),
        "delta_E_mix_V": float(result["E_mix"]) - float(result_no["E_mix"]),
        "i_mix_avg_with_A_per_m2": float(result["i_mix_avg_A_per_m2"]),
        "i_mix_avg_no_A_per_m2": float(result_no["i_mix_avg_A_per_m2"]),
        "delta_i_mix_avg_A_per_m2": float(result["i_mix_avg_A_per_m2"])
        - float(result_no["i_mix_avg_A_per_m2"]),
        "relative_balance_with": float(result["relative_balance_residual"]),
        "relative_balance_no": float(result_no["relative_balance_residual"]),
        "max_abs_phi_tilde_with_edl": float(result["max_abs_phi_tilde"]),
        **charge,
        "overlap_reference_L_support_nm": float(reference_case.value_nm),
        "overlap_delta_E_mix_vs_1000_V": delta_e_v,
        "overlap_abs_delta_E_mix_vs_1000_mV": 1000.0 * abs(delta_e_v),
        "overlap_delta_i_mix_avg_vs_1000_A_per_m2": delta_i,
        "overlap_delta_i_mix_avg_vs_1000_percent": delta_i_percent,
        "overlap_abs_delta_i_mix_avg_vs_1000_percent": abs(delta_i_percent),
        "phi_RP_Au_mean_mV": phi_au_mv,
        "phi_RP_Pd_mean_mV": phi_pd_mv,
        "overlap_delta_phi_RP_Au_mean_vs_1000_mV": delta_phi_au,
        "overlap_delta_phi_RP_Pd_mean_vs_1000_mV": delta_phi_pd,
        "overlap_delta_phi_RP_active_rms_vs_1000_mV": delta_phi_rms,
        "overlap_delta_phi_RP_active_max_material_mean_vs_1000_mV": max(
            abs(delta_phi_au), abs(delta_phi_pd)
        ),
    }


def _build_rows(cases: Sequence[Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference = _case_for_value(cases, REFERENCE_SUPPORT_NM)
    if reference is None:
        raise ValueError("The 1000 nm reference case is missing")
    rows = [
        _raw_row(case, reference)
        for value_nm in OVERLAP_SUPPORT_NM
        for case in [_case_for_value(cases, value_nm)]
        if case is not None
    ]
    if len(rows) != len(OVERLAP_SUPPORT_NM):
        raise ValueError("The support scan is incomplete")
    contact = rows[0]
    fraction_fields = (
        ("overlap_abs_delta_E_mix_vs_1000_mV", "overlap_fraction_E_mix"),
        (
            "overlap_delta_phi_RP_active_rms_vs_1000_mV",
            "overlap_fraction_phi_RP_active_rms",
        ),
        (
            "overlap_abs_delta_i_mix_avg_vs_1000_percent",
            "overlap_fraction_i_mix_avg",
        ),
    )
    for source, destination in fraction_fields:
        denominator = float(contact[source])
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError(f"Invalid L_support=0 overlap normalization for {source}")
        for row in rows:
            row[destination] = float(row[source]) / denominator
    reference_row = _raw_row(reference, reference)
    for _, destination in fraction_fields:
        reference_row[destination] = 0.0
    return rows, reference_row


def _exponential_approach(x: Any, e_inf: float, amplitude: float, decay_nm: float) -> Any:
    return e_inf + amplitude * np.exp(-np.asarray(x, dtype=float) / decay_nm)


def _crossing(rows: Sequence[Mapping[str, Any]]) -> float | None:
    finite = [row for row in rows if row["L_GC_C_nm"] is not None]
    for left, right in zip(finite[:-1], finite[1:]):
        x0 = float(left["L_support_nm"])
        x1 = float(right["L_support_nm"])
        f0 = x0 - float(left["L_GC_C_nm"])
        f1 = x1 - float(right["L_GC_C_nm"])
        if f0 == 0.0:
            return x0
        if f0 * f1 <= 0.0 and f1 != f0:
            return float(x0 - f0 * (x1 - x0) / (f1 - f0))
    return None


def _plateau_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fit_rows = [
        row
        for row in rows
        if PLATEAU_FIT_MIN_NM <= float(row["L_support_nm"]) <= 10.0
    ]
    x = np.asarray([row["L_support_nm"] for row in fit_rows], dtype=float)
    y = np.asarray([row["E_mix_with_V"] for row in fit_rows], dtype=float)
    span = max(float(np.ptp(y)), 1.0e-6)
    amplitude0 = float(y[0] - y[-1])
    if abs(amplitude0) < 1.0e-8:
        amplitude0 = math.copysign(span, amplitude0 if amplitude0 != 0.0 else 1.0)
    p0 = (float(y[-1]), amplitude0, 2.5)
    lower = (float(np.min(y) - 5.0 * span), -20.0 * span, 0.05)
    upper = (float(np.max(y) + 5.0 * span), 20.0 * span, 100.0)
    popt, covariance = curve_fit(
        _exponential_approach,
        x,
        y,
        p0=p0,
        bounds=(lower, upper),
        maxfev=50000,
    )
    e_inf, amplitude, decay_nm = [float(item) for item in popt]
    predicted = np.asarray(_exponential_approach(x, *popt), dtype=float)
    residual = y - predicted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    if abs(amplitude) <= PLATEAU_TOLERANCE_V:
        continuous_nm = 0.0
    else:
        continuous_nm = max(
            0.0,
            decay_nm * math.log(abs(amplitude) / PLATEAU_TOLERANCE_V),
        )
    sampled_nm = next(
        (
            float(row["L_support_nm"])
            for index, row in enumerate(rows)
            if max(
                abs(float(item["E_mix_with_V"]) - e_inf)
                for item in rows[index:]
            )
            <= PLATEAU_TOLERANCE_V
        ),
        None,
    )
    return {
        "model": "E_mix_with(L) = E_inf + amplitude * exp(-L/decay_length)",
        "fit_range_nm": [PLATEAU_FIT_MIN_NM, 10.0],
        "E_inf_V": e_inf,
        "amplitude_V": amplitude,
        "decay_length_nm": decay_nm,
        "fit_r_squared": r_squared,
        "max_fit_residual_V": float(np.max(np.abs(residual))),
        "fit_covariance": np.asarray(covariance, dtype=float).tolist(),
        "plateau_tolerance_V": PLATEAU_TOLERANCE_V,
        "plateau_continuous_nm": continuous_nm,
        "plateau_first_sampled_nm": sampled_nm,
        "lambda_D_nm": float(rows[0]["lambda_D_nm"]),
        "L_support_equals_L_GC_C_nm": _crossing(rows),
        "LGC_definition": "2 * epsilon_s * R * T / (F * mean_support_abs_sigma_C)",
        "debye_huckel_caveat": (
            "L_GC is a charge-derived diagnostic within the linearized-PB model; "
            "max_abs_phi_tilde may exceed 1."
        ),
    }


def _threshold_crossing(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    threshold: float,
) -> tuple[float | None, float | None]:
    x = np.asarray([row["L_support_nm"] for row in rows], dtype=float)
    y = np.asarray([abs(float(row[key])) for row in rows], dtype=float)
    index = next(
        (idx for idx in range(y.size) if float(np.max(y[idx:])) <= threshold),
        None,
    )
    if index is None:
        return None, None
    sampled = float(x[index])
    if index == 0:
        return sampled, sampled
    x0, x1 = float(x[index - 1]), float(x[index])
    y0, y1 = float(y[index - 1]), float(y[index])
    if y0 > 0.0 and y1 > 0.0 and not math.isclose(y0, y1):
        continuous = x0 + math.log(y0 / threshold) / math.log(y0 / y1) * (x1 - x0)
    elif not math.isclose(y0, y1):
        continuous = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
    else:
        continuous = sampled
    return float(continuous), sampled


def _interpolate(rows: Sequence[Mapping[str, Any]], x_nm: float, key: str) -> float | None:
    finite = [
        (float(row["L_support_nm"]), float(row[key]))
        for row in rows
        if row[key] is not None
    ]
    if not finite:
        return None
    return float(
        np.interp(
            x_nm,
            np.asarray([item[0] for item in finite]),
            np.asarray([item[1] for item in finite]),
        )
    )


def _overlap_summary(
    rows: Sequence[Mapping[str, Any]],
    reference_row: Mapping[str, Any],
) -> dict[str, Any]:
    specs = {
        "E_mix": (
            "overlap_abs_delta_E_mix_vs_1000_mV",
            "overlap_fraction_E_mix",
            OVERLAP_E_TOLERANCE_MV,
            "mV",
        ),
        "phi_RP_active_rms": (
            "overlap_delta_phi_RP_active_rms_vs_1000_mV",
            "overlap_fraction_phi_RP_active_rms",
            OVERLAP_PHI_TOLERANCE_MV,
            "mV",
        ),
        "i_mix_avg": (
            "overlap_abs_delta_i_mix_avg_vs_1000_percent",
            "overlap_fraction_i_mix_avg",
            OVERLAP_I_TOLERANCE_PERCENT,
            "% of 1000 nm reference",
        ),
    }
    metrics: dict[str, Any] = {}
    absolute_continuous: list[float] = []
    absolute_sampled: list[float] = []
    fraction_continuous: list[float] = []
    fraction_sampled: list[float] = []
    for name, (key, fraction_key, tolerance, units) in specs.items():
        continuous, sampled = _threshold_crossing(rows, key, tolerance)
        f_continuous, f_sampled = _threshold_crossing(
            rows,
            fraction_key,
            OVERLAP_FRACTION_TOLERANCE,
        )
        if continuous is not None:
            absolute_continuous.append(continuous)
        if sampled is not None:
            absolute_sampled.append(sampled)
        if f_continuous is not None:
            fraction_continuous.append(f_continuous)
        if f_sampled is not None:
            fraction_sampled.append(f_sampled)
        metrics[name] = {
            "absolute_tolerance": tolerance,
            "absolute_tolerance_continuous_nm": continuous,
            "absolute_tolerance_first_sampled_nm": sampled,
            "five_percent_continuous_nm": f_continuous,
            "five_percent_first_sampled_nm": f_sampled,
            "contact_overlap_magnitude": float(rows[0][key]),
            "units": units,
        }
    all_absolute = len(absolute_continuous) == len(specs)
    all_fraction = len(fraction_continuous) == len(specs)
    combined_absolute = max(absolute_continuous) if all_absolute else None
    combined_absolute_sampled = max(absolute_sampled) if all_absolute else None
    combined_fraction = max(fraction_continuous) if all_fraction else None
    combined_fraction_sampled = max(fraction_sampled) if all_fraction else None
    lambda_nm = float(rows[0]["lambda_D_nm"])
    return {
        "lambda_D_nm": lambda_nm,
        "absolute_tolerances": {
            "abs_delta_E_mix_mV": OVERLAP_E_TOLERANCE_MV,
            "delta_phi_RP_active_rms_mV": OVERLAP_PHI_TOLERANCE_MV,
            "abs_delta_i_mix_avg_percent_of_reference": OVERLAP_I_TOLERANCE_PERCENT,
        },
        "metrics": metrics,
        "combined_absolute_continuous_nm": combined_absolute,
        "combined_absolute_first_sampled_nm": combined_absolute_sampled,
        "combined_absolute_over_lambda_D": (
            None if combined_absolute is None else combined_absolute / lambda_nm
        ),
        "combined_absolute_L_GC_C_nm": (
            None if combined_absolute is None else _interpolate(rows, combined_absolute, "L_GC_C_nm")
        ),
        "combined_five_percent_continuous_nm": combined_fraction,
        "combined_five_percent_first_sampled_nm": combined_fraction_sampled,
        "combined_five_percent_over_lambda_D": (
            None if combined_fraction is None else combined_fraction / lambda_nm
        ),
        "combined_five_percent_L_GC_C_nm": (
            None if combined_fraction is None else _interpolate(rows, combined_fraction, "L_GC_C_nm")
        ),
        "reference": dict(reference_row),
        "definitions": {
            "delta_E_mix": "E_mix_with(L_support) - E_mix_with(1000 nm)",
            "delta_i_mix_avg": "i_mix_avg_with(L_support) - i_mix_avg_with(1000 nm)",
            "delta_phi_RP_active_rms": (
                "length-weighted RMS of Au/Pd material-mean differences from 1000 nm"
            ),
            "five_percent_boundary": (
                "all three overlap magnitudes are <=5% of their L_support=0 values"
            ),
        },
        "model_caveat": (
            "Operational overlap boundaries for this Au=Pd=2 nm linearized-PB "
            "parameter set, not universal transitions."
        ),
        "numerical_caveat": (
            "Material-mean reaction-plane potentials are used to avoid pointwise "
            "Fourier/Gibbs contamination at material boundaries."
        ),
    }


def _style_axes(ax: Any, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=7.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", top=False, right=False)


def _save_pair(fig: Any, stem: str) -> list[Path]:
    paths: list[Path] = []
    for suffix in ("png", "svg"):
        path = FIGURES_DIR / f"{stem}.{suffix}"
        fig.savefig(
            path,
            dpi=600,
            transparent=True,
            facecolor="none",
            edgecolor="none",
            bbox_inches="tight",
            pad_inches=0.04,
        )
        paths.append(path)
    plt.close(fig)
    return paths


def _marker_x(summary: Mapping[str, Any], value_nm: Any, x_key: str) -> float | None:
    if value_nm is None:
        return None
    value = float(value_nm)
    if x_key == "L_support_nm":
        return value
    if x_key == "L_support_over_lambda_D":
        return value / float(summary["lambda_D_nm"])
    raise ValueError(f"Unsupported x field: {x_key}")


def _plot_compare(
    rows: Sequence[Mapping[str, Any]],
    plateau: Mapping[str, Any],
    *,
    x_key: str,
    y_with_key: str,
    y_no_key: str,
    xlabel: str,
    ylabel: str,
    stem: str,
    show_markers: bool,
) -> list[Path]:
    x = np.asarray([row[x_key] for row in rows], dtype=float)
    y_with = np.asarray([row[y_with_key] for row in rows], dtype=float)
    y_no = np.asarray([row[y_no_key] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(4.35, 3.15), constrained_layout=False)
    ax.plot(x, y_with, color=WITH_COLOR, lw=2.0, marker="o", ms=3.0, label="with EDL")
    ax.plot(x, y_no, color=WITHOUT_COLOR, lw=1.8, marker="s", ms=2.8, label="w/o EDL")
    _style_axes(ax, xlabel, ylabel, r"$L_{\mathrm{support}}$ scan")
    ax.set_xlim(float(x[0]), float(x[-1]))
    ymin = min(float(np.min(y_with)), float(np.min(y_no)))
    ymax = max(float(np.max(y_with)), float(np.max(y_no)))
    pad = max(1e-8, 0.10 * (ymax - ymin))
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.legend(loc="center right", bbox_to_anchor=(0.985, 0.29), fontsize=7.7, handlelength=1.8)

    inset = ax.inset_axes([0.47, 0.52, 0.50, 0.39])
    inset.plot(x, y_with, color=WITH_COLOR, lw=1.45, marker="o", ms=2.1)
    detail_span = float(np.ptp(y_with))
    detail_pad = max(1e-8, 0.12 * detail_span)
    inset.set_xlim(float(x[0]), float(x[-1]))
    inset.set_ylim(float(np.min(y_with)) - detail_pad, float(np.max(y_with)) + detail_pad)
    inset.tick_params(length=2.3, width=0.7, pad=1.6, labelsize=6.2)
    inset.set_title("with EDL detail", loc="left", fontsize=6.8, pad=2.0)
    for spine in inset.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color(DARK)
    if show_markers:
        marker_specs = (
            (plateau["L_support_equals_L_GC_C_nm"], GRAY, (0, (2, 2)), r"$L=L_{\mathrm{GC},C}$"),
            (plateau["lambda_D_nm"], LAMBDA_COLOR, (0, (4, 2)), r"$\lambda_D$"),
            (
                plateau["plateau_first_sampled_nm"]
                if plateau["plateau_first_sampled_nm"] is not None
                else plateau["plateau_continuous_nm"],
                PLATEAU_COLOR,
                (0, (6, 2)),
                "plateau",
            ),
        )
        for value_nm, color, linestyle, label in marker_specs:
            xpos = _marker_x(plateau, value_nm, x_key)
            if xpos is None or not float(x[0]) <= xpos <= float(x[-1]):
                continue
            inset.axvline(xpos, color=color, lw=0.9, ls=linestyle, alpha=0.95)
            inset.text(
                xpos,
                0.96,
                label,
                transform=inset.get_xaxis_transform(),
                rotation=90,
                ha="right",
                va="top",
                fontsize=5.5,
                color=color,
            )
    fig.subplots_adjust(left=0.20, right=0.975, bottom=0.20, top=0.86)
    return _save_pair(fig, stem)


def _plot_support_charge(
    rows: Sequence[Mapping[str, Any]],
    plateau: Mapping[str, Any],
) -> list[Path]:
    defined = [row for row in rows if row["sigma_C_mean_abs_C_per_m2"] is not None]
    x = np.asarray([row["L_support_nm"] for row in defined], dtype=float)
    y = 100.0 * np.asarray([row["sigma_C_mean_abs_C_per_m2"] for row in defined], dtype=float)
    fig, ax = plt.subplots(figsize=(4.35, 3.15), constrained_layout=False)
    ax.plot(x, y, color=GRAY, lw=2.0, marker="o", ms=3.2, markeredgecolor=DARK, markeredgewidth=0.45)
    _style_axes(
        ax,
        r"$L_{\mathrm{support}}$ (nm)",
        r"$\langle|\sigma_C|\rangle$ ($\mu$C/cm$^2$)",
        r"$L_{\mathrm{support}}$ scan",
    )
    ax.set_xlim(0.0, 10.0)
    pad = max(1e-6, 0.10 * float(np.ptp(y)))
    ax.set_ylim(float(np.min(y)) - pad, float(np.max(y)) + pad)
    for xpos, color, label in (
        (plateau["L_support_equals_L_GC_C_nm"], GRAY, r"$L=L_{\mathrm{GC},C}$"),
        (plateau["lambda_D_nm"], LAMBDA_COLOR, r"$\lambda_D$"),
    ):
        if xpos is None or not 0.0 <= float(xpos) <= 10.0:
            continue
        ax.axvline(float(xpos), color=color, lw=1.0, ls=(0, (4, 2)))
        ax.text(float(xpos), 0.96, label, transform=ax.get_xaxis_transform(), rotation=90, ha="right", va="top", fontsize=7.0, color=color)
    ax.text(0.025, 0.055, "0 nm: N/A", transform=ax.transAxes, fontsize=7.2, color=DARK)
    fig.subplots_adjust(left=0.20, right=0.975, bottom=0.20, top=0.86)
    return _save_pair(
        fig,
        f"ofat_compare_L_support_mean_abs_sigma_C_0_10nm_{OUTPUT_TAG}",
    )


def _plot_lgc(
    rows: Sequence[Mapping[str, Any]],
    plateau: Mapping[str, Any],
) -> list[Path]:
    defined = [row for row in rows if row["L_GC_C_nm"] is not None]
    x = np.asarray([row["L_support_nm"] for row in defined], dtype=float)
    y = np.asarray([row["L_GC_C_nm"] for row in defined], dtype=float)
    fig, ax = plt.subplots(figsize=(4.35, 3.15), constrained_layout=False)
    ax.plot(x, y, color=GRAY, lw=2.0, marker="o", ms=3.2, markeredgecolor=DARK, markeredgewidth=0.45, label=r"$L_{\mathrm{GC},C}$")
    pad = max(1e-6, 0.10 * float(np.ptp(y)))
    ymin, ymax = float(np.min(y)) - pad, float(np.max(y)) + pad
    equality = np.linspace(max(0.0, ymin), min(10.0, ymax), 100)
    ax.plot(equality, equality, color=DARK, lw=1.2, ls=(0, (4, 2)), label=r"$L_{\mathrm{GC},C}=L_{\mathrm{support}}$")
    crossing = plateau["L_support_equals_L_GC_C_nm"]
    if crossing is not None:
        ax.scatter([crossing], [crossing], s=30, facecolor="white", edgecolor=PLATEAU_COLOR, linewidth=1.2, zorder=5)
        ax.annotate(
            rf"crossing = {float(crossing):.2f} nm",
            xy=(crossing, crossing),
            xytext=(min(8.2, float(crossing) + 1.6), ymin + 0.60 * (ymax - ymin)),
            fontsize=7.0,
            color=PLATEAU_COLOR,
            arrowprops={"arrowstyle": "->", "color": PLATEAU_COLOR, "lw": 0.8},
        )
    _style_axes(ax, r"$L_{\mathrm{support}}$ (nm)", r"$L_{\mathrm{GC},C}$ (nm)", r"$L_{\mathrm{support}}$ scan")
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(ymin, ymax)
    ax.text(0.12, ymin + 0.025 * (ymax - ymin), "0 nm: N/A", fontsize=7.2, color=DARK)
    ax.legend(loc="best", fontsize=7.0, handlelength=1.8)
    fig.subplots_adjust(left=0.20, right=0.975, bottom=0.20, top=0.86)
    return _save_pair(fig, f"ofat_compare_L_support_L_GC_C_0_10nm_{OUTPUT_TAG}")


def _positive_for_log(values: np.ndarray) -> np.ndarray:
    positive = values[values > 0.0]
    floor = max(1e-8, 0.2 * float(np.min(positive))) if positive.size else 1e-8
    return np.maximum(values, floor)


def _plot_overlap_metrics(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> list[Path]:
    x = np.asarray([row["L_support_nm"] for row in rows], dtype=float)
    specs = (
        ("overlap_abs_delta_E_mix_vs_1000_mV", r"$|\Delta E_{\mathrm{mix}}|$ (mV)", r"$E_{\mathrm{mix}}$ overlap", PLATEAU_COLOR, OVERLAP_E_TOLERANCE_MV),
        ("overlap_delta_phi_RP_active_rms_vs_1000_mV", r"$\Delta\phi_{\mathrm{RP,active}}^{\mathrm{RMS}}$ (mV)", "Reaction-plane overlap", LAMBDA_COLOR, OVERLAP_PHI_TOLERANCE_MV),
        ("overlap_abs_delta_i_mix_avg_vs_1000_percent", r"$|\Delta\bar{i}_{\mathrm{mix}}|/\bar{i}_{1000}$ (%)", "Current overlap", OVERLAP_I_COLOR, OVERLAP_I_TOLERANCE_PERCENT),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.55, 2.75), constrained_layout=False)
    strict_nm = summary["combined_absolute_continuous_nm"]
    for ax, (key, ylabel, title, color, tolerance) in zip(axes, specs, strict=True):
        y = _positive_for_log(np.asarray([row[key] for row in rows], dtype=float))
        ax.semilogy(x, y, color=color, lw=1.8, marker="o", ms=2.8, markeredgecolor=DARK, markeredgewidth=0.35)
        ax.axhline(tolerance, color=GRAY, lw=0.9, ls=(0, (3, 2)))
        ax.axvline(float(summary["lambda_D_nm"]), color=LAMBDA_COLOR, lw=0.85, ls=(0, (4, 2)))
        if strict_nm is not None:
            ax.axvline(float(strict_nm), color=PLATEAU_COLOR, lw=1.0, ls=(0, (6, 2)))
        _style_axes(ax, r"$L_{\mathrm{support}}$ (nm)", ylabel, title)
        ax.set_xlim(0.0, 15.0)
        ax.set_xticks([0.0, 3.0, 6.0, 9.0, 12.0, 15.0])
        ax.set_ylim(max(1e-5, 0.55 * min(float(np.min(y)), tolerance)), 1.45 * max(float(np.max(y)), tolerance))
    axes[0].text(float(summary["lambda_D_nm"]), 0.96, r"$\lambda_D$", transform=axes[0].get_xaxis_transform(), rotation=90, ha="right", va="top", fontsize=6.6, color=LAMBDA_COLOR)
    if strict_nm is not None:
        axes[2].text(float(strict_nm), 0.96, rf"strict = {float(strict_nm):.1f} nm", transform=axes[2].get_xaxis_transform(), rotation=90, ha="right", va="top", fontsize=6.6, color=PLATEAU_COLOR)
    axes[0].text(0.97, 0.07, "dashed: tolerance", transform=axes[0].transAxes, ha="right", fontsize=6.4, color=GRAY)
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.22, top=0.83, wspace=0.43)
    return _save_pair(fig, f"ofat_compare_L_support_overlap_metrics_vs_1000nm_{OUTPUT_TAG}")


def _plot_overlap_fraction(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> list[Path]:
    x = np.asarray([row["L_support_nm"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(4.6, 3.2), constrained_layout=False)
    specs = (
        ("overlap_fraction_E_mix", r"$E_{\mathrm{mix}}$", PLATEAU_COLOR, "o"),
        ("overlap_fraction_phi_RP_active_rms", r"$\phi_{\mathrm{RP,active}}$ RMS", LAMBDA_COLOR, "s"),
        ("overlap_fraction_i_mix_avg", r"$\bar{i}_{\mathrm{mix}}$", OVERLAP_I_COLOR, "^"),
    )
    for key, label, color, marker in specs:
        y = _positive_for_log(np.asarray([row[key] for row in rows], dtype=float))
        ax.semilogy(x, y, color=color, lw=1.75, marker=marker, ms=3.1, markeredgecolor=DARK, markeredgewidth=0.35, label=label)
    boundary = summary["combined_five_percent_continuous_nm"]
    ax.axhline(OVERLAP_FRACTION_TOLERANCE, color=GRAY, lw=0.95, ls=(0, (3, 2)))
    ax.axvline(float(summary["lambda_D_nm"]), color=LAMBDA_COLOR, lw=0.9, ls=(0, (4, 2)))
    if boundary is not None:
        ax.axvline(float(boundary), color=PLATEAU_COLOR, lw=1.0, ls=(0, (6, 2)))
    _style_axes(ax, r"$L_{\mathrm{support}}$ (nm)", "Residual overlap fraction", "EDL overlap relative to 1000 nm")
    ax.set_xlim(0.0, 15.0)
    ax.set_xticks([0.0, 3.0, 6.0, 9.0, 12.0, 15.0])
    ax.set_ylim(1e-3, 1.35)
    ax.text(0.15, OVERLAP_FRACTION_TOLERANCE * 1.08, "5%", fontsize=7.0, color=GRAY)
    if boundary is not None:
        ax.text(float(boundary), 0.96, rf"95% decayed = {float(boundary):.1f} nm", transform=ax.get_xaxis_transform(), rotation=90, ha="right", va="top", fontsize=6.8, color=PLATEAU_COLOR)
    ax.legend(loc="upper right", fontsize=7.0, handlelength=1.8)
    fig.subplots_adjust(left=0.18, right=0.975, bottom=0.20, top=0.86)
    return _save_pair(fig, f"ofat_compare_L_support_overlap_fraction_vs_1000nm_{OUTPUT_TAG}")


def _source_hashes() -> dict[str, str]:
    paths = (Path(__file__).resolve(), Path(engine.__file__).resolve())
    return {
        str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _save_inputs() -> list[Path]:
    short_params = engine.params_for_support_scan(0.0, n_modes=engine.SHORT_N_MODES)
    reference_params = engine.params_for_support_scan(
        REFERENCE_SUPPORT_NM,
        n_modes=engine.LONG_N_MODES,
    )
    config = {
        "model": "Au|C|Pd linear-PB support-length OFAT",
        "main_support_nm": MAIN_SUPPORT_NM,
        "overlap_extra_support_nm": OVERLAP_EXTRA_SUPPORT_NM,
        "reference_support_nm": REFERENCE_SUPPORT_NM,
        "physical_overrides": {
            "L_Au": engine.L_AU_M,
            "L_Pd_len": engine.L_PD_M,
            "C_H_Au_F_per_m2": engine.C_H_AU_F_PER_M2,
            "C_H_support_F_per_m2": engine.C_H_SUPPORT_F_PER_M2,
            "C_H_Pd_F_per_m2": engine.C_H_PD_F_PER_M2,
            "it0_1": engine.I0_EQUAL_A_PER_M2,
            "it0_2": engine.I0_EQUAL_A_PER_M2,
            "alpha1": engine.ALPHA_EQUAL,
            "alpha2": engine.ALPHA_EQUAL,
            "C_tot_mol_per_m3": 10.0,
        },
        "numerical": {
            "short_N_modes": engine.SHORT_N_MODES,
            "reference_N_modes": engine.LONG_N_MODES,
            "gauss_legendre_order_per_active_segment": engine.DEFAULT_GL_ORDER,
            "include_2d": False,
            "balance": "absolute current I_Au + I_Pd = 0",
        },
        "source_sha256": _source_hashes(),
    }
    return [
        _write_json(INPUTS_DIR / f"scan_config_Figure_L_support_{OUTPUT_TAG}.json", config),
        _write_json(INPUTS_DIR / f"params_short_scan_{OUTPUT_TAG}.json", short_params),
        _write_json(INPUTS_DIR / f"params_reference_1000nm_{OUTPUT_TAG}.json", reference_params),
    ]


def _validate_outputs(
    figures: Sequence[Path],
    rows: Sequence[Mapping[str, Any]],
    reference_row: Mapping[str, Any],
) -> dict[str, Any]:
    pngs = [path for path in figures if path.suffix.lower() == ".png"]
    svgs = [path for path in figures if path.suffix.lower() == ".svg"]
    if len(pngs) != 8 or len(svgs) != 8:
        raise RuntimeError(f"Expected 8 PNG + 8 SVG, got {len(pngs)} + {len(svgs)}")
    actual_pngs = set(FIGURES_DIR.glob("*.png"))
    actual_svgs = set(FIGURES_DIR.glob("*.svg"))
    expected_pngs = set(pngs)
    expected_svgs = set(svgs)
    if actual_pngs != expected_pngs or actual_svgs != expected_svgs:
        unexpected = sorted((actual_pngs - expected_pngs) | (actual_svgs - expected_svgs))
        absent = sorted((expected_pngs - actual_pngs) | (expected_svgs - actual_svgs))
        raise RuntimeError(
            "OFAT directory must contain exactly the generated 8 PNG + 8 SVG; "
            f"unexpected={unexpected}, absent={absent}"
        )
    missing = [path for path in figures if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Empty/missing OFAT figures: {missing}")
    noneditable = [path for path in svgs if "<text" not in path.read_text(encoding="utf-8")]
    if noneditable:
        raise RuntimeError(f"SVG text is not editable: {noneditable}")
    pdfs = sorted(OUT_DIR.rglob("*.pdf"))
    if pdfs:
        raise RuntimeError(f"Unexpected PDF outputs: {pdfs}")
    all_rows = list(rows) + [reference_row]
    max_balance = max(
        max(float(row["relative_balance_with"]), float(row["relative_balance_no"]))
        for row in all_rows
    )
    if max_balance >= engine.RELATIVE_BALANCE_TOL:
        raise RuntimeError(f"Current balance residual is too large: {max_balance:.3g}")
    return {
        "passed": True,
        "png_count": len(pngs),
        "svg_count": len(svgs),
        "pdf_count": 0,
        "svg_text_editable": True,
        "main_scan_points": len(MAIN_SUPPORT_NM),
        "overlap_scan_points": len(OVERLAP_SUPPORT_NM),
        "reference_support_nm": REFERENCE_SUPPORT_NM,
        "max_relative_current_balance_residual": max_balance,
        "debye_huckel_caveat": (
            "Linearized-PB applicability warnings are retained in the scan CSV "
            "through max_abs_phi_tilde_with_edl."
        ),
        "fourier_caveat": (
            "Material-mean overlap diagnostics are used near interfaces to reduce "
            "sensitivity to Fourier/Gibbs ringing."
        ),
    }


def build_figure_l_support_ofat(base_cases: Sequence[Any] | None = None) -> dict[str, Any]:
    """Calculate, export, and validate the complete support-length OFAT set.

    ``base_cases`` may contain already-computed formal cases.  Matching points,
    especially the N=7680 1000 nm reference, are reused after strict parameter,
    mode-count, quadrature, and current-balance validation.
    """

    for directory in (FIGURES_DIR, CSV_DIR, INPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    configure_style()
    cases = _get_or_build_cases(base_cases)
    overlap_rows, reference_row = _build_rows(cases)
    main_rows = [row for row in overlap_rows if float(row["L_support_nm"]) <= 10.0]
    plateau = _plateau_summary(main_rows)
    overlap = _overlap_summary(overlap_rows, reference_row)

    data_paths: list[Path] = []
    data_paths.extend(_save_inputs())
    data_paths.append(
        _write_csv(
            CSV_DIR / f"ofat_compare_L_support_dense_0_10nm_{OUTPUT_TAG}.csv",
            main_rows,
        )
    )
    data_paths.append(
        _write_csv(
            CSV_DIR / f"edl_overlap_vs_1000nm_L_support_{OUTPUT_TAG}.csv",
            overlap_rows,
        )
    )
    data_paths.append(
        _write_json(
            CSV_DIR / f"reference_1000nm_L_support_{OUTPUT_TAG}.json",
            reference_row,
        )
    )
    data_paths.append(
        _write_csv(
            CSV_DIR / f"plateau_length_scale_summary_{OUTPUT_TAG}.csv",
            [{key: value for key, value in plateau.items() if key != "fit_covariance"}],
        )
    )
    data_paths.append(
        _write_json(CSV_DIR / f"plateau_length_scale_summary_{OUTPUT_TAG}.json", plateau)
    )
    data_paths.append(
        _write_csv(
            CSV_DIR / f"edl_overlap_summary_vs_1000nm_L_support_{OUTPUT_TAG}.csv",
            [{
                "combined_absolute_continuous_nm": overlap["combined_absolute_continuous_nm"],
                "combined_absolute_first_sampled_nm": overlap["combined_absolute_first_sampled_nm"],
                "combined_five_percent_continuous_nm": overlap["combined_five_percent_continuous_nm"],
                "combined_five_percent_first_sampled_nm": overlap["combined_five_percent_first_sampled_nm"],
                "lambda_D_nm": overlap["lambda_D_nm"],
            }],
        )
    )
    data_paths.append(
        _write_json(
            CSV_DIR / f"edl_overlap_summary_vs_1000nm_L_support_{OUTPUT_TAG}.json",
            overlap,
        )
    )

    figures: list[Path] = []
    figures.extend(
        _plot_compare(
            main_rows,
            plateau,
            x_key="L_support_nm",
            y_with_key="E_mix_with_V",
            y_no_key="E_mix_no_V",
            xlabel=r"$L_{\mathrm{support}}$ (nm)",
            ylabel=r"$E_{\mathrm{mix}}$ (V vs. RHE)",
            stem=f"ofat_compare_L_support_E_mix_0_10nm_{OUTPUT_TAG}",
            show_markers=True,
        )
    )
    figures.extend(_plot_support_charge(main_rows, plateau))
    figures.extend(_plot_lgc(main_rows, plateau))
    figures.extend(_plot_overlap_metrics(overlap_rows, overlap))
    figures.extend(_plot_overlap_fraction(overlap_rows, overlap))
    figures.extend(
        _plot_compare(
            main_rows,
            plateau,
            x_key="L_support_nm",
            y_with_key="i_mix_avg_with_A_per_m2",
            y_no_key="i_mix_avg_no_A_per_m2",
            xlabel=r"$L_{\mathrm{support}}$ (nm)",
            ylabel=r"Mixed current density, $\bar{i}_{\mathrm{mix}}$ (A/m$^2$)",
            stem=f"ofat_compare_L_support_i_mix_avg_0_10nm_{OUTPUT_TAG}",
            show_markers=False,
        )
    )
    figures.extend(
        _plot_compare(
            main_rows,
            plateau,
            x_key="L_support_over_lambda_D",
            y_with_key="E_mix_with_V",
            y_no_key="E_mix_no_V",
            xlabel=r"$L_{\mathrm{support}}/\lambda_D$",
            ylabel=r"$E_{\mathrm{mix}}$ (V vs. RHE)",
            stem=f"ofat_compare_L_support_over_lambda_D_E_mix_0_10nm_{OUTPUT_TAG}",
            show_markers=True,
        )
    )
    figures.extend(
        _plot_compare(
            main_rows,
            plateau,
            x_key="L_support_over_lambda_D",
            y_with_key="i_mix_avg_with_A_per_m2",
            y_no_key="i_mix_avg_no_A_per_m2",
            xlabel=r"$L_{\mathrm{support}}/\lambda_D$",
            ylabel=r"Mixed current density, $\bar{i}_{\mathrm{mix}}$ (A/m$^2$)",
            stem=f"ofat_compare_L_support_over_lambda_D_i_mix_avg_0_10nm_{OUTPUT_TAG}",
            show_markers=False,
        )
    )

    validation = _validate_outputs(figures, overlap_rows, reference_row)
    validation["plateau"] = plateau
    validation["overlap"] = overlap
    validation_path = _write_json(OUT_DIR / "validation.json", validation)
    data_paths.append(validation_path)
    manifest = {
        "output_tag": OUTPUT_TAG,
        "figures": [str(path.relative_to(OUT_DIR)) for path in figures],
        "data": [str(path.relative_to(OUT_DIR)) for path in data_paths],
        "validation": validation,
    }
    manifest_path = _write_json(OUT_DIR / "manifest.json", manifest)
    data_paths.append(manifest_path)
    return {
        "figure_paths": figures,
        "data_paths": data_paths,
        "validation": validation,
        "rows": overlap_rows,
        "reference_row": reference_row,
    }


def main() -> None:
    result = build_figure_l_support_ofat()
    validation = result["validation"]
    print(
        f"Generated {validation['png_count']} PNG + {validation['svg_count']} SVG "
        f"in {FIGURES_DIR}"
    )
    print(
        "Max relative current-balance residual = "
        f"{validation['max_relative_current_balance_residual']:.3g}"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "MAIN_SUPPORT_NM",
    "OVERLAP_SUPPORT_NM",
    "REFERENCE_SUPPORT_NM",
    "build_figure_l_support_ofat",
]
