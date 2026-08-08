from __future__ import annotations

import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = ROOT / "Figures" / "Figure_same_length_i0_alpha"
OUT_DIR = Path(__file__).resolve().parent
CSV_DIR = OUT_DIR / "csv"
INPUTS_DIR = OUT_DIR / "inputs"

sys.path.insert(0, str(COMMON_DIR))
import same_length_i0_alpha_common as common  # noqa: E402


solver = common.solver
OUTPUT_TAG = common.OUTPUT_TAG

PNG_OUT = OUT_DIR / f"support_ch_effect_{OUTPUT_TAG}.png"
SVG_OUT = OUT_DIR / f"support_ch_effect_{OUTPUT_TAG}.svg"
SWEEP_CSV_OUT = CSV_DIR / f"support_ch_sweep_{OUTPUT_TAG}.csv"
PROFILE_CSV_OUT = CSV_DIR / f"support_ch_phi_rp_profiles_{OUTPUT_TAG}.csv"

PARAMS_OUT = INPUTS_DIR / f"params_{OUTPUT_TAG}.json"
OVERRIDES_OUT = INPUTS_DIR / f"overrides_{OUTPUT_TAG}.json"
SUMMARY_CSV_OUT = INPUTS_DIR / f"summary_compare_{OUTPUT_TAG}.csv"
SUMMARY_JSON_OUT = INPUTS_DIR / f"summary_compare_{OUTPUT_TAG}.json"
CONFIG_OUT = INPUTS_DIR / f"support_ch_scan_config_{OUTPUT_TAG}.json"
STUDY_SUMMARY_OUT = INPUTS_DIR / f"support_ch_study_summary_{OUTPUT_TAG}.json"

# The upper bound matches the existing Figure 5 C_H scans. A linear axis is
# required because C_H,support = 0 is a physically meaningful endpoint here.
SUPPORT_CH_MIN_F_PER_M2 = 0.0
SUPPORT_CH_MAX_F_PER_M2 = 1.0
N_SWEEP = 41
BASELINE_SUPPORT_CH_F_PER_M2 = 0.10
REPRESENTATIVE_SUPPORT_CH_F_PER_M2 = (0.0, BASELINE_SUPPORT_CH_F_PER_M2, 1.0)

# Use the converged Figure 3 resolution because the support affects the active
# materials only through lateral electrostatic coupling at piecewise boundaries.
N_MODES = 960
NX = 5000

COLORS = {
    "with_edl": "#F26B38",
    "no_edl": "#12355B",
    "au": "#B64342",
    "support": "#767676",
    "pd": "#3775BA",
    "case_zero": "#767676",
    "case_baseline": "#F26B38",
    "case_high": "#B64342",
    "baseline": "#272727",
    "boundary": "#8C8C8C",
    "au_fill": "#F6CFCB",
    "support_fill": "#E7E7E7",
    "pd_fill": "#DCEAF5",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 10.8,
            "axes.titlesize": 12.2,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 9.3,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Nimbus Sans",
            "mathtext.it": "Nimbus Sans:italic",
            "mathtext.bf": "Nimbus Sans:bold",
            "mathtext.cal": "Nimbus Sans",
            "mathtext.sf": "Nimbus Sans",
            "mathtext.tt": "Nimbus Sans",
        }
    )


def ensure_dirs() -> None:
    for path in (OUT_DIR, CSV_DIR, INPUTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def trapz(values: np.ndarray, coordinates: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, coordinates))
    return float(np.trapz(values, coordinates))


def segment_mean(
    values: np.ndarray,
    x_tilde: np.ndarray,
    mask: np.ndarray,
    segment_length_tilde: float,
) -> float:
    selected_values = np.asarray(values, dtype=float)[mask]
    selected_x = np.asarray(x_tilde, dtype=float)[mask]
    if selected_values.size < 2 or segment_length_tilde <= 0.0:
        raise ValueError("A non-zero segment with at least two grid points is required")
    return trapz(selected_values, selected_x) / float(segment_length_tilde)


def scan_values() -> np.ndarray:
    values = np.linspace(SUPPORT_CH_MIN_F_PER_M2, SUPPORT_CH_MAX_F_PER_M2, N_SWEEP)
    baseline_index = int(np.argmin(np.abs(values - BASELINE_SUPPORT_CH_F_PER_M2)))
    if abs(float(values[baseline_index]) - BASELINE_SUPPORT_CH_F_PER_M2) > 1.0e-12:
        raise ValueError("The support C_H scan must contain the baseline exactly")
    values[baseline_index] = BASELINE_SUPPORT_CH_F_PER_M2
    return values


def configure_params() -> dict[str, Any]:
    params = common.load_same_length_i0_alpha_params()
    common.validate_same_length_i0_alpha_params(params)
    if params.get("g_C") is not None:
        raise ValueError("g_C must remain None so Cdl_C controls support C_H")
    if not math.isclose(
        float(params["Cdl_C"]),
        BASELINE_SUPPORT_CH_F_PER_M2,
        rel_tol=0.0,
        abs_tol=1.0e-14,
    ):
        raise ValueError("Unexpected support C_H baseline")

    params["N_modes"] = N_MODES
    params["Nx"] = NX
    # The known violation is retained in every result row and in the figure
    # footnote; suppress only repetitive terminal warnings during the batch.
    params["dh_violation_action"] = "ignore"
    solver.validate_params(params)
    return params


def support_mask(x_tilde: np.ndarray, derived: dict[str, Any]) -> np.ndarray:
    left = float(derived["L_Au_tilde"])
    right = float(derived["L_C_tilde"])
    tolerance = 1.0e-12
    return (x_tilde >= left - tolerance) & (x_tilde <= right + tolerance)


def material_labels(x_tilde: np.ndarray, derived: dict[str, Any]) -> np.ndarray:
    left = float(derived["L_Au_tilde"])
    right = float(derived["L_C_tilde"])
    labels = np.full(x_tilde.shape, "support", dtype=object)
    labels[x_tilde < left] = "Au"
    labels[x_tilde > right] = "Pd"
    labels[np.isclose(x_tilde, left, rtol=0.0, atol=1.0e-12)] = "Au/support boundary"
    labels[np.isclose(x_tilde, right, rtol=0.0, atol=1.0e-12)] = "support/Pd boundary"
    return labels


def summarize_support_case(
    base_params: dict[str, Any],
    c_h_support: float,
    index: int,
    no_edl: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    params = copy.deepcopy(base_params)
    params["Cdl_C"] = float(c_h_support)
    result = solver.run_case(params, mode="FULL", return_profiles=True, use_edl=True)
    derived = solver.compute_derived_params(params)

    x_tilde = np.asarray(result["x_tilde"], dtype=float)
    phi_tilde = np.asarray(result["phi_tilde"], dtype=float)
    thermal_voltage = float(derived["R"]) * float(derived["T"]) / float(derived["F"])
    phi_rp_v = thermal_voltage * phi_tilde
    mask_au = np.asarray(result["mask_Au"], dtype=bool)
    mask_pd = np.asarray(result["mask_Pd"], dtype=bool)
    mask_support = support_mask(x_tilde, derived)

    length_au = float(derived["L_Au_tilde"])
    length_support = float(derived["L_C_tilde"] - derived["L_Au_tilde"])
    length_pd = float(derived["L_tilde"] - derived["L_C_tilde"])
    mean_phi_au_v = segment_mean(phi_rp_v, x_tilde, mask_au, length_au)
    mean_phi_support_v = segment_mean(phi_rp_v, x_tilde, mask_support, length_support)
    mean_phi_pd_v = segment_mean(phi_rp_v, x_tilde, mask_pd, length_pd)

    e_mix = float(result["E_mix"])
    support_compact_voltage_v = e_mix - phi_rp_v - float(params["pzc_C"])
    mean_support_compact_voltage_v = segment_mean(
        support_compact_voltage_v,
        x_tilde,
        mask_support,
        length_support,
    )
    support_sigma_c_per_m2 = float(c_h_support) * support_compact_voltage_v
    mean_support_sigma_c_per_m2 = segment_mean(
        support_sigma_c_per_m2,
        x_tilde,
        mask_support,
        length_support,
    )

    diffuse_capacitance_f_per_m2 = float(derived["epsilon_s"]) / float(derived["lambda_D"])
    no_e_mix = float(no_edl["E_mix"])
    no_i_mix_avg = float(no_edl["i_mix_avg_A_per_m2"])
    is_baseline = math.isclose(
        float(c_h_support),
        BASELINE_SUPPORT_CH_F_PER_M2,
        rel_tol=0.0,
        abs_tol=1.0e-14,
    )
    is_representative = any(
        math.isclose(float(c_h_support), value, rel_tol=0.0, abs_tol=1.0e-14)
        for value in REPRESENTATIVE_SUPPORT_CH_F_PER_M2
    )

    row = {
        "scan_index": int(index),
        "is_baseline": bool(is_baseline),
        "is_representative": bool(is_representative),
        # Cdl_C is retained only as the solver-compatible storage name.
        "Cdl_C_F_per_m2": float(c_h_support),
        "C_H_support_uF_per_cm2": 100.0 * float(c_h_support),
        "g_C": float(derived["g_C"]),
        "C_D_F_per_m2": diffuse_capacitance_f_per_m2,
        "E_mix_with_EDL_V": e_mix,
        "E_mix_without_EDL_V": no_e_mix,
        "EDL_shift_E_mix_mV": 1000.0 * (e_mix - no_e_mix),
        "delta_E_mix_from_C_H_zero_mV": float("nan"),
        "i_mix_avg_with_EDL_A_per_m2": float(result["i_mix_avg_A_per_m2"]),
        "i_mix_avg_without_EDL_A_per_m2": no_i_mix_avg,
        "ratio_i_mix_with_to_without": float(result["i_mix_avg_A_per_m2"]) / no_i_mix_avg,
        "pct_i_mix_with_vs_without": 100.0 * (float(result["i_mix_avg_A_per_m2"]) / no_i_mix_avg - 1.0),
        "pct_i_mix_from_C_H_zero": float("nan"),
        "i_mix_abs_with_EDL_A": float(result["i_mix_abs_A"]),
        "i_mix_abs_without_EDL_A": float(no_edl["i_mix_abs_A"]),
        "I_Au_abs_A": float(result["I_Au_abs_A"]),
        "I_Pd_abs_A": float(result["I_Pd_abs_A"]),
        "current_balance_residual_abs_A": float(result["residual_abs_A"]),
        "mean_phi_RP_Au_V": mean_phi_au_v,
        "mean_phi_RP_Au_mV": 1000.0 * mean_phi_au_v,
        "mean_phi_RP_support_V": mean_phi_support_v,
        "mean_phi_RP_support_mV": 1000.0 * mean_phi_support_v,
        "mean_phi_RP_Pd_V": mean_phi_pd_v,
        "mean_phi_RP_Pd_mV": 1000.0 * mean_phi_pd_v,
        "mean_support_compact_voltage_V": mean_support_compact_voltage_v,
        "mean_support_sigma_C_per_m2": mean_support_sigma_c_per_m2,
        "mean_support_sigma_uC_per_cm2": 100.0 * mean_support_sigma_c_per_m2,
        "max_abs_phi_tilde": float(result["max_abs_phi_tilde"]),
        "debye_huckel_ok": bool(result["debye_huckel_ok"]),
        "N_modes": int(params["N_modes"]),
        "Nx": int(params["Nx"]),
    }

    profile = {
        "Cdl_C_F_per_m2": float(c_h_support),
        "C_H_support_uF_per_cm2": 100.0 * float(c_h_support),
        "E_mix_with_EDL_V": e_mix,
        "x_tilde": x_tilde,
        "x_nm": x_tilde * float(derived["lambda_D"]) * 1.0e9,
        "material": material_labels(x_tilde, derived),
        "phi_tilde": phi_tilde,
        "phi_RP_V": phi_rp_v,
        "phi_RP_mV": 1000.0 * phi_rp_v,
        "i_Au_local_A_per_m2": np.asarray(result["i1"], dtype=float),
        "i_Pd_local_A_per_m2": np.asarray(result["i2"], dtype=float),
        "local_current_density_A_per_m2": np.asarray(result["i1"], dtype=float)
        + np.asarray(result["i2"], dtype=float),
        "support_mask": mask_support,
        "support_compact_voltage_V": support_compact_voltage_v,
        "support_sigma_C_per_m2": support_sigma_c_per_m2,
        "derived": derived,
    }
    return row, profile


def run_support_ch_scan(
    params: dict[str, Any],
    no_edl: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[float, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    representative_profiles: dict[float, dict[str, Any]] = {}

    for index, value in enumerate(scan_values()):
        print(
            f"[{index + 1:02d}/{N_SWEEP}] "
            f"C_H,support = {100.0 * float(value):.1f} uF/cm^2",
            flush=True,
        )
        row, profile = summarize_support_case(params, float(value), index, no_edl)
        rows.append(row)
        for representative_value in REPRESENTATIVE_SUPPORT_CH_F_PER_M2:
            if math.isclose(float(value), representative_value, rel_tol=0.0, abs_tol=1.0e-14):
                representative_profiles[representative_value] = profile

    zero_e_mix = float(rows[0]["E_mix_with_EDL_V"])
    zero_i_mix = float(rows[0]["i_mix_avg_with_EDL_A_per_m2"])
    for row in rows:
        row["delta_E_mix_from_C_H_zero_mV"] = 1000.0 * (
            float(row["E_mix_with_EDL_V"]) - zero_e_mix
        )
        row["pct_i_mix_from_C_H_zero"] = 100.0 * (
            float(row["i_mix_avg_with_EDL_A_per_m2"]) / zero_i_mix - 1.0
        )

    if set(representative_profiles) != set(REPRESENTATIVE_SUPPORT_CH_F_PER_M2):
        raise RuntimeError("Not all representative support C_H profiles were captured")
    return rows, representative_profiles


SWEEP_FIELDS = [
    "scan_index",
    "is_baseline",
    "is_representative",
    "Cdl_C_F_per_m2",
    "C_H_support_uF_per_cm2",
    "g_C",
    "C_D_F_per_m2",
    "E_mix_with_EDL_V",
    "E_mix_without_EDL_V",
    "EDL_shift_E_mix_mV",
    "delta_E_mix_from_C_H_zero_mV",
    "i_mix_avg_with_EDL_A_per_m2",
    "i_mix_avg_without_EDL_A_per_m2",
    "ratio_i_mix_with_to_without",
    "pct_i_mix_with_vs_without",
    "pct_i_mix_from_C_H_zero",
    "i_mix_abs_with_EDL_A",
    "i_mix_abs_without_EDL_A",
    "I_Au_abs_A",
    "I_Pd_abs_A",
    "current_balance_residual_abs_A",
    "mean_phi_RP_Au_V",
    "mean_phi_RP_Au_mV",
    "mean_phi_RP_support_V",
    "mean_phi_RP_support_mV",
    "mean_phi_RP_Pd_V",
    "mean_phi_RP_Pd_mV",
    "mean_support_compact_voltage_V",
    "mean_support_sigma_C_per_m2",
    "mean_support_sigma_uC_per_cm2",
    "max_abs_phi_tilde",
    "debye_huckel_ok",
    "N_modes",
    "Nx",
]


def profile_rows(profiles: dict[float, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c_h_support in REPRESENTATIVE_SUPPORT_CH_F_PER_M2:
        profile = profiles[c_h_support]
        mask_support = np.asarray(profile["support_mask"], dtype=bool)
        point_count = len(profile["x_nm"])
        for index in range(point_count):
            rows.append(
                {
                    "Cdl_C_F_per_m2": float(profile["Cdl_C_F_per_m2"]),
                    "C_H_support_uF_per_cm2": float(profile["C_H_support_uF_per_cm2"]),
                    "E_mix_with_EDL_V": float(profile["E_mix_with_EDL_V"]),
                    "x_nm": float(profile["x_nm"][index]),
                    "material": str(profile["material"][index]),
                    "is_support_mask": bool(mask_support[index]),
                    "phi_tilde": float(profile["phi_tilde"][index]),
                    "phi_RP_V": float(profile["phi_RP_V"][index]),
                    "phi_RP_mV": float(profile["phi_RP_mV"][index]),
                    "i_Au_local_A_per_m2": float(profile["i_Au_local_A_per_m2"][index]),
                    "i_Pd_local_A_per_m2": float(profile["i_Pd_local_A_per_m2"][index]),
                    "local_current_density_A_per_m2": float(
                        profile["local_current_density_A_per_m2"][index]
                    ),
                    "support_compact_voltage_V": (
                        float(profile["support_compact_voltage_V"][index]) if mask_support[index] else ""
                    ),
                    "support_sigma_C_per_m2": (
                        float(profile["support_sigma_C_per_m2"][index]) if mask_support[index] else ""
                    ),
                }
            )
    return rows


PROFILE_FIELDS = [
    "Cdl_C_F_per_m2",
    "C_H_support_uF_per_cm2",
    "E_mix_with_EDL_V",
    "x_nm",
    "material",
    "is_support_mask",
    "phi_tilde",
    "phi_RP_V",
    "phi_RP_mV",
    "i_Au_local_A_per_m2",
    "i_Pd_local_A_per_m2",
    "local_current_density_A_per_m2",
    "support_compact_voltage_V",
    "support_sigma_C_per_m2",
]


def baseline_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [row for row in rows if bool(row["is_baseline"])]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one baseline scan row, found {len(matches)}")
    return matches[0]


def validate_results(
    params: dict[str, Any],
    rows: list[dict[str, Any]],
    profiles: dict[float, dict[str, Any]],
    no_edl: dict[str, Any],
    high_resolution_summary: dict[str, str],
) -> dict[str, Any]:
    if len(rows) != N_SWEEP:
        raise RuntimeError(f"Expected {N_SWEEP} scan rows, found {len(rows)}")

    finite_fields = [
        field
        for field in SWEEP_FIELDS
        if field not in {"is_baseline", "is_representative", "debye_huckel_ok"}
    ]
    for field in finite_fields:
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite values found in {field}")

    e_mix = np.asarray([float(row["E_mix_with_EDL_V"]) for row in rows], dtype=float)
    i_mix = np.asarray([float(row["i_mix_avg_with_EDL_A_per_m2"]) for row in rows], dtype=float)
    max_phi = np.asarray([float(row["max_abs_phi_tilde"]) for row in rows], dtype=float)
    i_au_abs_a = np.asarray([float(row["I_Au_abs_A"]) for row in rows], dtype=float)
    i_pd_abs_a = np.asarray([float(row["I_Pd_abs_A"]) for row in rows], dtype=float)
    residual_abs_a = np.asarray([float(row["current_balance_residual_abs_A"]) for row in rows], dtype=float)
    i_mix_abs_a = np.asarray([float(row["i_mix_abs_with_EDL_A"]) for row in rows], dtype=float)
    if not np.all(np.diff(e_mix) < 0.0):
        raise ValueError("E_mix should decrease monotonically over this support C_H scan")
    if not np.all(np.diff(i_mix) > 0.0):
        raise ValueError("i_mix should increase monotonically over this support C_H scan")

    # Mixed potential is defined by total absolute-current balance. The solver's
    # *_abs_A keys mean currents converted to amperes using the stated width;
    # I_Pd_abs_A intentionally retains its cathodic (negative) sign.
    if not np.allclose(i_au_abs_a + i_pd_abs_a, residual_abs_a, rtol=0.0, atol=1.0e-24):
        raise ValueError("Stored absolute-current residual is inconsistent with I_Au + I_Pd")
    relative_balance_residual = np.abs(residual_abs_a) / np.maximum(i_mix_abs_a, np.finfo(float).tiny)
    if float(np.max(relative_balance_residual)) > 1.0e-7:
        raise ValueError("Absolute-current balance residual is too large")
    if not np.allclose(i_mix_abs_a, np.abs(i_au_abs_a), rtol=1.0e-12, atol=1.0e-24):
        raise ValueError("i_mix_abs must equal |I_Au| at mixed potential")
    reactive_area_m2 = (
        float(params["out_of_plane_width"])
        * (float(params["L_Au"]) + float(params["L_Pd_len"]))
    )
    if not np.allclose(i_mix, i_mix_abs_a / reactive_area_m2, rtol=1.0e-12, atol=1.0e-14):
        raise ValueError("Mixed current density does not use the common total reactive area")

    baseline = baseline_row(rows)
    if abs(float(baseline["E_mix_with_EDL_V"]) - float(high_resolution_summary["E_mix_with"])) > 5.0e-11:
        raise ValueError("Baseline E_mix in scan does not match the stored high-resolution summary")
    if (
        abs(
            float(baseline["i_mix_avg_with_EDL_A_per_m2"])
            - float(high_resolution_summary["i_mix_avg_with"])
        )
        > 5.0e-11
    ):
        raise ValueError("Baseline i_mix in scan does not match the stored high-resolution summary")

    # Directly verify the analytic expectation that w/o EDL ignores C_H.
    for endpoint in (SUPPORT_CH_MIN_F_PER_M2, SUPPORT_CH_MAX_F_PER_M2):
        endpoint_params = copy.deepcopy(params)
        endpoint_params["Cdl_C"] = endpoint
        endpoint_no_edl = solver.run_case(
            endpoint_params,
            mode="FULL",
            return_profiles=False,
            use_edl=False,
        )
        if abs(float(endpoint_no_edl["E_mix"]) - float(no_edl["E_mix"])) > 1.0e-12:
            raise ValueError("w/o EDL E_mix changed with support C_H")
        if (
            abs(float(endpoint_no_edl["i_mix_avg_A_per_m2"]) - float(no_edl["i_mix_avg_A_per_m2"]))
            > 1.0e-12
        ):
            raise ValueError("w/o EDL i_mix changed with support C_H")

    if any(bool(row["debye_huckel_ok"]) for row in rows):
        raise ValueError("The expected linear Debye-Huckel caveat was not present at every scan point")

    zero = rows[0]
    high = rows[-1]
    diffuse_capacitance = float(zero["C_D_F_per_m2"])
    return {
        "output_tag": OUTPUT_TAG,
        "scan_parameter_display": "C_H,support",
        "scan_parameter_internal": "Cdl_C",
        "scan_min_F_per_m2": SUPPORT_CH_MIN_F_PER_M2,
        "scan_max_F_per_m2": SUPPORT_CH_MAX_F_PER_M2,
        "scan_min_uF_per_cm2": 100.0 * SUPPORT_CH_MIN_F_PER_M2,
        "scan_max_uF_per_cm2": 100.0 * SUPPORT_CH_MAX_F_PER_M2,
        "scan_points": N_SWEEP,
        "scan_scale": "linear",
        "baseline_C_H_support_F_per_m2": BASELINE_SUPPORT_CH_F_PER_M2,
        "baseline_C_H_support_uF_per_cm2": 100.0 * BASELINE_SUPPORT_CH_F_PER_M2,
        "N_modes": N_MODES,
        "Nx": NX,
        "diffuse_capacitance_C_D_F_per_m2": diffuse_capacitance,
        "diffuse_capacitance_C_D_uF_per_cm2": 100.0 * diffuse_capacitance,
        "g_support_min": float(zero["g_C"]),
        "g_support_max": float(high["g_C"]),
        "E_mix_with_at_zero_V": float(zero["E_mix_with_EDL_V"]),
        "E_mix_with_at_baseline_V": float(baseline["E_mix_with_EDL_V"]),
        "E_mix_with_at_max_V": float(high["E_mix_with_EDL_V"]),
        "delta_E_mix_zero_to_max_mV": 1000.0
        * (float(high["E_mix_with_EDL_V"]) - float(zero["E_mix_with_EDL_V"])),
        "i_mix_avg_with_at_zero_A_per_m2": float(zero["i_mix_avg_with_EDL_A_per_m2"]),
        "i_mix_avg_with_at_baseline_A_per_m2": float(baseline["i_mix_avg_with_EDL_A_per_m2"]),
        "i_mix_avg_with_at_max_A_per_m2": float(high["i_mix_avg_with_EDL_A_per_m2"]),
        "pct_i_mix_zero_to_max": 100.0
        * (
            float(high["i_mix_avg_with_EDL_A_per_m2"])
            / float(zero["i_mix_avg_with_EDL_A_per_m2"])
            - 1.0
        ),
        "E_mix_without_EDL_V": float(no_edl["E_mix"]),
        "i_mix_avg_without_EDL_A_per_m2": float(no_edl["i_mix_avg_A_per_m2"]),
        "E_mix_monotonic": "decreasing",
        "i_mix_monotonic": "increasing",
        "max_abs_phi_tilde_min": float(np.min(max_phi)),
        "max_abs_phi_tilde_max": float(np.max(max_phi)),
        "max_abs_current_balance_residual_A": float(np.max(np.abs(residual_abs_a))),
        "max_relative_current_balance_residual": float(np.max(relative_balance_residual)),
        "debye_huckel_ok_all_points": False,
        "linear_DH_caveat": (
            "max |phi_tilde| exceeds 1 at every scan point; interpret the result as an internal "
            "sensitivity of the current linear Debye-Huckel model, not as a quantitatively validated "
            "electrostatic magnitude."
        ),
        "support_CH_zero_interpretation": (
            "C_H,support=0 gives g_support=0 and zero local compact-layer charge on the support. "
            "The 10 nm support geometry remains, and lateral fields from Au/Pd can still make "
            "phi_RP on the support non-zero."
        ),
        "representative_profile_points_per_case": int(len(profiles[0.0]["x_nm"])),
    }


def add_baseline_reference(ax: plt.Axes) -> None:
    ax.axvline(
        100.0 * BASELINE_SUPPORT_CH_F_PER_M2,
        color=COLORS["baseline"],
        linewidth=1.05,
        linestyle=(0, (2, 2)),
        alpha=0.72,
        zorder=1,
    )


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(direction="out", length=3.5, width=0.9)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def plot_support_ch_figure(
    rows: list[dict[str, Any]],
    profiles: dict[float, dict[str, Any]],
    study_summary: dict[str, Any],
) -> None:
    x = np.asarray([float(row["C_H_support_uF_per_cm2"]) for row in rows], dtype=float)
    e_with = np.asarray([float(row["E_mix_with_EDL_V"]) for row in rows], dtype=float)
    i_with = np.asarray([float(row["i_mix_avg_with_EDL_A_per_m2"]) for row in rows], dtype=float)
    i_no = np.asarray([float(row["i_mix_avg_without_EDL_A_per_m2"]) for row in rows], dtype=float)
    phi_au = np.asarray([float(row["mean_phi_RP_Au_mV"]) for row in rows], dtype=float)
    phi_support = np.asarray([float(row["mean_phi_RP_support_mV"]) for row in rows], dtype=float)
    phi_pd = np.asarray([float(row["mean_phi_RP_Pd_mV"]) for row in rows], dtype=float)

    baseline = baseline_row(rows)
    baseline_x = float(baseline["C_H_support_uF_per_cm2"])
    marker_every = max(1, (N_SWEEP - 1) // 10)
    x_label = r"$C_{\mathrm{H},\mathrm{support}}$ ($\mu$F cm$^{-2}$)"

    fig, axes = plt.subplots(2, 2, figsize=(8.1, 6.15))
    ax_e, ax_i, ax_phi, ax_profile = axes.ravel()

    ax_e.plot(
        x,
        e_with,
        color=COLORS["with_edl"],
        linewidth=2.25,
        marker="o",
        markersize=3.7,
        markevery=marker_every,
        label="with EDL",
        zorder=3,
    )
    ax_e.scatter(
        [baseline_x],
        [float(baseline["E_mix_with_EDL_V"])],
        marker="*",
        s=72,
        facecolor=COLORS["with_edl"],
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
        label="baseline",
    )
    add_baseline_reference(ax_e)
    e_margin = 0.08 * float(np.ptp(e_with))
    ax_e.set_ylim(float(np.min(e_with)) - e_margin, float(np.max(e_with)) + e_margin)
    ax_e.set_title("Mixed potential")
    ax_e.set_xlabel(x_label)
    ax_e.set_ylabel(r"$E_{\mathrm{mix}}$ (V vs. RHE)")
    ax_e.text(
        0.98,
        0.96,
        f"w/o EDL = {float(study_summary['E_mix_without_EDL_V']):.3f} V\n"
        "(constant; outside scale)",
        transform=ax_e.transAxes,
        ha="right",
        va="top",
        color=COLORS["no_edl"],
        fontsize=8.8,
    )
    ax_e.legend(loc="lower left", handlelength=2.2)

    ax_i.plot(
        x,
        i_with,
        color=COLORS["with_edl"],
        linewidth=2.25,
        marker="o",
        markersize=3.7,
        markevery=marker_every,
        label="with EDL",
        zorder=3,
    )
    ax_i.plot(
        x,
        i_no,
        color=COLORS["no_edl"],
        linewidth=2.0,
        linestyle=(0, (5, 2.5)),
        label="w/o EDL",
        zorder=2,
    )
    ax_i.scatter(
        [baseline_x],
        [float(baseline["i_mix_avg_with_EDL_A_per_m2"])],
        marker="*",
        s=72,
        facecolor=COLORS["with_edl"],
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
    )
    add_baseline_reference(ax_i)
    ax_i.set_title("Mixed current density")
    ax_i.set_xlabel(x_label)
    ax_i.set_ylabel(r"$\bar{i}_{\mathrm{mix}}$ (A m$^{-2}$)")
    ax_i.legend(loc="center right", handlelength=2.5)

    ax_phi.plot(x, phi_au, color=COLORS["au"], linewidth=2.15, label="Au")
    ax_phi.plot(x, phi_support, color=COLORS["support"], linewidth=2.15, label="support")
    ax_phi.plot(x, phi_pd, color=COLORS["pd"], linewidth=2.15, label="Pd")
    add_baseline_reference(ax_phi)
    ax_phi.set_title("Segment-mean reaction-plane potential")
    ax_phi.set_xlabel(x_label)
    ax_phi.set_ylabel(r"$\langle\phi_{\mathrm{RP}}\rangle$ (mV)")
    ax_phi.legend(loc="best", ncol=1, handlelength=2.2)

    representative_styles = {
        0.0: (COLORS["case_zero"], (0, (4, 2))),
        BASELINE_SUPPORT_CH_F_PER_M2: (COLORS["case_baseline"], "-"),
        SUPPORT_CH_MAX_F_PER_M2: (COLORS["case_high"], (0, (1.5, 1.5))),
    }
    for value in REPRESENTATIVE_SUPPORT_CH_F_PER_M2:
        profile = profiles[value]
        color, linestyle = representative_styles[value]
        ax_profile.plot(
            profile["x_nm"],
            profile["phi_RP_mV"],
            color=color,
            linewidth=2.0,
            linestyle=linestyle,
            label=rf"{100.0 * value:.0f} $\mu$F cm$^{{-2}}$",
            zorder=3,
        )

    reference_derived = profiles[BASELINE_SUPPORT_CH_F_PER_M2]["derived"]
    l_au_nm = float(reference_derived["L_Au"]) * 1.0e9
    l_support_end_nm = float(reference_derived["L_C"]) * 1.0e9
    l_total_nm = float(reference_derived["L_total"]) * 1.0e9
    ax_profile.axvspan(0.0, l_au_nm, color=COLORS["au_fill"], alpha=0.18, zorder=0)
    ax_profile.axvspan(
        l_au_nm,
        l_support_end_nm,
        color=COLORS["support_fill"],
        alpha=0.48,
        zorder=0,
    )
    ax_profile.axvspan(
        l_support_end_nm,
        l_total_nm,
        color=COLORS["pd_fill"],
        alpha=0.25,
        zorder=0,
    )
    for boundary in (l_au_nm, l_support_end_nm):
        ax_profile.axvline(
            boundary,
            color=COLORS["boundary"],
            linewidth=0.9,
            linestyle=(0, (2, 2)),
            zorder=1,
        )
    ax_profile.set_xlim(0.0, l_total_nm)
    ax_profile.set_title("Spatial reaction-plane potential")
    ax_profile.set_xlabel(r"$x$ (nm)")
    ax_profile.set_ylabel(r"$\phi_{\mathrm{RP}}$ (mV)")
    ax_profile.legend(loc="upper left", title=r"support $C_{\mathrm{H}}$", title_fontsize=9.0)

    for ax in axes.ravel():
        style_axis(ax)

    caveat = (
        "Linear Debye–Hückel model: max |φ̃| = "
        f"{float(study_summary['max_abs_phi_tilde_min']):.3f}–"
        f"{float(study_summary['max_abs_phi_tilde_max']):.3f} > 1; "
        "trends are within-model sensitivities."
    )
    fig.text(0.5, 0.018, caveat, ha="center", va="bottom", color="#555555", fontsize=8.4)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.125, top=0.945, wspace=0.30, hspace=0.38)
    fig.savefig(PNG_OUT, dpi=600, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(SVG_OUT, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def save_traceability_inputs(
    params: dict[str, Any],
    high_resolution_summary: dict[str, str],
    study_summary: dict[str, Any],
) -> None:
    write_json(PARAMS_OUT, params)
    overrides = dict(common.PARAM_OVERRIDES)
    overrides.update(
        {
            "N_modes": N_MODES,
            "Nx": NX,
            "dh_violation_action": "ignore",
        }
    )
    write_json(OVERRIDES_OUT, overrides)
    write_json(SUMMARY_JSON_OUT, high_resolution_summary)
    write_rows(SUMMARY_CSV_OUT, [high_resolution_summary], list(high_resolution_summary.keys()))

    derived = solver.compute_derived_params(params)
    config = {
        "output_tag": OUTPUT_TAG,
        "source_result_params": str(common.BASE_PARAMS_PATH.relative_to(ROOT)),
        "source_helper": str((COMMON_DIR / "same_length_i0_alpha_common.py").relative_to(ROOT)),
        "scan_parameter_display": "C_H,support",
        "scan_parameter_internal": "Cdl_C",
        "scan_min_F_per_m2": SUPPORT_CH_MIN_F_PER_M2,
        "scan_max_F_per_m2": SUPPORT_CH_MAX_F_PER_M2,
        "scan_min_uF_per_cm2": 100.0 * SUPPORT_CH_MIN_F_PER_M2,
        "scan_max_uF_per_cm2": 100.0 * SUPPORT_CH_MAX_F_PER_M2,
        "scan_points": N_SWEEP,
        "scan_scale": "linear",
        "baseline_F_per_m2": BASELINE_SUPPORT_CH_F_PER_M2,
        "baseline_uF_per_cm2": 100.0 * BASELINE_SUPPORT_CH_F_PER_M2,
        "representative_F_per_m2": list(REPRESENTATIVE_SUPPORT_CH_F_PER_M2),
        "representative_uF_per_cm2": [100.0 * value for value in REPRESENTATIVE_SUPPORT_CH_F_PER_M2],
        "N_modes": N_MODES,
        "Nx": NX,
        "mode": "FULL",
        "C_D_F_per_m2": float(derived["epsilon_s"]) / float(derived["lambda_D"]),
        "C_D_uF_per_cm2": 100.0 * float(derived["epsilon_s"]) / float(derived["lambda_D"]),
        "note": (
            "C_H,support=0 is distinct from L_gap=0 and from w/o EDL: the 10 nm support remains, "
            "but its local compact-layer charging coefficient is zero."
        ),
    }
    write_json(CONFIG_OUT, config)
    write_json(STUDY_SUMMARY_OUT, study_summary)


def assert_outputs() -> None:
    expected = (
        PNG_OUT,
        SVG_OUT,
        SWEEP_CSV_OUT,
        PROFILE_CSV_OUT,
        PARAMS_OUT,
        OVERRIDES_OUT,
        SUMMARY_CSV_OUT,
        SUMMARY_JSON_OUT,
        CONFIG_OUT,
        STUDY_SUMMARY_OUT,
    )
    missing = [path for path in expected if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing or empty support C_H outputs: {missing}")
    pdfs = sorted(OUT_DIR.glob("**/*.pdf"))
    if pdfs:
        raise RuntimeError(f"Expected zero PDF files, found {len(pdfs)}")


def main() -> None:
    apply_style()
    ensure_dirs()
    params = configure_params()

    print(f"Running high-resolution support C_H study for {OUTPUT_TAG}")
    baseline_pair = solver.run_edl_comparison_pair(params, mode="FULL")
    high_resolution_summary = common.summary_from_pair(baseline_pair)
    no_edl = baseline_pair["no_edl"]

    rows, profiles = run_support_ch_scan(params, no_edl)
    study_summary = validate_results(params, rows, profiles, no_edl, high_resolution_summary)

    write_rows(SWEEP_CSV_OUT, rows, SWEEP_FIELDS)
    write_rows(PROFILE_CSV_OUT, profile_rows(profiles), PROFILE_FIELDS)
    save_traceability_inputs(params, high_resolution_summary, study_summary)
    plot_support_ch_figure(rows, profiles, study_summary)
    assert_outputs()

    print(f"Output directory: {OUT_DIR.relative_to(ROOT)}")
    print(
        "0 -> 100 uF/cm^2: "
        f"delta E_mix = {float(study_summary['delta_E_mix_zero_to_max_mV']):.6f} mV; "
        f"delta i_mix = {float(study_summary['pct_i_mix_zero_to_max']):.4f}%"
    )
    print("Verified 1 PNG + 1 SVG + 2 CSV + traceability inputs + 0 PDF")


if __name__ == "__main__":
    main()
