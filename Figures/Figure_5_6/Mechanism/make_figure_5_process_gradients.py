from __future__ import annotations

import copy
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator


ROOT = Path(__file__).resolve().parents[3]
SOLVER_DIR = ROOT / "Mixed_Potential_Electrical_Double_Layer"
HEATMAP_INPUTS_DIR = ROOT / "Figures" / "Figure_5" / "Heatmap" / "inputs"
OUT_DIR = Path(__file__).resolve().parent
CSV_DIR = OUT_DIR / "csv"

OUTPUT_TAG = "same_length_i0_alpha050_au25_pd25_20260528_111255"
PARAMS_PATH = HEATMAP_INPUTS_DIR / f"params_{OUTPUT_TAG}.json"
SUMMARY_PATH = HEATMAP_INPUTS_DIR / f"summary_compare_{OUTPUT_TAG}.json"
PNG_OUT = OUT_DIR / f"figure_5_process_gradients_{OUTPUT_TAG}.png"
SVG_OUT = OUT_DIR / f"figure_5_process_gradients_{OUTPUT_TAG}.svg"
CSV_OUT = CSV_DIR / f"figure_5_process_sweeps_{OUTPUT_TAG}.csv"
SCHEMATIC_CSV_OUT = CSV_DIR / f"figure_5_polarization_schematic_{OUTPUT_TAG}.csv"
CURVE_CSV_OUT = CSV_DIR / f"figure_5_polarization_curves_{OUTPUT_TAG}.csv"

N_SWEEP = 41
N_POLARIZATION_POINTS = 480
BASELINE_ATOL = 5.0e-10

sys.path.insert(0, str(SOLVER_DIR))
import Solve_Emix_updating as solver  # noqa: E402


COLORS = {
    "au_dark": "#B64342",
    "au_mid": "#F26B38",
    "au_light": "#F6CFCB",
    "pd_dark": "#0F4D92",
    "pd_mid": "#3775BA",
    "pd_light": "#DDF3DE",
    "dark": "#272727",
    "gray": "#767676",
    "light_gray": "#CFCECE",
    "no_edl": "#12355B",
    "baseline": "#111111",
    "au_curve": "#3B7A57",
    "pd_curve": "#B64342",
    "case_low": "#767676",
    "case_base": "#272727",
    "case_high": "#0F4D92",
}

AU_CMAP = LinearSegmentedColormap.from_list("au_sweep", [COLORS["au_light"], COLORS["au_mid"], COLORS["au_dark"]])
PD_CMAP = LinearSegmentedColormap.from_list("pd_sweep", [COLORS["pd_light"], COLORS["pd_mid"], COLORS["pd_dark"]])


@dataclass(frozen=True)
class ScanSpec:
    key: str
    title: str
    side: str
    low: float
    baseline: float
    high: float
    scale: str
    display_factor: float
    display_unit: str
    display_fmt: str


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    scale: str
    fmt: str
    no_edl_key: str | None = None


SCAN_SPECS = (
    ScanSpec(
        key="Cdl_Au",
        title=r"$C_{\mathrm{H},\mathrm{Au}}$",
        side="Au",
        low=0.05,
        baseline=0.20,
        high=1.00,
        scale="log",
        display_factor=100.0,
        display_unit=r"$\mu$F cm$^{-2}$",
        display_fmt="{:.0f}",
    ),
    ScanSpec(
        key="Cdl_Pd",
        title=r"$C_{\mathrm{H},\mathrm{Pd}}$",
        side="Pd",
        low=0.05,
        baseline=0.40,
        high=1.00,
        scale="log",
        display_factor=100.0,
        display_unit=r"$\mu$F cm$^{-2}$",
        display_fmt="{:.0f}",
    ),
    ScanSpec(
        key="pzc_Au",
        title=r"$\mathrm{PZC}_{\mathrm{Au}}$",
        side="Au",
        low=0.63,
        baseline=0.93,
        high=1.23,
        scale="linear",
        display_factor=1.0,
        display_unit="V",
        display_fmt="{:.2f}",
    ),
    ScanSpec(
        key="pzc_Pd",
        title=r"$\mathrm{PZC}_{\mathrm{Pd}}$",
        side="Pd",
        low=0.48,
        baseline=0.78,
        high=1.08,
        scale="linear",
        display_factor=1.0,
        display_unit="V",
        display_fmt="{:.2f}",
    ),
)

METRIC_SPECS = (
    MetricSpec("sigma_uC_per_cm2", r"$\sigma$ ($\mu$C/cm$^2$)", "linear", "{:.2f}"),
    MetricSpec("phi_RP_mean_mV", r"$\langle\phi_{\mathrm{RP}}\rangle$ (mV)", "linear", "{:.0f}"),
    MetricSpec("reactant_mean", r"$\langle c_{\mathrm{react}}/c_{\mathrm{bulk}}\rangle$", "log", "{:.1e}"),
    MetricSpec("eta_mean_mV", r"$\langle\eta_{\mathrm{RP}}\rangle$ (mV)", "linear", "{:.0f}"),
    MetricSpec("bv_exp_mean", r"$\langle e^{f_\eta}\rangle$", "log", "{:.1e}"),
    MetricSpec("bv_conc_exp_mean", r"$\langle(c/c_b)e^{f_\eta}\rangle$", "log", "{:.1e}"),
    MetricSpec(
        "i_mix_avg_with_A_per_m2",
        r"$\bar{i}_{\mathrm{mix}}$ (A m$^{-2}$)",
        "log",
        "{:.2e}",
        "i_mix_avg_no_A_per_m2",
    ),
)

MECHANISM_FIGURE_STEMS = {
    "Cdl_Au": "figure_5_mechanism_ch_au_causal_chain",
    "Cdl_Pd": "figure_5_mechanism_ch_pd_causal_chain",
    "pzc_Au": "figure_5_mechanism_pzc_au_causal_chain",
    "pzc_Pd": "figure_5_mechanism_pzc_pd_causal_chain",
}

POLARIZATION_FIGURE_STEMS = {
    "Cdl_Au": "figure_5_polarization_ch_au",
    "Cdl_Pd": "figure_5_polarization_ch_pd",
    "pzc_Au": "figure_5_polarization_pzc_au",
    "pzc_Pd": "figure_5_polarization_pzc_pd",
}

MECHANISM_FIGURE_TITLES = {
    "Cdl_Au": r"Figure 5 mechanism: $C_{\mathrm{H},\mathrm{Au}}$ causal chain",
    "Cdl_Pd": r"Figure 5 mechanism: $C_{\mathrm{H},\mathrm{Pd}}$ causal chain",
    "pzc_Au": r"Figure 5 mechanism: $\mathrm{PZC}_{\mathrm{Au}}$ causal chain",
    "pzc_Pd": r"Figure 5 mechanism: $\mathrm{PZC}_{\mathrm{Pd}}$ causal chain",
}

POLARIZATION_FIGURE_TITLES = {
    "Cdl_Au": r"Figure 5 polarization: $C_{\mathrm{H},\mathrm{Au}}$ scan",
    "Cdl_Pd": r"Figure 5 polarization: $C_{\mathrm{H},\mathrm{Pd}}$ scan",
    "pzc_Au": r"Figure 5 polarization: $\mathrm{PZC}_{\mathrm{Au}}$ scan",
    "pzc_Pd": r"Figure 5 polarization: $\mathrm{PZC}_{\mathrm{Pd}}$ scan",
}

MECHANISM_FIGSIZE = (3.60, 11.55)
POLARIZATION_FIGSIZE = (6.1, 4.35)
POLARIZATION_FONT_SCALE = 1.45
PLOT_CURRENT_UNIT_A = 1.0e-9
PLOT_CURRENT_LABEL = r"Current (10$^{-3}$ $\mu$A)"

SCHEMATIC_CASE_SPECS = (
    ("low", 0.52, (0, (3, 2))),
    ("base", 1.0, "-"),
    ("high", 0.78, (0, (5, 2))),
)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 8.2,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.75,
            "axes.grid": False,
            "legend.frameon": False,
            "savefig.bbox": None,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Nimbus Sans",
            "mathtext.it": "Nimbus Sans:italic",
            "mathtext.bf": "Nimbus Sans:bold",
            "mathtext.cal": "Nimbus Sans",
            "mathtext.sf": "Nimbus Sans",
            "mathtext.tt": "Nimbus Sans",
        }
    )


def polarization_font_scale(spec: ScanSpec) -> float:
    return POLARIZATION_FONT_SCALE


def polarization_figsize(spec: ScanSpec) -> tuple[float, float]:
    return POLARIZATION_FIGSIZE


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def assert_close(label: str, actual: float, expected: float, atol: float) -> None:
    if abs(actual - expected) > atol:
        raise ValueError(f"{label}: recomputed {actual:.15g} differs from stored {expected:.15g}")


def split_axis(low: float, baseline: float, high: float, n: int, scale: str) -> np.ndarray:
    if n < 3 or n % 2 != 1:
        raise ValueError("N_SWEEP must be an odd integer >= 3 so the baseline is exact")
    half = n // 2
    if not (low < baseline < high):
        raise ValueError("Each sweep needs low < baseline < high")
    if scale == "log":
        lower = np.geomspace(low, baseline, half + 1)
        upper = np.geomspace(baseline, high, half + 1)[1:]
    elif scale == "linear":
        lower = np.linspace(low, baseline, half + 1)
        upper = np.linspace(baseline, high, half + 1)[1:]
    else:
        raise ValueError(f"Unsupported scale: {scale}")
    values = np.concatenate([lower, upper])
    values[half] = baseline
    return values


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def segment_mean(values: np.ndarray, x_tilde: np.ndarray, mask: np.ndarray, segment_length_tilde: float) -> float:
    vals = np.asarray(values, dtype=float)[mask]
    x_seg = np.asarray(x_tilde, dtype=float)[mask]
    if vals.size == 0 or segment_length_tilde <= 0.0:
        return float("nan")
    if vals.size == 1:
        return float(vals[0])
    return trapz(vals, x_seg) / float(segment_length_tilde)


def validate_baseline(params: dict[str, Any], summary: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    res_edl = solver.run_case(params, mode="FULL", return_profiles=False, use_edl=True)
    res_no = solver.run_case(params, mode="FULL", return_profiles=False, use_edl=False)
    assert_close("E_mix_with", float(res_edl["E_mix"]), float(summary["E_mix_with"]), BASELINE_ATOL)
    assert_close(
        "i_mix_avg_with",
        float(res_edl["i_mix_avg_A_per_m2"]),
        float(summary["i_mix_avg_with"]),
        BASELINE_ATOL,
    )
    assert_close("E_mix_no", float(res_no["E_mix"]), float(summary["E_mix_no"]), BASELINE_ATOL)
    assert_close(
        "i_mix_avg_no",
        float(res_no["i_mix_avg_A_per_m2"]),
        float(summary["i_mix_avg_no"]),
        BASELINE_ATOL,
    )
    return res_edl, res_no


def effective_cdl(params: dict[str, Any], derived: dict[str, Any], side: str) -> float:
    suffix = "Au" if side == "Au" else "Pd"
    g_key = f"g_{suffix}"
    return float(derived[g_key]) * float(derived["epsilon_s"]) / float(derived["lambda_D"])


def summarize_case(
    base_params: dict[str, Any],
    spec: ScanSpec,
    value: float,
    sweep_index: int,
    no_edl: dict[str, Any],
) -> dict[str, Any]:
    params = copy.deepcopy(base_params)
    params[spec.key] = float(value)

    res = solver.run_case(params, mode="FULL", return_profiles=True, use_edl=True)
    derived = solver.compute_derived_params(params)

    x_tilde = np.asarray(res["x_tilde"], dtype=float)
    phi_tilde = np.asarray(res["phi_tilde"], dtype=float)
    phi_rp_v = (float(derived["R"]) * float(derived["T"]) / float(derived["F"])) * phi_tilde
    e_mix = float(res["E_mix"])

    if spec.side == "Au":
        mask = np.asarray(res["mask_Au"], dtype=bool)
        segment_length = float(derived["L_Au_tilde"])
        pzc = float(params["pzc_Au"])
        reactant = "R1"
        reactant_values = np.asarray(solver.safe_exp(-float(params["z_R1"]) * phi_tilde), dtype=float)
        eta_values = e_mix - float(res["E1_eq_eff"]) - phi_rp_v
        beta = float(derived["F"]) / (float(derived["R"]) * float(derived["T"]))
        bv_exp_values = np.asarray(solver.safe_exp((1.0 - float(params["alpha1"])) * beta * eta_values), dtype=float)
        side_current_avg = float(res["I_Au_avg_A_per_m2"])
    else:
        mask = np.asarray(res["mask_Pd"], dtype=bool)
        segment_length = float(derived["L_tilde"] - derived["L_C_tilde"])
        pzc = float(params["pzc_Pd"])
        reactant = "O2"
        reactant_values = np.asarray(solver.safe_exp(-float(params["z_O2"]) * phi_tilde), dtype=float)
        eta_values = e_mix - float(res["E2_eq_eff"]) - phi_rp_v
        beta = float(derived["F"]) / (float(derived["R"]) * float(derived["T"]))
        bv_exp_values = np.asarray(solver.safe_exp(-float(params["alpha2"]) * beta * eta_values), dtype=float)
        side_current_avg = float(res["I_Pd_avg_A_per_m2"])

    phi_mean_v = segment_mean(phi_rp_v, x_tilde, mask, segment_length)
    reactant_mean = segment_mean(reactant_values, x_tilde, mask, segment_length)
    eta_mean_v = segment_mean(eta_values, x_tilde, mask, segment_length)
    bv_exp_mean = segment_mean(bv_exp_values, x_tilde, mask, segment_length)
    bv_conc_exp_values = reactant_values * bv_exp_values
    bv_conc_exp_mean = segment_mean(bv_conc_exp_values, x_tilde, mask, segment_length)
    sigma_c_per_m2 = effective_cdl(params, derived, spec.side) * (e_mix - phi_mean_v - pzc)

    return {
        "scan": spec.key,
        "side": spec.side,
        "reactant": reactant,
        "sweep_index": sweep_index,
        "is_baseline": bool(abs(float(value) - spec.baseline) <= max(1e-15, abs(spec.baseline) * 1e-12)),
        "parameter_value_SI": float(value),
        "parameter_value_display": spec.display_factor * float(value),
        "parameter_display_unit": spec.display_unit,
        "sigma_C_per_m2": float(sigma_c_per_m2),
        "sigma_mC_per_m2": float(1000.0 * sigma_c_per_m2),
        "sigma_uC_per_cm2": float(100.0 * sigma_c_per_m2),
        "phi_RP_mean_V": float(phi_mean_v),
        "phi_RP_mean_mV": float(1000.0 * phi_mean_v),
        "reactant_mean": float(reactant_mean),
        "log10_reactant_mean": float(np.log10(reactant_mean)),
        "eta_mean_V": float(eta_mean_v),
        "eta_mean_mV": float(1000.0 * eta_mean_v),
        "bv_exp_mean": float(bv_exp_mean),
        "log10_bv_exp_mean": float(np.log10(bv_exp_mean)),
        "bv_conc_exp_mean": float(bv_conc_exp_mean),
        "log10_bv_conc_exp_mean": float(np.log10(bv_conc_exp_mean)),
        "side_current_avg_A_per_m2": float(side_current_avg),
        "E_mix_with_V": e_mix,
        "E_mix_no_V": float(no_edl["E_mix"]),
        "i_mix_avg_with_A_per_m2": float(res["i_mix_avg_A_per_m2"]),
        "i_mix_avg_no_A_per_m2": float(no_edl["i_mix_avg_A_per_m2"]),
        "i_mix_abs_with_A": float(res["i_mix_abs_A"]),
        "i_mix_abs_no_A": float(no_edl["i_mix_abs_A"]),
        "max_abs_phi_tilde": float(res["max_abs_phi_tilde"]),
        "debye_huckel_ok": bool(res["debye_huckel_ok"]),
    }


def build_sweep_rows(params: dict[str, Any], no_edl: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SCAN_SPECS:
        values = split_axis(spec.low, spec.baseline, spec.high, N_SWEEP, spec.scale)
        for idx, value in enumerate(values):
            rows.append(summarize_case(params, spec, float(value), idx, no_edl))
    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scan",
        "side",
        "reactant",
        "sweep_index",
        "is_baseline",
        "parameter_value_SI",
        "parameter_value_display",
        "parameter_display_unit",
        "sigma_C_per_m2",
        "sigma_mC_per_m2",
        "sigma_uC_per_cm2",
        "phi_RP_mean_V",
        "phi_RP_mean_mV",
        "reactant_mean",
        "log10_reactant_mean",
        "eta_mean_V",
        "eta_mean_mV",
        "bv_exp_mean",
        "log10_bv_exp_mean",
        "bv_conc_exp_mean",
        "log10_bv_conc_exp_mean",
        "side_current_avg_A_per_m2",
        "E_mix_with_V",
        "E_mix_no_V",
        "i_mix_avg_with_A_per_m2",
        "i_mix_avg_no_A_per_m2",
        "i_mix_abs_with_A",
        "i_mix_abs_no_A",
        "max_abs_phi_tilde",
        "debye_huckel_ok",
    ]
    with CSV_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def schematic_value_for_case(spec: ScanSpec, case_name: str) -> float:
    if case_name == "low":
        return spec.low
    if case_name == "base":
        return spec.baseline
    if case_name == "high":
        return spec.high
    raise ValueError(f"Unsupported schematic case: {case_name}")


def average_current_density(params: dict[str, Any], current: np.ndarray | float) -> np.ndarray | float:
    derived = solver.compute_derived_params(params)
    reactive_length = float(derived["L_Au"]) + float(derived["L_Pd_len"])
    value = float(derived["lambda_D"]) * current / reactive_length
    return value


def plot_current_units(current_abs_A: np.ndarray | float) -> np.ndarray | float:
    values = np.asarray(current_abs_A, dtype=float) / PLOT_CURRENT_UNIT_A
    if np.ndim(current_abs_A) == 0:
        return float(values)
    return values


def build_schematic_case(
    base_params: dict[str, Any],
    spec: ScanSpec,
    case_name: str,
    e_values: np.ndarray | None = None,
) -> dict[str, Any]:
    value = schematic_value_for_case(spec, case_name)
    params = copy.deepcopy(base_params)
    params[spec.key] = float(value)
    result = solver.run_case(params, mode="FULL", return_profiles=False, use_edl=True)
    case: dict[str, Any] = {
        "scan": spec.key,
        "case": case_name,
        "parameter_value_SI": float(value),
        "parameter_value_display": spec.display_factor * float(value),
        "parameter_display_unit": spec.display_unit,
        "E_mix": float(result["E_mix"]),
        "i_mix_avg_A_per_m2": float(result["i_mix_avg_A_per_m2"]),
        "I_Au_avg_A_per_m2": float(result["I_Au_avg_A_per_m2"]),
        "I_Pd_avg_A_per_m2": float(result["I_Pd_avg_A_per_m2"]),
        "i_mix_current_1e_minus_3_uA": float(plot_current_units(result["i_mix_abs_A"])),
        "I_Au_current_1e_minus_3_uA": float(plot_current_units(result["I_Au_abs_A"])),
        "I_Pd_current_1e_minus_3_uA": float(plot_current_units(result["I_Pd_abs_A"])),
        "E1_eq_eff": float(result["E1_eq_eff"]),
        "E2_eq_eff": float(result["E2_eq_eff"]),
        "params": params,
    }
    if e_values is not None:
        curve = solver.compute_polarization_curve(
            params,
            mode="FULL",
            use_edl=True,
            E_values=e_values,
            use_affine_phi2=bool(params.get("use_affine_phi2", True)),
        )
        case.update(
            E=np.asarray(curve["E"], dtype=float),
            I_Au_avg_curve=np.asarray(average_current_density(params, curve["I_Au"]), dtype=float),
            I_Pd_avg_curve=np.asarray(average_current_density(params, curve["I_Pd"]), dtype=float),
            I_Au_current_curve=np.asarray(plot_current_units(curve["I_Au_abs_A"]), dtype=float),
            I_Pd_current_curve=np.asarray(plot_current_units(curve["I_Pd_abs_A"]), dtype=float),
        )
    return case


def build_polarization_schematic_data(base_params: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    schematic: dict[str, list[dict[str, Any]]] = {}
    for spec in SCAN_SPECS:
        schematic[spec.key] = [
            build_schematic_case(base_params, spec, case_name)
            for case_name, _alpha, _style in SCHEMATIC_CASE_SPECS
        ]
    return schematic


def write_schematic_csv(schematic: dict[str, list[dict[str, Any]]]) -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scan",
        "case",
        "parameter_value_SI",
        "parameter_value_display",
        "parameter_display_unit",
        "E_mix",
        "i_mix_avg_A_per_m2",
        "I_Au_avg_A_per_m2",
        "I_Pd_avg_A_per_m2",
        "i_mix_current_1e_minus_3_uA",
        "I_Au_current_1e_minus_3_uA",
        "I_Pd_current_1e_minus_3_uA",
    ]
    with SCHEMATIC_CSV_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for spec in SCAN_SPECS:
            for case in schematic[spec.key]:
                writer.writerow({key: case[key] for key in fieldnames})


def require_finite_schematic(schematic: dict[str, list[dict[str, Any]]]) -> None:
    numeric_keys = [
        "parameter_value_SI",
        "parameter_value_display",
        "E_mix",
        "i_mix_avg_A_per_m2",
        "I_Au_avg_A_per_m2",
        "I_Pd_avg_A_per_m2",
    ]
    total = sum(len(cases) for cases in schematic.values())
    if total != len(SCAN_SPECS) * len(SCHEMATIC_CASE_SPECS):
        raise RuntimeError(f"Expected {len(SCAN_SPECS) * len(SCHEMATIC_CASE_SPECS)} schematic rows, got {total}")
    for cases in schematic.values():
        for case in cases:
            for key in numeric_keys:
                if not np.isfinite(float(case[key])):
                    raise RuntimeError(f"{case['scan']} {case['case']} has non-finite {key}")
            for key in ("E", "I_Au_avg_curve", "I_Pd_avg_curve"):
                if key in case and not np.all(np.isfinite(np.asarray(case[key], dtype=float))):
                    raise RuntimeError(f"{case['scan']} {case['case']} has non-finite curve {key}")


def polarization_e_values(cases: list[dict[str, Any]]) -> np.ndarray:
    e_mix_values = np.asarray([float(case["E_mix"]) for case in cases], dtype=float)
    e_span = max(float(np.max(e_mix_values) - np.min(e_mix_values)), 0.10)
    pad = max(0.09, 0.65 * e_span)
    e_min = float(np.min(e_mix_values)) - pad
    e_max = float(np.max(e_mix_values)) + pad
    return np.linspace(e_min, e_max, N_POLARIZATION_POINTS)


def build_polarization_curve_data(
    schematic: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    curve_data: dict[str, list[dict[str, Any]]] = {}
    for spec in SCAN_SPECS:
        cases = schematic[spec.key]
        e_values = polarization_e_values(cases)
        out_cases: list[dict[str, Any]] = []
        for case in cases:
            params = case["params"]
            curve = solver.compute_polarization_curve(
                params,
                mode="FULL",
                use_edl=True,
                E_values=e_values,
                use_affine_phi2=bool(params.get("use_affine_phi2", True)),
            )
            out_cases.append(
                {
                    "scan": spec.key,
                    "case": case["case"],
                    "parameter_value_display": float(case["parameter_value_display"]),
                    "parameter_display_unit": case["parameter_display_unit"],
                    "E_mix": float(case["E_mix"]),
                    "i_mix_avg_A_per_m2": float(case["i_mix_avg_A_per_m2"]),
                    "I_Au_avg_A_per_m2": float(case["I_Au_avg_A_per_m2"]),
                    "I_Pd_avg_A_per_m2": float(case["I_Pd_avg_A_per_m2"]),
                    "i_mix_current_1e_minus_3_uA": float(case["i_mix_current_1e_minus_3_uA"]),
                    "I_Au_current_1e_minus_3_uA": float(case["I_Au_current_1e_minus_3_uA"]),
                    "I_Pd_current_1e_minus_3_uA": float(case["I_Pd_current_1e_minus_3_uA"]),
                    "E": np.asarray(curve["E"], dtype=float),
                    "I_Au_avg_curve": np.asarray(average_current_density(params, curve["I_Au"]), dtype=float),
                    "I_Pd_avg_curve": np.asarray(average_current_density(params, curve["I_Pd"]), dtype=float),
                    "I_Au_current_curve": np.asarray(plot_current_units(curve["I_Au_abs_A"]), dtype=float),
                    "I_Pd_current_curve": np.asarray(plot_current_units(curve["I_Pd_abs_A"]), dtype=float),
                }
            )

        base_case = next(case for case in out_cases if case["case"] == "base")
        base_au = np.asarray(base_case["I_Au_avg_curve"], dtype=float)
        base_pd = np.asarray(base_case["I_Pd_avg_curve"], dtype=float)
        for case in out_cases:
            au = np.asarray(case["I_Au_avg_curve"], dtype=float)
            pd = np.asarray(case["I_Pd_avg_curve"], dtype=float)
            delta_au = au - base_au
            delta_pd = pd - base_pd
            rel_au = delta_au / np.maximum(np.abs(base_au), 1.0e-30)
            rel_pd = delta_pd / np.maximum(np.abs(base_pd), 1.0e-30)
            case.update(
                delta_I_Au_vs_base=delta_au,
                delta_I_Pd_vs_base=delta_pd,
                rel_delta_I_Au_vs_base=rel_au,
                rel_delta_I_Pd_vs_base=rel_pd,
                max_abs_rel_delta_I_Au_vs_base=float(np.max(np.abs(rel_au))),
                max_abs_rel_delta_I_Pd_vs_base=float(np.max(np.abs(rel_pd))),
            )
        curve_data[spec.key] = out_cases
    return curve_data


def write_polarization_curve_csv(curve_data: dict[str, list[dict[str, Any]]]) -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scan",
        "case",
        "parameter_value_display",
        "E_V",
        "I_Au_avg_A_per_m2",
        "I_Pd_avg_A_per_m2",
        "I_Au_current_1e_minus_3_uA",
        "I_Pd_current_1e_minus_3_uA",
        "delta_I_Au_vs_base",
        "delta_I_Pd_vs_base",
        "rel_delta_I_Au_vs_base",
        "rel_delta_I_Pd_vs_base",
    ]
    with CURVE_CSV_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for spec in SCAN_SPECS:
            for case in curve_data[spec.key]:
                e_values = np.asarray(case["E"], dtype=float)
                for idx, e_value in enumerate(e_values):
                    writer.writerow(
                        {
                            "scan": spec.key,
                            "case": case["case"],
                            "parameter_value_display": case["parameter_value_display"],
                            "E_V": float(e_value),
                            "I_Au_avg_A_per_m2": float(case["I_Au_avg_curve"][idx]),
                            "I_Pd_avg_A_per_m2": float(case["I_Pd_avg_curve"][idx]),
                            "I_Au_current_1e_minus_3_uA": float(case["I_Au_current_curve"][idx]),
                            "I_Pd_current_1e_minus_3_uA": float(case["I_Pd_current_curve"][idx]),
                            "delta_I_Au_vs_base": float(case["delta_I_Au_vs_base"][idx]),
                            "delta_I_Pd_vs_base": float(case["delta_I_Pd_vs_base"][idx]),
                            "rel_delta_I_Au_vs_base": float(case["rel_delta_I_Au_vs_base"][idx]),
                            "rel_delta_I_Pd_vs_base": float(case["rel_delta_I_Pd_vs_base"][idx]),
                        }
                    )


def require_finite_curve_data(curve_data: dict[str, list[dict[str, Any]]]) -> None:
    total_cases = sum(len(cases) for cases in curve_data.values())
    expected_cases = len(SCAN_SPECS) * len(SCHEMATIC_CASE_SPECS)
    if total_cases != expected_cases:
        raise RuntimeError(f"Expected {expected_cases} polarization curve cases, got {total_cases}")
    array_keys = [
        "E",
        "I_Au_avg_curve",
        "I_Pd_avg_curve",
        "I_Au_current_curve",
        "I_Pd_current_curve",
        "delta_I_Au_vs_base",
        "delta_I_Pd_vs_base",
        "rel_delta_I_Au_vs_base",
        "rel_delta_I_Pd_vs_base",
    ]
    for cases in curve_data.values():
        for case in cases:
            for key in array_keys:
                values = np.asarray(case[key], dtype=float)
                if values.size != N_POLARIZATION_POINTS:
                    raise RuntimeError(f"{case['scan']} {case['case']} {key} has {values.size} points")
                if not np.all(np.isfinite(values)):
                    raise RuntimeError(f"{case['scan']} {case['case']} has non-finite {key}")
            for key in (
                "E_mix",
                "i_mix_avg_A_per_m2",
                "I_Au_avg_A_per_m2",
                "I_Pd_avg_A_per_m2",
                "i_mix_current_1e_minus_3_uA",
                "I_Au_current_1e_minus_3_uA",
                "I_Pd_current_1e_minus_3_uA",
                "max_abs_rel_delta_I_Au_vs_base",
                "max_abs_rel_delta_I_Pd_vs_base",
            ):
                if not np.isfinite(float(case[key])):
                    raise RuntimeError(f"{case['scan']} {case['case']} has non-finite {key}")


def require_finite_rows(rows: list[dict[str, Any]]) -> None:
    numeric_keys = [
        "parameter_value_SI",
        "parameter_value_display",
        "sigma_C_per_m2",
        "sigma_mC_per_m2",
        "sigma_uC_per_cm2",
        "phi_RP_mean_V",
        "phi_RP_mean_mV",
        "reactant_mean",
        "log10_reactant_mean",
        "eta_mean_V",
        "eta_mean_mV",
        "bv_exp_mean",
        "log10_bv_exp_mean",
        "bv_conc_exp_mean",
        "log10_bv_conc_exp_mean",
        "side_current_avg_A_per_m2",
        "E_mix_with_V",
        "E_mix_no_V",
        "i_mix_avg_with_A_per_m2",
        "i_mix_avg_no_A_per_m2",
        "i_mix_abs_with_A",
        "i_mix_abs_no_A",
        "max_abs_phi_tilde",
    ]
    if len(rows) != len(SCAN_SPECS) * N_SWEEP:
        raise RuntimeError(f"Expected {len(SCAN_SPECS) * N_SWEEP} rows, got {len(rows)}")
    for row in rows:
        for key in numeric_keys:
            if not np.isfinite(float(row[key])):
                raise RuntimeError(f"{row['scan']} row {row['sweep_index']} has non-finite {key}")


def rows_for_scan(rows: list[dict[str, Any]], scan_key: str) -> list[dict[str, Any]]:
    out = [row for row in rows if row["scan"] == scan_key]
    return sorted(out, key=lambda row: int(row["sweep_index"]))


def values_for(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def metric_xlim(rows: list[dict[str, Any]], metric: MetricSpec) -> tuple[float, float]:
    vals = values_for(rows, metric.key)
    if metric.no_edl_key is not None:
        vals = np.concatenate([vals, values_for(rows, metric.no_edl_key)])
    vals = vals[np.isfinite(vals)]
    if metric.scale == "log":
        vals = vals[vals > 0.0]
        if vals.size == 0:
            raise ValueError(f"{metric.key} has no positive values for a log axis")
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        return lo / 1.65, hi * 1.65

    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if lo < 0.0 < hi:
        span = max(abs(lo), abs(hi))
        return -1.22 * span, 1.22 * span
    pad = max(1.0e-9, 0.16 * (hi - lo if hi > lo else max(abs(hi), 1.0)))
    return lo - pad, hi + pad


def format_parameter_ticks(spec: ScanSpec) -> list[str]:
    return [
        spec.display_fmt.format(spec.display_factor * spec.low),
        spec.display_fmt.format(spec.display_factor * spec.baseline),
        spec.display_fmt.format(spec.display_factor * spec.high),
    ]


def parameter_axis_label(spec: ScanSpec) -> str:
    return f"{spec.title}\n({spec.display_unit})"


def reactant_symbol(spec: ScanSpec | None) -> str:
    if spec is None:
        return r"\mathrm{Red}_1/\mathrm{Ox}_2"
    return r"\mathrm{Red}_1" if spec.side == "Au" else r"\mathrm{Ox}_2"


def metric_label(metric: MetricSpec, spec: ScanSpec | None = None) -> str:
    if metric.key == "reactant_mean":
        return rf"$\langle c_{{{reactant_symbol(spec)}}}/c_{{\mathrm{{bulk}}}}\rangle$"
    if metric.key == "bv_conc_exp_mean":
        return rf"$\langle(c_{{{reactant_symbol(spec)}}}/c_b)e^{{f_\eta}}\rangle$"
    return metric.label


def formula_symbols(spec: ScanSpec | None) -> tuple[str, str]:
    if spec is None:
        return r"j", r"\mathrm{Red}_1/\mathrm{Ox}_2"
    side = r"\mathrm{Au}" if spec.side == "Au" else r"\mathrm{Pd}"
    reactant = reactant_symbol(spec)
    return side, reactant


def formula_for_metric(metric: MetricSpec, spec: ScanSpec | None = None) -> str:
    j, k = formula_symbols(spec)
    if metric.key == "sigma_uC_per_cm2":
        return (
            rf"$\sigma_{{{j}}}=C_{{H,{j}}}\Delta\phi_{{{j}}}$"
            "\n"
            rf"$\Delta\phi_{{{j}}}=E_{{\mathrm{{mix}}}}-\langle\phi_{{\mathrm{{RP}}}}\rangle_{{{j}}}-\mathrm{{PZC}}_{{{j}}}$"
        )
    if metric.key == "phi_RP_mean_mV":
        return (
            rf"$\langle\phi_{{\mathrm{{RP}}}}\rangle_{{{j}}}"
            rf"=\frac{{RT}}{{F}}\langle\tilde{{\phi}}_s(x,0)\rangle_{{{j}}}$"
        )
    if metric.key == "reactant_mean":
        return (
            rf"$\langle c_{{{k}}}/c_b\rangle_{{{j}}}"
            rf"=\langle e^{{-z_{{{k}}}\tilde{{\phi}}_s}}\rangle_{{{j}}}$"
        )
    if metric.key == "eta_mean_mV":
        if spec is None:
            return rf"$\eta_{{{j}}}=E_{{\mathrm{{mix}}}}-E_{{{j},\mathrm{{eq}}}}-\phi_{{\mathrm{{RP}}}}(x)$"
        if spec is not None and spec.side == "Au":
            return rf"$\eta_{{{j}}}=E_{{\mathrm{{mix}}}}-E_{{1,\mathrm{{eq}}}}-\phi_{{\mathrm{{RP}}}}(x)$"
        return rf"$\eta_{{{j}}}=E_{{\mathrm{{mix}}}}-E_{{2,\mathrm{{eq}}}}-\phi_{{\mathrm{{RP}}}}(x)$"
    if metric.key == "bv_exp_mean":
        if spec is None:
            return r"$f_\eta=(1-\alpha_1)F\eta_{\mathrm{Au}}/RT$ or $-\alpha_2F\eta_{\mathrm{Pd}}/RT$"
        if spec is not None and spec.side == "Au":
            return rf"$\langle e^{{f_\eta}}\rangle_{{{j}}}=\langle e^{{(1-\alpha_1)F\eta_{{{j}}}/RT}}\rangle_{{{j}}}$"
        return rf"$\langle e^{{f_\eta}}\rangle_{{{j}}}=\langle e^{{-\alpha_2F\eta_{{{j}}}/RT}}\rangle_{{{j}}}$"
    if metric.key == "bv_conc_exp_mean":
        return (
            rf"$\langle(c_{{{k}}}/c_b)e^{{f_\eta}}\rangle_{{{j}}}$"
            "\n"
            r"dimensionless BV driving weight"
        )
    if metric.key == "side_current_avg_A_per_m2":
        return (
            rf"$\bar{{I}}_{{{j}}}=\lambda_D I_{{{j}}}/L_{{\mathrm{{rxn}}}}$"
            "\n"
            rf"$I_{{{j}}}=\int_{{{j}}} i_{{{j}}}(\tilde{{x}})\,d\tilde{{x}}$"
        )
    if metric.key == "E_mix_with_V":
        return r"$I_{\mathrm{Au}}(E_{\mathrm{mix}})+I_{\mathrm{Pd}}(E_{\mathrm{mix}})=0$"
    if metric.key == "i_mix_avg_with_A_per_m2":
        return (
            r"$\bar{i}_{\mathrm{mix}}=\lambda_D |I_{\mathrm{Au}}|/L_{\mathrm{rxn}}$"
            "\n"
            r"$=\lambda_D |I_{\mathrm{Pd}}|/L_{\mathrm{rxn}}$"
        )
    raise ValueError(f"No formula configured for {metric.key}")


def style_formula_axis(ax: plt.Axes, metric: MetricSpec, formula: str, *, show_title: bool) -> None:
    ax.set_axis_off()
    if show_title:
        ax.text(
            0.0,
            1.0,
            "Formula",
            ha="left",
            va="top",
            fontsize=7.6,
            color=COLORS["dark"],
            fontweight="normal",
            transform=ax.transAxes,
        )
    ax.text(
        0.0,
        0.60,
        metric_label(metric),
        ha="left",
        va="center",
        fontsize=5.8,
        color=COLORS["gray"],
        transform=ax.transAxes,
    )
    formula_lines = formula.splitlines()
    y0 = 0.38 if len(formula_lines) == 1 else 0.43
    for idx, line in enumerate(formula_lines):
        ax.text(
            0.0,
            y0 - 0.18 * idx,
            line,
            ha="left",
            va="center",
            fontsize=6.25,
            color=COLORS["dark"],
            transform=ax.transAxes,
        )


def format_metric_value(metric: MetricSpec, value: float) -> str:
    return metric.fmt.format(value)


def add_gradient_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    cmap: LinearSegmentedColormap,
    linewidth: float = 1.9,
) -> None:
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=float(np.min(y)), vmax=float(np.max(y)))
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth, zorder=3)
    lc.set_array(0.5 * (y[:-1] + y[1:]))
    ax.add_collection(lc)


def annotate_low_base_high(ax: plt.Axes, rows: list[dict[str, Any]], metric: MetricSpec, color: str) -> None:
    y = values_for(rows, "parameter_value_display")
    x = values_for(rows, metric.key)
    labels = ((0, "low"), (len(rows) // 2, "base"), (len(rows) - 1, "high"))
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    for idx, tag in labels:
        ax.scatter([x[idx]], [y[idx]], s=15, color=color, edgecolor="white", linewidth=0.45, zorder=4)
        x_text = x[idx]
        y_text = y[idx]
        ha = "left"
        offset = (3, 0)
        if metric.scale == "log":
            xpos = (np.log10(x[idx]) - np.log10(x_min)) / (np.log10(x_max) - np.log10(x_min))
        else:
            xpos = (x[idx] - x_min) / (x_max - x_min)
        if xpos > 0.72:
            ha = "right"
            offset = (-3, 0)
        va = "center"
        if idx == 0:
            va = "bottom"
        elif idx == len(rows) - 1:
            va = "top"
        text = f"{tag}: {format_metric_value(metric, x[idx])}"
        ax.annotate(
            text,
            xy=(x_text, y_text),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=5.35,
            color=COLORS["dark"],
            zorder=5,
            clip_on=False,
        )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def style_metric_axis(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    spec: ScanSpec,
    metric: MetricSpec,
    xlim: tuple[float, float],
    row_idx: int,
    col_idx: int,
    show_column_title: bool | None = None,
    show_x_label: bool | None = None,
) -> None:
    if show_column_title is None:
        show_column_title = row_idx == 0
    if show_x_label is None:
        show_x_label = row_idx == len(METRIC_SPECS) - 1

    y_values = values_for(rows, "parameter_value_display")
    y_baseline = spec.display_factor * spec.baseline
    ax.set_xscale(metric.scale)
    ax.set_yscale(spec.scale)
    ax.set_xlim(*xlim)
    ax.set_ylim(float(np.min(y_values)), float(np.max(y_values)))
    ax.axhline(y_baseline, color=COLORS["baseline"], linewidth=0.8, linestyle=(0, (2, 2)), alpha=0.9, zorder=2)

    if metric.no_edl_key is not None:
        no_edl_val = float(rows[0][metric.no_edl_key])
        ax.axvline(no_edl_val, color=COLORS["no_edl"], linewidth=0.85, linestyle=(0, (3, 2)), alpha=0.95, zorder=1)
        if col_idx == 0:
            ax.text(
                no_edl_val,
                1.015,
                "w/o EDL",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=5.3,
                color=COLORS["no_edl"],
                rotation=90,
            )

    ax.tick_params(axis="both", length=2.4, width=0.65, pad=1.5, labelsize=5.9)
    ax.tick_params(axis="x", labelbottom=show_x_label)
    for spine in ("left", "right", "top", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.72)
        ax.spines[spine].set_color(COLORS["dark"])

    if show_column_title:
        ax.set_title(
            parameter_axis_label(spec),
            loc="center",
            fontsize=8.2,
            pad=6.0,
            color=COLORS["dark"],
        )

    if col_idx == 0:
        ax.set_ylabel(metric_label(metric, spec), fontsize=6.9, labelpad=3.8)
    else:
        ax.set_ylabel("")

    if show_x_label:
        ax.set_xlabel(metric_label(metric, spec), fontsize=6.6, labelpad=2.3)
    else:
        ax.set_xlabel("")

    ticks = [spec.display_factor * spec.low, spec.display_factor * spec.baseline, spec.display_factor * spec.high]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _pos: spec.display_fmt.format(val)))
    if metric.scale == "linear":
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))


def schematic_case_appearance(case_name: str) -> tuple[float, str | tuple[int, tuple[int, ...]], str]:
    for name, alpha, linestyle in SCHEMATIC_CASE_SPECS:
        if case_name == name:
            return alpha, linestyle, COLORS[f"case_{name}"]
    raise ValueError(f"Unsupported schematic case: {case_name}")


def schematic_case_label(spec: ScanSpec, case: dict[str, Any]) -> str:
    value = spec.display_fmt.format(float(case["parameter_value_display"]))
    return f"{case['case']}: {value} {spec.display_unit}"


def signed_current_key(side: str) -> str:
    if side == "Au":
        return "I_Au_avg_A_per_m2"
    if side == "Pd":
        return "I_Pd_avg_A_per_m2"
    raise ValueError(f"Unsupported side: {side}")


def side_color(side: str) -> str:
    if side == "Au":
        return COLORS["au_curve"]
    if side == "Pd":
        return COLORS["pd_curve"]
    raise ValueError(f"Unsupported side: {side}")


def side_label(side: str) -> str:
    return "Au anodic" if side == "Au" else "Pd cathodic"


def opposite_side(side: str) -> str:
    if side == "Au":
        return "Pd"
    if side == "Pd":
        return "Au"
    raise ValueError(f"Unsupported side: {side}")


def add_gradient_line_by_values(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color_values: np.ndarray,
    cmap: LinearSegmentedColormap,
    linewidth: float = 1.8,
) -> None:
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=float(np.min(color_values)), vmax=float(np.max(color_values)))
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth, zorder=3)
    lc.set_array(0.5 * (color_values[:-1] + color_values[1:]))
    ax.add_collection(lc)


def parameter_xlim(spec: ScanSpec) -> tuple[float, float]:
    low = spec.display_factor * spec.low
    high = spec.display_factor * spec.high
    if spec.scale == "log":
        return low / 1.05, high * 1.05
    pad = 0.045 * (high - low)
    return low - pad, high + pad


def format_parameter_axis(ax: plt.Axes, spec: ScanSpec) -> None:
    ticks = [spec.display_factor * spec.low, spec.display_factor * spec.baseline, spec.display_factor * spec.high]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _pos: spec.display_fmt.format(val)))


def annotate_causal_low_base_high(ax: plt.Axes, rows: list[dict[str, Any]], metric: MetricSpec, color: str) -> None:
    x = values_for(rows, "parameter_value_display")
    y = values_for(rows, metric.key)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    for idx, tag in ((0, "L"), (len(rows) // 2, "B"), (len(rows) - 1, "H")):
        ax.scatter([x[idx]], [y[idx]], s=20, color=color, edgecolor="white", linewidth=0.5, zorder=5)
        if spec_scale_from_axis(ax) == "log":
            xpos = (np.log10(x[idx]) - np.log10(x_min)) / (np.log10(x_max) - np.log10(x_min))
        else:
            xpos = (x[idx] - x_min) / (x_max - x_min)
        if metric.scale == "log":
            ypos = (np.log10(y[idx]) - np.log10(y_min)) / (np.log10(y_max) - np.log10(y_min))
        else:
            ypos = (y[idx] - y_min) / (y_max - y_min)

        x_offset = -4 if xpos > 0.70 else 4
        y_offset = -5 if ypos > 0.72 else 5
        xytext = (x_offset, y_offset)
        ha = "right" if x_offset < 0 else "left"
        va = "top" if y_offset < 0 else "bottom"
        ax.annotate(
            f"{tag}: {format_metric_value(metric, y[idx])}",
            xy=(x[idx], y[idx]),
            xytext=xytext,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=7.2,
            color=COLORS["dark"],
            clip_on=False,
            zorder=10,
        )


def spec_scale_from_axis(ax: plt.Axes) -> str:
    return "log" if ax.get_xscale() == "log" else "linear"


def data_fraction(value: float, limits: tuple[float, float], scale: str) -> float:
    lo, hi = limits
    if scale == "log":
        return (np.log10(value) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    return (value - lo) / (hi - lo)


def formula_text_position(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> tuple[float, float, str]:
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    x_scale = spec_scale_from_axis(ax)
    y_scale = "log" if ax.get_yscale() == "log" else "linear"
    point_positions = [
        (
            data_fraction(float(x[idx]), x_limits, x_scale),
            data_fraction(float(y[idx]), y_limits, y_scale),
        )
        for idx in (0, len(x) // 2, len(x) - 1)
    ]
    lower_left_is_busy = any(xpos < 0.42 and ypos < 0.38 for xpos, ypos in point_positions)
    if lower_left_is_busy:
        return 0.02, 0.95, "top"
    return 0.02, 0.04, "bottom"


def plot_causal_tile(
    ax: plt.Axes,
    scan_rows: list[dict[str, Any]],
    spec: ScanSpec,
    metric: MetricSpec,
    cmap: LinearSegmentedColormap,
    marker_color: str,
    show_xlabel: bool,
) -> None:
    x = values_for(scan_rows, "parameter_value_display")
    y = values_for(scan_rows, metric.key)
    ax.set_xscale(spec.scale)
    ax.set_yscale(metric.scale)
    ax.set_xlim(*parameter_xlim(spec))
    ax.set_ylim(*metric_xlim(scan_rows, metric))
    ax.axvline(
        spec.display_factor * spec.baseline,
        color=COLORS["baseline"],
        linewidth=0.75,
        linestyle=(0, (2, 2)),
        alpha=0.9,
        zorder=1,
    )
    if metric.no_edl_key is not None:
        no_edl_val = float(scan_rows[0][metric.no_edl_key])
        ax.axhline(
            no_edl_val,
            color=COLORS["no_edl"],
            linewidth=0.75,
            linestyle=(0, (3, 2)),
            alpha=0.8,
            zorder=1,
        )

    add_gradient_line_by_values(ax, x, y, x, cmap, linewidth=2.05)
    annotate_causal_low_base_high(ax, scan_rows, metric, marker_color)
    format_parameter_axis(ax, spec)
    ax.set_title(metric_label(metric, spec), loc="left", fontsize=11.0, pad=3.8, color=COLORS["dark"])
    if show_xlabel:
        ax.set_xlabel(f"{spec.title} ({spec.display_unit})", fontsize=9.6, labelpad=2.2)
    else:
        ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", length=3.0, width=0.78, pad=1.9, labelsize=9.0)
    ax.tick_params(axis="x", labelbottom=show_xlabel)
    if metric.scale == "linear":
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    formula = formula_for_metric(metric, spec)
    formula_x, formula_y, formula_va = formula_text_position(ax, x, y)
    ax.text(
        formula_x,
        formula_y,
        formula,
        transform=ax.transAxes,
        ha="left",
        va=formula_va,
        fontsize=7.7,
        color=COLORS["gray"],
        linespacing=1.02,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=1.3),
        zorder=5,
    )
    for spine in ("left", "right", "top", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.72)
        ax.spines[spine].set_color(COLORS["dark"])


def case_by_name(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(case["case"]): case for case in cases}


def polarization_ylim(cases: list[dict[str, Any]]) -> float:
    marker_currents = np.asarray(
        [
            value
            for case in cases
            for value in (
                float(case["I_Au_current_1e_minus_3_uA"]),
                float(case["I_Pd_current_1e_minus_3_uA"]),
                float(case["i_mix_current_1e_minus_3_uA"]),
            )
        ],
        dtype=float,
    )
    return max(0.13, 1.65 * float(np.max(np.abs(marker_currents))))


def polarization_summary_lines(spec: ScanSpec, cases: list[dict[str, Any]]) -> list[str]:
    coupled_side = opposite_side(spec.side)
    named = case_by_name(cases)
    e_low = float(named["low"]["E_mix"])
    e_high = float(named["high"]["E_mix"])
    lines = [rf"True EDL curves", rf"Only {spec.title} changes"]
    for case_name in ("low", "base", "high"):
        case = named[case_name]
        param_value = spec.display_fmt.format(float(case["parameter_value_display"]))
        lines.append(
            rf"{case_name}: {param_value}, "
            rf"$E$={float(case['E_mix']):.3f} V, "
            rf"$|I|$={float(case['i_mix_current_1e_minus_3_uA']):.2g}"
        )
    i_low = float(named["low"]["i_mix_current_1e_minus_3_uA"])
    i_high = float(named["high"]["i_mix_current_1e_minus_3_uA"])
    if i_low > 0.0:
        lines.append(rf"low$\to$high: $\Delta E$={e_high - e_low:+.3f} V, $|I|\times$={i_high / i_low:.2g}")
    rel_key = f"max_abs_rel_delta_I_{coupled_side}_vs_base"
    low_coupling_pct = 100.0 * float(named["low"][rel_key])
    high_coupling_pct = 100.0 * float(named["high"][rel_key])
    lines.append(rf"Opposite {coupled_side}:")
    lines.append(rf"max rel {low_coupling_pct:.3g}% (low), {high_coupling_pct:.3g}% (high)")
    return lines


def plot_polarization_summary_panel(ax: plt.Axes, spec: ScanSpec, cases: list[dict[str, Any]]) -> None:
    ax.set_axis_off()
    ax.text(
        0.0,
        0.98,
        "Curve parameters",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        color=COLORS["dark"],
    )
    ax.text(
        0.0,
        0.84,
        "\n".join(polarization_summary_lines(spec, cases)),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color=COLORS["dark"],
        linespacing=1.25,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor=COLORS["light_gray"], linewidth=0.65, alpha=0.94),
    )


def plot_true_polarization_panel(ax: plt.Axes, spec: ScanSpec, cases: list[dict[str, Any]]) -> None:
    font_scale = polarization_font_scale(spec)
    varied_side = spec.side
    coupled_side = opposite_side(spec.side)
    y_limit = polarization_ylim(cases)
    e_values = np.asarray(cases[0]["E"], dtype=float)
    ax.axhline(0.0, color=COLORS["dark"], linewidth=0.78, zorder=1)

    for side in ("Au", "Pd"):
        curve_key = f"I_{side}_current_curve"
        is_varied = side == varied_side
        for case in cases:
            case_name = str(case["case"])
            alpha, linestyle, case_color = schematic_case_appearance(case_name)
            linewidth = 2.35 if is_varied else 1.2
            case_alpha = alpha if is_varied else (0.36 if case_name != "base" else 0.55)
            ax.plot(
                np.asarray(case["E"], dtype=float),
                np.asarray(case[curve_key], dtype=float),
                color=side_color(side),
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=case_alpha,
                zorder=5 if is_varied else 3,
            )

            e_mix = float(case["E_mix"])
            ax.axvline(e_mix, color=case_color, linewidth=0.85, linestyle=linestyle, alpha=alpha, zorder=2)

    for case in cases:
        case_name = str(case["case"])
        alpha, _linestyle, case_color = schematic_case_appearance(case_name)
        e_mix = float(case["E_mix"])
        ax.scatter(
            [e_mix, e_mix],
            [
                float(case["I_Au_current_1e_minus_3_uA"]),
                float(case["I_Pd_current_1e_minus_3_uA"]),
            ],
            s=30 if case_name == "base" else 22,
            color=[COLORS["au_curve"], COLORS["pd_curve"]],
            edgecolor="white",
            linewidth=0.5,
            alpha=min(1.0, alpha + 0.10),
            zorder=8,
        )
        ax.scatter(
            [e_mix],
            [0.0],
            s=28 if case_name == "base" else 20,
            color=case_color,
            edgecolor="white",
            linewidth=0.5,
            alpha=min(1.0, alpha + 0.10),
            zorder=9,
        )

    named = case_by_name(cases)
    e_low = float(named["low"]["E_mix"])
    e_high = float(named["high"]["E_mix"])
    y_arrow = 0.80 * y_limit
    ax.annotate(
        "",
        xy=(e_high, y_arrow),
        xytext=(e_low, y_arrow),
        arrowprops=dict(arrowstyle="->", color=COLORS["dark"], linewidth=0.8, shrinkA=0, shrinkB=0),
        zorder=10,
    )
    ax.text(
        0.5 * (e_low + e_high),
        y_arrow + 0.04 * y_limit,
        r"$E_{\mathrm{mix}}$ shift",
        ha="center",
        va="bottom",
        fontsize=7.0 * font_scale,
        color=COLORS["dark"],
    )

    handles = [
        Line2D([0], [0], color=side_color(varied_side), linewidth=2.35, label=f"{side_label(varied_side)} highlighted"),
        Line2D([0], [0], color=side_color(coupled_side), linewidth=1.2, alpha=0.45, label=f"{side_label(coupled_side)} real coupling"),
        Line2D([0], [0], color=COLORS["case_low"], linewidth=1.3, linestyle=(0, (3, 2)), label="low"),
        Line2D([0], [0], color=COLORS["case_base"], linewidth=1.3, linestyle="-", label="base"),
        Line2D([0], [0], color=COLORS["case_high"], linewidth=1.3, linestyle=(0, (5, 2)), label="high"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=7.0 * font_scale,
        handlelength=2.4,
        ncols=1,
        columnspacing=0.85,
        labelspacing=0.34,
        borderaxespad=0.25,
    )
    ax.set_xlim(float(np.min(e_values)), float(np.max(e_values)))
    ax.set_ylim(-y_limit, y_limit)
    ax.set_title("True polarization curves", loc="left", fontsize=9.4 * font_scale, pad=4.0, color=COLORS["dark"])
    ax.set_xlabel("Potential (V vs. RHE)", fontsize=8.0 * font_scale, labelpad=2.5)
    ax.set_ylabel(PLOT_CURRENT_LABEL, fontsize=8.0 * font_scale, labelpad=3.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis="both", length=2.7, width=0.75, pad=1.6, labelsize=7.1 * font_scale)
    for spine in ("left", "right", "top", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.74)
        ax.spines[spine].set_color(COLORS["dark"])


def plot_process_gradients(rows: list[dict[str, Any]]) -> None:
    apply_style()
    metric_xlims = {metric.key: metric_xlim(rows, metric) for metric in METRIC_SPECS}

    fig = plt.figure(figsize=(14.2, 9.8), constrained_layout=False)
    gs = fig.add_gridspec(
        len(METRIC_SPECS),
        len(SCAN_SPECS) + 1,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 0.92],
    )
    axes = np.empty((len(METRIC_SPECS), len(SCAN_SPECS)), dtype=object)
    for row_idx, metric in enumerate(METRIC_SPECS):
        formula_ax = fig.add_subplot(gs[row_idx, len(SCAN_SPECS)])
        style_formula_axis(formula_ax, metric, formula_for_metric(metric), show_title=row_idx == 0)

    for col_idx, spec in enumerate(SCAN_SPECS):
        scan_rows = rows_for_scan(rows, spec.key)
        y = values_for(scan_rows, "parameter_value_display")
        cmap = AU_CMAP if spec.side == "Au" else PD_CMAP
        marker_color = COLORS["au_dark"] if spec.side == "Au" else COLORS["pd_dark"]

        for row_idx, metric in enumerate(METRIC_SPECS):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            axes[row_idx, col_idx] = ax
            x = values_for(scan_rows, metric.key)
            style_metric_axis(ax, scan_rows, spec, metric, metric_xlims[metric.key], row_idx, col_idx)
            add_gradient_line(ax, x, y, cmap)
            annotate_low_base_high(ax, scan_rows, metric, marker_color)

            if row_idx == 0:
                ax.text(
                    0.02,
                    0.98,
                    f"track {spec.side}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=5.7,
                    color=marker_color,
                )

    fig.suptitle(
        "Figure 5 mechanism: parameter changes propagate through EDL charging to mixed-potential outputs",
        x=0.02,
        y=0.992,
        ha="left",
        va="top",
        fontsize=11.0,
        color=COLORS["dark"],
    )
    fig.text(
        0.02,
        0.966,
        "Each column varies one parameter only. Dashed horizontal lines mark the Figure 5 baseline; dashed vertical lines in the bottom rows mark the w/o EDL output.",
        ha="left",
        va="top",
        fontsize=7.0,
        color=COLORS["gray"],
    )
    fig.text(
        0.515,
        0.028,
        "Scan direction is vertical from low to high parameter value; low/base/high labels give the exact row-quantity values at those three points.",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=COLORS["gray"],
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.075, top=0.925, hspace=0.48, wspace=0.25)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_OUT, dpi=600, bbox_inches="tight")
    fig.savefig(SVG_OUT, bbox_inches="tight")
    plt.close(fig)


def mechanism_output_paths(spec: ScanSpec) -> tuple[Path, Path]:
    stem = MECHANISM_FIGURE_STEMS[spec.key]
    return OUT_DIR / f"{stem}_{OUTPUT_TAG}.png", OUT_DIR / f"{stem}_{OUTPUT_TAG}.svg"


def polarization_output_paths(spec: ScanSpec) -> tuple[Path, Path]:
    stem = POLARIZATION_FIGURE_STEMS[spec.key]
    return OUT_DIR / f"{stem}_{OUTPUT_TAG}.png", OUT_DIR / f"{stem}_{OUTPUT_TAG}.svg"


def plot_single_mechanism_figure(
    rows: list[dict[str, Any]],
    spec: ScanSpec,
) -> tuple[Path, Path]:
    scan_rows = rows_for_scan(rows, spec.key)
    cmap = AU_CMAP if spec.side == "Au" else PD_CMAP
    marker_color = COLORS["au_dark"] if spec.side == "Au" else COLORS["pd_dark"]

    fig = plt.figure(figsize=MECHANISM_FIGSIZE, constrained_layout=False)
    gs = fig.add_gridspec(len(METRIC_SPECS), 1, hspace=0.53)

    for row_idx, metric in enumerate(METRIC_SPECS):
        ax = fig.add_subplot(gs[row_idx, 0])
        plot_causal_tile(
            ax,
            scan_rows,
            spec,
            metric,
            cmap,
            marker_color,
            show_xlabel=row_idx == len(METRIC_SPECS) - 1,
        )
        if row_idx == 0:
            ax.text(
                0.98,
                0.96,
                f"track {spec.side}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.6,
                color=marker_color,
            )

    fig.suptitle(
        MECHANISM_FIGURE_TITLES[spec.key].replace("Figure 5 mechanism: ", "Figure 5 mechanism:\n"),
        x=0.065,
        y=0.992,
        ha="left",
        va="top",
        fontsize=12.6,
        color=COLORS["dark"],
        linespacing=1.0,
    )
    fig.text(
        0.065,
        0.935,
        f"Only {spec.title} is scanned.\nPanels are side-averaged over the tracked active region.",
        ha="left",
        va="top",
        fontsize=7.9,
        color=COLORS["gray"],
        linespacing=1.05,
    )
    fig.text(
        0.50,
        0.018,
        "L, B, and H mark low, baseline, and high scan values.",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color=COLORS["gray"],
    )
    fig.subplots_adjust(left=0.16, right=0.965, bottom=0.075, top=0.865)

    png_path, svg_path = mechanism_output_paths(spec)
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def plot_single_polarization_figure(
    spec: ScanSpec,
    curve_cases: list[dict[str, Any]],
) -> tuple[Path, Path]:
    font_scale = polarization_font_scale(spec)
    fig, ax = plt.subplots(figsize=polarization_figsize(spec), constrained_layout=False)
    plot_true_polarization_panel(ax, spec, curve_cases)
    ax.set_title(POLARIZATION_FIGURE_TITLES[spec.key], loc="left", fontsize=9.6 * font_scale, pad=4.0, color=COLORS["dark"])
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.19, top=0.88)

    png_path, svg_path = polarization_output_paths(spec)
    fig.savefig(png_path, dpi=450)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def plot_mechanism_and_polarization_figures(
    rows: list[dict[str, Any]],
    curve_data: dict[str, list[dict[str, Any]]],
) -> list[Path]:
    saved: list[Path] = []
    for spec in SCAN_SPECS:
        mechanism_png, mechanism_svg = plot_single_mechanism_figure(rows, spec)
        polarization_png, polarization_svg = plot_single_polarization_figure(spec, curve_data[spec.key])
        saved.extend([mechanism_png, mechanism_svg, polarization_png, polarization_svg])
    return saved


def cleanup_stale_column_exports() -> None:
    for path in sorted(OUT_DIR.glob(f"figure_5_column_*_{OUTPUT_TAG}.*")):
        if path.suffix.lower() in {".png", ".svg"}:
            path.unlink()


def assert_outputs() -> None:
    expected_exports = {PNG_OUT.resolve(), SVG_OUT.resolve()}
    for spec in SCAN_SPECS:
        expected_exports.update(path.resolve() for path in mechanism_output_paths(spec))
        expected_exports.update(path.resolve() for path in polarization_output_paths(spec))
    actual_exports = {
        path.resolve()
        for path in OUT_DIR.glob(f"*_{OUTPUT_TAG}.*")
        if path.suffix.lower() in {".png", ".svg"}
    }
    if actual_exports != expected_exports:
        raise RuntimeError(f"Unexpected gradient exports: {sorted(str(path) for path in actual_exports)}")
    for path in (CSV_OUT, SCHEMATIC_CSV_OUT, CURVE_CSV_OUT):
        if not path.exists():
            raise RuntimeError(f"Expected CSV output was not generated: {path}")
    pdfs = sorted(OUT_DIR.glob("*.pdf"))
    if pdfs:
        raise RuntimeError(f"Expected no PDF outputs, found: {pdfs}")


def main() -> None:
    params = load_json(PARAMS_PATH)
    summary = load_json(SUMMARY_PATH)
    cleanup_stale_column_exports()
    _, no_edl = validate_baseline(params, summary)
    rows = build_sweep_rows(params, no_edl)
    require_finite_rows(rows)
    schematic = build_polarization_schematic_data(params)
    require_finite_schematic(schematic)
    curve_data = build_polarization_curve_data(schematic)
    require_finite_curve_data(curve_data)
    write_csv(rows)
    write_schematic_csv(schematic)
    write_polarization_curve_csv(curve_data)
    plot_process_gradients(rows)
    small_paths = plot_mechanism_and_polarization_figures(rows, curve_data)
    assert_outputs()

    print(f"Verified baseline E_mix_with = {float(summary['E_mix_with']):.15g} V")
    print(f"Verified baseline i_mix_avg_with = {float(summary['i_mix_avg_with']):.15g} A/m^2")
    print(f"Saved CSV rows = {len(rows)}")
    print(f"Saved schematic CSV rows = {len(SCAN_SPECS) * len(SCHEMATIC_CASE_SPECS)}")
    print(f"Saved polarization curve CSV rows = {len(SCAN_SPECS) * len(SCHEMATIC_CASE_SPECS) * N_POLARIZATION_POINTS}")
    print(f"Saved {PNG_OUT}")
    print(f"Saved {SVG_OUT}")
    for path in small_paths:
        print(f"Saved {path}")
    print(f"Saved {CSV_OUT}")
    print(f"Saved {SCHEMATIC_CSV_OUT}")
    print(f"Saved {CURVE_CSV_OUT}")


if __name__ == "__main__":
    main()
