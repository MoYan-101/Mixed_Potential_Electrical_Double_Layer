from __future__ import annotations

import copy
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import curve_fit


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = OUT_DIR / "inputs"
CSV_DIR = OUT_DIR / "csv"
OFAT_FIGURES_DIR = OUT_DIR / "OFAT"
CASE_FIGURES_DIR = OUT_DIR / "Case_Figures"

sys.path.insert(0, str(ROOT))

from Figures.Figure_3 import make_figure_3_panels as panel_base  # noqa: E402
from Figures.Figure_same_length_i0_alpha import same_length_i0_alpha_common as common  # noqa: E402
from Figures.Figrue_RP import make_phi_s_reactants_2d as rp_base  # noqa: E402


solver = common.solver
plt = panel_base.plt

L_SUPPORT_NM_VALUES = (0.0, 2.0, 3.0, 10.0, 1000.0)
ACTIVE_ZOOM_SUPPORT_NM = 1000.0
ACTIVE_WINDOW_NM = 60.0
STANDARD_N_MODES = 960
STANDARD_NX = 5000
LONG_N_MODES = 3840
LONG_NX = 7000
N_Y_2D = 320
MODE_CHUNK_SIZE = 512

OFAT_L_SUPPORT_NM_VALUES = (
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
OVERLAP_EXTRA_L_SUPPORT_NM_VALUES = (11.0, 12.0, 15.0)
OVERLAP_L_SUPPORT_NM_VALUES = OFAT_L_SUPPORT_NM_VALUES + OVERLAP_EXTRA_L_SUPPORT_NM_VALUES
OVERLAP_REFERENCE_NM = 1000.0
OVERLAP_RESIDUAL_FRACTION = 0.05
OVERLAP_E_TOLERANCE_MV = 0.1
OVERLAP_PHI_RP_RMS_TOLERANCE_MV = 0.1
OVERLAP_I_REL_TOLERANCE_PERCENT = 0.1
PLATEAU_FIT_MIN_NM = 5.0
PLATEAU_TOLERANCE_V = 0.1e-3
PLATEAU_CHECKPOINTS_NM = (30.0, 100.0, 1000.0)
PLATEAU_CHECKPOINT_TOLERANCE_V = 0.05e-3

CURRENT_EXPONENT = -3
PANEL_E_COMMON_YMIN = -400.0
FULL_PANEL_FIGSIZE = (5.8, 2.55)
FULL_PANEL_AXES_RECT = (0.12, 0.22, 0.85, 0.62)
EXPECTED_CASE_OUTPUT_PAIRS = 24
EXPECTED_OFAT_OUTPUT_PAIRS = 8
EXPECTED_OUTPUT_PAIRS = EXPECTED_CASE_OUTPUT_PAIRS + EXPECTED_OFAT_OUTPUT_PAIRS

OUTPUT_TAG = common.OUTPUT_TAG
OFAT_CSV = CSV_DIR / f"ofat_compare_L_support_dense_0_10nm_{OUTPUT_TAG}.csv"
CHECKPOINT_CSV = CSV_DIR / f"plateau_checkpoints_L_support_{OUTPUT_TAG}.csv"
PLATEAU_SUMMARY_CSV = CSV_DIR / f"plateau_length_scale_summary_{OUTPUT_TAG}.csv"
PLATEAU_SUMMARY_JSON = CSV_DIR / f"plateau_length_scale_summary_{OUTPUT_TAG}.json"
PHI_STATS_CSV = CSV_DIR / f"phi_s_stats_L_support_{OUTPUT_TAG}.csv"
SIGMA_PROFILES_CSV = CSV_DIR / f"sigma_profiles_L_support_{OUTPUT_TAG}.csv"
LENGTH_SCALE_CSV = CSV_DIR / f"length_scale_summary_L_support_{OUTPUT_TAG}.csv"
LENGTH_SCALE_JSON = CSV_DIR / f"length_scale_summary_L_support_{OUTPUT_TAG}.json"
OVERLAP_CSV = CSV_DIR / f"edl_overlap_vs_1000nm_L_support_{OUTPUT_TAG}.csv"
OVERLAP_SUMMARY_CSV = CSV_DIR / f"edl_overlap_summary_vs_1000nm_L_support_{OUTPUT_TAG}.csv"
OVERLAP_SUMMARY_JSON = CSV_DIR / f"edl_overlap_summary_vs_1000nm_L_support_{OUTPUT_TAG}.json"

COLORS = {
    "au": rp_base.COLORS["au"],
    "Au": rp_base.COLORS["au"],
    "support": rp_base.COLORS["support"],
    "pd": rp_base.COLORS["pd"],
    "Pd": rp_base.COLORS["pd"],
    "dark": panel_base.COLORS["dark"],
    "gray": panel_base.COLORS["gray"],
    "light_gray": panel_base.COLORS["light_gray"],
    "lambda": "#5A90C8",
    "lgc": "#8C8C8C",
    "plateau": "#D83A2E",
    "overlap_e": "#D83A2E",
    "overlap_phi": "#5A90C8",
    "overlap_i": "#4D8061",
    "current_i1": "#009E73",
    "current_i2": "#0072B2",
}


@dataclass(frozen=True)
class Phi2DData:
    params: dict[str, Any]
    res_edl: dict[str, Any]
    x_nm: np.ndarray
    y_nm: np.ndarray
    phi_s_mV: np.ndarray
    lambda_D_nm: float
    L_Au_nm: float
    L_C_nm: float
    L_total_nm: float


@dataclass(frozen=True)
class SigmaSegment:
    material: str
    x_nm: np.ndarray
    phi_rp_V: np.ndarray
    sigma_C_per_m2: np.ndarray


@dataclass(frozen=True)
class SigmaData:
    segments: tuple[SigmaSegment, ...]
    support_status: str
    sigma_C_signed_mean_C_per_m2: float | None
    sigma_C_mean_abs_C_per_m2: float | None
    sigma_C_midpoint_C_per_m2: float | None
    sigma_C_min_C_per_m2: float | None
    sigma_C_max_C_per_m2: float | None
    sigma_C_sign_change: bool | None
    L_GC_C_nm: float | None
    L_GC_C_midpoint_nm: float | None
    L_support_over_L_GC_C: float | None


@dataclass(frozen=True)
class LSupportCase:
    value_nm: float
    params: dict[str, Any]
    summary: dict[str, str]
    res_edl: dict[str, Any]
    res_no: dict[str, Any]
    panel_data: Any
    rp_data: Phi2DData
    sigma_data: SigmaData


def ensure_dirs() -> None:
    for path in (OUT_DIR, INPUTS_DIR, CSV_DIR, OFAT_FIGURES_DIR, CASE_FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def configure_style() -> None:
    panel_base.apply_style()
    panel_base.SINGLE_PANEL_FIGSIZE = FULL_PANEL_FIGSIZE
    panel_base.SINGLE_PANEL_AXES_RECT = FULL_PANEL_AXES_RECT


def format_nm_value(value_nm: float) -> str:
    if math.isclose(value_nm, round(value_nm), rel_tol=0.0, abs_tol=1e-9):
        return f"{int(round(value_nm))}"
    return f"{value_nm:.3g}"


def output_tag(value_nm: float) -> str:
    return f"L_support_{format_nm_value(value_nm).replace('.', 'p')}nm_{OUTPUT_TAG}"


def is_case(value_nm: float, target_nm: float) -> bool:
    return math.isclose(value_nm, target_nm, rel_tol=0.0, abs_tol=1e-9)


def resolution_for_l_support(value_nm: float) -> tuple[int, int]:
    if is_case(value_nm, ACTIVE_ZOOM_SUPPORT_NM):
        return LONG_N_MODES, LONG_NX
    return STANDARD_N_MODES, STANDARD_NX


def params_for_l_support(value_nm: float, *, long_resolution: bool | None = None) -> dict[str, Any]:
    params = common.load_same_length_i0_alpha_params()
    params["L_gap"] = float(value_nm) * 1.0e-9
    if long_resolution is None:
        n_modes, nx = resolution_for_l_support(value_nm)
    elif long_resolution:
        n_modes, nx = LONG_N_MODES, LONG_NX
    else:
        n_modes, nx = STANDARD_N_MODES, STANDARD_NX
    params["N_modes"] = n_modes
    params["Nx"] = nx
    validate_case_params(params, value_nm)
    return params


def validate_case_params(params: dict[str, Any], expected_l_support_nm: float) -> None:
    checks = {
        "L_Au": common.L_AU_SAME,
        "L_gap": expected_l_support_nm * 1.0e-9,
        "L_Pd_len": common.L_PD_SAME,
        "it0_1": common.I0_GEOM,
        "it0_2": common.I0_GEOM,
        "alpha1": common.ALPHA_EQUAL,
        "alpha2": common.ALPHA_EQUAL,
        "out_of_plane_width": common.OUT_OF_PLANE_WIDTH,
    }
    for key, expected in checks.items():
        actual = float(params[key])
        atol = max(1e-15, abs(expected) * 1e-12)
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol):
            raise ValueError(f"{key} should be {expected:.15g}, got {actual:.15g}")


def _safe_unlink(path: Path, allowed_root: Path) -> None:
    root_resolved = allowed_root.resolve()
    path_resolved = path.resolve(strict=False)
    try:
        path_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to remove path outside {root_resolved}: {path}") from exc
    if path.is_symlink():
        raise RuntimeError(f"Refusing to remove symlink: {path}")
    if not path.is_file():
        raise RuntimeError(f"Expected generated file, got non-file: {path}")
    path.unlink()


def remove_previous_generated_outputs() -> list[Path]:
    removed: list[Path] = []
    out_patterns = (
        "solution_phase_potential_2d_L_support_*.png",
        "solution_phase_potential_2d_L_support_*.svg",
        "solution_phase_potential_2d_active_zoom_L_support_*.png",
        "solution_phase_potential_2d_active_zoom_L_support_*.svg",
        "figure_3_panel_b_reaction_plane_potential_L_support_*.png",
        "figure_3_panel_b_reaction_plane_potential_L_support_*.svg",
        "figure_3_panel_e_local_current_density_L_support_*.png",
        "figure_3_panel_e_local_current_density_L_support_*.svg",
        "figure_3_panel_b_reaction_plane_potential_active_zoom_L_support_*.png",
        "figure_3_panel_b_reaction_plane_potential_active_zoom_L_support_*.svg",
        "figure_3_panel_e_local_current_density_active_zoom_L_support_*.png",
        "figure_3_panel_e_local_current_density_active_zoom_L_support_*.svg",
        "surface_charge_distribution_L_support_*.png",
        "surface_charge_distribution_L_support_*.svg",
        "surface_charge_distribution_active_zoom_L_support_*.png",
        "surface_charge_distribution_active_zoom_L_support_*.svg",
        "ofat_compare_L_gap*.png",
        "ofat_compare_L_gap*.svg",
        "ofat_compare_L_support*.png",
        "ofat_compare_L_support*.svg",
        "l_support_polarization_curves_*.png",
        "l_support_polarization_curves_*.svg",
    )
    input_patterns = (
        "params_L_support_*.json",
        "overrides_L_support_*.json",
        "summary_compare_L_support_*.csv",
        "summary_compare_L_support_*.json",
    )
    csv_patterns = (
        "phi_s_stats_L_support_*.csv",
        "ofat_compare_L_gap_highN_*.csv",
        "ofat_compare_L_support_*.csv",
        "l_support_polarization_curves_*.csv",
        "sigma_profiles_L_support_*.csv",
        "length_scale_summary_L_support_*.csv",
        "length_scale_summary_L_support_*.json",
        "plateau_checkpoints_L_support_*.csv",
        "plateau_length_scale_summary_*.csv",
        "plateau_length_scale_summary_*.json",
        "edl_overlap_vs_1000nm_L_support_*.csv",
        "edl_overlap_summary_vs_1000nm_L_support_*.csv",
        "edl_overlap_summary_vs_1000nm_L_support_*.json",
    )
    managed_groups = (
        (OUT_DIR, out_patterns),
        (OFAT_FIGURES_DIR, out_patterns),
        (CASE_FIGURES_DIR, out_patterns),
        (INPUTS_DIR, input_patterns),
        (CSV_DIR, csv_patterns),
    )
    for root, patterns in managed_groups:
        targets = sorted({path for pattern in patterns for path in root.glob(pattern)})
        for path in targets:
            _safe_unlink(path, root)
            removed.append(path)
    return removed


def _build_with_edl_result(params: dict[str, Any], *, return_profiles: bool) -> tuple[Any, dict[str, Any]]:
    edl = solver.EDLModel(params)
    use_affine = bool(params.get("use_affine_phi2", True))
    E_mix, i_mix, info = solver.solve_emix(
        edl=edl,
        params=params,
        mode="FULL",
        use_affine_phi2=use_affine,
        xtol=float(params.get("xtol", 1e-10)),
        max_bracket_expands=int(params.get("max_bracket_expands", 12)),
    )
    if not bool(info.get("converged", False)):
        raise RuntimeError(f"with-EDL root solve did not converge: {info}")

    out = solver._build_run_output(
        mode="FULL",
        E_mix=E_mix,
        i_mix=i_mix,
        info=info,
        derived=edl.derived,
        a1=float(edl.pre["a1"]),
        b1=float(edl.pre["b1"]),
        a2=float(edl.pre["a2"]),
        b2=float(edl.pre["b2"]),
    )
    currents = solver.full_mode_currents(float(E_mix), edl, params, return_profiles=return_profiles)
    out.update(currents)
    if return_profiles:
        phi2_1_mean, phi2_2_mean = edl.segment_mean_phi2(float(E_mix), use_affine_phi2=False)
        out["phi2_1_meanV"] = float(phi2_1_mean)
        out["phi2_2_meanV"] = float(phi2_2_mean)
        phi_for_dh = np.asarray(out["phi_tilde"], dtype=float)
    else:
        _, phi_for_dh = edl.phi_tilde_surface(float(E_mix))
    out.update(solver._compute_dh_status_from_phi(phi_for_dh, params))
    solver._attach_current_unit_outputs(
        out,
        float(edl.derived["lambda_D"]),
        float(params.get("out_of_plane_width", 1.0)),
        float(edl.derived["L_Au"] + edl.derived["L_Pd_len"]),
    )
    out.update(solver._effective_reaction_params(params))
    solver._handle_dh_violation(out, params, mode="FULL", use_edl=True)
    return edl, out


def summary_from_results(res_edl: dict[str, Any], res_no: dict[str, Any]) -> dict[str, str]:
    comparison = solver._make_edl_comparison_metrics(res_edl, res_no, mode="FULL")
    return common.summary_from_pair({"with_edl": res_edl, "no_edl": res_no, "comparison": comparison})


def save_case_inputs(value_nm: float, params: dict[str, Any], summary: dict[str, str]) -> None:
    tag = output_tag(value_nm)
    overrides = {
        **common.PARAM_OVERRIDES,
        "L_gap": value_nm * 1.0e-9,
        "N_modes": int(params["N_modes"]),
        "Nx": int(params["Nx"]),
    }
    _write_json(INPUTS_DIR / f"params_{tag}.json", params)
    _write_json(INPUTS_DIR / f"overrides_{tag}.json", overrides)
    _write_csv(INPUTS_DIR / f"summary_compare_{tag}.csv", [summary])
    _write_json(INPUTS_DIR / f"summary_compare_{tag}.json", summary)


def build_panel_data(
    params: dict[str, Any],
    summary: dict[str, str],
    res_edl: dict[str, Any],
    res_no: dict[str, Any],
) -> Any:
    derived_edl = solver.compute_derived_params(params)
    derived_no = solver.compute_derived_params(params)
    scale = float(derived_edl["R"]) * float(derived_edl["T"]) / float(derived_edl["F"])

    x_edl = np.asarray(res_edl["x_tilde"], dtype=float)
    x_no = np.asarray(res_no["x_tilde"], dtype=float)
    if not np.allclose(x_edl, x_no, rtol=0.0, atol=1e-14):
        raise ValueError("with/w/o EDL surface grids do not match")
    x_nm = x_edl * float(derived_edl["lambda_D"]) * 1.0e9
    phi_rp_edl = scale * np.asarray(res_edl["phi_tilde"], dtype=float)
    phi_rp_no = scale * np.asarray(res_no["phi_tilde"], dtype=float)

    mask_au = np.asarray(res_edl["mask_Au"], dtype=bool)
    mask_pd = np.asarray(res_edl["mask_Pd"], dtype=bool)
    mask_au_no = np.asarray(res_no["mask_Au"], dtype=bool)
    mask_pd_no = np.asarray(res_no["mask_Pd"], dtype=bool)

    eta_edl = np.full_like(x_nm, np.nan, dtype=float)
    eta_no = np.full_like(x_nm, np.nan, dtype=float)
    eta_edl[mask_au] = float(res_edl["E_mix"]) - float(res_edl["E1_eq_eff"]) - phi_rp_edl[mask_au]
    eta_edl[mask_pd] = float(res_edl["E_mix"]) - float(res_edl["E2_eq_eff"]) - phi_rp_edl[mask_pd]
    eta_no[mask_au_no] = float(res_no["E_mix"]) - float(res_no["E1_eq_eff"]) - phi_rp_no[mask_au_no]
    eta_no[mask_pd_no] = float(res_no["E_mix"]) - float(res_no["E2_eq_eff"]) - phi_rp_no[mask_pd_no]

    phi_tilde = np.asarray(res_edl["phi_tilde"], dtype=float)
    c_r1_norm = np.asarray(solver.safe_exp(-float(params["z_R1"]) * phi_tilde), dtype=float)
    c_o2_norm = np.asarray(solver.safe_exp(-float(params["z_O2"]) * phi_tilde), dtype=float)

    def segmented(values: Any, mask: np.ndarray) -> np.ndarray:
        source = np.asarray(values, dtype=float)
        target = np.full_like(source, np.nan, dtype=float)
        target[mask] = source[mask]
        return target

    return panel_base.Figure3Data(
        params=copy.deepcopy(params),
        summary=dict(summary),
        res_edl=res_edl,
        res_no=res_no,
        prof_edl=res_edl,
        prof_no=res_no,
        derived_edl=derived_edl,
        derived_no=derived_no,
        x_nm=x_nm,
        phi_rp_edl=phi_rp_edl,
        phi_rp_no=phi_rp_no,
        eta_edl=eta_edl,
        eta_no=eta_no,
        c_r1_norm=c_r1_norm,
        c_o2_norm=c_o2_norm,
        i1_edl_segment=segmented(res_edl["i1"], mask_au),
        i2_edl_segment=segmented(res_edl["i2"], mask_pd),
        i1_no_segment=segmented(res_no["i1"], mask_au_no),
        i2_no_segment=segmented(res_no["i2"], mask_pd_no),
        L_Au_nm=float(derived_edl["L_Au"]) * 1.0e9,
        L_C_nm=float(derived_edl["L_C"]) * 1.0e9,
    )


def build_phi_2d_data(params: dict[str, Any], res_edl: dict[str, Any], edl: Any) -> Phi2DData:
    derived = edl.derived
    E_mix = float(res_edl["E_mix"])
    beta = float(derived["beta"])
    thermal_V = float(derived["R"]) * float(derived["T"]) / float(derived["F"])
    x_tilde = np.asarray(edl.pre["x_tilde"], dtype=float)
    profile_x = np.asarray(res_edl["x_tilde"], dtype=float)
    if not np.allclose(x_tilde, profile_x, rtol=0.0, atol=1e-14):
        raise ValueError("2D and reaction-plane grids do not match")

    rho = np.asarray(edl.pre["rho"], dtype=float)
    gamma = np.asarray(edl.pre["gamma"], dtype=float)
    coefficients = np.asarray(edl.pre["A_M"], dtype=float) * beta * E_mix - np.asarray(edl.pre["A_pzc"], dtype=float)
    lambda_D_nm = float(derived["lambda_D"]) * 1.0e9
    y_nm = np.linspace(0.0, 5.0 * lambda_D_nm, N_Y_2D, dtype=float)
    y_tilde = y_nm / lambda_D_nm
    decay_modes = coefficients[:, None] * np.exp(-gamma[:, None] * y_tilde[None, :])
    phi_tilde_2d = np.empty((N_Y_2D, x_tilde.size), dtype=float)
    for start in range(0, x_tilde.size, MODE_CHUNK_SIZE):
        stop = min(start + MODE_CHUNK_SIZE, x_tilde.size)
        cos_modes = np.cos(np.outer(rho, x_tilde[start:stop]))
        phi_tilde_2d[:, start:stop] = decay_modes.T @ cos_modes
    if not np.allclose(phi_tilde_2d[0], np.asarray(res_edl["phi_tilde"], dtype=float), rtol=0.0, atol=1e-10):
        max_error = float(np.max(np.abs(phi_tilde_2d[0] - np.asarray(res_edl["phi_tilde"], dtype=float))))
        raise ValueError(f"Phi_s(x,0) does not match reaction-plane profile; max error={max_error:.3g}")
    phi_s_mV = 1000.0 * thermal_V * phi_tilde_2d
    if not np.all(np.isfinite(phi_s_mV)):
        raise ValueError("2D Phi_s contains non-finite values")
    return Phi2DData(
        params=copy.deepcopy(params),
        res_edl=res_edl,
        x_nm=x_tilde * lambda_D_nm,
        y_nm=y_nm,
        phi_s_mV=phi_s_mV,
        lambda_D_nm=lambda_D_nm,
        L_Au_nm=float(derived["L_Au"]) * 1.0e9,
        L_C_nm=float(derived["L_C"]) * 1.0e9,
        L_total_nm=float(derived["L_total"]) * 1.0e9,
    )


def build_sigma_data(params: dict[str, Any], res_edl: dict[str, Any]) -> SigmaData:
    derived = solver.compute_derived_params(params)
    x_m = np.asarray(res_edl["x_tilde"], dtype=float) * float(derived["lambda_D"])
    x_nm = x_m * 1.0e9
    thermal_V = float(derived["R"]) * float(derived["T"]) / float(derived["F"])
    phi_rp = thermal_V * np.asarray(res_edl["phi_tilde"], dtype=float)
    E_mix = float(res_edl["E_mix"])
    eps_s = float(derived["epsilon_s"])
    lambda_D = float(derived["lambda_D"])
    specs = (
        ("Au", 0.0, float(derived["L_Au"]), "g_Au", "pzc_Au"),
        ("support", float(derived["L_Au"]), float(derived["L_C"]), "g_C", "pzc_C"),
        ("Pd", float(derived["L_C"]), float(derived["L_total"]), "g_Pd", "pzc_Pd"),
    )
    segments: list[SigmaSegment] = []
    support_x: np.ndarray | None = None
    support_sigma: np.ndarray | None = None
    for material, x0, x1, g_key, pzc_key in specs:
        if x1 - x0 <= 1e-18:
            continue
        mask = (x_m >= x0 - 1e-18) & (x_m <= x1 + 1e-18)
        C_H_eff = float(derived[g_key]) * eps_s / lambda_D
        sigma = C_H_eff * (E_mix - float(params[pzc_key]) - phi_rp[mask])
        segment = SigmaSegment(
            material=material,
            x_nm=x_nm[mask].copy(),
            phi_rp_V=phi_rp[mask].copy(),
            sigma_C_per_m2=np.asarray(sigma, dtype=float),
        )
        segments.append(segment)
        if material == "support":
            support_x = x_m[mask]
            support_sigma = np.asarray(sigma, dtype=float)

    support_width = float(derived["L_gap"])
    if support_width <= 1e-18 or support_x is None or support_sigma is None:
        return SigmaData(tuple(segments), "not_applicable", None, None, None, None, None, None, None, None, None)
    if support_x.size < 2:
        raise ValueError("Support sigma profile requires at least two grid points")

    signed_mean = float(np.trapezoid(support_sigma, support_x) / support_width)
    mean_abs = float(np.trapezoid(np.abs(support_sigma), support_x) / support_width)
    midpoint_x = float(derived["L_Au"] + 0.5 * derived["L_gap"])
    midpoint = float(np.interp(midpoint_x, x_m, _sigma_all_for_material(params, derived, E_mix, phi_rp, "support")))
    sigma_min = float(np.min(support_sigma))
    sigma_max = float(np.max(support_sigma))
    sign_change = bool(sigma_min < 0.0 < sigma_max)
    if not math.isfinite(mean_abs) or mean_abs <= 0.0:
        raise ValueError("Support mean absolute charge must be finite and positive")
    prefactor_m_C_per_m2 = 2.0 * eps_s * float(params["R"]) * float(params["T"]) / float(params["F"])
    L_GC_nm = prefactor_m_C_per_m2 / mean_abs * 1.0e9
    L_GC_mid_nm = prefactor_m_C_per_m2 / abs(midpoint) * 1.0e9 if midpoint != 0.0 else None
    return SigmaData(
        segments=tuple(segments),
        support_status="defined",
        sigma_C_signed_mean_C_per_m2=signed_mean,
        sigma_C_mean_abs_C_per_m2=mean_abs,
        sigma_C_midpoint_C_per_m2=midpoint,
        sigma_C_min_C_per_m2=sigma_min,
        sigma_C_max_C_per_m2=sigma_max,
        sigma_C_sign_change=sign_change,
        L_GC_C_nm=float(L_GC_nm),
        L_GC_C_midpoint_nm=None if L_GC_mid_nm is None else float(L_GC_mid_nm),
        L_support_over_L_GC_C=float(support_width * 1.0e9 / L_GC_nm),
    )


def _sigma_all_for_material(
    params: dict[str, Any],
    derived: dict[str, Any],
    E_mix: float,
    phi_rp: np.ndarray,
    material: str,
) -> np.ndarray:
    mapping = {
        "Au": ("g_Au", "pzc_Au"),
        "support": ("g_C", "pzc_C"),
        "Pd": ("g_Pd", "pzc_Pd"),
    }
    g_key, pzc_key = mapping[material]
    C_H_eff = float(derived[g_key]) * float(derived["epsilon_s"]) / float(derived["lambda_D"])
    return C_H_eff * (E_mix - float(params[pzc_key]) - phi_rp)


def build_case(value_nm: float) -> LSupportCase:
    params = params_for_l_support(value_nm)
    edl, res_edl = _build_with_edl_result(params, return_profiles=True)
    res_no = solver.run_case(params, mode="FULL", return_profiles=True, use_edl=False)
    summary = summary_from_results(res_edl, res_no)
    panel_data = build_panel_data(params, summary, res_edl, res_no)
    sigma_data = build_sigma_data(params, res_edl)
    rp_data = build_phi_2d_data(params, res_edl, edl)
    save_case_inputs(value_nm, params, summary)
    return LSupportCase(value_nm, params, summary, res_edl, res_no, panel_data, rp_data, sigma_data)


def build_cases() -> list[LSupportCase]:
    return [build_case(value_nm) for value_nm in L_SUPPORT_NM_VALUES]


def unique_ticks(values: Iterable[float]) -> list[float]:
    ticks: list[float] = []
    for value in values:
        value = float(value)
        if not any(math.isclose(value, existing, rel_tol=0.0, abs_tol=1e-8) for existing in ticks):
            ticks.append(value)
    return ticks


def format_tick(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-6):
        return f"{int(round(value))}"
    return f"{value:.1f}"


def x_ticks_for_panel(data: Any) -> list[float]:
    total = float(data.x_nm[-1])
    if total > 200.0:
        return [0.0, 250.0, 500.0, 750.0, total]
    return unique_ticks([0.0, float(data.L_Au_nm), float(data.L_C_nm), total])


def x_ticks_for_2d(data: Phi2DData) -> list[float]:
    if data.L_total_nm > 200.0:
        return [0.0, 250.0, 500.0, 750.0, data.L_total_nm]
    return unique_ticks([0.0, data.L_Au_nm, data.L_C_nm, data.L_total_nm])


def add_unique_boundaries(ax: Any, data: Any, *, zorder: int = 1) -> None:
    total = float(data.x_nm[-1]) if hasattr(data, "x_nm") else float(data.L_total_nm)
    for xpos in unique_ticks([float(data.L_Au_nm), float(data.L_C_nm)]):
        if 0.0 < xpos < total:
            ax.axvline(xpos, linestyle=(0, (3, 2)), linewidth=0.9, color=COLORS["gray"], alpha=0.85, zorder=zorder)


def finite_range(*arrays: Any, pad_frac: float = 0.08, include_zero: bool = False) -> tuple[float, float]:
    vals = np.concatenate([np.ravel(np.asarray(arr, dtype=float)) for arr in arrays])
    vals = vals[np.isfinite(vals)]
    if include_zero:
        vals = np.concatenate([vals, np.array([0.0])])
    if vals.size == 0:
        raise ValueError("Cannot build finite range from empty data")
    ymin = float(np.min(vals))
    ymax = float(np.max(vals))
    span = ymax - ymin
    pad = max(1e-6, pad_frac * span)
    return ymin - pad, ymax + pad


def common_phi_ylim(cases: list[LSupportCase]) -> tuple[float, float]:
    return finite_range(
        *(arr for case in cases for arr in (case.panel_data.phi_rp_edl, case.panel_data.phi_rp_no)),
        include_zero=True,
    )


def scale_current(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float) / (10.0**CURRENT_EXPONENT)


def scaled_current_arrays(data: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        scale_current(data.i1_edl_segment),
        scale_current(data.i1_no_segment),
        scale_current(data.i2_edl_segment),
        scale_current(data.i2_no_segment),
    )


def common_current_ylim(cases: list[LSupportCase]) -> tuple[float, float]:
    ymin, ymax = finite_range(*(arr for case in cases for arr in scaled_current_arrays(case.panel_data)), include_zero=True)
    return min(ymin, PANEL_E_COMMON_YMIN), ymax


def common_sigma_ylim(cases: list[LSupportCase]) -> tuple[float, float]:
    arrays = [100.0 * segment.sigma_C_per_m2 for case in cases for segment in case.sigma_data.segments]
    return finite_range(*arrays, include_zero=True)


def save_panel_figure(fig: Any, stem: str, *, output_dir: Path = CASE_FIGURES_DIR) -> list[Path]:
    panel_base.make_transparent(fig)
    saved: list[Path] = []
    for ext in ("png", "svg"):
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(
            path,
            dpi=600,
            transparent=True,
            facecolor="none",
            edgecolor="none",
            bbox_inches="tight",
            pad_inches=0.04,
        )
        saved.append(path)
    plt.close(fig)
    return saved


def save_map_figure(fig: Any, stem: str, *, output_dir: Path = CASE_FIGURES_DIR) -> list[Path]:
    saved: list[Path] = []
    for ext in ("png", "svg"):
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=600, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)
    return saved


def annotate_l_support(ax: Any, value_nm: float, *, loc: str = "upper_right") -> None:
    locations = {
        "upper_right": (0.985, 0.955, "right", "top"),
        "lower_left": (0.035, 0.055, "left", "bottom"),
        "lower_center": (0.500, 0.055, "center", "bottom"),
        "lower_right": (0.965, 0.055, "right", "bottom"),
    }
    if loc not in locations:
        raise ValueError(f"Unsupported annotation location: {loc}")
    x, y, ha, va = locations[loc]
    ax.text(
        x,
        y,
        rf"$L_{{\mathrm{{support}}}}$ = {format_nm_value(value_nm)} nm",
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=7.8,
        color=COLORS["dark"],
    )


def plot_panel_b_case(case: LSupportCase, ylim: tuple[float, float]) -> list[Path]:
    data = case.panel_data
    fig, ax = panel_base.make_single_axis_panel()
    ax.plot(data.x_nm, data.phi_rp_edl, color=panel_base.COLORS["with"], lw=2.0, label="with EDL", zorder=3)
    ax.plot(data.x_nm, data.phi_rp_no, color=panel_base.COLORS["without"], lw=1.8, label="w/o EDL", zorder=2)
    add_unique_boundaries(ax, data)
    panel_base.style_axes(ax, "x (nm)", r"$\phi_{\mathrm{RP}}(x)$ (V)", "Reaction-plane potential")
    ax.set_xlim(0.0, float(data.x_nm[-1]))
    ax.set_xticks(x_ticks_for_panel(data))
    ax.set_xticklabels([format_tick(tick) for tick in ax.get_xticks()])
    ax.set_ylim(*ylim)
    annotate_l_support(
        ax,
        case.value_nm,
        loc="lower_center" if is_case(case.value_nm, ACTIVE_ZOOM_SUPPORT_NM) else "upper_right",
    )
    ax.legend(loc="center right", bbox_to_anchor=(0.98, 0.50), fontsize=8.0, handlelength=2.0)
    return save_panel_figure(fig, f"figure_3_panel_b_reaction_plane_potential_{output_tag(case.value_nm)}")


def plot_panel_e_case(case: LSupportCase, ylim: tuple[float, float]) -> list[Path]:
    data = case.panel_data
    i1_edl, i1_no, i2_edl, i2_no = scaled_current_arrays(data)
    fig, ax = panel_base.make_single_axis_panel()
    ax.fill_between(data.x_nm, 0.0, i1_edl, where=np.isfinite(i1_edl), color=COLORS["current_i1"], alpha=0.34, lw=0.0)
    ax.fill_between(data.x_nm, 0.0, i2_edl, where=np.isfinite(i2_edl), color=COLORS["current_i2"], alpha=0.38, lw=0.0)
    ax.axhline(0.0, color=COLORS["dark"], lw=0.55, alpha=0.78, zorder=2)
    ax.plot(data.x_nm, i1_edl, color=COLORS["current_i1"], lw=2.0, label=r"$i_1$ (Au), with EDL", zorder=4)
    ax.plot(data.x_nm, i2_edl, color=COLORS["current_i2"], lw=2.0, label=r"$i_2$ (Pd), with EDL", zorder=4)
    ax.plot(data.x_nm, i1_no, color=COLORS["current_i1"], lw=1.7, ls=(0, (4, 2)), label=r"$i_1$ (Au), w/o EDL", zorder=3)
    ax.plot(data.x_nm, i2_no, color=COLORS["current_i2"], lw=1.7, ls=(0, (2, 2)), label=r"$i_2$ (Pd), w/o EDL", zorder=3)
    add_unique_boundaries(ax, data, zorder=5)
    panel_base.style_axes(ax, "x (nm)", r"$i(x)$ ($10^{-3}$ A/m$^2$)", "Local current density at RP")
    ax.set_xlim(0.0, float(data.x_nm[-1]))
    ax.set_xticks(x_ticks_for_panel(data))
    ax.set_xticklabels([format_tick(tick) for tick in ax.get_xticks()])
    ax.set_ylim(*ylim)
    annotate_l_support(ax, case.value_nm, loc="lower_left")
    ax.legend(loc="upper right", fontsize=6.7, handlelength=1.6)
    return save_panel_figure(fig, f"figure_3_panel_e_local_current_density_{output_tag(case.value_nm)}")


def zoom_windows(data: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    total = float(data.x_nm[-1])
    return (0.0, min(ACTIVE_WINDOW_NM, total)), (max(0.0, total - ACTIVE_WINDOW_NM), total)


def window_ticks(xmin: float, xmax: float, data: Any) -> list[float]:
    candidates = [xmin, xmax]
    for xpos in (float(data.L_Au_nm), float(data.L_C_nm)):
        if xmin <= xpos <= xmax:
            candidates.append(xpos)
    if len(candidates) == 2 and xmax - xmin >= 50.0:
        candidates.append((xmin + xmax) / 2.0)
    return sorted(unique_ticks(candidates))


def plot_panel_b_active_zoom(case: LSupportCase, ylim: tuple[float, float]) -> list[Path]:
    data = case.panel_data
    windows = zoom_windows(data)
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.55), sharey=True)
    panel_base.make_transparent(fig)
    for ax, (xmin, xmax), title in zip(axes, windows, ("Au-side window", "Pd-side window"), strict=True):
        ax.plot(data.x_nm, data.phi_rp_edl, color=panel_base.COLORS["with"], lw=2.0, label="with EDL", zorder=3)
        ax.plot(data.x_nm, data.phi_rp_no, color=panel_base.COLORS["without"], lw=1.8, label="w/o EDL", zorder=2)
        add_unique_boundaries(ax, data)
        panel_base.style_axes(ax, "x (nm)", r"$\phi_{\mathrm{RP}}(x)$ (V)", title)
        ax.set_xlim(xmin, xmax)
        ax.set_xticks(window_ticks(xmin, xmax, data))
        ax.set_xticklabels([format_tick(tick) for tick in ax.get_xticks()])
        ax.set_ylim(*ylim)
    axes[1].set_ylabel("")
    axes[1].tick_params(labelleft=False)
    axes[1].legend(loc="center right", fontsize=7.7, handlelength=1.8)
    annotate_l_support(axes[1], case.value_nm)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.22, top=0.84, wspace=0.16)
    return save_panel_figure(fig, f"figure_3_panel_b_reaction_plane_potential_active_zoom_{output_tag(case.value_nm)}")


def plot_panel_e_active_zoom(case: LSupportCase, ylim: tuple[float, float]) -> list[Path]:
    data = case.panel_data
    i1_edl, i1_no, i2_edl, i2_no = scaled_current_arrays(data)
    windows = zoom_windows(data)
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.55), sharey=True)
    panel_base.make_transparent(fig)
    for ax, (xmin, xmax), title in zip(axes, windows, ("Au-side window", "Pd-side window"), strict=True):
        ax.fill_between(data.x_nm, 0.0, i1_edl, where=np.isfinite(i1_edl), color=COLORS["current_i1"], alpha=0.34, lw=0.0)
        ax.fill_between(data.x_nm, 0.0, i2_edl, where=np.isfinite(i2_edl), color=COLORS["current_i2"], alpha=0.38, lw=0.0)
        ax.axhline(0.0, color=COLORS["dark"], lw=0.55, alpha=0.78)
        ax.plot(data.x_nm, i1_edl, color=COLORS["current_i1"], lw=2.0, label=r"$i_1$ (Au), with EDL")
        ax.plot(data.x_nm, i2_edl, color=COLORS["current_i2"], lw=2.0, label=r"$i_2$ (Pd), with EDL")
        ax.plot(data.x_nm, i1_no, color=COLORS["current_i1"], lw=1.7, ls=(0, (4, 2)), label=r"$i_1$ (Au), w/o EDL")
        ax.plot(data.x_nm, i2_no, color=COLORS["current_i2"], lw=1.7, ls=(0, (2, 2)), label=r"$i_2$ (Pd), w/o EDL")
        add_unique_boundaries(ax, data, zorder=5)
        panel_base.style_axes(ax, "x (nm)", r"$i(x)$ ($10^{-3}$ A/m$^2$)", title)
        ax.set_xlim(xmin, xmax)
        ax.set_xticks(window_ticks(xmin, xmax, data))
        ax.set_xticklabels([format_tick(tick) for tick in ax.get_xticks()])
        ax.set_ylim(*ylim)
    axes[1].set_ylabel("")
    axes[1].tick_params(labelleft=False)
    axes[1].legend(loc="upper right", fontsize=6.3, handlelength=1.5)
    annotate_l_support(axes[0], case.value_nm, loc="lower_right")
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.22, top=0.84, wspace=0.16)
    return save_panel_figure(fig, f"figure_3_panel_e_local_current_density_active_zoom_{output_tag(case.value_nm)}")


def add_material_lane(ax: Any, data: Phi2DData, xmin: float | None = None, xmax: float | None = None) -> None:
    xmin = 0.0 if xmin is None else float(xmin)
    xmax = float(data.L_total_nm) if xmax is None else float(xmax)
    segments = (
        ("Au", 0.0, data.L_Au_nm, COLORS["au"], COLORS["dark"]),
        ("support", data.L_Au_nm, data.L_C_nm, COLORS["support"], "white"),
        ("Pd", data.L_C_nm, data.L_total_nm, COLORS["pd"], "white"),
    )
    visible_span = xmax - xmin
    for label, x0, x1, face, text_color in segments:
        x0_clip = max(x0, xmin)
        x1_clip = min(x1, xmax)
        width = x1_clip - x0_clip
        if width <= 1e-9:
            continue
        ax.add_patch(rp_base.Rectangle((x0_clip, 0.0), width, 1.0, facecolor=face, edgecolor="white", lw=0.8))
        if width >= 6.0 and width / visible_span >= 0.08:
            ax.text((x0_clip + x1_clip) / 2.0, 0.5, label, ha="center", va="center", fontsize=8.3, color=text_color)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()


def plot_phi_s_case(case: LSupportCase, phi_vlim: float) -> list[Path]:
    data = case.rp_data
    fig = plt.figure(figsize=(5.8, 3.0))
    gs = fig.add_gridspec(2, 2, width_ratios=(1.0, 0.038), height_ratios=(1.0, 0.12), hspace=0.24, wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    lane_ax = fig.add_subplot(gs[1, 0], sharex=ax)
    fig.add_subplot(gs[1, 1]).set_axis_off()
    levels = np.linspace(-phi_vlim, phi_vlim, 11)
    rp_base.add_heatmap(
        ax,
        cax,
        data,
        data.phi_s_mV,
        cmap="RdBu_r",
        norm=rp_base.TwoSlopeNorm(vmin=-phi_vlim, vcenter=0.0, vmax=phi_vlim),
        cbar_label=r"$\Phi_s$ (mV)",
        title=rf"Solution phase potential, $E_{{\mathrm{{mix}}}}$ = {float(data.res_edl['E_mix']):.2f} V",
        contour_levels=levels,
    )
    ax.text(0.985, 0.955, rf"$L_{{\mathrm{{support}}}}$ = {format_nm_value(case.value_nm)} nm", transform=ax.transAxes, ha="right", va="top", fontsize=8.2, color=COLORS["dark"])
    ax.set_xlim(0.0, data.L_total_nm)
    ax.set_ylim(0.0, data.y_nm[-1])
    ax.set_yticks([0.0, 5.0, 10.0, 15.0])
    ax.set_xticks(x_ticks_for_2d(data))
    ax.set_xticklabels([format_tick(tick) for tick in ax.get_xticks()])
    ax.tick_params(labelbottom=True)
    ax.set_xlabel("")
    add_material_lane(lane_ax, data)
    lane_ax.text(0.5, -0.58, "x (nm)", transform=lane_ax.transAxes, ha="center", va="top", fontsize=8.5, color=COLORS["dark"], clip_on=False)
    return save_map_figure(fig, f"solution_phase_potential_2d_{output_tag(case.value_nm)}")


def style_zoom_map_axis(ax: Any, title: str, *, show_ylabel: bool) -> None:
    ax.set_title(title, loc="left", pad=5, fontsize=9.4, fontweight="normal")
    ax.set_xlabel("")
    ax.set_ylabel("y (nm)" if show_ylabel else "")
    if not show_ylabel:
        ax.tick_params(labelleft=False)
    ax.tick_params(length=3.2, width=0.85, pad=2.5, labelsize=8.0)
    for spine in ("left", "bottom", "top", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.9)
        ax.spines[spine].set_color(COLORS["dark"])


def plot_phi_s_active_zoom(case: LSupportCase, phi_vlim: float) -> list[Path]:
    data = case.rp_data
    windows = zoom_windows(case.panel_data)
    fig = plt.figure(figsize=(5.95, 3.15))
    gs = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 0.050), height_ratios=(1.0, 0.12), hspace=0.24, wspace=0.14)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])
    lane_axes = [fig.add_subplot(gs[1, 0], sharex=axes[0]), fig.add_subplot(gs[1, 1], sharex=axes[1])]
    fig.add_subplot(gs[1, 2]).set_axis_off()
    norm = rp_base.TwoSlopeNorm(vmin=-phi_vlim, vcenter=0.0, vmax=phi_vlim)
    levels = np.linspace(-phi_vlim, phi_vlim, 11)
    mesh = None
    for ax, lane_ax, (xmin, xmax), title, show_ylabel in zip(
        axes,
        lane_axes,
        windows,
        ("Au-side active window", "Pd-side active window"),
        (True, False),
        strict=True,
    ):
        mesh = ax.pcolormesh(data.x_nm, data.y_nm, data.phi_s_mV, shading="auto", cmap="RdBu_r", norm=norm, rasterized=True)
        ax.contour(data.x_nm, data.y_nm, data.phi_s_mV, levels=levels, colors="black", lw=0.28, alpha=0.28)
        add_unique_boundaries(ax, data)
        style_zoom_map_axis(ax, title, show_ylabel=show_ylabel)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0.0, data.y_nm[-1])
        ax.set_yticks([0.0, 5.0, 10.0, 15.0])
        ax.set_xticks(window_ticks(xmin, xmax, case.panel_data))
        ax.set_xticklabels([format_tick(tick) for tick in ax.get_xticks()])
        ax.tick_params(labelbottom=True)
        add_material_lane(lane_ax, data, xmin, xmax)
    if mesh is None:
        raise RuntimeError("No Phi_s zoom mesh was created")
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label(r"$\Phi_s$ (mV)", labelpad=5)
    cbar.ax.tick_params(length=2.8, width=0.75, labelsize=7.8, pad=2.2)
    cbar.outline.set_linewidth(0.8)
    annotate_l_support(axes[1], case.value_nm)
    for lane_ax in lane_axes:
        lane_ax.text(0.5, -0.58, "x (nm)", transform=lane_ax.transAxes, ha="center", va="top", fontsize=8.5, color=COLORS["dark"], clip_on=False)
    return save_map_figure(fig, f"solution_phase_potential_2d_active_zoom_{output_tag(case.value_nm)}")


def sigma_annotation(case: LSupportCase) -> str:
    if case.sigma_data.L_GC_C_nm is None:
        return rf"$L_{{\mathrm{{support}}}}={format_nm_value(case.value_nm)}$ nm" "\n" r"$L_{\mathrm{GC},C}=$ N/A"
    mean_uC_cm2 = 100.0 * float(case.sigma_data.sigma_C_mean_abs_C_per_m2)
    return (
        rf"$L_{{\mathrm{{support}}}}={format_nm_value(case.value_nm)}$ nm" "\n"
        rf"$\langle|\sigma_C|\rangle={mean_uC_cm2:.2f}$ $\mu$C/cm$^2$" "\n"
        rf"$L_{{\mathrm{{GC}},C}}={float(case.sigma_data.L_GC_C_nm):.2f}$ nm"
    )


def plot_sigma_segments(ax: Any, case: LSupportCase, *, label: bool = True) -> None:
    for segment in case.sigma_data.segments:
        ax.plot(
            segment.x_nm,
            100.0 * segment.sigma_C_per_m2,
            color=COLORS[segment.material],
            lw=2.0,
            label=segment.material if label else None,
            zorder=3,
        )


def plot_sigma_case(case: LSupportCase, ylim: tuple[float, float]) -> list[Path]:
    data = case.panel_data
    fig, ax = panel_base.make_single_axis_panel()
    plot_sigma_segments(ax, case)
    ax.axhline(0.0, color=COLORS["dark"], lw=0.7, alpha=0.82, zorder=2)
    add_unique_boundaries(ax, data)
    panel_base.style_axes(ax, "x (nm)", r"$\sigma(x)$ ($\mu$C/cm$^2$)", "Surface charge distribution")
    ax.set_xlim(0.0, float(data.x_nm[-1]))
    ax.set_xticks(x_ticks_for_panel(data))
    ax.set_xticklabels([format_tick(tick) for tick in ax.get_xticks()])
    ax.set_ylim(*ylim)
    ax.text(0.985, 0.955, sigma_annotation(case), transform=ax.transAxes, ha="right", va="top", fontsize=7.3, color=COLORS["dark"])
    ax.legend(loc="upper left", fontsize=7.4, handlelength=1.8)
    return save_panel_figure(fig, f"surface_charge_distribution_{output_tag(case.value_nm)}")


def plot_sigma_active_zoom(case: LSupportCase, ylim: tuple[float, float]) -> list[Path]:
    data = case.panel_data
    windows = zoom_windows(data)
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.55), sharey=True)
    panel_base.make_transparent(fig)
    for ax, (xmin, xmax), title in zip(axes, windows, ("Au-side window", "Pd-side window"), strict=True):
        plot_sigma_segments(ax, case)
        ax.axhline(0.0, color=COLORS["dark"], lw=0.7, alpha=0.82)
        add_unique_boundaries(ax, data)
        panel_base.style_axes(ax, "x (nm)", r"$\sigma(x)$ ($\mu$C/cm$^2$)", title)
        ax.set_xlim(xmin, xmax)
        ax.set_xticks(window_ticks(xmin, xmax, data))
        ax.set_xticklabels([format_tick(tick) for tick in ax.get_xticks()])
        ax.set_ylim(*ylim)
    axes[1].set_ylabel("")
    axes[1].tick_params(labelleft=False)
    axes[1].legend(loc="lower right", fontsize=7.2, handlelength=1.7)
    axes[1].text(0.985, 0.955, sigma_annotation(case), transform=axes[1].transAxes, ha="right", va="top", fontsize=7.1, color=COLORS["dark"])
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.22, top=0.84, wspace=0.16)
    return save_panel_figure(fig, f"surface_charge_distribution_active_zoom_{output_tag(case.value_nm)}")


def _ofat_row(
    value_nm: float,
    res_edl: dict[str, Any],
    res_no: dict[str, Any],
    sigma: SigmaData,
    params: dict[str, Any],
    reference_case: LSupportCase,
) -> dict[str, Any]:
    derived = solver.compute_derived_params(params)
    reference = reference_case.res_edl
    delta_E_overlap_V = float(res_edl["E_mix"]) - float(reference["E_mix"])
    delta_i_overlap = float(res_edl["i_mix_avg_A_per_m2"]) - float(reference["i_mix_avg_A_per_m2"])
    delta_i_overlap_percent = 100.0 * delta_i_overlap / float(reference["i_mix_avg_A_per_m2"])
    phi_au_mean_mV = 1000.0 * float(res_edl["phi2_1_meanV"])
    phi_pd_mean_mV = 1000.0 * float(res_edl["phi2_2_meanV"])
    delta_phi_au_mean_mV = phi_au_mean_mV - 1000.0 * float(reference["phi2_1_meanV"])
    delta_phi_pd_mean_mV = phi_pd_mean_mV - 1000.0 * float(reference["phi2_2_meanV"])
    L_Au = float(derived["L_Au"])
    L_Pd = float(derived["L_Pd_len"])
    delta_phi_active_rms_mV = math.sqrt(
        (L_Au * delta_phi_au_mean_mV**2 + L_Pd * delta_phi_pd_mean_mV**2) / (L_Au + L_Pd)
    )
    return {
        "L_support_nm": float(value_nm),
        "L_gap_m": float(params["L_gap"]),
        "N_modes": int(params["N_modes"]),
        "Nx": int(params["Nx"]),
        "lambda_D_nm": float(derived["lambda_D"]) * 1.0e9,
        "L_support_over_lambda_D": float(value_nm) / (float(derived["lambda_D"]) * 1.0e9),
        "E_mix_with_V": float(res_edl["E_mix"]),
        "E_mix_no_V": float(res_no["E_mix"]),
        "delta_E_mix_V": float(res_edl["E_mix"]) - float(res_no["E_mix"]),
        "i_mix_avg_with_A_per_m2": float(res_edl["i_mix_avg_A_per_m2"]),
        "i_mix_avg_no_A_per_m2": float(res_no["i_mix_avg_A_per_m2"]),
        "delta_i_mix_avg_A_per_m2": float(res_edl["i_mix_avg_A_per_m2"]) - float(res_no["i_mix_avg_A_per_m2"]),
        "max_abs_phi_tilde_with_edl": float(res_edl["max_abs_phi_tilde"]),
        "support_charge_status": sigma.support_status,
        "sigma_C_signed_mean_C_per_m2": sigma.sigma_C_signed_mean_C_per_m2,
        "sigma_C_mean_abs_C_per_m2": sigma.sigma_C_mean_abs_C_per_m2,
        "sigma_C_midpoint_C_per_m2": sigma.sigma_C_midpoint_C_per_m2,
        "sigma_C_min_C_per_m2": sigma.sigma_C_min_C_per_m2,
        "sigma_C_max_C_per_m2": sigma.sigma_C_max_C_per_m2,
        "sigma_C_sign_change": sigma.sigma_C_sign_change,
        "L_GC_C_nm": sigma.L_GC_C_nm,
        "L_GC_C_midpoint_nm": sigma.L_GC_C_midpoint_nm,
        "L_support_over_L_GC_C": sigma.L_support_over_L_GC_C,
        "overlap_reference_L_support_nm": reference_case.value_nm,
        "overlap_delta_E_mix_vs_1000_V": delta_E_overlap_V,
        "overlap_abs_delta_E_mix_vs_1000_mV": 1000.0 * abs(delta_E_overlap_V),
        "overlap_delta_i_mix_avg_vs_1000_A_per_m2": delta_i_overlap,
        "overlap_delta_i_mix_avg_vs_1000_percent": delta_i_overlap_percent,
        "overlap_abs_delta_i_mix_avg_vs_1000_percent": abs(delta_i_overlap_percent),
        "phi_RP_Au_mean_mV": phi_au_mean_mV,
        "phi_RP_Pd_mean_mV": phi_pd_mean_mV,
        "overlap_delta_phi_RP_Au_mean_vs_1000_mV": delta_phi_au_mean_mV,
        "overlap_delta_phi_RP_Pd_mean_vs_1000_mV": delta_phi_pd_mean_mV,
        "overlap_delta_phi_RP_active_rms_vs_1000_mV": delta_phi_active_rms_mV,
        "overlap_delta_phi_RP_active_max_material_mean_vs_1000_mV": max(
            abs(delta_phi_au_mean_mV), abs(delta_phi_pd_mean_mV)
        ),
    }


def build_ofat_rows(cases: list[LSupportCase]) -> list[dict[str, Any]]:
    reference_case = next(case for case in cases if is_case(case.value_nm, OVERLAP_REFERENCE_NM))
    reusable = {case.value_nm: case for case in cases if not is_case(case.value_nm, ACTIVE_ZOOM_SUPPORT_NM)}
    rows: list[dict[str, Any]] = []
    for value_nm in OVERLAP_L_SUPPORT_NM_VALUES:
        case = next((item for key, item in reusable.items() if is_case(value_nm, key)), None)
        if case is not None:
            rows.append(_ofat_row(value_nm, case.res_edl, case.res_no, case.sigma_data, case.params, reference_case))
            continue
        params = params_for_l_support(value_nm, long_resolution=False)
        _, res_edl = _build_with_edl_result(params, return_profiles=True)
        res_no = solver.run_case(params, mode="FULL", return_profiles=False, use_edl=False)
        sigma = build_sigma_data(params, res_edl)
        rows.append(_ofat_row(value_nm, res_edl, res_no, sigma, params, reference_case))

    contact_row = next(row for row in rows if is_case(float(row["L_support_nm"]), 0.0))
    fraction_fields = (
        ("overlap_abs_delta_E_mix_vs_1000_mV", "overlap_fraction_E_mix"),
        ("overlap_delta_phi_RP_active_rms_vs_1000_mV", "overlap_fraction_phi_RP_active_rms"),
        ("overlap_abs_delta_i_mix_avg_vs_1000_percent", "overlap_fraction_i_mix_avg"),
    )
    for source_key, fraction_key in fraction_fields:
        contact_value = float(contact_row[source_key])
        if not math.isfinite(contact_value) or contact_value <= 0.0:
            raise ValueError(f"Contact overlap magnitude must be finite and positive for {source_key}")
        for row in rows:
            row[fraction_key] = float(row[source_key]) / contact_value
    return rows


def exponential_approach(length_nm: Any, E_inf: float, amplitude: float, decay_nm: float) -> Any:
    return E_inf + amplitude * np.exp(-np.asarray(length_nm, dtype=float) / decay_nm)


def analyze_plateau(rows: list[dict[str, Any]], cases: list[LSupportCase]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fit_rows = [row for row in rows if float(row["L_support_nm"]) >= PLATEAU_FIT_MIN_NM]
    x_fit = np.asarray([row["L_support_nm"] for row in fit_rows], dtype=float)
    y_fit = np.asarray([row["E_mix_with_V"] for row in fit_rows], dtype=float)
    p0 = (float(y_fit[-1]) - 0.04e-3, float(y_fit[0] - y_fit[-1]) + 0.04e-3, 2.5)
    popt, pcov = curve_fit(
        exponential_approach,
        x_fit,
        y_fit,
        p0=p0,
        bounds=([0.55, 0.0, 0.05], [0.65, 0.1, 100.0]),
        maxfev=20000,
    )
    E_inf, amplitude, decay_nm = [float(value) for value in popt]
    residuals = y_fit - exponential_approach(x_fit, *popt)
    max_fit_residual_V = float(np.max(np.abs(residuals)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    plateau_continuous_nm = max(0.0, decay_nm * math.log(amplitude / PLATEAU_TOLERANCE_V))

    plateau_sampled_nm: float | None = None
    for idx, row in enumerate(rows):
        differences = [abs(float(item["E_mix_with_V"]) - E_inf) for item in rows[idx:]]
        if differences and max(differences) <= PLATEAU_TOLERANCE_V:
            plateau_sampled_nm = float(row["L_support_nm"])
            break
    if plateau_sampled_nm is None:
        raise RuntimeError("No OFAT point satisfies the plateau criterion")

    finite_lgc = [row for row in rows if row["L_GC_C_nm"] is not None]
    crossover_nm: float | None = None
    for left, right in zip(finite_lgc[:-1], finite_lgc[1:]):
        f_left = float(left["L_support_nm"]) - float(left["L_GC_C_nm"])
        f_right = float(right["L_support_nm"]) - float(right["L_GC_C_nm"])
        if f_left == 0.0:
            crossover_nm = float(left["L_support_nm"])
            break
        if f_left * f_right <= 0.0:
            x0 = float(left["L_support_nm"])
            x1 = float(right["L_support_nm"])
            crossover_nm = x0 - f_left * (x1 - x0) / (f_right - f_left)
            break
    if crossover_nm is None:
        raise RuntimeError("Could not locate L_support = L_GC,C crossover")

    case_1000 = next(case for case in cases if is_case(case.value_nm, 1000.0))
    checkpoint_rows: list[dict[str, Any]] = []
    for value_nm in PLATEAU_CHECKPOINTS_NM:
        if is_case(value_nm, 1000.0):
            E_mix = float(case_1000.res_edl["E_mix"])
            n_modes = int(case_1000.params["N_modes"])
            nx = int(case_1000.params["Nx"])
        else:
            params = params_for_l_support(value_nm, long_resolution=False)
            _, result = _build_with_edl_result(params, return_profiles=False)
            E_mix = float(result["E_mix"])
            n_modes = int(params["N_modes"])
            nx = int(params["Nx"])
        difference = E_mix - E_inf
        checkpoint_rows.append(
            {
                "L_support_nm": value_nm,
                "N_modes": n_modes,
                "Nx": nx,
                "E_mix_with_V": E_mix,
                "E_mix_minus_E_inf_V": difference,
                "within_0p05_mV": abs(difference) <= PLATEAU_CHECKPOINT_TOLERANCE_V,
            }
        )
    if not all(bool(row["within_0p05_mV"]) for row in checkpoint_rows):
        raise RuntimeError(f"Long-support checkpoints do not validate E_inf within 0.05 mV: {checkpoint_rows}")
    if max_fit_residual_V > 1.0e-6:
        raise RuntimeError(f"Plateau exponential fit residual is too large: {max_fit_residual_V:.3g} V")

    lambda_D_nm = float(rows[0]["lambda_D_nm"])
    summary = {
        "model": "E_mix_with(L) = E_inf + amplitude * exp(-L/decay_length)",
        "fit_range_nm": [PLATEAU_FIT_MIN_NM, float(rows[-1]["L_support_nm"])],
        "E_inf_V": E_inf,
        "amplitude_V": amplitude,
        "decay_length_nm": decay_nm,
        "fit_r_squared": r_squared,
        "max_fit_residual_V": max_fit_residual_V,
        "plateau_tolerance_V": PLATEAU_TOLERANCE_V,
        "plateau_continuous_nm": plateau_continuous_nm,
        "plateau_first_sampled_nm": plateau_sampled_nm,
        "lambda_D_nm": lambda_D_nm,
        "plateau_over_lambda_D": plateau_sampled_nm / lambda_D_nm,
        "L_support_equals_L_GC_C_nm": crossover_nm,
        "LGC_definition": "2 * epsilon_s * R * T / (F * mean_support_abs_sigma_C)",
        "paper_source": "Zhang et al., ACS Catalysis 2026, PDF page 6 / journal page 3180",
        "debye_huckel_caveat": "L_GC is a charge-derived diagnostic within the current linearized-PB model; max_abs_phi_tilde exceeds 1.",
        "fit_covariance": np.asarray(pcov, dtype=float).tolist(),
    }
    return summary, checkpoint_rows


def _persistent_threshold_crossing(
    rows: list[dict[str, Any]],
    key: str,
    threshold: float,
) -> tuple[float, float]:
    x = np.asarray([row["L_support_nm"] for row in rows], dtype=float)
    y = np.asarray([abs(float(row[key])) for row in rows], dtype=float)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"Non-finite overlap values for {key}")
    index: int | None = None
    for idx in range(y.size):
        if np.max(y[idx:]) <= threshold:
            index = idx
            break
    if index is None:
        raise RuntimeError(f"Overlap threshold {threshold:g} was not reached for {key}")
    sampled_nm = float(x[index])
    if index == 0:
        return sampled_nm, sampled_nm
    x0, x1 = float(x[index - 1]), float(x[index])
    y0, y1 = float(y[index - 1]), float(y[index])
    if y0 <= 0.0 or y1 <= 0.0 or math.isclose(y0, y1, rel_tol=0.0, abs_tol=1e-30):
        continuous_nm = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
    else:
        continuous_nm = x0 + math.log(y0 / threshold) / math.log(y0 / y1) * (x1 - x0)
    return float(continuous_nm), sampled_nm


def _combined_first_sampled_nm(
    rows: list[dict[str, Any]],
    criteria: tuple[tuple[str, float], ...],
) -> float:
    for idx, row in enumerate(rows):
        trailing = rows[idx:]
        if all(max(abs(float(item[key])) for item in trailing) <= threshold for key, threshold in criteria):
            return float(row["L_support_nm"])
    raise RuntimeError(f"Combined overlap criteria were not reached: {criteria}")


def _interpolate_row_value(rows: list[dict[str, Any]], x_nm: float, key: str) -> float:
    finite = [(float(row["L_support_nm"]), float(row[key])) for row in rows if row[key] is not None]
    if not finite:
        raise ValueError(f"No finite values available for {key}")
    x = np.asarray([item[0] for item in finite], dtype=float)
    y = np.asarray([item[1] for item in finite], dtype=float)
    return float(np.interp(x_nm, x, y))


def analyze_overlap(rows: list[dict[str, Any]], reference_case: LSupportCase) -> dict[str, Any]:
    metric_specs = {
        "E_mix": {
            "key": "overlap_abs_delta_E_mix_vs_1000_mV",
            "fraction_key": "overlap_fraction_E_mix",
            "absolute_tolerance": OVERLAP_E_TOLERANCE_MV,
            "units": "mV",
        },
        "phi_RP_active_rms": {
            "key": "overlap_delta_phi_RP_active_rms_vs_1000_mV",
            "fraction_key": "overlap_fraction_phi_RP_active_rms",
            "absolute_tolerance": OVERLAP_PHI_RP_RMS_TOLERANCE_MV,
            "units": "mV",
        },
        "i_mix_avg": {
            "key": "overlap_abs_delta_i_mix_avg_vs_1000_percent",
            "fraction_key": "overlap_fraction_i_mix_avg",
            "absolute_tolerance": OVERLAP_I_REL_TOLERANCE_PERCENT,
            "units": "% of 1000 nm reference",
        },
    }
    metric_summary: dict[str, Any] = {}
    absolute_criteria: list[tuple[str, float]] = []
    fraction_criteria: list[tuple[str, float]] = []
    for label, spec in metric_specs.items():
        key = str(spec["key"])
        fraction_key = str(spec["fraction_key"])
        contact_value = float(rows[0][key])
        e_fold_continuous_nm, e_fold_first_sampled_nm = _persistent_threshold_crossing(
            rows, key, contact_value / math.e
        )
        residual_continuous_nm, residual_first_sampled_nm = _persistent_threshold_crossing(
            rows, fraction_key, OVERLAP_RESIDUAL_FRACTION
        )
        absolute_continuous_nm, absolute_first_sampled_nm = _persistent_threshold_crossing(
            rows, key, float(spec["absolute_tolerance"])
        )
        metric_summary[label] = {
            "contact_overlap_magnitude": contact_value,
            "units": spec["units"],
            "e_fold_continuous_nm": e_fold_continuous_nm,
            "e_fold_first_sampled_nm": e_fold_first_sampled_nm,
            "five_percent_continuous_nm": residual_continuous_nm,
            "five_percent_first_sampled_nm": residual_first_sampled_nm,
            "absolute_tolerance": spec["absolute_tolerance"],
            "absolute_tolerance_continuous_nm": absolute_continuous_nm,
            "absolute_tolerance_first_sampled_nm": absolute_first_sampled_nm,
        }
        absolute_criteria.append((key, float(spec["absolute_tolerance"])))
        fraction_criteria.append((fraction_key, OVERLAP_RESIDUAL_FRACTION))

    combined_five_percent_continuous_nm = max(
        float(item["five_percent_continuous_nm"]) for item in metric_summary.values()
    )
    combined_absolute_continuous_nm = max(
        float(item["absolute_tolerance_continuous_nm"]) for item in metric_summary.values()
    )
    combined_five_percent_first_sampled_nm = _combined_first_sampled_nm(rows, tuple(fraction_criteria))
    combined_absolute_first_sampled_nm = _combined_first_sampled_nm(rows, tuple(absolute_criteria))
    lambda_D_nm = float(rows[0]["lambda_D_nm"])
    L_GC_at_five_percent_nm = _interpolate_row_value(rows, combined_five_percent_continuous_nm, "L_GC_C_nm")
    L_GC_at_absolute_nm = _interpolate_row_value(rows, combined_absolute_continuous_nm, "L_GC_C_nm")
    return {
        "reference": {
            "L_support_nm": reference_case.value_nm,
            "N_modes": int(reference_case.params["N_modes"]),
            "Nx": int(reference_case.params["Nx"]),
            "E_mix_with_V": float(reference_case.res_edl["E_mix"]),
            "i_mix_avg_with_A_per_m2": float(reference_case.res_edl["i_mix_avg_A_per_m2"]),
            "phi_RP_Au_mean_mV": 1000.0 * float(reference_case.res_edl["phi2_1_meanV"]),
            "phi_RP_Pd_mean_mV": 1000.0 * float(reference_case.res_edl["phi2_2_meanV"]),
        },
        "definitions": {
            "delta_E_mix": "E_mix_with(L_support) - E_mix_with(1000 nm)",
            "delta_i_mix_avg": "i_mix_avg_with(L_support) - i_mix_avg_with(1000 nm)",
            "delta_phi_RP_Au_or_Pd": "material-mean phi_RP(L_support) - material-mean phi_RP(1000 nm), aligned by material rather than global x",
            "delta_phi_RP_active_rms": "length-weighted RMS of the Au and Pd material-mean delta_phi_RP values",
            "five_percent_boundary": "all three overlap magnitudes are <= 5% of their respective L_support=0 nm contact values",
            "absolute_negligible_boundary": "all three absolute tolerances are satisfied persistently",
        },
        "absolute_tolerances": {
            "abs_delta_E_mix_mV": OVERLAP_E_TOLERANCE_MV,
            "delta_phi_RP_active_rms_mV": OVERLAP_PHI_RP_RMS_TOLERANCE_MV,
            "abs_delta_i_mix_avg_percent_of_reference": OVERLAP_I_REL_TOLERANCE_PERCENT,
        },
        "metrics": metric_summary,
        "combined_five_percent_continuous_nm": combined_five_percent_continuous_nm,
        "combined_five_percent_first_sampled_nm": combined_five_percent_first_sampled_nm,
        "combined_five_percent_over_lambda_D": combined_five_percent_continuous_nm / lambda_D_nm,
        "combined_five_percent_L_GC_C_nm": L_GC_at_five_percent_nm,
        "combined_five_percent_over_L_GC_C": combined_five_percent_continuous_nm / L_GC_at_five_percent_nm,
        "combined_absolute_continuous_nm": combined_absolute_continuous_nm,
        "combined_absolute_first_sampled_nm": combined_absolute_first_sampled_nm,
        "combined_absolute_over_lambda_D": combined_absolute_continuous_nm / lambda_D_nm,
        "combined_absolute_L_GC_C_nm": L_GC_at_absolute_nm,
        "combined_absolute_over_L_GC_C": combined_absolute_continuous_nm / L_GC_at_absolute_nm,
        "lambda_D_nm": lambda_D_nm,
        "model_caveat": "These are operational overlap boundaries within the present linearized-PB model, not universal sharp transitions.",
        "numerical_caveat": "Material-mean phi_RP is used because pointwise interface maxima are contaminated by unequal Fourier resolution and Gibbs ringing.",
    }


def overlap_summary_csv_row(summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "reference_L_support_nm": summary["reference"]["L_support_nm"],
        "lambda_D_nm": summary["lambda_D_nm"],
        "combined_five_percent_continuous_nm": summary["combined_five_percent_continuous_nm"],
        "combined_five_percent_first_sampled_nm": summary["combined_five_percent_first_sampled_nm"],
        "combined_five_percent_over_lambda_D": summary["combined_five_percent_over_lambda_D"],
        "combined_five_percent_over_L_GC_C": summary["combined_five_percent_over_L_GC_C"],
        "combined_absolute_continuous_nm": summary["combined_absolute_continuous_nm"],
        "combined_absolute_first_sampled_nm": summary["combined_absolute_first_sampled_nm"],
        "combined_absolute_over_lambda_D": summary["combined_absolute_over_lambda_D"],
        "combined_absolute_over_L_GC_C": summary["combined_absolute_over_L_GC_C"],
    }
    for label, metric in summary["metrics"].items():
        for key in (
            "contact_overlap_magnitude",
            "e_fold_continuous_nm",
            "five_percent_continuous_nm",
            "five_percent_first_sampled_nm",
            "absolute_tolerance",
            "absolute_tolerance_continuous_nm",
            "absolute_tolerance_first_sampled_nm",
        ):
            row[f"{label}_{key}"] = metric[key]
    return row


def _marker_x(summary: dict[str, Any], value_nm: float, x_key: str) -> float:
    if x_key == "L_support_nm":
        return float(value_nm)
    return float(value_nm) / float(summary["lambda_D_nm"])


def plot_ofat(
    rows: list[dict[str, Any]],
    plateau: dict[str, Any],
    *,
    x_key: str,
    y_key_with: str,
    y_key_no: str,
    xlabel: str,
    ylabel: str,
    stem: str,
    show_length_markers: bool,
) -> list[Path]:
    x = np.asarray([row[x_key] for row in rows], dtype=float)
    y_with = np.asarray([row[y_key_with] for row in rows], dtype=float)
    y_no = np.asarray([row[y_key_no] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(4.35, 3.15), constrained_layout=False)
    ax.plot(x, y_with, color=panel_base.COLORS["with"], lw=2.0, marker="o", ms=3.0, label="with EDL")
    ax.plot(x, y_no, color=panel_base.COLORS["without"], lw=1.8, marker="s", ms=2.8, label="w/o EDL")
    panel_base.style_axes(ax, xlabel, ylabel, r"$L_{\mathrm{support}}$ scan")
    ax.set_xlim(float(x[0]), float(x[-1]))
    ymin = min(float(np.min(y_with)), float(np.min(y_no)))
    ymax = max(float(np.max(y_with)), float(np.max(y_no)))
    pad = max(1e-8, 0.10 * (ymax - ymin))
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.legend(loc="center right", bbox_to_anchor=(0.985, 0.30), fontsize=7.7, handlelength=1.8)

    inset = ax.inset_axes([0.47, 0.52, 0.50, 0.39])
    inset.plot(x, y_with, color=panel_base.COLORS["with"], lw=1.45, marker="o", ms=2.1)
    inset.set_xlim(float(x[0]), float(x[-1]))
    detail_span = float(np.max(y_with) - np.min(y_with))
    detail_pad = max(1e-8, 0.12 * detail_span)
    inset.set_ylim(float(np.min(y_with)) - detail_pad, float(np.max(y_with)) + detail_pad)
    inset.tick_params(length=2.3, width=0.7, pad=1.6, labelsize=6.2)
    inset.set_title("with EDL detail", loc="left", fontsize=6.8, pad=2.0)
    for spine in inset.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color(COLORS["dark"])

    marker_specs = (
        (float(plateau["L_support_equals_L_GC_C_nm"]), COLORS["lgc"], (0, (2, 2)), r"$L=L_{\mathrm{GC},C}$"),
        (float(plateau["lambda_D_nm"]), COLORS["lambda"], (0, (4, 2)), r"$\lambda_D$"),
        (float(plateau["plateau_first_sampled_nm"]), COLORS["plateau"], (0, (6, 2)), "plateau"),
    )
    for value_nm, color, linestyle, label in marker_specs:
        xpos = _marker_x(plateau, value_nm, x_key)
        if show_length_markers:
            inset.axvline(xpos, color=color, lw=0.9, ls=linestyle, alpha=0.95)
            inset.text(xpos, 0.96, label, transform=inset.get_xaxis_transform(), rotation=90, ha="right", va="top", fontsize=5.5, color=color)

    fig.subplots_adjust(left=0.20, right=0.975, bottom=0.20, top=0.86)
    return save_panel_figure(fig, stem, output_dir=OFAT_FIGURES_DIR)


def plot_support_mean_abs_sigma(rows: list[dict[str, Any]], plateau: dict[str, Any]) -> list[Path]:
    defined_rows = [row for row in rows if row["sigma_C_mean_abs_C_per_m2"] is not None]
    x_nm = np.asarray([row["L_support_nm"] for row in defined_rows], dtype=float)
    sigma_uC_cm2 = 100.0 * np.asarray([row["sigma_C_mean_abs_C_per_m2"] for row in defined_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(4.35, 3.15), constrained_layout=False)
    ax.plot(
        x_nm,
        sigma_uC_cm2,
        color=COLORS["support"],
        lw=2.0,
        marker="o",
        ms=3.2,
        markeredgecolor=COLORS["dark"],
        markeredgewidth=0.45,
    )
    panel_base.style_axes(
        ax,
        r"$L_{\mathrm{support}}$ (nm)",
        r"$\langle|\sigma_C|\rangle$ ($\mu$C/cm$^2$)",
        r"$L_{\mathrm{support}}$ scan",
    )
    ax.set_xlim(0.0, float(OFAT_L_SUPPORT_NM_VALUES[-1]))
    ymin = float(np.min(sigma_uC_cm2))
    ymax = float(np.max(sigma_uC_cm2))
    pad = max(1e-6, 0.10 * (ymax - ymin))
    ax.set_ylim(ymin - pad, ymax + pad)

    markers = (
        (float(plateau["L_support_equals_L_GC_C_nm"]), COLORS["lgc"], (0, (2, 2)), r"$L=L_{\mathrm{GC},C}$"),
        (float(plateau["lambda_D_nm"]), COLORS["lambda"], (0, (4, 2)), r"$\lambda_D$"),
    )
    for xpos, color, linestyle, label in markers:
        ax.axvline(xpos, color=color, lw=1.0, ls=linestyle, alpha=0.95)
        ax.text(
            xpos,
            0.96,
            label,
            transform=ax.get_xaxis_transform(),
            rotation=90,
            ha="right",
            va="top",
            fontsize=7.0,
            color=color,
        )
    ax.text(0.025, 0.055, "0 nm: N/A", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2, color=COLORS["dark"])
    fig.subplots_adjust(left=0.20, right=0.975, bottom=0.20, top=0.86)
    stem = f"ofat_compare_L_support_mean_abs_sigma_C_0_10nm_{OUTPUT_TAG}"
    return save_panel_figure(fig, stem, output_dir=OFAT_FIGURES_DIR)


def plot_support_lgc(rows: list[dict[str, Any]], plateau: dict[str, Any]) -> list[Path]:
    defined_rows = [row for row in rows if row["L_GC_C_nm"] is not None]
    x_nm = np.asarray([row["L_support_nm"] for row in defined_rows], dtype=float)
    lgc_nm = np.asarray([row["L_GC_C_nm"] for row in defined_rows], dtype=float)
    crossover_nm = float(plateau["L_support_equals_L_GC_C_nm"])

    fig, ax = plt.subplots(figsize=(4.35, 3.15), constrained_layout=False)
    ax.plot(
        x_nm,
        lgc_nm,
        color=COLORS["lgc"],
        lw=2.0,
        marker="o",
        ms=3.2,
        markeredgecolor=COLORS["dark"],
        markeredgewidth=0.45,
        label=r"$L_{\mathrm{GC},C}$",
        zorder=3,
    )
    ymin = float(np.min(lgc_nm))
    ymax = float(np.max(lgc_nm))
    pad = max(1e-6, 0.10 * (ymax - ymin))
    ylim = (ymin - pad, ymax + pad)
    equality_x = np.linspace(max(0.0, ylim[0]), min(float(OFAT_L_SUPPORT_NM_VALUES[-1]), ylim[1]), 100)
    ax.plot(
        equality_x,
        equality_x,
        color=COLORS["dark"],
        lw=1.2,
        ls=(0, (4, 2)),
        label=r"$L_{\mathrm{GC},C}=L_{\mathrm{support}}$",
        zorder=2,
    )
    ax.scatter(
        [crossover_nm],
        [crossover_nm],
        s=30,
        facecolor="white",
        edgecolor=COLORS["plateau"],
        linewidth=1.2,
        zorder=5,
    )
    ax.annotate(
        rf"crossing = {crossover_nm:.2f} nm",
        xy=(crossover_nm, crossover_nm),
        xytext=(3.8, 3.15),
        fontsize=7.0,
        color=COLORS["plateau"],
        arrowprops={"arrowstyle": "->", "color": COLORS["plateau"], "lw": 0.8},
    )
    panel_base.style_axes(
        ax,
        r"$L_{\mathrm{support}}$ (nm)",
        r"$L_{\mathrm{GC},C}$ (nm)",
        r"$L_{\mathrm{support}}$ scan",
    )
    ax.set_xlim(0.0, float(OFAT_L_SUPPORT_NM_VALUES[-1]))
    ax.set_ylim(*ylim)
    ax.text(
        0.12,
        ylim[0] + 0.025 * (ylim[1] - ylim[0]),
        "0 nm: N/A",
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=COLORS["dark"],
    )
    ax.legend(loc="lower right", fontsize=7.0, handlelength=1.8)
    fig.subplots_adjust(left=0.20, right=0.975, bottom=0.20, top=0.86)
    stem = f"ofat_compare_L_support_L_GC_C_0_10nm_{OUTPUT_TAG}"
    return save_panel_figure(fig, stem, output_dir=OFAT_FIGURES_DIR)


def plot_overlap_metrics(rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[Path]:
    x_nm = np.asarray([row["L_support_nm"] for row in rows], dtype=float)
    specs = (
        (
            "overlap_abs_delta_E_mix_vs_1000_mV",
            r"$|\Delta E_{\mathrm{mix}}|$ (mV)",
            r"$E_{\mathrm{mix}}$ overlap",
            COLORS["overlap_e"],
            OVERLAP_E_TOLERANCE_MV,
        ),
        (
            "overlap_delta_phi_RP_active_rms_vs_1000_mV",
            r"$\Delta\phi_{\mathrm{RP,active}}^{\mathrm{RMS}}$ (mV)",
            r"Reaction-plane overlap",
            COLORS["overlap_phi"],
            OVERLAP_PHI_RP_RMS_TOLERANCE_MV,
        ),
        (
            "overlap_abs_delta_i_mix_avg_vs_1000_percent",
            r"$|\Delta\bar{i}_{\mathrm{mix}}|/\bar{i}_{1000}$ (%)",
            r"Current overlap",
            COLORS["overlap_i"],
            OVERLAP_I_REL_TOLERANCE_PERCENT,
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.55, 2.75), constrained_layout=False)
    strict_nm = float(summary["combined_absolute_continuous_nm"])
    for ax, (key, ylabel, title, color, tolerance) in zip(axes, specs, strict=True):
        y = np.asarray([row[key] for row in rows], dtype=float)
        ax.semilogy(
            x_nm,
            y,
            color=color,
            lw=1.8,
            marker="o",
            ms=2.8,
            markeredgecolor=COLORS["dark"],
            markeredgewidth=0.35,
            zorder=3,
        )
        ax.axhline(tolerance, color=COLORS["gray"], lw=0.9, ls=(0, (3, 2)), zorder=1)
        ax.axvline(float(summary["lambda_D_nm"]), color=COLORS["lambda"], lw=0.85, ls=(0, (4, 2)), zorder=1)
        ax.axvline(strict_nm, color=COLORS["plateau"], lw=1.0, ls=(0, (6, 2)), zorder=2)
        panel_base.style_axes(ax, r"$L_{\mathrm{support}}$ (nm)", ylabel, title)
        ax.set_xlim(0.0, float(x_nm[-1]))
        ax.set_xticks([0.0, 3.0, 6.0, 9.0, 12.0, 15.0])
        ax.set_ylim(max(1e-4, 0.55 * min(float(np.min(y)), tolerance)), 1.45 * max(float(np.max(y)), tolerance))
    axes[0].text(
        float(summary["lambda_D_nm"]),
        0.96,
        r"$\lambda_D$",
        transform=axes[0].get_xaxis_transform(),
        rotation=90,
        ha="right",
        va="top",
        fontsize=6.6,
        color=COLORS["lambda"],
    )
    axes[2].text(
        strict_nm,
        0.96,
        rf"strict = {strict_nm:.1f} nm",
        transform=axes[2].get_xaxis_transform(),
        rotation=90,
        ha="right",
        va="top",
        fontsize=6.6,
        color=COLORS["plateau"],
    )
    axes[0].text(0.97, 0.07, "dashed: tolerance", transform=axes[0].transAxes, ha="right", va="bottom", fontsize=6.4, color=COLORS["gray"])
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.22, top=0.83, wspace=0.43)
    stem = f"ofat_compare_L_support_overlap_metrics_vs_1000nm_{OUTPUT_TAG}"
    return save_panel_figure(fig, stem, output_dir=OFAT_FIGURES_DIR)


def plot_overlap_fractions(rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[Path]:
    x_nm = np.asarray([row["L_support_nm"] for row in rows], dtype=float)
    specs = (
        ("overlap_fraction_E_mix", r"$E_{\mathrm{mix}}$", COLORS["overlap_e"], "o"),
        ("overlap_fraction_phi_RP_active_rms", r"$\phi_{\mathrm{RP,active}}$ RMS", COLORS["overlap_phi"], "s"),
        ("overlap_fraction_i_mix_avg", r"$\bar{i}_{\mathrm{mix}}$", COLORS["overlap_i"], "^"),
    )
    fig, ax = plt.subplots(figsize=(4.6, 3.2), constrained_layout=False)
    for key, label, color, marker in specs:
        y = np.asarray([row[key] for row in rows], dtype=float)
        ax.semilogy(
            x_nm,
            y,
            color=color,
            lw=1.75,
            marker=marker,
            ms=3.1,
            markeredgecolor=COLORS["dark"],
            markeredgewidth=0.35,
            label=label,
        )
    five_percent_nm = float(summary["combined_five_percent_continuous_nm"])
    ax.axhline(OVERLAP_RESIDUAL_FRACTION, color=COLORS["gray"], lw=0.95, ls=(0, (3, 2)))
    ax.axvline(float(summary["lambda_D_nm"]), color=COLORS["lambda"], lw=0.9, ls=(0, (4, 2)))
    ax.axvline(five_percent_nm, color=COLORS["plateau"], lw=1.0, ls=(0, (6, 2)))
    panel_base.style_axes(
        ax,
        r"$L_{\mathrm{support}}$ (nm)",
        "Residual overlap fraction",
        "EDL overlap relative to 1000 nm",
    )
    ax.set_xlim(0.0, float(x_nm[-1]))
    ax.set_xticks([0.0, 3.0, 6.0, 9.0, 12.0, 15.0])
    ax.set_ylim(1e-3, 1.35)
    ax.text(0.15, OVERLAP_RESIDUAL_FRACTION * 1.08, "5%", ha="left", va="bottom", fontsize=7.0, color=COLORS["gray"])
    ax.text(
        five_percent_nm,
        0.96,
        rf"95% decayed = {five_percent_nm:.1f} nm",
        transform=ax.get_xaxis_transform(),
        rotation=90,
        ha="right",
        va="top",
        fontsize=6.8,
        color=COLORS["plateau"],
    )
    ax.legend(loc="upper right", fontsize=7.0, handlelength=1.8)
    fig.subplots_adjust(left=0.18, right=0.975, bottom=0.20, top=0.86)
    stem = f"ofat_compare_L_support_overlap_fraction_vs_1000nm_{OUTPUT_TAG}"
    return save_panel_figure(fig, stem, output_dir=OFAT_FIGURES_DIR)


def save_case_csvs(cases: list[LSupportCase]) -> None:
    phi_rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []
    sigma_rows: list[dict[str, Any]] = []
    for case in cases:
        data = case.rp_data
        sigma = case.sigma_data
        phi_rows.append(
            {
                "L_support_nm": case.value_nm,
                "L_Au_nm": data.L_Au_nm,
                "L_Pd_len_nm": data.L_total_nm - data.L_C_nm,
                "L_total_nm": data.L_total_nm,
                "lambda_D_nm": data.lambda_D_nm,
                "E_mix_with_V": float(case.res_edl["E_mix"]),
                "E_mix_no_V": float(case.res_no["E_mix"]),
                "i_mix_avg_with_A_per_m2": float(case.res_edl["i_mix_avg_A_per_m2"]),
                "i_mix_avg_no_A_per_m2": float(case.res_no["i_mix_avg_A_per_m2"]),
                "max_abs_phi_tilde": float(case.res_edl["max_abs_phi_tilde"]),
                "phi_s_min_mV": float(np.min(data.phi_s_mV)),
                "phi_s_max_mV": float(np.max(data.phi_s_mV)),
            }
        )
        length_rows.append(
            {
                "L_support_nm": case.value_nm,
                "lambda_D_nm": data.lambda_D_nm,
                "L_support_over_lambda_D": case.value_nm / data.lambda_D_nm,
                "support_charge_status": sigma.support_status,
                "sigma_C_signed_mean_C_per_m2": sigma.sigma_C_signed_mean_C_per_m2,
                "sigma_C_mean_abs_C_per_m2": sigma.sigma_C_mean_abs_C_per_m2,
                "sigma_C_midpoint_C_per_m2": sigma.sigma_C_midpoint_C_per_m2,
                "sigma_C_min_C_per_m2": sigma.sigma_C_min_C_per_m2,
                "sigma_C_max_C_per_m2": sigma.sigma_C_max_C_per_m2,
                "sigma_C_sign_change": sigma.sigma_C_sign_change,
                "L_GC_C_nm": sigma.L_GC_C_nm,
                "L_GC_C_midpoint_nm": sigma.L_GC_C_midpoint_nm,
                "L_support_over_L_GC_C": sigma.L_support_over_L_GC_C,
                "E_mix_with_V": float(case.res_edl["E_mix"]),
                "max_abs_phi_tilde": float(case.res_edl["max_abs_phi_tilde"]),
            }
        )
        for segment in sigma.segments:
            for x_nm, phi_rp, charge in zip(segment.x_nm, segment.phi_rp_V, segment.sigma_C_per_m2, strict=True):
                sigma_rows.append(
                    {
                        "L_support_nm": case.value_nm,
                        "x_nm": float(x_nm),
                        "material": segment.material,
                        "phi_RP_V": float(phi_rp),
                        "sigma_C_per_m2": float(charge),
                        "sigma_uC_per_cm2": 100.0 * float(charge),
                    }
                )
    _write_csv(PHI_STATS_CSV, phi_rows)
    _write_csv(LENGTH_SCALE_CSV, length_rows)
    _write_json(
        LENGTH_SCALE_JSON,
        {
            "formula": "L_GC,C = 2 epsilon_s R T / (F <|sigma_C|>_support)",
            "sigma_formula": "sigma_i(x) = C_H,i^eff [E_mix - PZC_i - phi_RP(x)]",
            "paper_source": "MS/cs5c06754.pdf, PDF page 6 / journal page 3180",
            "cases": length_rows,
        },
    )
    _write_csv(SIGMA_PROFILES_CSV, sigma_rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_json(path: Path, data: Any) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return path


def assert_numerical_results(
    cases: list[LSupportCase],
    plateau: dict[str, Any],
    overlap: dict[str, Any],
) -> None:
    expected_lgc = {2.0: 2.276, 3.0: 2.550, 10.0: 3.72, 1000.0: 5.216}
    for value_nm, expected in expected_lgc.items():
        case = next(case for case in cases if is_case(case.value_nm, value_nm))
        actual = case.sigma_data.L_GC_C_nm
        if actual is None or abs(actual - expected) > 0.03:
            raise RuntimeError(f"L_GC,C regression for {value_nm:g} nm: expected about {expected}, got {actual}")
    case_zero = next(case for case in cases if is_case(case.value_nm, 0.0))
    if case_zero.sigma_data.L_GC_C_nm is not None:
        raise RuntimeError("0 nm support must have L_GC,C = N/A")
    lambda_D_nm = float(cases[0].rp_data.lambda_D_nm)
    if abs(lambda_D_nm - 3.041217889) > 1e-6:
        raise RuntimeError(f"Unexpected Debye length: {lambda_D_nm}")
    if abs(float(plateau["plateau_first_sampled_nm"]) - 8.0) > 1e-9:
        raise RuntimeError(f"Expected first sampled plateau at 8 nm, got {plateau['plateau_first_sampled_nm']}")
    if not 2.3 < float(plateau["L_support_equals_L_GC_C_nm"]) < 2.5:
        raise RuntimeError(f"Unexpected L_support=L_GC crossover: {plateau['L_support_equals_L_GC_C_nm']}")
    if not 8.5 < float(overlap["combined_five_percent_continuous_nm"]) < 10.0:
        raise RuntimeError(f"Unexpected 5% overlap boundary: {overlap['combined_five_percent_continuous_nm']}")
    if not 10.0 < float(overlap["combined_absolute_continuous_nm"]) < 12.0:
        raise RuntimeError(f"Unexpected strict overlap boundary: {overlap['combined_absolute_continuous_nm']}")


def assert_outputs(saved: list[Path]) -> None:
    missing = [path for path in saved if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing expected outputs: {missing}")
    root_pngs = sorted(OUT_DIR.glob("*.png"))
    root_svgs = sorted(OUT_DIR.glob("*.svg"))
    if root_pngs or root_svgs:
        raise RuntimeError(f"Figure files must be organized into subdirectories, found root outputs: {root_pngs + root_svgs}")
    case_pngs = sorted(CASE_FIGURES_DIR.glob("*.png"))
    case_svgs = sorted(CASE_FIGURES_DIR.glob("*.svg"))
    ofat_pngs = sorted(OFAT_FIGURES_DIR.glob("*.png"))
    ofat_svgs = sorted(OFAT_FIGURES_DIR.glob("*.svg"))
    if len(case_pngs) != EXPECTED_CASE_OUTPUT_PAIRS or len(case_svgs) != EXPECTED_CASE_OUTPUT_PAIRS:
        raise RuntimeError(
            f"Expected {EXPECTED_CASE_OUTPUT_PAIRS} case PNG/SVG pairs, got {len(case_pngs)} PNG and {len(case_svgs)} SVG"
        )
    if len(ofat_pngs) != EXPECTED_OFAT_OUTPUT_PAIRS or len(ofat_svgs) != EXPECTED_OFAT_OUTPUT_PAIRS:
        raise RuntimeError(
            f"Expected {EXPECTED_OFAT_OUTPUT_PAIRS} OFAT PNG/SVG pairs, got {len(ofat_pngs)} PNG and {len(ofat_svgs)} SVG"
        )
    if len(saved) != 2 * EXPECTED_OUTPUT_PAIRS:
        raise RuntimeError(f"Expected {2 * EXPECTED_OUTPUT_PAIRS} saved paths, got {len(saved)}")
    pdfs = sorted(OUT_DIR.rglob("*.pdf"))
    if pdfs:
        raise RuntimeError(f"Expected zero PDF outputs, found: {pdfs}")
    all_figures = (*case_pngs, *case_svgs, *ofat_pngs, *ofat_svgs)
    stale = [path for path in all_figures if "_L_support_30nm_" in path.name or "_L_support_100nm_" in path.name]
    if stale:
        raise RuntimeError(f"Stale 30/100 nm case outputs remain: {stale}")
    expected_tags = {output_tag(value_nm) for value_nm in L_SUPPORT_NM_VALUES}
    input_tag_sets = {
        "params": {path.stem.removeprefix("params_") for path in INPUTS_DIR.glob("params_L_support_*.json")},
        "overrides": {path.stem.removeprefix("overrides_") for path in INPUTS_DIR.glob("overrides_L_support_*.json")},
        "summary CSV": {path.stem.removeprefix("summary_compare_") for path in INPUTS_DIR.glob("summary_compare_L_support_*.csv")},
        "summary JSON": {path.stem.removeprefix("summary_compare_") for path in INPUTS_DIR.glob("summary_compare_L_support_*.json")},
    }
    for label, actual_tags in input_tag_sets.items():
        if actual_tags != expected_tags:
            raise RuntimeError(f"{label} tags do not match retained cases: {actual_tags} != {expected_tags}")


def main() -> None:
    ensure_dirs()
    removed = remove_previous_generated_outputs()
    configure_style()

    cases = build_cases()
    phi_ylim = common_phi_ylim(cases)
    current_ylim = common_current_ylim(cases)
    sigma_ylim = common_sigma_ylim(cases)
    phi_abs = max(float(np.max(np.abs(case.rp_data.phi_s_mV))) for case in cases)
    phi_vlim = max(10.0, math.ceil(phi_abs / 10.0) * 10.0)

    saved: list[Path] = []
    for case in cases:
        saved.extend(plot_panel_b_case(case, phi_ylim))
        saved.extend(plot_panel_e_case(case, current_ylim))
        saved.extend(plot_phi_s_case(case, phi_vlim))
        saved.extend(plot_sigma_case(case, sigma_ylim))

    case_1000 = next(case for case in cases if is_case(case.value_nm, ACTIVE_ZOOM_SUPPORT_NM))
    saved.extend(plot_panel_b_active_zoom(case_1000, phi_ylim))
    saved.extend(plot_panel_e_active_zoom(case_1000, current_ylim))
    saved.extend(plot_phi_s_active_zoom(case_1000, phi_vlim))
    saved.extend(plot_sigma_active_zoom(case_1000, sigma_ylim))

    save_case_csvs(cases)
    overlap_rows = build_ofat_rows(cases)
    ofat_rows = [row for row in overlap_rows if float(row["L_support_nm"]) <= float(OFAT_L_SUPPORT_NM_VALUES[-1])]
    plateau_summary, checkpoint_rows = analyze_plateau(ofat_rows, cases)
    reference_case = next(case for case in cases if is_case(case.value_nm, OVERLAP_REFERENCE_NM))
    overlap_summary = analyze_overlap(overlap_rows, reference_case)
    _write_csv(OFAT_CSV, ofat_rows)
    _write_csv(CHECKPOINT_CSV, checkpoint_rows)
    _write_csv(PLATEAU_SUMMARY_CSV, [{key: value for key, value in plateau_summary.items() if key != "fit_covariance"}])
    _write_json(PLATEAU_SUMMARY_JSON, plateau_summary)
    _write_csv(OVERLAP_CSV, overlap_rows)
    _write_csv(OVERLAP_SUMMARY_CSV, [overlap_summary_csv_row(overlap_summary)])
    _write_json(OVERLAP_SUMMARY_JSON, overlap_summary)

    saved.extend(
        plot_ofat(
            ofat_rows,
            plateau_summary,
            x_key="L_support_nm",
            y_key_with="E_mix_with_V",
            y_key_no="E_mix_no_V",
            xlabel=r"$L_{\mathrm{support}}$ (nm)",
            ylabel=r"$E_{\mathrm{mix}}$ (V vs. RHE)",
            stem=f"ofat_compare_L_support_E_mix_0_10nm_{OUTPUT_TAG}",
            show_length_markers=True,
        )
    )
    saved.extend(plot_support_mean_abs_sigma(ofat_rows, plateau_summary))
    saved.extend(plot_support_lgc(ofat_rows, plateau_summary))
    saved.extend(plot_overlap_metrics(overlap_rows, overlap_summary))
    saved.extend(plot_overlap_fractions(overlap_rows, overlap_summary))
    saved.extend(
        plot_ofat(
            ofat_rows,
            plateau_summary,
            x_key="L_support_nm",
            y_key_with="i_mix_avg_with_A_per_m2",
            y_key_no="i_mix_avg_no_A_per_m2",
            xlabel=r"$L_{\mathrm{support}}$ (nm)",
            ylabel=r"Mixed current density, $\bar{i}_{\mathrm{mix}}$ (A/m$^2$)",
            stem=f"ofat_compare_L_support_i_mix_avg_0_10nm_{OUTPUT_TAG}",
            show_length_markers=False,
        )
    )
    saved.extend(
        plot_ofat(
            ofat_rows,
            plateau_summary,
            x_key="L_support_over_lambda_D",
            y_key_with="E_mix_with_V",
            y_key_no="E_mix_no_V",
            xlabel=r"$L_{\mathrm{support}}/\lambda_D$",
            ylabel=r"$E_{\mathrm{mix}}$ (V vs. RHE)",
            stem=f"ofat_compare_L_support_over_lambda_D_E_mix_0_10nm_{OUTPUT_TAG}",
            show_length_markers=True,
        )
    )
    saved.extend(
        plot_ofat(
            ofat_rows,
            plateau_summary,
            x_key="L_support_over_lambda_D",
            y_key_with="i_mix_avg_with_A_per_m2",
            y_key_no="i_mix_avg_no_A_per_m2",
            xlabel=r"$L_{\mathrm{support}}/\lambda_D$",
            ylabel=r"Mixed current density, $\bar{i}_{\mathrm{mix}}$ (A/m$^2$)",
            stem=f"ofat_compare_L_support_over_lambda_D_i_mix_avg_0_10nm_{OUTPUT_TAG}",
            show_length_markers=False,
        )
    )

    assert_numerical_results(cases, plateau_summary, overlap_summary)
    assert_outputs(saved)

    print(f"Removed {len(removed)} prior generated files from New_100nm")
    print(f"same-length equal-i0 alpha=0.5 tag = {OUTPUT_TAG}")
    print(f"lambda_D = {float(plateau_summary['lambda_D_nm']):.9f} nm")
    print(f"L_support = L_GC,C crossover = {float(plateau_summary['L_support_equals_L_GC_C_nm']):.6g} nm")
    print(f"E_mix asymptote = {float(plateau_summary['E_inf_V']):.12f} V")
    print(f"continuous plateau threshold = {float(plateau_summary['plateau_continuous_nm']):.6g} nm")
    print(f"first sampled plateau point = {float(plateau_summary['plateau_first_sampled_nm']):.6g} nm")
    print(
        "EDL overlap <= 5% of contact values at "
        f"{float(overlap_summary['combined_five_percent_continuous_nm']):.6g} nm "
        f"(first sampled {float(overlap_summary['combined_five_percent_first_sampled_nm']):.6g} nm)"
    )
    print(
        "EDL overlap satisfies all strict tolerances at "
        f"{float(overlap_summary['combined_absolute_continuous_nm']):.6g} nm "
        f"(first sampled {float(overlap_summary['combined_absolute_first_sampled_nm']):.6g} nm)"
    )
    for case in cases:
        lgc_text = "N/A" if case.sigma_data.L_GC_C_nm is None else f"{case.sigma_data.L_GC_C_nm:.6g} nm"
        print(
            f"L_support={format_nm_value(case.value_nm)} nm: "
            f"N_modes={int(case.params['N_modes'])}, Nx={int(case.params['Nx'])}, "
            f"E_mix_with={float(case.res_edl['E_mix']):.12f} V, L_GC,C={lgc_text}, "
            f"max|phi_tilde|={float(case.res_edl['max_abs_phi_tilde']):.6g}"
        )
    print(f"Verified {EXPECTED_OUTPUT_PAIRS} PNG/SVG output pairs and zero PDF outputs")


if __name__ == "__main__":
    main()
