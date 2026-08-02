from __future__ import annotations

import copy
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[3]
SAME_LENGTH_I0_ALPHA_DIR = ROOT / "Figures" / "Figure_same_length_i0_alpha"
OUT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = OUT_DIR / "inputs"
CSV_DIR = OUT_DIR / "csv"

L_SUPPORT_NM_VALUES = (0.0, 3.0, 1000.0)
ACTIVE_ZOOM_SUPPORT_NM = 1000.0
ACTIVE_WINDOW_NM = 60.0
CURRENT_EXPONENT = -3
PANEL_E_COMMON_YMIN = -400.0
N_POLARIZATION_POINTS = 480
PLOT_CURRENT_UNIT_A = 1.0e-9
POLARIZATION_E_PAD_LEFT = 0.035
POLARIZATION_E_PAD_RIGHT = 0.040

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SAME_LENGTH_I0_ALPHA_DIR))

import same_length_i0_alpha_common as common  # noqa: E402
from Figures.Figure_3 import make_figure_3_panels as panel_base  # noqa: E402
from Figures.Figrue_RP import make_phi_s_reactants_2d as rp_base  # noqa: E402


solver = common.solver
POLARIZATION_STEM = f"l_support_polarization_curves_{common.OUTPUT_TAG}"
POLARIZATION_CURVE_CSV = CSV_DIR / f"{POLARIZATION_STEM}.csv"
POLARIZATION_COLORS = {
    "au_curve": "#3B7A57",
    "pd_curve": "#B64342",
    "case_0": "#767676",
    "case_3": "#272727",
    "case_1000": "#0F4D92",
    "dark": "#272727",
    "gray": "#767676",
}


@dataclass(frozen=True)
class LSupportCase:
    value_nm: float
    params: dict[str, Any]
    summary: dict[str, str]
    panel_data: Any
    rp_data: Any


def ensure_dirs() -> None:
    for path in (OUT_DIR, INPUTS_DIR, CSV_DIR):
        path.mkdir(parents=True, exist_ok=True)


def format_nm_value(value_nm: float) -> str:
    if math.isclose(value_nm, round(value_nm), rel_tol=0.0, abs_tol=1e-9):
        return f"{int(round(value_nm))}"
    return f"{value_nm:.3g}"


def format_nm_tag(value_nm: float) -> str:
    return format_nm_value(value_nm).replace(".", "p")


def output_tag(value_nm: float) -> str:
    return f"L_support_{format_nm_tag(value_nm)}nm_{common.OUTPUT_TAG}"


def remove_previous_outputs() -> None:
    patterns = (
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
        "l_support_polarization_curves_*.png",
        "l_support_polarization_curves_*.svg",
    )
    for pattern in patterns:
        for path in OUT_DIR.glob(pattern):
            path.unlink()
    for path in CSV_DIR.glob("l_support_polarization_curves_*.csv"):
        path.unlink()


def load_params_for_l_support(value_nm: float) -> dict[str, Any]:
    params = common.load_same_length_i0_alpha_params()
    params["L_gap"] = value_nm * 1.0e-9
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


def summary_from_params(params: dict[str, Any]) -> dict[str, str]:
    pair = solver.run_edl_comparison_pair(params, mode="FULL")
    return common.summary_from_pair(pair)


def save_inputs(value_nm: float, params: dict[str, Any], summary: dict[str, str]) -> None:
    tag = output_tag(value_nm)
    overrides = {
        **common.PARAM_OVERRIDES,
        "L_gap": value_nm * 1.0e-9,
    }
    with (INPUTS_DIR / f"params_{tag}.json").open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, sort_keys=True)
        f.write("\n")
    with (INPUTS_DIR / f"overrides_{tag}.json").open("w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2, sort_keys=True)
        f.write("\n")
    with (INPUTS_DIR / f"summary_compare_{tag}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    with (INPUTS_DIR / f"summary_compare_{tag}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")


def build_panel_data(params: dict[str, Any], summary: dict[str, str]) -> Any:
    old_load_params = panel_base.load_params
    old_load_summary = getattr(panel_base, "load_summary", None)

    def load_params() -> dict[str, Any]:
        return copy.deepcopy(params)

    def load_summary() -> dict[str, str]:
        return dict(summary)

    try:
        panel_base.load_params = load_params
        if old_load_summary is not None:
            panel_base.load_summary = load_summary
        return panel_base.build_data()
    finally:
        panel_base.load_params = old_load_params
        if old_load_summary is not None:
            panel_base.load_summary = old_load_summary


def build_2d_data(params: dict[str, Any], summary: dict[str, str]) -> Any:
    old_load_params = rp_base.load_params
    old_load_summary = rp_base.load_summary

    def load_params() -> dict[str, Any]:
        return copy.deepcopy(params)

    def load_summary() -> dict[str, str]:
        return dict(summary)

    try:
        rp_base.load_params = load_params
        rp_base.load_summary = load_summary
        return rp_base.build_2d_data()
    finally:
        rp_base.load_params = old_load_params
        rp_base.load_summary = old_load_summary


def build_cases() -> list[LSupportCase]:
    cases: list[LSupportCase] = []
    for value_nm in L_SUPPORT_NM_VALUES:
        params = load_params_for_l_support(value_nm)
        summary = summary_from_params(params)
        save_inputs(value_nm, params, summary)
        cases.append(
            LSupportCase(
                value_nm=value_nm,
                params=params,
                summary=summary,
                panel_data=build_panel_data(params, summary),
                rp_data=build_2d_data(params, summary),
            )
        )
    return cases


def unique_ticks(values: list[float]) -> list[float]:
    ticks: list[float] = []
    for value in values:
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
    if data.L_C_nm - data.L_Au_nm < 6.0:
        return unique_ticks([0.0, data.L_Au_nm, total])
    return unique_ticks([0.0, data.L_Au_nm, data.L_C_nm, total])


def add_unique_boundaries(ax: Any, data: Any, *, zorder: int = 1) -> None:
    total = float(data.x_nm[-1])
    positions: list[float] = []
    for xpos in (float(data.L_Au_nm), float(data.L_C_nm)):
        if xpos <= 0.0 or xpos >= total:
            continue
        if not any(math.isclose(xpos, existing, rel_tol=0.0, abs_tol=1e-8) for existing in positions):
            positions.append(xpos)
    for xpos in positions:
        ax.axvline(
            xpos,
            linestyle=(0, (3, 2)),
            linewidth=0.9,
            color=panel_base.COLORS["gray"],
            alpha=0.85,
            zorder=zorder,
        )


def finite_range(*arrays: Any, pad_frac: float = 0.08, include_zero: bool = False) -> tuple[float, float]:
    vals = panel_base.np.concatenate([panel_base.np.ravel(panel_base.np.asarray(arr, dtype=float)) for arr in arrays])
    vals = vals[panel_base.np.isfinite(vals)]
    if include_zero:
        vals = panel_base.np.concatenate([vals, panel_base.np.array([0.0], dtype=float)])
    if vals.size == 0:
        raise ValueError("Cannot build finite range from empty data")
    ymin = float(panel_base.np.min(vals))
    ymax = float(panel_base.np.max(vals))
    span = ymax - ymin
    pad = max(1e-6, pad_frac * span)
    return ymin - pad, ymax + pad


def common_phi_ylim(cases: list[LSupportCase]) -> tuple[float, float]:
    return finite_range(
        *(arr for case in cases for arr in (case.panel_data.phi_rp_edl, case.panel_data.phi_rp_no)),
        pad_frac=0.08,
        include_zero=True,
    )


def scale_current(values: Any) -> Any:
    return panel_base.np.asarray(values, dtype=float) / (10.0 ** CURRENT_EXPONENT)


def scaled_current_arrays(data: Any) -> tuple[Any, Any, Any, Any]:
    return (
        scale_current(data.i1_edl_segment),
        scale_current(data.i1_no_segment),
        scale_current(data.i2_edl_segment),
        scale_current(data.i2_no_segment),
    )


def common_current_ylim(cases: list[LSupportCase]) -> tuple[float, float]:
    ymin, ymax = finite_range(
        *(arr for case in cases for arr in scaled_current_arrays(case.panel_data)),
        pad_frac=0.08,
        include_zero=True,
    )
    return min(ymin, PANEL_E_COMMON_YMIN), ymax


def save_panel_figure(fig: Any, stem: str) -> list[Path]:
    saved: list[Path] = []
    panel_base.make_transparent(fig)
    for ext in ("png", "svg"):
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=600, transparent=True, facecolor="none", edgecolor="none")
        saved.append(path)
    panel_base.plt.close(fig)
    return saved


def save_map_figure(fig: Any, stem: str) -> list[Path]:
    saved: list[Path] = []
    for ext in ("png", "svg"):
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=600, bbox_inches="tight")
        saved.append(path)
    rp_base.plt.close(fig)
    return saved


def annotate_l_support(ax: Any, value_nm: float, *, loc: str = "upper_right") -> None:
    if loc == "upper_right":
        x, y = 0.985, 0.955
        ha, va = "right", "top"
    elif loc == "lower_left":
        x, y = 0.035, 0.055
        ha, va = "left", "bottom"
    elif loc == "lower_right":
        x, y = 0.965, 0.055
        ha, va = "right", "bottom"
    else:
        raise ValueError(f"Unsupported annotation location: {loc}")
    ax.text(
        x,
        y,
        rf"$L_{{\mathrm{{support}}}}$ = {format_nm_value(value_nm)} nm",
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=7.8,
        color=panel_base.COLORS["dark"],
    )


def plot_panel_b_case(case: LSupportCase, ylim: tuple[float, float]) -> list[Path]:
    data = case.panel_data
    fig, ax = panel_base.make_single_axis_panel()
    ax.plot(data.x_nm, data.phi_rp_edl, color=panel_base.COLORS["with"], linewidth=2.0, label="with EDL", zorder=3)
    ax.plot(data.x_nm, data.phi_rp_no, color=panel_base.COLORS["without"], linewidth=1.8, label="w/o EDL", zorder=2)
    add_unique_boundaries(ax, data)
    panel_base.style_axes(ax, "x (nm)", r"$\phi_{\mathrm{RP}}(x)$ (V)", "Reaction-plane potential")
    ax.set_xlim(0.0, float(data.x_nm[-1]))
    ax.set_xticks(x_ticks_for_panel(data))
    ax.set_xticklabels([format_tick(tick) for tick in ax.get_xticks()])
    ax.set_ylim(*ylim)
    annotate_l_support(ax, case.value_nm)
    ax.legend(loc="center right", bbox_to_anchor=(0.98, 0.50), fontsize=8.0, handlelength=2.0)
    return save_panel_figure(fig, f"figure_3_panel_b_reaction_plane_potential_{output_tag(case.value_nm)}")


def plot_panel_e_case(case: LSupportCase, ylim: tuple[float, float]) -> list[Path]:
    data = case.panel_data
    i1_edl, i1_no, i2_edl, i2_no = scaled_current_arrays(data)
    i_label = r"$i(x)$ [$10^{-3}$ A/m$^2$]"

    fig, ax = panel_base.make_single_axis_panel()
    mask_i1_edl = panel_base.np.isfinite(i1_edl)
    mask_i2_edl = panel_base.np.isfinite(i2_edl)
    ax.fill_between(
        data.x_nm,
        0.0,
        i1_edl,
        where=mask_i1_edl,
        interpolate=False,
        color=panel_base.COLORS["with"],
        alpha=0.34,
        linewidth=0.0,
        zorder=1,
    )
    ax.fill_between(
        data.x_nm,
        0.0,
        i2_edl,
        where=mask_i2_edl,
        interpolate=False,
        color=panel_base.COLORS["with_alt"],
        alpha=0.38,
        linewidth=0.0,
        zorder=1,
    )
    ax.axhline(0.0, color=panel_base.COLORS["dark"], linewidth=0.55, alpha=0.78, zorder=2)
    ax.plot(data.x_nm, i1_edl, color=panel_base.COLORS["with"], linewidth=2.0, label=r"$i_1$ (Au), with EDL", zorder=4)
    ax.plot(data.x_nm, i2_edl, color=panel_base.COLORS["with_alt"], linewidth=2.0, label=r"$i_2$ (Pd), with EDL", zorder=4)
    ax.plot(
        data.x_nm,
        i1_no,
        color=panel_base.COLORS["without"],
        linewidth=1.7,
        linestyle=(0, (4, 2)),
        label=r"$i_1$ (Au), w/o EDL",
        zorder=3,
    )
    ax.plot(
        data.x_nm,
        i2_no,
        color=panel_base.COLORS["without_alt"],
        linewidth=1.7,
        linestyle=(0, (2, 2)),
        label=r"$i_2$ (Pd), w/o EDL",
        zorder=3,
    )
    add_unique_boundaries(ax, data, zorder=5)
    panel_base.style_axes(ax, "x (nm)", i_label, "Local current density at RP")
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
    fig, axes = panel_base.plt.subplots(1, 2, figsize=(5.8, 2.55), sharey=True)
    panel_base.make_transparent(fig)

    for ax, (xmin, xmax), title in zip(axes, windows, ("Au-side window", "Pd-side window"), strict=True):
        ax.plot(data.x_nm, data.phi_rp_edl, color=panel_base.COLORS["with"], linewidth=2.0, label="with EDL", zorder=3)
        ax.plot(data.x_nm, data.phi_rp_no, color=panel_base.COLORS["without"], linewidth=1.8, label="w/o EDL", zorder=2)
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
    fig, axes = panel_base.plt.subplots(1, 2, figsize=(5.8, 2.55), sharey=True)
    panel_base.make_transparent(fig)

    for ax, (xmin, xmax), title in zip(axes, windows, ("Au-side window", "Pd-side window"), strict=True):
        ax.fill_between(data.x_nm, 0.0, i1_edl, where=panel_base.np.isfinite(i1_edl), color=panel_base.COLORS["with"], alpha=0.34, linewidth=0.0, zorder=1)
        ax.fill_between(data.x_nm, 0.0, i2_edl, where=panel_base.np.isfinite(i2_edl), color=panel_base.COLORS["with_alt"], alpha=0.38, linewidth=0.0, zorder=1)
        ax.axhline(0.0, color=panel_base.COLORS["dark"], linewidth=0.55, alpha=0.78, zorder=2)
        ax.plot(data.x_nm, i1_edl, color=panel_base.COLORS["with"], linewidth=2.0, label=r"$i_1$ (Au), with EDL", zorder=4)
        ax.plot(data.x_nm, i2_edl, color=panel_base.COLORS["with_alt"], linewidth=2.0, label=r"$i_2$ (Pd), with EDL", zorder=4)
        ax.plot(data.x_nm, i1_no, color=panel_base.COLORS["without"], linewidth=1.7, linestyle=(0, (4, 2)), label=r"$i_1$ (Au), w/o EDL", zorder=3)
        ax.plot(data.x_nm, i2_no, color=panel_base.COLORS["without_alt"], linewidth=1.7, linestyle=(0, (2, 2)), label=r"$i_2$ (Pd), w/o EDL", zorder=3)
        add_unique_boundaries(ax, data, zorder=5)
        panel_base.style_axes(ax, "x (nm)", r"$i(x)$ [$10^{-3}$ A/m$^2$]", title)
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


def x_ticks_for_2d(data: Any) -> list[float]:
    if data.L_total_nm > 200.0:
        return [0.0, 250.0, 500.0, 750.0, data.L_total_nm]
    support_width_nm = data.L_C_nm - data.L_Au_nm
    if support_width_nm < 6.0:
        return unique_ticks([0.0, data.L_Au_nm, data.L_total_nm])
    return unique_ticks([0.0, data.L_Au_nm, data.L_C_nm, data.L_total_nm])


def add_material_lane(ax: Any, data: Any) -> None:
    segments = [
        ("Au", 0.0, data.L_Au_nm, rp_base.COLORS["au"], rp_base.COLORS["dark"]),
        ("support", data.L_Au_nm, data.L_C_nm, rp_base.COLORS["support"], "white"),
        ("Pd", data.L_C_nm, data.L_total_nm, rp_base.COLORS["pd"], "white"),
    ]
    total = float(data.L_total_nm)
    for label, x0, x1, face, text_color in segments:
        width = x1 - x0
        if width <= 1e-9:
            continue
        ax.add_patch(rp_base.Rectangle((x0, 0.0), width, 1.0, facecolor=face, edgecolor="white", linewidth=0.8))
        width_fraction = width / total
        if width_fraction < 0.07 and label == "support":
            continue
        if width_fraction < 0.07:
            text_color = rp_base.COLORS["dark"]
        ax.text((x0 + x1) / 2.0, 0.5, label, ha="center", va="center", fontsize=8.3, color=text_color)
    ax.set_xlim(0.0, total)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()


def add_material_lane_window(ax: Any, data: Any, xmin: float, xmax: float) -> None:
    segments = [
        ("Au", 0.0, data.L_Au_nm, rp_base.COLORS["au"], rp_base.COLORS["dark"]),
        ("support", data.L_Au_nm, data.L_C_nm, rp_base.COLORS["support"], "white"),
        ("Pd", data.L_C_nm, data.L_total_nm, rp_base.COLORS["pd"], "white"),
    ]
    for label, x0, x1, face, text_color in segments:
        x0_clip = max(x0, xmin)
        x1_clip = min(x1, xmax)
        width = x1_clip - x0_clip
        if width <= 1e-9:
            continue
        ax.add_patch(rp_base.Rectangle((x0_clip, 0.0), width, 1.0, facecolor=face, edgecolor="white", linewidth=0.8))
        if width >= 6.0:
            ax.text((x0_clip + x1_clip) / 2.0, 0.5, label, ha="center", va="center", fontsize=8.3, color=text_color)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()


def plot_phi_s_case(case: LSupportCase, phi_vlim: float) -> list[Path]:
    data = case.rp_data
    fig = rp_base.plt.figure(figsize=(5.8, 3.0))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=(1.0, 0.038),
        height_ratios=(1.0, 0.12),
        hspace=0.24,
        wspace=0.08,
    )
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    lane_ax = fig.add_subplot(gs[1, 0], sharex=ax)
    fig.add_subplot(gs[1, 1]).set_axis_off()

    phi_levels = rp_base.np.linspace(-phi_vlim, phi_vlim, 11)
    rp_base.add_heatmap(
        ax,
        cax,
        data,
        data.phi_s_mV,
        cmap="RdBu_r",
        norm=rp_base.TwoSlopeNorm(vmin=-phi_vlim, vcenter=0.0, vmax=phi_vlim),
        cbar_label=r"$\Phi_s$ (mV)",
        title=rf"Solution phase potential, $E_{{\mathrm{{mix}}}}$ = {float(data.res_edl['E_mix']):.2f} V",
        show_xlabel=False,
        contour_levels=phi_levels,
    )
    ax.text(
        0.985,
        0.955,
        rf"$L_{{\mathrm{{support}}}}$ = {format_nm_value(case.value_nm)} nm",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        color=rp_base.COLORS["dark"],
    )
    ax.set_xlim(0.0, data.L_total_nm)
    ax.set_ylim(0.0, data.y_nm[-1])
    ax.set_yticks([0.0, 5.0, 10.0, 15.0])
    ax.set_xticks(x_ticks_for_2d(data))
    ax.set_xticklabels([format_tick(tick) for tick in ax.get_xticks()])
    ax.tick_params(labelbottom=True)
    ax.set_xlabel("")
    add_material_lane(lane_ax, data)
    lane_ax.text(
        0.5,
        -0.58,
        "x (nm)",
        transform=lane_ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color=rp_base.COLORS["dark"],
        clip_on=False,
    )
    fig.align_ylabels([ax])
    return save_map_figure(fig, f"solution_phase_potential_2d_{output_tag(case.value_nm)}")


def style_zoom_map_axis(ax: Any, title: str, *, show_ylabel: bool) -> None:
    ax.set_title(title, loc="left", pad=5, fontsize=9.4, fontweight="normal")
    ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("y (nm)")
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    ax.tick_params(length=3.2, width=0.85, pad=2.5, labelsize=8.0)
    for spine in ("left", "bottom", "top", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.9)
        ax.spines[spine].set_color(rp_base.COLORS["dark"])


def plot_phi_s_active_zoom(case: LSupportCase, phi_vlim: float) -> list[Path]:
    data = case.rp_data
    windows = zoom_windows(case.panel_data)
    fig = rp_base.plt.figure(figsize=(5.95, 3.15))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        width_ratios=(1.0, 1.0, 0.050),
        height_ratios=(1.0, 0.12),
        hspace=0.24,
        wspace=0.14,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1], sharey=None)]
    cax = fig.add_subplot(gs[0, 2])
    lane_axes = [fig.add_subplot(gs[1, 0], sharex=axes[0]), fig.add_subplot(gs[1, 1], sharex=axes[1])]
    fig.add_subplot(gs[1, 2]).set_axis_off()

    norm = rp_base.TwoSlopeNorm(vmin=-phi_vlim, vcenter=0.0, vmax=phi_vlim)
    phi_levels = rp_base.np.linspace(-phi_vlim, phi_vlim, 11)
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
        ax.contour(data.x_nm, data.y_nm, data.phi_s_mV, levels=phi_levels, colors="black", linewidths=0.28, alpha=0.28)
        rp_base.add_region_boundaries(ax, data)
        style_zoom_map_axis(ax, title, show_ylabel=show_ylabel)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0.0, data.y_nm[-1])
        ax.set_yticks([0.0, 5.0, 10.0, 15.0])
        ax.set_xticks(window_ticks(xmin, xmax, case.panel_data))
        ax.set_xticklabels([format_tick(tick) for tick in ax.get_xticks()])
        ax.tick_params(labelbottom=True)
        add_material_lane_window(lane_ax, data, xmin, xmax)

    if mesh is None:
        raise RuntimeError("No Phi_s zoom mesh was created")
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label(r"$\Phi_s$ (mV)", labelpad=5)
    cbar.ax.tick_params(length=2.8, width=0.75, labelsize=7.8, pad=2.2)
    cbar.outline.set_linewidth(0.8)
    axes[1].text(
        0.985,
        0.955,
        rf"$L_{{\mathrm{{support}}}}$ = {format_nm_value(case.value_nm)} nm",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        color=rp_base.COLORS["dark"],
    )
    for lane_ax in lane_axes:
        lane_ax.text(
            0.5,
            -0.58,
            "x (nm)",
            transform=lane_ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
            color=rp_base.COLORS["dark"],
            clip_on=False,
        )
    fig.align_ylabels([axes[0]])
    return save_map_figure(fig, f"solution_phase_potential_2d_active_zoom_{output_tag(case.value_nm)}")


def stats_row(case: LSupportCase) -> dict[str, Any]:
    data = case.rp_data
    summary = case.summary
    return {
        "L_support_nm": case.value_nm,
        "L_Au_nm": float(data.L_Au_nm),
        "L_Pd_len_nm": float(data.L_total_nm - data.L_C_nm),
        "L_total_nm": float(data.L_total_nm),
        "lambda_D_nm": float(data.lambda_D_nm),
        "E_mix_with_V": float(summary["E_mix_with"]),
        "E_mix_no_V": float(summary["E_mix_no"]),
        "i_mix_avg_with_A_per_m2": float(summary["i_mix_avg_with"]),
        "i_mix_avg_no_A_per_m2": float(summary["i_mix_avg_no"]),
        "max_abs_phi_tilde": float(summary["max_abs_phi_tilde_with_edl"]),
        "phi_s_min_mV": float(rp_base.np.min(data.phi_s_mV)),
        "phi_s_max_mV": float(rp_base.np.max(data.phi_s_mV)),
    }


def save_stats(cases: list[LSupportCase]) -> Path:
    rows = [stats_row(case) for case in cases]
    path = CSV_DIR / f"phi_s_stats_L_support_{common.OUTPUT_TAG}.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def l_support_case_appearance(value_nm: float) -> tuple[str, float, str | tuple[int, tuple[int, ...]], str]:
    if math.isclose(value_nm, 0.0, rel_tol=0.0, abs_tol=1e-9):
        return "0 nm", 0.55, (0, (3, 2)), POLARIZATION_COLORS["case_0"]
    if math.isclose(value_nm, 3.0, rel_tol=0.0, abs_tol=1e-9):
        return "3 nm", 0.90, "-", POLARIZATION_COLORS["case_3"]
    if math.isclose(value_nm, 1000.0, rel_tol=0.0, abs_tol=1e-9):
        return "1000 nm", 0.95, (0, (5, 2)), POLARIZATION_COLORS["case_1000"]
    return f"{format_nm_value(value_nm)} nm", 0.80, (0, (1, 1)), POLARIZATION_COLORS["gray"]


def current_1e_minus_3_uA(values: Any) -> Any:
    return panel_base.np.asarray(values, dtype=float) / PLOT_CURRENT_UNIT_A


def polarization_e_values(cases: list[LSupportCase]) -> Any:
    e_mix_values = panel_base.np.asarray([float(case.summary["E_mix_with"]) for case in cases], dtype=float)
    e_min = float(panel_base.np.min(e_mix_values)) - POLARIZATION_E_PAD_LEFT
    e_max = float(panel_base.np.max(e_mix_values)) + POLARIZATION_E_PAD_RIGHT
    return panel_base.np.linspace(e_min, e_max, N_POLARIZATION_POINTS, dtype=float)


def build_polarization_curve_data(cases: list[LSupportCase]) -> list[dict[str, Any]]:
    e_values = polarization_e_values(cases)
    curve_cases: list[dict[str, Any]] = []
    for case in cases:
        curve = solver.compute_polarization_curve(
            case.params,
            mode="FULL",
            use_edl=True,
            E_values=e_values,
            use_affine_phi2=bool(case.params.get("use_affine_phi2", True)),
        )
        i_mix_abs_A = float(case.summary["i_mix_abs_with"])
        i_mix_no_abs_A = float(case.summary["i_mix_abs_no"])
        curve_cases.append(
            {
                "L_support_nm": case.value_nm,
                "label": l_support_case_appearance(case.value_nm)[0],
                "E": panel_base.np.asarray(curve["E"], dtype=float),
                "I_Au_abs_A": panel_base.np.asarray(curve["I_Au_abs_A"], dtype=float),
                "I_Pd_abs_A": panel_base.np.asarray(curve["I_Pd_abs_A"], dtype=float),
                "I_total_abs_A": panel_base.np.asarray(curve["I_total_abs_A"], dtype=float),
                "I_Au_current": current_1e_minus_3_uA(curve["I_Au_abs_A"]),
                "I_Pd_current": current_1e_minus_3_uA(curve["I_Pd_abs_A"]),
                "I_total_current": current_1e_minus_3_uA(curve["I_total_abs_A"]),
                "E_mix_with": float(case.summary["E_mix_with"]),
                "E_mix_no": float(case.summary["E_mix_no"]),
                "i_mix_abs_A": i_mix_abs_A,
                "i_mix_current": i_mix_abs_A / PLOT_CURRENT_UNIT_A,
                "i_mix_no_abs_A": i_mix_no_abs_A,
                "i_mix_no_current": i_mix_no_abs_A / PLOT_CURRENT_UNIT_A,
                "i_mix_avg_with_A_per_m2": float(case.summary["i_mix_avg_with"]),
                "i_mix_avg_no_A_per_m2": float(case.summary["i_mix_avg_no"]),
            }
        )
    require_finite_polarization_data(curve_cases)
    return curve_cases


def require_finite_polarization_data(curve_cases: list[dict[str, Any]]) -> None:
    if len(curve_cases) != len(L_SUPPORT_NM_VALUES):
        raise RuntimeError(f"Expected {len(L_SUPPORT_NM_VALUES)} polarization cases, got {len(curve_cases)}")
    reference_e = panel_base.np.asarray(curve_cases[0]["E"], dtype=float)
    for case in curve_cases:
        e_values = panel_base.np.asarray(case["E"], dtype=float)
        if not panel_base.np.allclose(e_values, reference_e, rtol=0.0, atol=1e-14):
            raise ValueError("Polarization cases do not share the same E grid")
        for key in ("I_Au_abs_A", "I_Pd_abs_A", "I_total_abs_A", "I_Au_current", "I_Pd_current", "I_total_current"):
            arr = panel_base.np.asarray(case[key], dtype=float)
            if arr.shape != reference_e.shape:
                raise ValueError(f"{key} shape does not match E grid for L_support={case['L_support_nm']}")
            if not panel_base.np.all(panel_base.np.isfinite(arr)):
                raise ValueError(f"{key} contains non-finite values for L_support={case['L_support_nm']}")


def write_polarization_curve_csv(curve_cases: list[dict[str, Any]]) -> Path:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "L_support_nm",
        "case_label",
        "E_V",
        "I_Au_abs_A",
        "I_Pd_abs_A",
        "I_total_abs_A",
        "I_Au_current_1e_minus_3_uA",
        "I_Pd_current_1e_minus_3_uA",
        "I_total_current_1e_minus_3_uA",
        "E_mix_with_V",
        "i_mix_abs_with_A",
        "i_mix_current_1e_minus_3_uA",
        "i_mix_avg_with_A_per_m2",
        "E_mix_no_V",
        "i_mix_abs_no_A",
        "i_mix_current_no_1e_minus_3_uA",
        "i_mix_avg_no_A_per_m2",
    ]
    with POLARIZATION_CURVE_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in curve_cases:
            e_values = panel_base.np.asarray(case["E"], dtype=float)
            for idx, e_value in enumerate(e_values):
                writer.writerow(
                    {
                        "L_support_nm": float(case["L_support_nm"]),
                        "case_label": case["label"],
                        "E_V": float(e_value),
                        "I_Au_abs_A": float(case["I_Au_abs_A"][idx]),
                        "I_Pd_abs_A": float(case["I_Pd_abs_A"][idx]),
                        "I_total_abs_A": float(case["I_total_abs_A"][idx]),
                        "I_Au_current_1e_minus_3_uA": float(case["I_Au_current"][idx]),
                        "I_Pd_current_1e_minus_3_uA": float(case["I_Pd_current"][idx]),
                        "I_total_current_1e_minus_3_uA": float(case["I_total_current"][idx]),
                        "E_mix_with_V": float(case["E_mix_with"]),
                        "i_mix_abs_with_A": float(case["i_mix_abs_A"]),
                        "i_mix_current_1e_minus_3_uA": float(case["i_mix_current"]),
                        "i_mix_avg_with_A_per_m2": float(case["i_mix_avg_with_A_per_m2"]),
                        "E_mix_no_V": float(case["E_mix_no"]),
                        "i_mix_abs_no_A": float(case["i_mix_no_abs_A"]),
                        "i_mix_current_no_1e_minus_3_uA": float(case["i_mix_no_current"]),
                        "i_mix_avg_no_A_per_m2": float(case["i_mix_avg_no_A_per_m2"]),
                    }
                )
    return POLARIZATION_CURVE_CSV


def polarization_y_limit(curve_cases: list[dict[str, Any]]) -> float:
    curve_max = max(
        float(panel_base.np.nanmax(panel_base.np.abs(case[key])))
        for case in curve_cases
        for key in ("I_Au_current", "I_Pd_current")
    )
    marker_max = max(float(case["i_mix_current"]) for case in curve_cases)
    return max(0.13, 1.08 * curve_max, 1.65 * marker_max)


def plot_l_support_polarization(curve_cases: list[dict[str, Any]]) -> list[Path]:
    e_values = panel_base.np.asarray(curve_cases[0]["E"], dtype=float)
    y_limit = polarization_y_limit(curve_cases)
    fig, ax = panel_base.plt.subplots(figsize=(6.1, 4.35), constrained_layout=False)
    ax.axhline(0.0, color=POLARIZATION_COLORS["dark"], linewidth=0.78, zorder=1)

    for case in curve_cases:
        _label, alpha, linestyle, case_color = l_support_case_appearance(float(case["L_support_nm"]))
        ax.plot(
            e_values,
            panel_base.np.asarray(case["I_Au_current"], dtype=float),
            color=POLARIZATION_COLORS["au_curve"],
            linewidth=2.05,
            linestyle=linestyle,
            alpha=alpha,
            zorder=5,
        )
        ax.plot(
            e_values,
            panel_base.np.asarray(case["I_Pd_current"], dtype=float),
            color=POLARIZATION_COLORS["pd_curve"],
            linewidth=2.05,
            linestyle=linestyle,
            alpha=alpha,
            zorder=5,
        )
        e_mix = float(case["E_mix_with"])
        i_mix = float(case["i_mix_current"])
        ax.axvline(e_mix, color=case_color, linewidth=0.90, linestyle=linestyle, alpha=alpha, zorder=2)
        ax.scatter(
            [e_mix, e_mix],
            [i_mix, -i_mix],
            s=28,
            color=[POLARIZATION_COLORS["au_curve"], POLARIZATION_COLORS["pd_curve"]],
            edgecolor="white",
            linewidth=0.55,
            alpha=min(1.0, alpha + 0.12),
            zorder=8,
        )
        ax.scatter(
            [e_mix],
            [0.0],
            s=24,
            color=case_color,
            edgecolor="white",
            linewidth=0.55,
            alpha=min(1.0, alpha + 0.12),
            zorder=9,
        )

    low_case = next(case for case in curve_cases if math.isclose(float(case["L_support_nm"]), 0.0, rel_tol=0.0, abs_tol=1e-9))
    high_case = next(case for case in curve_cases if math.isclose(float(case["L_support_nm"]), 1000.0, rel_tol=0.0, abs_tol=1e-9))
    e_low = float(low_case["E_mix_with"])
    e_high = float(high_case["E_mix_with"])
    y_arrow = 0.80 * y_limit
    ax.annotate(
        "",
        xy=(e_high, y_arrow),
        xytext=(e_low, y_arrow),
        arrowprops=dict(arrowstyle="->", color=POLARIZATION_COLORS["dark"], linewidth=0.85, shrinkA=0, shrinkB=0),
        zorder=10,
    )
    ax.text(
        0.5 * (e_low + e_high),
        y_arrow + 0.04 * y_limit,
        r"$E_{\mathrm{mix}}$ shifts lower",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=POLARIZATION_COLORS["dark"],
    )
    current_ratio = float(high_case["i_mix_current"]) / float(low_case["i_mix_current"])
    ax.text(
        0.975,
        0.055,
        rf"$|I_{{\mathrm{{mix}}}}|$: {float(low_case['i_mix_current']):.3f} $\to$ {float(high_case['i_mix_current']):.3f}"
        "\n"
        rf"$\Delta E$ = {1000.0 * (e_high - e_low):+.1f} mV, $\times${current_ratio:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.2,
        color=POLARIZATION_COLORS["dark"],
        linespacing=1.12,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.8),
        zorder=11,
    )

    handles = [
        Line2D([0], [0], color=POLARIZATION_COLORS["au_curve"], linewidth=2.35, label="Au anodic"),
        Line2D([0], [0], color=POLARIZATION_COLORS["pd_curve"], linewidth=2.35, label="Pd cathodic"),
    ]
    for value_nm in L_SUPPORT_NM_VALUES:
        label, alpha, linestyle, case_color = l_support_case_appearance(value_nm)
        handles.append(
            Line2D(
                [0],
                [0],
                color=case_color,
                linewidth=1.45,
                linestyle=linestyle,
                alpha=alpha,
                label=rf"$L_{{\mathrm{{support}}}}$ = {label}",
            )
        )
    ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=9.1,
        handlelength=2.4,
        ncols=1,
        columnspacing=0.85,
        labelspacing=0.34,
        borderaxespad=0.25,
    )
    ax.set_xlim(float(panel_base.np.min(e_values)), float(panel_base.np.max(e_values)))
    ax.set_ylim(-y_limit, y_limit)
    ax.set_title(r"$L_{\mathrm{support}}$ polarization curves", loc="left", fontsize=13.9, pad=4.0, color=POLARIZATION_COLORS["dark"])
    ax.set_xlabel("Potential (V vs. RHE)", fontsize=11.6, labelpad=2.5)
    ax.set_ylabel(r"Current (10$^{-3}$ $\mu$A)", fontsize=11.6, labelpad=3.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis="both", length=2.7, width=0.75, pad=1.6, labelsize=10.3)
    for spine in ("left", "right", "top", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.74)
        ax.spines[spine].set_color(POLARIZATION_COLORS["dark"])
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.19, top=0.88)

    saved: list[Path] = []
    for ext in ("png", "svg"):
        path = OUT_DIR / f"{POLARIZATION_STEM}.{ext}"
        fig.savefig(path, dpi=600)
        saved.append(path)
    panel_base.plt.close(fig)
    return saved


def assert_outputs(saved: list[Path]) -> None:
    missing = [path for path in saved if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing expected outputs: {missing}")
    suffix_counts = {".png": 0, ".svg": 0}
    for path in saved:
        suffix_counts[path.suffix] += 1
    if suffix_counts[".png"] != suffix_counts[".svg"]:
        raise RuntimeError(f"PNG/SVG output count mismatch: {suffix_counts}")
    pdfs = sorted(OUT_DIR.glob("*.pdf"))
    if pdfs:
        raise RuntimeError(f"Expected zero PDF outputs, found {len(pdfs)}")


def main() -> None:
    ensure_dirs()
    remove_previous_outputs()
    panel_base.apply_style()
    rp_base.apply_style()

    cases = build_cases()
    phi_ylim = common_phi_ylim(cases)
    current_ylim = common_current_ylim(cases)
    phi_abs = max(float(rp_base.np.nanmax(rp_base.np.abs(case.rp_data.phi_s_mV))) for case in cases)
    phi_vlim = max(10.0, math.ceil(phi_abs / 10.0) * 10.0)

    saved: list[Path] = []
    for case in cases:
        saved.extend(plot_panel_b_case(case, phi_ylim))
        saved.extend(plot_panel_e_case(case, current_ylim))
        saved.extend(plot_phi_s_case(case, phi_vlim))

    zoom_case = next(case for case in cases if math.isclose(case.value_nm, ACTIVE_ZOOM_SUPPORT_NM, rel_tol=0.0, abs_tol=1e-9))
    saved.extend(plot_phi_s_active_zoom(zoom_case, phi_vlim))
    saved.extend(plot_panel_b_active_zoom(zoom_case, phi_ylim))
    saved.extend(plot_panel_e_active_zoom(zoom_case, current_ylim))

    stats_path = save_stats(cases)
    polarization_curve_data = build_polarization_curve_data(cases)
    polarization_csv_path = write_polarization_curve_csv(polarization_curve_data)
    saved.extend(plot_l_support_polarization(polarization_curve_data))
    assert_outputs(saved)

    print(f"same-length equal-i0 alpha=0.5 tag = {common.OUTPUT_TAG}")
    print(f"Saved stats: {stats_path.relative_to(ROOT)}")
    print(f"Saved polarization curves: {polarization_csv_path.relative_to(ROOT)}")
    print(f"Common panel b ylim = {phi_ylim[0]:.6g} to {phi_ylim[1]:.6g} V")
    print(f"Common panel e ylim = {current_ylim[0]:.6g} to {current_ylim[1]:.6g} x 10^-3 A/m^2")
    print(f"Common Phi_s color limit = +/- {phi_vlim:.6g} mV")
    for case in cases:
        print(f"L_support = {format_nm_value(case.value_nm)} nm")
        print(f"  E_mix_with = {float(case.summary['E_mix_with']):.15g} V")
        print(f"  E_mix_no = {float(case.summary['E_mix_no']):.15g} V")
        print(f"  i_mix_avg_with = {float(case.summary['i_mix_avg_with']):.15g} A/m^2")
        print(f"  i_mix_avg_no = {float(case.summary['i_mix_avg_no']):.15g} A/m^2")
        print(f"  max |phi_tilde| = {float(case.summary['max_abs_phi_tilde_with_edl']):.6g}")
    print(f"Verified {len(saved) // 2} PNG/SVG output pairs and zero PDF outputs")
    for path in saved:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
