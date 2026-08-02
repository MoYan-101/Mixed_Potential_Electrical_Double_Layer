from __future__ import annotations

import csv
import copy
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT_ID = "20260528_111255"
SOLVER_DIR = ROOT / "Mixed_Potential_Electrical_Double_Layer"
RESULT_DIR = SOLVER_DIR / "results" / RESULT_ID
PARAMS_PATH = RESULT_DIR / "params.json"
OUT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = OUT_DIR / "inputs"

sys.path.insert(0, str(SOLVER_DIR))
import Solve_Emix_updating as solver  # noqa: E402


COLORS = {
    "with": "#F26B38",
    "with_alt": "#D83A2E",
    "with_gold": "#F2B134",
    "without": "#12355B",
    "without_alt": "#2D5A7B",
    "without_deep": "#0B1F3A",
    "dark": "#272727",
    "gray": "#767676",
    "light_gray": "#CFCECE",
    "pzc_support": "#8C8C8C",
    "pzc_pd": "#5A90C8",
    "pzc_au": "#E4C133",
}

SINGLE_PANEL_FIGSIZE = (3.55, 2.75)
SINGLE_PANEL_AXES_RECT = (0.18, 0.20, 0.78, 0.66)
PANEL_A_FIGSIZE = (4.05, 2.75)
PANEL_F_FIGSIZE = (4.20, 3.25)
PANEL_D_TITLE = "Local overpotential"
PANEL_E_TITLE = "Local current density"
PANEL_E_YMIN: float | None = None
HIGH_RES_N_MODES = 960
HIGH_RES_NX = 5000
TRACEABILITY_STEM = f"figure_3_high_resolution_{RESULT_ID}"


def units_to_parentheses(label: str) -> str:
    return re.sub(r"\s+\[([^\]]+)\]", r" (\1)", label)


@dataclass(frozen=True)
class Figure3Data:
    params: dict[str, Any]
    summary: dict[str, str]
    res_edl: dict[str, Any]
    res_no: dict[str, Any]
    prof_edl: dict[str, Any]
    prof_no: dict[str, Any]
    derived_edl: dict[str, Any]
    derived_no: dict[str, Any]
    x_nm: np.ndarray
    phi_rp_edl: np.ndarray
    phi_rp_no: np.ndarray
    eta_edl: np.ndarray
    eta_no: np.ndarray
    c_r1_norm: np.ndarray
    c_o2_norm: np.ndarray
    i1_edl_segment: np.ndarray
    i2_edl_segment: np.ndarray
    i1_no_segment: np.ndarray
    i2_no_segment: np.ndarray
    L_Au_nm: float
    L_C_nm: float


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 9.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.9,
            "axes.grid": False,
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.edgecolor": "none",
            "savefig.transparent": True,
            "savefig.bbox": None,
            "legend.frameon": False,
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


def make_transparent(fig: plt.Figure) -> None:
    fig.patch.set_alpha(0.0)
    fig.patch.set_facecolor("none")
    for ax in fig.axes:
        ax.set_facecolor("none")


def make_single_axis_panel() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=SINGLE_PANEL_FIGSIZE)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes(SINGLE_PANEL_AXES_RECT)
    ax.set_facecolor("none")
    return fig, ax


def load_params() -> dict[str, Any]:
    with PARAMS_PATH.open("r", encoding="utf-8") as f:
        params_data = json.load(f)
    params = solver.apply_param_overrides(solver.default_params(), params_data, reset_lambda_D=False)
    params["N_modes"] = HIGH_RES_N_MODES
    params["Nx"] = HIGH_RES_NX
    return params


def summary_from_pair(pair: dict[str, Any]) -> dict[str, str]:
    res_edl = pair["with_edl"]
    res_no = pair["no_edl"]
    comparison = pair["comparison"]
    values: dict[str, Any] = {
        "pH": res_edl.get("pH"),
        "pH_ref": res_edl.get("pH_ref"),
        "delta_pH": res_edl.get("delta_pH"),
        "E1_eq_eff": res_edl.get("E1_eq_eff"),
        "E2_eq_eff": res_edl.get("E2_eq_eff"),
        "it0_1_eff": res_edl.get("it0_1_eff"),
        "it0_2_eff": res_edl.get("it0_2_eff"),
        "E_mix_with": res_edl.get("E_mix"),
        "i_mix_with": res_edl.get("i_mix"),
        "i_mix_norm_with": res_edl.get("i_mix"),
        "i_mix_phys_with": res_edl.get("i_mix_phys_A_per_m"),
        "i_mix_abs_with": res_edl.get("i_mix_abs_A"),
        "i_mix_avg_with": res_edl.get("i_mix_avg_A_per_m2"),
        "E_mix_no": res_no.get("E_mix"),
        "i_mix_no": res_no.get("i_mix"),
        "i_mix_norm_no": res_no.get("i_mix"),
        "i_mix_phys_no": res_no.get("i_mix_phys_A_per_m"),
        "i_mix_abs_no": res_no.get("i_mix_abs_A"),
        "i_mix_avg_no": res_no.get("i_mix_avg_A_per_m2"),
        "delta_E_mix": comparison.get("delta_E_mix"),
        "delta_i_mix_avg_A_per_m2": comparison.get("delta_i_mix_avg_A_per_m2"),
        "ratio_i_mix_avg": comparison.get("ratio_i_mix_avg"),
        "pct_i_mix_avg": comparison.get("pct_i_mix_avg"),
        "ratio_i_mix": comparison.get("ratio_i_mix"),
        "pct_i_mix": comparison.get("pct_i_mix"),
        "ratio_i_mix_phys": comparison.get("ratio_i_mix_phys"),
        "pct_i_mix_phys": comparison.get("pct_i_mix_phys"),
        "delta_i_mix_abs_A": comparison.get("delta_i_mix_abs_A"),
        "ratio_i_mix_abs": comparison.get("ratio_i_mix_abs"),
        "pct_i_mix_abs": comparison.get("pct_i_mix_abs"),
        "max_abs_phi_tilde_with_edl": res_edl.get("max_abs_phi_tilde"),
        "debye_huckel_ok_with_edl": res_edl.get("debye_huckel_ok"),
        "mode": comparison.get("mode", "FULL"),
    }
    return {key: "" if value is None else str(value) for key, value in values.items()}


def save_traceability_inputs(params: dict[str, Any], summary: dict[str, str]) -> None:
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    params_to_save = copy.deepcopy(params)
    overrides = {
        "source_params": str(PARAMS_PATH.relative_to(ROOT)),
        "N_modes": HIGH_RES_N_MODES,
        "Nx": HIGH_RES_NX,
    }
    with (INPUTS_DIR / f"params_{TRACEABILITY_STEM}.json").open("w", encoding="utf-8") as f:
        json.dump(params_to_save, f, indent=2, sort_keys=True)
        f.write("\n")
    with (INPUTS_DIR / f"overrides_{TRACEABILITY_STEM}.json").open("w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2, sort_keys=True)
        f.write("\n")
    with (INPUTS_DIR / f"summary_compare_{TRACEABILITY_STEM}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    with (INPUTS_DIR / f"summary_compare_{TRACEABILITY_STEM}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")


def build_data() -> Figure3Data:
    params = load_params()

    pair = solver.run_edl_comparison_pair(params, mode="FULL")
    res_edl = pair["with_edl"]
    res_no = pair["no_edl"]
    summary = summary_from_pair(pair)
    save_traceability_inputs(params, summary)

    prof_edl, derived_edl = solver.build_profiles_for_emix(params, float(res_edl["E_mix"]), use_edl=True)
    prof_no, derived_no = solver.build_profiles_for_emix(params, float(res_no["E_mix"]), use_edl=False)

    scale_edl = float(derived_edl["R"]) * float(derived_edl["T"]) / float(derived_edl["F"])
    scale_no = float(derived_no["R"]) * float(derived_no["T"]) / float(derived_no["F"])

    x_nm = np.asarray(prof_edl["x_tilde"], dtype=float) * float(derived_edl["lambda_D"]) * 1e9
    phi_rp_edl = scale_edl * np.asarray(prof_edl["phi_tilde"], dtype=float)
    phi_rp_no = scale_no * np.asarray(prof_no["phi_tilde"], dtype=float)

    mask_Au = np.asarray(prof_edl["mask_Au"], dtype=bool)
    mask_Pd = np.asarray(prof_edl["mask_Pd"], dtype=bool)
    mask_Au_no = np.asarray(prof_no["mask_Au"], dtype=bool)
    mask_Pd_no = np.asarray(prof_no["mask_Pd"], dtype=bool)

    E1_eq = float(res_edl["E1_eq_eff"])
    E2_eq = float(res_edl["E2_eq_eff"])
    eta_edl = np.full_like(x_nm, np.nan, dtype=float)
    eta_no = np.full_like(x_nm, np.nan, dtype=float)
    eta_edl[mask_Au] = float(res_edl["E_mix"]) - E1_eq - phi_rp_edl[mask_Au]
    eta_edl[mask_Pd] = float(res_edl["E_mix"]) - E2_eq - phi_rp_edl[mask_Pd]
    eta_no[mask_Au_no] = float(res_no["E_mix"]) - E1_eq - phi_rp_no[mask_Au_no]
    eta_no[mask_Pd_no] = float(res_no["E_mix"]) - E2_eq - phi_rp_no[mask_Pd_no]

    phi_tilde_edl = np.asarray(prof_edl["phi_tilde"], dtype=float)
    c_r1_norm = np.asarray(solver.safe_exp(-float(params["z_R1"]) * phi_tilde_edl), dtype=float)
    c_o2_norm = np.asarray(solver.safe_exp(-float(params["z_O2"]) * phi_tilde_edl), dtype=float)

    i1_edl = np.asarray(prof_edl["i1"], dtype=float)
    i2_edl = np.asarray(prof_edl["i2"], dtype=float)
    i1_no = np.asarray(prof_no["i1"], dtype=float)
    i2_no = np.asarray(prof_no["i2"], dtype=float)
    i1_edl_segment = np.full_like(i1_edl, np.nan, dtype=float)
    i2_edl_segment = np.full_like(i2_edl, np.nan, dtype=float)
    i1_no_segment = np.full_like(i1_no, np.nan, dtype=float)
    i2_no_segment = np.full_like(i2_no, np.nan, dtype=float)
    i1_edl_segment[mask_Au] = i1_edl[mask_Au]
    i2_edl_segment[mask_Pd] = i2_edl[mask_Pd]
    i1_no_segment[mask_Au_no] = i1_no[mask_Au_no]
    i2_no_segment[mask_Pd_no] = i2_no[mask_Pd_no]

    L_Au_nm = float(derived_edl["L_Au_tilde"]) * float(derived_edl["lambda_D"]) * 1e9
    L_C_nm = float(derived_edl["L_C_tilde"]) * float(derived_edl["lambda_D"]) * 1e9

    return Figure3Data(
        params=params,
        summary=summary,
        res_edl=res_edl,
        res_no=res_no,
        prof_edl=prof_edl,
        prof_no=prof_no,
        derived_edl=derived_edl,
        derived_no=derived_no,
        x_nm=x_nm,
        phi_rp_edl=phi_rp_edl,
        phi_rp_no=phi_rp_no,
        eta_edl=eta_edl,
        eta_no=eta_no,
        c_r1_norm=c_r1_norm,
        c_o2_norm=c_o2_norm,
        i1_edl_segment=i1_edl_segment,
        i2_edl_segment=i2_edl_segment,
        i1_no_segment=i1_no_segment,
        i2_no_segment=i2_no_segment,
        L_Au_nm=L_Au_nm,
        L_C_nm=L_C_nm,
    )


def style_axes(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_facecolor("none")
    ax.set_xlabel(units_to_parentheses(xlabel))
    ax.set_ylabel(units_to_parentheses(ylabel))
    ax.xaxis.label.set_size(10.4)
    ax.yaxis.label.set_size(10.4)
    ax.set_title(title, loc="left", pad=6, fontsize=11.0, fontweight="normal")
    ax.tick_params(length=3.4, width=0.85, pad=2.5, labelsize=9.5)
    for spine in ("left", "bottom", "top", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.9)
        ax.spines[spine].set_color(COLORS["dark"])


def add_boundaries(ax: plt.Axes, data: Figure3Data) -> None:
    for xpos in (data.L_Au_nm, data.L_C_nm):
        ax.axvline(xpos, linestyle=(0, (3, 2)), linewidth=0.9, color=COLORS["gray"], alpha=0.85, zorder=1)


def save_panel(fig: plt.Figure, stem: str) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in ("png", "svg"):
        path = OUT_DIR / f"{stem}_{RESULT_ID}.{ext}"
        make_transparent(fig)
        fig.savefig(path, dpi=600, transparent=True, facecolor="none", edgecolor="none")
        paths.append(path)
    plt.close(fig)
    return paths


def finite_ylim(ax: plt.Axes, *arrays: np.ndarray, pad_frac: float = 0.08) -> None:
    vals = np.concatenate([np.ravel(np.asarray(arr, dtype=float)) for arr in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    ymin = float(np.min(vals))
    ymax = float(np.max(vals))
    span = ymax - ymin
    pad = max(1e-6, pad_frac * span)
    ax.set_ylim(ymin - pad, ymax + pad)


def plot_panel_a(data: Figure3Data) -> list[Path]:
    values = np.array(
        [
            [float(data.res_no["E_mix"]), float(data.res_edl["E_mix"])],
            [float(data.res_no["i_mix_avg_A_per_m2"]), float(data.res_edl["i_mix_avg_A_per_m2"])],
        ],
        dtype=float,
    )
    (i_mix_plot,), i_label, _ = solver._scaled_current_display("i_mix_avg", values[1])
    i_label = i_label.replace("Average mixed current density, ", "")

    fig, axes = plt.subplots(1, 2, figsize=PANEL_A_FIGSIZE)
    make_transparent(fig)
    labels = ["w/o EDL", "with EDL"]
    x = np.array([0.0, 1.0], dtype=float)
    colors = [COLORS["without"], COLORS["with"]]

    axes[0].bar(x, values[0], width=0.58, color=colors, edgecolor=COLORS["dark"], linewidth=0.8)
    style_axes(axes[0], "", r"$E_{\mathrm{mix}}$ (V)", r"$E_{\mathrm{mix}}$")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    axes[0].set_xlim(-0.55, 1.55)
    axes[0].set_ylim(0.0, max(values[0]) * 1.18)

    axes[1].bar(x, i_mix_plot, width=0.58, color=colors, edgecolor=COLORS["dark"], linewidth=0.8)
    style_axes(axes[1], "", i_label, r"$i_{\mathrm{mix}}$")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    axes[1].set_xlim(-0.55, 1.55)
    axes[1].set_ylim(0.0, max(i_mix_plot) * 1.18)

    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.32, top=0.84, wspace=0.60)
    return save_panel(fig, "figure_3_panel_a_emix_imix")


def plot_panel_b(data: Figure3Data) -> list[Path]:
    fig, ax = make_single_axis_panel()
    ax.plot(data.x_nm, data.phi_rp_edl, color=COLORS["with"], linewidth=2.0, label="with EDL", zorder=3)
    ax.plot(data.x_nm, data.phi_rp_no, color=COLORS["without"], linewidth=1.8, label="w/o EDL", zorder=2)
    add_boundaries(ax, data)
    style_axes(ax, "x (nm)", r"$\phi_{\mathrm{RP}}(x)$ (V)", "Reaction-plane potential")
    ax.legend(loc="center right", bbox_to_anchor=(0.98, 0.50), fontsize=8.9, handlelength=2.0)
    return save_panel(fig, "figure_3_panel_b_reaction_plane_potential")


def plot_panel_c(data: Figure3Data) -> list[Path]:
    fig, ax = make_single_axis_panel()
    ax.plot(
        data.x_nm,
        data.c_r1_norm,
        color=COLORS["with"],
        linewidth=2.0,
        label=r"$C_{\mathrm{Red},1}/C_{\mathrm{bulk}}$ (with EDL)",
        zorder=3,
    )
    ax.plot(
        data.x_nm,
        data.c_o2_norm,
        color=COLORS["with_gold"],
        linewidth=2.0,
        label=r"$C_{\mathrm{Ox},2}/C_{\mathrm{bulk}}$ (with EDL)",
        zorder=3,
    )
    ax.plot(
        data.x_nm,
        np.ones_like(data.x_nm),
        color=COLORS["without"],
        linewidth=1.6,
        linestyle=(0, (4, 2)),
        label="w/o EDL",
        zorder=2,
    )
    add_boundaries(ax, data)
    ax.set_yscale("log")
    positive = np.concatenate(
        [
            data.c_r1_norm[np.isfinite(data.c_r1_norm) & (data.c_r1_norm > 0.0)],
            data.c_o2_norm[np.isfinite(data.c_o2_norm) & (data.c_o2_norm > 0.0)],
            np.array([1.0], dtype=float),
        ]
    )
    ax.set_ylim(float(np.min(positive)) / 1.25, float(np.max(positive)) * 8.0)
    style_axes(ax, "x (nm)", r"$c_i/c_{\mathrm{bulk}}$ (-)", "Local reactant concentration")
    ax.legend(loc="upper right", fontsize=7.9, handlelength=1.7)
    return save_panel(fig, "figure_3_panel_c_local_reactant_concentration")


def plot_panel_d(data: Figure3Data) -> list[Path]:
    fig, ax = make_single_axis_panel()
    ax.plot(data.x_nm, data.eta_edl, color=COLORS["with"], linewidth=2.0, label="with EDL", zorder=3)
    ax.plot(data.x_nm, data.eta_no, color=COLORS["without"], linewidth=1.8, label="w/o EDL", zorder=2)
    add_boundaries(ax, data)
    style_axes(ax, "x (nm)", solver._plot_axis_label("overpotential"), PANEL_D_TITLE)
    finite_ylim(ax, data.eta_edl, data.eta_no, pad_frac=0.08)
    ax.legend(loc="center right", fontsize=8.9, handlelength=2.0)
    return save_panel(fig, "figure_3_panel_d_local_overpotential")


def plot_panel_e(data: Figure3Data) -> list[Path]:
    (i1_edl, i1_no, i2_edl, i2_no), i_label, _ = solver._scaled_current_display(
        "local_current_density",
        data.i1_edl_segment,
        data.i1_no_segment,
        data.i2_edl_segment,
        data.i2_no_segment,
    )
    i_label = i_label.replace("Local current density, ", "")

    fig, ax = make_single_axis_panel()
    ax.plot(data.x_nm, i1_edl, color=COLORS["with"], linewidth=2.0, label=r"$i_1$ (Au), with EDL", zorder=4)
    ax.plot(data.x_nm, i2_edl, color=COLORS["with_alt"], linewidth=2.0, label=r"$i_2$ (Pd), with EDL", zorder=4)
    ax.plot(
        data.x_nm,
        i1_no,
        color=COLORS["without"],
        linewidth=1.7,
        linestyle=(0, (4, 2)),
        label=r"$i_1$ (Au), w/o EDL",
        zorder=3,
    )
    ax.plot(
        data.x_nm,
        i2_no,
        color=COLORS["without_alt"],
        linewidth=1.7,
        linestyle=(0, (2, 2)),
        label=r"$i_2$ (Pd), w/o EDL",
        zorder=3,
    )
    add_boundaries(ax, data)
    style_axes(ax, "x (nm)", i_label, PANEL_E_TITLE)
    finite_ylim(ax, i1_edl, i1_no, i2_edl, i2_no, pad_frac=0.08)
    if PANEL_E_YMIN is not None:
        _, ymax = ax.get_ylim()
        ax.set_ylim(PANEL_E_YMIN, ymax)
    ax.legend(loc="upper right", fontsize=7.8, handlelength=1.8)
    return save_panel(fig, "figure_3_panel_e_local_current_density")


def fmt_v(value: float) -> str:
    return f"{value:.2f} V"


def add_reference_marker(
    ax: plt.Axes,
    x: float,
    y: float,
    label: str,
    color: str,
    marker: str,
    text_y_offset: float,
    text_x_offset: float = 0.0,
    linewidth: float = 1.7,
    linestyle: str | tuple[int, tuple[int, ...]] = "-",
) -> None:
    ax.vlines(x, y - 0.09, y + 0.09, color=color, linewidth=linewidth, linestyle=linestyle, zorder=2)
    ax.scatter([x], [y], s=72, marker=marker, color=color, edgecolor="none", linewidth=0.0, zorder=4)
    ax.text(
        x + text_x_offset,
        y + text_y_offset,
        f"{label}\n{fmt_v(x)}",
        ha="center",
        va="bottom" if text_y_offset >= 0 else "top",
        fontsize=8.5,
        color=color,
        linespacing=1.1,
    )


def plot_panel_f(data: Figure3Data) -> list[Path]:
    values = {
        "E1_eq": float(data.summary["E1_eq_eff"]),
        "E2_eq": float(data.summary["E2_eq_eff"]),
        "E_mix_with": float(data.summary["E_mix_with"]),
        "E_mix_no": float(data.summary["E_mix_no"]),
        "pzc_C": float(data.params["pzc_C"]),
        "pzc_Pd": float(data.params["pzc_Pd"]),
        "pzc_Au": float(data.params["pzc_Au"]),
    }

    fig, ax = plt.subplots(figsize=PANEL_F_FIGSIZE)
    make_transparent(fig)
    xmin, xmax = 0.0, 1.0
    lane_y = {"eq": 0.42, "mix": 0.00, "pzc": -0.42}
    ax.hlines(0.0, xmin, xmax, color=COLORS["dark"], linewidth=1.1, zorder=1)

    add_reference_marker(ax, values["E1_eq"], lane_y["eq"], r"$E_{1,\mathrm{eq}}$", COLORS["gray"], "o", 0.12)
    add_reference_marker(ax, values["E2_eq"], lane_y["eq"], r"$E_{2,\mathrm{eq}}$", COLORS["gray"], "o", 0.12)
    add_reference_marker(
        ax,
        values["E_mix_no"],
        lane_y["mix"],
        r"$E_{\mathrm{mix}}$ w/o EDL",
        COLORS["without"],
        "D",
        -0.15,
        text_x_offset=-0.070,
        linewidth=2.0,
    )
    add_reference_marker(
        ax,
        values["E_mix_with"],
        lane_y["mix"],
        r"$E_{\mathrm{mix}}$ with EDL",
        COLORS["with"],
        "D",
        0.13,
        linewidth=2.0,
    )
    add_reference_marker(
        ax,
        values["pzc_C"],
        lane_y["pzc"],
        "PZC support",
        COLORS["pzc_support"],
        "^",
        -0.13,
        text_x_offset=0.035,
    )
    add_reference_marker(ax, values["pzc_Pd"], lane_y["pzc"], "PZC Pd", COLORS["pzc_pd"], "^", -0.13, text_x_offset=-0.02)
    add_reference_marker(ax, values["pzc_Au"], lane_y["pzc"], "PZC Au", COLORS["pzc_au"], "^", -0.13)

    ax.set_xlim(xmin - 0.02, xmax + 0.02)
    ax.set_ylim(-0.92, 0.80)
    ax.set_xlabel("Potential (V vs. RHE)")
    ax.xaxis.label.set_size(10.6)
    ax.set_yticks([])
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.tick_params(axis="x", length=3.4, width=0.85, labelsize=9.5)
    ax.set_title("Potential reference map", loc="left", fontsize=11.0, pad=6, fontweight="normal")
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_position(("data", -0.72))
    ax.spines["bottom"].set_linewidth(0.9)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.17, top=0.89)
    return save_panel(fig, "figure_3_panel_f_pzc_potential_reference_map")


def main() -> None:
    apply_style()
    data = build_data()

    saved: list[Path] = []
    saved.extend(plot_panel_a(data))
    saved.extend(plot_panel_b(data))
    saved.extend(plot_panel_c(data))
    saved.extend(plot_panel_d(data))
    saved.extend(plot_panel_e(data))
    saved.extend(plot_panel_f(data))

    if len(saved) != 12:
        raise RuntimeError(f"Expected 12 exported files, got {len(saved)}")

    print(f"Using high-resolution Figure 3 settings: N_modes = {HIGH_RES_N_MODES}, Nx = {HIGH_RES_NX}")
    print(f"E_mix_with = {float(data.res_edl['E_mix']):.15g} V")
    print(f"E_mix_no = {float(data.res_no['E_mix']):.15g} V")
    print(f"i_mix_avg_with = {float(data.res_edl['i_mix_avg_A_per_m2']):.15g} A/m^2")
    print(f"i_mix_avg_no = {float(data.res_no['i_mix_avg_A_per_m2']):.15g} A/m^2")
    print(f"Saved traceability inputs to {INPUTS_DIR}")
    print("Saved Figure 3 panels:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
