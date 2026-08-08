"""Traceable result writer and publication figures for independent EDLs."""

from __future__ import annotations

import csv
import json
import math
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D

from .model import (
    ELECTROSTATIC_BACKEND,
    MODEL_ID,
    RESULT_SCHEMA_VERSION,
    TOPOLOGY_ID,
    IndependentPlanarEDLModel,
    solve_comparison,
)


DPI = 600
COLORS = {
    "with_edl": "#F26B38",
    "without_edl": "#12355B",
    "au": "#D4A923",
    "pd": "#4599A3",
    "red1_i1": "#009E73",
    "ox2_i2": "#0072B2",
    "dark": "#242424",
    "gray": "#7F7F7F",
}
RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "svg.fonttype": "none",
    "axes.linewidth": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _jsonable(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _save(fig: plt.Figure, directory: Path, stem: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in ("png", "svg"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(
            path,
            dpi=DPI,
            bbox_inches="tight",
            pad_inches=0.04,
            transparent=True,
            facecolor="none",
            edgecolor="none",
        )
        paths.append(path)
    plt.close(fig)
    return paths


def _style_axis(ax: plt.Axes, ylabel: str, *, show_ylabel: bool) -> None:
    ax.set_facecolor("none")
    ax.set_xlabel("local x (nm)", fontsize=9.8)
    ax.set_ylabel(ylabel if show_ylabel else "", fontsize=9.8)
    if not show_ylabel:
        ax.tick_params(left=False, labelleft=False)
    ax.tick_params(length=3.3, width=0.85, pad=2.5, labelsize=8.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color(COLORS["dark"])


def _finite_ylim(axes: tuple[plt.Axes, plt.Axes], arrays: list[np.ndarray]) -> None:
    values = np.concatenate([np.ravel(array) for array in arrays])
    values = values[np.isfinite(values)]
    low, high = float(np.min(values)), float(np.max(values))
    pad = max(1.0e-6, 0.09 * (high - low))
    axes[0].set_ylim(low - pad, high + pad)


def _two_surface_panel(
    model: IndependentPlanarEDLModel,
    title: str,
    ylabel: str,
    plotter: Callable[[plt.Axes, str, np.ndarray], list[Line2D]],
    ylim_arrays: list[np.ndarray],
    output_dir: Path,
    stem: str,
    *,
    log_scale: bool = False,
) -> list[Path]:
    fig = plt.figure(figsize=(7.0, 2.85))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.0, 0.72),
        left=0.105,
        right=0.985,
        bottom=0.22,
        top=0.80,
        wspace=0.16,
    )
    axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]))
    axes[1].sharey(axes[0])
    legend_ax = fig.add_subplot(grid[0, 2])
    legend_ax.set_axis_off()
    lines: list[Line2D] = []
    for index, (ax, material) in enumerate(zip(axes, ("Au", "Pd"), strict=True)):
        length_nm = float(model.params[f"L_{material}"]) * 1.0e9
        x_nm = np.linspace(0.0, length_nm, 401)
        lines.extend(plotter(ax, material, x_nm))
        if log_scale:
            ax.set_yscale("log")
        _style_axis(ax, ylabel, show_ylabel=index == 0)
        ax.set_xlim(0.0, length_nm)
        ax.set_xticks([0.0, 0.5 * length_nm, length_nm])
        ax.set_title(
            f"{material} local interface",
            loc="left",
            pad=4,
            fontsize=8.4,
            fontweight="normal",
            color=COLORS["dark"],
        )
    if log_scale:
        positive = np.concatenate([array[array > 0.0] for array in ylim_arrays])
        axes[0].set_ylim(float(np.min(positive)) / 1.25, float(np.max(positive)) * 5.0)
    else:
        _finite_ylim(axes, ylim_arrays)
    unique: dict[str, Line2D] = {}
    for line in lines:
        label = line.get_label()
        if label and not label.startswith("_"):
            unique.setdefault(label, line)
    legend_ax.text(
        0.0,
        0.96,
        "Independent local EDLs",
        transform=legend_ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color=COLORS["dark"],
    )
    legend_ax.text(
        0.0,
        0.82,
        "no spatial EDL coupling",
        transform=legend_ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=COLORS["gray"],
    )
    legend_ax.legend(
        list(unique.values()),
        list(unique),
        loc="center left",
        bbox_to_anchor=(-0.02, 0.43),
        fontsize=7.6,
        handlelength=2.0,
        borderaxespad=0.0,
    )
    fig.suptitle(title, x=0.105, y=0.965, ha="left", fontsize=10.8)
    return _save(fig, output_dir, stem)


def _figure3(
    model: IndependentPlanarEDLModel,
    result: Mapping[str, Any],
    output_dir: Path,
) -> list[Path]:
    with_edl = result["with_edl"]
    without = result["without_edl"]
    E_with = float(with_edl["E_mix_V"])
    E_no = float(without["E_mix_V"])
    thermal = model.derived["thermal_voltage_V"]
    phi = {
        "Au": float(with_edl["phi_RP_Au_tilde"]),
        "Pd": float(with_edl["phi_RP_Pd_tilde"]),
    }
    reaction = model.reaction
    saved: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(4.05, 2.75))
    x = np.arange(2)
    values = (
        (E_no, E_with),
        (
            float(without["i_mix_avg_A_per_m2"]),
            float(with_edl["i_mix_avg_A_per_m2"]),
        ),
    )
    for ax, vals, title, ylabel in zip(
        axes,
        values,
        (r"$E_{\mathrm{mix}}$", r"$\bar{i}_{\mathrm{mix}}$"),
        (r"$E_{\mathrm{mix}}$ (V)", r"$\bar{i}_{\mathrm{mix}}$ (A/m$^2$)"),
        strict=True,
    ):
        ax.bar(
            x,
            vals,
            width=0.58,
            color=(COLORS["without_edl"], COLORS["with_edl"]),
            edgecolor=COLORS["dark"],
            linewidth=0.8,
        )
        ax.set_title(title, loc="left", fontsize=10.8)
        ax.set_ylabel(ylabel, fontsize=9.6)
        ax.set_xticks(x, ("w/o EDL", "with EDL"), rotation=45, ha="right")
        ax.tick_params(labelsize=8.7, length=3.2)
        ax.set_ylim(0.0, max(vals) * 1.18)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.32, top=0.84, wspace=0.60)
    saved.extend(_save(fig, output_dir, "figure_3_panel_a_emix_imix_independent_edls"))

    phi_v = {key: value * thermal for key, value in phi.items()}

    def plot_b(ax: plt.Axes, material: str, x_nm: np.ndarray) -> list[Line2D]:
        return [
            ax.plot(x_nm, np.full_like(x_nm, phi_v[material]), color=COLORS["with_edl"], lw=2.0, label="with EDL")[0],
            ax.plot(x_nm, np.zeros_like(x_nm), color=COLORS["without_edl"], lw=1.8, label="w/o EDL")[0],
        ]

    saved.extend(
        _two_surface_panel(
            model,
            "Independent reaction-plane potential",
            r"$\phi_{\mathrm{RP}}$ (V)",
            plot_b,
            [np.asarray(list(phi_v.values())), np.asarray([0.0])],
            output_dir,
            "figure_3_panel_b_reaction_plane_potential_independent_edls",
        )
    )

    concentration = {
        material: (
            math.exp(np.clip(-float(model.params["z_R1"]) * phi[material], -700.0, 700.0)),
            math.exp(np.clip(-float(model.params["z_O2"]) * phi[material], -700.0, 700.0)),
        )
        for material in ("Au", "Pd")
    }

    def plot_c(ax: plt.Axes, material: str, x_nm: np.ndarray) -> list[Line2D]:
        red1, ox2 = concentration[material]
        return [
            ax.plot(x_nm, np.full_like(x_nm, red1), color=COLORS["red1_i1"], lw=2.0, label=r"$c_{\mathrm{Red1}}/c_{\mathrm{bulk}}$ (with EDL)")[0],
            ax.plot(x_nm, np.full_like(x_nm, ox2), color=COLORS["ox2_i2"], lw=2.0, label=r"$c_{\mathrm{Ox2}}/c_{\mathrm{bulk}}$ (with EDL)")[0],
            ax.plot(x_nm, np.ones_like(x_nm), color=COLORS["without_edl"], lw=1.6, ls=(0, (4, 2)), label="w/o EDL")[0],
        ]

    concentration_arrays = [
        np.asarray([value for pair in concentration.values() for value in pair]),
        np.asarray([1.0]),
    ]
    saved.extend(
        _two_surface_panel(
            model,
            "Independent reactant concentration at RP",
            r"$c_i/c_{\mathrm{bulk}}$ (-)",
            plot_c,
            concentration_arrays,
            output_dir,
            "figure_3_panel_c_reactant_concentration_independent_edls",
            log_scale=True,
        )
    )

    eta_with = {
        "Au": E_with - reaction["E1_eq_eff"] - phi_v["Au"],
        "Pd": E_with - reaction["E2_eq_eff"] - phi_v["Pd"],
    }
    eta_no = {
        "Au": E_no - reaction["E1_eq_eff"],
        "Pd": E_no - reaction["E2_eq_eff"],
    }

    def plot_d(ax: plt.Axes, material: str, x_nm: np.ndarray) -> list[Line2D]:
        return [
            ax.plot(x_nm, np.full_like(x_nm, eta_with[material]), color=COLORS["with_edl"], lw=2.0, label="with EDL")[0],
            ax.plot(x_nm, np.full_like(x_nm, eta_no[material]), color=COLORS["without_edl"], lw=1.8, label="w/o EDL")[0],
        ]

    saved.extend(
        _two_surface_panel(
            model,
            "Independent local overpotential at RP",
            r"Local overpotential, $\eta$ (V)",
            plot_d,
            [np.asarray(list(eta_with.values())), np.asarray(list(eta_no.values()))],
            output_dir,
            "figure_3_panel_d_local_overpotential_independent_edls",
        )
    )

    j_with = {
        "Au": float(with_edl["j_Au_A_per_m2"]),
        "Pd": float(with_edl["j_Pd_A_per_m2"]),
    }
    j_no = {
        "Au": float(without["j_Au_A_per_m2"]),
        "Pd": float(without["j_Pd_A_per_m2"]),
    }

    def plot_e(ax: plt.Axes, material: str, x_nm: np.ndarray) -> list[Line2D]:
        color = COLORS["red1_i1"] if material == "Au" else COLORS["ox2_i2"]
        index = "1" if material == "Au" else "2"
        return [
            ax.plot(x_nm, np.full_like(x_nm, j_with[material]), color=color, lw=2.0, label=rf"$i_{index}$ ({material}), with EDL")[0],
            ax.plot(x_nm, np.full_like(x_nm, j_no[material]), color=color, lw=1.7, ls=(0, (4, 2)), label=rf"$i_{index}$ ({material}), w/o EDL")[0],
        ]

    saved.extend(
        _two_surface_panel(
            model,
            "Independent local current density at RP",
            r"Local current density (A/m$^2$)",
            plot_e,
            [np.asarray(list(j_with.values())), np.asarray(list(j_no.values()))],
            output_dir,
            "figure_3_panel_e_local_current_density_independent_edls",
        )
    )

    fig, ax = plt.subplots(figsize=(4.2, 3.25))
    lane = {"eq": 0.42, "mix": 0.0, "pzc": -0.42}
    ax.hlines(0.0, 0.0, 1.0, color=COLORS["dark"], linewidth=1.1)
    markers = (
        (reaction["E1_eq_eff"], lane["eq"], r"$E_{1,\mathrm{eq}}$", COLORS["gray"], "o"),
        (reaction["E2_eq_eff"], lane["eq"], r"$E_{2,\mathrm{eq}}$", COLORS["gray"], "o"),
        (E_no, lane["mix"], r"$E_{\mathrm{mix}}$ w/o EDL", COLORS["without_edl"], "D"),
        (E_with, lane["mix"], r"$E_{\mathrm{mix}}$ with EDL", COLORS["with_edl"], "D"),
        (float(model.params["pzc_Pd"]), lane["pzc"], "PZC Pd", "#5A90C8", "^"),
        (float(model.params["pzc_Au"]), lane["pzc"], "PZC Au", "#E4C133", "^"),
    )
    for index, (value, y, label, color, marker) in enumerate(markers):
        ax.vlines(value, y - 0.08, y + 0.08, color=color, lw=1.7)
        ax.scatter([value], [y], color=color, marker=marker, s=62, zorder=3)
        offset = 0.11 if y > 0.0 or (y == 0.0 and index == 3) else -0.12
        ax.text(value, y + offset, f"{label}\n{value:.2f} V", ha="center", va="bottom" if offset > 0 else "top", fontsize=7.7, color=color)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.90, 0.78)
    ax.set_xlabel("Potential (V vs. RHE)", fontsize=10.2)
    ax.set_yticks([])
    ax.set_title("Potential reference map", loc="left", fontsize=10.8)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_position(("data", -0.70))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.17, top=0.89)
    saved.extend(_save(fig, output_dir, "figure_3_panel_f_pzc_potential_reference_map_independent_edls"))
    return saved


def _rp_2d(
    model: IndependentPlanarEDLModel,
    result: Mapping[str, Any],
    output_dir: Path,
) -> tuple[list[Path], list[dict[str, float]]]:
    with_edl = result["with_edl"]
    E = float(with_edl["E_mix_V"])
    lambda_D = model.derived["lambda_D"]
    y_nm = np.linspace(0.0, 5.0 * lambda_D * 1.0e9, 241)
    fields: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, float]] = []
    for material in ("Au", "Pd"):
        length_nm = float(model.params[f"L_{material}"]) * 1.0e9
        x_nm = np.linspace(0.0, length_nm, 401)
        phi_profile = model.phi_tilde_profile(E, material, y_nm * 1.0e-9)
        phi_tilde = np.repeat(phi_profile[:, None], x_nm.size, axis=1)
        fields[material] = {
            "x_nm": x_nm,
            "phi_tilde": phi_tilde,
            "phi_mV": phi_tilde * model.derived["thermal_voltage_V"] * 1.0e3,
            "red1": np.exp(np.clip(-float(model.params["z_R1"]) * phi_tilde, -700.0, 700.0)),
            "ox2": np.exp(np.clip(-float(model.params["z_O2"]) * phi_tilde, -700.0, 700.0)),
        }
        for index, y in enumerate(y_nm):
            rows.append(
                {
                    "material": material,
                    "y_nm": float(y),
                    "phi_tilde": float(phi_profile[index]),
                    "phi_s_mV": float(phi_profile[index] * model.derived["thermal_voltage_V"] * 1.0e3),
                    "c_Red1_over_c_bulk": float(math.exp(np.clip(-float(model.params["z_R1"]) * phi_profile[index], -700.0, 700.0))),
                    "c_Ox2_over_c_bulk": float(math.exp(np.clip(-float(model.params["z_O2"]) * phi_profile[index], -700.0, 700.0))),
                }
            )

    maximum = max(float(np.max(np.abs(fields[m]["phi_mV"]))) for m in fields)
    potential_norm = TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)
    saved: list[Path] = []
    fig = plt.figure(figsize=(6.8, 3.25))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.065), wspace=0.16)
    axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]))
    cax = fig.add_subplot(grid[0, 2])
    mesh = None
    for index, (ax, material) in enumerate(zip(axes, ("Au", "Pd"), strict=True)):
        field = fields[material]
        mesh = ax.pcolormesh(
            field["x_nm"],
            y_nm,
            field["phi_mV"],
            shading="auto",
            cmap="RdBu_r",
            norm=potential_norm,
            rasterized=True,
        )
        ax.set_title(f"{material} independent local EDL", loc="left", fontsize=9.5)
        ax.set_xlabel("local x (nm)")
        ax.set_ylabel("distance into electrolyte (nm)" if index == 0 else "")
        if index:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=8.5)
    assert mesh is not None
    colorbar = fig.colorbar(mesh, cax=cax)
    colorbar.set_label(r"$\Phi_s$ (mV)", fontsize=9.5)
    fig.suptitle("Solution potential in two independent local half-spaces", x=0.105, ha="left", fontsize=10.8)
    fig.subplots_adjust(left=0.105, right=0.95, bottom=0.17, top=0.82)
    saved.extend(_save(fig, output_dir, "solution_phase_potential_2d_independent_edls"))

    concentration_values = np.concatenate(
        [np.ravel(fields[m][name]) for m in fields for name in ("red1", "ox2")]
    )
    concentration_norm = LogNorm(
        vmin=10.0 ** math.floor(math.log10(float(np.min(concentration_values)))),
        vmax=10.0 ** math.ceil(math.log10(float(np.max(concentration_values)))),
    )
    fig = plt.figure(figsize=(6.9, 6.5))
    grid = fig.add_gridspec(3, 3, width_ratios=(1.0, 1.0, 0.065), hspace=0.13, wspace=0.16)
    potential_mesh = concentration_mesh = None
    for row, key in enumerate(("phi_mV", "red1", "ox2")):
        for col, material in enumerate(("Au", "Pd")):
            ax = fig.add_subplot(grid[row, col])
            field = fields[material]
            if key == "phi_mV":
                potential_mesh = ax.pcolormesh(
                    field["x_nm"],
                    y_nm,
                    field[key],
                    shading="auto",
                    cmap="RdBu_r",
                    norm=potential_norm,
                    rasterized=True,
                )
            else:
                concentration_mesh = ax.pcolormesh(
                    field["x_nm"],
                    y_nm,
                    field[key],
                    shading="auto",
                    cmap="viridis",
                    norm=concentration_norm,
                    rasterized=True,
                )
            if row == 0:
                ax.set_title(f"{material} independent local EDL", loc="left", fontsize=9.2)
            if col == 0:
                labels = (r"$\Phi_s$", r"$c_{\mathrm{Red1}}/c_{\mathrm{bulk}}$", r"$c_{\mathrm{Ox2}}/c_{\mathrm{bulk}}$")
                ax.set_ylabel("distance into electrolyte (nm)", fontsize=8.8)
                ax.text(
                    0.035,
                    0.90,
                    labels[row],
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.8,
                    color="#202020",
                )
            else:
                ax.tick_params(labelleft=False)
            if row == 2:
                ax.set_xlabel("local x (nm)", fontsize=9.0)
            else:
                ax.tick_params(labelbottom=False)
            ax.tick_params(labelsize=8.0)
    assert potential_mesh is not None and concentration_mesh is not None
    potential_cax = fig.add_subplot(grid[0, 2])
    concentration_cax = fig.add_subplot(grid[1:, 2])
    pbar = fig.colorbar(potential_mesh, cax=potential_cax)
    pbar.set_label(r"$\Phi_s$ (mV)", fontsize=8.8)
    cbar = fig.colorbar(concentration_mesh, cax=concentration_cax)
    cbar.set_label(r"$c_i/c_{\mathrm{bulk}}$ (-)", fontsize=8.8)
    fig.suptitle("Potential and reactants in independent local EDLs", x=0.105, ha="left", fontsize=10.8)
    fig.subplots_adjust(left=0.105, right=0.95, bottom=0.08, top=0.91)
    saved.extend(_save(fig, output_dir, "phi_s_reactants_2d_independent_edls"))
    return saved, rows


def build_results(params: Mapping[str, Any], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    model = IndependentPlanarEDLModel(params)
    result = solve_comparison(model.params)
    figure3_dir = output / "figures" / "Figure_3"
    rp_dir = output / "figures" / "rp_2d"
    with plt.rc_context(RC):
        figure3_paths = _figure3(model, result, figure3_dir)
        rp_paths, profile_rows = _rp_2d(model, result, rp_dir)

    csv_dir = output / "csv"
    csv_dir.mkdir()
    summary_rows = []
    for key in ("with_edl", "without_edl"):
        case = result[key]
        summary_rows.append(
            {
                "condition": case["condition"],
                "E_mix_V": case["E_mix_V"],
                "I_Au_A": case["I_Au_A"],
                "I_Pd_A": case["I_Pd_A"],
                "i_mix_avg_A_per_m2": case["i_mix_avg_A_per_m2"],
                "phi_RP_Au_V": case["phi_RP_Au_V"],
                "phi_RP_Pd_V": case["phi_RP_Pd_V"],
                "max_abs_phi_tilde": case["debye_huckel_validity"]["max_abs_phi_tilde"],
            }
        )
    with (csv_dir / "summary_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (csv_dir / "independent_local_edl_profiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profile_rows[0]))
        writer.writeheader()
        writer.writerows(profile_rows)

    _write_json(output / "params.json", model.params)
    _write_json(output / "derived.json", model.derived)
    _write_json(output / "summary.json", result)
    E_with = float(result["with_edl"]["E_mix_V"])
    beta = float(model.derived["beta_per_V"])
    robin_residuals: dict[str, float] = {}
    for material in ("Au", "Pd"):
        phi_rp = float(model.surface_phi_tilde(E_with, material))
        g_value = float(model.derived[f"g_{material}"])
        driving = beta * (E_with - float(model.params[f"pzc_{material}"]))
        # For phi = phi_RP exp(-y_tilde), -d(phi)/d(y_tilde) = phi_RP at y=0.
        robin_residuals[material] = phi_rp + g_value * phi_rp - g_value * driving
    validation = {
        "max_abs_robin_residual": max(abs(value) for value in robin_residuals.values()),
        "max_abs_closed_form_minus_brent_V": max(
            abs(float(result[key]["closed_form_minus_brent_V"]))
            for key in ("with_edl", "without_edl")
        ),
        "max_relative_current_balance_residual": max(
            float(result[key]["relative_balance_residual"])
            for key in ("with_edl", "without_edl")
        ),
        "far_field_is_analytic_zero_limit": True,
        "robin_residuals": robin_residuals,
    }
    validation["passed"] = (
        validation["max_abs_robin_residual"] < 1.0e-11
        and validation["max_abs_closed_form_minus_brent_V"] < 5.0e-11
        and validation["max_relative_current_balance_residual"] < 1.0e-10
    )
    _write_json(output / "validation.json", validation)
    pngs = sorted(output.glob("figures/**/*.png"))
    svgs = sorted(output.glob("figures/**/*.svg"))
    pdfs = sorted(output.glob("**/*.pdf"))
    if len(pngs) != 8 or len(svgs) != 8 or pdfs:
        raise RuntimeError(
            f"Expected 8 PNG, 8 SVG, 0 PDF; got {len(pngs)}, {len(svgs)}, {len(pdfs)}"
        )
    artifacts = {
        "figure3": [str(path.relative_to(output)) for path in figure3_paths],
        "rp_2d": [str(path.relative_to(output)) for path in rp_paths],
        "csv": [
            "csv/summary_cases.csv",
            "csv/independent_local_edl_profiles.csv",
        ],
        "metadata": [
            "params.json",
            "derived.json",
            "summary.json",
            "validation.json",
        ],
    }
    _write_json(output / "artifacts.json", artifacts)
    manifest = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "topology_id": TOPOLOGY_ID,
        "electrostatic_backend": ELECTROSTATIC_BACKEND,
        "electronic_constraint": "ideal wire; shared E_mix; I_Au + I_Pd = 0",
        "electrolyte_reference": (
            "separate local half-spaces with a common bulk solution-potential reference"
        ),
        "created_local": datetime.now().astimezone().isoformat(),
        "python": platform.python_version(),
        "output_path_policy": "artifact paths are relative to run root",
        "figure_counts": {"png": len(pngs), "svg": len(svgs), "pdf": len(pdfs)},
        "debye_huckel_caveat": result["with_edl"]["debye_huckel_validity"],
    }
    _write_json(output / "run_manifest.json", manifest)
    return {"output": str(output), "manifest": manifest, "result": result}


def default_output(root: Path) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return root / "results" / stamp
