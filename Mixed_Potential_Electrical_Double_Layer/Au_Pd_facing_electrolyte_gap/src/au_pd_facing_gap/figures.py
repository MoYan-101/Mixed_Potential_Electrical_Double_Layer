"""Publication 2D maps and traceable result writer for the facing slit."""

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
    FacingElectrolyteSlitModel,
    canonical_params,
    compute_derived,
    solve_cases,
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


def _field_1d(
    model: FacingElectrolyteSlitModel,
    E_V: float,
    x_nm: np.ndarray,
) -> dict[str, np.ndarray]:
    phi = model.phi_tilde(E_V, x_nm * 1.0e-9)
    return {
        "x_nm": x_nm,
        "phi_tilde": phi,
        "phi_mV": phi * model.derived["thermal_voltage_V"] * 1.0e3,
        "red1": np.exp(np.clip(-float(model.params["z_R1"]) * phi, -700.0, 700.0)),
        "ox2": np.exp(np.clip(-float(model.params["z_O2"]) * phi, -700.0, 700.0)),
    }


def _plot_windows(model: FacingElectrolyteSlitModel) -> tuple[tuple[float, float], ...]:
    if model.gap_nm < 100.0:
        return ((0.0, model.gap_nm),)
    edge_nm = min(
        float(model.params["large_gap_window_lambda_D"])
        * model.derived["lambda_D"]
        * 1.0e9,
        0.49 * model.gap_nm,
    )
    return ((0.0, edge_nm), (model.gap_nm - edge_nm, model.gap_nm))


def _window_fields(
    model: FacingElectrolyteSlitModel,
    E_V: float,
) -> list[dict[str, np.ndarray]]:
    return [
        _field_1d(model, E_V, np.linspace(start, stop, 601))
        for start, stop in _plot_windows(model)
    ]


def _tile(values: np.ndarray, face_points: int) -> np.ndarray:
    return np.repeat(np.asarray(values, dtype=float)[None, :], face_points, axis=0)


def _style_map_axis(
    ax: plt.Axes,
    *,
    show_ylabel: bool,
    show_xlabel: bool,
    ylabel: str = "along electrode face, y (nm)",
) -> None:
    ax.set_ylabel(ylabel if show_ylabel else "", fontsize=9.3)
    ax.set_xlabel("normal gap coordinate, x (nm)" if show_xlabel else "", fontsize=9.3)
    if not show_ylabel:
        ax.tick_params(left=False, labelleft=False)
    if not show_xlabel:
        ax.tick_params(labelbottom=False)
    ax.tick_params(length=3.2, width=0.85, labelsize=8.2, pad=2.2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color(COLORS["dark"])


def _electrode_edge(ax: plt.Axes, side: str, *, label: bool) -> None:
    if side == "left":
        text, align, text_x = "Au", "right", -0.045
    elif side == "right":
        text, align, text_x = "Pd", "left", 1.045
    else:
        raise ValueError(side)
    if label:
        ax.text(
            text_x,
            1.02,
            text,
            transform=ax.transAxes,
            ha=align,
            va="bottom",
            fontsize=8.5,
            color=COLORS["dark"],
            clip_on=False,
        )


def _break_marks(left: plt.Axes, right: plt.Axes) -> None:
    left.spines["right"].set_visible(False)
    right.spines["left"].set_visible(False)
    mark = 0.013
    kwargs = {
        "color": COLORS["dark"],
        "clip_on": False,
        "linewidth": 0.9,
        "solid_capstyle": "butt",
        "zorder": 20,
    }
    left.plot((1 - mark, 1 + mark), (-mark, mark), transform=left.transAxes, **kwargs)
    left.plot((1 - mark, 1 + mark), (1 - mark, 1 + mark), transform=left.transAxes, **kwargs)
    right.plot((-mark, mark), (-mark, mark), transform=right.transAxes, **kwargs)
    right.plot((-mark, mark), (1 - mark, 1 + mark), transform=right.transAxes, **kwargs)


def _global_norms(
    params: Mapping[str, Any],
    cases: list[Mapping[str, Any]],
    model_class: type[FacingElectrolyteSlitModel] = FacingElectrolyteSlitModel,
) -> tuple[TwoSlopeNorm, LogNorm]:
    potentials: list[np.ndarray] = []
    concentrations: list[np.ndarray] = []
    for case in cases:
        model = model_class(params, float(case["gap_nm"]))
        for field in _window_fields(model, float(case["E_mix_V"])):
            potentials.append(field["phi_mV"])
            concentrations.extend((field["red1"], field["ox2"]))
    max_potential = max(float(np.max(np.abs(value))) for value in potentials)
    concentration = np.concatenate(concentrations)
    return (
        TwoSlopeNorm(vmin=-max_potential, vcenter=0.0, vmax=max_potential),
        LogNorm(
            vmin=10.0 ** math.floor(math.log10(float(np.min(concentration)))),
            vmax=10.0 ** math.ceil(math.log10(float(np.max(concentration)))),
        ),
    )


def _potential_only(
    model: FacingElectrolyteSlitModel,
    case: Mapping[str, Any],
    output_dir: Path,
    potential_norm: TwoSlopeNorm,
) -> list[Path]:
    windows = _plot_windows(model)
    fields = _window_fields(model, float(case["E_mix_V"]))
    face_nm = float(model.params["L_face"]) * 1.0e9
    face_points = int(model.params["plot_face_points"])
    y_nm = np.linspace(0.0, face_nm, face_points)
    token = f"d{int(round(model.gap_nm)):04d}nm"
    if len(windows) == 1:
        fig = plt.figure(figsize=(5.15, 3.25))
        grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 0.065), left=0.15, right=0.91, bottom=0.18, top=0.80, wspace=0.17)
        ax = fig.add_subplot(grid[0, 0])
        cax = fig.add_subplot(grid[0, 1])
        field = fields[0]
        mesh = ax.pcolormesh(
            field["x_nm"],
            y_nm,
            _tile(field["phi_mV"], face_points),
            shading="auto",
            cmap="RdBu_r",
            norm=potential_norm,
            rasterized=True,
        )
        _style_map_axis(ax, show_ylabel=True, show_xlabel=True)
        _electrode_edge(ax, "left", label=True)
        _electrode_edge(ax, "right", label=True)
        colorbar = fig.colorbar(mesh, cax=cax)
        colorbar.set_label(r"$\Phi_s$ (mV)", fontsize=9.3)
    else:
        fig = plt.figure(figsize=(7.0, 3.25))
        grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.065), left=0.11, right=0.93, bottom=0.18, top=0.80, wspace=0.16)
        axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]))
        axes[1].sharey(axes[0])
        cax = fig.add_subplot(grid[0, 2])
        mesh = None
        for index, (ax, field, title) in enumerate(zip(axes, fields, ("Au-side EDL window", "Pd-side EDL window"), strict=True)):
            mesh = ax.pcolormesh(
                field["x_nm"],
                y_nm,
                _tile(field["phi_mV"], face_points),
                shading="auto",
                cmap="RdBu_r",
                norm=potential_norm,
                rasterized=True,
            )
            _style_map_axis(ax, show_ylabel=index == 0, show_xlabel=True)
            ax.set_title(title, loc="left", fontsize=8.8, pad=4)
        _electrode_edge(axes[0], "left", label=True)
        _electrode_edge(axes[1], "right", label=True)
        _break_marks(axes[0], axes[1])
        assert mesh is not None
        colorbar = fig.colorbar(mesh, cax=cax)
        colorbar.set_label(r"$\Phi_s$ (mV)", fontsize=9.3)
        omitted = windows[1][0] - windows[0][1]
        fig.text(0.52, 0.855, f"{omitted:g} nm bulk middle omitted", ha="center", va="bottom", fontsize=7.7, color=COLORS["gray"])
    fig.text(
        0.91 if len(windows) == 1 else 0.93,
        0.90 if len(windows) == 1 else 0.915,
        "Spatial axes not drawn to equal scale",
        ha="right",
        va="bottom",
        fontsize=6.8,
        color=COLORS["gray"],
    )
    fig.suptitle(
        rf"Facing Au|electrolyte|Pd: solution potential, $d={model.gap_nm:g}$ nm",
        x=0.11 if len(windows) > 1 else 0.15,
        y=0.98,
        ha="left",
        fontsize=10.6,
    )
    return _save(fig, output_dir, f"facing_solution_potential_2d_{token}")


def _composite(
    model: FacingElectrolyteSlitModel,
    case: Mapping[str, Any],
    output_dir: Path,
    potential_norm: TwoSlopeNorm,
    concentration_norm: LogNorm,
) -> list[Path]:
    windows = _plot_windows(model)
    fields = _window_fields(model, float(case["E_mix_V"]))
    face_nm = float(model.params["L_face"]) * 1.0e9
    face_points = int(model.params["plot_face_points"])
    y_nm = np.linspace(0.0, face_nm, face_points)
    token = f"d{int(round(model.gap_nm)):04d}nm"
    keys = ("phi_mV", "red1", "ox2")
    row_labels = (r"$\Phi_s$", r"$c_{\mathrm{Red1}}/c_{\mathrm{bulk}}$", r"$c_{\mathrm{Ox2}}/c_{\mathrm{bulk}}$")
    potential_mesh = concentration_mesh = None
    if len(windows) == 1:
        fig = plt.figure(figsize=(5.25, 7.0))
        grid = fig.add_gridspec(3, 2, width_ratios=(1.0, 0.065), hspace=0.13, wspace=0.17, left=0.19, right=0.90, bottom=0.08, top=0.90)
        axes: list[plt.Axes] = []
        field = fields[0]
        for row, key in enumerate(keys):
            ax = fig.add_subplot(grid[row, 0])
            axes.append(ax)
            if key == "phi_mV":
                potential_mesh = ax.pcolormesh(
                    field["x_nm"],
                    y_nm,
                    _tile(field[key], face_points),
                    shading="auto",
                    cmap="RdBu_r",
                    norm=potential_norm,
                    rasterized=True,
                )
            else:
                concentration_mesh = ax.pcolormesh(
                    field["x_nm"],
                    y_nm,
                    _tile(field[key], face_points),
                    shading="auto",
                    cmap="viridis",
                    norm=concentration_norm,
                    rasterized=True,
                )
            _style_map_axis(
                ax,
                show_ylabel=True,
                show_xlabel=row == 2,
                ylabel="along electrode face, y (nm)",
            )
            ax.text(
                0.025,
                0.90,
                row_labels[row],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.2,
                color=COLORS["dark"],
            )
            _electrode_edge(ax, "left", label=row == 0)
            _electrode_edge(ax, "right", label=row == 0)
        p_cax = fig.add_subplot(grid[0, 1])
        c_cax = fig.add_subplot(grid[1:, 1])
    else:
        fig = plt.figure(figsize=(7.0, 7.0))
        grid = fig.add_gridspec(3, 3, width_ratios=(1.0, 1.0, 0.065), hspace=0.13, wspace=0.16, left=0.12, right=0.93, bottom=0.08, top=0.90)
        axes_by_row: list[tuple[plt.Axes, plt.Axes]] = []
        for row, key in enumerate(keys):
            axes = (fig.add_subplot(grid[row, 0]), fig.add_subplot(grid[row, 1]))
            axes[1].sharey(axes[0])
            axes_by_row.append(axes)
            for col, (ax, field) in enumerate(zip(axes, fields, strict=True)):
                if key == "phi_mV":
                    potential_mesh = ax.pcolormesh(
                        field["x_nm"],
                        y_nm,
                        _tile(field[key], face_points),
                        shading="auto",
                        cmap="RdBu_r",
                        norm=potential_norm,
                        rasterized=True,
                    )
                else:
                    concentration_mesh = ax.pcolormesh(
                        field["x_nm"],
                        y_nm,
                        _tile(field[key], face_points),
                        shading="auto",
                        cmap="viridis",
                        norm=concentration_norm,
                        rasterized=True,
                    )
                _style_map_axis(
                    ax,
                    show_ylabel=col == 0,
                    show_xlabel=row == 2,
                    ylabel="along electrode face, y (nm)",
                )
                if col == 0:
                    ax.text(
                        0.025,
                        0.90,
                        row_labels[row],
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8.2,
                        color=COLORS["dark"],
                    )
            _break_marks(axes[0], axes[1])
        axes_by_row[0][0].set_title("Au-side EDL window", loc="left", fontsize=8.8, pad=4)
        axes_by_row[0][1].set_title("Pd-side EDL window", loc="left", fontsize=8.8, pad=4)
        _electrode_edge(axes_by_row[0][0], "left", label=True)
        _electrode_edge(axes_by_row[0][1], "right", label=True)
        p_cax = fig.add_subplot(grid[0, 2])
        c_cax = fig.add_subplot(grid[1:, 2])
        omitted = windows[1][0] - windows[0][1]
        fig.text(0.50, 0.935, f"{omitted:g} nm bulk middle omitted", ha="center", va="bottom", fontsize=7.7, color=COLORS["gray"])
    assert potential_mesh is not None and concentration_mesh is not None
    pbar = fig.colorbar(potential_mesh, cax=p_cax)
    pbar.set_label(r"$\Phi_s$ (mV)", fontsize=8.8)
    cbar = fig.colorbar(concentration_mesh, cax=c_cax)
    cbar.set_label(r"$c_i/c_{\mathrm{bulk}}$ (-)", fontsize=8.8)
    fig.suptitle(
        rf"Facing Au|electrolyte|Pd: potential and reactants, $d={model.gap_nm:g}$ nm",
        x=0.12 if len(windows) > 1 else 0.19,
        y=0.985 if len(windows) == 1 else 0.997,
        ha="left",
        fontsize=10.6,
    )
    fig.text(
        0.90 if len(windows) == 1 else 0.93,
        0.945 if len(windows) == 1 else 0.96,
        "Spatial axes not drawn to equal scale",
        ha="right",
        va="bottom",
        fontsize=6.8,
        color=COLORS["gray"],
    )
    return _save(fig, output_dir, f"facing_phi_s_reactants_2d_{token}")


def _style_line_axis(
    ax: plt.Axes,
    *,
    ylabel: str = "",
    show_ylabel: bool = True,
) -> None:
    ax.set_ylabel(ylabel if show_ylabel else "", fontsize=9.4)
    if not show_ylabel:
        ax.tick_params(left=False, labelleft=False)
    ax.tick_params(length=3.2, width=0.85, pad=2.2, labelsize=8.2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color(COLORS["dark"])


def _figure3_profile_panel_six(
    params: Mapping[str, Any],
    cases: list[Mapping[str, Any]],
    output_dir: Path,
    *,
    quantity: str,
    model_class: type[FacingElectrolyteSlitModel],
    solution_label: str,
) -> list[Path]:
    """Two-row profile layout for five finite gaps plus one broken large gap."""

    ordered = sorted(cases, key=lambda case: float(case["gap_nm"]))
    if len(ordered) != 6 or float(ordered[-1]["gap_nm"]) < 100.0:
        raise ValueError(
            "Six-case Figure 3 profiles require five finite gaps and one large gap"
        )
    models = [model_class(params, float(case["gap_nm"])) for case in ordered]
    fields = [
        _window_fields(model, float(case["E_mix_V"]))
        for model, case in zip(models, ordered, strict=True)
    ]
    if any(len(case_fields) != 1 for case_fields in fields[:-1]) or len(fields[-1]) != 2:
        raise ValueError("Unexpected profile-window layout for six-gap Figure 3")

    if quantity == "potential":
        ylabel = r"$\Phi_s$ (mV)"
        title = "Solution potential across facing electrolyte slits"
        stem = "figure_3_panel_b_solution_potential_profiles_facing_gap"
        all_values = np.concatenate(
            [field["phi_mV"] for case_fields in fields for field in case_fields]
        )
        low = float(np.min(all_values))
        high = max(0.0, float(np.max(all_values)))
        padding = 0.08 * max(high - low, 1.0)
        y_limits = (low - padding, high + padding)

        def draw(ax: plt.Axes, field: Mapping[str, np.ndarray]) -> None:
            ax.plot(
                field["x_nm"],
                field["phi_mV"],
                color=COLORS["with_edl"],
                lw=2.0,
            )
            ax.axhline(
                0.0, color=COLORS["gray"], lw=1.0, ls=(0, (3, 2))
            )

        handles = [
            Line2D(
                [0], [0], color=COLORS["with_edl"], lw=2.0, label=r"$\Phi_s(x)$"
            ),
            Line2D(
                [0],
                [0],
                color=COLORS["gray"],
                lw=1.0,
                ls=(0, (3, 2)),
                label="bulk reference",
            ),
        ]
    elif quantity == "concentration":
        ylabel = r"$c_i/c_{\mathrm{bulk}}$ (-)"
        title = "Reactant concentration across facing electrolyte slits"
        stem = "figure_3_panel_c_reactant_concentration_profiles_facing_gap"
        all_values = np.concatenate(
            [
                field[key]
                for case_fields in fields
                for field in case_fields
                for key in ("red1", "ox2")
            ]
        )
        y_limits = (
            10.0 ** math.floor(math.log10(float(np.min(all_values)))),
            10.0 ** math.ceil(math.log10(float(np.max(all_values)))),
        )

        def draw(ax: plt.Axes, field: Mapping[str, np.ndarray]) -> None:
            ax.plot(
                field["x_nm"],
                field["red1"],
                color=COLORS["red1_i1"],
                lw=2.0,
            )
            ax.plot(
                field["x_nm"],
                field["ox2"],
                color=COLORS["ox2_i2"],
                lw=2.0,
            )
            ax.axhline(
                1.0, color=COLORS["gray"], lw=1.0, ls=(0, (3, 2))
            )

        handles = [
            Line2D(
                [0],
                [0],
                color=COLORS["red1_i1"],
                lw=2.0,
                label=r"$c_{\mathrm{Red1}}/c_{\mathrm{bulk}}$",
            ),
            Line2D(
                [0],
                [0],
                color=COLORS["ox2_i2"],
                lw=2.0,
                label=r"$c_{\mathrm{Ox2}}/c_{\mathrm{bulk}}$",
            ),
            Line2D(
                [0],
                [0],
                color=COLORS["gray"],
                lw=1.0,
                ls=(0, (3, 2)),
                label="bulk = 1",
            ),
        ]
    else:
        raise ValueError(quantity)

    fig = plt.figure(figsize=(9.25, 5.15))
    grid = fig.add_gridspec(
        2,
        4,
        left=0.075,
        right=0.985,
        bottom=0.13,
        top=0.86,
        hspace=0.44,
        wspace=0.18,
    )
    axes: list[plt.Axes] = []
    for row, col in ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)):
        axes.append(
            fig.add_subplot(
                grid[row, col],
                sharey=axes[0] if axes else None,
            )
        )
    legend_ax = fig.add_subplot(grid[1, 3])
    legend_ax.set_axis_off()

    for index, (ax, model, case_fields) in enumerate(
        zip(axes[:5], models[:5], fields[:5], strict=True)
    ):
        draw(ax, case_fields[0])
        show_ylabel = index in (0, 4)
        _style_line_axis(ax, ylabel=ylabel, show_ylabel=show_ylabel)
        ax.set_xlim(0.0, model.gap_nm)
        ax.set_title(rf"$d={model.gap_nm:g}$ nm", loc="center", fontsize=8.7, pad=4)
        ax.text(
            0.0,
            1.02,
            "Au",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.7,
        )
        ax.text(
            1.0,
            1.02,
            "Pd",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.7,
        )
        if quantity == "concentration":
            ax.set_yscale("log")

    for index, (ax, field) in enumerate(zip(axes[5:7], fields[-1], strict=True)):
        draw(ax, field)
        _style_line_axis(ax, ylabel=ylabel, show_ylabel=False)
        ax.set_xlim(float(field["x_nm"][0]), float(field["x_nm"][-1]))
        if quantity == "concentration":
            ax.set_yscale("log")
        if index == 0:
            ax.text(
                0.0,
                1.02,
                "Au",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.7,
            )
        else:
            ax.text(
                1.0,
                1.02,
                "Pd",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.7,
            )
    _break_marks(axes[5], axes[6])
    axes[0].set_ylim(*y_limits)
    large_left = axes[5].get_position()
    large_right = axes[6].get_position()
    fig.text(
        0.5 * (large_left.x0 + large_right.x1),
        large_left.y1 + 0.035,
        rf"$d={models[-1].gap_nm:g}$ nm (bulk middle omitted)",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color=COLORS["gray"],
    )
    legend_ax.text(0.0, 0.96, solution_label, ha="left", va="top", fontsize=8.2)
    legend_ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(-0.02, 0.47),
        fontsize=7.5,
        handlelength=2.0,
    )
    fig.supxlabel("normal gap coordinate, x (nm)", x=0.47, y=0.035, fontsize=9.5)
    fig.suptitle(title, x=0.075, y=0.975, ha="left", fontsize=10.8)
    return _save(fig, output_dir, stem)


def _figure3_profile_panel(
    params: Mapping[str, Any],
    cases: list[Mapping[str, Any]],
    output_dir: Path,
    *,
    quantity: str,
    model_class: type[FacingElectrolyteSlitModel] = FacingElectrolyteSlitModel,
    solution_label: str = "Exact 1D slit solution",
) -> list[Path]:
    if len(cases) == 6:
        return _figure3_profile_panel_six(
            params,
            cases,
            output_dir,
            quantity=quantity,
            model_class=model_class,
            solution_label=solution_label,
        )
    if len(cases) != 3 or float(cases[-1]["gap_nm"]) < 100.0:
        raise ValueError("Figure 3 profiles require two finite-gap cases and one large-gap case")
    models = [model_class(params, float(case["gap_nm"])) for case in cases]
    fields = [
        _window_fields(model, float(case["E_mix_V"]))
        for model, case in zip(models, cases, strict=True)
    ]
    fig = plt.figure(figsize=(9.2, 3.05))
    grid = fig.add_gridspec(
        1,
        5,
        width_ratios=(1.0, 1.0, 1.0, 1.0, 0.80),
        left=0.075,
        right=0.985,
        bottom=0.22,
        top=0.78,
        wspace=0.16,
    )
    axes = [fig.add_subplot(grid[0, 0])]
    axes.extend(fig.add_subplot(grid[0, index], sharey=axes[0]) for index in (1, 2, 3))
    legend_ax = fig.add_subplot(grid[0, 4])
    legend_ax.set_axis_off()

    if quantity == "potential":
        ylabel = r"$\Phi_s$ (mV)"
        title = "Solution potential across facing electrolyte slits"
        stem = "figure_3_panel_b_solution_potential_profiles_facing_gap"
        all_values = np.concatenate(
            [field["phi_mV"] for case_fields in fields for field in case_fields]
        )
        low = float(np.min(all_values))
        high = max(0.0, float(np.max(all_values)))
        padding = 0.08 * max(high - low, 1.0)
        y_limits = (low - padding, high + padding)

        def draw(ax: plt.Axes, field: Mapping[str, np.ndarray]) -> None:
            ax.plot(field["x_nm"], field["phi_mV"], color=COLORS["with_edl"], lw=2.0)
            ax.axhline(0.0, color=COLORS["gray"], lw=1.0, ls=(0, (3, 2)))

        handles = [
            Line2D([0], [0], color=COLORS["with_edl"], lw=2.0, label=r"$\Phi_s(x)$"),
            Line2D([0], [0], color=COLORS["gray"], lw=1.0, ls=(0, (3, 2)), label="bulk reference"),
        ]
    elif quantity == "concentration":
        ylabel = r"$c_i/c_{\mathrm{bulk}}$ (-)"
        title = "Reactant concentration across facing electrolyte slits"
        stem = "figure_3_panel_c_reactant_concentration_profiles_facing_gap"
        all_values = np.concatenate(
            [
                field[key]
                for case_fields in fields
                for field in case_fields
                for key in ("red1", "ox2")
            ]
        )
        y_limits = (
            10.0 ** math.floor(math.log10(float(np.min(all_values)))),
            10.0 ** math.ceil(math.log10(float(np.max(all_values)))),
        )

        def draw(ax: plt.Axes, field: Mapping[str, np.ndarray]) -> None:
            ax.plot(field["x_nm"], field["red1"], color=COLORS["red1_i1"], lw=2.0)
            ax.plot(field["x_nm"], field["ox2"], color=COLORS["ox2_i2"], lw=2.0)
            ax.axhline(1.0, color=COLORS["gray"], lw=1.0, ls=(0, (3, 2)))

        handles = [
            Line2D([0], [0], color=COLORS["red1_i1"], lw=2.0, label=r"$c_{\mathrm{Red1}}/c_{\mathrm{bulk}}$"),
            Line2D([0], [0], color=COLORS["ox2_i2"], lw=2.0, label=r"$c_{\mathrm{Ox2}}/c_{\mathrm{bulk}}$"),
            Line2D([0], [0], color=COLORS["gray"], lw=1.0, ls=(0, (3, 2)), label="bulk = 1"),
        ]
    else:
        raise ValueError(quantity)

    for index in (0, 1):
        ax = axes[index]
        draw(ax, fields[index][0])
        _style_line_axis(ax, ylabel=ylabel, show_ylabel=index == 0)
        ax.set_xlim(0.0, models[index].gap_nm)
        ax.set_title(rf"$d={models[index].gap_nm:g}$ nm", loc="center", fontsize=8.6, pad=4)
        ax.text(0.0, 1.02, "Au", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.8)
        ax.text(1.0, 1.02, "Pd", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.8)
        if quantity == "concentration":
            ax.set_yscale("log")

    for ax, field in zip(axes[2:4], fields[2], strict=True):
        draw(ax, field)
        _style_line_axis(ax, ylabel=ylabel, show_ylabel=False)
        ax.set_xlim(float(field["x_nm"][0]), float(field["x_nm"][-1]))
        if quantity == "concentration":
            ax.set_yscale("log")
    axes[2].text(0.0, 1.02, "Au", transform=axes[2].transAxes, ha="left", va="bottom", fontsize=7.8)
    axes[3].text(1.0, 1.02, "Pd", transform=axes[3].transAxes, ha="right", va="bottom", fontsize=7.8)
    _break_marks(axes[2], axes[3])
    axes[0].set_ylim(*y_limits)
    large_left = axes[2].get_position()
    large_right = axes[3].get_position()
    fig.text(
        0.5 * (large_left.x0 + large_right.x1),
        0.815,
        rf"$d={models[2].gap_nm:g}$ nm (bulk middle omitted)",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color=COLORS["gray"],
    )
    legend_ax.text(0.0, 0.96, solution_label, ha="left", va="top", fontsize=8.2)
    legend_ax.legend(handles=handles, loc="center left", bbox_to_anchor=(-0.02, 0.48), fontsize=7.5, handlelength=2.0)
    fig.supxlabel("normal gap coordinate, x (nm)", x=0.46, y=0.055, fontsize=9.5)
    fig.suptitle(title, x=0.075, y=0.97, ha="left", fontsize=10.8)
    return _save(fig, output_dir, stem)


def _figure3(
    params: Mapping[str, Any],
    cases: list[Mapping[str, Any]],
    output_dir: Path,
    *,
    model_class: type[FacingElectrolyteSlitModel] = FacingElectrolyteSlitModel,
    solution_label: str = "Exact 1D slit solution",
) -> tuple[list[Path], list[dict[str, float]]]:
    ordered = sorted(cases, key=lambda case: float(case["gap_nm"]))
    gaps = np.asarray([float(case["gap_nm"]) for case in ordered])
    emix = np.asarray([float(case["E_mix_V"]) for case in ordered])
    imix = np.asarray([float(case["i_mix_avg_A_per_m2"]) for case in ordered])
    thermal = compute_derived(params)["thermal_voltage_V"]
    reaction = model_class(params, float(gaps[0])).reaction
    rows: list[dict[str, float]] = []
    for case in ordered:
        phi_au = float(case["phi_RP_Au_tilde"])
        phi_pd = float(case["phi_RP_Pd_tilde"])
        phi_au_v = phi_au * thermal
        phi_pd_v = phi_pd * thermal
        rows.append(
            {
                "gap_nm": float(case["gap_nm"]),
                "E_mix_V": float(case["E_mix_V"]),
                "i_mix_avg_A_per_m2": float(case["i_mix_avg_A_per_m2"]),
                "phi_RP_Au_V": phi_au_v,
                "phi_RP_Pd_V": phi_pd_v,
                "c_Red1_at_Au_over_bulk": math.exp(np.clip(-float(params["z_R1"]) * phi_au, -700.0, 700.0)),
                "c_Ox2_at_Pd_over_bulk": math.exp(np.clip(-float(params["z_O2"]) * phi_pd, -700.0, 700.0)),
                "eta_Au_V": float(case["E_mix_V"]) - reaction["E1_eq_eff"] - phi_au_v,
                "eta_Pd_V": float(case["E_mix_V"]) - reaction["E2_eq_eff"] - phi_pd_v,
                "j_Au_A_per_m2": float(case["j_Au_A_per_m2"]),
                "j_Pd_A_per_m2": float(case["j_Pd_A_per_m2"]),
            }
        )

    saved: list[Path] = []
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.9))
    for ax, values, title, ylabel in zip(
        axes,
        (emix, imix),
        (r"$E_{\mathrm{mix}}$", r"$\bar{i}_{\mathrm{mix}}$"),
        (r"$E_{\mathrm{mix}}$ (V)", r"$\bar{i}_{\mathrm{mix}}$ (A/m$^2$)"),
        strict=True,
    ):
        ax.semilogx(gaps, values, color=COLORS["with_edl"], marker="o", ms=5.5, lw=1.8)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_xlabel(r"Gap, $d$ (nm)", fontsize=9.2)
        ax.set_ylabel(ylabel, fontsize=9.2)
        ax.set_xticks(gaps, [f"{gap:g}" for gap in gaps])
        ax.tick_params(labelsize=8.2, length=3.2)
        low, high = float(np.min(values)), float(np.max(values))
        pad = 0.14 * max(high - low, 1.0e-4)
        ax.set_ylim(low - pad, high + pad)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.22, top=0.86, wspace=0.48)
    saved.extend(_save(fig, output_dir, "figure_3_panel_a_emix_imix_vs_gap_facing_gap"))

    saved.extend(
        _figure3_profile_panel(
            params,
            ordered,
            output_dir,
            quantity="potential",
            model_class=model_class,
            solution_label=solution_label,
        )
    )
    saved.extend(
        _figure3_profile_panel(
            params,
            ordered,
            output_dir,
            quantity="concentration",
            model_class=model_class,
            solution_label=solution_label,
        )
    )

    fig, ax = plt.subplots(figsize=(4.4, 3.15))
    ax.semilogx(gaps, [row["eta_Au_V"] for row in rows], color=COLORS["au"], marker="o", lw=1.9, label="Au")
    ax.semilogx(gaps, [row["eta_Pd_V"] for row in rows], color=COLORS["pd"], marker="s", lw=1.9, label="Pd")
    ax.axhline(0.0, color=COLORS["gray"], lw=1.0, ls=(0, (3, 2)))
    ax.set_xticks(gaps, [f"{gap:g}" for gap in gaps])
    ax.set_xlabel(r"Gap, $d$ (nm)", fontsize=9.6)
    ax.set_ylabel(r"Interfacial overpotential, $\eta_M$ (V)", fontsize=9.6)
    ax.set_title("Local overpotential at facing reaction planes", loc="left", fontsize=10.6)
    ax.legend(loc="best", fontsize=8.2)
    ax.tick_params(labelsize=8.4, length=3.2)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.87)
    saved.extend(_save(fig, output_dir, "figure_3_panel_d_interfacial_overpotential_vs_gap"))

    fig, ax = plt.subplots(figsize=(4.4, 3.15))
    ax.semilogx(gaps, [row["j_Au_A_per_m2"] for row in rows], color=COLORS["au"], marker="o", lw=1.9, label="Au oxidation")
    ax.semilogx(gaps, [row["j_Pd_A_per_m2"] for row in rows], color=COLORS["pd"], marker="s", lw=1.9, label="Pd reduction")
    ax.axhline(0.0, color=COLORS["gray"], lw=1.0)
    ax.set_xticks(gaps, [f"{gap:g}" for gap in gaps])
    ax.set_xlabel(r"Gap, $d$ (nm)", fontsize=9.6)
    ax.set_ylabel(r"Local current density (A/m$^2$)", fontsize=9.6)
    ax.set_title("Signed local current density at facing interfaces", loc="left", fontsize=10.6)
    ax.legend(loc="best", fontsize=8.0)
    ax.tick_params(labelsize=8.4, length=3.2)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.87)
    saved.extend(_save(fig, output_dir, "figure_3_panel_e_interfacial_current_density_vs_gap"))

    dense_reference_map = len(gaps) > 3
    fig, ax = plt.subplots(figsize=(5.1, 3.75 if dense_reference_map else 3.35))
    lane = {
        "eq": 0.70 if dense_reference_map else 0.52,
        "mix": 0.0,
        "pzc": -0.70 if dense_reference_map else -0.52,
    }
    markers = [
        (reaction["E1_eq_eff"], lane["eq"], r"$E_{1,\mathrm{eq}}$", COLORS["gray"], "o"),
        (reaction["E2_eq_eff"], lane["eq"], r"$E_{2,\mathrm{eq}}$", COLORS["gray"], "o"),
        (float(params["pzc_Pd"]), lane["pzc"], "PZC Pd", "#5A90C8", "^"),
        (float(params["pzc_Au"]), lane["pzc"], "PZC Au", "#E4C133", "^"),
    ]
    for value, y, label, color, marker in markers:
        ax.vlines(value, y - 0.07, y + 0.07, color=color, lw=1.6)
        ax.scatter([value], [y], color=color, marker=marker, s=55, zorder=3)
        offset = 0.11 if y > 0.0 else -0.11
        ax.text(value, y + offset, f"{label}\n{value:.2f} V", ha="center", va="bottom" if offset > 0 else "top", fontsize=7.4, color=color)
    if len(gaps) == 3:
        mix_colors = ("#D83A2E", "#F26B38", COLORS["without_edl"])
        mix_y = (0.15, 0.0, -0.15)
    else:
        palette = (
            "#8F1D21",
            "#B52B2F",
            "#D83A2E",
            "#F26B38",
            "#D88A22",
            COLORS["without_edl"],
        )
        if len(gaps) > len(palette):
            raise ValueError("Potential reference map supports at most six gaps")
        mix_colors = palette[: len(gaps)]
        mix_y = tuple(np.linspace(0.34, -0.34, len(gaps)))
    for gap, value, y, color in zip(gaps, emix, mix_y, mix_colors, strict=True):
        ax.scatter([value], [y], color=color, marker="D", s=48, zorder=4)
        ax.text(value - 0.018, y, rf"{gap:g} nm, {value:.3f} V", ha="right", va="center", fontsize=7.2, color=color)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim((-1.22, 1.02) if dense_reference_map else (-1.04, 0.84))
    ax.set_xlabel("Potential (V vs. RHE)", fontsize=9.8)
    ax.set_yticks([])
    ax.set_title("Potential reference map for facing gaps", loc="left", fontsize=10.6)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_position(
        ("data", -1.03 if dense_reference_map else -0.86)
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.16, top=0.89)
    saved.extend(_save(fig, output_dir, "figure_3_panel_f_pzc_potential_reference_map_facing_gap"))
    return saved, rows


def build_results(
    params: Mapping[str, Any],
    output: Path,
    *,
    model_class: type[FacingElectrolyteSlitModel] = FacingElectrolyteSlitModel,
    solve_function: Callable[[Mapping[str, Any]], dict[str, Any]] = solve_cases,
    solution_label: str = "Exact 1D slit solution",
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    canonical = canonical_params(params)
    result = solve_function(canonical)
    cases = list(result["cases"])
    if not cases:
        raise RuntimeError("No gap cases were produced")
    nonlinear = all("nonlinear_pb_diagnostics" in case for case in cases)
    output.mkdir(parents=True, exist_ok=False)
    potential_norm, concentration_norm = _global_norms(
        canonical, cases, model_class=model_class
    )
    figure_dir = output / "figures" / "rp_2d"
    figure3_dir = output / "figures" / "Figure_3"
    figure_paths: list[Path] = []
    figure3_paths: list[Path] = []
    endpoint_rows: list[dict[str, float]] = []
    profile_files: list[str] = []
    csv_dir = output / "csv"
    csv_dir.mkdir()
    with plt.rc_context(RC):
        figure3_paths, endpoint_rows = _figure3(
            canonical,
            cases,
            figure3_dir,
            model_class=model_class,
            solution_label=solution_label,
        )
        for case in cases:
            model = model_class(canonical, float(case["gap_nm"]))
            figure_paths.extend(_potential_only(model, case, figure_dir, potential_norm))
            figure_paths.extend(_composite(model, case, figure_dir, potential_norm, concentration_norm))
            x_m = model.profile_grid_m()
            phi = model.phi_tilde(float(case["E_mix_V"]), x_m)
            rows = [
                {
                    "gap_nm": model.gap_nm,
                    "x_nm": float(x * 1e9),
                    "phi_tilde": float(value),
                    "phi_s_mV": float(value * model.derived["thermal_voltage_V"] * 1e3),
                    "c_Red1_over_c_bulk": float(math.exp(np.clip(-float(canonical["z_R1"]) * value, -700.0, 700.0))),
                    "c_Ox2_over_c_bulk": float(math.exp(np.clip(-float(canonical["z_O2"]) * value, -700.0, 700.0))),
                }
                for x, value in zip(x_m, phi, strict=True)
            ]
            filename = f"gap_profile_d{int(round(model.gap_nm)):04d}nm.csv"
            with (csv_dir / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            profile_files.append(f"csv/{filename}")

    summary_rows: list[dict[str, Any]] = []
    for case in cases:
        diagnostic = case.get("linearization_diagnostic") or case.get(
            "debye_huckel_validity", {}
        )
        row: dict[str, Any] = {
            "gap_nm": case["gap_nm"],
            "gap_over_lambda_D": case["gap_over_lambda_D"],
            "overlap_factor_exp_minus_d_over_lambda": case["overlap_factor_exp_minus_d_over_lambda"],
            "E_mix_V": case["E_mix_V"],
            "I_Au_A": case["I_Au_A"],
            "I_Pd_A": case["I_Pd_A"],
            "i_mix_avg_A_per_m2": case["i_mix_avg_A_per_m2"],
            "phi_RP_Au_V": case["phi_RP_Au_V"],
            "phi_RP_Pd_V": case["phi_RP_Pd_V"],
            "max_abs_phi_tilde": diagnostic["max_abs_phi_tilde"],
        }
        if "sigma_Au_C_per_m2" in case:
            row["sigma_Au_C_per_m2"] = case["sigma_Au_C_per_m2"]
            row["sigma_Pd_C_per_m2"] = case["sigma_Pd_C_per_m2"]
        if "nonlinear_pb_diagnostics" in case:
            row["nonlinear_pb_solver_method"] = case["nonlinear_pb_diagnostics"][
                "solver_method"
            ]
            row["nonlinear_pb_max_rms_residual"] = case[
                "nonlinear_pb_diagnostics"
            ]["max_rms_residual"]
        summary_rows.append(row)
    with (csv_dir / "summary_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (csv_dir / "figure3_endpoint_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(endpoint_rows[0]))
        writer.writeheader()
        writer.writerows(endpoint_rows)
    extra_csv: list[str] = []
    extra_metadata: list[str] = []
    if nonlinear:
        convergence_rows = list(result["numerical_convergence"]["rows"])
        convergence_name = "nonlinear_pb_convergence_check.csv"
        with (csv_dir / convergence_name).open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(convergence_rows[0])
            )
            writer.writeheader()
            writer.writerows(convergence_rows)
        extra_csv.append(f"csv/{convergence_name}")
        _write_json(
            output / "nonlinear_pb_convergence.json",
            result["numerical_convergence"],
        )
        extra_metadata.append("nonlinear_pb_convergence.json")

    _write_json(output / "params.json", canonical)
    _write_json(output / "derived.json", compute_derived(canonical))
    _write_json(output / "summary.json", result)
    validation: dict[str, Any] = {
        "electrostatic_backend": result["electrostatic_backend"],
        "all_gaps_strictly_positive": all(float(case["gap_nm"]) > 0.0 for case in cases),
        "max_abs_robin_residual": max(
            abs(value)
            for case in cases
            for value in case["robin_residuals"].values()
        ),
        "max_relative_current_balance_residual": max(
            float(case["relative_balance_residual"]) for case in cases
        ),
        "large_gap_reference": {
            "gap_nm": cases[-1]["gap_nm"],
            "overlap_factor": cases[-1]["overlap_factor_exp_minus_d_over_lambda"],
            "interpretation": "independent-planar-EDL limit",
        },
    }
    if nonlinear:
        comparison = dict(result["large_gap_comparison"])
        validation.update(
            {
                "all_nonlinear_pb_solves_successful": all(
                    int(case["nonlinear_pb_diagnostics"].get("status", 0)) == 0
                    for case in cases
                ),
                "max_nonlinear_pb_rms_residual": max(
                    float(case["nonlinear_pb_diagnostics"]["max_rms_residual"])
                    for case in cases
                ),
                "max_first_integral_span": max(
                    float(case["nonlinear_pb_diagnostics"]["first_integral_span"])
                    for case in cases
                ),
                "max_abs_integrated_charge_residual": max(
                    abs(
                        float(
                            case["nonlinear_pb_diagnostics"][
                                "integrated_charge_residual"
                            ]
                        )
                    )
                    for case in cases
                ),
                "large_gap_gouy_chapman_stern_comparison": comparison,
                "numerical_convergence": {
                    key: result["numerical_convergence"][key]
                    for key in (
                        "max_abs_delta_E_mix_V",
                        "max_abs_delta_phi_RP_tilde",
                        "max_abs_relative_i_mix_difference",
                    )
                },
            }
        )
        validation["passed"] = (
            validation["all_gaps_strictly_positive"]
            and validation["all_nonlinear_pb_solves_successful"]
            and validation["max_abs_robin_residual"] < 1.0e-8
            and validation["max_relative_current_balance_residual"] < 1.0e-8
            and validation["max_nonlinear_pb_rms_residual"] < 5.0e-7
            and validation["max_abs_integrated_charge_residual"] < 1.0e-5
            and abs(float(comparison["delta_E_mix_V"])) < 1.0e-9
            and abs(float(comparison["delta_phi_RP_Au_tilde"])) < 1.0e-8
            and abs(float(comparison["delta_phi_RP_Pd_tilde"])) < 1.0e-8
            and float(
                result["numerical_convergence"]["max_abs_delta_E_mix_V"]
            )
            < 1.0e-8
            and float(
                result["numerical_convergence"]["max_abs_delta_phi_RP_tilde"]
            )
            < 1.0e-6
            and float(
                result["numerical_convergence"][
                    "max_abs_relative_i_mix_difference"
                ]
            )
            < 1.0e-6
        )
    else:
        validation["max_abs_closed_form_minus_brent_V"] = max(
            abs(float(case["closed_form_minus_brent_V"])) for case in cases
        )
        validation["passed"] = (
            validation["all_gaps_strictly_positive"]
            and validation["max_abs_robin_residual"] < 1e-11
            and validation["max_abs_closed_form_minus_brent_V"] < 5e-11
            and validation["max_relative_current_balance_residual"] < 1e-10
        )
    _write_json(output / "validation.json", validation)
    _write_json(
        output / "reference_scope.json",
        {
            "reference": "Huang, Chen, Eikerling, PNAS 2023, e2307307120",
            "doi": "10.1073/pnas.2307307120",
            "used_for": "conceptual simultaneous two-electrode charging/current coupling",
            "not_used_as": "a spatial nanometre-gap PB-overlap solver",
            "nonlinear_relation_scope": (
                "The Gouy-Chapman asinh relation is used only for the isolated-planar "
                "large-gap reference; finite overlapping gaps solve full nonlinear PB."
            ),
        },
    )

    pngs = sorted(output.glob("figures/**/*.png"))
    svgs = sorted(output.glob("figures/**/*.svg"))
    pdfs = sorted(output.glob("**/*.pdf"))
    expected_per_format = 2 * len(cases) + 6
    if len(pngs) != expected_per_format or len(svgs) != expected_per_format or pdfs:
        raise RuntimeError(
            f"Expected {expected_per_format} PNG, {expected_per_format} SVG, 0 PDF; "
            f"got {len(pngs)}, {len(svgs)}, {len(pdfs)}"
        )
    artifacts = {
        "rp_2d": [str(path.relative_to(output)) for path in figure_paths],
        "figure3": [str(path.relative_to(output)) for path in figure3_paths],
        "csv": [
            "csv/summary_cases.csv",
            "csv/figure3_endpoint_metrics.csv",
            *profile_files,
            *extra_csv,
        ],
        "metadata": [
            "params.json",
            "derived.json",
            "summary.json",
            "validation.json",
            "reference_scope.json",
            *extra_metadata,
        ],
    }
    _write_json(output / "artifacts.json", artifacts)
    manifest = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "model_id": result["model_id"],
        "topology_id": result["topology_id"],
        "electrostatic_backend": result["electrostatic_backend"],
        "electronic_constraint": "ideal wire; shared E_mix; I_Au + I_Pd = 0",
        "electrolyte_domain": "finite positive-width slit; grand-canonical bulk concentration reference",
        "coordinate_and_normals": "Au at x=0 with outward normal -x; Pd at x=d with outward normal +x",
        "created_local": datetime.now().astimezone().isoformat(),
        "python": platform.python_version(),
        "output_path_policy": "artifact paths are relative to run root",
        "figure_counts": {"png": len(pngs), "svg": len(svgs), "pdf": len(pdfs)},
        "figure3_panel_count": 6,
        "distances_nm": [float(case["gap_nm"]) for case in cases],
    }
    if nonlinear:
        manifest.update(
            {
                "governing_equation": "d2(psi)/d(x/lambda_D)2 = sinh(psi)",
                "large_gap_method": (
                    "independent nonlinear Gouy-Chapman-asinh/Stern asymptote when "
                    "d/lambda_D exceeds the configured switch"
                ),
                "nonlinear_pb_caveat": (
                    "Mean-field point-ion continuum model with a grand-canonical bulk "
                    "chemical-potential reference; finite ion size, dielectric saturation, "
                    "ion correlations, ion-number conservation/Donnan shifts, transport, "
                    "and solution resistance are not included."
                ),
            }
        )
    else:
        manifest["debye_huckel_caveat"] = (
            "All retained cases exceed the configured small-potential threshold; "
            "interpret as linear-model geometry comparisons."
        )
    _write_json(output / "run_manifest.json", manifest)
    return {"output": str(output), "manifest": manifest, "result": result}


def default_output(root: Path) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return root / "results" / stamp
