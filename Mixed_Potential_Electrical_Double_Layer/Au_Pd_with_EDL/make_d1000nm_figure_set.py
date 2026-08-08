#!/usr/bin/env python3
"""Build the traceable d_Au-Pd=1000 nm high-resolution figure bundle.

The source physical parameters are read from the immutable 20260802_144648
result.  Only the Au--Pd separation and numerical resolution are overridden.
The full-build output directory must not already exist, so the parent result
is never rewritten accidentally.  ``--refresh-figure3-only`` is the explicit
in-place path for redrawing the saved bundle without rerunning the solver.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import Rectangle

from au_pd_edl.figure3 import (
    Figure3SavedData,
    generate_figure3_comparison_panels,
    load_figure3_saved_data,
)
from au_pd_edl.io import environment_metadata, make_run_tag
from au_pd_edl.kinetics import au_local_current_density, pd_local_current_density
from au_pd_edl.parameters import (
    ELECTROLYTE_DOMAIN_DESCRIPTION,
    ELECTROLYTE_DOMAIN_ID,
    ELECTROSTATIC_BACKEND,
    GEOMETRY_NAME,
    MODEL_NAME,
    RESULT_SCHEMA_VERSION,
    apply_param_overrides,
)
from au_pd_edl.plotting import COLORS, PUBLICATION_RCPARAMS
from au_pd_edl.solver import run_edl_comparison_pair


PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_RESULT = PACKAGE_ROOT / "results" / "20260802_144648"
DEFAULT_OUTPUT = SOURCE_RESULT / "figures" / "d1000nm_high_resolution"
SEPARATION_NM = 1000.0
SEPARATION_M = SEPARATION_NM * 1.0e-9
ACTIVE_WINDOW_NM = 30.0
FIELD_NX_PER_WINDOW = 601
FIELD_NY = 241
FIELD_Y_MAX_OVER_LAMBDA = 5.0
DPI = 600
FIGURE3_PZC_MARKER_Y = -0.35
FIGURE3_PZC_TEXT_Y = -0.50
FIGURE3_CUT_FIGSIZE = (3.35, 3.25)
RP_POTENTIAL_FIGSIZE = (3.35, 3.55)
RP_COMPOSITE_FIGSIZE = (3.35, 7.45)
RP_RED1_LIMITS = (1.0e-3, 1.0e1)
RP_OX2_LIMITS = (1.0e-1, 1.0e3)
RP_MODEL_AU_COLOR = "#EFC62E"
RP_MODEL_C_COLOR = "#8C8C8C"
RP_MODEL_PD_COLOR = "#5B8FCA"

# A staged refinement path makes the production override auditable.  The last
# level is four times the 20260802_144648 N_modes and approximately four times
# its Nx, while retaining about ten surface points per shortest cosine wave.
REFINEMENT_LEVELS = (
    ("baseline_long_domain", 960, 5001),
    ("intermediate", 1920, 10001),
    ("production", 3840, 20001),
)


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
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


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


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in fieldnames})


def _case_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    validity = result["debye_huckel_validity"]
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "electrostatic_backend": ELECTROSTATIC_BACKEND,
        "d_Au_Pd_nm": float(result["d_Au_Pd_nm"]),
        "condition": str(result["condition_label"]),
        "E_mix_V": float(result["E_mix_V"]),
        "I_Au_A": float(result["I_Au_A"]),
        "I_Pd_A": float(result["I_Pd_A"]),
        "i_mix_abs_A": float(result["i_mix_abs_A"]),
        "i_mix_avg_A_per_m2": float(result["i_mix_avg_A_per_m2"]),
        "relative_balance_residual": float(result["relative_balance_residual"]),
        "max_abs_phi_tilde": float(validity["max_abs_phi_tilde"]),
        "dh_threshold_exceeded": bool(validity["threshold_exceeded"]),
    }


def _active_x_nm(total_nm: float) -> np.ndarray:
    left = np.linspace(
        0.0,
        min(ACTIVE_WINDOW_NM, total_nm),
        FIELD_NX_PER_WINDOW,
    )
    right = np.linspace(
        max(0.0, total_nm - ACTIVE_WINDOW_NM),
        total_nm,
        FIELD_NX_PER_WINDOW,
    )
    return np.concatenate((left, right))


def _solve_refinement(
    source_params: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], Any, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_profile_mV: np.ndarray | None = None
    previous_E: float | None = None
    previous_i: float | None = None
    production_pair: dict[str, Any] | None = None
    production_model: Any | None = None
    production_params: dict[str, Any] | None = None

    for label, n_modes, n_x in REFINEMENT_LEVELS:
        params = apply_param_overrides(
            source_params,
            {
                "d_Au_Pd": SEPARATION_M,
                "N_modes": int(n_modes),
                "Nx": int(n_x),
                "dh_violation_action": "ignore",
            },
        )
        started = time.perf_counter()
        pair = run_edl_comparison_pair(params, return_model=True)
        model = pair["model_instance"]
        with_edl = pair["with_edl"]
        derived = with_edl["derived"]
        total_nm = float(derived["L_total"]) * 1.0e9
        sample_nm = _active_x_nm(total_nm)
        sample_tilde = sample_nm * 1.0e-9 / float(derived["lambda_D"])
        sample_phi = model.interpolate(
            float(with_edl["E_mix_V"]),
            sample_tilde,
            np.zeros_like(sample_tilde),
        )
        sample_phi_mV = (
            np.asarray(sample_phi, dtype=float)
            * float(derived["thermal_voltage_V"])
            * 1.0e3
        )
        shortest_wavelength_nm = 2.0 * total_nm / float(n_modes)
        target_dx_nm = total_nm / float(n_x - 1)
        E_mix = float(with_edl["E_mix_V"])
        i_mix = float(with_edl["i_mix_avg_A_per_m2"])
        row = {
            "level": label,
            "N_modes": int(n_modes),
            "n_coefficients": int(n_modes + 1),
            "Nx": int(n_x),
            "L_total_nm": total_nm,
            "shortest_cosine_wavelength_nm": shortest_wavelength_nm,
            "surface_target_dx_nm": target_dx_nm,
            "surface_points_per_shortest_wavelength": (
                shortest_wavelength_nm / target_dx_nm
            ),
            "E_mix_with_EDL_V": E_mix,
            "i_mix_avg_with_EDL_A_per_m2": i_mix,
            "max_abs_phi_tilde": float(
                with_edl["debye_huckel_validity"]["max_abs_phi_tilde"]
            ),
            "relative_current_balance_residual": float(
                with_edl["relative_balance_residual"]
            ),
            "affine_phi_m_relative_l2": float(
                with_edl["affine_residuals"]["phi_m_relative_l2"]
            ),
            "affine_phi_pzc_relative_l2": float(
                with_edl["affine_residuals"]["phi_pzc_relative_l2"]
            ),
            "delta_E_vs_previous_uV": (
                None if previous_E is None else (E_mix - previous_E) * 1.0e6
            ),
            "relative_delta_i_vs_previous_percent": (
                None
                if previous_i is None
                else 100.0 * (i_mix / previous_i - 1.0)
            ),
            "active_surface_max_abs_delta_vs_previous_mV": (
                None
                if previous_profile_mV is None
                else float(np.max(np.abs(sample_phi_mV - previous_profile_mV)))
            ),
            "elapsed_s": time.perf_counter() - started,
        }
        rows.append(row)
        previous_E = E_mix
        previous_i = i_mix
        previous_profile_mV = sample_phi_mV

        if label == "production":
            production_pair = pair
            production_model = model
            production_params = params
        else:
            del pair, model
            gc.collect()

    if production_pair is None or production_model is None or production_params is None:
        raise RuntimeError("Production refinement level was not solved")

    result = production_pair["with_edl"]
    derived = result["derived"]
    lambda_D = float(derived["lambda_D"])
    # A region at least 50 nm from either metal/substrate junction should be
    # effectively bulk at d=1000 nm.  Any residual peak-to-peak variation here
    # is a conservative Fourier-ringing diagnostic.
    gap_check_nm = np.linspace(75.0, 975.0, 3601)
    gap_phi = production_model.interpolate(
        float(result["E_mix_V"]),
        gap_check_nm * 1.0e-9 / lambda_D,
        np.zeros_like(gap_check_nm),
    )
    gap_phi_mV = (
        np.asarray(gap_phi, dtype=float)
        * float(derived["thermal_voltage_V"])
        * 1.0e3
    )
    ringing = {
        "definition": (
            "surface Phi_s over x=75-975 nm, at least 50 nm from either "
            "metal/substrate junction"
        ),
        "x_range_nm": [75.0, 975.0],
        "sample_count": int(gap_check_nm.size),
        "max_abs_phi_s_mV": float(np.max(np.abs(gap_phi_mV))),
        "peak_to_peak_phi_s_mV": float(np.ptp(gap_phi_mV)),
        "visibility_threshold_mV": 0.02,
        "passed": bool(float(np.ptp(gap_phi_mV)) < 0.02),
    }
    return rows, production_pair, production_model, {
        "params": production_params,
        "ringing": ringing,
    }


def _surface_profile(result: Mapping[str, Any], model: Any) -> dict[str, np.ndarray]:
    params = result["params"]
    derived = result["derived"]
    E_mix = float(result["E_mix_V"])
    x_tilde = np.asarray(model.surface_x_tilde, dtype=float)
    phi_tilde = np.asarray(model.surface_field(E_mix), dtype=float)
    x_m = x_tilde * float(derived["lambda_D"])
    x_nm = x_m * 1.0e9
    phi_s_V = phi_tilde * float(derived["thermal_voltage_V"])
    c_red1 = np.exp(np.clip(-float(params["z_R1"]) * phi_tilde, -700.0, 700.0))
    c_ox2 = np.exp(np.clip(-float(params["z_O2"]) * phi_tilde, -700.0, 700.0))
    j_au = np.full_like(phi_tilde, np.nan)
    j_pd = np.full_like(phi_tilde, np.nan)
    au_mask = x_m <= float(derived["x_Au_end"]) + 1.0e-15
    pd_mask = x_m >= float(derived["x_Pd_start"]) - 1.0e-15
    j_au[au_mask] = np.asarray(
        au_local_current_density(E_mix, phi_tilde[au_mask], params), dtype=float
    )
    j_pd[pd_mask] = np.asarray(
        pd_local_current_density(E_mix, phi_tilde[pd_mask], params), dtype=float
    )
    return {
        "x_m": x_m,
        "x_nm": x_nm,
        "phi_tilde": phi_tilde,
        "phi_s_V": phi_s_V,
        "c_Red1_over_c_bulk": c_red1,
        "c_Ox2_over_c_bulk": c_ox2,
        "j_Au_A_per_m2": j_au,
        "j_Pd_A_per_m2": j_pd,
    }


def _write_profile_csv(path: Path, profile: Mapping[str, np.ndarray]) -> None:
    keys = list(profile)
    rows = [
        {key: float(np.asarray(profile[key])[index]) for key in keys}
        for index in range(len(profile["x_nm"]))
    ]
    _write_rows(path, rows)


def _save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    transparent: bool = False,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for extension in ("png", "svg"):
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(
            path,
            dpi=DPI,
            bbox_inches="tight",
            pad_inches=0.04,
            transparent=transparent,
            facecolor="none" if transparent else "white",
            edgecolor="none" if transparent else "white",
        )
        saved.append(str(path.resolve()))
    plt.close(fig)
    return saved


def _active_windows(data: Figure3SavedData) -> tuple[tuple[float, float], ...]:
    return (
        (0.0, min(ACTIVE_WINDOW_NM, data.L_total_nm)),
        (max(0.0, data.L_total_nm - ACTIVE_WINDOW_NM), data.L_total_nm),
    )


def _window_ticks(
    data: Figure3SavedData, window: tuple[float, float]
) -> list[float]:
    xmin, xmax = window
    # Do not label the two endpoints adjacent to the break: in the compact
    # layout, "30" and "1020" collide.  The exact window limits are stated in
    # the panel headers, while the ticks retain the outer endpoint and the
    # physically useful metal/substrate boundary.
    if math.isclose(xmin, 0.0, abs_tol=1.0e-9):
        ticks = [xmin]
        if xmin < data.L_Au_nm < xmax:
            ticks.append(data.L_Au_nm)
    else:
        ticks = [xmax]
        if xmin < data.L_Pd_start_nm < xmax:
            ticks.append(data.L_Pd_start_nm)
    return sorted(round(value, 9) for value in ticks)


def _style_active_line_axis(
    ax: plt.Axes,
    ylabel: str,
    *,
    show_ylabel: bool,
) -> None:
    ax.set_facecolor("none")
    ax.set_ylabel(ylabel if show_ylabel else "", fontsize=9.8)
    ax.tick_params(axis="x", length=3.3, width=0.85, pad=2.5, labelsize=8.8)
    if show_ylabel:
        ax.tick_params(axis="y", which="both", length=3.3, width=0.85, pad=2.5, labelsize=8.8)
    else:
        # The right broken-axis window shares the left window's y limits but
        # must not repeat its major/minor scale.
        ax.tick_params(
            axis="y",
            which="both",
            left=False,
            right=False,
            labelleft=False,
            labelright=False,
        )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color(COLORS["dark"])


def _decorate_active_line_axis(
    ax: plt.Axes,
    data: Figure3SavedData,
    window: tuple[float, float],
) -> None:
    xmin, xmax = window
    gap_left = max(xmin, data.L_Au_nm)
    gap_right = min(xmax, data.L_Pd_start_nm)
    if gap_right > gap_left:
        ax.axvspan(
            gap_left,
            gap_right,
            color=COLORS["substrate"],
            alpha=0.45,
            linewidth=0.0,
            zorder=0,
        )
        if gap_right - gap_left >= 8.0:
            ax.text(
                0.5 * (gap_left + gap_right),
                0.08,
                "insulating substrate",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=6.6,
                color=COLORS["gray"],
                zorder=1,
            )
    for edge in (data.L_Au_nm, data.L_Pd_start_nm):
        if xmin <= edge <= xmax:
            ax.axvline(
                edge,
                color=COLORS["gray"],
                linewidth=0.85,
                linestyle=(0, (3, 2)),
                alpha=0.85,
                zorder=2,
            )
    ax.set_xlim(xmin, xmax)
    ticks = _window_ticks(data, window)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{value:g}" for value in ticks])


def _set_finite_ylim(ax: plt.Axes, arrays: Sequence[np.ndarray]) -> None:
    values = np.concatenate([np.ravel(np.asarray(array, dtype=float)) for array in arrays])
    values = values[np.isfinite(values)]
    if not values.size:
        return
    low = float(np.min(values))
    high = float(np.max(values))
    span = high - low
    pad = max(1.0e-6, 0.08 * span)
    ax.set_ylim(low - pad, high + pad)


def _add_broken_axis_marks(left: plt.Axes, right: plt.Axes) -> None:
    """Mark the omitted middle substrate interval on a paired x axis."""

    left.spines["right"].set_visible(False)
    right.spines["left"].set_visible(False)
    mark = 0.013
    common = {
        "color": COLORS["dark"],
        "clip_on": False,
        "linewidth": 0.9,
        "solid_capstyle": "butt",
        "zorder": 20,
    }
    left.plot((1.0 - mark, 1.0 + mark), (-mark, mark), transform=left.transAxes, **common)
    left.plot(
        (1.0 - mark, 1.0 + mark),
        (1.0 - mark, 1.0 + mark),
        transform=left.transAxes,
        **common,
    )
    right.plot((-mark, mark), (-mark, mark), transform=right.transAxes, **common)
    right.plot(
        (-mark, mark),
        (1.0 - mark, 1.0 + mark),
        transform=right.transAxes,
        **common,
    )


def _plot_figure3_cut_axis(
    data: Figure3SavedData,
    output_dir: Path,
    *,
    canonical_filenames: bool,
) -> dict[str, Any]:
    windows = _active_windows(data)
    active_mask = (data.x_nm <= windows[0][1]) | (data.x_nm >= windows[1][0])
    d_token = f"d{int(round(data.d_nm))}nm"
    saved: list[str] = []

    definitions: list[dict[str, Any]] = [
        {
            "panel": "b",
            "stem": "solution_potential_y0",
            "title": r"Solution potential along $\mathit{y}=0$",
            "ylabel": r"$\mathit{\phi}_{\mathrm{RP}}(\mathit{x})$ (V)",
            "plot": lambda ax: (
                ax.plot(data.x_nm, data.phi_s_V, color=COLORS["with_edl"], lw=2.0, label="with EDL", zorder=4),
                ax.plot(data.x_nm, np.zeros_like(data.x_nm), color=COLORS["without_edl"], lw=1.8, label="w/o EDL", zorder=3),
            ),
            "ylim_arrays": [data.phi_s_V[active_mask], np.asarray([0.0])],
            "legend_loc": "center right",
        },
        {
            "panel": "c",
            "stem": "reactant_concentration_y0",
            "title": r"Reactant concentration along $\mathit{y}=0$",
            "ylabel": r"$\mathit{c}_{\mathit{i}}/\mathit{c}_{\mathrm{bulk}}$ (-)",
            "plot": lambda ax: (
                ax.plot(data.x_nm, data.c_red1, color=COLORS["red1_i1"], lw=2.0, label=r"$\mathit{c}_{\mathrm{Red1}}/\mathit{c}_{\mathrm{bulk}}$ (with EDL)", zorder=4),
                ax.plot(data.x_nm, data.c_ox2, color=COLORS["ox2_i2"], lw=2.0, label=r"$\mathit{c}_{\mathrm{Ox2}}/\mathit{c}_{\mathrm{bulk}}$ (with EDL)", zorder=4),
                ax.plot(data.x_nm, np.ones_like(data.x_nm), color=COLORS["without_edl"], lw=1.6, ls=(0, (4, 2)), label="w/o EDL", zorder=3),
            ),
            "ylim_arrays": [data.c_red1[active_mask], data.c_ox2[active_mask], np.asarray([1.0])],
            "legend_loc": "upper right",
            "log": True,
        },
        {
            "panel": "d",
            "stem": "local_overpotential",
            "title": "Local overpotential at RP",
            "ylabel": r"$\mathit{\eta}(\mathit{x})$ (V)",
            "plot": lambda ax: (
                ax.plot(data.x_nm, data.eta_with, color=COLORS["with_edl"], lw=2.0, label="with EDL", zorder=4),
                ax.plot(data.x_nm, data.eta_no, color=COLORS["without_edl"], lw=1.8, label="w/o EDL", zorder=3),
            ),
            "ylim_arrays": [data.eta_with[active_mask], data.eta_no[active_mask]],
            "legend_loc": "center right",
        },
        {
            "panel": "e",
            "stem": "local_current_density",
            "title": "Local current density at RP",
            "ylabel": r"$\mathit{i}(\mathit{x})$ (A/m$^2$)",
            "plot": lambda ax: (
                ax.plot(data.x_nm, data.j_au_with, color=COLORS["red1_i1"], lw=2.0, label=r"$\mathit{i}_1$ (Au), with EDL", zorder=5),
                ax.plot(data.x_nm, data.j_pd_with, color=COLORS["ox2_i2"], lw=2.0, label=r"$\mathit{i}_2$ (Pd), with EDL", zorder=5),
                ax.plot(data.x_nm, data.j_au_no, color=COLORS["red1_i1"], lw=1.7, ls=(0, (4, 2)), label=r"$\mathit{i}_1$ (Au), w/o EDL", zorder=4),
                ax.plot(data.x_nm, data.j_pd_no, color=COLORS["ox2_i2"], lw=1.7, ls=(0, (4, 2)), label=r"$\mathit{i}_2$ (Pd), w/o EDL", zorder=4),
            ),
            "ylim_arrays": [
                data.j_au_with[active_mask],
                data.j_pd_with[active_mask],
                data.j_au_no[active_mask],
                data.j_pd_no[active_mask],
            ],
            "legend_loc": "upper right",
        },
    ]

    with plt.rc_context(PUBLICATION_RCPARAMS):
        for definition in definitions:
            fig = plt.figure(figsize=FIGURE3_CUT_FIGSIZE)
            grid = fig.add_gridspec(
                1,
                2,
                width_ratios=(1.0, 1.0),
                left=0.16,
                right=0.985,
                bottom=0.36,
                top=0.79,
                wspace=0.12,
            )
            axes = (
                fig.add_subplot(grid[0, 0]),
                fig.add_subplot(grid[0, 1]),
            )
            handles: list[Any] = []
            labels: list[str] = []
            for index, (ax, window, side_label) in enumerate(
                zip(
                    axes,
                    windows,
                    (
                        f"Au ({windows[0][0]:g}-{windows[0][1]:g} nm)",
                        f"Pd ({windows[1][0]:g}-{windows[1][1]:g} nm)",
                    ),
                    strict=True,
                )
            ):
                definition["plot"](ax)
                if index == 0:
                    handles, labels = ax.get_legend_handles_labels()
                if definition.get("log", False):
                    ax.set_yscale("log")
                _decorate_active_line_axis(ax, data, window)
                _style_active_line_axis(
                    ax,
                    definition["ylabel"],
                    show_ylabel=index == 0,
                )
                ax.set_title(
                    side_label,
                    loc="left",
                    pad=4,
                    fontsize=7.4,
                    fontweight="normal",
                )
                if definition.get("log", False):
                    positive = np.concatenate(
                        [
                            np.asarray(array)[np.asarray(array) > 0.0]
                            for array in definition["ylim_arrays"]
                        ]
                    )
                    ax.set_ylim(float(np.min(positive)) / 1.25, float(np.max(positive)) * 8.0)
                else:
                    _set_finite_ylim(ax, definition["ylim_arrays"])
            _add_broken_axis_marks(axes[0], axes[1])
            fig.suptitle(
                definition["title"],
                x=0.16,
                y=0.975,
                ha="left",
                fontsize=10.3,
                fontweight="normal",
            )
            fig.text(
                0.575,
                0.070,
                "x (nm)",
                ha="center",
                va="center",
                fontsize=9.2,
                color=COLORS["dark"],
            )
            fig.text(
                0.575,
                0.875,
                rf"$\mathit{{d}}_{{\mathrm{{Au-Pd}}}}$ = {data.d_nm:g} nm; 990 nm omitted",
                ha="center",
                va="center",
                fontsize=6.4,
                color=COLORS["gray"],
            )
            legend_columns = 2
            legend_fontsize = 5.8 if definition["panel"] in {"c", "e"} else 6.8
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.575, 0.275),
                ncols=legend_columns,
                fontsize=legend_fontsize,
                handlelength=1.7,
                columnspacing=0.9,
                labelspacing=0.25,
            )
            qualifier = "" if canonical_filenames else "active_zoom_"
            stem = (
                f"figure_3_panel_{definition['panel']}_{definition['stem']}_"
                f"{qualifier}{d_token}_{data.tag}"
            )
            saved.extend(_save_figure(fig, output_dir, stem, transparent=True))

    metadata = {
        "figure_type": "figure_3_broken_x_axis_spatial_panels",
        "canonical_filenames": canonical_filenames,
        "separation_nm": data.d_nm,
        "windows_nm": [list(window) for window in windows],
        "omitted_interval_nm": [windows[0][1], windows[1][0]],
        "omitted_width_nm": windows[1][0] - windows[0][1],
        "panels": [definition["panel"] for definition in definitions],
        "formats": ["png", "svg"],
        "dpi": DPI,
        "saved_paths": saved,
    }
    metadata_name = (
        "figure_3_cut_axis_metadata.json"
        if canonical_filenames
        else "figure_3_active_zoom_metadata.json"
    )
    _write_json(output_dir / metadata_name, metadata)
    return metadata


def _evaluate_active_fields(result: Mapping[str, Any], model: Any) -> dict[str, Any]:
    derived = result["derived"]
    params = result["params"]
    lambda_D = float(derived["lambda_D"])
    total_nm = float(derived["L_total"]) * 1.0e9
    windows = (
        np.linspace(0.0, min(ACTIVE_WINDOW_NM, total_nm), FIELD_NX_PER_WINDOW),
        np.linspace(
            max(0.0, total_nm - ACTIVE_WINDOW_NM),
            total_nm,
            FIELD_NX_PER_WINDOW,
        ),
    )
    y_nm = np.linspace(
        0.0,
        FIELD_Y_MAX_OVER_LAMBDA * lambda_D * 1.0e9,
        FIELD_NY,
    )
    y_tilde = y_nm * 1.0e-9 / lambda_D
    coefficients = model.coefficient_field(float(result["E_mix_V"]))
    decayed = np.exp(-np.outer(y_tilde, model.gamma)) * coefficients[None, :]
    phi_tilde: list[np.ndarray] = []
    for x_nm in windows:
        cosine = np.cos(np.outer(x_nm * 1.0e-9 / lambda_D, model.rho))
        phi_tilde.append(np.asarray(decayed @ cosine.T, dtype=float))
    thermal = float(derived["thermal_voltage_V"])
    phi_s_mV = [values * thermal * 1.0e3 for values in phi_tilde]
    c_red1 = [
        np.exp(np.clip(-float(params["z_R1"]) * values, -700.0, 700.0))
        for values in phi_tilde
    ]
    c_ox2 = [
        np.exp(np.clip(-float(params["z_O2"]) * values, -700.0, 700.0))
        for values in phi_tilde
    ]
    return {
        "x_nm": windows,
        "y_nm": y_nm,
        "phi_tilde": tuple(phi_tilde),
        "phi_s_mV": tuple(phi_s_mV),
        "c_red1": tuple(c_red1),
        "c_ox2": tuple(c_ox2),
        "lambda_D_nm": lambda_D * 1.0e9,
        "L_Au_nm": float(derived["L_Au"]) * 1.0e9,
        "L_Pd_start_nm": float(derived["x_Pd_start"]) * 1.0e9,
        "L_total_nm": total_nm,
        "E_mix_V": float(result["E_mix_V"]),
    }


def _write_active_window_fields(output: Path, fields: Mapping[str, Any]) -> Path:
    path = output / "fields" / "active_window_fields_d1000nm.npz"
    np.savez_compressed(
        path,
        x_Au_window_nm=fields["x_nm"][0],
        x_Pd_window_nm=fields["x_nm"][1],
        y_nm=fields["y_nm"],
        phi_tilde_Au_window=fields["phi_tilde"][0],
        phi_tilde_Pd_window=fields["phi_tilde"][1],
        phi_s_mV_Au_window=fields["phi_s_mV"][0],
        phi_s_mV_Pd_window=fields["phi_s_mV"][1],
        c_Red1_over_c_bulk_Au_window=fields["c_red1"][0],
        c_Red1_over_c_bulk_Pd_window=fields["c_red1"][1],
        c_Ox2_over_c_bulk_Au_window=fields["c_ox2"][0],
        c_Ox2_over_c_bulk_Pd_window=fields["c_ox2"][1],
    )
    return path


def _load_cropped_active_fields(
    output: Path,
    data: Figure3SavedData,
) -> dict[str, Any]:
    path = output / "fields" / "active_window_fields_d1000nm.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    target_windows = _active_windows(data)
    with np.load(path) as saved:
        x_all = (
            np.asarray(saved["x_Au_window_nm"], dtype=float),
            np.asarray(saved["x_Pd_window_nm"], dtype=float),
        )
        masks = tuple(
            (x_values >= window[0] - 1.0e-9)
            & (x_values <= window[1] + 1.0e-9)
            for x_values, window in zip(x_all, target_windows, strict=True)
        )
        x_nm = tuple(x_values[mask] for x_values, mask in zip(x_all, masks, strict=True))
        for values, window in zip(x_nm, target_windows, strict=True):
            if values.size < 2 or not math.isclose(values[0], window[0], abs_tol=1.0e-8) or not math.isclose(values[-1], window[1], abs_tol=1.0e-8):
                raise ValueError(
                    f"Saved active field does not cover requested window {window}"
                )

        def cropped_pair(au_key: str, pd_key: str) -> tuple[np.ndarray, np.ndarray]:
            return (
                np.asarray(saved[au_key], dtype=float)[:, masks[0]],
                np.asarray(saved[pd_key], dtype=float)[:, masks[1]],
            )

        phi_tilde = cropped_pair("phi_tilde_Au_window", "phi_tilde_Pd_window")
        phi_s_mV = cropped_pair("phi_s_mV_Au_window", "phi_s_mV_Pd_window")
        c_red1 = cropped_pair(
            "c_Red1_over_c_bulk_Au_window",
            "c_Red1_over_c_bulk_Pd_window",
        )
        c_ox2 = cropped_pair(
            "c_Ox2_over_c_bulk_Au_window",
            "c_Ox2_over_c_bulk_Pd_window",
        )
        y_nm = np.asarray(saved["y_nm"], dtype=float)

    derived = json.loads((output / "derived.json").read_text(encoding="utf-8"))
    return {
        "x_nm": x_nm,
        "y_nm": y_nm,
        "phi_tilde": phi_tilde,
        "phi_s_mV": phi_s_mV,
        "c_red1": c_red1,
        "c_ox2": c_ox2,
        "lambda_D_nm": float(derived["lambda_D"]) * 1.0e9,
        "L_Au_nm": data.L_Au_nm,
        "L_Pd_start_nm": data.L_Pd_start_nm,
        "L_total_nm": data.L_total_nm,
        "E_mix_V": data.E_mix_with,
    }


def _log_norm(arrays: Sequence[np.ndarray]) -> LogNorm:
    values = np.concatenate([np.ravel(np.asarray(array, dtype=float)) for array in arrays])
    values = values[np.isfinite(values) & (values > 0.0)]
    lower = 10.0 ** math.floor(math.log10(float(np.min(values))))
    upper = 10.0 ** math.ceil(math.log10(float(np.max(values))))
    return LogNorm(vmin=lower, vmax=upper)


def _map_axis_style(ax: plt.Axes, title: str, *, show_ylabel: bool) -> None:
    ax.set_title(title, loc="left", pad=5, fontsize=9.2, fontweight="normal")
    ax.set_xlabel("")
    ax.set_ylabel("y (nm)" if show_ylabel else "")
    ax.tick_params(axis="x", length=3.2, width=0.85, pad=2.5, labelsize=8.0)
    if show_ylabel:
        ax.tick_params(axis="y", length=3.2, width=0.85, pad=2.5, labelsize=8.0)
    else:
        ax.tick_params(
            axis="y",
            which="both",
            left=False,
            right=False,
            labelleft=False,
            labelright=False,
        )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color(COLORS["dark"])


def _add_map_edges(ax: plt.Axes, fields: Mapping[str, Any], window_index: int) -> None:
    xmin = float(fields["x_nm"][window_index][0])
    xmax = float(fields["x_nm"][window_index][-1])
    for edge in (float(fields["L_Au_nm"]), float(fields["L_Pd_start_nm"])):
        if xmin <= edge <= xmax:
            ax.axvline(
                edge,
                color=COLORS["gray"],
                linewidth=0.8,
                linestyle=(0, (3, 2)),
                alpha=0.80,
                zorder=5,
            )


def _add_material_lane_window(
    ax: plt.Axes,
    fields: Mapping[str, Any],
    window_index: int,
    *,
    show_xlabel: bool = True,
) -> None:
    xmin = float(fields["x_nm"][window_index][0])
    xmax = float(fields["x_nm"][window_index][-1])
    L_Au = float(fields["L_Au_nm"])
    x_Pd = float(fields["L_Pd_start_nm"])
    total = float(fields["L_total_nm"])
    segments = (
        ("Au", 0.0, L_Au, RP_MODEL_AU_COLOR, COLORS["dark"]),
        ("C", L_Au, x_Pd, RP_MODEL_C_COLOR, "white"),
        ("Pd", x_Pd, total, RP_MODEL_PD_COLOR, "white"),
    )
    for label, x0, x1, face, text_color in segments:
        left = max(xmin, x0)
        right = min(xmax, x1)
        if right <= left:
            continue
        ax.add_patch(
            Rectangle(
                (left, 0.0),
                right - left,
                1.0,
                facecolor=face,
                edgecolor="white",
                linewidth=0.8,
            )
        )
        if right - left >= 6.0:
            ax.text(
                0.5 * (left + right),
                0.5,
                label,
                ha="center",
                va="center",
                fontsize=7.4 if "substrate" in label else 8.3,
                color=text_color,
                linespacing=0.9,
            )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()
    if show_xlabel:
        ax.text(
            0.5,
            -0.58,
            "x (nm)",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
            color=COLORS["dark"],
            clip_on=False,
        )


def _map_ticks(fields: Mapping[str, Any], window_index: int) -> list[float]:
    xmin = float(fields["x_nm"][window_index][0])
    xmax = float(fields["x_nm"][window_index][-1])
    if window_index == 0:
        ticks = [xmin]
        boundary = float(fields["L_Au_nm"])
        if xmin < boundary < xmax:
            ticks.append(boundary)
    else:
        ticks = [xmax]
        boundary = float(fields["L_Pd_start_nm"])
        if xmin < boundary < xmax:
            ticks.append(boundary)
    return sorted(round(value, 9) for value in ticks)


def _plot_potential_active_zoom(
    fields: Mapping[str, Any], output_dir: Path, tag: str
) -> dict[str, Any]:
    max_abs = max(
        float(np.max(np.abs(np.asarray(values)))) for values in fields["phi_s_mV"]
    )
    limit = max(10.0, 10.0 * math.ceil(max_abs / 10.0))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    levels = np.linspace(-limit, limit, 11)
    # The exact zero contour turns sub-0.01 mV truncation noise over the bulk
    # gap into a visually prominent sawtooth.  The field itself is retained;
    # only this numerically meaningless neutral contour is omitted.
    visible_levels = levels[np.abs(levels) > 1.0e-12]
    with plt.rc_context(PUBLICATION_RCPARAMS):
        fig = plt.figure(figsize=RP_POTENTIAL_FIGSIZE)
        grid = fig.add_gridspec(
            nrows=2,
            ncols=3,
            width_ratios=(1.0, 1.0, 0.070),
            height_ratios=(1.0, 0.12),
            hspace=0.24,
            wspace=0.10,
            left=0.16,
            right=0.88,
            bottom=0.16,
            top=0.80,
        )
        axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
        cax = fig.add_subplot(grid[0, 2])
        lane_axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
        fig.add_subplot(grid[1, 2]).set_axis_off()
        mesh = None
        for index, (ax, title) in enumerate(
            zip(
                axes,
                (
                    f"Au side ({fields['x_nm'][0][0]:g}-{fields['x_nm'][0][-1]:g} nm)",
                    f"Pd side ({fields['x_nm'][1][0]:g}-{fields['x_nm'][1][-1]:g} nm)",
                ),
                strict=True,
            )
        ):
            mesh = ax.pcolormesh(
                fields["x_nm"][index],
                fields["y_nm"],
                fields["phi_s_mV"][index],
                shading="auto",
                cmap="RdBu_r",
                norm=norm,
                rasterized=True,
            )
            ax.contour(
                fields["x_nm"][index],
                fields["y_nm"],
                fields["phi_s_mV"][index],
                levels=visible_levels,
                colors="black",
                linewidths=0.28,
                alpha=0.28,
            )
            _add_map_edges(ax, fields, index)
            _map_axis_style(ax, title, show_ylabel=index == 0)
            ax.set_ylim(0.0, float(np.asarray(fields["y_nm"])[-1]))
            ax.set_yticks([0.0, 5.0, 10.0, 15.0])
            ticks = _map_ticks(fields, index)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{value:g}" for value in ticks])
            _add_material_lane_window(
                lane_axes[index],
                fields,
                index,
                show_xlabel=False,
            )
        _add_broken_axis_marks(axes[0], axes[1])
        if mesh is None:
            raise RuntimeError("No potential active-window mesh was created")
        colorbar = fig.colorbar(mesh, cax=cax)
        colorbar.set_label(r"$\Phi_s$ (mV)", labelpad=5)
        colorbar.ax.tick_params(length=2.8, width=0.75, labelsize=7.8, pad=2.2)
        colorbar.outline.set_linewidth(0.8)
        fig.suptitle(
            "Solution-phase potential",
            x=0.16,
            y=0.985,
            ha="left",
            fontsize=10.2,
            fontweight="normal",
        )
        fig.text(
            0.52,
            0.90,
            rf"$d_{{\mathrm{{Au-Pd}}}}$ = {SEPARATION_NM:g} nm; 990 nm omitted",
            ha="center",
            va="center",
            fontsize=6.8,
            color=COLORS["gray"],
        )
        fig.text(
            0.52,
            0.035,
            "x (nm)",
            ha="center",
            va="center",
            fontsize=9.0,
            color=COLORS["dark"],
        )
        stem = f"solution_phase_potential_2d_active_zoom_d1000nm_{tag}"
        saved = _save_figure(fig, output_dir, stem)
    return {
        "figure_type": "solution_phase_potential_2d_active_zoom",
        "phi_s_limit_mV": [-limit, limit],
        "saved_paths": saved,
    }


def _plot_composite_active_zoom(
    fields: Mapping[str, Any], output_dir: Path, tag: str
) -> dict[str, Any]:
    max_abs = max(
        float(np.max(np.abs(np.asarray(values)))) for values in fields["phi_s_mV"]
    )
    phi_limit = max(10.0, 10.0 * math.ceil(max_abs / 10.0))
    phi_norm = TwoSlopeNorm(vmin=-phi_limit, vcenter=0.0, vmax=phi_limit)
    # Keep the concentration scales identical to the preceding d=1000 nm
    # figure bundle.  Cropping the display window must not silently tighten
    # the normalization and make the revised maps incomparable to the source.
    red_norm = LogNorm(vmin=RP_RED1_LIMITS[0], vmax=RP_RED1_LIMITS[1])
    ox_norm = LogNorm(vmin=RP_OX2_LIMITS[0], vmax=RP_OX2_LIMITS[1])
    rows = (
        (
            "Solution-phase potential",
            fields["phi_s_mV"],
            "RdBu_r",
            phi_norm,
            np.linspace(-phi_limit, phi_limit, 11),
            r"$\Phi_s$ (mV)",
        ),
        (
            "Reactant Red1 distribution",
            fields["c_red1"],
            "viridis",
            red_norm,
            np.geomspace(float(red_norm.vmin), float(red_norm.vmax), 9),
            r"$c_{\mathrm{Red1}}/c_{\mathrm{bulk}}$",
        ),
        (
            "Reactant Ox2 distribution",
            fields["c_ox2"],
            "viridis",
            ox_norm,
            np.geomspace(float(ox_norm.vmin), float(ox_norm.vmax), 9),
            r"$c_{\mathrm{Ox2}}/c_{\mathrm{bulk}}$",
        ),
    )
    # As for the zero-potential contour, c/c_bulk=1 is the neutral bulk level.
    # Removing only that contour prevents tiny far-gap truncation noise from
    # being amplified into an apparent physical oscillation.
    rows = tuple(
        (
            row_title,
            values,
            cmap,
            norm,
            np.asarray(levels)[
                np.abs(np.asarray(levels) - (0.0 if row_index == 0 else 1.0))
                > 1.0e-12
            ],
            cbar_label,
        )
        for row_index, (row_title, values, cmap, norm, levels, cbar_label) in enumerate(rows)
    )
    with plt.rc_context(PUBLICATION_RCPARAMS):
        fig = plt.figure(figsize=RP_COMPOSITE_FIGSIZE)
        grid = fig.add_gridspec(
            nrows=3,
            ncols=3,
            width_ratios=(1.0, 1.0, 0.070),
            height_ratios=(1.0, 1.0, 1.0),
            hspace=0.31,
            wspace=0.10,
            left=0.16,
            right=0.88,
            bottom=0.075,
            top=0.91,
        )
        axes: list[list[plt.Axes]] = []
        for row_index, (row_title, values, cmap, norm, levels, cbar_label) in enumerate(rows):
            row_axes = [fig.add_subplot(grid[row_index, 0]), fig.add_subplot(grid[row_index, 1])]
            cax = fig.add_subplot(grid[row_index, 2])
            axes.append(row_axes)
            mesh = None
            for window_index, ax in enumerate(row_axes):
                mesh = ax.pcolormesh(
                    fields["x_nm"][window_index],
                    fields["y_nm"],
                    values[window_index],
                    shading="auto",
                    cmap=cmap,
                    norm=norm,
                    rasterized=True,
                )
                ax.contour(
                    fields["x_nm"][window_index],
                    fields["y_nm"],
                    values[window_index],
                    levels=levels,
                    colors="black",
                    linewidths=0.28,
                    alpha=0.27,
                )
                _add_map_edges(ax, fields, window_index)
                _map_axis_style(
                    ax,
                    row_title if window_index == 0 else "",
                    show_ylabel=window_index == 0,
                )
                ax.set_ylim(0.0, float(np.asarray(fields["y_nm"])[-1]))
                ax.set_yticks([0.0, 5.0, 10.0, 15.0])
                ticks = _map_ticks(fields, window_index)
                ax.set_xticks(ticks)
                if row_index == len(rows) - 1:
                    ax.set_xticklabels([f"{value:g}" for value in ticks])
                else:
                    ax.tick_params(labelbottom=False)
            _add_broken_axis_marks(row_axes[0], row_axes[1])
            if mesh is None:
                raise RuntimeError("No composite active-window mesh was created")
            colorbar = fig.colorbar(mesh, cax=cax)
            colorbar.set_label(cbar_label, labelpad=5)
            colorbar.ax.tick_params(length=2.8, width=0.75, labelsize=7.5, pad=2.2)
            colorbar.outline.set_linewidth(0.8)
        fig.text(
            0.16,
            0.985,
            (
                rf"$d_{{\mathrm{{Au-Pd}}}}$ = {SEPARATION_NM:g} nm; "
                f"windows {fields['x_nm'][0][0]:g}-{fields['x_nm'][0][-1]:g} nm"
                f" | {fields['x_nm'][1][0]:g}-{fields['x_nm'][1][-1]:g} nm"
            ),
            ha="left",
            va="top",
            fontsize=7.0,
            color=COLORS["gray"],
        )
        fig.text(
            0.52,
            0.025,
            "x (nm)",
            ha="center",
            va="center",
            fontsize=9.0,
            color=COLORS["dark"],
        )
        stem = f"phi_s_reactants_2d_active_zoom_d1000nm_{tag}"
        saved = _save_figure(fig, output_dir, stem)
    return {
        "figure_type": "phi_s_reactants_2d_active_zoom",
        "phi_s_limit_mV": [-phi_limit, phi_limit],
        "c_Red1_limits": [float(red_norm.vmin), float(red_norm.vmax)],
        "c_Ox2_limits": [float(ox_norm.vmin), float(ox_norm.vmax)],
        "saved_paths": saved,
    }


def _save_core_bundle(
    output: Path,
    source_params: Mapping[str, Any],
    refinement_rows: Sequence[Mapping[str, Any]],
    pair: Mapping[str, Any],
    model: Any,
    audit: Mapping[str, Any],
) -> tuple[str, dict[str, np.ndarray]]:
    params = audit["params"]
    result = pair["with_edl"]
    without = pair["without_edl"]
    tag = make_run_tag(params)
    profile = _surface_profile(result, model)

    _write_json(output / "params.json", params)
    _write_json(output / "derived.json", result["derived"])
    _write_json(output / "environment.json", environment_metadata(PACKAGE_ROOT))
    _write_json(output / "inputs" / "source_params_20260802_144648.json", source_params)
    _write_json(
        output / "inputs" / "numerical_overrides_d1000nm.json",
        {
            "source_result": str(SOURCE_RESULT.resolve()),
            "physical_override": {"d_Au_Pd_m": SEPARATION_M, "d_Au_Pd_nm": SEPARATION_NM},
            "numerical_override": {
                "N_modes": int(params["N_modes"]),
                "Nx": int(params["Nx"]),
            },
            "unchanged_physical_parameters": (
                "All source physical parameters except d_Au_Pd are unchanged."
            ),
        },
    )
    _write_rows(output / "convergence" / "resolution_refinement.csv", refinement_rows)
    _write_json(
        output / "convergence" / "resolution_refinement.json",
        {
            "levels": refinement_rows,
            "production_selection": {
                "N_modes": int(params["N_modes"]),
                "Nx": int(params["Nx"]),
                "reason": (
                    "Fourfold mode refinement and approximately fourfold Nx refinement "
                    "relative to 20260802_144648; about ten surface points per shortest "
                    "retained cosine wavelength."
                ),
            },
            "far_gap_ringing_check": audit["ringing"],
        },
    )

    summary_rows = [_case_summary(without), _case_summary(result)]
    _write_rows(output / "csv" / "summary_cases.csv", summary_rows)
    _write_profile_csv(output / "csv" / "top_surface_profiles_d1000nm.csv", profile)
    np.savez_compressed(
        output / "fields" / "spectral_field_d1000nm.npz",
        mode_index=model.mode_index,
        rho=model.rho,
        gamma=model.gamma,
        coefficients=model.coefficient_field(float(result["E_mix_V"])),
        affine_coefficients_m=model.phi_m,
        affine_coefficients_pzc=model.phi_pzc,
        surface_x_tilde=model.surface_x_tilde,
        surface_phi_tilde=profile["phi_tilde"],
        lambda_D_m=np.asarray(float(result["derived"]["lambda_D"])),
        E_mix_V=np.asarray(float(result["E_mix_V"])),
    )
    spectral_metadata = dict(result["electrostatics"])
    spectral_metadata.update(
        {
            "saved_surface_n_x_coordinates": int(len(profile["x_nm"])),
            "n_x_coordinates": int(len(profile["x_nm"])),
            "n_x_coordinates_role": "saved augmented diagnostic surface profile",
            "long_gap_resolution_audit": audit["ringing"],
        }
    )
    _write_json(
        output / "fields" / "spectral_metadata_d1000nm.json",
        {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "electrostatic_backend": ELECTROSTATIC_BACKEND,
            "case": _case_summary(result),
            "root": result["root"],
            "electrostatics": spectral_metadata,
            "affine_residuals": result["affine_residuals"],
            "derived": result["derived"],
            "effective_reaction": result["effective_reaction"],
        },
    )

    summary = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "electrostatic_backend": ELECTROSTATIC_BACKEND,
        "model": MODEL_NAME,
        "geometry": GEOMETRY_NAME,
        "tag": tag,
        "source_result": str(SOURCE_RESULT.resolve()),
        "retained_cases": {"1000.0": _case_summary(result)},
        "without_edl_reference": _case_summary(without),
        "production_resolution": {
            "N_modes": int(params["N_modes"]),
            "Nx": int(params["Nx"]),
        },
        "far_gap_ringing_check": audit["ringing"],
        "debye_huckel_caveat": (
            "max_abs_phi_tilde exceeds 1; this is a linearized-PB internal "
            "geometry-sensitivity result, not an absolute quantitative prediction."
        ),
    }
    _write_json(output / "summary.json", summary)
    manifest = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "electrostatic_backend": ELECTROSTATIC_BACKEND,
        "command": "make_d1000nm_figure_set.py",
        "created_local": datetime.now().astimezone().isoformat(),
        "model": MODEL_NAME,
        "geometry": GEOMETRY_NAME,
        "tag": tag,
        "parent_result": str(SOURCE_RESULT.resolve()),
        "output_dir": str(output.resolve()),
        "d_Au_Pd_nm": SEPARATION_NM,
        "N_modes": int(params["N_modes"]),
        "Nx": int(params["Nx"]),
        "formats": ["png", "svg"],
        "pdf_policy": "no PDF generated",
        "far_gap_ringing_check": audit["ringing"],
    }
    _write_json(output / "run_manifest.json", manifest)
    return tag, profile


def _merge_cut_axis_metadata(
    figure3_dir: Path,
    canonical_metadata: Mapping[str, Any],
    cut_axis_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    updated = dict(canonical_metadata)
    updated.update(
        {
            "spatial_axis_layout": "broken_x_axis_two_active_windows",
            "spatial_panels": ["b", "c", "d", "e"],
            "spatial_windows_nm": cut_axis_metadata["windows_nm"],
            "omitted_interval_nm": cut_axis_metadata["omitted_interval_nm"],
            "omitted_width_nm": cut_axis_metadata["omitted_width_nm"],
            "spatial_panel_rendering": cut_axis_metadata,
        }
    )
    _write_json(figure3_dir / "figure_3_metadata.json", updated)
    return updated


def _replace_string_tree(value: Any, old: str, new: str) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _replace_string_tree(item, old, new)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_string_tree(item, old, new) for item in value]
    if isinstance(value, str):
        return value.replace(old, new)
    return value


def _rebase_moved_bundle_metadata(output: Path) -> None:
    """Update absolute paths after the user moves an intact result bundle."""

    manifest_path = output / "run_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_output = manifest.get("output_dir")
    new_output = str(output.resolve())
    if not isinstance(old_output, str) or old_output == new_output:
        return
    if Path(old_output).name != output.name:
        raise ValueError(
            "Refusing to rebase metadata because the recorded and current "
            "bundle directory names differ"
        )
    for json_path in sorted(output.rglob("*.json")):
        value = json.loads(json_path.read_text(encoding="utf-8"))
        _write_json(json_path, _replace_string_tree(value, old_output, new_output))


def refresh_figure3_only(output: Path) -> dict[str, Any]:
    """Redraw Figure 3 and RP maps from saved data without rerunning the solver."""

    if not output.is_dir():
        raise FileNotFoundError(output)
    for required in (
        output / "params.json",
        output / "summary.json",
        output / "csv" / "summary_cases.csv",
        output / "csv" / "top_surface_profiles_d1000nm.csv",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    _rebase_moved_bundle_metadata(output)
    figure3_dir = output / "figures" / "Figure_3"
    canonical_metadata = generate_figure3_comparison_panels(
        output,
        figure3_dir,
        separation_nm=SEPARATION_NM,
        dpi=DPI,
        pzc_marker_y=FIGURE3_PZC_MARKER_Y,
        pzc_text_y=FIGURE3_PZC_TEXT_Y,
    )
    data = load_figure3_saved_data(output, SEPARATION_NM)
    cut_axis = _plot_figure3_cut_axis(
        data,
        figure3_dir,
        canonical_filenames=True,
    )
    canonical_metadata = _merge_cut_axis_metadata(
        figure3_dir,
        canonical_metadata,
        cut_axis,
    )
    active_zoom = _plot_figure3_cut_axis(
        data,
        figure3_dir / "active_zoom",
        canonical_filenames=False,
    )

    fields = _load_cropped_active_fields(output, data)
    source_field_path = _write_active_window_fields(output, fields)
    rp_dir = output / "figures" / "rp_2d"
    potential_2d = _plot_potential_active_zoom(fields, rp_dir, data.tag)
    composite_2d = _plot_composite_active_zoom(fields, rp_dir, data.tag)
    rp_metadata = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "electrostatic_backend": ELECTROSTATIC_BACKEND,
        "electrolyte_domain_id": ELECTROLYTE_DOMAIN_ID,
        "electrolyte_domain_description": ELECTROLYTE_DOMAIN_DESCRIPTION,
        "substrate_bc": "homogeneous Neumann",
        "separation_nm": SEPARATION_NM,
        "active_windows_nm": [
            [float(fields["x_nm"][0][0]), float(fields["x_nm"][0][-1])],
            [float(fields["x_nm"][1][0]), float(fields["x_nm"][1][-1])],
        ],
        "grid": {
            "n_x_per_window": int(np.asarray(fields["x_nm"][0]).size),
            "n_y": int(np.asarray(fields["y_nm"]).size),
            "y_max_over_lambda_D": FIELD_Y_MAX_OVER_LAMBDA,
            "lambda_D_nm": fields["lambda_D_nm"],
        },
        "potential_only": potential_2d,
        "potential_and_reactants": composite_2d,
        "source_field_npz": str(source_field_path.resolve()),
    }
    _write_json(rp_dir / "rp_2d_active_zoom_metadata.json", rp_metadata)

    pngs = sorted(output.glob("figures/**/*.png"))
    svgs = sorted(output.glob("figures/**/*.svg"))
    pdfs = sorted(output.glob("**/*.pdf"))
    if len(pngs) < 12 or len(svgs) < 12 or len(pngs) != len(svgs):
        raise RuntimeError(
            "Expected at least the 12 core PNG/SVG pairs and matching format "
            f"counts; got {len(pngs)} PNG and {len(svgs)} SVG"
        )
    if pdfs:
        raise RuntimeError(f"Unexpected PDF outputs: {pdfs}")

    bundle_path = output / "figure_bundle_metadata.json"
    bundle = (
        json.loads(bundle_path.read_text(encoding="utf-8"))
        if bundle_path.is_file()
        else {}
    )
    bundle.update(
        {
            "output_dir": str(output.resolve()),
            "canonical_figure3": canonical_metadata,
            "cut_axis_figure3": cut_axis,
            "active_zoom_figure3": active_zoom,
            "rp_2d": rp_metadata,
            "figure_counts": {
                "png": len(pngs),
                "svg": len(svgs),
                "pdf": len(pdfs),
            },
            "all_png": [str(path.resolve()) for path in pngs],
            "all_svg": [str(path.resolve()) for path in svgs],
        }
    )
    _write_json(bundle_path, bundle)
    return bundle


def build(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    source_params_path = SOURCE_RESULT / "params.json"
    if not source_params_path.is_file():
        raise FileNotFoundError(source_params_path)
    source_params = json.loads(source_params_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "fields").mkdir()

    refinement_rows, pair, model, audit = _solve_refinement(source_params)
    tag, profile = _save_core_bundle(
        output,
        source_params,
        refinement_rows,
        pair,
        model,
        audit,
    )

    figure3_dir = output / "figures" / "Figure_3"
    canonical_figure3 = generate_figure3_comparison_panels(
        output,
        figure3_dir,
        separation_nm=SEPARATION_NM,
        dpi=DPI,
        pzc_marker_y=FIGURE3_PZC_MARKER_Y,
        pzc_text_y=FIGURE3_PZC_TEXT_Y,
    )
    figure3_data = load_figure3_saved_data(output, SEPARATION_NM)
    cut_axis_figure3 = _plot_figure3_cut_axis(
        figure3_data,
        figure3_dir,
        canonical_filenames=True,
    )
    canonical_figure3 = _merge_cut_axis_metadata(
        figure3_dir,
        canonical_figure3,
        cut_axis_figure3,
    )
    active_figure3 = _plot_figure3_cut_axis(
        figure3_data,
        figure3_dir / "active_zoom",
        canonical_filenames=False,
    )

    fields = _evaluate_active_fields(pair["with_edl"], model)
    active_field_path = _write_active_window_fields(output, fields)
    rp_dir = output / "figures" / "rp_2d"
    potential_2d = _plot_potential_active_zoom(fields, rp_dir, tag)
    composite_2d = _plot_composite_active_zoom(fields, rp_dir, tag)
    rp_metadata = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "electrostatic_backend": ELECTROSTATIC_BACKEND,
        "electrolyte_domain_id": ELECTROLYTE_DOMAIN_ID,
        "electrolyte_domain_description": ELECTROLYTE_DOMAIN_DESCRIPTION,
        "substrate_bc": "homogeneous Neumann",
        "separation_nm": SEPARATION_NM,
        "active_windows_nm": [
            [float(fields["x_nm"][0][0]), float(fields["x_nm"][0][-1])],
            [float(fields["x_nm"][1][0]), float(fields["x_nm"][1][-1])],
        ],
        "grid": {
            "n_x_per_window": FIELD_NX_PER_WINDOW,
            "n_y": FIELD_NY,
            "y_max_over_lambda_D": FIELD_Y_MAX_OVER_LAMBDA,
            "lambda_D_nm": fields["lambda_D_nm"],
        },
        "potential_only": potential_2d,
        "potential_and_reactants": composite_2d,
        "source_field_npz": str(active_field_path.resolve()),
    }
    _write_json(rp_dir / "rp_2d_active_zoom_metadata.json", rp_metadata)

    pngs = sorted(output.glob("figures/**/*.png"))
    svgs = sorted(output.glob("figures/**/*.svg"))
    pdfs = sorted(output.glob("**/*.pdf"))
    if len(pngs) != 12 or len(svgs) != 12:
        raise RuntimeError(
            f"Expected 12 PNG and 12 SVG outputs; got {len(pngs)} PNG and {len(svgs)} SVG"
        )
    if pdfs:
        raise RuntimeError(f"Unexpected PDF outputs: {pdfs}")
    if not np.allclose(
        fields["phi_tilde"][0][0],
        np.interp(fields["x_nm"][0], profile["x_nm"], profile["phi_tilde"]),
        rtol=0.0,
        # The saved Nx=20001 profile and the 0.05 nm active-window grid are
        # distinct.  A 0.1 mV-equivalent interpolation tolerance is stricter
        # than the figure scale while accounting for the steep junction.
        atol=4.0e-3,
    ):
        raise RuntimeError("Au active-window y=0 field disagrees with saved surface profile")
    if not np.allclose(
        fields["phi_tilde"][1][0],
        np.interp(fields["x_nm"][1], profile["x_nm"], profile["phi_tilde"]),
        rtol=0.0,
        atol=4.0e-3,
    ):
        raise RuntimeError("Pd active-window y=0 field disagrees with saved surface profile")

    bundle_metadata = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "electrostatic_backend": ELECTROSTATIC_BACKEND,
        "output_dir": str(output.resolve()),
        "source_result": str(SOURCE_RESULT.resolve()),
        "d_Au_Pd_nm": SEPARATION_NM,
        "N_modes": int(audit["params"]["N_modes"]),
        "Nx": int(audit["params"]["Nx"]),
        "far_gap_ringing_check": audit["ringing"],
        "figure_counts": {"png": len(pngs), "svg": len(svgs), "pdf": len(pdfs)},
        "canonical_figure3": canonical_figure3,
        "cut_axis_figure3": cut_axis_figure3,
        "active_zoom_figure3": active_figure3,
        "rp_2d": rp_metadata,
        "all_png": [str(path.resolve()) for path in pngs],
        "all_svg": [str(path.resolve()) for path in svgs],
    }
    _write_json(output / "figure_bundle_metadata.json", bundle_metadata)
    return bundle_metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"new output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--refresh-figure3-only",
        action="store_true",
        help=(
            "redraw Figure 3 and RP maps from an existing saved bundle "
            "without rerunning the solver"
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output.resolve()
    metadata = (
        refresh_figure3_only(output)
        if args.refresh_figure3_only
        else build(output)
    )
    print(json.dumps(metadata["figure_counts"], sort_keys=True))
    print(metadata["output_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
