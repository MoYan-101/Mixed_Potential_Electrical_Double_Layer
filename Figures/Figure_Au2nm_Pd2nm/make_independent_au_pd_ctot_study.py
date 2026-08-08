"""Build the C_tot study for the fixed Au=Pd=2 nm independent-EDL model.

The numerical scan and the five reference figure classes are owned by the
``au_pd_independent_edls.ctot_study`` module.  This wrapper supplies the
Au2/Pd2 parameter lock, adapts conclusions that depend on those parameters,
uses the project colour/DPI conventions, and adds project-level regression
and artifact checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image


sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "Mixed_Potential_Electrical_Double_Layer" / "Au_Pd_independent_EDLs"
MODEL_SRC = MODEL_ROOT / "src"
PROJECT_DIR = Path(__file__).resolve().parent
INDEPENDENT_DIR = PROJECT_DIR / "Au_Pd_independent"
DEFAULT_OUTPUT = INDEPENDENT_DIR / "C_tot_study"
CHECKSUM_FILE = "checksums.sha256"
CHECKSUM_IGNORED_FILENAMES = {CHECKSUM_FILE, ".DS_Store"}
STUDY_ID = "independent_Au2nm_Pd2nm_CH50_Ctot_scan"
REFERENCE_RESULT = (
    "Mixed_Potential_Electrical_Double_Layer/Au_Pd_independent_EDLs/"
    "results/20260803_153355_ctot_study"
)
EXPECTED_FIGURE_PAIRS = 6
TREND_FIGSIZE_IN = (3.0, 3.45)
TREND_CANVAS_PX_AT_600_DPI = (1800, 2070)
TREND_SUBPLOT_ADJUST = {
    "left": 0.24,
    "right": 0.975,
    "bottom": 0.18,
    "top": 0.965,
}
IMIX_A_PER_M2_TO_MA_PER_CM2 = 0.1
IMIX_TREND_YLIM_MA_PER_CM2 = (0.0, 0.013)
POLARIZATION_OVERLAY_STEM = (
    "ctot_half_reaction_polarization_overlay_independent_edls"
)
PROFILE_SCHEME_STEM = (
    "ctot_phi_bar_profiles_0p01M_1M_with_without_edl_independent_edls"
)
PROFILE_SCHEME_SIDE_BY_SIDE_STEM = f"{PROFILE_SCHEME_STEM}_au_pd_side_by_side"
PROFILE_SCHEME_SIDE_BY_SIDE_HEIGHT_IN = 3.45
PROFILE_SCHEME_SIDE_BY_SIDE_HEIGHT_PX_AT_600_DPI = 2070

if str(MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_SRC))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from au_pd_independent_edls import ctot_study as ctot  # noqa: E402
from make_independent_au_pd import independent_params  # noqa: E402


PALETTE = {
    "with_edl": "#F26B38",
    "without_edl": "#12355B",
    "au": "#E4C133",
    "pd": "#5A90C8",
    "au_curve": "#009E73",
    "pd_curve": "#0072B2",
}

EXPECTED_REPRESENTATIVE: dict[float, dict[str, tuple[float, float]]] = {
    0.01: {
        "E_mix_with_EDL_V": (0.6249104327868923, 5.0e-11),
        "i_mix_avg_with_EDL_A_per_m2": (0.04315057812685645, 5.0e-12),
        "i_mix_abs_with_EDL_A": (1.7260231250742582e-12, 5.0e-22),
        "i_mix_avg_ratio_with_over_without": (0.3670504266936859, 5.0e-12),
    },
    1.0: {
        "E_mix_with_EDL_V": (0.5260483104771886, 5.0e-11),
        "i_mix_avg_with_EDL_A_per_m2": (0.09045095067661652, 5.0e-12),
        "i_mix_abs_with_EDL_A": (3.618038027064661e-12, 5.0e-22),
        "i_mix_avg_ratio_with_over_without": (0.7694001212011172, 5.0e-12),
    },
    10.0: {
        "E_mix_with_EDL_V": (0.4905802818826732, 5.0e-11),
        "i_mix_avg_with_EDL_A_per_m2": (0.10696013375585511, 5.0e-12),
        "i_mix_abs_with_EDL_A": (4.278405350234205e-12, 5.0e-22),
        "i_mix_avg_ratio_with_over_without": (0.9098316740712563, 5.0e-12),
    },
    1000.0: {
        "E_mix_with_EDL_V": (0.4696476642872442, 5.0e-11),
        "i_mix_avg_with_EDL_A_per_m2": (0.11638666360291064, 5.0e-12),
        "i_mix_abs_with_EDL_A": (4.655466544116426e-12, 5.0e-22),
        "i_mix_avg_ratio_with_over_without": (0.9900162730453549, 5.0e-12),
    },
}

EXPECTED_WITHOUT_EDL = {
    "E_mix_without_EDL_V": (0.46699999999999986, 5.0e-11),
    "i_mix_avg_without_EDL_A_per_m2": (0.11756035407872402, 5.0e-12),
    "i_mix_abs_without_EDL_A": (4.7024141631489615e-12, 5.0e-22),
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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    sources = {
        "study_wrapper": Path(__file__).resolve(),
        "parameter_wrapper": PROJECT_DIR / "make_independent_au_pd.py",
        "ctot_study": MODEL_SRC / "au_pd_independent_edls" / "ctot_study.py",
        "independent_model": MODEL_SRC / "au_pd_independent_edls" / "model.py",
    }
    return {name: _sha256(path) for name, path in sources.items()}


def _save_fixed_trend_canvas(
    fig: plt.Figure, directory: Path, stem: str
) -> list[Path]:
    """Save the two trend figures on one identical, explicitly sized canvas."""

    directory.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(*TREND_FIGSIZE_IN, forward=True)
    fig.subplots_adjust(**TREND_SUBPLOT_ADJUST)
    saved: list[Path] = []
    for suffix in ("png", "svg"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(
            path,
            dpi=ctot.DPI,
            bbox_inches=None,
            facecolor="white",
            edgecolor="none",
        )
        saved.append(path)
    plt.close(fig)
    return saved


def _save_fixed_profile_side_by_side_canvas(
    fig: plt.Figure,
    directory: Path,
    *,
    reference_width_px: int,
) -> list[Path]:
    """Save the horizontal Au/Pd scheme at the polarization PNG width."""

    directory.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(
        float(reference_width_px) / float(ctot.DPI),
        PROFILE_SCHEME_SIDE_BY_SIDE_HEIGHT_IN,
        forward=True,
    )
    fig.subplots_adjust(
        left=0.025,
        right=0.995,
        bottom=0.055,
        top=0.75,
        wspace=0.16,
    )
    saved: list[Path] = []
    for suffix in ("png", "svg"):
        path = directory / f"{PROFILE_SCHEME_SIDE_BY_SIDE_STEM}.{suffix}"
        fig.savefig(
            path,
            dpi=ctot.DPI,
            bbox_inches=None,
            facecolor="white",
            edgecolor="none",
        )
        saved.append(path)
    plt.close(fig)
    return saved


def _remove_formal_high_salt_text_annotation(ax: plt.Axes) -> None:
    """Remove only the 10^3 M text while retaining its marker and data point."""

    target = ctot.CASE_STYLES[1.0e3]["label"]
    matching = [artist for artist in ax.texts if artist.get_text() == target]
    if len(matching) != 1:
        raise RuntimeError(
            "Expected exactly one formal-high-salt trend annotation, "
            f"found {len(matching)}"
        )
    matching[0].remove()


def _target_plot_ctot_trends(
    rows: list[dict[str, Any]], output_dir: Path
) -> list[Path]:
    """Reference trend plots with a conclusion derived from the actual scan."""

    concentration = np.asarray([row["C_tot_M"] for row in rows], dtype=float)
    specifications = (
        {
            "stem": "ctot_emix_high_salt_regime_independent_edls",
            "with_key": "E_mix_with_EDL_V",
            "without_key": "E_mix_without_EDL_V",
            "display_scale": 1.0,
            "ylabel": r"$E_{\mathrm{mix}}$ (V vs. RHE)",
            "ylim": (0.40, 0.67),
            "offsets": {1.0e-2: -0.014, 1.0: 0.010, 1.0e3: 0.008},
            "legend_loc": "lower left",
            "legend_bbox": (0.01, 0.075),
        },
        {
            "stem": "ctot_imix_avg_high_salt_regime_independent_edls",
            "with_key": "i_mix_avg_with_EDL_A_per_m2",
            "without_key": "i_mix_avg_without_EDL_A_per_m2",
            "display_scale": IMIX_A_PER_M2_TO_MA_PER_CM2,
            "ylabel": r"$\bar{i}_{\mathrm{mix}}$ (mA/cm$^2$)",
            "ylim": IMIX_TREND_YLIM_MA_PER_CM2,
            "offsets": {1.0e-2: -0.0008, 1.0: 0.0006, 1.0e3: -0.0006},
            "legend_loc": "lower right",
            "legend_bbox": None,
        },
    )
    saved: list[Path] = []
    for specification in specifications:
        display_scale = float(specification["display_scale"])
        y_with = np.asarray(
            [row[str(specification["with_key"])] for row in rows], dtype=float
        ) * display_scale
        y_without = np.asarray(
            [row[str(specification["without_key"])] for row in rows], dtype=float
        ) * display_scale
        lower = float(min(np.min(y_with), np.min(y_without)))
        upper = float(max(np.max(y_with), np.max(y_without)))
        span = max(upper - lower, 1.0e-6)
        fig, ax = plt.subplots(figsize=TREND_FIGSIZE_IN)
        ctot._regime_background(ax)
        ctot._split_line(
            ax,
            concentration,
            y_with,
            color=ctot.COLORS["with_edl"],
            linewidth=2.25,
            linestyle="solid",
            label="with EDL",
            marker="o",
            markersize=2.8,
        )
        ctot._split_line(
            ax,
            concentration,
            y_without,
            color=ctot.COLORS["without_edl"],
            linewidth=1.8,
            linestyle=(0, (4.0, 2.5)),
            label="w/o EDL",
        )
        marker_key = str(specification["with_key"])
        marker_rows = [
            {
                **row,
                marker_key: float(row[marker_key]) * display_scale,
            }
            for row in rows
        ]
        ctot._representative_markers(
            ax,
            marker_rows,
            marker_key,
            specification["offsets"],
        )
        _remove_formal_high_salt_text_annotation(ax)
        ax.set_xscale("log")
        ax.set_xlim(1.0e-4, 1.0e3)
        if "ylim" in specification:
            ax.set_ylim(*specification["ylim"])
        else:
            ax.set_ylim(lower - 0.08 * span, upper + 0.15 * span)
        ax.set_xlabel(
            r"Electrolyte concentration, $C_{\mathrm{tot}}$ (M)", fontsize=8.8
        )
        ax.set_ylabel(str(specification["ylabel"]), fontsize=8.8)
        legend_kwargs: dict[str, Any] = {
            "loc": str(specification["legend_loc"]),
            "fontsize": 7.4,
            "handlelength": 2.4,
        }
        if specification["legend_bbox"] is not None:
            legend_kwargs["bbox_to_anchor"] = specification["legend_bbox"]
        ax.legend(**legend_kwargs)
        ax.tick_params(length=3.2, width=0.85, labelsize=8.0)
        saved.extend(
            _save_fixed_trend_canvas(fig, output_dir, str(specification["stem"]))
        )
    return saved


def _target_plot_polarization_overlay(
    polarization_rows: list[dict[str, Any]],
    cases: Mapping[float, Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Plot signed absolute half-reaction currents for the 2 nm electrodes."""

    fig, ax = plt.subplots(figsize=(7.2, 4.35))
    reference = [
        row
        for row in polarization_rows
        if row["condition"] == "w/o EDL reference"
    ]
    e_reference = np.asarray([row["E_V"] for row in reference], dtype=float)
    for key, color in (
        ("I_Au_1e_minus_3_uA", ctot.COLORS["au_curve"]),
        ("I_Pd_1e_minus_3_uA", ctot.COLORS["pd_curve"]),
    ):
        ax.plot(
            e_reference,
            np.asarray([row[key] for row in reference], dtype=float),
            color=color,
            linewidth=1.55,
            linestyle=(0, (4.0, 2.4)),
            alpha=0.30,
            zorder=1,
        )

    mixed_markers: list[float] = []
    for concentration_m in ctot.REPRESENTATIVE_CONCENTRATIONS_M:
        style = ctot.CASE_STYLES[concentration_m]
        subset = [
            row
            for row in polarization_rows
            if row["condition"] == "with EDL"
            and float(row["C_tot_M"]) == concentration_m
        ]
        potential = np.asarray([row["E_V"] for row in subset], dtype=float)
        alpha = 1.0 if concentration_m <= 1.0 else 0.42
        ax.plot(
            potential,
            np.asarray(
                [row["I_Au_1e_minus_3_uA"] for row in subset], dtype=float
            ),
            color=ctot.COLORS["au_curve"],
            linewidth=2.0,
            linestyle=style["linestyle"],
            alpha=alpha,
        )
        ax.plot(
            potential,
            np.asarray(
                [row["I_Pd_1e_minus_3_uA"] for row in subset], dtype=float
            ),
            color=ctot.COLORS["pd_curve"],
            linewidth=2.0,
            linestyle=style["linestyle"],
            alpha=alpha,
        )
        solution = cases[concentration_m]["solution"]
        e_mix = float(solution["E_mix_V"])
        i_mix = ctot.CURRENT_DISPLAY_SCALE * float(solution["i_mix_abs_A"])
        mixed_markers.append(i_mix)
        ax.axvline(e_mix, color=ctot.COLORS["gray"], linewidth=0.8, alpha=0.50)
        ax.scatter(
            [e_mix, e_mix],
            [i_mix, -i_mix],
            marker=str(style["marker"]),
            s=48,
            facecolor=ctot.COLORS["dark"],
            edgecolor="white",
            linewidth=0.7,
            alpha=alpha,
            zorder=5,
        )

    y_limit = max(0.008, 2.8 * max(mixed_markers))
    ax.axhline(0.0, color=ctot.COLORS["dark"], linewidth=0.9)
    ax.set_xlim(0.40, 0.64)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xlabel("Potential (V vs. RHE)")
    ax.set_ylabel(r"Current (10$^{-3}$ $\mu$A)")
    ax.set_title(
        r"Salt-dependent polarization curves explain $I_{\mathrm{mix}}$",
        loc="left",
        fontsize=10.2,
    )
    half_reaction_legend = ax.legend(
        handles=[
            Line2D(
                [0], [0], color=ctot.COLORS["au_curve"], lw=2.5, label="Au oxidation"
            ),
            Line2D(
                [0], [0], color=ctot.COLORS["pd_curve"], lw=2.5, label="Pd reduction"
            ),
            Line2D(
                [0],
                [0],
                color=ctot.COLORS["gray"],
                lw=1.7,
                ls=(0, (4, 2.4)),
                label="w/o EDL ref.",
            ),
        ],
        title="Half reaction",
        loc="upper left",
        fontsize=7.6,
        title_fontsize=7.8,
    )
    ax.add_artist(half_reaction_legend)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=ctot.COLORS["dark"],
                lw=1.8,
                ls=ctot.CASE_STYLES[concentration_m]["linestyle"],
                marker=ctot.CASE_STYLES[concentration_m]["marker"],
                label=ctot.CASE_STYLES[concentration_m]["label"],
                alpha=1.0 if concentration_m <= 1.0 else 0.42,
            )
            for concentration_m in ctot.REPRESENTATIVE_CONCENTRATIONS_M
        ],
        title=r"$C_{\mathrm{tot}}$",
        loc="upper right",
        fontsize=7.6,
        title_fontsize=7.8,
    )
    ax.tick_params(length=3.2, width=0.85, labelsize=8.1)
    return ctot._save_figure(
        fig,
        output_dir,
        "ctot_half_reaction_polarization_overlay_independent_edls",
    )


def _target_plot_profile_schematic(
    curves: Mapping[str, Mapping[float, Mapping[str, Any]]],
    distance_nm: np.ndarray,
    output_dir: Path,
    *,
    include_side_by_side: bool = False,
) -> list[Path]:
    """EDL profile schematic with distinct w/o-EDL and Phi=0 references."""

    def compact_tangent_start_tilde(curve_data: Mapping[str, Any]) -> float:
        phi = np.asarray(curve_data["phi_tilde"], dtype=float)
        lambda_d_nm = 1.0e9 * float(curve_data["model"].derived["lambda_D"])
        ohp_slope_per_nm = -float(phi[0]) / lambda_d_nm
        return float(phi[0]) + ohp_slope_per_nm * (
            ctot.COMPACT_LEFT_NM - ctot.RP_X_NM
        )

    curve_data_all = [
        curves[material][concentration_m]
        for material in ("Au", "Pd")
        for concentration_m in ctot.PROFILE_CONCENTRATIONS_M
    ]
    all_phi_tilde = np.concatenate(
        [
            np.asarray(curve_data["phi_tilde"], dtype=float)
            for curve_data in curve_data_all
        ]
        + [
            np.asarray([compact_tangent_start_tilde(curve_data)], dtype=float)
            for curve_data in curve_data_all
        ]
    )
    common_lower = min(float(np.min(all_phi_tilde)) * 1.10, -0.5)
    common_upper = 0.55
    def draw_profile_axis(
        ax: plt.Axes,
        material: str,
        *,
        show_ohp_label: bool,
    ) -> None:
        lower = common_lower
        upper = common_upper
        metal_color = ctot.COLORS["au"] if material == "Au" else ctot.COLORS["pd"]
        reaction_label = (
            r"$\mathrm{Red}_1^-$" if material == "Au" else r"$\mathrm{Ox}_2^+$"
        )
        reaction_color = (
            ctot.COLORS["with_edl"] if material == "Au" else "#D4A923"
        )
        ax.axvspan(
            ctot.METAL_LEFT_NM,
            ctot.COMPACT_LEFT_NM,
            color=metal_color,
            alpha=0.95,
            zorder=0,
        )
        ax.axvspan(
            ctot.COMPACT_LEFT_NM,
            ctot.RP_X_NM,
            color=ctot.COLORS["inner_layer"],
            alpha=1.0,
            zorder=0,
        )
        ax.axvspan(
            ctot.RP_X_NM,
            float(distance_nm[-1]),
            color=ctot.COLORS["electrolyte"],
            alpha=0.55,
            zorder=0,
        )
        ax.axvline(ctot.RP_X_NM, color=ctot.COLORS["dark"], linewidth=1.0, zorder=2)
        ax.axhline(
            0.0,
            color="black",
            linewidth=0.65,
            linestyle="solid",
            zorder=2,
        )
        for concentration_m, color in (
            (1.0e-2, ctot.COLORS["profile_low"]),
            (1.0, ctot.COLORS["profile_high"]),
        ):
            style = ctot.CASE_STYLES[concentration_m]
            curve_data = curves[material][concentration_m]
            curve = np.asarray(curve_data["phi_tilde"], dtype=float)
            ax.plot(
                [ctot.COMPACT_LEFT_NM, ctot.RP_X_NM],
                [compact_tangent_start_tilde(curve_data), float(curve[0])],
                color=color,
                linewidth=1.55,
                linestyle=style["linestyle"],
                zorder=3,
            )
            ax.plot(
                distance_nm,
                curve,
                color=color,
                linewidth=1.55,
                linestyle=style["linestyle"],
                label=f"with EDL, {style['label']}",
                zorder=3,
            )
            ax.scatter(
                [0.0],
                [curve[0]],
                s=24,
                color=color,
                edgecolor="white",
                linewidth=0.55,
                zorder=4,
            )
        # The w/o-EDL solution-phase curve is defined only from bulk to OHP/RP.
        # It ends at the reaction plane and does not extend into the compact layer.
        ax.plot(
            [ctot.RP_X_NM, float(distance_nm[-1])],
            [0.0, 0.0],
            color=ctot.COLORS["without_edl"],
            linewidth=1.45,
            linestyle=(0, (2.4, 1.25)),
            label="w/o EDL",
            zorder=6,
        )
        ax.set_ylim(lower, upper)
        ax.set_xlim(ctot.METAL_LEFT_NM, float(distance_nm[-1]))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(length=0)
        ax.text(
            0.5 * (ctot.METAL_LEFT_NM + ctot.COMPACT_LEFT_NM),
            0.50 * (lower + upper),
            material,
            ha="center",
            va="center",
            rotation=90,
            fontsize=9.0,
            fontweight="bold",
            color=ctot.COLORS["dark"],
        )
        if show_ohp_label:
            ax.text(
                ctot.RP_X_NM,
                upper + 0.035 * (upper - lower),
                "OHP/RP",
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
                color=ctot.COLORS["dark"],
                clip_on=False,
            )
        ax.text(
            0.74 * float(distance_nm[-1]),
            lower + 0.29 * (upper - lower),
            reaction_label,
            ha="center",
            va="center",
            fontsize=11.0,
            color=reaction_color,
        )

    vertical_fig, vertical_axes = plt.subplots(
        2,
        1,
        figsize=(4.15, 3.45),
        sharex=True,
        sharey=True,
    )
    for index, (ax, material) in enumerate(
        zip(vertical_axes, ("Au", "Pd"), strict=True)
    ):
        draw_profile_axis(ax, material, show_ohp_label=index == 0)
    vertical_handles, vertical_labels = vertical_axes[0].get_legend_handles_labels()
    vertical_fig.legend(
        vertical_handles,
        vertical_labels,
        loc="upper right",
        bbox_to_anchor=(0.965, 0.825),
        ncol=3,
        fontsize=6.1,
        handlelength=2.2,
        columnspacing=1.15,
    )
    vertical_fig.subplots_adjust(
        left=0.045,
        right=0.99,
        bottom=0.045,
        top=0.96,
        hspace=0.30,
    )
    saved = ctot._save_figure(
        vertical_fig,
        output_dir,
        PROFILE_SCHEME_STEM,
    )
    if not include_side_by_side:
        return saved

    polarization_png = (
        output_dir.parent
        / "Figure_4"
        / f"{POLARIZATION_OVERLAY_STEM}.png"
    )
    if not polarization_png.is_file():
        raise RuntimeError(
            "Cannot size the horizontal EDL scheme because the polarization "
            f"reference PNG is missing: {polarization_png}"
        )
    with Image.open(polarization_png) as reference_image:
        reference_width_px = int(reference_image.width)

    horizontal_fig, horizontal_axes = plt.subplots(
        1,
        2,
        figsize=(
            float(reference_width_px) / float(ctot.DPI),
            PROFILE_SCHEME_SIDE_BY_SIDE_HEIGHT_IN,
        ),
        sharex=True,
        sharey=True,
    )
    for ax, material in zip(horizontal_axes, ("Au", "Pd"), strict=True):
        draw_profile_axis(ax, material, show_ohp_label=True)
    horizontal_handles, horizontal_labels = horizontal_axes[0].get_legend_handles_labels()
    horizontal_fig.legend(
        horizontal_handles,
        horizontal_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=3,
        fontsize=7.0,
        handlelength=2.2,
        columnspacing=1.35,
    )
    saved.extend(
        _save_fixed_profile_side_by_side_canvas(
            horizontal_fig,
            output_dir,
            reference_width_px=reference_width_px,
        )
    )
    return saved


def _register_additional_scheme_artifacts(
    output: Path,
    paths: list[Path],
) -> None:
    """Register wrapper-owned scheme variants in the upstream artifact index."""

    artifact_path = output / "artifacts.json"
    artifacts = _read_json(artifact_path)
    scheme_entries = set(str(value) for value in artifacts["EDL_scheme"])
    for path in paths:
        relative = path.relative_to(output).as_posix()
        scheme_entries.add(relative)
    artifacts["EDL_scheme"] = sorted(scheme_entries)
    _write_json(artifact_path, artifacts)


def _configure_upstream() -> None:
    ctot.DPI = 600
    ctot.COLORS.update(PALETTE)
    ctot.plot_ctot_trends = _target_plot_ctot_trends
    ctot.plot_polarization_overlay = _target_plot_polarization_overlay
    ctot.plot_profile_schematic = _target_plot_profile_schematic


def _scan_checks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_grid = ctot.concentration_scan_values_m()
    actual_grid = np.asarray([float(row["C_tot_M"]) for row in rows], dtype=float)
    checks: dict[str, Any] = {
        "scan_row_count": len(rows),
        "scan_grid_exact": bool(
            actual_grid.shape == expected_grid.shape
            and np.allclose(actual_grid, expected_grid, rtol=0.0, atol=1.0e-13)
        ),
        "representative_regression": {},
    }
    representative_passed = True
    for concentration_m, fields in EXPECTED_REPRESENTATIVE.items():
        row = ctot.row_at(rows, concentration_m)
        case_checks: dict[str, Any] = {}
        for field, (expected, tolerance) in fields.items():
            actual = float(row[field])
            passed = abs(actual - expected) <= tolerance
            case_checks[field] = {
                "actual": actual,
                "expected": expected,
                "absolute_difference": abs(actual - expected),
                "absolute_tolerance": tolerance,
                "passed": passed,
            }
            representative_passed = representative_passed and passed
        checks["representative_regression"][f"{concentration_m:g}_M"] = case_checks

    without_checks: dict[str, Any] = {}
    without_passed = True
    for field, (expected, tolerance) in EXPECTED_WITHOUT_EDL.items():
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
        max_error = float(np.max(np.abs(values - expected)))
        passed = max_error <= tolerance and float(np.ptp(values)) < 1.0e-12
        without_checks[field] = {
            "expected": expected,
            "max_absolute_difference": max_error,
            "absolute_tolerance": tolerance,
            "variation": float(np.ptp(values)),
            "passed": passed,
        }
        without_passed = without_passed and passed
    checks["without_edl_regression"] = without_checks

    e_with = np.asarray([float(row["E_mix_with_EDL_V"]) for row in rows])
    i_with = np.asarray(
        [float(row["i_mix_avg_with_EDL_A_per_m2"]) for row in rows]
    )
    ratio = np.asarray(
        [float(row["i_mix_avg_ratio_with_over_without"]) for row in rows]
    )
    checks["target_trend"] = {
        "E_mix_strictly_decreases": bool(np.all(np.diff(e_with) < 0.0)),
        "i_mix_avg_strictly_increases": bool(np.all(np.diff(i_with) > 0.0)),
        "ratio_min": float(np.min(ratio)),
        "ratio_max": float(np.max(ratio)),
        "ratio_remains_below_one": bool(np.all(ratio < 1.0)),
        "unity_crossings_M": ctot._log_interpolated_crossing(
            rows, "i_mix_avg_ratio_with_over_without", 1.0
        ),
        "interpretation": "with-EDL current approaches the w/o-EDL limit from below",
    }
    checks["g_Au_equals_g_Pd"] = bool(
        np.allclose(
            [float(row["g_Au"]) for row in rows],
            [float(row["g_Pd"]) for row in rows],
            rtol=0.0,
            atol=1.0e-14,
        )
    )
    checks["passed"] = bool(
        len(rows) == 36
        and checks["scan_grid_exact"]
        and representative_passed
        and without_passed
        and checks["target_trend"]["E_mix_strictly_decreases"]
        and checks["target_trend"]["i_mix_avg_strictly_increases"]
        and checks["target_trend"]["ratio_remains_below_one"]
        and not checks["target_trend"]["unity_crossings_M"]
        and checks["g_Au_equals_g_Pd"]
    )
    return checks


def _debye_huckel_caveat(rows: list[dict[str, Any]]) -> dict[str, Any]:
    threshold = 1.0
    maxima = np.asarray(
        [
            max(abs(float(row["Au_phi_RP_tilde"])), abs(float(row["Pd_phi_RP_tilde"])))
            for row in rows
        ],
        dtype=float,
    )
    concentrations = np.asarray([float(row["C_tot_M"]) for row in rows], dtype=float)
    within = concentrations[maxima <= threshold]
    representative = {
        f"{value:g}_M": float(
            max(
                abs(float(ctot.row_at(rows, value)["Au_phi_RP_tilde"])),
                abs(float(ctot.row_at(rows, value)["Pd_phi_RP_tilde"])),
            )
        )
        for value in (0.01, 1.0, 10.0, 1000.0)
    }
    return {
        "threshold_max_abs_phi_tilde": threshold,
        "maximum_over_scan": float(np.max(maxima)),
        "threshold_exceeded": bool(np.any(maxima > threshold)),
        "scan_points_exceeding_threshold": int(np.count_nonzero(maxima > threshold)),
        "scan_points_total": int(len(rows)),
        "first_sampled_concentration_within_threshold_M": (
            float(within[0]) if within.size else None
        ),
        "representative_max_abs_phi_tilde": representative,
        "included_in_numerical_pass_fail": False,
        "interpretation": (
            "The numerical solution passed its internal identities, but the linear "
            "Debye-Huckel approximation is outside its nominal weak-potential range "
            "for the flagged scan points."
        ),
    }


def _artifact_checks(output: Path) -> dict[str, Any]:
    pngs = sorted(output.glob("figures/**/*.png"))
    svgs = sorted(output.glob("figures/**/*.svg"))
    pdfs = sorted(output.glob("**/*.pdf"))
    png_dpi: dict[str, list[float] | None] = {}
    png_dimensions: dict[str, list[int] | None] = {}
    png_decodable = True
    for path in pngs:
        try:
            with Image.open(path) as image:
                dpi_value = image.info.get("dpi")
                png_dpi[path.relative_to(output).as_posix()] = (
                    [float(dpi_value[0]), float(dpi_value[1])]
                    if dpi_value is not None
                    else None
                )
                png_dimensions[path.relative_to(output).as_posix()] = [
                    int(image.width),
                    int(image.height),
                ]
                image.verify()
        except Exception:
            png_decodable = False
            png_dpi[path.relative_to(output).as_posix()] = None
            png_dimensions[path.relative_to(output).as_posix()] = None
    dpi_passed = all(
        value is not None
        and abs(value[0] - 600.0) < 1.0
        and abs(value[1] - 600.0) < 1.0
        for value in png_dpi.values()
    )
    svg_parseable = True
    svg_editable = True
    svg_helvetica_first = True
    for path in svgs:
        text = path.read_text(encoding="utf-8")
        try:
            ET.parse(path)
        except ET.ParseError:
            svg_parseable = False
        svg_editable = svg_editable and "<text" in text
        svg_helvetica_first = svg_helvetica_first and "Helvetica" in text
    stems_png = {path.relative_to(output).with_suffix("").as_posix() for path in pngs}
    stems_svg = {path.relative_to(output).with_suffix("").as_posix() for path in svgs}
    trend_stems = (
        "figures/Figure_4/ctot_emix_high_salt_regime_independent_edls.png",
        "figures/Figure_4/ctot_imix_avg_high_salt_regime_independent_edls.png",
    )
    trend_canvas_expected = list(TREND_CANVAS_PX_AT_600_DPI)
    trend_canvases_match = all(
        png_dimensions.get(relative) == trend_canvas_expected
        for relative in trend_stems
    )
    polarization_relative = (
        f"figures/Figure_4/{POLARIZATION_OVERLAY_STEM}.png"
    )
    horizontal_scheme_relative = (
        f"figures/EDL_scheme/{PROFILE_SCHEME_SIDE_BY_SIDE_STEM}.png"
    )
    polarization_dimensions = png_dimensions.get(polarization_relative)
    horizontal_scheme_dimensions = png_dimensions.get(horizontal_scheme_relative)
    horizontal_scheme_width_matches_polarization = bool(
        polarization_dimensions is not None
        and horizontal_scheme_dimensions is not None
        and horizontal_scheme_dimensions[0] == polarization_dimensions[0]
        and horizontal_scheme_dimensions[1]
        == PROFILE_SCHEME_SIDE_BY_SIDE_HEIGHT_PX_AT_600_DPI
    )
    checks = {
        "figure4_pair_count": len(list((output / "figures" / "Figure_4").glob("*.png"))),
        "mechanism_pair_count": len(list((output / "figures" / "Mechanism").glob("*.png"))),
        "edl_scheme_pair_count": len(list((output / "figures" / "EDL_scheme").glob("*.png"))),
        "png_count": len(pngs),
        "svg_count": len(svgs),
        "pdf_count": len(pdfs),
        "all_files_nonempty": all(path.stat().st_size > 0 for path in (*pngs, *svgs)),
        "png_decodable": png_decodable,
        "png_dpi": png_dpi,
        "png_dimensions_px": png_dimensions,
        "trend_canvas_expected_px": trend_canvas_expected,
        "trend_canvases_identical_and_expected": trend_canvases_match,
        "polarization_overlay_dimensions_px": polarization_dimensions,
        "horizontal_scheme_dimensions_px": horizontal_scheme_dimensions,
        "horizontal_scheme_width_matches_polarization": (
            horizontal_scheme_width_matches_polarization
        ),
        "all_png_600_dpi": dpi_passed,
        "svg_parseable": svg_parseable,
        "svg_text_editable": svg_editable,
        "svg_helvetica_first": svg_helvetica_first,
        "png_svg_stems_match": stems_png == stems_svg,
        "no_delivery_pycache": not list(output.glob("**/__pycache__")),
    }
    checks["passed"] = bool(
        checks["figure4_pair_count"] == 3
        and checks["mechanism_pair_count"] == 1
        and checks["edl_scheme_pair_count"] == 2
        and len(pngs) == EXPECTED_FIGURE_PAIRS
        and len(svgs) == EXPECTED_FIGURE_PAIRS
        and not pdfs
        and checks["all_files_nonempty"]
        and png_decodable
        and dpi_passed
        and trend_canvases_match
        and horizontal_scheme_width_matches_polarization
        and svg_parseable
        and svg_editable
        and svg_helvetica_first
        and stems_png == stems_svg
        and checks["no_delivery_pycache"]
    )
    return checks


def _refresh_artifacts(output: Path) -> None:
    path = output / "artifacts.json"
    artifacts = _read_json(path)
    registered = sorted(
        {
            relative
            for category in ("Figure_4", "Mechanism", "EDL_scheme", "csv", "metadata")
            for relative in artifacts[category]
            if relative != "artifacts.json"
        }
    )
    for relative in registered:
        target = output / relative
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(f"Registered artifact is missing or empty: {relative}")
    artifacts["sha256"] = {relative: _sha256(output / relative) for relative in registered}
    artifacts["size_bytes"] = {
        relative: (output / relative).stat().st_size for relative in registered
    }
    artifacts["project_checksums"] = CHECKSUM_FILE
    _write_json(path, artifacts)


def _write_checksums(output: Path) -> dict[str, str]:
    paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name not in CHECKSUM_IGNORED_FILENAMES
    )
    checksums = {
        path.relative_to(output).as_posix(): _sha256(path)
        for path in paths
    }
    (output / CHECKSUM_FILE).write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in checksums.items()),
        encoding="utf-8",
    )
    return checksums


def _verify_checksums(output: Path) -> dict[str, str]:
    path = output / CHECKSUM_FILE
    if not path.is_file():
        raise RuntimeError(f"Missing {CHECKSUM_FILE}")
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if relative in checksums:
            raise RuntimeError(f"Duplicate checksum path: {relative}")
        target = output / relative
        if not target.is_file() or _sha256(target) != expected:
            raise RuntimeError(f"Checksum mismatch: {relative}")
        checksums[relative] = expected
    actual = {
        item.relative_to(output).as_posix()
        for item in output.rglob("*")
        if item.is_file() and item.name not in CHECKSUM_IGNORED_FILENAMES
    }
    if set(checksums) != actual:
        raise RuntimeError("C_tot study checksum coverage is incomplete")
    return checksums


def _verify_artifacts(output: Path) -> None:
    artifacts = _read_json(output / "artifacts.json")
    if set(artifacts["sha256"]) != set(artifacts["size_bytes"]):
        raise RuntimeError("artifacts.json hash and size registries differ")
    for relative, expected in artifacts["sha256"].items():
        target = output / relative
        if _sha256(target) != expected:
            raise RuntimeError(f"artifacts.json checksum mismatch: {relative}")
        if target.stat().st_size != int(artifacts["size_bytes"][relative]):
            raise RuntimeError(f"artifacts.json size mismatch: {relative}")


def _read_scan_csv(output: Path) -> list[dict[str, Any]]:
    with (output / "csv" / "ctot_scan.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _locked_parameter_checks(params: Mapping[str, Any]) -> dict[str, Any]:
    expected: dict[str, float | int | None] = {
        "L_Au": 2.0e-9,
        "L_Pd": 2.0e-9,
        "C_H_Au": 0.50,
        "C_H_Pd": 0.50,
        "active_faces_Au": 1,
        "active_faces_Pd": 1,
        "C_tot": 10.0,
        "it0_1": 1.852573885166257e-4,
        "it0_2": 1.852573885166257e-4,
        "alpha1": 0.5,
        "alpha2": 0.5,
        "out_of_plane_width": 0.01,
        "epsilon_s": None,
        "lambda_D": None,
        "g_Au": None,
        "g_Pd": None,
    }
    fields: dict[str, Any] = {}
    passed = True
    for key, expected_value in expected.items():
        actual = params.get(key)
        if expected_value is None:
            field_passed = actual is None
        else:
            field_passed = math.isclose(
                float(actual),
                float(expected_value),
                rel_tol=0.0,
                abs_tol=max(1.0e-15, abs(float(expected_value)) * 1.0e-12),
            )
        fields[key] = {"actual": actual, "expected": expected_value, "passed": field_passed}
        passed = passed and field_passed
    return {"fields": fields, "passed": passed}


def _augment_result(output: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    scan_checks = _scan_checks(rows)
    artifact_checks = _artifact_checks(output)
    params = _read_json(output / "params.json")
    parameter_checks = _locked_parameter_checks(params)
    caveat = _debye_huckel_caveat(rows)

    validation_path = output / "validation.json"
    validation = _read_json(validation_path)
    upstream_passed = bool(validation.get("passed"))
    validation.update(
        {
            "study_id": STUDY_ID,
            "upstream_numerical_validation_passed": upstream_passed,
            "locked_parameter_checks": parameter_checks,
            "target_scan_checks": scan_checks,
            "artifact_checks": artifact_checks,
            "debye_huckel_applicability_caveat": caveat,
            "publication_figure_semantics": {
                "emix_y_axis_V": [0.40, 0.67],
                "trend_figsize_inches": list(TREND_FIGSIZE_IN),
                "trend_canvas_px_at_600_dpi": list(TREND_CANVAS_PX_AT_600_DPI),
                "emix_and_imix_trend_canvases_identical": True,
                "trend_export_bbox_inches": None,
                "trend_export_uses_fixed_canvas": True,
                "formal_high_salt_marker_visible": True,
                "formal_high_salt_text_annotation_visible": False,
                "trend_explanatory_titles_visible": False,
                "imix_trend_source_unit": "A/m^2",
                "imix_trend_display_unit": "mA/cm^2",
                "imix_A_per_m2_to_mA_per_cm2": IMIX_A_PER_M2_TO_MA_PER_CM2,
                "imix_y_axis_mA_per_cm2": list(IMIX_TREND_YLIM_MA_PER_CM2),
                "polarization_source_currents_are_signed": True,
                "polarization_display_is_magnitude": False,
                "polarization_quantity": "signed absolute half-reaction current",
                "polarization_normalization": "none",
                "polarization_display_unit": "10^-3 uA",
                "polarization_y_axis": "signed and symmetric about zero",
                "polarization_y_label": "Current (10^-3 uA)",
                "without_edl_profile_segment": "bulk solution to OHP/RP only",
                "phi_zero_reference": "thin black solid line",
                "profile_side_by_side_layout": "Au and Pd in one horizontal row",
                "profile_side_by_side_canvas_px": artifact_checks[
                    "horizontal_scheme_dimensions_px"
                ],
                "profile_side_by_side_width_reference": (
                    "ctot_half_reaction_polarization_overlay_independent_edls.png"
                ),
                "profile_side_by_side_export_bbox_inches": None,
            },
            "topology_scope_caveat": (
                "The 2 nm lengths set active areas only. The analytic model uses two "
                "independent infinite planar half-spaces and does not resolve finite-size "
                "edge fields or lateral EDL overlap."
            ),
            "passed": bool(
                upstream_passed
                and parameter_checks["passed"]
                and scan_checks["passed"]
                and artifact_checks["passed"]
            ),
        }
    )
    _write_json(validation_path, validation)
    if not validation["passed"]:
        raise RuntimeError(f"Target C_tot validation failed: {validation}")

    summary_path = output / "summary.json"
    summary = _read_json(summary_path)
    summary.update(
        {
            "study_id": STUDY_ID,
            "fixed_parameters": {
                "L_Au_nm": 2.0,
                "L_Pd_nm": 2.0,
                "C_H_Au_uF_per_cm2": 50.0,
                "C_H_Pd_uF_per_cm2": 50.0,
                "active_faces_Au": 1,
                "active_faces_Pd": 1,
            },
            "current_trend_interpretation": scan_checks["target_trend"]["interpretation"],
            "publication_figure_quantity_definitions": {
                "imix_trend_y": (
                    "i_mix_avg_A_per_m2 multiplied by 0.1 and reported in mA/cm^2; "
                    "the displayed y axis begins at zero"
                ),
                "polarization_y": (
                    "Signed absolute half-reaction currents I_Au and I_Pd, "
                    "reported in 10^-3 uA"
                ),
                "polarization_mixed_marker_y": (
                    "+/- i_mix_abs_A multiplied by 10^9 to report 10^-3 uA"
                ),
                "polarization_csv_source": (
                    "csv/ctot_polarization_curves.csv signed "
                    "I_Au_1e_minus_3_uA and I_Pd_1e_minus_3_uA"
                ),
                "profile_side_by_side": (
                    "Same analytic Au/Pd profiles as the vertical scheme, arranged "
                    "as a 1x2 figure whose fixed PNG width matches the half-reaction "
                    "polarization overlay"
                ),
            },
        }
    )
    _write_json(summary_path, summary)

    scan_config_path = output / "scan_config.json"
    scan_config = _read_json(scan_config_path)
    scan_config.update(
        {
            "target_study_id": STUDY_ID,
            "reference_result_for_layout_and_method_only": REFERENCE_RESULT,
            "target_figure_adaptations": {
                "current_trend_interpretation_is_data_dependent": True,
                "trend_explanatory_titles_visible": False,
                "emix_y_axis_V": [0.40, 0.67],
                "trend_figsize_inches": list(TREND_FIGSIZE_IN),
                "trend_canvas_px_at_600_dpi": list(TREND_CANVAS_PX_AT_600_DPI),
                "trend_canvases_fixed_and_identical": True,
                "trend_export_bbox_inches": None,
                "formal_high_salt_marker_visible": True,
                "formal_high_salt_text_annotation_visible": False,
                "imix_trend_display_unit": "mA/cm^2",
                "imix_trend_unit_conversion": "1 A/m^2 = 0.1 mA/cm^2",
                "imix_trend_y_axis_mA_per_cm2": list(IMIX_TREND_YLIM_MA_PER_CM2),
                "polarization_y_axis_uses_signed_absolute_current": True,
                "polarization_normalization": "none",
                "polarization_display_unit": "10^-3 uA",
                "polarization_y_axis_is_symmetric_about_zero": True,
                "polarization_y_label": "Current (10^-3 uA)",
                "without_edl_profile_dense_dash": [2.4, 1.25],
                "without_edl_profile_domain": "OHP/RP to bulk solution",
                "phi_zero_reference_style": "thin black solid line",
                "mechanism_Au_Pd_corresponding_metrics_share_y_scale": True,
                "profile_side_by_side_layout": "1x2 Au/Pd",
                "profile_side_by_side_height_inches": (
                    PROFILE_SCHEME_SIDE_BY_SIDE_HEIGHT_IN
                ),
                "profile_side_by_side_width_reference": POLARIZATION_OVERLAY_STEM,
                "profile_side_by_side_export_bbox_inches": None,
                "dpi_png": 600,
                "palette": PALETTE,
            },
        }
    )
    _write_json(scan_config_path, scan_config)

    manifest_path = output / "run_manifest.json"
    manifest = _read_json(manifest_path)
    manifest.update(
        {
            "study_id": STUDY_ID,
            "parent_study_id": "independent_Au2nm_Pd2nm_CH50",
            "study_parameters": {
                "L_Au_nm": 2.0,
                "L_Pd_nm": 2.0,
                "C_H_Au_uF_per_cm2": 50.0,
                "C_H_Pd_uF_per_cm2": 50.0,
                "active_faces_Au": 1,
                "active_faces_Pd": 1,
                "equal_i0_A_per_m2": float(params["it0_1"]),
                "alpha1": 0.5,
                "alpha2": 0.5,
                "out_of_plane_width_m": 0.01,
            },
            "parameter_provenance": (
                "Figures/Figure_Au2nm_Pd2nm/make_independent_au_pd.py::"
                "independent_params"
            ),
            "reference_result_for_layout_and_method_only": REFERENCE_RESULT,
            "source_sha256": _source_hashes(),
            "checksums_file": CHECKSUM_FILE,
            "export": {
                "dpi_png": 600,
                "formats": ["png", "svg"],
                "svg_fonttype": "none",
                "pdf_enabled": False,
            },
            "target_result_interpretation": (
                "For C_H,Au=C_H,Pd=50 uF/cm^2, i_mix with EDL remains below "
                "the w/o-EDL value over the sampled grid and approaches it from below."
            ),
            "formal_high_salt_note": (
                "10-1000 M is retained only as a mathematical high-salt extension; "
                "10^3 M is not a physically realizable electrolyte concentration."
            ),
            "publication_figure_adjustments": {
                "ctot_emix_high_salt_regime": (
                    "E_mix axis fixed to 0.40-0.67 V on the shared 3 x 3.45 in "
                    "fixed canvas; bbox_inches=None; the 10^3 M endpoint remains "
                    "but its text annotation is omitted; no explanatory title"
                ),
                "ctot_imix_avg_high_salt_regime": (
                    "i_mix converted from A/m^2 to mA/cm^2, y axis fixed to "
                    "0-0.013 mA/cm^2, on the same 3 x 3.45 in fixed canvas as "
                    "E_mix; bbox_inches=None; the 10^3 M endpoint remains but its "
                    "text annotation is omitted; no explanatory title"
                ),
                "ctot_half_reaction_polarization_overlay": (
                    "signed absolute Au/Pd half-reaction currents in 10^-3 uA; "
                    "y axis is symmetric about zero and labeled Current"
                ),
                "ctot_phi_bar_profiles": (
                    "denser w/o-EDL dash from bulk to OHP/RP, plus a thin black "
                    "Phi=0 reference; retain the original 2x1 version and add a "
                    "fixed-canvas 1x2 Au/Pd version whose PNG width matches the "
                    "half-reaction polarization overlay"
                ),
            },
            "validation_passed": True,
        }
    )
    _write_json(manifest_path, manifest)

    _refresh_artifacts(output)
    checksums = _write_checksums(output)
    _verify_artifacts(output)
    _verify_checksums(output)
    return {
        "output": str(output),
        "summary": summary,
        "validation": validation,
        "manifest": manifest,
        "checksums": checksums,
    }


def build_independent_au_pd_ctot_study(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if output == DEFAULT_OUTPUT.resolve() and not (INDEPENDENT_DIR / "params.json").is_file():
        raise RuntimeError(
            "Build the independent Au/Pd baseline first; refusing to create an incomplete "
            f"parent result at {INDEPENDENT_DIR}"
        )
    params = independent_params()
    parameter_checks = _locked_parameter_checks(params)
    if not parameter_checks["passed"]:
        raise RuntimeError(f"Target parameter lock failed: {parameter_checks}")
    _configure_upstream()
    preview_rows = ctot.compute_scan_rows(params)
    preview_checks = _scan_checks(preview_rows)
    if not preview_checks["passed"]:
        raise RuntimeError(f"Target C_tot preview regression failed: {preview_checks}")
    ctot.build_ctot_results(params, output)
    _, profile_curves, distance_nm = ctot.compute_profile_rows(params)
    with plt.rc_context(ctot.RC):
        profile_paths = _target_plot_profile_schematic(
            profile_curves,
            distance_nm,
            output / "figures" / "EDL_scheme",
            include_side_by_side=True,
        )
    side_by_side_paths = [
        path
        for path in profile_paths
        if path.stem == PROFILE_SCHEME_SIDE_BY_SIDE_STEM
    ]
    if len(side_by_side_paths) != 2:
        raise RuntimeError(
            "Expected one PNG/SVG pair for the horizontal EDL scheme, "
            f"got {side_by_side_paths}"
        )
    _register_additional_scheme_artifacts(output, side_by_side_paths)
    rows = _read_scan_csv(output)
    return _augment_result(output, rows)


def validate_existing_independent_au_pd_ctot_study(
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    if not output.is_dir():
        raise FileNotFoundError(f"Missing C_tot study output: {output}")
    params = _read_json(output / "params.json")
    parameter_checks = _locked_parameter_checks(params)
    rows = _read_scan_csv(output)
    scan_checks = _scan_checks(rows)
    artifact_checks = _artifact_checks(output)
    validation = _read_json(output / "validation.json")
    manifest = _read_json(output / "run_manifest.json")
    if not parameter_checks["passed"]:
        raise RuntimeError("Existing C_tot study parameter lock failed")
    if not scan_checks["passed"]:
        raise RuntimeError("Existing C_tot study scan regression failed")
    if not artifact_checks["passed"]:
        raise RuntimeError("Existing C_tot study artifacts failed validation")
    if validation.get("study_id") != STUDY_ID or not bool(validation.get("passed")):
        raise RuntimeError("Existing C_tot study validation metadata is invalid")
    if manifest.get("source_sha256") != _source_hashes():
        raise RuntimeError("Existing C_tot study source hashes do not match current code")
    _verify_artifacts(output)
    checksums = _verify_checksums(output)
    return {
        "output": str(output),
        "reused": True,
        "validation": validation,
        "manifest": manifest,
        "checksums": checksums,
    }


def build_or_reuse_independent_au_pd_ctot_study(
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        return validate_existing_independent_au_pd_ctot_study(output)
    return build_independent_au_pd_ctot_study(output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="New or reusable output directory (default: %(default)s)",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = build_or_reuse_independent_au_pd_ctot_study(args.output)
    summary = (
        result["summary"]
        if "summary" in result
        else _read_json(Path(result["output"]) / "summary.json")
    )
    print(f"output = {result['output']}")
    for label in ("0.01 M", "1 M", "$10^3$ M"):
        row = summary["representative_cases"][label]
        print(
            f"{label}: E_mix={float(row['E_mix_with_EDL_V']):.12f} V, "
            f"i_mix_avg={float(row['i_mix_avg_with_EDL_A_per_m2']):.12g} A/m^2, "
            f"I_mix={float(row['i_mix_abs_with_EDL_A']):.12g} A"
        )
    print("figures = 6 PNG + 6 SVG; PDF = 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
