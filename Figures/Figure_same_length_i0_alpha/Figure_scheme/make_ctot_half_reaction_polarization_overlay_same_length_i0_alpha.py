from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]
SAME_LENGTH_I0_ALPHA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SAME_LENGTH_I0_ALPHA_DIR))

import same_length_i0_alpha_common as common  # noqa: E402
from Figures.Figure_scheme.make_edl_vs_no_edl_bv_mixed_potential import (  # noqa: E402
    PALETTE,
    assert_mixed_current,
    compute_signed_half_currents,
    current_at_mix,
    fmt_v,
    solver,
)
from Figures.Figure_same_length_i0_alpha.Figure_scheme.make_emix_up_imix_down_schematic_same_length_i0_alpha import (  # noqa: E402
    CURRENT_UNIT,
    apply_style,
    current_density_to_display_current,
)


OUT_PATH = (
    common.FIGURE_SCHEME_DIR
    / f"ctot_half_reaction_polarization_overlay_{common.OUTPUT_TAG}.png"
)
REPRESENTATIVE_C_M = (1.0e-2, 1.0, 1.0e3)
E_MIN = 0.40
E_MAX = 0.63
Y_LIMIT = 0.16

CASE_STYLES = {
    1.0e-2: {"label": "0.01 M", "linestyle": "solid", "marker": "o"},
    1.0: {"label": "1 M", "linestyle": (0, (5.0, 2.0, 1.2, 2.0)), "marker": "s"},
    1.0e3: {"label": r"$10^3$ M", "linestyle": (0, (1.0, 2.0)), "marker": "^"},
}
NO_EDL_STYLE = (0, (4.0, 2.6))


def case_params(base_params: dict[str, Any], c_m: float) -> dict[str, Any]:
    params = copy.deepcopy(base_params)
    params["C_tot"] = solver.concentration_M_to_mol_per_m3(float(c_m))
    params["lambda_D"] = None
    return params


def compute_case(base_params: dict[str, Any], c_m: float, e_values: np.ndarray) -> dict[str, Any]:
    params = case_params(base_params, c_m)
    pair = solver.run_edl_comparison_pair(params, mode="FULL")
    with_edl = pair["with_edl"]
    no_edl = pair["no_edl"]

    use_affine_phi2 = bool(params.get("use_affine_phi2", True))
    lambda_d = float(with_edl["lambda_D"])

    au_with, pd_with = compute_signed_half_currents(
        params, lambda_d, e_values, use_edl=True, use_affine_phi2=use_affine_phi2
    )
    au_no, pd_no = compute_signed_half_currents(
        params, lambda_d, e_values, use_edl=False, use_affine_phi2=use_affine_phi2
    )
    i_au_with, i_pd_with = current_at_mix(
        params, lambda_d, float(with_edl["E_mix"]), use_edl=True, use_affine_phi2=use_affine_phi2
    )
    i_au_no, i_pd_no = current_at_mix(
        params, lambda_d, float(no_edl["E_mix"]), use_edl=False, use_affine_phi2=use_affine_phi2
    )
    assert_mixed_current("with EDL", i_au_with, i_pd_with, float(with_edl["i_mix_avg_A_per_m2"]))
    assert_mixed_current("w/o EDL", i_au_no, i_pd_no, float(no_edl["i_mix_avg_A_per_m2"]))

    return {
        "C_M": float(c_m),
        "with_edl": with_edl,
        "no_edl": no_edl,
        "au_with": current_density_to_display_current(params, au_with),
        "pd_with": current_density_to_display_current(params, pd_with),
        "au_no": current_density_to_display_current(params, au_no),
        "pd_no": current_density_to_display_current(params, pd_no),
        "i_au_with": float(current_density_to_display_current(params, i_au_with)),
        "i_pd_with": float(current_density_to_display_current(params, i_pd_with)),
        "i_au_no": float(current_density_to_display_current(params, i_au_no)),
        "i_pd_no": float(current_density_to_display_current(params, i_pd_no)),
        "E_with": float(with_edl["E_mix"]),
        "E_no": float(no_edl["E_mix"]),
        "I_with": float(with_edl["i_mix_abs_A"]) * 1.0e9,
        "I_no": float(no_edl["i_mix_abs_A"]) * 1.0e9,
        "I_ratio": float(with_edl["i_mix_abs_A"]) / float(no_edl["i_mix_abs_A"]),
        "K_Au_ratio": float(with_edl["K_Au"]) / float(no_edl["K_Au"]),
        "K_Pd_ratio": float(with_edl["K_Pd"]) / float(no_edl["K_Pd"]),
        "delta_E_mV": 1000.0 * (float(with_edl["E_mix"]) - float(no_edl["E_mix"])),
    }


def compute_cases(base_params: dict[str, Any], e_values: np.ndarray) -> list[dict[str, Any]]:
    cases = [compute_case(base_params, c_m, e_values) for c_m in REPRESENTATIVE_C_M]
    ref = cases[0]
    for case in cases[1:]:
        if not math.isclose(case["E_no"], ref["E_no"], rel_tol=0.0, abs_tol=1.0e-10):
            raise ValueError(f"w/o EDL E_mix changed with C_tot: {case['E_no']:.12g} vs {ref['E_no']:.12g}")
        if not math.isclose(case["I_no"], ref["I_no"], rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(f"w/o EDL I_mix changed with C_tot: {case['I_no']:.12g} vs {ref['I_no']:.12g}")
    return cases


def plot_reference_curves(ax: plt.Axes, e_values: np.ndarray, ref_case: dict[str, Any]) -> None:
    ax.plot(
        e_values,
        ref_case["au_no"],
        color=PALETTE["green"],
        linewidth=1.55,
        linestyle=NO_EDL_STYLE,
        alpha=0.38,
        zorder=1,
    )
    ax.plot(
        e_values,
        ref_case["pd_no"],
        color=PALETTE["orange"],
        linewidth=1.55,
        linestyle=NO_EDL_STYLE,
        alpha=0.38,
        zorder=1,
    )
    ax.axvline(ref_case["E_no"], color=PALETTE["gray"], linewidth=0.8, linestyle=(0, (2, 3)), alpha=0.55)
    ax.scatter(
        [ref_case["E_no"], ref_case["E_no"]],
        [ref_case["i_au_no"], ref_case["i_pd_no"]],
        s=44,
        marker="D",
        color=PALETTE["gray"],
        edgecolor="white",
        linewidth=0.6,
        zorder=8,
    )
    ax.annotate(
        "w/o EDL",
        xy=(ref_case["E_no"], ref_case["i_au_no"]),
        xytext=(0.444, 0.117),
        arrowprops=dict(arrowstyle="->", color=PALETTE["gray"], linewidth=0.8),
        ha="center",
        va="center",
        fontsize=7.1,
        color=PALETTE["gray"],
    )


def plot_case_curves(ax: plt.Axes, e_values: np.ndarray, case: dict[str, Any]) -> None:
    style = CASE_STYLES[case["C_M"]]
    linestyle = style["linestyle"]
    ax.plot(
        e_values,
        case["au_with"],
        color=PALETTE["green"],
        linewidth=2.05,
        linestyle=linestyle,
        alpha=0.94,
        zorder=4,
    )
    ax.plot(
        e_values,
        case["pd_with"],
        color=PALETTE["orange"],
        linewidth=2.05,
        linestyle=linestyle,
        alpha=0.94,
        zorder=4,
    )
    ax.axvline(case["E_with"], color=PALETTE["dark"], linewidth=0.72, linestyle=linestyle, alpha=0.38, zorder=2)
    ax.scatter(
        [case["E_with"], case["E_with"]],
        [case["i_au_with"], case["i_pd_with"]],
        s=52,
        marker=style["marker"],
        color=PALETTE["dark"],
        edgecolor="white",
        linewidth=0.7,
        zorder=9,
    )


def add_point_annotations(ax: plt.Axes, cases: list[dict[str, Any]]) -> None:
    targets = {
        1.0e-2: (0.612, 0.095, "0.01 M"),
        1.0: (0.526, 0.109, "1 M"),
        1.0e3: (0.432, -0.104, r"$10^3$ M"),
    }
    for case in cases:
        x_text, y_text, label = targets[case["C_M"]]
        ax.annotate(
            label,
            xy=(case["E_with"], case["i_au_with"]),
            xytext=(x_text, y_text),
            arrowprops=dict(arrowstyle="->", color=PALETTE["dark"], linewidth=0.82),
            ha="center",
            va="center",
            fontsize=7.1,
            color=PALETTE["dark"],
        )


def add_regime_labels(ax: plt.Axes) -> None:
    labels = [
        (0.580, -0.078, "EDL suppresses Au", "#FDE9E0"),
        (0.514, 0.135, r"$E_{mix}$ shift dominates", "#E9F1E8"),
        (0.448, -0.135, "EDL terms fade", "#E8EEF7"),
    ]
    for x, y, text, facecolor in labels:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=7.3,
            color=PALETTE["dark"],
            bbox=dict(facecolor=facecolor, edgecolor="none", alpha=0.88, boxstyle="round,pad=0.20"),
            zorder=10,
        )


def add_readout_box(ax: plt.Axes, cases: list[dict[str, Any]]) -> None:
    lines = [
        r"$C_{\mathrm{tot}}$     $\Delta E_{\mathrm{mix}}$    $I_{\mathrm{with}}/I_{\mathrm{no}}$    $K_{\mathrm{Au}}/K_{\mathrm{Au,no}}$"
    ]
    for case in cases:
        label = CASE_STYLES[case["C_M"]]["label"].replace("$", "")
        if case["C_M"] == 1.0e3:
            label = "10^3 M"
        lines.append(
            f"{label:<7} {case['delta_E_mV']:>6.1f} mV       "
            f"{case['I_ratio']:>5.3f}              {case['K_Au_ratio']:>6.4g}"
        )
    ax.text(
        0.985,
        0.055,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.75,
        color=PALETTE["dark"],
        linespacing=1.18,
        family="monospace",
        bbox=dict(facecolor="white", edgecolor="#D8D8D8", linewidth=0.7, alpha=0.92, boxstyle="round,pad=0.32"),
        zorder=12,
    )


def add_legends(ax: plt.Axes) -> None:
    reaction_handles = [
        Line2D([0], [0], color=PALETTE["green"], linewidth=2.2, label=r"Au oxidation"),
        Line2D([0], [0], color=PALETTE["orange"], linewidth=2.2, label=r"Pd reduction"),
        Line2D([0], [0], color=PALETTE["gray"], linewidth=1.7, linestyle=NO_EDL_STYLE, label="w/o EDL ref."),
    ]
    c_handles = [
        Line2D(
            [0],
            [0],
            color=PALETTE["dark"],
            linewidth=1.8,
            linestyle=CASE_STYLES[c_m]["linestyle"],
            marker=CASE_STYLES[c_m]["marker"],
            markerfacecolor=PALETTE["dark"],
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=5.8,
            label=CASE_STYLES[c_m]["label"],
        )
        for c_m in REPRESENTATIVE_C_M
    ]

    legend_reaction = ax.legend(
        handles=reaction_handles,
        loc="upper left",
        fontsize=7.0,
        handlelength=2.6,
        frameon=True,
        framealpha=0.90,
        facecolor="white",
        edgecolor="none",
        title="Half reaction",
        title_fontsize=7.2,
    )
    ax.add_artist(legend_reaction)
    legend_c = ax.legend(
        handles=c_handles,
        loc="upper right",
        fontsize=6.8,
        handlelength=2.4,
        frameon=True,
        framealpha=0.90,
        facecolor="white",
        edgecolor="none",
        title=r"$C_{\mathrm{tot}}$",
        title_fontsize=7.2,
    )
    ax.add_artist(legend_c)


def plot_overlay(ax: plt.Axes, cases: list[dict[str, Any]], e_values: np.ndarray) -> None:
    plot_reference_curves(ax, e_values, cases[0])
    for case in cases:
        plot_case_curves(ax, e_values, case)
    ax.axhline(0.0, color=PALETTE["dark"], linewidth=0.85, zorder=0)
    add_point_annotations(ax, cases)
    add_regime_labels(ax)
    add_readout_box(ax, cases)
    add_legends(ax)

    ax.set_xlim(E_MIN, E_MAX)
    ax.set_ylim(-Y_LIMIT, Y_LIMIT)
    ax.set_xlabel("Potential (V vs. RHE)")
    ax.set_ylabel(f"Half-reaction current ({CURRENT_UNIT})")
    ax.set_title(r"Salt-dependent polarization curves explain $I_{\mathrm{mix}}$", loc="left", fontsize=9.7, pad=5)
    ax.tick_params(length=3.2, width=0.85, labelsize=8.0)


def main() -> None:
    common.ensure_output_dirs()
    base_params = common.load_same_length_i0_alpha_params()
    e_values = np.linspace(E_MIN, E_MAX, 1300)
    apply_style()

    cases = compute_cases(base_params, e_values)
    fig, ax = plt.subplots(figsize=(7.2, 4.35))
    plot_overlay(ax, cases, e_values)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for path in (OUT_PATH, OUT_PATH.with_suffix(".svg")):
        fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH.relative_to(ROOT)}")
    print(f"Saved {OUT_PATH.with_suffix('.svg').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
