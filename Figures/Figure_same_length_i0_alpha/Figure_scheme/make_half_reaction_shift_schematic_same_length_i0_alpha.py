from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


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
    / f"half_reaction_shift_schematic_{common.OUTPUT_TAG}.png"
)


def compute_half_reaction_data(params: dict[str, Any], summary: dict[str, str]) -> dict[str, Any]:
    values = {
        "E1_eq": float(summary["E1_eq_eff"]),
        "E2_eq": float(summary["E2_eq_eff"]),
        "E_mix_with": float(summary["E_mix_with"]),
        "E_mix_no": float(summary["E_mix_no"]),
        "i_mix_avg_with": float(summary["i_mix_avg_with"]),
        "i_mix_avg_no": float(summary["i_mix_avg_no"]),
    }
    use_affine_phi2 = bool(params.get("use_affine_phi2", True))
    lambda_d = float(solver.compute_derived_params(params)["lambda_D"])
    e_values = np.linspace(0.38, 0.68, 1400)

    au_edl, pd_edl = compute_signed_half_currents(
        params, lambda_d, e_values, use_edl=True, use_affine_phi2=use_affine_phi2
    )
    au_no, pd_no = compute_signed_half_currents(
        params, lambda_d, e_values, use_edl=False, use_affine_phi2=use_affine_phi2
    )

    i_au_with, i_pd_with = current_at_mix(
        params, lambda_d, values["E_mix_with"], use_edl=True, use_affine_phi2=use_affine_phi2
    )
    i_au_no, i_pd_no = current_at_mix(
        params, lambda_d, values["E_mix_no"], use_edl=False, use_affine_phi2=use_affine_phi2
    )
    i_au_edl_at_no, i_pd_edl_at_no = current_at_mix(
        params, lambda_d, values["E_mix_no"], use_edl=True, use_affine_phi2=use_affine_phi2
    )
    assert_mixed_current("with EDL", i_au_with, i_pd_with, values["i_mix_avg_with"])
    assert_mixed_current("w/o EDL", i_au_no, i_pd_no, values["i_mix_avg_no"])

    return {
        "values": values,
        "e_values": e_values,
        "au_edl": current_density_to_display_current(params, au_edl),
        "pd_edl_abs": np.abs(current_density_to_display_current(params, pd_edl)),
        "au_no": current_density_to_display_current(params, au_no),
        "pd_no_abs": np.abs(current_density_to_display_current(params, pd_no)),
        "i_au_with": float(current_density_to_display_current(params, i_au_with)),
        "i_pd_with_abs": abs(float(current_density_to_display_current(params, i_pd_with))),
        "i_au_no": float(current_density_to_display_current(params, i_au_no)),
        "i_pd_no_abs": abs(float(current_density_to_display_current(params, i_pd_no))),
        "i_au_edl_at_no": float(current_density_to_display_current(params, i_au_edl_at_no)),
        "i_pd_edl_at_no_abs": abs(float(current_density_to_display_current(params, i_pd_edl_at_no))),
    }


def plot_half_reaction_panel(ax: plt.Axes, data: dict[str, Any]) -> None:
    values = data["values"]
    e_values = data["e_values"]
    no_edl_style = (0, (4, 3))

    ax.plot(
        e_values,
        data["au_no"],
        color=PALETTE["green"],
        linewidth=1.55,
        linestyle=no_edl_style,
        alpha=0.62,
        label="Au oxidation, w/o EDL",
    )
    ax.plot(
        e_values,
        data["pd_no_abs"],
        color=PALETTE["orange"],
        linewidth=1.55,
        linestyle=no_edl_style,
        alpha=0.62,
        label="Pd reduction, w/o EDL",
    )
    ax.plot(
        e_values,
        data["au_edl"],
        color=PALETTE["green"],
        linewidth=2.2,
        label="Au oxidation, with EDL",
    )
    ax.plot(
        e_values,
        data["pd_edl_abs"],
        color=PALETTE["orange"],
        linewidth=2.2,
        label="Pd reduction, with EDL",
    )

    e_no = values["E_mix_no"]
    e_with = values["E_mix_with"]
    i_no = data["i_au_no"]
    i_with = data["i_au_with"]
    au_at_no = data["i_au_edl_at_no"]
    pd_at_no = data["i_pd_edl_at_no_abs"]

    ax.axvline(e_no, color=PALETTE["gray"], linestyle=(0, (2, 3)), linewidth=0.95, alpha=0.78)
    ax.axvline(e_with, color=PALETTE["dark"], linestyle=(0, (3, 2)), linewidth=0.95, alpha=0.95)
    ax.scatter([e_no], [i_no], s=54, marker="s", color=PALETTE["gray"], edgecolor="white", linewidth=0.7, zorder=7)
    ax.scatter([e_with], [i_with], s=60, marker="o", color=PALETTE["dark"], edgecolor="white", linewidth=0.7, zorder=8)
    ax.scatter(
        [e_no, e_no],
        [au_at_no, pd_at_no],
        s=42,
        marker="^",
        color=[PALETTE["green"], PALETTE["orange"]],
        edgecolor="white",
        linewidth=0.65,
        zorder=8,
    )

    bracket_x = e_no + 0.017
    ax.annotate(
        "",
        xy=(bracket_x, pd_at_no),
        xytext=(bracket_x, au_at_no),
        arrowprops=dict(arrowstyle="<->", color=PALETTE["dark"], linewidth=0.9),
    )
    ax.text(
        bracket_x + 0.012,
        math.sqrt(au_at_no * pd_at_no),
        "EDL at old $E_{mix}$\n$|I_{Pd}| \\gg I_{Au}$\nnet reduction",
        ha="left",
        va="center",
        fontsize=7.1,
        color=PALETTE["dark"],
        linespacing=1.08,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, boxstyle="round,pad=0.16"),
    )

    arrow_y = 2.6e-4
    ax.annotate(
        "",
        xy=(e_with, arrow_y),
        xytext=(e_no, arrow_y),
        arrowprops=dict(arrowstyle="->", color=PALETTE["dark"], linewidth=1.0),
    )
    ax.text(
        0.5 * (e_no + e_with),
        1.55e-4,
        "raise potential until\n$ I_{Au}=|I_{Pd}|$",
        ha="center",
        va="top",
        fontsize=7.1,
        color=PALETTE["dark"],
        linespacing=1.05,
    )

    ax.text(
        e_no - 0.013,
        i_no * 1.65,
        "w/o EDL\nbalance",
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=PALETTE["gray"],
        linespacing=1.08,
    )
    ax.text(
        e_with + 0.013,
        i_with * 1.45,
        "with-EDL\nbalance",
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=PALETTE["dark"],
        linespacing=1.08,
    )

    ax.set_yscale("log")
    ax.set_xlim(0.38, 0.68)
    ax.set_ylim(6.0e-5, 60.0)
    ax.set_xlabel("Potential (V vs. RHE)")
    ax.set_ylabel(f"Half-reaction current magnitude ({CURRENT_UNIT})")
    ax.set_title("Actual half-reaction curves set the shift", loc="left", fontsize=9.7, pad=5)
    ax.legend(
        loc="upper right",
        fontsize=6.6,
        handlelength=2.4,
        ncols=2,
        columnspacing=0.85,
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="none",
    )
    ax.tick_params(length=3.2, width=0.85, labelsize=8.0)


def plot_balance_readout(ax: plt.Axes, data: dict[str, Any]) -> None:
    values = data["values"]
    e_no = values["E_mix_no"]
    e_with = values["E_mix_with"]
    d_e = e_with - e_no
    rows = [
        (
            "1. No-EDL balance",
            f"$E_{{mix}}$ = {fmt_v(e_no)}\n"
            rf"$I_{{Au}}=|I_{{Pd}}|$ = {data['i_au_no']:.4f}",
            "#F3F3F3",
        ),
        (
            "2. EDL at the old potential",
            f"$I_{{Au}}$ = {data['i_au_edl_at_no']:.4f}\n"
            rf"$|I_{{Pd}}|$ = {data['i_pd_edl_at_no_abs']:.3f}"
            "\n"
            "cathodic imbalance",
            "#FDE9E0",
        ),
        (
            "3. New with-EDL balance",
            f"$E_{{mix}}$ = {fmt_v(e_with)}\n"
            rf"$I_{{Au}}=|I_{{Pd}}|$ = {data['i_au_with']:.4f}"
            "\n"
            rf"$\Delta E_{{mix}}$ = +{d_e:.3f} V",
            "#E9F1E8",
        ),
    ]

    ax.set_axis_off()
    ax.text(
        0.02,
        0.99,
        rf"Balance criterion: $I_{{Au}}+I_{{Pd}}=0$"
        "\n"
        rf"same as $I_{{Au}}=|I_{{Pd}}|$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color=PALETTE["dark"],
        linespacing=1.12,
    )
    for (title, body, facecolor), y in zip(rows, (0.79, 0.52, 0.23)):
        ax.text(
            0.02,
            y,
            f"{title}\n{body}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.55,
            color=PALETTE["dark"],
            linespacing=1.18,
            bbox=dict(facecolor=facecolor, edgecolor="#D8D8D8", linewidth=0.7, boxstyle="round,pad=0.40"),
        )

def main() -> None:
    common.ensure_output_dirs()
    params, summary = common.load_inputs_for_scripts()
    common.assert_expected_values(summary)
    apply_style()

    data = compute_half_reaction_data(params, summary)
    fig = plt.figure(figsize=(7.25, 3.85))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=(1.72, 0.88), wspace=0.18)
    plot_half_reaction_panel(fig.add_subplot(gs[0, 0]), data)
    plot_balance_readout(fig.add_subplot(gs[0, 1]), data)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for path in (OUT_PATH, OUT_PATH.with_suffix(".svg")):
        fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH.relative_to(ROOT)}")
    print(f"Saved {OUT_PATH.with_suffix('.svg').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
