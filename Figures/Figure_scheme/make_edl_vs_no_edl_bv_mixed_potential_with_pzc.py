from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from Figures.Figure_scheme.make_edl_vs_no_edl_bv_mixed_potential import (  # noqa: E402
    OUT_DIR,
    PALETTE,
    RESULT_ID,
    add_mixed_current_annotations,
    add_potential_marker,
    assert_mixed_current,
    compute_signed_half_currents,
    current_at_mix,
    fmt_v,
    load_inputs,
    solver,
)


PZC_COLOR = "#C9A227"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "legend.frameon": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Nimbus Sans",
            "mathtext.it": "Nimbus Sans:italic",
            "mathtext.bf": "Nimbus Sans:bold",
            "mathtext.cal": "Nimbus Sans",
            "mathtext.sf": "Nimbus Sans",
            "mathtext.tt": "Nimbus Sans",
        }
    )


def add_pzc_lane(
    ax: plt.Axes,
    pzcs: list[tuple[str, float]],
    e_min: float,
    e_max: float,
) -> None:
    ax.set_xlim(e_min, e_max)
    ax.set_ylim(-0.35, 0.65)
    ax.hlines(0.0, e_min, e_max, color=PALETTE["dark"], linewidth=0.75)
    for label, x in pzcs:
        ax.vlines(x, -0.12, 0.12, color=PZC_COLOR, linewidth=0.85)
        ax.scatter([x], [0.0], marker="^", s=34, color=PZC_COLOR, edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(
            x,
            0.26,
            f"PZC {label}\n{fmt_v(x)}",
            ha="center",
            va="bottom",
            fontsize=6.9,
            color=PALETTE["dark"],
            linespacing=1.05,
        )
    ax.text(e_min, 0.0, "PZC", ha="right", va="center", fontsize=7.4, color=PALETTE["gray"])
    ax.set_yticks([])
    ax.set_xlabel("Potential (V vs. RHE)", fontsize=9.0)
    ax.set_xticks([0.10, 0.30, 0.50, 0.70, 0.90])
    ax.tick_params(axis="x", length=3.2, width=0.85, labelsize=8.0)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.85)


def main() -> None:
    params, summary = load_inputs()

    values = {
        "E1_eq": float(summary["E1_eq_eff"]),
        "E2_eq": float(summary["E2_eq_eff"]),
        "E_mix_with": float(summary["E_mix_with"]),
        "E_mix_no": float(summary["E_mix_no"]),
        "i_mix_avg_with": float(summary["i_mix_avg_with"]),
        "i_mix_avg_no": float(summary["i_mix_avg_no"]),
        "pzc_C": float(params["pzc_C"]),
        "pzc_Pd": float(params["pzc_Pd"]),
        "pzc_Au": float(params["pzc_Au"]),
    }
    pzcs = [
        ("support", values["pzc_C"]),
        ("Pd", values["pzc_Pd"]),
        ("Au", values["pzc_Au"]),
    ]

    use_affine_phi2 = bool(params.get("use_affine_phi2", True))
    lambda_d = float(solver.compute_derived_params(params)["lambda_D"])

    e_min = values["E1_eq"] - 0.06
    e_max = max(values["E2_eq"] + 0.06, values["pzc_Au"] + 0.04)
    e_values = np.linspace(e_min, e_max, 1300)

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
    assert_mixed_current("with EDL", i_au_with, i_pd_with, values["i_mix_avg_with"])
    assert_mixed_current("w/o EDL", i_au_no, i_pd_no, values["i_mix_avg_no"])

    apply_style()

    fig, (ax, ax_pzc) = plt.subplots(
        2,
        1,
        figsize=(6.3, 4.05),
        sharex=True,
        gridspec_kw={"height_ratios": [4.4, 0.65], "hspace": 0.06},
    )

    y_limit = max(0.13, 2.35 * max(values["i_mix_avg_with"], values["i_mix_avg_no"]))
    no_edl_style = (0, (4, 3))

    for _, x in pzcs:
        ax.axvline(x, color=PZC_COLOR, linestyle=(0, (1, 2)), linewidth=0.85, alpha=0.55, zorder=0)

    ax.plot(e_values, au_edl, color=PALETTE["green"], linewidth=2.2, label=r"Au anodic, with EDL", zorder=4)
    ax.plot(e_values, pd_edl, color=PALETTE["orange"], linewidth=2.2, label=r"Pd cathodic, with EDL", zorder=4)
    ax.plot(
        e_values,
        au_no,
        color=PALETTE["green"],
        linewidth=1.7,
        linestyle=no_edl_style,
        alpha=0.58,
        label=r"Au anodic, w/o EDL",
        zorder=3,
    )
    ax.plot(
        e_values,
        pd_no,
        color=PALETTE["orange"],
        linewidth=1.7,
        linestyle=no_edl_style,
        alpha=0.58,
        label=r"Pd cathodic, w/o EDL",
        zorder=3,
    )

    ax.axhline(0.0, color=PALETTE["dark"], linewidth=0.9, zorder=1)
    add_mixed_current_annotations(
        ax,
        values["E_mix_with"],
        i_au_with,
        i_pd_with,
        color=PALETTE["dark"],
        linestyle=(0, (3, 2)),
        alpha=1.0,
        zorder=6,
    )
    add_mixed_current_annotations(
        ax,
        values["E_mix_no"],
        i_au_no,
        i_pd_no,
        color=PALETTE["gray"],
        linestyle=(0, (2, 3)),
        alpha=0.72,
        zorder=5,
    )

    ax.hlines(
        [i_au_with, i_pd_with],
        values["E1_eq"],
        values["E_mix_with"],
        colors=[PALETTE["green"], PALETTE["orange"]],
        linestyles=(0, (3, 3)),
        linewidth=0.9,
        alpha=0.82,
        zorder=2,
    )
    ax.hlines(
        [i_au_no, i_pd_no],
        values["E1_eq"],
        values["E_mix_no"],
        colors=[PALETTE["green"], PALETTE["orange"]],
        linestyles=no_edl_style,
        linewidth=0.8,
        alpha=0.45,
        zorder=2,
    )

    ax.scatter([values["E_mix_with"]], [0.0], s=50, color=PALETTE["dark"], edgecolor="white", linewidth=0.6, zorder=8)
    ax.scatter([values["E_mix_no"]], [0.0], s=44, color=PALETTE["gray"], edgecolor="white", linewidth=0.6, zorder=7)

    add_potential_marker(ax, values["E1_eq"], r"$E_{1,\mathrm{eq}}$", PALETTE["blue"], 0.33 * y_limit)
    add_potential_marker(
        ax,
        values["E2_eq"],
        r"$E_{2,\mathrm{eq}}$",
        PALETTE["blue"],
        -0.33 * y_limit,
        text_x_offset=-0.015,
    )

    ax.text(
        values["E_mix_with"] + 0.012,
        0.55 * y_limit,
        rf"with EDL: $E_{{\mathrm{{mix}}}}={fmt_v(values['E_mix_with'])}$",
        ha="left",
        va="bottom",
        fontsize=8.4,
        color=PALETTE["dark"],
    )
    ax.text(
        values["E_mix_with"] + 0.012,
        i_au_with,
        rf"$|i|={i_au_with:.3f}$ A m$^{{-2}}$",
        ha="left",
        va="center",
        fontsize=8.0,
        color=PALETTE["dark"],
    )
    ax.text(
        values["E_mix_no"] - 0.012,
        -0.55 * y_limit,
        rf"w/o EDL: $E_{{\mathrm{{mix}}}}={fmt_v(values['E_mix_no'])}$",
        ha="right",
        va="top",
        fontsize=8.2,
        color=PALETTE["gray"],
    )
    ax.text(
        values["E_mix_no"] - 0.012,
        i_pd_no + 0.010,
        rf"$|i|={i_au_no:.3f}$ A m$^{{-2}}$",
        ha="right",
        va="bottom",
        fontsize=7.8,
        color=PALETTE["gray"],
    )

    ax.set_xlim(e_min, e_max)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_ylabel(r"Average half-reaction current density (A m$^{-2}$)", fontsize=9.0)
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.tick_params(axis="y", length=3.2, width=0.85, labelsize=8.0)
    ax.set_title("EDL shifts the mixed potential", loc="left", fontsize=9.8, pad=6)
    ax.text(
        0.995,
        1.015,
        RESULT_ID,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        color=PALETTE["gray"],
    )
    ax.legend(loc="upper left", fontsize=7.4, handlelength=2.6, ncols=2, columnspacing=1.0)

    add_pzc_lane(ax_pzc, pzcs, e_min, e_max)

    fig.subplots_adjust(left=0.13, right=0.985, top=0.92, bottom=0.14, hspace=0.06)
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"edl_vs_no_edl_bv_mixed_potential_with_pzc_{RESULT_ID}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"E_mix_with = {values['E_mix_with']:.15g} V")
    print(f"E_mix_no = {values['E_mix_no']:.15g} V")
    print(f"pzc_C = {values['pzc_C']:.15g} V")
    print(f"pzc_Pd = {values['pzc_Pd']:.15g} V")
    print(f"pzc_Au = {values['pzc_Au']:.15g} V")
    print(f"Saved {OUT_DIR / f'edl_vs_no_edl_bv_mixed_potential_with_pzc_{RESULT_ID}.png'}")
    print(f"Saved {OUT_DIR / f'edl_vs_no_edl_bv_mixed_potential_with_pzc_{RESULT_ID}.svg'}")


if __name__ == "__main__":
    main()
