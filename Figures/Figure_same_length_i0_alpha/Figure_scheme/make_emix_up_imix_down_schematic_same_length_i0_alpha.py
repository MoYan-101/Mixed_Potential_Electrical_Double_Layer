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
    add_mixed_current_annotations,
    assert_mixed_current,
    compute_signed_half_currents,
    current_at_mix,
    fmt_v,
    solver,
)


OUT_PATH = (
    common.FIGURE_SCHEME_DIR
    / f"emix_up_imix_down_schematic_{common.OUTPUT_TAG}.png"
)
CURRENT_SCALE = 1e9
CURRENT_UNIT = r"10$^{-3}$ uA"
REFERENCE_XMIN = -0.02
REFERENCE_XMAX = 1.02
REFERENCE_XTICKS = [0.0, 0.25, 0.50, 0.75, 1.0]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 8.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Nimbus Sans",
            "mathtext.it": "Nimbus Sans:italic",
            "mathtext.bf": "Nimbus Sans:bold",
            "mathtext.cal": "Nimbus Sans",
            "mathtext.sf": "Nimbus Sans",
            "mathtext.tt": "Nimbus Sans",
        }
    )


def segment_average(values: np.ndarray, x: np.ndarray, mask: np.ndarray) -> float:
    x_seg = np.asarray(x[mask], dtype=float)
    y_seg = np.asarray(values[mask], dtype=float)
    if x_seg.size < 2:
        return float(np.nanmean(y_seg))
    width = float(x_seg[-1] - x_seg[0])
    if width <= 0.0:
        return float(np.nanmean(y_seg))
    return float(np.trapz(y_seg, x_seg) / width)


def single_period_reactive_area(params: dict[str, Any]) -> float:
    reactive_length = float(params["L_Au"]) + float(params["L_Pd_len"])
    return reactive_length * float(params.get("out_of_plane_width", 1.0))


def current_density_to_display_current(params: dict[str, Any], current_density: np.ndarray | float) -> np.ndarray:
    return np.asarray(current_density, dtype=float) * single_period_reactive_area(params) * CURRENT_SCALE


def compute_explanation_values(params: dict[str, Any], summary: dict[str, str]) -> dict[str, float]:
    pair = solver.run_edl_comparison_pair(params, mode="FULL")
    common.assert_expected_values(common.summary_from_pair(pair))

    res_with = pair["with_edl"]
    res_no = pair["no_edl"]
    beta = float(params["F"]) / (float(params["R"]) * float(params["T"]))
    alpha1 = float(params["alpha1"])
    alpha2 = float(params["alpha2"])
    dE = float(summary["E_mix_with"]) - float(summary["E_mix_no"])

    au_e_factor = math.exp((1.0 - alpha1) * beta * dE)
    pd_e_factor = math.exp(-alpha2 * beta * dE)
    au_edl_factor = float(res_with["K_Au"]) / float(res_no["K_Au"])
    pd_edl_factor = float(res_with["K_Pd"]) / float(res_no["K_Pd"])
    current_ratio = float(summary["i_mix_avg_with"]) / float(summary["i_mix_avg_no"])

    prof_with, derived_with = solver.build_profiles_for_emix(params, float(summary["E_mix_with"]), use_edl=True)
    x_m = np.asarray(prof_with["x_tilde"], dtype=float) * float(derived_with["lambda_D"])
    phi_tilde = np.asarray(prof_with["phi_tilde"], dtype=float)
    phi_v = phi_tilde * float(params["R"]) * float(params["T"]) / float(params["F"])
    mask_au = np.asarray(prof_with["mask_Au"], dtype=bool)
    mask_pd = np.asarray(prof_with["mask_Pd"], dtype=bool)

    c_r1_norm = np.exp(-float(params["z_R1"]) * phi_tilde)
    c_o2_norm = np.exp(-float(params["z_O2"]) * phi_tilde)

    return {
        "dE": dE,
        "au_e_factor": au_e_factor,
        "pd_e_factor": pd_e_factor,
        "au_edl_factor": au_edl_factor,
        "pd_edl_factor": pd_edl_factor,
        "current_ratio": current_ratio,
        "current_pct": (current_ratio - 1.0) * 100.0,
        "phi_au_mV": 1000.0 * segment_average(phi_v, x_m, mask_au),
        "phi_pd_mV": 1000.0 * segment_average(phi_v, x_m, mask_pd),
        "c_r1_au": segment_average(c_r1_norm, x_m, mask_au),
        "c_o2_pd": segment_average(c_o2_norm, x_m, mask_pd),
    }


def plot_polarization_panel(
    ax: plt.Axes,
    params: dict[str, Any],
    summary: dict[str, str],
) -> tuple[float, float]:
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
    e_min = REFERENCE_XMIN
    e_max = REFERENCE_XMAX
    e_values = np.linspace(e_min, e_max, 1200)

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

    au_edl = current_density_to_display_current(params, au_edl)
    pd_edl = current_density_to_display_current(params, pd_edl)
    au_no = current_density_to_display_current(params, au_no)
    pd_no = current_density_to_display_current(params, pd_no)
    i_au_with = float(current_density_to_display_current(params, i_au_with))
    i_pd_with = float(current_density_to_display_current(params, i_pd_with))
    i_au_no = float(current_density_to_display_current(params, i_au_no))
    i_pd_no = float(current_density_to_display_current(params, i_pd_no))
    i_mix_with = float(current_density_to_display_current(params, values["i_mix_avg_with"]))
    i_mix_no = float(current_density_to_display_current(params, values["i_mix_avg_no"]))

    y_limit = 2.25 * max(i_mix_with, i_mix_no)
    no_edl_style = (0, (4, 3))

    ax.plot(e_values, au_edl, color=PALETTE["green"], linewidth=2.1, label="Oxidation on Au, with EDL")
    ax.plot(e_values, pd_edl, color=PALETTE["orange"], linewidth=2.1, label="Reduction on Pd, with EDL")
    ax.plot(
        e_values,
        au_no,
        color=PALETTE["green"],
        linewidth=1.55,
        linestyle=no_edl_style,
        alpha=0.58,
        label="Oxidation on Au, w/o EDL",
    )
    ax.plot(
        e_values,
        pd_no,
        color=PALETTE["orange"],
        linewidth=1.55,
        linestyle=no_edl_style,
        alpha=0.58,
        label="Reduction on Pd, w/o EDL",
    )
    ax.axhline(0.0, color=PALETTE["dark"], linewidth=0.85)

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

    ax.annotate(
        "",
        xy=(values["E_mix_with"], -0.72 * y_limit),
        xytext=(values["E_mix_no"], -0.72 * y_limit),
        arrowprops=dict(arrowstyle="->", color=PALETTE["dark"], linewidth=1.0),
    )
    ax.text(
        0.5 * (values["E_mix_no"] + values["E_mix_with"]),
        -0.83 * y_limit,
        rf"$E_{{\mathrm{{mix}}}}$ shifts up by {values['E_mix_with'] - values['E_mix_no']:.3f} V",
        ha="center",
        va="top",
        fontsize=8.0,
        color=PALETTE["dark"],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, boxstyle="round,pad=0.12"),
    )

    drop_arrow_x = e_max - 0.03
    ax.annotate(
        "",
        xy=(drop_arrow_x, i_mix_with),
        xytext=(drop_arrow_x, i_mix_no),
        arrowprops=dict(arrowstyle="->", color=PALETTE["dark"], linewidth=1.05),
    )
    ax.text(
        drop_arrow_x - 0.018,
        0.5 * (i_mix_with + i_mix_no),
        r"$I_{\mathrm{mix}}$ drops",
        ha="right",
        va="center",
        fontsize=7.7,
        color=PALETTE["dark"],
        linespacing=1.05,
    )

    ax.text(
        values["E_mix_with"] + 0.012,
        0.56 * y_limit,
        rf"with EDL: {fmt_v(values['E_mix_with'])}, $|I|={i_mix_with:.3f}$ {CURRENT_UNIT}",
        ha="left",
        va="center",
        fontsize=7.7,
        color=PALETTE["dark"],
    )
    ax.text(
        values["E_mix_no"] - 0.012,
        -0.50 * y_limit,
        rf"w/o EDL: {fmt_v(values['E_mix_no'])}, $|I|={i_mix_no:.3f}$ {CURRENT_UNIT}",
        ha="right",
        va="center",
        fontsize=7.7,
        color=PALETTE["gray"],
    )

    ax.set_xlim(e_min, e_max)
    ax.set_xticks(REFERENCE_XTICKS)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xlabel("Potential (V vs. RHE)")
    ax.set_ylabel(f"Current ({CURRENT_UNIT})")
    ax.tick_params(length=3.2, width=0.85, labelsize=8.0)
    ax.set_title(r"Mixed-potential balance", loc="left", fontsize=9.7, pad=5)
    ax.legend(loc="upper left", fontsize=6.6, handlelength=2.5, ncols=2, columnspacing=0.85)
    return e_min, e_max


def plot_factor_panel(ax: plt.Axes, explanation: dict[str, float]) -> None:
    factors = np.array(
        [
            1.0,
            explanation["au_e_factor"],
            explanation["au_e_factor"] * explanation["au_edl_factor"],
        ],
        dtype=float,
    )
    labels = [
        "w/o EDL\nbaseline",
        "higher\n$E_{mix}$ only",
        "with EDL\nfinal",
    ]
    colors = [PALETTE["gray"], PALETTE["green"], PALETTE["dark"]]
    x = np.arange(len(factors), dtype=float)

    ax.bar(x, factors, width=0.58, color=colors, edgecolor=PALETTE["dark"], linewidth=0.8)
    ax.set_yscale("log")
    ax.axhline(1.0, color=PALETTE["gray"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Au current factor vs w/o EDL")
    ax.set_ylim(0.035, 25.0)
    ax.tick_params(axis="both", length=3.2, width=0.85, labelsize=8.0)
    ax.set_title(
        rf"Why $i_{{\mathrm{{mix}}}}$ decreases"
        "\n"
        rf"$\times${explanation['au_e_factor']:.1f}"
        rf" $\times$ {explanation['au_edl_factor']:.4f}"
        rf" $=$ $\times${explanation['current_ratio']:.3f}",
        loc="left",
        fontsize=9.2,
        pad=5,
    )

    ax.text(x[0], factors[0] * 1.22, "1.00", ha="center", va="bottom", fontsize=8.0, color=PALETTE["dark"])
    ax.text(
        x[1],
        factors[1] * 1.20,
        f"{explanation['au_e_factor']:.1f}",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=PALETTE["green"],
    )
    ax.text(
        x[2],
        factors[2] * 1.18,
        f"{explanation['current_ratio']:.3f}",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=PALETTE["dark"],
    )
    ax.text(
        0.045,
        0.055,
        "EDL local term on Au\n"
        rf"$\langle\phi_{{RP}}\rangle={explanation['phi_au_mV']:.0f}$ mV; "
        rf"$c_{{R1}}/c_b={explanation['c_r1_au']:.4f}$" "\n"
        rf"$K_{{Au}}^{{EDL}}/K_{{Au}}^{{no}}={explanation['au_edl_factor']:.4f}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.85,
        color=PALETTE["dark"],
        linespacing=1.18,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, boxstyle="round,pad=0.20"),
    )


def main() -> None:
    common.ensure_output_dirs()
    params, summary = common.load_inputs_for_scripts()
    common.assert_expected_values(summary)

    apply_style()
    fig, ax = plt.subplots(figsize=(5.35, 3.65))
    plot_polarization_panel(ax, params, summary)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for path in (OUT_PATH, OUT_PATH.with_suffix(".svg")):
        fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH.relative_to(ROOT)}")
    print(f"Saved {OUT_PATH.with_suffix('.svg').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
