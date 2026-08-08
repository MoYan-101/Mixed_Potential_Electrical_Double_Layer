from __future__ import annotations

import copy
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
from Figures.Figure_scheme.make_edl_vs_no_edl_bv_mixed_potential import PALETTE, solver  # noqa: E402


OUT_PATH = (
    common.FIGURE_SCHEME_DIR
    / f"ctot_high_salt_regime_schematic_{common.OUTPUT_TAG}.png"
)
CURRENT_SCALE = 1.0e9
CURRENT_UNIT = r"10$^{-3}$ uA"
REPRESENTATIVE_C_M = (1.0e-2, 1.0, 1.0e3)


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


def ctot_scan_values() -> np.ndarray:
    values = np.concatenate(
        [
            np.logspace(-4, 3, 33),
            np.asarray(REPRESENTATIVE_C_M, dtype=float),
        ]
    )
    return np.asarray(sorted(set(np.round(values, 14))), dtype=float)


def compute_ctot_rows(params: dict[str, Any], c_values_m: np.ndarray) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for c_m in c_values_m:
        p = copy.deepcopy(params)
        p["C_tot"] = solver.concentration_M_to_mol_per_m3(float(c_m))
        p["lambda_D"] = None
        pair = solver.run_edl_comparison_pair(p, mode="FULL")
        with_edl = pair["with_edl"]
        no_edl = pair["no_edl"]

        rows.append(
            {
                "C_M": float(c_m),
                "lambda_D_nm": float(with_edl["lambda_D"]) * 1.0e9,
                "E_with": float(with_edl["E_mix"]),
                "E_no": float(no_edl["E_mix"]),
                "I_with": float(with_edl["i_mix_abs_A"]) * CURRENT_SCALE,
                "I_no": float(no_edl["i_mix_abs_A"]) * CURRENT_SCALE,
                "I_ratio": float(with_edl["i_mix_abs_A"]) / float(no_edl["i_mix_abs_A"]),
                "K_Au_ratio": float(with_edl["K_Au"]) / float(no_edl["K_Au"]),
                "K_Pd_ratio": float(with_edl["K_Pd"]) / float(no_edl["K_Pd"]),
                "delta_E_mV": 1000.0 * (float(with_edl["E_mix"]) - float(no_edl["E_mix"])),
                "max_abs_phi_tilde": float(with_edl["max_abs_phi_tilde"]),
            }
        )
    return rows


def row_at(rows: list[dict[str, float]], c_m: float) -> dict[str, float]:
    for row in rows:
        if np.isclose(row["C_M"], c_m, rtol=0.0, atol=max(1.0e-14, 1.0e-12 * c_m)):
            return row
    raise ValueError(f"Missing representative C_tot = {c_m:g} M")


def setup_regime_background(ax: plt.Axes) -> None:
    spans = [
        (1.0e-4, 7.0e-2, "#FDE9E0", "low salt\nEDL suppresses Au"),
        (7.0e-2, 8.0, "#E9F1E8", "$E_{mix}$ shift\ndominates"),
        (8.0, 1.0e3, "#E8EEF7", "formal high salt\nEDL terms fade"),
    ]
    for xmin, xmax, color, label in spans:
        ax.axvspan(xmin, xmax, color=color, alpha=0.55, linewidth=0, zorder=0)
        x_mid = 10.0 ** (0.5 * (np.log10(xmin) + np.log10(xmax)))
        ax.text(
            x_mid,
            0.98,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.85,
            color=PALETTE["dark"],
            linespacing=1.05,
        )


def plot_trend_panel(ax: plt.Axes, rows: list[dict[str, float]]) -> None:
    c = np.asarray([row["C_M"] for row in rows], dtype=float)
    i_with = np.asarray([row["I_with"] for row in rows], dtype=float)
    i_no = np.asarray([row["I_no"] for row in rows], dtype=float)

    setup_regime_background(ax)
    ax.plot(c, i_with, color="#F26B38", linewidth=2.25, marker="o", markersize=3.0, label="with EDL")
    ax.plot(c, i_no, color="#12355B", linewidth=1.8, linestyle=(0, (4, 2.5)), label="w/o EDL")

    for c_m, label, y_offset, ha in (
        (1.0e-2, "0.01 M", -0.009, "center"),
        (1.0, "1 M", -0.008, "center"),
        (1.0e3, "$10^3$ M", 0.006, "right"),
    ):
        row = row_at(rows, c_m)
        ax.scatter(
            [row["C_M"]],
            [row["I_with"]],
            s=44,
            color="#F26B38",
            edgecolor="white",
            linewidth=0.75,
            zorder=5,
        )
        ax.text(
            row["C_M"],
            row["I_with"] + y_offset,
            label,
            ha=ha,
            va="bottom" if y_offset > 0 else "top",
            fontsize=7.2,
            color=PALETTE["dark"],
        )

    ax.annotate(
        "overshoot",
        xy=(1.0, row_at(rows, 1.0)["I_with"]),
        xytext=(0.20, row_at(rows, 1.0)["I_with"] + 0.017),
        arrowprops=dict(arrowstyle="->", color=PALETTE["dark"], linewidth=0.85),
        fontsize=7.2,
        color=PALETTE["dark"],
        ha="center",
    )
    ax.annotate(
        "returns toward w/o EDL",
        xy=(1.0e3, row_at(rows, 1.0e3)["I_with"]),
        xytext=(32.0, row_at(rows, 1.0e3)["I_with"] - 0.023),
        arrowprops=dict(arrowstyle="->", color=PALETTE["dark"], linewidth=0.85),
        fontsize=7.2,
        color=PALETTE["dark"],
        ha="center",
    )

    ax.set_xscale("log")
    ax.set_xlim(1.0e-4, 1.0e3)
    ax.set_ylim(0.017, 0.074)
    ax.set_xlabel(r"Electrolyte concentration, $C_{\mathrm{tot}}$ (M)")
    ax.set_ylabel(f"Current ({CURRENT_UNIT})")
    ax.set_title(r"$I_{\mathrm{mix}}$ overshoots before the high-salt limit", loc="left", fontsize=9.7, pad=5)
    ax.legend(loc="lower right", fontsize=7.4, handlelength=2.4)
    ax.tick_params(length=3.2, width=0.85, labelsize=8.0)


def format_card_value(row: dict[str, float]) -> str:
    return (
        rf"$K_{{Au}}/L_{{Au}}$ = {row['K_Au_ratio']:.3g}" "\n"
        rf"$K_{{Pd}}/L_{{Pd}}$ = {row['K_Pd_ratio']:.3g}" "\n"
        rf"$\Delta E_{{mix}}$ = {row['delta_E_mV']:.1f} mV" "\n"
        rf"$I_{{with}}/I_{{no}}$ = {row['I_ratio']:.3f}"
    )


def plot_mechanism_cards(ax: plt.Axes, rows: list[dict[str, float]]) -> None:
    ax.set_axis_off()
    cards = [
        (
            "Low salt: local EDL term wins",
            "Au oxidation is strongly suppressed;\nPd reduction is strongly enhanced.",
            row_at(rows, 1.0e-2),
            "#FDE9E0",
        ),
        (
            r"Intermediate salt: $E_{mix}$ shift wins",
            r"Direct Au suppression weakens,"
            "\n"
            r"but the positive $E_{mix}$ shift remains.",
            row_at(rows, 1.0),
            "#E9F1E8",
        ),
        (
            "Formal high salt: EDL terms fade",
            r"K factors approach 1 and $E_{mix}$"
            "\n"
            "returns toward the w/o EDL value.",
            row_at(rows, 1.0e3),
            "#E8EEF7",
        ),
    ]

    y_positions = [0.98, 0.66, 0.34]
    for (title, description, row, facecolor), y in zip(cards, y_positions):
        body = (
            f"{title}\n"
            f"{description}\n\n"
            f"{format_card_value(row)}"
        )
        ax.text(
            0.02,
            y,
            body,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.35,
            color=PALETTE["dark"],
            linespacing=1.14,
            bbox=dict(facecolor=facecolor, edgecolor="#D9D9D9", linewidth=0.7, boxstyle="round,pad=0.38"),
        )


def main() -> None:
    common.ensure_output_dirs()
    params, _summary = common.load_inputs_for_scripts()
    apply_style()

    rows = compute_ctot_rows(params, ctot_scan_values())

    fig = plt.figure(figsize=(7.35, 4.15))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=(1.55, 1.0), wspace=0.18)
    trend_ax = fig.add_subplot(gs[0, 0])
    card_ax = fig.add_subplot(gs[0, 1])
    plot_trend_panel(trend_ax, rows)
    plot_mechanism_cards(card_ax, rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for path in (OUT_PATH, OUT_PATH.with_suffix(".svg")):
        fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH.relative_to(ROOT)}")
    print(f"Saved {OUT_PATH.with_suffix('.svg').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
