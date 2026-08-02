from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SAME_LENGTH_I0_ALPHA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SAME_LENGTH_I0_ALPHA_DIR))

import same_length_i0_alpha_common as common  # noqa: E402
from Figures.Figure_same_length_i0_alpha.Figure_scheme.make_ctot_high_salt_regime_schematic_same_length_i0_alpha import (  # noqa: E402
    PALETTE,
    apply_style,
    compute_ctot_rows,
    ctot_scan_values,
    row_at,
)


OUT_PATH = (
    common.FIGURE_SCHEME_DIR
    / f"ctot_emix_high_salt_regime_schematic_{common.OUTPUT_TAG}.png"
)


def setup_regime_background(ax: plt.Axes) -> None:
    spans = [
        (1.0e-4, 7.0e-2, "#FDE9E0", "low salt\nlarge positive shift"),
        (7.0e-2, 8.0, "#E9F1E8", "$E_{mix}$ shift\nweakens"),
        (8.0, 1.0e3, "#E8EEF7", "formal high salt\nreturns to w/o EDL"),
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
    e_with = np.asarray([row["E_with"] for row in rows], dtype=float)
    e_no = np.asarray([row["E_no"] for row in rows], dtype=float)

    setup_regime_background(ax)
    ax.plot(c, e_with, color="#F26B38", linewidth=2.25, marker="o", markersize=3.0, label="with EDL")
    ax.plot(c, e_no, color="#12355B", linewidth=1.8, linestyle=(0, (4, 2.5)), label="w/o EDL")

    for c_m, label, y_offset, ha in (
        (1.0e-2, "0.01 M", -0.014, "center"),
        (1.0, "1 M", 0.010, "center"),
        (1.0e3, "$10^3$ M", 0.009, "right"),
    ):
        row = row_at(rows, c_m)
        ax.scatter(
            [row["C_M"]],
            [row["E_with"]],
            s=44,
            color="#F26B38",
            edgecolor="white",
            linewidth=0.75,
            zorder=5,
        )
        ax.text(
            row["C_M"],
            row["E_with"] + y_offset,
            label,
            ha=ha,
            va="bottom" if y_offset > 0 else "top",
            fontsize=7.2,
            color=PALETTE["dark"],
        )

    ax.annotate(
        "largest shift",
        xy=(1.0e-2, row_at(rows, 1.0e-2)["E_with"]),
        xytext=(1.7e-3, 0.627),
        arrowprops=dict(arrowstyle="->", color=PALETTE["dark"], linewidth=0.85),
        fontsize=7.2,
        color=PALETTE["dark"],
        ha="center",
    )
    ax.annotate(
        "shift decays",
        xy=(1.0, row_at(rows, 1.0)["E_with"]),
        xytext=(0.23, 0.536),
        arrowprops=dict(arrowstyle="->", color=PALETTE["dark"], linewidth=0.85),
        fontsize=7.2,
        color=PALETTE["dark"],
        ha="center",
    )
    ax.annotate(
        "returns toward w/o EDL",
        xy=(1.0e3, row_at(rows, 1.0e3)["E_with"]),
        xytext=(35.0, 0.488),
        arrowprops=dict(arrowstyle="->", color=PALETTE["dark"], linewidth=0.85),
        fontsize=7.2,
        color=PALETTE["dark"],
        ha="center",
    )

    ax.set_xscale("log")
    ax.set_xlim(1.0e-4, 1.0e3)
    ax.set_ylim(0.455, 0.655)
    ax.set_xlabel(r"Electrolyte concentration, $C_{\mathrm{tot}}$ (M)")
    ax.set_ylabel(r"$E_{\mathrm{mix}}$ (V vs. RHE)")
    ax.set_title(r"$E_{\mathrm{mix}}$ shift fades in the high-salt limit", loc="left", fontsize=9.7, pad=5)
    ax.legend(loc="lower left", fontsize=7.4, handlelength=2.4)
    ax.tick_params(length=3.2, width=0.85, labelsize=8.0)


def format_card_value(row: dict[str, float]) -> str:
    return (
        rf"$E_{{mix,with}}$ = {row['E_with']:.3f} V" "\n"
        rf"$E_{{mix,no}}$ = {row['E_no']:.3f} V" "\n"
        rf"$\Delta E_{{mix}}$ = {row['delta_E_mV']:.1f} mV" "\n"
        rf"$I_{{with}}/I_{{no}}$ = {row['I_ratio']:.3f}"
    )


def plot_mechanism_cards(ax: plt.Axes, rows: list[dict[str, float]]) -> None:
    ax.set_axis_off()
    cards = [
        (
            r"Low salt: largest positive $E_{mix}$ shift",
            "EDL strongly shifts the mixed potential;\nAu depletion still lowers the current.",
            row_at(rows, 1.0e-2),
            "#FDE9E0",
        ),
        (
            r"Intermediate salt: shift remains",
            r"The positive $E_{mix}$ shift weakens,"
            "\n"
            "but it is still large enough to drive overshoot.",
            row_at(rows, 1.0),
            "#E9F1E8",
        ),
        (
            r"Formal high salt: $E_{mix}$ returns",
            "EDL potentials are screened out;\nwith-EDL approaches the w/o EDL value.",
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
