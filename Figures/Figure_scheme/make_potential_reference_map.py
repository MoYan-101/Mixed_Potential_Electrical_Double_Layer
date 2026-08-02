from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULT_ID = "20260528_111255"
PARAMS_PATH = ROOT / "Mixed_Potential_Electrical_Double_Layer" / "results" / RESULT_ID / "params.json"
SUMMARY_PATH = ROOT / "Mixed_Potential_Electrical_Double_Layer" / "results" / RESULT_ID / "csv" / "summary_compare.csv"
OUT_DIR = Path(__file__).resolve().parent


PALETTE = {
    "blue": "#0F4D92",
    "blue_light": "#3775BA",
    "red": "#B64342",
    "green": "#3B7A57",
    "gold": "#C9A227",
    "gray": "#767676",
    "dark": "#272727",
    "light_gray": "#CFCECE",
}


def load_inputs() -> tuple[dict[str, float], dict[str, str]]:
    with PARAMS_PATH.open("r", encoding="utf-8") as f:
        params = json.load(f)
    with SUMMARY_PATH.open("r", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    return params, row


def fmt_v(value: float) -> str:
    return f"{value:.2f} V"


def add_marker(
    ax: plt.Axes,
    x: float,
    y: float,
    label: str,
    value: float,
    color: str,
    marker: str,
    text_y_offset: float,
    linestyle: str = "-",
    linewidth: float = 2.0,
    alpha: float = 1.0,
    text_x_offset: float = 0.0,
) -> None:
    ax.vlines(x, y - 0.10, y + 0.10, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, zorder=2)
    ax.scatter([x], [y], s=95, marker=marker, color=color, edgecolor="white", linewidth=0.8, alpha=alpha, zorder=4)
    va = "bottom" if text_y_offset >= 0 else "top"
    ax.text(
        x + text_x_offset,
        y + text_y_offset,
        f"{label}\n{fmt_v(value)}",
        ha="center",
        va=va,
        fontsize=9.5,
        color=color,
        linespacing=1.15,
    )


def main() -> None:
    params, summary = load_inputs()

    values = {
        "E1_eq": float(summary["E1_eq_eff"]),
        "E2_eq": float(summary["E2_eq_eff"]),
        "E_mix_with": float(summary["E_mix_with"]),
        "E_mix_no": float(summary["E_mix_no"]),
        "pzc_Au": float(params["pzc_Au"]),
        "pzc_C": float(params["pzc_C"]),
        "pzc_Pd": float(params["pzc_Pd"]),
    }

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.1,
            "svg.fonttype": "none",
            "legend.frameon": False,
        }
    )

    fig, ax = plt.subplots(figsize=(7.25, 3.15))

    xmin, xmax = 0.0, 1.0
    y_axis = 0.0
    ax.hlines(y_axis, xmin, xmax, color=PALETTE["dark"], linewidth=1.4, zorder=1)

    lane_y = {
        "eq": 0.28,
        "mix": 0.00,
        "pzc": -0.28,
    }

    add_marker(ax, values["E1_eq"], lane_y["eq"], r"$E_{1,\mathrm{eq}}$", values["E1_eq"], PALETTE["blue"], "o", 0.13)
    add_marker(ax, values["E2_eq"], lane_y["eq"], r"$E_{2,\mathrm{eq}}$", values["E2_eq"], PALETTE["blue"], "o", 0.13)

    add_marker(
        ax,
        values["E_mix_no"],
        lane_y["mix"],
        r"$E_{\mathrm{mix}}$ w/o EDL",
        values["E_mix_no"],
        PALETTE["gray"],
        "D",
        -0.17,
        linestyle=(0, (3, 2)),
        linewidth=1.5,
        alpha=0.85,
        text_x_offset=-0.065,
    )
    add_marker(
        ax,
        values["E_mix_with"],
        lane_y["mix"],
        r"$E_{\mathrm{mix}}$ with EDL",
        values["E_mix_with"],
        PALETTE["red"],
        "D",
        0.16,
        linewidth=2.4,
    )

    add_marker(ax, values["pzc_C"], lane_y["pzc"], "PZC support", values["pzc_C"], PALETTE["green"], "^", -0.14, text_x_offset=0.03)
    add_marker(ax, values["pzc_Pd"], lane_y["pzc"], "PZC Pd", values["pzc_Pd"], PALETTE["gold"], "^", -0.14, text_x_offset=-0.02)
    add_marker(ax, values["pzc_Au"], lane_y["pzc"], "PZC Au", values["pzc_Au"], PALETTE["green"], "^", -0.14)

    ax.text(-0.015, lane_y["eq"], "Equilibrium\npotentials", ha="right", va="center", fontsize=9.5, color=PALETTE["dark"])
    ax.text(-0.015, lane_y["mix"], "Mixed\npotential", ha="right", va="center", fontsize=9.5, color=PALETTE["dark"])
    ax.text(-0.015, lane_y["pzc"], "PZC", ha="right", va="center", fontsize=9.5, color=PALETTE["dark"])

    ax.set_xlim(xmin - 0.05, xmax + 0.02)
    ax.set_ylim(-0.72, 0.62)
    ax.set_xlabel("Potential (V vs. RHE)")
    ax.set_yticks([])
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.tick_params(axis="x", length=4, width=1.0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_position(("data", -0.58))
    ax.set_title(f"Relative potential positions used for {RESULT_ID}", loc="left", fontsize=11.5, pad=8)

    fig.tight_layout(pad=1.0)
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"potential_reference_map_{RESULT_ID}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
