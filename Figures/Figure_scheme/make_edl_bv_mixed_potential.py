from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT_ID = "20260528_111255"
SOLVER_DIR = ROOT / "Mixed_Potential_Electrical_Double_Layer"
PARAMS_PATH = SOLVER_DIR / "results" / RESULT_ID / "params.json"
SUMMARY_PATH = SOLVER_DIR / "results" / RESULT_ID / "csv" / "summary_compare.csv"
OUT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(SOLVER_DIR))
import Solve_Emix_updating as solver  # noqa: E402


PALETTE = {
    "blue": "#0F4D92",
    "red": "#B64342",
    "orange": "#E65F2A",
    "green": "#3B7A57",
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


def average_current_density(params: dict[str, float], lambda_d: float, current: np.ndarray) -> np.ndarray:
    reactive_length = float(params["L_Au"]) + float(params["L_Pd_len"])
    return lambda_d * current / reactive_length


def add_potential_marker(
    ax: plt.Axes,
    x: float,
    label: str,
    color: str,
    y_text: float,
    marker: str = "o",
    text_x_offset: float = 0.0,
) -> None:
    ax.scatter([x], [0.0], s=64, color=color, marker=marker, edgecolor="white", linewidth=0.8, zorder=5)
    ax.text(
        x + text_x_offset,
        y_text,
        f"{label}\n{fmt_v(x)}",
        ha="center",
        va="center",
        fontsize=9.5,
        color=color,
        linespacing=1.15,
    )


def main() -> None:
    params, summary = load_inputs()

    values = {
        "E1_eq": float(summary["E1_eq_eff"]),
        "E2_eq": float(summary["E2_eq_eff"]),
        "E_mix": float(summary["E_mix_with"]),
        "i_mix_avg": float(summary["i_mix_avg_with"]),
    }

    use_affine_phi2 = bool(params.get("use_affine_phi2", True))
    derived = solver.compute_derived_params(params)
    lambda_d = float(derived["lambda_D"])

    e_min = values["E1_eq"] - 0.06
    e_max = values["E2_eq"] + 0.06
    e_values = np.linspace(e_min, e_max, 1200)
    curve = solver.compute_polarization_curve(
        params,
        mode="FULL",
        use_edl=True,
        E_values=e_values,
        use_affine_phi2=use_affine_phi2,
    )

    au_current = average_current_density(params, lambda_d, curve["I_Au"])
    pd_current = average_current_density(params, lambda_d, curve["I_Pd"])

    mix_curve = solver.compute_polarization_curve(
        params,
        mode="FULL",
        use_edl=True,
        E_values=np.array([values["E_mix"]], dtype=float),
        use_affine_phi2=use_affine_phi2,
    )
    i_au_mix = float(average_current_density(params, lambda_d, mix_curve["I_Au"])[0])
    i_pd_mix = float(average_current_density(params, lambda_d, mix_curve["I_Pd"])[0])
    if abs(i_au_mix - values["i_mix_avg"]) > 1e-8:
        raise ValueError(
            f"Computed i_mix_avg {i_au_mix:.12g} differs from summary {values['i_mix_avg']:.12g}"
        )
    if abs(i_au_mix + i_pd_mix) > 1e-8:
        raise ValueError(f"Current balance failed at E_mix: I_Au + I_Pd = {i_au_mix + i_pd_mix:.12g}")

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

    fig, ax = plt.subplots(figsize=(7.35, 4.15))

    y_limit = max(0.13, 2.35 * values["i_mix_avg"])
    ax.plot(
        curve["E"],
        au_current,
        color=PALETTE["green"],
        linewidth=2.6,
        label=r"Au anodic half-reaction, $i_{\mathrm{an}}$",
        clip_on=True,
        zorder=3,
    )
    ax.plot(
        curve["E"],
        pd_current,
        color=PALETTE["orange"],
        linewidth=2.6,
        label=r"Pd cathodic half-reaction, $i_{\mathrm{ca}}$",
        clip_on=True,
        zorder=3,
    )

    ax.axhline(0.0, color=PALETTE["dark"], linewidth=1.25, zorder=1)
    ax.axvline(values["E_mix"], color=PALETTE["dark"], linestyle=(0, (3, 2)), linewidth=1.2, zorder=2)
    ax.scatter([values["E_mix"]], [0.0], s=72, color=PALETTE["dark"], edgecolor="white", linewidth=0.8, zorder=6)

    ax.annotate(
        "",
        xy=(values["E_mix"], i_au_mix),
        xytext=(values["E_mix"], 0.0),
        arrowprops=dict(arrowstyle="->", color=PALETTE["green"], linewidth=1.6),
        zorder=5,
    )
    ax.annotate(
        "",
        xy=(values["E_mix"], i_pd_mix),
        xytext=(values["E_mix"], 0.0),
        arrowprops=dict(arrowstyle="->", color=PALETTE["orange"], linewidth=1.6),
        zorder=5,
    )
    ax.hlines(
        [i_au_mix, i_pd_mix],
        values["E1_eq"],
        values["E_mix"],
        colors=[PALETTE["green"], PALETTE["orange"]],
        linestyles=(0, (3, 3)),
        linewidth=1.2,
        alpha=0.85,
        zorder=2,
    )
    ax.scatter(
        [values["E_mix"], values["E_mix"]],
        [i_au_mix, i_pd_mix],
        s=58,
        color=[PALETTE["green"], PALETTE["orange"]],
        edgecolor="white",
        linewidth=0.7,
        zorder=6,
    )

    add_potential_marker(ax, values["E1_eq"], r"$E_{1,\mathrm{eq}}$", PALETTE["blue"], 0.31 * y_limit)
    add_potential_marker(
        ax,
        values["E2_eq"],
        r"$E_{2,\mathrm{eq}}$",
        PALETTE["blue"],
        -0.31 * y_limit,
        text_x_offset=-0.015,
    )

    ax.text(
        values["E_mix"] + 0.012,
        0.48 * y_limit,
        rf"$E_{{\mathrm{{mix}}}}={fmt_v(values['E_mix'])}$",
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=PALETTE["dark"],
    )
    ax.text(
        values["E_mix"] + 0.012,
        i_au_mix,
        rf"$|i_{{\mathrm{{an}}}}|=|i_{{\mathrm{{ca}}}}|={i_au_mix:.3f}$ A m$^{{-2}}$",
        ha="left",
        va="center",
        fontsize=9.5,
        color=PALETTE["dark"],
    )
    ax.text(
        values["E1_eq"] + 0.035,
        0.72 * y_limit,
        r"$\mathrm{Red}_1 \rightarrow \mathrm{Ox}_1 + ne^-$",
        color=PALETTE["green"],
        fontsize=11,
        ha="left",
        va="center",
    )
    ax.text(
        values["E2_eq"] - 0.035,
        -0.72 * y_limit,
        r"$\mathrm{Ox}_2 + ne^- \rightarrow \mathrm{Red}_2$",
        color=PALETTE["orange"],
        fontsize=11,
        ha="right",
        va="center",
    )

    ax.set_xlim(e_min, e_max)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xlabel("Potential (V vs. RHE)")
    ax.set_ylabel(r"Average half-reaction current density (A m$^{-2}$)")
    ax.set_xticks([0.10, 0.30, 0.50, 0.70, 0.90])
    ax.tick_params(axis="both", length=4, width=1.0)
    ax.set_title(f"EDL-corrected Butler-Volmer mixed potential ({RESULT_ID})", loc="left", fontsize=11.5, pad=8)
    ax.legend(loc="upper left", fontsize=9.0, handlelength=2.6)

    fig.tight_layout(pad=1.0)
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"edl_bv_mixed_potential_{RESULT_ID}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"E_mix_with = {values['E_mix']:.15g} V")
    print(f"i_mix_avg_with = {i_au_mix:.15g} A/m^2")
    print(f"Saved {OUT_DIR / f'edl_bv_mixed_potential_{RESULT_ID}.png'}")
    print(f"Saved {OUT_DIR / f'edl_bv_mixed_potential_{RESULT_ID}.svg'}")


if __name__ == "__main__":
    main()
