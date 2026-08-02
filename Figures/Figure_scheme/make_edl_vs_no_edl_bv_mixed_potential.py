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
    "orange": "#B64342",
    "green": "#3B7A57",
    "gray": "#767676",
    "dark": "#272727",
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


def compute_signed_half_currents(
    params: dict[str, float],
    lambda_d: float,
    e_values: np.ndarray,
    use_edl: bool,
    use_affine_phi2: bool,
) -> tuple[np.ndarray, np.ndarray]:
    curve = solver.compute_polarization_curve(
        params,
        mode="FULL",
        use_edl=use_edl,
        E_values=e_values,
        use_affine_phi2=use_affine_phi2,
    )
    au_current = average_current_density(params, lambda_d, curve["I_Au"])
    pd_current = average_current_density(params, lambda_d, curve["I_Pd"])
    return au_current, pd_current


def current_at_mix(
    params: dict[str, float],
    lambda_d: float,
    e_mix: float,
    use_edl: bool,
    use_affine_phi2: bool,
) -> tuple[float, float]:
    au_current, pd_current = compute_signed_half_currents(
        params,
        lambda_d,
        np.array([e_mix], dtype=float),
        use_edl=use_edl,
        use_affine_phi2=use_affine_phi2,
    )
    return float(au_current[0]), float(pd_current[0])


def assert_mixed_current(
    label: str,
    i_au: float,
    i_pd: float,
    expected_i_mix: float,
    atol: float = 1e-8,
) -> None:
    if abs(i_au - expected_i_mix) > atol:
        raise ValueError(f"{label}: computed i_mix_avg {i_au:.12g} differs from summary {expected_i_mix:.12g}")
    if abs(i_au + i_pd) > atol:
        raise ValueError(f"{label}: current balance failed at E_mix: I_Au + I_Pd = {i_au + i_pd:.12g}")


def add_potential_marker(
    ax: plt.Axes,
    x: float,
    label: str,
    color: str,
    y_text: float,
    marker: str = "o",
    text_x_offset: float = 0.0,
) -> None:
    ax.scatter([x], [0.0], s=42, color=color, marker=marker, edgecolor="white", linewidth=0.6, zorder=6)
    ax.text(
        x + text_x_offset,
        y_text,
        f"{label}\n{fmt_v(x)}",
        ha="center",
        va="center",
        fontsize=8.0,
        color=color,
        linespacing=1.15,
    )


def add_mixed_current_annotations(
    ax: plt.Axes,
    e_mix: float,
    i_au: float,
    i_pd: float,
    color: str,
    linestyle: str | tuple[int, tuple[int, ...]],
    alpha: float,
    zorder: int,
) -> None:
    arrow_common = dict(arrowstyle="->", linewidth=1.1, alpha=alpha, linestyle=linestyle)
    ax.annotate(
        "",
        xy=(e_mix, i_au),
        xytext=(e_mix, 0.0),
        arrowprops={**arrow_common, "color": PALETTE["green"]},
        zorder=zorder,
    )
    ax.annotate(
        "",
        xy=(e_mix, i_pd),
        xytext=(e_mix, 0.0),
        arrowprops={**arrow_common, "color": PALETTE["orange"]},
        zorder=zorder,
    )
    ax.scatter(
        [e_mix, e_mix],
        [i_au, i_pd],
        s=36,
        color=[PALETTE["green"], PALETTE["orange"]],
        edgecolor="white",
        linewidth=0.5,
        alpha=alpha,
        zorder=zorder + 1,
    )
    ax.axvline(e_mix, color=color, linestyle=linestyle, linewidth=0.95, alpha=alpha, zorder=zorder - 1)


def main() -> None:
    params, summary = load_inputs()

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

    e_min = values["E1_eq"] - 0.06
    e_max = values["E2_eq"] + 0.06
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

    fig, ax = plt.subplots(figsize=(6.3, 3.6))

    y_limit = max(0.13, 2.35 * max(values["i_mix_avg_with"], values["i_mix_avg_no"]))
    no_edl_style = (0, (4, 3))

    ax.plot(
        e_values,
        au_edl,
        color=PALETTE["green"],
        linewidth=2.2,
        label=r"Au anodic, with EDL",
        zorder=4,
    )
    ax.plot(
        e_values,
        pd_edl,
        color=PALETTE["orange"],
        linewidth=2.2,
        label=r"Pd cathodic, with EDL",
        zorder=4,
    )
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
    ax.set_xlabel("Potential (V vs. RHE)", fontsize=9.0)
    ax.set_ylabel(r"Average half-reaction current density (A m$^{-2}$)", fontsize=9.0)
    ax.set_xticks([0.10, 0.30, 0.50, 0.70, 0.90])
    ax.tick_params(axis="both", length=3.2, width=0.85, labelsize=8.0)
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

    fig.tight_layout(pad=1.0)
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"edl_vs_no_edl_bv_mixed_potential_{RESULT_ID}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"E_mix_with = {values['E_mix_with']:.15g} V")
    print(f"E_mix_no = {values['E_mix_no']:.15g} V")
    print(f"i_mix_avg_with = {i_au_with:.15g} A/m^2")
    print(f"i_mix_avg_no = {i_au_no:.15g} A/m^2")
    print(f"Saved {OUT_DIR / f'edl_vs_no_edl_bv_mixed_potential_{RESULT_ID}.png'}")
    print(f"Saved {OUT_DIR / f'edl_vs_no_edl_bv_mixed_potential_{RESULT_ID}.svg'}")


if __name__ == "__main__":
    main()
