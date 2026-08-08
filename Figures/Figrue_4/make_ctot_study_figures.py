from __future__ import annotations

import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
SAME_LENGTH_I0_ALPHA_DIR = ROOT / "Figures" / "Figure_same_length_i0_alpha"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SAME_LENGTH_I0_ALPHA_DIR))

import same_length_i0_alpha_common as common  # noqa: E402
from Figures.Figure_scheme.make_edl_vs_no_edl_bv_mixed_potential import (  # noqa: E402
    PALETTE,
    assert_mixed_current,
    compute_signed_half_currents,
    current_at_mix,
    solver,
)


OUT_DIR = ROOT / "Figures" / "Figrue_4"
INPUTS_DIR = OUT_DIR / "inputs"
CSV_DIR = OUT_DIR / "csv"
OUTPUT_TAG = common.OUTPUT_TAG

CURRENT_SCALE = 1.0e9
CURRENT_UNIT = r"10$^{-3}$ uA"
REPRESENTATIVE_C_M = (1.0e-2, 1.0, 1.0e3)
HIGH_SALT_LIGHT_START_M = 1.0
EXTRA_CTOT_SCAN_VALUES_M = (10.0,)

E_MIN = 0.40
E_MAX = 0.63
Y_LIMIT = 0.16
NO_EDL_STYLE = (0, (4.0, 2.6))
TREND_FIGSIZE = (4.15, 3.45)
POLARIZATION_FIGSIZE = (7.2, 4.35)

CASE_STYLES = {
    1.0e-2: {"label": "0.01 M", "linestyle": "solid", "marker": "o"},
    1.0: {"label": "1 M", "linestyle": (0, (5.0, 2.0, 1.2, 2.0)), "marker": "s"},
    1.0e3: {"label": r"$10^3$ M", "linestyle": (0, (1.0, 2.0)), "marker": "^"},
}

TREND_SPECS = {
    "emix": {
        "filename": f"ctot_emix_high_salt_regime_schematic_{OUTPUT_TAG}",
        "y_key_with": "E_with",
        "y_key_no": "E_no",
        "ylabel": r"$E_{\mathrm{mix}}$ (V vs. RHE)",
        "title": r"$E_{\mathrm{mix}}$ shift fades in the high-salt limit",
        "ylim": (0.455, 0.655),
        "label_offsets": {1.0e-2: -0.014, 1.0: 0.010, 1.0e3: 0.009},
        "legend_loc": "lower left",
    },
    "current": {
        "filename": f"ctot_high_salt_regime_schematic_{OUTPUT_TAG}",
        "y_key_with": "I_with",
        "y_key_no": "I_no",
        "ylabel": f"Current ({CURRENT_UNIT})",
        "title": r"$I_{\mathrm{mix}}$ overshoots before the high-salt limit",
        "ylim": (0.017, 0.074),
        "label_offsets": {1.0e-2: -0.009, 1.0: -0.008, 1.0e3: 0.006},
        "legend_loc": "lower right",
    },
}

CTOT_CSV_FIELDS = [
    "C_M",
    "lambda_D_nm",
    "E_with",
    "E_no",
    "I_with",
    "I_no",
    "I_ratio",
    "K_Au_ratio",
    "K_Pd_ratio",
    "delta_E_mV",
    "max_abs_phi_tilde",
]


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


def ensure_output_dirs() -> None:
    for path in (OUT_DIR, INPUTS_DIR, CSV_DIR):
        path.mkdir(parents=True, exist_ok=True)


def ctot_scan_values() -> np.ndarray:
    explicit_values = REPRESENTATIVE_C_M + (HIGH_SALT_LIGHT_START_M,) + EXTRA_CTOT_SCAN_VALUES_M
    values = np.concatenate(
        [
            np.logspace(-4, 3, 33),
            np.asarray(explicit_values, dtype=float),
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


def validate_representative_rows(rows: list[dict[str, float]]) -> None:
    expected = {
        1.0e-2: {"E_with": 0.5979829354430014, "E_no": 0.4670000000000001, "I_ratio": 0.7132},
        1.0: {"E_with": 0.504487, "E_no": 0.4670000000000001, "I_ratio": 1.0689},
        1.0e3: {"E_with": 0.468453, "E_no": 0.4670000000000001, "I_ratio": 1.0041},
    }
    for c_m, values in expected.items():
        row = row_at(rows, c_m)
        for key, expected_value in values.items():
            actual = row[key]
            atol = 5.0e-4 if key == "I_ratio" else 5.0e-5
            if abs(actual - expected_value) > atol:
                raise ValueError(f"{key} drift at C_tot={c_m:g} M: expected {expected_value:.12g}, got {actual:.12g}")


def save_traceability_inputs(params: dict[str, Any], summary: dict[str, str], rows: list[dict[str, float]]) -> None:
    ensure_output_dirs()
    with (INPUTS_DIR / f"params_{OUTPUT_TAG}.json").open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, sort_keys=True)
        f.write("\n")
    with (INPUTS_DIR / f"overrides_{OUTPUT_TAG}.json").open("w", encoding="utf-8") as f:
        json.dump(common.PARAM_OVERRIDES, f, indent=2, sort_keys=True)
        f.write("\n")
    with (INPUTS_DIR / f"summary_compare_{OUTPUT_TAG}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    with (INPUTS_DIR / f"summary_compare_{OUTPUT_TAG}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    with (CSV_DIR / f"ctot_scan_{OUTPUT_TAG}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CTOT_CSV_FIELDS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in CTOT_CSV_FIELDS} for row in rows)


def setup_regime_background(ax: plt.Axes) -> None:
    for xmin, xmax, color in (
        (1.0e-4, 1.0e-2, "#FDE9E0"),
        (1.0e-2, 1.0, "#E9F1E8"),
        (1.0, 1.0e3, "#E8EEF7"),
    ):
        ax.axvspan(xmin, xmax, color=color, alpha=0.55, linewidth=0, zorder=0)


def plot_split_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    linewidth: float,
    linestyle: str | tuple[int, tuple[float, ...]],
    label: str,
    marker: str | None = None,
    markersize: float = 0.0,
) -> None:
    normal = x <= HIGH_SALT_LIGHT_START_M
    formal = x >= HIGH_SALT_LIGHT_START_M
    normal_x = x[normal]
    formal_x = x[formal]
    normal_markevery = None
    formal_markevery = None
    if marker is not None:
        if normal_x.size and np.isclose(normal_x[-1], HIGH_SALT_LIGHT_START_M, rtol=0.0, atol=1.0e-12):
            normal_markevery = slice(None, -1)
        if formal_x.size and np.isclose(formal_x[0], HIGH_SALT_LIGHT_START_M, rtol=0.0, atol=1.0e-12):
            formal_markevery = slice(1, None)
    ax.plot(
        normal_x,
        y[normal],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        marker=marker,
        markersize=markersize,
        markevery=normal_markevery,
        label=label,
        zorder=3,
    )
    ax.plot(
        formal_x,
        y[formal],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        marker=marker,
        markersize=markersize,
        markevery=formal_markevery,
        alpha=0.32,
        label="_nolegend_",
        zorder=3,
    )


def plot_representative_points(
    ax: plt.Axes,
    rows: list[dict[str, float]],
    y_key: str,
    label_offsets: dict[float, float],
) -> None:
    for c_m, label, ha in (
        (1.0e-2, "0.01 M", "center"),
        (1.0, "1 M", "center"),
        (1.0e3, r"$10^3$ M", "right"),
    ):
        row = row_at(rows, c_m)
        y_value = row[y_key]
        y_offset = label_offsets[c_m]
        ax.scatter(
            [row["C_M"]],
            [y_value],
            s=44,
            color="#F26B38",
            edgecolor="white",
            linewidth=0.75,
            zorder=5,
        )
        ax.text(
            row["C_M"],
            y_value + y_offset,
            label,
            ha=ha,
            va="bottom" if y_offset > 0 else "top",
            fontsize=7.2,
            color=PALETTE["dark"],
        )


def plot_trend_figure(rows: list[dict[str, float]], spec_key: str) -> Path:
    spec = TREND_SPECS[spec_key]
    c = np.asarray([row["C_M"] for row in rows], dtype=float)
    y_with = np.asarray([row[spec["y_key_with"]] for row in rows], dtype=float)
    y_no = np.asarray([row[spec["y_key_no"]] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=TREND_FIGSIZE)
    setup_regime_background(ax)
    plot_split_line(
        ax,
        c,
        y_with,
        color="#F26B38",
        linewidth=2.25,
        linestyle="solid",
        marker="o",
        markersize=3.0,
        label="with EDL",
    )
    plot_split_line(
        ax,
        c,
        y_no,
        color="#12355B",
        linewidth=1.8,
        linestyle=(0, (4, 2.5)),
        label="w/o EDL",
    )
    plot_representative_points(ax, rows, str(spec["y_key_with"]), spec["label_offsets"])

    ax.set_xscale("log")
    ax.set_xlim(1.0e-4, 1.0e3)
    ax.set_ylim(*spec["ylim"])
    ax.set_xlabel(r"Electrolyte concentration, $C_{\mathrm{tot}}$ (M)")
    ax.set_ylabel(str(spec["ylabel"]))
    ax.set_title(str(spec["title"]), loc="left", fontsize=9.7, pad=5)
    ax.legend(loc=str(spec["legend_loc"]), fontsize=7.4, handlelength=2.4)
    ax.tick_params(length=3.2, width=0.85, labelsize=8.0)

    out_path = OUT_DIR / f"{spec['filename']}.png"
    for path in (out_path, out_path.with_suffix(".svg")):
        fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    return out_path


def single_period_reactive_area(params: dict[str, Any]) -> float:
    reactive_length = float(params["L_Au"]) + float(params["L_Pd_len"])
    return reactive_length * float(params.get("out_of_plane_width", 1.0))


def current_density_to_display_current(params: dict[str, Any], current_density: np.ndarray | float) -> np.ndarray:
    return np.asarray(current_density, dtype=float) * single_period_reactive_area(params) * CURRENT_SCALE


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
        "I_with": float(with_edl["i_mix_abs_A"]) * CURRENT_SCALE,
        "I_no": float(no_edl["i_mix_abs_A"]) * CURRENT_SCALE,
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


def plot_polarization_overlay(base_params: dict[str, Any]) -> Path:
    e_values = np.linspace(E_MIN, E_MAX, 1300)
    cases = compute_cases(base_params, e_values)

    fig, ax = plt.subplots(figsize=POLARIZATION_FIGSIZE)
    plot_reference_curves(ax, e_values, cases[0])
    for case in cases:
        plot_case_curves(ax, e_values, case)
    ax.axhline(0.0, color=PALETTE["dark"], linewidth=0.85, zorder=0)
    add_legends(ax)

    ax.set_xlim(E_MIN, E_MAX)
    ax.set_ylim(-Y_LIMIT, Y_LIMIT)
    ax.set_xlabel("Potential (V vs. RHE)")
    ax.set_ylabel(f"Half-reaction current ({CURRENT_UNIT})")
    ax.set_title(r"Salt-dependent polarization curves explain $I_{\mathrm{mix}}$", loc="left", fontsize=9.7, pad=5)
    ax.tick_params(length=3.2, width=0.85, labelsize=8.0)

    out_path = OUT_DIR / f"ctot_half_reaction_polarization_overlay_{OUTPUT_TAG}.png"
    for path in (out_path, out_path.with_suffix(".svg")):
        fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ensure_output_dirs()
    apply_style()

    params = common.load_same_length_i0_alpha_params()
    pair = solver.run_edl_comparison_pair(params, mode="FULL")
    summary = common.summary_from_pair(pair)
    common.assert_expected_values(summary)

    rows = compute_ctot_rows(params, ctot_scan_values())
    validate_representative_rows(rows)
    save_traceability_inputs(params, summary, rows)

    outputs = [
        plot_trend_figure(rows, "emix"),
        plot_trend_figure(rows, "current"),
        plot_polarization_overlay(params),
    ]

    for path in outputs:
        print(f"Saved {path.relative_to(ROOT)}")
        print(f"Saved {path.with_suffix('.svg').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
