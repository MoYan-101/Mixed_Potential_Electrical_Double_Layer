from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
RESULT_ID = "20260528_111255"
SOLVER_DIR = ROOT / "Mixed_Potential_Electrical_Double_Layer"
RESULT_DIR = SOLVER_DIR / "results" / RESULT_ID
PARAMS_PATH = RESULT_DIR / "params.json"
SUMMARY_PATH = RESULT_DIR / "csv" / "summary_compare.csv"
OUT_DIR = Path(__file__).resolve().parent
OUT_STEM = "phi_s_reactants_2d"

sys.path.insert(0, str(SOLVER_DIR))
import Solve_Emix_updating as solver  # noqa: E402


COLORS = {
    "dark": "#272727",
    "gray": "#767676",
    "light_gray": "#CFCECE",
    "au": "#C9A227",
    "support": "#8C8C8C",
    "pd": "#42949E",
}


@dataclass(frozen=True)
class RP2DData:
    params: dict[str, Any]
    summary: dict[str, str]
    res_edl: dict[str, Any]
    x_nm: np.ndarray
    y_nm: np.ndarray
    phi_tilde: np.ndarray
    phi_s_mV: np.ndarray
    c_r1_norm: np.ndarray
    c_o2_norm: np.ndarray
    lambda_D_nm: float
    L_Au_nm: float
    L_C_nm: float
    L_total_nm: float


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 8.5,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.9,
            "axes.grid": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Nimbus Sans",
            "mathtext.it": "Nimbus Sans:italic",
            "mathtext.bf": "Nimbus Sans:bold",
            "mathtext.cal": "Nimbus Sans",
            "mathtext.sf": "Nimbus Sans",
            "mathtext.tt": "Nimbus Sans",
        }
    )


def load_params() -> dict[str, Any]:
    with PARAMS_PATH.open("r", encoding="utf-8") as f:
        params_data = json.load(f)
    return solver.apply_param_overrides(solver.default_params(), params_data, reset_lambda_D=False)


def load_summary() -> dict[str, str]:
    with SUMMARY_PATH.open("r", encoding="utf-8") as f:
        return next(csv.DictReader(f))


def assert_close(label: str, actual: float, expected: float, atol: float) -> None:
    if abs(actual - expected) > atol:
        raise ValueError(f"{label}: recomputed {actual:.15g} differs from summary {expected:.15g}")


def validate_against_summary(res_edl: dict[str, Any], summary: dict[str, str]) -> None:
    assert_close("E_mix_with", float(res_edl["E_mix"]), float(summary["E_mix_with"]), 1e-10)
    assert_close(
        "i_mix_avg_with",
        float(res_edl["i_mix_avg_A_per_m2"]),
        float(summary["i_mix_avg_with"]),
        1e-8,
    )


def validate_finite_positive(label: str, values: np.ndarray, positive: bool = False) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} contains non-finite values")
    if positive and not np.all(values > 0.0):
        raise ValueError(f"{label} must be strictly positive")


def build_2d_data(n_y: int = 320) -> RP2DData:
    params = load_params()
    summary = load_summary()

    pair = solver.run_edl_comparison_pair(params, mode="FULL")
    res_edl = pair["with_edl"]
    validate_against_summary(res_edl, summary)

    edl = solver.EDLModel(params)
    E_mix = float(res_edl["E_mix"])
    beta = float(edl.derived["beta"])
    scale = float(edl.derived["R"]) * float(edl.derived["T"]) / float(edl.derived["F"])

    x_tilde = np.asarray(edl.pre["x_tilde"], dtype=float)
    rho = np.asarray(edl.pre["rho"], dtype=float)
    gamma = np.asarray(edl.pre["gamma"], dtype=float)
    A = np.asarray(edl.pre["A_M"], dtype=float) * beta * E_mix - np.asarray(edl.pre["A_pzc"], dtype=float)

    lambda_D = float(edl.derived["lambda_D"])
    lambda_D_nm = lambda_D * 1e9
    y_nm = np.linspace(0.0, 5.0 * lambda_D_nm, n_y, dtype=float)
    y_tilde = y_nm / lambda_D_nm

    cos_modes = np.cos(np.outer(rho, x_tilde))
    decay_modes = A[:, None] * np.exp(-gamma[:, None] * y_tilde[None, :])
    phi_tilde = decay_modes.T @ cos_modes
    phi_s_mV = 1000.0 * scale * phi_tilde

    prof_edl, _ = solver.build_profiles_for_emix(params, E_mix, use_edl=True)
    profile_x = np.asarray(prof_edl["x_tilde"], dtype=float)
    profile_phi_tilde = np.asarray(prof_edl["phi_tilde"], dtype=float)
    if not np.allclose(x_tilde, profile_x, rtol=0.0, atol=1e-14):
        raise ValueError("2D grid x_tilde does not match the Figure 3 reaction-plane grid")
    if not np.allclose(phi_tilde[0, :], profile_phi_tilde, rtol=0.0, atol=1e-11):
        raise ValueError("Phi_s(x, y=0) does not match the Figure 3 reaction-plane potential")

    c_r1_norm = np.asarray(solver.safe_exp(-float(params["z_R1"]) * phi_tilde), dtype=float)
    c_o2_norm = np.asarray(solver.safe_exp(-float(params["z_O2"]) * phi_tilde), dtype=float)

    validate_finite_positive("Phi_s", phi_s_mV)
    validate_finite_positive("c_R1/c_bulk", c_r1_norm, positive=True)
    validate_finite_positive("c_O2/c_bulk", c_o2_norm, positive=True)

    return RP2DData(
        params=params,
        summary=summary,
        res_edl=res_edl,
        x_nm=x_tilde * lambda_D_nm,
        y_nm=y_nm,
        phi_tilde=phi_tilde,
        phi_s_mV=phi_s_mV,
        c_r1_norm=c_r1_norm,
        c_o2_norm=c_o2_norm,
        lambda_D_nm=lambda_D_nm,
        L_Au_nm=float(edl.derived["L_Au_tilde"]) * lambda_D_nm,
        L_C_nm=float(edl.derived["L_C_tilde"]) * lambda_D_nm,
        L_total_nm=float(edl.derived["L_tilde"]) * lambda_D_nm,
    )


def style_map_axis(ax: plt.Axes, title: str, show_xlabel: bool = False) -> None:
    ax.set_ylabel("y (nm)")
    if show_xlabel:
        ax.set_xlabel("x (nm)")
    else:
        ax.tick_params(labelbottom=False)
    ax.set_title(title, loc="left", pad=5, fontsize=9.6, fontweight="normal")
    ax.tick_params(length=3.2, width=0.85, pad=2.5, labelsize=8.0)
    for spine in ("left", "bottom", "top", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.9)
        ax.spines[spine].set_color(COLORS["dark"])


def add_region_boundaries(ax: plt.Axes, data: RP2DData) -> None:
    positions: list[float] = []
    for xpos in (data.L_Au_nm, data.L_C_nm):
        if xpos <= 0.0 or xpos >= data.L_total_nm:
            continue
        if not any(math.isclose(xpos, existing, rel_tol=0.0, abs_tol=1e-9) for existing in positions):
            positions.append(xpos)
    for xpos in positions:
        ax.axvline(xpos, linestyle=(0, (3, 2)), linewidth=0.8, color=COLORS["gray"], alpha=0.8, zorder=5)


def log_norm(values: np.ndarray) -> LogNorm:
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        raise ValueError("Cannot build LogNorm from empty positive data")
    vmin = 10.0 ** math.floor(math.log10(float(np.min(positive))))
    vmax = 10.0 ** math.ceil(math.log10(float(np.max(positive))))
    return LogNorm(vmin=vmin, vmax=vmax)


def log_contour_levels(values: np.ndarray, count: int = 8) -> np.ndarray:
    positive = values[np.isfinite(values) & (values > 0.0)]
    return np.logspace(math.log10(float(np.min(positive))), math.log10(float(np.max(positive))), count)


def add_material_lane(ax: plt.Axes, data: RP2DData) -> None:
    segments = [
        ("Au", 0.0, data.L_Au_nm, COLORS["au"], COLORS["dark"]),
        ("support", data.L_Au_nm, data.L_C_nm, COLORS["support"], "white"),
        ("Pd", data.L_C_nm, data.L_total_nm, COLORS["pd"], "white"),
    ]
    for label, x0, x1, face, text_color in segments:
        if x1 - x0 <= 1e-9:
            continue
        ax.add_patch(Rectangle((x0, 0.0), x1 - x0, 1.0, facecolor=face, edgecolor="white", linewidth=0.8))
        ax.text((x0 + x1) / 2.0, 0.5, label, ha="center", va="center", fontsize=8.3, color=text_color)
    ax.set_xlim(0.0, data.L_total_nm)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()


def add_heatmap(
    ax: plt.Axes,
    cax: plt.Axes,
    data: RP2DData,
    values: np.ndarray,
    *,
    cmap: str,
    norm: LogNorm | TwoSlopeNorm,
    cbar_label: str,
    title: str,
    show_xlabel: bool = False,
    contour_levels: np.ndarray | None = None,
) -> None:
    mesh = ax.pcolormesh(data.x_nm, data.y_nm, values, shading="auto", cmap=cmap, norm=norm, rasterized=True)
    if contour_levels is not None:
        ax.contour(data.x_nm, data.y_nm, values, levels=contour_levels, colors="black", linewidths=0.28, alpha=0.28)
    add_region_boundaries(ax, data)
    style_map_axis(ax, title, show_xlabel=show_xlabel)
    cbar = ax.figure.colorbar(mesh, cax=cax)
    cbar.set_label(cbar_label, labelpad=5)
    cbar.ax.tick_params(length=2.8, width=0.75, labelsize=7.8, pad=2.2)
    cbar.outline.set_linewidth(0.8)


def save_figure(fig: plt.Figure) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for ext in ("png", "svg"):
        path = OUT_DIR / f"{OUT_STEM}_{RESULT_ID}.{ext}"
        fig.savefig(path, dpi=600, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)

    if (OUT_DIR / f"{OUT_STEM}_{RESULT_ID}.pdf").exists():
        raise RuntimeError("Unexpected PDF output exists for Figure_RP")
    if len(saved) != 2:
        raise RuntimeError(f"Expected 2 exported files, got {len(saved)}")
    return saved


def plot_phi_s_reactants(data: RP2DData) -> list[Path]:
    fig = plt.figure(figsize=(5.8, 6.75))
    gs = fig.add_gridspec(
        nrows=4,
        ncols=2,
        width_ratios=(1.0, 0.038),
        height_ratios=(1.0, 1.0, 1.0, 0.12),
        hspace=0.20,
        wspace=0.08,
    )
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    caxes = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    lane_ax = fig.add_subplot(gs[3, 0], sharex=axes[-1])
    fig.add_subplot(gs[3, 1]).set_axis_off()

    phi_abs = float(np.nanmax(np.abs(data.phi_s_mV)))
    phi_vlim = max(10.0, math.ceil(phi_abs / 10.0) * 10.0)
    phi_levels = np.linspace(-phi_vlim, phi_vlim, 11)

    add_heatmap(
        axes[0],
        caxes[0],
        data,
        data.phi_s_mV,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-phi_vlim, vcenter=0.0, vmax=phi_vlim),
        cbar_label=r"$\Phi_s$ (mV)",
        title=rf"Solution phase potential, $E_{{\mathrm{{mix}}}}$ = {float(data.res_edl['E_mix']):.2f} V",
        contour_levels=phi_levels,
    )
    add_heatmap(
        axes[1],
        caxes[1],
        data,
        data.c_r1_norm,
        cmap="viridis",
        norm=log_norm(data.c_r1_norm),
        cbar_label=r"$c_{\mathrm{R1}}/c_{\mathrm{bulk}}$",
        title=r"Reactant R1 distribution",
        contour_levels=log_contour_levels(data.c_r1_norm),
    )
    add_heatmap(
        axes[2],
        caxes[2],
        data,
        data.c_o2_norm,
        cmap="viridis",
        norm=log_norm(data.c_o2_norm),
        cbar_label=r"$c_{\mathrm{O2}}/c_{\mathrm{bulk}}$",
        title=r"Reactant O2 distribution",
        show_xlabel=True,
        contour_levels=log_contour_levels(data.c_o2_norm),
    )

    for ax in axes:
        ax.set_xlim(0.0, data.L_total_nm)
        ax.set_ylim(0.0, data.y_nm[-1])
        ax.set_yticks([0.0, 5.0, 10.0, 15.0])
    axes[-1].set_xticks([0.0, 11.0, 21.0, 40.0, 58.0])

    add_material_lane(lane_ax, data)
    fig.align_ylabels(axes)
    return save_figure(fig)


def main() -> None:
    apply_style()
    data = build_2d_data()
    saved = plot_phi_s_reactants(data)

    if len(list(OUT_DIR.glob(f"{OUT_STEM}_{RESULT_ID}.png"))) != 1:
        raise RuntimeError("Expected exactly one PNG output")
    if len(list(OUT_DIR.glob(f"{OUT_STEM}_{RESULT_ID}.svg"))) != 1:
        raise RuntimeError("Expected exactly one SVG output")
    if len(list(OUT_DIR.glob(f"{OUT_STEM}_{RESULT_ID}.pdf"))) != 0:
        raise RuntimeError("Expected zero PDF outputs")

    print("Verified recomputed values against summary_compare.csv")
    print(f"E_mix_with = {float(data.res_edl['E_mix']):.15g} V")
    print(f"i_mix_avg_with = {float(data.res_edl['i_mix_avg_A_per_m2']):.15g} A/m^2")
    print(f"lambda_D = {data.lambda_D_nm:.6g} nm")
    print(f"max |phi_tilde| = {float(np.max(np.abs(data.phi_tilde))):.6g}")
    print(f"Phi_s range = {float(np.min(data.phi_s_mV)):.6g} to {float(np.max(data.phi_s_mV)):.6g} mV")
    print(f"c_R1/c_bulk range = {float(np.min(data.c_r1_norm)):.6g} to {float(np.max(data.c_r1_norm)):.6g}")
    print(f"c_O2/c_bulk range = {float(np.min(data.c_o2_norm)):.6g} to {float(np.max(data.c_o2_norm)):.6g}")
    print("Saved Figure_RP outputs:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
