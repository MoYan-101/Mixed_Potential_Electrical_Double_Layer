from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_TAG = "same_length_i0_alpha050_au25_pd25_20260528_111255"

HEATMAP_INPUTS_DIR = ROOT / "Figures" / "Figure_5" / "Heatmap" / "inputs"
PARAMS_PATH = HEATMAP_INPUTS_DIR / f"params_{OUTPUT_TAG}.json"
SUMMARY_PATH = HEATMAP_INPUTS_DIR / f"summary_compare_{OUTPUT_TAG}.json"

OUT_DIR = Path(__file__).resolve().parent
PNG_OUT = OUT_DIR / f"pzc_scan_range_reference_map_{OUTPUT_TAG}.png"
SVG_OUT = OUT_DIR / f"pzc_scan_range_reference_map_{OUTPUT_TAG}.svg"
PNG_PROJECTED_OUT = OUT_DIR / f"pzc_scan_range_reference_map_projected_band_{OUTPUT_TAG}.png"
SVG_PROJECTED_OUT = OUT_DIR / f"pzc_scan_range_reference_map_projected_band_{OUTPUT_TAG}.svg"

COLORS = {
    "dark": "#272727",
    "gray": "#767676",
    "light_gray": "#CFCECE",
    "pzc_support": "#8C8C8C",
    "pzc_pd": "#5A90C8",
    "pzc_au": "#E4C133",
}

FIGSIZE = (5.35, 2.95)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.edgecolor": "none",
            "savefig.transparent": True,
            "savefig.bbox": None,
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


def make_transparent(fig: plt.Figure) -> None:
    fig.patch.set_alpha(0.0)
    for ax in fig.axes:
        ax.patch.set_alpha(0.0)


def fmt_v(value: float) -> str:
    return f"{value:.2f} V"


def validate_inputs(params: dict[str, Any], summary: dict[str, Any]) -> None:
    required_params = (
        "pzc_C",
        "pzc_Pd",
        "pzc_Au",
        "heatmap_pzc_minus_span",
        "heatmap_pzc_plus_span",
    )
    required_summary = ("E1_eq_eff", "E2_eq_eff")
    missing = [key for key in required_params if key not in params]
    missing.extend(key for key in required_summary if key not in summary)
    if missing:
        raise KeyError(f"Missing required input keys: {missing}")


def gradient_scan_band(
    ax: plt.Axes,
    *,
    low: float,
    base: float,
    high: float,
    y: float,
    color: str,
    label: str,
    label_y_offset: float,
) -> None:
    if not low < base < high:
        raise ValueError(f"Need low < base < high for {label}: {low}, {base}, {high}")

    height = 0.115
    xs = np.linspace(low, high, 900)
    left_span = base - low
    right_span = high - base
    distance = np.where(xs <= base, (base - xs) / left_span, (xs - base) / right_span)
    distance = np.clip(distance, 0.0, 1.0)
    strength = 0.10 + 0.82 * (1.0 - distance) ** 1.15
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    rgba = np.zeros((18, xs.size, 4), dtype=float)
    rgba[:, :, :3] = 1.0 - strength[np.newaxis, :, np.newaxis] * (1.0 - rgb)
    rgba[:, :, 3] = 0.92

    ax.imshow(
        rgba,
        extent=(low, high, y - height / 2.0, y + height / 2.0),
        aspect="auto",
        interpolation="bicubic",
        zorder=1.6,
    )
    ax.hlines(y, low, high, color=color, linewidth=0.85, alpha=0.38, zorder=2)
    for x, tag, text_offset in ((low, "L", -0.095), (high, "H", -0.095)):
        ax.vlines(x, y - 0.085, y + 0.085, color=color, linewidth=1.0, alpha=0.72, zorder=2.5)
        ax.text(
            x,
            y + text_offset,
            f"{tag}: {fmt_v(x)}",
            ha="center",
            va="top",
            fontsize=6.8,
            color=color,
        )
    ax.vlines(base, y - 0.115, y + 0.115, color=color, linewidth=1.45, alpha=0.95, zorder=3)
    ax.scatter([base], [y], s=80, marker="^", color=color, edgecolor="white", linewidth=0.6, zorder=4)
    ax.text(
        base,
        y + label_y_offset,
        f"{label}\n{fmt_v(base)}",
        ha="center",
        va="bottom" if label_y_offset >= 0.0 else "top",
        fontsize=7.7,
        color=color,
        linespacing=1.05,
        zorder=5,
    )


def projected_scan_band(
    ax: plt.Axes,
    *,
    low: float,
    base: float,
    high: float,
    y: float,
    color: str,
    label: str,
    label_y_offset: float,
) -> None:
    if not low < base < high:
        raise ValueError(f"Need low < base < high for {label}: {low}, {base}, {high}")

    half_top_width = min(0.055, 0.16 * min(base - low, high - base))
    top_left = base - half_top_width
    top_right = base + half_top_width
    y_bottom = y - 0.080
    y_shoulder = y + 0.032
    y_top = y + 0.172
    vertices = np.asarray(
        [
            [low, y_bottom],
            [high, y_bottom],
            [high, y_shoulder],
            [top_right, y_top],
            [top_left, y_top],
            [low, y_shoulder],
        ],
        dtype=float,
    )
    patch = Polygon(vertices, closed=True, facecolor="none", edgecolor="none")
    ax.add_patch(patch)

    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    rows = 220
    strength = np.linspace(0.92, 0.18, rows)[:, np.newaxis]
    rgba = np.zeros((rows, 2, 4), dtype=float)
    rgba[:, :, :3] = 1.0 - strength[:, :, np.newaxis] * (1.0 - rgb)
    rgba[:, :, 3] = 0.96
    image = ax.imshow(
        rgba,
        extent=(low, high, y_bottom, y_top),
        aspect="auto",
        interpolation="bicubic",
        zorder=1.7,
    )
    image.set_clip_path(patch)
    ax.add_patch(
        Polygon(vertices, closed=True, facecolor="none", edgecolor=color, linewidth=0.7, alpha=0.42, zorder=2.0)
    )

    for x, tag, text_offset in ((low, "L", -0.072), (high, "H", -0.072)):
        ax.vlines(x, y_bottom - 0.012, y_shoulder + 0.016, color=color, linewidth=1.0, alpha=0.82, zorder=2.5)
        ax.text(
            x,
            y_bottom + text_offset,
            f"{tag}: {fmt_v(x)}",
            ha="center",
            va="top",
            fontsize=6.8,
            color=color,
        )
    marker_y = y_bottom + 0.012
    ax.vlines(base, y_bottom - 0.03, y_top + 0.045, color=color, linewidth=1.45, alpha=0.95, zorder=3)
    ax.scatter([base], [marker_y], s=80, marker="^", color=color, edgecolor="white", linewidth=0.6, zorder=4)
    ax.text(
        base,
        marker_y + label_y_offset,
        f"{label}\n{fmt_v(base)}",
        ha="center",
        va="bottom" if label_y_offset >= 0.0 else "top",
        fontsize=7.7,
        color=color,
        linespacing=1.05,
        zorder=5,
    )


def add_reference_marker(
    ax: plt.Axes,
    x: float,
    y: float,
    label: str,
    color: str,
    marker: str,
    text_y_offset: float,
    text_x_offset: float = 0.0,
) -> None:
    ax.vlines(x, y - 0.08, y + 0.08, color=color, linewidth=1.25, zorder=3)
    ax.scatter([x], [y], s=68, marker=marker, color=color, edgecolor="none", linewidth=0.0, zorder=4)
    ax.text(
        x + text_x_offset,
        y + text_y_offset,
        f"{label}\n{fmt_v(x)}",
        ha="center",
        va="bottom" if text_y_offset >= 0 else "top",
        fontsize=7.6,
        color=color,
        linespacing=1.1,
        zorder=5,
    )


def build_values(params: dict[str, Any], summary: dict[str, Any]) -> dict[str, float]:
    minus_span = float(params["heatmap_pzc_minus_span"])
    plus_span = float(params["heatmap_pzc_plus_span"])
    pzc_pd = float(params["pzc_Pd"])
    pzc_au = float(params["pzc_Au"])
    return {
        "E1_eq": float(summary["E1_eq_eff"]),
        "E2_eq": float(summary["E2_eq_eff"]),
        "pzc_C": float(params["pzc_C"]),
        "pzc_Pd": pzc_pd,
        "pzc_Au": pzc_au,
        "pzc_Pd_low": pzc_pd - minus_span,
        "pzc_Pd_high": pzc_pd + plus_span,
        "pzc_Au_low": pzc_au - minus_span,
        "pzc_Au_high": pzc_au + plus_span,
    }


def plot_reference_map(values: dict[str, float], *, projected_band: bool = False) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    make_transparent(fig)

    xmin = 0.0
    xmax = 1.25
    lane_y = {
        "eq": 0.47,
        "support": -0.14,
        "pzc_pd": -0.42,
        "pzc_au": -0.66,
    }

    ax.hlines(0.0, xmin, xmax, color=COLORS["dark"], linewidth=1.05, zorder=1)
    ax.hlines(lane_y["eq"], xmin, xmax, color=COLORS["light_gray"], linewidth=0.75, alpha=0.45, zorder=1)
    ax.hlines(-0.53, xmin, xmax, color=COLORS["light_gray"], linewidth=0.75, alpha=0.35, zorder=1)

    add_reference_marker(ax, values["E1_eq"], lane_y["eq"], r"$E_{1,\mathrm{eq}}$", COLORS["gray"], "o", 0.12)
    add_reference_marker(ax, values["E2_eq"], lane_y["eq"], r"$E_{2,\mathrm{eq}}$", COLORS["gray"], "o", 0.12)
    add_reference_marker(
        ax,
        values["pzc_C"],
        lane_y["support"],
        "PZC support",
        COLORS["pzc_support"],
        "^",
        0.23,
        text_x_offset=0.018,
    )

    scan_band = projected_scan_band if projected_band else gradient_scan_band

    scan_band(
        ax,
        low=values["pzc_Pd_low"],
        base=values["pzc_Pd"],
        high=values["pzc_Pd_high"],
        y=lane_y["pzc_pd"],
        color=COLORS["pzc_pd"],
        label="PZC Pd",
        label_y_offset=0.31 if projected_band else 0.12,
    )
    scan_band(
        ax,
        low=values["pzc_Au_low"],
        base=values["pzc_Au"],
        high=values["pzc_Au_high"],
        y=lane_y["pzc_au"],
        color=COLORS["pzc_au"],
        label="PZC Au",
        label_y_offset=-0.13,
    )

    ax.text(
        0.015,
        lane_y["eq"] + 0.045,
        r"$E_{\mathrm{eq}}$",
        ha="left",
        va="bottom",
        fontsize=8.0,
        color=COLORS["gray"],
    )
    ax.text(
        0.015,
        -0.53 + 0.045,
        "PZC scan range",
        ha="left",
        va="bottom",
        fontsize=8.0,
        color=COLORS["gray"],
    )

    ax.set_xlim(xmin - 0.025, xmax + 0.035)
    ax.set_ylim(-1.02, 0.82)
    ax.set_xlabel("Potential (V vs. RHE)")
    ax.set_yticks([])
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.00, 1.25])
    ax.tick_params(axis="x", length=3.4, width=0.85, labelsize=8.2)
    ax.set_title("PZC scan range reference map", loc="left", fontsize=10.0, pad=6, fontweight="normal")
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_position(("data", -0.86))
    ax.spines["bottom"].set_linewidth(0.9)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.88)

    png_out = PNG_PROJECTED_OUT if projected_band else PNG_OUT
    svg_out = SVG_PROJECTED_OUT if projected_band else SVG_OUT
    fig.savefig(png_out, dpi=600)
    fig.savefig(svg_out)
    plt.close(fig)


def assert_outputs() -> None:
    for path in (PNG_OUT, SVG_OUT, PNG_PROJECTED_OUT, SVG_PROJECTED_OUT):
        if not path.exists() or path.stat().st_size <= 0:
            raise RuntimeError(f"Missing output: {path}")
    pdfs = sorted(OUT_DIR.glob("*.pdf"))
    if pdfs:
        raise RuntimeError(f"Expected no PDF outputs, found: {pdfs}")


def main() -> None:
    apply_style()
    params = load_json(PARAMS_PATH)
    summary = load_json(SUMMARY_PATH)
    validate_inputs(params, summary)
    values = build_values(params, summary)
    plot_reference_map(values)
    plot_reference_map(values, projected_band=True)
    assert_outputs()

    print(f"E1_eq = {values['E1_eq']:.3f} V")
    print(f"E2_eq = {values['E2_eq']:.3f} V")
    print(f"PZC support = {values['pzc_C']:.3f} V")
    print(f"PZC Pd = {values['pzc_Pd']:.3f} V; scan {values['pzc_Pd_low']:.3f}-{values['pzc_Pd_high']:.3f} V")
    print(f"PZC Au = {values['pzc_Au']:.3f} V; scan {values['pzc_Au_low']:.3f}-{values['pzc_Au_high']:.3f} V")
    print(f"Wrote {PNG_OUT}")
    print(f"Wrote {SVG_OUT}")
    print(f"Wrote {PNG_PROJECTED_OUT}")
    print(f"Wrote {SVG_PROJECTED_OUT}")


if __name__ == "__main__":
    main()
