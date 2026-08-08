from __future__ import annotations

import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SAME_LENGTH_I0_ALPHA_DIR = ROOT / "Figures" / "Figure_same_length_i0_alpha"
OUT_DIR = ROOT / "Figures" / "Figure_scan" / "figures" / "L_support"
INPUTS_DIR = OUT_DIR / "inputs"
CSV_DIR = OUT_DIR / "csv"

L_SUPPORT_NM_VALUES = (0.0, 3.0, 1000.0)
OUT_STEM = "solution_phase_potential_2d"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SAME_LENGTH_I0_ALPHA_DIR))

import same_length_i0_alpha_common as common  # noqa: E402
from Figures.Figrue_RP import make_phi_s_reactants_2d as base  # noqa: E402


solver = common.solver


def ensure_dirs() -> None:
    for path in (OUT_DIR, INPUTS_DIR, CSV_DIR):
        path.mkdir(parents=True, exist_ok=True)


def format_nm_value(value_nm: float) -> str:
    if math.isclose(value_nm, round(value_nm), rel_tol=0.0, abs_tol=1e-9):
        return f"{int(round(value_nm))}"
    return f"{value_nm:.3g}"


def format_nm_tag(value_nm: float) -> str:
    return format_nm_value(value_nm).replace(".", "p")


def output_tag(value_nm: float) -> str:
    return f"L_support_{format_nm_tag(value_nm)}nm_{common.OUTPUT_TAG}"


def output_stem(value_nm: float) -> str:
    return f"{OUT_STEM}_{output_tag(value_nm)}"


def remove_previous_outputs() -> None:
    for pattern in (
        f"{OUT_STEM}_L_support_*.png",
        f"{OUT_STEM}_L_support_*.svg",
    ):
        for path in OUT_DIR.glob(pattern):
            path.unlink()
    for pattern in (
        "params_L_support_*.json",
        "overrides_L_support_*.json",
        "summary_compare_L_support_*.csv",
        "summary_compare_L_support_*.json",
    ):
        for path in INPUTS_DIR.glob(pattern):
            path.unlink()
    for path in CSV_DIR.glob("phi_s_stats_L_support*.csv"):
        path.unlink()


def load_params_for_l_support(value_nm: float) -> dict[str, Any]:
    params = common.load_same_length_i0_alpha_params()
    params["L_gap"] = value_nm * 1.0e-9
    validate_case_params(params, value_nm)
    return params


def validate_case_params(params: dict[str, Any], expected_l_support_nm: float) -> None:
    checks = {
        "L_Au": common.L_AU_SAME,
        "L_gap": expected_l_support_nm * 1.0e-9,
        "L_Pd_len": common.L_PD_SAME,
        "it0_1": common.I0_GEOM,
        "it0_2": common.I0_GEOM,
        "alpha1": common.ALPHA_EQUAL,
        "alpha2": common.ALPHA_EQUAL,
        "out_of_plane_width": common.OUT_OF_PLANE_WIDTH,
    }
    for key, expected in checks.items():
        actual = float(params[key])
        atol = max(1e-15, abs(expected) * 1e-12)
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol):
            raise ValueError(f"{key} should be {expected:.15g}, got {actual:.15g}")


def summary_from_params(params: dict[str, Any]) -> dict[str, str]:
    pair = solver.run_edl_comparison_pair(params, mode="FULL")
    return common.summary_from_pair(pair)


def save_inputs(value_nm: float, params: dict[str, Any], summary: dict[str, str]) -> None:
    tag = output_tag(value_nm)
    overrides = {
        **common.PARAM_OVERRIDES,
        "L_gap": value_nm * 1.0e-9,
    }
    with (INPUTS_DIR / f"params_{tag}.json").open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, sort_keys=True)
        f.write("\n")
    with (INPUTS_DIR / f"overrides_{tag}.json").open("w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2, sort_keys=True)
        f.write("\n")
    with (INPUTS_DIR / f"summary_compare_{tag}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    with (INPUTS_DIR / f"summary_compare_{tag}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")


def build_2d_data(params: dict[str, Any], summary: dict[str, str]) -> Any:
    old_load_params = base.load_params
    old_load_summary = base.load_summary

    def load_params() -> dict[str, Any]:
        return copy.deepcopy(params)

    def load_summary() -> dict[str, str]:
        return dict(summary)

    try:
        base.load_params = load_params
        base.load_summary = load_summary
        return base.build_2d_data()
    finally:
        base.load_params = old_load_params
        base.load_summary = old_load_summary


def unique_ticks(values: list[float]) -> list[float]:
    ticks: list[float] = []
    for value in values:
        if not any(math.isclose(value, existing, rel_tol=0.0, abs_tol=1e-8) for existing in ticks):
            ticks.append(value)
    return ticks


def format_tick(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-6):
        return f"{int(round(value))}"
    return f"{value:.1f}"


def x_ticks_for_data(data: Any) -> list[float]:
    if data.L_total_nm > 200.0:
        return [0.0, 250.0, 500.0, 750.0, data.L_total_nm]
    support_width_nm = data.L_C_nm - data.L_Au_nm
    if support_width_nm < 6.0:
        return unique_ticks([0.0, data.L_Au_nm, data.L_total_nm])
    return unique_ticks([0.0, data.L_Au_nm, data.L_C_nm, data.L_total_nm])


def add_material_lane(ax: Any, data: Any) -> None:
    segments = [
        ("Au", 0.0, data.L_Au_nm, base.COLORS["au"], base.COLORS["dark"]),
        ("support", data.L_Au_nm, data.L_C_nm, base.COLORS["support"], "white"),
        ("Pd", data.L_C_nm, data.L_total_nm, base.COLORS["pd"], "white"),
    ]
    total = float(data.L_total_nm)
    for label, x0, x1, face, text_color in segments:
        width = x1 - x0
        if width <= 1e-9:
            continue
        ax.add_patch(
            base.Rectangle((x0, 0.0), width, 1.0, facecolor=face, edgecolor="white", linewidth=0.8)
        )
        width_fraction = width / total
        if width_fraction < 0.07 and label == "support":
            continue
        if width_fraction < 0.07:
            text_color = base.COLORS["dark"]
        ax.text((x0 + x1) / 2.0, 0.5, label, ha="center", va="center", fontsize=8.3, color=text_color)
    ax.set_xlim(0.0, total)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()


def save_case_figure(fig: Any, stem: str) -> list[Path]:
    saved: list[Path] = []
    for ext in ("png", "svg"):
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=600, bbox_inches="tight")
        saved.append(path)
    base.plt.close(fig)
    return saved


def plot_phi_s_case(data: Any, value_nm: float, phi_vlim: float) -> list[Path]:
    fig = base.plt.figure(figsize=(5.8, 3.0))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=(1.0, 0.038),
        height_ratios=(1.0, 0.12),
        hspace=0.24,
        wspace=0.08,
    )
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    lane_ax = fig.add_subplot(gs[1, 0], sharex=ax)
    fig.add_subplot(gs[1, 1]).set_axis_off()

    phi_levels = base.np.linspace(-phi_vlim, phi_vlim, 11)
    base.add_heatmap(
        ax,
        cax,
        data,
        data.phi_s_mV,
        cmap="RdBu_r",
        norm=base.TwoSlopeNorm(vmin=-phi_vlim, vcenter=0.0, vmax=phi_vlim),
        cbar_label=r"$\Phi_s$ (mV)",
        title=rf"Solution phase potential, $E_{{\mathrm{{mix}}}}$ = {float(data.res_edl['E_mix']):.2f} V",
        show_xlabel=False,
        contour_levels=phi_levels,
    )
    ax.text(
        0.985,
        0.955,
        rf"$L_{{\mathrm{{support}}}}$ = {format_nm_value(value_nm)} nm",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        color=base.COLORS["dark"],
    )
    ax.set_xlim(0.0, data.L_total_nm)
    ax.set_ylim(0.0, data.y_nm[-1])
    ax.set_yticks([0.0, 5.0, 10.0, 15.0])

    ticks = x_ticks_for_data(data)
    ax.set_xticks(ticks)
    ax.set_xticklabels([format_tick(tick) for tick in ticks])
    ax.tick_params(labelbottom=True)
    ax.set_xlabel("")
    add_material_lane(lane_ax, data)
    lane_ax.text(
        0.5,
        -0.58,
        "x (nm)",
        transform=lane_ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color=base.COLORS["dark"],
        clip_on=False,
    )
    fig.align_ylabels([ax])
    return save_case_figure(fig, output_stem(value_nm))


def stats_row(value_nm: float, data: Any, summary: dict[str, str]) -> dict[str, Any]:
    return {
        "L_support_nm": value_nm,
        "L_Au_nm": float(data.L_Au_nm),
        "L_Pd_len_nm": float(data.L_total_nm - data.L_C_nm),
        "L_total_nm": float(data.L_total_nm),
        "lambda_D_nm": float(data.lambda_D_nm),
        "E_mix_with_V": float(summary["E_mix_with"]),
        "E_mix_no_V": float(summary["E_mix_no"]),
        "i_mix_avg_with_A_per_m2": float(summary["i_mix_avg_with"]),
        "i_mix_avg_no_A_per_m2": float(summary["i_mix_avg_no"]),
        "max_abs_phi_tilde": float(summary["max_abs_phi_tilde_with_edl"]),
        "phi_s_min_mV": float(base.np.min(data.phi_s_mV)),
        "phi_s_max_mV": float(base.np.max(data.phi_s_mV)),
    }


def save_stats(rows: list[dict[str, Any]]) -> Path:
    path = CSV_DIR / f"phi_s_stats_L_support_{common.OUTPUT_TAG}.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def assert_outputs() -> tuple[list[Path], list[Path]]:
    pngs = sorted(OUT_DIR.glob(f"{OUT_STEM}_L_support_*.png"))
    svgs = sorted(OUT_DIR.glob(f"{OUT_STEM}_L_support_*.svg"))
    pdfs = sorted(OUT_DIR.glob("*.pdf"))
    if len(pngs) != len(L_SUPPORT_NM_VALUES) or len(svgs) != len(L_SUPPORT_NM_VALUES):
        raise RuntimeError(f"Expected 3 PNG and 3 SVG outputs, got {len(pngs)} PNG and {len(svgs)} SVG")
    if pdfs:
        raise RuntimeError(f"Expected zero PDF outputs, found {len(pdfs)}")
    return pngs, svgs


def main() -> None:
    ensure_dirs()
    remove_previous_outputs()
    base.apply_style()

    cases: list[tuple[float, dict[str, Any], dict[str, str], Any]] = []
    stats_rows: list[dict[str, Any]] = []
    for value_nm in L_SUPPORT_NM_VALUES:
        params = load_params_for_l_support(value_nm)
        summary = summary_from_params(params)
        save_inputs(value_nm, params, summary)
        data = build_2d_data(params, summary)
        cases.append((value_nm, params, summary, data))
        stats_rows.append(stats_row(value_nm, data, summary))

    phi_abs = max(float(base.np.nanmax(base.np.abs(data.phi_s_mV))) for _, _, _, data in cases)
    phi_vlim = max(10.0, math.ceil(phi_abs / 10.0) * 10.0)

    saved: list[Path] = []
    for value_nm, _params, summary, data in cases:
        saved.extend(plot_phi_s_case(data, value_nm, phi_vlim))
        print(f"L_support = {format_nm_value(value_nm)} nm")
        print(f"  E_mix_with = {float(summary['E_mix_with']):.15g} V")
        print(f"  i_mix_avg_with = {float(summary['i_mix_avg_with']):.15g} A/m^2")
        print(f"  max |phi_tilde| = {float(summary['max_abs_phi_tilde_with_edl']):.6g}")

    stats_path = save_stats(stats_rows)
    pngs, svgs = assert_outputs()

    print(f"Common Phi_s color limit = +/- {phi_vlim:.6g} mV")
    print(f"Saved stats: {stats_path.relative_to(ROOT)}")
    print(f"Verified {len(pngs)} PNG outputs, {len(svgs)} SVG outputs, and zero PDF outputs")
    for path in saved:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
