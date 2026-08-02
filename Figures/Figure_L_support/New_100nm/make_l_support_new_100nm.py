from __future__ import annotations

import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = OUT_DIR / "inputs"
CSV_DIR = OUT_DIR / "csv"

sys.path.insert(0, str(ROOT))

from Figures.Figure_L_support.Old import make_l_support_figure_panels_same_length_i0_alpha as base  # noqa: E402


L_SUPPORT_NM_VALUES = (0.0, 3.0, 10.0, 30.0, 100.0, 1000.0)
ACTIVE_ZOOM_SUPPORT_NM_VALUES = (30.0, 100.0, 1000.0)
HIGH_RES_L_SUPPORT_NM_VALUES = L_SUPPORT_NM_VALUES
HIGH_RES_N_MODES = 960
HIGH_RES_NX = 5000
OFAT_POINTS = 15
OFAT_MAX_NM = 100.0
OFAT_LONG_MAX_NM = 1000.0
FULL_PANEL_FIGSIZE = (5.8, 2.55)
FULL_PANEL_AXES_RECT = (0.12, 0.22, 0.85, 0.62)
OFAT_STEM = "ofat_compare_L_gap_i_mix_avg_A_per_m2"
OFAT_EMIX_STEM = "ofat_compare_L_gap_E_mix"
OFAT_LONG_STEM = "ofat_compare_L_gap_i_mix_avg_A_per_m2_0_1000nm"
OFAT_LONG_EMIX_STEM = "ofat_compare_L_gap_E_mix_0_1000nm"
OFAT_CSV = CSV_DIR / f"ofat_compare_L_gap_highN_{base.common.OUTPUT_TAG}.csv"
OFAT_LONG_CSV = CSV_DIR / f"ofat_compare_L_gap_highN_0_1000nm_{base.common.OUTPUT_TAG}.csv"


def ensure_dirs() -> None:
    for path in (OUT_DIR, INPUTS_DIR, CSV_DIR):
        path.mkdir(parents=True, exist_ok=True)


def configure_parent_module() -> None:
    base.OUT_DIR = OUT_DIR
    base.INPUTS_DIR = INPUTS_DIR
    base.CSV_DIR = CSV_DIR
    base.L_SUPPORT_NM_VALUES = L_SUPPORT_NM_VALUES
    base.POLARIZATION_CURVE_CSV = CSV_DIR / f"{base.POLARIZATION_STEM}.csv"
    base.panel_base.SINGLE_PANEL_FIGSIZE = FULL_PANEL_FIGSIZE
    base.panel_base.SINGLE_PANEL_AXES_RECT = FULL_PANEL_AXES_RECT
    base.x_ticks_for_panel = x_ticks_for_panel
    base.x_ticks_for_2d = x_ticks_for_2d


def remove_previous_outputs() -> None:
    patterns = (
        "solution_phase_potential_2d_L_support_*.png",
        "solution_phase_potential_2d_L_support_*.svg",
        "solution_phase_potential_2d_active_zoom_L_support_*.png",
        "solution_phase_potential_2d_active_zoom_L_support_*.svg",
        "figure_3_panel_b_reaction_plane_potential_L_support_*.png",
        "figure_3_panel_b_reaction_plane_potential_L_support_*.svg",
        "figure_3_panel_e_local_current_density_L_support_*.png",
        "figure_3_panel_e_local_current_density_L_support_*.svg",
        "figure_3_panel_b_reaction_plane_potential_active_zoom_L_support_*.png",
        "figure_3_panel_b_reaction_plane_potential_active_zoom_L_support_*.svg",
        "figure_3_panel_e_local_current_density_active_zoom_L_support_*.png",
        "figure_3_panel_e_local_current_density_active_zoom_L_support_*.svg",
        "ofat_compare_L_gap*.png",
        "ofat_compare_L_gap*.svg",
    )
    for pattern in patterns:
        for path in OUT_DIR.glob(pattern):
            path.unlink()
    for pattern in ("phi_s_stats_L_support_*.csv", "ofat_compare_L_gap_highN_*.csv"):
        for path in CSV_DIR.glob(pattern):
            path.unlink()


def use_high_res(value_nm: float) -> bool:
    return any(math.isclose(value_nm, target, rel_tol=0.0, abs_tol=1e-9) for target in HIGH_RES_L_SUPPORT_NM_VALUES)


def params_for_l_support(value_nm: float, *, high_res: bool | None = None) -> dict[str, Any]:
    params = base.common.load_same_length_i0_alpha_params()
    params["L_gap"] = value_nm * 1.0e-9
    if use_high_res(value_nm) if high_res is None else high_res:
        params["N_modes"] = HIGH_RES_N_MODES
        params["Nx"] = HIGH_RES_NX
    base.validate_case_params(params, value_nm)
    return params


def save_inputs(value_nm: float, params: dict[str, Any], summary: dict[str, str]) -> None:
    tag = base.output_tag(value_nm)
    overrides = {
        **base.common.PARAM_OVERRIDES,
        "L_gap": value_nm * 1.0e-9,
    }
    if use_high_res(value_nm):
        overrides["N_modes"] = HIGH_RES_N_MODES
        overrides["Nx"] = HIGH_RES_NX

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


def build_cases() -> list[Any]:
    cases: list[Any] = []
    for value_nm in L_SUPPORT_NM_VALUES:
        params = params_for_l_support(value_nm)
        summary = base.summary_from_params(params)
        save_inputs(value_nm, params, summary)
        cases.append(
            base.LSupportCase(
                value_nm=value_nm,
                params=params,
                summary=summary,
                panel_data=base.build_panel_data(params, summary),
                rp_data=base.build_2d_data(params, summary),
            )
        )
    return cases


def x_ticks_for_panel(data: Any) -> list[float]:
    total = float(data.x_nm[-1])
    if total > 200.0:
        return [0.0, 250.0, 500.0, 750.0, total]
    return base.unique_ticks([0.0, float(data.L_Au_nm), float(data.L_C_nm), total])


def x_ticks_for_2d(data: Any) -> list[float]:
    total = float(data.L_total_nm)
    if total > 200.0:
        return [0.0, 250.0, 500.0, 750.0, total]
    return base.unique_ticks([0.0, float(data.L_Au_nm), float(data.L_C_nm), total])


def ofat_l_support_values(max_nm: float) -> Any:
    return base.panel_base.np.linspace(0.0, max_nm, OFAT_POINTS, dtype=float)


def build_ofat_l_gap_rows(max_nm: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value_nm in ofat_l_support_values(max_nm):
        params = params_for_l_support(float(value_nm), high_res=True)
        pair = base.solver.run_edl_comparison_pair(params, mode="FULL")
        with_edl = pair["with_edl"]
        no_edl = pair["no_edl"]
        rows.append(
            {
                "L_support_nm": float(value_nm),
                "L_gap_m": float(params["L_gap"]),
                "N_modes": int(params["N_modes"]),
                "Nx": int(params["Nx"]),
                "E_mix_with_V": float(with_edl["E_mix"]),
                "E_mix_no_V": float(no_edl["E_mix"]),
                "delta_E_mix_V": float(with_edl["E_mix"]) - float(no_edl["E_mix"]),
                "i_mix_avg_with_A_per_m2": float(with_edl["i_mix_avg_A_per_m2"]),
                "i_mix_avg_no_A_per_m2": float(no_edl["i_mix_avg_A_per_m2"]),
                "delta_i_mix_avg_A_per_m2": float(with_edl["i_mix_avg_A_per_m2"]) - float(no_edl["i_mix_avg_A_per_m2"]),
                "max_abs_phi_tilde_with_edl": float(with_edl["max_abs_phi_tilde"]),
            }
        )
    return rows


def write_ofat_l_gap_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_ofat_l_gap(rows: list[dict[str, Any]], stem: str) -> list[Path]:
    x = base.panel_base.np.asarray([row["L_support_nm"] for row in rows], dtype=float)
    y_with = base.panel_base.np.asarray([row["i_mix_avg_with_A_per_m2"] for row in rows], dtype=float)
    y_no = base.panel_base.np.asarray([row["i_mix_avg_no_A_per_m2"] for row in rows], dtype=float)

    fig, ax = base.panel_base.plt.subplots(figsize=(4.05, 2.75), constrained_layout=False)
    ax.plot(x, y_with, color=base.panel_base.COLORS["with"], linewidth=2.0, marker="o", markersize=3.4, label="with EDL")
    ax.plot(x, y_no, color=base.panel_base.COLORS["without"], linewidth=1.8, marker="s", markersize=3.0, label="w/o EDL")
    base.panel_base.style_axes(
        ax,
        r"$L_{\mathrm{support}}$ (nm)",
        r"Mixed current density, $\bar{i}_{\mathrm{mix}}$ (A/m$^2$)",
        r"$L_{\mathrm{support}}$ scan",
    )
    ax.set_xlim(float(x[0]), float(x[-1]))
    ymin = min(float(y_with.min()), float(y_no.min()))
    ymax = max(float(y_with.max()), float(y_no.max()))
    pad = 0.10 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.legend(loc="center right", fontsize=8.0, handlelength=2.0)
    fig.subplots_adjust(left=0.20, right=0.975, bottom=0.21, top=0.86)

    saved: list[Path] = []
    for ext in ("png", "svg"):
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=600, transparent=True, facecolor="none", edgecolor="none")
        saved.append(path)
    base.panel_base.plt.close(fig)
    return saved


def plot_ofat_l_gap_emix(rows: list[dict[str, Any]], stem: str) -> list[Path]:
    x = base.panel_base.np.asarray([row["L_support_nm"] for row in rows], dtype=float)
    y_with = base.panel_base.np.asarray([row["E_mix_with_V"] for row in rows], dtype=float)
    y_no = base.panel_base.np.asarray([row["E_mix_no_V"] for row in rows], dtype=float)

    fig, ax = base.panel_base.plt.subplots(figsize=(4.05, 2.75), constrained_layout=False)
    ax.plot(x, y_with, color=base.panel_base.COLORS["with"], linewidth=2.0, marker="o", markersize=3.4, label="with EDL")
    ax.plot(x, y_no, color=base.panel_base.COLORS["without"], linewidth=1.8, marker="s", markersize=3.0, label="w/o EDL")
    base.panel_base.style_axes(
        ax,
        r"$L_{\mathrm{support}}$ (nm)",
        r"$E_{\mathrm{mix}}$ (V vs. RHE)",
        r"$L_{\mathrm{support}}$ scan",
    )
    ax.set_xlim(float(x[0]), float(x[-1]))
    ymin = min(float(y_with.min()), float(y_no.min()))
    ymax = max(float(y_with.max()), float(y_no.max()))
    pad = 0.10 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.legend(loc="center right", fontsize=8.0, handlelength=2.0)
    fig.subplots_adjust(left=0.20, right=0.975, bottom=0.21, top=0.86)

    saved: list[Path] = []
    for ext in ("png", "svg"):
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=600, transparent=True, facecolor="none", edgecolor="none")
        saved.append(path)
    base.panel_base.plt.close(fig)
    return saved


def assert_outputs(saved: list[Path], ofat_row_sets: list[list[dict[str, Any]]]) -> None:
    missing = [path for path in saved if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing expected outputs: {missing}")
    suffix_counts = {".png": 0, ".svg": 0}
    for path in saved:
        suffix_counts[path.suffix] += 1
    if suffix_counts[".png"] != suffix_counts[".svg"]:
        raise RuntimeError(f"PNG/SVG output count mismatch: {suffix_counts}")
    expected_pairs = 31
    if suffix_counts[".png"] != expected_pairs:
        raise RuntimeError(f"Expected {expected_pairs} PNG/SVG output pairs, got {suffix_counts['.png']}")
    pdfs = sorted(OUT_DIR.glob("*.pdf"))
    if pdfs:
        raise RuntimeError(f"Expected zero PDF outputs, found {len(pdfs)}")
    for ofat_rows in ofat_row_sets:
        if len(ofat_rows) != OFAT_POINTS:
            raise RuntimeError(f"Expected {OFAT_POINTS} OFAT rows, got {len(ofat_rows)}")
        y_with = base.panel_base.np.asarray([row["i_mix_avg_with_A_per_m2"] for row in ofat_rows], dtype=float)
        if not base.panel_base.np.all(base.panel_base.np.diff(y_with) >= -1e-12):
            raise RuntimeError("High-N L_support OFAT i_mix_avg_with is not monotonic nondecreasing")


def main() -> None:
    configure_parent_module()
    ensure_dirs()
    remove_previous_outputs()
    base.panel_base.apply_style()
    base.rp_base.apply_style()

    cases = build_cases()
    phi_ylim = base.common_phi_ylim(cases)
    current_ylim = base.common_current_ylim(cases)
    phi_abs = max(float(base.rp_base.np.nanmax(base.rp_base.np.abs(case.rp_data.phi_s_mV))) for case in cases)
    phi_vlim = max(10.0, math.ceil(phi_abs / 10.0) * 10.0)

    saved: list[Path] = []
    for case in cases:
        saved.extend(base.plot_panel_b_case(case, phi_ylim))
        saved.extend(base.plot_panel_e_case(case, current_ylim))
        saved.extend(base.plot_phi_s_case(case, phi_vlim))

    for case in cases:
        if any(math.isclose(case.value_nm, zoom_nm, rel_tol=0.0, abs_tol=1e-9) for zoom_nm in ACTIVE_ZOOM_SUPPORT_NM_VALUES):
            saved.extend(base.plot_phi_s_active_zoom(case, phi_vlim))
            saved.extend(base.plot_panel_b_active_zoom(case, phi_ylim))
            saved.extend(base.plot_panel_e_active_zoom(case, current_ylim))

    stats_path = base.save_stats(cases)
    ofat_rows = build_ofat_l_gap_rows(OFAT_MAX_NM)
    ofat_csv_path = write_ofat_l_gap_csv(ofat_rows, OFAT_CSV)
    saved.extend(plot_ofat_l_gap_emix(ofat_rows, OFAT_EMIX_STEM))
    saved.extend(plot_ofat_l_gap(ofat_rows, OFAT_STEM))

    ofat_long_rows = build_ofat_l_gap_rows(OFAT_LONG_MAX_NM)
    ofat_long_csv_path = write_ofat_l_gap_csv(ofat_long_rows, OFAT_LONG_CSV)
    saved.extend(plot_ofat_l_gap_emix(ofat_long_rows, OFAT_LONG_EMIX_STEM))
    saved.extend(plot_ofat_l_gap(ofat_long_rows, OFAT_LONG_STEM))
    assert_outputs(saved, [ofat_rows, ofat_long_rows])

    print(f"same-length equal-i0 alpha=0.5 tag = {base.common.OUTPUT_TAG}")
    print(f"Saved stats: {stats_path.relative_to(ROOT)}")
    print(f"Saved high-N OFAT: {ofat_csv_path.relative_to(ROOT)}")
    print(f"Saved high-N OFAT 0-1000 nm: {ofat_long_csv_path.relative_to(ROOT)}")
    print(f"Common panel b ylim = {phi_ylim[0]:.6g} to {phi_ylim[1]:.6g} V")
    print(f"Common panel e ylim = {current_ylim[0]:.6g} to {current_ylim[1]:.6g} x 10^-3 A/m^2")
    print(f"Common Phi_s color limit = +/- {phi_vlim:.6g} mV")
    for case in cases:
        print(f"L_support = {base.format_nm_value(case.value_nm)} nm")
        print(f"  N_modes = {int(case.params['N_modes'])}, Nx = {int(case.params['Nx'])}")
        print(f"  E_mix_with = {float(case.summary['E_mix_with']):.15g} V")
        print(f"  E_mix_no = {float(case.summary['E_mix_no']):.15g} V")
        print(f"  i_mix_avg_with = {float(case.summary['i_mix_avg_with']):.15g} A/m^2")
        print(f"  i_mix_avg_no = {float(case.summary['i_mix_avg_no']):.15g} A/m^2")
        print(f"  max |phi_tilde| = {float(case.summary['max_abs_phi_tilde_with_edl']):.6g}")
    print(f"Verified {len(saved) // 2} PNG/SVG output pairs and zero PDF outputs")
    for path in saved:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
