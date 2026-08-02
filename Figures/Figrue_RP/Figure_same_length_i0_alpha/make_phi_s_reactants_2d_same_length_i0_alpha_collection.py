from __future__ import annotations

import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SAME_LENGTH_I0_ALPHA_DIR = ROOT / "Figures" / "Figure_same_length_i0_alpha"
OUT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = OUT_DIR / "inputs"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SAME_LENGTH_I0_ALPHA_DIR))

import same_length_i0_alpha_common as common  # noqa: E402
from Figures.Figrue_RP import make_phi_s_reactants_2d as base  # noqa: E402


solver = common.solver

BASELINE_EXPECTED = {
    "E_mix_with": (0.5979829354430014, 5e-10),
    "E_mix_no": (0.4670000000000001, 5e-10),
    "i_mix_avg_with": (0.08384634584416407, 5e-10),
    "i_mix_avg_no": (0.11756035407872442, 5e-10),
    "i_mix_abs_with": (4.192317292208204e-11, 5e-18),
    "i_mix_abs_no": (5.878017703936221e-11, 5e-18),
    "max_abs_phi_tilde_with_edl": (6.038944611092431, 5e-10),
}
SUPPORT0_OVERRIDES = {
    **common.PARAM_OVERRIDES,
    "L_gap": 0.0,
}
SUPPORT0_TAG = f"same_length_i0_alpha050_au25_pd25_support0_{common.BASE_RESULT_ID}"
SUPPORT0_EXPECTED = {
    "E_mix_with": (0.600693745661775, 5e-10),
    "E_mix_no": (0.4670000000000001, 5e-10),
    "i_mix_avg_with": (0.08160442531462271, 5e-10),
    "i_mix_avg_no": (0.11756035407872442, 5e-10),
    "i_mix_abs_with": (4.080221265731136e-11, 5e-18),
    "i_mix_abs_no": (5.878017703936221e-11, 5e-18),
    "max_abs_phi_tilde_with_edl": (5.985937034969025, 5e-10),
}
FIGSIZE = (2.6, 7.45)
PANEL_HSPACE = 0.36


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_params_with_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    params = solver.apply_param_overrides(
        solver.default_params(),
        common.load_baseline_params_json(),
        reset_lambda_D=False,
    )
    return solver.apply_param_overrides(params, overrides, reset_lambda_D=False)


def validate_case_params(params: dict[str, Any], expected_l_gap: float) -> None:
    checks = {
        "L_Au": common.L_AU_SAME,
        "L_gap": expected_l_gap,
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


def assert_expected_values(summary: dict[str, str], expected: dict[str, tuple[float, float]]) -> None:
    for key, (expected_value, atol) in expected.items():
        actual = float(summary[key])
        if abs(actual - expected_value) > atol:
            raise ValueError(f"{key}: expected {expected_value:.15g}, got {actual:.15g}")


def save_inputs(tag: str, params: dict[str, Any], overrides: dict[str, Any], summary: dict[str, str]) -> None:
    ensure_dirs()
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


def unique_ticks(values: list[float]) -> list[float]:
    ticks: list[float] = []
    for value in values:
        if not any(math.isclose(value, existing, rel_tol=0.0, abs_tol=1e-9) for existing in ticks):
            ticks.append(value)
    return ticks


def plot_phi_s_reactants_case(data: Any) -> list[Path]:
    fig = base.plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(
        nrows=4,
        ncols=2,
        width_ratios=(1.0, 0.038),
        height_ratios=(1.0, 1.0, 1.0, 0.12),
        hspace=PANEL_HSPACE,
        wspace=0.08,
    )
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    caxes = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    lane_ax = fig.add_subplot(gs[3, 0], sharex=axes[-1])
    fig.add_subplot(gs[3, 1]).set_axis_off()

    phi_abs = float(base.np.nanmax(base.np.abs(data.phi_s_mV)))
    phi_vlim = max(10.0, base.math.ceil(phi_abs / 10.0) * 10.0)
    phi_levels = base.np.linspace(-phi_vlim, phi_vlim, 11)

    base.add_heatmap(
        axes[0],
        caxes[0],
        data,
        data.phi_s_mV,
        cmap="RdBu_r",
        norm=base.TwoSlopeNorm(vmin=-phi_vlim, vcenter=0.0, vmax=phi_vlim),
        cbar_label=r"$\Phi_s$ (mV)",
        title=rf"Solution phase potential, $E_{{\mathrm{{mix}}}}$ = {float(data.res_edl['E_mix']):.2f} V",
        contour_levels=phi_levels,
    )
    base.add_heatmap(
        axes[1],
        caxes[1],
        data,
        data.c_r1_norm,
        cmap="viridis",
        norm=base.log_norm(data.c_r1_norm),
        cbar_label=r"$c_{\mathrm{Red1}}/c_{\mathrm{bulk}}$",
        title=r"Reactant Red1 distribution",
        contour_levels=base.log_contour_levels(data.c_r1_norm),
    )
    base.add_heatmap(
        axes[2],
        caxes[2],
        data,
        data.c_o2_norm,
        cmap="viridis",
        norm=base.log_norm(data.c_o2_norm),
        cbar_label=r"$c_{\mathrm{Ox2}}/c_{\mathrm{bulk}}$",
        title=r"Reactant Ox2 distribution",
        show_xlabel=True,
        contour_levels=base.log_contour_levels(data.c_o2_norm),
    )

    for ax in axes:
        ax.set_xlim(0.0, data.L_total_nm)
        ax.set_ylim(0.0, data.y_nm[-1])
        ax.set_yticks([0.0, 5.0, 10.0, 15.0])

    xticks = unique_ticks([0.0, data.L_Au_nm, data.L_C_nm, data.L_total_nm])
    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels([f"{tick:.0f}" for tick in xticks])

    base.add_material_lane(lane_ax, data)
    for text in lane_ax.texts:
        if text.get_text() == "support":
            text.set_fontsize(5.8)
    fig.align_ylabels(axes)
    return base.save_figure(fig)


def run_case(
    *,
    tag: str,
    overrides: dict[str, Any],
    expected_l_gap: float,
    expected_summary: dict[str, tuple[float, float]] | None = None,
) -> tuple[dict[str, Any], dict[str, str], list[Path]]:
    params = load_params_with_overrides(overrides)
    validate_case_params(params, expected_l_gap)
    summary = summary_from_params(params)
    if expected_summary is not None:
        assert_expected_values(summary, expected_summary)
    save_inputs(tag, params, overrides, summary)

    def load_params() -> dict[str, Any]:
        return copy.deepcopy(params)

    def load_summary() -> dict[str, str]:
        return dict(summary)

    base.RESULT_ID = tag
    base.OUT_DIR = OUT_DIR
    base.load_params = load_params
    base.load_summary = load_summary
    base.plot_phi_s_reactants = plot_phi_s_reactants_case

    data = base.build_2d_data()
    saved = plot_phi_s_reactants_case(data)
    return params, summary, saved


def main() -> None:
    ensure_dirs()
    base.apply_style()

    cases = [
        {
            "label": "same-length equal-i0 alpha=0.5, support=10 nm",
            "tag": common.OUTPUT_TAG,
            "overrides": dict(common.PARAM_OVERRIDES),
            "expected_l_gap": 10.0e-9,
            "expected_summary": BASELINE_EXPECTED,
        },
        {
            "label": "same-length equal-i0 alpha=0.5, support=0 nm",
            "tag": SUPPORT0_TAG,
            "overrides": SUPPORT0_OVERRIDES,
            "expected_l_gap": 0.0,
            "expected_summary": SUPPORT0_EXPECTED,
        },
    ]

    all_saved: list[Path] = []
    for case in cases:
        params, summary, saved = run_case(
            tag=str(case["tag"]),
            overrides=copy.deepcopy(case["overrides"]),
            expected_l_gap=float(case["expected_l_gap"]),
            expected_summary=case["expected_summary"],
        )
        all_saved.extend(saved)

        print(case["label"])
        print(f"  tag = {case['tag']}")
        print(f"  L_Au = {float(params['L_Au']) * 1e9:.6g} nm")
        print(f"  L_gap = {float(params['L_gap']) * 1e9:.6g} nm")
        print(f"  L_Pd_len = {float(params['L_Pd_len']) * 1e9:.6g} nm")
        print(f"  E_mix_with = {float(summary['E_mix_with']):.15g} V")
        print(f"  E_mix_no = {float(summary['E_mix_no']):.15g} V")
        print(f"  i_mix_avg_with = {float(summary['i_mix_avg_with']):.15g} A/m^2")
        print(f"  i_mix_avg_no = {float(summary['i_mix_avg_no']):.15g} A/m^2")
        print(f"  max |phi_tilde| = {float(summary['max_abs_phi_tilde_with_edl']):.6g}")
        for path in saved:
            print(f"  saved {path}")

    pngs = sorted(OUT_DIR.glob("phi_s_reactants_2d_*.png"))
    svgs = sorted(OUT_DIR.glob("phi_s_reactants_2d_*.svg"))
    pdfs = sorted(OUT_DIR.glob("*.pdf"))
    if len(pngs) != 2 or len(svgs) != 2:
        raise RuntimeError(f"Expected 2 PNG and 2 SVG outputs, got {len(pngs)} PNG and {len(svgs)} SVG")
    if pdfs:
        raise RuntimeError(f"Expected zero PDF outputs, found {len(pdfs)}")
    if len(all_saved) != 4:
        raise RuntimeError(f"Expected 4 saved paths, got {len(all_saved)}")


if __name__ == "__main__":
    main()
