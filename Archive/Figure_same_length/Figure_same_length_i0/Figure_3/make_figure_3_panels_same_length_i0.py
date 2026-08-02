from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SAME_LENGTH_I0_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SAME_LENGTH_I0_DIR))

import same_length_i0_common as common  # noqa: E402
from Figures.Figure_3 import make_figure_3_panels as base  # noqa: E402


SPECIES_COLORS = {
    # Colorblind-safe green/blue pairing chosen to echo the viridis-based
    # Red1/Ox2 2D distributions while keeping phi/eta orange-yellow free.
    "red1_i1": "#009E73",
    "ox2_i2": "#0072B2",
}
TRACEABILITY_STEM = f"figure_3_{common.OUTPUT_TAG}"


def save_figure3_traceability(params: dict[str, Any], summary: dict[str, str]) -> None:
    common.ensure_output_dirs()
    overrides = {
        **common.PARAM_OVERRIDES,
        "source_params": str(common.BASE_PARAMS_PATH.relative_to(ROOT)),
        "N_modes": int(params["N_modes"]),
        "Nx": int(params["Nx"]),
    }
    with (common.INPUTS_DIR / f"params_{TRACEABILITY_STEM}.json").open("w", encoding="utf-8") as f:
        json.dump(copy.deepcopy(params), f, indent=2, sort_keys=True)
        f.write("\n")
    with (common.INPUTS_DIR / f"overrides_{TRACEABILITY_STEM}.json").open("w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2, sort_keys=True)
        f.write("\n")
    with (common.INPUTS_DIR / f"summary_compare_{TRACEABILITY_STEM}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    with (common.INPUTS_DIR / f"summary_compare_{TRACEABILITY_STEM}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")


def plot_panel_c_same_length_i0(data: Any) -> list[Path]:
    fig, ax = base.make_single_axis_panel()
    ax.plot(
        data.x_nm,
        data.c_r1_norm,
        color=SPECIES_COLORS["red1_i1"],
        linewidth=2.0,
        label=r"$C_{\mathrm{Red},1}/C_{\mathrm{bulk}}$ (with EDL)",
        zorder=3,
    )
    ax.plot(
        data.x_nm,
        data.c_o2_norm,
        color=SPECIES_COLORS["ox2_i2"],
        linewidth=2.0,
        label=r"$C_{\mathrm{Ox},2}/C_{\mathrm{bulk}}$ (with EDL)",
        zorder=3,
    )
    ax.plot(
        data.x_nm,
        base.np.ones_like(data.x_nm),
        color=base.COLORS["without"],
        linewidth=1.6,
        linestyle=(0, (4, 2)),
        label="w/o EDL",
        zorder=2,
    )
    base.add_boundaries(ax, data)
    ax.set_yscale("log")
    positive = base.np.concatenate(
        [
            data.c_r1_norm[base.np.isfinite(data.c_r1_norm) & (data.c_r1_norm > 0.0)],
            data.c_o2_norm[base.np.isfinite(data.c_o2_norm) & (data.c_o2_norm > 0.0)],
            base.np.array([1.0], dtype=float),
        ]
    )
    ax.set_ylim(float(base.np.min(positive)) / 1.25, float(base.np.max(positive)) * 30.0)
    base.style_axes(ax, "x (nm)", r"$c_i/c_{\mathrm{bulk}}$ (-)", "Local reactant concentration")
    ax.legend(
        loc="upper right",
        fontsize=6.9,
        handlelength=1.7,
        ncols=1,
        borderaxespad=0.0,
    )
    return base.save_panel(fig, "figure_3_panel_c_local_reactant_concentration")


def plot_panel_e_same_length_i0(data: Any) -> list[Path]:
    (i1_edl, i1_no, i2_edl, i2_no), i_label, _ = base.solver._scaled_current_display(
        "local_current_density",
        data.i1_edl_segment,
        data.i1_no_segment,
        data.i2_edl_segment,
        data.i2_no_segment,
    )
    i_label = i_label.replace("Local current density, ", "")

    fig, ax = base.make_single_axis_panel()
    ax.plot(
        data.x_nm,
        i1_edl,
        color=SPECIES_COLORS["red1_i1"],
        linewidth=2.0,
        label=r"$i_1$ (Au), with EDL",
        zorder=4,
    )
    ax.plot(
        data.x_nm,
        i2_edl,
        color=SPECIES_COLORS["ox2_i2"],
        linewidth=2.0,
        label=r"$i_2$ (Pd), with EDL",
        zorder=4,
    )
    ax.plot(
        data.x_nm,
        i1_no,
        color=SPECIES_COLORS["red1_i1"],
        linewidth=1.7,
        linestyle=(0, (4, 2)),
        label=r"$i_1$ (Au), w/o EDL",
        zorder=3,
    )
    ax.plot(
        data.x_nm,
        i2_no,
        color=SPECIES_COLORS["ox2_i2"],
        linewidth=1.7,
        linestyle=(0, (4, 2)),
        label=r"$i_2$ (Pd), w/o EDL",
        zorder=3,
    )
    base.add_boundaries(ax, data)
    base.style_axes(ax, "x (nm)", i_label, base.PANEL_E_TITLE)
    base.finite_ylim(ax, i1_edl, i1_no, i2_edl, i2_no, pad_frac=0.08)
    if base.PANEL_E_YMIN is not None:
        _, ymax = ax.get_ylim()
        ax.set_ylim(base.PANEL_E_YMIN, ymax)
    ax.legend(loc="upper right", fontsize=7.8, handlelength=1.8)
    return base.save_panel(fig, "figure_3_panel_e_local_current_density")


def main() -> None:
    common.ensure_output_dirs()
    params, summary = common.load_inputs_for_scripts()
    common.assert_expected_values(summary)

    def load_params() -> dict[str, Any]:
        return copy.deepcopy(params)

    def load_summary() -> dict[str, str]:
        return dict(summary)

    base.RESULT_ID = common.OUTPUT_TAG
    base.OUT_DIR = common.FIGURE_3_DIR
    base.INPUTS_DIR = common.INPUTS_DIR
    base.TRACEABILITY_STEM = TRACEABILITY_STEM
    base.load_params = load_params
    base.load_summary = load_summary
    base.save_traceability_inputs = save_figure3_traceability
    base.plot_panel_c = plot_panel_c_same_length_i0
    base.plot_panel_e = plot_panel_e_same_length_i0
    base.main()


if __name__ == "__main__":
    main()
