from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SAME_LENGTH_I0_ALPHA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SAME_LENGTH_I0_ALPHA_DIR))

import same_length_i0_alpha_common as common  # noqa: E402
from Figures.Figrue_RP import make_phi_s_reactants_2d as base  # noqa: E402


FIGSIZE = (2.6, 7.45)
PANEL_HSPACE = 0.36


def plot_phi_s_reactants_same_length_i0_alpha(data: Any) -> list[Path]:
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
        cbar_label=r"Φ$_s$ $\mathrm{(mV)}$",
        title=rf"Solution phase potential, $E_{{\mathrm{{mix}}}}$ = {float(data.res_edl['E_mix']):.2f} V",
        contour_levels=phi_levels,
    )
    caxes[0].yaxis.label.set_fontstyle("italic")
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
    axes[-1].set_xticks([0.0, data.L_Au_nm, data.L_C_nm, data.L_total_nm])
    axes[-1].set_xticklabels(
        [
            "0",
            f"{data.L_Au_nm:.0f}",
            f"{data.L_C_nm:.0f}",
            f"{data.L_total_nm:.0f}",
        ]
    )

    base.add_material_lane(lane_ax, data)
    for text in lane_ax.texts:
        if text.get_text() == "support":
            text.set_fontsize(5.8)
    fig.align_ylabels(axes)
    return base.save_figure(fig)


def main() -> None:
    common.ensure_output_dirs()
    params, summary = common.load_inputs_for_scripts()
    common.assert_expected_values(summary)

    def load_params() -> dict[str, Any]:
        return copy.deepcopy(params)

    def load_summary() -> dict[str, str]:
        return dict(summary)

    base.RESULT_ID = common.OUTPUT_TAG
    base.OUT_DIR = common.FIGRUE_RP_DIR
    base.load_params = load_params
    base.load_summary = load_summary
    base.plot_phi_s_reactants = plot_phi_s_reactants_same_length_i0_alpha
    base.main()


if __name__ == "__main__":
    main()
