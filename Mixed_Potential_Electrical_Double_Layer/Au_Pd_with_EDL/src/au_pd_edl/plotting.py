"""Publication figures for flush Au/Pd on an insulating substrate.

The functions in this module deliberately accept solved ``cases`` rather than
running the spectral solver.  A case mapping has the form::

    {d_nm: {"result": result_dict, "model": LinearPBModel}}

This keeps plotting reproducible and prevents a figure export from silently
changing physical or numerical parameters.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import Rectangle

from .kinetics import emix_closed_form_no_edl, no_edl_absolute_currents
from .parameters import ELECTROLYTE_DOMAIN_DESCRIPTION, ELECTROLYTE_DOMAIN_ID


RP_SEPARATIONS_NM = (0.0, 2.0, 3.0, 10.0)

COLORS = {
    "dark": "#272727",
    "gray": "#767676",
    "light_gray": "#CFCECE",
    "au": "#C9A227",
    "substrate": "#D9D9D9",
    "pd": "#42949E",
    "with_edl": "#F26B38",
    "with_edl_alt": "#D83A2E",
    "without_edl": "#12355B",
    "reference": "#E4C133",
    # Species colors used consistently by Figure 3 concentration/current
    # pairs.  Green/blue echo the viridis-based Red1/Ox2 2D distributions.
    "red1_i1": "#009E73",
    "ox2_i2": "#0072B2",
}

PUBLICATION_RCPARAMS: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Helvetica",
        "Nimbus Sans",
        "Arial",
        "DejaVu Sans",
        "sans-serif",
    ],
    "font.size": 8.5,
    "axes.linewidth": 0.9,
    "axes.grid": False,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # ``custom`` with an unavailable Nimbus Sans can trigger a FreeType
    # division-by-zero in headless environments.  DejaVu Sans is the final
    # project-approved fallback and is always bundled with Matplotlib.
    "mathtext.fontset": "dejavusans",
    "mathtext.default": "regular",
}


def _validate_tag(tag: str) -> str:
    text = str(tag).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", text):
        raise ValueError(
            "tag must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_', or '-'"
        )
    return text


def _as_finite_float(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _extract_number(
    values: Mapping[str, Any],
    names: Sequence[str],
    label: str,
) -> float:
    for name in names:
        if name in values:
            return _as_finite_float(values[name], label)
    raise KeyError(f"Missing {label}; expected one of: {', '.join(names)}")


def _lookup_case(
    cases: Mapping[float, Mapping[str, Any]],
    requested_nm: float,
    *,
    atol_nm: float = 1.0e-6,
) -> tuple[float, Mapping[str, Any]]:
    matches: list[tuple[float, Mapping[str, Any]]] = []
    for raw_d, case in cases.items():
        d_nm = _as_finite_float(raw_d, "case separation")
        if math.isclose(d_nm, requested_nm, rel_tol=0.0, abs_tol=atol_nm):
            matches.append((d_nm, case))
    if not matches:
        raise KeyError(f"Missing required d = {requested_nm:g} nm case")
    if len(matches) > 1:
        raise ValueError(f"Multiple cases match d = {requested_nm:g} nm")
    return matches[0]


def _case_parts(case: Mapping[str, Any]) -> tuple[Mapping[str, Any], Any]:
    if "result" not in case or "model" not in case:
        raise KeyError("Each case must contain 'result' and 'model'")
    result = case["result"]
    if not isinstance(result, Mapping):
        raise TypeError("case['result'] must be a mapping")
    model = case["model"]
    if not callable(getattr(model, "upper_grid", None)):
        raise TypeError("case['model'] must provide upper_grid()")
    return result, model


def _result_params_and_derived(
    result: Mapping[str, Any],
    model: Any,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    params = result.get("params", getattr(model, "params", None))
    derived = result.get("derived", getattr(model, "derived", None))
    if not isinstance(params, Mapping):
        raise TypeError("result['params'] (or model.params) must be a mapping")
    if not isinstance(derived, Mapping):
        raise TypeError("result['derived'] (or model.derived) must be a mapping")
    return params, derived


def _lambda_D_m(derived: Mapping[str, Any]) -> float:
    value = _extract_number(derived, ("lambda_D", "lambda_D_m"), "lambda_D")
    if value <= 0.0:
        raise ValueError("lambda_D must be positive")
    return value


def _emix_with_edl(result: Mapping[str, Any]) -> float:
    return _extract_number(
        result,
        ("E_mix_V", "E_mix", "E_mix_with_V", "E_mix_with"),
        "with-EDL mixed potential",
    )


def _imix_with_edl(result: Mapping[str, Any]) -> float:
    value = _extract_number(
        result,
        (
            "i_mix_avg_A_per_m2",
            "i_mix_avg",
            "i_mix_avg_with_A_per_m2",
            "i_mix_avg_with",
        ),
        "with-EDL mixed current density",
    )
    if value < 0.0:
        raise ValueError("mixed current density must be non-negative")
    return value


def _emix_without_edl(
    result: Mapping[str, Any],
    params: Mapping[str, Any],
    derived: Mapping[str, Any],
) -> float:
    for name in (
        "E_mix_no_edl_V",
        "E_mix_without_edl_V",
        "E_mix_no_V",
        "E_mix_no",
        "E_mix_without_edl",
    ):
        if name in result:
            return _as_finite_float(result[name], "w/o-EDL mixed potential")
    return float(emix_closed_form_no_edl(params, derived))


def _imix_without_edl(
    result: Mapping[str, Any],
    params: Mapping[str, Any],
    derived: Mapping[str, Any],
    E_mix_no: float,
) -> float:
    for name in (
        "i_mix_avg_no_edl_A_per_m2",
        "i_mix_avg_without_edl_A_per_m2",
        "i_mix_avg_no_A_per_m2",
        "i_mix_avg_no",
        "i_mix_avg_without_edl",
    ):
        if name in result:
            value = _as_finite_float(result[name], "w/o-EDL mixed current density")
            if value < 0.0:
                raise ValueError("w/o-EDL mixed current density must be non-negative")
            return value
    return float(no_edl_absolute_currents(E_mix_no, params, derived)["i_mix_avg_A_per_m2"])


def _shared_log_norm(arrays: Sequence[np.ndarray]) -> LogNorm:
    positives = [
        np.asarray(values, dtype=float)[
            np.isfinite(values) & (np.asarray(values, dtype=float) > 0.0)
        ]
        for values in arrays
    ]
    positives = [values for values in positives if values.size]
    if not positives:
        raise ValueError("Cannot construct LogNorm without positive finite values")
    all_positive = np.concatenate(positives)
    lower = 10.0 ** math.floor(math.log10(float(np.min(all_positive))))
    upper = 10.0 ** math.ceil(math.log10(float(np.max(all_positive))))
    if math.isclose(lower, upper, rel_tol=1.0e-14, abs_tol=0.0):
        lower /= 10.0
        upper *= 10.0
    return LogNorm(vmin=lower, vmax=upper)


def _prepare_rp_case(
    d_nm: float,
    case: Mapping[str, Any],
    *,
    n_x: int,
    n_y: int,
    y_max_over_lambda: float,
) -> dict[str, Any]:
    result, model = _case_parts(case)
    params, derived = _result_params_and_derived(result, model)
    E_mix = _emix_with_edl(result)
    grid = model.upper_grid(
        E_mix,
        n_x=int(n_x),
        n_y=int(n_y),
        y_max_over_lambda=float(y_max_over_lambda),
    )
    x_tilde = np.asarray(grid["x_tilde"], dtype=float)
    y_tilde = np.asarray(grid["y_tilde"], dtype=float)
    phi_tilde = np.asarray(grid["phi_tilde"], dtype=float)
    if x_tilde.shape != (n_x,) or y_tilde.shape != (n_y,):
        raise ValueError("model.upper_grid returned unexpected coordinate dimensions")
    if phi_tilde.shape != (n_y, n_x):
        raise ValueError(
            f"model.upper_grid phi_tilde shape {phi_tilde.shape} != {(n_y, n_x)}"
        )
    if not np.all(np.isfinite(phi_tilde)):
        raise ValueError(f"d={d_nm:g} nm phi_tilde contains non-finite values")

    # Verify the displayed y=0 plane against the solver's dedicated top profile
    # when that diagnostic interface is available.
    if callable(getattr(model, "top_profile", None)):
        profile = model.top_profile(E_mix, n_x=int(n_x))
        profile_x = np.asarray(profile["x_tilde"], dtype=float)
        profile_phi = np.asarray(profile["phi_tilde"], dtype=float)
        if not np.allclose(profile_x, x_tilde, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"d={d_nm:g} nm top-profile x grid does not match upper_grid")
        if not np.allclose(profile_phi, phi_tilde[0], rtol=0.0, atol=1.0e-10):
            raise ValueError(f"d={d_nm:g} nm y=0 field does not match top_profile")

    lambda_D = _lambda_D_m(derived)
    beta = _extract_number(derived, ("beta",), "beta")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    z_red1 = _extract_number(params, ("z_R1",), "z_R1")
    z_ox2 = _extract_number(params, ("z_O2",), "z_O2")
    phi_s_mV = 1000.0 * phi_tilde / beta
    c_red1 = np.exp(np.clip(-z_red1 * phi_tilde, -700.0, 700.0))
    c_ox2 = np.exp(np.clip(-z_ox2 * phi_tilde, -700.0, 700.0))
    if not np.all(np.isfinite(phi_s_mV)):
        raise ValueError(f"d={d_nm:g} nm Phi_s contains non-finite values")
    if not np.all(np.isfinite(c_red1)) or not np.all(c_red1 > 0.0):
        raise ValueError(f"d={d_nm:g} nm c_Red1/c_bulk must be finite and positive")
    if not np.all(np.isfinite(c_ox2)) or not np.all(c_ox2 > 0.0):
        raise ValueError(f"d={d_nm:g} nm c_Ox2/c_bulk must be finite and positive")

    L_Au_nm = _extract_number(derived, ("L_Au",), "L_Au") * 1.0e9
    separation_from_params_nm = _extract_number(
        derived, ("d_Au_Pd",), "d_Au_Pd"
    ) * 1.0e9
    L_Pd_nm = _extract_number(derived, ("L_Pd",), "L_Pd") * 1.0e9
    if not math.isclose(separation_from_params_nm, d_nm, rel_tol=0.0, abs_tol=1.0e-5):
        raise ValueError(
            f"case key d={d_nm:g} nm disagrees with derived d_Au_Pd="
            f"{separation_from_params_nm:.9g} nm"
        )
    L_total_nm = L_Au_nm + d_nm + L_Pd_nm
    x_nm = x_tilde * lambda_D * 1.0e9
    y_nm = y_tilde * lambda_D * 1.0e9
    if not math.isclose(float(x_nm[-1]), L_total_nm, rel_tol=0.0, abs_tol=1.0e-5):
        raise ValueError(
            "upper_grid x extent does not match the flush Au-substrate-Pd geometry"
        )

    return {
        "d_nm": d_nm,
        "result": result,
        "params": params,
        "derived": derived,
        "E_mix_V": E_mix,
        "x_nm": x_nm,
        "y_nm": y_nm,
        "phi_tilde": phi_tilde,
        "phi_s_mV": phi_s_mV,
        "c_red1_norm": c_red1,
        "c_ox2_norm": c_ox2,
        "lambda_D_nm": lambda_D * 1.0e9,
        "L_Au_nm": L_Au_nm,
        "L_Pd_nm": L_Pd_nm,
        "L_total_nm": L_total_nm,
    }


def _style_map_axis(ax: plt.Axes, title: str, *, show_xlabel: bool) -> None:
    ax.set_ylabel("y (nm)")
    ax.set_title(title, loc="left", pad=4.5, fontsize=9.3, fontweight="normal")
    if show_xlabel:
        ax.set_xlabel("x (nm)")
    else:
        ax.tick_params(labelbottom=False)
    ax.tick_params(length=3.2, width=0.85, pad=2.5, labelsize=8.0)
    for name in ("left", "bottom", "top", "right"):
        ax.spines[name].set_visible(True)
        ax.spines[name].set_linewidth(0.9)
        ax.spines[name].set_color(COLORS["dark"])


def _edge_positions(data: Mapping[str, Any]) -> list[float]:
    first = float(data["L_Au_nm"])
    second = first + float(data["d_nm"])
    if math.isclose(first, second, rel_tol=0.0, abs_tol=1.0e-9):
        return [first]
    return [first, second]


def _add_edge_lines(ax: plt.Axes, data: Mapping[str, Any]) -> None:
    for position in _edge_positions(data):
        ax.axvline(
            position,
            color=COLORS["gray"],
            linewidth=0.8,
            linestyle=(0, (3, 2)),
            alpha=0.80,
            zorder=5,
        )


def _material_lane_ticks(data: Mapping[str, Any]) -> list[float]:
    d_nm = float(data["d_nm"])
    L_Au_nm = float(data["L_Au_nm"])
    L_total_nm = float(data["L_total_nm"])
    ticks = [0.0, L_Au_nm]
    # For 2 and 3 nm, omit the second edge tick while retaining both edge lines.
    if not math.isclose(d_nm, 0.0, abs_tol=1.0e-9) and not math.isclose(
        d_nm, 2.0, abs_tol=1.0e-9
    ) and not math.isclose(d_nm, 3.0, abs_tol=1.0e-9):
        ticks.append(L_Au_nm + d_nm)
    ticks.append(L_total_nm)
    return ticks


def _add_material_lane(ax: plt.Axes, data: Mapping[str, Any]) -> None:
    L_Au_nm = float(data["L_Au_nm"])
    d_nm = float(data["d_nm"])
    L_total_nm = float(data["L_total_nm"])
    x_pd = L_Au_nm + d_nm
    segments = [
        ("Au", 0.0, L_Au_nm, COLORS["au"], COLORS["dark"]),
        ("", L_Au_nm, x_pd, COLORS["substrate"], COLORS["dark"]),
        ("Pd", x_pd, L_total_nm, COLORS["pd"], "white"),
    ]
    for label, x0, x1, face, text_color in segments:
        if x1 - x0 <= 1.0e-9:
            continue
        ax.add_patch(
            Rectangle(
                (x0, 0.0),
                x1 - x0,
                1.0,
                facecolor=face,
                edgecolor="white",
                linewidth=0.75,
            )
        )
        if label:
            ax.text(
                0.5 * (x0 + x1),
                0.5,
                label,
                ha="center",
                va="center",
                fontsize=8.3,
                color=text_color,
            )
    if d_nm > 0.0:
        substrate_center = L_Au_nm + 0.5 * d_nm
        if d_nm >= 8.0:
            ax.text(
                substrate_center,
                0.5,
                "insulating\nsubstrate",
                ha="center",
                va="center",
                fontsize=5.3,
                color=COLORS["dark"],
                linespacing=0.9,
            )
        else:
            # The 2/3 nm segment cannot fit a horizontal phrase; retain a
            # legible material identity with a vertical short label.
            ax.text(
                substrate_center,
                0.5,
                "substrate",
                ha="center",
                va="center",
                rotation=90,
                fontsize=5.2,
                color=COLORS["dark"],
            )
    ax.set_xlim(0.0, L_total_nm)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()


def _draw_field_panel(
    ax: plt.Axes,
    cax: plt.Axes,
    data: Mapping[str, Any],
    values: np.ndarray,
    *,
    cmap: str,
    norm: LogNorm | TwoSlopeNorm,
    levels: np.ndarray,
    title: str,
    colorbar_label: str,
    show_xlabel: bool,
) -> None:
    field_artist = ax.pcolormesh(
        data["x_nm"],
        data["y_nm"],
        values,
        shading="auto",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    ax.contour(
        data["x_nm"],
        data["y_nm"],
        values,
        levels=levels,
        colors="black",
        linewidths=0.28,
        alpha=0.27,
    )
    _add_edge_lines(ax, data)
    _style_map_axis(ax, title, show_xlabel=show_xlabel)
    colorbar = ax.figure.colorbar(field_artist, cax=cax)
    colorbar.set_label(colorbar_label, labelpad=5)
    colorbar.ax.tick_params(length=2.8, width=0.75, labelsize=7.8, pad=2.2)
    colorbar.outline.set_linewidth(0.8)


def _save_png_svg(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    dpi: int,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for extension in ("png", "svg"):
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(
            path,
            dpi=int(dpi),
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )
        paths.append(str(path.resolve()))
    plt.close(fig)
    unexpected_pdf = output_dir / f"{stem}.pdf"
    if unexpected_pdf.exists():
        raise RuntimeError(f"Unexpected PDF exists for this output stem: {unexpected_pdf}")
    return paths


def _plot_one_rp_case(
    data: Mapping[str, Any],
    output_dir: Path,
    tag: str,
    *,
    phi_norm: TwoSlopeNorm,
    red_norm: LogNorm,
    ox_norm: LogNorm,
    phi_levels: np.ndarray,
    red_levels: np.ndarray,
    ox_levels: np.ndarray,
    dpi: int,
) -> list[str]:
    fig = plt.figure(figsize=(5.8, 6.9))
    grid = fig.add_gridspec(
        nrows=4,
        ncols=2,
        width_ratios=(1.0, 0.038),
        height_ratios=(1.0, 1.0, 1.0, 0.12),
        hspace=0.22,
        wspace=0.08,
    )
    axes = [fig.add_subplot(grid[index, 0]) for index in range(3)]
    color_axes = [fig.add_subplot(grid[index, 1]) for index in range(3)]
    lane_axis = fig.add_subplot(grid[3, 0], sharex=axes[-1])
    fig.add_subplot(grid[3, 1]).set_axis_off()

    _draw_field_panel(
        axes[0],
        color_axes[0],
        data,
        np.asarray(data["phi_s_mV"]),
        cmap="RdBu_r",
        norm=phi_norm,
        levels=phi_levels,
        title="Solution-phase potential",
        colorbar_label=r"$\Phi_s$ (mV)",
        show_xlabel=False,
    )
    _draw_field_panel(
        axes[1],
        color_axes[1],
        data,
        np.asarray(data["c_red1_norm"]),
        cmap="viridis",
        norm=red_norm,
        levels=red_levels,
        title="Reactant Red1 distribution",
        colorbar_label=r"$c_{\mathrm{Red1}}/c_{\mathrm{bulk}}$",
        show_xlabel=False,
    )
    _draw_field_panel(
        axes[2],
        color_axes[2],
        data,
        np.asarray(data["c_ox2_norm"]),
        cmap="viridis",
        norm=ox_norm,
        levels=ox_levels,
        title="Reactant Ox2 distribution",
        colorbar_label=r"$c_{\mathrm{Ox2}}/c_{\mathrm{bulk}}$",
        show_xlabel=False,
    )

    for ax in axes:
        ax.set_xlim(0.0, float(data["L_total_nm"]))
        ax.set_ylim(0.0, float(np.asarray(data["y_nm"])[-1]))
        y_stop = float(np.asarray(data["y_nm"])[-1])
        ax.set_yticks(np.arange(0.0, math.floor(y_stop / 5.0) * 5.0 + 0.1, 5.0))
    x_ticks = _material_lane_ticks(data)
    axes[-1].set_xticks(x_ticks)
    axes[-1].set_xticklabels([f"{value:g}" for value in x_ticks])
    _add_material_lane(lane_axis, data)
    lane_axis.text(
        0.5,
        -0.55,
        "x (nm)",
        transform=lane_axis.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        clip_on=False,
    )
    fig.align_ylabels(axes)
    fig.suptitle(
        rf"Au–Pd separation $d$ = {float(data['d_nm']):g} nm; "
        rf"$E_{{\mathrm{{mix}}}}$ = {float(data['E_mix_V']):.6f} V",
        fontsize=10.0,
        y=0.995,
    )
    fig.subplots_adjust(top=0.93)

    d_code = int(round(float(data["d_nm"])))
    stem = f"phi_s_reactants_2d_d{d_code:03d}nm_{tag}"
    return _save_png_svg(fig, output_dir, stem, dpi=dpi)


def generate_rp_2d_figures(
    cases: Mapping[float, Mapping[str, Any]],
    output_dir: str | Path,
    tag: str,
    *,
    n_x: int = 600,
    n_y: int = 320,
    y_max_over_lambda: float = 5.0,
    dpi: int = 600,
) -> dict[str, Any]:
    """Export the required d=0/2/3/10 nm three-layer field figures.

    All four figures use the same grid dimensions, y/lambda range, symmetric
    potential color scale, concentration LogNorm limits, and contour levels.
    The return value is JSON-safe metadata suitable for a run manifest.
    """

    clean_tag = _validate_tag(tag)
    if int(n_x) < 2 or int(n_y) < 2:
        raise ValueError("n_x and n_y must both be >= 2")
    if float(y_max_over_lambda) <= 0.0:
        raise ValueError("y_max_over_lambda must be positive")
    if int(dpi) <= 0:
        raise ValueError("dpi must be positive")

    prepared: list[dict[str, Any]] = []
    for requested in RP_SEPARATIONS_NM:
        actual, case = _lookup_case(cases, requested)
        prepared.append(
            _prepare_rp_case(
                actual,
                case,
                n_x=int(n_x),
                n_y=int(n_y),
                y_max_over_lambda=float(y_max_over_lambda),
            )
        )

    max_abs_phi_mV = max(
        float(np.max(np.abs(np.asarray(data["phi_s_mV"])))) for data in prepared
    )
    phi_limit_mV = max(10.0, 10.0 * math.ceil(max_abs_phi_mV / 10.0))
    phi_norm = TwoSlopeNorm(vmin=-phi_limit_mV, vcenter=0.0, vmax=phi_limit_mV)
    red_norm = _shared_log_norm(
        [np.asarray(data["c_red1_norm"]) for data in prepared]
    )
    ox_norm = _shared_log_norm([np.asarray(data["c_ox2_norm"]) for data in prepared])
    phi_levels = np.linspace(-phi_limit_mV, phi_limit_mV, 11)
    red_levels = np.geomspace(float(red_norm.vmin), float(red_norm.vmax), 9)
    ox_levels = np.geomspace(float(ox_norm.vmin), float(ox_norm.vmax), 9)

    output = Path(output_dir)
    saved_by_separation: dict[str, list[str]] = {}
    case_metadata: list[dict[str, Any]] = []
    with plt.rc_context(PUBLICATION_RCPARAMS):
        for data in prepared:
            d_label = f"{int(round(float(data['d_nm']))):03d}"
            saved_by_separation[d_label] = _plot_one_rp_case(
                data,
                output,
                clean_tag,
                phi_norm=phi_norm,
                red_norm=red_norm,
                ox_norm=ox_norm,
                phi_levels=phi_levels,
                red_levels=red_levels,
                ox_levels=ox_levels,
                dpi=int(dpi),
            )
            case_metadata.append(
                {
                    "d_nm": float(data["d_nm"]),
                    "E_mix_V": float(data["E_mix_V"]),
                    "lambda_D_nm": float(data["lambda_D_nm"]),
                    "x_range_nm": [0.0, float(data["L_total_nm"])],
                    "y_range_nm": [0.0, float(np.asarray(data["y_nm"])[-1])],
                    "max_abs_phi_tilde": float(
                        np.max(np.abs(np.asarray(data["phi_tilde"])))
                    ),
                }
            )

    all_paths = [path for paths in saved_by_separation.values() for path in paths]
    if len(all_paths) != 8 or sum(path.endswith(".png") for path in all_paths) != 4:
        raise RuntimeError("Expected exactly 4 PNG and 4 SVG RP outputs")
    if sum(path.endswith(".svg") for path in all_paths) != 4:
        raise RuntimeError("Expected exactly 4 PNG and 4 SVG RP outputs")
    return {
        "figure_type": "phi_s_reactants_2d_batch",
        "topology_id": "flush_coplanar_on_uncharged_insulating_substrate",
        "electrolyte_domain_id": ELECTROLYTE_DOMAIN_ID,
        "electrolyte_domain_description": ELECTROLYTE_DOMAIN_DESCRIPTION,
        "substrate_bc": "homogeneous_neumann",
        "tag": clean_tag,
        "formats": ["png", "svg"],
        "dpi": int(dpi),
        "grid": {
            "n_x": int(n_x),
            "n_y": int(n_y),
            "y_max_over_lambda": float(y_max_over_lambda),
        },
        "shared_color_limits": {
            "phi_s_mV": [-phi_limit_mV, phi_limit_mV],
            "c_Red1_over_c_bulk": [float(red_norm.vmin), float(red_norm.vmax)],
            "c_Ox2_over_c_bulk": [float(ox_norm.vmin), float(ox_norm.vmax)],
        },
        "shared_contour_levels": {
            "phi_s_mV": phi_levels.tolist(),
            "c_Red1_over_c_bulk": red_levels.tolist(),
            "c_Ox2_over_c_bulk": ox_levels.tolist(),
        },
        "cases": case_metadata,
        "saved_paths_by_separation": saved_by_separation,
        "saved_paths": all_paths,
    }


def _collect_trend_series(
    cases_or_scan_result: Mapping[Any, Any],
) -> dict[str, np.ndarray]:
    rows: list[tuple[float, float, float, float, float]] = []
    if "rows" in cases_or_scan_result:
        raw_rows = cases_or_scan_result["rows"]
        if isinstance(raw_rows, (str, bytes)) or not isinstance(raw_rows, Sequence):
            raise TypeError("scan_result['rows'] must be a sequence of mappings")
        for index, row in enumerate(raw_rows):
            if not isinstance(row, Mapping):
                raise TypeError(f"scan_result['rows'][{index}] must be a mapping")
            rows.append(
                (
                    _extract_number(
                        row,
                        ("d_Au_Pd_nm", "d_nm"),
                        f"scan row {index} separation",
                    ),
                    _extract_number(
                        row,
                        ("E_mix_with_EDL_V", "E_mix_with_V", "E_mix_V"),
                        f"scan row {index} with-EDL mixed potential",
                    ),
                    _extract_number(
                        row,
                        (
                            "E_mix_without_EDL_V",
                            "E_mix_no_edl_V",
                            "E_mix_no_V",
                        ),
                        f"scan row {index} w/o-EDL mixed potential",
                    ),
                    _extract_number(
                        row,
                        (
                            "i_mix_avg_with_EDL_A_per_m2",
                            "i_mix_avg_with_A_per_m2",
                            "i_mix_avg_A_per_m2",
                        ),
                        f"scan row {index} with-EDL mixed current density",
                    ),
                    _extract_number(
                        row,
                        (
                            "i_mix_avg_without_EDL_A_per_m2",
                            "i_mix_avg_no_edl_A_per_m2",
                            "i_mix_avg_no_A_per_m2",
                        ),
                        f"scan row {index} w/o-EDL mixed current density",
                    ),
                )
            )
    else:
        # Direct cases input remains useful for small programmatic studies.
        for raw_d, case in cases_or_scan_result.items():
            d_nm = _as_finite_float(raw_d, "case separation")
            result, model = _case_parts(case)
            params, derived = _result_params_and_derived(result, model)
            E_with = _emix_with_edl(result)
            i_with = _imix_with_edl(result)
            E_no = _emix_without_edl(result, params, derived)
            i_no = _imix_without_edl(result, params, derived, E_no)
            rows.append((d_nm, E_with, E_no, i_with, i_no))
    if not rows:
        raise ValueError("cases must not be empty")
    rows.sort(key=lambda row: row[0])
    values = np.asarray(rows, dtype=float)
    if np.any(values[:, 0] < 0.0) or np.any(values[:, 0] > 100.0 + 1.0e-9):
        raise ValueError("trend separations must lie within 0-100 nm")
    if np.any(values[:, 3:] < 0.0):
        raise ValueError("trend mixed current densities must be non-negative")
    return {
        "d_nm": values[:, 0],
        "E_with_V": values[:, 1],
        "E_without_V": values[:, 2],
        "i_with_A_per_m2": values[:, 3],
        "i_without_A_per_m2": values[:, 4],
    }


def plot_separation_trends(
    cases_or_scan_result: Mapping[Any, Any],
    output_dir: str | Path,
    tag: str,
    *,
    dpi: int = 600,
    x_max_nm: float = 100.0,
) -> dict[str, Any]:
    """Plot mixed potential/current density versus Au-Pd separation.

    The two panels show with-EDL values, the separation-independent w/o-EDL
    control, and the with-EDL d=100 nm reference as a horizontal line.  Input
    may be the complete ``run_separation_scan`` result (preferred, no retained
    spectral models needed) or the original direct cases mapping.  ``x_max_nm``
    changes only the visible separation range; the d=100 nm case remains the
    no-overlap reference and the full scan continues to determine the y scale.
    """

    clean_tag = _validate_tag(tag)
    x_max = _as_finite_float(x_max_nm, "x_max_nm")
    if not 0.0 < x_max <= 100.0:
        raise ValueError("x_max_nm must be within (0, 100]")
    series = _collect_trend_series(cases_or_scan_result)
    d_100_index = np.flatnonzero(np.isclose(series["d_nm"], 100.0, atol=1.0e-6))
    if d_100_index.size != 1:
        raise ValueError("Trend figure requires exactly one d = 100 nm case")
    index_100 = int(d_100_index[0])
    E_ref = float(series["E_with_V"][index_100])
    i_ref = float(series["i_with_A_per_m2"][index_100])
    visible = series["d_nm"] <= x_max + 1.0e-9
    if not np.any(visible):
        raise ValueError(f"No trend separations lie within 0-{x_max:g} nm")
    visible_series = {name: values[visible] for name, values in series.items()}
    full_y_limits: list[tuple[float, float]] = []
    for values in (
        np.concatenate((series["E_with_V"], series["E_without_V"], [E_ref])),
        np.concatenate(
            ((series["i_with_A_per_m2"], series["i_without_A_per_m2"], [i_ref]))
        ),
    ):
        lower = float(np.min(values))
        upper = float(np.max(values))
        span = upper - lower
        if span <= 0.0:
            padding = 0.05 * abs(lower) if lower != 0.0 else 0.05
        else:
            padding = 0.10 * span
        full_y_limits.append((lower - padding, upper + padding))

    with plt.rc_context({**PUBLICATION_RCPARAMS, "font.size": 9.5}):
        fig, axes = plt.subplots(2, 1, figsize=(6.2, 6.1), sharex=True)
        line_with, = axes[0].plot(
            visible_series["d_nm"],
            visible_series["E_with_V"],
            color=COLORS["with_edl"],
            linewidth=2.0,
            marker="o",
            markersize=3.8,
            markeredgewidth=0.0,
            label="with EDL",
            zorder=4,
        )
        line_without, = axes[0].plot(
            visible_series["d_nm"],
            visible_series["E_without_V"],
            color=COLORS["without_edl"],
            linewidth=1.8,
            linestyle=(0, (5, 2.5)),
            label="w/o EDL",
            zorder=3,
        )
        line_reference = axes[0].axhline(
            E_ref,
            color=COLORS["reference"],
            linewidth=1.5,
            linestyle=(0, (1.5, 2.0)),
            label="with EDL at 100 nm",
            zorder=2,
        )

        axes[1].plot(
            visible_series["d_nm"],
            visible_series["i_with_A_per_m2"],
            color=COLORS["with_edl"],
            linewidth=2.0,
            marker="o",
            markersize=3.8,
            markeredgewidth=0.0,
            zorder=4,
        )
        axes[1].plot(
            visible_series["d_nm"],
            visible_series["i_without_A_per_m2"],
            color=COLORS["without_edl"],
            linewidth=1.8,
            linestyle=(0, (5, 2.5)),
            zorder=3,
        )
        axes[1].axhline(
            i_ref,
            color=COLORS["reference"],
            linewidth=1.5,
            linestyle=(0, (1.5, 2.0)),
            zorder=2,
        )

        axes[0].set_ylabel(r"Mixed potential, $E_{\mathrm{mix}}$ (V vs. RHE)")
        axes[1].set_ylabel(
            r"Mixed current density, $\bar{i}_{\mathrm{mix}}$ (A/m$^2$)"
        )
        axes[1].set_xlabel(r"Au–Pd separation, $d$ (nm)")
        axes[0].set_title("Mixed potential", loc="left", fontsize=10.2)
        axes[1].set_title("Mixed current density", loc="left", fontsize=10.2)
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(length=3.5, width=0.9, direction="out")
            ax.set_xlim(0.0, x_max)
            ax.margins(y=0.10)
        if not math.isclose(x_max, 100.0, rel_tol=0.0, abs_tol=1.0e-9):
            for ax, limits in zip(axes, full_y_limits, strict=True):
                ax.set_ylim(*limits)
        if math.isclose(x_max, 20.0, rel_tol=0.0, abs_tol=1.0e-9):
            axes[1].set_xticks([0.0, 5.0, 10.0, 15.0, 20.0])
        fig.legend(
            [line_with, line_without, line_reference],
            ["with EDL", "w/o EDL", "with EDL at 100 nm"],
            loc="upper center",
            bbox_to_anchor=(0.53, 0.995),
            ncol=3,
            handlelength=2.7,
            columnspacing=1.5,
        )
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.10, top=0.88, hspace=0.36)
        if math.isclose(x_max, 100.0, rel_tol=0.0, abs_tol=1.0e-9):
            stem = f"emix_imix_vs_separation_{clean_tag}"
        elif math.isclose(x_max, 20.0, rel_tol=0.0, abs_tol=1.0e-9):
            stem = f"emix_imix_vs_separation_0_20nm_{clean_tag}"
        else:
            x_code = f"{x_max:g}".replace(".", "p")
            stem = f"emix_imix_vs_separation_0_{x_code}nm_{clean_tag}"
        saved_paths = _save_png_svg(fig, Path(output_dir), stem, dpi=int(dpi))

    metadata = {
        "figure_type": "emix_imix_vs_separation",
        "tag": clean_tag,
        "formats": ["png", "svg"],
        "dpi": int(dpi),
        "saved_paths": saved_paths,
        "separations_nm": visible_series["d_nm"].tolist(),
        "series": {
            "E_mix_with_EDL_V": visible_series["E_with_V"].tolist(),
            "E_mix_without_EDL_V": visible_series["E_without_V"].tolist(),
            "i_mix_avg_with_EDL_A_per_m2": visible_series[
                "i_with_A_per_m2"
            ].tolist(),
            "i_mix_avg_without_EDL_A_per_m2": visible_series[
                "i_without_A_per_m2"
            ].tolist(),
        },
        "reference_100nm": {
            "E_mix_with_EDL_V": E_ref,
            "i_mix_avg_with_EDL_A_per_m2": i_ref,
        },
    }
    if not math.isclose(x_max, 100.0, rel_tol=0.0, abs_tol=1.0e-9):
        metadata["x_range_nm"] = [0.0, x_max]
        metadata["source_row_count"] = int(series["d_nm"].size)
        metadata["visible_row_count"] = int(np.count_nonzero(visible))
        metadata["n_visible_points"] = metadata["visible_row_count"]
        if math.isclose(x_max, 20.0, rel_tol=0.0, abs_tol=1.0e-9):
            metadata["x_ticks_nm"] = [0.0, 5.0, 10.0, 15.0, 20.0]
    return metadata


__all__ = [
    "COLORS",
    "PUBLICATION_RCPARAMS",
    "RP_SEPARATIONS_NM",
    "generate_rp_2d_figures",
    "plot_separation_trends",
]
