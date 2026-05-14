"""
EDL ↔ mixed potential ↔ kinetics self-consistent solver (PDF-faithful)
======================================================================

Implements the equations in:
- Derivation_260115_for me.pdf (equation numbers like S1-xx, S3-xx, D5-xx)
- Mixed potential _Potential distribution_251230.pdf (main narrative; repeats key relations)

Requirements:
    pip install numpy scipy pandas matplotlib

Outputs (auto-created):
    ./results/<timestamp>/
        results_summary.csv
        sensitivities.csv
        profiles.npz
        figures/*.png
        ofat_<param>.csv
        heatmap_*.csv

Core assumptions (match the PDFs):
- Debye–Hückel / linearized Poisson–Boltzmann (Eq. (S1-4))
- Linear charging boundary condition (Eq. (S-12a))
- Irreversible Frumkin-corrected Butler–Volmer kinetics (Eqs. (S3-4),(S3-5) or (S3-8),(S3-9))
- Mixed potential defined by net-current = 0 (Eq. (S3-7b))

If you later switch to nonlinear PB or potential-dependent C_dl, the affine EDL decomposition
(Eqs. (D1-4)–(D3-2)) no longer holds (see D4 "Scope and limitations").
"""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Callable, cast

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

try:
    import plotly.graph_objects as go  # pyright: ignore[reportMissingImports]
except Exception:
    go = None

import scipy.linalg as la
from scipy.optimize import root_scalar

SCRIPT_DIR = Path(__file__).resolve().parent
NATURE_COLORS = {
    "blue": "#355C7D",
    "orange": "#C06C52",
    "green": "#3B7A57",
    "gold": "#C9A227",
    "gray": "#6B7280",
    "black": "#111827",
}
NATURE_SINGLE_FIGSIZE = (3.35, 2.55)
NATURE_WIDE_FIGSIZE = (4.5, 2.9)
NATURE_DOUBLE_FIGSIZE = (6.9, 2.9)
LEGEND_WITH_EDL = "with EDL"
LEGEND_WITHOUT_EDL = "without EDL"
HEATMAP_TITLE_FONTSIZE = 14.2
HEATMAP_AXIS_LABEL_FONTSIZE = 13.8
HEATMAP_TICK_LABEL_FONTSIZE = 12.4
HEATMAP_COLORBAR_LABEL_FONTSIZE = 11.8
HEATMAP_COLORBAR_TICK_FONTSIZE = 10.8
HEATMAP_BASELINE_MARKER_SIZE = 22.0


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": NATURE_COLORS["black"],
            "axes.labelcolor": NATURE_COLORS["black"],
            "axes.linewidth": 0.9,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": False,
            "axes.axisbelow": True,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.color": NATURE_COLORS["black"],
            "ytick.color": NATURE_COLORS["black"],
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "lines.linewidth": 2.0,
            "patch.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "font.size": 8.5,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        }
    )


_configure_matplotlib()


# -----------------------------
# Utilities
# -----------------------------

def safe_exp(x: np.ndarray | float, clip: float = 700.0) -> np.ndarray | float:
    """Safe exp() with clipping to avoid overflow."""
    return np.exp(np.clip(x, -clip, clip))


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def trapz_compat(y: np.ndarray | float, x: np.ndarray | float) -> float:
    """Compatibility wrapper: use np.trapezoid if available, else np.trapz."""
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    func = getattr(np, "trapezoid", None)
    if func is None:
        func = getattr(np, "trapz")
    trapz_func = cast(Callable[[np.ndarray, np.ndarray], float], func)
    return float(trapz_func(y_arr, x_arr))


def J_n(n: int, a: float, b: float, rho_n: float) -> float:
    """Eq. (S1-26): J_n(a,b) = ∫_a^b cos(rho_n x̃) dx̃"""
    if n == 0:
        return b - a
    return (math.sin(rho_n * b) - math.sin(rho_n * a)) / rho_n


def I_mn(m: int, n: int, a: float, b: float, rho_m: float, rho_n: float) -> float:
    """Eq. (S1-27): I_mn(a,b) = ∫_a^b cos(rho_m x̃) cos(rho_n x̃) dx̃"""
    if m == n:
        rm = rho_m
        return 0.5 * (b - a) + (math.sin(2 * rm * b) - math.sin(2 * rm * a)) / (4 * rm)
    denom1 = rho_m - rho_n
    denom2 = rho_m + rho_n
    term1 = (math.sin(denom1 * b) - math.sin(denom1 * a)) / denom1
    term2 = (math.sin(denom2 * b) - math.sin(denom2 * a)) / denom2
    return 0.5 * (term1 + term2)


def make_edges(vals: np.ndarray, scale: str) -> np.ndarray:
    """Build bin edges from centers for pcolormesh heatmaps."""
    v = np.asarray(vals, dtype=float)
    if v.ndim != 1 or len(v) < 2:
        raise ValueError("make_edges needs a 1D array of length >= 2")
    if scale == "log":
        if np.any(v <= 0):
            raise ValueError("log-scale edges require positive values")
        lv = np.log(v)
        edges = np.empty(len(v) + 1, dtype=float)
        edges[1:-1] = 0.5 * (lv[:-1] + lv[1:])
        edges[0] = lv[0] - (edges[1] - lv[0])
        edges[-1] = lv[-1] + (lv[-1] - edges[-2])
        return np.exp(edges)
    edges = np.empty(len(v) + 1, dtype=float)
    edges[1:-1] = 0.5 * (v[:-1] + v[1:])
    edges[0] = v[0] - (edges[1] - v[0])
    edges[-1] = v[-1] + (v[-1] - edges[-2])
    return edges


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a (possibly nested) dict for CSV saving."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        kk = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_dict(v, prefix=kk + "."))
        else:
            out[kk] = v
    return out


LAMBDA_D_DEP_KEYS = {"C_tot", "epsilon_r", "epsilon0", "epsilon_s", "T", "R", "F"}


def apply_param_overrides(
    base_params: Dict[str, Any],
    overrides: Optional[Dict[str, Any]],
    reset_lambda_D: bool = True,
) -> Dict[str, Any]:
    """
    Return a NEW params dict with overrides applied.

    Beginner note:
    - If you change C_tot / epsilon_r / T, the Debye length should be recalculated.
    - To avoid mistakes, this function auto-sets lambda_D=None when needed,
      unless you explicitly provide "lambda_D" in overrides.
    """
    p = copy.deepcopy(base_params)
    if not overrides:
        return p

    unknown = sorted(set(overrides) - set(base_params))
    if unknown:
        raise KeyError(f"Unknown parameter override(s): {', '.join(unknown)}")

    for k, v in overrides.items():
        p[k] = v

    if reset_lambda_D and "lambda_D" not in overrides:
        if any(k in overrides for k in LAMBDA_D_DEP_KEYS):
            # Safety: force recompute if user changed inputs that affect lambda_D.
            p["lambda_D"] = None

    return p


def load_overrides_json(path: str | Path) -> Dict[str, Any]:
    """
    Load a JSON file that only contains parameter overrides.
    Example JSON:
        {"L_gap": 20e-9, "C_tot": 200.0}
    """
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("JSON overrides must be a dict at the top level.")
    return data


HOVER_PARAM_KEYS = [
    "C_tot", "lambda_D", "epsilon_r", "T",
    "pH",
    "L_Au", "L_gap", "L_Pd_len",
    "Cdl_Au", "Cdl_C", "Cdl_Pd",
    "pzc_Au", "pzc_C", "pzc_Pd",
    "it0_1", "it0_2", "alpha1", "alpha2",
    "z_R1", "z_O2", "E1_eq", "E2_eq",
]

_PLOTLY_WARNED = False
_NO_EDL_WARNED = False
_DH_WARNED = False


def format_hover_params(params: Dict[str, Any], keys: Optional[List[str]] = None) -> str:
    """Format parameter dict for Plotly hover text."""
    flat = flatten_dict(params)
    rxn = _effective_reaction_params(params)
    if keys is None:
        keys = sorted(flat.keys())
    lines: List[str] = []
    for key in keys:
        if key not in flat:
            continue
        val = flat[key]
        if key == "C_tot":
            lines.append(f"{key}={concentration_mol_per_m3_to_M(float(val)):.6g} M")
            continue
        if isinstance(val, float):
            lines.append(f"{key}={val:.6g}")
        else:
            lines.append(f"{key}={val}")
    lines.append(f"E1_eq_eff={rxn['E1_eq_eff']:.6g}")
    lines.append(f"E2_eq_eff={rxn['E2_eq_eff']:.6g}")
    lines.append(f"it0_1_eff={rxn['it0_1_eff']:.6g}")
    lines.append(f"it0_2_eff={rxn['it0_2_eff']:.6g}")
    return "<br>".join(lines)


def _surface_grid_with_boundaries(L_total: float, L_Au: float, L_C: float, Nx: int) -> np.ndarray:
    """
    Uniform surface grid with exact material boundaries inserted.

    This removes the O(dx) integration ambiguity at x=L_Au and x=L_C when
    FULL-mode segment integrals are evaluated by trapezoidal quadrature.
    """
    x = np.linspace(0.0, L_total, Nx, dtype=float)
    x = np.concatenate([x, np.array([0.0, L_Au, L_C, L_total], dtype=float)])
    x = np.unique(np.round(x, decimals=15))
    x.sort()
    return x


def _new_figure(figsize: Tuple[float, float] = NATURE_SINGLE_FIGSIZE) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _style_axes(
    ax: Axes,
    xlabel: str,
    ylabel: str,
    title: str,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=6, fontweight="semibold")
    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)
    ax.tick_params(length=3.5, width=0.8, pad=2)
    for spine in ("left", "bottom", "top", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.9)
        ax.spines[spine].set_color(NATURE_COLORS["black"])


def _add_vertical_boundaries(ax: Axes, *positions: float) -> None:
    for xpos in positions:
        ax.axvline(xpos, linestyle=(0, (3, 2)), linewidth=0.9, color=NATURE_COLORS["gray"], alpha=0.9)


def _finalize_figure(fig: Figure, path: Path) -> None:
    fig.tight_layout(pad=0.35)
    fig.savefig(path)
    plt.close(fig)


def _style_plotly_figure(fig, title: str, xaxis_title: str, yaxis_title: str) -> None:
    fig.update_layout(
        template="simple_white",
        title=dict(text=title, x=0.0, xanchor="left"),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        font=dict(family="Arial, Helvetica, sans-serif", size=13, color=NATURE_COLORS["black"]),
        colorway=[NATURE_COLORS["blue"], NATURE_COLORS["orange"], NATURE_COLORS["green"], NATURE_COLORS["gold"]],
        width=760,
        height=470,
        margin=dict(l=72, r=24, t=56, b=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0),
        hovermode="closest",
    )
    fig.update_xaxes(showline=True, mirror=True, linewidth=1.0, linecolor=NATURE_COLORS["black"], ticks="outside", showgrid=False, zeroline=False)
    fig.update_yaxes(showline=True, mirror=True, linewidth=1.0, linecolor=NATURE_COLORS["black"], ticks="outside", showgrid=False, zeroline=False)


def _add_plotly_boundaries(fig, *positions: float) -> None:
    for xpos in positions:
        fig.add_vline(x=xpos, line_dash="dash", line_color=NATURE_COLORS["gray"], line_width=1.0)


def _maybe_warn_plotly() -> None:
    global _PLOTLY_WARNED
    if go is None and not _PLOTLY_WARNED:
        print("Plotly not installed; skipping interactive HTML. Install with: pip install plotly")
        _PLOTLY_WARNED = True


def write_plotly_html(fig, path: Path) -> None:
    if go is None:
        _maybe_warn_plotly()
        return
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)


PLOT_AXIS_LABELS = {
    "phi2": r"Solution reaction-plane potential, $\phi_2(x)$ [V]",
    "metal_potential": r"Metal potential, $E_{\mathrm{m}}$ [V]",
    "local_current_density": r"Local current density, $i(x)$ [A/m$^2$]",
    "E_mix": r"Mixed potential, $E_{\mathrm{mix}}$ [V]",
    "overpotential": r"Local overpotential, $\eta(x)$ [V]",
    "i_mix_norm": r"Normalized mixed current, $i_{\mathrm{mix}}$ [A/m$^2$]",
    "i_mix_phys": r"Mixed current per unit depth, $i_{\mathrm{mix}}$ [A/m]",
    "i_mix_abs": r"Mixed current, $i_{\mathrm{mix}}$ [A]",
    "i_mix_avg": r"Average mixed current density, $\bar{i}_{\mathrm{mix}}$ [A/m$^2$]",
    "I_net_norm": r"Normalized net current, $I_{\mathrm{net}}$ [A/m$^2$]",
    "I_net_phys": r"Net current per unit depth, $I_{\mathrm{net}}$ [A/m]",
    "I_net_abs": r"Net current, $I_{\mathrm{net}}$ [A]",
    "I_net_avg": r"Average net current density, $\bar{I}_{\mathrm{net}}$ [A/m$^2$]",
    "S_E": r"$S_E$ [-]",
    "S_I_norm": r"$S_{I,\mathrm{norm}}$ [-]",
    "S_I_phys": r"$S_{I,\mathrm{phys}}$ [-]",
    "S_I_abs": r"$S_{I,\mathrm{abs}}$ [-]",
}

CURRENT_AXIS_KEYS = {
    "local_current_density",
    "i_mix_norm",
    "i_mix_phys",
    "i_mix_abs",
    "i_mix_avg",
    "I_net_norm",
    "I_net_phys",
    "I_net_abs",
    "I_net_avg",
}

MOLAR_TO_MOL_PER_M3 = 1000.0
LENGTH_TO_NM = 1e9
CDL_TO_UF_PER_CM2 = 100.0
MIN_PHYSICAL_CDL_UF_PER_CM2 = 1.0
MIN_PHYSICAL_CDL_F_PER_M2 = MIN_PHYSICAL_CDL_UF_PER_CM2 / CDL_TO_UF_PER_CM2


def concentration_M_to_mol_per_m3(value_M: float) -> float:
    return float(value_M) * MOLAR_TO_MOL_PER_M3


def concentration_mol_per_m3_to_M(value_mol_per_m3: float) -> float:
    return float(value_mol_per_m3) / MOLAR_TO_MOL_PER_M3


def _display_param_value(name: str, value: float) -> float:
    if name == "C_tot":
        return concentration_mol_per_m3_to_M(value)
    if name in {"lambda_D", "L_Au", "L_gap", "L_Pd_len"}:
        return float(value) * LENGTH_TO_NM
    if name in {"Cdl_Au", "Cdl_C", "Cdl_Pd"}:
        return float(value) * CDL_TO_UF_PER_CM2
    return float(value)


def _display_param_values(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if name == "C_tot":
        return arr / MOLAR_TO_MOL_PER_M3
    if name in {"lambda_D", "L_Au", "L_gap", "L_Pd_len"}:
        return arr * LENGTH_TO_NM
    if name in {"Cdl_Au", "Cdl_C", "Cdl_Pd"}:
        return arr * CDL_TO_UF_PER_CM2
    return arr


PARAM_AXIS_LABELS = {
    "C_tot": r"Electrolyte concentration, $C_{\mathrm{tot}}$ [M]",
    "lambda_D": r"$\lambda_D$ [nm]",
    "epsilon_r": r"$\varepsilon_r$ [-]",
    "T": "T [K]",
    "pH": "pH [-]",
    "L_Au": r"$L_{\mathrm{Au}}$ [nm]",
    "L_gap": r"$L_{\mathrm{support}}$ [nm]",
    "L_Pd_len": r"$L_{\mathrm{Pd}}$ [nm]",
    "Cdl_Au": r"$C_{\mathrm{dl,Au}}$ [$\mu$F/cm$^2$]",
    "Cdl_C": r"$C_{\mathrm{dl,support}}$ [$\mu$F/cm$^2$]",
    "Cdl_Pd": r"$C_{\mathrm{dl,Pd}}$ [$\mu$F/cm$^2$]",
    "pzc_Au": r"$\mathrm{pzc}_{\mathrm{Au}}$ [V]",
    "pzc_C": r"$\mathrm{pzc}_{\mathrm{support}}$ [V]",
    "pzc_Pd": r"$\mathrm{pzc}_{\mathrm{Pd}}$ [V]",
    "it0_1": r"$i_{0,1}$ [A/m$^2$]",
    "it0_2": r"$i_{0,2}$ [A/m$^2$]",
    "alpha1": r"$\alpha_1$ [-]",
    "alpha2": r"$\alpha_2$ [-]",
    "E1_eq": r"$E_{\mathrm{eq},1}$ [V]",
    "E2_eq": r"$E_{\mathrm{eq},2}$ [V]",
    "z_R1": r"$z_{\mathrm{R},1}$ [-]",
    "z_O2": r"$z_{\mathrm{O},2}$ [-]",
}


def _plot_axis_label(name: str) -> str:
    return PLOT_AXIS_LABELS.get(name, name)


def _param_axis_label(name: str) -> str:
    return PARAM_AXIS_LABELS.get(name, name)


def _delta_pzc_label() -> str:
    return r"$\Delta \mathrm{pzc} = \mathrm{pzc}_{\mathrm{Au}} - \mathrm{pzc}_{\mathrm{Pd}}$ [V]"


def _pzc_gap_label() -> str:
    return r"$\mathrm{pzc}_{\mathrm{support}}$ [V]"


def _format_signed_float_tag(value: float, decimals: int = 3) -> str:
    return f"{value:+.{decimals}f}".replace("+", "p").replace("-", "m").replace(".", "p")


def _split_label_unit(label: str) -> Tuple[str, str]:
    if label.endswith("]") and " [" in label:
        base, unit = label.rsplit(" [", 1)
        return base, unit[:-1]
    return label, ""


def _label_symbol(label: str) -> str:
    base, _ = _split_label_unit(label)
    return base.split(", ")[-1]


def _label_with_power_of_ten(label: str, exponent: int) -> str:
    if exponent == 0:
        return label
    base, unit = _split_label_unit(label)
    if unit:
        return f"{base} [$10^{{{exponent}}}$ {unit}]"
    return f"{label} [$10^{{{exponent}}}$]"


def _scientific_exponent(values: Any, step: int = 3) -> int:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    finite = finite[np.abs(finite) > 0.0]
    if finite.size == 0:
        return 0
    max_abs = float(np.max(np.abs(finite)))
    raw_exp = int(math.floor(math.log10(max_abs)))
    scaled_exp = int(step * math.floor(raw_exp / step))
    return 0 if abs(scaled_exp) < step else scaled_exp


def _scale_by_exponent(values: Any, exponent: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if exponent == 0:
        return arr
    return arr / (10.0 ** exponent)


def _scaled_current_display(axis_key: str, *values: Any) -> Tuple[List[np.ndarray], str, int]:
    if axis_key not in CURRENT_AXIS_KEYS:
        raise ValueError(f"Unsupported current axis key: {axis_key}")
    exponent = _scientific_exponent(np.concatenate([np.ravel(np.asarray(v, dtype=float)) for v in values if np.size(v) > 0]))
    scaled = [_scale_by_exponent(v, exponent) for v in values]
    label = _label_with_power_of_ten(_plot_axis_label(axis_key), exponent)
    return scaled, label, exponent


def _style_colorbar(
    cbar,
    label: str,
    ticks: Optional[np.ndarray | List[float]] = None,
    use_max_locator: bool = True,
    labelsize: float = 7.5,
    label_fontsize: Optional[float] = None,
) -> None:
    cbar.set_label(label)
    if label_fontsize is not None:
        cbar.ax.yaxis.label.set_fontsize(label_fontsize)
    cbar.ax.tick_params(length=2.8, width=0.7, pad=2, labelsize=labelsize)
    cbar.outline.set_linewidth(0.8)
    if ticks is not None:
        cbar.set_ticks(np.asarray(ticks, dtype=float))
    elif use_max_locator:
        cbar.locator = MaxNLocator(nbins=5)
    cbar.update_ticks()


def _style_heatmap_axes(ax: Axes) -> None:
    ax.title.set_fontsize(HEATMAP_TITLE_FONTSIZE)
    ax.title.set_fontweight("semibold")
    ax.xaxis.label.set_fontsize(HEATMAP_AXIS_LABEL_FONTSIZE)
    ax.yaxis.label.set_fontsize(HEATMAP_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=HEATMAP_TICK_LABEL_FONTSIZE, length=4.8, width=1.05, pad=3.2)
    ax.tick_params(axis="both", which="minor", length=3.0, width=0.85)


def _style_heatmap_colorbar(
    cbar,
    label: str,
    ticks: Optional[np.ndarray | List[float]] = None,
    use_max_locator: bool = True,
) -> None:
    _style_colorbar(
        cbar,
        label,
        ticks=ticks,
        use_max_locator=use_max_locator,
        labelsize=HEATMAP_COLORBAR_TICK_FONTSIZE,
        label_fontsize=HEATMAP_COLORBAR_LABEL_FONTSIZE,
    )


def _ofat_x_values(dfp: pd.DataFrame, pname: str) -> np.ndarray:
    if pname == "C_tot" and "value_M" in dfp.columns:
        return np.asarray(dfp["value_M"], dtype=float)
    return _display_param_values(pname, np.asarray(dfp["value"], dtype=float))


def _expand_mode_selection(mode_setting: str, allow_both: bool = True) -> List[str]:
    mode_upper = str(mode_setting).upper()
    if allow_both and mode_upper == "BOTH":
        return ["MEAN", "FULL"]
    if mode_upper in {"MEAN", "FULL"}:
        return [mode_upper]
    raise ValueError(f"Unsupported mode setting: {mode_setting}")


# -----------------------------
# Default parameters (edit here)
# -----------------------------

def default_params() -> Dict[str, Any]:
    """
    Parameter dictionary `params` (requested by the user).

    Units:
    - Lengths: meters
    - Potentials: volts (E_mix, E_eq, pzc must share the same reference)
    - C_tot: mol/m^3 of each ionic species for a symmetric 1:1 electrolyte
    - Capacitance Cdl: F/m^2 (or set dimensionless g_* directly)
    - Exchange current densities it0_*: A/m^2
    - out_of_plane_width: meters; default 1 m (unit width) for converting A/m to A

    Dimensionless (PDF definitions):
    - g_i = (λ_D/ε_s) C_dl,i (Eq. (S-11c))
    - x̃ = x/λ_D, L̃ = L/λ_D (Eq. (S1-3))
    - φ̃ = (F/RT)(φ - φ_b) (Eq. (S1-1))
    """
    return dict(
        # constants
        R=8.314,
        F=96485.0,
        T=298.0,

        # permittivity
        epsilon0=8.8541878128e-12,  # F/m (TODO check PDF table if needed)
        epsilon_r=78.5,
        epsilon_s=None,             # override if desired

        # electrolyte
        C_tot=10.0,                 # mol/m^3 of each ion; 10.0 corresponds to 10 mM
        lambda_D=None,              # optional override; set to None to auto-recompute

        # geometry
        L_Au=11e-9,                 # m
        L_gap=10e-9,                # m
        L_Pd_len=37e-9,             # m
        out_of_plane_width=1.0,     # m, unit out-of-plane width by default

        # interfacial electrostatics (Cdl or g)
        Cdl_Au=0.23,                # F/m^2 = 23 uF/cm^2; displayed as uF/cm^2 in figures/tables
        Cdl_C=0.1,                  # F/m^2 = 10 uF/cm^2; displayed as uF/cm^2 in figures/tables
        Cdl_Pd=0.377,               # F/m^2 = 37.7 uF/cm^2; displayed as uF/cm^2 in figures/tables
        g_Au=None, g_C=None, g_Pd=None,  # if set, overrides Cdl_* via Eq. (S-11c)

        # pzc (V)
        pzc_Au=0.513,
        pzc_C=0.361,
        pzc_Pd=0.371,

        # kinetics
        it0_1=8.85e-5,              # A/m^2
        it0_2=3.878e-4,             # A/m^2
        alpha1=0.5,
        alpha2=0.37,
        z_R1=-1.0,
        z_O2=1.0,
        E1_eq=0.1,                  # V
        E2_eq=0.834,                # V
        pH=7.0,                     # baseline pH
        pH_ref=7.0,                 # reference pH at which E*_eq and it0_* are defined
        E1_eq_pH_slope_V_per_pH=0.0, # E1_eq(pH) = E1_eq + slope * (pH - pH_ref)
        E2_eq_pH_slope_V_per_pH=-(math.log(10.0) * 8.314 * 298.0 / 96485.0), # reaction 2 treated as H+ + e- -> 1/2 H2
        it0_1_pH_order=0.0,         # it0_1(pH) = it0_1 * 10^(-order * (pH - pH_ref))
        it0_2_pH_order=1.0,         # reaction 2 treated as first-order in H+

        # numerics
        N_modes=80,
        Nx=1200,
        xtol=1e-10,
        max_bracket_expands=12,

        # switches
        use_edl=True,                       # True=with EDL, False=no EDL
        use_affine_phi2=True,              # Eq. (D5-3)
        use_closed_form_when_affine=True,  # Eq. (D5-6)/(D5-8)
        do_self_checks=False,              # run optional self-checks (slow)
        do_convergence_check=False,        # run optional grid/mode convergence check (slow)
        dh_warn_threshold=1.0,             # warn/guard when max |phi_tilde| exceeds this
        dh_violation_action="warn",        # "ignore", "warn", or "raise"

        # what to run
        do_ofat=True,
        do_heatmaps=True,
        do_sensitivities=False,

        # scan sizes
        ofat_n=15,
        ofat_L_gap_n=15,
        heatmap_nx=25,
        heatmap_ny=25,
        scan_mode="BOTH",  # "MEAN" or "BOTH"
        heatmap_mode="BOTH",       # "MEAN", "FULL", or "BOTH"
        sensitivity_mode="BOTH",   # "MEAN", "FULL", or "BOTH"
        ofat_pH_min=0.0,
        ofat_pH_max=14.0,
        ofat_C_tot_min=concentration_M_to_mol_per_m3(1.0e-4),    # 0.1 mM
        ofat_C_tot_max=concentration_M_to_mol_per_m3(10.0),      # 10 M
        heatmap_C_tot_min=concentration_M_to_mol_per_m3(1.0e-4), # 0.1 mM
        heatmap_C_tot_max=concentration_M_to_mol_per_m3(10.0),   # 10 M
        heatmap_L_min=2e-9,        # m, 2 nm lower bound for Au/Pd length heatmaps
        heatmap_L_max=1000e-9,     # m, 1000 nm upper bound for Au/Pd length heatmaps
        heatmap_Cdl_C_min=0.01,      # F/m^2 = 1 uF/cm^2
        heatmap_Cdl_C_max=10.0,      # F/m^2
        heatmap_pzc_C_offsets=[-0.10, 0.0, 0.10],  # V offsets around baseline pzc_C
        ofat_L_gap_min=0.0,        # m, allow L_gap=0 in OFAT
        ofat_L_gap_max=1000e-9,    # m, default OFAT range for nanoscale/sub-micron gaps
    )


def validate_params(params: Dict[str, Any]) -> None:
    """Validate parameter completeness and basic physical ranges."""
    required_keys = set(default_params())
    missing = sorted(required_keys - set(params))
    if missing:
        raise KeyError(f"Missing required parameter(s): {', '.join(missing)}")

    def require_finite(name: str) -> float:
        val = float(params[name])
        if not math.isfinite(val):
            raise ValueError(f"{name} must be finite")
        return val

    for name in ("R", "F", "T", "epsilon0", "epsilon_r", "L_Au", "L_gap", "L_Pd_len", "it0_1", "it0_2", "xtol", "pH", "pH_ref"):
        require_finite(name)

    if require_finite("R") <= 0 or require_finite("F") <= 0 or require_finite("T") <= 0:
        raise ValueError("R, F, and T must be positive")
    if require_finite("epsilon0") <= 0 or require_finite("epsilon_r") <= 0:
        raise ValueError("epsilon0 and epsilon_r must be positive")
    if params.get("epsilon_s") is not None and float(params["epsilon_s"]) <= 0:
        raise ValueError("epsilon_s must be positive when provided")

    if params.get("lambda_D") is None:
        if require_finite("C_tot") <= 0:
            raise ValueError("C_tot must be positive when lambda_D is auto-calculated")
    elif float(params["lambda_D"]) <= 0:
        raise ValueError("lambda_D must be positive when provided")

    for name in ("it0_1", "it0_2"):
        if require_finite(name) <= 0:
            raise ValueError(f"{name} must be positive")

    if require_finite("L_Au") <= 0 or require_finite("L_gap") < 0 or require_finite("L_Pd_len") <= 0:
        raise ValueError("Geometry lengths must satisfy L_Au>0, L_gap>=0, L_Pd_len>0")

    for name in ("Cdl_Au", "Cdl_C", "Cdl_Pd"):
        if require_finite(name) < 0:
            raise ValueError(f"{name} must be non-negative")
    for name in ("g_Au", "g_C", "g_Pd"):
        if params.get(name) is not None and float(params[name]) < 0:
            raise ValueError(f"{name} must be non-negative when provided")

    for name in ("alpha1", "alpha2"):
        alpha = require_finite(name)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"{name} must be within [0, 1]")

    for name in (
        "pzc_Au", "pzc_C", "pzc_Pd", "z_R1", "z_O2", "E1_eq", "E2_eq",
        "E1_eq_pH_slope_V_per_pH", "E2_eq_pH_slope_V_per_pH", "it0_1_pH_order", "it0_2_pH_order",
    ):
        require_finite(name)
    if require_finite("out_of_plane_width") <= 0:
        raise ValueError("out_of_plane_width must be positive")

    if int(params["N_modes"]) < 1:
        raise ValueError("N_modes must be >= 1")
    if int(params["Nx"]) < 2:
        raise ValueError("Nx must be >= 2")
    if int(params.get("ofat_n", 1)) < 2:
        raise ValueError("ofat_n must be >= 2")
    if int(params.get("ofat_L_gap_n", 1)) < 2:
        raise ValueError("ofat_L_gap_n must be >= 2")
    if float(params["xtol"]) <= 0:
        raise ValueError("xtol must be positive")
    if int(params["max_bracket_expands"]) < 0:
        raise ValueError("max_bracket_expands must be >= 0")
    if float(params.get("dh_warn_threshold", 1.0)) <= 0:
        raise ValueError("dh_warn_threshold must be positive")
    if require_finite("heatmap_L_min") <= 0 or require_finite("heatmap_L_max") <= 0:
        raise ValueError("heatmap_L_min and heatmap_L_max must be positive")
    if float(params["heatmap_L_max"]) <= float(params["heatmap_L_min"]):
        raise ValueError("heatmap_L_max must be larger than heatmap_L_min")
    if str(params.get("dh_violation_action", "warn")).lower() not in {"ignore", "warn", "raise"}:
        raise ValueError("dh_violation_action must be one of: ignore, warn, raise")
    for name in ("ofat_pH_min", "ofat_pH_max"):
        require_finite(name)
    if float(params.get("ofat_pH_max", 14.0)) < float(params.get("ofat_pH_min", 0.0)):
        raise ValueError("ofat_pH_max must be >= ofat_pH_min")
    for name in (
        "ofat_C_tot_min", "ofat_C_tot_max", "heatmap_C_tot_min", "heatmap_C_tot_max",
        "heatmap_Cdl_C_min", "heatmap_Cdl_C_max",
    ):
        if require_finite(name) <= 0:
            raise ValueError(f"{name} must be positive")
    if float(params.get("ofat_C_tot_max", 1.0)) < float(params.get("ofat_C_tot_min", 1.0)):
        raise ValueError("ofat_C_tot_max must be >= ofat_C_tot_min")
    if float(params.get("heatmap_C_tot_max", 1.0)) < float(params.get("heatmap_C_tot_min", 1.0)):
        raise ValueError("heatmap_C_tot_max must be >= heatmap_C_tot_min")
    if float(params.get("heatmap_Cdl_C_max", 1.0)) < float(params.get("heatmap_Cdl_C_min", 1.0)):
        raise ValueError("heatmap_Cdl_C_max must be >= heatmap_Cdl_C_min")
    pzc_offsets = params.get("heatmap_pzc_C_offsets", [-0.10, 0.0, 0.10])
    if not isinstance(pzc_offsets, list) or len(pzc_offsets) == 0:
        raise ValueError("heatmap_pzc_C_offsets must be a non-empty list")
    if not all(np.isfinite(float(v)) for v in pzc_offsets):
        raise ValueError("heatmap_pzc_C_offsets must contain only finite values")
    if str(params.get("scan_mode", "BOTH")).upper() not in {"MEAN", "FULL", "BOTH"}:
        raise ValueError("scan_mode must be MEAN, FULL, or BOTH")
    for mode_key in ("heatmap_mode", "sensitivity_mode"):
        if str(params.get(mode_key, "BOTH")).upper() not in {"MEAN", "FULL", "BOTH"}:
            raise ValueError(f"{mode_key} must be MEAN, FULL, or BOTH")


# -----------------------------
# EDL model: cosine expansion + matrix solve
# -----------------------------

def compute_derived_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Derived quantities used by both EDL and no-EDL paths."""
    validate_params(params)
    p = params
    R_gas = float(p["R"]); F = float(p["F"]); T = float(p["T"])
    beta = F / (R_gas * T)

    # epsilon_s
    if p.get("epsilon_s") is not None:
        eps_s = float(p["epsilon_s"])
    else:
        eps_s = float(p["epsilon_r"]) * float(p["epsilon0"])

    # lambda_D (Eq. (S1-2))
    # Here C_tot follows the SI convention c_+ = c_- = C_tot for a symmetric 1:1 electrolyte.
    if p.get("lambda_D") is not None:
        lambda_D = float(p["lambda_D"])
    else:
        C_tot = float(p["C_tot"])
        lambda_D = math.sqrt(eps_s * R_gas * T / (2.0 * F**2 * C_tot))

    # geometry (m)
    L_Au = float(p["L_Au"]); L_gap = float(p["L_gap"]); L_Pd_len = float(p["L_Pd_len"])
    if not (L_Au > 0 and L_gap >= 0 and L_Pd_len > 0):
        raise ValueError("Geometry lengths must be positive (L_gap can be zero)")
    L_C = L_Au + L_gap
    L_total = L_C + L_Pd_len

    # dimensionless lengths (Eq. (S1-3))
    L_tilde = L_total / lambda_D
    L_Au_tilde = L_Au / lambda_D
    L_C_tilde = L_C / lambda_D

    # g_i = (lambda_D/epsilon_s) Cdl_i (Eq. (S-11c)), unless overridden
    def g_from_Cdl(Cdl: float) -> float:
        return (lambda_D / eps_s) * Cdl

    g_Au = p.get("g_Au"); g_C = p.get("g_C"); g_Pd = p.get("g_Pd")
    if g_Au is None: g_Au = g_from_Cdl(float(p["Cdl_Au"]))
    if g_C is None:  g_C  = g_from_Cdl(float(p["Cdl_C"]))
    if g_Pd is None: g_Pd = g_from_Cdl(float(p["Cdl_Pd"]))

    # pzc (V) -> phi_pzc_tilde (Eq. (S1-1), phi_b=0)
    pzc_Au = float(p["pzc_Au"]); pzc_C = float(p["pzc_C"]); pzc_Pd = float(p["pzc_Pd"])
    pzc_Au_tilde = beta * pzc_Au
    pzc_C_tilde = beta * pzc_C
    pzc_Pd_tilde = beta * pzc_Pd

    return dict(
        R=R_gas, F=F, T=T, beta=beta,
        epsilon_s=eps_s, lambda_D=lambda_D,
        L_Au=L_Au, L_gap=L_gap, L_Pd_len=L_Pd_len,
        L_C=L_C, L_total=L_total,
        L_tilde=L_tilde, L_Au_tilde=L_Au_tilde, L_C_tilde=L_C_tilde,
        g_Au=g_Au, g_C=g_C, g_Pd=g_Pd,
        pzc_Au=pzc_Au, pzc_C=pzc_C, pzc_Pd=pzc_Pd,
        pzc_Au_tilde=pzc_Au_tilde, pzc_C_tilde=pzc_C_tilde, pzc_Pd_tilde=pzc_Pd_tilde,
    )

class EDLModel:
    """
    Linear EDL model (Debye–Hückel) with lateral heterogeneity.

    Governing PDE:
        ∂²φ̃_s/∂x̃² + ∂²φ̃_s/∂ỹ² = φ̃_s                      (Eq. (S1-4))

    BCs:
        ∂φ̃_s/∂x̃|_{x̃=0} = 0,  ∂φ̃_s/∂x̃|_{x̃=L̃} = 0          (Eq. (S1-5))
        φ̃_s(x̃,ỹ→∞) = 0                                      (Eq. (S1-6))
        ∂φ̃_s/∂ỹ|_{ỹ=0} = -g_i(φ̃_M - φ̃_s(x̃,0) - φ̃_pzc,i)   (Eq. (S-12a))
        with piecewise constants g_i, φ̃_pzc,i on Au/C/Pd       (Eq. (S-12b))

    Cosine expansion:
        φ̃_s(x̃,ỹ) = Σ A_n cos(ρ_n x̃) exp(-γ_n ỹ)            (Eq. (S1-13))
        ρ_n = nπ/L̃, γ_n = √(1+ρ_n²)                           (Eq. (S1-14))
        φ̃_s(x̃,0) = Σ A_n cos(ρ_n x̃)                          (Eq. (S1-15))

    Coefficients from matrix system:
        [γ + S]A = R                                           (Eq. (S1-24))
        with explicit S and R elements (Eqs. (S1-28)–(S1-32)).

    Affine reuse:
        A(φ̃_M) = A_M φ̃_M - A_pzc                              (Eq. (D3-1)),
        enabling fast updates during outer mixed-potential iterations.
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = copy.deepcopy(params)
        validate_params(self.params)
        self.derived: Dict[str, Any] = {}
        self.pre: Dict[str, Any] = {}
        self._build()

    def _compute_derived(self) -> None:
        self.derived = compute_derived_params(self.params)

    def _build(self) -> None:
        self._compute_derived()
        p = self.params; d = self.derived

        N = int(p["N_modes"])
        Nx = int(p["Nx"])

        L = d["L_tilde"]
        L_Au = d["L_Au_tilde"]
        L_C = d["L_C_tilde"]

        n = np.arange(N + 1, dtype=float)
        rho = n * math.pi / L                 # Eq. (S1-14)
        gamma = np.sqrt(1.0 + rho**2)         # Eq. (S1-14)

        segs = [
            ("Au", 0.0,  L_Au, d["g_Au"], d["pzc_Au_tilde"]),
            ("C",  L_Au, L_C,  d["g_C"],  d["pzc_C_tilde"]),
            ("Pd", L_C,  L,    d["g_Pd"], d["pzc_Pd_tilde"]),
        ]

        # Build S matrix (Eqs. (S1-28)–(S1-31))
        S = np.zeros((N + 1, N + 1), dtype=float)

        for nn in range(N + 1):
            acc = 0.0
            for _, a, b, gseg, _ in segs:
                acc += gseg * J_n(nn, a, b, float(rho[nn]))
            S[0, nn] = acc / L  # Eq. (S1-28)

        for mm in range(1, N + 1):
            acc0 = 0.0
            for _, a, b, gseg, _ in segs:
                acc0 += gseg * J_n(mm, a, b, float(rho[mm]))
            S[mm, 0] = 2.0 * acc0 / L  # Eq. (S1-30)

            for nn in range(1, N + 1):
                acc = 0.0
                for _, a, b, gseg, _ in segs:
                    acc += gseg * I_mn(mm, nn, a, b, float(rho[mm]), float(rho[nn]))
                S[mm, nn] = 2.0 * acc / L  # Eq. (S1-31)

        # Matrix system: [γ + S]A = R (Eq. (S1-24))
        M = np.diag(gamma) + S
        lu, piv = la.lu_factor(M)

        # R = rM φ̃_M - r_pzc, using R0 (S1-29) and Rm (S1-32); see also Eq. (D1-4)–(D1-5)
        rM = np.zeros(N + 1, dtype=float)
        r_pzc = np.zeros(N + 1, dtype=float)

        accM0 = 0.0; accp0 = 0.0
        for _, a, b, gseg, pzc_t in segs:
            dx = b - a
            accM0 += gseg * dx
            accp0 += gseg * pzc_t * dx
        rM[0] = accM0 / L
        r_pzc[0] = accp0 / L

        for mm in range(1, N + 1):
            accM = 0.0; accp = 0.0
            for _, a, b, gseg, pzc_t in segs:
                Jm = J_n(mm, a, b, float(rho[mm]))
                accM += gseg * Jm
                accp += gseg * pzc_t * Jm
            rM[mm] = 2.0 * accM / L
            r_pzc[mm] = 2.0 * accp / L

        # Affine decomposition A = A_M φ̃_M - A_pzc (Eq. (D3-1))
        A_M = la.lu_solve((lu, piv), rM)
        A_pzc = la.lu_solve((lu, piv), r_pzc)

        # Segment-average weights (Eqs. (S4-6) and (S4-9))
        c_Au = np.zeros(N + 1, dtype=float)
        c_Pd = np.zeros(N + 1, dtype=float)
        c_Au[0] = 1.0; c_Pd[0] = 1.0
        L_Pd = L - L_C
        for nn in range(1, N + 1):
            c_Au[nn] = math.sin(float(rho[nn]) * L_Au) / (L_Au * float(rho[nn]))
            c_Pd[nn] = -math.sin(float(rho[nn]) * L_C) / (L_Pd * float(rho[nn]))

        # Affine φ2,i = a_i E + b_i (Eq. (D5-3))
        a1 = float(np.dot(c_Au, A_M))
        a2 = float(np.dot(c_Pd, A_M))
        b1 = -(d["R"] * d["T"] / d["F"]) * float(np.dot(c_Au, A_pzc))
        b2 = -(d["R"] * d["T"] / d["F"]) * float(np.dot(c_Pd, A_pzc))

        # Precompute surface basis φ̃_M(x̃) and φ̃_pzc(x̃) for FULL mode (Eq. (S1-15) + affine (D3-1))
        x = _surface_grid_with_boundaries(L, L_Au, L_C, Nx)
        cos_mat = np.cos(np.outer(x, rho))
        phi_tilde_M = cos_mat @ A_M
        phi_tilde_pzc = cos_mat @ A_pzc

        self.pre = dict(
            N=N, Nx=Nx,
            rho=rho, gamma=gamma,
            segs=segs,
            S=S, M=M,
            rM=rM, r_pzc=r_pzc,
            A_M=A_M, A_pzc=A_pzc,
            c_Au=c_Au, c_Pd=c_Pd,
            a1=a1, a2=a2, b1=b1, b2=b2,
            x_tilde=x, phi_tilde_M=phi_tilde_M, phi_tilde_pzc=phi_tilde_pzc,
        )

    def phi_tilde_surface(self, E_mix: float) -> Tuple[np.ndarray, np.ndarray]:
        """φ̃_s(x̃,0) using affine decomposition (Eq. (S1-15) + Eq. (D3-1))."""
        beta = self.derived["beta"]
        phiM_tilde = beta * E_mix
        x = self.pre["x_tilde"]
        phi = self.pre["phi_tilde_M"] * phiM_tilde - self.pre["phi_tilde_pzc"]
        return x, phi

    def segment_mean_phi2(self, E_mix: float, use_affine_phi2: bool) -> Tuple[float, float]:
        """Segment-mean φ2 (V). Either affine (D5-3) or direct (S3-16)."""
        if use_affine_phi2:
            return self.pre["a1"] * E_mix + self.pre["b1"], self.pre["a2"] * E_mix + self.pre["b2"]
        Au_t, Pd_t = self.segment_mean_phi_tilde(E_mix)
        scale = self.derived["R"] * self.derived["T"] / self.derived["F"]
        return scale * Au_t, scale * Pd_t

    def segment_mean_phi_tilde(self, E_mix: float) -> Tuple[float, float]:
        """Segment-mean φ̃ (Eq. (S4-5))."""
        beta = self.derived["beta"]
        phiM_tilde = beta * E_mix
        A = self.pre["A_M"] * phiM_tilde - self.pre["A_pzc"]
        return float(np.dot(self.pre["c_Au"], A)), float(np.dot(self.pre["c_Pd"], A))


# -----------------------------
# Kinetics and mixed potential
# -----------------------------

def _effective_reaction_params(params: Dict[str, Any]) -> Dict[str, float]:
    pH = float(params.get("pH", params.get("pH_ref", 7.0)))
    pH_ref = float(params.get("pH_ref", pH))
    delta_pH = pH - pH_ref

    E1_eq_base = float(params["E1_eq"])
    E2_eq_base = float(params["E2_eq"])
    E1_slope = float(params.get("E1_eq_pH_slope_V_per_pH", 0.0))
    E2_slope = float(params.get("E2_eq_pH_slope_V_per_pH", 0.0))
    order1 = float(params.get("it0_1_pH_order", 0.0))
    order2 = float(params.get("it0_2_pH_order", 0.0))

    E1_eq_eff = E1_eq_base + E1_slope * delta_pH
    E2_eq_eff = E2_eq_base + E2_slope * delta_pH
    it0_1_eff = float(params["it0_1"]) * safe_exp(-math.log(10.0) * order1 * delta_pH)
    it0_2_eff = float(params["it0_2"]) * safe_exp(-math.log(10.0) * order2 * delta_pH)

    return dict(
        pH=pH,
        pH_ref=pH_ref,
        delta_pH=delta_pH,
        E1_eq_eff=float(E1_eq_eff),
        E2_eq_eff=float(E2_eq_eff),
        it0_1_eff=float(it0_1_eff),
        it0_2_eff=float(it0_2_eff),
        E1_eq_pH_slope_V_per_pH=E1_slope,
        E2_eq_pH_slope_V_per_pH=E2_slope,
        it0_1_pH_order=order1,
        it0_2_pH_order=order2,
    )


def _kinetics_context(E: float, params: Dict[str, Any]) -> Dict[str, float]:
    R_gas = float(params["R"]); F = float(params["F"]); T = float(params["T"])
    beta = F / (R_gas * T)
    alpha1 = float(params["alpha1"]); alpha2 = float(params["alpha2"])
    z_R1 = float(params["z_R1"]); z_O2 = float(params["z_O2"])
    rxn = _effective_reaction_params(params)
    eta1 = E - rxn["E1_eq_eff"]
    eta2 = E - rxn["E2_eq_eff"]
    return dict(
        beta=beta,
        it0_1=rxn["it0_1_eff"],
        it0_2=rxn["it0_2_eff"],
        alpha1=alpha1,
        alpha2=alpha2,
        eta1=eta1,
        eta2=eta2,
        E1_eq_eff=rxn["E1_eq_eff"],
        E2_eq_eff=rxn["E2_eq_eff"],
        pH=rxn["pH"],
        pH_ref=rxn["pH_ref"],
        Gamma1=(1.0 - alpha1) + z_R1,
        Gamma2=alpha2 - z_O2,
    )


def _segment_masks(x: np.ndarray, L_Au: float, L_C: float, L_total: float) -> Tuple[np.ndarray, np.ndarray]:
    mask_Au = (x >= 0.0) & (x <= L_Au + 1e-12)
    mask_Pd = (x >= L_C - 1e-12) & (x <= L_total + 1e-12)
    return mask_Au, mask_Pd


def _build_run_output(
    mode: str,
    E_mix: float,
    i_mix: float,
    info: Dict[str, Any],
    derived: Dict[str, Any],
    a1: float,
    b1: float,
    a2: float,
    b2: float,
) -> Dict[str, Any]:
    return dict(
        mode=mode,
        E_mix=float(E_mix),
        i_mix=float(i_mix),
        i_mix_norm_A_per_m2=float(i_mix),
        i_mix_phys_A_per_m=float(derived["lambda_D"]) * float(i_mix),
        residual=float(info.get("residual_at_root", float("nan"))),
        residual_norm_A_per_m2=float(info.get("residual_at_root", float("nan"))),
        residual_phys_A_per_m=float(derived["lambda_D"]) * float(info.get("residual_at_root", float("nan"))),
        converged=bool(info.get("converged", False)),
        method=str(info.get("method", "")),
        iterations=int(info.get("iterations", -1)),
        a1=float(a1),
        b1=float(b1),
        a2=float(a2),
        b2=float(b2),
        lambda_D=float(derived["lambda_D"]),
        reactive_length_m=float(derived["L_Au"] + derived["L_Pd_len"]),
        g_Au=float(derived["g_Au"]),
        g_C=float(derived["g_C"]),
        g_Pd=float(derived["g_Pd"]),
        L_tilde=float(derived["L_tilde"]),
        L_Au_tilde=float(derived["L_Au_tilde"]),
        L_C_tilde=float(derived["L_C_tilde"]),
    )


def _attach_current_unit_outputs(out: Dict[str, Any], lambda_D: float, out_of_plane_width: float, reactive_length_m: float) -> None:
    out["out_of_plane_width_m"] = float(out_of_plane_width)
    out["reactive_length_m"] = float(reactive_length_m)
    reactive_area_m2 = float(reactive_length_m) * float(out_of_plane_width)
    out["reactive_area_m2"] = reactive_area_m2
    for base_key in ("I_Au", "I_Pd", "residual", "i_mix"):
        if base_key not in out:
            continue
        val = float(out[base_key])
        out[f"{base_key}_norm_A_per_m2"] = val
        phys = lambda_D * val
        out[f"{base_key}_phys_A_per_m"] = phys
        out[f"{base_key}_abs_A"] = out_of_plane_width * phys
        out[f"{base_key}_avg_A_per_m2"] = (out[f"{base_key}_abs_A"] / reactive_area_m2) if reactive_area_m2 > 0.0 else float("nan")


def _compute_dh_status_from_phi(phi_tilde: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    max_abs_phi_tilde = float(np.max(np.abs(phi_tilde))) if phi_tilde.size else 0.0
    threshold = float(params.get("dh_warn_threshold", 1.0))
    ok = bool(max_abs_phi_tilde < threshold)
    return dict(
        max_abs_phi_tilde=max_abs_phi_tilde,
        dh_warn_threshold=threshold,
        debye_huckel_ok=ok,
    )


def _handle_dh_violation(run_info: Dict[str, Any], params: Dict[str, Any], mode: str, use_edl: bool) -> None:
    global _DH_WARNED
    if not use_edl:
        return
    if bool(run_info.get("debye_huckel_ok", True)):
        return
    action = str(params.get("dh_violation_action", "warn")).lower()
    msg = (
        f"Debye-Huckel validity warning in run_case(mode={mode}, use_edl={use_edl}): "
        f"max |phi_tilde| = {run_info['max_abs_phi_tilde']:.6g} exceeds "
        f"dh_warn_threshold = {run_info['dh_warn_threshold']:.6g}."
    )
    if action == "warn":
        if not _DH_WARNED:
            print(f"WARNING: {msg}")
            _DH_WARNED = True
    elif action == "raise":
        raise ValueError(msg)


def _solve_root_problem(
    f: Callable[[float], float],
    E1_eq: float,
    E2_eq: float,
    xtol: float,
    max_bracket_expands: int,
    E_guess: Optional[float] = None,
    bracket: Optional[Tuple[float, float]] = None,
    info: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
    if E_guess is None:
        E_guess = 0.5 * (E1_eq + E2_eq)
    if bracket is None:
        bracket = (min(E1_eq, E2_eq) - 0.5, max(E1_eq, E2_eq) + 0.5)

    a, b = float(bracket[0]), float(bracket[1])
    fa, fb = f(a), f(b)
    expands = 0
    while np.sign(fa) == np.sign(fb) and expands < max_bracket_expands:
        mid = 0.5 * (a + b)
        span = b - a
        a = mid - 1.5 * span
        b = mid + 1.5 * span
        fa, fb = f(a), f(b)
        expands += 1

    out_info = {} if info is None else dict(info)
    out_info.update(bracket_a=a, bracket_b=b, f_a=float(fa), f_b=float(fb), expands=expands)

    try:
        if np.isfinite(fa) and np.isfinite(fb) and np.sign(fa) != np.sign(fb):
            sol = root_scalar(f, bracket=(a, b), method="brentq", xtol=xtol)
            method = "brentq"
        else:
            sol = root_scalar(f, x0=E_guess, x1=E_guess + 0.05, method="secant", xtol=xtol, maxiter=200)
            method = "secant"
        out_info.update(method=method, converged=bool(sol.converged), iterations=int(sol.iterations))
        if not sol.converged:
            return float("nan"), out_info
        return float(sol.root), out_info
    except Exception as exc:
        out_info.update(converged=False, error=repr(exc))
        return float("nan"), out_info

def currents_mean_field(E: float, phi2_1: float, phi2_2: float, params: Dict[str, Any]) -> Tuple[float, float]:
    """Mean-field irreversible Frumkin-BV currents, Eqs. (S3-4) & (S3-5)."""
    ctx = _kinetics_context(E, params)
    log_i1 = math.log(ctx["it0_1"]) + (1.0 - ctx["alpha1"]) * ctx["beta"] * ctx["eta1"] - ctx["Gamma1"] * ctx["beta"] * phi2_1
    log_i2 = math.log(ctx["it0_2"]) - ctx["alpha2"] * ctx["beta"] * ctx["eta2"] + ctx["Gamma2"] * ctx["beta"] * phi2_2

    i1 = float(safe_exp(log_i1))
    i2 = -float(safe_exp(log_i2))
    return i1, i2


def full_mode_currents(E: float, edl: EDLModel, params: Dict[str, Any], return_profiles: bool) -> Dict[str, Any]:
    """
    FULL mode:
    - local i1(x̃), i2(x̃): Eqs. (S3-8),(S3-9)
    - K integrals: Eqs. (S3-11),(S3-12)
    - net current: Eq. (S3-7b)
    """
    ctx = _kinetics_context(E, params)

    x, phi_tilde = edl.phi_tilde_surface(E)

    L_Au = edl.derived["L_Au_tilde"]
    L_C = edl.derived["L_C_tilde"]
    L = edl.derived["L_tilde"]

    mask_Au, mask_Pd = _segment_masks(x, L_Au, L_C, L)

    K_Au = trapz_compat(safe_exp(-ctx["Gamma1"] * phi_tilde[mask_Au]), x[mask_Au])  # Eq. (S3-12)
    K_Pd = trapz_compat(safe_exp(ctx["Gamma2"] * phi_tilde[mask_Pd]), x[mask_Pd])   # Eq. (S3-11)

    # These are integrals over d x_tilde rather than over physical dx.
    # Because x_tilde is dimensionless, I_Au/I_Pd/i_mix retain A/m^2 units.
    # Multiply by lambda_D (m) to obtain current per unit depth in A/m.
    pref1 = ctx["it0_1"] * safe_exp((1.0 - ctx["alpha1"]) * ctx["beta"] * ctx["eta1"])
    pref2 = -ctx["it0_2"] * safe_exp(-ctx["alpha2"] * ctx["beta"] * ctx["eta2"])
    I_Au = float(pref1 * K_Au)
    I_Pd = float(pref2 * K_Pd)

    residual = I_Au + I_Pd  # Eq. (S3-7b)
    i_mix = abs(I_Au)       # definition used here: |int_Au i1 d x_tilde|

    out: Dict[str, Any] = dict(I_Au=I_Au, I_Pd=I_Pd, residual=residual, i_mix=i_mix, K_Au=K_Au, K_Pd=K_Pd)

    if return_profiles:
        i1 = np.zeros_like(x)
        i2 = np.zeros_like(x)
        i1[mask_Au] = pref1 * safe_exp(-ctx["Gamma1"] * phi_tilde[mask_Au])
        i2[mask_Pd] = pref2 * safe_exp(ctx["Gamma2"] * phi_tilde[mask_Pd])
        out.update(dict(x_tilde=x, phi_tilde=phi_tilde, i1=i1, i2=i2, mask_Au=mask_Au, mask_Pd=mask_Pd))
    return out


def full_mode_currents_no_edl(
    E: float,
    derived: Dict[str, Any],
    params: Dict[str, Any],
    return_profiles: bool,
) -> Dict[str, Any]:
    """
    FULL mode with EDL disabled:
    - phi_tilde(x) = 0 everywhere
    - K_Au = L_Au_tilde, K_Pd = L_Pd_tilde
    - local i1/i2 are uniform within segments
    """
    ctx = _kinetics_context(E, params)

    L_Au = float(derived["L_Au_tilde"])
    L_C = float(derived["L_C_tilde"])
    L = float(derived["L_tilde"])
    L_Pd = L - L_C

    K_Au = L_Au
    K_Pd = L_Pd

    pref1 = ctx["it0_1"] * safe_exp((1.0 - ctx["alpha1"]) * ctx["beta"] * ctx["eta1"])
    pref2 = -ctx["it0_2"] * safe_exp(-ctx["alpha2"] * ctx["beta"] * ctx["eta2"])
    I_Au = float(pref1 * K_Au)
    I_Pd = float(pref2 * K_Pd)

    residual = I_Au + I_Pd
    i_mix = abs(I_Au)

    out: Dict[str, Any] = dict(I_Au=I_Au, I_Pd=I_Pd, residual=residual, i_mix=i_mix, K_Au=K_Au, K_Pd=K_Pd)

    if return_profiles:
        Nx = int(params["Nx"])
        x = _surface_grid_with_boundaries(L, L_Au, L_C, Nx)
        phi_tilde = np.zeros_like(x)
        mask_Au, mask_Pd = _segment_masks(x, L_Au, L_C, L)

        i1 = np.zeros_like(x)
        i2 = np.zeros_like(x)
        i1[mask_Au] = pref1
        i2[mask_Pd] = pref2
        out.update(dict(x_tilde=x, phi_tilde=phi_tilde, i1=i1, i2=i2, mask_Au=mask_Au, mask_Pd=mask_Pd))
    return out


def mean_mode_residual_no_edl(E: float, derived: Dict[str, Any], params: Dict[str, Any]) -> float:
    """Mean-field residual with EDL disabled (phi2_1=phi2_2=0)."""
    i1, i2 = currents_mean_field(E, 0.0, 0.0, params)
    L_Au = float(derived["L_Au_tilde"])
    L_Pd = float(derived["L_tilde"] - derived["L_C_tilde"])
    return float(L_Au * i1 + L_Pd * i2)


def emix_closed_form_no_edl(derived: Dict[str, Any], params: Dict[str, Any]) -> float:
    """Closed-form Emix for no-EDL mean-field model (phi2_1=phi2_2=0)."""
    R_gas = float(params["R"]); F = float(params["F"]); T = float(params["T"])
    rxn = _effective_reaction_params(params)
    it0_1 = rxn["it0_1_eff"]; it0_2 = rxn["it0_2_eff"]
    alpha1 = float(params["alpha1"]); alpha2 = float(params["alpha2"])
    E1_eq = rxn["E1_eq_eff"]; E2_eq = rxn["E2_eq_eff"]

    kappa = 1.0 - alpha1 + alpha2
    L_Au = float(derived["L_Au_tilde"])
    L_Pd = float(derived["L_tilde"] - derived["L_C_tilde"])

    E_base = ((1.0 - alpha1) * E1_eq + alpha2 * E2_eq) / kappa
    E_base += (R_gas * T / (F * kappa)) * math.log((L_Pd * it0_2) / (L_Au * it0_1))
    return float(E_base)


def solve_emix_no_edl(
    params: Dict[str, Any],
    derived: Dict[str, Any],
    mode: str,
    xtol: float,
    max_bracket_expands: int,
    E_guess: Optional[float] = None,
    bracket: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """Root-find Emix for no-EDL kinetics (phi2_1=phi2_2=0)."""
    mode = mode.upper()
    rxn = _effective_reaction_params(params)
    if mode == "FULL":
        f = lambda E: float(full_mode_currents_no_edl(E, derived, params, return_profiles=False)["residual"])
    elif mode == "MEAN":
        f = lambda E: float(mean_mode_residual_no_edl(E, derived, params))
    else:
        raise ValueError("mode must be FULL or MEAN")
    E_mix, info = _solve_root_problem(
        f=f,
        E1_eq=rxn["E1_eq_eff"],
        E2_eq=rxn["E2_eq_eff"],
        xtol=xtol,
        max_bracket_expands=max_bracket_expands,
        E_guess=E_guess,
        bracket=bracket,
        info=dict(mode=mode),
    )
    if not info.get("converged", False):
        return float("nan"), float("nan"), info

    if mode == "FULL":
        cur = full_mode_currents_no_edl(E_mix, derived, params, return_profiles=False)
        i_mix = float(cur["i_mix"])
        resid = float(cur["residual"])
    else:
        i1, _ = currents_mean_field(E_mix, 0.0, 0.0, params)
        i_mix = abs(float(derived["L_Au_tilde"]) * i1)
        resid = mean_mode_residual_no_edl(E_mix, derived, params)

    info["residual_at_root"] = float(resid)
    return E_mix, i_mix, info


def mean_mode_residual(E: float, edl: EDLModel, params: Dict[str, Any], use_affine_phi2: bool) -> float:
    """Mean-field residual: L̃_Au i1 + (L̃-L̃_C) i2 = 0 (Eq. (S3-13b))."""
    phi2_1, phi2_2 = edl.segment_mean_phi2(E, use_affine_phi2=use_affine_phi2)
    i1, i2 = currents_mean_field(E, phi2_1, phi2_2, params)
    L_Au = edl.derived["L_Au_tilde"]
    L_Pd = edl.derived["L_tilde"] - edl.derived["L_C_tilde"]
    return float(L_Au * i1 + L_Pd * i2)


def emix_closed_form_affine(edl: EDLModel, params: Dict[str, Any]) -> float:
    """
    Closed-form Emix when φ2,i = a_i E + b_i (Eq. (D5-3)),
    equivalent to Eq. (D5-6)/(D5-8) but with base term consistent with Eq. (S3-14).
    TODO: verify whether your SI's E0 definition already includes the length ratio term.
    """
    R_gas = float(params["R"]); F = float(params["F"]); T = float(params["T"])
    rxn = _effective_reaction_params(params)
    it0_1 = rxn["it0_1_eff"]; it0_2 = rxn["it0_2_eff"]
    alpha1 = float(params["alpha1"]); alpha2 = float(params["alpha2"])
    z_R1 = float(params["z_R1"]); z_O2 = float(params["z_O2"])
    E1_eq = rxn["E1_eq_eff"]; E2_eq = rxn["E2_eq_eff"]

    kappa = 1.0 - alpha1 + alpha2
    rho = (1.0 - alpha1) + z_R1
    chi = alpha2 - z_O2

    a1 = float(edl.pre["a1"]); a2 = float(edl.pre["a2"])
    b1 = float(edl.pre["b1"]); b2 = float(edl.pre["b2"])

    L_Au = edl.derived["L_Au_tilde"]
    L_Pd = edl.derived["L_tilde"] - edl.derived["L_C_tilde"]

    E_base = ((1.0 - alpha1) * E1_eq + alpha2 * E2_eq) / kappa
    E_base += (R_gas * T / (F * kappa)) * math.log((L_Pd * it0_2) / (L_Au * it0_1))

    kappa_eff = kappa - rho * a1 - chi * a2
    if abs(kappa_eff) < 1e-12:
        raise ValueError("Closed-form Emix denominator kappa_eff is too close to zero")
    return float((kappa * E_base + chi * b2 + rho * b1) / kappa_eff)


def solve_emix(
    edl: EDLModel,
    params: Dict[str, Any],
    mode: str,
    use_affine_phi2: bool,
    xtol: float,
    max_bracket_expands: int,
    E_guess: Optional[float] = None,
    bracket: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """Root-find Emix from zero-net-current condition (Eq. (S3-7b)/(S3-13b))."""
    mode = mode.upper()
    rxn = _effective_reaction_params(params)
    if mode == "FULL":
        f = lambda E: float(full_mode_currents(E, edl, params, return_profiles=False)["residual"])
    elif mode == "MEAN":
        f = lambda E: float(mean_mode_residual(E, edl, params, use_affine_phi2=use_affine_phi2))
    else:
        raise ValueError("mode must be FULL or MEAN")
    E_mix, info = _solve_root_problem(
        f=f,
        E1_eq=rxn["E1_eq_eff"],
        E2_eq=rxn["E2_eq_eff"],
        xtol=xtol,
        max_bracket_expands=max_bracket_expands,
        E_guess=E_guess,
        bracket=bracket,
        info=dict(mode=mode, use_affine_phi2=use_affine_phi2),
    )
    if not info.get("converged", False):
        return float("nan"), float("nan"), info

    if mode == "FULL":
        cur = full_mode_currents(E_mix, edl, params, return_profiles=False)
        i_mix = float(cur["i_mix"])
        resid = float(cur["residual"])
    else:
        phi2_1, phi2_2 = edl.segment_mean_phi2(E_mix, use_affine_phi2=use_affine_phi2)
        i1, _ = currents_mean_field(E_mix, phi2_1, phi2_2, params)
        i_mix = abs(edl.derived["L_Au_tilde"] * i1)
        resid = mean_mode_residual(E_mix, edl, params, use_affine_phi2=use_affine_phi2)

    info["residual_at_root"] = float(resid)
    return E_mix, i_mix, info


# -----------------------------
# Single-case runner
# -----------------------------

def run_case(
    params: Dict[str, Any],
    mode: str,
    return_profiles: bool,
    use_edl: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Run one parameter set.

    Returned units:
    - E_mix, phi2_*: V
    - local profiles i1(x), i2(x): A/m^2
    - i_mix, residual, I_Au, I_Pd: int i d x_tilde quantities in A/m^2
      (multiply by lambda_D to obtain current per unit depth in A/m)
    - *_abs_A outputs: absolute current in A after multiplying by out_of_plane_width
    """
    mode = mode.upper()
    p = copy.deepcopy(params)

    if use_edl is None:
        use_edl = bool(p.get("use_edl", True))
    else:
        use_edl = bool(use_edl)
        p["use_edl"] = use_edl

    if not use_edl:
        global _NO_EDL_WARNED
        if not _NO_EDL_WARNED:
            print("WARNING: use_edl=False skips the EDL PDE solve and uses a no-EDL comparison model.")
            _NO_EDL_WARNED = True

    # EDL entry points:
    # - currents_mean_field(...): phi2_1/phi2_2 in the Frumkin term (-Gamma*beta*phi2)
    # - full_mode_currents(...): phi_tilde(x) in Boltzmann factors exp(+/-Gamma*phi_tilde)
    # no-EDL handling: set phi2_1/phi2_2 = 0, phi_tilde = 0, K_Au/L_Au_tilde, K_Pd/L_Pd_tilde, and skip EDL solve.
    use_affine_phi2 = bool(p.get("use_affine_phi2", True))
    use_closed = bool(p.get("use_closed_form_when_affine", True))
    if return_profiles and mode != "FULL":
        raise ValueError("return_profiles=True only supported for FULL mode")
    rxn = _effective_reaction_params(p)

    if use_edl:
        edl = EDLModel(p)
        L_Au_tilde = float(edl.derived["L_Au_tilde"])
        L_Pd_tilde = float(edl.derived["L_tilde"] - edl.derived["L_C_tilde"])

        if mode == "MEAN" and use_affine_phi2 and use_closed:
            E_mix = emix_closed_form_affine(edl, p)
            resid = mean_mode_residual(E_mix, edl, p, use_affine_phi2=True)
            info = dict(converged=True, method="closed_form(D5-6/D5-8)", iterations=0, residual_at_root=float(resid))
            # i_mix
            phi2_1, phi2_2 = edl.segment_mean_phi2(E_mix, use_affine_phi2=True)
            i1, _ = currents_mean_field(E_mix, phi2_1, phi2_2, p)
            i_mix = abs(edl.derived["L_Au_tilde"] * i1)
        else:
            E_mix, i_mix, info = solve_emix(
                edl=edl,
                params=p,
                mode=mode,
                use_affine_phi2=use_affine_phi2,
                xtol=float(p.get("xtol", 1e-10)),
                max_bracket_expands=int(p.get("max_bracket_expands", 12)),
            )

        out = _build_run_output(
            mode=mode,
            E_mix=E_mix,
            i_mix=i_mix,
            info=info,
            derived=edl.derived,
            a1=float(edl.pre["a1"]),
            b1=float(edl.pre["b1"]),
            a2=float(edl.pre["a2"]),
            b2=float(edl.pre["b2"]),
        )

        if mode == "FULL":
            cur = full_mode_currents(float(E_mix), edl, p, return_profiles=return_profiles)
            out.update(cur)
        else:
            phi2_1, phi2_2 = edl.segment_mean_phi2(float(E_mix), use_affine_phi2=use_affine_phi2)
            i1_loc, i2_loc = currents_mean_field(float(E_mix), phi2_1, phi2_2, p)
            I_Au = L_Au_tilde * i1_loc
            I_Pd = L_Pd_tilde * i2_loc
            out.update(I_Au=float(I_Au), I_Pd=float(I_Pd), residual=float(I_Au + I_Pd))

        if return_profiles:
            # segment-mean phi2 values for reporting
            phi2_1_m, phi2_2_m = edl.segment_mean_phi2(float(E_mix), use_affine_phi2=False)
            out["phi2_1_meanV"] = float(phi2_1_m)
            out["phi2_2_meanV"] = float(phi2_2_m)
            phi_tilde_for_dh = out["phi_tilde"]
        else:
            _, phi_tilde_for_dh = edl.phi_tilde_surface(float(E_mix))

        out.update(_compute_dh_status_from_phi(np.asarray(phi_tilde_for_dh, dtype=float), p))
        _attach_current_unit_outputs(
            out,
            float(edl.derived["lambda_D"]),
            float(p.get("out_of_plane_width", 1.0)),
            float(edl.derived["L_Au"] + edl.derived["L_Pd_len"]),
        )
    else:
        derived = compute_derived_params(p)
        L_Au_tilde = float(derived["L_Au_tilde"])
        L_Pd_tilde = float(derived["L_tilde"] - derived["L_C_tilde"])

        if mode == "MEAN" and use_closed:
            E_mix = emix_closed_form_no_edl(derived, p)
            resid = mean_mode_residual_no_edl(E_mix, derived, p)
            info = dict(converged=True, method="closed_form(no_edl)", iterations=0, residual_at_root=float(resid))
            i1, _ = currents_mean_field(E_mix, 0.0, 0.0, p)
            i_mix = abs(derived["L_Au_tilde"] * i1)
        else:
            E_mix, i_mix, info = solve_emix_no_edl(
                params=p,
                derived=derived,
                mode=mode,
                xtol=float(p.get("xtol", 1e-10)),
                max_bracket_expands=int(p.get("max_bracket_expands", 12)),
            )

        out = _build_run_output(
            mode=mode,
            E_mix=E_mix,
            i_mix=i_mix,
            info=info,
            derived=derived,
            a1=0.0,
            b1=0.0,
            a2=0.0,
            b2=0.0,
        )

        if mode == "FULL":
            cur = full_mode_currents_no_edl(float(E_mix), derived, p, return_profiles=return_profiles)
            out.update(cur)
        else:
            i1_loc, i2_loc = currents_mean_field(float(E_mix), 0.0, 0.0, p)
            I_Au = L_Au_tilde * i1_loc
            I_Pd = L_Pd_tilde * i2_loc
            out.update(I_Au=float(I_Au), I_Pd=float(I_Pd), residual=float(I_Au + I_Pd))

        if return_profiles:
            out["phi2_1_meanV"] = 0.0
            out["phi2_2_meanV"] = 0.0

        out.update(max_abs_phi_tilde=0.0, dh_warn_threshold=float(p.get("dh_warn_threshold", 1.0)), debye_huckel_ok=True)
        _attach_current_unit_outputs(
            out,
            float(derived["lambda_D"]),
            float(p.get("out_of_plane_width", 1.0)),
            float(derived["L_Au"] + derived["L_Pd_len"]),
        )

    out.update(rxn)
    _handle_dh_violation(out, p, mode=mode, use_edl=use_edl)

    return out


# -----------------------------
# Saving + plotting
# -----------------------------

def plot_baseline_profiles(case_full: Dict[str, Any], params: Dict[str, Any], out_dir: Path) -> None:
    """Save baseline FULL-mode potential and current profiles as PNG."""
    fig_dir = ensure_dir(out_dir / "figures")

    R_gas = float(params["R"]); F = float(params["F"]); T = float(params["T"])
    scale = R_gas * T / F

    x_tilde = case_full["x_tilde"]
    x_nm = x_tilde * case_full["lambda_D"] * 1e9
    phi2 = scale * case_full["phi_tilde"]

    i1 = case_full["i1"]
    i2 = case_full["i2"]
    (i1_plot, i2_plot), i_label, _ = _scaled_current_display("local_current_density", i1, i2)

    L_Au_nm = case_full["L_Au_tilde"] * case_full["lambda_D"] * 1e9
    L_C_nm = case_full["L_C_tilde"] * case_full["lambda_D"] * 1e9

    fig_phi, ax_phi = _new_figure()
    ax_phi.plot(x_nm, phi2, color=NATURE_COLORS["blue"])
    _add_vertical_boundaries(ax_phi, L_Au_nm, L_C_nm)
    _style_axes(ax_phi, "x [nm]", _plot_axis_label("phi2"), "Reaction-plane potential along surface")
    _finalize_figure(fig_phi, fig_dir / "baseline_phi2.png")

    fig_i, ax_i = _new_figure()
    ax_i.plot(x_nm, i1_plot, label="i1 (Au)", color=NATURE_COLORS["blue"])
    ax_i.plot(x_nm, i2_plot, label="i2 (Pd)", color=NATURE_COLORS["orange"])
    _add_vertical_boundaries(ax_i, L_Au_nm, L_C_nm)
    _style_axes(ax_i, "x [nm]", i_label, "Local current density profiles")
    ax_i.legend(loc="best")
    _finalize_figure(fig_i, fig_dir / "baseline_currents.png")

    plot_baseline_profiles_html(
        x_nm=x_nm,
        phi2=phi2,
        i1=i1,
        i2=i2,
        L_Au_nm=L_Au_nm,
        L_C_nm=L_C_nm,
        params=params,
        fig_dir=fig_dir,
    )


def plot_baseline_profiles_html(
    x_nm: np.ndarray,
    phi2: np.ndarray,
    i1: np.ndarray,
    i2: np.ndarray,
    L_Au_nm: float,
    L_C_nm: float,
    params: Dict[str, Any],
    fig_dir: Path,
) -> None:
    if go is None:
        _maybe_warn_plotly()
        return

    hover_params = format_hover_params(params, keys=HOVER_PARAM_KEYS)
    hover_text = [hover_params] * len(x_nm)

    fig_phi = go.Figure()
    fig_phi.add_trace(
        go.Scatter(
            x=x_nm,
            y=phi2,
            mode="lines",
            name="phi2",
            line=dict(color=NATURE_COLORS["blue"], width=2.5),
            text=hover_text,
            hovertemplate="x=%{x:.6g} nm<br>phi2=%{y:.6g} V<br>%{text}<extra></extra>",
        )
    )
    _add_plotly_boundaries(fig_phi, L_Au_nm, L_C_nm)
    _style_plotly_figure(fig_phi, "Reaction-plane potential along surface", "x [nm]", _plot_axis_label("phi2"))
    write_plotly_html(fig_phi, fig_dir / "baseline_phi2.html")

    fig_i = go.Figure()
    fig_i.add_trace(
        go.Scatter(
            x=x_nm,
            y=i1,
            mode="lines",
            name="i1 (Au)",
            line=dict(color=NATURE_COLORS["blue"], width=2.4),
            text=hover_text,
            hovertemplate="x=%{x:.6g} nm<br>%{fullData.name}=%{y:.6g} A/m^2<br>%{text}<extra></extra>",
        )
    )
    fig_i.add_trace(
        go.Scatter(
            x=x_nm,
            y=i2,
            mode="lines",
            name="i2 (Pd)",
            line=dict(color=NATURE_COLORS["orange"], width=2.4),
            text=hover_text,
            hovertemplate="x=%{x:.6g} nm<br>%{fullData.name}=%{y:.6g} A/m^2<br>%{text}<extra></extra>",
        )
    )
    _add_plotly_boundaries(fig_i, L_Au_nm, L_C_nm)
    _style_plotly_figure(fig_i, "Local current density profiles", "x [nm]", _plot_axis_label("local_current_density"))
    write_plotly_html(fig_i, fig_dir / "baseline_currents.html")


def plot_ofat_html(
    dfp_plot: pd.DataFrame,
    pname: str,
    metric: str,
    ylab: str,
    xscale: str,
    modes: List[str],
    fig_dir: Path,
) -> None:
    if go is None:
        _maybe_warn_plotly()
        return
    if dfp_plot.empty:
        return
    x_col = "value_M" if pname == "C_tot" and "value_M" in dfp_plot.columns else "value"
    x_label = _param_axis_label(pname)

    fig = go.Figure()
    for mode in modes:
        sub = dfp_plot[dfp_plot["mode"] == mode]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub[x_col],
                y=sub[metric],
                mode="lines+markers",
                name=mode,
                line=dict(width=2.2),
                marker=dict(size=6),
                text=sub["hover_params"],
                hovertemplate=f"{x_label}=%{{x:.6g}}<br>{metric}=%{{y:.6g}}<br>%{{text}}<extra>{mode}</extra>",
            )
        )

    _style_plotly_figure(fig, f"{metric} vs {pname}", x_label, ylab)
    if xscale == "log":
        fig.update_xaxes(type="log")
    write_plotly_html(fig, fig_dir / f"ofat_{pname}_{metric}.html")


def _format_nm_tag(val_m: float) -> str:
    nm = float(val_m) * 1e9
    if abs(nm - round(nm)) < 1e-6:
        return str(int(round(nm)))
    return f"{nm:.3g}".replace(".", "p")


def _case_tag_from_params(params: Dict[str, Any]) -> str:
    return "LAu{0}_Lgap{1}_LPd{2}".format(
        _format_nm_tag(params["L_Au"]),
        _format_nm_tag(params["L_gap"]),
        _format_nm_tag(params["L_Pd_len"]),
    )


def build_profiles_for_emix(
    params: Dict[str, Any],
    E_mix: float,
    use_edl: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if use_edl:
        edl = EDLModel(params)
        prof = full_mode_currents(float(E_mix), edl, params, return_profiles=True)
        derived = edl.derived
        return prof, derived
    derived = compute_derived_params(params)
    prof = full_mode_currents_no_edl(float(E_mix), derived, params, return_profiles=True)
    return prof, derived


def _save_profiles_npz(path: Path, prof: Dict[str, Any], derived: Dict[str, Any]) -> None:
    scale = float(derived["R"]) * float(derived["T"]) / float(derived["F"])
    x_tilde = prof["x_tilde"]
    x_m = x_tilde * float(derived["lambda_D"])
    phi2_V = scale * prof["phi_tilde"]

    np.savez(
        path,
        x_tilde=x_tilde,
        x_m=x_m,
        phi_tilde=prof["phi_tilde"],
        phi2_V=phi2_V,
        i1=prof["i1"],
        i2=prof["i2"],
        mask_Au=prof["mask_Au"],
        mask_Pd=prof["mask_Pd"],
        L_Au_tilde=float(derived["L_Au_tilde"]),
        L_C_tilde=float(derived["L_C_tilde"]),
        lambda_D=float(derived["lambda_D"]),
    )


def _polarization_E_values(params: Dict[str, Any], solver_settings: Dict[str, Any]) -> np.ndarray:
    E_min = solver_settings.get("E_min")
    E_max = solver_settings.get("E_max")
    n_E = int(solver_settings.get("n_E", 200))
    if E_min is None or E_max is None:
        span = float(solver_settings.get("E_span", 0.5))
        rxn = _effective_reaction_params(params)
        E1_eq = rxn["E1_eq_eff"]
        E2_eq = rxn["E2_eq_eff"]
        E_min = min(E1_eq, E2_eq) - span
        E_max = max(E1_eq, E2_eq) + span
    return np.linspace(float(E_min), float(E_max), n_E)


def _compare_polarization_E_values(
    params: Dict[str, Any],
    solver_settings: Dict[str, Any],
    E_mix_edl: float,
    E_mix_no: float,
) -> np.ndarray:
    E_min = solver_settings.get("E_min")
    E_max = solver_settings.get("E_max")
    n_E = int(solver_settings.get("n_E", 200))
    if E_min is not None and E_max is not None:
        return np.linspace(float(E_min), float(E_max), n_E)
    halfspan = float(solver_settings.get("E_window_halfspan", 0.10))
    lo = min(float(E_mix_edl), float(E_mix_no)) - halfspan
    hi = max(float(E_mix_edl), float(E_mix_no)) + halfspan
    return np.linspace(lo, hi, n_E)


def compute_polarization_curve(
    params: Dict[str, Any],
    mode: str,
    use_edl: bool,
    E_values: np.ndarray,
    use_affine_phi2: bool,
) -> Dict[str, np.ndarray]:
    """
    Polarization curve arrays versus E.

    I_Au, I_Pd, and I_total are x_tilde-integrated currents, i.e.
    int i d x_tilde in A/m^2. Multiply by lambda_D for A/m.
    """
    mode = mode.upper()
    I_Au = np.zeros_like(E_values, dtype=float)
    I_Pd = np.zeros_like(E_values, dtype=float)
    I_total = np.zeros_like(E_values, dtype=float)
    lambda_D = float(compute_derived_params(params)["lambda_D"])
    out_of_plane_width = float(params.get("out_of_plane_width", 1.0))
    reactive_length_m = float(params["L_Au"]) + float(params["L_Pd_len"])

    if use_edl:
        edl = EDLModel(params)
        lambda_D = float(edl.derived["lambda_D"])
        L_Au = float(edl.derived["L_Au_tilde"])
        L_Pd = float(edl.derived["L_tilde"] - edl.derived["L_C_tilde"])
        for i, E in enumerate(E_values):
            if mode == "FULL":
                cur = full_mode_currents(float(E), edl, params, return_profiles=False)
                I_Au[i] = float(cur["I_Au"])
                I_Pd[i] = float(cur["I_Pd"])
                I_total[i] = float(cur["residual"])
            elif mode == "MEAN":
                phi2_1, phi2_2 = edl.segment_mean_phi2(float(E), use_affine_phi2=use_affine_phi2)
                i1, i2 = currents_mean_field(float(E), phi2_1, phi2_2, params)
                I_Au[i] = L_Au * i1
                I_Pd[i] = L_Pd * i2
                I_total[i] = I_Au[i] + I_Pd[i]
            else:
                raise ValueError("mode must be FULL or MEAN")
    else:
        derived = compute_derived_params(params)
        lambda_D = float(derived["lambda_D"])
        L_Au = float(derived["L_Au_tilde"])
        L_Pd = float(derived["L_tilde"] - derived["L_C_tilde"])
        for i, E in enumerate(E_values):
            if mode == "FULL":
                cur = full_mode_currents_no_edl(float(E), derived, params, return_profiles=False)
                I_Au[i] = float(cur["I_Au"])
                I_Pd[i] = float(cur["I_Pd"])
                I_total[i] = float(cur["residual"])
            elif mode == "MEAN":
                i1, i2 = currents_mean_field(float(E), 0.0, 0.0, params)
                I_Au[i] = L_Au * i1
                I_Pd[i] = L_Pd * i2
                I_total[i] = I_Au[i] + I_Pd[i]
            else:
                raise ValueError("mode must be FULL or MEAN")

    return dict(
        E=E_values,
        I_total=I_total,
        I_total_phys_A_per_m=lambda_D * I_total,
        I_total_abs_A=out_of_plane_width * lambda_D * I_total,
        I_total_avg_A_per_m2=(out_of_plane_width * lambda_D * I_total) / (reactive_length_m * out_of_plane_width),
        I_Au=I_Au,
        I_Au_phys_A_per_m=lambda_D * I_Au,
        I_Au_abs_A=out_of_plane_width * lambda_D * I_Au,
        I_Pd=I_Pd,
        I_Pd_phys_A_per_m=lambda_D * I_Pd,
        I_Pd_abs_A=out_of_plane_width * lambda_D * I_Pd,
    )


def plot_compare_polarization_curve(
    curve_edl: Dict[str, np.ndarray],
    curve_no: Dict[str, np.ndarray],
    E_mix_edl: float,
    E_mix_no: float,
    out_path: Path,
    title: str,
) -> None:
    (I_edl_plot, I_no_plot), y_label, _ = _scaled_current_display(
        "I_net_avg",
        curve_edl["I_total_avg_A_per_m2"],
        curve_no["I_total_avg_A_per_m2"],
    )
    fig, ax = _new_figure(NATURE_WIDE_FIGSIZE)
    ax.plot(curve_edl["E"], I_edl_plot, label=LEGEND_WITH_EDL, color=NATURE_COLORS["blue"])
    ax.plot(curve_no["E"], I_no_plot, label=LEGEND_WITHOUT_EDL, color=NATURE_COLORS["orange"])
    ax.axhline(0.0, color=NATURE_COLORS["black"], linewidth=0.9)
    ax.axvline(E_mix_edl, linestyle=(0, (3, 2)), linewidth=1.0, color=NATURE_COLORS["blue"], alpha=0.9)
    ax.axvline(E_mix_no, linestyle=(0, (3, 2)), linewidth=1.0, color=NATURE_COLORS["orange"], alpha=0.9)
    _style_axes(ax, "E [V]", y_label, title)
    ax.legend(loc="best")
    _finalize_figure(fig, out_path)


def plot_compare_emix_imix(
    E_mix_edl: float,
    i_mix_avg_edl: float,
    E_mix_no: float,
    i_mix_avg_no: float,
    out_path: Path,
    title: str,
) -> None:
    (i_mix_plot,), i_label, _ = _scaled_current_display("i_mix_avg", np.array([i_mix_avg_edl, i_mix_avg_no], dtype=float))
    fig, axes = plt.subplots(1, 2, figsize=NATURE_DOUBLE_FIGSIZE)
    labels = [LEGEND_WITH_EDL, LEGEND_WITHOUT_EDL]
    colors = [NATURE_COLORS["blue"], NATURE_COLORS["orange"]]
    axes[0].bar(labels, [E_mix_edl, E_mix_no], color=colors, edgecolor=NATURE_COLORS["black"])
    axes[1].bar(labels, i_mix_plot, color=colors, edgecolor=NATURE_COLORS["black"])
    _style_axes(axes[0], "", _plot_axis_label("E_mix"), r"$E_{\mathrm{mix}}$")
    _style_axes(axes[1], "", i_label, r"$i_{\mathrm{mix}}$")
    for ax in axes:
        ax.tick_params(axis="x", rotation=0)
    fig.suptitle(title, x=0.02, y=1.02, ha="left", fontsize=9, fontweight="semibold")
    _finalize_figure(fig, out_path)


def plot_compare_phi2(
    prof_edl: Dict[str, Any],
    derived_edl: Dict[str, Any],
    prof_no: Dict[str, Any],
    derived_no: Dict[str, Any],
    out_path: Path,
    title: str,
) -> None:
    scale_edl = float(derived_edl["R"]) * float(derived_edl["T"]) / float(derived_edl["F"])
    scale_no = float(derived_no["R"]) * float(derived_no["T"]) / float(derived_no["F"])
    x_nm = prof_edl["x_tilde"] * float(derived_edl["lambda_D"]) * 1e9
    phi2_edl = scale_edl * prof_edl["phi_tilde"]
    phi2_no = scale_no * prof_no["phi_tilde"]

    L_Au_nm = float(derived_edl["L_Au_tilde"]) * float(derived_edl["lambda_D"]) * 1e9
    L_C_nm = float(derived_edl["L_C_tilde"]) * float(derived_edl["lambda_D"]) * 1e9

    fig, ax = _new_figure()
    ax.plot(x_nm, phi2_edl, label=LEGEND_WITH_EDL, color=NATURE_COLORS["blue"])
    ax.plot(x_nm, phi2_no, label=LEGEND_WITHOUT_EDL, color=NATURE_COLORS["orange"])
    _add_vertical_boundaries(ax, L_Au_nm, L_C_nm)
    _style_axes(ax, "x [nm]", _plot_axis_label("phi2"), title)
    ax.legend(loc="best")
    _finalize_figure(fig, out_path)


def plot_compare_potentials_overpotential(
    prof_edl: Dict[str, Any],
    derived_edl: Dict[str, Any],
    prof_no: Dict[str, Any],
    derived_no: Dict[str, Any],
    params: Dict[str, Any],
    E_mix_edl: float,
    E_mix_no: float,
    out_path: Path,
    title: str,
) -> None:
    rxn = _effective_reaction_params(params)
    scale_edl = float(derived_edl["R"]) * float(derived_edl["T"]) / float(derived_edl["F"])
    scale_no = float(derived_no["R"]) * float(derived_no["T"]) / float(derived_no["F"])
    x_nm = prof_edl["x_tilde"] * float(derived_edl["lambda_D"]) * 1e9
    phi2_edl = scale_edl * prof_edl["phi_tilde"]
    phi2_no = scale_no * prof_no["phi_tilde"]

    metal_edl = np.full_like(x_nm, float(E_mix_edl), dtype=float)
    metal_no = np.full_like(x_nm, float(E_mix_no), dtype=float)

    eta_edl = np.full_like(x_nm, np.nan, dtype=float)
    eta_no = np.full_like(x_nm, np.nan, dtype=float)
    mask_Au = np.asarray(prof_edl["mask_Au"], dtype=bool)
    mask_Pd = np.asarray(prof_edl["mask_Pd"], dtype=bool)
    E1_eq = rxn["E1_eq_eff"]
    E2_eq = rxn["E2_eq_eff"]
    eta_edl[mask_Au] = float(E_mix_edl) - E1_eq - phi2_edl[mask_Au]
    eta_edl[mask_Pd] = float(E_mix_edl) - E2_eq - phi2_edl[mask_Pd]
    eta_no[mask_Au] = float(E_mix_no) - E1_eq - phi2_no[mask_Au]
    eta_no[mask_Pd] = float(E_mix_no) - E2_eq - phi2_no[mask_Pd]

    L_Au_nm = float(derived_edl["L_Au_tilde"]) * float(derived_edl["lambda_D"]) * 1e9
    L_C_nm = float(derived_edl["L_C_tilde"]) * float(derived_edl["lambda_D"]) * 1e9

    fig, axes = plt.subplots(1, 3, figsize=(10.3, 2.9))

    axes[0].plot(x_nm, metal_edl, label=LEGEND_WITH_EDL, color=NATURE_COLORS["blue"])
    axes[0].plot(x_nm, metal_no, label=LEGEND_WITHOUT_EDL, color=NATURE_COLORS["orange"])
    _add_vertical_boundaries(axes[0], L_Au_nm, L_C_nm)
    _style_axes(axes[0], "x [nm]", _plot_axis_label("metal_potential"), "Metal potential")
    axes[0].legend(loc="best")

    axes[1].plot(x_nm, phi2_edl, label=LEGEND_WITH_EDL, color=NATURE_COLORS["blue"])
    axes[1].plot(x_nm, phi2_no, label=LEGEND_WITHOUT_EDL, color=NATURE_COLORS["orange"])
    _add_vertical_boundaries(axes[1], L_Au_nm, L_C_nm)
    _style_axes(axes[1], "x [nm]", _plot_axis_label("phi2"), "Reaction-plane potential")

    axes[2].plot(x_nm, eta_edl, label=LEGEND_WITH_EDL, color=NATURE_COLORS["blue"])
    axes[2].plot(x_nm, eta_no, label=LEGEND_WITHOUT_EDL, color=NATURE_COLORS["orange"])
    _add_vertical_boundaries(axes[2], L_Au_nm, L_C_nm)
    _style_axes(axes[2], "x [nm]", _plot_axis_label("overpotential"), "Local overpotential")

    fig.suptitle(title, x=0.02, y=1.02, ha="left", fontsize=9, fontweight="semibold")
    _finalize_figure(fig, out_path)


def _add_panel_label(ax: Axes, label: str) -> None:
    ax.text(
        -0.17,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=NATURE_COLORS["black"],
        clip_on=False,
    )


def _style_publication_axes(ax: Axes, xlabel: str, ylabel: str, title: str) -> None:
    _style_axes(ax, xlabel, ylabel, title)
    ax.title.set_fontsize(12.5)
    ax.xaxis.label.set_fontsize(11.5)
    ax.yaxis.label.set_fontsize(11.5)
    ax.tick_params(labelsize=10.5, length=4.0, width=0.9, pad=3.0)


def plot_publication_compare_panels(
    prof_edl: Dict[str, Any],
    derived_edl: Dict[str, Any],
    prof_no: Dict[str, Any],
    derived_no: Dict[str, Any],
    params: Dict[str, Any],
    E_mix_edl: float,
    E_mix_no: float,
    out_base: Path,
    i_mix_abs_edl: Optional[float] = None,
    i_mix_abs_no: Optional[float] = None,
    export_formats: Tuple[str, ...] = ("png", "pdf", "svg"),
) -> Dict[str, str]:
    """
    Publication-style compare figure:
    (a) compact bar-chart summary of mixed potential and mixed current
    (b) reaction-plane potential
    (c) normalized local reactant concentration
    (d) local overpotential
    (e) local current density profile

    All panels compare with/without EDL at their converged mixed potentials.
    """
    out_base = Path(out_base)
    ensure_dir(out_base.parent)
    export_formats = tuple(fmt.lower() for fmt in export_formats)
    allowed_formats = {"png", "pdf", "svg"}
    unknown_formats = sorted(set(export_formats) - allowed_formats)
    if unknown_formats:
        raise ValueError(f"Unsupported export format(s): {', '.join(unknown_formats)}")

    rxn = _effective_reaction_params(params)
    scale_edl = float(derived_edl["R"]) * float(derived_edl["T"]) / float(derived_edl["F"])
    scale_no = float(derived_no["R"]) * float(derived_no["T"]) / float(derived_no["F"])

    x_nm = np.asarray(prof_edl["x_tilde"], dtype=float) * float(derived_edl["lambda_D"]) * 1e9
    phi2_edl = scale_edl * np.asarray(prof_edl["phi_tilde"], dtype=float)
    phi2_no = scale_no * np.asarray(prof_no["phi_tilde"], dtype=float)

    eta_edl = np.full_like(x_nm, np.nan, dtype=float)
    eta_no = np.full_like(x_nm, np.nan, dtype=float)
    mask_Au = np.asarray(prof_edl["mask_Au"], dtype=bool)
    mask_Pd = np.asarray(prof_edl["mask_Pd"], dtype=bool)
    E1_eq = float(rxn["E1_eq_eff"])
    E2_eq = float(rxn["E2_eq_eff"])
    eta_edl[mask_Au] = float(E_mix_edl) - E1_eq - phi2_edl[mask_Au]
    eta_edl[mask_Pd] = float(E_mix_edl) - E2_eq - phi2_edl[mask_Pd]
    eta_no[mask_Au] = float(E_mix_no) - E1_eq - phi2_no[mask_Au]
    eta_no[mask_Pd] = float(E_mix_no) - E2_eq - phi2_no[mask_Pd]

    phi_tilde_edl = np.asarray(prof_edl["phi_tilde"], dtype=float)
    z_R1 = float(params["z_R1"])
    z_O2 = float(params["z_O2"])
    c_R1_norm = np.asarray(safe_exp(-z_R1 * phi_tilde_edl), dtype=float)
    c_O2_norm = np.asarray(safe_exp(-z_O2 * phi_tilde_edl), dtype=float)

    i1_edl = np.asarray(prof_edl["i1"], dtype=float)
    i2_edl = np.asarray(prof_edl["i2"], dtype=float)
    i1_no = np.asarray(prof_no["i1"], dtype=float)
    i2_no = np.asarray(prof_no["i2"], dtype=float)
    mask_Au_no = np.asarray(prof_no["mask_Au"], dtype=bool)
    mask_Pd_no = np.asarray(prof_no["mask_Pd"], dtype=bool)

    i1_edl_segment = np.full_like(i1_edl, np.nan, dtype=float)
    i2_edl_segment = np.full_like(i2_edl, np.nan, dtype=float)
    i1_no_segment = np.full_like(i1_no, np.nan, dtype=float)
    i2_no_segment = np.full_like(i2_no, np.nan, dtype=float)
    i1_edl_segment[mask_Au] = i1_edl[mask_Au]
    i2_edl_segment[mask_Pd] = i2_edl[mask_Pd]
    i1_no_segment[mask_Au_no] = i1_no[mask_Au_no]
    i2_no_segment[mask_Pd_no] = i2_no[mask_Pd_no]
    (i1_edl_plot, i1_no_plot, i2_edl_plot, i2_no_plot), i_label, _ = _scaled_current_display(
        "local_current_density",
        i1_edl_segment,
        i1_no_segment,
        i2_edl_segment,
        i2_no_segment,
    )
    reactive_area_m2 = (float(params["L_Au"]) + float(params["L_Pd_len"])) * float(params.get("out_of_plane_width", 1.0))
    i_mix_avg_vals = np.array(
        [
            np.nan if i_mix_abs_no is None else float(i_mix_abs_no) / reactive_area_m2,
            np.nan if i_mix_abs_edl is None else float(i_mix_abs_edl) / reactive_area_m2,
        ],
        dtype=float,
    )
    (i_mix_abs_plot,), i_mix_abs_label, _ = _scaled_current_display("i_mix_avg", i_mix_avg_vals)

    L_Au_nm = float(derived_edl["L_Au_tilde"]) * float(derived_edl["lambda_D"]) * 1e9
    L_C_nm = float(derived_edl["L_C_tilde"]) * float(derived_edl["lambda_D"]) * 1e9

    fig = plt.figure(figsize=(12.0, 6.8))
    gs = fig.add_gridspec(2, 11, height_ratios=[1.0, 1.0], wspace=0.25, hspace=0.62)
    gs_a = gs[0, 2:5].subgridspec(1, 3, width_ratios=[1.0, 0.82, 1.0], wspace=0.0)

    ax_a1 = fig.add_subplot(gs_a[0, 0])
    ax_a2 = fig.add_subplot(gs_a[0, 2])
    ax_b = fig.add_subplot(gs[0, 6:9])
    ax_c = fig.add_subplot(gs[1, 0:3])
    ax_d = fig.add_subplot(gs[1, 4:7])
    ax_e = fig.add_subplot(gs[1, 8:11])

    labels = [LEGEND_WITHOUT_EDL, LEGEND_WITH_EDL]
    categories = np.array([0.0, 1.0], dtype=float)
    width = 0.56
    bar_colors = [NATURE_COLORS["orange"], NATURE_COLORS["blue"]]

    ax_a1.bar(
        categories,
        [float(E_mix_no), float(E_mix_edl)],
        width=width,
        color=bar_colors,
        edgecolor=NATURE_COLORS["black"],
        linewidth=0.8,
        zorder=3,
    )
    _style_publication_axes(ax_a1, "", r"$E_{\mathrm{mix}}$ [V]", r"$E_{\mathrm{mix}}$")
    ax_a1.set_xticks(categories)
    ax_a1.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax_a1.set_xlim(-0.6, 1.6)
    ax_a1.yaxis.labelpad = 6.0
    ax_a1.tick_params(axis="x", labelsize=9.5)

    ax_a2.bar(
        categories,
        i_mix_abs_plot,
        width=width,
        color=bar_colors,
        edgecolor=NATURE_COLORS["black"],
        linewidth=0.8,
        zorder=3,
    )
    i_mix_abs_label_short = i_mix_abs_label.replace("Average mixed current density, ", "")
    _style_publication_axes(ax_a2, "", i_mix_abs_label_short, r"$i_{\mathrm{mix}}$")
    ax_a2.set_xticks(categories)
    ax_a2.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax_a2.set_xlim(-0.6, 1.6)
    ax_a2.yaxis.set_label_position("left")
    ax_a2.yaxis.tick_left()
    ax_a2.tick_params(axis="y", labelleft=True, left=True, labelright=False, right=False, pad=3.5)
    ax_a2.tick_params(axis="x", labelsize=9.5)
    ax_a2.yaxis.labelpad = 6.0

    ax_b.plot(x_nm, phi2_edl, label=LEGEND_WITH_EDL, color=NATURE_COLORS["blue"])
    ax_b.plot(x_nm, phi2_no, label=LEGEND_WITHOUT_EDL, color=NATURE_COLORS["orange"])
    _add_vertical_boundaries(ax_b, L_Au_nm, L_C_nm)
    _style_publication_axes(ax_b, "x [nm]", r"Reaction-plane potential, $\phi_{\mathrm{RP}}(x)$ [V]", "Reaction-plane potential")
    ax_b.yaxis.labelpad = 5.0
    ax_b.legend(loc="upper right", bbox_to_anchor=(0.98, 0.90), borderaxespad=0.15, fontsize=10.0, handlelength=2.0)

    ax_c.plot(x_nm, c_R1_norm, label=r"$c_{\mathrm{R1}}/c_{\mathrm{bulk}}$ (with EDL)", color=NATURE_COLORS["green"])
    ax_c.plot(x_nm, c_O2_norm, label=r"$c_{\mathrm{O2}}/c_{\mathrm{bulk}}$ (with EDL)", color=NATURE_COLORS["gold"])
    ax_c.plot(x_nm, np.ones_like(x_nm), label=LEGEND_WITHOUT_EDL, color=NATURE_COLORS["orange"], linestyle="--")
    _add_vertical_boundaries(ax_c, L_Au_nm, L_C_nm)
    _style_publication_axes(ax_c, "x [nm]", r"$c_i/c_{\mathrm{bulk}}$ [-]", "Local reactant concentration")
    ax_c.yaxis.labelpad = 5.0
    ax_c.legend(loc="lower right", bbox_to_anchor=(0.98, 0.30), borderaxespad=0.15, fontsize=9.3, handlelength=2.0)

    ax_d.plot(x_nm, eta_edl, label=LEGEND_WITH_EDL, color=NATURE_COLORS["blue"])
    ax_d.plot(x_nm, eta_no, label=LEGEND_WITHOUT_EDL, color=NATURE_COLORS["orange"])
    _add_vertical_boundaries(ax_d, L_Au_nm, L_C_nm)
    _style_publication_axes(ax_d, "x [nm]", _plot_axis_label("overpotential"), "Local overpotential")
    ax_d.yaxis.labelpad = 5.0
    ax_d.legend(loc="center right", bbox_to_anchor=(0.98, 0.50), borderaxespad=0.15, fontsize=10.0, handlelength=2.0)

    ax_e.plot(x_nm, i1_edl_plot, label=r"$i_1$ (Au), with EDL", color=NATURE_COLORS["green"])
    ax_e.plot(x_nm, i1_no_plot, label=r"$i_1$ (Au), without EDL", color=NATURE_COLORS["green"], linestyle="--")
    ax_e.plot(x_nm, i2_edl_plot, label=r"$i_2$ (Pd), with EDL", color=NATURE_COLORS["gold"])
    ax_e.plot(x_nm, i2_no_plot, label=r"$i_2$ (Pd), without EDL", color=NATURE_COLORS["gold"], linestyle="--")
    _add_vertical_boundaries(ax_e, L_Au_nm, L_C_nm)
    current_label_short = i_label.replace("Local current density, ", "")
    _style_publication_axes(ax_e, "x [nm]", current_label_short, "Local current density")
    ax_e.yaxis.labelpad = 5.0
    ax_e.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), borderaxespad=0.15, fontsize=8.4, handlelength=1.8)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.12, top=0.93, wspace=0.25, hspace=0.62)

    panel_specs = [("a", ax_a1), ("b", ax_b), ("c", ax_c), ("d", ax_d), ("e", ax_e)]
    for label, ax in panel_specs:
        pos = ax.get_position()
        fig.text(
            pos.x0 - 0.028,
            pos.y1 + 0.014,
            label,
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="bottom",
            color=NATURE_COLORS["black"],
        )

    paths: Dict[str, str] = {}
    if "png" in export_formats:
        png_path = out_base.with_suffix(".png")
        fig.savefig(png_path, dpi=600)
        paths["png"] = str(png_path)
    if "pdf" in export_formats:
        pdf_path = out_base.with_suffix(".pdf")
        fig.savefig(pdf_path)
        paths["pdf"] = str(pdf_path)
    if "svg" in export_formats:
        svg_path = out_base.with_suffix(".svg")
        fig.savefig(svg_path)
        paths["svg"] = str(svg_path)
    plt.close(fig)
    return paths


def _safe_ratio_pct(numerator: float, denominator: float) -> Tuple[float, float]:
    if denominator == 0.0:
        return float("nan"), float("nan")
    ratio = numerator / denominator
    pct = 100.0 * (numerator - denominator) / denominator
    return float(ratio), float(pct)


def _make_edl_comparison_metrics(res_edl: Dict[str, Any], res_no: Dict[str, Any], mode: str) -> Dict[str, float | str | bool]:
    ratio_i_mix, pct_i_mix = _safe_ratio_pct(
        float(res_edl["i_mix_norm_A_per_m2"]),
        float(res_no["i_mix_norm_A_per_m2"]),
    )
    ratio_i_mix_phys, pct_i_mix_phys = _safe_ratio_pct(
        float(res_edl["i_mix_phys_A_per_m"]),
        float(res_no["i_mix_phys_A_per_m"]),
    )
    ratio_i_mix_abs, pct_i_mix_abs = _safe_ratio_pct(
        float(res_edl["i_mix_abs_A"]),
        float(res_no["i_mix_abs_A"]),
    )
    ratio_i_mix_avg, pct_i_mix_avg = _safe_ratio_pct(
        float(res_edl["i_mix_avg_A_per_m2"]),
        float(res_no["i_mix_avg_A_per_m2"]),
    )
    return dict(
        mode=mode,
        delta_E_mix=float(res_edl["E_mix"] - res_no["E_mix"]),
        delta_i_mix_norm_A_per_m2=float(res_edl["i_mix_norm_A_per_m2"] - res_no["i_mix_norm_A_per_m2"]),
        delta_i_mix_phys_A_per_m=float(res_edl["i_mix_phys_A_per_m"] - res_no["i_mix_phys_A_per_m"]),
        delta_i_mix_abs_A=float(res_edl["i_mix_abs_A"] - res_no["i_mix_abs_A"]),
        delta_i_mix_avg_A_per_m2=float(res_edl["i_mix_avg_A_per_m2"] - res_no["i_mix_avg_A_per_m2"]),
        ratio_i_mix=float(ratio_i_mix),
        pct_i_mix=float(pct_i_mix),
        ratio_i_mix_phys=float(ratio_i_mix_phys),
        pct_i_mix_phys=float(pct_i_mix_phys),
        ratio_i_mix_abs=float(ratio_i_mix_abs),
        pct_i_mix_abs=float(pct_i_mix_abs),
        ratio_i_mix_avg=float(ratio_i_mix_avg),
        pct_i_mix_avg=float(pct_i_mix_avg),
        max_abs_phi_tilde_with_edl=float(res_edl["max_abs_phi_tilde"]),
        debye_huckel_ok_with_edl=bool(res_edl["debye_huckel_ok"]),
    )


def run_edl_comparison_pair(
    base_params: Dict[str, Any],
    mode: str = "FULL",
) -> Dict[str, Any]:
    p0 = copy.deepcopy(base_params)
    mode = mode.upper()
    res_edl = run_case(p0, mode=mode, return_profiles=False, use_edl=True)
    res_no = run_case(p0, mode=mode, return_profiles=False, use_edl=False)
    comparison = _make_edl_comparison_metrics(res_edl, res_no, mode=mode)
    return dict(with_edl=res_edl, no_edl=res_no, comparison=comparison)


def compare_edl_effects(
    base_params: Dict[str, Any],
    outputs: Tuple[str, ...] = ("E_mix", "i_mix", "phi2_vs_x", "polarization_curve"),
    save_dir: str | Path = "results/compare_case1",
    solver_settings: Optional[Dict[str, Any]] = None,
    save_data: bool = True,
    save_fig: bool = True,
) -> Dict[str, Any]:
    p0 = copy.deepcopy(base_params)
    outputs_set = {o.lower() for o in outputs}
    solver_settings = solver_settings or {}

    mode = str(solver_settings.get("mode", "FULL")).upper()
    use_affine_phi2 = bool(p0.get("use_affine_phi2", True))

    pair = run_edl_comparison_pair(p0, mode=mode)
    res_edl = pair["with_edl"]
    res_no = pair["no_edl"]
    comparison = pair["comparison"]
    out: Dict[str, Any] = dict(with_edl=res_edl, no_edl=res_no, comparison=comparison, paths={})

    prof_edl: Optional[Dict[str, Any]] = None
    prof_no: Optional[Dict[str, Any]] = None
    derived_edl: Optional[Dict[str, Any]] = None
    derived_no: Optional[Dict[str, Any]] = None

    if save_data or ("phi2_vs_x" in outputs_set):
        prof_edl, derived_edl = build_profiles_for_emix(p0, float(res_edl["E_mix"]), use_edl=True)
        prof_no, derived_no = build_profiles_for_emix(p0, float(res_no["E_mix"]), use_edl=False)

        if "phi2_vs_x" in outputs_set:
            scale_edl = float(derived_edl["R"]) * float(derived_edl["T"]) / float(derived_edl["F"])
            scale_no = float(derived_no["R"]) * float(derived_no["T"]) / float(derived_no["F"])
            out["with_edl"]["phi2_vs_x"] = dict(
                x_m=prof_edl["x_tilde"] * float(derived_edl["lambda_D"]),
                phi2_V=scale_edl * prof_edl["phi_tilde"],
            )
            out["no_edl"]["phi2_vs_x"] = dict(
                x_m=prof_no["x_tilde"] * float(derived_no["lambda_D"]),
                phi2_V=scale_no * prof_no["phi_tilde"],
            )

    curve_edl: Optional[Dict[str, np.ndarray]] = None
    curve_no: Optional[Dict[str, np.ndarray]] = None
    if "polarization_curve" in outputs_set:
        E_values = _compare_polarization_E_values(
            p0,
            solver_settings,
            E_mix_edl=float(res_edl["E_mix"]),
            E_mix_no=float(res_no["E_mix"]),
        )
        curve_edl = compute_polarization_curve(p0, mode=mode, use_edl=True, E_values=E_values, use_affine_phi2=use_affine_phi2)
        curve_no = compute_polarization_curve(p0, mode=mode, use_edl=False, E_values=E_values, use_affine_phi2=use_affine_phi2)
        out["with_edl"]["polarization_curve"] = curve_edl
        out["no_edl"]["polarization_curve"] = curve_no

    if save_data or save_fig:
        out_dir = ensure_dir(Path(save_dir))
        paths: Dict[str, str] = {"save_dir": str(out_dir)}

        if save_data:
            with_dir = ensure_dir(out_dir / "with_edl")
            no_dir = ensure_dir(out_dir / "no_edl")
            csv_dir = ensure_dir(out_dir / "csv")
            ensure_dir(with_dir / "figures")
            ensure_dir(no_dir / "figures")

            params_path = out_dir / "params.json"
            with params_path.open("w", encoding="utf-8") as f:
                json.dump(p0, f, indent=2, sort_keys=True, default=str)

            summary_path = csv_dir / "summary_compare.csv"
            df_sum = pd.DataFrame([dict(
                pH=res_edl["pH"],
                pH_ref=res_edl["pH_ref"],
                delta_pH=res_edl["delta_pH"],
                E1_eq_eff=res_edl["E1_eq_eff"],
                E2_eq_eff=res_edl["E2_eq_eff"],
                it0_1_eff=res_edl["it0_1_eff"],
                it0_2_eff=res_edl["it0_2_eff"],
                E_mix_with=res_edl["E_mix"],
                i_mix_with=res_edl["i_mix"],
                i_mix_norm_with=res_edl["i_mix_norm_A_per_m2"],
                i_mix_phys_with=res_edl["i_mix_phys_A_per_m"],
                i_mix_abs_with=res_edl["i_mix_abs_A"],
                i_mix_avg_with=res_edl["i_mix_avg_A_per_m2"],
                E_mix_no=res_no["E_mix"],
                i_mix_no=res_no["i_mix"],
                i_mix_norm_no=res_no["i_mix_norm_A_per_m2"],
                i_mix_phys_no=res_no["i_mix_phys_A_per_m"],
                i_mix_abs_no=res_no["i_mix_abs_A"],
                i_mix_avg_no=res_no["i_mix_avg_A_per_m2"],
                delta_E_mix=comparison["delta_E_mix"],
                delta_i_mix_avg_A_per_m2=comparison["delta_i_mix_avg_A_per_m2"],
                ratio_i_mix_avg=comparison["ratio_i_mix_avg"],
                pct_i_mix_avg=comparison["pct_i_mix_avg"],
                ratio_i_mix=comparison["ratio_i_mix"],
                pct_i_mix=comparison["pct_i_mix"],
                ratio_i_mix_phys=comparison["ratio_i_mix_phys"],
                pct_i_mix_phys=comparison["pct_i_mix_phys"],
                delta_i_mix_abs_A=comparison["delta_i_mix_abs_A"],
                ratio_i_mix_abs=comparison["ratio_i_mix_abs"],
                pct_i_mix_abs=comparison["pct_i_mix_abs"],
                mode=mode,
            )])
            df_sum.to_csv(summary_path, index=False)

            if prof_edl is None or prof_no is None or derived_edl is None or derived_no is None:
                prof_edl, derived_edl = build_profiles_for_emix(p0, float(res_edl["E_mix"]), use_edl=True)
                prof_no, derived_no = build_profiles_for_emix(p0, float(res_no["E_mix"]), use_edl=False)

            _save_profiles_npz(with_dir / "profiles.npz", prof_edl, derived_edl)
            _save_profiles_npz(no_dir / "profiles.npz", prof_no, derived_no)

            paths.update(
                params_json=str(params_path),
                summary_compare=str(summary_path),
                with_edl_profiles=str(with_dir / "profiles.npz"),
                no_edl_profiles=str(no_dir / "profiles.npz"),
            )

        if save_fig:
            fig_dir_main = ensure_dir(out_dir / "figures")
            fig_dir_with = ensure_dir(out_dir / "with_edl" / "figures")
            fig_dir_no = ensure_dir(out_dir / "no_edl" / "figures")
            fig_dirs = [fig_dir_main, fig_dir_with, fig_dir_no]

            tag = _case_tag_from_params(p0)
            title = "EDL compare"

            if curve_edl is None or curve_no is None:
                E_values = _compare_polarization_E_values(
                    p0,
                    solver_settings,
                    E_mix_edl=float(res_edl["E_mix"]),
                    E_mix_no=float(res_no["E_mix"]),
                )
                curve_edl = compute_polarization_curve(p0, mode=mode, use_edl=True, E_values=E_values, use_affine_phi2=use_affine_phi2)
                curve_no = compute_polarization_curve(p0, mode=mode, use_edl=False, E_values=E_values, use_affine_phi2=use_affine_phi2)

            for fig_dir in fig_dirs:
                plot_compare_polarization_curve(
                    curve_edl=curve_edl,
                    curve_no=curve_no,
                    E_mix_edl=float(res_edl["E_mix"]),
                    E_mix_no=float(res_no["E_mix"]),
                    out_path=fig_dir / f"compare_polcurve_{tag}.png",
                    title=title,
                )

                plot_compare_emix_imix(
                    E_mix_edl=float(res_edl["E_mix"]),
                    i_mix_avg_edl=float(res_edl["i_mix_avg_A_per_m2"]),
                    E_mix_no=float(res_no["E_mix"]),
                    i_mix_avg_no=float(res_no["i_mix_avg_A_per_m2"]),
                    out_path=fig_dir / f"compare_emix_imix_{tag}.png",
                    title=title,
                )

                if "phi2_vs_x" in outputs_set:
                    if prof_edl is None or prof_no is None or derived_edl is None or derived_no is None:
                        prof_edl, derived_edl = build_profiles_for_emix(p0, float(res_edl["E_mix"]), use_edl=True)
                        prof_no, derived_no = build_profiles_for_emix(p0, float(res_no["E_mix"]), use_edl=False)
                    plot_compare_phi2(
                        prof_edl=prof_edl,
                        derived_edl=derived_edl,
                        prof_no=prof_no,
                        derived_no=derived_no,
                        out_path=fig_dir / f"compare_phi2_{tag}.png",
                        title=title,
                    )
                    plot_compare_potentials_overpotential(
                        prof_edl=prof_edl,
                        derived_edl=derived_edl,
                        prof_no=prof_no,
                        derived_no=derived_no,
                        params=p0,
                        E_mix_edl=float(res_edl["E_mix"]),
                        E_mix_no=float(res_no["E_mix"]),
                        out_path=fig_dir / f"compare_potentials_overpotential_{tag}.png",
                        title=title,
                    )

            paths["figures_dir"] = str(fig_dir_main)

        out["paths"] = paths

    return out


def self_check_with_reference(
    params: Dict[str, Any],
    ref: Dict[str, float],
    mode: str = "FULL",
    atol_E: float = 1e-10,
    atol_I: float = 1e-8,
) -> Dict[str, float]:
    """Compare current with-EDL results to a saved reference (from older code)."""
    res = run_case(params, mode=mode, return_profiles=False, use_edl=True)
    dE = float(abs(res["E_mix"] - float(ref["E_mix"])))
    dI = float(abs(res["i_mix"] - float(ref["i_mix"])))
    ok = (dE <= atol_E) and (dI <= atol_I)
    print(f"self_check_with_reference: dE={dE:.6e}, dI={dI:.6e}, ok={ok}")
    return dict(dE=dE, dI=dI, ok=bool(ok))


def make_summary_row(run_tag: str, params: Dict[str, Any], result: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Row for results_summary.csv (includes full parameters + key outputs)."""
    row = dict(run=run_tag, **flatten_dict(params))
    row.update(
        mode=result.get("mode"), # pyright: ignore[reportArgumentType]
        E_mix=result.get("E_mix"), # pyright: ignore[reportArgumentType]
        i_mix=result.get("i_mix"), # pyright: ignore[reportArgumentType]
        i_mix_norm_A_per_m2=result.get("i_mix_norm_A_per_m2"), # pyright: ignore[reportArgumentType]
        i_mix_phys_A_per_m=result.get("i_mix_phys_A_per_m"), # pyright: ignore[reportArgumentType]
        i_mix_abs_A=result.get("i_mix_abs_A"), # pyright: ignore[reportArgumentType]
        i_mix_avg_A_per_m2=result.get("i_mix_avg_A_per_m2"), # pyright: ignore[reportArgumentType]
        residual=result.get("residual"), # pyright: ignore[reportArgumentType]
        residual_norm_A_per_m2=result.get("residual_norm_A_per_m2"), # pyright: ignore[reportArgumentType]
        residual_phys_A_per_m=result.get("residual_phys_A_per_m"), # pyright: ignore[reportArgumentType]
        residual_abs_A=result.get("residual_abs_A"), # pyright: ignore[reportArgumentType]
        residual_avg_A_per_m2=result.get("residual_avg_A_per_m2"), # pyright: ignore[reportArgumentType]
        converged=result.get("converged"), # pyright: ignore[reportArgumentType]
        method=result.get("method"), # pyright: ignore[reportArgumentType]
        iterations=result.get("iterations"), # pyright: ignore[reportArgumentType]
        lambda_D=result.get("lambda_D"), # pyright: ignore[reportArgumentType]
        out_of_plane_width_m=result.get("out_of_plane_width_m"), # pyright: ignore[reportArgumentType]
        reactive_length_m=result.get("reactive_length_m"), # pyright: ignore[reportArgumentType]
        reactive_area_m2=result.get("reactive_area_m2"), # pyright: ignore[reportArgumentType]
        g_Au=result.get("g_Au"), g_C=result.get("g_C"), g_Pd=result.get("g_Pd"), # pyright: ignore[reportArgumentType]
        a1=result.get("a1"), b1=result.get("b1"), a2=result.get("a2"), b2=result.get("b2"), # pyright: ignore[reportArgumentType]
        max_abs_phi_tilde=result.get("max_abs_phi_tilde"), # pyright: ignore[reportArgumentType]
        debye_huckel_ok=result.get("debye_huckel_ok"), # pyright: ignore[reportArgumentType]
        pH=result.get("pH"), # pyright: ignore[reportArgumentType]
        pH_ref=result.get("pH_ref"), # pyright: ignore[reportArgumentType]
        delta_pH=result.get("delta_pH"), # pyright: ignore[reportArgumentType]
        E1_eq_eff=result.get("E1_eq_eff"), # pyright: ignore[reportArgumentType]
        E2_eq_eff=result.get("E2_eq_eff"), # pyright: ignore[reportArgumentType]
        it0_1_eff=result.get("it0_1_eff"), # pyright: ignore[reportArgumentType]
        it0_2_eff=result.get("it0_2_eff"), # pyright: ignore[reportArgumentType]
    )
    if extra:
        row.update(extra)
    if "C_tot" in row and row["C_tot"] is not None:
        row["C_tot_M"] = concentration_mol_per_m3_to_M(float(row["C_tot"]))
    if row.get("scan_param") == "C_tot" and "scan_value" in row:
        row["scan_value_M"] = concentration_mol_per_m3_to_M(float(row["scan_value"]))
    if "phi2_1_meanV" in result:
        row["phi2_1_meanV"] = result["phi2_1_meanV"]
        row["phi2_2_meanV"] = result["phi2_2_meanV"]
    return row


# -----------------------------
# Scans and sensitivities
# -----------------------------

def make_ofat_specs(p0: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Default OFAT scan specs (editable)."""
    n = int(p0["ofat_n"])
    n_lgap = int(p0.get("ofat_L_gap_n", n))
    pH_min = float(p0.get("ofat_pH_min", 0.0))
    pH_max = float(p0.get("ofat_pH_max", 14.0))
    C_tot_min = float(p0.get("ofat_C_tot_min", concentration_M_to_mol_per_m3(1.0e-4)))
    C_tot_max = float(p0.get("ofat_C_tot_max", concentration_M_to_mol_per_m3(10.0)))
    L_gap_min = float(p0.get("ofat_L_gap_min", 0.0))
    L_gap_max = float(p0.get("ofat_L_gap_max", 1.0e-2))
    def cdl_scan_bounds(key: str) -> Tuple[float, float]:
        baseline = float(p0[key])
        # Use a fixed physically meaningful lower bound of 1 uF/cm^2 for OFAT,
        # so all Cdl scans start from the same experimental floor.
        vmin = MIN_PHYSICAL_CDL_F_PER_M2
        vmax = max(vmin * 10.0, baseline * 10.0)
        return vmin, vmax
    Cdl_Au_min, Cdl_Au_max = cdl_scan_bounds("Cdl_Au")
    Cdl_C_min, Cdl_C_max = cdl_scan_bounds("Cdl_C")
    Cdl_Pd_min, Cdl_Pd_max = cdl_scan_bounds("Cdl_Pd")
    return {
        "pH":         {"type": "linear", "min": pH_min, "max": pH_max, "n": n},
        "C_tot":      {"type": "log",    "min": C_tot_min, "max": C_tot_max, "n": n},
        "lambda_D":   {"type": "log",    "span": 10.0, "n": n},
        "epsilon_r":  {"type": "linear", "span": 20.0, "n": n},
        "T":          {"type": "linear", "span": 30.0, "n": n},
        "L_Au":       {"type": "log",    "span": 10.0, "n": n},
        "L_gap":      {"type": "linear", "min": L_gap_min, "max": L_gap_max, "n": n_lgap},
        "L_Pd_len":   {"type": "log",    "span": 10.0, "n": n},
        "Cdl_Au":     {"type": "log",    "min": Cdl_Au_min, "max": Cdl_Au_max, "n": n},
        "Cdl_C":      {"type": "log",    "min": Cdl_C_min, "max": Cdl_C_max, "n": n},
        "Cdl_Pd":     {"type": "log",    "min": Cdl_Pd_min, "max": Cdl_Pd_max, "n": n},
        "pzc_Au":     {"type": "linear", "span": 0.2,  "n": n},
        "pzc_C":      {"type": "linear", "span": 0.2,  "n": n},
        "pzc_Pd":     {"type": "linear", "span": 0.2,  "n": n},
        "it0_1":      {"type": "log",    "span": 10.0, "n": n},
        "it0_2":      {"type": "log",    "span": 10.0, "n": n},
        "alpha1":     {"type": "linear", "span": 0.2,  "n": n},
        "alpha2":     {"type": "linear", "span": 0.2,  "n": n},
        "E1_eq":      {"type": "linear", "span": 0.3,  "n": n},
        "E2_eq":      {"type": "linear", "span": 0.3,  "n": n},
        "z_R1":       {"type": "linear", "span": 1.0,  "n": n},
        "z_O2":       {"type": "linear", "span": 1.0,  "n": n},
    }


def make_scan_values(p0_val: float, spec: Dict[str, Any]) -> np.ndarray:
    kind = spec["type"]; n = int(spec["n"])
    if "min" in spec or "max" in spec:
        vmin = float(spec.get("min", p0_val))
        vmax = float(spec.get("max", p0_val))
        if vmax < vmin:
            raise ValueError("scan spec max must be >= min")
        if kind == "log":
            if vmin <= 0 or vmax <= 0:
                raise ValueError("log scan requires positive min/max")
            return np.logspace(np.log10(vmin), np.log10(vmax), n)
        return np.linspace(vmin, vmax, n)
    span = float(spec["span"])
    if kind == "log":
        if p0_val <= 0:
            raise ValueError("log scan requires positive baseline value")
        return np.logspace(np.log10(p0_val / span), np.log10(p0_val * span), n)
    if kind == "linear":
        return np.linspace(p0_val - span, p0_val + span, n)
    raise ValueError(f"Unknown scan type: {kind}")


def plot_ofat_edl_comparison_html(
    dfp_plot: pd.DataFrame,
    pname: str,
    with_col: str,
    no_col: str,
    metric_label: str,
    ylab: str,
    xscale: str,
    fig_dir: Path,
) -> None:
    if go is None:
        _maybe_warn_plotly()
        return
    if dfp_plot.empty:
        return
    x_label = _param_axis_label(pname)
    x_vals = _ofat_x_values(dfp_plot, pname)
    title_metric = _label_symbol(ylab)
    title_param = _label_symbol(x_label)

    fig = go.Figure()
    for label, column, color in [
        (LEGEND_WITH_EDL, with_col, NATURE_COLORS["blue"]),
        (LEGEND_WITHOUT_EDL, no_col, NATURE_COLORS["orange"]),
    ]:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=dfp_plot[column],
                mode="lines+markers",
                name=label,
                line=dict(width=2.2, color=color),
                marker=dict(size=6, color=color),
                text=dfp_plot["hover_params"],
                hovertemplate=f"{x_label}=%{{x:.6g}}<br>{metric_label}=%{{y:.6g}}<br>%{{text}}<extra>{label}</extra>",
            )
        )

    _style_plotly_figure(fig, rf"{title_metric} vs {title_param}", x_label, ylab)
    if xscale == "log":
        fig.update_xaxes(type="log")
    write_plotly_html(fig, fig_dir / f"ofat_compare_{pname}_{metric_label}.html")


def _plot_ofat_edl_comparison(
    dfp: pd.DataFrame,
    pname: str,
    with_col: str,
    no_col: str,
    metric_label: str,
    ylab: str,
    spec_type: str,
    fig_dir: Path,
    y_axis_key: Optional[str] = None,
) -> None:
    fig, ax = _new_figure(NATURE_WIDE_FIGSIZE)
    x_vals = _ofat_x_values(dfp, pname)
    y_with = np.asarray(dfp[with_col], dtype=float)
    y_no = np.asarray(dfp[no_col], dtype=float)
    if y_axis_key is not None:
        (y_with, y_no), ylab, _ = _scaled_current_display(y_axis_key, y_with, y_no)
    ax.plot(x_vals, y_with, marker="o", linestyle="-", label=LEGEND_WITH_EDL, color=NATURE_COLORS["blue"], markersize=4.5)
    ax.plot(x_vals, y_no, marker="o", linestyle="-", label=LEGEND_WITHOUT_EDL, color=NATURE_COLORS["orange"], markersize=4.5)
    title_metric = _label_symbol(ylab)
    title_param = _label_symbol(_param_axis_label(pname))
    _style_axes(
        ax,
        _param_axis_label(pname),
        ylab,
        rf"{title_metric} vs {title_param}",
        xscale="log" if spec_type == "log" else None,
    )
    ax.legend(loc="best")
    _finalize_figure(fig, fig_dir / f"ofat_compare_{pname}_{metric_label}.png")


def _heatmap_vmin_vmax(Z_with: np.ndarray, Z_no: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    finite_with = Z_with[np.isfinite(Z_with)]
    finite_no = Z_no[np.isfinite(Z_no)]
    if finite_with.size == 0 and finite_no.size == 0:
        return None, None
    finite = np.concatenate([finite_with, finite_no]) if finite_with.size and finite_no.size else (finite_with if finite_with.size else finite_no)
    return float(np.min(finite)), float(np.max(finite))


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray, atol: float = 1e-30) -> np.ndarray:
    ratio = np.full_like(numerator, np.nan, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > atol)
    np.divide(numerator, denominator, out=ratio, where=valid)
    return ratio


def _safe_log10_abs(values: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    valid = np.isfinite(arr) & (np.abs(arr) > floor)
    out[valid] = np.log10(np.abs(arr[valid]))
    return out


def _plot_heatmap_single(
    Z: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    xlabel: str,
    ylabel: str,
    cbarlab: str,
    title: str,
    out_path: Path,
    cmap: str = "viridis",
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    center: Optional[float] = None,
    symmetric_about_center: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    norm: Optional[Any] = None,
    cbar_ticks: Optional[np.ndarray | List[float]] = None,
    cbar_use_max_locator: bool = True,
    baseline_point: Optional[Tuple[float, float]] = None,
) -> None:
    fig = plt.figure(figsize=(5.8, 4.65))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.065], wspace=0.20)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    X, Y = np.meshgrid(x_edges, y_edges)
    finite = Z[np.isfinite(Z)]
    mesh_kwargs: Dict[str, Any] = {"shading": "auto", "cmap": cmap}

    if norm is not None:
        mesh_kwargs["norm"] = norm
    elif vmin is not None or vmax is not None:
        if vmin is not None:
            mesh_kwargs["vmin"] = float(vmin)
        if vmax is not None:
            mesh_kwargs["vmax"] = float(vmax)
    elif finite.size > 0:
        zmin = float(np.min(finite))
        zmax = float(np.max(finite))
        if center is not None and symmetric_about_center:
            span = max(abs(zmin - center), abs(zmax - center))
            if span == 0.0:
                span = max(1e-12, abs(center) * 1e-6, 1e-12)
            mesh_kwargs["vmin"] = center - span
            mesh_kwargs["vmax"] = center + span
        elif center is not None and zmin < center < zmax:
            mesh_kwargs["norm"] = TwoSlopeNorm(vmin=zmin, vcenter=center, vmax=zmax)
        elif zmin == zmax:
            eps = max(1e-12, abs(zmin) * 1e-6, 1e-12)
            mesh_kwargs["vmin"] = zmin - eps
            mesh_kwargs["vmax"] = zmax + eps
        else:
            mesh_kwargs["vmin"] = zmin
            mesh_kwargs["vmax"] = zmax

    mesh = ax.pcolormesh(X, Y, Z, **mesh_kwargs)
    _style_axes(ax, xlabel, ylabel, title, xscale=xscale, yscale=yscale)
    _style_heatmap_axes(ax)
    if xscale is None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    if yscale is None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    if baseline_point is not None:
        bx, by = baseline_point
        ax.plot(
            [bx],
            [by],
            marker="*",
            markersize=HEATMAP_BASELINE_MARKER_SIZE,
            markerfacecolor="white",
            markeredgecolor=NATURE_COLORS["black"],
            markeredgewidth=1.9,
            color=NATURE_COLORS["black"],
            linestyle="None",
            zorder=6,
        )
    cbar = fig.colorbar(mesh, cax=cax)
    _style_heatmap_colorbar(cbar, cbarlab, ticks=cbar_ticks, use_max_locator=cbar_use_max_locator)
    fig.subplots_adjust(left=0.16, right=0.94, bottom=0.18, top=0.90)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_heatmap_edl_comparison(
    Z_with: np.ndarray,
    Z_no: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    xlabel: str,
    ylabel: str,
    cbarlab: str,
    title: str,
    out_path: Path,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    baseline_point: Optional[Tuple[float, float]] = None,
) -> None:
    fig = plt.figure(figsize=(11.0, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.055], wspace=0.22)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])
    X, Y = np.meshgrid(x_edges, y_edges)
    vmin, vmax = _heatmap_vmin_vmax(Z_with, Z_no)
    mesh = None
    for idx, (ax, panel_title, Z) in enumerate(zip(
        axes,
        [LEGEND_WITH_EDL, LEGEND_WITHOUT_EDL],
        [Z_with, Z_no],
    )):
        mesh = ax.pcolormesh(X, Y, Z, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        _style_axes(ax, xlabel, ylabel if idx == 0 else "", panel_title, xscale=xscale, yscale=yscale)
        _style_heatmap_axes(ax)
        if xscale is None:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        if yscale is None:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if baseline_point is not None:
            bx, by = baseline_point
            ax.plot(
                [bx],
                [by],
                marker="*",
                markersize=HEATMAP_BASELINE_MARKER_SIZE,
                markerfacecolor="white",
                markeredgecolor=NATURE_COLORS["black"],
                markeredgewidth=1.9,
                color=NATURE_COLORS["black"],
                linestyle="None",
                zorder=6,
            )
    fig.suptitle(title, x=0.02, y=1.02, ha="left", fontsize=14.0, fontweight="semibold")
    if mesh is not None:
        cbar = fig.colorbar(mesh, cax=cax)
        _style_heatmap_colorbar(cbar, cbarlab)
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.18, top=0.82)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_heatmap_slice_panels(
    Z_panels: List[np.ndarray],
    panel_titles: List[str],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    xlabel: str,
    ylabel: str,
    cbarlab: str,
    title: str,
    out_path: Path,
    cmap: str = "viridis",
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    center: Optional[float] = None,
    symmetric_about_center: bool = False,
) -> None:
    n_panels = len(Z_panels)
    if n_panels == 0:
        return
    fig = plt.figure(figsize=(4.0 * n_panels + 1.1, 4.7))
    gs = fig.add_gridspec(1, n_panels + 1, width_ratios=[1.0] * n_panels + [0.06], wspace=0.24)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]
    cax = fig.add_subplot(gs[0, n_panels])
    X, Y = np.meshgrid(x_edges, y_edges)

    finite_blocks = [np.asarray(Z, dtype=float)[np.isfinite(Z)] for Z in Z_panels]
    finite_blocks = [blk for blk in finite_blocks if blk.size > 0]
    mesh_kwargs: Dict[str, Any] = {"shading": "auto", "cmap": cmap}
    if finite_blocks:
        finite = np.concatenate(finite_blocks)
        zmin = float(np.min(finite))
        zmax = float(np.max(finite))
        if center is not None and symmetric_about_center:
            span = max(abs(zmin - center), abs(zmax - center))
            if span == 0.0:
                span = max(1e-12, abs(center) * 1e-6, 1e-12)
            mesh_kwargs["vmin"] = center - span
            mesh_kwargs["vmax"] = center + span
        elif center is not None and zmin < center < zmax:
            mesh_kwargs["norm"] = TwoSlopeNorm(vmin=zmin, vcenter=center, vmax=zmax)
        elif zmin == zmax:
            eps = max(1e-12, abs(zmin) * 1e-6, 1e-12)
            mesh_kwargs["vmin"] = zmin - eps
            mesh_kwargs["vmax"] = zmax + eps
        else:
            mesh_kwargs["vmin"] = zmin
            mesh_kwargs["vmax"] = zmax

    mesh = None
    for idx, (ax, Z, panel_title) in enumerate(zip(axes, Z_panels, panel_titles)):
        mesh = ax.pcolormesh(X, Y, Z, **mesh_kwargs)
        _style_axes(ax, xlabel, ylabel if idx == 0 else "", panel_title, xscale=xscale, yscale=yscale)
        _style_heatmap_axes(ax)
        if xscale is None:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        if yscale is None:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    if mesh is not None:
        cbar = fig.colorbar(mesh, cax=cax)
        _style_heatmap_colorbar(cbar, cbarlab)
    fig.suptitle(title, x=0.02, y=1.02, ha="left", fontsize=14.0, fontweight="semibold")
    fig.subplots_adjust(left=0.09, right=0.95, bottom=0.18, top=0.82)
    fig.savefig(out_path)
    plt.close(fig)


def run_ofat(base_params: Dict[str, Any], out_dir: Path, summary_rows: List[Dict[str, Any]]) -> None:
    """OFAT comparison scan: with EDL vs without EDL across parameter values."""
    fig_dir = ensure_dir(out_dir / "figures")
    csv_dir = ensure_dir(out_dir / "csv")
    specs = make_ofat_specs(base_params)

    for pname, spec in specs.items():
        p0 = base_params.get(pname)
        if pname == "lambda_D" and (p0 is None):
            p0 = EDLModel(base_params).derived["lambda_D"]

        if p0 is None:
            continue

        vals = make_scan_values(float(p0), spec)
        if pname in ("alpha1", "alpha2"):
            vals = np.clip(vals, 0.01, 0.99)

        rows_param: List[Dict[str, Any]] = []

        for v in vals:
            pvar = copy.deepcopy(base_params)

            if pname == "lambda_D":
                pvar["lambda_D"] = float(v)
            else:
                pvar[pname] = float(v)
                if base_params.get("lambda_D") is None:
                    pvar["lambda_D"] = None

            pair = run_edl_comparison_pair(pvar, mode="FULL")
            res_edl = pair["with_edl"]
            res_no = pair["no_edl"]
            comp = pair["comparison"]

            row_param = dict(
                param=pname,
                value=float(v),
                pH=res_edl["pH"],
                E1_eq_eff=res_edl["E1_eq_eff"],
                E2_eq_eff=res_edl["E2_eq_eff"],
                it0_1_eff=res_edl["it0_1_eff"],
                it0_2_eff=res_edl["it0_2_eff"],
                E_mix_with_edl_FULL=res_edl["E_mix"],
                E_mix_without_edl=res_no["E_mix"],
                delta_E_mix=comp["delta_E_mix"],
                i_mix_phys_with_edl_FULL_A_per_m=res_edl["i_mix_phys_A_per_m"],
                i_mix_phys_without_edl_A_per_m=res_no["i_mix_phys_A_per_m"],
                delta_i_mix_phys_A_per_m=comp["delta_i_mix_phys_A_per_m"],
                i_mix_abs_with_edl_FULL_A=res_edl["i_mix_abs_A"],
                i_mix_abs_without_edl_A=res_no["i_mix_abs_A"],
                delta_i_mix_abs_A=comp["delta_i_mix_abs_A"],
                i_mix_avg_with_edl_FULL_A_per_m2=res_edl["i_mix_avg_A_per_m2"],
                i_mix_avg_without_edl_A_per_m2=res_no["i_mix_avg_A_per_m2"],
                delta_i_mix_avg_A_per_m2=comp["delta_i_mix_avg_A_per_m2"],
                ratio_i_mix_avg=comp["ratio_i_mix_avg"],
                pct_i_mix_avg=comp["pct_i_mix_avg"],
                ratio_i_mix_abs=comp["ratio_i_mix_abs"],
                pct_i_mix_abs=comp["pct_i_mix_abs"],
                ratio_i_mix_phys=comp["ratio_i_mix_phys"],
                pct_i_mix_phys=comp["pct_i_mix_phys"],
                max_abs_phi_tilde_with_edl=res_edl["max_abs_phi_tilde"],
                debye_huckel_ok_with_edl=res_edl["debye_huckel_ok"],
                hover_params=format_hover_params(pvar, keys=HOVER_PARAM_KEYS),
            )
            if pname == "C_tot":
                row_param["value_M"] = concentration_mol_per_m3_to_M(float(v))
            rows_param.append(row_param)
            extra_with = {"scan_param": pname, "scan_value": float(v), "use_edl_case": "with_edl_FULL"}
            extra_no = {"scan_param": pname, "scan_value": float(v), "use_edl_case": "without_edl"}
            if pname == "C_tot":
                extra_with["scan_value_M"] = concentration_mol_per_m3_to_M(float(v))
                extra_no["scan_value_M"] = concentration_mol_per_m3_to_M(float(v))
            summary_rows.append(make_summary_row(f"ofat_compare:{pname}:with_edl_FULL", pvar, res_edl, extra=extra_with))
            summary_rows.append(make_summary_row(f"ofat_compare:{pname}:without_edl", pvar, res_no, extra=extra_no))

        dfp = pd.DataFrame(rows_param)
        dfp.to_csv(csv_dir / f"ofat_compare_{pname}.csv", index=False)

        _plot_ofat_edl_comparison(
            dfp,
            pname,
            with_col="E_mix_with_edl_FULL",
            no_col="E_mix_without_edl",
            metric_label="E_mix",
            ylab=_plot_axis_label("E_mix"),
            spec_type=spec["type"],
            fig_dir=fig_dir,
        )
        _plot_ofat_edl_comparison(
            dfp,
            pname,
            with_col="i_mix_avg_with_edl_FULL_A_per_m2",
            no_col="i_mix_avg_without_edl_A_per_m2",
            metric_label="i_mix_avg_A_per_m2",
            ylab=_plot_axis_label("i_mix_avg"),
            spec_type=spec["type"],
            fig_dir=fig_dir,
            y_axis_key="i_mix_avg",
        )
        plot_ofat_edl_comparison_html(
            dfp,
            pname,
            with_col="E_mix_with_edl_FULL",
            no_col="E_mix_without_edl",
            metric_label="E_mix",
            ylab=_plot_axis_label("E_mix"),
            xscale=spec["type"],
            fig_dir=fig_dir,
        )
        plot_ofat_edl_comparison_html(
            dfp,
            pname,
            with_col="i_mix_avg_with_edl_FULL_A_per_m2",
            no_col="i_mix_avg_without_edl_A_per_m2",
            metric_label="i_mix_avg_A_per_m2",
            ylab=_plot_axis_label("i_mix_avg"),
            xscale=spec["type"],
            fig_dir=fig_dir,
        )


def run_heatmaps(base_params: Dict[str, Any], out_dir: Path, summary_rows: List[Dict[str, Any]]) -> None:
    """2D comparison heatmaps: with EDL vs without EDL."""
    fig_dir = ensure_dir(out_dir / "figures")
    csv_dir = ensure_dir(out_dir / "csv")
    nx = int(base_params["heatmap_nx"])
    ny = int(base_params["heatmap_ny"])

    def _storage_axis_name(name: str) -> str:
        if name == "C_tot":
            return "C_tot_M"
        if name in {"L_Au", "L_gap", "L_Pd_len"}:
            return f"{name}_nm"
        if name in {"Cdl_Au", "Cdl_C", "Cdl_Pd"}:
            return f"{name}_uF_per_cm2"
        return f"{name}_V" if name.startswith("pzc_") else name

    def _finite_minmax(*arrays: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        finite_blocks = []
        for arr in arrays:
            finite = np.asarray(arr, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                finite_blocks.append(finite)
        if not finite_blocks:
            return None, None
        finite = np.concatenate(finite_blocks)
        return float(np.min(finite)), float(np.max(finite))

    def _expand_range(vmin: Optional[float], vmax: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        if vmin is None or vmax is None:
            return None, None
        if math.isclose(vmin, vmax, rel_tol=0.0, abs_tol=1e-15):
            eps = max(1e-12, abs(vmin) * 1e-6, 1e-12)
            return vmin - eps, vmax + eps
        return vmin, vmax

    def _paired_compare_heatmap_data(
        *,
        tag: str,
        x_name: str,
        x_vals: np.ndarray,
        y_name: str,
        y_vals: np.ndarray,
        xscale: str,
        yscale: str,
        assign_fn: Callable[[Dict[str, Any], float, float], None],
        title_emix_with: str,
        title_emix_delta: str,
        title_imix_with: str,
        title_imix_delta: str,
    ) -> Dict[str, Any]:
        Emix_with = np.full((len(y_vals), len(x_vals)), np.nan)
        Emix_no = np.full((len(y_vals), len(x_vals)), np.nan)
        imix_avg_with = np.full((len(y_vals), len(x_vals)), np.nan)
        imix_avg_no = np.full((len(y_vals), len(x_vals)), np.nan)

        for iy, yv in enumerate(y_vals):
            for ix, xv in enumerate(x_vals):
                pvar = copy.deepcopy(base_params)
                assign_fn(pvar, float(xv), float(yv))
                pair = run_edl_comparison_pair(pvar, mode="FULL")
                res_edl = pair["with_edl"]
                res_no = pair["no_edl"]
                Emix_with[iy, ix] = res_edl["E_mix"]
                Emix_no[iy, ix] = res_no["E_mix"]
                imix_avg_with[iy, ix] = res_edl["i_mix_avg_A_per_m2"]
                imix_avg_no[iy, ix] = res_no["i_mix_avg_A_per_m2"]

                extra_common = {
                    x_name: float(xv),
                    y_name: float(yv),
                    _storage_axis_name(x_name): float(_display_param_value(x_name, float(xv))),
                    _storage_axis_name(y_name): float(_display_param_value(y_name, float(yv))),
                }
                extra_with = dict(extra_common, use_edl_case="with_edl_FULL")
                extra_no = dict(extra_common, use_edl_case="without_edl")
                summary_rows.append(make_summary_row(f"heatmap_compare:{tag}:with_edl_FULL", pvar, res_edl, extra=extra_with))
                summary_rows.append(make_summary_row(f"heatmap_compare:{tag}:without_edl", pvar, res_no, extra=extra_no))

        x_disp = _display_param_values(x_name, x_vals)
        y_disp = _display_param_values(y_name, y_vals)
        df_Emix_with = pd.DataFrame(Emix_with, index=y_disp, columns=x_disp)
        df_Emix_no = pd.DataFrame(Emix_no, index=y_disp, columns=x_disp)
        df_Emix_delta = pd.DataFrame(Emix_with - Emix_no, index=y_disp, columns=x_disp)
        df_imix_with = pd.DataFrame(imix_avg_with, index=y_disp, columns=x_disp)
        df_imix_no = pd.DataFrame(imix_avg_no, index=y_disp, columns=x_disp)
        df_delta_imix = pd.DataFrame(imix_avg_with - imix_avg_no, index=y_disp, columns=x_disp)
        df_log10_imix_with = pd.DataFrame(_safe_log10_abs(imix_avg_with), index=y_disp, columns=x_disp)
        df_log10_imix_no = pd.DataFrame(_safe_log10_abs(imix_avg_no), index=y_disp, columns=x_disp)
        imix_ratio = _safe_ratio(imix_avg_with, imix_avg_no)
        df_imix_ratio = pd.DataFrame(imix_ratio, index=y_disp, columns=x_disp)
        df_delta_log10_imix = pd.DataFrame(
            _safe_log10_abs(imix_avg_with) - _safe_log10_abs(imix_avg_no),
            index=y_disp,
            columns=x_disp,
        )
        for df in (
            df_Emix_with,
            df_Emix_no,
            df_Emix_delta,
            df_imix_with,
            df_imix_no,
            df_delta_imix,
            df_log10_imix_with,
            df_log10_imix_no,
            df_imix_ratio,
            df_delta_log10_imix,
        ):
            df.index.name = _storage_axis_name(y_name)
            df.columns.name = _storage_axis_name(x_name)

        df_Emix_with.to_csv(csv_dir / f"heatmap_compare_{tag}_Emix_with_edl_FULL.csv")
        df_Emix_no.to_csv(csv_dir / f"heatmap_compare_{tag}_Emix_without_edl.csv")
        df_Emix_delta.to_csv(csv_dir / f"heatmap_compare_{tag}_delta_Emix.csv")
        df_imix_with.to_csv(csv_dir / f"heatmap_compare_{tag}_imix_avg_with_edl_FULL.csv")
        df_imix_no.to_csv(csv_dir / f"heatmap_compare_{tag}_imix_avg_without_edl.csv")
        df_delta_imix.to_csv(csv_dir / f"heatmap_compare_{tag}_delta_i_mix_avg.csv")
        df_log10_imix_with.to_csv(csv_dir / f"heatmap_compare_{tag}_log10_imix_avg_with_edl_FULL.csv")
        df_log10_imix_no.to_csv(csv_dir / f"heatmap_compare_{tag}_log10_imix_avg_without_edl.csv")
        df_imix_ratio.to_csv(csv_dir / f"heatmap_compare_{tag}_ratio_i_mix_avg.csv")
        df_delta_log10_imix.to_csv(csv_dir / f"heatmap_compare_{tag}_delta_log10_i_mix_avg.csv")

        return {
            "tag": tag,
            "x_name": x_name,
            "y_name": y_name,
            "xscale": xscale,
            "yscale": yscale,
            "x_disp": x_disp,
            "y_disp": y_disp,
            "x_edges": make_edges(x_disp, xscale),
            "y_edges": make_edges(y_disp, yscale),
            "baseline_point": (
                _display_param_value(x_name, float(base_params[x_name])),
                _display_param_value(y_name, float(base_params[y_name])),
            ),
            "Emix_with": Emix_with,
            "Emix_no": Emix_no,
            "delta_Emix": Emix_with - Emix_no,
            "imix_avg_with": imix_avg_with,
            "imix_avg_no": imix_avg_no,
            "delta_imix_avg": imix_avg_with - imix_avg_no,
            "log10_imix_avg_with": _safe_log10_abs(imix_avg_with),
            "log10_imix_avg_no": _safe_log10_abs(imix_avg_no),
            "delta_log10_imix_avg": _safe_log10_abs(imix_avg_with) - _safe_log10_abs(imix_avg_no),
            "imix_ratio": imix_ratio,
            "title_emix_with": title_emix_with,
            "title_emix_delta": title_emix_delta,
            "title_imix_with": title_imix_with,
            "title_imix_delta": title_imix_delta,
        }

    # Only keep the new Au/Pd coupled heatmaps. Legacy support/ionic-strength
    # heatmaps are intentionally disabled.
    def _cdl_max(baseline: float) -> float:
        return max(MIN_PHYSICAL_CDL_F_PER_M2 * 10.0, float(baseline) * 10.0)

    heatmap_data: List[Dict[str, Any]] = []

    Cdl_Au_vals = np.logspace(np.log10(MIN_PHYSICAL_CDL_F_PER_M2), np.log10(_cdl_max(float(base_params["Cdl_Au"]))), nx)
    Cdl_Pd_vals = np.logspace(np.log10(MIN_PHYSICAL_CDL_F_PER_M2), np.log10(_cdl_max(float(base_params["Cdl_Pd"]))), ny)
    heatmap_data.append(_paired_compare_heatmap_data(
        tag="CdlAu_vs_CdlPd",
        x_name="Cdl_Au",
        x_vals=Cdl_Au_vals,
        y_name="Cdl_Pd",
        y_vals=Cdl_Pd_vals,
        xscale="log",
        yscale="log",
        assign_fn=lambda pvar, xv, yv: (pvar.__setitem__("Cdl_Au", xv), pvar.__setitem__("Cdl_Pd", yv)),
        title_emix_with=r"$E_{\mathrm{mix}}$",
        title_emix_delta=r"$\Delta E_{\mathrm{mix}}$",
        title_imix_with=r"$\bar{i}_{\mathrm{mix}}$",
        title_imix_delta=r"$\Delta \bar{i}_{\mathrm{mix}}$",
    ))

    L_heatmap_min = float(base_params["heatmap_L_min"])
    L_heatmap_max = float(base_params["heatmap_L_max"])
    L_Au_vals = np.logspace(np.log10(L_heatmap_min), np.log10(L_heatmap_max), nx)
    L_Pd_vals = np.logspace(np.log10(L_heatmap_min), np.log10(L_heatmap_max), ny)
    heatmap_data.append(_paired_compare_heatmap_data(
        tag="LAu_vs_LPd",
        x_name="L_Au",
        x_vals=L_Au_vals,
        y_name="L_Pd_len",
        y_vals=L_Pd_vals,
        xscale="log",
        yscale="log",
        assign_fn=lambda pvar, xv, yv: (pvar.__setitem__("L_Au", xv), pvar.__setitem__("L_Pd_len", yv)),
        title_emix_with=r"$E_{\mathrm{mix}}$",
        title_emix_delta=r"$\Delta E_{\mathrm{mix}}$",
        title_imix_with=r"$\bar{i}_{\mathrm{mix}}$",
        title_imix_delta=r"$\Delta \bar{i}_{\mathrm{mix}}$",
    ))

    pzc_Au0 = float(base_params["pzc_Au"])
    pzc_Pd0 = float(base_params["pzc_Pd"])
    pzc_Au_vals = np.linspace(pzc_Au0 - 0.2, pzc_Au0 + 0.2, nx)
    pzc_Pd_vals = np.linspace(pzc_Pd0 - 0.2, pzc_Pd0 + 0.2, ny)
    heatmap_data.append(_paired_compare_heatmap_data(
        tag="pzcAu_vs_pzcPd",
        x_name="pzc_Au",
        x_vals=pzc_Au_vals,
        y_name="pzc_Pd",
        y_vals=pzc_Pd_vals,
        xscale="linear",
        yscale="linear",
        assign_fn=lambda pvar, xv, yv: (pvar.__setitem__("pzc_Au", xv), pvar.__setitem__("pzc_Pd", yv)),
        title_emix_with=r"$E_{\mathrm{mix}}$",
        title_emix_delta=r"$\Delta E_{\mathrm{mix}}$",
        title_imix_with=r"$\bar{i}_{\mathrm{mix}}$",
        title_imix_delta=r"$\Delta \bar{i}_{\mathrm{mix}}$",
    ))

    for entry in heatmap_data:
        xscale_plot = entry["xscale"] if entry["xscale"] != "linear" else None
        yscale_plot = entry["yscale"] if entry["yscale"] != "linear" else None
        (imix_avg_with_plot,), imix_avg_label, _ = _scaled_current_display("i_mix_avg", entry["imix_avg_with"])
        (delta_imix_avg_plot,), _, delta_imix_exp = _scaled_current_display("i_mix_avg", entry["delta_imix_avg"])
        delta_imix_abs_label = _label_with_power_of_ten(r"$\Delta \bar{i}_{\mathrm{mix}}$ [A/m$^2$]", delta_imix_exp)
        for obsolete in (
            fig_dir / f"heatmap_compare_{entry['tag']}_ratio_i_mix_abs_FULL.png",
            fig_dir / f"heatmap_compare_{entry['tag']}_log10_imix_abs_with_edl_FULL.png",
            fig_dir / f"heatmap_compare_{entry['tag']}_delta_log10_i_mix_abs_FULL.png",
            fig_dir / f"heatmap_compare_{entry['tag']}_imix_abs_with_edl_FULL.png",
            fig_dir / f"heatmap_compare_{entry['tag']}_delta_i_mix_abs_FULL.png",
        ):
            if obsolete.exists():
                obsolete.unlink()

        _plot_heatmap_single(
            entry["Emix_with"],
            entry["x_edges"],
            entry["y_edges"],
            xlabel=_param_axis_label(entry["x_name"]),
            ylabel=_param_axis_label(entry["y_name"]),
            cbarlab=_plot_axis_label("E_mix"),
            title=entry["title_emix_with"],
            out_path=fig_dir / f"heatmap_compare_{entry['tag']}_Emix_with_edl_FULL.png",
            xscale=xscale_plot,
            yscale=yscale_plot,
            baseline_point=entry["baseline_point"],
        )
        _plot_heatmap_single(
            entry["delta_Emix"],
            entry["x_edges"],
            entry["y_edges"],
            xlabel=_param_axis_label(entry["x_name"]),
            ylabel=_param_axis_label(entry["y_name"]),
            cbarlab=r"$\Delta E_{\mathrm{mix}}$ [V]",
            title=entry["title_emix_delta"],
            out_path=fig_dir / f"heatmap_compare_{entry['tag']}_delta_Emix_FULL.png",
            cmap="coolwarm",
            xscale=xscale_plot,
            yscale=yscale_plot,
            center=0.0,
            symmetric_about_center=True,
            baseline_point=entry["baseline_point"],
        )
        _plot_heatmap_single(
            imix_avg_with_plot,
            entry["x_edges"],
            entry["y_edges"],
            xlabel=_param_axis_label(entry["x_name"]),
            ylabel=_param_axis_label(entry["y_name"]),
            cbarlab=imix_avg_label,
            title=entry["title_imix_with"],
            out_path=fig_dir / f"heatmap_compare_{entry['tag']}_imix_avg_with_edl_FULL.png",
            xscale=xscale_plot,
            yscale=yscale_plot,
            baseline_point=entry["baseline_point"],
        )
        _plot_heatmap_single(
            delta_imix_avg_plot,
            entry["x_edges"],
            entry["y_edges"],
            xlabel=_param_axis_label(entry["x_name"]),
            ylabel=_param_axis_label(entry["y_name"]),
            cbarlab=delta_imix_abs_label,
            title=entry["title_imix_delta"],
            out_path=fig_dir / f"heatmap_compare_{entry['tag']}_delta_i_mix_avg_FULL.png",
            cmap="coolwarm",
            xscale=xscale_plot,
            yscale=yscale_plot,
            center=0.0,
            symmetric_about_center=True,
            baseline_point=entry["baseline_point"],
        )

    return

    # --- Pair 1: C_tot vs Δpzc ---
    C_min = float(base_params.get("heatmap_C_tot_min", concentration_M_to_mol_per_m3(1.0e-4)))
    C_max = float(base_params.get("heatmap_C_tot_max", concentration_M_to_mol_per_m3(10.0)))
    C_vals = np.logspace(np.log10(C_min), np.log10(C_max), nx)
    C_vals_M = _display_param_values("C_tot", C_vals)
    delta0 = float(base_params["pzc_Au"]) - float(base_params["pzc_Pd"])
    delta_vals = np.linspace(delta0 - 0.2, delta0 + 0.2, ny)

    Emix_with = np.full((ny, nx), np.nan)
    Emix_no = np.full((ny, nx), np.nan)
    imix_abs_with = np.full((ny, nx), np.nan)
    imix_abs_no = np.full((ny, nx), np.nan)

    for iy, dlt in enumerate(delta_vals):
        for ix, C in enumerate(C_vals):
            pvar = copy.deepcopy(base_params)
            pvar["C_tot"] = float(C)
            pvar["lambda_D"] = None
            pvar["pzc_Au"] = float(pvar["pzc_Pd"]) + float(dlt)
            pair = run_edl_comparison_pair(pvar, mode="FULL")
            res_edl = pair["with_edl"]
            res_no = pair["no_edl"]
            Emix_with[iy, ix] = res_edl["E_mix"]
            Emix_no[iy, ix] = res_no["E_mix"]
            imix_abs_with[iy, ix] = res_edl["i_mix_abs_A"]
            imix_abs_no[iy, ix] = res_no["i_mix_abs_A"]
            extra_with = {"C_tot": float(C), "C_tot_M": concentration_mol_per_m3_to_M(float(C)), "delta_pzc": float(dlt), "use_edl_case": "with_edl_FULL"}
            extra_no = {"C_tot": float(C), "C_tot_M": concentration_mol_per_m3_to_M(float(C)), "delta_pzc": float(dlt), "use_edl_case": "without_edl"}
            summary_rows.append(make_summary_row("heatmap_compare:Ctot_vs_deltapzc:with_edl_FULL", pvar, res_edl, extra=extra_with))
            summary_rows.append(make_summary_row("heatmap_compare:Ctot_vs_deltapzc:without_edl", pvar, res_no, extra=extra_no))

    df_Emix_with = pd.DataFrame(Emix_with, index=delta_vals, columns=C_vals_M)
    df_Emix_no = pd.DataFrame(Emix_no, index=delta_vals, columns=C_vals_M)
    df_Emix_delta = pd.DataFrame(Emix_with - Emix_no, index=delta_vals, columns=C_vals_M)
    df_imix_with = pd.DataFrame(imix_abs_with, index=delta_vals, columns=C_vals_M)
    df_imix_no = pd.DataFrame(imix_abs_no, index=delta_vals, columns=C_vals_M)
    df_imix_delta = pd.DataFrame(imix_abs_with - imix_abs_no, index=delta_vals, columns=C_vals_M)
    imix_ratio = _safe_ratio(imix_abs_with, imix_abs_no)
    df_imix_ratio = pd.DataFrame(imix_ratio, index=delta_vals, columns=C_vals_M)
    for df in (df_Emix_with, df_Emix_no, df_Emix_delta, df_imix_with, df_imix_no, df_imix_delta, df_imix_ratio):
        df.index.name = "delta_pzc_V"
        df.columns.name = "C_tot_M"
    df_Emix_with.to_csv(out_dir / "heatmap_compare_Ctot_vs_deltapzc_Emix_with_edl_FULL.csv")
    df_Emix_no.to_csv(out_dir / "heatmap_compare_Ctot_vs_deltapzc_Emix_without_edl.csv")
    df_Emix_delta.to_csv(out_dir / "heatmap_compare_Ctot_vs_deltapzc_delta_Emix.csv")
    df_imix_with.to_csv(out_dir / "heatmap_compare_Ctot_vs_deltapzc_imix_abs_with_edl_FULL.csv")
    df_imix_no.to_csv(out_dir / "heatmap_compare_Ctot_vs_deltapzc_imix_abs_without_edl.csv")
    df_imix_delta.to_csv(out_dir / "heatmap_compare_Ctot_vs_deltapzc_delta_imix_abs.csv")
    df_imix_ratio.to_csv(out_dir / "heatmap_compare_Ctot_vs_deltapzc_ratio_i_mix_abs.csv")

    x_edges = make_edges(C_vals_M, "log")
    y_edges = make_edges(delta_vals, "linear")
    baseline_ctot_delta = (concentration_mol_per_m3_to_M(float(base_params["C_tot"])), delta0)
    (imix_abs_with_plot, imix_abs_no_plot), imix_abs_label, _ = _scaled_current_display("i_mix_abs", imix_abs_with, imix_abs_no)
    _plot_heatmap_edl_comparison(
        Emix_with,
        Emix_no,
        x_edges,
        y_edges,
        xlabel=_param_axis_label("C_tot"),
        ylabel=_delta_pzc_label(),
        cbarlab=_plot_axis_label("E_mix"),
        title=r"EDL comparison: $E_{\mathrm{mix}}(C_{\mathrm{tot}}, \Delta \mathrm{pzc})$",
        out_path=fig_dir / "heatmap_compare_Ctot_vs_deltapzc_Emix_FULL.png",
        xscale="log",
        baseline_point=baseline_ctot_delta,
    )
    _plot_heatmap_edl_comparison(
        imix_abs_with_plot,
        imix_abs_no_plot,
        x_edges,
        y_edges,
        xlabel=_param_axis_label("C_tot"),
        ylabel=_delta_pzc_label(),
        cbarlab=imix_abs_label,
        title=r"EDL comparison: $i_{\mathrm{mix}}(C_{\mathrm{tot}}, \Delta \mathrm{pzc})$",
        out_path=fig_dir / "heatmap_compare_Ctot_vs_deltapzc_imix_abs_FULL.png",
        xscale="log",
        baseline_point=baseline_ctot_delta,
    )
    _plot_heatmap_single(
        Emix_with - Emix_no,
        x_edges,
        y_edges,
        xlabel=_param_axis_label("C_tot"),
        ylabel=_delta_pzc_label(),
        cbarlab=r"$\Delta E_{\mathrm{mix}}$ [V]",
        title=r"EDL effect: $\Delta E_{\mathrm{mix}}(C_{\mathrm{tot}}, \Delta \mathrm{pzc})$",
        out_path=fig_dir / "heatmap_compare_Ctot_vs_deltapzc_delta_Emix_FULL.png",
        cmap="coolwarm",
        xscale="log",
        center=0.0,
        symmetric_about_center=True,
    )
    _plot_heatmap_single(
        imix_ratio,
        x_edges,
        y_edges,
        xlabel=_param_axis_label("C_tot"),
        ylabel=_delta_pzc_label(),
        cbarlab=r"$i_{\mathrm{mix}}(\mathrm{with\ EDL}) / i_{\mathrm{mix}}(\mathrm{without\ EDL})$ [-]",
        title=r"EDL effect: $i_{\mathrm{mix}}$ ratio$(C_{\mathrm{tot}}, \Delta \mathrm{pzc})$",
        out_path=fig_dir / "heatmap_compare_Ctot_vs_deltapzc_ratio_i_mix_abs_FULL.png",
        cmap="coolwarm",
        xscale="log",
        center=1.0,
    )

    # --- Pair 2: L_gap vs Cdl_C, sliced by support-region pzc ---
    L_gap_min = float(base_params.get("ofat_L_gap_min", 0.0))
    L_gap_max = float(base_params.get("ofat_L_gap_max", 1000e-9))
    L_gap_vals = np.linspace(L_gap_min, L_gap_max, nx)
    L_gap_vals_nm = L_gap_vals * 1e9
    Cdl_C_min = float(base_params.get("heatmap_Cdl_C_min", max(1e-6, float(base_params["Cdl_C"]) / 10.0)))
    Cdl_C_max = float(base_params.get("heatmap_Cdl_C_max", float(base_params["Cdl_C"]) * 10.0))
    Cdl_C_vals = np.logspace(np.log10(Cdl_C_min), np.log10(Cdl_C_max), ny)
    pzc_C_offsets = [float(v) for v in cast(List[Any], base_params.get("heatmap_pzc_C_offsets", [-0.10, 0.0, 0.10]))]
    pzc_C_base = float(base_params["pzc_C"])
    pzc_C_vals = [pzc_C_base + offset for offset in pzc_C_offsets]

    Emix2_with_panels: List[np.ndarray] = []
    Emix2_no_panels: List[np.ndarray] = []
    delta_Emix2_panels: List[np.ndarray] = []
    ratio_imix2_panels: List[np.ndarray] = []
    panel_titles = [rf"$\mathrm{{pzc}}_{{\mathrm{{support}}}} = {val:.3f}\ \mathrm{{V}}$" for val in pzc_C_vals]

    for pzc_C_val in pzc_C_vals:
        Emix2_with = np.full((ny, nx), np.nan)
        Emix2_no = np.full((ny, nx), np.nan)
        imix2_abs_with = np.full((ny, nx), np.nan)
        imix2_abs_no = np.full((ny, nx), np.nan)

        for iy, Cdl_C in enumerate(Cdl_C_vals):
            for ix, L_gap in enumerate(L_gap_vals):
                pvar = copy.deepcopy(base_params)
                pvar["L_gap"] = float(L_gap)
                pvar["Cdl_C"] = float(Cdl_C)
                pvar["pzc_C"] = float(pzc_C_val)
                pair = run_edl_comparison_pair(pvar, mode="FULL")
                res_edl = pair["with_edl"]
                res_no = pair["no_edl"]
                Emix2_with[iy, ix] = res_edl["E_mix"]
                Emix2_no[iy, ix] = res_no["E_mix"]
                imix2_abs_with[iy, ix] = res_edl["i_mix_abs_A"]
                imix2_abs_no[iy, ix] = res_no["i_mix_abs_A"]
                extra_with = {
                    "L_gap": float(L_gap),
                    "L_gap_nm": float(L_gap) * 1e9,
                    "Cdl_C": float(Cdl_C),
                    "pzc_C": float(pzc_C_val),
                    "use_edl_case": "with_edl_FULL",
                }
                extra_no = {
                    "L_gap": float(L_gap),
                    "L_gap_nm": float(L_gap) * 1e9,
                    "Cdl_C": float(Cdl_C),
                    "pzc_C": float(pzc_C_val),
                    "use_edl_case": "without_edl",
                }
                summary_rows.append(make_summary_row("heatmap_compare:Lgap_vs_Cdlgap:with_edl_FULL", pvar, res_edl, extra=extra_with))
                summary_rows.append(make_summary_row("heatmap_compare:Lgap_vs_Cdlgap:without_edl", pvar, res_no, extra=extra_no))

        tag = _format_signed_float_tag(pzc_C_val)
        df2_Emix_with = pd.DataFrame(Emix2_with, index=Cdl_C_vals, columns=L_gap_vals_nm)
        df2_Emix_no = pd.DataFrame(Emix2_no, index=Cdl_C_vals, columns=L_gap_vals_nm)
        df2_Emix_delta = pd.DataFrame(Emix2_with - Emix2_no, index=Cdl_C_vals, columns=L_gap_vals_nm)
        df2_imix_with = pd.DataFrame(imix2_abs_with, index=Cdl_C_vals, columns=L_gap_vals_nm)
        df2_imix_no = pd.DataFrame(imix2_abs_no, index=Cdl_C_vals, columns=L_gap_vals_nm)
        df2_imix_ratio = pd.DataFrame(_safe_ratio(imix2_abs_with, imix2_abs_no), index=Cdl_C_vals, columns=L_gap_vals_nm)
        for df in (df2_Emix_with, df2_Emix_no, df2_Emix_delta, df2_imix_with, df2_imix_no, df2_imix_ratio):
            df.index.name = "Cdl_support_F_per_m2"
            df.columns.name = "L_gap_nm"
        df2_Emix_with.to_csv(out_dir / f"heatmap_compare_Lgap_vs_Cdlgap_pzcgap_{tag}_Emix_with_edl_FULL.csv")
        df2_Emix_no.to_csv(out_dir / f"heatmap_compare_Lgap_vs_Cdlgap_pzcgap_{tag}_Emix_without_edl.csv")
        df2_Emix_delta.to_csv(out_dir / f"heatmap_compare_Lgap_vs_Cdlgap_pzcgap_{tag}_delta_Emix.csv")
        df2_imix_with.to_csv(out_dir / f"heatmap_compare_Lgap_vs_Cdlgap_pzcgap_{tag}_imix_abs_with_edl_FULL.csv")
        df2_imix_no.to_csv(out_dir / f"heatmap_compare_Lgap_vs_Cdlgap_pzcgap_{tag}_imix_abs_without_edl.csv")
        df2_imix_ratio.to_csv(out_dir / f"heatmap_compare_Lgap_vs_Cdlgap_pzcgap_{tag}_ratio_i_mix_abs.csv")

        Emix2_with_panels.append(Emix2_with)
        Emix2_no_panels.append(Emix2_no)
        delta_Emix2_panels.append(Emix2_with - Emix2_no)
        ratio_imix2_panels.append(_safe_ratio(imix2_abs_with, imix2_abs_no))

    x_edges = make_edges(L_gap_vals_nm, "linear")
    y_edges = make_edges(Cdl_C_vals, "log")
    _plot_heatmap_slice_panels(
        Emix2_with_panels,
        panel_titles,
        x_edges,
        y_edges,
        xlabel=r"$L_{\mathrm{support}}$ [nm]",
        ylabel=_param_axis_label("Cdl_C"),
        cbarlab=_plot_axis_label("E_mix"),
        title=r"EDL comparison: $E_{\mathrm{mix}}(L_{\mathrm{support}}, C_{\mathrm{dl,support}})$ at sliced $\mathrm{pzc}_{\mathrm{support}}$",
        out_path=fig_dir / "heatmap_compare_Lgap_vs_Cdlgap_Emix_with_edl_pzcgap_slices.png",
        yscale="log",
    )
    _plot_heatmap_slice_panels(
        Emix2_no_panels,
        panel_titles,
        x_edges,
        y_edges,
        xlabel=r"$L_{\mathrm{support}}$ [nm]",
        ylabel=_param_axis_label("Cdl_C"),
        cbarlab=_plot_axis_label("E_mix"),
        title=r"Reference comparison: $E_{\mathrm{mix}}(L_{\mathrm{support}}, C_{\mathrm{dl,support}})$ at sliced $\mathrm{pzc}_{\mathrm{support}}$",
        out_path=fig_dir / "heatmap_compare_Lgap_vs_Cdlgap_Emix_without_edl_pzcgap_slices.png",
        yscale="log",
    )
    _plot_heatmap_slice_panels(
        delta_Emix2_panels,
        panel_titles,
        x_edges,
        y_edges,
        xlabel=r"$L_{\mathrm{support}}$ [nm]",
        ylabel=_param_axis_label("Cdl_C"),
        cbarlab=r"$\Delta E_{\mathrm{mix}}$ [V]",
        title=r"EDL effect: $\Delta E_{\mathrm{mix}}(L_{\mathrm{support}}, C_{\mathrm{dl,support}})$ at sliced $\mathrm{pzc}_{\mathrm{support}}$",
        out_path=fig_dir / "heatmap_compare_Lgap_vs_Cdlgap_delta_Emix_pzcgap_slices.png",
        cmap="coolwarm",
        yscale="log",
        center=0.0,
        symmetric_about_center=True,
    )
    _plot_heatmap_slice_panels(
        ratio_imix2_panels,
        panel_titles,
        x_edges,
        y_edges,
        xlabel=r"$L_{\mathrm{support}}$ [nm]",
        ylabel=_param_axis_label("Cdl_C"),
        cbarlab=r"$i_{\mathrm{mix}}(\mathrm{with\ EDL}) / i_{\mathrm{mix}}(\mathrm{without\ EDL})$ [-]",
        title=r"EDL effect: $i_{\mathrm{mix}}$ ratio$(L_{\mathrm{support}}, C_{\mathrm{dl,support}})$ at sliced $\mathrm{pzc}_{\mathrm{support}}$",
        out_path=fig_dir / "heatmap_compare_Lgap_vs_Cdlgap_ratio_i_mix_abs_pzcgap_slices.png",
        cmap="coolwarm",
        yscale="log",
        center=1.0,
    )

    # --- Pair 3-5: Au vs Pd coupled sweeps ---
    def _cdl_max(baseline: float) -> float:
        return max(MIN_PHYSICAL_CDL_F_PER_M2 * 10.0, float(baseline) * 10.0)

    Cdl_Au_vals = np.logspace(np.log10(MIN_PHYSICAL_CDL_F_PER_M2), np.log10(_cdl_max(float(base_params["Cdl_Au"]))), nx)
    Cdl_Pd_vals = np.logspace(np.log10(MIN_PHYSICAL_CDL_F_PER_M2), np.log10(_cdl_max(float(base_params["Cdl_Pd"]))), ny)
    _paired_compare_heatmap(
        tag="CdlAu_vs_CdlPd",
        x_name="Cdl_Au",
        x_vals=Cdl_Au_vals,
        y_name="Cdl_Pd",
        y_vals=Cdl_Pd_vals,
        xscale="log",
        yscale="log",
        assign_fn=lambda pvar, xv, yv: (pvar.__setitem__("Cdl_Au", xv), pvar.__setitem__("Cdl_Pd", yv)),
        title_emix=r"EDL comparison: $E_{\mathrm{mix}}(C_{\mathrm{dl,Au}}, C_{\mathrm{dl,Pd}})$",
        title_imix=r"EDL comparison: $i_{\mathrm{mix}}(C_{\mathrm{dl,Au}}, C_{\mathrm{dl,Pd}})$",
    )

    L_Au_vals = np.linspace(max(1e-9, 0.5 * float(base_params["L_Au"])), 3.0 * float(base_params["L_Au"]), nx)
    L_Pd_vals = np.linspace(max(1e-9, 0.5 * float(base_params["L_Pd_len"])), 3.0 * float(base_params["L_Pd_len"]), ny)
    _paired_compare_heatmap(
        tag="LAu_vs_LPd",
        x_name="L_Au",
        x_vals=L_Au_vals,
        y_name="L_Pd_len",
        y_vals=L_Pd_vals,
        xscale="linear",
        yscale="linear",
        assign_fn=lambda pvar, xv, yv: (pvar.__setitem__("L_Au", xv), pvar.__setitem__("L_Pd_len", yv)),
        title_emix=r"EDL comparison: $E_{\mathrm{mix}}(L_{\mathrm{Au}}, L_{\mathrm{Pd}})$",
        title_imix=r"EDL comparison: $i_{\mathrm{mix}}(L_{\mathrm{Au}}, L_{\mathrm{Pd}})$",
    )

    pzc_Au0 = float(base_params["pzc_Au"])
    pzc_Pd0 = float(base_params["pzc_Pd"])
    pzc_Au_vals = np.linspace(pzc_Au0 - 0.2, pzc_Au0 + 0.2, nx)
    pzc_Pd_vals = np.linspace(pzc_Pd0 - 0.2, pzc_Pd0 + 0.2, ny)
    _paired_compare_heatmap(
        tag="pzcAu_vs_pzcPd",
        x_name="pzc_Au",
        x_vals=pzc_Au_vals,
        y_name="pzc_Pd",
        y_vals=pzc_Pd_vals,
        xscale="linear",
        yscale="linear",
        assign_fn=lambda pvar, xv, yv: (pvar.__setitem__("pzc_Au", xv), pvar.__setitem__("pzc_Pd", yv)),
        title_emix=r"EDL comparison: $E_{\mathrm{mix}}(\mathrm{pzc}_{\mathrm{Au}}, \mathrm{pzc}_{\mathrm{Pd}})$",
        title_imix=r"EDL comparison: $i_{\mathrm{mix}}(\mathrm{pzc}_{\mathrm{Au}}, \mathrm{pzc}_{\mathrm{Pd}})$",
    )


def compute_sensitivities(base_params: Dict[str, Any], out_dir: Path, mode: str, summary_rows: List[Dict[str, Any]], rel_step: float = 0.01) -> pd.DataFrame:
    """
    Normalized sensitivities:
        S_p^E = ∂ln(|E_mix|)/∂ln(p)
        S_p^I_norm = ∂ln(i_mix_norm)/∂ln(p)
        S_p^I_abs = ∂ln(i_mix_abs)/∂ln(p)
    Also appends the +/- perturbed runs to results_summary rows.
    """
    mode = mode.upper()
    specs = make_ofat_specs(base_params)

    base_res = run_case(base_params, mode=mode, return_profiles=False)
    E0 = float(base_res["E_mix"])
    I0_norm = float(base_res["i_mix_norm_A_per_m2"])
    I0_abs = float(base_res["i_mix_abs_A"])

    rows: List[Dict[str, Any]] = []

    for pname in specs.keys():
        p0 = base_params.get(pname)
        if pname == "lambda_D" and (p0 is None):
            p0 = EDLModel(base_params).derived["lambda_D"]
        if p0 is None:
            continue
        p0 = float(p0)
        if p0 == 0.0:
            continue

        p_plus = p0 * (1.0 + rel_step)
        p_minus = p0 * (1.0 - rel_step)
        if pname in ("alpha1", "alpha2"):
            p_plus = float(np.clip(p_plus, 0.01, 0.99))
            p_minus = float(np.clip(p_minus, 0.01, 0.99))

        def eval_case(tag: str, pval: float) -> Tuple[float, float, float]:
            pvar = copy.deepcopy(base_params)
            if pname == "lambda_D":
                pvar["lambda_D"] = float(pval)
            else:
                pvar[pname] = float(pval)
                if base_params.get("lambda_D") is None:
                    pvar["lambda_D"] = None
            res = run_case(pvar, mode=mode, return_profiles=False)
            summary_rows.append(make_summary_row(run_tag=tag, params=pvar, result=res, extra={"sens_param": pname}))
            return float(res["E_mix"]), float(res["i_mix_norm_A_per_m2"]), float(res["i_mix_abs_A"])

        E_plus, I_plus_norm, I_plus_abs = eval_case(f"sens:+:{pname}", p_plus)
        E_minus, I_minus_norm, I_minus_abs = eval_case(f"sens:-:{pname}", p_minus)

        def ln_abs(x: float) -> float:
            if not np.isfinite(x) or x == 0.0:
                return float("nan")
            return math.log(abs(x))

        denom = math.log(abs(p_plus)) - math.log(abs(p_minus))
        S_E = (ln_abs(E_plus) - ln_abs(E_minus)) / denom if denom != 0 else float("nan")
        S_I_norm = (math.log(I_plus_norm) - math.log(I_minus_norm)) / denom if (I_plus_norm > 0 and I_minus_norm > 0 and denom != 0) else float("nan")
        S_I_abs = (math.log(I_plus_abs) - math.log(I_minus_abs)) / denom if (I_plus_abs > 0 and I_minus_abs > 0 and denom != 0) else float("nan")

        rows.append(
            dict(
                param=pname,
                p0=p0,
                E_mix_0=E0,
                i_mix_norm_0=I0_norm,
                i_mix_abs_0=I0_abs,
                S_E=S_E,
                S_I_norm=S_I_norm,
                S_I_abs=S_I_abs,
            )
        )

    df = pd.DataFrame(rows)
    df["abs_S_E"] = df["S_E"].abs()
    df["abs_S_I_norm"] = df["S_I_norm"].abs()
    df["abs_S_I_abs"] = df["S_I_abs"].abs()
    df = df.sort_values(by=["abs_S_E", "abs_S_I_abs", "abs_S_I_norm"], ascending=False)

    csv_dir = ensure_dir(out_dir / "csv")
    df.to_csv(csv_dir / "sensitivities.csv", index=False)

    fig_dir = ensure_dir(out_dir / "figures")
    for col in ("S_E", "S_I_abs"):
        fig, ax = _new_figure((6.8, 3.0))
        sub = df.head(20)
        colors = [NATURE_COLORS["blue"] if val >= 0 else NATURE_COLORS["orange"] for val in sub[col].fillna(0.0)]
        ax.bar(sub["param"], sub[col], color=colors, edgecolor=NATURE_COLORS["black"])
        _style_axes(ax, "", _plot_axis_label(col), f"Top-20 normalized sensitivities ({col}, mode={mode})")
        ax.tick_params(axis="x", rotation=60)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        _finalize_figure(fig, fig_dir / f"sensitivity_{col}_{mode}.png")

    return df


# -----------------------------
# Self-checks (optional)
# -----------------------------

def run_self_checks(
    params: Optional[Dict[str, Any]] = None,
    E_test: float = 0.35,
    rel_tol: float = 1e-8,
    bc_tol: float = 1e-12,
    robin_tol: float = 1e-10,
    phi2_tol: float = 1e-6,
    farfield_ratio: float = 1e-3,
    tail_tol: float = 1e-2,
    dh_warn_threshold: float = 1.0,
    run_convergence: bool = False,
    raise_on_fail: bool = False,
    print_summary: bool = True,
) -> Dict[str, Any]:
    """
    Run solver-consistency checks for the EDL model.

    The checks below are designed around the spectral/Galerkin formulation used
    in this solver. Pointwise BC checks near material boundaries are intentionally
    avoided because Gibbs oscillations make them a poor regression criterion for
    a truncated cosine expansion.
    """
    p = default_params() if params is None else copy.deepcopy(params)
    results: Dict[str, Any] = {}

    def _record(name: str, criterion_met: bool, value: Any, warn_only: bool = False) -> None:
        warn = bool(warn_only and not criterion_met)
        results[name] = dict(ok=bool(criterion_met or warn_only), criterion_met=bool(criterion_met), warn=warn, value=value)
        if not criterion_met and not warn_only and raise_on_fail:
            raise AssertionError(f"Self-check failed: {name} -> {value}")

    # Test 1 & 2: mixed potential consistency (FULL mode)
    case = run_case(p, mode="FULL", return_profiles=True, use_edl=True)
    I_Au = float(case["I_Au"]); I_Pd = float(case["I_Pd"])
    resid = I_Au + I_Pd
    scale = abs(I_Au) + abs(I_Pd) + 1e-30
    rel_resid = abs(resid) / scale
    _record("mixed_potential_zero_current", rel_resid < rel_tol, rel_resid)
    _record("internal_current_balance",
            (abs(resid) < rel_tol * scale)
            and (abs(abs(I_Au) - abs(I_Pd)) < rel_tol * scale)
            and (float(case["i_mix"]) >= 0.0),
            dict(residual=resid, I_Au=I_Au, I_Pd=I_Pd, i_mix=float(case["i_mix"])))

    # Test 3: sidewall Neumann condition from the spectral derivative
    edl = EDLModel(p)
    x, phi = edl.phi_tilde_surface(E_test)
    dx = x[1] - x[0]
    dphi_left_fd = (phi[1] - phi[0]) / dx
    dphi_right_fd = (phi[-1] - phi[-2]) / dx

    beta = edl.derived["beta"]
    phiM_t = beta * E_test
    A = edl.pre["A_M"] * phiM_t - edl.pre["A_pzc"]
    rho = edl.pre["rho"]
    gamma = edl.pre["gamma"]
    L_tilde = float(edl.derived["L_tilde"])

    dphi_dx_left = float(-np.dot(A * rho, np.sin(rho * 0.0)))
    dphi_dx_right = float(-np.dot(A * rho, np.sin(rho * L_tilde)))
    _record(
        "sidewall_neumann_spectral",
        (abs(dphi_dx_left) < bc_tol) and (abs(dphi_dx_right) < bc_tol),
        dict(
            dphi_dx_left=float(dphi_dx_left),
            dphi_dx_right=float(dphi_dx_right),
            dphi_dx_left_fd=float(dphi_left_fd),
            dphi_dx_right_fd=float(dphi_right_fd),
        ),
    )

    # Test 4: Robin BC in the projected/Galerkin sense used by the solver
    projected_resid = edl.pre["M"] @ A - (edl.pre["rM"] * phiM_t - edl.pre["r_pzc"])
    projected_rel = float(np.linalg.norm(projected_resid) / (np.linalg.norm(edl.pre["rM"] * phiM_t - edl.pre["r_pzc"]) + 1e-30))
    projected_inf = float(np.max(np.abs(projected_resid)))
    _record(
        "robin_bc_projected_residual",
        projected_rel < robin_tol,
        dict(relative_l2=projected_rel, absolute_inf=projected_inf),
    )

    # Test 5: spectral tail should be small if N_modes is adequate
    tail_n = min(10, max(3, len(A) // 8))
    tail_ratio = float(np.max(np.abs(A[-tail_n:])) / (np.max(np.abs(A)) + 1e-30))
    tail_l2 = float(np.linalg.norm(A[-tail_n:]) / (np.linalg.norm(A) + 1e-30))
    _record(
        "spectral_tail_decay",
        tail_ratio < tail_tol,
        dict(tail_n=tail_n, tail_ratio=tail_ratio, tail_l2_ratio=tail_l2),
    )

    # Test 6: far-field decay (y~ -> infinity)
    cos_mat = np.cos(np.outer(x, rho))
    def phi_y(y: float) -> np.ndarray:
        return cos_mat @ (A * np.exp(-gamma * y))

    m0 = float(np.max(np.abs(phi_y(0.0))))
    m10 = float(np.max(np.abs(phi_y(10.0))))
    _record("far_field_decay", m10 < farfield_ratio * m0, dict(m0=m0, m10=m10))

    # Test 7: affine vs direct segment-mean phi2
    phi2_aff = edl.segment_mean_phi2(E_test, use_affine_phi2=True)
    phi2_dir = edl.segment_mean_phi2(E_test, use_affine_phi2=False)
    diff = max(abs(phi2_aff[0] - phi2_dir[0]), abs(phi2_aff[1] - phi2_dir[1]))
    _record("phi2_affine_vs_direct", diff < phi2_tol, diff)

    # Test 8: Debye–Hückel linearization (warning only)
    max_phi = float(np.max(np.abs(case["phi_tilde"])))
    ok_linear = max_phi < dh_warn_threshold
    if not ok_linear:
        print(
            f"WARNING: |phi_tilde| exceeded the recommended Debye-Huckel range "
            f"(max={max_phi:.6g}, threshold={dh_warn_threshold:.6g})."
        )
    _record(
        "debye_huckel_linearization",
        ok_linear,
        dict(max_abs_phi_tilde=max_phi, recommended_threshold=float(dh_warn_threshold)),
        warn_only=True,
    )

    # Test 9: grid/mode convergence (optional, no assert)
    if run_convergence:
        conv_rows: List[Dict[str, Any]] = []
        Nx = max(int(p.get("Nx", 1200)), 2000)
        for Nm in [40, 80, 160]:
            pvar = copy.deepcopy(p)
            pvar["N_modes"] = Nm
            pvar["Nx"] = Nx
            res = run_case(pvar, mode="FULL", return_profiles=False, use_edl=True)
            conv_rows.append(dict(N_modes=Nm, Nx=Nx, E_mix=float(res["E_mix"]), i_mix=float(res["i_mix"])))
        results["convergence_scan"] = conv_rows

    if print_summary:
        print("\n=== Self-check summary ===")
        for name, item in results.items():
            if name == "convergence_scan":
                print("convergence_scan:", item)
                continue
            status = "warn" if item.get("warn", False) else ("ok" if item["ok"] else "fail")
            print(f"{name}: status={status}, criterion_met={item.get('criterion_met')}, value={item['value']}")

    return results


# -----------------------------
# Full workflow helper
# -----------------------------

def run_full_workflow(params: Dict[str, Any], out_dir: str | Path, print_summary: bool = True) -> Dict[str, Any]:
    """
    Run the same workflow as main(), but with user-provided params and output folder.
    This is used by run_cases.py to batch-run multiple parameter sets.
    """
    p = copy.deepcopy(params)
    out_dir = ensure_dir(Path(out_dir))
    ensure_dir(out_dir / "figures")
    csv_dir = ensure_dir(out_dir / "csv")

    summary_rows: List[Dict[str, Any]] = []

    # Baseline with EDL (FULL numerical solution)
    case_full = run_case(p, mode="FULL", return_profiles=True, use_edl=True)
    baseline_compare = compare_edl_effects(
        p,
        save_dir=out_dir,
        solver_settings={"mode": "FULL"},
        save_data=True,
        save_fig=True,
    )
    case_no_edl = baseline_compare["no_edl"]

    summary_rows.append(make_summary_row("baseline:with_edl_FULL", p, case_full, extra={"use_edl_case": "with_edl_FULL"}))
    summary_rows.append(make_summary_row("baseline:without_edl", p, case_no_edl, extra={"use_edl_case": "without_edl"}))

    # Save baseline local profiles for the with-EDL FULL case
    R_gas = float(p["R"]); F = float(p["F"]); T = float(p["T"])
    scale = R_gas * T / F
    x_tilde = case_full["x_tilde"]
    x_m = x_tilde * case_full["lambda_D"]
    phi2_V = scale * case_full["phi_tilde"]

    np.savez(
        out_dir / "profiles.npz",
        x_tilde=x_tilde,
        x_m=x_m,
        phi_tilde=case_full["phi_tilde"],
        phi2_V=phi2_V,
        i1=case_full["i1"],
        i2=case_full["i2"],
        mask_Au=case_full["mask_Au"],
        mask_Pd=case_full["mask_Pd"],
        L_Au_tilde=case_full["L_Au_tilde"],
        L_C_tilde=case_full["L_C_tilde"],
        lambda_D=case_full["lambda_D"],
    )

    plot_baseline_profiles(case_full, p, out_dir)

    # Print comparison (optional)
    if print_summary:
        df_cmp = pd.DataFrame([
            dict(
                scenario="with_edl_FULL",
                E_mix=case_full["E_mix"],
                i_mix_abs_A=case_full["i_mix_abs_A"],
                residual=case_full["residual"],
                method=case_full["method"],
                max_abs_phi_tilde=case_full["max_abs_phi_tilde"],
                debye_huckel_ok=case_full["debye_huckel_ok"],
            ),
            dict(
                scenario="without_edl",
                E_mix=case_no_edl["E_mix"],
                i_mix_abs_A=case_no_edl["i_mix_abs_A"],
                residual=case_no_edl["residual"],
                method=case_no_edl["method"],
                max_abs_phi_tilde=case_no_edl["max_abs_phi_tilde"],
                debye_huckel_ok=case_no_edl["debye_huckel_ok"],
            ),
        ])
        print("\n=== Baseline comparison: with EDL vs without EDL ===")
        print(df_cmp.to_string(index=False))
        print("\nComparison summary:")
        print(pd.DataFrame([baseline_compare["comparison"]]).to_string(index=False))

    # Optional self-checks
    if bool(p.get("do_self_checks", False)):
        run_self_checks(
            p,
            run_convergence=bool(p.get("do_convergence_check", False)),
            raise_on_fail=False,
            print_summary=True,
        )

    # OFAT
    if bool(p.get("do_ofat", True)):
        run_ofat(p, out_dir=out_dir, summary_rows=summary_rows)

    # Heatmaps
    if bool(p.get("do_heatmaps", True)):
        run_heatmaps(p, out_dir=out_dir, summary_rows=summary_rows)

    # Sensitivities
    if bool(p.get("do_sensitivities", True)):
        compute_sensitivities(p, out_dir=out_dir, mode="FULL", summary_rows=summary_rows, rel_step=0.01)

    # Save master summary (required)
    pd.DataFrame(summary_rows).to_csv(csv_dir / "results_summary.csv", index=False)

    return dict(out_dir=str(out_dir), case_full=case_full, case_no_edl=case_no_edl, baseline_compare=baseline_compare)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    params = default_params()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = SCRIPT_DIR / "results" / timestamp
    run_full_workflow(params, out_dir=out_dir, print_summary=True)
    print(f"\nAll results saved under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
