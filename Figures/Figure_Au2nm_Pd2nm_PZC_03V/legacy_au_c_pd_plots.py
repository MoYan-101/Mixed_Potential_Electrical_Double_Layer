"""Publication plots for the legacy Au | support | Pd cases.

The numerical driver owns the case data.  This module deliberately accepts a
duck-typed case object (or mapping) so plotting stays independent of the
solver/integration grid implementation.  See :func:`plot_all_cases` for the
minimum public entry point.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator


DEFAULT_OUTPUT_TAG = "au2_pd2_20260528_111255"
ACTIVE_ZOOM_SUPPORT_NM = 1000.0
ACTIVE_WINDOW_NM = 20.0
CURRENT_EXPONENT = -3

COLORS = {
    "with": "#F26B38",
    "with_alt": "#D83A2E",
    "without": "#12355B",
    "without_alt": "#2D5A7B",
    "current_Au": "#009E73",
    "current_Pd": "#0072B2",
    "dark": "#272727",
    "gray": "#767676",
    "light_gray": "#CFCECE",
    "Au": "#E4C133",
    "support": "#8C8C8C",
    "Pd": "#5A90C8",
}

_MISSING = object()


@dataclass(frozen=True)
class PlotRanges:
    """Shared ranges used for direct comparison across support lengths."""

    phi_rp_V: tuple[float, float]
    current_scaled_A_per_m2: tuple[float, float]
    sigma_uC_per_cm2: tuple[float, float]
    phi_s_vlim_mV: float
    current_exponent: int = CURRENT_EXPONENT


@dataclass(frozen=True)
class _Geometry:
    L_Au_nm: float
    L_support_nm: float
    L_Pd_nm: float
    L_total_nm: float
    au_support_boundary_nm: float
    support_pd_boundary_nm: float


@dataclass(frozen=True)
class _SigmaSegment:
    material: str
    x_nm: np.ndarray
    sigma_uC_per_cm2: np.ndarray


def configure_style() -> None:
    """Apply the project Helvetica-first, editable-SVG plotting style."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 9.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.9,
            "axes.grid": False,
            "figure.facecolor": "none",
            "axes.facecolor": "none",
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


def _read_one(source: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(source, Mapping) and name in source:
        return source[name]
    if hasattr(source, name):
        return getattr(source, name)
    if default is not _MISSING:
        return default
    raise AttributeError(name)


def _find(
    case: Any,
    names: Sequence[str],
    *,
    nested: Sequence[str] = (),
    default: Any = _MISSING,
) -> Any:
    sources = [case]
    for parent_name in nested:
        parent = _read_one(case, parent_name, None)
        if parent is not None:
            sources.append(parent)
    for source in sources:
        for name in names:
            try:
                return _read_one(source, name)
            except AttributeError:
                continue
    if default is not _MISSING:
        return default
    joined = ", ".join(names)
    raise AttributeError(f"Case does not provide any of: {joined}")


def _array(values: Any, label: str, *, ndim: int = 1) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != ndim:
        raise ValueError(f"{label} must be {ndim}D; got shape {result.shape}")
    if result.size == 0:
        raise ValueError(f"{label} is empty")
    return result


def _line_x(case: Any) -> np.ndarray:
    return _array(_find(case, ("x_nm", "profile_x_nm"), nested=("panel_data",)), "x_nm")


def _phi_rp(case: Any, with_edl: bool) -> np.ndarray:
    if with_edl:
        names = ("phi_rp_with_V", "phi_rp_edl_V", "phi_rp_edl", "phi_rp_with")
        label = "phi_rp_with_V"
    else:
        names = ("phi_rp_no_V", "phi_rp_without_V", "phi_rp_no", "phi_rp_without")
        label = "phi_rp_no_V"
    result = _array(_find(case, names, nested=("panel_data",)), label)
    _validate_profile_length(case, result, label)
    return result


def _mask(case: Any, reaction: int, with_edl: bool) -> np.ndarray | None:
    material = "Au" if reaction == 1 else "Pd"
    suffixes = ("with", "edl") if with_edl else ("no", "without", "no_edl")
    names = [f"mask_{material}_{suffix}" for suffix in suffixes]
    names.extend((f"mask_{material}", f"mask_{material.lower()}"))
    raw = _find(case, tuple(names), nested=("panel_data", "prof_edl", "prof_no"), default=None)
    if raw is None:
        return None
    mask = np.asarray(raw, dtype=bool)
    if mask.shape != _line_x(case).shape:
        raise ValueError(f"{material} mask shape {mask.shape} does not match x grid")
    return mask


def _current(case: Any, reaction: int, with_edl: bool) -> np.ndarray:
    prefix = f"i{reaction}"
    if with_edl:
        names = (
            f"{prefix}_with_A_per_m2",
            f"{prefix}_with",
            f"{prefix}_edl_segment",
            f"{prefix}_edl",
        )
        label = f"{prefix}_with"
    else:
        names = (
            f"{prefix}_no_A_per_m2",
            f"{prefix}_without_A_per_m2",
            f"{prefix}_no",
            f"{prefix}_no_segment",
            f"{prefix}_without",
        )
        label = f"{prefix}_no"
    values = _array(_find(case, names, nested=("panel_data",)), label)
    _validate_profile_length(case, values, label)
    mask = _mask(case, reaction, with_edl)
    if mask is None:
        return values
    segmented = np.full_like(values, np.nan, dtype=float)
    segmented[mask] = values[mask]
    return segmented


def _validate_profile_length(case: Any, values: np.ndarray, label: str) -> None:
    x_nm = _line_x(case)
    if values.shape != x_nm.shape:
        raise ValueError(f"{label} shape {values.shape} does not match x_nm shape {x_nm.shape}")


def _float_field(case: Any, names: Sequence[str], *, nested: Sequence[str] = (), default: Any = _MISSING) -> float:
    value = _find(case, names, nested=nested, default=default)
    if value is _MISSING:
        raise AttributeError(", ".join(names))
    return float(value)


def _geometry(case: Any) -> _Geometry:
    x_nm = _line_x(case)
    total = _float_field(
        case,
        ("L_total_nm", "total_length_nm"),
        nested=("panel_data", "rp_data"),
        default=float(x_nm[-1]),
    )
    L_Au = _float_field(case, ("L_Au_nm", "au_length_nm"), nested=("panel_data", "rp_data"))
    L_Pd = _float_field(
        case,
        ("L_Pd_nm", "L_Pd_len_nm", "pd_length_nm"),
        nested=("panel_data", "rp_data"),
        default=np.nan,
    )
    support = _float_field(
        case,
        ("L_support_nm", "support_length_nm", "L_gap_nm"),
        default=float(_find(case, ("value_nm",))),
    )
    boundary_1 = _float_field(
        case,
        ("au_support_boundary_nm", "x_Au_end_nm", "boundary_Au_C_nm"),
        nested=("panel_data", "rp_data"),
        default=L_Au,
    )
    boundary_2_raw = _find(
        case,
        ("support_pd_boundary_nm", "x_Pd_start_nm", "x_support_end_nm", "L_Au_plus_C_nm"),
        nested=("panel_data", "rp_data"),
        default=None,
    )
    if boundary_2_raw is not None:
        boundary_2 = float(boundary_2_raw)
    elif np.isfinite(L_Pd):
        boundary_2 = total - L_Pd
    elif np.isfinite(support):
        boundary_2 = boundary_1 + support
    else:
        # Legacy Figure3 data use L_C_nm for the cumulative support/Pd boundary.
        boundary_2 = _float_field(case, ("L_C_nm",), nested=("panel_data", "rp_data"))
    if not np.isfinite(L_Pd):
        L_Pd = total - boundary_2
    derived_support = boundary_2 - boundary_1
    if not np.isclose(derived_support, support, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"Inconsistent support geometry: value/length={support:g} nm, "
            f"boundaries imply {derived_support:g} nm"
        )
    if not (0.0 <= boundary_1 <= boundary_2 <= total):
        raise ValueError(
            f"Invalid boundaries 0 <= {boundary_1:g} <= {boundary_2:g} <= {total:g} nm"
        )
    return _Geometry(L_Au, support, L_Pd, total, boundary_1, boundary_2)


def _map_arrays(case: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_raw = _find(case, ("x_2d_nm", "map_x_nm"), nested=("rp_data", "data_2d"), default=None)
    if x_raw is None:
        x_raw = _find(case, ("x_nm",), nested=("rp_data", "data_2d"))
    y_raw = _find(case, ("y_2d_nm", "map_y_nm", "y_nm"), nested=("rp_data", "data_2d"))
    phi_raw = _find(case, ("phi_s_mV", "phi_2d_mV"), nested=("rp_data", "data_2d"))
    x_nm = _array(x_raw, "x_2d_nm")
    y_nm = _array(y_raw, "y_2d_nm")
    phi_mV = _array(phi_raw, "phi_s_mV", ndim=2)
    if phi_mV.shape == (x_nm.size, y_nm.size) and phi_mV.shape != (y_nm.size, x_nm.size):
        phi_mV = phi_mV.T
    if phi_mV.shape != (y_nm.size, x_nm.size):
        raise ValueError(
            f"phi_s_mV shape {phi_mV.shape} must equal (len(y), len(x)) "
            f"= {(y_nm.size, x_nm.size)}"
        )
    return x_nm, y_nm, phi_mV


def _normalise_material(value: Any) -> str:
    label = str(value).strip().lower()
    if label == "au":
        return "Au"
    if label == "pd":
        return "Pd"
    if label in {"c", "support", "carbon"}:
        return "support"
    raise ValueError(f"Unsupported sigma material: {value!r}")


def _sigma_segments(case: Any) -> tuple[_SigmaSegment, ...]:
    raw_segments = _find(case, ("sigma_segments", "segments"), nested=("sigma_data",))
    result: list[_SigmaSegment] = []
    for raw in raw_segments:
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, Mapping)) and len(raw) == 3:
            material_raw, x_raw, sigma_raw = raw
            unit = "C_per_m2"
        else:
            material_raw = _find(raw, ("material", "label", "name"))
            x_raw = _find(raw, ("x_nm", "x"))
            sigma_u_raw = _find(raw, ("sigma_uC_per_cm2", "sigma_uC_cm2"), default=None)
            if sigma_u_raw is None:
                sigma_raw = _find(raw, ("sigma_C_per_m2", "sigma"))
                unit = "C_per_m2"
            else:
                sigma_raw = sigma_u_raw
                unit = "uC_per_cm2"
        x_nm = _array(x_raw, "sigma x_nm")
        sigma = _array(sigma_raw, "sigma")
        if sigma.shape != x_nm.shape:
            raise ValueError(f"sigma shape {sigma.shape} does not match segment x shape {x_nm.shape}")
        if unit == "C_per_m2":
            sigma = 100.0 * sigma
        result.append(_SigmaSegment(_normalise_material(material_raw), x_nm, sigma))
    if not result:
        raise ValueError("sigma_segments is empty")
    return tuple(result)


def _finite_range(*arrays: Any, pad_frac: float = 0.08, include_zero: bool = True) -> tuple[float, float]:
    finite_parts: list[np.ndarray] = []
    for values in arrays:
        arr = np.ravel(np.asarray(values, dtype=float))
        finite = arr[np.isfinite(arr)]
        if finite.size:
            finite_parts.append(finite)
    if include_zero:
        finite_parts.append(np.array([0.0]))
    if not finite_parts:
        raise ValueError("Cannot calculate a range from empty/non-finite data")
    values = np.concatenate(finite_parts)
    low = float(np.min(values))
    high = float(np.max(values))
    span = high - low
    if span <= 0.0:
        pad = max(1e-6, 0.08 * max(abs(low), 1.0))
    else:
        pad = pad_frac * span
    return low - pad, high + pad


def compute_common_ranges(cases: Sequence[Any], *, current_exponent: int = CURRENT_EXPONENT) -> PlotRanges:
    """Calculate common line/map ranges for all supplied cases."""

    if not cases:
        raise ValueError("At least one case is required")
    phi_arrays = [_phi_rp(case, state) for case in cases for state in (True, False)]
    current_arrays = [
        _current(case, reaction, state) / (10.0**current_exponent)
        for case in cases
        for reaction in (1, 2)
        for state in (True, False)
    ]
    sigma_arrays = [segment.sigma_uC_per_cm2 for case in cases for segment in _sigma_segments(case)]
    map_abs = max(float(np.nanmax(np.abs(_map_arrays(case)[2]))) for case in cases)
    if not np.isfinite(map_abs):
        raise ValueError("2D Phi_s data contain no finite values")
    phi_vlim = max(10.0, 10.0 * math.ceil(map_abs / 10.0))
    return PlotRanges(
        phi_rp_V=_finite_range(*phi_arrays),
        current_scaled_A_per_m2=_finite_range(*current_arrays),
        sigma_uC_per_cm2=_finite_range(*sigma_arrays),
        phi_s_vlim_mV=phi_vlim,
        current_exponent=current_exponent,
    )


def format_nm(value_nm: float) -> str:
    if math.isclose(value_nm, round(value_nm), rel_tol=0.0, abs_tol=1e-9):
        return str(int(round(value_nm)))
    return f"{value_nm:.3g}"


def case_output_tag(
    case: Any,
    *,
    output_tag: str = DEFAULT_OUTPUT_TAG,
    tag_getter: Callable[[Any], str] | None = None,
) -> str:
    """Return the complete filename tag for one support-length case."""

    if tag_getter is not None:
        tag = str(tag_getter(case)).strip("_ ")
        if not tag:
            raise ValueError("tag_getter returned an empty case tag")
        return tag
    explicit = _find(case, ("figure_tag", "case_tag"), default=None)
    if explicit is not None:
        tag = str(explicit).strip("_ ")
        if tag:
            return tag
    value_nm = float(_find(case, ("value_nm",)))
    suffix = str(output_tag).strip("_ ")
    prefix = f"L_support_{format_nm(value_nm)}nm"
    return f"{prefix}_{suffix}" if suffix else prefix


def save_figure_pair(
    fig: plt.Figure,
    output_dir: str | Path,
    stem: str,
    *,
    transparent: bool,
    pad_inches: float = 0.04,
) -> list[Path]:
    """Save one figure as 600 dpi PNG and editable-text SVG only."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if transparent:
        fig.patch.set_alpha(0.0)
        fig.patch.set_facecolor("none")
        for ax in fig.axes:
            ax.set_facecolor("none")
    saved: list[Path] = []
    for extension in ("png", "svg"):
        path = output_path / f"{stem}.{extension}"
        fig.savefig(
            path,
            dpi=600,
            bbox_inches="tight",
            pad_inches=pad_inches,
            transparent=transparent,
            facecolor="none" if transparent else "white",
            edgecolor="none",
        )
        saved.append(path)
    plt.close(fig)
    return saved


def _style_axis(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, labelpad=5)
    ax.set_ylabel(ylabel, labelpad=5)
    ax.set_title(title, loc="left", pad=7, fontsize=10.4, fontweight="normal")
    ax.tick_params(length=3.4, width=0.85, pad=3.0, labelsize=8.7)
    for spine in ("left", "bottom", "top", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.9)
        ax.spines[spine].set_color(COLORS["dark"])


def _single_axis() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(5.8, 2.55))
    ax = fig.add_axes((0.12, 0.22, 0.85, 0.62))
    return fig, ax


def _unique(values: Sequence[float]) -> list[float]:
    result: list[float] = []
    for value in values:
        value = float(value)
        if not any(math.isclose(value, old, rel_tol=0.0, abs_tol=1e-8) for old in result):
            result.append(value)
    return result


def _full_ticks(geometry: _Geometry) -> list[float]:
    if geometry.L_total_nm > 200.0:
        return np.linspace(0.0, geometry.L_total_nm, 5).tolist()
    return _unique(
        [0.0, geometry.au_support_boundary_nm, geometry.support_pd_boundary_nm, geometry.L_total_nm]
    )


def _format_ticks(ticks: Sequence[float]) -> list[str]:
    return [format_nm(float(tick)) if abs(float(tick)) < 1e4 else f"{float(tick):.3g}" for tick in ticks]


def _add_boundaries(ax: plt.Axes, geometry: _Geometry) -> None:
    for position in _unique((geometry.au_support_boundary_nm, geometry.support_pd_boundary_nm)):
        if 0.0 < position < geometry.L_total_nm:
            ax.axvline(
                position,
                color=COLORS["gray"],
                linewidth=0.85,
                linestyle=(0, (3, 2)),
                alpha=0.82,
                zorder=5,
            )


def _annotate_support(ax: plt.Axes, case: Any, *, location: str = "upper right") -> None:
    value_nm = float(_find(case, ("value_nm",)))
    positions = {
        "upper right": (0.985, 0.955, "right", "top"),
        "lower left": (0.025, 0.055, "left", "bottom"),
        "lower right": (0.975, 0.055, "right", "bottom"),
        "lower center": (0.500, 0.055, "center", "bottom"),
    }
    x, y, horizontal, vertical = positions[location]
    ax.text(
        x,
        y,
        rf"$L_{{\mathrm{{support}}}}$ = {format_nm(value_nm)} nm",
        transform=ax.transAxes,
        ha=horizontal,
        va=vertical,
        fontsize=7.8,
        color=COLORS["dark"],
    )


def _apply_full_x(ax: plt.Axes, geometry: _Geometry) -> None:
    ticks = _full_ticks(geometry)
    ax.set_xlim(0.0, geometry.L_total_nm)
    ax.set_xticks(ticks)
    ax.set_xticklabels(_format_ticks(ticks))


def _window_ticks(xmin: float, xmax: float, geometry: _Geometry) -> list[float]:
    candidates = [xmin, xmax]
    for boundary in (geometry.au_support_boundary_nm, geometry.support_pd_boundary_nm):
        if xmin <= boundary <= xmax:
            # Four-digit labels separated by only the 2 nm active width overlap
            # at the Pd edge; the dashed boundary and material lane still mark it.
            near_endpoint = min(boundary - xmin, xmax - boundary) < 0.15 * (xmax - xmin)
            if near_endpoint and max(abs(xmin), abs(xmax)) >= 100.0:
                continue
            candidates.append(boundary)
    if len(_unique(candidates)) == 2:
        candidates.append(0.5 * (xmin + xmax))
    return sorted(_unique(candidates))


def _active_windows(geometry: _Geometry, width_nm: float) -> tuple[tuple[float, float], tuple[float, float]]:
    width_nm = min(float(width_nm), geometry.L_total_nm)
    if width_nm <= 0.0:
        raise ValueError("active_window_nm must be positive")
    return (0.0, width_nm), (geometry.L_total_nm - width_nm, geometry.L_total_nm)


def plot_reaction_plane_potential(
    case: Any,
    output_dir: str | Path,
    ylim: tuple[float, float],
    *,
    tag: str,
) -> list[Path]:
    x_nm = _line_x(case)
    geometry = _geometry(case)
    fig, ax = _single_axis()
    ax.plot(x_nm, _phi_rp(case, True), color=COLORS["with"], lw=2.0, label="with EDL", zorder=3)
    ax.plot(x_nm, _phi_rp(case, False), color=COLORS["without"], lw=1.8, label="w/o EDL", zorder=2)
    _add_boundaries(ax, geometry)
    _style_axis(ax, "x (nm)", r"$\phi_{\mathrm{RP}}(x)$ (V)", "Reaction-plane potential")
    _apply_full_x(ax, geometry)
    ax.set_ylim(*ylim)
    _annotate_support(ax, case, location="lower center" if geometry.L_total_nm > 200.0 else "upper right")
    ax.legend(loc="center right", bbox_to_anchor=(0.98, 0.50), fontsize=8.0, handlelength=2.0)
    return save_figure_pair(
        fig,
        output_dir,
        f"figure_3_panel_b_reaction_plane_potential_{tag}",
        transparent=True,
    )


def _current_arrays(case: Any, exponent: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scale = 10.0**exponent
    return (
        _current(case, 1, True) / scale,
        _current(case, 2, True) / scale,
        _current(case, 1, False) / scale,
        _current(case, 2, False) / scale,
    )


def _draw_currents(ax: plt.Axes, case: Any, exponent: int) -> None:
    x_nm = _line_x(case)
    i1_with, i2_with, i1_no, i2_no = _current_arrays(case, exponent)
    ax.axhline(0.0, color=COLORS["dark"], lw=0.55, alpha=0.78, zorder=2)
    ax.plot(
        x_nm,
        i1_with,
        color=COLORS["current_Au"],
        lw=2.0,
        label=r"$i_1$ (Au), with EDL",
        zorder=4,
    )
    ax.plot(
        x_nm,
        i1_no,
        color=COLORS["current_Au"],
        lw=1.7,
        ls=(0, (4, 2)),
        label=r"$i_1$ (Au), w/o EDL",
        zorder=3,
    )
    ax.plot(
        x_nm,
        i2_with,
        color=COLORS["current_Pd"],
        lw=2.0,
        label=r"$i_2$ (Pd), with EDL",
        zorder=4,
    )
    ax.plot(
        x_nm,
        i2_no,
        color=COLORS["current_Pd"],
        lw=1.7,
        ls=(0, (4, 2)),
        label=r"$i_2$ (Pd), w/o EDL",
        zorder=3,
    )


def _current_ylabel(exponent: int) -> str:
    if exponent == 0:
        return r"$i(x)$ (A/m$^2$)"
    return rf"$i(x)$ ($10^{{{exponent}}}$ A/m$^2$)"


def plot_local_current_density(
    case: Any,
    output_dir: str | Path,
    ylim: tuple[float, float],
    *,
    tag: str,
    current_exponent: int = CURRENT_EXPONENT,
) -> list[Path]:
    geometry = _geometry(case)
    fig, ax = _single_axis()
    _draw_currents(ax, case, current_exponent)
    _add_boundaries(ax, geometry)
    _style_axis(ax, "x (nm)", _current_ylabel(current_exponent), "Local current density at RP")
    _apply_full_x(ax, geometry)
    ax.set_ylim(*ylim)
    _annotate_support(ax, case, location="lower left")
    ax.legend(loc="upper right", fontsize=6.7, handlelength=1.6)
    return save_figure_pair(
        fig,
        output_dir,
        f"figure_3_panel_e_local_current_density_{tag}",
        transparent=True,
    )


def _summary_float(case: Any, names: Sequence[str], default: float = np.nan) -> float:
    summary = _find(case, ("summary",), default={})
    value = _find(summary, names, default=default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _add_material_lane(
    ax: plt.Axes,
    geometry: _Geometry,
    *,
    xmin: float = 0.0,
    xmax: float | None = None,
) -> None:
    xmax = geometry.L_total_nm if xmax is None else float(xmax)
    span = xmax - xmin
    segments = (
        ("Au", 0.0, geometry.au_support_boundary_nm, COLORS["Au"], COLORS["dark"]),
        ("C", geometry.au_support_boundary_nm, geometry.support_pd_boundary_nm, COLORS["support"], "white"),
        ("Pd", geometry.support_pd_boundary_nm, geometry.L_total_nm, COLORS["Pd"], "white"),
    )
    for label, start, stop, face, text_color in segments:
        start_clip = max(start, xmin)
        stop_clip = min(stop, xmax)
        width = stop_clip - start_clip
        if width <= 1e-9:
            continue
        ax.add_patch(Rectangle((start_clip, 0.0), width, 1.0, facecolor=face, edgecolor="white", lw=0.8))
        if span > 0.0 and width >= 1.0 and width / span >= 0.075:
            ax.text(
                0.5 * (start_clip + stop_clip),
                0.5,
                label,
                ha="center",
                va="center",
                fontsize=8.0,
                color=text_color,
            )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()


def _draw_map(
    ax: plt.Axes,
    case: Any,
    norm: TwoSlopeNorm,
    *,
    levels: np.ndarray,
) -> Any:
    x_nm, y_nm, phi_mV = _map_arrays(case)
    mesh = ax.pcolormesh(x_nm, y_nm, phi_mV, shading="auto", cmap="RdBu_r", norm=norm, rasterized=True)
    finite = phi_mV[np.isfinite(phi_mV)]
    if finite.size and float(np.ptp(finite)) > 1e-12:
        inside = levels[(levels > float(np.min(finite))) & (levels < float(np.max(finite)))]
        if inside.size:
            ax.contour(x_nm, y_nm, phi_mV, levels=inside, colors="black", linewidths=0.28, alpha=0.28)
    return mesh


def _style_map_axis(ax: plt.Axes, title: str, *, show_ylabel: bool = True) -> None:
    ax.set_title(title, loc="left", pad=5, fontsize=9.4, fontweight="normal")
    ax.set_ylabel("y (nm)" if show_ylabel else "")
    if not show_ylabel:
        ax.tick_params(labelleft=False)
    ax.tick_params(length=3.2, width=0.85, pad=2.5, labelsize=8.0)
    for spine in ("left", "bottom", "top", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.9)
        ax.spines[spine].set_color(COLORS["dark"])


def plot_solution_phase_potential_2d(
    case: Any,
    output_dir: str | Path,
    phi_vlim_mV: float,
    *,
    tag: str,
) -> list[Path]:
    x_nm, y_nm, _ = _map_arrays(case)
    geometry = _geometry(case)
    fig = plt.figure(figsize=(5.8, 3.0), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=(1.0, 0.038), height_ratios=(1.0, 0.12), hspace=0.24, wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    lane_ax = fig.add_subplot(gs[1, 0], sharex=ax)
    fig.add_subplot(gs[1, 1]).set_axis_off()
    norm = TwoSlopeNorm(vmin=-phi_vlim_mV, vcenter=0.0, vmax=phi_vlim_mV)
    levels = np.linspace(-phi_vlim_mV, phi_vlim_mV, 11)
    mesh = _draw_map(ax, case, norm, levels=levels)
    emix = _summary_float(case, ("E_mix_with", "E_mix_with_V", "E_mix"))
    title = "Solution phase potential"
    if np.isfinite(emix):
        title += rf", $E_{{\mathrm{{mix}}}}$ = {emix:.2f} V"
    _style_map_axis(ax, title)
    ax.set_xlim(float(x_nm[0]), float(x_nm[-1]))
    ax.set_ylim(float(y_nm[0]), float(y_nm[-1]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ticks = _full_ticks(geometry)
    ax.set_xticks(ticks)
    ax.set_xticklabels(_format_ticks(ticks))
    ax.set_xlabel("")
    _add_boundaries(ax, geometry)
    _annotate_support(ax, case)
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label(r"$\Phi_s$ (mV)", labelpad=5)
    cbar.ax.tick_params(length=2.8, width=0.75, labelsize=7.8, pad=2.2)
    cbar.outline.set_linewidth(0.8)
    _add_material_lane(lane_ax, geometry)
    lane_ax.text(
        0.5,
        -0.58,
        "x (nm)",
        transform=lane_ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color=COLORS["dark"],
        clip_on=False,
    )
    return save_figure_pair(fig, output_dir, f"solution_phase_potential_2d_{tag}", transparent=False)


def _draw_sigma(ax: plt.Axes, case: Any, *, labels: bool) -> None:
    for segment in _sigma_segments(case):
        visible_label = "C" if segment.material == "support" else segment.material
        ax.plot(
            segment.x_nm,
            segment.sigma_uC_per_cm2,
            color=COLORS[segment.material],
            lw=2.0,
            label=visible_label if labels else None,
            zorder=3,
        )


def plot_surface_charge_distribution(
    case: Any,
    output_dir: str | Path,
    ylim: tuple[float, float],
    *,
    tag: str,
) -> list[Path]:
    geometry = _geometry(case)
    fig, ax = _single_axis()
    _draw_sigma(ax, case, labels=True)
    ax.axhline(0.0, color=COLORS["dark"], lw=0.7, alpha=0.82, zorder=2)
    _add_boundaries(ax, geometry)
    _style_axis(ax, "x (nm)", r"$\sigma(x)$ ($\mu$C/cm$^2$)", "Surface charge distribution")
    _apply_full_x(ax, geometry)
    ax.set_ylim(*ylim)
    _annotate_support(ax, case)
    ax.legend(loc="upper left", fontsize=7.4, handlelength=1.8)
    return save_figure_pair(fig, output_dir, f"surface_charge_distribution_{tag}", transparent=True)


def plot_reaction_plane_potential_active_zoom(
    case: Any,
    output_dir: str | Path,
    ylim: tuple[float, float],
    *,
    tag: str,
    active_window_nm: float = ACTIVE_WINDOW_NM,
) -> list[Path]:
    x_nm = _line_x(case)
    geometry = _geometry(case)
    windows = _active_windows(geometry, active_window_nm)
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.55), sharey=True)
    for ax, (xmin, xmax), title in zip(axes, windows, ("Au-side active window", "Pd-side active window"), strict=True):
        ax.plot(x_nm, _phi_rp(case, True), color=COLORS["with"], lw=2.0, label="with EDL", zorder=3)
        ax.plot(x_nm, _phi_rp(case, False), color=COLORS["without"], lw=1.8, label="w/o EDL", zorder=2)
        _add_boundaries(ax, geometry)
        _style_axis(ax, "x (nm)", r"$\phi_{\mathrm{RP}}(x)$ (V)", title)
        ticks = _window_ticks(xmin, xmax, geometry)
        ax.set_xlim(xmin, xmax)
        ax.set_xticks(ticks)
        ax.set_xticklabels(_format_ticks(ticks))
        ax.set_ylim(*ylim)
    axes[1].set_ylabel("")
    axes[1].tick_params(labelleft=False)
    axes[1].legend(loc="center right", fontsize=7.7, handlelength=1.8)
    _annotate_support(axes[1], case)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.22, top=0.84, wspace=0.16)
    return save_figure_pair(
        fig,
        output_dir,
        f"figure_3_panel_b_reaction_plane_potential_active_zoom_{tag}",
        transparent=True,
    )


def plot_local_current_density_active_zoom(
    case: Any,
    output_dir: str | Path,
    ylim: tuple[float, float],
    *,
    tag: str,
    current_exponent: int = CURRENT_EXPONENT,
    active_window_nm: float = ACTIVE_WINDOW_NM,
) -> list[Path]:
    geometry = _geometry(case)
    windows = _active_windows(geometry, active_window_nm)
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.55), sharey=True)
    for ax, (xmin, xmax), title in zip(axes, windows, ("Au-side active window", "Pd-side active window"), strict=True):
        _draw_currents(ax, case, current_exponent)
        _add_boundaries(ax, geometry)
        _style_axis(ax, "x (nm)", _current_ylabel(current_exponent), title)
        ticks = _window_ticks(xmin, xmax, geometry)
        ax.set_xlim(xmin, xmax)
        ax.set_xticks(ticks)
        ax.set_xticklabels(_format_ticks(ticks))
        ax.set_ylim(*ylim)
    axes[1].set_ylabel("")
    axes[1].tick_params(labelleft=False)
    axes[1].legend(loc="upper right", fontsize=6.3, handlelength=1.5)
    _annotate_support(axes[0], case, location="lower right")
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.22, top=0.84, wspace=0.16)
    return save_figure_pair(
        fig,
        output_dir,
        f"figure_3_panel_e_local_current_density_active_zoom_{tag}",
        transparent=True,
    )


def plot_solution_phase_potential_2d_active_zoom(
    case: Any,
    output_dir: str | Path,
    phi_vlim_mV: float,
    *,
    tag: str,
    active_window_nm: float = ACTIVE_WINDOW_NM,
) -> list[Path]:
    _, y_nm, _ = _map_arrays(case)
    geometry = _geometry(case)
    windows = _active_windows(geometry, active_window_nm)
    fig = plt.figure(figsize=(5.95, 3.15), facecolor="white")
    gs = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 0.050), height_ratios=(1.0, 0.12), hspace=0.24, wspace=0.14)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])
    lane_axes = [fig.add_subplot(gs[1, 0], sharex=axes[0]), fig.add_subplot(gs[1, 1], sharex=axes[1])]
    fig.add_subplot(gs[1, 2]).set_axis_off()
    norm = TwoSlopeNorm(vmin=-phi_vlim_mV, vcenter=0.0, vmax=phi_vlim_mV)
    levels = np.linspace(-phi_vlim_mV, phi_vlim_mV, 11)
    mesh = None
    for ax, lane_ax, (xmin, xmax), title, show_ylabel in zip(
        axes,
        lane_axes,
        windows,
        ("Au-side active window", "Pd-side active window"),
        (True, False),
        strict=True,
    ):
        mesh = _draw_map(ax, case, norm, levels=levels)
        _style_map_axis(ax, title, show_ylabel=show_ylabel)
        _add_boundaries(ax, geometry)
        ticks = _window_ticks(xmin, xmax, geometry)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(float(y_nm[0]), float(y_nm[-1]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_xticks(ticks)
        ax.set_xticklabels(_format_ticks(ticks))
        _add_material_lane(lane_ax, geometry, xmin=xmin, xmax=xmax)
        lane_ax.text(
            0.5,
            -0.58,
            "x (nm)",
            transform=lane_ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
            color=COLORS["dark"],
            clip_on=False,
        )
    if mesh is None:
        raise RuntimeError("No 2D active-window mesh was created")
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label(r"$\Phi_s$ (mV)", labelpad=5)
    cbar.ax.tick_params(length=2.8, width=0.75, labelsize=7.8, pad=2.2)
    cbar.outline.set_linewidth(0.8)
    _annotate_support(axes[1], case)
    return save_figure_pair(
        fig,
        output_dir,
        f"solution_phase_potential_2d_active_zoom_{tag}",
        transparent=False,
    )


def plot_surface_charge_distribution_active_zoom(
    case: Any,
    output_dir: str | Path,
    ylim: tuple[float, float],
    *,
    tag: str,
    active_window_nm: float = ACTIVE_WINDOW_NM,
) -> list[Path]:
    geometry = _geometry(case)
    windows = _active_windows(geometry, active_window_nm)
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.55), sharey=True)
    for ax, (xmin, xmax), title in zip(axes, windows, ("Au-side active window", "Pd-side active window"), strict=True):
        _draw_sigma(ax, case, labels=True)
        ax.axhline(0.0, color=COLORS["dark"], lw=0.7, alpha=0.82)
        _add_boundaries(ax, geometry)
        _style_axis(ax, "x (nm)", r"$\sigma(x)$ ($\mu$C/cm$^2$)", title)
        ticks = _window_ticks(xmin, xmax, geometry)
        ax.set_xlim(xmin, xmax)
        ax.set_xticks(ticks)
        ax.set_xticklabels(_format_ticks(ticks))
        ax.set_ylim(*ylim)
    axes[1].set_ylabel("")
    axes[1].tick_params(labelleft=False)
    axes[1].legend(loc="lower right", fontsize=7.2, handlelength=1.7)
    _annotate_support(axes[1], case)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.22, top=0.84, wspace=0.16)
    return save_figure_pair(
        fig,
        output_dir,
        f"surface_charge_distribution_active_zoom_{tag}",
        transparent=True,
    )


def plot_all_cases(
    cases: Sequence[Any],
    output_dir: str | Path,
    *,
    output_tag: str = DEFAULT_OUTPUT_TAG,
    tag_getter: Callable[[Any], str] | None = None,
    active_zoom_support_nm: float = ACTIVE_ZOOM_SUPPORT_NM,
    active_window_nm: float = ACTIVE_WINDOW_NM,
    current_exponent: int = CURRENT_EXPONENT,
) -> list[Path]:
    """Render the four case plots and the four long-support active zooms.

    Each case must expose the direct fields described in the module docstring,
    either as attributes or mapping keys.  The default naming scheme is
    ``L_support_{n}nm_au2_pd2_20260528_111255``.  Pass ``tag_getter`` when the
    numerical driver already owns the complete case tag.  Figure 3-style line
    panels are written below ``Figure_3``; 2D reaction-plane maps and surface
    charge plots are written below ``Figure_RP``.
    """

    cases = tuple(cases)
    if not cases:
        raise ValueError("At least one legacy Au | support | Pd case is required")
    configure_style()
    output_path = Path(output_dir)
    figure_3_dir = output_path / "Figure_3"
    figure_rp_dir = output_path / "Figure_RP"
    ranges = compute_common_ranges(cases, current_exponent=current_exponent)
    saved: list[Path] = []
    zoom_case: Any | None = None
    for case in cases:
        tag = case_output_tag(case, output_tag=output_tag, tag_getter=tag_getter)
        saved.extend(plot_reaction_plane_potential(case, figure_3_dir, ranges.phi_rp_V, tag=tag))
        saved.extend(
            plot_local_current_density(
                case,
                figure_3_dir,
                ranges.current_scaled_A_per_m2,
                tag=tag,
                current_exponent=current_exponent,
            )
        )
        saved.extend(plot_solution_phase_potential_2d(case, figure_rp_dir, ranges.phi_s_vlim_mV, tag=tag))
        saved.extend(
            plot_surface_charge_distribution(case, figure_rp_dir, ranges.sigma_uC_per_cm2, tag=tag)
        )
        if math.isclose(
            float(_find(case, ("value_nm",))),
            float(active_zoom_support_nm),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            zoom_case = case
    if zoom_case is None:
        raise ValueError(f"No case found for active zoom at L_support={active_zoom_support_nm:g} nm")
    zoom_tag = case_output_tag(zoom_case, output_tag=output_tag, tag_getter=tag_getter)
    saved.extend(
        plot_reaction_plane_potential_active_zoom(
            zoom_case,
            figure_3_dir,
            ranges.phi_rp_V,
            tag=zoom_tag,
            active_window_nm=active_window_nm,
        )
    )
    saved.extend(
        plot_local_current_density_active_zoom(
            zoom_case,
            figure_3_dir,
            ranges.current_scaled_A_per_m2,
            tag=zoom_tag,
            current_exponent=current_exponent,
            active_window_nm=active_window_nm,
        )
    )
    saved.extend(
        plot_solution_phase_potential_2d_active_zoom(
            zoom_case,
            figure_rp_dir,
            ranges.phi_s_vlim_mV,
            tag=zoom_tag,
            active_window_nm=active_window_nm,
        )
    )
    saved.extend(
        plot_surface_charge_distribution_active_zoom(
            zoom_case,
            figure_rp_dir,
            ranges.sigma_uC_per_cm2,
            tag=zoom_tag,
            active_window_nm=active_window_nm,
        )
    )
    expected_paths = 2 * (4 * len(cases) + 4)
    if len(saved) != expected_paths:
        raise RuntimeError(f"Expected {expected_paths} PNG/SVG paths, generated {len(saved)}")
    pdfs = sorted(output_path.rglob("*.pdf"))
    if pdfs:
        raise RuntimeError(f"Legacy plot output directory must contain zero PDF files; found {pdfs}")
    return saved


__all__ = [
    "ACTIVE_WINDOW_NM",
    "ACTIVE_ZOOM_SUPPORT_NM",
    "COLORS",
    "CURRENT_EXPONENT",
    "DEFAULT_OUTPUT_TAG",
    "PlotRanges",
    "case_output_tag",
    "compute_common_ranges",
    "configure_style",
    "plot_all_cases",
    "plot_local_current_density",
    "plot_local_current_density_active_zoom",
    "plot_reaction_plane_potential",
    "plot_reaction_plane_potential_active_zoom",
    "plot_solution_phase_potential_2d",
    "plot_solution_phase_potential_2d_active_zoom",
    "plot_surface_charge_distribution",
    "plot_surface_charge_distribution_active_zoom",
    "save_figure_pair",
]
