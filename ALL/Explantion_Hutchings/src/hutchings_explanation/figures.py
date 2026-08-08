"""Publication figures for the four-case Hutchings explanation collection.

The module consumes the normalized dictionaries returned by
``hutchings_explanation.cases.load_all_cases``.  It deliberately contains no
solver calls: every plotted value is traceable to the case-loading layer.

Two public entry points are provided:

``generate_summary_figures``
    Write comparisons of mixed potential and absolute mixed current with one
    common w/o-EDL baseline and four case-specific with-EDL bars.

``generate_case_figures``
    Write six Figure 3 panels and three reaction-plane figures for one case.

All figures are exported as 600 dpi PNG plus editable-text SVG.  No PDF output
is produced.  Dense two-dimensional fields are rasterized within SVG while
axes, labels, and annotations remain editable vector objects.
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator


COLORS = {
    "with": "#F26B38",
    "with_alt": "#D83A2E",
    "with_gold": "#F2B134",
    "without": "#12355B",
    "without_alt": "#2D5A7B",
    "current_Au": "#009E73",
    "current_Pd": "#0072B2",
    "Au": "#E4C133",
    "C": "#8C8C8C",
    "Pd": "#5A90C8",
    "dark": "#272727",
    "gray": "#767676",
    "light_gray": "#CFCECE",
}

PUBLICATION_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Helvetica",
        "Nimbus Sans",
        "Arial",
        "DejaVu Sans",
        "sans-serif",
    ],
    "font.size": 9.8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.linewidth": 0.9,
    "axes.grid": False,
    "legend.frameon": False,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.edgecolor": "none",
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

_MISSING = object()
_NO_DEFAULT = object()


@dataclass(frozen=True)
class _Segment:
    segment_id: str
    material: str
    start_nm: float
    stop_nm: float

    @property
    def width_nm(self) -> float:
        return self.stop_nm - self.start_nm


@dataclass(frozen=True)
class _ContinuousSurface:
    x_nm: np.ndarray
    segment_id: np.ndarray
    material: np.ndarray
    phi_with_V: np.ndarray
    phi_no_V: np.ndarray
    c_r1_with: np.ndarray
    c_o2_with: np.ndarray
    c_r1_no: np.ndarray
    c_o2_no: np.ndarray
    eta_au_with_V: np.ndarray
    eta_pd_with_V: np.ndarray
    eta_au_no_V: np.ndarray
    eta_pd_no_V: np.ndarray
    j_au_with: np.ndarray
    j_pd_with: np.ndarray
    j_au_no: np.ndarray
    j_pd_no: np.ndarray


@dataclass(frozen=True)
class _Grid2D:
    x_nm: np.ndarray
    y_nm: np.ndarray
    phi_s_mV: np.ndarray
    c_r1: np.ndarray
    c_o2: np.ndarray


@dataclass(frozen=True)
class _ChargeSegment:
    segment_id: str
    material: str
    x_nm: np.ndarray
    sigma_C_per_m2: np.ndarray


def _read(source: Any, *names: str, default: Any = _NO_DEFAULT) -> Any:
    """Return the first available mapping key or object attribute."""

    for name in names:
        if isinstance(source, Mapping) and name in source:
            return source[name]
        if hasattr(source, name):
            return getattr(source, name)
    if default is not _NO_DEFAULT:
        return default
    raise KeyError(f"Missing required field; tried {', '.join(names)}")


def _finite_scalar(source: Any, *names: str, default: Any = _MISSING) -> float:
    raw = _read(source, *names, default=default)
    if raw is _MISSING:
        raise KeyError(f"Missing scalar field; tried {', '.join(names)}")
    array = np.asarray(raw)
    if array.size != 1:
        raise ValueError(f"Expected one scalar for {names!r}; got shape {array.shape}")
    value = float(array.reshape(-1)[0])
    if not np.isfinite(value):
        raise ValueError(f"Expected a finite scalar for {names!r}; got {value!r}")
    return value


def _normalise_material(value: Any) -> str:
    label = str(value).strip().lower()
    if label == "au" or label.startswith("au_"):
        return "Au"
    if label == "pd" or label.startswith("pd_"):
        return "Pd"
    if label in {"c", "support", "carbon", "c_left", "c_right"} or label.startswith("c_"):
        return "C"
    raise ValueError(f"Unsupported material label: {value!r}")


def _as_1d(raw: Any, name: str, *, dtype: Any = float) -> np.ndarray:
    array = np.asarray(raw, dtype=dtype)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array; got {array.shape}")
    return array


def _broadcast_1d(raw: Any, size: int, name: str, *, default: Any = _MISSING) -> np.ndarray:
    if raw is _MISSING:
        if default is _MISSING:
            raise KeyError(name)
        raw = default
    array = np.asarray(raw, dtype=float)
    if array.ndim == 0 or array.size == 1:
        return np.full(size, float(array.reshape(-1)[0]), dtype=float)
    array = np.ravel(array).astype(float, copy=False)
    if array.size != size:
        raise ValueError(f"{name} has length {array.size}; expected {size}")
    return array


def _case_sequence(cases: Any) -> list[Mapping[str, Any]]:
    if isinstance(cases, Mapping):
        values = list(cases.values())
    elif isinstance(cases, Sequence) and not isinstance(cases, (str, bytes)):
        values = list(cases)
    else:
        raise TypeError("cases must be a mapping or a sequence of case mappings")
    if not values:
        raise ValueError("At least one case is required")
    if not all(isinstance(case, Mapping) for case in values):
        raise TypeError("Every case must be a mapping")
    return values


def _case_tag(case: Mapping[str, Any]) -> str:
    tag = str(_read(case, "artifact_tag", "key", default="case")).strip()
    if not tag:
        raise ValueError("case artifact_tag must not be empty")
    return tag.replace(" ", "_")


def _case_label(case: Mapping[str, Any]) -> str:
    label = str(_read(case, "display_label", "key", default="Case")).strip()
    return "\n".join(textwrap.wrap(label, width=21, break_long_words=False))


def _is_independent(case: Mapping[str, Any]) -> bool:
    topology = str(_read(case, "topology_kind", default="")).strip().lower()
    return topology.startswith("independent") or bool(_read(case, "independent_surfaces", default=None))


def _segment_coordinate(record: Any, nm_names: Sequence[str], m_names: Sequence[str]) -> float:
    for name in nm_names:
        raw = _read(record, name, default=_MISSING)
        if raw is not _MISSING:
            return float(raw)
    for name in m_names:
        raw = _read(record, name, default=_MISSING)
        if raw is not _MISSING:
            return 1.0e9 * float(raw)
    raise KeyError(f"Missing segment coordinate; tried {(*nm_names, *m_names)!r}")


def _load_segments(case: Mapping[str, Any]) -> tuple[_Segment, ...]:
    raw_segments = _read(case, "segments")
    if not isinstance(raw_segments, Sequence) or isinstance(raw_segments, (str, bytes)):
        raise TypeError("case.segments must be a sequence")
    segments: list[_Segment] = []
    for index, record in enumerate(raw_segments):
        material = _normalise_material(_read(record, "material"))
        segment_id = str(_read(record, "segment_id", "name", default=f"segment_{index}"))
        start_nm = _segment_coordinate(
            record,
            ("start_nm", "x_start_nm", "x0_nm"),
            ("start_m", "x_start_m", "x0_m"),
        )
        stop_nm = _segment_coordinate(
            record,
            ("stop_nm", "end_nm", "x_end_nm", "x1_nm"),
            ("stop_m", "end_m", "x_end_m", "x1_m"),
        )
        if not np.isfinite(start_nm) or not np.isfinite(stop_nm) or stop_nm < start_nm:
            raise ValueError(f"Invalid segment interval {start_nm!r}--{stop_nm!r} nm")
        if stop_nm > start_nm:
            segments.append(_Segment(segment_id, material, start_nm, stop_nm))
    if not segments:
        raise ValueError("Continuous case has no positive-width segments")
    segments.sort(key=lambda segment: segment.start_nm)
    tolerance = max(1e-8, 1e-8 * (segments[-1].stop_nm - segments[0].start_nm))
    for left, right in zip(segments[:-1], segments[1:], strict=True):
        if not math.isclose(left.stop_nm, right.start_nm, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(
                f"Segment gap/overlap between {left.segment_id!r} and {right.segment_id!r}"
            )
    return tuple(segments)


def _surface_value(source: Mapping[str, Any], size: int, *names: str, default: Any = _MISSING) -> np.ndarray:
    raw = _read(source, *names, default=_MISSING)
    return _broadcast_1d(raw, size, names[0], default=default)


def _load_surface(case: Mapping[str, Any]) -> _ContinuousSurface:
    source = _read(case, "surface")
    x_nm = _as_1d(_read(source, "x_nm"), "surface.x_nm")
    if np.any(~np.isfinite(x_nm)) or np.any(np.diff(x_nm) <= 0.0):
        raise ValueError("surface.x_nm must be finite and strictly increasing")
    size = x_nm.size
    material_raw = np.asarray(_read(source, "material"), dtype=object)
    if material_raw.ndim == 0:
        material_raw = np.full(size, material_raw.item(), dtype=object)
    if material_raw.size != size:
        raise ValueError("surface.material length does not match surface.x_nm")
    material = np.asarray([_normalise_material(item) for item in material_raw], dtype=object)
    segment_raw = np.asarray(_read(source, "segment_id", default=material), dtype=object)
    if segment_raw.ndim == 0:
        segment_raw = np.full(size, segment_raw.item(), dtype=object)
    if segment_raw.size != size:
        raise ValueError("surface.segment_id length does not match surface.x_nm")
    return _ContinuousSurface(
        x_nm=x_nm,
        segment_id=segment_raw,
        material=material,
        phi_with_V=_surface_value(
            source,
            size,
            "phi_RP_with_V",
            "phi_rp_with_V",
            "phi_with_V",
        ),
        phi_no_V=_surface_value(
            source,
            size,
            "phi_RP_no_V",
            "phi_rp_no_V",
            "phi_no_V",
            default=0.0,
        ),
        c_r1_with=_surface_value(source, size, "c_R1_with", "c_r1_with"),
        c_o2_with=_surface_value(source, size, "c_O2_with", "c_o2_with"),
        c_r1_no=_surface_value(source, size, "c_R1_no", "c_r1_no", default=1.0),
        c_o2_no=_surface_value(source, size, "c_O2_no", "c_o2_no", default=1.0),
        eta_au_with_V=_surface_value(
            source,
            size,
            "eta_Au_with_V",
            "eta_au_with_V",
            default=np.nan,
        ),
        eta_pd_with_V=_surface_value(
            source,
            size,
            "eta_Pd_with_V",
            "eta_pd_with_V",
            default=np.nan,
        ),
        eta_au_no_V=_surface_value(
            source,
            size,
            "eta_Au_no_V",
            "eta_au_no_V",
            default=np.nan,
        ),
        eta_pd_no_V=_surface_value(
            source,
            size,
            "eta_Pd_no_V",
            "eta_pd_no_V",
            default=np.nan,
        ),
        j_au_with=_surface_value(
            source,
            size,
            "j_Au_with_A_per_m2",
            "j_Au_with",
            "j_au_with",
            "i1_with",
        ),
        j_pd_with=_surface_value(
            source,
            size,
            "j_Pd_with_A_per_m2",
            "j_Pd_with",
            "j_pd_with",
            "i2_with",
        ),
        j_au_no=_surface_value(
            source,
            size,
            "j_Au_no_A_per_m2",
            "j_Au_no",
            "j_au_no",
            "i1_no",
        ),
        j_pd_no=_surface_value(
            source,
            size,
            "j_Pd_no_A_per_m2",
            "j_Pd_no",
            "j_pd_no",
            "i2_no",
        ),
    )


def _field_2d(source: Mapping[str, Any], name: str, aliases: Sequence[str], ny: int, nx: int) -> np.ndarray:
    raw = _read(source, name, *aliases)
    array = np.asarray(raw, dtype=float)
    if array.shape == (nx, ny) and array.shape != (ny, nx):
        array = array.T
    if array.shape != (ny, nx):
        raise ValueError(f"grid_2d.{name} has shape {array.shape}; expected {(ny, nx)}")
    return array


def _load_grid(case: Mapping[str, Any]) -> _Grid2D:
    source = _read(case, "grid_2d")
    x_nm = _as_1d(_read(source, "x_nm"), "grid_2d.x_nm")
    y_nm = _as_1d(_read(source, "y_nm"), "grid_2d.y_nm")
    if np.any(np.diff(x_nm) <= 0.0) or np.any(np.diff(y_nm) <= 0.0):
        raise ValueError("grid_2d coordinates must be strictly increasing")
    ny, nx = y_nm.size, x_nm.size
    return _Grid2D(
        x_nm=x_nm,
        y_nm=y_nm,
        phi_s_mV=_field_2d(
            source,
            "phi_s_with_mV",
            ("phi_s_mV", "phi_mV", "phi"),
            ny,
            nx,
        ),
        c_r1=_field_2d(
            source,
            "c_R1_with",
            ("c_R1_norm", "c_R1", "c_r1_norm", "c_r1"),
            ny,
            nx,
        ),
        c_o2=_field_2d(
            source,
            "c_O2_with",
            ("c_O2_norm", "c_O2", "c_o2_norm", "c_o2"),
            ny,
            nx,
        ),
    )


def _load_charges(case: Mapping[str, Any]) -> tuple[_ChargeSegment, ...]:
    raw = _read(case, "charge_segments", "surface_charge_segments")
    if isinstance(raw, Mapping):
        records = list(raw.values()) if not {"x_nm", "sigma_C_per_m2"}.issubset(raw) else [raw]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        records = list(raw)
    else:
        raise TypeError("charge_segments must be a mapping or sequence")
    result: list[_ChargeSegment] = []
    for index, record in enumerate(records):
        x_nm = np.atleast_1d(np.asarray(_read(record, "x_nm"), dtype=float))
        sigma = np.atleast_1d(
            np.asarray(_read(record, "sigma_C_per_m2", "sigma"), dtype=float)
        )
        if x_nm.ndim != 1 or sigma.shape != x_nm.shape or x_nm.size == 0:
            raise ValueError(f"Invalid charge segment {index}")
        material = _normalise_material(_read(record, "material"))
        segment_id = str(_read(record, "segment_id", "name", default=f"charge_{index}"))
        result.append(_ChargeSegment(segment_id, material, x_nm, sigma))
    if not result:
        raise ValueError("charge_segments is empty")
    return tuple(result)


def _style_axis(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, labelpad=5)
    ax.set_ylabel(ylabel, labelpad=5)
    ax.set_title(title, loc="left", pad=7, fontsize=10.5, fontweight="normal")
    ax.tick_params(length=3.4, width=0.85, pad=3.0, labelsize=8.8)
    for spine in ("left", "bottom", "top", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.9)
        ax.spines[spine].set_color(COLORS["dark"])


def _save_pair(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    transparent: bool,
    pad_inches: float = 0.05,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / f"{stem}.{extension}" for extension in ("png", "svg")]
    if transparent:
        fig.patch.set_alpha(0.0)
        fig.patch.set_facecolor("none")
        for axis in fig.axes:
            axis.set_facecolor("none")
    for target in targets:
        fig.savefig(
            target,
            dpi=600,
            bbox_inches="tight",
            pad_inches=pad_inches,
            transparent=transparent,
            facecolor="none" if transparent else "white",
            edgecolor="none",
        )
    plt.close(fig)
    return targets


def _finite_ylim(ax: plt.Axes, *arrays: Any, include_zero: bool = False) -> None:
    parts: list[np.ndarray] = []
    for raw in arrays:
        values = np.ravel(np.asarray(raw, dtype=float))
        finite = values[np.isfinite(values)]
        if finite.size:
            parts.append(finite)
    if include_zero:
        parts.append(np.array([0.0]))
    if not parts:
        raise ValueError("Cannot determine limits from empty data")
    values = np.concatenate(parts)
    low, high = float(np.min(values)), float(np.max(values))
    span = high - low
    pad = max(1e-9, 0.08 * span, 0.02 * max(abs(low), abs(high), 1e-9))
    ax.set_ylim(low - pad, high + pad)


def _potential_norm(values: np.ndarray) -> TwoSlopeNorm:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Potential field has no finite values")
    maximum = float(np.max(np.abs(finite)))
    limit = max(1.0, 10.0 * math.ceil(maximum / 10.0))
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def _log_norm(*arrays: np.ndarray) -> LogNorm:
    parts = [array[np.isfinite(array) & (array > 0.0)] for array in arrays]
    parts = [part for part in parts if part.size]
    if not parts:
        raise ValueError("Concentration fields have no positive finite values")
    values = np.concatenate(parts)
    low, high = float(np.min(values)), float(np.max(values))
    if math.isclose(low, high, rel_tol=1e-12, abs_tol=0.0):
        low, high = low / 1.25, high * 1.25
    return LogNorm(vmin=low, vmax=high)


def _current_scale(*arrays: np.ndarray) -> tuple[float, int]:
    parts = [np.abs(array[np.isfinite(array)]) for array in arrays]
    parts = [part for part in parts if part.size]
    if not parts:
        return 1.0, 0
    maximum = float(np.max(np.concatenate(parts)))
    if maximum == 0.0 or 1.0e-2 <= maximum < 1.0e3:
        return 1.0, 0
    exponent = int(3 * math.floor(math.log10(maximum) / 3.0))
    return 10.0**exponent, exponent


def _current_ylabel(exponent: int) -> str:
    if exponent == 0:
        return r"$i(x)$ (A m$^{-2}$)"
    return rf"$i(x)$ ($10^{{{exponent}}}$ A m$^{{-2}}$)"


def _summary_plot_values(
    no_values: np.ndarray,
    with_values: np.ndarray,
    *,
    metric: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Collapse identical w/o-EDL values to one explicitly common baseline."""

    no_values = np.asarray(no_values, dtype=float)
    with_values = np.asarray(with_values, dtype=float)
    if no_values.ndim != 1 or with_values.ndim != 1 or no_values.size != with_values.size:
        raise ValueError("Summary series must be one-dimensional arrays of equal length")
    if no_values.size == 0 or not np.all(np.isfinite(no_values)):
        raise ValueError("Summary w/o-EDL values must be non-empty and finite")
    if not np.all(np.isfinite(with_values)):
        raise ValueError("Summary with-EDL values must be finite")
    if not np.allclose(no_values, no_values[0], rtol=1.0e-12, atol=1.0e-12):
        spread = float(np.max(no_values) - np.min(no_values))
        raise ValueError(
            f"Cannot use one common w/o-EDL {metric} baseline; "
            f"the four-case spread is {spread:.6g}"
        )

    values = np.concatenate(([float(np.mean(no_values))], with_values))
    conditions = ("without",) + ("with",) * with_values.size
    return values, conditions


def _summary_bar(
    cases: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    metric: str,
) -> list[Path]:
    case_labels = [_case_label(case) for case in cases]
    if metric == "E":
        no_values = np.array([_finite_scalar(case, "E_no_V") for case in cases])
        with_values = np.array([_finite_scalar(case, "E_with_V") for case in cases])
        ylabel = r"$E_{\mathrm{mix}}$ (V)"
        title = "Mixed potential comparison"
        stem = "summary_E_mix_comparison"
        value_format = ".3f"
    elif metric == "I":
        no_values = 1.0e12 * np.array([_finite_scalar(case, "I_no_A") for case in cases])
        with_values = 1.0e12 * np.array([_finite_scalar(case, "I_with_A") for case in cases])
        ylabel = r"$I_{\mathrm{mix}}$ (pA)"
        title = "Area-matched absolute mixed current comparison"
        stem = "summary_I_mix_absolute_comparison"
        value_format = ".3g"
    else:
        raise ValueError(f"Unsupported summary metric {metric!r}")

    values, _conditions = _summary_plot_values(no_values, with_values, metric=metric)
    labels = ["Common\nw/o EDL", *case_labels]
    positions = np.concatenate(([0.0], 1.35 + np.arange(len(cases), dtype=float)))
    width = 0.62

    fig_width = max(7.4, 1.45 * len(values))
    fig, ax = plt.subplots(figsize=(fig_width, 3.9))
    bars_no = ax.bar(
        positions[:1],
        values[:1],
        width,
        color=COLORS["without"],
        edgecolor=COLORS["dark"],
        linewidth=0.8,
        hatch="//",
        label="w/o EDL",
    )
    bars_with = ax.bar(
        positions[1:],
        values[1:],
        width,
        color=COLORS["with"],
        edgecolor=COLORS["dark"],
        linewidth=0.8,
        label="with EDL",
    )
    _style_axis(ax, "", ylabel, title)
    ax.set_xticks(positions, labels)
    ax.tick_params(axis="x", labelsize=8.2, pad=5)
    ax.legend(loc="upper right", ncols=2, fontsize=8.4, handlelength=1.8)
    low = min(0.0, float(np.min(values)))
    high = max(0.0, float(np.max(values)))
    span = max(high - low, 1e-12)
    ax.set_ylim(low - 0.03 * span, high + 0.20 * span)
    for bars in (bars_no, bars_with):
        ax.bar_label(bars, fmt=f"%{value_format}", padding=3, fontsize=7.4)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=COLORS["light_gray"], lw=0.55, alpha=0.55)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.24, top=0.90)
    return _save_pair(fig, output_dir, stem, transparent=True)


def generate_summary_figures(cases: Any, output_root: str | Path) -> list[Path]:
    """Generate one common w/o-EDL bar and four with-EDL case bars.

    ``I_with_A`` and ``I_no_A`` are assumed to have already been normalized
    by the case layer to one common Au(4 nm)+Pd(4 nm) comparison area.
    """

    case_list = _case_sequence(cases)
    output_dir = Path(output_root)
    saved: list[Path] = []
    with matplotlib.rc_context(PUBLICATION_RCPARAMS):
        saved.extend(_summary_bar(case_list, output_dir, metric="E"))
        saved.extend(_summary_bar(case_list, output_dir, metric="I"))
    _validate_artifact_list(saved, expected_pairs=2)
    return saved


def _plot_panel_a(case: Mapping[str, Any], output_dir: Path) -> list[Path]:
    e_values = np.array([_finite_scalar(case, "E_no_V"), _finite_scalar(case, "E_with_V")])
    raw_current = case.get("raw_current")
    if not isinstance(raw_current, Mapping):
        raise TypeError("Case is missing its native/source-cell current record")
    i_values = 1.0e12 * np.array(
        [float(raw_current["no_A"]), float(raw_current["with_A"])], dtype=float
    )
    labels = ["w/o EDL", "with EDL"]
    colors = [COLORS["without"], COLORS["with"]]
    positions = np.arange(2, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(4.5, 2.9))
    specs = (
        (axes[0], e_values, r"$E_{\mathrm{mix}}$ (V)", r"$E_{\mathrm{mix}}$", ".3f"),
        (
            axes[1],
            i_values,
            r"$I_{\mathrm{mix}}$ (pA)",
            r"$I_{\mathrm{mix}}$ (native cell)",
            ".3g",
        ),
    )
    for ax, values, ylabel, title, fmt in specs:
        bars = ax.bar(
            positions,
            values,
            width=0.58,
            color=colors,
            edgecolor=COLORS["dark"],
            linewidth=0.8,
        )
        _style_axis(ax, "", ylabel, title)
        ax.set_xticks(positions, labels, rotation=38, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.55, 1.55)
        low, high = min(0.0, float(np.min(values))), max(0.0, float(np.max(values)))
        span = max(high - low, 1e-12)
        ax.set_ylim(low - 0.04 * span, high + 0.22 * span)
        ax.bar_label(bars, fmt=f"%{fmt}", padding=3, fontsize=7.5)
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.29, top=0.84, wspace=0.57)
    return _save_pair(
        fig,
        output_dir,
        f"figure_3_panel_a_emix_imix_absolute_{_case_tag(case)}",
        transparent=True,
    )


def _profile_figure() -> tuple[plt.Figure, plt.Axes]:
    """Return a compact profile layout without a redundant material lane."""

    return plt.subplots(figsize=(5.8, 3.0))


def _segment_edges(segments: Sequence[_Segment]) -> list[float]:
    return [segments[0].start_nm, *(segment.stop_nm for segment in segments)]


def _add_boundaries(ax: plt.Axes, segments: Sequence[_Segment]) -> None:
    for position in _segment_edges(segments)[1:-1]:
        ax.axvline(position, color=COLORS["gray"], lw=0.8, ls=(0, (3, 2)), alpha=0.82)


def _set_spatial_x(ax: plt.Axes, segments: Sequence[_Segment], *, labels: bool = True) -> None:
    edges = _segment_edges(segments)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xticks(edges)
    if not labels:
        ax.tick_params(labelbottom=False)


def _add_material_lane(ax: plt.Axes, segments: Sequence[_Segment]) -> None:
    total = segments[-1].stop_nm - segments[0].start_nm
    for segment in segments:
        text_color = COLORS["dark"] if segment.material == "Au" else "white"
        ax.add_patch(
            Rectangle(
                (segment.start_nm, 0.0),
                segment.width_nm,
                1.0,
                facecolor=COLORS[segment.material],
                edgecolor="white",
                lw=0.8,
            )
        )
        if segment.width_nm / total >= 0.055:
            ax.text(
                0.5 * (segment.start_nm + segment.stop_nm),
                0.5,
                segment.material,
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.2,
            )
    ax.set_xlim(segments[0].start_nm, segments[-1].stop_nm)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()
    ax.text(
        0.5,
        -0.55,
        r"$x$ (nm)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        clip_on=False,
    )


def _finish_continuous_profile(
    fig: plt.Figure,
    ax: plt.Axes,
    segments: Sequence[_Segment],
) -> None:
    _add_boundaries(ax, segments)
    _set_spatial_x(ax, segments)
    ax.set_xlabel(r"$x$ (nm)")
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.19, top=0.90)


def _plot_continuous_b(
    case: Mapping[str, Any],
    surface: _ContinuousSurface,
    segments: Sequence[_Segment],
    output_dir: Path,
) -> list[Path]:
    fig, ax = _profile_figure()
    ax.plot(surface.x_nm, surface.phi_with_V, color=COLORS["with"], lw=2.0, label="with EDL")
    ax.plot(
        surface.x_nm,
        surface.phi_no_V,
        color=COLORS["without"],
        lw=1.8,
        ls=(0, (4, 2)),
        label="w/o EDL",
    )
    _style_axis(ax, "", r"$\phi_{\mathrm{RP}}(x)$ (V)", "Reaction-plane potential")
    _finite_ylim(ax, surface.phi_with_V, surface.phi_no_V)
    ax.legend(loc="best", fontsize=8.1, handlelength=2.0)
    _finish_continuous_profile(fig, ax, segments)
    return _save_pair(
        fig,
        output_dir,
        f"figure_3_panel_b_reaction_plane_potential_{_case_tag(case)}",
        transparent=True,
    )


def _plot_continuous_c(
    case: Mapping[str, Any],
    surface: _ContinuousSurface,
    segments: Sequence[_Segment],
    output_dir: Path,
) -> list[Path]:
    fig, ax = _profile_figure()
    series = (
        (surface.c_r1_with, COLORS["with"], "-", r"$\mathrm{Red}_1^-$, with EDL"),
        (surface.c_r1_no, COLORS["without"], (0, (4, 2)), r"$\mathrm{Red}_1^-$, w/o EDL"),
        (surface.c_o2_with, COLORS["with_gold"], "-", r"$\mathrm{Ox}_2^+$, with EDL"),
        (surface.c_o2_no, COLORS["without_alt"], (0, (4, 2)), r"$\mathrm{Ox}_2^+$, w/o EDL"),
    )
    positives: list[np.ndarray] = []
    for values, color, linestyle, label in series:
        plotted = np.where(np.isfinite(values) & (values > 0.0), values, np.nan)
        positives.append(plotted)
        ax.plot(surface.x_nm, plotted, color=color, lw=1.8, ls=linestyle, label=label)
    finite = np.concatenate([values[np.isfinite(values)] for values in positives])
    ax.set_yscale("log")
    ax.set_ylim(float(np.min(finite)) / 1.25, float(np.max(finite)) * 1.25)
    _style_axis(ax, "", r"$c_i/c_{\mathrm{bulk}}$", "Reactant concentration at RP")
    ax.legend(loc="best", fontsize=7.1, handlelength=1.7, ncols=2, columnspacing=0.9)
    _finish_continuous_profile(fig, ax, segments)
    return _save_pair(
        fig,
        output_dir,
        f"figure_3_panel_c_reactant_concentration_{_case_tag(case)}",
        transparent=True,
    )


def _mask(surface: _ContinuousSurface, material: str) -> np.ndarray:
    return np.asarray(surface.material == material, dtype=bool)


def _plot_continuous_d(
    case: Mapping[str, Any],
    surface: _ContinuousSurface,
    segments: Sequence[_Segment],
    output_dir: Path,
) -> list[Path]:
    mask_au, mask_pd = _mask(surface, "Au"), _mask(surface, "Pd")
    fig, ax = _profile_figure()
    plotted: list[np.ndarray] = []
    for condition, color, linestyle, au, pd in (
        ("with EDL", COLORS["with"], "-", surface.eta_au_with_V, surface.eta_pd_with_V),
        ("w/o EDL", COLORS["without"], (0, (4, 2)), surface.eta_au_no_V, surface.eta_pd_no_V),
    ):
        au_values = np.where(mask_au, au, np.nan)
        pd_values = np.where(mask_pd, pd, np.nan)
        plotted.extend((au_values, pd_values))
        ax.plot(surface.x_nm, au_values, color=color, lw=1.9, ls=linestyle, label=condition)
        ax.plot(surface.x_nm, pd_values, color=color, lw=1.9, ls=linestyle)
    _style_axis(ax, "", r"$\eta(x)$ (V)", "Overpotential at RP")
    _finite_ylim(ax, *plotted)
    ax.legend(loc="best", fontsize=8.1, handlelength=2.0)
    _finish_continuous_profile(fig, ax, segments)
    return _save_pair(
        fig,
        output_dir,
        f"figure_3_panel_d_overpotential_{_case_tag(case)}",
        transparent=True,
    )


def _plot_continuous_e(
    case: Mapping[str, Any],
    surface: _ContinuousSurface,
    segments: Sequence[_Segment],
    output_dir: Path,
) -> list[Path]:
    mask_au, mask_pd = _mask(surface, "Au"), _mask(surface, "Pd")
    raw = (
        np.where(mask_au, surface.j_au_with, np.nan),
        np.where(mask_au, surface.j_au_no, np.nan),
        np.where(mask_pd, surface.j_pd_with, np.nan),
        np.where(mask_pd, surface.j_pd_no, np.nan),
    )
    scale, exponent = _current_scale(*raw)
    au_with, au_no, pd_with, pd_no = (values / scale for values in raw)
    fig, ax = _profile_figure()
    ax.axhline(0.0, color=COLORS["dark"], lw=0.6, alpha=0.78)
    ax.plot(surface.x_nm, au_with, color=COLORS["current_Au"], lw=2.0, label="Au with EDL")
    ax.plot(surface.x_nm, au_no, color=COLORS["current_Au"], lw=1.7, ls=(0, (4, 2)), label="Au w/o EDL")
    ax.plot(surface.x_nm, pd_with, color=COLORS["current_Pd"], lw=2.0, label="Pd with EDL")
    ax.plot(surface.x_nm, pd_no, color=COLORS["current_Pd"], lw=1.7, ls=(0, (4, 2)), label="Pd w/o EDL")
    _style_axis(ax, "", _current_ylabel(exponent), "Current density at RP")
    _finite_ylim(ax, au_with, au_no, pd_with, pd_no, include_zero=True)
    ax.legend(loc="best", fontsize=7.2, handlelength=1.8, ncols=2, columnspacing=0.9)
    _finish_continuous_profile(fig, ax, segments)
    return _save_pair(
        fig,
        output_dir,
        f"figure_3_panel_e_current_density_{_case_tag(case)}",
        transparent=True,
    )


def _independent_records(case: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = _read(case, "independent_surfaces")
    if not isinstance(raw, Mapping):
        raise TypeError("independent_surfaces must be a mapping keyed by Au and Pd")
    records: dict[str, Mapping[str, Any]] = {}
    for key, value in raw.items():
        material = _normalise_material(key)
        if material in {"Au", "Pd"}:
            if not isinstance(value, Mapping):
                raise TypeError(f"independent_surfaces[{key!r}] must be a mapping")
            records[material] = value
    if set(records) != {"Au", "Pd"}:
        raise ValueError("independent_surfaces must contain exactly Au and Pd")
    return records


def _ind_x(record: Mapping[str, Any]) -> np.ndarray:
    surface = _read(record, "surface", default=record)
    return _as_1d(_read(surface, "x_nm"), "independent surface x_nm")


def _ind_array(record: Mapping[str, Any], x: np.ndarray, *names: str, default: Any = _MISSING) -> np.ndarray:
    surface = _read(record, "surface", default=record)
    return _broadcast_1d(
        _read(surface, *names, default=_MISSING),
        x.size,
        names[0],
        default=default,
    )


def _independent_profile_layout(title: str, ylabel: str) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(1, 2, figsize=(6.15, 2.9), sharey=True)
    for ax, material in zip(axes, ("Au", "Pd"), strict=True):
        _style_axis(ax, r"$x$ (nm)", ylabel if material == "Au" else "", f"{material} interface")
    fig.suptitle(title, x=0.08, y=0.985, ha="left", fontsize=10.5)
    return fig, axes


def _plot_independent_b(case: Mapping[str, Any], output_dir: Path) -> list[Path]:
    records = _independent_records(case)
    fig, axes = _independent_profile_layout("Reaction-plane potential", r"$\phi_{\mathrm{RP}}$ (V)")
    plotted: list[np.ndarray] = []
    for ax, material in zip(axes, ("Au", "Pd"), strict=True):
        record, x = records[material], _ind_x(records[material])
        with_values = _ind_array(record, x, "phi_RP_with_V", "phi_rp_with_V", "phi_with_V")
        no_values = _ind_array(record, x, "phi_RP_no_V", "phi_rp_no_V", "phi_no_V", default=0.0)
        plotted.extend((with_values, no_values))
        ax.plot(x, with_values, color=COLORS["with"], lw=2.0, label="with EDL")
        ax.plot(x, no_values, color=COLORS["without"], lw=1.8, ls=(0, (4, 2)), label="w/o EDL")
        ax.set_xlim(float(x[0]), float(x[-1]))
    low = min(float(np.nanmin(values)) for values in plotted)
    high = max(float(np.nanmax(values)) for values in plotted)
    span = max(high - low, 1e-9)
    axes[0].set_ylim(low - 0.08 * span, high + 0.08 * span)
    axes[1].legend(loc="best", fontsize=8.0, handlelength=2.0)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.19, top=0.82, wspace=0.12)
    return _save_pair(fig, output_dir, f"figure_3_panel_b_reaction_plane_potential_{_case_tag(case)}", transparent=True)


def _plot_independent_c(case: Mapping[str, Any], output_dir: Path) -> list[Path]:
    records = _independent_records(case)
    fig, axes = _independent_profile_layout("Reactant concentration at RP", r"$c_i/c_{\mathrm{bulk}}$")
    all_values: list[np.ndarray] = []
    specs = {
        "Au": ("c_R1_with", "c_R1_no", r"$\mathrm{Red}_1^-$"),
        "Pd": ("c_O2_with", "c_O2_no", r"$\mathrm{Ox}_2^+$"),
    }
    for ax, material in zip(axes, ("Au", "Pd"), strict=True):
        record, x = records[material], _ind_x(records[material])
        with_name, no_name, species = specs[material]
        with_values = _ind_array(record, x, with_name, with_name.lower())
        no_values = _ind_array(record, x, no_name, no_name.lower(), default=1.0)
        all_values.extend((with_values, no_values))
        ax.plot(x, with_values, color=COLORS["with"], lw=2.0, label=f"{species}, with EDL")
        ax.plot(x, no_values, color=COLORS["without"], lw=1.8, ls=(0, (4, 2)), label=f"{species}, w/o EDL")
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.set_yscale("log")
        ax.legend(loc="best", fontsize=7.4, handlelength=1.8)
    positive = np.concatenate([values[np.isfinite(values) & (values > 0.0)] for values in all_values])
    axes[0].set_ylim(float(np.min(positive)) / 1.25, float(np.max(positive)) * 1.25)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.19, top=0.82, wspace=0.12)
    return _save_pair(fig, output_dir, f"figure_3_panel_c_reactant_concentration_{_case_tag(case)}", transparent=True)


def _plot_independent_d(case: Mapping[str, Any], output_dir: Path) -> list[Path]:
    records = _independent_records(case)
    fig, axes = _independent_profile_layout("Overpotential at RP", r"$\eta$ (V)")
    plotted: list[np.ndarray] = []
    for ax, material in zip(axes, ("Au", "Pd"), strict=True):
        record, x = records[material], _ind_x(records[material])
        lower = material.lower()
        with_values = _ind_array(record, x, f"eta_{material}_with_V", f"eta_{lower}_with_V", "eta_with_V")
        no_values = _ind_array(record, x, f"eta_{material}_no_V", f"eta_{lower}_no_V", "eta_no_V")
        plotted.extend((with_values, no_values))
        ax.plot(x, with_values, color=COLORS["with"], lw=2.0, label="with EDL")
        ax.plot(x, no_values, color=COLORS["without"], lw=1.8, ls=(0, (4, 2)), label="w/o EDL")
        ax.set_xlim(float(x[0]), float(x[-1]))
    low = min(float(np.nanmin(values)) for values in plotted)
    high = max(float(np.nanmax(values)) for values in plotted)
    span = max(high - low, 1e-9)
    axes[0].set_ylim(low - 0.08 * span, high + 0.08 * span)
    axes[1].legend(loc="best", fontsize=8.0, handlelength=2.0)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.19, top=0.82, wspace=0.12)
    return _save_pair(fig, output_dir, f"figure_3_panel_d_overpotential_{_case_tag(case)}", transparent=True)


def _plot_independent_e(case: Mapping[str, Any], output_dir: Path) -> list[Path]:
    records = _independent_records(case)
    raw: list[np.ndarray] = []
    per_material: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for material in ("Au", "Pd"):
        record, x = records[material], _ind_x(records[material])
        lower = material.lower()
        with_values = _ind_array(
            record,
            x,
            f"j_{material}_with_A_per_m2",
            f"j_{material}_with",
            f"j_{lower}_with",
            "current_density_with_A_per_m2",
            "j_with",
        )
        no_values = _ind_array(
            record,
            x,
            f"j_{material}_no_A_per_m2",
            f"j_{material}_no",
            f"j_{lower}_no",
            "current_density_no_A_per_m2",
            "j_no",
        )
        raw.extend((with_values, no_values))
        per_material[material] = (x, with_values, no_values)
    scale, exponent = _current_scale(*raw)
    fig, axes = _independent_profile_layout("Current density at RP", _current_ylabel(exponent))
    for ax, material in zip(axes, ("Au", "Pd"), strict=True):
        x, with_values, no_values = per_material[material]
        color = COLORS[f"current_{material}"]
        ax.axhline(0.0, color=COLORS["dark"], lw=0.6, alpha=0.78)
        ax.plot(x, with_values / scale, color=color, lw=2.0, label=f"{material} with EDL")
        ax.plot(x, no_values / scale, color=color, lw=1.8, ls=(0, (4, 2)), label=f"{material} w/o EDL")
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.legend(loc="best", fontsize=7.5, handlelength=1.8)
    scaled = [values / scale for values in raw]
    low = min(0.0, *(float(np.nanmin(values)) for values in scaled))
    high = max(0.0, *(float(np.nanmax(values)) for values in scaled))
    span = max(high - low, 1e-9)
    axes[0].set_ylim(low - 0.08 * span, high + 0.08 * span)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.19, top=0.82, wspace=0.12)
    return _save_pair(fig, output_dir, f"figure_3_panel_e_current_density_{_case_tag(case)}", transparent=True)


def _parameter(case: Mapping[str, Any], *names: str, default: Any = _MISSING) -> float:
    params = _read(case, "params", default={})
    for source in (case, params):
        raw = _read(source, *names, default=_MISSING)
        if raw is not _MISSING and raw is not None:
            array = np.asarray(raw)
            if array.size == 1 and np.isfinite(float(array.reshape(-1)[0])):
                return float(array.reshape(-1)[0])
    if default is not _MISSING:
        return float(default)
    raise KeyError(f"Missing parameter {names!r}")


def _effective_equilibrium(case: Mapping[str, Any], reaction: int) -> float:
    direct_names = (
        ("E1_eq_eff_V", "E1_eq_eff"),
        ("E2_eq_eff_V", "E2_eq_eff"),
    )[reaction - 1]
    try:
        return _parameter(case, *direct_names)
    except KeyError:
        base = _parameter(case, f"E{reaction}_eq_V", f"E{reaction}_eq")
        slope = _parameter(
            case,
            f"E{reaction}_eq_pH_slope_V_per_pH",
            default=0.0,
        )
        p_h = _parameter(case, "pH", default=7.0)
        p_h_ref = _parameter(case, "pH_ref", default=p_h)
        return base + slope * (p_h - p_h_ref)


def _reference_marker(
    ax: plt.Axes,
    x: float,
    y: float,
    label: str,
    color: str,
    marker: str,
    dy: float,
    *,
    dx: float = 0.0,
) -> None:
    ax.vlines(x, y - 0.08, y + 0.08, color=color, lw=1.7)
    ax.scatter([x], [y], s=56, marker=marker, color=color, edgecolor="none", zorder=4)
    ax.text(
        x + dx,
        y + dy,
        f"{label}\n{x:.2f} V",
        ha="center",
        va="bottom" if dy >= 0.0 else "top",
        fontsize=7.8,
        color=color,
        linespacing=1.08,
    )


def _plot_panel_f(case: Mapping[str, Any], output_dir: Path) -> list[Path]:
    e1, e2 = _effective_equilibrium(case, 1), _effective_equilibrium(case, 2)
    e_no, e_with = _finite_scalar(case, "E_no_V"), _finite_scalar(case, "E_with_V")
    values: list[tuple[float, str, str]] = []
    materials = {segment.material for segment in _load_segments(case)} if not _is_independent(case) else {"Au", "Pd"}
    for material in ("C", "Pd", "Au"):
        if material not in materials:
            continue
        try:
            pzc = _parameter(case, f"pzc_{material}", f"PZC_{material}_V")
        except KeyError:
            if material != "C":
                raise
            pzc = _parameter(case, "pzc_support", "PZC_support_V")
        values.append((pzc, f"PZC {material}", COLORS[material]))

    fig, ax = plt.subplots(figsize=(5.35, 3.35))
    y_eq, y_mix, y_pzc = 0.44, 0.0, -0.44
    ax.hlines(0.0, 0.0, 1.0, color=COLORS["dark"], lw=1.0)
    _reference_marker(ax, e1, y_eq, r"$E_{1,\mathrm{eq}}$", COLORS["gray"], "o", 0.115)
    _reference_marker(ax, e2, y_eq, r"$E_{2,\mathrm{eq}}$", COLORS["gray"], "o", 0.115)
    _reference_marker(ax, e_no, y_mix, r"$E_{\mathrm{mix}}$ w/o EDL", COLORS["without"], "D", -0.135, dx=-0.05)
    _reference_marker(ax, e_with, y_mix, r"$E_{\mathrm{mix}}$ with EDL", COLORS["with"], "D", 0.12)
    pzc_offsets = {"C": 0.025, "Pd": -0.018, "Au": 0.0}
    for pzc, label, color in values:
        material = label.split()[-1]
        _reference_marker(ax, pzc, y_pzc, label, color, "^", -0.125, dx=pzc_offsets[material])
    all_x = np.array([e1, e2, e_no, e_with, *(value[0] for value in values)])
    ax.set_xlim(min(0.0, float(np.min(all_x)) - 0.04), max(1.0, float(np.max(all_x)) + 0.04))
    ax.set_ylim(-0.98, 0.79)
    ax.set_xlabel("Potential (V vs. RHE)")
    ax.set_yticks([])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis="x", length=3.4, width=0.85, labelsize=8.8)
    ax.set_title("Potential reference map", loc="left", pad=7, fontsize=10.5)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_position(("data", -0.80))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.16, top=0.90)
    return _save_pair(
        fig,
        output_dir,
        f"figure_3_panel_f_pzc_potential_reference_map_{_case_tag(case)}",
        transparent=True,
    )


def _style_map_axis(
    ax: plt.Axes,
    title: str,
    segments: Sequence[_Segment],
    *,
    show_xlabels: bool,
) -> None:
    ax.set_title(title, loc="left", pad=5, fontsize=9.5)
    ax.set_ylabel(r"$y$ (nm)")
    ax.tick_params(length=3.2, width=0.85, pad=2.5, labelsize=8.0)
    _set_spatial_x(ax, segments, labels=show_xlabels)
    _add_boundaries(ax, segments)
    for spine in ("left", "bottom", "top", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.9)


def _add_colorbar(fig: plt.Figure, mesh: Any, cax: plt.Axes, label: str) -> None:
    colorbar = fig.colorbar(mesh, cax=cax)
    colorbar.set_label(label, labelpad=5)
    colorbar.ax.tick_params(length=2.8, width=0.75, labelsize=7.7, pad=2.2)
    colorbar.outline.set_linewidth(0.8)


def _plot_continuous_potential_2d(
    case: Mapping[str, Any],
    grid: _Grid2D,
    segments: Sequence[_Segment],
    output_dir: Path,
) -> list[Path]:
    fig = plt.figure(figsize=(5.9, 3.25), facecolor="white")
    layout = fig.add_gridspec(2, 2, width_ratios=(1.0, 0.038), height_ratios=(1.0, 0.11), hspace=0.20, wspace=0.08)
    ax, cax = fig.add_subplot(layout[0, 0]), fig.add_subplot(layout[0, 1])
    lane_ax = fig.add_subplot(layout[1, 0], sharex=ax)
    fig.add_subplot(layout[1, 1]).set_axis_off()
    mesh = ax.pcolormesh(grid.x_nm, grid.y_nm, grid.phi_s_mV, shading="auto", cmap="RdBu_r", norm=_potential_norm(grid.phi_s_mV), rasterized=True)
    _style_map_axis(ax, rf"Solution-phase potential, $E_{{\mathrm{{mix}}}}$ = {_finite_scalar(case, 'E_with_V'):.2f} V", segments, show_xlabels=True)
    ax.set_ylim(float(grid.y_nm[0]), float(grid.y_nm[-1]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    _add_colorbar(fig, mesh, cax, r"$\mathit{\Phi}_{\mathrm{s}}$ (mV)")
    _add_material_lane(lane_ax, segments)
    fig.subplots_adjust(left=0.11, right=0.92, bottom=0.13, top=0.89)
    return _save_pair(fig, output_dir, f"solution_phase_potential_2d_{_case_tag(case)}", transparent=False)


def _plot_continuous_composite_2d(
    case: Mapping[str, Any],
    grid: _Grid2D,
    segments: Sequence[_Segment],
    output_dir: Path,
) -> list[Path]:
    fig = plt.figure(figsize=(5.9, 7.0), facecolor="white")
    layout = fig.add_gridspec(4, 2, width_ratios=(1.0, 0.038), height_ratios=(1.0, 1.0, 1.0, 0.11), hspace=0.20, wspace=0.08)
    axes = [fig.add_subplot(layout[row, 0]) for row in range(3)]
    caxes = [fig.add_subplot(layout[row, 1]) for row in range(3)]
    lane_ax = fig.add_subplot(layout[3, 0], sharex=axes[-1])
    fig.add_subplot(layout[3, 1]).set_axis_off()
    c_norm = _log_norm(grid.c_r1, grid.c_o2)
    specs = (
        (grid.phi_s_mV, "RdBu_r", _potential_norm(grid.phi_s_mV), rf"Solution-phase potential, $E_{{\mathrm{{mix}}}}$ = {_finite_scalar(case, 'E_with_V'):.2f} V", r"$\mathit{\Phi}_{\mathrm{s}}$ (mV)"),
        (grid.c_r1, "viridis", c_norm, r"$\mathrm{Red}_1^-$ distribution", r"$c(\mathrm{Red}_1^-)/c_{\mathrm{bulk}}$"),
        (grid.c_o2, "viridis", c_norm, r"$\mathrm{Ox}_2^+$ distribution", r"$c(\mathrm{Ox}_2^+)/c_{\mathrm{bulk}}$"),
    )
    for index, (ax, cax, spec) in enumerate(zip(axes, caxes, specs, strict=True)):
        values, cmap, norm, title, label = spec
        mesh = ax.pcolormesh(grid.x_nm, grid.y_nm, values, shading="auto", cmap=cmap, norm=norm, rasterized=True)
        _style_map_axis(ax, title, segments, show_xlabels=index == 2)
        ax.set_ylim(float(grid.y_nm[0]), float(grid.y_nm[-1]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        _add_colorbar(fig, mesh, cax, label)
    _add_material_lane(lane_ax, segments)
    fig.subplots_adjust(left=0.11, right=0.92, bottom=0.07, top=0.96)
    return _save_pair(fig, output_dir, f"solution_phase_potential_and_reactants_2d_{_case_tag(case)}", transparent=False)


def _plot_continuous_charge(
    case: Mapping[str, Any],
    charges: Sequence[_ChargeSegment],
    segments: Sequence[_Segment],
    output_dir: Path,
) -> list[Path]:
    fig, ax = _profile_figure()
    plotted: list[np.ndarray] = []
    seen: set[str] = set()
    for segment in charges:
        values = 100.0 * segment.sigma_C_per_m2
        plotted.append(values)
        label = segment.material if segment.material not in seen else None
        seen.add(segment.material)
        ax.plot(segment.x_nm, values, color=COLORS[segment.material], lw=2.0, label=label)
    ax.axhline(0.0, color=COLORS["dark"], lw=0.65, alpha=0.8)
    _style_axis(ax, "", r"$\sigma(x)$ (µC cm$^{-2}$)", "Surface charge distribution")
    _finite_ylim(ax, *plotted, include_zero=True)
    ax.legend(loc="best", fontsize=7.8, handlelength=1.8, ncols=3)
    _finish_continuous_profile(fig, ax, segments)
    return _save_pair(fig, output_dir, f"surface_charge_distribution_{_case_tag(case)}", transparent=True)


def _independent_depth_fields(
    case: Mapping[str, Any],
    material: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    record = _independent_records(case)[material]
    surface = _read(record, "surface", default=record)
    grid = _read(record, "grid_2d", default=record)
    x_nm = _as_1d(_read(grid, "x_nm", default=_read(surface, "x_nm")), "independent grid x_nm")
    y_raw = _read(grid, "y_nm", default=_MISSING)
    if y_raw is _MISSING:
        lambda_nm = 1.0e9 * _parameter(case, "lambda_D", default=3.041217889e-9)
        y_nm = np.linspace(0.0, 5.0 * lambda_nm, 320)
    else:
        y_nm = _as_1d(y_raw, f"independent_surfaces.{material}.y_nm")
    phi_raw = _read(
        grid,
        "phi_s_with_mV",
        "phi_s_mV",
        "phi_s_mV_y",
        "phi_mV_y",
        default=_MISSING,
    )
    if phi_raw is _MISSING:
        phi0 = _finite_scalar(surface, "phi_RP_with_V", "phi_rp_with_V", "phi_with_V")
        lambda_nm = 1.0e9 * _parameter(case, "lambda_D", default=3.041217889e-9)
        phi_y = 1.0e3 * phi0 * np.exp(-y_nm / lambda_nm)
    else:
        phi_array = np.asarray(phi_raw, dtype=float)
        if phi_array.shape == (x_nm.size, y_nm.size) and phi_array.shape != (
            y_nm.size,
            x_nm.size,
        ):
            phi_array = phi_array.T
        if phi_array.shape == (y_nm.size, x_nm.size):
            phi_2d = phi_array
        else:
            phi_y = _broadcast_1d(phi_array, y_nm.size, "phi_s_mV_y")
            phi_2d = np.broadcast_to(phi_y[:, None], (y_nm.size, x_nm.size)).copy()
    if phi_raw is _MISSING:
        phi_2d = np.broadcast_to(phi_y[:, None], (y_nm.size, x_nm.size)).copy()
    c_r1_raw = _read(grid, "c_R1_with", "c_R1_norm", default=_MISSING)
    c_o2_raw = _read(grid, "c_O2_with", "c_O2_norm", default=_MISSING)
    if c_r1_raw is _MISSING or c_o2_raw is _MISSING:
        beta = _parameter(case, "beta", "beta_per_V", default=38.943368749727554)
        phi_tilde = beta * phi_2d / 1.0e3
        c_r1 = np.exp(phi_tilde)
        c_o2 = np.exp(-phi_tilde)
    else:
        c_r1 = np.asarray(c_r1_raw, dtype=float)
        c_o2 = np.asarray(c_o2_raw, dtype=float)
        if c_r1.shape == (x_nm.size, y_nm.size) and c_r1.shape != (y_nm.size, x_nm.size):
            c_r1 = c_r1.T
        if c_o2.shape == (x_nm.size, y_nm.size) and c_o2.shape != (y_nm.size, x_nm.size):
            c_o2 = c_o2.T
        if c_r1.shape != phi_2d.shape or c_o2.shape != phi_2d.shape:
            raise ValueError("Independent concentration grids do not match the potential grid")
    return x_nm, y_nm, phi_2d, c_r1, c_o2


def _add_independent_material_lane(
    ax: plt.Axes,
    material: str,
    x_min_nm: float,
    x_max_nm: float,
) -> None:
    text_color = COLORS["dark"] if material == "Au" else "white"
    ax.add_patch(
        Rectangle(
            (x_min_nm, 0.0),
            x_max_nm - x_min_nm,
            1.0,
            facecolor=COLORS[material],
            edgecolor="white",
            lw=0.7,
        )
    )
    ax.text(
        0.5 * (x_min_nm + x_max_nm),
        0.5,
        material,
        ha="center",
        va="center",
        fontsize=8.0,
        color=text_color,
    )
    ax.set_xlim(x_min_nm, x_max_nm)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()
    ax.text(
        0.5,
        -0.55,
        r"$x$ (nm)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        clip_on=False,
    )


def _plot_independent_potential_2d(case: Mapping[str, Any], output_dir: Path) -> list[Path]:
    fields = {material: _independent_depth_fields(case, material) for material in ("Au", "Pd")}
    norm = _potential_norm(np.concatenate([fields[m][2].ravel() for m in ("Au", "Pd")]))
    fig = plt.figure(figsize=(6.3, 3.45), facecolor="white")
    layout = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.0, 1.0, 0.045),
        height_ratios=(1.0, 0.085),
        hspace=0.24,
        wspace=0.20,
    )
    axes = [fig.add_subplot(layout[0, index]) for index in range(2)]
    lane_axes = [fig.add_subplot(layout[1, index], sharex=axes[index]) for index in range(2)]
    cax = fig.add_subplot(layout[0, 2])
    fig.add_subplot(layout[1, 2]).set_axis_off()
    mesh = None
    for ax, lane_ax, material in zip(axes, lane_axes, ("Au", "Pd"), strict=True):
        x, y, phi, _c1, _c2 = fields[material]
        mesh = ax.pcolormesh(x, y, phi, shading="auto", cmap="RdBu_r", norm=norm, rasterized=True)
        _style_axis(ax, "", r"$y$ (nm)" if material == "Au" else "", f"{material} half-space")
        if material == "Pd":
            ax.tick_params(labelleft=False)
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.set_ylim(float(y[0]), float(y[-1]))
        _add_independent_material_lane(lane_ax, material, float(x[0]), float(x[-1]))
    assert mesh is not None
    _add_colorbar(fig, mesh, cax, r"$\mathit{\Phi}_{\mathrm{s}}$ (mV)")
    fig.suptitle(rf"Solution-phase potential, $E_{{\mathrm{{mix}}}}$ = {_finite_scalar(case, 'E_with_V'):.2f} V", x=0.08, y=0.99, ha="left", fontsize=10.5)
    fig.subplots_adjust(left=0.10, right=0.91, bottom=0.13, top=0.84)
    return _save_pair(fig, output_dir, f"solution_phase_potential_2d_{_case_tag(case)}", transparent=False)


def _plot_independent_composite_2d(case: Mapping[str, Any], output_dir: Path) -> list[Path]:
    fields = {material: _independent_depth_fields(case, material) for material in ("Au", "Pd")}
    p_norm = _potential_norm(np.concatenate([fields[m][2].ravel() for m in ("Au", "Pd")]))
    c_norm = _log_norm(*(field for material in ("Au", "Pd") for field in fields[material][3:5]))
    fig = plt.figure(figsize=(6.6, 7.0), facecolor="white")
    layout = fig.add_gridspec(
        4,
        3,
        width_ratios=(1.0, 1.0, 0.045),
        height_ratios=(1.0, 1.0, 1.0, 0.085),
        hspace=0.24,
        wspace=0.18,
    )
    axes = np.empty((3, 2), dtype=object)
    caxes = [fig.add_subplot(layout[row, 2]) for row in range(3)]
    lane_axes = [fig.add_subplot(layout[3, col]) for col in range(2)]
    fig.add_subplot(layout[3, 2]).set_axis_off()
    row_specs = (
        (2, "RdBu_r", p_norm, r"$\mathit{\Phi}_{\mathrm{s}}$ (mV)", "Solution-phase potential"),
        (3, "viridis", c_norm, r"$c(\mathrm{Red}_1^-)/c_{\mathrm{bulk}}$", r"$\mathrm{Red}_1^-$ distribution"),
        (4, "viridis", c_norm, r"$c(\mathrm{Ox}_2^+)/c_{\mathrm{bulk}}$", r"$\mathrm{Ox}_2^+$ distribution"),
    )
    for row, (field_index, cmap, norm, label, row_title) in enumerate(row_specs):
        mesh = None
        for col, material in enumerate(("Au", "Pd")):
            ax = fig.add_subplot(layout[row, col])
            axes[row, col] = ax
            x, y = fields[material][0], fields[material][1]
            values = fields[material][field_index]
            mesh = ax.pcolormesh(x, y, values, shading="auto", cmap=cmap, norm=norm, rasterized=True)
            _style_axis(ax, "", r"$y$ (nm)" if col == 0 else "", f"{row_title}: {material}")
            if col == 1:
                ax.tick_params(labelleft=False)
            ax.set_xlim(float(x[0]), float(x[-1]))
            ax.set_ylim(float(y[0]), float(y[-1]))
            if row != 2:
                ax.tick_params(labelbottom=False)
        assert mesh is not None
        _add_colorbar(fig, mesh, caxes[row], label)
    for lane_ax, material in zip(lane_axes, ("Au", "Pd"), strict=True):
        x = fields[material][0]
        _add_independent_material_lane(lane_ax, material, float(x[0]), float(x[-1]))
    fig.subplots_adjust(left=0.10, right=0.91, bottom=0.07, top=0.97)
    return _save_pair(fig, output_dir, f"solution_phase_potential_and_reactants_2d_{_case_tag(case)}", transparent=False)


def _plot_independent_charge(case: Mapping[str, Any], output_dir: Path) -> list[Path]:
    records = _independent_records(case)
    fig, axes = _independent_profile_layout("Surface charge distribution", r"$\sigma$ (µC cm$^{-2}$)")
    plotted: list[np.ndarray] = []
    for ax, material in zip(axes, ("Au", "Pd"), strict=True):
        record, x = records[material], _ind_x(records[material])
        charge = _read(record, "charge", default=record)
        sigma = _broadcast_1d(
            _read(charge, "sigma_C_per_m2", "sigma_with_C_per_m2", "sigma"),
            x.size,
            "sigma_C_per_m2",
        )
        scaled = 100.0 * sigma
        plotted.append(scaled)
        ax.axhline(0.0, color=COLORS["dark"], lw=0.65, alpha=0.8)
        ax.plot(x, scaled, color=COLORS[material], lw=2.0, label=material)
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.legend(loc="best", fontsize=8.0)
    low = min(0.0, *(float(np.nanmin(values)) for values in plotted))
    high = max(0.0, *(float(np.nanmax(values)) for values in plotted))
    span = max(high - low, 1e-9)
    axes[0].set_ylim(low - 0.08 * span, high + 0.08 * span)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.19, top=0.82, wspace=0.12)
    return _save_pair(fig, output_dir, f"surface_charge_distribution_{_case_tag(case)}", transparent=True)


def _validate_artifact_list(paths: Sequence[Path], *, expected_pairs: int) -> None:
    expected = 2 * expected_pairs
    if len(paths) != expected:
        raise RuntimeError(f"Expected {expected} figure artifacts; generated {len(paths)}")
    pngs = [path for path in paths if path.suffix.lower() == ".png"]
    svgs = [path for path in paths if path.suffix.lower() == ".svg"]
    if len(pngs) != expected_pairs or len(svgs) != expected_pairs:
        raise RuntimeError(
            f"Expected {expected_pairs} PNG and SVG files; got {len(pngs)} and {len(svgs)}"
        )
    if any(path.suffix.lower() == ".pdf" for path in paths):
        raise RuntimeError("PDF output is forbidden")
    missing = [path for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing or empty figure files: {missing}")
    noneditable = [path for path in svgs if "<text" not in path.read_text(encoding="utf-8")]
    if noneditable:
        raise RuntimeError(f"SVG text is not editable in: {noneditable}")


def generate_case_figures(case: Mapping[str, Any], output_root: str | Path) -> list[Path]:
    """Generate six Figure 3 and three reaction-plane figure pairs for one case."""

    if not isinstance(case, Mapping):
        raise TypeError("case must be a mapping")
    output = Path(output_root)
    figure_3 = output / "Figure_3"
    figure_rp = output / "Figure_RP"
    saved: list[Path] = []
    with matplotlib.rc_context(PUBLICATION_RCPARAMS):
        saved.extend(_plot_panel_a(case, figure_3))
        if _is_independent(case):
            saved.extend(_plot_independent_b(case, figure_3))
            saved.extend(_plot_independent_c(case, figure_3))
            saved.extend(_plot_independent_d(case, figure_3))
            saved.extend(_plot_independent_e(case, figure_3))
            saved.extend(_plot_panel_f(case, figure_3))
            saved.extend(_plot_independent_potential_2d(case, figure_rp))
            saved.extend(_plot_independent_composite_2d(case, figure_rp))
            saved.extend(_plot_independent_charge(case, figure_rp))
        else:
            segments = _load_segments(case)
            surface = _load_surface(case)
            grid = _load_grid(case)
            charges = _load_charges(case)
            saved.extend(_plot_continuous_b(case, surface, segments, figure_3))
            saved.extend(_plot_continuous_c(case, surface, segments, figure_3))
            saved.extend(_plot_continuous_d(case, surface, segments, figure_3))
            saved.extend(_plot_continuous_e(case, surface, segments, figure_3))
            saved.extend(_plot_panel_f(case, figure_3))
            saved.extend(_plot_continuous_potential_2d(case, grid, segments, figure_rp))
            saved.extend(_plot_continuous_composite_2d(case, grid, segments, figure_rp))
            saved.extend(_plot_continuous_charge(case, charges, segments, figure_rp))
    _validate_artifact_list(saved, expected_pairs=9)
    return saved


__all__ = [
    "COLORS",
    "PUBLICATION_RCPARAMS",
    "generate_case_figures",
    "generate_summary_figures",
]
