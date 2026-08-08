"""Publication figures for the C-supported Au--Pd Janus mirror cell.

The plotting layer intentionally depends only on the public result-bundle
schema.  It does not import the solver, which keeps saved result bundles
renderable after numerical internals change.

``generate_all_figures`` writes exactly nine PNG/SVG pairs below the supplied
run directory (or below a supplied ``figures`` directory).  Dense 2D fields
are rasterized inside SVG files while labels and annotations remain editable
vector text.
"""

from __future__ import annotations

import math
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


OUTPUT_TAG = "c5_au4_pd4_c5_mirror"

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
    "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans", "sans-serif"],
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


@dataclass(frozen=True)
class _GeometrySegment:
    segment_id: str
    material: str
    start_nm: float
    stop_nm: float

    @property
    def width_nm(self) -> float:
        return self.stop_nm - self.start_nm


@dataclass(frozen=True)
class _Geometry:
    segments: tuple[_GeometrySegment, ...]
    edges_nm: tuple[float, ...]
    start_nm: float
    stop_nm: float


@dataclass(frozen=True)
class _SurfaceData:
    x_nm: np.ndarray
    segment_id: np.ndarray
    material: np.ndarray
    phi_rp_with_V: np.ndarray
    phi_rp_no_V: np.ndarray
    c_R1_with: np.ndarray
    c_O2_with: np.ndarray
    c_R1_no: np.ndarray
    c_O2_no: np.ndarray
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
    c_R1_norm: np.ndarray
    c_O2_norm: np.ndarray


@dataclass(frozen=True)
class _ChargeSegment:
    segment_id: str
    material: str
    x_nm: np.ndarray
    sigma_C_per_m2: np.ndarray


def _read(source: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(source, Mapping) and name in source:
        return source[name]
    if hasattr(source, name):
        return getattr(source, name)
    if default is not _MISSING:
        return default
    raise KeyError(name)


def _require_mapping_field(source: Any, name: str) -> Any:
    try:
        value = _read(source, name)
    except KeyError:
        raise KeyError(f"Result bundle is missing required field {name!r}")
    return value


def _first_scalar(
    sources: Sequence[Any],
    names: Sequence[str],
    *,
    default: Any = _MISSING,
) -> float:
    for source in sources:
        if source is None:
            continue
        for name in names:
            try:
                value = _read(source, name)
            except KeyError:
                continue
            if value is None:
                continue
            array = np.asarray(value)
            if array.size != 1:
                continue
            result = float(array.reshape(-1)[0])
            if np.isfinite(result):
                return result
    if default is not _MISSING:
        return float(default)
    raise KeyError(f"Could not find a finite scalar under any of: {', '.join(names)}")


def _float_1d(source: Any, name: str) -> np.ndarray:
    raw = _require_mapping_field(source, name)
    result = np.asarray(raw, dtype=float)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array; got {result.shape}")
    return result


def _object_1d(source: Any, name: str) -> np.ndarray:
    raw = _require_mapping_field(source, name)
    result = np.asarray(raw, dtype=object)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array; got {result.shape}")
    return result


def _normalise_material(value: Any) -> str:
    label = str(value).strip().lower()
    if label == "au" or label.startswith("au_"):
        return "Au"
    if label == "pd" or label.startswith("pd_"):
        return "Pd"
    if label in {"c", "support", "carbon", "c_left", "c_right"} or label.startswith("c_"):
        return "C"
    raise ValueError(f"Unsupported material label: {value!r}")


def _segment_coordinate_nm(record: Any, nm_names: Sequence[str], m_names: Sequence[str]) -> float:
    for name in nm_names:
        try:
            return float(_read(record, name))
        except KeyError:
            continue
    for name in m_names:
        try:
            return 1.0e9 * float(_read(record, name))
        except KeyError:
            continue
    raise KeyError(f"Segment lacks coordinate field(s): {', '.join((*nm_names, *m_names))}")


def _load_geometry(bundle: Any) -> _Geometry:
    derived = _require_mapping_field(bundle, "derived")
    raw_segments = _require_mapping_field(derived, "segments")
    if not isinstance(raw_segments, Sequence) or isinstance(raw_segments, (str, bytes)):
        raise TypeError("derived.segments must be a sequence")
    segments: list[_GeometrySegment] = []
    for index, record in enumerate(raw_segments):
        try:
            segment_id = str(_read(record, "name"))
        except KeyError:
            segment_id = str(_read(record, "segment_id"))
        material = _normalise_material(_read(record, "material"))
        start_nm = _segment_coordinate_nm(
            record,
            ("x_start_nm", "start_nm", "x0_nm"),
            ("x_start_m", "start_m", "x0_m"),
        )
        stop_nm = _segment_coordinate_nm(
            record,
            ("x_end_nm", "stop_nm", "x1_nm"),
            ("x_end_m", "stop_m", "x1_m"),
        )
        if not np.isfinite(start_nm) or not np.isfinite(stop_nm) or stop_nm < start_nm:
            raise ValueError(
                f"Invalid derived.segments[{index}] interval {start_nm:g}--{stop_nm:g} nm"
            )
        segments.append(_GeometrySegment(segment_id, material, start_nm, stop_nm))
    if not segments:
        raise ValueError("derived.segments is empty")
    tolerance = max(1e-9, 1e-9 * max(abs(segments[0].start_nm), abs(segments[-1].stop_nm), 1.0))
    for left, right in zip(segments[:-1], segments[1:], strict=True):
        if not math.isclose(left.stop_nm, right.start_nm, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(
                f"Geometry gap/overlap between {left.segment_id!r} and {right.segment_id!r}: "
                f"{left.stop_nm:g} vs {right.start_nm:g} nm"
            )
    expected_materials = ("C", "Au", "Pd", "C")
    if tuple(segment.material for segment in segments) != expected_materials:
        raise ValueError(
            "derived.segments must preserve the C|Au|Pd|C topology; got "
            f"{tuple(segment.material for segment in segments)!r}"
        )
    edges = (segments[0].start_nm, *(segment.stop_nm for segment in segments))
    geometry = _Geometry(tuple(segments), tuple(float(value) for value in edges), edges[0], edges[-1])
    if geometry.stop_nm <= geometry.start_nm:
        raise ValueError("Total geometry length must be positive")
    reported_total = _first_scalar((derived,), ("L_total_nm", "L_halfcell_nm"), default=geometry.stop_nm - geometry.start_nm)
    if not math.isclose(
        reported_total,
        geometry.stop_nm - geometry.start_nm,
        rel_tol=1e-9,
        abs_tol=tolerance,
    ):
        raise ValueError(
            f"derived L_total={reported_total:g} nm disagrees with segment span "
            f"{geometry.stop_nm - geometry.start_nm:g} nm"
        )
    return geometry


def _load_surface(bundle: Any, geometry: _Geometry) -> _SurfaceData:
    source = _require_mapping_field(bundle, "surface")
    numeric_names = (
        "x_nm",
        "phi_rp_with_V",
        "phi_rp_no_V",
        "c_R1_with",
        "c_O2_with",
        "c_R1_no",
        "c_O2_no",
        "eta_au_with_V",
        "eta_pd_with_V",
        "eta_au_no_V",
        "eta_pd_no_V",
        "j_au_with",
        "j_pd_with",
        "j_au_no",
        "j_pd_no",
    )
    arrays = {name: _float_1d(source, name) for name in numeric_names}
    segment_id = _object_1d(source, "segment_id")
    material_raw = _object_1d(source, "material")
    expected = arrays["x_nm"].size
    for name, values in (*arrays.items(), ("segment_id", segment_id), ("material", material_raw)):
        if values.size != expected:
            raise ValueError(f"surface.{name} has length {values.size}; expected {expected}")
    x_nm = arrays["x_nm"]
    if np.any(~np.isfinite(x_nm)) or np.any(np.diff(x_nm) <= 0.0):
        raise ValueError("surface.x_nm must be finite and strictly increasing")
    span_tolerance = max(1e-7, 1e-8 * (geometry.stop_nm - geometry.start_nm))
    if not np.isclose(x_nm[0], geometry.start_nm, atol=span_tolerance) or not np.isclose(
        x_nm[-1], geometry.stop_nm, atol=span_tolerance
    ):
        raise ValueError(
            f"surface.x_nm must span {geometry.start_nm:g}--{geometry.stop_nm:g} nm; "
            f"got {x_nm[0]:g}--{x_nm[-1]:g} nm"
        )
    material = np.asarray([_normalise_material(item) for item in material_raw], dtype=object)
    return _SurfaceData(
        segment_id=segment_id,
        material=material,
        **arrays,
    )


def _field_2d(source: Any, name: str, ny: int, nx: int) -> np.ndarray:
    result = np.asarray(_require_mapping_field(source, name), dtype=float)
    if result.shape == (nx, ny) and result.shape != (ny, nx):
        result = result.T
    if result.shape != (ny, nx):
        raise ValueError(f"grid_2d.{name} must have shape {(ny, nx)}; got {result.shape}")
    return result


def _load_grid_2d(bundle: Any, geometry: _Geometry) -> _Grid2D:
    source = _require_mapping_field(bundle, "grid_2d")
    x_nm = _float_1d(source, "x_nm")
    y_nm = _float_1d(source, "y_nm")
    if np.any(~np.isfinite(x_nm)) or np.any(np.diff(x_nm) <= 0.0):
        raise ValueError("grid_2d.x_nm must be finite and strictly increasing")
    if np.any(~np.isfinite(y_nm)) or np.any(np.diff(y_nm) <= 0.0):
        raise ValueError("grid_2d.y_nm must be finite and strictly increasing")
    span_tolerance = max(1e-7, 1e-8 * (geometry.stop_nm - geometry.start_nm))
    if not np.isclose(x_nm[0], geometry.start_nm, atol=span_tolerance) or not np.isclose(
        x_nm[-1], geometry.stop_nm, atol=span_tolerance
    ):
        raise ValueError(
            f"grid_2d.x_nm must span {geometry.start_nm:g}--{geometry.stop_nm:g} nm; "
            f"got {x_nm[0]:g}--{x_nm[-1]:g} nm"
        )
    ny, nx = y_nm.size, x_nm.size
    return _Grid2D(
        x_nm=x_nm,
        y_nm=y_nm,
        phi_s_mV=_field_2d(source, "phi_s_mV", ny, nx),
        c_R1_norm=_field_2d(source, "c_R1_norm", ny, nx),
        c_O2_norm=_field_2d(source, "c_O2_norm", ny, nx),
    )


def _broadcast_object(raw: Any, length: int, label: str) -> np.ndarray:
    result = np.asarray(raw, dtype=object)
    if result.ndim == 0:
        return np.full(length, result.item(), dtype=object)
    if result.ndim != 1 or result.size != length:
        raise ValueError(f"{label} must be scalar or have length {length}; got {result.shape}")
    return result


def _charge_rows(raw: Any) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    mappings: list[Any]
    if isinstance(raw, Mapping) or hasattr(raw, "x_nm"):
        mappings = [raw]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        mappings = list(raw)
    else:
        raise TypeError("surface_charge_segments must be a mapping or a sequence of records")
    rows: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for index, record in enumerate(mappings):
        x = np.asarray(_require_mapping_field(record, "x_nm"), dtype=float)
        sigma = np.asarray(_require_mapping_field(record, "sigma_C_per_m2"), dtype=float)
        x = np.atleast_1d(x)
        sigma = np.atleast_1d(sigma)
        if x.ndim != 1 or sigma.ndim != 1 or x.size == 0 or sigma.shape != x.shape:
            raise ValueError(
                f"surface_charge_segments[{index}] x/sigma arrays must be non-empty and equal; "
                f"got {x.shape} and {sigma.shape}"
            )
        ids = _broadcast_object(_require_mapping_field(record, "segment_id"), x.size, "segment_id")
        materials = _broadcast_object(_require_mapping_field(record, "material"), x.size, "material")
        for segment_id in dict.fromkeys(str(value) for value in ids):
            id_mask = np.asarray([str(value) == segment_id for value in ids], dtype=bool)
            material_values = {_normalise_material(value) for value in materials[id_mask]}
            if len(material_values) != 1:
                raise ValueError(f"Segment {segment_id!r} contains multiple materials: {material_values}")
            material = material_values.pop()
            rows.append((segment_id, material, x[id_mask], sigma[id_mask]))
    return rows


def _load_charge_segments(bundle: Any) -> tuple[_ChargeSegment, ...]:
    raw = _require_mapping_field(bundle, "surface_charge_segments")
    grouped: dict[tuple[str, str], tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for segment_id, material, x, sigma in _charge_rows(raw):
        xs, sigmas = grouped.setdefault((segment_id, material), ([], []))
        xs.append(x)
        sigmas.append(sigma)
    result: list[_ChargeSegment] = []
    for (segment_id, material), (x_parts, sigma_parts) in grouped.items():
        x = np.concatenate(x_parts).astype(float, copy=False)
        sigma = np.concatenate(sigma_parts).astype(float, copy=False)
        order = np.argsort(x, kind="stable")
        x, sigma = x[order], sigma[order]
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(sigma)):
            raise ValueError(f"Surface-charge segment {segment_id!r} contains non-finite values")
        result.append(_ChargeSegment(segment_id, material, x, sigma))
    if not result:
        raise ValueError("surface_charge_segments is empty")
    result.sort(key=lambda segment: float(np.min(segment.x_nm)))
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


def _unique_positions(values: Sequence[float], *, tolerance: float = 1e-9) -> list[float]:
    unique: list[float] = []
    for raw in values:
        value = float(raw)
        if not any(math.isclose(value, previous, rel_tol=0.0, abs_tol=tolerance) for previous in unique):
            unique.append(value)
    return unique


def _add_boundaries(ax: plt.Axes, geometry: _Geometry) -> None:
    tolerance = max(1e-9, 1e-9 * (geometry.stop_nm - geometry.start_nm))
    for position in _unique_positions(geometry.edges_nm[1:-1], tolerance=tolerance):
        if math.isclose(position, geometry.start_nm, rel_tol=0.0, abs_tol=tolerance) or math.isclose(
            position, geometry.stop_nm, rel_tol=0.0, abs_tol=tolerance
        ):
            continue
        ax.axvline(
            position,
            color=COLORS["gray"],
            linewidth=0.8,
            linestyle=(0, (3, 2)),
            alpha=0.82,
            zorder=5,
        )


def _set_spatial_x(ax: plt.Axes, geometry: _Geometry, *, show_labels: bool = True) -> None:
    tolerance = max(1e-9, 1e-9 * (geometry.stop_nm - geometry.start_nm))
    ticks = _unique_positions(geometry.edges_nm, tolerance=tolerance)
    ax.set_xlim(geometry.start_nm, geometry.stop_nm)
    ax.set_xticks(ticks)
    if not show_labels:
        ax.tick_params(labelbottom=False)


def _add_material_lane(ax: plt.Axes, geometry: _Geometry) -> None:
    total_width = geometry.stop_nm - geometry.start_nm
    for segment in geometry.segments:
        label, start, stop = segment.material, segment.start_nm, segment.stop_nm
        if segment.width_nm <= max(1e-10, 1e-10 * total_width):
            continue
        text_color = COLORS["dark"] if label == "Au" else "white"
        ax.add_patch(
            Rectangle(
                (start, 0.0),
                stop - start,
                1.0,
                facecolor=COLORS[label],
                edgecolor="white",
                linewidth=0.8,
            )
        )
        if segment.width_nm / total_width >= 0.055:
            ax.text(
                0.5 * (start + stop),
                0.5,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.2,
            )
    ax.set_xlim(geometry.start_nm, geometry.stop_nm)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()


def _add_lane_xlabel(ax: plt.Axes) -> None:
    ax.text(
        0.5,
        -0.55,
        r"$x$ (nm)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color=COLORS["dark"],
        clip_on=False,
    )


def _profile_figure() -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    fig = plt.figure(figsize=(5.8, 3.0))
    # Leave enough room for the main axis tick numerals before the material
    # lane begins.  A smaller gap lets the lane paint over the lower halves of
    # endpoint labels (most visibly ``0`` and the final cell edge at 600 dpi).
    grid = fig.add_gridspec(2, 1, height_ratios=(1.0, 0.105), hspace=0.36)
    ax = fig.add_subplot(grid[0, 0])
    lane_ax = fig.add_subplot(grid[1, 0], sharex=ax)
    return fig, ax, lane_ax


def _finite_ylim(ax: plt.Axes, *arrays: Any, include_zero: bool = False, pad_frac: float = 0.08) -> None:
    finite_parts: list[np.ndarray] = []
    for raw in arrays:
        values = np.ravel(np.asarray(raw, dtype=float))
        finite = values[np.isfinite(values)]
        if finite.size:
            finite_parts.append(finite)
    if include_zero:
        finite_parts.append(np.array([0.0]))
    if not finite_parts:
        raise ValueError("Cannot set y limits from empty/non-finite data")
    values = np.concatenate(finite_parts)
    low, high = float(np.min(values)), float(np.max(values))
    span = high - low
    pad = max(1e-8, pad_frac * span, 0.02 * max(abs(low), abs(high), 1e-8))
    ax.set_ylim(low - pad, high + pad)


def _resolve_figure_root(output_root: str | Path) -> Path:
    root = Path(output_root)
    return root if root.name == "figures" else root / "figures"


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
    pdf_target = output_dir / f"{stem}.pdf"
    existing = [path for path in (*targets, pdf_target) if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing figure artifact(s): {joined}")
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


def _plot_panel_a(bundle: Any, output_dir: Path) -> list[Path]:
    with_edl = _require_mapping_field(bundle, "with_edl")
    without_edl = _require_mapping_field(bundle, "without_edl")
    e_values = np.array(
        [
            _first_scalar((without_edl,), ("E_mix_V", "E_mix")),
            _first_scalar((with_edl,), ("E_mix_V", "E_mix")),
        ]
    )
    i_values = np.array(
        [
            _first_scalar((without_edl,), ("i_mix_avg_A_per_m2", "i_mix_avg")),
            _first_scalar((with_edl,), ("i_mix_avg_A_per_m2", "i_mix_avg")),
        ]
    )
    labels = ["w/o EDL", "with EDL"]
    colors = [COLORS["without"], COLORS["with"]]
    positions = np.arange(2, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(4.35, 2.85))
    for ax, values, ylabel, title, value_format in (
        (axes[0], e_values, r"$E_{\mathrm{mix}}$ (V)", r"$E_{\mathrm{mix}}$", ".3f"),
        (
            axes[1],
            i_values,
            r"$\bar{i}_{\mathrm{mix}}$ (A m$^{-2}$)",
            r"$\bar{i}_{\mathrm{mix}}$",
            ".3g",
        ),
    ):
        bars = ax.bar(
            positions,
            values,
            width=0.58,
            color=colors,
            edgecolor=COLORS["dark"],
            linewidth=0.8,
        )
        _style_axis(ax, "", ylabel, title)
        ax.set_xticks(positions, labels, rotation=42, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.55, 1.55)
        low = min(0.0, float(np.min(values)))
        high = max(0.0, float(np.max(values)))
        span = max(high - low, 1e-12)
        ax.set_ylim(low - 0.04 * span, high + 0.22 * span)
        ax.bar_label(bars, fmt=f"%{value_format}", padding=3, fontsize=7.6, color=COLORS["dark"])
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.29, top=0.84, wspace=0.57)
    return _save_pair(
        fig,
        output_dir,
        f"figure_3_panel_a_emix_imix_{OUTPUT_TAG}",
        transparent=True,
    )


def _plot_panel_b(surface: _SurfaceData, geometry: _Geometry, output_dir: Path) -> list[Path]:
    fig, ax, lane_ax = _profile_figure()
    ax.plot(surface.x_nm, surface.phi_rp_with_V, color=COLORS["with"], lw=2.0, label="with EDL")
    ax.plot(
        surface.x_nm,
        surface.phi_rp_no_V,
        color=COLORS["without"],
        lw=1.8,
        ls=(0, (4, 2)),
        label="w/o EDL",
    )
    _add_boundaries(ax, geometry)
    _set_spatial_x(ax, geometry)
    _style_axis(ax, "", r"$\phi_{\mathrm{RP}}(x)$ (V)", "Reaction-plane potential")
    _finite_ylim(ax, surface.phi_rp_with_V, surface.phi_rp_no_V)
    ax.legend(loc="best", fontsize=8.1, handlelength=2.0)
    _add_material_lane(lane_ax, geometry)
    _add_lane_xlabel(lane_ax)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.11, top=0.90)
    return _save_pair(
        fig,
        output_dir,
        f"figure_3_panel_b_reaction_plane_potential_{OUTPUT_TAG}",
        transparent=True,
    )


def _positive_for_log(values: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values) & (values > 0.0), values, np.nan)


def _plot_panel_c(surface: _SurfaceData, geometry: _Geometry, output_dir: Path) -> list[Path]:
    fig, ax, lane_ax = _profile_figure()
    series = (
        (surface.c_R1_with, COLORS["with"], "-", r"$\mathrm{Red}_1^-$, with EDL"),
        (surface.c_R1_no, COLORS["without"], (0, (4, 2)), r"$\mathrm{Red}_1^-$, w/o EDL"),
        (surface.c_O2_with, COLORS["with_gold"], "-", r"$\mathrm{Ox}_2^+$, with EDL"),
        (surface.c_O2_no, COLORS["without_alt"], (0, (4, 2)), r"$\mathrm{Ox}_2^+$, w/o EDL"),
    )
    positive_arrays: list[np.ndarray] = []
    for values, color, linestyle, label in series:
        positive = _positive_for_log(values)
        positive_arrays.append(positive)
        ax.plot(surface.x_nm, positive, color=color, lw=1.8, ls=linestyle, label=label)
    finite = np.concatenate([values[np.isfinite(values)] for values in positive_arrays])
    if finite.size == 0:
        raise ValueError("Reactant concentration profiles contain no positive finite values")
    ax.set_yscale("log")
    low, high = float(np.min(finite)), float(np.max(finite))
    if math.isclose(low, high, rel_tol=1e-12, abs_tol=0.0):
        ax.set_ylim(low / 1.4, high * 1.4)
    else:
        ax.set_ylim(low / 1.25, high * 1.25)
    _add_boundaries(ax, geometry)
    _set_spatial_x(ax, geometry)
    _style_axis(ax, "", r"$c_i/c_{\mathrm{bulk}}$", "Reactant concentration at RP")
    ax.legend(loc="best", fontsize=7.1, handlelength=1.7, ncols=2, columnspacing=0.9)
    _add_material_lane(lane_ax, geometry)
    _add_lane_xlabel(lane_ax)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.11, top=0.90)
    return _save_pair(
        fig,
        output_dir,
        f"figure_3_panel_c_local_reactant_concentration_{OUTPUT_TAG}",
        transparent=True,
    )


def _material_mask(surface: _SurfaceData, material: str) -> np.ndarray:
    return np.asarray(surface.material == material, dtype=bool)


def _masked(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, values, np.nan)


def _plot_panel_d(surface: _SurfaceData, geometry: _Geometry, output_dir: Path) -> list[Path]:
    fig, ax, lane_ax = _profile_figure()
    mask_au = _material_mask(surface, "Au")
    mask_pd = _material_mask(surface, "Pd")
    plotted: list[np.ndarray] = []
    for condition, color, linestyle, au_values, pd_values in (
        (
            "with EDL",
            COLORS["with"],
            "-",
            surface.eta_au_with_V,
            surface.eta_pd_with_V,
        ),
        (
            "w/o EDL",
            COLORS["without"],
            (0, (4, 2)),
            surface.eta_au_no_V,
            surface.eta_pd_no_V,
        ),
    ):
        au_segment = _masked(au_values, mask_au)
        pd_segment = _masked(pd_values, mask_pd)
        plotted.extend((au_segment, pd_segment))
        ax.plot(surface.x_nm, au_segment, color=color, lw=1.9, ls=linestyle, label=condition)
        ax.plot(surface.x_nm, pd_segment, color=color, lw=1.9, ls=linestyle)
    _add_boundaries(ax, geometry)
    _set_spatial_x(ax, geometry)
    _style_axis(ax, "", r"$\eta(x)$ (V)", "Overpotential at RP")
    _finite_ylim(ax, *plotted)
    ax.legend(loc="best", fontsize=8.1, handlelength=2.0)
    _add_material_lane(lane_ax, geometry)
    _add_lane_xlabel(lane_ax)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.11, top=0.90)
    return _save_pair(
        fig,
        output_dir,
        f"figure_3_panel_d_local_overpotential_{OUTPUT_TAG}",
        transparent=True,
    )


def _current_scale(*arrays: np.ndarray) -> tuple[float, int]:
    finite = np.concatenate(
        [np.abs(values[np.isfinite(values)]) for values in arrays if np.any(np.isfinite(values))]
    )
    if finite.size == 0 or float(np.max(finite)) == 0.0:
        return 1.0, 0
    maximum = float(np.max(finite))
    if 1e-2 <= maximum < 1e3:
        return 1.0, 0
    exponent = int(3 * math.floor(math.log10(maximum) / 3.0))
    return 10.0**exponent, exponent


def _current_ylabel(exponent: int) -> str:
    if exponent == 0:
        return r"$i(x)$ (A m$^{-2}$)"
    return rf"$i(x)$ ($10^{{{exponent}}}$ A m$^{{-2}}$)"


def _plot_panel_e(surface: _SurfaceData, geometry: _Geometry, output_dir: Path) -> list[Path]:
    mask_au = _material_mask(surface, "Au")
    mask_pd = _material_mask(surface, "Pd")
    raw = (
        _masked(surface.j_au_with, mask_au),
        _masked(surface.j_au_no, mask_au),
        _masked(surface.j_pd_with, mask_pd),
        _masked(surface.j_pd_no, mask_pd),
    )
    scale, exponent = _current_scale(*raw)
    au_with, au_no, pd_with, pd_no = (values / scale for values in raw)
    fig, ax, lane_ax = _profile_figure()
    ax.axhline(0.0, color=COLORS["dark"], lw=0.6, alpha=0.78, zorder=1)
    ax.plot(surface.x_nm, au_with, color=COLORS["current_Au"], lw=2.0, label="Au with EDL")
    ax.plot(
        surface.x_nm,
        au_no,
        color=COLORS["current_Au"],
        lw=1.7,
        ls=(0, (4, 2)),
        label="Au w/o EDL",
    )
    ax.plot(surface.x_nm, pd_with, color=COLORS["current_Pd"], lw=2.0, label="Pd with EDL")
    ax.plot(
        surface.x_nm,
        pd_no,
        color=COLORS["current_Pd"],
        lw=1.7,
        ls=(0, (4, 2)),
        label="Pd w/o EDL",
    )
    _add_boundaries(ax, geometry)
    _set_spatial_x(ax, geometry)
    _style_axis(ax, "", _current_ylabel(exponent), "Current density at RP")
    _finite_ylim(ax, au_with, au_no, pd_with, pd_no, include_zero=True)
    ax.legend(loc="best", fontsize=7.2, handlelength=1.8, ncols=2, columnspacing=0.9)
    _add_material_lane(lane_ax, geometry)
    _add_lane_xlabel(lane_ax)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.11, top=0.90)
    return _save_pair(
        fig,
        output_dir,
        f"figure_3_panel_e_local_current_density_{OUTPUT_TAG}",
        transparent=True,
    )


def _fmt_voltage(value: float) -> str:
    return f"{value:.2f} V"


def _reference_marker(
    ax: plt.Axes,
    x: float,
    y: float,
    label: str,
    color: str,
    marker: str,
    text_offset_y: float,
    *,
    text_offset_x: float = 0.0,
    linestyle: Any = "-",
) -> None:
    ax.vlines(x, y - 0.085, y + 0.085, color=color, lw=1.7, ls=linestyle, zorder=2)
    ax.scatter([x], [y], s=60, marker=marker, color=color, edgecolor="none", zorder=4)
    ax.text(
        x + text_offset_x,
        y + text_offset_y,
        f"{label}\n{_fmt_voltage(x)}",
        ha="center",
        va="bottom" if text_offset_y >= 0.0 else "top",
        fontsize=8.0,
        color=color,
        linespacing=1.08,
    )


def _plot_panel_f(bundle: Any, output_dir: Path) -> list[Path]:
    params = _require_mapping_field(bundle, "params")
    derived = _require_mapping_field(bundle, "derived")
    with_edl = _require_mapping_field(bundle, "with_edl")
    without_edl = _require_mapping_field(bundle, "without_edl")
    sources = (with_edl, derived, params)
    e1 = _first_scalar(sources, ("E1_eq_eff_V", "E1_eq_eff", "E1_eq_V", "E1_eq"))
    e2 = _first_scalar(sources, ("E2_eq_eff_V", "E2_eq_eff", "E2_eq_V", "E2_eq"))
    e_with = _first_scalar((with_edl,), ("E_mix_V", "E_mix"))
    e_no = _first_scalar((without_edl,), ("E_mix_V", "E_mix"))
    pzc_c = _first_scalar((params, derived), ("pzc_C", "PZC_C_V", "pzc_support"))
    pzc_pd = _first_scalar((params, derived), ("pzc_Pd", "PZC_Pd_V"))
    pzc_au = _first_scalar((params, derived), ("pzc_Au", "PZC_Au_V"))

    fig, ax = plt.subplots(figsize=(5.25, 3.35))
    lane_y = {"eq": 0.44, "mix": 0.0, "pzc": -0.44}
    ax.hlines(0.0, 0.0, 1.0, color=COLORS["dark"], lw=1.0, zorder=1)
    _reference_marker(ax, e1, lane_y["eq"], r"$E_{1,\mathrm{eq}}$", COLORS["gray"], "o", 0.115)
    _reference_marker(ax, e2, lane_y["eq"], r"$E_{2,\mathrm{eq}}$", COLORS["gray"], "o", 0.115)
    _reference_marker(
        ax,
        e_no,
        lane_y["mix"],
        r"$E_{\mathrm{mix}}$ w/o EDL",
        COLORS["without"],
        "D",
        -0.135,
        text_offset_x=-0.055,
    )
    _reference_marker(
        ax,
        e_with,
        lane_y["mix"],
        r"$E_{\mathrm{mix}}$ with EDL",
        COLORS["with"],
        "D",
        0.12,
    )
    _reference_marker(
        ax,
        pzc_c,
        lane_y["pzc"],
        "PZC C",
        COLORS["C"],
        "^",
        -0.125,
        text_offset_x=0.025,
    )
    _reference_marker(
        ax,
        pzc_pd,
        lane_y["pzc"],
        "PZC Pd",
        COLORS["Pd"],
        "^",
        -0.125,
        text_offset_x=-0.018,
    )
    _reference_marker(ax, pzc_au, lane_y["pzc"], "PZC Au", COLORS["Au"], "^", -0.125)
    all_values = np.array([e1, e2, e_with, e_no, pzc_c, pzc_pd, pzc_au], dtype=float)
    xmin = min(0.0, float(np.min(all_values)) - 0.04)
    xmax = max(1.0, float(np.max(all_values)) + 0.04)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.98, 0.79)
    ax.set_xlabel("Potential (V vs. RHE)")
    ax.set_yticks([])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis="x", length=3.4, width=0.85, labelsize=8.8)
    ax.set_title("Potential reference map", loc="left", pad=7, fontsize=10.5, fontweight="normal")
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_position(("data", -0.80))
    ax.spines["bottom"].set_linewidth(0.9)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.16, top=0.90)
    return _save_pair(
        fig,
        output_dir,
        f"figure_3_panel_f_pzc_potential_reference_map_{OUTPUT_TAG}",
        transparent=True,
    )


def _potential_norm(values: np.ndarray) -> TwoSlopeNorm:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("phi_s_mV contains no finite values")
    maximum = float(np.max(np.abs(finite)))
    limit = max(10.0, 10.0 * math.ceil(maximum / 10.0))
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def _log_norm(*arrays: np.ndarray) -> LogNorm:
    positive_parts = [values[np.isfinite(values) & (values > 0.0)] for values in arrays]
    positive_parts = [values for values in positive_parts if values.size]
    if not positive_parts:
        raise ValueError("Reactant 2D fields contain no positive finite values")
    values = np.concatenate(positive_parts)
    low, high = float(np.min(values)), float(np.max(values))
    if math.isclose(low, high, rel_tol=1e-10, abs_tol=0.0):
        low, high = low / 1.25, high * 1.25
    return LogNorm(vmin=low, vmax=high)


def _style_map_axis(
    ax: plt.Axes,
    title: str,
    geometry: _Geometry,
    *,
    show_xlabels: bool,
) -> None:
    ax.set_title(title, loc="left", pad=5, fontsize=9.5, fontweight="normal")
    ax.set_ylabel(r"$y$ (nm)")
    ax.tick_params(length=3.2, width=0.85, pad=2.5, labelsize=8.0)
    _set_spatial_x(ax, geometry, show_labels=show_xlabels)
    _add_boundaries(ax, geometry)
    for spine in ("left", "bottom", "top", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.9)
        ax.spines[spine].set_color(COLORS["dark"])


def _add_colorbar(fig: plt.Figure, mesh: Any, cax: plt.Axes, label: str) -> None:
    colorbar = fig.colorbar(mesh, cax=cax)
    colorbar.set_label(label, labelpad=5)
    colorbar.ax.tick_params(length=2.8, width=0.75, labelsize=7.7, pad=2.2)
    colorbar.outline.set_linewidth(0.8)


def _plot_potential_2d(
    bundle: Any,
    grid: _Grid2D,
    geometry: _Geometry,
    output_dir: Path,
) -> list[Path]:
    with_edl = _require_mapping_field(bundle, "with_edl")
    e_mix = _first_scalar((with_edl,), ("E_mix_V", "E_mix"))
    fig = plt.figure(figsize=(5.9, 3.25), facecolor="white")
    layout = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 0.038),
        height_ratios=(1.0, 0.11),
        hspace=0.20,
        wspace=0.08,
    )
    ax = fig.add_subplot(layout[0, 0])
    cax = fig.add_subplot(layout[0, 1])
    lane_ax = fig.add_subplot(layout[1, 0], sharex=ax)
    fig.add_subplot(layout[1, 1]).set_axis_off()
    mesh = ax.pcolormesh(
        grid.x_nm,
        grid.y_nm,
        grid.phi_s_mV,
        shading="auto",
        cmap="RdBu_r",
        norm=_potential_norm(grid.phi_s_mV),
        rasterized=True,
    )
    _style_map_axis(
        ax,
        rf"Solution phase potential, $E_{{\mathrm{{mix}}}}$ = {e_mix:.2f} V",
        geometry,
        show_xlabels=True,
    )
    ax.set_ylim(float(grid.y_nm[0]), float(grid.y_nm[-1]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    _add_colorbar(fig, mesh, cax, r"$\mathit{\Phi}_{\mathrm{s}}$ (mV)")
    _add_material_lane(lane_ax, geometry)
    _add_lane_xlabel(lane_ax)
    fig.subplots_adjust(left=0.11, right=0.92, bottom=0.13, top=0.89)
    return _save_pair(
        fig,
        output_dir,
        f"solution_phase_potential_2d_{OUTPUT_TAG}",
        transparent=False,
    )


def _plot_potential_and_reactants_2d(
    bundle: Any,
    grid: _Grid2D,
    geometry: _Geometry,
    output_dir: Path,
) -> list[Path]:
    with_edl = _require_mapping_field(bundle, "with_edl")
    e_mix = _first_scalar((with_edl,), ("E_mix_V", "E_mix"))
    fig = plt.figure(figsize=(5.9, 7.0), facecolor="white")
    layout = fig.add_gridspec(
        4,
        2,
        width_ratios=(1.0, 0.038),
        height_ratios=(1.0, 1.0, 1.0, 0.11),
        hspace=0.20,
        wspace=0.08,
    )
    axes = [fig.add_subplot(layout[row, 0]) for row in range(3)]
    color_axes = [fig.add_subplot(layout[row, 1]) for row in range(3)]
    lane_ax = fig.add_subplot(layout[3, 0], sharex=axes[-1])
    fig.add_subplot(layout[3, 1]).set_axis_off()
    concentration_norm = _log_norm(grid.c_R1_norm, grid.c_O2_norm)
    specifications = (
        (
            grid.phi_s_mV,
            "RdBu_r",
            _potential_norm(grid.phi_s_mV),
            rf"Solution phase potential, $E_{{\mathrm{{mix}}}}$ = {e_mix:.2f} V",
            r"$\mathit{\Phi}_{\mathrm{s}}$ (mV)",
        ),
        (
            grid.c_R1_norm,
            "viridis",
            concentration_norm,
            r"$\mathrm{Red}_1^-$ distribution",
            r"$c(\mathrm{Red}_1^-)/c_{\mathrm{bulk}}$",
        ),
        (
            grid.c_O2_norm,
            "viridis",
            concentration_norm,
            r"$\mathrm{Ox}_2^+$ distribution",
            r"$c(\mathrm{Ox}_2^+)/c_{\mathrm{bulk}}$",
        ),
    )
    for index, (ax, cax, specification) in enumerate(zip(axes, color_axes, specifications, strict=True)):
        values, cmap, norm, title, colorbar_label = specification
        mesh = ax.pcolormesh(
            grid.x_nm,
            grid.y_nm,
            values,
            shading="auto",
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
        _style_map_axis(ax, title, geometry, show_xlabels=index == 2)
        ax.set_ylim(float(grid.y_nm[0]), float(grid.y_nm[-1]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        _add_colorbar(fig, mesh, cax, colorbar_label)
    _add_material_lane(lane_ax, geometry)
    _add_lane_xlabel(lane_ax)
    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.11, right=0.92, bottom=0.07, top=0.96)
    return _save_pair(
        fig,
        output_dir,
        f"solution_phase_potential_and_reactants_2d_{OUTPUT_TAG}",
        transparent=False,
    )


def _plot_surface_charge(
    segments: Sequence[_ChargeSegment],
    geometry: _Geometry,
    output_dir: Path,
) -> list[Path]:
    fig, ax, lane_ax = _profile_figure()
    seen_materials: set[str] = set()
    sigma_scaled: list[np.ndarray] = []
    for segment in segments:
        values = 100.0 * segment.sigma_C_per_m2
        sigma_scaled.append(values)
        label = segment.material if segment.material not in seen_materials else None
        seen_materials.add(segment.material)
        ax.plot(segment.x_nm, values, color=COLORS[segment.material], lw=2.0, label=label, zorder=3)
    ax.axhline(0.0, color=COLORS["dark"], lw=0.65, alpha=0.8, zorder=1)
    _add_boundaries(ax, geometry)
    _set_spatial_x(ax, geometry)
    _style_axis(
        ax,
        "",
        r"$\sigma(x)$ (µC cm$^{-2}$)",
        "Surface charge distribution",
    )
    _finite_ylim(ax, *sigma_scaled, include_zero=True)
    ax.legend(loc="best", fontsize=7.8, handlelength=1.8, ncols=3)
    _add_material_lane(lane_ax, geometry)
    _add_lane_xlabel(lane_ax)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.11, top=0.90)
    return _save_pair(
        fig,
        output_dir,
        f"surface_charge_distribution_{OUTPUT_TAG}",
        transparent=True,
    )


def generate_all_figures(bundle: Mapping[str, Any], output_root: str | Path) -> list[Path]:
    """Generate the canonical nine publication figures.

    Parameters
    ----------
    bundle:
        Result mapping with ``params``, ``derived``, ``with_edl``,
        ``without_edl``, ``surface``, ``grid_2d`` and
        ``surface_charge_segments`` fields.
    output_root:
        Result-run directory.  Outputs are written below ``figures/Figure_3``
        and ``figures/Figure_RP``.  Passing the ``figures`` directory itself is
        also supported.

    Returns
    -------
    list[pathlib.Path]
        Eighteen paths in deterministic figure order (PNG then SVG for each
        of nine figures).
    """

    if not isinstance(bundle, Mapping):
        raise TypeError("bundle must be a mapping")
    for required in (
        "params",
        "derived",
        "with_edl",
        "without_edl",
        "surface",
        "grid_2d",
        "surface_charge_segments",
    ):
        _require_mapping_field(bundle, required)

    geometry = _load_geometry(bundle)
    surface = _load_surface(bundle, geometry)
    grid = _load_grid_2d(bundle, geometry)
    charge_segments = _load_charge_segments(bundle)
    figure_root = _resolve_figure_root(output_root)
    figure_3_dir = figure_root / "Figure_3"
    figure_rp_dir = figure_root / "Figure_RP"

    saved: list[Path] = []
    with matplotlib.rc_context(PUBLICATION_RCPARAMS):
        saved.extend(_plot_panel_a(bundle, figure_3_dir))
        saved.extend(_plot_panel_b(surface, geometry, figure_3_dir))
        saved.extend(_plot_panel_c(surface, geometry, figure_3_dir))
        saved.extend(_plot_panel_d(surface, geometry, figure_3_dir))
        saved.extend(_plot_panel_e(surface, geometry, figure_3_dir))
        saved.extend(_plot_panel_f(bundle, figure_3_dir))
        saved.extend(_plot_potential_2d(bundle, grid, geometry, figure_rp_dir))
        saved.extend(_plot_potential_and_reactants_2d(bundle, grid, geometry, figure_rp_dir))
        saved.extend(_plot_surface_charge(charge_segments, geometry, figure_rp_dir))

    if len(saved) != 18:
        raise RuntimeError(f"Expected 18 figure artifacts, generated {len(saved)}")
    png_count = sum(path.suffix.lower() == ".png" for path in saved)
    svg_count = sum(path.suffix.lower() == ".svg" for path in saved)
    if png_count != 9 or svg_count != 9:
        raise RuntimeError(f"Expected 9 PNG and 9 SVG artifacts; got {png_count} PNG and {svg_count} SVG")
    if any(path.suffix.lower() == ".pdf" for path in saved):
        raise RuntimeError("PDF output is forbidden for this figure package")
    return saved


__all__ = ["OUTPUT_TAG", "generate_all_figures"]
