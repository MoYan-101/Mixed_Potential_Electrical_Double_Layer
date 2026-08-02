"""Figure-3-style panels for flush Au/Pd on an insulating substrate.

The default comparison uses ``d = 10 nm`` because the legacy reference
Figure 3 has the length-matched Au|C|Pd geometry 25|10|25 nm.  Reading the
saved top-surface profile avoids silently rerunning the spectral model while a
figure is exported.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .kinetics import (
    au_local_current_density,
    effective_reaction_params,
    pd_local_current_density,
)
from .parameters import (
    ELECTROLYTE_DOMAIN_DESCRIPTION,
    ELECTROLYTE_DOMAIN_ID,
    ELECTROSTATIC_BACKEND,
    RESULT_SCHEMA_VERSION,
)
from .plotting import COLORS, PUBLICATION_RCPARAMS, _validate_tag


PANEL_FIGSIZE = (3.55, 2.75)
PANEL_A_FIGSIZE = (4.05, 2.75)
PANEL_F_FIGSIZE = (4.20, 3.25)
PANEL_F_PZC_MARKER_Y = -0.42
PANEL_F_PZC_TEXT_Y = -0.50


@dataclass(frozen=True)
class Figure3SavedData:
    params: Mapping[str, Any]
    tag: str
    d_nm: float
    x_nm: np.ndarray
    phi_tilde: np.ndarray
    phi_s_V: np.ndarray
    c_red1: np.ndarray
    c_ox2: np.ndarray
    j_au_with: np.ndarray
    j_pd_with: np.ndarray
    j_au_no: np.ndarray
    j_pd_no: np.ndarray
    eta_with: np.ndarray
    eta_no: np.ndarray
    E_mix_with: float
    E_mix_no: float
    i_mix_with: float
    i_mix_no: float
    E1_eq_eff: float
    E2_eq_eff: float
    L_Au_nm: float
    L_Pd_start_nm: float
    L_total_nm: float


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _find_summary_row(
    rows: list[dict[str, str]], condition: str, d_nm: float | None = None
) -> dict[str, str]:
    matches: list[dict[str, str]] = []
    for row in rows:
        if row.get("condition") != condition:
            continue
        if d_nm is not None and not math.isclose(
            float(row["d_Au_Pd_nm"]), d_nm, rel_tol=0.0, abs_tol=1.0e-6
        ):
            continue
        matches.append(row)
    if len(matches) != 1:
        target = condition if d_nm is None else f"{condition}, d={d_nm:g} nm"
        raise ValueError(f"Expected exactly one summary row for {target}; got {len(matches)}")
    return matches[0]


def _profile_filename(d_nm: float) -> str:
    rounded = round(d_nm)
    if not math.isclose(d_nm, rounded, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("Figure 3 profile naming currently requires an integer separation")
    return f"top_surface_profiles_d{int(rounded):03d}nm.csv"


def _load_and_validate_result_contract(
    root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
    """Require schema-v2 spectral metadata before reading numerical CSV data."""

    expected = (RESULT_SCHEMA_VERSION, ELECTROSTATIC_BACKEND)
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise ValueError(
            "Figure 3 requires summary.json with an explicit schema-v2 "
            "semi-infinite spectral result contract"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise ValueError("summary.json must contain a JSON object")
    summary_identity = (
        summary.get("result_schema_version"),
        summary.get("electrostatic_backend"),
    )
    if summary_identity != expected:
        raise ValueError(
            "Unsupported summary.json result contract: "
            f"got version/backend {summary_identity!r}, expected {expected!r}"
        )

    manifest_path = root / "run_manifest.json"
    manifest: Mapping[str, Any] | None = None
    if manifest_path.is_file():
        loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_manifest, Mapping):
            raise ValueError("run_manifest.json must contain a JSON object")
        manifest = loaded_manifest
        manifest_identity = (
            manifest.get("result_schema_version"),
            manifest.get("electrostatic_backend"),
        )
        if manifest_identity != summary_identity:
            raise ValueError(
                "run_manifest.json and summary.json disagree on result "
                f"version/backend: {manifest_identity!r} != {summary_identity!r}"
            )
        if manifest_identity != expected:
            raise ValueError(
                "Unsupported run_manifest.json result contract: "
                f"got {manifest_identity!r}, expected {expected!r}"
            )
    return summary, manifest


def load_figure3_saved_data(run_dir: str | Path, separation_nm: float = 10.0) -> Figure3SavedData:
    """Load a saved profile and derive the matching w/o-EDL local fields."""

    root = Path(run_dir).resolve()
    summary, manifest = _load_and_validate_result_contract(root)
    params = json.loads((root / "params.json").read_text(encoding="utf-8"))
    tag = _validate_tag(str(summary["tag"]))
    if manifest is not None and str(manifest.get("tag")) != tag:
        raise ValueError("run_manifest.json and summary.json disagree on tag")
    d_nm = float(separation_nm)

    summary_rows = _read_csv_rows(root / "csv" / "summary_cases.csv")
    with_row = _find_summary_row(summary_rows, "with EDL", d_nm)
    no_row = _find_summary_row(summary_rows, "w/o EDL")
    profile_rows = _read_csv_rows(root / "csv" / _profile_filename(d_nm))
    if len(profile_rows) < 2:
        raise ValueError("Saved top-surface profile must contain at least two points")

    def column(name: str) -> np.ndarray:
        values = np.asarray([float(row[name]) for row in profile_rows], dtype=float)
        if name not in {"j_Au_A_per_m2", "j_Pd_A_per_m2"} and not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite values in saved profile column {name}")
        return values

    x_nm = column("x_nm")
    phi_tilde = column("phi_tilde")
    phi_s = column("phi_s_V")
    c_red1 = column("c_Red1_over_c_bulk")
    c_ox2 = column("c_Ox2_over_c_bulk")
    j_au_with = column("j_Au_A_per_m2")
    j_pd_with = column("j_Pd_A_per_m2")

    L_Au_nm = float(params["L_Au"]) * 1.0e9
    L_Pd_start_nm = L_Au_nm + d_nm
    L_total_nm = L_Pd_start_nm + float(params["L_Pd"]) * 1.0e9
    if not math.isclose(float(x_nm[0]), 0.0, abs_tol=1.0e-9) or not math.isclose(
        float(x_nm[-1]), L_total_nm, rel_tol=0.0, abs_tol=1.0e-6
    ):
        raise ValueError(
            "Saved profile x range does not match flush Au|substrate|Pd geometry"
        )

    au_mask = x_nm <= L_Au_nm + 1.0e-9
    pd_mask = x_nm >= L_Pd_start_nm - 1.0e-9
    E_with = float(with_row["E_mix_V"])
    E_no = float(no_row["E_mix_V"])
    rxn = effective_reaction_params(params)

    j_au_no = np.full_like(x_nm, np.nan)
    j_pd_no = np.full_like(x_nm, np.nan)
    j_au_no[au_mask] = float(au_local_current_density(E_no, 0.0, params))
    j_pd_no[pd_mask] = float(pd_local_current_density(E_no, 0.0, params))

    eta_with = np.full_like(x_nm, np.nan)
    eta_no = np.full_like(x_nm, np.nan)
    eta_with[au_mask] = E_with - rxn["E1_eq_eff"] - phi_s[au_mask]
    eta_with[pd_mask] = E_with - rxn["E2_eq_eff"] - phi_s[pd_mask]
    eta_no[au_mask] = E_no - rxn["E1_eq_eff"]
    eta_no[pd_mask] = E_no - rxn["E2_eq_eff"]

    return Figure3SavedData(
        params=params,
        tag=tag,
        d_nm=d_nm,
        x_nm=x_nm,
        phi_tilde=phi_tilde,
        phi_s_V=phi_s,
        c_red1=c_red1,
        c_ox2=c_ox2,
        j_au_with=j_au_with,
        j_pd_with=j_pd_with,
        j_au_no=j_au_no,
        j_pd_no=j_pd_no,
        eta_with=eta_with,
        eta_no=eta_no,
        E_mix_with=E_with,
        E_mix_no=E_no,
        i_mix_with=float(with_row["i_mix_avg_A_per_m2"]),
        i_mix_no=float(no_row["i_mix_avg_A_per_m2"]),
        E1_eq_eff=float(rxn["E1_eq_eff"]),
        E2_eq_eff=float(rxn["E2_eq_eff"]),
        L_Au_nm=L_Au_nm,
        L_Pd_start_nm=L_Pd_start_nm,
        L_total_nm=L_total_nm,
    )


def _single_axis() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=PANEL_FIGSIZE)
    ax = fig.add_axes((0.18, 0.20, 0.78, 0.66))
    return fig, ax


def _style_axes(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_facecolor("none")
    ax.set_xlabel(xlabel, fontsize=10.4)
    ax.set_ylabel(ylabel, fontsize=10.4)
    ax.set_title(title, loc="left", pad=6, fontsize=11.0, fontweight="normal")
    ax.tick_params(length=3.4, width=0.85, pad=2.5, labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color(COLORS["dark"])


def _add_substrate_gap(ax: plt.Axes, data: Figure3SavedData) -> None:
    ax.axvspan(
        data.L_Au_nm,
        data.L_Pd_start_nm,
        color=COLORS["substrate"],
        alpha=0.45,
        linewidth=0.0,
        zorder=0,
    )
    for xpos in (data.L_Au_nm, data.L_Pd_start_nm):
        ax.axvline(
            xpos,
            linestyle=(0, (3, 2)),
            linewidth=0.9,
            color=COLORS["gray"],
            alpha=0.85,
            zorder=1,
        )
    ax.set_xlim(0.0, data.L_total_nm)


def _finite_ylim(ax: plt.Axes, *arrays: np.ndarray, pad_fraction: float = 0.08) -> None:
    values = np.concatenate([np.ravel(np.asarray(array, dtype=float)) for array in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    low, high = float(np.min(values)), float(np.max(values))
    span = high - low
    pad = max(1.0e-6, pad_fraction * span)
    ax.set_ylim(low - pad, high + pad)


def _make_transparent(fig: plt.Figure) -> None:
    fig.patch.set_alpha(0.0)
    for ax in fig.axes:
        ax.patch.set_alpha(0.0)


def _save_panel(
    fig: plt.Figure, output_dir: Path, stem: str, data: Figure3SavedData, dpi: int
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    d_token = f"d{int(round(data.d_nm)):03d}nm"
    paths: list[Path] = []
    for extension in ("png", "svg"):
        path = output_dir / f"{stem}_{d_token}_{data.tag}.{extension}"
        _make_transparent(fig)
        fig.savefig(
            path,
            dpi=dpi,
            transparent=True,
            facecolor="none",
            edgecolor="none",
        )
        paths.append(path)
    plt.close(fig)
    return paths


def _panel_a(data: Figure3SavedData, output_dir: Path, dpi: int) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=PANEL_A_FIGSIZE)
    labels = ["w/o EDL", "with EDL"]
    x = np.array([0.0, 1.0])
    colors = [COLORS["without_edl"], COLORS["with_edl"]]
    values = ((data.E_mix_no, data.E_mix_with), (data.i_mix_no, data.i_mix_with))
    titles = (r"$E_{\mathrm{mix}}$", r"$\bar{i}_{\mathrm{mix}}$")
    ylabels = (r"$E_{\mathrm{mix}}$ (V)", r"$\bar{i}_{\mathrm{mix}}$ (A/m$^2$)")
    for ax, vals, title, ylabel in zip(axes, values, titles, ylabels, strict=True):
        ax.bar(x, vals, width=0.58, color=colors, edgecolor=COLORS["dark"], linewidth=0.8)
        _style_axes(ax, "", ylabel, title)
        ax.set_xticks(x, labels, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(0.0, max(vals) * 1.18)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.32, top=0.84, wspace=0.60)
    return _save_panel(fig, output_dir, "figure_3_panel_a_emix_imix", data, dpi)


def _panel_b(data: Figure3SavedData, output_dir: Path, dpi: int) -> list[Path]:
    fig, ax = _single_axis()
    ax.plot(data.x_nm, data.phi_s_V, color=COLORS["with_edl"], lw=2.0, label="with EDL", zorder=3)
    ax.plot(data.x_nm, np.zeros_like(data.x_nm), color=COLORS["without_edl"], lw=1.8, label="w/o EDL", zorder=2)
    _add_substrate_gap(ax, data)
    _style_axes(ax, "x (nm)", r"$\phi_s(x,y=0)$ (V)", "Solution potential along y = 0")
    ax.text(
        0.5 * (data.L_Au_nm + data.L_Pd_start_nm),
        0.38,
        "insulating\nsubstrate",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="center",
        fontsize=7.2,
        color=COLORS["gray"],
        linespacing=1.0,
    )
    ax.legend(loc="center right", bbox_to_anchor=(0.98, 0.50), fontsize=8.9, handlelength=2.0)
    return _save_panel(fig, output_dir, "figure_3_panel_b_solution_potential_y0", data, dpi)


def _panel_c(data: Figure3SavedData, output_dir: Path, dpi: int) -> list[Path]:
    fig, ax = _single_axis()
    ax.plot(data.x_nm, data.c_red1, color=COLORS["red1_i1"], lw=2.0, label=r"$c_{\mathrm{Red1}}/c_{\mathrm{bulk}}$ (with EDL)", zorder=3)
    ax.plot(data.x_nm, data.c_ox2, color=COLORS["ox2_i2"], lw=2.0, label=r"$c_{\mathrm{Ox2}}/c_{\mathrm{bulk}}$ (with EDL)", zorder=3)
    ax.plot(data.x_nm, np.ones_like(data.x_nm), color=COLORS["without_edl"], lw=1.6, ls=(0, (4, 2)), label="w/o EDL", zorder=2)
    _add_substrate_gap(ax, data)
    ax.set_yscale("log")
    positive = np.concatenate((data.c_red1[data.c_red1 > 0], data.c_ox2[data.c_ox2 > 0], [1.0]))
    ax.set_ylim(float(np.min(positive)) / 1.25, float(np.max(positive)) * 8.0)
    _style_axes(ax, "x (nm)", r"$c_i/c_{\mathrm{bulk}}$ (-)", "Reactant concentration along y = 0")
    ax.legend(loc="upper right", fontsize=7.6, handlelength=1.7)
    return _save_panel(fig, output_dir, "figure_3_panel_c_reactant_concentration_y0", data, dpi)


def _panel_d(data: Figure3SavedData, output_dir: Path, dpi: int) -> list[Path]:
    fig, ax = _single_axis()
    ax.plot(data.x_nm, data.eta_with, color=COLORS["with_edl"], lw=2.0, label="with EDL", zorder=3)
    ax.plot(data.x_nm, data.eta_no, color=COLORS["without_edl"], lw=1.8, label="w/o EDL", zorder=2)
    _add_substrate_gap(ax, data)
    _style_axes(ax, "x (nm)", r"Local overpotential, $\eta$ (V)", "Local overpotential at RP")
    _finite_ylim(ax, data.eta_with, data.eta_no)
    ax.legend(loc="center right", fontsize=8.9, handlelength=2.0)
    return _save_panel(fig, output_dir, "figure_3_panel_d_local_overpotential", data, dpi)


def _panel_e(data: Figure3SavedData, output_dir: Path, dpi: int) -> list[Path]:
    fig, ax = _single_axis()
    ax.plot(data.x_nm, data.j_au_with, color=COLORS["red1_i1"], lw=2.0, label=r"$i_1$ (Au), with EDL", zorder=4)
    ax.plot(data.x_nm, data.j_pd_with, color=COLORS["ox2_i2"], lw=2.0, label=r"$i_2$ (Pd), with EDL", zorder=4)
    ax.plot(data.x_nm, data.j_au_no, color=COLORS["red1_i1"], lw=1.7, ls=(0, (4, 2)), label=r"$i_1$ (Au), w/o EDL", zorder=3)
    ax.plot(data.x_nm, data.j_pd_no, color=COLORS["ox2_i2"], lw=1.7, ls=(0, (4, 2)), label=r"$i_2$ (Pd), w/o EDL", zorder=3)
    _add_substrate_gap(ax, data)
    _style_axes(ax, "x (nm)", r"Local current density (A/m$^2$)", "Local current density at RP")
    _finite_ylim(ax, data.j_au_with, data.j_pd_with, data.j_au_no, data.j_pd_no)
    ax.legend(loc="upper right", fontsize=7.6, handlelength=1.8)
    return _save_panel(fig, output_dir, "figure_3_panel_e_local_current_density", data, dpi)


def _add_reference_marker(
    ax: plt.Axes,
    x: float,
    y: float,
    label: str,
    color: str,
    marker: str,
    text_y_offset: float,
    text_x_offset: float = 0.0,
) -> None:
    ax.vlines(x, y - 0.09, y + 0.09, color=color, linewidth=1.8, zorder=2)
    ax.scatter([x], [y], s=72, marker=marker, color=color, edgecolor="none", zorder=4)
    ax.text(
        x + text_x_offset,
        y + text_y_offset,
        f"{label}\n{x:.2f} V",
        ha="center",
        va="bottom" if text_y_offset >= 0 else "top",
        fontsize=8.5,
        color=color,
        linespacing=1.1,
    )


def _panel_f(
    data: Figure3SavedData,
    output_dir: Path,
    dpi: int,
    *,
    pzc_marker_y: float,
    pzc_text_y: float,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=PANEL_F_FIGSIZE)
    lane = {"eq": 0.42, "mix": 0.0, "pzc": pzc_marker_y}
    pzc_text_y_offset = pzc_text_y - pzc_marker_y
    ax.hlines(0.0, 0.0, 1.0, color=COLORS["dark"], linewidth=1.1, zorder=1)
    _add_reference_marker(ax, data.E1_eq_eff, lane["eq"], r"$\mathit{E}_{1,\mathrm{eq}}$", COLORS["gray"], "o", 0.12)
    _add_reference_marker(ax, data.E2_eq_eff, lane["eq"], r"$\mathit{E}_{2,\mathrm{eq}}$", COLORS["gray"], "o", 0.12)
    _add_reference_marker(ax, data.E_mix_no, lane["mix"], r"$\mathit{E}_{\mathrm{mix}}$ w/o EDL", COLORS["without_edl"], "D", -0.15, -0.07)
    _add_reference_marker(ax, data.E_mix_with, lane["mix"], r"$\mathit{E}_{\mathrm{mix}}$ with EDL", COLORS["with_edl"], "D", 0.13)
    _add_reference_marker(
        ax,
        float(data.params["pzc_Pd"]),
        lane["pzc"],
        "PZC Pd",
        "#5A90C8",
        "^",
        pzc_text_y_offset,
        -0.02,
    )
    _add_reference_marker(
        ax,
        float(data.params["pzc_Au"]),
        lane["pzc"],
        "PZC Au",
        "#E4C133",
        "^",
        pzc_text_y_offset,
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.92, 0.80)
    ax.set_xlabel("Potential (V vs. RHE)", fontsize=10.6)
    ax.set_yticks([])
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.tick_params(axis="x", length=3.4, width=0.85, labelsize=9.5)
    ax.set_title("Potential reference map", loc="left", fontsize=11.0, pad=6, fontweight="normal")
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_position(("data", -0.72))
    ax.spines["bottom"].set_linewidth(0.9)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.17, top=0.89)
    return _save_panel(fig, output_dir, "figure_3_panel_f_pzc_potential_reference_map", data, dpi)


def generate_figure3_comparison_panels(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    separation_nm: float = 10.0,
    dpi: int = 600,
    pzc_marker_y: float = PANEL_F_PZC_MARKER_Y,
    pzc_text_y: float = PANEL_F_PZC_TEXT_Y,
) -> dict[str, Any]:
    """Generate six matched Figure-3-style panels from saved run artifacts."""

    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if not math.isfinite(pzc_marker_y) or not math.isfinite(pzc_text_y):
        raise ValueError("PZC marker/text y positions must be finite")
    root = Path(run_dir).resolve()
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / "figures" / "Figure_3"
    )
    data = load_figure3_saved_data(root, separation_nm)
    with plt.rc_context(PUBLICATION_RCPARAMS):
        saved: list[Path] = []
        saved.extend(_panel_a(data, destination, dpi))
        saved.extend(_panel_b(data, destination, dpi))
        saved.extend(_panel_c(data, destination, dpi))
        saved.extend(_panel_d(data, destination, dpi))
        saved.extend(_panel_e(data, destination, dpi))
        saved.extend(
            _panel_f(
                data,
                destination,
                dpi,
                pzc_marker_y=pzc_marker_y,
                pzc_text_y=pzc_text_y,
            )
        )

    matching_png = list(destination.glob(f"figure_3_panel_*_d{int(round(data.d_nm)):03d}nm_{data.tag}.png"))
    matching_svg = list(destination.glob(f"figure_3_panel_*_d{int(round(data.d_nm)):03d}nm_{data.tag}.svg"))
    if len(saved) != 12 or len(matching_png) != 6 or len(matching_svg) != 6:
        raise RuntimeError("Figure 3 output count must be exactly 6 PNG and 6 SVG")
    if list(destination.glob("*.pdf")):
        raise RuntimeError("Figure 3 output directory must not contain PDF files")

    metadata: dict[str, Any] = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "electrostatic_backend": ELECTROSTATIC_BACKEND,
        "figure_type": "figure_3_style_flush_au_insulating_substrate_pd",
        "topology_id": "flush_coplanar_on_uncharged_insulating_substrate",
        "electrolyte_domain_id": ELECTROLYTE_DOMAIN_ID,
        "electrolyte_domain_description": ELECTROLYTE_DOMAIN_DESCRIPTION,
        "substrate_bc": "homogeneous_neumann",
        "comparison_basis": (
            "length-matched against legacy Au|C|Pd 25|10|25 nm Figure 3; "
            "the middle span is now an ideal uncharged insulating boundary"
        ),
        "separation_nm": data.d_nm,
        "geometry_nm": {
            "L_Au": data.L_Au_nm,
            "insulating_substrate_gap": data.d_nm,
            "L_Pd": data.L_total_nm - data.L_Pd_start_nm,
        },
        "E_mix_with_EDL_V": data.E_mix_with,
        "E_mix_without_EDL_V": data.E_mix_no,
        "i_mix_with_EDL_A_per_m2": data.i_mix_with,
        "i_mix_without_EDL_A_per_m2": data.i_mix_no,
        "formats": ["png", "svg"],
        "dpi": int(dpi),
        "panel_count": 6,
        "pdf_count": 0,
        "species_color_mapping": {
            "Red1_and_i1": COLORS["red1_i1"],
            "Ox2_and_i2": COLORS["ox2_i2"],
        },
        "condition_line_styles": {
            "with_EDL": "solid",
            "without_EDL": "dashed",
        },
        "panel_f_pzc_layout": {
            "marker_y": float(pzc_marker_y),
            "text_y": float(pzc_text_y),
        },
        "saved_paths": [str(path) for path in saved],
        "source_profile": str(root / "csv" / _profile_filename(data.d_nm)),
        "source_summary": str(root / "csv" / "summary_cases.csv"),
    }
    (destination / "figure_3_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


__all__ = [
    "Figure3SavedData",
    "generate_figure3_comparison_panels",
    "load_figure3_saved_data",
]
