"""Electrolyte-concentration study for two independent planar Au/Pd EDLs.

The scan mirrors the visual logic of ``Figures/Figrue_4`` while using only the
analytic independent-planar model in this package.  It also writes the
surface-resolved mechanism quantities requested for Au and Pd and a
quantitative 0.01/1 M potential-profile schematic with an explicit w/o-EDL
zero-potential reference.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.lines import Line2D

from .model import (
    ELECTROSTATIC_BACKEND,
    MODEL_ID,
    RESULT_SCHEMA_VERSION,
    TOPOLOGY_ID,
    IndependentPlanarEDLModel,
    canonical_params,
    load_params,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]

DPI = 450
CURRENT_DISPLAY_SCALE = 1.0e9  # A -> 10^-3 microampere
FORMAL_HIGH_SALT_START_M = 10.0
REPRESENTATIVE_CONCENTRATIONS_M = (1.0e-2, 1.0, 1.0e3)
PROFILE_CONCENTRATIONS_M = (1.0e-2, 1.0)
POLARIZATION_E_VALUES_V = np.linspace(0.40, 0.64, 960)
METAL_LEFT_NM = -2.45
COMPACT_LEFT_NM = -1.35
RP_X_NM = 0.0

COLORS = {
    "with_edl": "#F26B38",
    "without_edl": "#12355B",
    "au": "#F2B134",
    "au_curve": "#3B7A57",
    "pd": "#5A90C8",
    "pd_curve": "#B64342",
    "dark": "#272727",
    "gray": "#767676",
    "light_gray": "#CFCECE",
    "profile_low": "#0F4D92",
    "profile_high": "#5A90C8",
    "electrolyte": "#EAF5FA",
    "inner_layer": "#F4F4F4",
}

CASE_STYLES = {
    1.0e-2: {"label": "0.01 M", "linestyle": "solid", "marker": "o"},
    1.0: {
        "label": "1 M",
        "linestyle": (0, (5.0, 2.0, 1.2, 2.0)),
        "marker": "s",
    },
    1.0e3: {
        "label": r"$10^3$ M",
        "linestyle": (0, (1.0, 2.0)),
        "marker": "^",
    },
}

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Helvetica",
        "Nimbus Sans",
        "Arial",
        "DejaVu Sans",
        "sans-serif",
    ],
    "font.size": 8.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "axes.grid": False,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Nimbus Sans",
    "mathtext.it": "Nimbus Sans:italic",
    "mathtext.bf": "Nimbus Sans:bold",
    "mathtext.cal": "Nimbus Sans",
    "mathtext.sf": "Nimbus Sans",
    "mathtext.tt": "Nimbus Sans",
    "xtick.direction": "out",
    "ytick.direction": "out",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _jsonable(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields) if fields is not None else list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_figure(fig: plt.Figure, directory: Path, stem: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in ("png", "svg"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(
            path,
            dpi=DPI,
            bbox_inches="tight",
            pad_inches=0.06,
            facecolor="white",
            edgecolor="none",
        )
        paths.append(path)
    plt.close(fig)
    return paths


def concentration_scan_values_m() -> np.ndarray:
    """Return the Figure-4-compatible scan grid, including exact checkpoints."""

    values = np.concatenate(
        [
            np.logspace(-4.0, 3.0, 33),
            np.asarray((*REPRESENTATIVE_CONCENTRATIONS_M, 10.0), dtype=float),
        ]
    )
    return np.asarray(sorted(set(np.round(values, 14))), dtype=float)


def params_at_concentration(base_params: Mapping[str, Any], concentration_m: float) -> dict[str, Any]:
    """Set concentration in SI and release all concentration-dependent overrides."""

    if not math.isfinite(float(concentration_m)) or float(concentration_m) <= 0.0:
        raise ValueError("concentration_m must be finite and positive")
    params = copy.deepcopy(canonical_params(base_params))
    params["C_tot"] = 1000.0 * float(concentration_m)
    params["lambda_D"] = None
    params["g_Au"] = None
    params["g_Pd"] = None
    return params


def _safe_exp(value: float) -> float:
    return float(math.exp(float(np.clip(value, -700.0, 700.0))))


def _surface_mechanism_metrics(
    model: IndependentPlanarEDLModel,
    with_edl: Mapping[str, Any],
    material: str,
) -> dict[str, float]:
    """Return uniform-plane means and exact kinetic reconstruction for one side."""

    if material not in ("Au", "Pd"):
        raise ValueError("material must be 'Au' or 'Pd'")
    p = model.params
    d = model.derived
    r = model.reaction
    beta = float(d["beta_per_V"])
    E_mix = float(with_edl["E_mix_V"])
    phi_tilde = float(with_edl[f"phi_RP_{material}_tilde"])
    phi_v = float(with_edl[f"phi_RP_{material}_V"])
    pzc = float(p[f"pzc_{material}"])
    c_h = float(p[f"C_H_{material}"])
    sigma = c_h * (E_mix - pzc - phi_v)
    sigma_from_diffuse = float(d["epsilon_s"]) * phi_v / float(d["lambda_D"])

    c_red1_star = _safe_exp(-float(p["z_R1"]) * phi_tilde)
    c_ox2_star = _safe_exp(-float(p["z_O2"]) * phi_tilde)
    if material == "Au":
        eta_v = E_mix - float(r["E1_eq_eff"]) - phi_v
        f_eta = (1.0 - float(p["alpha1"])) * beta * eta_v
        reactant_star = c_red1_star
        kinetic_weight = reactant_star * _safe_exp(f_eta)
        reconstructed_j = float(r["it0_1_eff"]) * kinetic_weight
        model_j = float(with_edl["j_Au_A_per_m2"])
    else:
        eta_v = E_mix - float(r["E2_eq_eff"]) - phi_v
        f_eta = -float(p["alpha2"]) * beta * eta_v
        reactant_star = c_ox2_star
        kinetic_weight = reactant_star * _safe_exp(f_eta)
        reconstructed_j = -float(r["it0_2_eff"]) * kinetic_weight
        model_j = float(with_edl["j_Pd_A_per_m2"])

    return {
        "sigma_C_per_m2": sigma,
        "sigma_uC_per_cm2": 100.0 * sigma,
        "sigma_from_diffuse_C_per_m2": sigma_from_diffuse,
        "charge_relation_residual_C_per_m2": sigma - sigma_from_diffuse,
        "phi_RP_tilde": phi_tilde,
        "phi_RP_V": phi_v,
        "phi_RP_mV": 1000.0 * phi_v,
        "eta_RP_V": eta_v,
        "eta_RP_mV": 1000.0 * eta_v,
        "eta_RP_star_beta_eta": beta * eta_v,
        "f_eta": f_eta,
        "exp_f_eta": _safe_exp(f_eta),
        "c_Red1_star": c_red1_star,
        "c_Ox2_star": c_ox2_star,
        "reactant_star": reactant_star,
        "reactant_times_exp_f_eta": kinetic_weight,
        "j_reconstructed_A_per_m2": reconstructed_j,
        "j_model_A_per_m2": model_j,
        "j_reconstruction_residual_A_per_m2": reconstructed_j - model_j,
    }


def compute_scan_row(base_params: Mapping[str, Any], concentration_m: float) -> dict[str, Any]:
    params = params_at_concentration(base_params, concentration_m)
    model = IndependentPlanarEDLModel(params)
    with_edl = model.solve(use_edl=True)
    without_edl = model.solve(use_edl=False)
    au = _surface_mechanism_metrics(model, with_edl, "Au")
    pd = _surface_mechanism_metrics(model, with_edl, "Pd")
    row: dict[str, Any] = {
        "C_tot_M": float(concentration_m),
        "C_tot_mol_per_m3": float(params["C_tot"]),
        "lambda_D_nm": 1.0e9 * float(model.derived["lambda_D"]),
        "C_D_F_per_m2": float(model.derived["epsilon_s"]) / float(model.derived["lambda_D"]),
        "g_Au": float(model.derived["g_Au"]),
        "g_Pd": float(model.derived["g_Pd"]),
        "q_Au": float(model.derived["q_Au"]),
        "q_Pd": float(model.derived["q_Pd"]),
        "E_mix_with_EDL_V": float(with_edl["E_mix_V"]),
        "E_mix_without_EDL_V": float(without_edl["E_mix_V"]),
        "delta_E_mix_mV": 1000.0
        * (float(with_edl["E_mix_V"]) - float(without_edl["E_mix_V"])),
        "i_mix_abs_with_EDL_A": float(with_edl["i_mix_abs_A"]),
        "i_mix_abs_without_EDL_A": float(without_edl["i_mix_abs_A"]),
        "i_mix_abs_with_EDL_1e_minus_3_uA": CURRENT_DISPLAY_SCALE
        * float(with_edl["i_mix_abs_A"]),
        "i_mix_abs_without_EDL_1e_minus_3_uA": CURRENT_DISPLAY_SCALE
        * float(without_edl["i_mix_abs_A"]),
        "i_mix_avg_with_EDL_A_per_m2": float(with_edl["i_mix_avg_A_per_m2"]),
        "i_mix_avg_without_EDL_A_per_m2": float(without_edl["i_mix_avg_A_per_m2"]),
        "i_mix_avg_ratio_with_over_without": float(with_edl["i_mix_avg_A_per_m2"])
        / float(without_edl["i_mix_avg_A_per_m2"]),
        "I_Au_with_EDL_A": float(with_edl["I_Au_A"]),
        "I_Pd_with_EDL_A": float(with_edl["I_Pd_A"]),
        "relative_current_balance_residual_with_EDL": float(
            with_edl["relative_balance_residual"]
        ),
        "relative_current_balance_residual_without_EDL": float(
            without_edl["relative_balance_residual"]
        ),
        "closed_form_minus_brent_with_EDL_V": float(
            with_edl["closed_form_minus_brent_V"]
        ),
        "closed_form_minus_brent_without_EDL_V": float(
            without_edl["closed_form_minus_brent_V"]
        ),
    }
    for prefix, metrics in (("Au", au), ("Pd", pd)):
        row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
    return row


def compute_scan_rows(
    base_params: Mapping[str, Any], concentrations_m: np.ndarray | None = None
) -> list[dict[str, Any]]:
    values = concentration_scan_values_m() if concentrations_m is None else concentrations_m
    return [compute_scan_row(base_params, float(value)) for value in values]


def row_at(rows: list[dict[str, Any]], concentration_m: float) -> dict[str, Any]:
    for row in rows:
        if np.isclose(
            float(row["C_tot_M"]),
            float(concentration_m),
            rtol=0.0,
            atol=max(1.0e-14, 1.0e-12 * float(concentration_m)),
        ):
            return row
    raise ValueError(f"Missing C_tot = {concentration_m:g} M")


def _log_interpolated_crossing(
    rows: list[dict[str, Any]], key: str, target: float
) -> list[float]:
    x = np.asarray([row["C_tot_M"] for row in rows], dtype=float)
    y = np.asarray([row[key] for row in rows], dtype=float) - target
    crossings: list[float] = []
    for index in range(len(x) - 1):
        if y[index] == 0.0:
            crossings.append(float(x[index]))
        elif y[index] * y[index + 1] < 0.0:
            fraction = -y[index] / (y[index + 1] - y[index])
            log_value = math.log10(x[index]) + fraction * (
                math.log10(x[index + 1]) - math.log10(x[index])
            )
            crossings.append(10.0**log_value)
    return crossings


def _regime_background(ax: plt.Axes) -> None:
    for xmin, xmax, color in (
        (1.0e-4, 1.0e-2, "#FDE9E0"),
        (1.0e-2, 1.0, "#E9F1E8"),
        (1.0, 1.0e3, "#E8EEF7"),
    ):
        ax.axvspan(xmin, xmax, color=color, alpha=0.55, linewidth=0.0, zorder=0)


def _split_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    linewidth: float,
    linestyle: str | tuple[int, tuple[float, ...]],
    label: str,
    marker: str | None = None,
    markersize: float = 0.0,
) -> None:
    normal = x <= FORMAL_HIGH_SALT_START_M
    formal = x >= FORMAL_HIGH_SALT_START_M
    ax.plot(
        x[normal],
        y[normal],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        marker=marker,
        markersize=markersize,
        label=label,
        zorder=3,
    )
    ax.plot(
        x[formal],
        y[formal],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        marker=marker,
        markersize=markersize,
        alpha=0.32,
        label="_nolegend_",
        zorder=3,
    )


def _representative_markers(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    key: str,
    offsets: Mapping[float, float],
) -> None:
    for concentration_m, horizontal_alignment in (
        (1.0e-2, "center"),
        (1.0, "center"),
        (1.0e3, "right"),
    ):
        row = row_at(rows, concentration_m)
        value = float(row[key])
        offset = float(offsets[concentration_m])
        ax.scatter(
            [concentration_m],
            [value],
            s=42,
            color=COLORS["with_edl"],
            edgecolor="white",
            linewidth=0.75,
            zorder=5,
        )
        ax.text(
            concentration_m,
            value + offset,
            CASE_STYLES[concentration_m]["label"],
            ha=horizontal_alignment,
            va="bottom" if offset > 0.0 else "top",
            fontsize=7.2,
            color=COLORS["dark"],
        )


def plot_ctot_trends(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    c = np.asarray([row["C_tot_M"] for row in rows], dtype=float)
    specifications = (
        {
            "stem": "ctot_emix_high_salt_regime_independent_edls",
            "with_key": "E_mix_with_EDL_V",
            "without_key": "E_mix_without_EDL_V",
            "ylabel": r"$E_{\mathrm{mix}}$ (V vs. RHE)",
            "title": r"$E_{\mathrm{mix}}$ shift fades in the high-salt limit",
            "ylim": (0.455, 0.665),
            "offsets": {1.0e-2: -0.014, 1.0: 0.010, 1.0e3: 0.008},
            "legend_loc": "lower left",
        },
        {
            "stem": "ctot_imix_avg_high_salt_regime_independent_edls",
            "with_key": "i_mix_avg_with_EDL_A_per_m2",
            "without_key": "i_mix_avg_without_EDL_A_per_m2",
            "ylabel": r"$\bar{i}_{\mathrm{mix}}$ (A/m$^2$)",
            "title": r"$\bar{i}_{\mathrm{mix}}$ overshoots before the high-salt limit",
            "ylim": (0.025, 0.137),
            "offsets": {1.0e-2: -0.008, 1.0: 0.006, 1.0e3: -0.006},
            "legend_loc": "lower right",
        },
    )
    saved: list[Path] = []
    for specification in specifications:
        y_with = np.asarray([row[str(specification["with_key"])] for row in rows], dtype=float)
        y_without = np.asarray(
            [row[str(specification["without_key"])] for row in rows], dtype=float
        )
        fig, ax = plt.subplots(figsize=(4.15, 3.45))
        _regime_background(ax)
        _split_line(
            ax,
            c,
            y_with,
            color=COLORS["with_edl"],
            linewidth=2.25,
            linestyle="solid",
            label="with EDL",
            marker="o",
            markersize=2.8,
        )
        _split_line(
            ax,
            c,
            y_without,
            color=COLORS["without_edl"],
            linewidth=1.8,
            linestyle=(0, (4.0, 2.5)),
            label="w/o EDL",
        )
        _representative_markers(
            ax,
            rows,
            str(specification["with_key"]),
            specification["offsets"],
        )
        ax.set_xscale("log")
        ax.set_xlim(1.0e-4, 1.0e3)
        ax.set_ylim(*specification["ylim"])
        ax.set_xlabel(r"Electrolyte concentration, $C_{\mathrm{tot}}$ (M)")
        ax.set_ylabel(str(specification["ylabel"]))
        ax.set_title(str(specification["title"]), loc="left", fontsize=9.7, pad=5.0)
        ax.legend(
            loc=str(specification["legend_loc"]),
            fontsize=7.4,
            handlelength=2.4,
        )
        ax.tick_params(length=3.2, width=0.85, labelsize=8.0)
        saved.extend(_save_figure(fig, output_dir, str(specification["stem"])))
    return saved


def compute_polarization_rows(
    base_params: Mapping[str, Any],
    e_values_v: np.ndarray = POLARIZATION_E_VALUES_V,
) -> tuple[list[dict[str, Any]], dict[float, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    cases: dict[float, dict[str, Any]] = {}
    for concentration_m in REPRESENTATIVE_CONCENTRATIONS_M:
        model = IndependentPlanarEDLModel(
            params_at_concentration(base_params, concentration_m)
        )
        solution = model.solve(use_edl=True)
        cases[concentration_m] = {"model": model, "solution": solution}
        for E_v in np.asarray(e_values_v, dtype=float):
            current = model.current_components(float(E_v), use_edl=True)
            rows.append(
                {
                    "condition": "with EDL",
                    "C_tot_M": concentration_m,
                    "E_V": float(E_v),
                    "I_Au_A": float(current["I_Au_A"]),
                    "I_Pd_A": float(current["I_Pd_A"]),
                    "I_Au_1e_minus_3_uA": CURRENT_DISPLAY_SCALE
                    * float(current["I_Au_A"]),
                    "I_Pd_1e_minus_3_uA": CURRENT_DISPLAY_SCALE
                    * float(current["I_Pd_A"]),
                    "j_Au_A_per_m2": float(current["j_Au_A_per_m2"]),
                    "j_Pd_A_per_m2": float(current["j_Pd_A_per_m2"]),
                }
            )
    reference_model = IndependentPlanarEDLModel(
        params_at_concentration(base_params, REPRESENTATIVE_CONCENTRATIONS_M[0])
    )
    for E_v in np.asarray(e_values_v, dtype=float):
        current = reference_model.current_components(float(E_v), use_edl=False)
        rows.append(
            {
                "condition": "w/o EDL reference",
                "C_tot_M": "",
                "E_V": float(E_v),
                "I_Au_A": float(current["I_Au_A"]),
                "I_Pd_A": float(current["I_Pd_A"]),
                "I_Au_1e_minus_3_uA": CURRENT_DISPLAY_SCALE
                * float(current["I_Au_A"]),
                "I_Pd_1e_minus_3_uA": CURRENT_DISPLAY_SCALE
                * float(current["I_Pd_A"]),
                "j_Au_A_per_m2": float(current["j_Au_A_per_m2"]),
                "j_Pd_A_per_m2": float(current["j_Pd_A_per_m2"]),
            }
        )
    return rows, cases


def plot_polarization_overlay(
    polarization_rows: list[dict[str, Any]],
    cases: Mapping[float, Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.2, 4.35))
    reference = [row for row in polarization_rows if row["condition"] == "w/o EDL reference"]
    e_reference = np.asarray([row["E_V"] for row in reference], dtype=float)
    for key, color in (
        ("I_Au_1e_minus_3_uA", COLORS["au_curve"]),
        ("I_Pd_1e_minus_3_uA", COLORS["pd_curve"]),
    ):
        ax.plot(
            e_reference,
            np.asarray([row[key] for row in reference], dtype=float),
            color=color,
            linewidth=1.55,
            linestyle=(0, (4.0, 2.4)),
            alpha=0.30,
            zorder=1,
        )

    for concentration_m in REPRESENTATIVE_CONCENTRATIONS_M:
        style = CASE_STYLES[concentration_m]
        subset = [
            row
            for row in polarization_rows
            if row["condition"] == "with EDL"
            and float(row["C_tot_M"]) == concentration_m
        ]
        e = np.asarray([row["E_V"] for row in subset], dtype=float)
        alpha = 1.0 if concentration_m <= 1.0 else 0.42
        ax.plot(
            e,
            np.asarray([row["I_Au_1e_minus_3_uA"] for row in subset], dtype=float),
            color=COLORS["au_curve"],
            linewidth=2.0,
            linestyle=style["linestyle"],
            alpha=alpha,
        )
        ax.plot(
            e,
            np.asarray([row["I_Pd_1e_minus_3_uA"] for row in subset], dtype=float),
            color=COLORS["pd_curve"],
            linewidth=2.0,
            linestyle=style["linestyle"],
            alpha=alpha,
        )
        solution = cases[concentration_m]["solution"]
        e_mix = float(solution["E_mix_V"])
        i_mix = CURRENT_DISPLAY_SCALE * float(solution["i_mix_abs_A"])
        ax.axvline(e_mix, color=COLORS["gray"], linewidth=0.8, alpha=0.50, zorder=0)
        ax.scatter(
            [e_mix, e_mix],
            [i_mix, -i_mix],
            marker=str(style["marker"]),
            s=48,
            facecolor=COLORS["dark"],
            edgecolor="white",
            linewidth=0.7,
            alpha=alpha,
            zorder=5,
        )

    ax.axhline(0.0, color=COLORS["dark"], linewidth=0.9)
    ax.set_xlim(0.40, 0.64)
    ax.set_ylim(-0.16, 0.16)
    ax.set_xlabel("Potential (V vs. RHE)")
    ax.set_ylabel(r"Half-reaction current (10$^{-3}$ $\mu$A)")
    ax.set_title(
        r"Salt-dependent polarization curves explain $I_{\mathrm{mix}}$",
        loc="left",
        fontsize=10.2,
    )
    half_reaction_legend = ax.legend(
        handles=[
            Line2D([0], [0], color=COLORS["au_curve"], lw=2.5, label="Au oxidation"),
            Line2D([0], [0], color=COLORS["pd_curve"], lw=2.5, label="Pd reduction"),
            Line2D(
                [0],
                [0],
                color=COLORS["gray"],
                lw=1.7,
                ls=(0, (4, 2.4)),
                label="w/o EDL ref.",
            ),
        ],
        title="Half reaction",
        loc="upper left",
        fontsize=7.6,
        title_fontsize=7.8,
    )
    ax.add_artist(half_reaction_legend)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=COLORS["dark"],
                lw=1.8,
                ls=CASE_STYLES[c]["linestyle"],
                marker=CASE_STYLES[c]["marker"],
                label=CASE_STYLES[c]["label"],
                alpha=1.0 if c <= 1.0 else 0.42,
            )
            for c in REPRESENTATIVE_CONCENTRATIONS_M
        ],
        title=r"$C_{\mathrm{tot}}$",
        loc="upper right",
        fontsize=7.6,
        title_fontsize=7.8,
    )
    ax.tick_params(length=3.2, width=0.85, labelsize=8.1)
    return _save_figure(
        fig,
        output_dir,
        "ctot_half_reaction_polarization_overlay_independent_edls",
    )


def _mechanism_curve(
    ax: plt.Axes,
    c: np.ndarray,
    values: np.ndarray,
    color: str,
) -> None:
    _regime_background(ax)
    _split_line(
        ax,
        c,
        values,
        color=color,
        linewidth=1.9,
        linestyle="solid",
        label="with EDL",
        marker=None,
    )
    for concentration_m in REPRESENTATIVE_CONCENTRATIONS_M:
        index = int(np.argmin(np.abs(c - concentration_m)))
        ax.scatter(
            [c[index]],
            [values[index]],
            s=18,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=5,
        )
    ax.set_xscale("log")
    ax.set_xlim(1.0e-4, 1.0e3)
    ax.tick_params(length=2.7, width=0.75, labelsize=7.0, pad=1.8)


def plot_mechanism(rows: list[dict[str, Any]], base_params: Mapping[str, Any], output_dir: Path) -> list[Path]:
    c = np.asarray([row["C_tot_M"] for row in rows], dtype=float)
    metric_rows = (
        (
            "sigma_uC_per_cm2",
            r"$\langle\sigma\rangle$ ($\mu$C/cm$^2$)",
            "linear",
            r"$\sigma=C_{\mathrm{H}}(E_{\mathrm{mix}}-\mathrm{PZC}-\phi_{\mathrm{RP}})$",
        ),
        (
            "phi_RP_tilde",
            r"$\langle\phi^*_{\mathrm{RP}}\rangle$ (-)",
            "linear",
            r"$\phi^*_{\mathrm{RP}}=q\beta(E_{\mathrm{mix}}-\mathrm{PZC})$",
        ),
        (
            "eta_RP_mV",
            r"$\langle\eta_{\mathrm{RP}}\rangle$ (mV)",
            "linear",
            r"$\eta_{\mathrm{RP}}=E_{\mathrm{mix}}-E_{\mathrm{eq}}-\phi_{\mathrm{RP}}$",
        ),
        (
            "exp_f_eta",
            r"$\langle e^{f_\eta}\rangle$ (-)",
            "log",
            r"$f_{\eta,\mathrm{Au}}=(1-\alpha_1)\beta\eta$; $f_{\eta,\mathrm{Pd}}=-\alpha_2\beta\eta$",
        ),
        (
            "reactant_star",
            "reactant",
            "log",
            r"$c_i^*=c_i/c_{i,\mathrm{bulk}}=e^{-z_i\phi^*_{\mathrm{RP}}}$",
        ),
        (
            "reactant_times_exp_f_eta",
            "weight",
            "log",
            r"$\langle c_i^*e^{f_\eta}\rangle$",
        ),
    )
    fig = plt.figure(figsize=(7.0, 11.8))
    grid = fig.add_gridspec(
        7,
        2,
        height_ratios=(1, 1, 1, 1, 1, 1, 1.08),
        left=0.105,
        right=0.98,
        bottom=0.065,
        top=0.875,
        hspace=0.53,
        wspace=0.30,
    )
    for row_index, (metric, generic_label, scale, formula) in enumerate(metric_rows):
        shared_y_axis = None
        for column, material in enumerate(("Au", "Pd")):
            # The paired Au/Pd panels show the same physical quantity, so they
            # must use identical y limits and ticks for a direct comparison.
            ax = fig.add_subplot(
                grid[row_index, column],
                sharey=shared_y_axis,
            )
            if shared_y_axis is None:
                shared_y_axis = ax
            key = f"{material}_{metric}"
            values = np.asarray([row[key] for row in rows], dtype=float)
            color = COLORS["with_edl"] if material == "Au" else COLORS["profile_low"]
            _mechanism_curve(ax, c, values, color)
            if scale == "log":
                ax.set_yscale("log")
            if generic_label == "reactant":
                secondary_key = (
                    f"{material}_c_Ox2_star"
                    if material == "Au"
                    else f"{material}_c_Red1_star"
                )
                secondary = np.asarray([row[secondary_key] for row in rows], dtype=float)
                _split_line(
                    ax,
                    c,
                    secondary,
                    color=COLORS["gray"],
                    linewidth=1.25,
                    linestyle=(0, (3.2, 2.2)),
                    label="secondary species",
                )
                label = r"$\langle c_i^*\rangle$ (-)"
                primary_label = (
                    r"$c^*_{\mathrm{Red}_1}$ (Au reactant)"
                    if material == "Au"
                    else r"$c^*_{\mathrm{Ox}_2}$ (Pd reactant)"
                )
                secondary_label = (
                    r"$c^*_{\mathrm{Ox}_2}$"
                    if material == "Au"
                    else r"$c^*_{\mathrm{Red}_1}$"
                )
                ax.legend(
                    handles=[
                        Line2D([0], [0], color=color, lw=1.8, label=primary_label),
                        Line2D(
                            [0],
                            [0],
                            color=COLORS["gray"],
                            lw=1.25,
                            ls=(0, (3.2, 2.2)),
                            label=secondary_label,
                        ),
                    ],
                    loc="upper right",
                    fontsize=5.2,
                    handlelength=1.8,
                )
            elif generic_label == "weight":
                label = (
                    r"$\langle c^*_{\mathrm{Red}_1}e^{f_\eta}\rangle$ (-)"
                    if material == "Au"
                    else r"$\langle c^*_{\mathrm{Ox}_2}e^{f_\eta}\rangle$ (-)"
                )
            else:
                label = generic_label
            ax.set_ylabel(label, fontsize=7.6, labelpad=2.6)
            if row_index < len(metric_rows) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(r"$C_{\mathrm{tot}}$ (M)", fontsize=7.8)
            if row_index == 0:
                title = (
                    r"Au oxidation: $\mathrm{Red}_1^-$"
                    if material == "Au"
                    else r"Pd reduction: $\mathrm{Ox}_2^+$"
                )
                ax.set_title(title, color=color, fontsize=9.2, pad=7.0)
            ax.text(
                0.985,
                0.08,
                formula,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=5.7,
                color=COLORS["gray"],
            )

    current_ax = fig.add_subplot(grid[6, :])
    _regime_background(current_ax)
    i_with = np.asarray(
        [row["i_mix_avg_with_EDL_A_per_m2"] for row in rows], dtype=float
    )
    i_without = np.asarray(
        [row["i_mix_avg_without_EDL_A_per_m2"] for row in rows], dtype=float
    )
    _split_line(
        current_ax,
        c,
        i_with,
        color=COLORS["with_edl"],
        linewidth=2.0,
        linestyle="solid",
        label="with EDL",
    )
    _split_line(
        current_ax,
        c,
        i_without,
        color=COLORS["without_edl"],
        linewidth=1.6,
        linestyle=(0, (4.0, 2.4)),
        label="w/o EDL",
    )
    for concentration_m in REPRESENTATIVE_CONCENTRATIONS_M:
        point = row_at(rows, concentration_m)
        current_ax.scatter(
            [concentration_m],
            [point["i_mix_avg_with_EDL_A_per_m2"]],
            s=26,
            color=COLORS["with_edl"],
            edgecolor="white",
            linewidth=0.55,
            zorder=5,
        )
    current_ax.set_xscale("log")
    current_ax.set_yscale("log")
    current_ax.set_xlim(1.0e-4, 1.0e3)
    current_ax.set_xlabel(r"Electrolyte concentration, $C_{\mathrm{tot}}$ (M)")
    current_ax.set_ylabel(r"$\bar{i}_{\mathrm{mix}}$ (A/m$^2$)")
    current_ax.legend(loc="lower right", fontsize=7.3, ncol=2, handlelength=2.2)
    current_ax.tick_params(length=3.0, width=0.8, labelsize=7.2)
    current_ax.text(
        0.01,
        0.08,
        r"$I_{\mathrm{Au}}+I_{\mathrm{Pd}}=0$; "
        r"$\bar{i}_{\mathrm{mix}}=|I_{\mathrm{Au}}|/(A_{\mathrm{Au}}+A_{\mathrm{Pd}})$",
        transform=current_ax.transAxes,
        fontsize=6.2,
        color=COLORS["gray"],
    )

    p = canonical_params(base_params)
    fig.suptitle(
        r"$C_{\mathrm{tot}}$ mechanism in two independent planar EDLs",
        x=0.105,
        y=0.975,
        ha="left",
        fontsize=12.0,
    )
    fig.text(
        0.105,
        0.932,
        (
            r"$C_{\mathrm{H,Au}}=$"
            f"{100.0 * float(p['C_H_Au']):.0f} and "
            r"$C_{\mathrm{H,Pd}}=$"
            f"{100.0 * float(p['C_H_Pd']):.0f} "
            r"$\mu$F cm$^{-2}$; PZC values are fixed. "
            r"$C_{\mathrm{tot}}$ changes $\lambda_D$, $g=C_{\mathrm{H}}/C_D$, and $q=g/(1+g)$."
        ),
        ha="left",
        va="top",
        fontsize=7.4,
        color=COLORS["gray"],
    )
    fig.text(
        0.105,
        0.905,
        "Each angle-bracket quantity equals the analytic local value because each independent planar face is uniform; both ionic species are shown, with the reacting species in the surface color.",
        ha="left",
        va="top",
        fontsize=7.1,
        color=COLORS["gray"],
    )
    return _save_figure(
        fig,
        output_dir,
        "ctot_surface_mechanism_causal_chain_independent_edls",
    )


def compute_profile_rows(
    base_params: Mapping[str, Any],
    n_distance: int = 321,
) -> tuple[list[dict[str, Any]], dict[str, dict[float, dict[str, Any]]], np.ndarray]:
    low_model = IndependentPlanarEDLModel(
        params_at_concentration(base_params, PROFILE_CONCENTRATIONS_M[0])
    )
    max_distance_nm = 5.0 * float(low_model.derived["lambda_D"]) * 1.0e9
    distance_nm = np.linspace(0.0, max_distance_nm, n_distance)
    curves: dict[str, dict[float, dict[str, Any]]] = {"Au": {}, "Pd": {}}
    rows: list[dict[str, Any]] = []
    for concentration_m in PROFILE_CONCENTRATIONS_M:
        model = IndependentPlanarEDLModel(
            params_at_concentration(base_params, concentration_m)
        )
        solution = model.solve(use_edl=True)
        for material in ("Au", "Pd"):
            phi_tilde = model.phi_tilde_profile(
                float(solution["E_mix_V"]), material, distance_nm * 1.0e-9
            )
            phi_v = phi_tilde * float(model.derived["thermal_voltage_V"])
            curves[material][concentration_m] = {
                "model": model,
                "solution": solution,
                "phi_tilde": phi_tilde,
                "phi_v": phi_v,
            }
            for index, distance in enumerate(distance_nm):
                rows.append(
                    {
                        "material": material,
                        "condition": "with EDL",
                        "curve_source": "analytic independent-planar linear PB",
                        "C_tot_M": concentration_m,
                        "C_tot_mol_per_m3": 1000.0 * concentration_m,
                        "E_mix_V": float(solution["E_mix_V"]),
                        "lambda_D_nm": 1.0e9 * float(model.derived["lambda_D"]),
                        "distance_from_RP_nm": float(distance),
                        "phi_bar_tilde": float(phi_tilde[index]),
                        "phi_bar_V": float(phi_v[index]),
                        "phi_bar_mV": 1000.0 * float(phi_v[index]),
                    }
                )
    reference_model = IndependentPlanarEDLModel(
        params_at_concentration(base_params, PROFILE_CONCENTRATIONS_M[0])
    )
    reference_solution = reference_model.solve(use_edl=False)
    for material in ("Au", "Pd"):
        for distance in distance_nm:
            rows.append(
                {
                    "material": material,
                    "condition": "w/o EDL",
                    "curve_source": "schematic zero line required by use_edl=False definition",
                    "C_tot_M": "",
                    "C_tot_mol_per_m3": "",
                    "E_mix_V": float(reference_solution["E_mix_V"]),
                    "lambda_D_nm": "",
                    "distance_from_RP_nm": float(distance),
                    "phi_bar_tilde": 0.0,
                    "phi_bar_V": 0.0,
                    "phi_bar_mV": 0.0,
                }
            )
    return rows, curves, distance_nm


def plot_profile_schematic(
    curves: Mapping[str, Mapping[float, Mapping[str, Any]]],
    distance_nm: np.ndarray,
    output_dir: Path,
) -> list[Path]:
    def compact_tangent_start_tilde(
        curve_data: Mapping[str, Any],
    ) -> float:
        phi = np.asarray(curve_data["phi_tilde"], dtype=float)
        lambda_d_nm = 1.0e9 * float(curve_data["model"].derived["lambda_D"])
        ohp_slope_per_nm = -float(phi[0]) / lambda_d_nm
        return float(phi[0]) + ohp_slope_per_nm * (COMPACT_LEFT_NM - RP_X_NM)

    curve_data_all = [
        curves[material][concentration_m]
        for material in ("Au", "Pd")
        for concentration_m in PROFILE_CONCENTRATIONS_M
    ]
    all_phi_tilde = np.concatenate(
        [np.asarray(curve_data["phi_tilde"], dtype=float) for curve_data in curve_data_all]
        + [
            np.asarray([compact_tangent_start_tilde(curve_data)], dtype=float)
            for curve_data in curve_data_all
        ]
    )
    common_lower = min(float(np.min(all_phi_tilde)) * 1.10, -0.5)
    common_upper = 0.55
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(4.15, 3.45),
        sharex=True,
        sharey=True,
    )
    for index, (ax, material) in enumerate(zip(axes, ("Au", "Pd"), strict=True)):
        lower = common_lower
        upper = common_upper
        metal_color = COLORS["au"] if material == "Au" else COLORS["pd"]
        reaction_label = r"$\mathrm{Red}_1^-$" if material == "Au" else r"$\mathrm{Ox}_2^+$"
        reaction_color = COLORS["with_edl"] if material == "Au" else "#D4A923"
        ax.axvspan(METAL_LEFT_NM, COMPACT_LEFT_NM, color=metal_color, alpha=0.95, zorder=0)
        ax.axvspan(COMPACT_LEFT_NM, RP_X_NM, color=COLORS["inner_layer"], alpha=1.0, zorder=0)
        ax.axvspan(RP_X_NM, float(distance_nm[-1]), color=COLORS["electrolyte"], alpha=0.55, zorder=0)
        ax.axvline(RP_X_NM, color=COLORS["dark"], linewidth=1.0, zorder=2)
        for concentration_m, color in (
            (1.0e-2, COLORS["profile_low"]),
            (1.0, COLORS["profile_high"]),
        ):
            style = CASE_STYLES[concentration_m]
            curve_data = curves[material][concentration_m]
            curve = np.asarray(curve_data["phi_tilde"], dtype=float)
            ax.plot(
                [COMPACT_LEFT_NM, RP_X_NM],
                [compact_tangent_start_tilde(curve_data), float(curve[0])],
                color=color,
                linewidth=1.55,
                linestyle=style["linestyle"],
                zorder=3,
            )
            ax.plot(
                distance_nm,
                curve,
                color=color,
                linewidth=1.55,
                linestyle=style["linestyle"],
                label=f"with EDL, {style['label']}",
                zorder=3,
            )
            ax.scatter(
                [0.0],
                [curve[0]],
                s=24,
                color=color,
                edgecolor="white",
                linewidth=0.55,
                zorder=4,
            )
        ax.plot(
            [COMPACT_LEFT_NM, float(distance_nm[-1])],
            [0.0, 0.0],
            color=COLORS["without_edl"],
            linewidth=1.45,
            linestyle=(0, (5.0, 2.6)),
            label="w/o EDL",
            zorder=6,
        )
        ax.set_ylim(lower, upper)
        ax.set_xlim(METAL_LEFT_NM, float(distance_nm[-1]))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(length=0)
        ax.text(
            0.5 * (METAL_LEFT_NM + COMPACT_LEFT_NM),
            0.50 * (lower + upper),
            material,
            ha="center",
            va="center",
            rotation=90,
            fontsize=9.0,
            fontweight="bold",
            color=COLORS["dark"],
        )
        if index == 0:
            ax.text(
                RP_X_NM,
                upper + 0.035 * (upper - lower),
                "OHP/RP",
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
                color=COLORS["dark"],
                clip_on=False,
            )
        ax.text(
            0.74 * float(distance_nm[-1]),
            lower + 0.29 * (upper - lower),
            reaction_label,
            ha="center",
            va="center",
            fontsize=11.0,
            color=reaction_color,
        )
        if index == 0:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper right",
                bbox_to_anchor=(0.965, 0.825),
                ncol=3,
                fontsize=6.1,
                handlelength=2.2,
                columnspacing=1.15,
            )
    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.045, top=0.96, hspace=0.30)
    return _save_figure(
        fig,
        output_dir,
        "ctot_phi_bar_profiles_0p01M_1M_with_without_edl_independent_edls",
    )

def _validation(rows: list[dict[str, Any]], profile_rows: list[dict[str, Any]]) -> dict[str, Any]:
    e_no = np.asarray([row["E_mix_without_EDL_V"] for row in rows], dtype=float)
    i_no = np.asarray(
        [row["i_mix_avg_without_EDL_A_per_m2"] for row in rows], dtype=float
    )
    max_charge_residual = max(
        abs(float(row[f"{material}_charge_relation_residual_C_per_m2"]))
        for row in rows
        for material in ("Au", "Pd")
    )
    max_kinetic_residual = max(
        abs(float(row[f"{material}_j_reconstruction_residual_A_per_m2"]))
        for row in rows
        for material in ("Au", "Pd")
    )
    max_balance = max(
        max(
            float(row["relative_current_balance_residual_with_EDL"]),
            float(row["relative_current_balance_residual_without_EDL"]),
        )
        for row in rows
    )
    max_root_difference = max(
        max(
            abs(float(row["closed_form_minus_brent_with_EDL_V"])),
            abs(float(row["closed_form_minus_brent_without_EDL_V"])),
        )
        for row in rows
    )
    max_without_profile = max(
        abs(float(row["phi_bar_tilde"]))
        for row in profile_rows
        if row["condition"] == "w/o EDL"
    )
    validation = {
        "scan_rows": len(rows),
        "profile_rows": len(profile_rows),
        "max_no_edl_E_mix_variation_V": float(np.ptp(e_no)),
        "max_no_edl_i_mix_avg_variation_A_per_m2": float(np.ptp(i_no)),
        "max_abs_charge_relation_residual_C_per_m2": max_charge_residual,
        "max_abs_kinetic_reconstruction_residual_A_per_m2": max_kinetic_residual,
        "max_relative_current_balance_residual": max_balance,
        "max_abs_closed_form_minus_brent_V": max_root_difference,
        "max_abs_without_edl_profile_phi_tilde": max_without_profile,
    }
    validation["passed"] = bool(
        validation["max_no_edl_E_mix_variation_V"] < 1.0e-12
        and validation["max_no_edl_i_mix_avg_variation_A_per_m2"] < 1.0e-12
        and validation["max_abs_charge_relation_residual_C_per_m2"] < 1.0e-12
        and validation["max_abs_kinetic_reconstruction_residual_A_per_m2"] < 1.0e-12
        and validation["max_relative_current_balance_residual"] < 1.0e-10
        and validation["max_abs_closed_form_minus_brent_V"] < 5.0e-11
        and validation["max_abs_without_edl_profile_phi_tilde"] == 0.0
    )
    return validation


def _build_ctot_results_in_place(
    params: Mapping[str, Any], output: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    base_params = canonical_params(params)
    scan_rows = compute_scan_rows(base_params)
    polarization_rows, polarization_cases = compute_polarization_rows(base_params)
    profile_rows, profile_curves, distance_nm = compute_profile_rows(base_params)

    figure4_dir = output / "figures" / "Figure_4"
    mechanism_dir = output / "figures" / "Mechanism"
    scheme_dir = output / "figures" / "EDL_scheme"
    with plt.rc_context(RC):
        figure4_paths = plot_ctot_trends(scan_rows, figure4_dir)
        figure4_paths.extend(
            plot_polarization_overlay(
                polarization_rows,
                polarization_cases,
                figure4_dir,
            )
        )
        mechanism_paths = plot_mechanism(scan_rows, base_params, mechanism_dir)
        scheme_paths = plot_profile_schematic(profile_curves, distance_nm, scheme_dir)

    csv_dir = output / "csv"
    _write_csv(csv_dir / "ctot_scan.csv", scan_rows)
    mechanism_fields = [
        "C_tot_M",
        "C_tot_mol_per_m3",
        "lambda_D_nm",
        "C_D_F_per_m2",
        "g_Au",
        "g_Pd",
        "q_Au",
        "q_Pd",
        "E_mix_with_EDL_V",
        "E_mix_without_EDL_V",
        "i_mix_avg_with_EDL_A_per_m2",
        "i_mix_avg_without_EDL_A_per_m2",
        "i_mix_avg_ratio_with_over_without",
    ]
    for material in ("Au", "Pd"):
        mechanism_fields.extend(
            [
                f"{material}_sigma_C_per_m2",
                f"{material}_sigma_uC_per_cm2",
                f"{material}_phi_RP_tilde",
                f"{material}_phi_RP_V",
                f"{material}_phi_RP_mV",
                f"{material}_eta_RP_V",
                f"{material}_eta_RP_mV",
                f"{material}_eta_RP_star_beta_eta",
                f"{material}_f_eta",
                f"{material}_exp_f_eta",
                f"{material}_c_Red1_star",
                f"{material}_c_Ox2_star",
                f"{material}_reactant_star",
                f"{material}_reactant_times_exp_f_eta",
                f"{material}_j_model_A_per_m2",
            ]
        )
    _write_csv(
        csv_dir / "ctot_mechanism_metrics_Au_Pd.csv",
        scan_rows,
        mechanism_fields,
    )
    representative_rows = [
        row_at(scan_rows, concentration_m)
        for concentration_m in (*REPRESENTATIVE_CONCENTRATIONS_M, 10.0)
    ]
    _write_csv(csv_dir / "ctot_representative_cases.csv", representative_rows)
    _write_csv(csv_dir / "ctot_polarization_curves.csv", polarization_rows)
    _write_csv(
        csv_dir / "ctot_phi_bar_profiles_0p01M_1M_with_without_edl.csv",
        profile_rows,
    )

    baseline_model = IndependentPlanarEDLModel(
        params_at_concentration(base_params, 1.0e-2)
    )
    peak_row = max(scan_rows, key=lambda row: float(row["i_mix_avg_with_EDL_A_per_m2"]))
    representative = {
        CASE_STYLES[concentration_m]["label"]: row_at(scan_rows, concentration_m)
        for concentration_m in REPRESENTATIVE_CONCENTRATIONS_M
    }
    summary = {
        "representative_cases": representative,
        "scan_peak_on_sampled_grid": {
            "C_tot_M": peak_row["C_tot_M"],
            "E_mix_with_EDL_V": peak_row["E_mix_with_EDL_V"],
            "i_mix_avg_with_EDL_A_per_m2": peak_row[
                "i_mix_avg_with_EDL_A_per_m2"
            ],
            "i_mix_avg_ratio_with_over_without": peak_row[
                "i_mix_avg_ratio_with_over_without"
            ],
        },
        "i_mix_ratio_unity_crossings_M_log_interpolated": _log_interpolated_crossing(
            scan_rows, "i_mix_avg_ratio_with_over_without", 1.0
        ),
        "formal_high_salt_extension": {
            "starts_at_M": FORMAL_HIGH_SALT_START_M,
            "ends_at_M": 1.0e3,
            "interpretation": (
                "Used only to display the mathematical high-salt approach; "
                "10^3 M is not a physically realizable electrolyte concentration."
            ),
        },
    }
    validation = _validation(scan_rows, profile_rows)
    if not validation["passed"]:
        raise RuntimeError(f"C_tot study validation failed: {validation}")

    scan_config = {
        "scan_parameter": "C_tot",
        "input_unit": "M",
        "model_parameter_unit": "mol/m^3",
        "conversion": "C_tot_mol_per_m3 = 1000 * C_tot_M",
        "scan_values_M": concentration_scan_values_m(),
        "representative_concentrations_M": REPRESENTATIVE_CONCENTRATIONS_M,
        "profile_concentrations_M": PROFILE_CONCENTRATIONS_M,
        "formal_high_salt_start_M": FORMAL_HIGH_SALT_START_M,
        "released_overrides_each_scan_point": ["lambda_D", "g_Au", "g_Pd"],
        "surface_average_convention": (
            "Each independent planar face is uniform, so angle-bracket means equal "
            "the corresponding analytic local scalar."
        ),
        "without_edl_profile_convention": (
            "Hand-drawn/model-defined reference phi_tilde(x)=0 throughout the solution."
        ),
        "mechanism_formulas": {
            "sigma_M": "C_H,M * (E_mix - PZC_M - phi_RP,M)",
            "phi_RP_M_tilde": "q_M * beta * (E_mix - PZC_M)",
            "eta_Au": "E_mix - E1_eq_eff - phi_RP,Au",
            "eta_Pd": "E_mix - E2_eq_eff - phi_RP,Pd",
            "exp_f_eta_Au": "exp((1-alpha1)*beta*eta_Au)",
            "exp_f_eta_Pd": "exp(-alpha2*beta*eta_Pd)",
            "c_Red1_star": "exp(-z_R1*phi_RP_tilde)",
            "c_Ox2_star": "exp(-z_O2*phi_RP_tilde)",
            "kinetic_weight_Au": "c_Red1_star * exp_f_eta_Au",
            "kinetic_weight_Pd": "c_Ox2_star * exp_f_eta_Pd",
            "i_mix_avg": "abs(I_Au)/(area_Au+area_Pd)",
        },
    }

    _write_json(output / "params.json", base_params)
    _write_json(output / "derived_at_0p01M.json", baseline_model.derived)
    _write_json(output / "scan_config.json", scan_config)
    _write_json(output / "summary.json", summary)
    _write_json(output / "validation.json", validation)

    pngs = sorted(output.glob("figures/**/*.png"))
    svgs = sorted(output.glob("figures/**/*.svg"))
    pdfs = sorted(output.glob("**/*.pdf"))
    if len(pngs) != 5 or len(svgs) != 5 or pdfs:
        raise RuntimeError(
            f"Expected 5 PNG, 5 SVG, 0 PDF; got {len(pngs)}, {len(svgs)}, {len(pdfs)}"
        )
    source_files = {
        "ctot_study.py": Path(__file__).resolve(),
        "model.py": Path(__file__).resolve().with_name("model.py"),
    }
    manifest = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "study": "electrolyte_concentration_C_tot",
        "model_id": MODEL_ID,
        "topology_id": TOPOLOGY_ID,
        "electrostatic_backend": ELECTROSTATIC_BACKEND,
        "electronic_constraint": "ideal wire; shared E_mix; I_Au + I_Pd = 0",
        "electrolyte_reference": (
            "separate local half-spaces with a common bulk solution-potential reference"
        ),
        "created_local": datetime.now().astimezone().isoformat(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "execution": {
            "argv": sys.argv,
            "atomic_output_staging": True,
        },
        "content_sha256": {
            "params.json": _sha256_file(output / "params.json"),
            "scan_config.json": _sha256_file(output / "scan_config.json"),
            "source": {
                label: _sha256_file(path) for label, path in source_files.items()
            },
        },
        "output_path_policy": "artifact paths are relative to run root",
        "figure_counts": {"png": len(pngs), "svg": len(svgs), "pdf": len(pdfs)},
        "scan_range_M": [float(concentration_scan_values_m()[0]), float(concentration_scan_values_m()[-1])],
        "scan_points": len(scan_rows),
        "formal_high_salt_extension_starts_M": FORMAL_HIGH_SALT_START_M,
    }
    _write_json(output / "run_manifest.json", manifest)
    artifacts = {
        "Figure_4": [str(path.relative_to(output)) for path in figure4_paths],
        "Mechanism": [str(path.relative_to(output)) for path in mechanism_paths],
        "EDL_scheme": [str(path.relative_to(output)) for path in scheme_paths],
        "csv": [
            "csv/ctot_scan.csv",
            "csv/ctot_mechanism_metrics_Au_Pd.csv",
            "csv/ctot_representative_cases.csv",
            "csv/ctot_polarization_curves.csv",
            "csv/ctot_phi_bar_profiles_0p01M_1M_with_without_edl.csv",
        ],
        "metadata": [
            "params.json",
            "derived_at_0p01M.json",
            "scan_config.json",
            "summary.json",
            "validation.json",
            "run_manifest.json",
            "artifacts.json",
        ],
        "self_checksum_note": (
            "artifacts.json lists itself but cannot contain a stable checksum of itself"
        ),
    }
    registered_paths = sorted(
        {
            relative
            for category in ("Figure_4", "Mechanism", "EDL_scheme", "csv", "metadata")
            for relative in artifacts[category]
            if relative != "artifacts.json"
        }
    )
    artifacts["sha256"] = {
        relative: _sha256_file(output / relative) for relative in registered_paths
    }
    artifacts["size_bytes"] = {
        relative: (output / relative).stat().st_size for relative in registered_paths
    }
    _write_json(output / "artifacts.json", artifacts)
    return {
        "output": str(output),
        "manifest": manifest,
        "summary": summary,
        "validation": validation,
    }


def build_ctot_results(params: Mapping[str, Any], output: Path) -> dict[str, Any]:
    """Build in a temporary sibling and expose the final run atomically."""

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.",
        dir=output.parent,
    ) as temporary_parent:
        staged_output = Path(temporary_parent) / output.name
        result = _build_ctot_results_in_place(params, staged_output)
        staged_output.replace(output)
    result["output"] = str(output)
    return result


def default_output(root: Path = PACKAGE_ROOT) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return root / "results" / f"{stamp}_ctot_study"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--params",
        type=Path,
        default=PACKAGE_ROOT / "params_template.json",
    )
    result.add_argument("--output", type=Path, default=None)
    return result


def main() -> int:
    args = parser().parse_args()
    params = load_params(args.params.resolve())
    output = (
        args.output.resolve()
        if args.output is not None
        else default_output(PACKAGE_ROOT).resolve()
    )
    result = build_ctot_results(params, output)
    print(json.dumps(result["manifest"]["figure_counts"], sort_keys=True))
    print(result["output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
