"""PZC study for two independent planar Au/Pd electrical double layers.

The two one-factor scans mirror the PZC mechanism, EDL-scheme, and
polarization figures in ``Figures/Figure_5_6``.  A point-by-point comparison
against the existing Au|support|Pd data is included in every result bundle.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, MaxNLocator

from .ctot_study import (
    COLORS,
    CURRENT_DISPLAY_SCALE,
    DPI,
    PACKAGE_ROOT,
    RC,
    _save_figure,
    _sha256_file,
    _surface_mechanism_metrics,
    _write_csv,
    _write_json,
)
from .model import (
    ELECTROSTATIC_BACKEND,
    MODEL_ID,
    RESULT_SCHEMA_VERSION,
    TOPOLOGY_ID,
    IndependentPlanarEDLModel,
    canonical_params,
    load_params,
)


PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_REFERENCE_SWEEP_CSV = (
    PROJECT_ROOT
    / "Figures"
    / "Figure_5_6"
    / "Mechanism"
    / "csv"
    / "figure_5_process_sweeps_same_length_i0_alpha050_au25_pd25_20260528_111255.csv"
)

N_SWEEP = 41
N_POLARIZATION_POINTS = 480
N_PROFILE_POINTS = 181
PROFILE_END_LAMBDA = 5.0
METAL_X0_NM = -2.35
COMPACT_X0_NM = -1.18
RP_X_NM = 0.0

PZC_COLORS = {
    "dark": "#272727",
    "gray": "#767676",
    "light_gray": "#D2D2D2",
    "with_edl": "#F26B38",
    "without_edl": "#12355B",
    "au": "#E9AE35",
    "pd": "#5A90C8",
    "au_curve": "#3B7A57",
    "pd_curve": "#B64342",
    "low": "#767676",
    "base": "#111111",
    "high": "#0F4D92",
    "electrolyte": "#EAF5FA",
    "compact": "#F4F4F4",
}

CASE_STYLES: dict[str, dict[str, Any]] = {
    "low": {
        "color": PZC_COLORS["low"],
        "linestyle": (0, (3.0, 2.0)),
        "linewidth": 1.55,
        "alpha": 0.62,
    },
    "base": {
        "color": PZC_COLORS["base"],
        "linestyle": "solid",
        "linewidth": 1.95,
        "alpha": 1.0,
    },
    "high": {
        "color": PZC_COLORS["high"],
        "linestyle": (0, (5.0, 2.0)),
        "linewidth": 1.75,
        "alpha": 0.82,
    },
}

AU_CMAP = LinearSegmentedColormap.from_list(
    "independent_pzc_au",
    ["#F7B49A", "#F26B38", "#B83A31"],
)
PD_CMAP = LinearSegmentedColormap.from_list(
    "independent_pzc_pd",
    ["#AFCBE5", "#5A90C8", "#0F4D92"],
)


@dataclass(frozen=True)
class PZCScanSpec:
    key: str
    side: str
    low: float
    baseline: float
    high: float
    title: str
    reactant: str

    @property
    def opposite_side(self) -> str:
        return "Pd" if self.side == "Au" else "Au"


SCAN_SPECS = (
    PZCScanSpec(
        key="pzc_Au",
        side="Au",
        low=0.63,
        baseline=0.93,
        high=1.23,
        title=r"$\mathrm{PZC}_{\mathrm{Au}}$",
        reactant=r"$\mathrm{Red}_1^-$",
    ),
    PZCScanSpec(
        key="pzc_Pd",
        side="Pd",
        low=0.48,
        baseline=0.78,
        high=1.08,
        title=r"$\mathrm{PZC}_{\mathrm{Pd}}$",
        reactant=r"$\mathrm{Ox}_2^+$",
    ),
)
SPEC_BY_KEY = {spec.key: spec for spec in SCAN_SPECS}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    scale: str
    value_format: str
    formula_au: str
    formula_pd: str
    without_edl_key: str | None = None


MECHANISM_METRICS = (
    MetricSpec(
        "sigma_uC_per_cm2",
        r"$\sigma$ ($\mu$C/cm$^2$)",
        "linear",
        ".2f",
        r"$\sigma_{\rm Au}=C_{\rm H,Au}(E_{\rm mix}-{\rm PZC}_{\rm Au}-\phi_{\rm RP})$",
        r"$\sigma_{\rm Pd}=C_{\rm H,Pd}(E_{\rm mix}-{\rm PZC}_{\rm Pd}-\phi_{\rm RP})$",
    ),
    MetricSpec(
        "phi_RP_tilde",
        r"$\langle\phi^*_{\mathrm{RP}}\rangle$ (-)",
        "linear",
        ".2f",
        r"$\phi^*_{\rm RP}=q_{\rm Au}\beta(E_{\rm mix}-{\rm PZC}_{\rm Au})$",
        r"$\phi^*_{\rm RP}=q_{\rm Pd}\beta(E_{\rm mix}-{\rm PZC}_{\rm Pd})$",
    ),
    MetricSpec(
        "reactant_star",
        r"$\langle c^*_{\mathrm{react}}\rangle$ (-)",
        "log",
        ".1e",
        r"$c^*_{\mathrm{Red}_1}=e^{-z_{\mathrm{Red}_1}\phi^*_{\rm RP}}$",
        r"$c^*_{\mathrm{Ox}_2}=e^{-z_{\mathrm{Ox}_2}\phi^*_{\rm RP}}$",
    ),
    MetricSpec(
        "eta_RP_mV",
        r"$\langle\eta_{\mathrm{RP}}\rangle$ (mV)",
        "linear",
        ".0f",
        r"$\eta_{\rm Au}=E_{\rm mix}-E_{1,\rm eq}-\phi_{\rm RP}$",
        r"$\eta_{\rm Pd}=E_{\rm mix}-E_{2,\rm eq}-\phi_{\rm RP}$",
    ),
    MetricSpec(
        "exp_f_eta",
        r"$\langle e^{f_\eta}\rangle$ (-)",
        "log",
        ".1e",
        r"$f_{\eta,\rm Au}=(1-\alpha_1)\beta\eta_{\rm Au}$",
        r"$f_{\eta,\rm Pd}=-\alpha_2\beta\eta_{\rm Pd}$",
    ),
    MetricSpec(
        "reactant_times_exp_f_eta",
        r"$\langle c^*_{\mathrm{react}}e^{f_\eta}\rangle$ (-)",
        "log",
        ".1e",
        r"$\langle c^*_{\mathrm{Red}_1}e^{f_\eta}\rangle$",
        r"$\langle c^*_{\mathrm{Ox}_2}e^{f_\eta}\rangle$",
    ),
    MetricSpec(
        "i_mix_avg_with_EDL_A_per_m2",
        r"$\bar{i}_{\mathrm{mix}}$ (A/m$^2$)",
        "log",
        ".2e",
        r"$\bar{i}_{\rm mix}=|I_{\rm Au}|/(A_{\rm Au}+A_{\rm Pd})$",
        r"$\bar{i}_{\rm mix}=|I_{\rm Pd}|/(A_{\rm Au}+A_{\rm Pd})$",
        "i_mix_avg_without_EDL_A_per_m2",
    ),
)


def pzc_scan_values(spec: PZCScanSpec) -> np.ndarray:
    """Return 41 linear points with the baseline inserted exactly."""

    half = N_SWEEP // 2
    lower = np.linspace(spec.low, spec.baseline, half + 1)
    upper = np.linspace(spec.baseline, spec.high, half + 1)[1:]
    values = np.concatenate((lower, upper))
    values[half] = spec.baseline
    return values


def params_at_pzc(
    base_params: Mapping[str, Any], spec: PZCScanSpec, pzc_v: float
) -> dict[str, Any]:
    if not math.isfinite(float(pzc_v)):
        raise ValueError("PZC value must be finite")
    params = copy.deepcopy(canonical_params(base_params))
    params[spec.key] = float(pzc_v)
    return params


def _mechanism_aliases(
    metrics: Mapping[str, float], material: str
) -> dict[str, float]:
    reactant = (
        float(metrics["c_Red1_star"])
        if material == "Au"
        else float(metrics["c_Ox2_star"])
    )
    return {
        "sigma_C_per_m2": float(metrics["sigma_C_per_m2"]),
        "sigma_uC_per_cm2": float(metrics["sigma_uC_per_cm2"]),
        "phi_RP_tilde": float(metrics["phi_RP_tilde"]),
        "phi_RP_mean_V": float(metrics["phi_RP_V"]),
        "phi_RP_mean_mV": float(metrics["phi_RP_mV"]),
        "reactant_star": reactant,
        "eta_RP_V": float(metrics["eta_RP_V"]),
        "eta_RP_mV": float(metrics["eta_RP_mV"]),
        "f_eta": float(metrics["f_eta"]),
        "exp_f_eta": float(metrics["exp_f_eta"]),
        "c_Red1_star": float(metrics["c_Red1_star"]),
        "c_Ox2_star": float(metrics["c_Ox2_star"]),
        "reactant_times_exp_f_eta": float(
            metrics["reactant_times_exp_f_eta"]
        ),
        "j_model_A_per_m2": float(metrics["j_model_A_per_m2"]),
        "charge_relation_residual_C_per_m2": float(
            metrics["charge_relation_residual_C_per_m2"]
        ),
        "j_reconstruction_residual_A_per_m2": float(
            metrics["j_reconstruction_residual_A_per_m2"]
        ),
    }


def compute_scan_row(
    base_params: Mapping[str, Any],
    spec: PZCScanSpec,
    pzc_v: float,
    sweep_index: int,
) -> dict[str, Any]:
    params = params_at_pzc(base_params, spec, pzc_v)
    model = IndependentPlanarEDLModel(params)
    with_edl = model.solve(use_edl=True)
    without_edl = model.solve(use_edl=False)
    surface_metrics = {
        material: _surface_mechanism_metrics(model, with_edl, material)
        for material in ("Au", "Pd")
    }
    tracked = _mechanism_aliases(surface_metrics[spec.side], spec.side)
    row: dict[str, Any] = {
        "scan": spec.key,
        "side": spec.side,
        "reactant": "Red_1" if spec.side == "Au" else "Ox_2",
        "sweep_index": int(sweep_index),
        "is_baseline": bool(
            math.isclose(float(pzc_v), spec.baseline, rel_tol=0.0, abs_tol=1e-14)
        ),
        "parameter_value_SI": float(pzc_v),
        "parameter_value_display": float(pzc_v),
        "parameter_display_unit": "V",
        "fixed_other_pzc_V": float(params[f"pzc_{spec.opposite_side}"]),
        "E_mix_with_EDL_V": float(with_edl["E_mix_V"]),
        "E_mix_without_EDL_V": float(without_edl["E_mix_V"]),
        "delta_E_mix_with_minus_without_mV": 1000.0
        * (float(with_edl["E_mix_V"]) - float(without_edl["E_mix_V"])),
        "i_mix_abs_with_EDL_A": float(with_edl["i_mix_abs_A"]),
        "i_mix_abs_without_EDL_A": float(without_edl["i_mix_abs_A"]),
        "i_mix_avg_with_EDL_A_per_m2": float(
            with_edl["i_mix_avg_A_per_m2"]
        ),
        "i_mix_avg_without_EDL_A_per_m2": float(
            without_edl["i_mix_avg_A_per_m2"]
        ),
        "i_mix_avg_ratio_with_over_without": float(
            with_edl["i_mix_avg_A_per_m2"]
            / without_edl["i_mix_avg_A_per_m2"]
        ),
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
        **tracked,
    }
    for material in ("Au", "Pd"):
        aliases = _mechanism_aliases(surface_metrics[material], material)
        row.update({f"{material}_{key}": value for key, value in aliases.items()})
    return row


def compute_scan_rows(base_params: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SCAN_SPECS:
        for index, value in enumerate(pzc_scan_values(spec)):
            rows.append(
                compute_scan_row(base_params, spec, float(value), sweep_index=index)
            )
    return rows


def rows_for_scan(
    rows: Iterable[Mapping[str, Any]], scan_key: str
) -> list[dict[str, Any]]:
    subset = [dict(row) for row in rows if str(row["scan"]) == scan_key]
    subset.sort(
        key=lambda row: float(
            row.get("parameter_value_SI", row.get("parameter_value_V"))
        )
    )
    if len(subset) != N_SWEEP:
        raise ValueError(f"Expected {N_SWEEP} rows for {scan_key}, got {len(subset)}")
    return subset


def representative_case_name(spec: PZCScanSpec, value: float) -> str | None:
    for name, expected in (
        ("low", spec.low),
        ("base", spec.baseline),
        ("high", spec.high),
    ):
        if math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-13):
            return name
    return None


def representative_scan_rows(
    scan_rows: Iterable[Mapping[str, Any]], spec: PZCScanSpec
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in scan_rows:
        case = representative_case_name(spec, float(row["parameter_value_SI"]))
        if case is not None:
            copy_row = dict(row)
            copy_row["case"] = case
            selected.append(copy_row)
    selected.sort(key=lambda row: ("low", "base", "high").index(str(row["case"])))
    if len(selected) != 3:
        raise ValueError(f"Missing representative cases for {spec.key}")
    return selected


def _case_models(
    base_params: Mapping[str, Any], spec: PZCScanSpec
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for case, value in (
        ("low", spec.low),
        ("base", spec.baseline),
        ("high", spec.high),
    ):
        model = IndependentPlanarEDLModel(params_at_pzc(base_params, spec, value))
        solution = model.solve(use_edl=True)
        cases.append(
            {
                "case": case,
                "parameter_value_V": value,
                "model": model,
                "solution": solution,
            }
        )
    return cases


def compute_polarization_rows(
    base_params: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    cases_by_scan: dict[str, list[dict[str, Any]]] = {}
    for spec in SCAN_SPECS:
        cases = _case_models(base_params, spec)
        emix_values = [float(case["solution"]["E_mix_V"]) for case in cases]
        e_values = np.linspace(
            min(emix_values) - 0.09,
            max(emix_values) + 0.09,
            N_POLARIZATION_POINTS,
        )
        for case in cases:
            model = case["model"]
            case["E_values_V"] = e_values
            case["I_Au_display"] = []
            case["I_Pd_display"] = []
            for E_v in e_values:
                current = model.current_components(float(E_v), use_edl=True)
                i_au_display = CURRENT_DISPLAY_SCALE * float(current["I_Au_A"])
                i_pd_display = CURRENT_DISPLAY_SCALE * float(current["I_Pd_A"])
                case["I_Au_display"].append(i_au_display)
                case["I_Pd_display"].append(i_pd_display)
                rows.append(
                    {
                        "scan": spec.key,
                        "case": case["case"],
                        "condition": "with EDL",
                        "parameter_value_V": case["parameter_value_V"],
                        "E_V": float(E_v),
                        "I_Au_A": float(current["I_Au_A"]),
                        "I_Pd_A": float(current["I_Pd_A"]),
                        "I_Au_1e_minus_3_uA": i_au_display,
                        "I_Pd_1e_minus_3_uA": i_pd_display,
                        "j_Au_A_per_m2": float(current["j_Au_A_per_m2"]),
                        "j_Pd_A_per_m2": float(current["j_Pd_A_per_m2"]),
                    }
                )
            case["I_Au_display"] = np.asarray(case["I_Au_display"], dtype=float)
            case["I_Pd_display"] = np.asarray(case["I_Pd_display"], dtype=float)

        reference_model = IndependentPlanarEDLModel(
            params_at_pzc(base_params, spec, spec.baseline)
        )
        reference_case: dict[str, Any] = {
            "case": "without_edl",
            "parameter_value_V": "",
            "model": reference_model,
            "solution": reference_model.solve(use_edl=False),
            "E_values_V": e_values,
            "I_Au_display": [],
            "I_Pd_display": [],
        }
        for E_v in e_values:
            current = reference_model.current_components(float(E_v), use_edl=False)
            i_au_display = CURRENT_DISPLAY_SCALE * float(current["I_Au_A"])
            i_pd_display = CURRENT_DISPLAY_SCALE * float(current["I_Pd_A"])
            reference_case["I_Au_display"].append(i_au_display)
            reference_case["I_Pd_display"].append(i_pd_display)
            rows.append(
                {
                    "scan": spec.key,
                    "case": "without_edl",
                    "condition": "w/o EDL",
                    "parameter_value_V": "",
                    "E_V": float(E_v),
                    "I_Au_A": float(current["I_Au_A"]),
                    "I_Pd_A": float(current["I_Pd_A"]),
                    "I_Au_1e_minus_3_uA": i_au_display,
                    "I_Pd_1e_minus_3_uA": i_pd_display,
                    "j_Au_A_per_m2": float(current["j_Au_A_per_m2"]),
                    "j_Pd_A_per_m2": float(current["j_Pd_A_per_m2"]),
                }
            )
        reference_case["I_Au_display"] = np.asarray(
            reference_case["I_Au_display"], dtype=float
        )
        reference_case["I_Pd_display"] = np.asarray(
            reference_case["I_Pd_display"], dtype=float
        )
        cases_by_scan[spec.key] = [*cases, reference_case]
    return rows, cases_by_scan


def compute_profile_rows(
    base_params: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    cases_by_scan: dict[str, list[dict[str, Any]]] = {}
    for spec in SCAN_SPECS:
        cases = _case_models(base_params, spec)
        for case in cases:
            model = case["model"]
            solution = case["solution"]
            lambda_nm = 1.0e9 * float(model.derived["lambda_D"])
            distance_nm = np.linspace(
                0.0,
                PROFILE_END_LAMBDA * lambda_nm,
                N_PROFILE_POINTS,
            )
            phi_tilde = model.phi_tilde_profile(
                float(solution["E_mix_V"]),
                spec.side,
                distance_nm * 1.0e-9,
            )
            thermal_v = float(model.derived["thermal_voltage_V"])
            phi_mv = 1000.0 * thermal_v * phi_tilde
            surface_phi_mv = float(phi_mv[0])
            surface_slope_mv_per_nm = -surface_phi_mv / lambda_nm
            compact_start_mv = surface_phi_mv + surface_slope_mv_per_nm * (
                COMPACT_X0_NM - RP_X_NM
            )
            compact_start_tilde = compact_start_mv / (1000.0 * thermal_v)
            case.update(
                {
                    "distance_nm": distance_nm,
                    "phi_tilde": phi_tilde,
                    "phi_mV": phi_mv,
                    "compact_start_phi_tilde": compact_start_tilde,
                    "compact_start_phi_mV": compact_start_mv,
                    "metal_minus_pzc_mV": 1000.0
                    * (
                        float(solution["E_mix_V"])
                        - float(model.params[spec.key])
                    ),
                    "lambda_D_nm": lambda_nm,
                }
            )
            common = {
                "scan": spec.key,
                "side": spec.side,
                "case": case["case"],
                "condition": "with EDL",
                "parameter_value_V": case["parameter_value_V"],
                "E_mix_V": float(solution["E_mix_V"]),
                "lambda_D_nm": lambda_nm,
            }
            for distance, phi_star, phi_value_mv in zip(
                distance_nm, phi_tilde, phi_mv, strict=True
            ):
                rows.append(
                    {
                        **common,
                        "region": "diffuse",
                        "distance_from_RP_nm": float(distance),
                        "phi_bar_tilde": float(phi_star),
                        "phi_bar_mV": float(phi_value_mv),
                    }
                )
            for distance, phi_star, phi_value_mv in (
                (
                    COMPACT_X0_NM,
                    compact_start_tilde,
                    compact_start_mv,
                ),
                (RP_X_NM, float(phi_tilde[0]), surface_phi_mv),
            ):
                rows.append(
                    {
                        **common,
                        "region": "compact_linear_OHP_tangent",
                        "distance_from_RP_nm": distance,
                        "phi_bar_tilde": phi_star,
                        "phi_bar_mV": phi_value_mv,
                    }
                )
        reference_model = IndependentPlanarEDLModel(
            params_at_pzc(base_params, spec, spec.baseline)
        )
        reference_solution = reference_model.solve(use_edl=False)
        x_end_nm = PROFILE_END_LAMBDA * 1.0e9 * float(
            reference_model.derived["lambda_D"]
        )
        for distance in (COMPACT_X0_NM, RP_X_NM, x_end_nm):
            rows.append(
                {
                    "scan": spec.key,
                    "side": spec.side,
                    "case": "without_edl",
                    "condition": "w/o EDL",
                    "parameter_value_V": "",
                    "E_mix_V": float(reference_solution["E_mix_V"]),
                    "lambda_D_nm": "",
                    "region": "zero_reference",
                    "distance_from_RP_nm": distance,
                    "phi_bar_tilde": 0.0,
                    "phi_bar_mV": 0.0,
                }
            )
        cases_by_scan[spec.key] = cases
    return rows, cases_by_scan


def _load_reference_pzc_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Au|C|Pd reference sweep CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if str(row.get("scan")) in SPEC_BY_KEY
        ]
    if len(rows) != len(SCAN_SPECS) * N_SWEEP:
        raise ValueError(
            f"Expected {len(SCAN_SPECS) * N_SWEEP} PZC reference rows, got {len(rows)}"
        )
    return rows


def build_comparison_rows(
    independent_rows: list[dict[str, Any]], reference_csv: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    reference_rows = _load_reference_pzc_rows(reference_csv)
    reference_lookup = {
        (str(row["scan"]), round(float(row["parameter_value_SI"]), 12)): row
        for row in reference_rows
    }
    comparison: list[dict[str, Any]] = []
    for row in independent_rows:
        key = (str(row["scan"]), round(float(row["parameter_value_SI"]), 12))
        if key not in reference_lookup:
            raise ValueError(f"Reference scan does not contain {key}")
        ref = reference_lookup[key]
        e_ind = float(row["E_mix_with_EDL_V"])
        e_ref = float(ref["E_mix_with_V"])
        i_ind = float(row["i_mix_avg_with_EDL_A_per_m2"])
        i_ref = float(ref["i_mix_avg_with_A_per_m2"])
        phi_ind = float(row["phi_RP_mean_mV"])
        phi_ref = float(ref["phi_RP_mean_mV"])
        react_ind = float(row["reactant_star"])
        react_ref = float(ref["reactant_mean"])
        comparison.append(
            {
                "scan": row["scan"],
                "side": row["side"],
                "parameter_value_V": row["parameter_value_SI"],
                "is_baseline": row["is_baseline"],
                "E_mix_independent_V": e_ind,
                "E_mix_Au_C_Pd_V": e_ref,
                "delta_E_mix_independent_minus_Au_C_Pd_mV": 1000.0
                * (e_ind - e_ref),
                "i_mix_avg_independent_A_per_m2": i_ind,
                "i_mix_avg_Au_C_Pd_A_per_m2": i_ref,
                "delta_i_mix_avg_independent_minus_Au_C_Pd_A_per_m2": i_ind
                - i_ref,
                "relative_delta_i_mix_avg_percent": 100.0 * (i_ind - i_ref) / i_ref,
                "ratio_i_mix_avg_independent_over_Au_C_Pd": i_ind / i_ref,
                "phi_RP_independent_mV": phi_ind,
                "phi_RP_Au_C_Pd_mean_mV": phi_ref,
                "delta_phi_RP_independent_minus_Au_C_Pd_mV": phi_ind
                - phi_ref,
                "reactant_independent": react_ind,
                "reactant_Au_C_Pd_mean": react_ref,
                "ratio_reactant_independent_over_Au_C_Pd_mean": react_ind
                / react_ref,
                "eta_RP_independent_mV": row["eta_RP_mV"],
                "eta_RP_Au_C_Pd_mean_mV": float(ref["eta_mean_mV"]),
                "exp_f_eta_independent": row["exp_f_eta"],
                "exp_f_eta_Au_C_Pd_mean": float(ref["bv_exp_mean"]),
                "kinetic_weight_independent": row[
                    "reactant_times_exp_f_eta"
                ],
                "kinetic_weight_Au_C_Pd_mean": float(
                    ref["bv_conc_exp_mean"]
                ),
            }
        )

    summary: dict[str, Any] = {}
    for spec in SCAN_SPECS:
        subset = rows_for_scan(comparison, spec.key)
        e_ind = np.asarray([row["E_mix_independent_V"] for row in subset])
        e_ref = np.asarray([row["E_mix_Au_C_Pd_V"] for row in subset])
        i_ind = np.asarray(
            [row["i_mix_avg_independent_A_per_m2"] for row in subset]
        )
        i_ref = np.asarray(
            [row["i_mix_avg_Au_C_Pd_A_per_m2"] for row in subset]
        )
        e_delta_mv = 1000.0 * (e_ind - e_ref)
        i_relative_percent = 100.0 * (i_ind - i_ref) / i_ref
        phi_delta_mv = np.asarray(
            [
                row["delta_phi_RP_independent_minus_Au_C_Pd_mV"]
                for row in subset
            ]
        )
        e_corr = float(np.corrcoef(e_ind, e_ref)[0, 1])
        log_i_corr = float(np.corrcoef(np.log10(i_ind), np.log10(i_ref))[0, 1])
        macro_agreement = bool(
            np.max(np.abs(e_delta_mv)) < 10.0
            and np.max(np.abs(i_relative_percent)) < 20.0
            and e_corr >= 0.995
            and log_i_corr >= 0.995
        )
        summary[spec.key] = {
            "max_abs_delta_E_mix_mV": float(np.max(np.abs(e_delta_mv))),
            "mean_abs_delta_E_mix_mV": float(np.mean(np.abs(e_delta_mv))),
            "max_abs_relative_delta_i_mix_percent": float(
                np.max(np.abs(i_relative_percent))
            ),
            "mean_abs_relative_delta_i_mix_percent": float(
                np.mean(np.abs(i_relative_percent))
            ),
            "max_abs_delta_phi_RP_mV": float(np.max(np.abs(phi_delta_mv))),
            "correlation_E_mix": e_corr,
            "correlation_log10_i_mix": log_i_corr,
            "macro_trend_and_scale_agreement": macro_agreement,
            "agreement_reporting_thresholds": {
                "max_abs_delta_E_mix_mV": 10.0,
                "max_abs_relative_delta_i_mix_percent": 20.0,
                "minimum_correlation": 0.995,
            },
        }
    summary["overall_macro_trend_and_scale_agreement"] = all(
        bool(summary[spec.key]["macro_trend_and_scale_agreement"])
        for spec in SCAN_SPECS
    )
    summary["interpretation"] = (
        "E_mix and average mixed-current trends and scales agree closely, but the "
        "models are not pointwise identical.  The independent model removes the "
        "weak lateral/support coupling present in Au|C|Pd and makes each face uniform."
    )
    return comparison, summary, reference_rows


def _side_color(material: str) -> str:
    return PZC_COLORS["au_curve"] if material == "Au" else PZC_COLORS["pd_curve"]


def _side_fill(material: str) -> str:
    return PZC_COLORS["au"] if material == "Au" else PZC_COLORS["pd"]


def _case_by_name(cases: Iterable[Mapping[str, Any]], name: str) -> dict[str, Any]:
    for case in cases:
        if str(case["case"]) == name:
            return dict(case)
    raise KeyError(name)


def _add_gradient_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    cmap: LinearSegmentedColormap,
    linewidth: float = 2.05,
) -> None:
    points = np.column_stack((x, y)).reshape(-1, 1, 2)
    segments = np.concatenate((points[:-1], points[1:]), axis=1)
    norm = Normalize(vmin=float(np.min(x)), vmax=float(np.max(x)))
    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=linewidth,
        zorder=3,
    )
    collection.set_array(0.5 * (x[:-1] + x[1:]))
    ax.add_collection(collection)


def _metric_values(
    rows: list[dict[str, Any]], metric: MetricSpec
) -> np.ndarray:
    return np.asarray([float(row[metric.key]) for row in rows], dtype=float)


def _metric_limits(
    rows: list[dict[str, Any]], metric: MetricSpec
) -> tuple[float, float]:
    values = _metric_values(rows, metric)
    if metric.without_edl_key is not None:
        values = np.append(values, float(rows[0][metric.without_edl_key]))
    if metric.scale == "log":
        low = float(np.min(values[values > 0.0]))
        high = float(np.max(values))
        if math.isclose(low, high):
            return low / 1.5, high * 1.5
        factor = 10.0 ** (0.10 * (math.log10(high) - math.log10(low)))
        return low / factor, high * factor
    low = float(np.min(values))
    high = float(np.max(values))
    span = high - low
    if span <= 0.0:
        span = max(abs(low), 1.0)
    return low - 0.12 * span, high + 0.12 * span


def _annotate_low_base_high(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    metric: MetricSpec,
    color: str,
) -> None:
    indices = (0, len(x) // 2, len(x) - 1)
    labels = ("L", "B", "H")
    for index, label in zip(indices, labels, strict=True):
        ax.scatter(
            [x[index]],
            [y[index]],
            s=23,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=6,
        )
        if index == 0:
            offset = (4, -7 if y[0] > y[len(y) // 2] else 5)
            ha = "left"
        elif index == len(x) - 1:
            offset = (-4, -7 if y[-1] > y[len(y) // 2] else 5)
            ha = "right"
        else:
            offset = (4, 5)
            ha = "left"
        ax.annotate(
            f"{label}: {format(float(y[index]), metric.value_format)}",
            xy=(x[index], y[index]),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va="bottom" if offset[1] > 0 else "top",
            fontsize=6.8,
            color=PZC_COLORS["dark"],
            clip_on=False,
            zorder=8,
        )


def plot_mechanism_figure(
    scan_rows: list[dict[str, Any]],
    spec: PZCScanSpec,
    output_dir: Path,
) -> list[Path]:
    x = np.asarray([float(row["parameter_value_SI"]) for row in scan_rows])
    cmap = AU_CMAP if spec.side == "Au" else PD_CMAP
    marker_color = _side_color(spec.side)
    fig, axes = plt.subplots(
        len(MECHANISM_METRICS),
        1,
        figsize=(3.72, 11.55),
        sharex=True,
    )
    for index, (ax, metric) in enumerate(zip(axes, MECHANISM_METRICS, strict=True)):
        y = _metric_values(scan_rows, metric)
        ax.set_yscale(metric.scale)
        ax.set_xlim(spec.low - 0.027, spec.high + 0.027)
        ax.set_ylim(*_metric_limits(scan_rows, metric))
        ax.axvline(
            spec.baseline,
            color=PZC_COLORS["dark"],
            linewidth=0.75,
            linestyle=(0, (2.2, 2.2)),
            alpha=0.88,
            zorder=1,
        )
        if metric.without_edl_key is not None:
            ax.axhline(
                float(scan_rows[0][metric.without_edl_key]),
                color=PZC_COLORS["without_edl"],
                linewidth=0.85,
                linestyle=(0, (3.4, 2.2)),
                alpha=0.88,
                zorder=1,
            )
        _add_gradient_line(ax, x, y, cmap)
        _annotate_low_base_high(ax, x, y, metric, marker_color)
        ax.set_title(metric.label, loc="left", fontsize=9.2, pad=3.2)
        formula = metric.formula_au if spec.side == "Au" else metric.formula_pd
        ax.text(
            0.02,
            0.05,
            formula,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=5.7,
            color=PZC_COLORS["gray"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.0),
            zorder=7,
        )
        ax.tick_params(length=2.8, width=0.75, labelsize=7.4, pad=1.7)
        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=3) if metric.scale == "linear" else ax.yaxis.get_major_locator()
        )
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(0.72)
        if index < len(MECHANISM_METRICS) - 1:
            ax.tick_params(labelbottom=False)
    axes[-1].xaxis.set_major_locator(
        FixedLocator([spec.low, spec.baseline, spec.high])
    )
    axes[-1].set_xlabel(f"{spec.title} (V)", fontsize=8.8, labelpad=2.4)
    fig.suptitle(
        rf"Independent EDL mechanism: {spec.title} causal chain",
        x=0.09,
        y=0.988,
        ha="left",
        fontsize=11.0,
    )
    fig.text(
        0.09,
        0.954,
        (
            f"Only {spec.title} is scanned; the {spec.side} face is tracked. "
            "The independent planar face is uniform, so each angle-bracket value equals its local face value."
        ),
        ha="left",
        va="top",
        fontsize=6.8,
        color=PZC_COLORS["gray"],
        wrap=True,
    )
    fig.text(
        0.09,
        0.014,
        "L, B, and H mark low, baseline, and high scan values.",
        ha="left",
        va="bottom",
        fontsize=6.7,
        color=PZC_COLORS["gray"],
    )
    fig.subplots_adjust(
        left=0.20,
        right=0.985,
        bottom=0.065,
        top=0.905,
        hspace=0.51,
    )
    stem = f"pzc_mechanism_{spec.side.lower()}_causal_chain_independent_edls"
    return _save_figure(fig, output_dir, stem)


def _polarization_ylim(cases: list[dict[str, Any]]) -> float:
    values = np.concatenate(
        [
            np.asarray(case[f"I_{side}_display"], dtype=float)
            for case in cases
            if str(case["case"]) != "without_edl"
            for side in ("Au", "Pd")
        ]
    )
    finite = values[np.isfinite(values)]
    # Limit the plotting range around the mixed-current operating points rather
    # than around the exponentially large curve tails.
    mixed_values = np.asarray(
        [
            CURRENT_DISPLAY_SCALE * float(case["solution"]["i_mix_abs_A"])
            for case in cases
            if str(case["case"]) != "without_edl"
        ]
    )
    limit = max(0.13, 1.70 * float(np.max(mixed_values)))
    if finite.size:
        limit = min(limit, max(0.13, 0.45 * float(np.max(np.abs(finite)))))
    return limit


def plot_polarization_figure(
    cases: list[dict[str, Any]],
    spec: PZCScanSpec,
    output_dir: Path,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.15, 4.35))
    with_cases = [case for case in cases if str(case["case"]) != "without_edl"]
    reference = _case_by_name(cases, "without_edl")
    y_limit = _polarization_ylim(cases)
    ax.axhline(0.0, color=PZC_COLORS["dark"], linewidth=0.82, zorder=1)
    ax.plot(
        reference["E_values_V"],
        reference["I_Au_display"],
        color=PZC_COLORS["without_edl"],
        linewidth=1.0,
        linestyle=(0, (2.3, 2.3)),
        alpha=0.34,
        zorder=1,
    )
    ax.plot(
        reference["E_values_V"],
        reference["I_Pd_display"],
        color=PZC_COLORS["without_edl"],
        linewidth=1.0,
        linestyle=(0, (2.3, 2.3)),
        alpha=0.34,
        zorder=1,
    )

    for material in ("Au", "Pd"):
        is_varied = material == spec.side
        for case in with_cases:
            style = CASE_STYLES[str(case["case"])]
            ax.plot(
                case["E_values_V"],
                case[f"I_{material}_display"],
                color=_side_color(material),
                linewidth=2.35 if is_varied else 1.18,
                linestyle=style["linestyle"],
                alpha=style["alpha"] if is_varied else 0.46,
                zorder=5 if is_varied else 3,
            )
    for case in with_cases:
        style = CASE_STYLES[str(case["case"])]
        e_mix = float(case["solution"]["E_mix_V"])
        i_mix = CURRENT_DISPLAY_SCALE * float(case["solution"]["i_mix_abs_A"])
        ax.axvline(
            e_mix,
            color=style["color"],
            linewidth=0.85,
            linestyle=style["linestyle"],
            alpha=style["alpha"],
            zorder=2,
        )
        ax.scatter(
            [e_mix, e_mix],
            [i_mix, -i_mix],
            s=31 if case["case"] == "base" else 23,
            color=[PZC_COLORS["au_curve"], PZC_COLORS["pd_curve"]],
            edgecolor="white",
            linewidth=0.55,
            alpha=min(1.0, float(style["alpha"]) + 0.12),
            zorder=8,
        )
        ax.scatter(
            [e_mix],
            [0.0],
            s=28 if case["case"] == "base" else 21,
            color=style["color"],
            edgecolor="white",
            linewidth=0.5,
            zorder=9,
        )
    low = _case_by_name(with_cases, "low")
    high = _case_by_name(with_cases, "high")
    e_low = float(low["solution"]["E_mix_V"])
    e_high = float(high["solution"]["E_mix_V"])
    y_arrow = 0.80 * y_limit
    ax.annotate(
        "",
        xy=(e_high, y_arrow),
        xytext=(e_low, y_arrow),
        arrowprops=dict(
            arrowstyle="->",
            color=PZC_COLORS["dark"],
            linewidth=0.8,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=10,
    )
    ax.text(
        0.5 * (e_low + e_high),
        y_arrow + 0.04 * y_limit,
        r"$E_{\mathrm{mix}}$ shift",
        ha="center",
        va="bottom",
        fontsize=8.0,
    )
    side_handles = [
        Line2D(
            [0],
            [0],
            color=_side_color(spec.side),
            lw=2.35,
            label=f"{spec.side} varied-side current",
        ),
        Line2D(
            [0],
            [0],
            color=_side_color(spec.opposite_side),
            lw=1.2,
            alpha=0.48,
            label=f"{spec.opposite_side} fixed-PZC current",
        ),
        Line2D(
            [0],
            [0],
            color=PZC_COLORS["without_edl"],
            lw=1.0,
            ls=(0, (2.3, 2.3)),
            alpha=0.5,
            label="w/o EDL reference",
        ),
    ]
    side_legend = ax.legend(
        handles=side_handles,
        loc="upper left",
        fontsize=7.2,
        handlelength=2.4,
        labelspacing=0.35,
    )
    ax.add_artist(side_legend)
    case_handles = [
        Line2D(
            [0],
            [0],
            color=CASE_STYLES[name]["color"],
            lw=1.6,
            ls=CASE_STYLES[name]["linestyle"],
            label=name,
        )
        for name in ("low", "base", "high")
    ]
    ax.legend(
        handles=case_handles,
        title=spec.title,
        loc="upper right",
        fontsize=7.0,
        title_fontsize=7.4,
        handlelength=2.2,
        labelspacing=0.30,
    )
    ax.set_xlim(
        float(np.min(with_cases[0]["E_values_V"])),
        float(np.max(with_cases[0]["E_values_V"])),
    )
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xlabel("Potential (V vs. RHE)")
    ax.set_ylabel(r"Half-reaction current (10$^{-3}$ $\mu$A)")
    ax.set_title(
        rf"Independent EDL polarization: {spec.title} scan",
        loc="left",
        fontsize=10.2,
        pad=5.0,
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(length=3.0, width=0.78, labelsize=8.0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.78)
    stem = f"pzc_polarization_{spec.side.lower()}_independent_edls"
    return _save_figure(fig, output_dir, stem)


def _scheme_y_limits(cases_by_scan: Mapping[str, list[dict[str, Any]]]) -> tuple[float, float]:
    values: list[float] = [0.0]
    for cases in cases_by_scan.values():
        for case in cases:
            values.extend(np.asarray(case["phi_mV"], dtype=float).tolist())
            values.append(float(case["compact_start_phi_mV"]))
    low = min(values)
    high = max(values)
    span = high - low
    if span <= 0.0:
        span = 1.0
    return low - 0.08 * span, high + 0.12 * span


def _metal_x_limits(cases_by_scan: Mapping[str, list[dict[str, Any]]]) -> tuple[float, float]:
    values = [
        float(case["metal_minus_pzc_mV"])
        for cases in cases_by_scan.values()
        for case in cases
    ]
    low, high = min(values), max(values)
    span = high - low
    return low - 0.08 * span, high + 0.08 * span


def _draw_metal_panel(
    ax: plt.Axes,
    spec: PZCScanSpec,
    cases: list[dict[str, Any]],
    x_limits: tuple[float, float],
) -> None:
    ax.set_xlim(*x_limits)
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.50, color=PZC_COLORS["dark"], linewidth=0.85, zorder=1)
    ax.add_patch(
        Rectangle(
            (x_limits[0], 0.32),
            x_limits[1] - x_limits[0],
            0.36,
            facecolor=_side_fill(spec.side),
            edgecolor="none",
            alpha=0.18,
            zorder=0,
        )
    )
    for case in cases:
        style = CASE_STYLES[str(case["case"])]
        phi_m = float(case["metal_minus_pzc_mV"])
        ax.plot(
            [phi_m, phi_m],
            [0.28, 0.72],
            color=style["color"],
            linewidth=1.5,
            linestyle=style["linestyle"],
            zorder=3,
        )
        ax.scatter(
            [phi_m],
            [0.50],
            s=25,
            color=style["color"],
            edgecolor="white",
            linewidth=0.55,
            zorder=4,
        )
    low = _case_by_name(cases, "low")
    high = _case_by_name(cases, "high")
    ax.annotate(
        "",
        xy=(float(high["metal_minus_pzc_mV"]), 0.82),
        xytext=(float(low["metal_minus_pzc_mV"]), 0.82),
        arrowprops=dict(arrowstyle="->", linewidth=0.8, color=PZC_COLORS["dark"]),
    )
    ax.text(
        0.02,
        0.96,
        r"$\phi_M=E_{\mathrm{mix}}-\mathrm{PZC}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
    )
    ax.set_yticks([])
    ax.set_xlabel(r"Converted metal potential, $\phi_M$ (mV)", fontsize=6.7, labelpad=1.5)
    ax.tick_params(axis="x", length=2.6, width=0.7, labelsize=6.6, pad=1.6)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)


def _draw_profile_panel(
    ax: plt.Axes,
    spec: PZCScanSpec,
    cases: list[dict[str, Any]],
    y_limits: tuple[float, float],
    show_ylabel: bool,
) -> None:
    x_end = max(float(np.max(case["distance_nm"])) for case in cases)
    ax.axvspan(METAL_X0_NM, COMPACT_X0_NM, color=_side_fill(spec.side), alpha=0.92, zorder=0)
    ax.axvspan(COMPACT_X0_NM, RP_X_NM, color=PZC_COLORS["compact"], alpha=1.0, zorder=0)
    ax.axvspan(RP_X_NM, x_end, color=PZC_COLORS["electrolyte"], alpha=0.62, zorder=0)
    ax.axvline(RP_X_NM, color=PZC_COLORS["dark"], linewidth=0.9, zorder=2)
    for case in cases:
        style = CASE_STYLES[str(case["case"])]
        surface_mv = float(np.asarray(case["phi_mV"])[0])
        ax.plot(
            [COMPACT_X0_NM, RP_X_NM],
            [float(case["compact_start_phi_mV"]), surface_mv],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            zorder=5,
        )
        ax.plot(
            case["distance_nm"],
            case["phi_mV"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            zorder=5,
        )
        ax.scatter(
            [0.0],
            [surface_mv],
            s=22,
            color=style["color"],
            edgecolor="white",
            linewidth=0.5,
            zorder=6,
        )
    ax.plot(
        [COMPACT_X0_NM, x_end],
        [0.0, 0.0],
        color=PZC_COLORS["without_edl"],
        linewidth=1.55,
        linestyle=(0, (3.5, 2.2)),
        label="w/o EDL",
        zorder=6,
    )
    case_lines = []
    for case in cases:
        case_lines.append(
            f"{case['case']:<4} {float(case['parameter_value_V']):.2f}  "
            f"{float(np.asarray(case['phi_mV'])[0]):+6.0f}  "
            f"{float(case['solution']['E_mix_V']):.3f}"
        )
    ax.text(
        0.97,
        0.96,
        (
            f"{spec.title} (V)\n"
            r"case  value  $\phi_{\rm RP}$ (mV)  $E_{\rm mix}$"
            "\n"
            + "\n".join(case_lines)
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        linespacing=1.23,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2.2),
        zorder=8,
    )
    high = _case_by_name(cases, "high")
    ax.annotate(
        r"higher PZC: more negative $\phi_{\rm RP}$",
        xy=(0.20, float(np.asarray(high["phi_mV"])[0])),
        xytext=(0.37 * x_end, y_limits[0] + 0.18 * (y_limits[1] - y_limits[0])),
        arrowprops=dict(arrowstyle="->", linewidth=0.75, color=PZC_COLORS["dark"]),
        fontsize=6.2,
        ha="left",
        va="center",
        zorder=7,
    )
    ax.text(
        0.09,
        y_limits[1] - 0.07 * (y_limits[1] - y_limits[0]),
        "OHP/RP",
        ha="left",
        va="top",
        fontsize=6.4,
    )
    ax.text(
        0.50 * (METAL_X0_NM + COMPACT_X0_NM),
        y_limits[0] + 0.12 * (y_limits[1] - y_limits[0]),
        spec.side,
        rotation=90,
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )
    ax.set_xlim(METAL_X0_NM, x_end)
    ax.set_ylim(*y_limits)
    ax.set_title(spec.title, loc="left", fontsize=9.3, pad=4.8)
    ax.set_xlabel("Distance from OHP/RP, x (nm)")
    ax.set_ylabel(r"$\bar{\phi}(x)$ (mV)" if show_ylabel else "")
    ax.tick_params(length=2.8, width=0.75, labelsize=7.4, pad=1.8)
    if not show_ylabel:
        ax.tick_params(labelleft=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.75)


def plot_edl_scheme(
    cases_by_scan: Mapping[str, list[dict[str, Any]]],
    output_dir: Path,
) -> list[Path]:
    y_limits = _scheme_y_limits(cases_by_scan)
    metal_limits = _metal_x_limits(cases_by_scan)
    fig = plt.figure(figsize=(7.35, 4.48))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(0.68, 3.25),
        hspace=0.48,
        wspace=0.18,
    )
    metal_axes = [fig.add_subplot(grid[0, column]) for column in range(2)]
    profile_axes: list[plt.Axes] = []
    for column in range(2):
        shared = profile_axes[0] if profile_axes else None
        profile_axes.append(fig.add_subplot(grid[1, column], sharey=shared))
    for column, spec in enumerate(SCAN_SPECS):
        cases = cases_by_scan[spec.key]
        _draw_metal_panel(metal_axes[column], spec, cases, metal_limits)
        _draw_profile_panel(
            profile_axes[column],
            spec,
            cases,
            y_limits,
            show_ylabel=column == 0,
        )
    handles = [
        Line2D(
            [0],
            [0],
            color=CASE_STYLES[name]["color"],
            linestyle=CASE_STYLES[name]["linestyle"],
            linewidth=CASE_STYLES[name]["linewidth"],
            label=name,
        )
        for name in ("low", "base", "high")
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color=PZC_COLORS["without_edl"],
            linestyle=(0, (3.5, 2.2)),
            linewidth=1.55,
            label="w/o EDL",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.50, 0.012),
        fontsize=7.1,
        handlelength=2.5,
    )
    fig.suptitle(
        r"Independent EDL scheme: PZC controls $\phi_M$ and $\phi_{\mathrm{RP}}$",
        x=0.06,
        y=0.985,
        ha="left",
        fontsize=11.0,
    )
    fig.text(
        0.06,
        0.928,
        "Low, base, and high cases use the same mixed-potential balance. Compact-layer segments are linear OHP tangents; diffuse profiles are analytic exponentials; both panels use one potential scale.",
        ha="left",
        va="top",
        fontsize=7.0,
        color=PZC_COLORS["gray"],
    )
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.16, top=0.82)
    return _save_figure(
        fig,
        output_dir,
        "pzc_edl_scheme_phi_rp_emix_independent_edls",
    )


def plot_model_comparison(
    comparison_rows: list[dict[str, Any]],
    comparison_summary: Mapping[str, Any],
    output_dir: Path,
) -> list[Path]:
    fig, axes = plt.subplots(3, 2, figsize=(7.25, 7.05), sharex="col")
    for column, spec in enumerate(SCAN_SPECS):
        subset = rows_for_scan(comparison_rows, spec.key)
        x = np.asarray([float(row["parameter_value_V"]) for row in subset])
        e_ind = np.asarray([float(row["E_mix_independent_V"]) for row in subset])
        e_ref = np.asarray([float(row["E_mix_Au_C_Pd_V"]) for row in subset])
        i_ind = np.asarray(
            [float(row["i_mix_avg_independent_A_per_m2"]) for row in subset]
        )
        i_ref = np.asarray(
            [float(row["i_mix_avg_Au_C_Pd_A_per_m2"]) for row in subset]
        )
        phi_ind = np.asarray(
            [float(row["phi_RP_independent_mV"]) for row in subset]
        )
        phi_ref = np.asarray(
            [float(row["phi_RP_Au_C_Pd_mean_mV"]) for row in subset]
        )
        for row_index, (independent, reference, ylabel) in enumerate(
            (
                (e_ind, e_ref, r"$E_{\mathrm{mix}}$ (V vs. RHE)"),
                (i_ind, i_ref, r"$\bar{i}_{\mathrm{mix}}$ (A/m$^2$)"),
                (phi_ind, phi_ref, r"$\phi_{\mathrm{RP}}$ (mV)"),
            )
        ):
            ax = axes[row_index, column]
            ax.plot(
                x,
                independent,
                color=PZC_COLORS["with_edl"],
                linewidth=2.1,
                label="independent EDLs",
                zorder=3,
            )
            ax.plot(
                x,
                reference,
                color=PZC_COLORS["without_edl"],
                linewidth=1.75,
                linestyle=(0, (4.2, 2.3)),
                label="Au | support | Pd",
                zorder=2,
            )
            ax.axvline(
                spec.baseline,
                color=PZC_COLORS["gray"],
                linewidth=0.7,
                linestyle=(0, (2.0, 2.0)),
                zorder=1,
            )
            if row_index == 1:
                ax.set_yscale("log")
            ax.set_ylabel(ylabel if column == 0 else "", fontsize=8.0)
            ax.tick_params(length=2.8, width=0.75, labelsize=7.5, pad=1.8)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_linewidth(0.72)
            if row_index == 0:
                stats = comparison_summary[spec.key]
                ax.set_title(spec.title, fontsize=9.4, pad=5.0)
                ax.text(
                    0.03,
                    0.06,
                    (
                        rf"max $|\Delta E|$ = {float(stats['max_abs_delta_E_mix_mV']):.2f} mV"
                        "\n"
                        rf"max $|\Delta \bar{{i}}|/\bar{{i}}$ = {float(stats['max_abs_relative_delta_i_mix_percent']):.1f}%"
                    ),
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=6.5,
                    color=PZC_COLORS["gray"],
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=1.2),
                )
        axes[-1, column].set_xlabel(f"{spec.title} (V)", fontsize=8.2)
        axes[-1, column].xaxis.set_major_locator(
            FixedLocator([spec.low, spec.baseline, spec.high])
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.51, 0.928),
        ncol=2,
        fontsize=7.6,
        handlelength=2.7,
    )
    fig.suptitle(
        r"PZC scan comparison: independent EDLs vs. Au $|$ support $|$ Pd",
        x=0.08,
        y=0.986,
        ha="left",
        fontsize=11.0,
    )
    fig.text(
        0.08,
        0.947,
        "Both use the same reactive lengths, kinetic parameters, interfacial capacitances, and total reactive-area normalization. The support and lateral field coupling exist only in the Au | support | Pd model.",
        ha="left",
        va="top",
        fontsize=6.9,
        color=PZC_COLORS["gray"],
    )
    fig.subplots_adjust(
        left=0.12,
        right=0.985,
        bottom=0.075,
        top=0.885,
        hspace=0.30,
        wspace=0.20,
    )
    return _save_figure(
        fig,
        output_dir,
        "pzc_independent_vs_au_support_pd_comparison",
    )


def _validation(
    scan_rows: list[dict[str, Any]],
    polarization_cases: Mapping[str, list[dict[str, Any]]],
    profile_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    comparison_summary: Mapping[str, Any],
) -> dict[str, Any]:
    grid_errors: dict[str, float] = {}
    for spec in SCAN_SPECS:
        subset = rows_for_scan(scan_rows, spec.key)
        observed = np.asarray(
            [float(row["parameter_value_SI"]) for row in subset], dtype=float
        )
        expected = pzc_scan_values(spec)
        grid_errors[spec.key] = float(np.max(np.abs(observed - expected)))

    baseline_rows = [row for row in scan_rows if bool(row["is_baseline"])]
    baseline_e_spread = float(
        np.ptp([float(row["E_mix_with_EDL_V"]) for row in baseline_rows])
    )
    baseline_i_spread = float(
        np.ptp(
            [float(row["i_mix_avg_with_EDL_A_per_m2"]) for row in baseline_rows]
        )
    )
    no_edl_e_spread = float(
        np.ptp([float(row["E_mix_without_EDL_V"]) for row in scan_rows])
    )
    no_edl_i_spread = float(
        np.ptp(
            [
                float(row["i_mix_avg_without_EDL_A_per_m2"])
                for row in scan_rows
            ]
        )
    )
    max_charge_residual = max(
        abs(float(row[f"{material}_charge_relation_residual_C_per_m2"]))
        for row in scan_rows
        for material in ("Au", "Pd")
    )
    max_kinetic_residual = max(
        abs(float(row[f"{material}_j_reconstruction_residual_A_per_m2"]))
        for row in scan_rows
        for material in ("Au", "Pd")
    )
    max_balance_residual = max(
        max(
            float(row["relative_current_balance_residual_with_EDL"]),
            float(row["relative_current_balance_residual_without_EDL"]),
        )
        for row in scan_rows
    )
    max_root_difference = max(
        max(
            abs(float(row["closed_form_minus_brent_with_EDL_V"])),
            abs(float(row["closed_form_minus_brent_without_EDL_V"])),
        )
        for row in scan_rows
    )
    max_without_profile = max(
        abs(float(row["phi_bar_tilde"]))
        for row in profile_rows
        if row["condition"] == "w/o EDL"
    )
    surface_profile_errors: list[float] = []
    compact_join_errors: list[float] = []
    for spec in SCAN_SPECS:
        scan_lookup = {
            round(float(row["parameter_value_SI"]), 12): row
            for row in rows_for_scan(scan_rows, spec.key)
        }
        for case_name, value in (
            ("low", spec.low),
            ("base", spec.baseline),
            ("high", spec.high),
        ):
            diffuse_surface = next(
                row
                for row in profile_rows
                if row["scan"] == spec.key
                and row["case"] == case_name
                and row["region"] == "diffuse"
                and float(row["distance_from_RP_nm"]) == 0.0
            )
            compact_surface = next(
                row
                for row in profile_rows
                if row["scan"] == spec.key
                and row["case"] == case_name
                and row["region"] == "compact_linear_OHP_tangent"
                and float(row["distance_from_RP_nm"]) == 0.0
            )
            scan_row = scan_lookup[round(value, 12)]
            surface_profile_errors.append(
                abs(
                    float(diffuse_surface["phi_bar_tilde"])
                    - float(scan_row["phi_RP_tilde"])
                )
            )
            compact_join_errors.append(
                abs(
                    float(diffuse_surface["phi_bar_tilde"])
                    - float(compact_surface["phi_bar_tilde"])
                )
            )

    opposite_curve_variations: dict[str, float] = {}
    for spec in SCAN_SPECS:
        cases = [
            case
            for case in polarization_cases[spec.key]
            if str(case["case"]) != "without_edl"
        ]
        arrays = [
            np.asarray(case[f"I_{spec.opposite_side}_display"], dtype=float)
            for case in cases
        ]
        opposite_curve_variations[spec.key] = float(
            max(np.max(np.abs(array - arrays[1])) for array in arrays)
        )

    validation: dict[str, Any] = {
        "scan_rows": len(scan_rows),
        "polarization_case_count": sum(
            len(cases) for cases in polarization_cases.values()
        ),
        "profile_rows": len(profile_rows),
        "comparison_rows": len(comparison_rows),
        "max_scan_grid_error_V": max(grid_errors.values()),
        "scan_grid_errors_V": grid_errors,
        "baseline_E_mix_spread_between_scans_V": baseline_e_spread,
        "baseline_i_mix_spread_between_scans_A_per_m2": baseline_i_spread,
        "max_no_edl_E_mix_variation_V": no_edl_e_spread,
        "max_no_edl_i_mix_avg_variation_A_per_m2": no_edl_i_spread,
        "max_abs_charge_relation_residual_C_per_m2": max_charge_residual,
        "max_abs_kinetic_reconstruction_residual_A_per_m2": max_kinetic_residual,
        "max_relative_current_balance_residual": max_balance_residual,
        "max_abs_closed_form_minus_brent_V": max_root_difference,
        "max_abs_without_edl_profile_phi_tilde": max_without_profile,
        "max_surface_profile_minus_scan_phi_tilde": max(surface_profile_errors),
        "max_compact_diffuse_join_error_phi_tilde": max(compact_join_errors),
        "max_opposite_side_fixed_E_polarization_variation_display_units": max(
            opposite_curve_variations.values()
        ),
        "opposite_side_curve_variations": opposite_curve_variations,
        "comparison_macro_agreement": bool(
            comparison_summary["overall_macro_trend_and_scale_agreement"]
        ),
    }
    validation["passed"] = bool(
        validation["scan_rows"] == len(SCAN_SPECS) * N_SWEEP
        and validation["comparison_rows"] == len(SCAN_SPECS) * N_SWEEP
        and validation["max_scan_grid_error_V"] < 1.0e-14
        and validation["baseline_E_mix_spread_between_scans_V"] < 1.0e-13
        and validation["baseline_i_mix_spread_between_scans_A_per_m2"] < 1.0e-13
        and validation["max_no_edl_E_mix_variation_V"] < 1.0e-13
        and validation["max_no_edl_i_mix_avg_variation_A_per_m2"] < 1.0e-13
        and validation["max_abs_charge_relation_residual_C_per_m2"] < 1.0e-12
        and validation["max_abs_kinetic_reconstruction_residual_A_per_m2"]
        < 1.0e-12
        and validation["max_relative_current_balance_residual"] < 1.0e-10
        and validation["max_abs_closed_form_minus_brent_V"] < 5.0e-11
        and validation["max_abs_without_edl_profile_phi_tilde"] == 0.0
        and validation["max_surface_profile_minus_scan_phi_tilde"] < 1.0e-13
        and validation["max_compact_diffuse_join_error_phi_tilde"] < 1.0e-13
        and validation[
            "max_opposite_side_fixed_E_polarization_variation_display_units"
        ]
        < 1.0e-14
        and validation["comparison_macro_agreement"]
    )
    return validation


def _representative_summary(
    scan_rows: list[dict[str, Any]], spec: PZCScanSpec
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in representative_scan_rows(rows_for_scan(scan_rows, spec.key), spec):
        result[str(row["case"])] = {
            "PZC_V": row["parameter_value_SI"],
            "E_mix_with_EDL_V": row["E_mix_with_EDL_V"],
            "i_mix_avg_with_EDL_A_per_m2": row[
                "i_mix_avg_with_EDL_A_per_m2"
            ],
            "sigma_uC_per_cm2": row["sigma_uC_per_cm2"],
            "phi_RP_tilde": row["phi_RP_tilde"],
            "phi_RP_mean_mV": row["phi_RP_mean_mV"],
            "reactant_star": row["reactant_star"],
            "eta_RP_mV": row["eta_RP_mV"],
            "exp_f_eta": row["exp_f_eta"],
            "reactant_times_exp_f_eta": row[
                "reactant_times_exp_f_eta"
            ],
        }
    return result


def _summary(
    scan_rows: list[dict[str, Any]], comparison_summary: Mapping[str, Any]
) -> dict[str, Any]:
    baseline = next(row for row in scan_rows if row["scan"] == "pzc_Au" and row["is_baseline"])
    representatives = {
        spec.key: _representative_summary(scan_rows, spec) for spec in SCAN_SPECS
    }
    trend_factors: dict[str, Any] = {}
    for spec in SCAN_SPECS:
        low = representatives[spec.key]["low"]
        high = representatives[spec.key]["high"]
        trend_factors[spec.key] = {
            "delta_E_mix_high_minus_low_mV": 1000.0
            * (
                float(high["E_mix_with_EDL_V"])
                - float(low["E_mix_with_EDL_V"])
            ),
            "i_mix_high_over_low": float(high["i_mix_avg_with_EDL_A_per_m2"])
            / float(low["i_mix_avg_with_EDL_A_per_m2"]),
            "direction": (
                "higher PZC_Au raises E_mix and lowers i_mix"
                if spec.side == "Au"
                else "higher PZC_Pd raises both E_mix and i_mix"
            ),
        }
    return {
        "baseline_independent_edls": {
            "pzc_Au_V": 0.93,
            "pzc_Pd_V": 0.78,
            "E_mix_with_EDL_V": baseline["E_mix_with_EDL_V"],
            "E_mix_without_EDL_V": baseline["E_mix_without_EDL_V"],
            "i_mix_avg_with_EDL_A_per_m2": baseline[
                "i_mix_avg_with_EDL_A_per_m2"
            ],
            "i_mix_avg_without_EDL_A_per_m2": baseline[
                "i_mix_avg_without_EDL_A_per_m2"
            ],
        },
        "representative_cases": representatives,
        "scan_trends": trend_factors,
        "comparison_vs_Au_support_Pd": comparison_summary,
        "comparison_conclusion": (
            "The two models give the same PZC trend directions and closely matching "
            "E_mix/current scales, while remaining quantitatively distinct because "
            "the independent model has uniform, non-overlapping faces and no support/lateral coupling."
        ),
        "polarization_topology_note": (
            "At fixed E, changing one PZC leaves the opposite-side polarization "
            "curve exactly unchanged in the independent model; the shared E_mix still moves."
        ),
    }


def _scan_config(reference_csv: Path) -> dict[str, Any]:
    return {
        "study": "one-factor PZC scans",
        "reference_family": "Figures/Figure_5_6 PZC mechanism/scheme/polarization",
        "reference_sweep_csv": str(reference_csv.resolve()),
        "scan_definitions": {
            spec.key: {
                "side": spec.side,
                "low_V": spec.low,
                "baseline_V": spec.baseline,
                "high_V": spec.high,
                "points": N_SWEEP,
                "values_V": pzc_scan_values(spec),
                "fixed_other_PZC_V": (
                    SPEC_BY_KEY[f"pzc_{spec.opposite_side}"].baseline
                ),
            }
            for spec in SCAN_SPECS
        },
        "scan_rule": "Only one material PZC is changed in each OFAT path.",
        "profile": {
            "diffuse_formula": "phi_bar_star(x)=phi_RP_star*exp(-x/lambda_D)",
            "distance_range": "0 to 5 lambda_D",
            "points": N_PROFILE_POINTS,
            "compact_layer": (
                "linear continuation using the analytic diffuse-profile tangent at OHP/RP"
            ),
            "without_edl": "phi_bar_star(x)=0 reference",
            "common_Au_Pd_y_scale": True,
        },
        "surface_average_convention": (
            "Each independent planar face is uniform; angle-bracket means equal the local face scalar."
        ),
        "current_balance": "I_Au + I_Pd = 0",
        "mixed_current_density": "abs(I_Au)/(area_Au+area_Pd)",
        "comparison_scope": (
            "same 41-point one-factor grids; identical reactive lengths, kinetic "
            "parameters, C_H values, and total reactive-area normalization"
        ),
    }


def _build_pzc_results_in_place(
    params: Mapping[str, Any], output: Path, reference_csv: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    base_params = canonical_params(params)
    scan_rows = compute_scan_rows(base_params)
    polarization_rows, polarization_cases = compute_polarization_rows(base_params)
    profile_rows, profile_cases = compute_profile_rows(base_params)
    comparison_rows, comparison_summary, reference_rows = build_comparison_rows(
        scan_rows, reference_csv
    )
    validation = _validation(
        scan_rows,
        polarization_cases,
        profile_rows,
        comparison_rows,
        comparison_summary,
    )
    if not validation["passed"]:
        raise RuntimeError(f"PZC study validation failed: {validation}")

    mechanism_dir = output / "figures" / "Mechanism"
    scheme_dir = output / "figures" / "EDL_scheme"
    polarization_dir = output / "figures" / "Polarization"
    comparison_dir = output / "figures" / "Comparison"
    mechanism_paths: list[Path] = []
    polarization_paths: list[Path] = []
    with plt.rc_context(RC):
        for spec in SCAN_SPECS:
            mechanism_paths.extend(
                plot_mechanism_figure(
                    rows_for_scan(scan_rows, spec.key), spec, mechanism_dir
                )
            )
            polarization_paths.extend(
                plot_polarization_figure(
                    polarization_cases[spec.key], spec, polarization_dir
                )
            )
        scheme_paths = plot_edl_scheme(profile_cases, scheme_dir)
        comparison_paths = plot_model_comparison(
            comparison_rows, comparison_summary, comparison_dir
        )

    csv_dir = output / "csv"
    _write_csv(csv_dir / "pzc_scan_independent_edls.csv", scan_rows)
    mechanism_fields = [
        "scan",
        "side",
        "reactant",
        "sweep_index",
        "is_baseline",
        "parameter_value_SI",
        "fixed_other_pzc_V",
        "E_mix_with_EDL_V",
        "E_mix_without_EDL_V",
        "i_mix_avg_with_EDL_A_per_m2",
        "i_mix_avg_without_EDL_A_per_m2",
    ]
    for material in ("Au", "Pd"):
        mechanism_fields.extend(
            [
                f"{material}_sigma_uC_per_cm2",
                f"{material}_phi_RP_tilde",
                f"{material}_phi_RP_mean_V",
                f"{material}_phi_RP_mean_mV",
                f"{material}_eta_RP_V",
                f"{material}_eta_RP_mV",
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
        csv_dir / "pzc_mechanism_metrics_Au_Pd.csv",
        scan_rows,
        mechanism_fields,
    )
    representative_rows = [
        row
        for spec in SCAN_SPECS
        for row in representative_scan_rows(rows_for_scan(scan_rows, spec.key), spec)
    ]
    _write_csv(csv_dir / "pzc_representative_cases.csv", representative_rows)
    _write_csv(csv_dir / "pzc_polarization_curves.csv", polarization_rows)
    _write_csv(csv_dir / "pzc_edl_scheme_profiles.csv", profile_rows)
    _write_csv(
        csv_dir / "pzc_comparison_vs_Au_support_Pd.csv", comparison_rows
    )
    _write_csv(
        csv_dir / "pzc_reference_Au_support_Pd_source_rows.csv", reference_rows
    )

    baseline_model = IndependentPlanarEDLModel(base_params)
    summary = _summary(scan_rows, comparison_summary)
    scan_config = _scan_config(reference_csv)
    _write_json(output / "params.json", base_params)
    _write_json(output / "derived.json", baseline_model.derived)
    _write_json(output / "scan_config.json", scan_config)
    _write_json(output / "summary.json", summary)
    _write_json(output / "comparison_summary.json", comparison_summary)
    _write_json(output / "validation.json", validation)

    pngs = sorted(output.glob("figures/**/*.png"))
    svgs = sorted(output.glob("figures/**/*.svg"))
    pdfs = sorted(output.glob("**/*.pdf"))
    if len(pngs) != 6 or len(svgs) != 6 or pdfs:
        raise RuntimeError(
            f"Expected 6 PNG, 6 SVG, 0 PDF; got {len(pngs)}, {len(svgs)}, {len(pdfs)}"
        )
    source_files = {
        "pzc_study.py": Path(__file__).resolve(),
        "ctot_study.py": Path(__file__).resolve().with_name("ctot_study.py"),
        "model.py": Path(__file__).resolve().with_name("model.py"),
    }
    manifest = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "study": "PZC_one_factor_Au_Pd",
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
        "execution": {"argv": sys.argv, "atomic_output_staging": True},
        "reference_Au_support_Pd": {
            "source_csv": str(reference_csv.resolve()),
            "sha256": _sha256_file(reference_csv),
            "comparison_rows": len(comparison_rows),
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
        "scan_points_per_path": N_SWEEP,
        "scan_paths": [spec.key for spec in SCAN_SPECS],
        "comparison_macro_agreement": bool(
            comparison_summary["overall_macro_trend_and_scale_agreement"]
        ),
    }
    _write_json(output / "run_manifest.json", manifest)
    artifacts: dict[str, Any] = {
        "Mechanism": [str(path.relative_to(output)) for path in mechanism_paths],
        "EDL_scheme": [str(path.relative_to(output)) for path in scheme_paths],
        "Polarization": [
            str(path.relative_to(output)) for path in polarization_paths
        ],
        "Comparison": [
            str(path.relative_to(output)) for path in comparison_paths
        ],
        "csv": [
            "csv/pzc_scan_independent_edls.csv",
            "csv/pzc_mechanism_metrics_Au_Pd.csv",
            "csv/pzc_representative_cases.csv",
            "csv/pzc_polarization_curves.csv",
            "csv/pzc_edl_scheme_profiles.csv",
            "csv/pzc_comparison_vs_Au_support_Pd.csv",
            "csv/pzc_reference_Au_support_Pd_source_rows.csv",
        ],
        "metadata": [
            "params.json",
            "derived.json",
            "scan_config.json",
            "summary.json",
            "comparison_summary.json",
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
            for category in (
                "Mechanism",
                "EDL_scheme",
                "Polarization",
                "Comparison",
                "csv",
                "metadata",
            )
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


def build_pzc_results(
    params: Mapping[str, Any], output: Path, reference_csv: Path = DEFAULT_REFERENCE_SWEEP_CSV
) -> dict[str, Any]:
    """Build in a temporary sibling and expose the validated result atomically."""

    output = Path(output)
    reference_csv = Path(reference_csv).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.",
        dir=output.parent,
    ) as temporary_parent:
        staged_output = Path(temporary_parent) / output.name
        result = _build_pzc_results_in_place(
            params,
            staged_output,
            reference_csv,
        )
        staged_output.replace(output)
    result["output"] = str(output)
    return result


def default_output(root: Path = PACKAGE_ROOT) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return root / "results" / f"{stamp}_pzc_study"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--params",
        type=Path,
        default=PACKAGE_ROOT / "params_template.json",
    )
    result.add_argument(
        "--reference-csv",
        type=Path,
        default=DEFAULT_REFERENCE_SWEEP_CSV,
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
    result = build_pzc_results(params, output, args.reference_csv.resolve())
    print(json.dumps(result["manifest"]["figure_counts"], sort_keys=True))
    print(result["output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
