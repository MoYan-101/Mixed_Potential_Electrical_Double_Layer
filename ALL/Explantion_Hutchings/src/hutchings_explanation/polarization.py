"""Five-condition polarization overlay for the Hutchings Summary.

The four with-EDL curves are evaluated with the same numerical backends used
to generate their source cases.  Currents are then converted to the Summary's
common Au(4 nm) + Pd(4 nm) reactive area.  The w/o-EDL limit is analytic and
identical for all four cases on that common area, so it is plotted once.
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


CURRENT_DEFINITION = "I_mix = |I_Au| = |I_Pd| at I_Au + I_Pd = 0"
CURRENT_DISPLAY_SCALE = 1.0e9  # A -> 10^-3 microampere
# Preserve the original ~0.25 mV sampling density while extending the shared
# traceable curve data far enough for the 0.45--0.75 V view.
POTENTIAL_VALUES_V = np.linspace(0.40, 0.75, 1401)
FIGURE_STEM = "summary_half_reaction_polarization_overlay"
FIGURE_STEM_XMIN045 = f"{FIGURE_STEM}_xmin045V"
X_AXIS_LABEL = "Electrode potential (V vs. RHE)"
Y_AXIS_LABEL = r"Current (10$^{-3}$ µA)"
AU_LEGEND_LABEL = "Oxidation on Au"
PD_LEGEND_LABEL = "Reduction on Pd"
FIGURE_VARIANTS = (
    {
        "variant_key": "xmin040V",
        "stem": FIGURE_STEM,
        "x_min_V": 0.40,
        "x_max_V": 0.64,
    },
    {
        "variant_key": "xmin045V",
        "stem": FIGURE_STEM_XMIN045,
        "x_min_V": 0.45,
        "x_max_V": 0.75,
    },
)
CSV_NAME = "four_case_polarization_curves.csv"
SUMMARY_NAME = "polarization_summary.json"
DPI = 600

COLORS = {
    "au_curve": "#009E73",
    "pd_curve": "#0072B2",
    "dark": "#272727",
    "gray": "#767676",
    "light_gray": "#CFCECE",
}

CASE_STYLES: dict[str, dict[str, Any]] = {
    "au_pd_independent": {
        "linestyle": "solid",
        "marker": "o",
        "label": "Independent Au/Pd",
        "alpha": 1.0,
    },
    "janus_au_pd": {
        "linestyle": (0, (5.0, 2.0, 1.2, 2.0)),
        "marker": "s",
        "label": "Janus Au|Pd",
        "alpha": 1.0,
    },
    "physical_mixture_c10": {
        "linestyle": (0, (1.0, 2.0)),
        "marker": "^",
        "label": "Physical mixture (C = 10 nm)",
        "alpha": 0.78,
    },
    "janus_on_c_support": {
        "linestyle": (0, (7.0, 2.2)),
        "marker": "D",
        "label": "C-supported Janus",
        "alpha": 0.92,
    },
    "without_edl_common": {
        "linestyle": (0, (4.0, 2.4)),
        "marker": "P",
        "label": "w/o EDL reference",
        "alpha": 0.34,
    },
}

RC_PARAMS = {
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

EXPECTED_KEYS = (
    "au_pd_independent",
    "janus_au_pd",
    "physical_mixture_c10",
    "janus_on_c_support",
)


CurrentEvaluator = Callable[[float], tuple[float, float]]


def _insert_import_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _ordered_cases(cases: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    by_key = {str(case["key"]): case for case in cases}
    if tuple(by_key) != EXPECTED_KEYS and set(by_key) != set(EXPECTED_KEYS):
        raise ValueError(f"Expected cases {EXPECTED_KEYS!r}; got {tuple(by_key)!r}")
    return [by_key[key] for key in EXPECTED_KEYS]


def _validate_potentials(values: np.ndarray | Sequence[float] | None) -> np.ndarray:
    potentials = np.asarray(POTENTIAL_VALUES_V if values is None else values, dtype=float)
    if potentials.ndim != 1 or potentials.size < 2:
        raise ValueError("potentials_V must be a 1D array with at least two values")
    if np.any(~np.isfinite(potentials)) or np.any(np.diff(potentials) <= 0.0):
        raise ValueError("potentials_V must be finite and strictly increasing")
    return potentials


def _effective_reaction(params: Mapping[str, Any]) -> dict[str, float]:
    delta_p_h = float(params["pH"]) - float(params["pH_ref"])
    return {
        "E1_eq_eff": float(params["E1_eq"])
        + float(params["E1_eq_pH_slope_V_per_pH"]) * delta_p_h,
        "E2_eq_eff": float(params["E2_eq"])
        + float(params["E2_eq_pH_slope_V_per_pH"]) * delta_p_h,
        "it0_1_eff": float(params["it0_1"])
        * math.exp(-math.log(10.0) * float(params["it0_1_pH_order"]) * delta_p_h),
        "it0_2_eff": float(params["it0_2"])
        * math.exp(-math.log(10.0) * float(params["it0_2_pH_order"]) * delta_p_h),
    }


def _without_edl_evaluator(case: Mapping[str, Any]) -> CurrentEvaluator:
    """Exact uniform-interface limit on the common Summary reactive area."""

    params = case["params"]
    reaction = _effective_reaction(params)
    beta = float(params["F"]) / (float(params["R"]) * float(params["T"]))
    alpha1 = float(params["alpha1"])
    alpha2 = float(params["alpha2"])
    half_area = 0.5 * float(case["comparison_reactive_area_m2"])

    def evaluate(E_V: float) -> tuple[float, float]:
        i_au = half_area * reaction["it0_1_eff"] * math.exp(
            float(
                np.clip(
                    (1.0 - alpha1) * beta * (float(E_V) - reaction["E1_eq_eff"]),
                    -700.0,
                    700.0,
                )
            )
        )
        i_pd = -half_area * reaction["it0_2_eff"] * math.exp(
            float(
                np.clip(
                    -alpha2 * beta * (float(E_V) - reaction["E2_eq_eff"]),
                    -700.0,
                    700.0,
                )
            )
        )
        return float(i_au), float(i_pd)

    return evaluate


def _independent_evaluator(
    case: Mapping[str, Any], workspace_root: Path
) -> CurrentEvaluator:
    source_root = (
        workspace_root
        / "Mixed_Potential_Electrical_Double_Layer"
        / "Au_Pd_independent_EDLs"
        / "src"
    )
    _insert_import_path(source_root)
    model_module = importlib.import_module("au_pd_independent_edls.model")
    model = model_module.IndependentPlanarEDLModel(case["params"])
    scale = float(case["scale_factor"])

    def evaluate(E_V: float) -> tuple[float, float]:
        currents = model.current_components(float(E_V), use_edl=True)
        return (
            scale * float(currents["I_Au_A"]),
            scale * float(currents["I_Pd_A"]),
        )

    return evaluate


def _legacy_evaluator(
    case: Mapping[str, Any], workspace_root: Path
) -> CurrentEvaluator:
    module_root = workspace_root / "Figures" / "Figure_Au2nm_Pd2nm"
    expected_module = module_root / "legacy_au_c_pd_engine.py"
    _insert_import_path(module_root)
    legacy = importlib.import_module("legacy_au_c_pd_engine")
    if Path(legacy.__file__).resolve() != expected_module.resolve():
        raise ImportError(f"Unexpected legacy engine source: {legacy.__file__}")

    params = dict(case["params"])
    model = legacy._build_coefficient_model(params)
    quadrature = legacy._build_gauss_data(model, int(legacy.DEFAULT_GL_ORDER))
    current_scale = (
        float(model.derived["lambda_D"])
        * float(params["out_of_plane_width"])
        * float(case["scale_factor"])
    )

    def evaluate(E_V: float) -> tuple[float, float]:
        currents = legacy._gauss_currents(float(E_V), quadrature, model, params)
        return (
            current_scale * float(currents["I_Au"]),
            current_scale * float(currents["I_Pd"]),
        )

    return evaluate


def _janus_on_c_evaluator(
    case: Mapping[str, Any], workspace_root: Path
) -> CurrentEvaluator:
    source_root = workspace_root / "Figures" / "Figure_Janus_on_C_support" / "src"
    _insert_import_path(source_root)
    electrostatics = importlib.import_module("janus_on_c_support.electrostatics")
    solver = importlib.import_module("janus_on_c_support.solver")
    params = dict(case["params"])
    model = electrostatics.LinearPBModel(params)
    quadrature = solver._prepare_quadrature(model)
    constants = solver._kinetic_constants(params, model.derived)
    scale = float(case["scale_factor"])

    def evaluate(E_V: float) -> tuple[float, float]:
        currents = solver._integrated_currents(
            float(E_V),
            params=params,
            derived=model.derived,
            constants=constants,
            use_edl=True,
            quadrature=quadrature,
        )
        return (
            scale * float(currents["I_Au_A"]),
            scale * float(currents["I_Pd_A"]),
        )

    return evaluate


def _with_edl_evaluators(
    cases: Sequence[Mapping[str, Any]], workspace_root: Path
) -> dict[str, CurrentEvaluator]:
    by_key = {str(case["key"]): case for case in cases}
    return {
        "au_pd_independent": _independent_evaluator(
            by_key["au_pd_independent"], workspace_root
        ),
        "janus_au_pd": _legacy_evaluator(by_key["janus_au_pd"], workspace_root),
        "physical_mixture_c10": _legacy_evaluator(
            by_key["physical_mixture_c10"], workspace_root
        ),
        "janus_on_c_support": _janus_on_c_evaluator(
            by_key["janus_on_c_support"], workspace_root
        ),
    }


def _relative_balance(i_au: float, i_pd: float) -> float:
    return float(abs(i_au + i_pd) / (abs(i_au) + abs(i_pd) + 1.0e-300))


def _row(
    *,
    order: int,
    key: str,
    label: str,
    edl_condition: str,
    E_V: float,
    i_au_A: float,
    i_pd_A: float,
) -> dict[str, Any]:
    return {
        "condition_order": int(order),
        "condition_key": key,
        "display_label": label,
        "edl_condition": edl_condition,
        "E_V": float(E_V),
        "I_Au_A": float(i_au_A),
        "I_Pd_A": float(i_pd_A),
        "I_Au_1e_minus_3_uA": CURRENT_DISPLAY_SCALE * float(i_au_A),
        "I_Pd_1e_minus_3_uA": CURRENT_DISPLAY_SCALE * float(i_pd_A),
        "net_current_A": float(i_au_A + i_pd_A),
    }


def compute_summary_polarization(
    cases: Sequence[Mapping[str, Any]],
    workspace_root: str | Path,
    potentials_V: np.ndarray | Sequence[float] | None = None,
) -> dict[str, Any]:
    """Compute four with-EDL curves plus one common w/o-EDL reference."""

    ordered = _ordered_cases(cases)
    workspace = Path(workspace_root).expanduser().resolve()
    potentials = _validate_potentials(potentials_V)
    evaluators = _with_edl_evaluators(ordered, workspace)
    without_evaluators = {
        str(case["key"]): _without_edl_evaluator(case) for case in ordered
    }

    rows: list[dict[str, Any]] = []
    conditions: list[dict[str, Any]] = []
    marker_checks: list[dict[str, Any]] = []

    for order, case in enumerate(ordered, start=1):
        key = str(case["key"])
        label = str(case["display_label"])
        evaluate = evaluators[key]
        for E_V in potentials:
            i_au, i_pd = evaluate(float(E_V))
            rows.append(
                _row(
                    order=order,
                    key=key,
                    label=label,
                    edl_condition="with EDL",
                    E_V=float(E_V),
                    i_au_A=i_au,
                    i_pd_A=i_pd,
                )
            )

        E_mix = float(case["E_with_V"])
        expected_I = float(case["I_with_A"])
        marker_au, marker_pd = evaluate(E_mix)
        balance = _relative_balance(marker_au, marker_pd)
        au_error = abs(abs(marker_au) - expected_I)
        pd_error = abs(abs(marker_pd) - expected_I)
        marker_checks.append(
            {
                "condition_key": key,
                "E_mix_V": E_mix,
                "expected_I_mix_A": expected_I,
                "computed_I_Au_A": marker_au,
                "computed_I_Pd_A": marker_pd,
                "relative_balance_residual": balance,
                "Au_magnitude_error_A": au_error,
                "Pd_magnitude_error_A": pd_error,
                "passed": bool(
                    balance < 1.0e-10 and au_error < 1.0e-22 and pd_error < 1.0e-22
                ),
            }
        )
        conditions.append(
            {
                "condition_order": order,
                "condition_key": key,
                "display_label": label,
                "edl_condition": "with EDL",
                "E_mix_V": E_mix,
                "I_mix_A": expected_I,
                "I_mix_pA": 1.0e12 * expected_I,
                "source_current_scale_factor": float(case["scale_factor"]),
            }
        )

    reference_case = ordered[0]
    reference_evaluator = without_evaluators[str(reference_case["key"])]
    reference_au = np.empty(potentials.size, dtype=float)
    reference_pd = np.empty(potentials.size, dtype=float)
    for index, E_V in enumerate(potentials):
        i_au, i_pd = reference_evaluator(float(E_V))
        reference_au[index] = i_au
        reference_pd[index] = i_pd
        rows.append(
            _row(
                order=5,
                key="without_edl_common",
                label="Common w/o EDL",
                edl_condition="w/o EDL",
                E_V=float(E_V),
                i_au_A=i_au,
                i_pd_A=i_pd,
            )
        )

    E_no_values = np.asarray([float(case["E_no_V"]) for case in ordered])
    I_no_values = np.asarray([float(case["I_no_A"]) for case in ordered])
    if not np.allclose(E_no_values, E_no_values[0], rtol=1.0e-12, atol=1.0e-12):
        raise RuntimeError("The cases do not share one w/o-EDL mixed potential")
    if not np.allclose(I_no_values, I_no_values[0], rtol=1.0e-12, atol=1.0e-24):
        raise RuntimeError("The cases do not share one w/o-EDL mixed current")
    E_no = float(np.mean(E_no_values))
    I_no = float(np.mean(I_no_values))
    no_marker_au, no_marker_pd = reference_evaluator(E_no)
    no_balance = _relative_balance(no_marker_au, no_marker_pd)
    no_au_error = abs(abs(no_marker_au) - I_no)
    no_pd_error = abs(abs(no_marker_pd) - I_no)
    marker_checks.append(
        {
            "condition_key": "without_edl_common",
            "E_mix_V": E_no,
            "expected_I_mix_A": I_no,
            "computed_I_Au_A": no_marker_au,
            "computed_I_Pd_A": no_marker_pd,
            "relative_balance_residual": no_balance,
            "Au_magnitude_error_A": no_au_error,
            "Pd_magnitude_error_A": no_pd_error,
            "passed": bool(
                no_balance < 1.0e-10
                and no_au_error < 1.0e-22
                and no_pd_error < 1.0e-22
            ),
        }
    )
    conditions.append(
        {
            "condition_order": 5,
            "condition_key": "without_edl_common",
            "display_label": "Common w/o EDL",
            "edl_condition": "w/o EDL",
            "E_mix_V": E_no,
            "I_mix_A": I_no,
            "I_mix_pA": 1.0e12 * I_no,
            "source_current_scale_factor": None,
        }
    )

    cross_case_curve_checks: list[dict[str, Any]] = []
    for case in ordered:
        key = str(case["key"])
        evaluator = without_evaluators[key]
        candidate = np.asarray([evaluator(float(E_V)) for E_V in potentials], dtype=float)
        candidate_au = candidate[:, 0]
        candidate_pd = candidate[:, 1]
        absolute_error = float(
            max(
                np.max(np.abs(candidate_au - reference_au)),
                np.max(np.abs(candidate_pd - reference_pd)),
            )
        )
        reference_scale = max(
            float(np.max(np.abs(reference_au))),
            float(np.max(np.abs(reference_pd))),
            1.0e-300,
        )
        relative_error = absolute_error / reference_scale
        cross_case_curve_checks.append(
            {
                "case_key": key,
                "max_absolute_difference_A": absolute_error,
                "max_relative_difference": relative_error,
                "passed": bool(relative_error < 1.0e-12),
            }
        )

    validation = {
        "current_definition": CURRENT_DEFINITION,
        "curve_count": 5,
        "half_reaction_curve_count": 10,
        "figure_variant_count": len(FIGURE_VARIANTS),
        "potential_point_count_per_condition": int(potentials.size),
        "csv_row_count": len(rows),
        "common_reactive_area_m2": float(reference_case["comparison_reactive_area_m2"]),
        "marker_checks": marker_checks,
        "without_edl_cross_case_curve_checks": cross_case_curve_checks,
        "maximum_marker_magnitude_error_A": float(
            max(
                max(float(check["Au_magnitude_error_A"]), float(check["Pd_magnitude_error_A"]))
                for check in marker_checks
            )
        ),
        "maximum_marker_relative_balance_residual": float(
            max(float(check["relative_balance_residual"]) for check in marker_checks)
        ),
        "passed": bool(
            all(bool(check["passed"]) for check in marker_checks)
            and all(bool(check["passed"]) for check in cross_case_curve_checks)
            and len(rows) == 5 * potentials.size
        ),
    }
    if not validation["passed"]:
        raise RuntimeError(f"Polarization validation failed: {validation}")

    return {
        "rows": rows,
        "conditions": conditions,
        "validation": validation,
        "potential_grid": {
            "minimum_V": float(potentials[0]),
            "maximum_V": float(potentials[-1]),
            "point_count": int(potentials.size),
            "spacing": "linear",
        },
        "method": {
            "au_pd_independent": "IndependentPlanarEDLModel.current_components",
            "janus_au_pd": "legacy cosine-Galerkin model with GL128 current quadrature",
            "physical_mixture_c10": (
                "legacy cosine-Galerkin model with GL128 current quadrature"
            ),
            "janus_on_c_support": (
                "LinearPBModel with cached GL current quadrature"
            ),
            "without_edl_common": "analytic uniform-interface limit",
            "normalization": "none; signed absolute currents on common Au4+Pd4 area",
        },
    }


def _condition_rows(
    rows: Sequence[Mapping[str, Any]], condition_key: str
) -> list[Mapping[str, Any]]:
    subset = [row for row in rows if str(row["condition_key"]) == condition_key]
    if not subset:
        raise ValueError(f"Missing polarization rows for {condition_key}")
    return subset


def generate_summary_polarization_figure(
    data: Mapping[str, Any], output_dir: str | Path
) -> list[Path]:
    """Render the five conditions in the 0.40--0.64 and 0.45--0.75 V views."""

    rows = data["rows"]
    conditions = data["conditions"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    curve_min_V = min(float(row["E_V"]) for row in rows)
    curve_max_V = max(float(row["E_V"]) for row in rows)

    for variant in FIGURE_VARIANTS:
        if (
            float(variant["x_min_V"]) < curve_min_V - 1.0e-12
            or float(variant["x_max_V"]) > curve_max_V + 1.0e-12
        ):
            raise ValueError(
                f"Figure variant {variant['variant_key']} requests "
                f"{variant['x_min_V']}--{variant['x_max_V']} V, but curve data "
                f"cover only {curve_min_V}--{curve_max_V} V"
            )
        with matplotlib.rc_context(RC_PARAMS):
            fig, ax = plt.subplots(figsize=(7.6, 4.55))
            for condition in conditions:
                key = str(condition["condition_key"])
                style = CASE_STYLES[key]
                subset = _condition_rows(rows, key)
                potential = np.asarray([row["E_V"] for row in subset], dtype=float)
                line_width = 1.55 if key == "without_edl_common" else 2.0
                zorder = 1 if key == "without_edl_common" else 2
                for field, color in (
                    ("I_Au_1e_minus_3_uA", COLORS["au_curve"]),
                    ("I_Pd_1e_minus_3_uA", COLORS["pd_curve"]),
                ):
                    ax.plot(
                        potential,
                        np.asarray([row[field] for row in subset], dtype=float),
                        color=color,
                        linewidth=line_width,
                        linestyle=style["linestyle"],
                        alpha=float(style["alpha"]),
                        zorder=zorder,
                    )

            unique_mixed_potentials: list[float] = []
            for condition in conditions:
                E_mix = float(condition["E_mix_V"])
                if not any(
                    abs(E_mix - value) < 1.0e-10
                    for value in unique_mixed_potentials
                ):
                    unique_mixed_potentials.append(E_mix)
                    ax.axvline(
                        E_mix,
                        color=COLORS["gray"],
                        linewidth=0.8,
                        alpha=0.45,
                        zorder=0,
                    )

            marker_values: list[float] = []
            for condition in conditions:
                key = str(condition["condition_key"])
                style = CASE_STYLES[key]
                E_mix = float(condition["E_mix_V"])
                displayed_I = CURRENT_DISPLAY_SCALE * float(condition["I_mix_A"])
                marker_values.append(abs(displayed_I))
                if key == "without_edl_common":
                    facecolor = "white"
                    edgecolor = COLORS["gray"]
                else:
                    facecolor = COLORS["dark"]
                    edgecolor = "white"
                ax.scatter(
                    [E_mix, E_mix],
                    [displayed_I, -displayed_I],
                    marker=str(style["marker"]),
                    s=48,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=0.8,
                    alpha=max(float(style["alpha"]), 0.72),
                    zorder=5,
                )

            ax.axhline(0.0, color=COLORS["dark"], linewidth=0.9)
            y_limit = max(0.008, 2.8 * max(marker_values))
            ax.set_xlim(float(variant["x_min_V"]), float(variant["x_max_V"]))
            ax.set_ylim(-y_limit, y_limit)
            ax.set_xlabel(X_AXIS_LABEL)
            ax.set_ylabel(Y_AXIS_LABEL)
            ax.set_title(
                r"Topology-dependent polarization curves explain $I_{\mathrm{mix}}$",
                loc="left",
                fontsize=10.2,
            )

            half_reaction_legend = ax.legend(
                handles=[
                    Line2D(
                        [0],
                        [0],
                        color=COLORS["au_curve"],
                        lw=2.5,
                        label=AU_LEGEND_LABEL,
                    ),
                    Line2D(
                        [0],
                        [0],
                        color=COLORS["pd_curve"],
                        lw=2.5,
                        label=PD_LEGEND_LABEL,
                    ),
                ],
                loc="upper left",
                fontsize=7.6,
            )
            ax.add_artist(half_reaction_legend)
            configuration_legend_location = (
                "lower right"
                if float(variant["x_max_V"]) > 0.64
                else "upper right"
            )
            ax.legend(
                handles=[
                    Line2D(
                        [0],
                        [0],
                        color=(
                            COLORS["gray"]
                            if str(condition["condition_key"])
                            == "without_edl_common"
                            else COLORS["dark"]
                        ),
                        lw=1.8,
                        ls=CASE_STYLES[str(condition["condition_key"])][
                            "linestyle"
                        ],
                        marker=CASE_STYLES[str(condition["condition_key"])][
                            "marker"
                        ],
                        markerfacecolor=(
                            "white"
                            if str(condition["condition_key"])
                            == "without_edl_common"
                            else COLORS["dark"]
                        ),
                        label=CASE_STYLES[str(condition["condition_key"])]["label"],
                        alpha=max(
                            float(
                                CASE_STYLES[str(condition["condition_key"])][
                                    "alpha"
                                ]
                            ),
                            0.72,
                        ),
                    )
                    for condition in conditions
                ],
                title="Configuration",
                loc=configuration_legend_location,
                fontsize=7.2,
                title_fontsize=7.6,
                handlelength=2.6,
                labelspacing=0.55,
            )
            ax.tick_params(length=3.2, width=0.85, labelsize=8.1)

            for suffix in ("png", "svg"):
                path = output / f"{variant['stem']}.{suffix}"
                fig.savefig(
                    path,
                    dpi=DPI,
                    bbox_inches="tight",
                    pad_inches=0.06,
                    facecolor="white",
                    edgecolor="none",
                )
                saved.append(path)
            plt.close(fig)
    return saved


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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if not rows:
        raise ValueError("Cannot write an empty polarization CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
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
    return path


def build_summary_polarization(
    cases: Sequence[Mapping[str, Any]],
    workspace_root: str | Path,
    summary_root: str | Path,
) -> dict[str, Any]:
    """Compute, validate, and export the traceable Summary overlay package."""

    summary = Path(summary_root)
    data = compute_summary_polarization(cases, workspace_root)
    csv_path = _write_csv(summary / "csv" / CSV_NAME, data["rows"])
    json_payload = {
        "figure_stem": FIGURE_STEM,
        "figure_stems": [str(variant["stem"]) for variant in FIGURE_VARIANTS],
        "figure_variants": FIGURE_VARIANTS,
        "x_axis_label": X_AXIS_LABEL,
        "y_axis_label": Y_AXIS_LABEL,
        "reaction_legend_labels": [AU_LEGEND_LABEL, PD_LEGEND_LABEL],
        "curve_csv": f"csv/{CSV_NAME}",
        "current_definition": CURRENT_DEFINITION,
        "potential_grid": data["potential_grid"],
        "conditions": data["conditions"],
        "method": data["method"],
        "validation": data["validation"],
    }
    json_path = _write_json(summary / SUMMARY_NAME, json_payload)
    figure_paths = generate_summary_polarization_figure(
        data, summary / "figures"
    )
    return {
        **data,
        "data_paths": [csv_path, json_path],
        "figure_paths": figure_paths,
    }


__all__ = [
    "CSV_NAME",
    "AU_LEGEND_LABEL",
    "FIGURE_STEM",
    "FIGURE_STEM_XMIN045",
    "FIGURE_VARIANTS",
    "PD_LEGEND_LABEL",
    "SUMMARY_NAME",
    "X_AXIS_LABEL",
    "Y_AXIS_LABEL",
    "build_summary_polarization",
    "compute_summary_polarization",
    "generate_summary_polarization_figure",
]
