from __future__ import annotations

import copy
import json
import math
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SAME_LENGTH_I0_ALPHA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SAME_LENGTH_I0_ALPHA_DIR))

import same_length_i0_alpha_common as common  # noqa: E402


solver = common.solver

HEATMAP_DIR = common.OUT_BASE / "Heatmap"
CSV_DIR = HEATMAP_DIR / "csv"
INPUTS_DIR = HEATMAP_DIR / "inputs"

PNG_OUT = HEATMAP_DIR / "heatmap_combined_panel_log.png"
SVG_OUT = HEATMAP_DIR / "heatmap_combined_panel_log.svg"
EXPECTED_CSV_PER_SCAN = 10
SCAN_TAGS = ("pzcAu_vs_pzcPd", "CdlAu_vs_CdlPd", "LAu_vs_LPd")


def units_to_parentheses(label: str) -> str:
    return re.sub(r"\s+\[([^\]]+)\]", r" (\1)", label)


def apply_unit_label_style() -> None:
    for labels in (solver.PLOT_AXIS_LABELS, solver.PARAM_AXIS_LABELS):
        for key, label in list(labels.items()):
            labels[key] = units_to_parentheses(label)

    original_style_heatmap_colorbar = solver._style_heatmap_colorbar

    def style_heatmap_colorbar_with_parentheses(cbar: Any, label: str, *args: Any, **kwargs: Any) -> Any:
        return original_style_heatmap_colorbar(cbar, units_to_parentheses(label), *args, **kwargs)

    solver._style_heatmap_colorbar = style_heatmap_colorbar_with_parentheses


def ensure_heatmap_dirs() -> None:
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)


def save_heatmap_inputs(params: dict[str, Any], summary: dict[str, str]) -> None:
    ensure_heatmap_dirs()
    with (INPUTS_DIR / f"params_{common.OUTPUT_TAG}.json").open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, sort_keys=True)
        f.write("\n")
    with (INPUTS_DIR / f"overrides_{common.OUTPUT_TAG}.json").open("w", encoding="utf-8") as f:
        json.dump(common.PARAM_OVERRIDES, f, indent=2, sort_keys=True)
        f.write("\n")
    pd.DataFrame([summary]).to_csv(INPUTS_DIR / f"summary_compare_{common.OUTPUT_TAG}.csv", index=False)
    with (INPUTS_DIR / f"summary_compare_{common.OUTPUT_TAG}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")


def assert_baseline(params: dict[str, Any], summary: dict[str, str]) -> None:
    common.validate_same_length_i0_alpha_params(params)
    common.assert_expected_values(summary)
    checks = {
        "L_Au": common.L_AU_SAME,
        "L_Pd_len": common.L_PD_SAME,
        "L_gap": 10.0e-9,
        "it0_1": common.I0_GEOM,
        "it0_2": common.I0_GEOM,
        "alpha1": common.ALPHA_EQUAL,
        "alpha2": common.ALPHA_EQUAL,
        "out_of_plane_width": common.OUT_OF_PLANE_WIDTH,
    }
    for key, expected in checks.items():
        actual = float(params[key])
        atol = max(1e-15, abs(expected) * 1e-12)
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol):
            raise ValueError(f"{key}: expected {expected:.15g}, got {actual:.15g}")


def storage_axis_name(name: str) -> str:
    if name == "C_tot":
        return "C_tot_M"
    if name in {"L_Au", "L_gap", "L_Pd_len"}:
        return f"{name}_nm"
    if name in {"Cdl_Au", "Cdl_C", "Cdl_Pd"}:
        return f"{name}_uF_per_cm2"
    return f"{name}_V" if name.startswith("pzc_") else name


def axis_values(vmin: float, vmax: float, count: int, scale: str) -> np.ndarray:
    if scale == "log":
        if vmin <= 0.0 or vmax <= vmin:
            raise ValueError("log axis values need a positive increasing range")
        return np.logspace(np.log10(vmin), np.log10(vmax), count)
    if vmax <= vmin:
        raise ValueError("linear axis values need an increasing range")
    return np.linspace(vmin, vmax, count)


def safe_arrays_are_finite(entry: dict[str, Any]) -> None:
    array_keys = (
        "Emix_with",
        "Emix_no",
        "delta_Emix",
        "imix_avg_with",
        "imix_avg_no",
        "delta_imix_avg",
        "log10_imix_avg_with",
        "log10_imix_avg_no",
        "delta_log10_imix_avg",
        "imix_ratio",
    )
    for key in array_keys:
        arr = np.asarray(entry[key], dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{entry['tag']} {key} contains non-finite values")


def write_heatmap_csvs(entry: dict[str, Any]) -> None:
    tag = entry["tag"]
    x_disp = np.asarray(entry["x_disp"], dtype=float)
    y_disp = np.asarray(entry["y_disp"], dtype=float)
    x_axis = storage_axis_name(entry["x_name"])
    y_axis = storage_axis_name(entry["y_name"])

    csv_specs = {
        f"heatmap_compare_{tag}_Emix_with_edl_FULL.csv": entry["Emix_with"],
        f"heatmap_compare_{tag}_Emix_without_edl.csv": entry["Emix_no"],
        f"heatmap_compare_{tag}_delta_Emix.csv": entry["delta_Emix"],
        f"heatmap_compare_{tag}_imix_avg_with_edl_FULL.csv": entry["imix_avg_with"],
        f"heatmap_compare_{tag}_imix_avg_without_edl.csv": entry["imix_avg_no"],
        f"heatmap_compare_{tag}_delta_i_mix_avg.csv": entry["delta_imix_avg"],
        f"heatmap_compare_{tag}_log10_imix_avg_with_edl_FULL.csv": entry["log10_imix_avg_with"],
        f"heatmap_compare_{tag}_log10_imix_avg_without_edl.csv": entry["log10_imix_avg_no"],
        f"heatmap_compare_{tag}_ratio_i_mix_avg.csv": entry["imix_ratio"],
        f"heatmap_compare_{tag}_delta_log10_i_mix_avg.csv": entry["delta_log10_imix_avg"],
    }

    for filename, values in csv_specs.items():
        df = pd.DataFrame(np.asarray(values, dtype=float), index=y_disp, columns=x_disp)
        df.index.name = y_axis
        df.columns.name = x_axis
        df.to_csv(CSV_DIR / filename)


def paired_compare_heatmap_data(
    *,
    base_params: dict[str, Any],
    tag: str,
    x_name: str,
    x_vals: np.ndarray,
    y_name: str,
    y_vals: np.ndarray,
    xscale: str,
    yscale: str,
    assign_fn: Callable[[dict[str, Any], float, float], None],
) -> dict[str, Any]:
    Emix_with = np.full((len(y_vals), len(x_vals)), np.nan)
    Emix_no = np.full((len(y_vals), len(x_vals)), np.nan)
    imix_avg_with = np.full((len(y_vals), len(x_vals)), np.nan)
    imix_avg_no = np.full((len(y_vals), len(x_vals)), np.nan)

    total = len(x_vals) * len(y_vals)
    completed = 0
    for iy, yv in enumerate(y_vals):
        for ix, xv in enumerate(x_vals):
            pvar = copy.deepcopy(base_params)
            assign_fn(pvar, float(xv), float(yv))
            pair = solver.run_edl_comparison_pair(pvar, mode="FULL")
            res_edl = pair["with_edl"]
            res_no = pair["no_edl"]
            Emix_with[iy, ix] = float(res_edl["E_mix"])
            Emix_no[iy, ix] = float(res_no["E_mix"])
            imix_avg_with[iy, ix] = float(res_edl["i_mix_avg_A_per_m2"])
            imix_avg_no[iy, ix] = float(res_no["i_mix_avg_A_per_m2"])

            completed += 1
            if completed in {1, total} or completed % max(1, total // 5) == 0:
                print(f"{tag}: {completed}/{total} cells")

    x_disp = solver._display_param_values(x_name, x_vals)
    y_disp = solver._display_param_values(y_name, y_vals)
    log10_imix_with = solver._safe_log10_abs(imix_avg_with)
    log10_imix_no = solver._safe_log10_abs(imix_avg_no)
    entry = {
        "tag": tag,
        "x_name": x_name,
        "y_name": y_name,
        "xscale": xscale,
        "yscale": yscale,
        "x_disp": x_disp,
        "y_disp": y_disp,
        "x_edges": solver.make_edges(x_disp, xscale),
        "y_edges": solver.make_edges(y_disp, yscale),
        "baseline_point": (
            solver._display_param_value(x_name, float(base_params[x_name])),
            solver._display_param_value(y_name, float(base_params[y_name])),
        ),
        "Emix_with": Emix_with,
        "Emix_no": Emix_no,
        "delta_Emix": Emix_with - Emix_no,
        "imix_avg_with": imix_avg_with,
        "imix_avg_no": imix_avg_no,
        "delta_imix_avg": imix_avg_with - imix_avg_no,
        "log10_imix_avg_with": log10_imix_with,
        "log10_imix_avg_no": log10_imix_no,
        "delta_log10_imix_avg": log10_imix_with - log10_imix_no,
        "imix_ratio": solver._safe_ratio(imix_avg_with, imix_avg_no),
    }
    safe_arrays_are_finite(entry)
    write_heatmap_csvs(entry)
    return entry


def build_log_heatmap_data(params: dict[str, Any]) -> list[dict[str, Any]]:
    nx = int(params["heatmap_nx"])
    ny = int(params["heatmap_ny"])
    if nx <= 1 or ny <= 1:
        raise ValueError("heatmap_nx and heatmap_ny must both be > 1")

    pzc_span = float(params.get("heatmap_pzc_span", 0.30))
    pzc_Au0 = float(params["pzc_Au"])
    pzc_Pd0 = float(params["pzc_Pd"])

    cdl_min = float(params.get("heatmap_Cdl_C_min", 0.05))
    cdl_max = float(params.get("heatmap_Cdl_C_max", 1.0))
    l_min = float(params["heatmap_L_min"])
    l_max = float(params["heatmap_L_max"])

    entries = [
        paired_compare_heatmap_data(
            base_params=params,
            tag="pzcAu_vs_pzcPd",
            x_name="pzc_Au",
            x_vals=np.linspace(pzc_Au0 - pzc_span, pzc_Au0 + pzc_span, nx),
            y_name="pzc_Pd",
            y_vals=np.linspace(pzc_Pd0 - pzc_span, pzc_Pd0 + pzc_span, ny),
            xscale="linear",
            yscale="linear",
            assign_fn=lambda pvar, xv, yv: (pvar.__setitem__("pzc_Au", xv), pvar.__setitem__("pzc_Pd", yv)),
        ),
        paired_compare_heatmap_data(
            base_params=params,
            tag="CdlAu_vs_CdlPd",
            x_name="Cdl_Au",
            x_vals=axis_values(cdl_min, cdl_max, nx, "log"),
            y_name="Cdl_Pd",
            y_vals=axis_values(cdl_min, cdl_max, ny, "log"),
            xscale="log",
            yscale="log",
            assign_fn=lambda pvar, xv, yv: (pvar.__setitem__("Cdl_Au", xv), pvar.__setitem__("Cdl_Pd", yv)),
        ),
        paired_compare_heatmap_data(
            base_params=params,
            tag="LAu_vs_LPd",
            x_name="L_Au",
            x_vals=axis_values(l_min, l_max, nx, "log"),
            y_name="L_Pd_len",
            y_vals=axis_values(l_min, l_max, ny, "log"),
            xscale="log",
            yscale="log",
            assign_fn=lambda pvar, xv, yv: (pvar.__setitem__("L_Au", xv), pvar.__setitem__("L_Pd_len", yv)),
        ),
    ]
    return entries


def assert_csv_outputs(params: dict[str, Any]) -> None:
    expected_shape = (int(params["heatmap_ny"]), int(params["heatmap_nx"]))
    csv_files = sorted(CSV_DIR.glob("heatmap_compare_*.csv"))
    expected_count = EXPECTED_CSV_PER_SCAN * len(SCAN_TAGS)
    if len(csv_files) != expected_count:
        raise RuntimeError(f"Expected {expected_count} heatmap CSVs, got {len(csv_files)}")

    for path in csv_files:
        df = pd.read_csv(path, index_col=0)
        if tuple(df.shape) != expected_shape:
            raise RuntimeError(f"{path} shape should be {expected_shape}, got {tuple(df.shape)}")
        values = df.to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"{path} contains non-finite values")


def assert_figure_outputs() -> None:
    for path in (PNG_OUT, SVG_OUT):
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Missing output figure: {path}")
    pdfs = sorted(HEATMAP_DIR.glob("*.pdf")) + sorted(HEATMAP_DIR.glob("*/*.pdf"))
    if pdfs:
        raise RuntimeError(f"Expected zero PDF outputs in {HEATMAP_DIR}, found {len(pdfs)}")
    unwanted_linear = sorted(HEATMAP_DIR.glob("heatmap_combined_panel_linear.*"))
    if unwanted_linear:
        raise RuntimeError(f"Linear heatmap outputs are not expected: {unwanted_linear}")


def main() -> None:
    apply_unit_label_style()
    ensure_heatmap_dirs()
    params, summary = common.load_inputs_for_scripts()
    assert_baseline(params, summary)
    save_heatmap_inputs(params, summary)

    print(f"Building log heatmap for {common.OUTPUT_TAG}")
    print(f"heatmap grid = {int(params['heatmap_nx'])} x {int(params['heatmap_ny'])}")
    entries = build_log_heatmap_data(params)
    star_summary = solver._baseline_star_edl_summary(params)

    solver._plot_combined_heatmap_panel(entries, PNG_OUT, "", star_summary=star_summary)
    solver._plot_combined_heatmap_panel(entries, SVG_OUT, "", star_summary=star_summary)

    assert_csv_outputs(params)
    assert_figure_outputs()

    print(f"Wrote {PNG_OUT}")
    print(f"Wrote {SVG_OUT}")
    print(f"Wrote {EXPECTED_CSV_PER_SCAN * len(SCAN_TAGS)} heatmap CSV files to {CSV_DIR}")
    print(f"E_mix_with = {float(summary['E_mix_with']):.16g} V")
    print(f"E_mix_no = {float(summary['E_mix_no']):.16g} V")
    print(f"i_mix_avg_with = {float(summary['i_mix_avg_with']):.16g} A/m^2")
    print(f"i_mix_avg_no = {float(summary['i_mix_avg_no']):.16g} A/m^2")


if __name__ == "__main__":
    main()
