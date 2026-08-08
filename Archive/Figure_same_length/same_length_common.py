from __future__ import annotations

import csv
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASE_RESULT_ID = "20260528_111255"
OUTPUT_TAG = f"same_length_au25_pd25_{BASE_RESULT_ID}"

SOLVER_DIR = ROOT / "Mixed_Potential_Electrical_Double_Layer"
BASE_RESULT_DIR = SOLVER_DIR / "results" / BASE_RESULT_ID
BASE_PARAMS_PATH = BASE_RESULT_DIR / "params.json"

OUT_BASE = ROOT / "Figures" / "Figure_same_length"
INPUTS_DIR = OUT_BASE / "inputs"
FIGURE_3_DIR = OUT_BASE / "Figure_3"
FIGURE_SCHEME_DIR = OUT_BASE / "Figure_scheme"
FIGRUE_RP_DIR = OUT_BASE / "Figrue_RP"

L_AU_SAME = 25.0e-9
L_PD_SAME = 25.0e-9
OUT_OF_PLANE_WIDTH = 0.01
LENGTH_OVERRIDES = {
    "L_Au": L_AU_SAME,
    "L_Pd_len": L_PD_SAME,
    "out_of_plane_width": OUT_OF_PLANE_WIDTH,
}
EXPECTED_IMAGE_COUNT = 16

sys.path.insert(0, str(SOLVER_DIR))
import Solve_Emix_updating as solver  # noqa: E402


def ensure_output_dirs() -> None:
    for path in (INPUTS_DIR, FIGURE_3_DIR, FIGURE_SCHEME_DIR, FIGRUE_RP_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_baseline_params_json() -> dict[str, Any]:
    with BASE_PARAMS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_same_length_params() -> dict[str, Any]:
    params = solver.apply_param_overrides(
        solver.default_params(),
        load_baseline_params_json(),
        reset_lambda_D=False,
    )
    params = solver.apply_param_overrides(params, LENGTH_OVERRIDES, reset_lambda_D=False)
    validate_same_length_params(params)
    return params


def validate_same_length_params(params: dict[str, Any]) -> None:
    if not math.isclose(float(params["L_Au"]), L_AU_SAME, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"L_Au should be {L_AU_SAME} m, got {params['L_Au']}")
    if not math.isclose(float(params["L_Pd_len"]), L_PD_SAME, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"L_Pd_len should be {L_PD_SAME} m, got {params['L_Pd_len']}")
    if not math.isclose(float(params["L_gap"]), 10.0e-9, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"L_gap should remain 10 nm, got {params['L_gap']}")
    if not math.isclose(float(params["out_of_plane_width"]), OUT_OF_PLANE_WIDTH, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"out_of_plane_width should be {OUT_OF_PLANE_WIDTH} m, got {params['out_of_plane_width']}")


def compute_same_length_pair() -> dict[str, Any]:
    params = load_same_length_params()
    return solver.run_edl_comparison_pair(params, mode="FULL")


def summary_from_pair(pair: dict[str, Any]) -> dict[str, str]:
    res_edl = pair["with_edl"]
    res_no = pair["no_edl"]
    comparison = pair["comparison"]
    values: dict[str, Any] = {
        "pH": res_edl.get("pH"),
        "pH_ref": res_edl.get("pH_ref"),
        "delta_pH": res_edl.get("delta_pH"),
        "E1_eq_eff": res_edl.get("E1_eq_eff"),
        "E2_eq_eff": res_edl.get("E2_eq_eff"),
        "it0_1_eff": res_edl.get("it0_1_eff"),
        "it0_2_eff": res_edl.get("it0_2_eff"),
        "E_mix_with": res_edl.get("E_mix"),
        "i_mix_with": res_edl.get("i_mix"),
        "i_mix_norm_with": res_edl.get("i_mix"),
        "i_mix_phys_with": res_edl.get("i_mix_phys_A_per_m"),
        "i_mix_abs_with": res_edl.get("i_mix_abs_A"),
        "i_mix_avg_with": res_edl.get("i_mix_avg_A_per_m2"),
        "E_mix_no": res_no.get("E_mix"),
        "i_mix_no": res_no.get("i_mix"),
        "i_mix_norm_no": res_no.get("i_mix"),
        "i_mix_phys_no": res_no.get("i_mix_phys_A_per_m"),
        "i_mix_abs_no": res_no.get("i_mix_abs_A"),
        "i_mix_avg_no": res_no.get("i_mix_avg_A_per_m2"),
        "delta_E_mix": comparison.get("delta_E_mix"),
        "delta_i_mix_avg_A_per_m2": comparison.get("delta_i_mix_avg_A_per_m2"),
        "ratio_i_mix_avg": comparison.get("ratio_i_mix_avg"),
        "pct_i_mix_avg": comparison.get("pct_i_mix_avg"),
        "ratio_i_mix": comparison.get("ratio_i_mix"),
        "pct_i_mix": comparison.get("pct_i_mix"),
        "ratio_i_mix_phys": comparison.get("ratio_i_mix_phys"),
        "pct_i_mix_phys": comparison.get("pct_i_mix_phys"),
        "delta_i_mix_abs_A": comparison.get("delta_i_mix_abs_A"),
        "ratio_i_mix_abs": comparison.get("ratio_i_mix_abs"),
        "pct_i_mix_abs": comparison.get("pct_i_mix_abs"),
        "mode": comparison.get("mode", "FULL"),
    }
    return {key: "" if value is None else str(value) for key, value in values.items()}


def load_summary_for_scripts() -> dict[str, str]:
    return summary_from_pair(compute_same_length_pair())


def load_inputs_for_scripts() -> tuple[dict[str, Any], dict[str, str]]:
    params = load_same_length_params()
    summary = load_summary_for_scripts()
    save_traceability_inputs(params, summary)
    return params, summary


def save_traceability_inputs(params: dict[str, Any] | None = None, summary: dict[str, str] | None = None) -> None:
    ensure_output_dirs()
    params = copy.deepcopy(params) if params is not None else load_same_length_params()
    summary = copy.deepcopy(summary) if summary is not None else load_summary_for_scripts()

    with (INPUTS_DIR / f"params_{OUTPUT_TAG}.json").open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, sort_keys=True)
        f.write("\n")
    with (INPUTS_DIR / f"overrides_{OUTPUT_TAG}.json").open("w", encoding="utf-8") as f:
        json.dump(LENGTH_OVERRIDES, f, indent=2, sort_keys=True)
        f.write("\n")
    with (INPUTS_DIR / f"summary_compare_{OUTPUT_TAG}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    with (INPUTS_DIR / f"summary_compare_{OUTPUT_TAG}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")


def assert_expected_values(summary: dict[str, str]) -> None:
    expected = {
        "E_mix_with": 0.6127292791768433,
        "E_mix_no": 0.4557694000946099,
        "i_mix_avg_with": 0.060891938513790864,
        "i_mix_avg_no": 0.045129203350147216,
    }
    for key, expected_value in expected.items():
        actual = float(summary[key])
        if abs(actual - expected_value) > 5e-10:
            raise ValueError(f"{key}: expected {expected_value:.15g}, got {actual:.15g}")


def assert_output_counts() -> list[Path]:
    images = sorted(OUT_BASE.glob(f"**/*_{OUTPUT_TAG}.png")) + sorted(OUT_BASE.glob(f"**/*_{OUTPUT_TAG}.svg"))
    pdfs = sorted(OUT_BASE.glob("**/*.pdf"))
    if len(images) != EXPECTED_IMAGE_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_IMAGE_COUNT} PNG/SVG outputs, got {len(images)}")
    if pdfs:
        raise RuntimeError(f"Expected zero PDF outputs, found {len(pdfs)}")
    return images


def print_summary(summary: dict[str, str]) -> None:
    print(f"same-length tag = {OUTPUT_TAG}")
    print("L_Au = 25 nm")
    print("L_Pd_len = 25 nm")
    print("out_of_plane_width = 1 cm")
    print(f"E_mix_with = {float(summary['E_mix_with']):.15g} V")
    print(f"E_mix_no = {float(summary['E_mix_no']):.15g} V")
    print(f"i_mix_avg_with = {float(summary['i_mix_avg_with']):.15g} A/m^2")
    print(f"i_mix_avg_no = {float(summary['i_mix_avg_no']):.15g} A/m^2")
