from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


sys.dont_write_bytecode = True

OUT_DIR = Path(__file__).resolve().parent
LEGACY_DIR = OUT_DIR / "Au_C_Pd"
LEGACY_FIGURE_DIR = LEGACY_DIR / "Case_Figures"
LEGACY_FIGURE3_DIR = LEGACY_FIGURE_DIR / "Figure_3"
LEGACY_RP_DIR = LEGACY_FIGURE_DIR / "Figure_RP"
LEGACY_INPUTS_DIR = LEGACY_DIR / "inputs"
LEGACY_CSV_DIR = LEGACY_DIR / "csv"
LEGACY_OFAT_DIR = LEGACY_DIR / "Figure_L_support" / "OFAT"
SUPPORT_PZC_STUDY_DIR = LEGACY_DIR / "PZC_support_study"
INDEPENDENT_DIR = OUT_DIR / "Au_Pd_independent"
INDEPENDENT_CTOT_DIR = INDEPENDENT_DIR / "C_tot_study"
INDEPENDENT_POLARIZATION_DIR = INDEPENDENT_DIR / "figures" / "Polarization_Scheme"
INDEPENDENT_UNIFORM_BARS_DIR = (
    INDEPENDENT_DIR / "figures" / "Figure_3" / "Uniform_Bar_Comparison"
)

BASE_RESULT_ID = "20260528_111255"
OUTPUT_TAG = f"au2_pd2_{BASE_RESULT_ID}"
EXPECTED_LEGACY_FIGURE_TYPES = 28
EXPECTED_LEGACY_FIGURE3_TYPES = 14
EXPECTED_LEGACY_RP_TYPES = 14
EXPECTED_LEGACY_OFAT_TYPES = 8
EXPECTED_SUPPORT_PZC_STUDY_FIGURE_TYPES = 27
EXPECTED_INDEPENDENT_FIGURE_TYPES = 8
EXPECTED_INDEPENDENT_CTOT_FIGURE_TYPES = 6
EXPECTED_INDEPENDENT_POLARIZATION_FIGURE_TYPES = 1
EXPECTED_INDEPENDENT_UNIFORM_BAR_FIGURE_TYPES = 2

from legacy_au_c_pd_engine import LegacyCase, build_cases, run_convergence_checks  # noqa: E402
from legacy_au_c_pd_plots import plot_all_cases  # noqa: E402
from figure_l_support_ofat import build_figure_l_support_ofat  # noqa: E402
from make_support_pzc_study import build_or_reuse_support_pzc_study  # noqa: E402
from make_independent_au_pd import (  # noqa: E402
    EXPECTED_RESULTS,
    _source_hashes as independent_source_hashes,
    build_independent_au_pd,
)
from make_independent_au_pd_ctot_study import (  # noqa: E402
    build_or_reuse_independent_au_pd_ctot_study,
)
from make_independent_au_pd_polarization import (  # noqa: E402
    build_or_reuse_independent_au_pd_polarization,
)
from make_independent_au_pd_uniform_bar_comparison import (  # noqa: E402
    build_or_reuse_independent_au_pd_uniform_bar_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Au=Pd=2 nm two-model figure collection.")
    parser.add_argument("--skip-legacy", action="store_true", help="Do not rebuild the Au|C|Pd cases.")
    parser.add_argument(
        "--skip-support-pzc-study",
        action="store_true",
        help="Do not build the Au|C|Pd support-PZC parameter study.",
    )
    parser.add_argument("--skip-independent", action="store_true", help="Do not build the independent Au|Pd set.")
    parser.add_argument(
        "--skip-independent-ctot",
        action="store_true",
        help="Do not build the independent Au|Pd C_tot study.",
    )
    parser.add_argument(
        "--skip-independent-polarization",
        action="store_true",
        help="Do not build the independent Au|Pd polarization figure.",
    )
    parser.add_argument(
        "--skip-independent-uniform-bars",
        action="store_true",
        help="Do not build the independent Au|Pd uniform-interface bar figure.",
    )
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return path


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    materialized = [{str(key): _jsonable(value) for key, value in row.items()} for row in rows]
    if not materialized:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in materialized:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)
    return path


def _format_nm(value_nm: float) -> str:
    if math.isclose(value_nm, round(value_nm), rel_tol=0.0, abs_tol=1.0e-9):
        return str(int(round(value_nm)))
    return f"{value_nm:.3g}".replace(".", "p")


def case_tag(case: LegacyCase) -> str:
    return f"L_support_{_format_nm(float(case.value_nm))}nm_{OUTPUT_TAG}"


def _migrate_legacy_case_figure_layout() -> None:
    """Move only known generated case figures into the requested subfolders."""

    LEGACY_FIGURE3_DIR.mkdir(parents=True, exist_ok=True)
    LEGACY_RP_DIR.mkdir(parents=True, exist_ok=True)
    root_artifacts = sorted(
        path
        for path in LEGACY_FIGURE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".svg"}
    )
    for path in root_artifacts:
        if path.name.startswith("figure_3_panel_"):
            destination_dir = LEGACY_FIGURE3_DIR
        elif path.name.startswith(("solution_phase_potential_2d_", "surface_charge_distribution_")):
            destination_dir = LEGACY_RP_DIR
        else:
            raise RuntimeError(f"Unknown case-figure artifact; refusing to move it: {path}")
        destination = destination_dir / path.name
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite an existing migrated figure: {destination}")
        path.replace(destination)


def _validate_legacy_parameters(cases: list[LegacyCase]) -> None:
    """Fail closed if any production case drifts from the locked study design."""

    expected_support_nm = (0.0, 1.0, 2.0, 3.0, 10.0, 1000.0)
    actual_support_nm = tuple(float(case.value_nm) for case in cases)
    if actual_support_nm != expected_support_nm:
        raise ValueError(
            f"Legacy support cases are {actual_support_nm}, expected {expected_support_nm}"
        )

    locked = {
        "L_Au": 2.0e-9,
        "L_Pd_len": 2.0e-9,
        "Cdl_Au": 0.50,
        "Cdl_C": 0.20,
        "Cdl_Pd": 0.50,
        "it0_1": 1.852573885166257e-4,
        "it0_2": 1.852573885166257e-4,
        "alpha1": 0.5,
        "alpha2": 0.5,
        "out_of_plane_width": 0.01,
        "C_tot": 10.0,
    }
    for case in cases:
        expected_modes = 7680 if float(case.value_nm) == 1000.0 else 960
        for key, expected in locked.items():
            actual = float(case.params[key])
            tolerance = max(1.0e-15, abs(expected) * 1.0e-12)
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
                raise ValueError(
                    f"L_support={case.value_nm:g} nm has {key}={actual:.15g}; "
                    f"expected {expected:.15g}"
                )
        expected_gap = float(case.value_nm) * 1.0e-9
        if not math.isclose(
            float(case.params["L_gap"]),
            expected_gap,
            rel_tol=0.0,
            abs_tol=max(1.0e-15, abs(expected_gap) * 1.0e-12),
        ):
            raise ValueError(f"L_support={case.value_nm:g} nm has an inconsistent L_gap")
        if int(case.params["N_modes"]) != expected_modes:
            raise ValueError(
                f"L_support={case.value_nm:g} nm uses N_modes={case.params['N_modes']}; "
                f"expected {expected_modes}"
            )
        if int(getattr(case, "gl_order", -1)) != 128:
            raise ValueError(
                f"L_support={case.value_nm:g} nm uses GL order "
                f"{getattr(case, 'gl_order', None)}; expected 128"
            )
        for key in ("g_Au", "g_C", "g_Pd"):
            if case.params.get(key) is not None:
                raise ValueError(f"{key} must remain auto-derived from C_H, not overridden")


def _result_value(result: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in result and result[key] is not None:
            return float(result[key])
    return None


def _relative_balance(result: Mapping[str, Any]) -> float:
    i_au = float(result["I_Au"])
    i_pd = float(result["I_Pd"])
    return abs(i_au + i_pd) / (abs(i_au) + abs(i_pd) + 1.0e-300)


def summary_row(case: LegacyCase) -> dict[str, Any]:
    with_edl = case.res_edl
    without = case.res_no
    numerical = dict(case.convergence)
    return {
        "model_id": "legacy_Au_C_Pd_linear_PB_piecewise_Robin",
        "L_Au_nm": float(case.L_Au_nm),
        "L_support_nm": float(case.value_nm),
        "L_Pd_nm": float(case.L_Pd_nm),
        "L_total_nm": float(case.L_total_nm),
        "C_H_Au_uF_per_cm2": 100.0 * float(case.params["Cdl_Au"]),
        "C_H_C_uF_per_cm2": 100.0 * float(case.params["Cdl_C"]),
        "C_H_Pd_uF_per_cm2": 100.0 * float(case.params["Cdl_Pd"]),
        "N_modes": int(case.params["N_modes"]),
        "coefficient_grid_Nx": int(case.params["Nx"]),
        "gl_order": int(getattr(case, "gl_order", 128)),
        "root_xtol_V": float(case.params["xtol"]),
        "profile_grid_kind": str(numerical["profile_grid_kind"]),
        "profile_grid_points": int(numerical["profile_grid_points"]),
        "Ny_2d": int(numerical["Ny_2d"]),
        "lambda_D_nm": float(case.derived["lambda_D"]) * 1.0e9,
        "E_mix_with_V": _result_value(with_edl, "E_mix", "E_mix_V"),
        "E_mix_no_V": _result_value(without, "E_mix", "E_mix_V"),
        "i_mix_abs_with_A": _result_value(with_edl, "i_mix_abs_A"),
        "i_mix_abs_no_A": _result_value(without, "i_mix_abs_A"),
        "i_mix_avg_with_A_per_m2": _result_value(with_edl, "i_mix_avg_A_per_m2"),
        "i_mix_avg_no_A_per_m2": _result_value(without, "i_mix_avg_A_per_m2"),
        "relative_balance_with": _relative_balance(with_edl),
        "relative_balance_no": _relative_balance(without),
        "max_abs_phi_tilde_with": _result_value(with_edl, "max_abs_phi_tilde"),
        "debye_huckel_threshold": float(case.params.get("dh_warn_threshold", 1.0)),
        "debye_huckel_threshold_exceeded": bool(
            float(_result_value(with_edl, "max_abs_phi_tilde") or 0.0)
            > float(case.params.get("dh_warn_threshold", 1.0))
        ),
        "phi_2d_surface_max_error": float(case.phi_2d_surface_max_error),
    }


def _material_labels(case: LegacyCase) -> np.ndarray:
    labels = np.full(np.asarray(case.x_nm).shape, "support", dtype=object)
    labels[np.asarray(case.mask_Au, dtype=bool)] = "Au"
    labels[np.asarray(case.mask_Pd, dtype=bool)] = "Pd"
    return labels


def _sigma_on_profile_grid(case: LegacyCase) -> np.ndarray:
    x_nm = np.asarray(case.x_nm, dtype=float)
    sigma = np.full(x_nm.shape, np.nan, dtype=float)
    for segment in case.sigma_segments:
        sx = np.asarray(getattr(segment, "x_nm"), dtype=float)
        values = np.asarray(
            getattr(segment, "sigma_C_per_m2", getattr(segment, "sigma", np.full(sx.shape, np.nan))),
            dtype=float,
        )
        if sx.size == 0:
            continue
        lo = float(np.min(sx)) - 1.0e-10
        hi = float(np.max(sx)) + 1.0e-10
        mask = (x_nm >= lo) & (x_nm <= hi)
        sigma[mask] = np.interp(x_nm[mask], sx, values)
    return sigma


def profile_rows(case: LegacyCase) -> list[dict[str, Any]]:
    x_nm = np.asarray(case.x_nm, dtype=float)
    material = _material_labels(case)
    sigma = _sigma_on_profile_grid(case)
    arrays = {
        "phi_RP_with_V": np.asarray(case.phi_rp_with_V, dtype=float),
        "phi_RP_no_V": np.asarray(case.phi_rp_no_V, dtype=float),
        "i1_with_A_per_m2": np.asarray(case.i1_with, dtype=float),
        "i2_with_A_per_m2": np.asarray(case.i2_with, dtype=float),
        "i1_no_A_per_m2": np.asarray(case.i1_no, dtype=float),
        "i2_no_A_per_m2": np.asarray(case.i2_no, dtype=float),
        "c_R1_over_bulk": np.asarray(case.c_R1_norm, dtype=float),
        "c_O2_over_bulk": np.asarray(case.c_O2_norm, dtype=float),
    }
    for name, values in arrays.items():
        if values.shape != x_nm.shape:
            raise ValueError(f"{name} shape {values.shape} does not match x grid {x_nm.shape}")
    rows: list[dict[str, Any]] = []
    for index, x_value in enumerate(x_nm):
        row: dict[str, Any] = {
            "L_support_nm": float(case.value_nm),
            "x_nm": float(x_value),
            "material": str(material[index]),
            "sigma_C_per_m2": float(sigma[index]) if math.isfinite(float(sigma[index])) else None,
            "sigma_uC_per_cm2": 100.0 * float(sigma[index]) if math.isfinite(float(sigma[index])) else None,
        }
        row.update({name: float(values[index]) for name, values in arrays.items()})
        rows.append(row)
    return rows


def save_legacy_traceability(cases: list[LegacyCase], convergence: Any) -> list[Path]:
    for directory in (LEGACY_FIGURE_DIR, LEGACY_INPUTS_DIR, LEGACY_CSV_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    summaries: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    for case in cases:
        tag = case_tag(case)
        summary = summary_row(case)
        case_profiles = profile_rows(case)
        summaries.append(summary)
        profiles.extend(case_profiles)
        overrides = {
            "L_Au": 2.0e-9,
            "L_gap": float(case.value_nm) * 1.0e-9,
            "L_Pd_len": 2.0e-9,
            "Cdl_Au": 0.50,
            "Cdl_C": 0.20,
            "Cdl_Pd": 0.50,
            "g_Au": None,
            "g_C": None,
            "g_Pd": None,
            "it0_1": float(case.params["it0_1"]),
            "it0_2": float(case.params["it0_2"]),
            "alpha1": 0.5,
            "alpha2": 0.5,
            "out_of_plane_width": 0.01,
            "N_modes": int(case.params["N_modes"]),
            "Nx": int(case.params["Nx"]),
            "xtol": float(case.params["xtol"]),
        }
        saved.extend(
            (
                _write_json(LEGACY_INPUTS_DIR / f"params_{tag}.json", case.params),
                _write_json(LEGACY_INPUTS_DIR / f"overrides_{tag}.json", overrides),
                _write_csv(LEGACY_INPUTS_DIR / f"summary_compare_{tag}.csv", [summary]),
                _write_json(LEGACY_INPUTS_DIR / f"summary_compare_{tag}.json", summary),
                _write_csv(LEGACY_CSV_DIR / f"profile_{tag}.csv", case_profiles),
            )
        )
    saved.extend(
        (
            _write_csv(LEGACY_CSV_DIR / f"case_summary_{OUTPUT_TAG}.csv", summaries),
            _write_json(LEGACY_CSV_DIR / f"case_summary_{OUTPUT_TAG}.json", summaries),
            _write_csv(LEGACY_CSV_DIR / f"surface_profiles_{OUTPUT_TAG}.csv", profiles),
            _write_json(LEGACY_CSV_DIR / f"numerical_convergence_{OUTPUT_TAG}.json", convergence),
        )
    )
    return saved


def validate_legacy(
    cases: list[LegacyCase],
    figure_paths: list[Path],
    convergence: Any,
    ofat_result: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_legacy_parameters(cases)
    balances = [
        {
            "L_support_nm": float(case.value_nm),
            "with_edl": _relative_balance(case.res_edl),
            "without_edl": _relative_balance(case.res_no),
        }
        for case in cases
    ]
    if any(max(row["with_edl"], row["without_edl"]) >= 1.0e-10 for row in balances):
        raise RuntimeError(f"Mixed-current balance validation failed: {balances}")
    surface_errors = [float(case.phi_2d_surface_max_error) for case in cases]
    if any(error > 1.0e-9 for error in surface_errors):
        raise RuntimeError(f"2D y=0 surface reconstruction failed: {surface_errors}")
    root_pngs = sorted(LEGACY_FIGURE_DIR.glob("*.png"))
    root_svgs = sorted(LEGACY_FIGURE_DIR.glob("*.svg"))
    if root_pngs or root_svgs:
        raise RuntimeError(f"Case_Figures root must be empty after categorization: {root_pngs + root_svgs}")
    figure3_pngs = sorted(LEGACY_FIGURE3_DIR.glob("*.png"))
    figure3_svgs = sorted(LEGACY_FIGURE3_DIR.glob("*.svg"))
    rp_pngs = sorted(LEGACY_RP_DIR.glob("*.png"))
    rp_svgs = sorted(LEGACY_RP_DIR.glob("*.svg"))
    ofat_pngs = sorted(LEGACY_OFAT_DIR.glob("*.png"))
    ofat_svgs = sorted(LEGACY_OFAT_DIR.glob("*.svg"))
    pngs = sorted(LEGACY_FIGURE_DIR.rglob("*.png"))
    svgs = sorted(LEGACY_FIGURE_DIR.rglob("*.svg"))
    pdfs = sorted(LEGACY_DIR.glob("**/*.pdf"))
    if len(pngs) != EXPECTED_LEGACY_FIGURE_TYPES or len(svgs) != EXPECTED_LEGACY_FIGURE_TYPES:
        raise RuntimeError(
            f"Expected {EXPECTED_LEGACY_FIGURE_TYPES} legacy PNG/SVG pairs, "
            f"got {len(pngs)} PNG and {len(svgs)} SVG"
        )
    if len(figure3_pngs) != EXPECTED_LEGACY_FIGURE3_TYPES or len(figure3_svgs) != EXPECTED_LEGACY_FIGURE3_TYPES:
        raise RuntimeError(
            f"Expected {EXPECTED_LEGACY_FIGURE3_TYPES} Figure_3 PNG/SVG pairs, "
            f"got {len(figure3_pngs)} PNG and {len(figure3_svgs)} SVG"
        )
    if len(rp_pngs) != EXPECTED_LEGACY_RP_TYPES or len(rp_svgs) != EXPECTED_LEGACY_RP_TYPES:
        raise RuntimeError(
            f"Expected {EXPECTED_LEGACY_RP_TYPES} Figure_RP PNG/SVG pairs, "
            f"got {len(rp_pngs)} PNG and {len(rp_svgs)} SVG"
        )
    if len(ofat_pngs) != EXPECTED_LEGACY_OFAT_TYPES or len(ofat_svgs) != EXPECTED_LEGACY_OFAT_TYPES:
        raise RuntimeError(
            f"Expected {EXPECTED_LEGACY_OFAT_TYPES} Figure_L_support OFAT PNG/SVG pairs, "
            f"got {len(ofat_pngs)} PNG and {len(ofat_svgs)} SVG"
        )
    if not bool(ofat_result.get("validation", {}).get("passed")):
        raise RuntimeError("Figure_L_support OFAT validation did not pass")
    if any(not path.is_file() or path.stat().st_size == 0 for path in figure_paths):
        raise RuntimeError("One or more legacy figure outputs are missing or empty")
    if any(path.stat().st_size == 0 for path in (*pngs, *svgs)):
        raise RuntimeError("One or more legacy PNG/SVG outputs are empty")
    if not all("<text" in path.read_text(encoding="utf-8") for path in svgs):
        raise RuntimeError("One or more legacy SVG outputs do not retain editable text")
    if pdfs:
        raise RuntimeError(f"Unexpected legacy PDF files: {pdfs}")
    validation = {
        "model_id": "legacy_Au_C_Pd_linear_PB_piecewise_Robin",
        "mixed_current_balance": balances,
        "phi_2d_surface_max_errors": surface_errors,
        "numerical_convergence": convergence,
        "debye_huckel_warning_retained": True,
        "fourier_gibbs_warning_retained": True,
        "figure_counts": {
            "Figure_3": {"png": len(figure3_pngs), "svg": len(figure3_svgs)},
            "Figure_RP": {"png": len(rp_pngs), "svg": len(rp_svgs)},
            "Figure_L_support_OFAT": {"png": len(ofat_pngs), "svg": len(ofat_svgs)},
            "total_legacy": {
                "png": len(pngs) + len(ofat_pngs),
                "svg": len(svgs) + len(ofat_svgs),
                "pdf": len(pdfs),
            },
        },
        "passed": True,
    }
    _write_json(LEGACY_DIR / "validation.json", validation)
    return validation


def build_legacy_outputs() -> dict[str, Any]:
    cases = build_cases()
    _validate_legacy_parameters(cases)
    convergence = run_convergence_checks()
    save_legacy_traceability(cases, convergence)
    _migrate_legacy_case_figure_layout()
    figure_paths = plot_all_cases(
        cases,
        LEGACY_FIGURE_DIR,
        output_tag=f"_{OUTPUT_TAG}",
        active_zoom_support_nm=1000.0,
        active_window_nm=20.0,
    )
    ofat_result = build_figure_l_support_ofat(cases)
    all_legacy_figures = figure_paths + list(ofat_result["figure_paths"])
    validation = validate_legacy(cases, all_legacy_figures, convergence, ofat_result)
    return {
        "cases": cases,
        "figures": all_legacy_figures,
        "ofat": ofat_result,
        "validation": validation,
    }


def _validate_existing_independent() -> dict[str, Any]:
    params_path = INDEPENDENT_DIR / "params.json"
    summary_path = INDEPENDENT_DIR / "summary.json"
    if not params_path.is_file() or not summary_path.is_file():
        raise FileExistsError(
            f"Independent output directory exists but is incomplete; refusing to overwrite: {INDEPENDENT_DIR}"
        )
    with params_path.open("r", encoding="utf-8") as handle:
        params = json.load(handle)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    expected = {"L_Au": 2.0e-9, "L_Pd": 2.0e-9, "C_H_Au": 0.50, "C_H_Pd": 0.50}
    for key, value in expected.items():
        if not math.isclose(float(params[key]), value, rel_tol=0.0, abs_tol=max(1.0e-15, abs(value) * 1.0e-12)):
            raise ValueError(f"Existing independent output has {key}={params[key]}, expected {value}")
    baseline_figure_dirs = (
        INDEPENDENT_DIR / "figures" / "Figure_3",
        INDEPENDENT_DIR / "figures" / "rp_2d",
    )
    pngs = sorted(path for directory in baseline_figure_dirs for path in directory.glob("*.png"))
    svgs = sorted(path for directory in baseline_figure_dirs for path in directory.glob("*.svg"))
    pdfs = sorted(INDEPENDENT_DIR.glob("**/*.pdf"))
    if len(pngs) != EXPECTED_INDEPENDENT_FIGURE_TYPES or len(svgs) != EXPECTED_INDEPENDENT_FIGURE_TYPES:
        raise RuntimeError(f"Existing independent output is incomplete: {len(pngs)} PNG, {len(svgs)} SVG")
    if pdfs:
        raise RuntimeError(f"Existing independent output contains unexpected PDF files: {pdfs}")
    if not all("<text" in path.read_text(encoding="utf-8") for path in svgs):
        raise RuntimeError("Existing independent SVG output does not retain editable text")
    for condition, fields in EXPECTED_RESULTS.items():
        result = summary[condition]
        for field, (expected_value, tolerance) in fields.items():
            actual = float(result[field])
            if abs(actual - expected_value) > tolerance:
                raise RuntimeError(
                    f"Existing independent {condition}.{field}={actual:.15g}, "
                    f"expected {expected_value:.15g} +/- {tolerance:.3g}"
                )
    validation_path = INDEPENDENT_DIR / "validation.json"
    if not validation_path.is_file():
        raise RuntimeError("Existing independent output is missing validation.json")
    with validation_path.open("r", encoding="utf-8") as handle:
        validation = json.load(handle)
    if not bool(validation.get("passed")):
        raise RuntimeError("Existing independent validation.json is not passed")
    checksum_path = INDEPENDENT_DIR / "checksums.sha256"
    if not checksum_path.is_file():
        raise RuntimeError("Existing independent output is missing checksums.sha256")
    checksum_lines = checksum_path.read_text(encoding="utf-8").splitlines()
    if not checksum_lines:
        raise RuntimeError("Existing independent checksums.sha256 is empty")
    listed_paths: set[str] = set()
    for line in checksum_lines:
        expected_hash, relative = line.split("  ", 1)
        if relative in listed_paths:
            raise RuntimeError(f"Existing independent checksum path is duplicated: {relative}")
        listed_paths.add(relative)
        target = INDEPENDENT_DIR / relative
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(f"Existing independent artifact is missing or empty: {relative}")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != expected_hash:
            raise RuntimeError(f"Existing independent checksum mismatch: {relative}")
    actual_paths = {
        path.relative_to(INDEPENDENT_DIR).as_posix()
        for path in INDEPENDENT_DIR.rglob("*")
        if path.is_file()
        and path.name != checksum_path.name
        and path.name != ".DS_Store"
        and INDEPENDENT_CTOT_DIR not in path.parents
        and INDEPENDENT_POLARIZATION_DIR not in path.parents
        and INDEPENDENT_UNIFORM_BARS_DIR not in path.parents
    }
    if listed_paths != actual_paths:
        missing = sorted(actual_paths - listed_paths)
        stale = sorted(listed_paths - actual_paths)
        raise RuntimeError(
            f"Existing independent checksum coverage mismatch; missing={missing}, stale={stale}"
        )
    manifest_path = INDEPENDENT_DIR / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Existing independent output is missing run_manifest.json")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("source_sha256") != independent_source_hashes():
        raise RuntimeError("Existing independent source hashes do not match the current model code")
    return {
        "reused": True,
        "validation": validation,
        "figure_counts": {"png": len(pngs), "svg": len(svgs), "pdf": len(pdfs)},
    }


def build_or_reuse_independent() -> dict[str, Any]:
    if INDEPENDENT_DIR.exists():
        return _validate_existing_independent()
    return build_independent_au_pd(INDEPENDENT_DIR)


def validate_total_outputs() -> dict[str, int]:
    pngs = sorted(OUT_DIR.glob("**/*.png"))
    svgs = sorted(OUT_DIR.glob("**/*.svg"))
    pdfs = sorted(OUT_DIR.glob("**/*.pdf"))
    expected = (
        EXPECTED_LEGACY_FIGURE_TYPES
        + EXPECTED_LEGACY_OFAT_TYPES
        + EXPECTED_SUPPORT_PZC_STUDY_FIGURE_TYPES
        + EXPECTED_INDEPENDENT_FIGURE_TYPES
        + EXPECTED_INDEPENDENT_CTOT_FIGURE_TYPES
        + EXPECTED_INDEPENDENT_POLARIZATION_FIGURE_TYPES
        + EXPECTED_INDEPENDENT_UNIFORM_BAR_FIGURE_TYPES
    )
    if len(pngs) != expected or len(svgs) != expected:
        raise RuntimeError(f"Expected {expected} PNG/SVG pairs, got {len(pngs)} PNG and {len(svgs)} SVG")
    if pdfs:
        raise RuntimeError(f"Expected zero PDF outputs, found {pdfs}")
    if any(path.stat().st_size == 0 for path in (*pngs, *svgs)):
        raise RuntimeError(f"One or more of the {2 * expected} figure artifacts are empty")
    if not all("<text" in path.read_text(encoding="utf-8") for path in svgs):
        raise RuntimeError("One or more SVG outputs do not retain editable text")
    return {"png": len(pngs), "svg": len(svgs), "pdf": len(pdfs)}


def main() -> int:
    args = parse_args()
    legacy_result = None if args.skip_legacy else build_legacy_outputs()
    support_pzc_study_result = (
        None
        if args.skip_support_pzc_study
        else build_or_reuse_support_pzc_study(SUPPORT_PZC_STUDY_DIR)
    )
    independent_result = None if args.skip_independent else build_or_reuse_independent()
    independent_ctot_result = (
        None
        if args.skip_independent or args.skip_independent_ctot
        else build_or_reuse_independent_au_pd_ctot_study(INDEPENDENT_CTOT_DIR)
    )
    independent_polarization_result = (
        None
        if args.skip_independent or args.skip_independent_polarization
        else build_or_reuse_independent_au_pd_polarization(INDEPENDENT_POLARIZATION_DIR)
    )
    independent_uniform_bars_result = (
        None
        if args.skip_independent or args.skip_independent_uniform_bars
        else build_or_reuse_independent_au_pd_uniform_bar_comparison(
            INDEPENDENT_UNIFORM_BARS_DIR
        )
    )
    counts = None
    if (
        not args.skip_legacy
        and not args.skip_support_pzc_study
        and not args.skip_independent
        and not args.skip_independent_ctot
        and not args.skip_independent_polarization
        and not args.skip_independent_uniform_bars
    ):
        counts = validate_total_outputs()
    print(f"Output directory: {OUT_DIR}")
    if legacy_result is not None:
        for case in legacy_result["cases"]:
            row = summary_row(case)
            print(
                f"Au|C|Pd L_support={row['L_support_nm']:g} nm: "
                f"E_mix={row['E_mix_with_V']:.12f} V, "
                f"i_mix_avg={row['i_mix_avg_with_A_per_m2']:.12g} A/m^2, "
                f"max|phi_tilde|={row['max_abs_phi_tilde_with']:.6g}"
            )
        ofat_validation = legacy_result["ofat"]["validation"]
        print(
            "Figure_L_support OFAT: "
            f"{ofat_validation['main_scan_points']} main points, "
            f"{ofat_validation['overlap_scan_points']} overlap points, "
            f"{ofat_validation['png_count']} PNG + {ofat_validation['svg_count']} SVG"
        )
    if support_pzc_study_result is not None:
        print("Au|C|Pd support-PZC parameter study validated")
    if independent_result is not None:
        print("Independent Au|Pd output validated")
    if independent_ctot_result is not None:
        print("Independent Au|Pd C_tot study validated")
    if independent_polarization_result is not None:
        print("Independent Au|Pd polarization figure validated")
    if independent_uniform_bars_result is not None:
        print("Independent Au|Pd uniform-interface bar figure validated")
    if counts is not None:
        print(f"Verified {counts['png']} PNG + {counts['svg']} SVG + {counts['pdf']} PDF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
