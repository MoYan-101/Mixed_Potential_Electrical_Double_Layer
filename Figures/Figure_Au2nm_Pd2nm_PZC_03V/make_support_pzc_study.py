"""Support-PZC study for the Au=Pd=2 nm Au|C|Pd legacy model.

The main scan evaluates four finite support lengths at 41 support-PZC values.
Three 1000 nm cases act as far-field anchors and two zero-length cases are a
negative control.  All mixed potentials use the FULL absolute-current balance
and separate 128-point Gauss--Legendre integration on Au and Pd.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator
from PIL import Image


sys.dont_write_bytecode = True

PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR.parents[1]
DEFAULT_OUTPUT = PROJECT_DIR / "Au_C_Pd" / "PZC_support_study"
REFERENCE_OFAT_CSV = (
    PROJECT_DIR
    / "Au_C_Pd"
    / "Figure_L_support"
    / "csv"
    / f"edl_overlap_vs_1000nm_L_support_au2_pd2_pzc03V_20260528_111255.csv"
)
CHECKSUM_FILE = "checksums.sha256"
STUDY_ID = "Au2nm_C_Pd2nm_PZC03V_support_PZC_study"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import legacy_au_c_pd_engine as engine  # noqa: E402


MAIN_LENGTHS_NM = (1.0, 3.0, 6.0, 15.0)
PZC_VALUES_V = tuple(round(float(value), 12) for value in np.linspace(0.10, 0.90, 41))
REPRESENTATIVE_PZC_V = (0.10, 0.50, 0.90)
ANCHOR_LENGTH_NM = 1000.0
NEGATIVE_CONTROL_LENGTH_NM = 0.0

REQUESTED_SHORT_HIGH_MODES = 960
REQUESTED_SHORT_LOW_MODES = 480
REQUESTED_LONG_HIGH_MODES = 7680
REQUESTED_LONG_LOW_MODES = 5760
PRODUCTION_MODES_BY_LENGTH = {
    1.0: 960,
    3.0: 960,
    6.0: 1920,
    15.0: 3840,
    1000.0: 11520,
}
CONVERGENCE_LOW_MODES_BY_LENGTH = {
    1.0: 480,
    3.0: 480,
    6.0: 960,
    15.0: 1920,
    1000.0: 9600,
}
HIGH_GL_ORDER = 128
LOW_GL_ORDER = 64
BALANCE_TOL = 1.0e-10

LENGTH_COLORS = {
    1.0: "#4D4D4D",
    3.0: "#F26B38",
    6.0: "#009E73",
    15.0: "#5A90C8",
}
PROFILE_STYLES = {
    0.10: {"color": "#767676", "linestyle": (0, (1.2, 2.0)), "label": "0.10 V"},
    0.50: {"color": "#272727", "linestyle": "solid", "label": "0.50 V"},
    0.90: {"color": "#3775BA", "linestyle": (0, (5.0, 2.5)), "label": "0.90 V"},
}
MATERIAL_COLORS = {"Au": "#E4C133", "C": "#8C8C8C", "Pd": "#5A90C8"}
WITHOUT_EDL_COLOR = "#12355B"
ANCHOR_COLOR = "#8C8C8C"
DARK = "#272727"

FIGURE_STEMS = (
    "support_pzc_trends_au2_pd2",
    "support_pzc_phi_RP_profiles_au2_pd2",
    "support_pzc_surface_charge_profiles_au2_pd2",
)
PZC_2D_OUTPUT_SUBDIR = Path("Figure_RP")


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": [
                "Helvetica",
                "Nimbus Sans",
                "Arial",
                "DejaVu Sans",
                "sans-serif",
            ],
            "font.size": 9.4,
            "axes.titlesize": 10.2,
            "axes.labelsize": 10.0,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "svg.fonttype": "none",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Nimbus Sans",
            "mathtext.it": "Nimbus Sans:italic",
            "mathtext.bf": "Nimbus Sans:bold",
            "mathtext.cal": "Nimbus Sans",
            "mathtext.sf": "Nimbus Sans",
            "mathtext.tt": "Nimbus Sans",
            "savefig.facecolor": "white",
        }
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _jsonable(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            if list(row) != columns:
                raise ValueError(f"Inconsistent columns in {path}")
            writer.writerow(_jsonable(row))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    solver_path = ROOT / "Mixed_Potential_Electrical_Double_Layer" / "Solve_Emix_updating.py"
    sources = {
        "study_generator": Path(__file__).resolve(),
        "legacy_engine": PROJECT_DIR / "legacy_au_c_pd_engine.py",
        "linear_solver": solver_path,
        "baseline_params": engine.BASE_PARAMS_PATH,
        "support_length_reference_csv": REFERENCE_OFAT_CSV,
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing traceability source(s): {missing}")
    return {name: _sha256(path) for name, path in sources.items()}


def _modes_for_length(table: Mapping[float, int], length_nm: float) -> int:
    matches = [
        modes
        for configured_length, modes in table.items()
        if math.isclose(
            float(length_nm), configured_length, rel_tol=0.0, abs_tol=1.0e-9
        )
    ]
    if len(matches) != 1:
        raise ValueError(f"No unique mode setting for L_support={length_nm:g} nm")
    return int(matches[0])


def _support_segment(case: Any) -> Any:
    segment = next(
        (item for item in case.sigma_segments if str(item.material).lower() == "support"),
        None,
    )
    if segment is None:
        raise RuntimeError(
            f"Missing C/support charge segment at L={case.value_nm:g} nm"
        )
    return segment


def _signed_support_sigma(case: Any) -> float:
    segment = _support_segment(case)
    x_nm = np.asarray(segment.x_nm, dtype=float)
    sigma = np.asarray(segment.sigma_C_per_m2, dtype=float)
    if x_nm.size < 2 or sigma.shape != x_nm.shape:
        raise RuntimeError("Invalid support surface-charge profile")
    return float(np.trapezoid(sigma, x_nm) / float(case.L_support_nm))


def _case_metrics(case: Any) -> dict[str, Any]:
    result = case.res_edl
    no_edl = case.res_no
    return {
        "L_support_nm": float(case.L_support_nm),
        "pzc_support_V": float(case.params["pzc_C"]),
        "N_modes": int(case.params["N_modes"]),
        "gl_order": int(case.gl_order),
        "lambda_D_nm": 1.0e9 * float(case.derived["lambda_D"]),
        "E_mix_with_V": float(result["E_mix"]),
        "E_mix_no_V": float(no_edl["E_mix"]),
        "i_mix_avg_with_A_per_m2": float(result["i_mix_avg_A_per_m2"]),
        "i_mix_avg_no_A_per_m2": float(no_edl["i_mix_avg_A_per_m2"]),
        "i_mix_abs_with_A": float(result["i_mix_abs_A"]),
        "i_mix_abs_no_A": float(no_edl["i_mix_abs_A"]),
        "phi_RP_Au_mean_mV": 1.0e3 * float(result["phi2_1_meanV"]),
        "phi_RP_Pd_mean_mV": 1.0e3 * float(result["phi2_2_meanV"]),
        "sigma_support_signed_mean_C_per_m2": _signed_support_sigma(case),
        "sigma_support_signed_mean_uC_per_cm2": 100.0 * _signed_support_sigma(case),
        "relative_balance_with": float(result["relative_balance_residual"]),
        "relative_balance_no": float(no_edl["relative_balance_residual"]),
        "max_abs_phi_tilde_with_edl": float(result["max_abs_phi_tilde"]),
        "debye_huckel_ok_with_edl": bool(result["debye_huckel_ok"]),
    }


def _add_overlap(row: dict[str, Any], reference: Mapping[str, Any]) -> None:
    delta_au = float(row["phi_RP_Au_mean_mV"]) - float(reference["phi_RP_Au_mean_mV"])
    delta_pd = float(row["phi_RP_Pd_mean_mV"]) - float(reference["phi_RP_Pd_mean_mV"])
    # The active lengths are both 2 nm, so this is the equal-active-length RMS.
    row["overlap_reference_L_support_nm"] = 15.0
    row["overlap_delta_phi_RP_Au_mean_vs_15nm_mV"] = delta_au
    row["overlap_delta_phi_RP_Pd_mean_vs_15nm_mV"] = delta_pd
    row["overlap_active_RP_RMS_vs_15nm_mV"] = math.sqrt(
        0.5 * (delta_au**2 + delta_pd**2)
    )


def _build_main_scan() -> tuple[
    list[dict[str, Any]],
    dict[tuple[float, float], Any],
    dict[tuple[float, float], Any],
]:
    rows: list[dict[str, Any]] = []
    representative_cases: dict[tuple[float, float], Any] = {}
    representative_low_gl_cases: dict[tuple[float, float], Any] = {}
    metrics: dict[tuple[float, float], dict[str, Any]] = {}
    for length_nm in MAIN_LENGTHS_NM:
        family = engine.build_support_pzc_scan_family(
            length_nm,
            PZC_VALUES_V,
            n_modes=_modes_for_length(PRODUCTION_MODES_BY_LENGTH, length_nm),
            gl_orders=(LOW_GL_ORDER, HIGH_GL_ORDER),
            include_2d=False,
        )
        for pzc_v in PZC_VALUES_V:
            case = family[(pzc_v, HIGH_GL_ORDER)]
            metrics[(length_nm, pzc_v)] = _case_metrics(case)
            if pzc_v in REPRESENTATIVE_PZC_V:
                representative_cases[(length_nm, pzc_v)] = case
                representative_low_gl_cases[(length_nm, pzc_v)] = family[
                    (pzc_v, LOW_GL_ORDER)
                ]
        del family
    for pzc_v in PZC_VALUES_V:
        reference = metrics[(15.0, pzc_v)]
        for length_nm in MAIN_LENGTHS_NM:
            row = metrics[(length_nm, pzc_v)]
            _add_overlap(row, reference)
            rows.append(row)
    return rows, representative_cases, representative_low_gl_cases


def _build_representative_2d_cases() -> dict[tuple[float, float], Any]:
    """Rebuild only the 12 representative points with their 2-D fields."""

    cases: dict[tuple[float, float], Any] = {}
    for length_nm in MAIN_LENGTHS_NM:
        family = engine.build_support_pzc_scan_family(
            length_nm,
            REPRESENTATIVE_PZC_V,
            n_modes=_modes_for_length(PRODUCTION_MODES_BY_LENGTH, length_nm),
            gl_orders=(HIGH_GL_ORDER,),
            include_2d=True,
        )
        for pzc_v in REPRESENTATIVE_PZC_V:
            cases[(length_nm, pzc_v)] = family[(pzc_v, HIGH_GL_ORDER)]
    return cases


def _build_anchors(
    representatives: Mapping[tuple[float, float], Any],
) -> tuple[list[dict[str, Any]], dict[float, Any], dict[float, Any]]:
    rows: list[dict[str, Any]] = []
    cases: dict[float, Any] = {}
    low_gl_cases: dict[float, Any] = {}
    family = engine.build_support_pzc_scan_family(
        ANCHOR_LENGTH_NM,
        REPRESENTATIVE_PZC_V,
        n_modes=_modes_for_length(PRODUCTION_MODES_BY_LENGTH, ANCHOR_LENGTH_NM),
        gl_orders=(LOW_GL_ORDER, HIGH_GL_ORDER),
        include_2d=False,
    )
    for pzc_v in REPRESENTATIVE_PZC_V:
        case = family[(pzc_v, HIGH_GL_ORDER)]
        row = _case_metrics(case)
        reference = _case_metrics(representatives[(15.0, pzc_v)])
        _add_overlap(row, reference)
        rows.append(row)
        cases[pzc_v] = case
        low_gl_cases[pzc_v] = family[(pzc_v, LOW_GL_ORDER)]
    return rows, cases, low_gl_cases


def _build_negative_controls() -> tuple[list[dict[str, Any]], dict[float, Any]]:
    rows: list[dict[str, Any]] = []
    cases: dict[float, Any] = {}
    family = engine.build_support_pzc_scan_family(
        NEGATIVE_CONTROL_LENGTH_NM,
        (0.10, 0.90),
        n_modes=REQUESTED_SHORT_HIGH_MODES,
        gl_orders=(HIGH_GL_ORDER,),
        include_2d=False,
    )
    for pzc_v in (0.10, 0.90):
        case = family[(pzc_v, HIGH_GL_ORDER)]
        result = case.res_edl
        no_edl = case.res_no
        rows.append(
            {
                "L_support_nm": 0.0,
                "pzc_support_V": pzc_v,
                "E_mix_with_V": float(result["E_mix"]),
                "i_mix_avg_with_A_per_m2": float(result["i_mix_avg_A_per_m2"]),
                "phi_RP_Au_mean_mV": 1.0e3 * float(result["phi2_1_meanV"]),
                "phi_RP_Pd_mean_mV": 1.0e3 * float(result["phi2_2_meanV"]),
                "relative_balance_with": float(result["relative_balance_residual"]),
                "E_mix_no_V": float(no_edl["E_mix"]),
                "i_mix_avg_no_A_per_m2": float(no_edl["i_mix_avg_A_per_m2"]),
                "relative_balance_no": float(no_edl["relative_balance_residual"]),
                "max_abs_phi_tilde_with_edl": float(result["max_abs_phi_tilde"]),
            }
        )
        cases[pzc_v] = case
    return rows, cases


def _relative_percent(a: float, b: float) -> float:
    return 100.0 * abs(float(a) - float(b)) / max(abs(float(b)), 1.0e-300)


def _convergence_row(
    production: Any,
    low_mode: Any,
    low_gl: Any,
    *,
    length_nm: float,
    pzc_v: float,
) -> dict[str, Any]:
    is_long = math.isclose(length_nm, ANCHOR_LENGTH_NM, rel_tol=0.0, abs_tol=1.0e-9)
    low_modes = _modes_for_length(CONVERGENCE_LOW_MODES_BY_LENGTH, length_nm)
    high_modes = _modes_for_length(PRODUCTION_MODES_BY_LENGTH, length_nm)
    e_tolerance_mv = 0.03 if is_long else 0.002
    i_tolerance_percent = 0.1 if is_long else 0.005
    high = production.res_edl
    mode_low = low_mode.res_edl
    gl_low = low_gl.res_edl
    mode_delta_e = 1.0e3 * abs(float(high["E_mix"]) - float(mode_low["E_mix"]))
    mode_delta_i = _relative_percent(
        float(high["i_mix_avg_A_per_m2"]),
        float(mode_low["i_mix_avg_A_per_m2"]),
    )
    gl_delta_e = 1.0e3 * abs(float(high["E_mix"]) - float(gl_low["E_mix"]))
    gl_delta_i = _relative_percent(
        float(high["i_mix_avg_A_per_m2"]),
        float(gl_low["i_mix_avg_A_per_m2"]),
    )
    balance = float(high["relative_balance_residual"])
    low_mode_balance = float(mode_low["relative_balance_residual"])
    low_gl_balance = float(gl_low["relative_balance_residual"])
    passed = bool(
        mode_delta_e < e_tolerance_mv
        and mode_delta_i < i_tolerance_percent
        and gl_delta_e < e_tolerance_mv
        and gl_delta_i < i_tolerance_percent
        and balance < BALANCE_TOL
        and low_mode_balance < BALANCE_TOL
        and low_gl_balance < BALANCE_TOL
    )
    return {
        "L_support_nm": length_nm,
        "pzc_support_V": pzc_v,
        "low_N_modes": low_modes,
        "high_N_modes": high_modes,
        "low_gl_order": LOW_GL_ORDER,
        "high_gl_order": HIGH_GL_ORDER,
        "E_mix_low_modes_V": float(mode_low["E_mix"]),
        "E_mix_high_modes_V": float(high["E_mix"]),
        "i_mix_avg_low_modes_A_per_m2": float(mode_low["i_mix_avg_A_per_m2"]),
        "i_mix_avg_high_modes_A_per_m2": float(high["i_mix_avg_A_per_m2"]),
        "mode_delta_E_mV": mode_delta_e,
        "mode_delta_i_percent": mode_delta_i,
        "E_mix_gl64_V": float(gl_low["E_mix"]),
        "E_mix_gl128_V": float(high["E_mix"]),
        "i_mix_avg_gl64_A_per_m2": float(gl_low["i_mix_avg_A_per_m2"]),
        "i_mix_avg_gl128_A_per_m2": float(high["i_mix_avg_A_per_m2"]),
        "quadrature_delta_E_mV": gl_delta_e,
        "quadrature_delta_i_percent": gl_delta_i,
        "relative_balance_high": balance,
        "relative_balance_low_modes": low_mode_balance,
        "relative_balance_gl64": low_gl_balance,
        "E_tolerance_mV": e_tolerance_mv,
        "i_tolerance_percent": i_tolerance_percent,
        "passed": passed,
    }


def _run_convergence(
    representatives: Mapping[tuple[float, float], Any],
    representative_low_gl_cases: Mapping[tuple[float, float], Any],
    anchors: Mapping[float, Any],
    anchor_low_gl_cases: Mapping[float, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for length_nm in MAIN_LENGTHS_NM + (ANCHOR_LENGTH_NM,):
        is_long = length_nm == ANCHOR_LENGTH_NM
        low_modes = _modes_for_length(CONVERGENCE_LOW_MODES_BY_LENGTH, length_nm)
        low_mode_family = engine.build_support_pzc_scan_family(
            length_nm,
            REPRESENTATIVE_PZC_V,
            n_modes=low_modes,
            gl_orders=(HIGH_GL_ORDER,),
            include_2d=False,
        )
        for pzc_v in REPRESENTATIVE_PZC_V:
            production = (
                anchors[pzc_v]
                if is_long
                else representatives[(length_nm, pzc_v)]
            )
            low_gl = (
                anchor_low_gl_cases[pzc_v]
                if is_long
                else representative_low_gl_cases[(length_nm, pzc_v)]
            )
            rows.append(
                _convergence_row(
                    production,
                    low_mode_family[(pzc_v, HIGH_GL_ORDER)],
                    low_gl,
                    length_nm=length_nm,
                    pzc_v=pzc_v,
                )
            )
    failures = [row for row in rows if not bool(row["passed"])]
    if failures:
        detail = "; ".join(
            f"L={row['L_support_nm']:g} nm, PZC={row['pzc_support_V']:.2f} V: "
            f"mode dE={row['mode_delta_E_mV']:.4g} mV, "
            f"mode di={row['mode_delta_i_percent']:.4g}%, "
            f"GL dE={row['quadrature_delta_E_mV']:.4g} mV, "
            f"GL di={row['quadrature_delta_i_percent']:.4g}%"
            for row in failures
        )
        raise RuntimeError(f"Support-PZC convergence thresholds failed: {detail}")
    return rows


def _requested_resolution_audit_row(
    low_case: Any,
    high_case: Any,
    *,
    length_nm: float,
    pzc_v: float,
    low_modes: int,
    high_modes: int,
) -> dict[str, Any]:
    is_long = math.isclose(
        length_nm, ANCHOR_LENGTH_NM, rel_tol=0.0, abs_tol=1.0e-9
    )
    e_tolerance_mv = 0.03 if is_long else 0.002
    i_tolerance_percent = 0.1 if is_long else 0.005
    low = low_case.res_edl
    high = high_case.res_edl
    delta_e_mv = 1.0e3 * abs(float(high["E_mix"]) - float(low["E_mix"]))
    delta_i_percent = _relative_percent(
        float(high["i_mix_avg_A_per_m2"]),
        float(low["i_mix_avg_A_per_m2"]),
    )
    low_balance = float(low["relative_balance_residual"])
    high_balance = float(high["relative_balance_residual"])
    passed = bool(
        delta_e_mv < e_tolerance_mv
        and delta_i_percent < i_tolerance_percent
        and low_balance < BALANCE_TOL
        and high_balance < BALANCE_TOL
    )
    return {
        "check_kind": "original_requested_resolution_audit",
        "L_support_nm": float(length_nm),
        "pzc_support_V": float(pzc_v),
        "low_N_modes": int(low_modes),
        "high_N_modes": int(high_modes),
        "E_mix_low_modes_V": float(low["E_mix"]),
        "E_mix_high_modes_V": float(high["E_mix"]),
        "i_mix_avg_low_modes_A_per_m2": float(low["i_mix_avg_A_per_m2"]),
        "i_mix_avg_high_modes_A_per_m2": float(high["i_mix_avg_A_per_m2"]),
        "mode_delta_E_mV": delta_e_mv,
        "mode_delta_i_percent": delta_i_percent,
        "relative_balance_low_modes": low_balance,
        "relative_balance_high_modes": high_balance,
        "E_tolerance_mV": e_tolerance_mv,
        "i_tolerance_percent": i_tolerance_percent,
        "passed": passed,
    }


def _run_requested_resolution_audit() -> tuple[
    list[dict[str, Any]],
    dict[tuple[float, float], Any],
]:
    """Evaluate the mode pairs specified in the original plan without hiding failures."""

    rows: list[dict[str, Any]] = []
    short_high_cases: dict[tuple[float, float], Any] = {}
    configurations = (
        *(
            (
                length_nm,
                REQUESTED_SHORT_LOW_MODES,
                REQUESTED_SHORT_HIGH_MODES,
            )
            for length_nm in MAIN_LENGTHS_NM
        ),
        (
            ANCHOR_LENGTH_NM,
            REQUESTED_LONG_LOW_MODES,
            REQUESTED_LONG_HIGH_MODES,
        ),
    )
    for length_nm, low_modes, high_modes in configurations:
        low_family = engine.build_support_pzc_scan_family(
            length_nm,
            REPRESENTATIVE_PZC_V,
            n_modes=low_modes,
            gl_orders=(HIGH_GL_ORDER,),
            include_2d=False,
        )
        high_family = engine.build_support_pzc_scan_family(
            length_nm,
            REPRESENTATIVE_PZC_V,
            n_modes=high_modes,
            gl_orders=(HIGH_GL_ORDER,),
            include_2d=False,
        )
        for pzc_v in REPRESENTATIVE_PZC_V:
            high_case = high_family[(pzc_v, HIGH_GL_ORDER)]
            rows.append(
                _requested_resolution_audit_row(
                    low_family[(pzc_v, HIGH_GL_ORDER)],
                    high_case,
                    length_nm=length_nm,
                    pzc_v=pzc_v,
                    low_modes=low_modes,
                    high_modes=high_modes,
                )
            )
            if length_nm in MAIN_LENGTHS_NM:
                short_high_cases[(length_nm, pzc_v)] = high_case
    return rows, short_high_cases


def _profile_stem(length_nm: float, pzc_v: float) -> str:
    pzc_tag = f"{pzc_v:.2f}".replace(".", "p")
    return f"profile_L_support_{length_nm:g}nm_pzc_C_{pzc_tag}V"


def _solution_potential_2d_stem(length_nm: float, pzc_v: float) -> str:
    pzc_tag = f"{pzc_v:.2f}".replace(".", "p")
    return (
        "solution_phase_potential_2d_"
        f"L_support_{length_nm:g}nm_pzc_support_{pzc_tag}V_au2_pd2"
    )


def _phi_s_reactants_2d_stem(length_nm: float, pzc_v: float) -> str:
    pzc_tag = f"{pzc_v:.2f}".replace(".", "p")
    return (
        "phi_s_reactants_2d_"
        f"L_support_{length_nm:g}nm_pzc_support_{pzc_tag}V_au2_pd2"
    )


def _profile_rows(case: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    material_map = {"Au": "Au", "support": "C", "Pd": "Pd"}
    for segment in case.sigma_segments:
        material = material_map[str(segment.material)]
        x_nm = np.asarray(segment.x_nm, dtype=float)
        phi_mv = 1.0e3 * np.asarray(segment.phi_rp_V, dtype=float)
        sigma = np.asarray(segment.sigma_uC_per_cm2, dtype=float)
        for x_value, phi_value, sigma_value in zip(x_nm, phi_mv, sigma, strict=True):
            rows.append(
                {
                    "L_support_nm": float(case.L_support_nm),
                    "pzc_support_V": float(case.params["pzc_C"]),
                    "material": material,
                    "x_nm": float(x_value),
                    "phi_RP_mV": float(phi_value),
                    "sigma_uC_per_cm2": float(sigma_value),
                }
            )
    return rows


def _write_representative_profiles(
    output: Path,
    representatives: Mapping[tuple[float, float], Any],
) -> list[str]:
    relative_paths: list[str] = []
    for length_nm in MAIN_LENGTHS_NM:
        for pzc_v in REPRESENTATIVE_PZC_V:
            relative = f"csv/profiles/{_profile_stem(length_nm, pzc_v)}.csv"
            _write_csv(output / relative, _profile_rows(representatives[(length_nm, pzc_v)]))
            relative_paths.append(relative)
    return relative_paths


def _save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> list[Path]:
    paths = [figure_dir / f"{stem}.png", figure_dir / f"{stem}.svg"]
    fig.savefig(paths[0], dpi=600, bbox_inches="tight", pad_inches=0.07)
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=0.07)
    plt.close(fig)
    return paths


def _shared_solution_potential_vlim_mV(
    cases: Mapping[tuple[float, float], Any],
) -> float:
    raw_limit = max(
        float(np.max(np.abs(np.asarray(case.phi_s_mV, dtype=float))))
        for case in cases.values()
    )
    if not math.isfinite(raw_limit) or raw_limit <= 0.0:
        raise RuntimeError(f"Invalid representative 2-D potential range: {raw_limit!r}")
    return 10.0 * math.ceil(raw_limit / 10.0)


def _reactant_distributions(case: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return dimensionless potential and normalized reactant concentrations."""

    thermal_v = (
        float(case.derived["R"])
        * float(case.derived["T"])
        / float(case.derived["F"])
    )
    phi_tilde = 1.0e-3 * np.asarray(case.phi_s_mV, dtype=float) / thermal_v
    c_red1_norm = np.exp(
        np.clip(-float(case.params["z_R1"]) * phi_tilde, -700.0, 700.0)
    )
    c_ox2_norm = np.exp(
        np.clip(-float(case.params["z_O2"]) * phi_tilde, -700.0, 700.0)
    )
    return phi_tilde, c_red1_norm, c_ox2_norm


def _shared_reactant_log_limits(
    cases: Mapping[tuple[float, float], Any],
) -> tuple[float, float]:
    """Build one reciprocal log scale shared by all cases and both reactants."""

    raw_min = math.inf
    raw_max = 0.0
    for case in cases.values():
        _phi_tilde, c_red1_norm, c_ox2_norm = _reactant_distributions(case)
        for values in (c_red1_norm, c_ox2_norm):
            if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                raise RuntimeError("Representative 2-D reactant fields must be finite and positive")
            raw_min = min(raw_min, float(np.min(values)))
            raw_max = max(raw_max, float(np.max(values)))
    if not math.isfinite(raw_min) or not math.isfinite(raw_max) or raw_min <= 0.0:
        raise RuntimeError(
            f"Invalid representative 2-D reactant range: {raw_min!r}, {raw_max!r}"
        )
    exponent = max(
        1,
        int(math.ceil(abs(math.log10(raw_min)))),
        int(math.ceil(abs(math.log10(raw_max)))),
    )
    return 10.0 ** (-exponent), 10.0**exponent


def _representative_2d_rows(
    cases: Mapping[tuple[float, float], Any],
    references: Mapping[tuple[float, float], Any],
    shared_vlim_mV: float,
    shared_reactant_vmin: float,
    shared_reactant_vmax: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for length_nm in MAIN_LENGTHS_NM:
        for pzc_v in REPRESENTATIVE_PZC_V:
            case = cases[(length_nm, pzc_v)]
            reference = references[(length_nm, pzc_v)]
            phi_mV = np.asarray(case.phi_s_mV, dtype=float)
            surface_mV = 1.0e3 * np.asarray(case.phi_rp_with_V, dtype=float)
            phi_tilde, c_red1_norm, c_ox2_norm = _reactant_distributions(case)
            rows.append(
                {
                    "L_support_nm": length_nm,
                    "pzc_support_V": pzc_v,
                    "N_modes": int(case.params["N_modes"]),
                    "gl_order": int(case.gl_order),
                    "Nx_2d": int(np.asarray(case.x_2d_nm).size),
                    "Ny_2d": int(np.asarray(case.y_2d_nm).size),
                    "y_max_nm": float(np.max(case.y_2d_nm)),
                    "phi_s_min_mV": float(np.min(phi_mV)),
                    "phi_s_max_mV": float(np.max(phi_mV)),
                    "shared_abs_color_limit_mV": shared_vlim_mV,
                    "phi_tilde_min": float(np.min(phi_tilde)),
                    "phi_tilde_max": float(np.max(phi_tilde)),
                    "c_Red1_over_c_bulk_min": float(np.min(c_red1_norm)),
                    "c_Red1_over_c_bulk_max": float(np.max(c_red1_norm)),
                    "c_Ox2_over_c_bulk_min": float(np.min(c_ox2_norm)),
                    "c_Ox2_over_c_bulk_max": float(np.max(c_ox2_norm)),
                    "shared_reactant_log_vmin": shared_reactant_vmin,
                    "shared_reactant_log_vmax": shared_reactant_vmax,
                    "reactant_reciprocity_max_error": float(
                        np.max(np.abs(c_red1_norm * c_ox2_norm - 1.0))
                    ),
                    "surface_reconstruction_max_error_phi_tilde": float(
                        case.phi_2d_surface_max_error
                    ),
                    "surface_reconstruction_max_error_mV": float(
                        np.max(np.abs(phi_mV[0] - surface_mV))
                    ),
                    "E_mix_2d_case_V": float(case.res_edl["E_mix"]),
                    "E_mix_difference_from_main_scan_V": float(
                        case.res_edl["E_mix"] - reference.res_edl["E_mix"]
                    ),
                    "i_mix_avg_2d_case_A_per_m2": float(
                        case.res_edl["i_mix_avg_A_per_m2"]
                    ),
                    "i_mix_avg_difference_from_main_scan_A_per_m2": float(
                        case.res_edl["i_mix_avg_A_per_m2"]
                        - reference.res_edl["i_mix_avg_A_per_m2"]
                    ),
                }
            )
    return rows


def _validate_representative_2d(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_count = len(MAIN_LENGTHS_NM) * len(REPRESENTATIVE_PZC_V)
    max_surface_tilde = max(
        abs(float(row["surface_reconstruction_max_error_phi_tilde"])) for row in rows
    )
    max_surface_mV = max(
        abs(float(row["surface_reconstruction_max_error_mV"])) for row in rows
    )
    max_e_mix_difference = max(
        abs(float(row["E_mix_difference_from_main_scan_V"])) for row in rows
    )
    max_i_mix_difference = max(
        abs(float(row["i_mix_avg_difference_from_main_scan_A_per_m2"]))
        for row in rows
    )
    color_limit_contains_all = all(
        max(abs(float(row["phi_s_min_mV"])), abs(float(row["phi_s_max_mV"])))
        <= float(row["shared_abs_color_limit_mV"]) + 1.0e-12
        for row in rows
    )
    reactant_limits_contain_all = all(
        float(row["shared_reactant_log_vmin"])
        <= min(
            float(row["c_Red1_over_c_bulk_min"]),
            float(row["c_Ox2_over_c_bulk_min"]),
        )
        and max(
            float(row["c_Red1_over_c_bulk_max"]),
            float(row["c_Ox2_over_c_bulk_max"]),
        )
        <= float(row["shared_reactant_log_vmax"])
        for row in rows
    )
    max_reciprocity_error = max(
        abs(float(row["reactant_reciprocity_max_error"])) for row in rows
    )
    finite = all(
        math.isfinite(float(value))
        for row in rows
        for value in row.values()
    )
    passed = bool(
        len(rows) == expected_count
        and finite
        and color_limit_contains_all
        and reactant_limits_contain_all
        and max_reciprocity_error < 1.0e-12
        and max_surface_tilde <= engine.SURFACE_RECONSTRUCTION_TOL
        and max_surface_mV < 1.0e-8
        and max_e_mix_difference < 5.0e-11
        and max_i_mix_difference < 5.0e-11
    )
    return {
        "case_count": len(rows),
        "expected_case_count": expected_count,
        "all_values_finite": finite,
        "shared_color_limit_contains_all_fields": color_limit_contains_all,
        "shared_reactant_log_limits_contain_all_fields": reactant_limits_contain_all,
        "max_reactant_reciprocity_error": max_reciprocity_error,
        "max_surface_reconstruction_error_phi_tilde": max_surface_tilde,
        "surface_reconstruction_tolerance_phi_tilde": engine.SURFACE_RECONSTRUCTION_TOL,
        "max_surface_reconstruction_error_mV": max_surface_mV,
        "max_E_mix_difference_from_main_scan_V": max_e_mix_difference,
        "max_i_mix_avg_difference_from_main_scan_A_per_m2": max_i_mix_difference,
        "passed": passed,
    }


def _format_nm_tick(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1.0e-9):
        return str(int(round(value)))
    return f"{value:g}"


def _add_2d_material_lane(ax: plt.Axes, length_nm: float) -> None:
    total_nm = 4.0 + length_nm
    segments = (
        ("Au", 0.0, 2.0, MATERIAL_COLORS["Au"], DARK),
        ("C", 2.0, 2.0 + length_nm, MATERIAL_COLORS["C"], "white"),
        ("Pd", 2.0 + length_nm, total_nm, MATERIAL_COLORS["Pd"], "white"),
    )
    for label, left, right, color, text_color in segments:
        ax.add_patch(
            Rectangle(
                (left, 0.0),
                right - left,
                1.0,
                facecolor=color,
                edgecolor="white",
                linewidth=0.8,
            )
        )
        ax.text(
            0.5 * (left + right),
            0.5,
            label,
            ha="center",
            va="center",
            color=text_color,
            fontsize=8.0,
        )
    ax.set_xlim(0.0, total_nm)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()


def _plot_solution_phase_potential_2d(
    case: Any,
    figure_dir: Path,
    shared_vlim_mV: float,
) -> list[Path]:
    length_nm = float(case.L_support_nm)
    pzc_v = float(case.params["pzc_C"])
    x_nm = np.asarray(case.x_2d_nm, dtype=float)
    y_nm = np.asarray(case.y_2d_nm, dtype=float)
    phi_mV = np.asarray(case.phi_s_mV, dtype=float)
    total_nm = float(case.L_total_nm)

    fig = plt.figure(figsize=(5.8, 3.0), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 0.038),
        height_ratios=(1.0, 0.12),
        hspace=0.24,
        wspace=0.08,
    )
    ax = fig.add_subplot(grid[0, 0])
    colorbar_ax = fig.add_subplot(grid[0, 1])
    lane_ax = fig.add_subplot(grid[1, 0], sharex=ax)
    fig.add_subplot(grid[1, 1]).set_axis_off()

    norm = TwoSlopeNorm(
        vmin=-shared_vlim_mV,
        vcenter=0.0,
        vmax=shared_vlim_mV,
    )
    mesh = ax.pcolormesh(
        x_nm,
        y_nm,
        phi_mV,
        shading="auto",
        cmap="RdBu_r",
        norm=norm,
        rasterized=True,
    )
    contour_levels = np.linspace(-shared_vlim_mV, shared_vlim_mV, 11)
    finite = phi_mV[np.isfinite(phi_mV)]
    inside = contour_levels[
        (contour_levels > float(np.min(finite)))
        & (contour_levels < float(np.max(finite)))
    ]
    if inside.size:
        ax.contour(
            x_nm,
            y_nm,
            phi_mV,
            levels=inside,
            colors="black",
            linewidths=0.28,
            alpha=0.28,
        )

    ax.set_title(
        rf"Solution phase potential, $E_{{\mathrm{{mix}}}}={float(case.res_edl['E_mix']):.3f}\,\mathrm{{V}}$",
        loc="left",
        pad=5.0,
        fontsize=9.4,
        fontweight="normal",
    )
    ax.set_ylabel(r"$y$ ($\mathrm{nm}$)")
    ax.set_xlim(0.0, total_nm)
    ax.set_ylim(float(y_nm[0]), float(y_nm[-1]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    boundaries = (2.0, 2.0 + length_nm)
    ticks = sorted({0.0, *boundaries, total_nm})
    ax.set_xticks(ticks)
    ax.set_xticklabels([_format_nm_tick(value) for value in ticks])
    ax.set_xlabel("")
    ax.tick_params(length=3.2, width=0.85, pad=2.5, labelsize=8.0)
    for boundary in boundaries:
        ax.axvline(
            boundary,
            color="#767676",
            linewidth=0.85,
            linestyle=(0, (3.0, 2.0)),
            alpha=0.82,
            zorder=5,
        )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color(DARK)
    ax.text(
        0.985,
        0.955,
        rf"$L_{{\mathrm{{support}}}}={length_nm:g}\,\mathrm{{nm}}$"
        "\n"
        rf"$\mathrm{{PZC}}_{{\mathrm{{support}}}}={pzc_v:.2f}\,\mathrm{{V}}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.6,
        color=DARK,
        zorder=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.8},
    )

    colorbar = fig.colorbar(mesh, cax=colorbar_ax)
    colorbar.set_label(r"$\Phi_s$ ($\mathrm{mV}$)", labelpad=5)
    colorbar.ax.tick_params(length=2.8, width=0.75, labelsize=7.8, pad=2.2)
    colorbar.outline.set_linewidth(0.8)
    _add_2d_material_lane(lane_ax, length_nm)
    lane_ax.text(
        0.5,
        -0.58,
        r"$x$ ($\mathrm{nm}$)",
        transform=lane_ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color=DARK,
        clip_on=False,
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    return _save_figure(
        fig,
        figure_dir,
        _solution_potential_2d_stem(length_nm, pzc_v),
    )


def _plot_representative_solution_potentials_2d(
    cases: Mapping[tuple[float, float], Any],
    figure_dir: Path,
    shared_vlim_mV: float,
) -> list[Path]:
    paths: list[Path] = []
    for length_nm in MAIN_LENGTHS_NM:
        for pzc_v in REPRESENTATIVE_PZC_V:
            paths.extend(
                _plot_solution_phase_potential_2d(
                    cases[(length_nm, pzc_v)],
                    figure_dir / PZC_2D_OUTPUT_SUBDIR,
                    shared_vlim_mV,
                )
            )
    return paths


def _reactant_colorbar_ticks(vmin: float, vmax: float) -> np.ndarray:
    low_exp = int(round(math.log10(vmin)))
    high_exp = int(round(math.log10(vmax)))
    step = max(1, int(math.ceil((high_exp - low_exp) / 4.0)))
    exponents = set(range(low_exp, high_exp + 1, step))
    exponents.update((low_exp, high_exp))
    if low_exp <= 0 <= high_exp:
        exponents.add(0)
    return np.asarray([10.0**value for value in sorted(exponents)], dtype=float)


def _plot_phi_s_reactants_2d(
    case: Any,
    figure_dir: Path,
    shared_phi_vlim_mV: float,
    shared_reactant_vmin: float,
    shared_reactant_vmax: float,
) -> list[Path]:
    """Plot potential and the two Boltzmann reactant fields for one case."""

    length_nm = float(case.L_support_nm)
    pzc_v = float(case.params["pzc_C"])
    x_nm = np.asarray(case.x_2d_nm, dtype=float)
    y_nm = np.asarray(case.y_2d_nm, dtype=float)
    phi_mV = np.asarray(case.phi_s_mV, dtype=float)
    _phi_tilde, c_red1_norm, c_ox2_norm = _reactant_distributions(case)
    total_nm = float(case.L_total_nm)
    boundaries = (2.0, 2.0 + length_nm)
    ticks = sorted({0.0, *boundaries, total_nm})

    fig = plt.figure(figsize=(5.8, 6.75), facecolor="white")
    grid = fig.add_gridspec(
        4,
        2,
        width_ratios=(1.0, 0.038),
        height_ratios=(1.0, 1.0, 1.0, 0.12),
        hspace=0.24,
        wspace=0.08,
    )
    axes = [fig.add_subplot(grid[index, 0]) for index in range(3)]
    colorbar_axes = [fig.add_subplot(grid[index, 1]) for index in range(3)]
    lane_ax = fig.add_subplot(grid[3, 0], sharex=axes[-1])
    fig.add_subplot(grid[3, 1]).set_axis_off()

    phi_norm = TwoSlopeNorm(
        vmin=-shared_phi_vlim_mV,
        vcenter=0.0,
        vmax=shared_phi_vlim_mV,
    )
    reactant_norm = LogNorm(vmin=shared_reactant_vmin, vmax=shared_reactant_vmax)
    panel_specs = (
        (
            phi_mV,
            "RdBu_r",
            phi_norm,
            (
                "Solution phase potential, "
                rf"$E_{{\mathrm{{mix}}}}={float(case.res_edl['E_mix']):.3f}\,\mathrm{{V}}$"
            ),
            r"$\Phi_s$ ($\mathrm{mV}$)",
        ),
        (
            c_red1_norm,
            "viridis",
            reactant_norm,
            "Reactant Red1 distribution",
            r"$c_{\mathrm{Red}_1}/c_{\mathrm{bulk}}$",
        ),
        (
            c_ox2_norm,
            "viridis",
            reactant_norm,
            "Reactant Ox2 distribution",
            r"$c_{\mathrm{Ox}_2}/c_{\mathrm{bulk}}$",
        ),
    )
    reactant_ticks = _reactant_colorbar_ticks(
        shared_reactant_vmin, shared_reactant_vmax
    )

    for panel_index, (ax, colorbar_ax, spec) in enumerate(
        zip(axes, colorbar_axes, panel_specs, strict=True)
    ):
        values, cmap, norm, title, colorbar_label = spec
        mesh = ax.pcolormesh(
            x_nm,
            y_nm,
            values,
            shading="auto",
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
        finite = values[np.isfinite(values)]
        if panel_index == 0:
            candidate_levels = np.linspace(
                -shared_phi_vlim_mV, shared_phi_vlim_mV, 11
            )
        else:
            candidate_levels = np.logspace(
                math.log10(shared_reactant_vmin),
                math.log10(shared_reactant_vmax),
                9,
            )
        inside = candidate_levels[
            (candidate_levels > float(np.min(finite)))
            & (candidate_levels < float(np.max(finite)))
        ]
        if inside.size:
            ax.contour(
                x_nm,
                y_nm,
                values,
                levels=inside,
                colors="black",
                linewidths=0.27,
                alpha=0.26,
            )
        ax.set_title(title, loc="left", pad=4.0, fontsize=9.4, fontweight="normal")
        ax.set_ylabel(r"$y$ ($\mathrm{nm}$)")
        ax.set_xlim(0.0, total_nm)
        ax.set_ylim(float(y_nm[0]), float(y_nm[-1]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_xticks(ticks)
        ax.set_xticklabels([_format_nm_tick(value) for value in ticks])
        ax.tick_params(length=3.2, width=0.85, pad=2.5, labelsize=8.0)
        if panel_index < 2:
            ax.tick_params(labelbottom=False)
        for boundary in boundaries:
            ax.axvline(
                boundary,
                color="#767676",
                linewidth=0.85,
                linestyle=(0, (3.0, 2.0)),
                alpha=0.82,
                zorder=5,
            )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.9)
            spine.set_color(DARK)
        colorbar = fig.colorbar(mesh, cax=colorbar_ax)
        colorbar.set_label(colorbar_label, labelpad=5)
        if panel_index > 0:
            colorbar.set_ticks(reactant_ticks)
        colorbar.ax.tick_params(length=2.8, width=0.75, labelsize=7.6, pad=2.2)
        colorbar.outline.set_linewidth(0.8)

    axes[0].text(
        0.985,
        0.945,
        rf"$L_{{\mathrm{{support}}}}={length_nm:g}\,\mathrm{{nm}}$"
        "\n"
        rf"$\mathrm{{PZC}}_{{\mathrm{{support}}}}={pzc_v:.2f}\,\mathrm{{V}}$",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=7.6,
        color=DARK,
        zorder=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.8},
    )
    _add_2d_material_lane(lane_ax, length_nm)
    lane_ax.text(
        0.5,
        -0.58,
        r"$x$ ($\mathrm{nm}$)",
        transform=lane_ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color=DARK,
        clip_on=False,
    )
    fig.align_ylabels(axes)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return _save_figure(
        fig,
        figure_dir,
        _phi_s_reactants_2d_stem(length_nm, pzc_v),
    )


def _plot_representative_phi_s_reactants_2d(
    cases: Mapping[tuple[float, float], Any],
    figure_dir: Path,
    shared_phi_vlim_mV: float,
    shared_reactant_vmin: float,
    shared_reactant_vmax: float,
) -> list[Path]:
    paths: list[Path] = []
    for length_nm in MAIN_LENGTHS_NM:
        for pzc_v in REPRESENTATIVE_PZC_V:
            paths.extend(
                _plot_phi_s_reactants_2d(
                    cases[(length_nm, pzc_v)],
                    figure_dir / PZC_2D_OUTPUT_SUBDIR,
                    shared_phi_vlim_mV,
                    shared_reactant_vmin,
                    shared_reactant_vmax,
                )
            )
    return paths


def _format_profile_axis(ax: plt.Axes, length_nm: float, ylabel: str) -> None:
    ax.set_title(
        rf"$L_{{\mathrm{{support}}}} = {length_nm:g}\,\mathrm{{nm}}$",
        loc="left",
        pad=4.0,
    )
    ax.set_xlabel(r"$x$ ($\mathrm{nm}$)")
    ax.set_ylabel(ylabel)
    ax.axvline(2.0, color=DARK, linewidth=0.65, alpha=0.45)
    ax.axvline(2.0 + length_nm, color=DARK, linewidth=0.65, alpha=0.45)
    ax.tick_params(pad=2.5)


def _material_background(ax: plt.Axes, length_nm: float) -> None:
    spans = ((0.0, 2.0, "Au"), (2.0, 2.0 + length_nm, "C"), (2.0 + length_nm, 4.0 + length_nm, "Pd"))
    for left, right, material in spans:
        ax.axvspan(left, right, color=MATERIAL_COLORS[material], alpha=0.075, linewidth=0)


def _plot_trends(
    rows: Sequence[Mapping[str, Any]],
    anchors: Sequence[Mapping[str, Any]],
    figure_dir: Path,
) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.45), sharex=True)
    specifications = (
        ("E_mix_with_V", r"$E_{\mathrm{mix}}$ ($\mathrm{V\ vs.\ RHE}$)", "Mixed potential"),
        ("i_mix_avg_with_A_per_m2", r"$\bar{i}_{\mathrm{mix}}$ ($\mathrm{A\,m^{-2}}$)", "Mixed current density"),
        ("sigma_support_signed_mean_uC_per_cm2", r"$\langle\sigma_{\mathrm{support}}\rangle$ ($\mathrm{\mu C\,cm^{-2}}$)", "Mean signed support charge"),
    )
    for ax, (key, ylabel, title) in zip(axes, specifications, strict=True):
        for length_nm in MAIN_LENGTHS_NM:
            subset = sorted(
                (row for row in rows if float(row["L_support_nm"]) == length_nm),
                key=lambda row: float(row["pzc_support_V"]),
            )
            ax.plot(
                [float(row["pzc_support_V"]) for row in subset],
                [float(row[key]) for row in subset],
                color=LENGTH_COLORS[length_nm],
                linewidth=1.9,
            )
        ax.scatter(
            [float(row["pzc_support_V"]) for row in anchors],
            [float(row[key]) for row in anchors],
            s=35,
            facecolors="white",
            edgecolors=ANCHOR_COLOR,
            linewidths=1.25,
            zorder=5,
        )
        if key in {"E_mix_with_V", "i_mix_avg_with_A_per_m2"}:
            no_key = "E_mix_no_V" if key == "E_mix_with_V" else "i_mix_avg_no_A_per_m2"
            baseline = float(rows[0][no_key])
            ax.axhline(
                baseline,
                color=WITHOUT_EDL_COLOR,
                linestyle=(0, (4.0, 2.0)),
                linewidth=1.4,
            )
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=4.0)
        ax.set_xlim(0.08, 0.92)
        ax.tick_params(pad=2.5)
    for ax in axes:
        ax.set_xlabel(r"$\mathrm{PZC}_{\mathrm{support}}$ ($\mathrm{V\ vs.\ RHE}$)")
    handles = [
        Line2D([0], [0], color=LENGTH_COLORS[length], lw=2.2, label=rf"{length:g} $\mathrm{{nm}}$")
        for length in MAIN_LENGTHS_NM
    ]
    handles.extend(
        [
            Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor=ANCHOR_COLOR, color="none", label=r"1000 $\mathrm{nm}$ anchors"),
            Line2D([0], [0], color=WITHOUT_EDL_COLOR, lw=1.5, linestyle=(0, (4, 2)), label="w/o EDL"),
        ]
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.015), ncol=6, handlelength=2.4, columnspacing=1.1, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=1.35)
    return _save_figure(fig, figure_dir, FIGURE_STEMS[0])


def _plot_phi_profiles(
    representatives: Mapping[tuple[float, float], Any], figure_dir: Path
) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.1))
    for ax, length_nm in zip(axes.flat, MAIN_LENGTHS_NM, strict=True):
        _material_background(ax, length_nm)
        for pzc_v in REPRESENTATIVE_PZC_V:
            case = representatives[(length_nm, pzc_v)]
            style = PROFILE_STYLES[pzc_v]
            ax.plot(case.x_nm, 1.0e3 * case.phi_rp_with_V, color=style["color"], linestyle=style["linestyle"], linewidth=1.75)
        ax.axhline(0.0, color=WITHOUT_EDL_COLOR, linestyle=(0, (4, 2)), linewidth=1.2)
        _format_profile_axis(ax, length_nm, r"$\phi_{\mathrm{RP}}$ ($\mathrm{mV}$)")
    handles = [Line2D([0], [0], color=PROFILE_STYLES[pzc]["color"], linestyle=PROFILE_STYLES[pzc]["linestyle"], lw=2.0, label=rf"$\mathrm{{PZC}}_{{\mathrm{{support}}}}={pzc:.2f}\,\mathrm{{V}}$") for pzc in REPRESENTATIVE_PZC_V]
    handles.append(Line2D([0], [0], color=WITHOUT_EDL_COLOR, linestyle=(0, (4, 2)), lw=1.5, label=r"w/o EDL ($\phi_{\mathrm{RP}}=0$)"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=4, handlelength=3.0, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.5, w_pad=1.5)
    return _save_figure(fig, figure_dir, FIGURE_STEMS[1])


def _plot_sigma_profiles(
    representatives: Mapping[tuple[float, float], Any], figure_dir: Path
) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.1))
    for ax, length_nm in zip(axes.flat, MAIN_LENGTHS_NM, strict=True):
        _material_background(ax, length_nm)
        for pzc_v in REPRESENTATIVE_PZC_V:
            case = representatives[(length_nm, pzc_v)]
            style = PROFILE_STYLES[pzc_v]
            for segment in case.sigma_segments:
                ax.plot(segment.x_nm, segment.sigma_uC_per_cm2, color=style["color"], linestyle=style["linestyle"], linewidth=1.75)
        ax.axhline(0.0, color="#B8B8B8", linewidth=0.7)
        _format_profile_axis(ax, length_nm, r"$\sigma$ ($\mathrm{\mu C\,cm^{-2}}$)")
    pzc_handles = [Line2D([0], [0], color=PROFILE_STYLES[pzc]["color"], linestyle=PROFILE_STYLES[pzc]["linestyle"], lw=2.0, label=rf"$\mathrm{{PZC}}_{{\mathrm{{support}}}}={pzc:.2f}\,\mathrm{{V}}$") for pzc in REPRESENTATIVE_PZC_V]
    material_handles = [
        Patch(
            facecolor=MATERIAL_COLORS[name],
            edgecolor="none",
            alpha=0.55,
            label="support" if name == "C" else name,
        )
        for name in ("Au", "C", "Pd")
    ]
    fig.legend(handles=pzc_handles + material_handles, loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=6, handlelength=2.8, fontsize=8.3)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.5, w_pad=1.5)
    return _save_figure(fig, figure_dir, FIGURE_STEMS[2])


def _row_lookup(
    rows: Sequence[Mapping[str, Any]], length_nm: float, pzc_v: float
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if math.isclose(float(row["L_support_nm"]), length_nm, abs_tol=1.0e-10)
        and math.isclose(float(row["pzc_support_V"]), pzc_v, abs_tol=1.0e-12)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one scan row at L={length_nm:g} nm, PZC={pzc_v:.2f} V; "
            f"found {len(matches)}"
        )
    return matches[0]


def _parameter_lock(case: Any) -> dict[str, Any]:
    expected = {
        "L_Au": engine.L_AU_M,
        "L_Pd_len": engine.L_PD_M,
        "Cdl_Au": engine.C_H_AU_F_PER_M2,
        "Cdl_C": engine.C_H_SUPPORT_F_PER_M2,
        "Cdl_Pd": engine.C_H_PD_F_PER_M2,
        "it0_1": engine.I0_EQUAL_A_PER_M2,
        "it0_2": engine.I0_EQUAL_A_PER_M2,
        "alpha1": engine.ALPHA_EQUAL,
        "alpha2": engine.ALPHA_EQUAL,
        "out_of_plane_width": engine.OUT_OF_PLANE_WIDTH_M,
        "C_tot": 10.0,
    }
    checks: dict[str, Any] = {}
    passed = True
    for key, target in expected.items():
        actual = float(case.params[key])
        tolerance = max(1.0e-15, abs(target) * 1.0e-12)
        item_passed = math.isclose(actual, target, rel_tol=0.0, abs_tol=tolerance)
        checks[key] = {
            "actual": actual,
            "expected": target,
            "absolute_tolerance": tolerance,
            "passed": item_passed,
        }
        passed = passed and item_passed
    return {"checks": checks, "passed": passed}


def _baseline_regression(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reference_rows = _read_csv(REFERENCE_OFAT_CSV)
    fields = (
        ("E_mix_with_V", "E_mix_with_V"),
        ("E_mix_no_V", "E_mix_no_V"),
        ("i_mix_avg_with_A_per_m2", "i_mix_avg_with_A_per_m2"),
        ("i_mix_avg_no_A_per_m2", "i_mix_avg_no_A_per_m2"),
        ("phi_RP_Au_mean_mV", "phi_RP_Au_mean_mV"),
        ("phi_RP_Pd_mean_mV", "phi_RP_Pd_mean_mV"),
        ("sigma_support_signed_mean_C_per_m2", "sigma_C_signed_mean_C_per_m2"),
    )
    tolerances = {
        "E_mix_with_V": 5.0e-10,
        "E_mix_no_V": 5.0e-10,
        "i_mix_avg_with_A_per_m2": 5.0e-10,
        "i_mix_avg_no_A_per_m2": 5.0e-10,
        "phi_RP_Au_mean_mV": 5.0e-7,
        "phi_RP_Pd_mean_mV": 5.0e-7,
        "sigma_support_signed_mean_C_per_m2": 5.0e-9,
    }
    results: dict[str, Any] = {}
    passed = True
    for length_nm in MAIN_LENGTHS_NM:
        actual = _row_lookup(rows, length_nm, 0.50)
        references = [
            row
            for row in reference_rows
            if math.isclose(float(row["L_support_nm"]), length_nm, abs_tol=1.0e-10)
        ]
        if len(references) != 1:
            raise RuntimeError(f"Missing unique OFAT reference at L={length_nm:g} nm")
        reference = references[0]
        length_results: dict[str, Any] = {}
        for actual_field, reference_field in fields:
            actual_value = float(actual[actual_field])
            expected_value = float(reference[reference_field])
            difference = abs(actual_value - expected_value)
            item_passed = difference <= tolerances[actual_field]
            length_results[actual_field] = {
                "actual": actual_value,
                "reference": expected_value,
                "absolute_difference": difference,
                "absolute_tolerance": tolerances[actual_field],
                "passed": item_passed,
            }
            passed = passed and item_passed
        results[f"{length_nm:g}_nm"] = length_results
    return {
        "reference_file": str(REFERENCE_OFAT_CSV),
        "comparison_resolution_N_modes": REQUESTED_SHORT_HIGH_MODES,
        "note": (
            "Regression is evaluated at the existing Figure_L_support N=960 "
            "resolution; production rows use the independently validated adaptive "
            "mode policy."
        ),
        "checks": results,
        "passed": passed,
    }


def _stern_and_profile_checks(
    rows: Sequence[Mapping[str, Any]],
    representatives: Mapping[tuple[float, float], Any],
) -> dict[str, Any]:
    stern_max_error = 0.0
    scalar_max_error = 0.0
    all_finite = True
    keys = {
        "Au": ("Cdl_Au", "pzc_Au"),
        "support": ("Cdl_C", "pzc_C"),
        "Pd": ("Cdl_Pd", "pzc_Pd"),
    }
    for (length_nm, pzc_v), case in representatives.items():
        scan_row = _row_lookup(rows, length_nm, pzc_v)
        scalar_max_error = max(
            scalar_max_error,
            abs(_signed_support_sigma(case) - float(scan_row["sigma_support_signed_mean_C_per_m2"])),
            abs(1.0e3 * float(case.res_edl["phi2_1_meanV"]) - float(scan_row["phi_RP_Au_mean_mV"])),
            abs(1.0e3 * float(case.res_edl["phi2_2_meanV"]) - float(scan_row["phi_RP_Pd_mean_mV"])),
        )
        for segment in case.sigma_segments:
            c_h_key, pzc_key = keys[str(segment.material)]
            expected = float(case.params[c_h_key]) * (
                float(case.res_edl["E_mix"])
                - float(case.params[pzc_key])
                - np.asarray(segment.phi_rp_V, dtype=float)
            )
            observed = np.asarray(segment.sigma_C_per_m2, dtype=float)
            stern_max_error = max(stern_max_error, float(np.max(np.abs(expected - observed))))
            all_finite = all_finite and bool(
                np.all(np.isfinite(observed))
                and np.all(np.isfinite(segment.phi_rp_V))
            )
    passed = bool(all_finite and stern_max_error < 1.0e-13 and scalar_max_error < 1.0e-12)
    return {
        "representative_profile_count": len(representatives),
        "all_profile_values_finite": all_finite,
        "stern_relation_max_abs_error_C_per_m2": stern_max_error,
        "stern_relation_tolerance_C_per_m2": 1.0e-13,
        "profile_to_scan_scalar_max_abs_error": scalar_max_error,
        "profile_to_scan_scalar_tolerance": 1.0e-12,
        "passed": passed,
    }


def _numeric_validation(
    rows: Sequence[Mapping[str, Any]],
    anchor_rows: Sequence[Mapping[str, Any]],
    negative_rows: Sequence[Mapping[str, Any]],
    convergence_rows: Sequence[Mapping[str, Any]],
    requested_resolution_rows: Sequence[Mapping[str, Any]],
    regression_rows: Sequence[Mapping[str, Any]],
    representatives: Mapping[tuple[float, float], Any],
) -> dict[str, Any]:
    scan_shape_passed = bool(
        len(rows) == 164
        and sorted({float(row["L_support_nm"]) for row in rows}) == list(MAIN_LENGTHS_NM)
        and sorted({round(float(row["pzc_support_V"]), 12) for row in rows})
        == list(PZC_VALUES_V)
    )
    numeric_fields = (
        "E_mix_with_V",
        "E_mix_no_V",
        "i_mix_avg_with_A_per_m2",
        "i_mix_avg_no_A_per_m2",
        "phi_RP_Au_mean_mV",
        "phi_RP_Pd_mean_mV",
        "sigma_support_signed_mean_C_per_m2",
        "overlap_active_RP_RMS_vs_15nm_mV",
        "relative_balance_with",
        "relative_balance_no",
    )
    finite = all(
        math.isfinite(float(row[field])) for row in rows for field in numeric_fields
    ) and all(
        math.isfinite(float(row[field]))
        for row in anchor_rows
        for field in numeric_fields
    ) and all(
        math.isfinite(float(value))
        for row in negative_rows
        for key, value in row.items()
        if key not in {"L_support_nm", "pzc_support_V"}
    ) and all(
        math.isfinite(float(value))
        for row in convergence_rows
        for value in row.values()
    ) and all(
        math.isfinite(float(value))
        for row in requested_resolution_rows
        for key, value in row.items()
        if key != "check_kind"
    )
    condition_balance = max(
        float(row[key])
        for row in tuple(rows) + tuple(anchor_rows) + tuple(negative_rows)
        for key in ("relative_balance_with", "relative_balance_no")
    )
    convergence_balance = max(
        float(row[key])
        for row in convergence_rows
        for key in (
            "relative_balance_high",
            "relative_balance_low_modes",
            "relative_balance_gl64",
        )
    )
    requested_resolution_balance = max(
        float(row[key])
        for row in requested_resolution_rows
        for key in ("relative_balance_low_modes", "relative_balance_high_modes")
    )
    max_balance = max(
        condition_balance,
        convergence_balance,
        requested_resolution_balance,
    )
    all_conditions = tuple(rows) + tuple(anchor_rows) + tuple(negative_rows)
    no_e = np.asarray([float(row["E_mix_no_V"]) for row in all_conditions], dtype=float)
    no_i = np.asarray(
        [float(row["i_mix_avg_no_A_per_m2"]) for row in all_conditions], dtype=float
    )
    no_edl_invariance = {
        "E_mix_range_V": float(np.ptp(no_e)),
        "E_mix_tolerance_V": 1.0e-11,
        "i_mix_avg_range_A_per_m2": float(np.ptp(no_i)),
        "i_mix_avg_tolerance_A_per_m2": 1.0e-11,
    }
    no_edl_invariance["passed"] = bool(
        no_edl_invariance["E_mix_range_V"] < no_edl_invariance["E_mix_tolerance_V"]
        and no_edl_invariance["i_mix_avg_range_A_per_m2"]
        < no_edl_invariance["i_mix_avg_tolerance_A_per_m2"]
    )
    negative_a, negative_b = negative_rows
    negative_differences = {
        field: abs(float(negative_a[field]) - float(negative_b[field]))
        for field in (
            "E_mix_with_V",
            "i_mix_avg_with_A_per_m2",
            "phi_RP_Au_mean_mV",
            "phi_RP_Pd_mean_mV",
        )
    }
    negative_passed = bool(max(negative_differences.values()) < 1.0e-10)
    convergence_passed = bool(
        len(convergence_rows) == 15 and all(bool(row["passed"]) for row in convergence_rows)
    )
    baseline = _baseline_regression(regression_rows)
    profiles = _stern_and_profile_checks(rows, representatives)
    parameter_lock = _parameter_lock(representatives[(1.0, 0.50)])
    requested_failures = [
        {
            "L_support_nm": float(row["L_support_nm"]),
            "pzc_support_V": float(row["pzc_support_V"]),
            "mode_delta_E_mV": float(row["mode_delta_E_mV"]),
            "mode_delta_i_percent": float(row["mode_delta_i_percent"]),
        }
        for row in requested_resolution_rows
        if not bool(row["passed"])
    ]
    requested_audit_integrity = bool(
        len(requested_resolution_rows) == 15
        and requested_resolution_balance < BALANCE_TOL
    )
    production_mode_policy = bool(
        all(
            int(row["N_modes"])
            == _modes_for_length(
                PRODUCTION_MODES_BY_LENGTH, float(row["L_support_nm"])
            )
            for row in tuple(rows) + tuple(anchor_rows)
        )
        and all(
            int(row["low_N_modes"])
            == _modes_for_length(
                CONVERGENCE_LOW_MODES_BY_LENGTH,
                float(row["L_support_nm"]),
            )
            and int(row["high_N_modes"])
            == _modes_for_length(
                PRODUCTION_MODES_BY_LENGTH,
                float(row["L_support_nm"]),
            )
            for row in convergence_rows
        )
    )
    reference_differences = {
        f"pzc_{pzc_v:.2f}_V": {
            "delta_E_mix_1000_minus_15_mV": 1.0e3
            * (
                float(_row_lookup(anchor_rows, ANCHOR_LENGTH_NM, pzc_v)["E_mix_with_V"])
                - float(_row_lookup(rows, 15.0, pzc_v)["E_mix_with_V"])
            ),
            "delta_i_mix_avg_1000_minus_15_percent": 100.0
            * (
                float(_row_lookup(anchor_rows, ANCHOR_LENGTH_NM, pzc_v)["i_mix_avg_with_A_per_m2"])
                / float(_row_lookup(rows, 15.0, pzc_v)["i_mix_avg_with_A_per_m2"])
                - 1.0
            ),
            "active_RP_RMS_1000_vs_15_mV": float(
                _row_lookup(anchor_rows, ANCHOR_LENGTH_NM, pzc_v)["overlap_active_RP_RMS_vs_15nm_mV"]
            ),
        }
        for pzc_v in REPRESENTATIVE_PZC_V
    }
    passed = bool(
        scan_shape_passed
        and finite
        and max_balance < BALANCE_TOL
        and no_edl_invariance["passed"]
        and negative_passed
        and convergence_passed
        and requested_audit_integrity
        and production_mode_policy
        and baseline["passed"]
        and profiles["passed"]
        and parameter_lock["passed"]
    )
    return {
        "main_scan_shape": {"rows": len(rows), "expected_rows": 164, "passed": scan_shape_passed},
        "all_computed_values_finite": finite,
        "current_balance": {
            "maximum_relative_residual": max_balance,
            "maximum_scan_anchor_control_residual": condition_balance,
            "maximum_convergence_case_residual": convergence_balance,
            "maximum_requested_resolution_audit_residual": requested_resolution_balance,
            "threshold": BALANCE_TOL,
            "passed": max_balance < BALANCE_TOL,
        },
        "without_edl_invariance": no_edl_invariance,
        "zero_length_negative_control": {"endpoint_differences": negative_differences, "absolute_tolerance": 1.0e-10, "passed": negative_passed},
        "pzc_0p50_regression_to_Figure_L_support": baseline,
        "representative_profiles": profiles,
        "parameter_lock": parameter_lock,
        "convergence": {"row_count": len(convergence_rows), "expected_row_count": 15, "all_passed": convergence_passed},
        "original_requested_resolution_audit": {
            "row_count": len(requested_resolution_rows),
            "expected_row_count": 15,
            "all_requested_pairs_passed": not requested_failures,
            "failed_pair_count": len(requested_failures),
            "failed_pairs": requested_failures,
            "integrity_passed": requested_audit_integrity,
            "disposition": (
                "Production modes were raised only where the originally requested "
                "mode pairs failed their stated tolerances."
            ),
        },
        "production_resolution_policy": {
            "production_N_modes_by_L_support_nm": {
                f"{length:g}": modes
                for length, modes in PRODUCTION_MODES_BY_LENGTH.items()
            },
            "convergence_low_N_modes_by_L_support_nm": {
                f"{length:g}": modes
                for length, modes in CONVERGENCE_LOW_MODES_BY_LENGTH.items()
            },
            "passed": production_mode_policy,
        },
        "reference_15nm_vs_1000nm": reference_differences,
        "passed": passed,
    }


def _expected_figure_relative_files() -> set[str]:
    files = {
        f"figures/{stem}.{suffix}"
        for stem in FIGURE_STEMS
        for suffix in ("png", "svg")
    }
    files.update(
        (
            "figures/"
            f"{PZC_2D_OUTPUT_SUBDIR.as_posix()}/"
            f"{_solution_potential_2d_stem(length_nm, pzc_v)}.{suffix}"
        )
        for length_nm in MAIN_LENGTHS_NM
        for pzc_v in REPRESENTATIVE_PZC_V
        for suffix in ("png", "svg")
    )
    files.update(
        (
            "figures/"
            f"{PZC_2D_OUTPUT_SUBDIR.as_posix()}/"
            f"{_phi_s_reactants_2d_stem(length_nm, pzc_v)}.{suffix}"
        )
        for length_nm in MAIN_LENGTHS_NM
        for pzc_v in REPRESENTATIVE_PZC_V
        for suffix in ("png", "svg")
    )
    return files


def _verify_artifacts(output: Path) -> dict[str, Any]:
    png_paths = sorted((output / "figures").rglob("*.png"))
    svg_paths = sorted((output / "figures").rglob("*.svg"))
    pdf_paths = sorted(output.rglob("*.pdf"))
    expected_names = _expected_figure_relative_files()
    expected_figure_count = len(expected_names) // 2
    actual_names = {
        path.relative_to(output).as_posix() for path in png_paths + svg_paths
    }
    png_checks: dict[str, Any] = {}
    png_passed = len(png_paths) == expected_figure_count
    for path in png_paths:
        relative = path.relative_to(output).as_posix()
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0.0, 0.0))
            width, height = image.size
        item_passed = bool(
            path.stat().st_size > 10_000
            and width >= 2000
            and height >= 1500
            and 590.0 <= float(dpi[0]) <= 610.0
            and 590.0 <= float(dpi[1]) <= 610.0
        )
        png_checks[relative] = {
            "size_bytes": path.stat().st_size,
            "pixel_dimensions": [width, height],
            "dpi": [float(dpi[0]), float(dpi[1])],
            "passed": item_passed,
        }
        png_passed = png_passed and item_passed
    svg_checks: dict[str, Any] = {}
    svg_passed = len(svg_paths) == expected_figure_count
    for path in svg_paths:
        relative = path.relative_to(output).as_posix()
        is_2d_map = relative.startswith(
            f"figures/{PZC_2D_OUTPUT_SUBDIR.as_posix()}/"
        )
        ET.parse(path)
        text = path.read_text(encoding="utf-8")
        item_passed = bool(
            path.stat().st_size > 5_000
            and "<text" in text
            and "Helvetica" in text
            and (("<image" in text) if is_2d_map else ("<image" not in text))
        )
        svg_checks[relative] = {
            "size_bytes": path.stat().st_size,
            "editable_text": "<text" in text,
            "helvetica_first_stack_present": "Helvetica" in text,
            "rasterized_2d_field_expected": is_2d_map,
            "embedded_raster_present": "<image" in text,
            "passed": item_passed,
        }
        svg_passed = svg_passed and item_passed
    profile_csv_count = len(list((output / "csv" / "profiles").glob("*.csv")))
    two_d_summary_path = output / "csv" / "support_pzc_2d_summary.csv"
    two_d_summary_count = (
        len(_read_csv(two_d_summary_path)) if two_d_summary_path.is_file() else 0
    )
    passed = bool(
        actual_names == expected_names
        and png_passed
        and svg_passed
        and not pdf_paths
        and profile_csv_count == 12
        and two_d_summary_count == 12
    )
    return {
        "figure_counts": {"png": len(png_paths), "svg": len(svg_paths), "pdf": len(pdf_paths)},
        "expected_figure_names": sorted(expected_names),
        "actual_figure_names": sorted(actual_names),
        "png": png_checks,
        "svg": svg_checks,
        "representative_profile_csv_count": profile_csv_count,
        "expected_representative_profile_csv_count": 12,
        "representative_2d_summary_csv_count": two_d_summary_count,
        "expected_representative_2d_summary_csv_count": 12,
        "svg_policy": (
            "Line/profile SVGs contain vector graphics and editable text. "
            "The 12 potential-only maps and 12 potential-plus-reactant composites "
            "rasterize only their dense pcolormesh fields while retaining editable "
            "SVG text and vector annotations."
        ),
        "passed": passed,
    }


def _write_checksums(output: Path) -> dict[str, str]:
    paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != CHECKSUM_FILE
    )
    checksums = {
        path.relative_to(output).as_posix(): _sha256(path) for path in paths
    }
    (output / CHECKSUM_FILE).write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in checksums.items()),
        encoding="utf-8",
    )
    return checksums


def _verify_checksums(output: Path) -> dict[str, str]:
    checksum_path = output / CHECKSUM_FILE
    if not checksum_path.is_file():
        raise RuntimeError("Missing support-PZC checksums.sha256")
    listed: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if relative in listed:
            raise RuntimeError(f"Duplicate checksum path: {relative}")
        target = output / relative
        if not target.is_file() or _sha256(target) != expected:
            raise RuntimeError(f"Support-PZC checksum mismatch: {relative}")
        listed[relative] = expected
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != CHECKSUM_FILE
    }
    if set(listed) != actual:
        raise RuntimeError("Support-PZC checksum coverage does not match output tree")
    return listed


def _scan_config() -> dict[str, Any]:
    return {
        "study_id": STUDY_ID,
        "main_L_support_nm": list(MAIN_LENGTHS_NM),
        "pzc_support_V": {
            "start": 0.10,
            "stop": 0.90,
            "count": 41,
            "values": list(PZC_VALUES_V),
        },
        "representative_pzc_support_V": list(REPRESENTATIVE_PZC_V),
        "far_field_anchor": {
            "L_support_nm": ANCHOR_LENGTH_NM,
            "pzc_support_V": list(REPRESENTATIVE_PZC_V),
        },
        "negative_control": {
            "L_support_nm": NEGATIVE_CONTROL_LENGTH_NM,
            "pzc_support_V": [0.10, 0.90],
        },
        "overlap_definition": (
            "Equal-active-length RMS of Au and Pd material-mean reaction-plane "
            "potential differences relative to L_support=15 nm at the same support PZC."
        ),
        "trend_figure_metrics": [
            "E_mix_with_V",
            "i_mix_avg_with_A_per_m2",
            "sigma_support_signed_mean_uC_per_cm2",
        ],
        "overlap_metric_visibility": (
            "Retained in CSV and validation metadata; omitted from the trend figure."
        ),
        "numerics": {
            "original_requested_mode_pairs": {
                "short": {
                    "low_N_modes": REQUESTED_SHORT_LOW_MODES,
                    "high_N_modes": REQUESTED_SHORT_HIGH_MODES,
                },
                "1000nm": {
                    "low_N_modes": REQUESTED_LONG_LOW_MODES,
                    "high_N_modes": REQUESTED_LONG_HIGH_MODES,
                },
            },
            "production_N_modes_by_L_support_nm": {
                f"{length:g}": modes
                for length, modes in PRODUCTION_MODES_BY_LENGTH.items()
            },
            "production_convergence_low_N_modes_by_L_support_nm": {
                f"{length:g}": modes
                for length, modes in CONVERGENCE_LOW_MODES_BY_LENGTH.items()
            },
            "production_GL_order": HIGH_GL_ORDER,
            "convergence_GL_order": LOW_GL_ORDER,
            "mode_and_GL_thresholds": {
                "short": {"E_mix_mV": 0.002, "i_mix_percent": 0.005},
                "1000nm": {"E_mix_mV": 0.03, "i_mix_percent": 0.1},
            },
        },
        "model": "FULL linear-PB piecewise-Robin with absolute-current balance",
        "resolution_policy": (
            "The original requested mode pairs are preserved in a separate audit. "
            "Production modes are raised only where that audit exceeds the stated "
            "E_mix or i_mix convergence tolerance."
        ),
        "two_dimensional_solution_potential": {
            "main_scan_include_2d": False,
            "representative_cases_include_2d": True,
            "L_support_nm": list(MAIN_LENGTHS_NM),
            "pzc_support_V": list(REPRESENTATIVE_PZC_V),
            "case_count": len(MAIN_LENGTHS_NM) * len(REPRESENTATIVE_PZC_V),
            "Ny": engine.NY_2D,
            "y_max_lambda_D": engine.Y_MAX_LAMBDA_D,
            "color_scale": "shared symmetric TwoSlopeNorm centered at 0 mV",
            "dense_field_svg_policy": (
                "Rasterized pcolormesh with editable SVG text and vector annotations"
            ),
        },
        "two_dimensional_reactants": {
            "source_cases": "same 12 representative 2-D solution-potential fields",
            "reactant_definition": "c_i/c_bulk = exp(-z_i F Phi_s / RT)",
            "reactant_color_scale": (
                "one shared reciprocal LogNorm for Red1 and Ox2 across all cases"
            ),
            "species": ["Red1", "Ox2"],
            "case_count": len(MAIN_LENGTHS_NM) * len(REPRESENTATIVE_PZC_V),
            "output": "solution-phase-potential-plus-Red1-plus-Ox2 composite",
            "dense_field_svg_policy": (
                "Rasterized pcolormesh with editable SVG text and vector annotations"
            ),
        },
    }


def _locked_overrides() -> dict[str, Any]:
    return {
        "L_Au_m": engine.L_AU_M,
        "L_Pd_m": engine.L_PD_M,
        "C_H_Au_F_per_m2": engine.C_H_AU_F_PER_M2,
        "C_H_support_F_per_m2": engine.C_H_SUPPORT_F_PER_M2,
        "C_H_Pd_F_per_m2": engine.C_H_PD_F_PER_M2,
        "PZC_Au_SHE_V": engine.PZC_AU_SHE_V,
        "PZC_Pd_SHE_V": engine.PZC_PD_SHE_V,
        "PZC_Au_RHE_V": engine.PZC_AU_RHE_V,
        "PZC_Pd_RHE_V": engine.PZC_PD_RHE_V,
        "PZC_Au_minus_PZC_Pd_V": engine.PZC_AU_RHE_V - engine.PZC_PD_RHE_V,
        "i0_Au_A_per_m2": engine.I0_EQUAL_A_PER_M2,
        "i0_Pd_A_per_m2": engine.I0_EQUAL_A_PER_M2,
        "alpha_Au": engine.ALPHA_EQUAL,
        "alpha_Pd": engine.ALPHA_EQUAL,
        "out_of_plane_width_m": engine.OUT_OF_PLANE_WIDTH_M,
        "C_tot_mol_per_m3": 10.0,
        "C_tot_mM": 10.0,
    }


def _expected_relative_files() -> set[str]:
    files = {
        *_expected_figure_relative_files(),
        "csv/support_pzc_scan.csv",
        "csv/support_pzc_1000nm_anchors.csv",
        "csv/support_pzc_zero_length_negative_control.csv",
        "csv/support_pzc_convergence.csv",
        "csv/support_pzc_original_resolution_audit.csv",
        "csv/support_pzc_2d_summary.csv",
        "inputs/baseline_params.json",
        "inputs/locked_overrides.json",
        "inputs/scan_config.json",
        "summary.json",
        "validation.json",
        "manifest.json",
    }
    files.update(
        f"csv/profiles/{_profile_stem(length_nm, pzc_v)}.csv"
        for length_nm in MAIN_LENGTHS_NM
        for pzc_v in REPRESENTATIVE_PZC_V
    )
    return files


def build_support_pzc_study(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".support-pzc-staging-", dir=output.parent))
    try:
        _configure_style()
        rows, representatives, representative_low_gl_cases = _build_main_scan()
        representative_2d_cases = _build_representative_2d_cases()
        shared_2d_vlim_mV = _shared_solution_potential_vlim_mV(
            representative_2d_cases
        )
        shared_reactant_vmin, shared_reactant_vmax = _shared_reactant_log_limits(
            representative_2d_cases
        )
        representative_2d_rows = _representative_2d_rows(
            representative_2d_cases,
            representatives,
            shared_2d_vlim_mV,
            shared_reactant_vmin,
            shared_reactant_vmax,
        )
        anchor_rows, anchors, anchor_low_gl_cases = _build_anchors(representatives)
        negative_rows, _negative_cases = _build_negative_controls()
        convergence_rows = _run_convergence(
            representatives,
            representative_low_gl_cases,
            anchors,
            anchor_low_gl_cases,
        )
        requested_resolution_rows, requested_short_high_cases = (
            _run_requested_resolution_audit()
        )
        regression_rows = [
            _case_metrics(requested_short_high_cases[(length_nm, 0.50)])
            for length_nm in MAIN_LENGTHS_NM
        ]

        _write_csv(staging / "csv" / "support_pzc_scan.csv", rows)
        _write_csv(staging / "csv" / "support_pzc_1000nm_anchors.csv", anchor_rows)
        _write_csv(
            staging / "csv" / "support_pzc_zero_length_negative_control.csv",
            negative_rows,
        )
        _write_csv(staging / "csv" / "support_pzc_convergence.csv", convergence_rows)
        _write_csv(
            staging / "csv" / "support_pzc_original_resolution_audit.csv",
            requested_resolution_rows,
        )
        _write_csv(
            staging / "csv" / "support_pzc_2d_summary.csv",
            representative_2d_rows,
        )
        profile_files = _write_representative_profiles(staging, representatives)
        baseline_params = engine.load_baseline_params()
        baseline_params.update(
            {
                "pzc_Au": engine.PZC_AU_RHE_V,
                "pzc_Pd": engine.PZC_PD_RHE_V,
            }
        )
        _write_json(staging / "inputs" / "baseline_params.json", baseline_params)
        _write_json(staging / "inputs" / "locked_overrides.json", _locked_overrides())
        _write_json(staging / "inputs" / "scan_config.json", _scan_config())

        figure_dir = staging / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        figure_paths = []
        figure_paths.extend(_plot_trends(rows, anchor_rows, figure_dir))
        figure_paths.extend(_plot_phi_profiles(representatives, figure_dir))
        figure_paths.extend(_plot_sigma_profiles(representatives, figure_dir))
        figure_paths.extend(
            _plot_representative_solution_potentials_2d(
                representative_2d_cases,
                figure_dir,
                shared_2d_vlim_mV,
            )
        )
        figure_paths.extend(
            _plot_representative_phi_s_reactants_2d(
                representative_2d_cases,
                figure_dir,
                shared_2d_vlim_mV,
                shared_reactant_vmin,
                shared_reactant_vmax,
            )
        )

        numeric = _numeric_validation(
            rows,
            anchor_rows,
            negative_rows,
            convergence_rows,
            requested_resolution_rows,
            regression_rows,
            representatives,
        )
        representative_2d_validation = _validate_representative_2d(
            representative_2d_rows
        )
        dh_max = max(float(row["max_abs_phi_tilde_with_edl"]) for row in rows)
        dh_failures = sum(not bool(row["debye_huckel_ok_with_edl"]) for row in rows)
        summary = {
            "study_id": STUDY_ID,
            "model": "Au|C|Pd linear-PB piecewise-Robin",
            "parameters": _locked_overrides(),
            "scan_point_count": len(rows),
            "representative_profile_count": len(profile_files),
            "representative_2d_case_count": len(representative_2d_rows),
            "far_field_anchor_count": len(anchor_rows),
            "negative_control_count": len(negative_rows),
            "convergence_check_count": len(convergence_rows),
            "original_requested_resolution_audit_count": len(
                requested_resolution_rows
            ),
            "original_requested_resolution_failed_pair_count": numeric[
                "original_requested_resolution_audit"
            ]["failed_pair_count"],
            "figure_counts": {
                "png": len(figure_paths) // 2,
                "svg": len(figure_paths) // 2,
                "pdf": 0,
            },
            "trend_figure_panels": [
                "Mixed potential",
                "Mixed current density",
                "Mean signed support charge",
            ],
            "overlap_metric_output": (
                "Retained in support_pzc_scan.csv and validation metadata; "
                "not displayed in support_pzc_trends_au2_pd2."
            ),
            "representative_2d_solution_potential": {
                "L_support_nm": list(MAIN_LENGTHS_NM),
                "pzc_support_V": list(REPRESENTATIVE_PZC_V),
                "shared_symmetric_color_limit_mV": shared_2d_vlim_mV,
                "y_max_lambda_D": engine.Y_MAX_LAMBDA_D,
                "output_subdirectory": (
                    f"figures/{PZC_2D_OUTPUT_SUBDIR.as_posix()}"
                ),
            },
            "representative_2d_reactant_distributions": {
                "definition": "c_i/c_bulk = exp(-z_i F Phi_s / RT)",
                "species": ["Red1", "Ox2"],
                "shared_log_color_limits": [
                    shared_reactant_vmin,
                    shared_reactant_vmax,
                ],
                "case_count": len(representative_2d_rows),
                "output_subdirectory": (
                    f"figures/{PZC_2D_OUTPUT_SUBDIR.as_posix()}"
                ),
            },
            "overlap_reference": (
                "15 nm is a finite-length reference, not a universal no-overlap boundary; "
                "1000 nm anchors quantify its residual distance from the far-field limit."
            ),
            "reference_15nm_vs_1000nm": numeric["reference_15nm_vs_1000nm"],
            "debye_huckel_applicability": {
                "maximum_abs_phi_tilde_in_main_scan": dh_max,
                "main_scan_points_above_weak_field_threshold": dh_failures,
                "threshold": 1.0,
                "note": (
                    "Applicability warnings are retained separately from root balance "
                    "and numerical-convergence validation."
                ),
            },
            "fourier_caveat": (
                "Piecewise boundaries are represented by a truncated cosine series; "
                "boundary-local ringing must not be interpreted as a physical oscillation."
            ),
            "visible_material_colors": MATERIAL_COLORS,
            "profile_pzc_styles": {
                f"{pzc:.2f} V": {
                    "color": PROFILE_STYLES[pzc]["color"],
                    "linestyle": str(PROFILE_STYLES[pzc]["linestyle"]),
                }
                for pzc in REPRESENTATIVE_PZC_V
            },
        }
        _write_json(staging / "summary.json", summary)

        artifacts = _verify_artifacts(staging)
        validation = {
            "study_id": STUDY_ID,
            "numerical": numeric,
            "representative_2d": representative_2d_validation,
            "artifacts": artifacts,
            "applicability_warnings": summary["debye_huckel_applicability"],
            "passed": bool(
                numeric["passed"]
                and representative_2d_validation["passed"]
                and artifacts["passed"]
            ),
        }
        if not validation["passed"]:
            raise RuntimeError(f"Support-PZC study validation failed: {validation}")
        _write_json(staging / "validation.json", validation)

        manifest = {
            "study_id": STUDY_ID,
            "source_sha256": _source_hashes(),
            "figure_files": sorted(path.relative_to(staging).as_posix() for path in figure_paths),
            "data_files": sorted(
                [
                    "csv/support_pzc_scan.csv",
                    "csv/support_pzc_1000nm_anchors.csv",
                    "csv/support_pzc_zero_length_negative_control.csv",
                    "csv/support_pzc_convergence.csv",
                    "csv/support_pzc_original_resolution_audit.csv",
                    "csv/support_pzc_2d_summary.csv",
                    *profile_files,
                ]
            ),
            "input_files": [
                "inputs/baseline_params.json",
                "inputs/locked_overrides.json",
                "inputs/scan_config.json",
            ],
            "metadata_files": ["summary.json", "validation.json", "manifest.json"],
            "formats": ["png", "svg"],
            "png_dpi": 600,
            "svg_text_editable": True,
            "pdf_enabled": False,
        }
        _write_json(staging / "manifest.json", manifest)
        actual_files = {
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file()
        }
        expected_files = _expected_relative_files()
        if actual_files != expected_files:
            raise RuntimeError(
                "Support-PZC output coverage mismatch before checksum: "
                f"missing={sorted(expected_files-actual_files)}, "
                f"extra={sorted(actual_files-expected_files)}"
            )
        checksums = _write_checksums(staging)
        _verify_checksums(staging)
        staging.rename(output)
    except Exception:
        if (
            staging.exists()
            and staging.parent == output.parent
            and staging.name.startswith(".support-pzc-staging-")
        ):
            shutil.rmtree(staging)
        raise
    return {
        "output": str(output),
        "output_dir": output,
        "reused": False,
        "summary": summary,
        "validation": validation,
        "manifest": manifest,
        "checksums": checksums,
    }


def validate_existing_support_pzc_study(
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    if not output.is_dir():
        raise FileNotFoundError(f"Missing support-PZC study: {output}")
    actual_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != CHECKSUM_FILE
    }
    expected_files = _expected_relative_files()
    if actual_files != expected_files:
        raise RuntimeError(
            "Existing support-PZC output coverage mismatch: "
            f"missing={sorted(expected_files-actual_files)}, "
            f"extra={sorted(actual_files-expected_files)}"
        )
    summary = _read_json(output / "summary.json")
    validation = _read_json(output / "validation.json")
    manifest = _read_json(output / "manifest.json")
    if summary.get("study_id") != STUDY_ID or manifest.get("study_id") != STUDY_ID:
        raise RuntimeError("Existing support-PZC study identifier is invalid")
    if not bool(validation.get("passed")):
        raise RuntimeError("Existing support-PZC validation.json is not passed")
    if manifest.get("source_sha256") != _source_hashes():
        raise RuntimeError("Existing support-PZC source hashes do not match current code")
    artifacts = _verify_artifacts(output)
    if not artifacts["passed"]:
        raise RuntimeError(f"Existing support-PZC figure validation failed: {artifacts}")
    checksums = _verify_checksums(output)
    return {
        "output": str(output),
        "output_dir": output,
        "reused": True,
        "summary": summary,
        "validation": validation,
        "manifest": manifest,
        "checksums": checksums,
    }


def build_or_reuse_support_pzc_study(
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        return validate_existing_support_pzc_study(output)
    return build_support_pzc_study(output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="New or reusable study output directory (default: %(default)s)",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = build_or_reuse_support_pzc_study(args.output)
    summary = result["summary"]
    print(f"output = {result['output']}")
    print(f"reused = {result['reused']}")
    print(
        f"scan = {summary['scan_point_count']} points; "
        f"profiles = {summary['representative_profile_count']}; "
        f"figures = {summary['figure_counts']['png']} PNG + "
        f"{summary['figure_counts']['svg']} SVG; "
        f"PDF = {summary['figure_counts']['pdf']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
