from __future__ import annotations

import copy
import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import to_rgb


ROOT = Path(__file__).resolve().parents[3]
SOLVER_DIR = ROOT / "Mixed_Potential_Electrical_Double_Layer"
HEATMAP_INPUTS_DIR = ROOT / "Figures" / "Figure_5" / "Heatmap" / "inputs"
OUT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = OUT_DIR / "inputs"
CSV_DIR = OUT_DIR / "csv"

OUTPUT_TAG = "same_length_i0_alpha050_au25_pd25_20260528_111255"
PARAMS_PATH = HEATMAP_INPUTS_DIR / f"params_{OUTPUT_TAG}.json"
SUMMARY_PATH = HEATMAP_INPUTS_DIR / f"summary_compare_{OUTPUT_TAG}.json"

CH_PNG_OUT = OUT_DIR / f"edl_scheme_CH_phi_rp_emix_{OUTPUT_TAG}.png"
CH_SVG_OUT = OUT_DIR / f"edl_scheme_CH_phi_rp_emix_{OUTPUT_TAG}.svg"
CH_VECTOR_SVG_OUT = OUT_DIR / f"edl_scheme_CH_phi_rp_emix_{OUTPUT_TAG}_vector.svg"
CH_ILLUSTRATOR_SVG_OUT = OUT_DIR / f"edl_scheme_CH_phi_rp_emix_{OUTPUT_TAG}_illustrator_safe.svg"
PZC_PNG_OUT = OUT_DIR / f"edl_scheme_PZC_phi_rp_emix_{OUTPUT_TAG}.png"
PZC_SVG_OUT = OUT_DIR / f"edl_scheme_PZC_phi_rp_emix_{OUTPUT_TAG}.svg"
PZC_VECTOR_SVG_OUT = OUT_DIR / f"edl_scheme_PZC_phi_rp_emix_{OUTPUT_TAG}_vector.svg"
PZC_ILLUSTRATOR_SVG_OUT = OUT_DIR / f"edl_scheme_PZC_phi_rp_emix_{OUTPUT_TAG}_illustrator_safe.svg"
CASE_CSV_OUT = CSV_DIR / f"edl_scheme_case_summary_{OUTPUT_TAG}.csv"
PROFILE_CSV_OUT = CSV_DIR / f"edl_scheme_phi_profiles_{OUTPUT_TAG}.csv"
CONFIG_OUT = INPUTS_DIR / f"edl_scheme_config_{OUTPUT_TAG}.json"

BASELINE_ATOL = 5.0e-10
N_DISTANCE = 180
DISTANCE_LAMBDA_MAX = 5.0
METAL_X0_NM = -2.35
METAL_X1_NM = -1.18
RP_X_NM = 0.0

sys.path.insert(0, str(SOLVER_DIR))
import Solve_Emix_updating as solver  # noqa: E402


COLORS = {
    "au": "#F2B134",
    "au_dark": "#B64342",
    "pd": "#5A90C8",
    "pd_dark": "#0F4D92",
    "electrolyte": "#EAF5FA",
    "bulk": "#DDF3DE",
    "inner_layer": "#F4F4F4",
    "dark": "#272727",
    "gray": "#767676",
    "light_gray": "#CFCECE",
    "low": "#767676",
    "base": "#111111",
    "high": "#0F4D92",
}

GRADIENT_COLORS = (COLORS["electrolyte"], "#F4FBF8", COLORS["bulk"])


@dataclass(frozen=True)
class ScanSpec:
    group: str
    key: str
    side: str
    title: str
    low: float
    baseline: float
    high: float
    display_factor: float
    display_unit: str
    display_fmt: str
    trend_text: str


SCAN_SPECS = (
    ScanSpec(
        group="CH",
        key="Cdl_Au",
        side="Au",
        title=r"$C_{\mathrm{H},\mathrm{Au}}$",
        low=0.05,
        baseline=0.20,
        high=1.00,
        display_factor=100.0,
        display_unit=r"$\mu$F cm$^{-2}$",
        display_fmt="{:.0f}",
        trend_text=r"higher $C_{\mathrm{H}}$: more negative $\phi_{\mathrm{RP}}$",
    ),
    ScanSpec(
        group="CH",
        key="Cdl_Pd",
        side="Pd",
        title=r"$C_{\mathrm{H},\mathrm{Pd}}$",
        low=0.05,
        baseline=0.40,
        high=1.00,
        display_factor=100.0,
        display_unit=r"$\mu$F cm$^{-2}$",
        display_fmt="{:.0f}",
        trend_text=r"higher $C_{\mathrm{H}}$: more negative $\phi_{\mathrm{RP}}$",
    ),
    ScanSpec(
        group="PZC",
        key="pzc_Au",
        side="Au",
        title=r"$\mathrm{PZC}_{\mathrm{Au}}$",
        low=0.63,
        baseline=0.93,
        high=1.23,
        display_factor=1.0,
        display_unit="V",
        display_fmt="{:.2f}",
        trend_text=r"higher PZC: more negative $\phi_{\mathrm{RP}}$",
    ),
    ScanSpec(
        group="PZC",
        key="pzc_Pd",
        side="Pd",
        title=r"$\mathrm{PZC}_{\mathrm{Pd}}$",
        low=0.48,
        baseline=0.78,
        high=1.08,
        display_factor=1.0,
        display_unit="V",
        display_fmt="{:.2f}",
        trend_text=r"higher PZC: more negative $\phi_{\mathrm{RP}}$",
    ),
)

CASE_VALUES = ("low", "base", "high")
CASE_STYLES = {
    "low": {"color": COLORS["low"], "linestyle": (0, (3, 2)), "linewidth": 1.45},
    "base": {"color": COLORS["base"], "linestyle": "-", "linewidth": 1.85},
    "high": {"color": COLORS["high"], "linestyle": (0, (5, 2)), "linewidth": 1.65},
}
EXPECTED_PNGS = (CH_PNG_OUT, PZC_PNG_OUT)
EXPECTED_SVGS = (
    CH_SVG_OUT,
    CH_VECTOR_SVG_OUT,
    CH_ILLUSTRATOR_SVG_OUT,
    PZC_SVG_OUT,
    PZC_VECTOR_SVG_OUT,
    PZC_ILLUSTRATOR_SVG_OUT,
)
SVG_NS = "http://www.w3.org/2000/svg"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 8.4,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "legend.frameon": False,
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
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def clean_style_attribute(style: str) -> str:
    text_only_keys = {
        "font-family",
        "font-size",
        "font-stretch",
        "font-style",
        "font-variant",
        "font-weight",
        "text-anchor",
    }
    parts: list[str] = []
    for item in style.split(";"):
        item = item.strip()
        if not item:
            continue
        key = item.split(":", 1)[0].strip()
        if key.startswith("-inkscape-") or key in text_only_keys:
            continue
        parts.append(item)
    return ";".join(parts)


def clean_svg_for_illustrator(tmp_svg: Path, out_svg: Path) -> None:
    ET.register_namespace("", SVG_NS)
    tree = ET.parse(tmp_svg)
    root = tree.getroot()

    def recurse(parent: ET.Element) -> None:
        for child in list(parent):
            if local_name(child.tag) in {"metadata", "defs"}:
                parent.remove(child)
                continue
            recurse(child)

        for attr in list(parent.attrib):
            attr_name = local_name(attr)
            value = parent.attrib[attr]
            if attr_name == "clip-path" or attr_name.startswith("aria-"):
                del parent.attrib[attr]
            elif attr_name == "style":
                cleaned = clean_style_attribute(value)
                if cleaned:
                    parent.attrib[attr] = cleaned
                else:
                    del parent.attrib[attr]

    recurse(root)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_svg, encoding="utf-8", xml_declaration=True)


def make_illustrator_safe_svg(source_svg: Path, out_svg: Path) -> None:
    inkscape = shutil.which("inkscape")
    if inkscape is None:
        raise RuntimeError("Inkscape is required to write Illustrator-safe SVG outputs")

    tmp_svg = out_svg.with_name(f"{out_svg.stem}.tmp.svg")
    actions = (
        "select-all;"
        "clone-unlink-recursively;"
        "object-unlink-clones;"
        "object-to-path;"
        "object-stroke-to-path;"
        f"export-filename:{tmp_svg};"
        "export-plain-svg;"
        "export-text-to-path;"
        "export-do"
    )
    try:
        subprocess.run(
            [inkscape, str(source_svg), f"--actions={actions}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        clean_svg_for_illustrator(tmp_svg, out_svg)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to write Illustrator-safe SVG for {source_svg}") from exc
    finally:
        tmp_svg.unlink(missing_ok=True)


def remove_stale_illustrator_svgs() -> None:
    for path in OUT_DIR.glob(f"edl_scheme_*_{OUTPUT_TAG}_illustrator*.svg"):
        if path not in {CH_ILLUSTRATOR_SVG_OUT, PZC_ILLUSTRATOR_SVG_OUT}:
            path.unlink(missing_ok=True)


def assert_close(label: str, actual: float, expected: float, atol: float) -> None:
    if abs(actual - expected) > atol:
        raise ValueError(f"{label}: recomputed {actual:.15g} differs from stored {expected:.15g}")


def validate_baseline(params: dict[str, Any], summary: dict[str, Any]) -> None:
    res_edl = solver.run_case(params, mode="FULL", return_profiles=False, use_edl=True)
    res_no = solver.run_case(params, mode="FULL", return_profiles=False, use_edl=False)
    assert_close("E_mix_with", float(res_edl["E_mix"]), float(summary["E_mix_with"]), BASELINE_ATOL)
    assert_close(
        "i_mix_avg_with",
        float(res_edl["i_mix_avg_A_per_m2"]),
        float(summary["i_mix_avg_with"]),
        BASELINE_ATOL,
    )
    assert_close("E_mix_no", float(res_no["E_mix"]), float(summary["E_mix_no"]), BASELINE_ATOL)
    assert_close(
        "i_mix_avg_no",
        float(res_no["i_mix_avg_A_per_m2"]),
        float(summary["i_mix_avg_no"]),
        BASELINE_ATOL,
    )


def case_value(spec: ScanSpec, case_name: str) -> float:
    if case_name == "low":
        return spec.low
    if case_name == "base":
        return spec.baseline
    if case_name == "high":
        return spec.high
    raise ValueError(f"Unsupported case: {case_name}")


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def segment_mean(values: np.ndarray, x_tilde: np.ndarray, mask: np.ndarray, segment_length_tilde: float) -> float:
    vals = np.asarray(values, dtype=float)[mask]
    x_seg = np.asarray(x_tilde, dtype=float)[mask]
    if vals.size == 0 or segment_length_tilde <= 0.0:
        return float("nan")
    if vals.size == 1:
        return float(vals[0])
    return trapz(vals, x_seg) / float(segment_length_tilde)


def side_segment_length_tilde(derived: dict[str, Any], side: str) -> float:
    if side == "Au":
        return float(derived["L_Au_tilde"])
    if side == "Pd":
        return float(derived["L_tilde"] - derived["L_C_tilde"])
    raise ValueError(f"Unsupported side: {side}")


def side_coefficients(edl: Any, side: str) -> np.ndarray:
    if side == "Au":
        return np.asarray(edl.pre["c_Au"], dtype=float)
    if side == "Pd":
        return np.asarray(edl.pre["c_Pd"], dtype=float)
    raise ValueError(f"Unsupported side: {side}")


def side_mask(result: dict[str, Any], side: str) -> np.ndarray:
    if side == "Au":
        return np.asarray(result["mask_Au"], dtype=bool)
    if side == "Pd":
        return np.asarray(result["mask_Pd"], dtype=bool)
    raise ValueError(f"Unsupported side: {side}")


def side_color(side: str) -> str:
    return COLORS["au"] if side == "Au" else COLORS["pd"]


def side_pzc(params: dict[str, Any], side: str) -> float:
    if side == "Au":
        return float(params["pzc_Au"])
    if side == "Pd":
        return float(params["pzc_Pd"])
    raise ValueError(f"Unsupported side: {side}")


def build_case(base_params: dict[str, Any], spec: ScanSpec, case_name: str) -> dict[str, Any]:
    params = copy.deepcopy(base_params)
    value = case_value(spec, case_name)
    params[spec.key] = float(value)

    result = solver.run_case(params, mode="FULL", return_profiles=True, use_edl=True)
    edl = solver.EDLModel(params)
    derived = edl.derived
    e_mix = float(result["E_mix"])
    beta = float(derived["beta"])
    scale = float(derived["R"]) * float(derived["T"]) / float(derived["F"])

    distance_tilde = np.linspace(0.0, DISTANCE_LAMBDA_MAX, N_DISTANCE, dtype=float)
    distance_nm = distance_tilde * float(derived["lambda_D"]) * 1.0e9
    gamma = np.asarray(edl.pre["gamma"], dtype=float)
    coeffs = side_coefficients(edl, spec.side)
    amplitude = np.asarray(edl.pre["A_M"], dtype=float) * beta * e_mix - np.asarray(edl.pre["A_pzc"], dtype=float)
    phi_tilde_profile = (coeffs * amplitude) @ np.exp(-np.outer(gamma, distance_tilde))
    phi_profile_mV = 1000.0 * scale * np.asarray(phi_tilde_profile, dtype=float)

    phi_side_affine = edl.segment_mean_phi2(e_mix, use_affine_phi2=False)
    side_index = 0 if spec.side == "Au" else 1
    phi_rp_mean_v = float(phi_side_affine[side_index])

    x_tilde = np.asarray(result["x_tilde"], dtype=float)
    phi_surface_v = scale * np.asarray(result["phi_tilde"], dtype=float)
    phi_rp_profile_mean_v = segment_mean(
        phi_surface_v,
        x_tilde,
        side_mask(result, spec.side),
        side_segment_length_tilde(derived, spec.side),
    )
    if not math.isclose(float(phi_profile_mV[0]) / 1000.0, phi_rp_mean_v, rel_tol=0.0, abs_tol=1e-11):
        raise ValueError(f"{spec.key} {case_name}: y=0 mode profile does not match segment mean")
    if not math.isclose(phi_rp_profile_mean_v, phi_rp_mean_v, rel_tol=0.0, abs_tol=2e-6):
        raise ValueError(f"{spec.key} {case_name}: surface profile mean does not match segment mean")

    return {
        "group": spec.group,
        "scan": spec.key,
        "side": spec.side,
        "case": case_name,
        "parameter_value_SI": float(value),
        "parameter_value_display": spec.display_factor * float(value),
        "parameter_display_unit": spec.display_unit,
        "E_mix_V": e_mix,
        "pzc_side_V": side_pzc(params, spec.side),
        "metal_minus_pzc_mV": 1000.0 * (e_mix - side_pzc(params, spec.side)),
        "i_mix_avg_A_per_m2": float(result["i_mix_avg_A_per_m2"]),
        "i_mix_abs_A": float(result["i_mix_abs_A"]),
        "phi_RP_mean_V": phi_rp_mean_v,
        "phi_RP_mean_mV": 1000.0 * phi_rp_mean_v,
        "phi_RP_profile_mean_V": phi_rp_profile_mean_v,
        "lambda_D_nm": float(derived["lambda_D"]) * 1.0e9,
        "distance_nm": distance_nm,
        "distance_over_lambda_D": distance_tilde,
        "phi_profile_mV": phi_profile_mV,
        "max_abs_phi_tilde": float(result["max_abs_phi_tilde"]),
        "debye_huckel_ok": bool(result["debye_huckel_ok"]),
    }


def build_all_cases(params: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    cases: dict[str, list[dict[str, Any]]] = {}
    for spec in SCAN_SPECS:
        cases[spec.key] = [build_case(params, spec, case_name) for case_name in CASE_VALUES]
    return cases


def require_finite_cases(cases: dict[str, list[dict[str, Any]]]) -> None:
    expected = len(SCAN_SPECS) * len(CASE_VALUES)
    actual = sum(len(values) for values in cases.values())
    if actual != expected:
        raise RuntimeError(f"Expected {expected} cases, got {actual}")
    scalar_keys = [
        "parameter_value_SI",
        "parameter_value_display",
        "E_mix_V",
        "pzc_side_V",
        "metal_minus_pzc_mV",
        "i_mix_avg_A_per_m2",
        "i_mix_abs_A",
        "phi_RP_mean_V",
        "phi_RP_mean_mV",
        "phi_RP_profile_mean_V",
        "lambda_D_nm",
        "max_abs_phi_tilde",
    ]
    array_keys = ["distance_nm", "distance_over_lambda_D", "phi_profile_mV"]
    for spec in SCAN_SPECS:
        if spec.key not in cases:
            raise RuntimeError(f"Missing cases for {spec.key}")
        for case in cases[spec.key]:
            for key in scalar_keys:
                if not np.isfinite(float(case[key])):
                    raise RuntimeError(f"{spec.key} {case['case']} has non-finite {key}")
            for key in array_keys:
                values = np.asarray(case[key], dtype=float)
                if values.size != N_DISTANCE:
                    raise RuntimeError(f"{spec.key} {case['case']} {key} has {values.size} points")
                if not np.all(np.isfinite(values)):
                    raise RuntimeError(f"{spec.key} {case['case']} has non-finite {key}")


def write_case_csv(cases: dict[str, list[dict[str, Any]]]) -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "scan",
        "side",
        "case",
        "parameter_value_SI",
        "parameter_value_display",
        "parameter_display_unit",
        "E_mix_V",
        "pzc_side_V",
        "metal_minus_pzc_mV",
        "i_mix_avg_A_per_m2",
        "i_mix_abs_A",
        "phi_RP_mean_V",
        "phi_RP_mean_mV",
        "phi_RP_profile_mean_V",
        "lambda_D_nm",
        "max_abs_phi_tilde",
        "debye_huckel_ok",
    ]
    with CASE_CSV_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for spec in SCAN_SPECS:
            for case in cases[spec.key]:
                writer.writerow({key: case[key] for key in fieldnames})


def write_profile_csv(cases: dict[str, list[dict[str, Any]]]) -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "scan",
        "side",
        "case",
        "parameter_value_display",
        "distance_nm",
        "distance_over_lambda_D",
        "phi_profile_mV",
    ]
    with PROFILE_CSV_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for spec in SCAN_SPECS:
            for case in cases[spec.key]:
                distance_nm = np.asarray(case["distance_nm"], dtype=float)
                distance_tilde = np.asarray(case["distance_over_lambda_D"], dtype=float)
                phi = np.asarray(case["phi_profile_mV"], dtype=float)
                for idx in range(distance_nm.size):
                    writer.writerow(
                        {
                            "group": case["group"],
                            "scan": case["scan"],
                            "side": case["side"],
                            "case": case["case"],
                            "parameter_value_display": case["parameter_value_display"],
                            "distance_nm": float(distance_nm[idx]),
                            "distance_over_lambda_D": float(distance_tilde[idx]),
                            "phi_profile_mV": float(phi[idx]),
                        }
                    )


def save_traceability_inputs(params: dict[str, Any], summary: dict[str, Any]) -> None:
    write_json(INPUTS_DIR / f"params_{OUTPUT_TAG}.json", params)
    write_json(INPUTS_DIR / f"summary_compare_{OUTPUT_TAG}.json", summary)
    config = {
        "output_tag": OUTPUT_TAG,
        "description": "Semi-quantitative EDL scheme for Figure 5 C_H and PZC scans; no IHP is represented.",
        "source_params": str(PARAMS_PATH.relative_to(ROOT)),
        "source_summary": str(SUMMARY_PATH.relative_to(ROOT)),
        "distance_lambda_max": DISTANCE_LAMBDA_MAX,
        "n_distance": N_DISTANCE,
        "scans": [
            {
                "group": spec.group,
                "key": spec.key,
                "side": spec.side,
                "low": spec.low,
                "baseline": spec.baseline,
                "high": spec.high,
                "display_factor": spec.display_factor,
                "display_unit": spec.display_unit,
            }
            for spec in SCAN_SPECS
        ],
        "profile_formula": (
            "side-average phi_s(y) = (RT/F) sum_n c_side,n A_n exp(-gamma_n y/lambda_D); "
            "the compact-side guide line is a linear OHP tangent and does not introduce an IHP."
        ),
        "visible_naming_note": "Use C_H in figure labels and output names; keep Cdl_* only as internal solver keys.",
    }
    write_json(CONFIG_OUT, config)


def ohp_slope_mV_per_nm(case: dict[str, Any]) -> float:
    distance_nm = np.asarray(case["distance_nm"], dtype=float)
    phi = np.asarray(case["phi_profile_mV"], dtype=float)
    if distance_nm.size < 2 or not distance_nm[1] > distance_nm[0]:
        raise ValueError(f"{case['scan']} {case['case']}: cannot estimate OHP slope")
    return float((phi[1] - phi[0]) / (distance_nm[1] - distance_nm[0]))


def compact_tangent_start_mV(case: dict[str, Any]) -> float:
    phi_rp = float(case["phi_RP_mean_mV"])
    return phi_rp + ohp_slope_mV_per_nm(case) * (METAL_X1_NM - RP_X_NM)


def profile_ylim(group_cases: list[dict[str, Any]]) -> tuple[float, float]:
    profile_values = [np.asarray(case["phi_profile_mV"], dtype=float) for case in group_cases]
    compact_values = [np.asarray([compact_tangent_start_mV(case)], dtype=float) for case in group_cases]
    values = np.concatenate(profile_values + compact_values)
    values = values[np.isfinite(values)]
    lo = float(np.min(values))
    hi = float(np.max(values))
    pad = max(16.0, 0.10 * (hi - lo if hi > lo else max(abs(hi), 1.0)))
    lo = min(lo - pad, -18.0)
    hi = max(hi + pad, 18.0)
    return lo, hi


def metal_coordinate_xlim(group_cases: list[dict[str, Any]]) -> tuple[float, float]:
    values = np.asarray([float(case["metal_minus_pzc_mV"]) for case in group_cases], dtype=float)
    span = float(np.max(values) - np.min(values))
    pad = max(22.0, 0.16 * span)
    return float(np.min(values) - pad), float(np.max(values) + pad)


def format_case_label(spec: ScanSpec, case: dict[str, Any]) -> str:
    param = spec.display_fmt.format(float(case["parameter_value_display"]))
    phi = float(case["phi_RP_mean_mV"])
    e_mix = float(case["E_mix_V"])
    return f"{case['case']:<4} {param:>5}   {phi:>+5.0f}   {e_mix:>.3f}"


def draw_material_slab(ax: plt.Axes, side: str, y_limits: tuple[float, float]) -> None:
    y0, y1 = y_limits
    face = side_color(side)
    ax.add_patch(
        Rectangle(
            (METAL_X0_NM, y0),
            METAL_X1_NM - METAL_X0_NM,
            y1 - y0,
            facecolor=face,
            edgecolor=COLORS["dark"],
            linewidth=0.75,
            zorder=1,
        )
    )
    ax.text(
        0.5 * (METAL_X0_NM + METAL_X1_NM),
        y1 - 0.08 * (y1 - y0),
        side,
        ha="center",
        va="top",
        fontsize=8.4,
        color=COLORS["dark"],
        fontweight="bold",
        zorder=4,
    )
    charge = "+" if y1 > abs(y0) else "-"
    for frac in (0.28, 0.45, 0.62, 0.79):
        ax.text(
            0.5 * (METAL_X0_NM + METAL_X1_NM),
            y0 + frac * (y1 - y0),
            charge,
            ha="center",
            va="center",
            fontsize=10.0,
            color=COLORS["dark"],
            zorder=4,
        )


def interpolate_color(stops: tuple[str, ...], fraction: float) -> tuple[float, float, float]:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    rgb = [np.asarray(to_rgb(color), dtype=float) for color in stops]
    if len(rgb) == 1:
        return tuple(float(v) for v in rgb[0])
    scaled = fraction * (len(rgb) - 1)
    idx = min(int(math.floor(scaled)), len(rgb) - 2)
    local = scaled - idx
    color = (1.0 - local) * rgb[idx] + local * rgb[idx + 1]
    return tuple(float(v) for v in color)


def draw_vector_gradient_background(ax: plt.Axes, x0: float, x1: float, y0: float, y1: float) -> None:
    n_steps = 96
    width = (x1 - x0) / n_steps
    for idx in range(n_steps):
        left = x0 + idx * width
        # Slight overlap avoids hairline seams in SVG/PDF viewers.
        rect_width = width * 1.025
        color = interpolate_color(GRADIENT_COLORS, idx / max(n_steps - 1, 1))
        ax.add_patch(
            Rectangle(
                (left, y0),
                rect_width,
                y1 - y0,
                facecolor=color,
                edgecolor="none",
                linewidth=0.0,
                alpha=0.72,
                zorder=0,
            )
        )


def draw_profile_panel(
    ax: plt.Axes,
    spec: ScanSpec,
    cases: list[dict[str, Any]],
    y_limits: tuple[float, float],
    *,
    show_ylabel: bool,
) -> None:
    y0, y1 = y_limits
    lambda_d_nm = float(cases[0]["lambda_D_nm"])
    x_max = DISTANCE_LAMBDA_MAX * lambda_d_nm
    ax.set_xlim(METAL_X0_NM, x_max)
    ax.set_ylim(*y_limits)

    draw_vector_gradient_background(ax, 0.0, x_max, y0, y1)
    ax.axvspan(METAL_X1_NM, 0.0, facecolor=COLORS["inner_layer"], edgecolor="none", zorder=0)
    draw_material_slab(ax, spec.side, y_limits)
    ax.axvline(RP_X_NM, color=COLORS["dark"], linewidth=0.85, zorder=3)
    ax.axhline(0.0, color=COLORS["dark"], linewidth=0.75, zorder=2)
    ax.text(
        0.18,
        y1 - 0.08 * (y1 - y0),
        "OHP / RP",
        ha="left",
        va="top",
        fontsize=6.6,
        color=COLORS["dark"],
        rotation=90,
    )
    ax.text(
        x_max - 0.12 * (x_max + 1.85),
        0.0 + 0.025 * (y1 - y0),
        r"$\phi_{\mathrm{bulk}}=0$",
        ha="right",
        va="bottom",
        fontsize=6.8,
        color=COLORS["gray"],
    )

    for case in cases:
        style = CASE_STYLES[str(case["case"])]
        distance_nm = np.asarray(case["distance_nm"], dtype=float)
        phi = np.asarray(case["phi_profile_mV"], dtype=float)
        phi_rp = float(case["phi_RP_mean_mV"])
        compact_start_y = compact_tangent_start_mV(case)
        ax.plot(
            [METAL_X1_NM, RP_X_NM],
            [compact_start_y, phi_rp],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            zorder=5,
        )
        ax.plot(
            distance_nm,
            phi,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            zorder=5,
        )
        ax.scatter(
            [0.0],
            [float(case["phi_RP_mean_mV"])],
            s=20,
            color=style["color"],
            edgecolor="white",
            linewidth=0.5,
            zorder=6,
        )
        if case["case"] == "base":
            ax.annotate(
                r"$\phi_{\mathrm{RP}}$",
                xy=(0.0, phi_rp),
                xytext=(8, 1),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7.1,
                color=COLORS["dark"],
                zorder=8,
            )

    text = (
        f"{spec.title} ({spec.display_unit})\n"
        r"case  value  $\langle\phi_{\mathrm{RP}}\rangle$  $E_{\mathrm{mix}}$"
        "\n"
        + "\n".join(format_case_label(spec, case) for case in cases)
    )
    ax.text(
        0.97,
        0.96,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.9,
        color=COLORS["dark"],
        linespacing=1.25,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=2.4),
        zorder=8,
    )
    ax.annotate(
        spec.trend_text,
        xy=(0.06 * x_max, float(cases[-1]["phi_RP_mean_mV"])),
        xytext=(0.33 * x_max, y0 + 0.18 * (y1 - y0)),
        arrowprops=dict(arrowstyle="->", color=COLORS["dark"], linewidth=0.75),
        ha="left",
        va="center",
        fontsize=6.4,
        color=COLORS["dark"],
        zorder=7,
    )
    ax.set_title(spec.title, loc="left", fontsize=9.6, pad=5.0)
    ax.set_xlabel(r"Distance from OHP/RP (nm)")
    if show_ylabel:
        ax.set_ylabel("Potential")
    else:
        ax.set_ylabel("")
    ax.set_yticks([])
    ax.tick_params(axis="x", length=3.0, width=0.75, pad=2.0, labelsize=7.6)
    ax.set_xticks(np.arange(0.0, x_max + 0.1, 2.5))
    for spine in ("bottom", "top", "right"):
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color(COLORS["dark"])
    ax.spines["left"].set_visible(False)
    if show_ylabel:
        ax.annotate(
            "",
            xy=(METAL_X0_NM + 0.12, y1 - 0.08 * (y1 - y0)),
            xytext=(METAL_X0_NM + 0.12, y0 + 0.12 * (y1 - y0)),
            arrowprops=dict(arrowstyle="->", color=COLORS["dark"], linewidth=0.85),
            zorder=8,
        )


def draw_metal_coordinate_panel(
    ax: plt.Axes,
    spec: ScanSpec,
    cases: list[dict[str, Any]],
    x_limits: tuple[float, float],
) -> None:
    ax.set_xlim(*x_limits)
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.50, color=COLORS["dark"], linewidth=0.85, zorder=1)
    ax.add_patch(
        Rectangle(
            (x_limits[0], 0.32),
            x_limits[1] - x_limits[0],
            0.36,
            facecolor=side_color(spec.side),
            edgecolor="none",
            alpha=0.13,
            zorder=0,
        )
    )
    for case in cases:
        style = CASE_STYLES[str(case["case"])]
        phi_m = float(case["metal_minus_pzc_mV"])
        ax.plot([phi_m, phi_m], [0.28, 0.72], color=style["color"], linewidth=1.5, zorder=3)
        ax.scatter([phi_m], [0.50], s=24, color=style["color"], edgecolor="white", linewidth=0.55, zorder=4)

    low_phi_m = float(cases[0]["metal_minus_pzc_mV"])
    high_phi_m = float(cases[-1]["metal_minus_pzc_mV"])
    y_arrow = 0.82
    ax.annotate(
        "",
        xy=(high_phi_m, y_arrow),
        xytext=(low_phi_m, y_arrow),
        arrowprops=dict(arrowstyle="->", color=COLORS["dark"], linewidth=0.8),
        zorder=4,
    )
    ax.text(
        0.02,
        0.94,
        r"$\phi_M=E_{\mathrm{mix}}-\mathrm{PZC}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=COLORS["dark"],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.2),
        zorder=6,
    )
    ax.text(
        0.98,
        0.94,
        "converted metal-potential shift",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color=COLORS["gray"],
    )
    ax.set_yticks([])
    ax.set_xlabel(r"Converted metal potential, $\phi_M$ (mV)", fontsize=6.7, labelpad=1.5)
    ax.tick_params(axis="x", length=2.6, width=0.7, pad=1.8, labelsize=6.6)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.spines["bottom"].set_color(COLORS["dark"])


def group_specs(group: str) -> list[ScanSpec]:
    return [spec for spec in SCAN_SPECS if spec.group == group]


def plot_group(
    group: str,
    cases: dict[str, list[dict[str, Any]]],
    png_out: Path,
    svg_out: Path,
    vector_svg_out: Path,
    illustrator_svg_out: Path,
) -> None:
    specs = group_specs(group)
    group_case_values = [case for spec in specs for case in cases[spec.key]]
    y_limits = profile_ylim(group_case_values)
    metal_limits = metal_coordinate_xlim(group_case_values)

    title = (
        r"Figure 5 EDL scheme: $C_{\mathrm{H}}$ controls $\phi_M$ and $\phi_{\mathrm{RP}}$"
        if group == "CH"
        else r"Figure 5 EDL scheme: PZC controls $\phi_M$ and $\phi_{\mathrm{RP}}$"
    )
    subtitle = (
        r"Low, base, and high cases use the same mixed-potential solver; $E_{\mathrm{mix}}$ is converted to $\phi_M=E_{\mathrm{mix}}-\mathrm{PZC}$."
    )

    fig = plt.figure(figsize=(7.35, 4.35))
    grid = fig.add_gridspec(2, 2, height_ratios=(0.68, 3.25), hspace=0.35, wspace=0.18)
    metal_axes = [fig.add_subplot(grid[0, idx]) for idx in range(2)]
    profile_axes: list[plt.Axes] = []
    for idx in range(2):
        shared = profile_axes[0] if profile_axes else None
        profile_axes.append(fig.add_subplot(grid[1, idx], sharey=shared))

    for idx, spec in enumerate(specs):
        draw_metal_coordinate_panel(metal_axes[idx], spec, cases[spec.key], metal_limits)
        draw_profile_panel(profile_axes[idx], spec, cases[spec.key], y_limits, show_ylabel=idx == 0)

    handles = [
        Line2D(
            [0],
            [0],
            color=CASE_STYLES[name]["color"],
            linestyle=CASE_STYLES[name]["linestyle"],
            linewidth=CASE_STYLES[name]["linewidth"],
            label=name,
        )
        for name in CASE_VALUES
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.50, 0.018), fontsize=7.2)
    fig.suptitle(title, x=0.055, y=0.985, ha="left", fontsize=11.2)
    fig.text(0.055, 0.915, subtitle, ha="left", va="top", fontsize=7.6, color=COLORS["gray"])
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.82, wspace=0.18, hspace=0.52)
    fig.savefig(png_out, dpi=350, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(svg_out, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(vector_svg_out, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    make_illustrator_safe_svg(vector_svg_out, illustrator_svg_out)


def assert_outputs() -> None:
    pdfs = sorted(OUT_DIR.glob("*.pdf"))
    missing = [path for path in (*EXPECTED_PNGS, *EXPECTED_SVGS) if not path.exists()]
    if missing:
        labels = ", ".join(str(path.relative_to(ROOT)) for path in missing)
        raise RuntimeError(f"Missing expected outputs: {labels}")
    if pdfs:
        raise RuntimeError(f"Expected zero PDF outputs, found {len(pdfs)}")


def main() -> None:
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    remove_stale_illustrator_svgs()

    params = load_json(PARAMS_PATH)
    summary = load_json(SUMMARY_PATH)
    validate_baseline(params, summary)
    save_traceability_inputs(params, summary)

    cases = build_all_cases(params)
    require_finite_cases(cases)
    write_case_csv(cases)
    write_profile_csv(cases)
    plot_group("CH", cases, CH_PNG_OUT, CH_SVG_OUT, CH_VECTOR_SVG_OUT, CH_ILLUSTRATOR_SVG_OUT)
    plot_group("PZC", cases, PZC_PNG_OUT, PZC_SVG_OUT, PZC_VECTOR_SVG_OUT, PZC_ILLUSTRATOR_SVG_OUT)
    assert_outputs()

    print(f"Wrote {CH_PNG_OUT.relative_to(ROOT)}")
    print(f"Wrote {CH_SVG_OUT.relative_to(ROOT)}")
    print(f"Wrote {CH_VECTOR_SVG_OUT.relative_to(ROOT)}")
    print(f"Wrote {CH_ILLUSTRATOR_SVG_OUT.relative_to(ROOT)}")
    print(f"Wrote {PZC_PNG_OUT.relative_to(ROOT)}")
    print(f"Wrote {PZC_SVG_OUT.relative_to(ROOT)}")
    print(f"Wrote {PZC_VECTOR_SVG_OUT.relative_to(ROOT)}")
    print(f"Wrote {PZC_ILLUSTRATOR_SVG_OUT.relative_to(ROOT)}")
    print(f"Wrote {CASE_CSV_OUT.relative_to(ROOT)}")
    print(f"Wrote {PROFILE_CSV_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
