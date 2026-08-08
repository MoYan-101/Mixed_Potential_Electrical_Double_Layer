"""Build the C_tot study for the fixed Au=Pd=2 nm independent-EDL model.

The numerical scan and the five reference figure classes are owned by the
``au_pd_independent_edls.ctot_study`` module.  This wrapper supplies the
Au2/Pd2 parameter lock, adapts conclusions that depend on those parameters,
uses the project colour/DPI conventions, and adds project-level regression
and artifact checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image


sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "Mixed_Potential_Electrical_Double_Layer" / "Au_Pd_independent_EDLs"
MODEL_SRC = MODEL_ROOT / "src"
PROJECT_DIR = Path(__file__).resolve().parent
INDEPENDENT_DIR = PROJECT_DIR / "Au_Pd_independent"
DEFAULT_OUTPUT = INDEPENDENT_DIR / "C_tot_study"
CHECKSUM_FILE = "checksums.sha256"
STUDY_ID = "independent_Au2nm_Pd2nm_CH50_PZC03V_Ctot_scan"
REFERENCE_RESULT = (
    "Mixed_Potential_Electrical_Double_Layer/Au_Pd_independent_EDLs/"
    "results/20260803_153355_ctot_study"
)
EXPECTED_FIGURE_PAIRS = 5

if str(MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_SRC))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from au_pd_independent_edls import ctot_study as ctot  # noqa: E402
from make_independent_au_pd import independent_params  # noqa: E402


PALETTE = {
    "with_edl": "#F26B38",
    "without_edl": "#12355B",
    "au": "#E4C133",
    "pd": "#5A90C8",
    "au_curve": "#009E73",
    "pd_curve": "#0072B2",
}

EXPECTED_REPRESENTATIVE: dict[float, dict[str, tuple[float, float]]] = {
    0.01: {
        "E_mix_with_EDL_V": (0.5918979952021372, 5.0e-11),
        "i_mix_avg_with_EDL_A_per_m2": (0.015838438113541905, 5.0e-12),
        "i_mix_abs_with_EDL_A": (6.335375245416762e-13, 5.0e-22),
        "i_mix_avg_ratio_with_over_without": (0.13472601573601703, 5.0e-12),
    },
    1.0: {
        "E_mix_with_EDL_V": (0.5137037894109704, 5.0e-11),
        "i_mix_avg_with_EDL_A_per_m2": (0.06959297241334496, 5.0e-12),
        "i_mix_abs_with_EDL_A": (2.7837188965337985e-12, 5.0e-22),
        "i_mix_avg_ratio_with_over_without": (0.5919765465042933, 5.0e-12),
    },
    10.0: {
        "E_mix_with_EDL_V": (0.4856506355626403, 5.0e-11),
        "i_mix_avg_with_EDL_A_per_m2": (0.09731571755397438, 5.0e-12),
        "i_mix_abs_with_EDL_A": (3.8926287021589755e-12, 5.0e-22),
        "i_mix_avg_ratio_with_over_without": (0.8277936751432982, 5.0e-12),
    },
    1000.0: {
        "E_mix_with_EDL_V": (0.4690941489147291, 5.0e-11),
        "i_mix_avg_with_EDL_A_per_m2": (0.11522469093233681, 5.0e-12),
        "i_mix_abs_with_EDL_A": (4.608987637293473e-12, 5.0e-22),
        "i_mix_avg_ratio_with_over_without": (0.9801322208946127, 5.0e-12),
    },
}

EXPECTED_WITHOUT_EDL = {
    "E_mix_without_EDL_V": (0.46699999999999986, 5.0e-11),
    "i_mix_avg_without_EDL_A_per_m2": (0.11756035407872402, 5.0e-12),
    "i_mix_abs_without_EDL_A": (4.7024141631489615e-12, 5.0e-22),
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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    sources = {
        "study_wrapper": Path(__file__).resolve(),
        "parameter_wrapper": PROJECT_DIR / "make_independent_au_pd.py",
        "ctot_study": MODEL_SRC / "au_pd_independent_edls" / "ctot_study.py",
        "independent_model": MODEL_SRC / "au_pd_independent_edls" / "model.py",
    }
    return {name: _sha256(path) for name, path in sources.items()}


def _target_plot_ctot_trends(
    rows: list[dict[str, Any]], output_dir: Path
) -> list[Path]:
    """Reference trend plots with a conclusion derived from the actual scan."""

    concentration = np.asarray([row["C_tot_M"] for row in rows], dtype=float)
    ratio = np.asarray(
        [row["i_mix_avg_ratio_with_over_without"] for row in rows], dtype=float
    )
    current_title = (
        r"$\bar{i}_{\mathrm{mix}}$ overshoots before the high-salt limit"
        if np.any(ratio > 1.0 + 1.0e-10)
        else r"$\bar{i}_{\mathrm{mix}}$ approaches w/o EDL from below"
    )
    specifications = (
        {
            "stem": "ctot_emix_high_salt_regime_independent_edls",
            "with_key": "E_mix_with_EDL_V",
            "without_key": "E_mix_without_EDL_V",
            "ylabel": r"$E_{\mathrm{mix}}$ (V vs. RHE)",
            "title": r"$E_{\mathrm{mix}}$ shift fades in the high-salt limit",
            "offsets": {1.0e-2: -0.014, 1.0: 0.010, 1.0e3: 0.008},
            "legend_loc": "lower left",
            "legend_bbox": (0.01, 0.075),
        },
        {
            "stem": "ctot_imix_avg_high_salt_regime_independent_edls",
            "with_key": "i_mix_avg_with_EDL_A_per_m2",
            "without_key": "i_mix_avg_without_EDL_A_per_m2",
            "ylabel": r"$\bar{i}_{\mathrm{mix}}$ (A/m$^2$)",
            "title": current_title,
            "offsets": {1.0e-2: -0.008, 1.0: 0.006, 1.0e3: -0.006},
            "legend_loc": "lower right",
            "legend_bbox": None,
        },
    )
    saved: list[Path] = []
    for specification in specifications:
        y_with = np.asarray(
            [row[str(specification["with_key"])] for row in rows], dtype=float
        )
        y_without = np.asarray(
            [row[str(specification["without_key"])] for row in rows], dtype=float
        )
        lower = float(min(np.min(y_with), np.min(y_without)))
        upper = float(max(np.max(y_with), np.max(y_without)))
        span = max(upper - lower, 1.0e-6)
        fig, ax = plt.subplots(figsize=(4.15, 3.45))
        ctot._regime_background(ax)
        ctot._split_line(
            ax,
            concentration,
            y_with,
            color=ctot.COLORS["with_edl"],
            linewidth=2.25,
            linestyle="solid",
            label="with EDL",
            marker="o",
            markersize=2.8,
        )
        ctot._split_line(
            ax,
            concentration,
            y_without,
            color=ctot.COLORS["without_edl"],
            linewidth=1.8,
            linestyle=(0, (4.0, 2.5)),
            label="w/o EDL",
        )
        ctot._representative_markers(
            ax,
            rows,
            str(specification["with_key"]),
            specification["offsets"],
        )
        ax.set_xscale("log")
        ax.set_xlim(1.0e-4, 1.0e3)
        ax.set_ylim(lower - 0.08 * span, upper + 0.15 * span)
        ax.set_xlabel(r"Electrolyte concentration, $C_{\mathrm{tot}}$ (M)")
        ax.set_ylabel(str(specification["ylabel"]))
        ax.set_title(str(specification["title"]), loc="left", fontsize=9.7, pad=5.0)
        legend_kwargs: dict[str, Any] = {
            "loc": str(specification["legend_loc"]),
            "fontsize": 7.4,
            "handlelength": 2.4,
        }
        if specification["legend_bbox"] is not None:
            legend_kwargs["bbox_to_anchor"] = specification["legend_bbox"]
        ax.legend(**legend_kwargs)
        ax.tick_params(length=3.2, width=0.85, labelsize=8.0)
        saved.extend(ctot._save_figure(fig, output_dir, str(specification["stem"])))
    return saved


def _target_plot_polarization_overlay(
    polarization_rows: list[dict[str, Any]],
    cases: Mapping[float, Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Reference polarization plot with an absolute-current scale suited to 2 nm."""

    fig, ax = plt.subplots(figsize=(7.2, 4.35))
    reference = [
        row
        for row in polarization_rows
        if row["condition"] == "w/o EDL reference"
    ]
    e_reference = np.asarray([row["E_V"] for row in reference], dtype=float)
    for key, color in (
        ("I_Au_1e_minus_3_uA", ctot.COLORS["au_curve"]),
        ("I_Pd_1e_minus_3_uA", ctot.COLORS["pd_curve"]),
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

    mixed_markers: list[float] = []
    for concentration_m in ctot.REPRESENTATIVE_CONCENTRATIONS_M:
        style = ctot.CASE_STYLES[concentration_m]
        subset = [
            row
            for row in polarization_rows
            if row["condition"] == "with EDL"
            and float(row["C_tot_M"]) == concentration_m
        ]
        potential = np.asarray([row["E_V"] for row in subset], dtype=float)
        alpha = 1.0 if concentration_m <= 1.0 else 0.42
        ax.plot(
            potential,
            np.asarray(
                [row["I_Au_1e_minus_3_uA"] for row in subset], dtype=float
            ),
            color=ctot.COLORS["au_curve"],
            linewidth=2.0,
            linestyle=style["linestyle"],
            alpha=alpha,
        )
        ax.plot(
            potential,
            np.asarray(
                [row["I_Pd_1e_minus_3_uA"] for row in subset], dtype=float
            ),
            color=ctot.COLORS["pd_curve"],
            linewidth=2.0,
            linestyle=style["linestyle"],
            alpha=alpha,
        )
        solution = cases[concentration_m]["solution"]
        e_mix = float(solution["E_mix_V"])
        i_mix = ctot.CURRENT_DISPLAY_SCALE * float(solution["i_mix_abs_A"])
        mixed_markers.append(i_mix)
        ax.axvline(e_mix, color=ctot.COLORS["gray"], linewidth=0.8, alpha=0.50)
        ax.scatter(
            [e_mix, e_mix],
            [i_mix, -i_mix],
            marker=str(style["marker"]),
            s=48,
            facecolor=ctot.COLORS["dark"],
            edgecolor="white",
            linewidth=0.7,
            alpha=alpha,
            zorder=5,
        )

    y_limit = max(0.008, 2.8 * max(mixed_markers))
    ax.axhline(0.0, color=ctot.COLORS["dark"], linewidth=0.9)
    ax.set_xlim(0.40, 0.64)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xlabel("Potential (V vs. RHE)")
    ax.set_ylabel(r"Half-reaction current (10$^{-3}$ $\mu$A)")
    ax.set_title(
        r"Salt-dependent polarization curves explain $I_{\mathrm{mix}}$",
        loc="left",
        fontsize=10.2,
    )
    half_reaction_legend = ax.legend(
        handles=[
            Line2D(
                [0], [0], color=ctot.COLORS["au_curve"], lw=2.5, label="Au oxidation"
            ),
            Line2D(
                [0], [0], color=ctot.COLORS["pd_curve"], lw=2.5, label="Pd reduction"
            ),
            Line2D(
                [0],
                [0],
                color=ctot.COLORS["gray"],
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
                color=ctot.COLORS["dark"],
                lw=1.8,
                ls=ctot.CASE_STYLES[concentration_m]["linestyle"],
                marker=ctot.CASE_STYLES[concentration_m]["marker"],
                label=ctot.CASE_STYLES[concentration_m]["label"],
                alpha=1.0 if concentration_m <= 1.0 else 0.42,
            )
            for concentration_m in ctot.REPRESENTATIVE_CONCENTRATIONS_M
        ],
        title=r"$C_{\mathrm{tot}}$",
        loc="upper right",
        fontsize=7.6,
        title_fontsize=7.8,
    )
    ax.tick_params(length=3.2, width=0.85, labelsize=8.1)
    return ctot._save_figure(
        fig,
        output_dir,
        "ctot_half_reaction_polarization_overlay_independent_edls",
    )


def _configure_upstream() -> None:
    ctot.DPI = 600
    ctot.COLORS.update(PALETTE)
    ctot.plot_ctot_trends = _target_plot_ctot_trends
    ctot.plot_polarization_overlay = _target_plot_polarization_overlay


def _scan_checks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_grid = ctot.concentration_scan_values_m()
    actual_grid = np.asarray([float(row["C_tot_M"]) for row in rows], dtype=float)
    checks: dict[str, Any] = {
        "scan_row_count": len(rows),
        "scan_grid_exact": bool(
            actual_grid.shape == expected_grid.shape
            and np.allclose(actual_grid, expected_grid, rtol=0.0, atol=1.0e-13)
        ),
        "representative_regression": {},
    }
    representative_passed = True
    for concentration_m, fields in EXPECTED_REPRESENTATIVE.items():
        row = ctot.row_at(rows, concentration_m)
        case_checks: dict[str, Any] = {}
        for field, (expected, tolerance) in fields.items():
            actual = float(row[field])
            passed = abs(actual - expected) <= tolerance
            case_checks[field] = {
                "actual": actual,
                "expected": expected,
                "absolute_difference": abs(actual - expected),
                "absolute_tolerance": tolerance,
                "passed": passed,
            }
            representative_passed = representative_passed and passed
        checks["representative_regression"][f"{concentration_m:g}_M"] = case_checks

    without_checks: dict[str, Any] = {}
    without_passed = True
    for field, (expected, tolerance) in EXPECTED_WITHOUT_EDL.items():
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
        max_error = float(np.max(np.abs(values - expected)))
        passed = max_error <= tolerance and float(np.ptp(values)) < 1.0e-12
        without_checks[field] = {
            "expected": expected,
            "max_absolute_difference": max_error,
            "absolute_tolerance": tolerance,
            "variation": float(np.ptp(values)),
            "passed": passed,
        }
        without_passed = without_passed and passed
    checks["without_edl_regression"] = without_checks

    e_with = np.asarray([float(row["E_mix_with_EDL_V"]) for row in rows])
    i_with = np.asarray(
        [float(row["i_mix_avg_with_EDL_A_per_m2"]) for row in rows]
    )
    ratio = np.asarray(
        [float(row["i_mix_avg_ratio_with_over_without"]) for row in rows]
    )
    checks["target_trend"] = {
        "E_mix_strictly_decreases": bool(np.all(np.diff(e_with) < 0.0)),
        "i_mix_avg_strictly_increases": bool(np.all(np.diff(i_with) > 0.0)),
        "ratio_min": float(np.min(ratio)),
        "ratio_max": float(np.max(ratio)),
        "ratio_remains_below_one": bool(np.all(ratio < 1.0)),
        "unity_crossings_M": ctot._log_interpolated_crossing(
            rows, "i_mix_avg_ratio_with_over_without", 1.0
        ),
        "interpretation": "with-EDL current approaches the w/o-EDL limit from below",
    }
    checks["g_Au_equals_g_Pd"] = bool(
        np.allclose(
            [float(row["g_Au"]) for row in rows],
            [float(row["g_Pd"]) for row in rows],
            rtol=0.0,
            atol=1.0e-14,
        )
    )
    checks["passed"] = bool(
        len(rows) == 36
        and checks["scan_grid_exact"]
        and representative_passed
        and without_passed
        and checks["target_trend"]["E_mix_strictly_decreases"]
        and checks["target_trend"]["i_mix_avg_strictly_increases"]
        and checks["target_trend"]["ratio_remains_below_one"]
        and not checks["target_trend"]["unity_crossings_M"]
        and checks["g_Au_equals_g_Pd"]
    )
    return checks


def _debye_huckel_caveat(rows: list[dict[str, Any]]) -> dict[str, Any]:
    threshold = 1.0
    maxima = np.asarray(
        [
            max(abs(float(row["Au_phi_RP_tilde"])), abs(float(row["Pd_phi_RP_tilde"])))
            for row in rows
        ],
        dtype=float,
    )
    concentrations = np.asarray([float(row["C_tot_M"]) for row in rows], dtype=float)
    within = concentrations[maxima <= threshold]
    representative = {
        f"{value:g}_M": float(
            max(
                abs(float(ctot.row_at(rows, value)["Au_phi_RP_tilde"])),
                abs(float(ctot.row_at(rows, value)["Pd_phi_RP_tilde"])),
            )
        )
        for value in (0.01, 1.0, 10.0, 1000.0)
    }
    return {
        "threshold_max_abs_phi_tilde": threshold,
        "maximum_over_scan": float(np.max(maxima)),
        "threshold_exceeded": bool(np.any(maxima > threshold)),
        "scan_points_exceeding_threshold": int(np.count_nonzero(maxima > threshold)),
        "scan_points_total": int(len(rows)),
        "first_sampled_concentration_within_threshold_M": (
            float(within[0]) if within.size else None
        ),
        "representative_max_abs_phi_tilde": representative,
        "included_in_numerical_pass_fail": False,
        "interpretation": (
            "The numerical solution passed its internal identities, but the linear "
            "Debye-Huckel approximation is outside its nominal weak-potential range "
            "for the flagged scan points."
        ),
    }


def _artifact_checks(output: Path) -> dict[str, Any]:
    pngs = sorted(output.glob("figures/**/*.png"))
    svgs = sorted(output.glob("figures/**/*.svg"))
    pdfs = sorted(output.glob("**/*.pdf"))
    png_dpi: dict[str, list[float] | None] = {}
    png_decodable = True
    for path in pngs:
        try:
            with Image.open(path) as image:
                dpi_value = image.info.get("dpi")
                png_dpi[path.relative_to(output).as_posix()] = (
                    [float(dpi_value[0]), float(dpi_value[1])]
                    if dpi_value is not None
                    else None
                )
                image.verify()
        except Exception:
            png_decodable = False
            png_dpi[path.relative_to(output).as_posix()] = None
    dpi_passed = all(
        value is not None
        and abs(value[0] - 600.0) < 1.0
        and abs(value[1] - 600.0) < 1.0
        for value in png_dpi.values()
    )
    svg_parseable = True
    svg_editable = True
    svg_helvetica_first = True
    for path in svgs:
        text = path.read_text(encoding="utf-8")
        try:
            ET.parse(path)
        except ET.ParseError:
            svg_parseable = False
        svg_editable = svg_editable and "<text" in text
        svg_helvetica_first = svg_helvetica_first and "Helvetica" in text
    stems_png = {path.relative_to(output).with_suffix("").as_posix() for path in pngs}
    stems_svg = {path.relative_to(output).with_suffix("").as_posix() for path in svgs}
    checks = {
        "figure4_pair_count": len(list((output / "figures" / "Figure_4").glob("*.png"))),
        "mechanism_pair_count": len(list((output / "figures" / "Mechanism").glob("*.png"))),
        "edl_scheme_pair_count": len(list((output / "figures" / "EDL_scheme").glob("*.png"))),
        "png_count": len(pngs),
        "svg_count": len(svgs),
        "pdf_count": len(pdfs),
        "all_files_nonempty": all(path.stat().st_size > 0 for path in (*pngs, *svgs)),
        "png_decodable": png_decodable,
        "png_dpi": png_dpi,
        "all_png_600_dpi": dpi_passed,
        "svg_parseable": svg_parseable,
        "svg_text_editable": svg_editable,
        "svg_helvetica_first": svg_helvetica_first,
        "png_svg_stems_match": stems_png == stems_svg,
        "no_delivery_pycache": not list(output.glob("**/__pycache__")),
    }
    checks["passed"] = bool(
        checks["figure4_pair_count"] == 3
        and checks["mechanism_pair_count"] == 1
        and checks["edl_scheme_pair_count"] == 1
        and len(pngs) == EXPECTED_FIGURE_PAIRS
        and len(svgs) == EXPECTED_FIGURE_PAIRS
        and not pdfs
        and checks["all_files_nonempty"]
        and png_decodable
        and dpi_passed
        and svg_parseable
        and svg_editable
        and svg_helvetica_first
        and stems_png == stems_svg
        and checks["no_delivery_pycache"]
    )
    return checks


def _refresh_artifacts(output: Path) -> None:
    path = output / "artifacts.json"
    artifacts = _read_json(path)
    registered = sorted(
        {
            relative
            for category in ("Figure_4", "Mechanism", "EDL_scheme", "csv", "metadata")
            for relative in artifacts[category]
            if relative != "artifacts.json"
        }
    )
    for relative in registered:
        target = output / relative
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(f"Registered artifact is missing or empty: {relative}")
    artifacts["sha256"] = {relative: _sha256(output / relative) for relative in registered}
    artifacts["size_bytes"] = {
        relative: (output / relative).stat().st_size for relative in registered
    }
    artifacts["project_checksums"] = CHECKSUM_FILE
    _write_json(path, artifacts)


def _write_checksums(output: Path) -> dict[str, str]:
    paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != CHECKSUM_FILE
    )
    checksums = {
        path.relative_to(output).as_posix(): _sha256(path)
        for path in paths
    }
    (output / CHECKSUM_FILE).write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in checksums.items()),
        encoding="utf-8",
    )
    return checksums


def _verify_checksums(output: Path) -> dict[str, str]:
    path = output / CHECKSUM_FILE
    if not path.is_file():
        raise RuntimeError(f"Missing {CHECKSUM_FILE}")
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if relative in checksums:
            raise RuntimeError(f"Duplicate checksum path: {relative}")
        target = output / relative
        if not target.is_file() or _sha256(target) != expected:
            raise RuntimeError(f"Checksum mismatch: {relative}")
        checksums[relative] = expected
    actual = {
        item.relative_to(output).as_posix()
        for item in output.rglob("*")
        if item.is_file() and item.name != CHECKSUM_FILE
    }
    if set(checksums) != actual:
        raise RuntimeError("C_tot study checksum coverage is incomplete")
    return checksums


def _verify_artifacts(output: Path) -> None:
    artifacts = _read_json(output / "artifacts.json")
    if set(artifacts["sha256"]) != set(artifacts["size_bytes"]):
        raise RuntimeError("artifacts.json hash and size registries differ")
    for relative, expected in artifacts["sha256"].items():
        target = output / relative
        if _sha256(target) != expected:
            raise RuntimeError(f"artifacts.json checksum mismatch: {relative}")
        if target.stat().st_size != int(artifacts["size_bytes"][relative]):
            raise RuntimeError(f"artifacts.json size mismatch: {relative}")


def _read_scan_csv(output: Path) -> list[dict[str, Any]]:
    with (output / "csv" / "ctot_scan.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _locked_parameter_checks(params: Mapping[str, Any]) -> dict[str, Any]:
    expected: dict[str, float | int | None] = {
        "L_Au": 2.0e-9,
        "L_Pd": 2.0e-9,
        "C_H_Au": 0.50,
        "C_H_Pd": 0.50,
        "active_faces_Au": 1,
        "active_faces_Pd": 1,
        "C_tot": 10.0,
        "it0_1": 1.852573885166257e-4,
        "it0_2": 1.852573885166257e-4,
        "alpha1": 0.5,
        "alpha2": 0.5,
        "out_of_plane_width": 0.01,
        "epsilon_s": None,
        "lambda_D": None,
        "g_Au": None,
        "g_Pd": None,
    }
    fields: dict[str, Any] = {}
    passed = True
    for key, expected_value in expected.items():
        actual = params.get(key)
        if expected_value is None:
            field_passed = actual is None
        else:
            field_passed = math.isclose(
                float(actual),
                float(expected_value),
                rel_tol=0.0,
                abs_tol=max(1.0e-15, abs(float(expected_value)) * 1.0e-12),
            )
        fields[key] = {"actual": actual, "expected": expected_value, "passed": field_passed}
        passed = passed and field_passed
    return {"fields": fields, "passed": passed}


def _augment_result(output: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    scan_checks = _scan_checks(rows)
    artifact_checks = _artifact_checks(output)
    params = _read_json(output / "params.json")
    parameter_checks = _locked_parameter_checks(params)
    caveat = _debye_huckel_caveat(rows)

    validation_path = output / "validation.json"
    validation = _read_json(validation_path)
    upstream_passed = bool(validation.get("passed"))
    validation.update(
        {
            "study_id": STUDY_ID,
            "upstream_numerical_validation_passed": upstream_passed,
            "locked_parameter_checks": parameter_checks,
            "target_scan_checks": scan_checks,
            "artifact_checks": artifact_checks,
            "debye_huckel_applicability_caveat": caveat,
            "topology_scope_caveat": (
                "The 2 nm lengths set active areas only. The analytic model uses two "
                "independent infinite planar half-spaces and does not resolve finite-size "
                "edge fields or lateral EDL overlap."
            ),
            "passed": bool(
                upstream_passed
                and parameter_checks["passed"]
                and scan_checks["passed"]
                and artifact_checks["passed"]
            ),
        }
    )
    _write_json(validation_path, validation)
    if not validation["passed"]:
        raise RuntimeError(f"Target C_tot validation failed: {validation}")

    summary_path = output / "summary.json"
    summary = _read_json(summary_path)
    summary.update(
        {
            "study_id": STUDY_ID,
            "fixed_parameters": {
                "L_Au_nm": 2.0,
                "L_Pd_nm": 2.0,
                "C_H_Au_uF_per_cm2": 50.0,
                "C_H_Pd_uF_per_cm2": 50.0,
                "active_faces_Au": 1,
                "active_faces_Pd": 1,
            },
            "current_trend_interpretation": scan_checks["target_trend"]["interpretation"],
        }
    )
    _write_json(summary_path, summary)

    scan_config_path = output / "scan_config.json"
    scan_config = _read_json(scan_config_path)
    scan_config.update(
        {
            "target_study_id": STUDY_ID,
            "reference_result_for_layout_and_method_only": REFERENCE_RESULT,
            "target_figure_adaptations": {
                "current_title_is_data_dependent": True,
                "polarization_y_axis_uses_absolute_current_and_is_zoomed_for_2_nm": True,
                "mechanism_Au_Pd_corresponding_metrics_share_y_scale": True,
                "dpi_png": 600,
                "palette": PALETTE,
            },
        }
    )
    _write_json(scan_config_path, scan_config)

    manifest_path = output / "run_manifest.json"
    manifest = _read_json(manifest_path)
    manifest.update(
        {
            "study_id": STUDY_ID,
            "parent_study_id": "independent_Au2nm_Pd2nm_CH50_PZC03V",
            "study_parameters": {
                "L_Au_nm": 2.0,
                "L_Pd_nm": 2.0,
                "C_H_Au_uF_per_cm2": 50.0,
                "C_H_Pd_uF_per_cm2": 50.0,
                "active_faces_Au": 1,
                "active_faces_Pd": 1,
                "equal_i0_A_per_m2": float(params["it0_1"]),
                "alpha1": 0.5,
                "alpha2": 0.5,
                "out_of_plane_width_m": 0.01,
            },
            "parameter_provenance": (
                "Figures/Figure_Au2nm_Pd2nm/make_independent_au_pd.py::"
                "independent_params"
            ),
            "reference_result_for_layout_and_method_only": REFERENCE_RESULT,
            "source_sha256": _source_hashes(),
            "checksums_file": CHECKSUM_FILE,
            "export": {
                "dpi_png": 600,
                "formats": ["png", "svg"],
                "svg_fonttype": "none",
                "pdf_enabled": False,
            },
            "target_result_interpretation": (
                "For C_H,Au=C_H,Pd=50 uF/cm^2, i_mix with EDL remains below "
                "the w/o-EDL value over the sampled grid and approaches it from below."
            ),
            "formal_high_salt_note": (
                "10-1000 M is retained only as a mathematical high-salt extension; "
                "10^3 M is not a physically realizable electrolyte concentration."
            ),
            "validation_passed": True,
        }
    )
    _write_json(manifest_path, manifest)

    _refresh_artifacts(output)
    checksums = _write_checksums(output)
    _verify_artifacts(output)
    _verify_checksums(output)
    return {
        "output": str(output),
        "summary": summary,
        "validation": validation,
        "manifest": manifest,
        "checksums": checksums,
    }


def build_independent_au_pd_ctot_study(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if output == DEFAULT_OUTPUT.resolve() and not (INDEPENDENT_DIR / "params.json").is_file():
        raise RuntimeError(
            "Build the independent Au/Pd baseline first; refusing to create an incomplete "
            f"parent result at {INDEPENDENT_DIR}"
        )
    params = independent_params()
    parameter_checks = _locked_parameter_checks(params)
    if not parameter_checks["passed"]:
        raise RuntimeError(f"Target parameter lock failed: {parameter_checks}")
    _configure_upstream()
    preview_rows = ctot.compute_scan_rows(params)
    preview_checks = _scan_checks(preview_rows)
    if not preview_checks["passed"]:
        raise RuntimeError(f"Target C_tot preview regression failed: {preview_checks}")
    ctot.build_ctot_results(params, output)
    rows = _read_scan_csv(output)
    return _augment_result(output, rows)


def validate_existing_independent_au_pd_ctot_study(
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    if not output.is_dir():
        raise FileNotFoundError(f"Missing C_tot study output: {output}")
    params = _read_json(output / "params.json")
    parameter_checks = _locked_parameter_checks(params)
    rows = _read_scan_csv(output)
    scan_checks = _scan_checks(rows)
    artifact_checks = _artifact_checks(output)
    validation = _read_json(output / "validation.json")
    manifest = _read_json(output / "run_manifest.json")
    if not parameter_checks["passed"]:
        raise RuntimeError("Existing C_tot study parameter lock failed")
    if not scan_checks["passed"]:
        raise RuntimeError("Existing C_tot study scan regression failed")
    if not artifact_checks["passed"]:
        raise RuntimeError("Existing C_tot study artifacts failed validation")
    if validation.get("study_id") != STUDY_ID or not bool(validation.get("passed")):
        raise RuntimeError("Existing C_tot study validation metadata is invalid")
    if manifest.get("source_sha256") != _source_hashes():
        raise RuntimeError("Existing C_tot study source hashes do not match current code")
    _verify_artifacts(output)
    checksums = _verify_checksums(output)
    return {
        "output": str(output),
        "reused": True,
        "validation": validation,
        "manifest": manifest,
        "checksums": checksums,
    }


def build_or_reuse_independent_au_pd_ctot_study(
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        return validate_existing_independent_au_pd_ctot_study(output)
    return build_independent_au_pd_ctot_study(output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="New or reusable output directory (default: %(default)s)",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = build_or_reuse_independent_au_pd_ctot_study(args.output)
    summary = (
        result["summary"]
        if "summary" in result
        else _read_json(Path(result["output"]) / "summary.json")
    )
    print(f"output = {result['output']}")
    for label in ("0.01 M", "1 M", "$10^3$ M"):
        row = summary["representative_cases"][label]
        print(
            f"{label}: E_mix={float(row['E_mix_with_EDL_V']):.12f} V, "
            f"i_mix_avg={float(row['i_mix_avg_with_EDL_A_per_m2']):.12g} A/m^2, "
            f"I_mix={float(row['i_mix_abs_with_EDL_A']):.12g} A"
        )
    print("figures = 5 PNG + 5 SVG; PDF = 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
