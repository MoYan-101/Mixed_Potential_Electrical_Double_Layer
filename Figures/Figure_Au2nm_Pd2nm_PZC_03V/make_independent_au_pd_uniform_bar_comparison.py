"""Bar comparison for spatially uniform independent Au/Pd interfaces."""

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
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from PIL import Image


sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_SRC = (
    ROOT
    / "Mixed_Potential_Electrical_Double_Layer"
    / "Au_Pd_independent_EDLs"
    / "src"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "Au_Pd_independent"
    / "figures"
    / "Figure_3"
    / "Uniform_Bar_Comparison"
)
CHECKSUM_FILE = "checksums.sha256"
FIGURE_STEM = "figure_3_uniform_interface_bar_comparison_independent_edls"

if str(MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_SRC))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from au_pd_independent_edls.model import IndependentPlanarEDLModel  # noqa: E402
from make_independent_au_pd import EXPECTED_RESULTS, independent_params  # noqa: E402


COLORS = {
    "with_edl": "#F26B38",
    "without_edl": "#12355B",
    "dark": "#272727",
    "gray": "#767676",
    "zero": "#B8B8B8",
}
CONDITIONS = ("w/o EDL", "with EDL")


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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    sources = {
        "bar_wrapper": Path(__file__).resolve(),
        "parameter_wrapper": PROJECT_DIR / "make_independent_au_pd.py",
        "independent_model": MODEL_SRC / "au_pd_independent_edls" / "model.py",
    }
    return {name: _sha256(path) for name, path in sources.items()}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica",
                "Nimbus Sans",
                "Arial",
                "DejaVu Sans",
                "sans-serif",
            ],
            "font.size": 9.2,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
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
        }
    )


def _build_values(
    model: IndependentPlanarEDLModel,
    with_edl: Mapping[str, Any],
    without_edl: Mapping[str, Any],
) -> dict[str, Any]:
    phi_tilde = {
        "Au": float(with_edl["phi_RP_Au_tilde"]),
        "Pd": float(with_edl["phi_RP_Pd_tilde"]),
    }
    phi_v = {
        "Au": float(with_edl["phi_RP_Au_V"]),
        "Pd": float(with_edl["phi_RP_Pd_V"]),
    }
    concentration_with = {
        "Au": math.exp(
            np.clip(-float(model.params["z_R1"]) * phi_tilde["Au"], -700.0, 700.0)
        ),
        "Pd": math.exp(
            np.clip(-float(model.params["z_O2"]) * phi_tilde["Pd"], -700.0, 700.0)
        ),
    }
    eta_with = {
        "Au": (
            float(with_edl["E_mix_V"])
            - float(model.reaction["E1_eq_eff"])
            - phi_v["Au"]
        ),
        "Pd": (
            float(with_edl["E_mix_V"])
            - float(model.reaction["E2_eq_eff"])
            - phi_v["Pd"]
        ),
    }
    eta_without = {
        "Au": float(without_edl["E_mix_V"]) - float(model.reaction["E1_eq_eff"]),
        "Pd": float(without_edl["E_mix_V"]) - float(model.reaction["E2_eq_eff"]),
    }
    return {
        "E_mix_V": {
            "w/o EDL": float(without_edl["E_mix_V"]),
            "with EDL": float(with_edl["E_mix_V"]),
        },
        "i_mix_avg_A_per_m2": {
            "w/o EDL": float(without_edl["i_mix_avg_A_per_m2"]),
            "with EDL": float(with_edl["i_mix_avg_A_per_m2"]),
        },
        "phi_RP_mV": {
            "Au": {"w/o EDL": 0.0, "with EDL": 1.0e3 * phi_v["Au"]},
            "Pd": {"w/o EDL": 0.0, "with EDL": 1.0e3 * phi_v["Pd"]},
        },
        "reactant_concentration_over_bulk": {
            "Au": {"reactant": "Red1", "w/o EDL": 1.0, "with EDL": concentration_with["Au"]},
            "Pd": {"reactant": "Ox2", "w/o EDL": 1.0, "with EDL": concentration_with["Pd"]},
        },
        "eta_RP_V": {
            "Au": {"w/o EDL": eta_without["Au"], "with EDL": eta_with["Au"]},
            "Pd": {"w/o EDL": eta_without["Pd"], "with EDL": eta_with["Pd"]},
        },
        "local_current_density_A_per_m2": {
            "Au": {
                "reaction": "i1",
                "w/o EDL": float(without_edl["j_Au_A_per_m2"]),
                "with EDL": float(with_edl["j_Au_A_per_m2"]),
            },
            "Pd": {
                "reaction": "i2",
                "w/o EDL": float(without_edl["j_Pd_A_per_m2"]),
                "with EDL": float(with_edl["j_Pd_A_per_m2"]),
            },
        },
    }


def _rows(values: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        rows.append(
            {
                "metric": "E_mix",
                "scope": "mixed",
                "species_or_reaction": "",
                "condition": condition,
                "value": values["E_mix_V"][condition],
                "unit": "V vs. RHE",
            }
        )
        rows.append(
            {
                "metric": "i_mix_avg",
                "scope": "mixed",
                "species_or_reaction": "",
                "condition": condition,
                "value": values["i_mix_avg_A_per_m2"][condition],
                "unit": "A/m^2",
            }
        )
    grouped = (
        ("phi_RP", "phi_RP_mV", "mV", ""),
        ("c_i_over_c_bulk", "reactant_concentration_over_bulk", "dimensionless", "reactant"),
        ("eta_RP", "eta_RP_V", "V", ""),
        ("local_current_density", "local_current_density_A_per_m2", "A/m^2", "reaction"),
    )
    for metric, key, unit, detail_key in grouped:
        for material in ("Au", "Pd"):
            detail = values[key][material].get(detail_key, "") if detail_key else ""
            for condition in CONDITIONS:
                rows.append(
                    {
                        "metric": metric,
                        "scope": material,
                        "species_or_reaction": detail,
                        "condition": condition,
                        "value": values[key][material][condition],
                        "unit": unit,
                    }
                )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Uniform-interface bar CSV cannot be empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _style_axis(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=10.4, pad=5.0)
    ax.set_ylabel(ylabel, fontsize=9.8)
    ax.tick_params(axis="both", labelsize=8.8, length=3.2, width=0.85, pad=2.5)
    ax.spines["left"].set_color(COLORS["dark"])
    ax.spines["bottom"].set_color(COLORS["dark"])
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)


def _label_linear_bars(
    ax: plt.Axes,
    bars: Any,
    values: list[float],
    formatter: Any,
    *,
    pad_fraction: float = 0.026,
) -> None:
    low, high = ax.get_ylim()
    pad = pad_fraction * (high - low)
    for bar, value in zip(bars, values, strict=True):
        if value > 0.0:
            y, va = value + pad, "bottom"
        elif value < 0.0:
            y, va = value - pad, "top"
        else:
            y, va = pad, "bottom"
        ax.text(
            bar.get_x() + 0.5 * bar.get_width(),
            y,
            formatter(value),
            ha="center",
            va=va,
            fontsize=8.0,
            color=COLORS["dark"],
        )


def _condition_panel(
    ax: plt.Axes,
    values: Mapping[str, float],
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
    formatter: Any,
) -> None:
    numeric = [float(values[condition]) for condition in CONDITIONS]
    bars = ax.bar(
        np.arange(2),
        numeric,
        width=0.58,
        color=(COLORS["without_edl"], COLORS["with_edl"]),
        edgecolor=COLORS["dark"],
        linewidth=0.85,
    )
    ax.set_xticks(np.arange(2), CONDITIONS)
    ax.set_ylim(*ylim)
    _style_axis(ax, title, ylabel)
    _label_linear_bars(ax, bars, numeric, formatter)


def _grouped_panel(
    ax: plt.Axes,
    values: Mapping[str, Mapping[str, float]],
    labels: tuple[str, str],
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
    formatter: Any,
) -> None:
    x = np.arange(2, dtype=float)
    width = 0.34
    without = [float(values[material]["w/o EDL"]) for material in ("Au", "Pd")]
    with_edl = [float(values[material]["with EDL"]) for material in ("Au", "Pd")]
    bars_without = ax.bar(
        x - 0.5 * width,
        without,
        width=width,
        color=COLORS["without_edl"],
        edgecolor=COLORS["dark"],
        linewidth=0.8,
    )
    bars_with = ax.bar(
        x + 0.5 * width,
        with_edl,
        width=width,
        color=COLORS["with_edl"],
        edgecolor=COLORS["dark"],
        linewidth=0.8,
    )
    ax.axhline(0.0, color=COLORS["zero"], linewidth=0.8, zorder=0)
    ax.set_xticks(x, labels)
    ax.set_ylim(*ylim)
    _style_axis(ax, title, ylabel)
    _label_linear_bars(ax, bars_without, without, formatter)
    _label_linear_bars(ax, bars_with, with_edl, formatter)


def _concentration_panel(ax: plt.Axes, values: Mapping[str, Mapping[str, Any]]) -> None:
    x = np.arange(2, dtype=float)
    width = 0.34
    lower = 1.0e-4
    without = [float(values[material]["w/o EDL"]) for material in ("Au", "Pd")]
    with_edl = [float(values[material]["with EDL"]) for material in ("Au", "Pd")]
    bars_without = ax.bar(
        x - 0.5 * width,
        np.asarray(without) - lower,
        bottom=lower,
        width=width,
        color=COLORS["without_edl"],
        edgecolor=COLORS["dark"],
        linewidth=0.8,
    )
    bars_with = ax.bar(
        x + 0.5 * width,
        np.asarray(with_edl) - lower,
        bottom=lower,
        width=width,
        color=COLORS["with_edl"],
        edgecolor=COLORS["dark"],
        linewidth=0.8,
    )
    ax.set_yscale("log")
    ax.set_ylim(lower, 180.0)
    ax.set_xticks(
        x,
        (r"Au ($\mathrm{Red1}$)", r"Pd ($\mathrm{Ox2}$)"),
    )
    _style_axis(
        ax,
        "Reactant concentration at RP",
        r"$c_i/c_{\mathrm{bulk}}$ (-)",
    )

    def label(bar: Any, value: float) -> None:
        if value < 1.0e-2:
            exponent = int(math.floor(math.log10(value)))
            coefficient = value / 10.0**exponent
            text = rf"${coefficient:.2f}\times10^{{{exponent}}}$"
        elif value >= 10.0:
            text = f"{value:.1f}"
        else:
            text = f"{value:.2f}"
        ax.text(
            bar.get_x() + 0.5 * bar.get_width(),
            value * 1.35,
            text,
            ha="center",
            va="bottom",
            fontsize=7.8,
            color=COLORS["dark"],
        )

    for bars, numeric in ((bars_without, without), (bars_with, with_edl)):
        for bar, value in zip(bars, numeric, strict=True):
            label(bar, value)


def _plot(values: Mapping[str, Any], output: Path) -> list[Path]:
    _apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(10.25, 6.35))

    _condition_panel(
        axes[0, 0],
        values["E_mix_V"],
        "Mixed potential",
        r"$E_{\mathrm{mix}}$ (V vs. RHE)",
        (0.0, 0.72),
        lambda value: f"{value:.3f}",
    )
    _condition_panel(
        axes[0, 1],
        values["i_mix_avg_A_per_m2"],
        "Mixed current density",
        r"$\bar{i}_{\mathrm{mix}}$ ($\mathrm{A\,m^{-2}}$)",
        (0.0, 0.135),
        lambda value: f"{value:.3f}",
    )
    _grouped_panel(
        axes[0, 2],
        values["phi_RP_mV"],
        ("Au", "Pd"),
        "Reaction-plane potential",
        r"$\phi_{\mathrm{RP}}$ (mV)",
        (-245.0, 25.0),
        lambda value: f"{value:.0f}",
    )
    _concentration_panel(axes[1, 0], values["reactant_concentration_over_bulk"])
    _grouped_panel(
        axes[1, 1],
        values["eta_RP_V"],
        ("Au", "Pd"),
        "Local overpotential at RP",
        r"$\eta_{\mathrm{RP}}$ (V)",
        (-0.46, 0.84),
        lambda value: f"{value:.3f}",
    )
    _grouped_panel(
        axes[1, 2],
        values["local_current_density_A_per_m2"],
        (r"Au ($i_1$)", r"Pd ($i_2$)"),
        "Local current density at RP",
        r"$i_{\mathrm{local}}$ ($\mathrm{A\,m^{-2}}$)",
        (-0.285, 0.285),
        lambda value: f"{value:.3f}",
    )

    legend_handles = [
        Patch(
            facecolor=COLORS["without_edl"],
            edgecolor=COLORS["dark"],
            linewidth=0.8,
            label="w/o EDL",
        ),
        Patch(
            facecolor=COLORS["with_edl"],
            edgecolor=COLORS["dark"],
            linewidth=0.8,
            label="with EDL",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncols=2,
        fontsize=9.3,
        handlelength=1.8,
        columnspacing=1.8,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.09,
        top=0.905,
        wspace=0.39,
        hspace=0.48,
    )

    saved: list[Path] = []
    for suffix in ("png", "svg"):
        path = output / f"{FIGURE_STEM}.{suffix}"
        fig.savefig(path, dpi=600, facecolor="white", transparent=False)
        saved.append(path)
    plt.close(fig)
    return saved


def _verify_artifacts(output: Path) -> dict[str, Any]:
    pngs = sorted(output.glob("*.png"))
    svgs = sorted(output.glob("*.svg"))
    pdfs = sorted(output.rglob("*.pdf"))
    if len(pngs) != 1 or len(svgs) != 1 or pdfs:
        raise RuntimeError(
            f"Expected 1 PNG + 1 SVG + 0 PDF, got {len(pngs)}, {len(svgs)}, {len(pdfs)}"
        )
    if any(path.stat().st_size == 0 for path in (*pngs, *svgs)):
        raise RuntimeError("A uniform-interface bar artifact is empty")
    with Image.open(pngs[0]) as image:
        image.verify()
    with Image.open(pngs[0]) as image:
        dpi = image.info.get("dpi", (0.0, 0.0))
        dpi_ok = all(595.0 <= float(value) <= 605.0 for value in dpi)
        size = [int(image.width), int(image.height)]
    svg_text = svgs[0].read_text(encoding="utf-8")
    ET.parse(svgs[0])
    editable = "<text" in svg_text and "Helvetica" in svg_text
    if not dpi_ok or not editable:
        raise RuntimeError(f"Bar export validation failed: dpi={dpi}, editable_svg={editable}")
    return {
        "png_count": 1,
        "svg_count": 1,
        "pdf_count": 0,
        "png_dimensions_px": size,
        "png_dpi": [float(value) for value in dpi],
        "svg_text_editable": editable,
        "passed": True,
    }


def _numeric_validation(
    model: IndependentPlanarEDLModel,
    with_edl: Mapping[str, Any],
    without_edl: Mapping[str, Any],
    values: Mapping[str, Any],
) -> dict[str, Any]:
    expected_checks: dict[str, Any] = {}
    expected_passed = True
    for condition, result in (("with_edl", with_edl), ("without_edl", without_edl)):
        for field, (expected, tolerance) in EXPECTED_RESULTS[condition].items():
            actual = float(result[field])
            passed = abs(actual - expected) <= tolerance
            expected_checks[f"{condition}.{field}"] = {
                "actual": actual,
                "expected": expected,
                "absolute_tolerance": tolerance,
                "passed": passed,
            }
            expected_passed = expected_passed and passed

    balance_checks: dict[str, Any] = {}
    balance_passed = True
    for condition, use_edl, result in (
        ("with_edl", True, with_edl),
        ("without_edl", False, without_edl),
    ):
        currents = model.current_components(float(result["E_mix_V"]), use_edl=use_edl)
        relative = float(currents["relative_balance_residual"])
        passed = relative < 1.0e-10
        balance_checks[condition] = {
            "relative_balance_residual": relative,
            "threshold": 1.0e-10,
            "passed": passed,
        }
        balance_passed = balance_passed and passed

    all_values = [float(row["value"]) for row in _rows(values)]
    finite = bool(np.all(np.isfinite(all_values)))
    no_edl_reference_passed = (
        values["phi_RP_mV"]["Au"]["w/o EDL"] == 0.0
        and values["phi_RP_mV"]["Pd"]["w/o EDL"] == 0.0
        and values["reactant_concentration_over_bulk"]["Au"]["w/o EDL"] == 1.0
        and values["reactant_concentration_over_bulk"]["Pd"]["w/o EDL"] == 1.0
    )
    signed_current_passed = (
        float(values["local_current_density_A_per_m2"]["Au"]["with EDL"]) > 0.0
        and float(values["local_current_density_A_per_m2"]["Pd"]["with EDL"]) < 0.0
        and float(values["local_current_density_A_per_m2"]["Au"]["w/o EDL"]) > 0.0
        and float(values["local_current_density_A_per_m2"]["Pd"]["w/o EDL"]) < 0.0
    )
    passed = (
        expected_passed
        and balance_passed
        and finite
        and no_edl_reference_passed
        and signed_current_passed
    )
    return {
        "expected_result_checks": expected_checks,
        "current_balance_checks": balance_checks,
        "all_bar_values_finite": finite,
        "without_edl_phi_zero_and_concentration_unity": no_edl_reference_passed,
        "signed_local_current_direction_correct": signed_current_passed,
        "uniformity_basis": (
            "Each electrode is an analytic planar half-space; all RP quantities "
            "are constant over its 2 nm active length and are represented by one bar."
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
        path.relative_to(output).as_posix(): _sha256(path)
        for path in paths
    }
    (output / CHECKSUM_FILE).write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in checksums.items()),
        encoding="utf-8",
    )
    return checksums


def build_independent_au_pd_uniform_bar_comparison(
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".uniform-bars-staging-", dir=output.parent))
    try:
        params = independent_params()
        model = IndependentPlanarEDLModel(params)
        with_edl = model.solve(use_edl=True)
        without_edl = model.solve(use_edl=False)
        values = _build_values(model, with_edl, without_edl)
        rows = _rows(values)

        csv_dir = staging / "csv"
        csv_dir.mkdir()
        _write_csv(csv_dir / "uniform_interface_bar_values.csv", rows)
        figure_paths = _plot(values, staging)

        summary = {
            "study_id": "independent_Au2nm_Pd2nm_CH50_uniform_bar_comparison",
            "model_id": with_edl["model_id"],
            "topology_id": with_edl["topology_id"],
            "parameters": {
                "L_Au_nm": 1.0e9 * float(params["L_Au"]),
                "L_Pd_nm": 1.0e9 * float(params["L_Pd"]),
                "C_H_Au_uF_per_cm2": 100.0 * float(params["C_H_Au"]),
                "C_H_Pd_uF_per_cm2": 100.0 * float(params["C_H_Pd"]),
                "C_tot_mM": float(params["C_tot"]),
            },
            "bar_values": values,
            "condition_colors": {
                "with EDL": COLORS["with_edl"],
                "w/o EDL": COLORS["without_edl"],
            },
            "reactant_mapping": {
                "Au": "Red1, z_R1=-1",
                "Pd": "Ox2, z_O2=+1",
            },
            "typography": {
                "variables": "italic math",
                "descriptive_subscripts_and_units": "upright roman",
            },
            "applicability_caveat": (
                "The with-EDL result exceeds the linear Debye-Huckel weak-field "
                "threshold; this is a model-internal comparison."
            ),
        }
        _write_json(staging / "summary.json", summary)

        numeric = _numeric_validation(model, with_edl, without_edl, values)
        artifacts = _verify_artifacts(staging)
        validation = {
            "numerical": numeric,
            "artifacts": artifacts,
            "debye_huckel_applicability": with_edl["debye_huckel_validity"],
            "passed": bool(numeric["passed"] and artifacts["passed"]),
        }
        if not validation["passed"]:
            raise RuntimeError(f"Uniform-interface bar validation failed: {validation}")
        _write_json(staging / "validation.json", validation)

        manifest = {
            "study_id": summary["study_id"],
            "source_sha256": _source_hashes(),
            "figure_files": [path.name for path in figure_paths],
            "data_files": ["csv/uniform_interface_bar_values.csv"],
            "metadata_files": ["summary.json", "validation.json", "manifest.json"],
            "formats": ["png", "svg"],
            "dpi_png": 600,
            "pdf_enabled": False,
        }
        _write_json(staging / "manifest.json", manifest)
        _write_checksums(staging)
        staging.rename(output)
    except Exception:
        if (
            staging.exists()
            and staging.parent == output.parent
            and staging.name.startswith(".uniform-bars-staging-")
        ):
            shutil.rmtree(staging)
        raise
    return {
        "reused": False,
        "output_dir": output,
        "summary": summary,
        "validation": validation,
    }


def _validate_existing(output: Path) -> dict[str, Any]:
    required = {
        f"{FIGURE_STEM}.png",
        f"{FIGURE_STEM}.svg",
        "csv/uniform_interface_bar_values.csv",
        "summary.json",
        "validation.json",
        "manifest.json",
    }
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != CHECKSUM_FILE
    }
    if actual != required:
        raise RuntimeError(
            f"Existing bar output coverage mismatch; missing={sorted(required-actual)}, "
            f"extra={sorted(actual-required)}"
        )
    validation = _read_json(output / "validation.json")
    if not bool(validation.get("passed")):
        raise RuntimeError("Existing bar validation.json is not passed")
    manifest = _read_json(output / "manifest.json")
    if manifest.get("source_sha256") != _source_hashes():
        raise RuntimeError("Existing bar source hashes do not match current code")

    checksum_path = output / CHECKSUM_FILE
    if not checksum_path.is_file():
        raise RuntimeError("Existing bar output is missing checksums.sha256")
    listed: set[str] = set()
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected_hash, relative = line.split("  ", 1)
        if relative in listed:
            raise RuntimeError(f"Duplicate bar checksum path: {relative}")
        listed.add(relative)
        target = output / relative
        if not target.is_file() or _sha256(target) != expected_hash:
            raise RuntimeError(f"Bar checksum mismatch: {relative}")
    if listed != actual:
        raise RuntimeError("Bar checksum coverage does not match its output tree")
    _verify_artifacts(output)
    return {
        "reused": True,
        "output_dir": output,
        "summary": _read_json(output / "summary.json"),
        "validation": validation,
    }


def build_or_reuse_independent_au_pd_uniform_bar_comparison(
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        return _validate_existing(output)
    return build_independent_au_pd_uniform_bar_comparison(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the uniform-interface Au=Pd=2 nm bar comparison."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_or_reuse_independent_au_pd_uniform_bar_comparison(args.output)
    values = result["summary"]["bar_values"]
    print(f"Output directory: {result['output_dir']}")
    print(
        "Uniform bars: "
        f"E_mix={values['E_mix_V']['with EDL']:.12f} V, "
        f"i_mix_avg={values['i_mix_avg_A_per_m2']['with EDL']:.12g} A/m^2, "
        f"reused={result['reused']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
