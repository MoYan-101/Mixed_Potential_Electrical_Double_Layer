"""Publication polarization curve for the fixed Au=Pd=2 nm model."""

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
    / "Polarization_Scheme"
)
CHECKSUM_FILE = "checksums.sha256"
FIGURE_STEM = "independent_au_pd_polarization_with_without_edl"
CURRENT_UNIT_A = 1.0e-9  # 10^-3 microampere
N_POTENTIAL_POINTS = 1201

if str(MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_SRC))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from au_pd_independent_edls.model import IndependentPlanarEDLModel  # noqa: E402
from make_independent_au_pd import EXPECTED_RESULTS, independent_params  # noqa: E402


COLORS = {
    "au": "#3B7A57",
    "pd": "#B64342",
    "dark": "#272727",
    "gray": "#767676",
    "accent": "#8E169A",
    "zero": "#B8B8B8",
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
        "polarization_wrapper": Path(__file__).resolve(),
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
            "font.size": 9.0,
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


def _current_rows(
    model: IndependentPlanarEDLModel,
    potentials: np.ndarray,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for potential in potentials:
        with_edl = model.current_components(float(potential), use_edl=True)
        without_edl = model.current_components(float(potential), use_edl=False)
        rows.append(
            {
                "E_V": float(potential),
                "I_Au_with_A": float(with_edl["I_Au_A"]),
                "I_Pd_with_A": float(with_edl["I_Pd_A"]),
                "I_Au_without_A": float(without_edl["I_Au_A"]),
                "I_Pd_without_A": float(without_edl["I_Pd_A"]),
                "I_Au_with_1e_minus_3_uA": float(with_edl["I_Au_A"]) / CURRENT_UNIT_A,
                "I_Pd_with_1e_minus_3_uA": float(with_edl["I_Pd_A"]) / CURRENT_UNIT_A,
                "I_Au_without_1e_minus_3_uA": float(without_edl["I_Au_A"]) / CURRENT_UNIT_A,
                "I_Pd_without_1e_minus_3_uA": float(without_edl["I_Pd_A"]) / CURRENT_UNIT_A,
            }
        )
    return rows


def _write_curve_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        raise ValueError("Polarization CSV cannot be empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    rows: list[dict[str, float]],
    with_edl: Mapping[str, Any],
    without_edl: Mapping[str, Any],
    output: Path,
) -> list[Path]:
    _apply_style()
    potential = np.asarray([row["E_V"] for row in rows], dtype=float)
    curves = {
        key: np.asarray([row[key] for row in rows], dtype=float)
        for key in (
            "I_Au_with_1e_minus_3_uA",
            "I_Pd_with_1e_minus_3_uA",
            "I_Au_without_1e_minus_3_uA",
            "I_Pd_without_1e_minus_3_uA",
        )
    }

    e_with = float(with_edl["E_mix_V"])
    e_without = float(without_edl["E_mix_V"])
    i_with = float(with_edl["i_mix_abs_A"]) / CURRENT_UNIT_A
    i_without = float(without_edl["i_mix_abs_A"]) / CURRENT_UNIT_A
    drop_percent = 100.0 * (1.0 - i_with / i_without)
    y_limit = 3.0 * i_without
    no_edl_style = (0, (5.0, 2.7))

    fig, ax = plt.subplots(figsize=(7.20, 4.75))
    ax.plot(
        potential,
        curves["I_Au_with_1e_minus_3_uA"],
        color=COLORS["au"],
        linewidth=2.35,
        label="Au oxidation, with EDL",
        zorder=3,
    )
    ax.plot(
        potential,
        curves["I_Pd_with_1e_minus_3_uA"],
        color=COLORS["pd"],
        linewidth=2.35,
        label="Pd reduction, with EDL",
        zorder=3,
    )
    ax.plot(
        potential,
        curves["I_Au_without_1e_minus_3_uA"],
        color=COLORS["au"],
        linewidth=1.85,
        linestyle=no_edl_style,
        label="Au oxidation, w/o EDL",
        zorder=2,
    )
    ax.plot(
        potential,
        curves["I_Pd_without_1e_minus_3_uA"],
        color=COLORS["pd"],
        linewidth=1.85,
        linestyle=no_edl_style,
        label="Pd reduction, w/o EDL",
        zorder=2,
    )
    ax.axhline(0.0, color=COLORS["zero"], linewidth=0.8, zorder=0)

    for e_mix, magnitude, color, dash, alpha in (
        (e_without, i_without, COLORS["gray"], (0, (2.0, 2.6)), 0.78),
        (e_with, i_with, COLORS["dark"], (0, (3.0, 2.2)), 0.92),
    ):
        ax.axvline(e_mix, color=color, linewidth=1.0, linestyle=dash, alpha=alpha, zorder=1)
        ax.scatter(
            [e_mix, e_mix],
            [magnitude, -magnitude],
            s=49,
            color=[COLORS["au"], COLORS["pd"]],
            edgecolor="white",
            linewidth=0.8,
            zorder=6,
        )

    ax.text(
        0.275,
        -0.33 * y_limit,
        "w/o EDL\n"
        + rf"$E_{{\mathrm{{mix}}}} = {e_without:.3f}\ \mathrm{{V}}$"
        + "\n"
        + rf"$I_{{\mathrm{{mix}}}} = {1.0e12 * float(without_edl['i_mix_abs_A']):.2f}\ \mathrm{{pA}}$",
        ha="left",
        va="center",
        color=COLORS["gray"],
        fontsize=8.6,
        linespacing=1.22,
    )
    ax.text(
        0.735,
        0.34 * y_limit,
        "with EDL\n"
        + rf"$E_{{\mathrm{{mix}}}} = {e_with:.3f}\ \mathrm{{V}}$"
        + "\n"
        + rf"$I_{{\mathrm{{mix}}}} = {1.0e12 * float(with_edl['i_mix_abs_A']):.2f}\ \mathrm{{pA}}$",
        ha="left",
        va="center",
        color=COLORS["dark"],
        fontsize=8.6,
        linespacing=1.22,
    )

    shift_y = -0.72 * y_limit
    ax.annotate(
        "",
        xy=(e_with, shift_y),
        xytext=(e_without, shift_y),
        arrowprops={"arrowstyle": "->", "color": COLORS["dark"], "linewidth": 1.05},
    )
    ax.text(
        0.5 * (e_with + e_without),
        shift_y - 0.075 * y_limit,
        rf"$E_{{\mathrm{{mix}}}}$ shifts up by {e_with - e_without:.3f} $\mathrm{{V}}$",
        ha="center",
        va="top",
        fontsize=8.8,
        color=COLORS["dark"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 1.2},
    )

    drop_x = e_with - 0.018
    ax.annotate(
        "",
        xy=(drop_x, i_with),
        xytext=(drop_x, i_without),
        arrowprops={"arrowstyle": "->", "color": COLORS["accent"], "linewidth": 1.25},
        zorder=7,
    )
    ax.text(
        drop_x - 0.012,
        0.5 * (i_with + i_without),
        rf"$I_{{\mathrm{{mix}}}}$ drops" + "\n" + f"by {drop_percent:.1f}%",
        ha="right",
        va="center",
        color=COLORS["accent"],
        fontsize=8.8,
        linespacing=1.12,
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xlabel("Potential (V vs. RHE)", fontsize=10.5)
    ax.set_ylabel(r"Current ($10^{-3}\,\mathrm{\mu A}$)", fontsize=10.5)
    ax.tick_params(axis="both", which="major", labelsize=9.2, length=3.5, width=0.85)
    ax.legend(
        loc="upper left",
        ncols=1,
        fontsize=8.1,
        handlelength=3.0,
        borderaxespad=0.7,
    )
    fig.subplots_adjust(left=0.115, right=0.975, bottom=0.16, top=0.965)

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
        raise RuntimeError("A polarization figure artifact is empty")
    with Image.open(pngs[0]) as image:
        image.verify()
    with Image.open(pngs[0]) as image:
        dpi = image.info.get("dpi", (0.0, 0.0))
        dpi_ok = all(595.0 <= float(value) <= 605.0 for value in dpi)
        png_size = [int(image.width), int(image.height)]
    svg_text = svgs[0].read_text(encoding="utf-8")
    ET.parse(svgs[0])
    svg_editable = "<text" in svg_text and "Helvetica" in svg_text
    if not dpi_ok or not svg_editable:
        raise RuntimeError(
            f"Figure export validation failed: dpi={dpi}, editable_svg={svg_editable}"
        )
    return {
        "png_count": 1,
        "svg_count": 1,
        "pdf_count": 0,
        "png_dimensions_px": png_size,
        "png_dpi": [float(value) for value in dpi],
        "svg_text_editable": svg_editable,
        "passed": True,
    }


def _numeric_validation(
    model: IndependentPlanarEDLModel,
    rows: list[dict[str, float]],
    with_edl: Mapping[str, Any],
    without_edl: Mapping[str, Any],
) -> dict[str, Any]:
    expected_checks: dict[str, Any] = {}
    all_expected = True
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
            all_expected = all_expected and passed

    balances: dict[str, Any] = {}
    all_balanced = True
    for condition, use_edl, result in (
        ("with_edl", True, with_edl),
        ("without_edl", False, without_edl),
    ):
        currents = model.current_components(float(result["E_mix_V"]), use_edl=use_edl)
        relative = float(currents["relative_balance_residual"])
        passed = relative < 1.0e-10
        balances[condition] = {
            "I_Au_A": float(currents["I_Au_A"]),
            "I_Pd_A": float(currents["I_Pd_A"]),
            "I_Au_plus_I_Pd_A": float(currents["residual_A"]),
            "relative_balance_residual": relative,
            "threshold": 1.0e-10,
            "passed": passed,
        }
        all_balanced = all_balanced and passed

    numeric_values = np.asarray(
        [[value for value in row.values()] for row in rows], dtype=float
    )
    finite = bool(np.all(np.isfinite(numeric_values)))
    point_count_ok = len(rows) == N_POTENTIAL_POINTS
    root_range_ok = all(
        0.0 <= float(result["E_mix_V"]) <= 1.0
        for result in (with_edl, without_edl)
    )
    passed = all_expected and all_balanced and finite and point_count_ok and root_range_ok
    return {
        "expected_result_checks": expected_checks,
        "current_balance_checks": balances,
        "curve_point_count": len(rows),
        "curve_point_count_expected": N_POTENTIAL_POINTS,
        "all_curve_values_finite": finite,
        "mixed_potentials_inside_plot_range": root_range_ok,
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


def build_independent_au_pd_polarization(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".polarization-staging-", dir=output.parent))
    try:
        params = independent_params()
        model = IndependentPlanarEDLModel(params)
        with_edl = model.solve(use_edl=True)
        without_edl = model.solve(use_edl=False)
        potentials = np.linspace(0.0, 1.0, N_POTENTIAL_POINTS)
        rows = _current_rows(model, potentials)

        csv_dir = staging / "csv"
        csv_dir.mkdir()
        _write_curve_csv(csv_dir / "independent_au_pd_polarization_curves.csv", rows)
        figure_paths = _plot(rows, with_edl, without_edl, staging)

        delta_e = float(with_edl["E_mix_V"]) - float(without_edl["E_mix_V"])
        current_ratio = float(with_edl["i_mix_abs_A"]) / float(without_edl["i_mix_abs_A"])
        summary = {
            "study_id": "independent_Au2nm_Pd2nm_CH50_polarization",
            "model_id": with_edl["model_id"],
            "topology_id": with_edl["topology_id"],
            "parameters": {
                "L_Au_nm": 1.0e9 * float(params["L_Au"]),
                "L_Pd_nm": 1.0e9 * float(params["L_Pd"]),
                "C_H_Au_uF_per_cm2": 100.0 * float(params["C_H_Au"]),
                "C_H_Pd_uF_per_cm2": 100.0 * float(params["C_H_Pd"]),
                "C_tot_mM": float(params["C_tot"]),
                "active_faces_Au": int(params["active_faces_Au"]),
                "active_faces_Pd": int(params["active_faces_Pd"]),
                "out_of_plane_width_m": float(params["out_of_plane_width"]),
            },
            "with_edl": with_edl,
            "without_edl": without_edl,
            "delta_E_mix_V": delta_e,
            "I_mix_ratio_with_over_without": current_ratio,
            "I_mix_drop_fraction": 1.0 - current_ratio,
            "I_mix_drop_percent": 100.0 * (1.0 - current_ratio),
            "plot_current_unit": "10^-3 uA = 10^-9 A",
            "x_label": "Potential (V vs. RHE)",
            "y_label": "Current (10^-3 uA)",
            "curve_styles": {
                "Au": COLORS["au"],
                "Pd": COLORS["pd"],
                "with EDL": "solid",
                "w/o EDL": "dashed",
            },
            "applicability_caveat": (
                "The with-EDL result exceeds the linear Debye-Huckel weak-field "
                "threshold; this is a model-internal comparison."
            ),
        }
        _write_json(staging / "summary.json", summary)

        numeric = _numeric_validation(model, rows, with_edl, without_edl)
        artifacts = _verify_artifacts(staging)
        validation = {
            "numerical": numeric,
            "artifacts": artifacts,
            "debye_huckel_applicability": with_edl["debye_huckel_validity"],
            "passed": bool(numeric["passed"] and artifacts["passed"]),
        }
        if not validation["passed"]:
            raise RuntimeError(f"Polarization validation failed: {validation}")
        _write_json(staging / "validation.json", validation)
        manifest = {
            "study_id": summary["study_id"],
            "source_sha256": _source_hashes(),
            "figure_files": [path.name for path in figure_paths],
            "data_files": ["csv/independent_au_pd_polarization_curves.csv"],
            "metadata_files": ["summary.json", "validation.json", "manifest.json"],
        }
        _write_json(staging / "manifest.json", manifest)
        _write_checksums(staging)
        staging.rename(output)
    except Exception:
        if staging.exists() and staging.parent == output.parent and staging.name.startswith(".polarization-staging-"):
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
        "csv/independent_au_pd_polarization_curves.csv",
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
            f"Existing polarization output coverage mismatch; missing={sorted(required-actual)}, "
            f"extra={sorted(actual-required)}"
        )
    validation = _read_json(output / "validation.json")
    if not bool(validation.get("passed")):
        raise RuntimeError("Existing polarization validation.json is not passed")
    manifest = _read_json(output / "manifest.json")
    if manifest.get("source_sha256") != _source_hashes():
        raise RuntimeError("Existing polarization source hashes do not match current code")

    checksum_path = output / CHECKSUM_FILE
    if not checksum_path.is_file():
        raise RuntimeError("Existing polarization output is missing checksums.sha256")
    listed: set[str] = set()
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected_hash, relative = line.split("  ", 1)
        if relative in listed:
            raise RuntimeError(f"Duplicate polarization checksum path: {relative}")
        listed.add(relative)
        target = output / relative
        if not target.is_file() or _sha256(target) != expected_hash:
            raise RuntimeError(f"Polarization checksum mismatch: {relative}")
    if listed != actual:
        raise RuntimeError("Polarization checksum coverage does not match its output tree")
    _verify_artifacts(output)
    return {
        "reused": True,
        "output_dir": output,
        "summary": _read_json(output / "summary.json"),
        "validation": validation,
    }


def build_or_reuse_independent_au_pd_polarization(
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        return _validate_existing(output)
    return build_independent_au_pd_polarization(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Au=Pd=2 nm independent-model polarization curve."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_or_reuse_independent_au_pd_polarization(args.output)
    summary = result["summary"]
    print(f"Output directory: {result['output_dir']}")
    print(
        "Independent Au|Pd polarization: "
        f"delta E_mix={summary['delta_E_mix_V']:.12f} V, "
        f"I_mix drop={summary['I_mix_drop_percent']:.3f}%, "
        f"reused={result['reused']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
