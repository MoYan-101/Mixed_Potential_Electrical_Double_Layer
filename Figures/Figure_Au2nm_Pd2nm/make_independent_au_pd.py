"""Build the traceable Au=Pd=2 nm independent-planar-EDL figure set.

This is a parameter-locked wrapper around ``Au_Pd_independent_EDLs``.  The
electrostatic model and the eight base figure classes remain owned by that
package.  This module fixes the study parameters, verifies the agreed
reference results, and applies one study-local publication reflow to the
solution-potential 2D figure without changing the shared figure package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = (
    ROOT
    / "Mixed_Potential_Electrical_Double_Layer"
    / "Au_Pd_independent_EDLs"
)
MODEL_SRC = MODEL_ROOT / "src"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "Au_Pd_independent"
CHECKSUM_FILE = "checksums.sha256"
PUBLICATION_DPI = 600
SOLUTION_PHASE_STEM = "solution_phase_potential_2d_independent_edls"
SOLUTION_PHASE_LAYOUT = "main-study-only_narrow_side_by_side"
SOLUTION_PHASE_CANVAS_PX = (2026, 1852)

if str(MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_SRC))

from au_pd_independent_edls.figures import RC, build_results  # noqa: E402
from au_pd_independent_edls.model import (  # noqa: E402
    IndependentPlanarEDLModel,
    default_params,
    solve_comparison,
)


STUDY_ID = "independent_Au2nm_Pd2nm_CH50"
PARAM_OVERRIDES: dict[str, float | int | None] = {
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
    # Recalculate the electrolyte properties and charging parameters from the
    # fixed concentration and capacitances instead of inheriting overrides.
    "lambda_D": None,
    "g_Au": None,
    "g_Pd": None,
}

EXPECTED_RESULTS: dict[str, dict[str, tuple[float, float]]] = {
    "with_edl": {
        "E_mix_V": (0.6249104327868923, 5.0e-11),
        "i_mix_avg_A_per_m2": (0.04315057812685645, 5.0e-12),
        "i_mix_abs_A": (1.7260231250742582e-12, 5.0e-22),
    },
    "without_edl": {
        "E_mix_V": (0.46699999999999986, 5.0e-11),
        "i_mix_avg_A_per_m2": (0.11756035407872402, 5.0e-12),
        "i_mix_abs_A": (4.7024141631489615e-12, 5.0e-22),
    },
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def independent_params() -> dict[str, Any]:
    """Return the canonical parameter set for this fixed study."""

    params = default_params()
    params.update(PARAM_OVERRIDES)
    for name, expected in PARAM_OVERRIDES.items():
        actual = params[name]
        if expected is None:
            if actual is not None:
                raise ValueError(f"{name} must remain unset for recalculation")
        elif not math.isclose(
            float(actual),
            float(expected),
            rel_tol=0.0,
            abs_tol=max(1.0e-15, abs(float(expected)) * 1.0e-12),
        ):
            raise ValueError(f"{name}: expected {expected!r}, got {actual!r}")
    return params


def _expected_value_checks(result: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    all_passed = True
    for condition, fields in EXPECTED_RESULTS.items():
        case = result[condition]
        for field, (expected, tolerance) in fields.items():
            actual = float(case[field])
            passed = abs(actual - expected) <= tolerance
            checks[f"{condition}.{field}"] = {
                "actual": actual,
                "expected": expected,
                "absolute_difference": abs(actual - expected),
                "absolute_tolerance": tolerance,
                "passed": passed,
            }
            all_passed = all_passed and passed
    checks["passed"] = all_passed
    return checks


def _source_hashes() -> dict[str, str]:
    sources = {
        "study_wrapper": Path(__file__).resolve(),
        "independent_model": MODEL_SRC / "au_pd_independent_edls" / "model.py",
        "independent_figures": MODEL_SRC / "au_pd_independent_edls" / "figures.py",
    }
    return {name: _sha256(path) for name, path in sources.items()}


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
    checksum_path = output / CHECKSUM_FILE
    checksum_path.write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in checksums.items()),
        encoding="utf-8",
    )
    for relative, expected in checksums.items():
        actual = _sha256(output / relative)
        if actual != expected:
            raise RuntimeError(f"Checksum verification failed for {relative}")
    return checksums


def _save_fixed_canvas(fig: plt.Figure, directory: Path, stem: str) -> list[Path]:
    """Save a publication figure without tight-bbox resizing the canvas."""

    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in ("png", "svg"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(
            path,
            dpi=PUBLICATION_DPI,
            transparent=True,
            facecolor="none",
            edgecolor="none",
        )
        paths.append(path)
    plt.close(fig)
    return paths


def _png_dimensions(path: Path) -> tuple[int, int]:
    """Read PNG IHDR dimensions without introducing another dependency."""

    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Not a valid PNG with an IHDR header: {path}")
    return struct.unpack(">II", header[16:24])


def _reflow_solution_phase_potential_2d(
    model: IndependentPlanarEDLModel,
    result: Mapping[str, Any],
    output: Path,
) -> list[Path]:
    """Reflow one base figure for this study only.

    The shared independent-EDL package deliberately retains its original
    horizontal two-panel layout.  This local postprocessor keeps Au and Pd
    side by side but makes both panels narrow, so that the canvas is half as
    wide while preserving the original height, field normalization, and
    editable SVG text.
    """

    if SOLUTION_PHASE_LAYOUT != "main-study-only_narrow_side_by_side":
        raise ValueError(f"Unsupported solution-potential layout: {SOLUTION_PHASE_LAYOUT}")

    E_mix = float(result["with_edl"]["E_mix_V"])
    lambda_D = float(model.derived["lambda_D"])
    thermal_voltage = float(model.derived["thermal_voltage_V"])
    y_nm = np.linspace(0.0, 5.0 * lambda_D * 1.0e9, 241)
    fields: dict[str, dict[str, np.ndarray]] = {}
    for material in ("Au", "Pd"):
        length_nm = float(model.params[f"L_{material}"]) * 1.0e9
        x_nm = np.linspace(0.0, length_nm, 401)
        phi_profile = model.phi_tilde_profile(
            E_mix,
            material,
            y_nm * 1.0e-9,
        )
        phi_tilde = np.repeat(phi_profile[:, None], x_nm.size, axis=1)
        fields[material] = {
            "x_nm": x_nm,
            "phi_mV": phi_tilde * thermal_voltage * 1.0e3,
        }

    maximum = max(
        float(np.max(np.abs(field["phi_mV"])))
        for field in fields.values()
    )
    potential_norm = TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)
    width_px, height_px = SOLUTION_PHASE_CANVAS_PX
    # The 0.1-pixel height guard avoids a floating-point floor to 1851 pixels
    # in some Matplotlib/Pillow combinations; the exported raster is 1852 px.
    figsize = (
        width_px / PUBLICATION_DPI,
        (height_px + 0.1) / PUBLICATION_DPI,
    )
    with plt.rc_context(RC):
        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(
            1,
            3,
            width_ratios=(1.0, 1.0, 0.075),
            left=0.16,
            right=0.81,
            bottom=0.18,
            top=0.80,
            wspace=0.18,
        )
        axes = (
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[0, 1]),
        )
        axes[1].sharey(axes[0])
        cax = fig.add_subplot(grid[0, 2])
        mesh = None
        for index, (ax, material) in enumerate(
            zip(axes, ("Au", "Pd"), strict=True)
        ):
            field = fields[material]
            mesh = ax.pcolormesh(
                field["x_nm"],
                y_nm,
                field["phi_mV"],
                shading="auto",
                cmap="RdBu_r",
                norm=potential_norm,
                rasterized=True,
            )
            ax.set_title(
                f"{material} independent\nlocal EDL",
                loc="center",
                fontsize=8.3,
                linespacing=1.02,
                pad=3.0,
            )
            ax.set_xticks([0.0, 1.0, 2.0])
            ax.tick_params(labelsize=7.5, pad=2.0)
            ax.set_xlabel("local x (nm)", fontsize=8.1, labelpad=3.0)
            if index == 1:
                ax.tick_params(labelleft=False)
        if mesh is None:
            raise RuntimeError("No solution-potential field was plotted")
        colorbar = fig.colorbar(mesh, cax=cax)
        colorbar.set_label(r"$\Phi_s$ (mV)", fontsize=8.1, labelpad=2.0)
        colorbar.ax.tick_params(labelsize=7.3, pad=2.0)
        axes[0].set_ylabel(
            "distance into electrolyte (nm)", fontsize=8.5, labelpad=4.0
        )
        fig.suptitle(
            "Solution potential in two independent\nlocal half-spaces",
            x=0.08,
            y=0.97,
            ha="left",
            va="top",
            fontsize=10.0,
        )
        paths = _save_fixed_canvas(
            fig,
            output / "figures" / "rp_2d",
            SOLUTION_PHASE_STEM,
        )

    png_path = output / "figures" / "rp_2d" / f"{SOLUTION_PHASE_STEM}.png"
    actual_size = _png_dimensions(png_path)
    if actual_size != SOLUTION_PHASE_CANVAS_PX:
        raise RuntimeError(
            f"Unexpected reflowed PNG dimensions {actual_size}; "
            f"expected {SOLUTION_PHASE_CANVAS_PX}"
        )
    return paths


def build_independent_au_pd(output_dir: str | Path) -> dict[str, Any]:
    """Generate the independent Au/Pd results in a new ``output_dir``.

    The destination must not already exist.  This preserves the upstream
    package's non-overwrite policy and prevents a partial rerun from mixing
    artifacts from different parameter sets.
    """

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    params = independent_params()
    preview = solve_comparison(params)
    preview_checks = _expected_value_checks(preview)
    if not preview_checks["passed"]:
        raise RuntimeError("Independent-model reference values changed")

    built = build_results(params, output)
    result = built["result"]
    expected_checks = _expected_value_checks(result)
    if not expected_checks["passed"]:
        raise RuntimeError("Generated independent-model values failed validation")

    publication_paths = _reflow_solution_phase_potential_2d(
        IndependentPlanarEDLModel(params),
        result,
        output,
    )

    pngs = sorted(output.glob("figures/**/*.png"))
    svgs = sorted(output.glob("figures/**/*.svg"))
    pdfs = sorted(output.glob("**/*.pdf"))
    figure3_pngs = sorted((output / "figures" / "Figure_3").glob("*.png"))
    rp_pngs = sorted((output / "figures" / "rp_2d").glob("*.png"))
    solution_png = (
        output / "figures" / "rp_2d" / f"{SOLUTION_PHASE_STEM}.png"
    )
    solution_png_size = _png_dimensions(solution_png)
    editable_svg_text = all("<text" in path.read_text(encoding="utf-8") for path in svgs)
    artifact_checks = {
        "figure3_png_count": len(figure3_pngs),
        "rp_2d_png_count": len(rp_pngs),
        "png_count": len(pngs),
        "svg_count": len(svgs),
        "pdf_count": len(pdfs),
        "all_svg_text_editable": editable_svg_text,
        "solution_phase_layout": SOLUTION_PHASE_LAYOUT,
        "solution_phase_canvas_px": list(solution_png_size),
        "solution_phase_reflowed_files": [
            path.relative_to(output).as_posix() for path in publication_paths
        ],
        "passed": (
            len(figure3_pngs) == 6
            and len(rp_pngs) == 2
            and len(pngs) == 8
            and len(svgs) == 8
            and not pdfs
            and editable_svg_text
            and solution_png_size == SOLUTION_PHASE_CANVAS_PX
        ),
    }
    if not artifact_checks["passed"]:
        raise RuntimeError(f"Unexpected independent-model artifacts: {artifact_checks}")

    validation_path = output / "validation.json"
    validation = _read_json(validation_path)
    upstream_passed = bool(validation.get("passed"))
    validation.update(
        {
            "study_id": STUDY_ID,
            "upstream_validation_passed": upstream_passed,
            "expected_reference_values": expected_checks,
            "artifact_checks": artifact_checks,
            "passed": upstream_passed
            and bool(expected_checks["passed"])
            and bool(artifact_checks["passed"]),
        }
    )
    _write_json(validation_path, validation)
    if not validation["passed"]:
        raise RuntimeError("Independent-model validation did not pass")

    artifacts_path = output / "artifacts.json"
    artifacts = _read_json(artifacts_path)
    artifacts["checksums"] = CHECKSUM_FILE
    _write_json(artifacts_path, artifacts)

    manifest_path = output / "run_manifest.json"
    manifest = _read_json(manifest_path)
    manifest.update(
        {
            "study_id": STUDY_ID,
            "study_parameters": {
                "L_Au_nm": 2.0,
                "L_Pd_nm": 2.0,
                "C_H_Au_uF_per_cm2": 50.0,
                "C_H_Pd_uF_per_cm2": 50.0,
                "active_faces_Au": 1,
                "active_faces_Pd": 1,
                "C_tot_mM": 10.0,
                "equal_i0_A_per_m2": float(params["it0_1"]),
                "alpha1": 0.5,
                "alpha2": 0.5,
            },
            "parameter_provenance": (
                "au_pd_independent_edls.default_params plus locked "
                "Au=Pd=2 nm, C_H, equal-i0, and alpha overrides"
            ),
            "publication_figure_adjustments": {
                "solution_phase_potential_2d": {
                    "layout": SOLUTION_PHASE_LAYOUT,
                    "canvas_px": list(SOLUTION_PHASE_CANVAS_PX),
                    "canvas_change": (
                        "width reduced from 4051 px to the nearest integer "
                        "at 50% (2026 px); height retained at 1852 px"
                    ),
                    "implementation_scope": (
                        "local postprocessor in this study wrapper; the shared "
                        "au_pd_independent_edls figure layout remains unchanged"
                    ),
                }
            },
            "source_sha256": _source_hashes(),
            "checksums_file": CHECKSUM_FILE,
            "export": {
                "dpi_png": 600,
                "formats": ["png", "svg"],
                "svg_fonttype": "none",
                "pdf_enabled": False,
            },
            "validation_passed": True,
        }
    )
    _write_json(manifest_path, manifest)

    checksums = _write_checksums(output)
    return {
        "output": str(output),
        "manifest": manifest,
        "result": result,
        "validation": validation,
        "checksums": checksums,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="New output directory (default: %(default)s)",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    built = build_independent_au_pd(args.output)
    result = built["result"]
    print(f"output = {built['output']}")
    for condition in ("with_edl", "without_edl"):
        case = result[condition]
        print(
            f"{condition}: E_mix={float(case['E_mix_V']):.12f} V, "
            f"i_mix_avg={float(case['i_mix_avg_A_per_m2']):.12g} A/m^2, "
            f"I_mix={float(case['i_mix_abs_A']):.12g} A"
        )
    print("figures = 8 PNG + 8 SVG; PDF = 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
