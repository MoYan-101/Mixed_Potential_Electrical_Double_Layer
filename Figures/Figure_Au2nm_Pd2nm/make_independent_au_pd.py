"""Build the traceable Au=Pd=2 nm independent-planar-EDL figure set.

This is a thin, parameter-locked wrapper around
``Au_Pd_independent_EDLs``.  The electrostatic model and all eight figure
classes remain owned by that package; this module only fixes the study
parameters, verifies the agreed reference results, and adds a checksum
manifest for the copied study output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = (
    ROOT
    / "Mixed_Potential_Electrical_Double_Layer"
    / "Au_Pd_independent_EDLs"
)
MODEL_SRC = MODEL_ROOT / "src"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "Au_Pd_independent"
CHECKSUM_FILE = "checksums.sha256"

if str(MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_SRC))

from au_pd_independent_edls.figures import build_results  # noqa: E402
from au_pd_independent_edls.model import (  # noqa: E402
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

    pngs = sorted(output.glob("figures/**/*.png"))
    svgs = sorted(output.glob("figures/**/*.svg"))
    pdfs = sorted(output.glob("**/*.pdf"))
    figure3_pngs = sorted((output / "figures" / "Figure_3").glob("*.png"))
    rp_pngs = sorted((output / "figures" / "rp_2d").glob("*.png"))
    editable_svg_text = all("<text" in path.read_text(encoding="utf-8") for path in svgs)
    artifact_checks = {
        "figure3_png_count": len(figure3_pngs),
        "rp_2d_png_count": len(rp_pngs),
        "png_count": len(pngs),
        "svg_count": len(svgs),
        "pdf_count": len(pdfs),
        "all_svg_text_editable": editable_svg_text,
        "passed": (
            len(figure3_pngs) == 6
            and len(rp_pngs) == 2
            and len(pngs) == 8
            and len(svgs) == 8
            and not pdfs
            and editable_svg_text
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
