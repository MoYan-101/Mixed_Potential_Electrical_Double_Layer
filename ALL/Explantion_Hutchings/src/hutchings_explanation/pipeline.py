"""Build and validate the four-case Hutchings explanation collection.

The collection is deliberately a reporting layer: numerical fields are loaded
by :mod:`hutchings_explanation.cases`, normalized to one explicit Au(4 nm) +
Pd(4 nm) reference reactive area for the Summary, and then rendered by
:mod:`hutchings_explanation.figures`.  This module owns only durable export,
validation, provenance, and atomic publication.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree

import matplotlib
import numpy as np

from .cases import (
    COMPARISON_AU_LENGTH_NM,
    COMPARISON_PD_LENGTH_NM,
    COMPARISON_REACTIVE_AREA_M2,
    load_all_cases,
)
from .figures import generate_case_figures, generate_summary_figures
from .polarization import (
    AU_LEGEND_LABEL,
    FIGURE_VARIANTS,
    PD_LEGEND_LABEL,
    X_AXIS_LABEL,
    Y_AXIS_LABEL,
    build_summary_polarization,
)
from .io import (
    CHECKSUM_FILENAME,
    assert_targets_absent,
    publish_staged_children,
    regular_files,
    sha256_file,
    staging_directory,
    verify_checksums,
    write_checksums,
    write_csv_rows,
    write_json,
    write_npz,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT
SCHEMA_VERSION = "1.0"
CURRENT_DEFINITION = "I_mix = |I_Au| = |I_Pd| at I_Au + I_Pd = 0"
EXPECTED_CASE_COUNT = 4
EXPECTED_SUMMARY_PAIRS = 4
EXPECTED_CASE_PAIRS = 9
EXPECTED_PNG_COUNT = 40
EXPECTED_SVG_COUNT = 40
EXPECTED_PDF_COUNT = 0
BALANCE_RELATIVE_TOLERANCE = 1.0e-10
C10_C1000_E_TOLERANCE_MV = 0.3
C10_C1000_I_TOLERANCE_PERCENT = 0.05
EXPECTED_DISPLAY_SEGMENTS = {
    "au_pd_independent": (
        ("Au", "Au", 0.0, 2.0),
        ("Pd", "Pd", 0.0, 2.0),
    ),
    "janus_au_pd": (
        ("Au", "Au", 0.0, 2.0),
        ("Pd", "Pd", 2.0, 4.0),
    ),
    "physical_mixture_c10": (
        ("Au", "Au", 0.0, 2.0),
        ("C", "C", 2.0, 12.0),
        ("Pd", "Pd", 12.0, 14.0),
    ),
    "janus_on_c_support": (
        ("C_left", "C", 0.0, 5.0),
        ("Au", "Au", 5.0, 9.0),
        ("Pd", "Pd", 9.0, 13.0),
        ("C_right", "C", 13.0, 18.0),
    ),
}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite; got {result!r}")
    return result


def _close(actual: float, expected: float, *, name: str, atol: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=atol):
        raise RuntimeError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def _case_row(index: int, case: Mapping[str, Any]) -> dict[str, Any]:
    area = _finite(case["comparison_reactive_area_m2"], "comparison area")
    i_with = _finite(case["I_with_A"], "I_with_A")
    i_no = _finite(case["I_no_A"], "I_no_A")
    return {
        "case_order": index,
        "case_key": str(case["key"]),
        "directory_name": str(case["directory_name"]),
        "display_label": str(case["display_label"]),
        "topology_kind": str(case["topology_kind"]),
        "E_mix_without_edl_V": _finite(case["E_no_V"], "E_no_V"),
        "E_mix_with_edl_V": _finite(case["E_with_V"], "E_with_V"),
        "I_mix_without_edl_A": i_no,
        "I_mix_with_edl_A": i_with,
        "I_mix_without_edl_pA": 1.0e12 * i_no,
        "I_mix_with_edl_pA": 1.0e12 * i_with,
        "common_reactive_area_m2": area,
        "comparison_Au_length_nm": COMPARISON_AU_LENGTH_NM,
        "comparison_Pd_length_nm": COMPARISON_PD_LENGTH_NM,
        "source_reactive_area_m2": _finite(
            case["raw_current"]["source_reactive_area_m2"], "source reactive area"
        ),
        "source_I_mix_without_edl_A": _finite(
            case["raw_current"]["no_A"], "source I no"
        ),
        "source_I_mix_with_edl_A": _finite(
            case["raw_current"]["with_A"], "source I with"
        ),
        "common_area_current_density_without_edl_A_per_m2": i_no / area,
        "common_area_current_density_with_edl_A_per_m2": i_with / area,
        "source_current_scale_factor": _finite(case["scale_factor"], "scale_factor"),
        "I_mix_definition": CURRENT_DEFINITION,
    }


def _c_has_no_faradaic_current(case: Mapping[str, Any]) -> bool:
    surface = case.get("surface")
    if not isinstance(surface, Mapping):
        return True
    material = np.asarray(surface.get("material", []), dtype=str)
    c_mask = material == "C"
    if not np.any(c_mask):
        return True
    current_fields = (
        "j_Au_with_A_per_m2",
        "j_Pd_with_A_per_m2",
        "j_Au_no_A_per_m2",
        "j_Pd_no_A_per_m2",
        "current_density_with_A_per_m2",
        "current_density_no_A_per_m2",
    )
    for field in current_fields:
        values = np.asarray(surface[field], dtype=float)
        if values.shape != material.shape or np.any(np.isfinite(values[c_mask])):
            return False
    return True


def _validate_cases(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(cases) != EXPECTED_CASE_COUNT:
        raise RuntimeError(f"Expected four comparison cases; got {len(cases)}")
    expected_keys = (
        "au_pd_independent",
        "janus_au_pd",
        "physical_mixture_c10",
        "janus_on_c_support",
    )
    actual_keys = tuple(str(case["key"]) for case in cases)
    if actual_keys != expected_keys:
        raise RuntimeError(f"Unexpected case order: {actual_keys!r}")
    directories = [str(case["directory_name"]) for case in cases]
    if len(directories) != len(set(directories)):
        raise RuntimeError("Case output directory names are not unique")

    reports: list[dict[str, Any]] = []
    for case in cases:
        key = str(case["key"])
        area = _finite(case["comparison_reactive_area_m2"], f"{key} area")
        _close(
            area,
            COMPARISON_REACTIVE_AREA_M2,
            name=f"{key} common reactive area",
            atol=1.0e-24,
        )
        diagnostics = case["diagnostics"]
        au_length = _finite(diagnostics["effective_Au_length_nm"], "Au length")
        pd_length = _finite(diagnostics["effective_Pd_length_nm"], "Pd length")
        _close(
            au_length,
            COMPARISON_AU_LENGTH_NM,
            name=f"{key} comparison Au width",
            atol=1.0e-12,
        )
        _close(
            pd_length,
            COMPARISON_PD_LENGTH_NM,
            name=f"{key} comparison Pd width",
            atol=1.0e-12,
        )
        expected_segments = EXPECTED_DISPLAY_SEGMENTS[key]
        actual_segments = case["segments"]
        if len(actual_segments) != len(expected_segments):
            raise RuntimeError(
                f"{key} display segment count mismatch: "
                f"expected {len(expected_segments)}, got {len(actual_segments)}"
            )
        display_segment_report: list[dict[str, Any]] = []
        for segment, (name, material, start_nm, end_nm) in zip(
            actual_segments, expected_segments, strict=True
        ):
            if str(segment["name"]) != name or str(segment["material"]) != material:
                raise RuntimeError(
                    f"{key} display segment identity mismatch: expected "
                    f"{name}/{material}, got {segment['name']}/{segment['material']}"
                )
            actual_start = _finite(segment["x_start_nm"], f"{key} {name} start")
            actual_end = _finite(segment["x_end_nm"], f"{key} {name} end")
            _close(
                actual_start,
                start_nm,
                name=f"{key} {name} display start",
                atol=1.0e-12,
            )
            _close(
                actual_end,
                end_nm,
                name=f"{key} {name} display end",
                atol=1.0e-12,
            )
            display_segment_report.append(
                {
                    "name": name,
                    "material": material,
                    "x_start_nm": actual_start,
                    "x_end_nm": actual_end,
                }
            )

        raw = case["raw_current"]
        scale = _finite(raw["scale_factor_to_comparison"], f"{key} scale")
        _close(scale, _finite(case["scale_factor"], f"{key} scale"), name=f"{key} scale", atol=1.0e-15)
        source_area = _finite(raw["source_reactive_area_m2"], f"{key} source area")
        _close(
            scale,
            COMPARISON_REACTIVE_AREA_M2 / source_area,
            name=f"{key} reference-area scale",
            atol=1.0e-15,
        )
        raw_with = _finite(raw["with_A"], f"{key} raw with-EDL current")
        raw_no = _finite(raw["no_A"], f"{key} raw w/o-EDL current")
        i_with = _finite(case["I_with_A"], f"{key} with-EDL current")
        i_no = _finite(case["I_no_A"], f"{key} w/o-EDL current")
        if min(raw_with, raw_no, i_with, i_no, scale) <= 0.0:
            raise RuntimeError(f"{key} contains a non-positive mixed current or scale")
        _close(i_with, scale * raw_with, name=f"{key} with-EDL area conversion", atol=1.0e-24)
        _close(i_no, scale * raw_no, name=f"{key} w/o-EDL area conversion", atol=1.0e-24)

        summary = case["summary"]
        _close(
            _finite(summary["with_edl"]["I_mix_A"], "summary I with"),
            i_with,
            name=f"{key} with-EDL summary current",
            atol=1.0e-24,
        )
        _close(
            _finite(summary["without_edl"]["I_mix_A"], "summary I no"),
            i_no,
            name=f"{key} w/o-EDL summary current",
            atol=1.0e-24,
        )
        balance_with = abs(_finite(diagnostics["relative_balance_with"], "balance with"))
        balance_no = abs(_finite(diagnostics["relative_balance_no"], "balance no"))
        if balance_with >= BALANCE_RELATIVE_TOLERANCE or balance_no >= BALANCE_RELATIVE_TOLERANCE:
            raise RuntimeError(
                f"{key} mixed-current balance exceeds {BALANCE_RELATIVE_TOLERANCE:g}"
            )
        c_no_current = _c_has_no_faradaic_current(case)
        if not c_no_current:
            raise RuntimeError(f"{key} contains finite Faradaic current on C")
        reports.append(
            {
                "case_key": key,
                "effective_Au_length_nm": au_length,
                "effective_Pd_length_nm": pd_length,
                "comparison_reactive_area_m2": area,
                "source_reactive_area_m2": source_area,
                "display_segments": display_segment_report,
                "current_scale_factor": scale,
                "reported_current_equals_raw_times_scale": True,
                "relative_balance_with": balance_with,
                "relative_balance_without": balance_no,
                "C_has_no_faradaic_current": c_no_current,
                "passed": True,
            }
        )

    physical = next(case for case in cases if case["key"] == "physical_mixture_c10")
    plateau = physical["diagnostics"].get("C10_vs_C1000_plateau")
    if not isinstance(plateau, Mapping):
        raise RuntimeError("Physical-mixture case lacks its C10/C1000 plateau comparison")
    delta_e = abs(_finite(plateau["delta_E_C10_minus_C1000_mV"], "C10/C1000 delta E"))
    delta_i = abs(_finite(plateau["delta_I_C10_minus_C1000_percent"], "C10/C1000 delta I"))
    plateau_passed = (
        delta_e < C10_C1000_E_TOLERANCE_MV
        and delta_i < C10_C1000_I_TOLERANCE_PERCENT
    )
    if not plateau_passed:
        raise RuntimeError(
            "C10 has not reached the C1000 plateau at the required thresholds: "
            f"delta E={delta_e:.6g} mV, delta I={delta_i:.6g}%"
        )
    e_no_values = np.asarray([_finite(case["E_no_V"], "E_no_V") for case in cases])
    i_no_values = np.asarray([_finite(case["I_no_A"], "I_no_A") for case in cases])
    if not np.allclose(e_no_values, e_no_values[0], rtol=1.0e-12, atol=1.0e-12):
        raise RuntimeError("The four cases do not share one common w/o-EDL E_mix baseline")
    if not np.allclose(i_no_values, i_no_values[0], rtol=1.0e-12, atol=1.0e-24):
        raise RuntimeError("The four cases do not share one common w/o-EDL I_mix baseline")
    return {
        "common_area": {
            "out_of_plane_width_m": 0.01,
            "effective_Au_length_nm": COMPARISON_AU_LENGTH_NM,
            "effective_Pd_length_nm": COMPARISON_PD_LENGTH_NM,
            "reactive_area_m2": COMPARISON_REACTIVE_AREA_M2,
            "current_definition": CURRENT_DEFINITION,
            "passed": True,
        },
        "case_checks": reports,
        "C10_vs_C1000_plateau": {
            "delta_E_mV": delta_e,
            "delta_I_percent": delta_i,
            "E_tolerance_mV_strict": C10_C1000_E_TOLERANCE_MV,
            "I_tolerance_percent_strict": C10_C1000_I_TOLERANCE_PERCENT,
            "interpretation": "plateau-equivalent at the stated plotting precision, not bit-identical",
            "passed": True,
        },
        "common_without_edl_baseline": {
            "E_mix_V": float(np.mean(e_no_values)),
            "I_mix_A": float(np.mean(i_no_values)),
            "case_count": len(cases),
            "summary_bar_count_per_metric": 1,
            "passed": True,
        },
        "passed": True,
    }


def _surface_rows(surface: Mapping[str, Any], **constants: Any) -> list[dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    size: int | None = None
    for name, raw in surface.items():
        array = np.asarray(raw)
        if array.ndim != 1:
            continue
        if size is None:
            size = int(array.size)
        if array.size == size:
            arrays[str(name)] = array
    if size is None:
        return []
    return [
        {**constants, **{name: values[index] for name, values in arrays.items()}}
        for index in range(size)
    ]


def _charge_rows(segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in segments:
        x = np.asarray(segment["x_nm"])
        phi = np.asarray(segment["phi_RP_V"])
        sigma = np.asarray(segment["sigma_C_per_m2"])
        if not (x.ndim == phi.ndim == sigma.ndim == 1 and x.size == phi.size == sigma.size):
            raise RuntimeError(f"Invalid charge arrays for segment {segment['name']}")
        rows.extend(
            {
                "segment_id": str(segment["name"]),
                "material": str(segment["material"]),
                "x_nm": x[index],
                "phi_RP_V": phi[index],
                "sigma_C_per_m2": sigma[index],
            }
            for index in range(x.size)
        )
    return rows


def _write_case_data(case: Mapping[str, Any], case_root: Path) -> list[Path]:
    data = case_root / "data"
    written: list[Path] = []
    summary = {
        "case_key": case["key"],
        "display_label": case["display_label"],
        "topology_kind": case["topology_kind"],
        "current_definition": CURRENT_DEFINITION,
        "comparison_reactive_area_m2": case["comparison_reactive_area_m2"],
        "E_mix_without_edl_V": case["E_no_V"],
        "E_mix_with_edl_V": case["E_with_V"],
        "comparison_I_mix_without_edl_A": case["I_no_A"],
        "comparison_I_mix_with_edl_A": case["I_with_A"],
        "comparison_I_mix_without_edl_pA": 1.0e12 * float(case["I_no_A"]),
        "comparison_I_mix_with_edl_pA": 1.0e12 * float(case["I_with_A"]),
        "native_I_mix_without_edl_A": case["raw_current"]["no_A"],
        "native_I_mix_with_edl_A": case["raw_current"]["with_A"],
        "native_I_mix_without_edl_pA": 1.0e12
        * float(case["raw_current"]["no_A"]),
        "native_I_mix_with_edl_pA": 1.0e12
        * float(case["raw_current"]["with_A"]),
        "figure_3_panel_a_current_basis": "native/source computational cell",
        "summary_current_basis": "area-matched Au(4 nm)+Pd(4 nm) reference cell",
        "raw_current": case["raw_current"],
        "segments": case["segments"],
        "diagnostics": case["diagnostics"],
        "source_paths": case["source_paths"],
    }
    written.extend(
        (
            write_json(data / "summary.json", summary),
            write_json(data / "parameters.json", case["params"]),
            write_json(data / "segments.json", case["segments"]),
            write_csv_rows(data / "segments.csv", list(case["segments"])),
        )
    )

    surface_rows: list[dict[str, Any]] = []
    surface_npz: dict[str, Any] = {}
    grid_npz: dict[str, Any] = {}
    if case["topology_kind"] == "continuous":
        surface = case["surface"]
        grid = case["grid_2d"]
        surface_rows = _surface_rows(
            surface, coordinate_system="continuous_Neumann_computational_halfcell"
        )
        surface_npz = dict(surface)
        grid_npz = dict(grid)
    else:
        for material, record in case["independent_surfaces"].items():
            prefix = str(material)
            surface_rows.extend(
                _surface_rows(
                    record["surface"],
                    coordinate_system=record["coordinate_system"],
                )
            )
            surface_npz.update(
                {f"{prefix}_{name}": value for name, value in record["surface"].items()}
            )
            grid_npz.update(
                {f"{prefix}_{name}": value for name, value in record["grid_2d"].items()}
            )
    written.extend(
        (
            write_csv_rows(data / "surface_profiles.csv", surface_rows),
            write_npz(data / "surface_profiles.npz", surface_npz),
            write_npz(data / "display_grid_2d.npz", grid_npz),
        )
    )
    charge_rows = _charge_rows(case["charge_segments"])
    written.append(write_csv_rows(data / "surface_charge_distribution.csv", charge_rows))
    written.append(
        write_npz(
            data / "surface_charge_distribution.npz",
            {
                "segment_id": np.asarray([row["segment_id"] for row in charge_rows], dtype=str),
                "material": np.asarray([row["material"] for row in charge_rows], dtype=str),
                "x_nm": np.asarray([row["x_nm"] for row in charge_rows], dtype=float),
                "phi_RP_V": np.asarray([row["phi_RP_V"] for row in charge_rows], dtype=float),
                "sigma_C_per_m2": np.asarray(
                    [row["sigma_C_per_m2"] for row in charge_rows], dtype=float
                ),
            },
        )
    )
    written.append(
        write_json(
            data / "data_manifest.json",
            {
                "case_key": case["key"],
                "surface_coordinate_convention": (
                    "separate Au/Pd half-spaces"
                    if case["topology_kind"] == "independent"
                    else "one continuous Neumann computational half-cell"
                ),
                "files": [
                    "summary.json",
                    "parameters.json",
                    "segments.json",
                    "segments.csv",
                    "surface_profiles.csv",
                    "surface_profiles.npz",
                    "surface_charge_distribution.csv",
                    "surface_charge_distribution.npz",
                    "display_grid_2d.npz",
                ],
            },
        )
    )
    return written


def _workspace_relative(path: Path, workspace_root: Path) -> str:
    try:
        return path.resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _source_registry(
    cases: Sequence[Mapping[str, Any]], workspace_root: Path
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case in cases:
        for role, raw_path in case["source_paths"].items():
            path = Path(str(raw_path)).expanduser().resolve()
            token = str(path)
            if token in seen:
                continue
            if not path.is_file():
                raise FileNotFoundError(f"Missing source artifact: {path}")
            seen.add(token)
            records.append(
                {
                    "kind": "model_input",
                    "case_key": case["key"],
                    "role": role,
                    "path": _workspace_relative(path, workspace_root),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    package_files = sorted((PACKAGE_ROOT / "src" / "hutchings_explanation").glob("*.py"))
    entrypoint = PACKAGE_ROOT / "make_all_explantion_hutchings.py"
    if entrypoint.is_file():
        package_files.append(entrypoint)
    source_hashes: dict[str, str] = {}
    for path in package_files:
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        digest = sha256_file(path)
        source_hashes[relative] = digest
        records.append(
            {
                "kind": "collection_source",
                "role": "implementation",
                "path": _workspace_relative(path, workspace_root),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    combined = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    paper = workspace_root / "MS" / "s41586-022-04397-7.pdf"
    if paper.is_file():
        records.append(
            {
                "kind": "interpretive_reference",
                "role": "Hutchings 2022 paper",
                "path": _workspace_relative(paper, workspace_root),
                "size_bytes": paper.stat().st_size,
                "sha256": sha256_file(paper),
            }
        )
    return {
        "workspace_root": str(workspace_root),
        "package_source_sha256": combined,
        "record_count": len(records),
        "records": records,
    }


def _environment_snapshot() -> dict[str, Any]:
    return {
        "created_at": _now(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "working_directory": os.getcwd(),
    }


def _visible_svg_text(path: Path) -> str:
    root = ElementTree.parse(path).getroot()
    parts: list[str] = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == "text":
            parts.append("".join(element.itertext()))
    return " ".join(parts)


def _validate_figures(
    staging_root: Path,
    cases: Sequence[Mapping[str, Any]],
    returned_paths: Sequence[Path],
) -> dict[str, Any]:
    pngs = sorted(staging_root.rglob("*.png"))
    svgs = sorted(staging_root.rglob("*.svg"))
    pdfs = sorted(staging_root.rglob("*.pdf"))
    expected_returned = EXPECTED_PNG_COUNT + EXPECTED_SVG_COUNT
    if len(returned_paths) != expected_returned:
        raise RuntimeError(
            f"Figure functions returned {len(returned_paths)} paths, "
            f"expected {expected_returned}"
        )
    if len(pngs) != EXPECTED_PNG_COUNT or len(svgs) != EXPECTED_SVG_COUNT or pdfs:
        raise RuntimeError(
            f"Expected {EXPECTED_PNG_COUNT} PNG + {EXPECTED_SVG_COUNT} SVG + "
            f"{EXPECTED_PDF_COUNT} PDF; got {len(pngs)}, {len(svgs)}, {len(pdfs)}"
        )
    summary_png = list((staging_root / "Summary" / "figures").glob("*.png"))
    summary_svg = list((staging_root / "Summary" / "figures").glob("*.svg"))
    if len(summary_png) != EXPECTED_SUMMARY_PAIRS or len(summary_svg) != EXPECTED_SUMMARY_PAIRS:
        raise RuntimeError(
            f"Summary must contain {EXPECTED_SUMMARY_PAIRS} PNG/SVG figure pairs"
        )
    polarization_svgs = [
        staging_root / "Summary" / "figures" / f"{variant['stem']}.svg"
        for variant in FIGURE_VARIANTS
    ]
    missing_polarization_svgs = [path for path in polarization_svgs if not path.is_file()]
    if missing_polarization_svgs:
        raise RuntimeError(
            f"Missing polarization SVG variant(s): {missing_polarization_svgs}"
        )
    mislabeled_polarization_svgs = [
        path for path in polarization_svgs if X_AXIS_LABEL not in _visible_svg_text(path)
    ]
    if mislabeled_polarization_svgs:
        raise RuntimeError(
            "Polarization SVG x-axis label must be "
            f"{X_AXIS_LABEL!r}: {mislabeled_polarization_svgs}"
        )
    polarization_label_errors: list[str] = []
    for path in polarization_svgs:
        visible_text = _visible_svg_text(path)
        raw_svg = path.read_text(encoding="utf-8")
        if AU_LEGEND_LABEL not in visible_text or PD_LEGEND_LABEL not in visible_text:
            polarization_label_errors.append(f"{path}: reaction legend labels")
        if "Half reaction" in visible_text:
            polarization_label_errors.append(f"{path}: obsolete Half reaction title")
        micro_styles = re.findall(
            r'<tspan\b[^>]*style="([^"]*)"[^>]*>µ</tspan>', raw_svg
        )
        if not micro_styles or any("italic" in style.lower() for style in micro_styles):
            polarization_label_errors.append(f"{path}: µA unit is not explicitly upright")
    if polarization_label_errors:
        raise RuntimeError(
            "Polarization SVG label validation failed: "
            + "; ".join(polarization_label_errors)
        )
    case_counts: list[dict[str, Any]] = []
    for case in cases:
        case_root = staging_root / str(case["directory_name"])
        f3_png = list((case_root / "Figure_3").glob("*.png"))
        f3_svg = list((case_root / "Figure_3").glob("*.svg"))
        rp_png = list((case_root / "Figure_RP").glob("*.png"))
        rp_svg = list((case_root / "Figure_RP").glob("*.svg"))
        if tuple(map(len, (f3_png, f3_svg, rp_png, rp_svg))) != (6, 6, 3, 3):
            raise RuntimeError(f"Wrong figure counts for {case['key']}")
        case_counts.append(
            {
                "case_key": case["key"],
                "Figure_3_png": 6,
                "Figure_3_svg": 6,
                "Figure_RP_png": 3,
                "Figure_RP_svg": 3,
                "passed": True,
            }
        )
    noneditable = [path for path in svgs if "<text" not in path.read_text(encoding="utf-8")]
    if noneditable:
        raise RuntimeError(f"SVG files without editable text: {noneditable}")
    visible_local = [
        path for path in svgs if re.search(r"\bLocal\b", _visible_svg_text(path), re.IGNORECASE)
    ]
    if visible_local:
        raise RuntimeError(f"Visible word 'Local' remains in SVG figures: {visible_local}")
    empty = [path for path in (*pngs, *svgs) if path.stat().st_size == 0]
    if empty:
        raise RuntimeError(f"Empty figure artifact(s): {empty}")
    return {
        "returned_path_count": len(returned_paths),
        "png_count": len(pngs),
        "svg_count": len(svgs),
        "pdf_count": len(pdfs),
        "summary_pairs": EXPECTED_SUMMARY_PAIRS,
        "summary_bar_layout": {
            "without_edl_common_bars_per_figure": 1,
            "with_edl_case_bars_per_figure": len(cases),
            "passed": True,
        },
        "polarization_figure_layout": {
            "variant_count": len(FIGURE_VARIANTS),
            "variants": FIGURE_VARIANTS,
            "x_axis_label": X_AXIS_LABEL,
            "y_axis_label": Y_AXIS_LABEL,
            "reaction_legend_labels": [AU_LEGEND_LABEL, PD_LEGEND_LABEL],
            "obsolete_half_reaction_title_absent": True,
            "current_unit_upright": True,
            "passed": True,
        },
        "case_counts": case_counts,
        "all_svg_text_editable": True,
        "visible_word_Local_absent": True,
        "expected": (
            f"{EXPECTED_PNG_COUNT} PNG + {EXPECTED_SVG_COUNT} editable-text SVG + "
            f"{EXPECTED_PDF_COUNT} PDF"
        ),
        "passed": True,
    }


def _relative_files(paths: Iterable[Path], root: Path) -> list[str]:
    return sorted({Path(path).resolve().relative_to(root.resolve()).as_posix() for path in paths})


def _write_summary(cases: Sequence[Mapping[str, Any]], root: Path) -> list[Path]:
    rows = [_case_row(index, case) for index, case in enumerate(cases, start=1)]
    summary = {
        "comparison_name": "Hutchings 2022 four-case mixed-potential/EDL comparison",
        "current_definition": CURRENT_DEFINITION,
        "common_reactive_area_m2": COMPARISON_REACTIVE_AREA_M2,
        "comparison_reference_geometry": {
            "out_of_plane_width_m": 0.01,
            "effective_Au_length_nm": COMPARISON_AU_LENGTH_NM,
            "effective_Pd_length_nm": COMPARISON_PD_LENGTH_NM,
        },
        "summary_figure_layout": {
            "without_edl_common_bars_per_figure": 1,
            "with_edl_case_bars_per_figure": len(cases),
            "polarization_with_edl_cases": len(cases),
            "polarization_without_edl_references": 1,
            "polarization_figure_variants": len(FIGURE_VARIANTS),
            "polarization_x_axis_label": X_AXIS_LABEL,
            "polarization_y_axis_label": Y_AXIS_LABEL,
            "polarization_reaction_legend_labels": [
                AU_LEGEND_LABEL,
                PD_LEGEND_LABEL,
            ],
        },
        "case_results": rows,
        "physical_mixture_C10_note": (
            "C10 is plateau-equivalent to C1000 at <0.3 mV and <0.05%, "
            "but the values are not bit-identical."
        ),
    }
    return [
        write_csv_rows(root / "Summary" / "csv" / "four_case_comparison.csv", rows),
        write_json(root / "Summary" / "summary.json", summary),
    ]


def build_collection(
    workspace_root: Path,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Generate, validate, and atomically publish the complete collection.

    The destination is intentionally non-overwriting.  Existing generated
    targets cause an immediate :class:`FileExistsError`; package source and
    documentation beside those targets are left untouched.
    """

    workspace = Path(workspace_root).expanduser().resolve()
    if not (workspace / "Figures").is_dir():
        raise FileNotFoundError(f"Not a 2026 workspace root: {workspace}")
    destination = Path(output_root or DEFAULT_OUTPUT_ROOT).expanduser().resolve()
    case_directories = [
        "01_Au_Pd_independent",
        "02_Janus_Au2_Pd2",
        "03_Physical_mixture_C10",
        "04_Janus_on_C_support",
    ]
    publication_names = [
        "Summary",
        *case_directories,
        "inputs",
        "manifest.json",
        "validation.json",
        "artifacts.json",
        "environment.json",
        CHECKSUM_FILENAME,
    ]
    assert_targets_absent(destination, publication_names)

    cases = load_all_cases(workspace)
    actual_directories = [str(case["directory_name"]) for case in cases]
    if actual_directories != case_directories:
        raise RuntimeError(
            f"Case directory contract changed: expected {case_directories}, got {actual_directories}"
        )
    case_validation = _validate_cases(cases)
    created_at = _now()

    with staging_directory(destination) as staging:
        written: list[Path] = []
        written.extend(_write_summary(cases, staging))
        figure_paths: list[Path] = []
        figure_paths.extend(generate_summary_figures(cases, staging / "Summary" / "figures"))
        polarization = build_summary_polarization(cases, workspace, staging / "Summary")
        written.extend(Path(path) for path in polarization["data_paths"])
        figure_paths.extend(Path(path) for path in polarization["figure_paths"])
        for case in cases:
            case_root = staging / str(case["directory_name"])
            written.extend(_write_case_data(case, case_root))
            figure_paths.extend(generate_case_figures(case, case_root))

        source_registry = _source_registry(cases, workspace)
        source_path = write_json(staging / "inputs" / "source_registry.json", source_registry)
        written.append(source_path)
        figure_validation = _validate_figures(staging, cases, figure_paths)
        validation = {
            "schema_version": SCHEMA_VERSION,
            "created_at": created_at,
            "case_validation": case_validation,
            "figure_validation": figure_validation,
            "polarization_validation": polarization["validation"],
            "checksum_validation": {
                "algorithm": "SHA-256",
                "scope": "all generated files except checksums.sha256 itself",
                "complete": True,
                "passed": True,
            },
            "physical_caveats": {
                "linear_Debye_Huckel": (
                    "All with-EDL cases exceed the weak-potential |phi_tilde| <= 1 regime."
                ),
                "Fourier_Gibbs": (
                    "Cosine/Fourier material discontinuities may retain boundary ringing."
                ),
                "geometry": (
                    "The spatial models are coplanar stripe surrogates, not curved 3D particles."
                ),
            },
            "passed": True,
        }
        validation_path = write_json(staging / "validation.json", validation)
        environment_path = write_json(staging / "environment.json", _environment_snapshot())
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "collection_name": "Explantion_Hutchings",
            "created_at": created_at,
            "workspace_root": str(workspace),
            "output_directory": str(destination),
            "generated_by": "hutchings_explanation.pipeline.build_collection",
            "case_directories": case_directories,
            "current_definition": CURRENT_DEFINITION,
            "common_reactive_area_m2": COMPARISON_REACTIVE_AREA_M2,
            "figure_formats": ["png", "svg"],
            "png_dpi": 600,
            "svg_fonttype": "none",
            "pdf_enabled": False,
            "expected_figure_artifacts": (
                f"{EXPECTED_PNG_COUNT} PNG + {EXPECTED_SVG_COUNT} SVG + "
                f"{EXPECTED_PDF_COUNT} PDF"
            ),
            "summary_figure_layout": {
                "without_edl_common_bars_per_figure": 1,
                "with_edl_case_bars_per_figure": len(cases),
                "polarization_with_edl_cases": len(cases),
                "polarization_without_edl_references": 1,
                "polarization_figure_variants": len(FIGURE_VARIANTS),
                "polarization_x_axis_label": X_AXIS_LABEL,
                "polarization_y_axis_label": Y_AXIS_LABEL,
                "polarization_reaction_legend_labels": [
                    AU_LEGEND_LABEL,
                    PD_LEGEND_LABEL,
                ],
            },
            "source_registry": "inputs/source_registry.json",
            "package_source_sha256": source_registry["package_source_sha256"],
            "validation": "validation.json",
            "artifacts": "artifacts.json",
            "checksums": CHECKSUM_FILENAME,
            "validation_passed": True,
        }
        manifest_path = write_json(staging / "manifest.json", manifest)
        written.extend((validation_path, environment_path, manifest_path))

        payload_files = regular_files(staging, exclude=("artifacts.json", CHECKSUM_FILENAME))
        payload_relatives = _relative_files(payload_files, staging)
        artifacts = {
            "schema_version": SCHEMA_VERSION,
            "created_at": created_at,
            "payload_file_count_before_artifact_registry": len(payload_relatives),
            "payload_files": payload_relatives,
            "figure_files": _relative_files(figure_paths, staging),
            "figure_counts": {
                "png": EXPECTED_PNG_COUNT,
                "svg": EXPECTED_SVG_COUNT,
                "pdf": EXPECTED_PDF_COUNT,
            },
            "checksums_file": CHECKSUM_FILENAME,
            "checksum_scope_note": (
                "checksums.sha256 covers every generated file including artifacts.json, "
                "and excludes only itself; static package source/README files at the destination "
                "are outside the generated-artifact scope."
            ),
        }
        artifacts_path = write_json(staging / "artifacts.json", artifacts)
        checksum_path, staged_checksums = write_checksums(staging)
        verified_stage = verify_checksums(staging, require_complete=True)
        if verified_stage != staged_checksums:
            raise RuntimeError("Staged checksum verification did not reproduce the registry")
        expected_checksum_paths = {
            path.relative_to(staging).as_posix()
            for path in regular_files(staging, exclude=(CHECKSUM_FILENAME,))
        }
        if set(staged_checksums) != expected_checksum_paths:
            raise RuntimeError("Staged checksum registry is incomplete")

        published = publish_staged_children(staging, destination, publication_names)

    verified_destination = verify_checksums(destination, require_complete=False)
    if verified_destination != staged_checksums:
        raise RuntimeError("Published checksums differ from the validated staging registry")
    if set(verified_destination) != expected_checksum_paths:
        raise RuntimeError("Published checksum registry does not cover every generated artifact")

    return {
        "output_root": str(destination),
        "case_directories": case_directories,
        "summary_rows": [_case_row(index, case) for index, case in enumerate(cases, start=1)],
        "validation": validation,
        "source_registry": source_registry,
        "figure_paths": [
            str(destination / relative)
            for relative in artifacts["figure_files"]
        ],
        "published_targets": [str(path) for path in published],
        "checksums_file": str(destination / checksum_path.name),
        "artifact_registry": str(destination / artifacts_path.name),
    }


__all__ = ["DEFAULT_OUTPUT_ROOT", "build_collection"]
