"""Traceable serialization for Au|Pd separation runs."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .electrostatics import LinearPBModel
from .parameters import (
    ELECTROSTATIC_BACKEND,
    GEOMETRY_NAME,
    MODEL_NAME,
    PACKAGE_VERSION,
    RESULT_SCHEMA_VERSION,
    apply_param_overrides,
    compute_derived_params,
)
from .scan import build_gouy_chapman_analysis
from .solver import top_surface_profiles


def make_run_tag(params: Mapping[str, Any]) -> str:
    """Return a stable physical-condition tag without embedding a timestamp."""

    canonical = apply_param_overrides(params)
    c_mM = float(canonical["C_tot"])
    alpha = int(round(100.0 * float(canonical["alpha1"])))
    au_nm = int(round(float(canonical["L_Au"]) * 1.0e9))
    pd_nm = int(round(float(canonical["L_Pd"]) * 1.0e9))
    return (
        f"flush_au{au_nm}_pd{pd_nm}_equal_i0_"
        f"alpha{alpha:03d}_ctot{c_mM:g}mM"
    )


def timestamped_run_directory(root: str | Path) -> Path:
    """Create no files; return a second-resolution result-directory path."""

    return Path(root) / datetime.now().strftime("%Y%m%d_%H%M%S")


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
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in fieldnames})


def _prepare_new_output_directory(path: str | Path) -> Path:
    """Return a new/empty output directory and reject every overwrite attempt."""

    destination = Path(path).resolve()
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty result directory: {destination}"
            )
    else:
        destination.mkdir(parents=True, exist_ok=False)
    return destination


def _git_metadata(package_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=package_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short", "--", str(package_root)],
            cwd=package_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {"commit": commit, "package_path_status": status}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"unavailable": str(exc)}


def environment_metadata(package_root: str | Path) -> dict[str, Any]:
    distributions = {}
    for name in ("numpy", "scipy", "matplotlib"):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
    root = Path(package_root).resolve()
    return {
        "source_package_version": PACKAGE_VERSION,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": distributions,
        "git": _git_metadata(root),
    }


def _case_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "electrostatic_backend": ELECTROSTATIC_BACKEND,
        "d_Au_Pd_nm": float(result["d_Au_Pd_nm"]),
        "condition": str(result["condition_label"]),
        "E_mix_V": float(result["E_mix_V"]),
        "I_Au_A": float(result["I_Au_A"]),
        "I_Pd_A": float(result["I_Pd_A"]),
        "i_mix_abs_A": float(result["i_mix_abs_A"]),
        "i_mix_avg_A_per_m2": float(result["i_mix_avg_A_per_m2"]),
        "relative_balance_residual": float(result["relative_balance_residual"]),
        "max_abs_phi_tilde": float(
            result["debye_huckel_validity"]["max_abs_phi_tilde"]
        ),
        "dh_threshold_exceeded": bool(
            result["debye_huckel_validity"]["threshold_exceeded"]
        ),
    }


def _validate_scan_result_contract(scan_result: Mapping[str, Any]) -> None:
    """Reject legacy/non-spectral scan payloads before assigning schema v2."""

    identity = (
        scan_result.get("result_schema_version"),
        scan_result.get("electrostatic_backend"),
    )
    expected = (RESULT_SCHEMA_VERSION, ELECTROSTATIC_BACKEND)
    if identity != expected:
        raise ValueError(
            "scan_result is not a result-schema-v2 semi-infinite cosine/Fourier "
            f"payload: got version/backend {identity!r}, expected {expected!r}"
        )

    rows = scan_result.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("scan_result rows must be a sequence")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"scan_result row {index} must be a mapping")
        row_identity = (
            row.get("result_schema_version"),
            row.get("electrostatic_backend"),
        )
        if row_identity != expected:
            raise ValueError(
                f"scan_result row {index} has incompatible version/backend "
                f"{row_identity!r}; expected {expected!r}"
            )

    spectral_results: list[tuple[str, Any]] = [
        ("no_overlap_100nm_reference", scan_result.get("no_overlap_100nm_reference"))
    ]
    cases = scan_result.get("cases", {})
    if not isinstance(cases, Mapping):
        raise ValueError("scan_result cases must be a mapping")
    spectral_results.extend(
        (f"cases[{key!r}]", case.get("result") if isinstance(case, Mapping) else None)
        for key, case in cases.items()
    )
    for label, result in spectral_results:
        electrostatics = result.get("electrostatics") if isinstance(result, Mapping) else None
        backend = (
            electrostatics.get("electrostatic_backend")
            if isinstance(electrostatics, Mapping)
            else None
        )
        if backend != ELECTROSTATIC_BACKEND:
            raise ValueError(
                f"{label} does not identify the required spectral backend: {backend!r}"
            )


def _separation_stem(d_nm: float) -> str:
    rounded = round(float(d_nm))
    if abs(float(d_nm) - rounded) <= 1.0e-9:
        return f"d{int(rounded):03d}nm"
    compact = f"{float(d_nm):.9f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"d{compact}nm"


def _save_case_fields(
    output_dir: Path,
    d_nm: float,
    result: Mapping[str, Any],
    model: LinearPBModel,
    *,
    display_nx: int,
    display_ny: int,
) -> dict[str, str]:
    stem = _separation_stem(d_nm)
    fields_dir = output_dir / "fields"
    csv_dir = output_dir / "csv"
    fields_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    E_mix = float(result["E_mix_V"])
    derived = result["derived"]
    params = result["params"]
    lambda_D = float(derived["lambda_D"])

    full_path = fields_dir / f"spectral_field_{stem}.npz"
    surface = model.top_profile(E_mix, n_x=int(params["Nx"]))
    saved_surface_n_x = int(np.asarray(surface["x_tilde"]).size)
    if int(np.asarray(surface["phi_tilde"]).size) != saved_surface_n_x:
        raise RuntimeError(
            f"Saved spectral surface coordinate/value length mismatch for {stem}"
        )
    np.savez_compressed(
        full_path,
        mode_index=model.mode_index,
        rho=model.rho,
        gamma=model.gamma,
        coefficients=model.coefficient_field(E_mix),
        affine_coefficients_m=model.phi_m,
        affine_coefficients_pzc=model.phi_pzc,
        surface_x_tilde=surface["x_tilde"],
        surface_phi_tilde=surface["phi_tilde"],
        lambda_D_m=np.asarray(lambda_D),
        E_mix_V=np.asarray(E_mix),
    )
    electrostatics_metadata = dict(result["electrostatics"])
    diagnostic_surface_n_x = int(model.surface_x_tilde.size)
    recorded_diagnostic_n_x = electrostatics_metadata.get(
        "diagnostic_surface_n_x_coordinates"
    )
    if recorded_diagnostic_n_x != diagnostic_surface_n_x:
        raise RuntimeError(
            "Result electrostatics metadata does not match the model's "
            f"diagnostic surface sampling for {stem}: "
            f"{recorded_diagnostic_n_x!r} != {diagnostic_surface_n_x}"
        )
    electrostatics_metadata.update(
        {
            "diagnostic_surface_n_x_coordinates": diagnostic_surface_n_x,
            "saved_surface_n_x_coordinates": saved_surface_n_x,
            # Retained for consumers of the v2 artifact metadata.  In this
            # file it refers only to the arrays saved in spectral_field_*.npz.
            "n_x_coordinates": saved_surface_n_x,
            "n_x_coordinates_role": (
                "saved spectral_field surface_x_tilde coordinate count"
            ),
        }
    )
    metadata_path = fields_dir / f"spectral_metadata_{stem}.json"
    _write_json(
        metadata_path,
        {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "electrostatic_backend": ELECTROSTATIC_BACKEND,
            "case": _case_summary(result),
            "root": result["root"],
            "electrostatics": electrostatics_metadata,
            "affine_residuals": result["affine_residuals"],
            "derived": result["derived"],
            "effective_reaction": result["effective_reaction"],
            "electronic_coupling": result["electronic_coupling"],
            "pb_model": result["pb_model"],
        },
    )

    grid = model.upper_grid(
        E_mix,
        n_x=int(display_nx),
        n_y=int(display_ny),
        y_max_over_lambda=5.0,
    )
    phi = np.asarray(grid["phi_tilde"], dtype=float)
    c_red = np.exp(
        np.clip(-float(params["z_R1"]) * phi, -700.0, 700.0)
    )
    c_ox = np.exp(
        np.clip(-float(params["z_O2"]) * phi, -700.0, 700.0)
    )
    top_check = model.top_profile(E_mix, n_x=int(display_nx))
    if not np.allclose(phi[0], top_check["phi_tilde"], rtol=0.0, atol=1.0e-10):
        raise RuntimeError(
            f"Display-grid y=0 field does not match the top profile for {stem}"
        )
    if not (
        np.all(np.isfinite(phi))
        and np.all(np.isfinite(c_red))
        and np.all(np.isfinite(c_ox))
        and np.all(c_red > 0.0)
        and np.all(c_ox > 0.0)
    ):
        raise RuntimeError(f"Non-finite or non-positive display field for {stem}")
    x_nm = np.asarray(grid["x_tilde"]) * lambda_D * 1.0e9
    y_nm = np.asarray(grid["y_tilde"]) * lambda_D * 1.0e9
    display_path = fields_dir / f"display_grid_{stem}.npz"
    np.savez_compressed(
        display_path,
        x_nm=x_nm,
        y_nm=y_nm,
        phi_tilde=phi,
        phi_s_V=phi * float(derived["thermal_voltage_V"]),
        c_Red1_over_c_bulk=c_red,
        c_Ox2_over_c_bulk=c_ox,
    )

    display_csv = csv_dir / f"display_grid_{stem}.csv"
    with display_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "x_nm",
                "y_nm",
                "phi_tilde",
                "phi_s_V",
                "c_Red1_over_c_bulk",
                "c_Ox2_over_c_bulk",
            ]
        )
        thermal = float(derived["thermal_voltage_V"])
        for iy, y_value in enumerate(y_nm):
            for ix, x_value in enumerate(x_nm):
                writer.writerow(
                    [
                        float(x_value),
                        float(y_value),
                        float(phi[iy, ix]),
                        float(phi[iy, ix] * thermal),
                        float(c_red[iy, ix]),
                        float(c_ox[iy, ix]),
                    ]
                )

    profiles = top_surface_profiles(result, model, n_x=max(2001, display_nx))
    profile_csv = csv_dir / f"top_surface_profiles_{stem}.csv"
    profile_rows = [
        {key: float(np.asarray(values)[index]) for key, values in profiles.items()}
        for index in range(len(profiles["x_nm"]))
    ]
    _write_rows(profile_csv, profile_rows)
    return {
        "full_spectral_npz": str(full_path),
        "spectral_metadata_json": str(metadata_path),
        "display_grid_npz": str(display_path),
        "display_grid_csv": str(display_csv),
        "top_surface_profiles_csv": str(profile_csv),
    }


def save_gouy_chapman_outputs(
    output_dir: str | Path,
    scan_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Write GC scan rows and length-scale analysis without rewriting scan data."""

    destination = Path(output_dir).resolve()
    csv_path = destination / "csv" / "gouy_chapman_length_vs_separation.csv"
    analysis_path = destination / "gouy_chapman_analysis.json"
    gc_rows = list(scan_result.get("gouy_chapman_rows", []))
    if not gc_rows:
        raise ValueError("scan_result gouy_chapman_rows must not be empty")
    analysis = build_gouy_chapman_analysis(scan_result)
    _write_rows(csv_path, gc_rows)
    _write_json(analysis_path, analysis)
    return {
        "csv": str(csv_path),
        "analysis_json": str(analysis_path),
        "n_rows": len(gc_rows),
        "analysis": analysis,
    }


def save_run(
    output_dir: str | Path,
    params: Mapping[str, Any],
    scan_result: Mapping[str, Any],
    *,
    source_baseline: str | Path | None = None,
    package_root: str | Path | None = None,
    display_nx: int = 600,
    display_ny: int = 320,
) -> dict[str, Any]:
    """Save a complete traceable run into a new or empty directory.

    This public API owns the no-overwrite guarantee.  A non-directory path or
    a directory containing any entry is rejected before parameters are
    validated or files are written.
    """

    destination = _prepare_new_output_directory(output_dir)
    _validate_scan_result_contract(scan_result)
    canonical = apply_param_overrides(params)
    root = (
        Path(package_root).resolve()
        if package_root is not None
        else Path(__file__).resolve().parents[2]
    )

    _write_json(destination / "params.json", canonical)
    _write_json(destination / "derived.json", compute_derived_params(canonical))
    _write_json(destination / "environment.json", environment_metadata(root))
    if source_baseline is not None:
        source = Path(source_baseline).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        inputs_dir = destination / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, inputs_dir / f"source_baseline_{source.name}")

    rows = list(scan_result["rows"])
    _write_rows(destination / "csv" / "separation_scan.csv", rows)
    gouy_chapman_outputs = save_gouy_chapman_outputs(destination, scan_result)
    summary_rows = [
        _case_summary(scan_result["without_edl_reference"]),
        _case_summary(scan_result["no_overlap_100nm_reference"]),
    ]
    for d_nm, case in sorted(scan_result.get("cases", {}).items()):
        summary_rows.append(_case_summary(case["result"]))
    _write_rows(destination / "csv" / "summary_cases.csv", summary_rows)

    summary_json = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "electrostatic_backend": ELECTROSTATIC_BACKEND,
        "model": MODEL_NAME,
        "geometry": GEOMETRY_NAME,
        "tag": make_run_tag(canonical),
        "plateau_check": scan_result["plateau_check"],
        "without_edl_reference": _case_summary(
            scan_result["without_edl_reference"]
        ),
        "no_overlap_100nm_reference": _case_summary(
            scan_result["no_overlap_100nm_reference"]
        ),
        "retained_cases": {
            str(d_nm): _case_summary(case["result"])
            for d_nm, case in sorted(scan_result.get("cases", {}).items())
        },
        "debye_huckel_caveat": (
            "Runs with max_abs_phi_tilde > 1 are linearized-PB internal geometry "
            "sensitivity results and are not claimed as absolute quantitative predictions."
        ),
        "gouy_chapman_analysis": {
            "csv": gouy_chapman_outputs["csv"],
            "analysis_json": gouy_chapman_outputs["analysis_json"],
            "n_rows": gouy_chapman_outputs["n_rows"],
            "interpretation": gouy_chapman_outputs["analysis"]["interpretation"],
        },
    }
    _write_json(destination / "summary.json", summary_json)

    case_artifacts: dict[str, Any] = {}
    for d_nm, case in sorted(scan_result.get("cases", {}).items()):
        model = case.get("model")
        if model is None:
            continue
        case_artifacts[str(d_nm)] = _save_case_fields(
            destination,
            float(d_nm),
            case["result"],
            model,
            display_nx=display_nx,
            display_ny=display_ny,
        )
    artifacts = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "electrostatic_backend": ELECTROSTATIC_BACKEND,
        "cases": case_artifacts,
    }
    _write_json(destination / "artifacts.json", artifacts)
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "electrostatic_backend": ELECTROSTATIC_BACKEND,
        "output_dir": str(destination),
        "tag": make_run_tag(canonical),
        "artifacts": artifacts,
        "summary": summary_json,
        "gouy_chapman": gouy_chapman_outputs,
    }
