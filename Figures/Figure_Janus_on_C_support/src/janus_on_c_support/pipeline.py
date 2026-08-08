"""End-to-end result builder for the C-supported Janus mirror half-cell."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.polynomial.legendre import leggauss

from .electrostatics import LinearPBModel, Segment
from .figures import generate_all_figures
from .io import (
    CHECKSUM_FILENAME,
    artifact_hashes,
    environment_snapshot,
    make_run_directory,
    relative_paths,
    source_hashes,
    write_checksums,
    write_csv,
    write_json,
    write_npz,
)
from .kinetics import (
    local_current_density_A_per_m2,
    local_overpotential_V,
)
from .parameters import (
    ELECTROSTATIC_BACKEND,
    GEOMETRY_NAME,
    MODEL_NAME,
    PACKAGE_VERSION,
    RESULT_SCHEMA_VERSION,
    TOPOLOGY_ID,
    apply_param_overrides,
    compute_derived_params,
    default_params,
    validate_params,
)
from .solver import solve_case


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "results"
DISPLAY_NX = 1601
DISPLAY_CSV_MAX_NX = 401
DISPLAY_CSV_MAX_NY = 161
FIGURE_DPI = 600
BALANCE_TOLERANCE = 1.0e-10
Y0_ABSOLUTE_TOLERANCE = 5.0e-11
CHARGE_ABSOLUTE_TOLERANCE_C_PER_M2 = 5.0e-12
ROBIN_INTERIOR_RMS_RELATIVE_TOLERANCE = 3.0e-2
ROBIN_ZERO_CHARGE_RMS_THRESHOLD_C_PER_M2 = 1.0e-12
ROBIN_ZERO_CHARGE_ABSOLUTE_RMS_TOLERANCE_C_PER_M2 = 5.0e-4
NEUMANN_ABSOLUTE_TOLERANCE = 1.0e-10
CONVERGENCE_MIN_MODES = 960
CONVERGENCE_MIN_GL_ORDER = 128
CONVERGENCE_MAX_MODES = 3840
CONVERGENCE_MAX_GL_ORDER = 512
CONVERGENCE_E_TOLERANCE_V = 1.0e-4
CONVERGENCE_I_RELATIVE_TOLERANCE = 1.0e-3


def _case_value(case: Mapping[str, Any], *names: str) -> float:
    for name in names:
        if name in case:
            value = float(case[name])
            if math.isfinite(value):
                return value
    raise KeyError(f"None of the result fields is available: {', '.join(names)}")


def _convergence_delta(
    lower: Mapping[str, Any],
    upper: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two with-EDL mixed-potential solutions."""

    E_lower = _case_value(lower, "E_mix_V", "E_mix")
    E_upper = _case_value(upper, "E_mix_V", "E_mix")
    i_lower = _case_value(lower, "i_mix_avg_A_per_m2")
    i_upper = _case_value(upper, "i_mix_avg_A_per_m2")
    delta_E = abs(E_upper - E_lower)
    delta_i_relative = abs(i_upper - i_lower) / max(
        abs(i_upper), np.finfo(float).tiny
    )
    passed = (
        delta_E < CONVERGENCE_E_TOLERANCE_V
        and delta_i_relative < CONVERGENCE_I_RELATIVE_TOLERANCE
    )
    return {
        "E_mix_lower_V": E_lower,
        "E_mix_upper_V": E_upper,
        "i_mix_lower_A_per_m2": i_lower,
        "i_mix_upper_A_per_m2": i_upper,
        "delta_E_mix_V": delta_E,
        "delta_E_mix_mV": 1.0e3 * delta_E,
        "delta_i_mix_relative": delta_i_relative,
        "delta_i_mix_percent": 100.0 * delta_i_relative,
        "E_tolerance_V_strict": CONVERGENCE_E_TOLERANCE_V,
        "i_relative_tolerance_strict": CONVERGENCE_I_RELATIVE_TOLERANCE,
        "passed": passed,
    }


def _resolve_converged_solution(
    params: Mapping[str, Any],
    *,
    perform: bool | None,
) -> tuple[dict[str, Any], dict[str, Any], LinearPBModel, dict[str, Any]]:
    """Select a converged spectral/quadrature resolution and solve it.

    The canonical production resolution is checked as ``480 -> 960`` modes
    and ``GL64 -> GL128``.  A failed dimension is doubled independently up to
    the documented hard limits.  Deliberately small test/debug overrides are
    skipped in ``auto`` mode; callers can force the production study with
    ``perform=True``.
    """

    requested = dict(params)
    requested_modes = int(requested["N_modes"])
    requested_gl = int(requested["gl_order"])
    enabled = (
        requested_modes >= CONVERGENCE_MIN_MODES
        and requested_gl >= CONVERGENCE_MIN_GL_ORDER
        if perform is None
        else bool(perform)
    )
    if not enabled:
        model = LinearPBModel(requested)
        solution = solve_case(use_edl=True, model=model)
        report = {
            "performed": False,
            "mode": "auto" if perform is None else "disabled",
            "reason": (
                "Numerical overrides are below the canonical production "
                "resolution; the expensive production convergence study was skipped."
            ),
            "requested_N_modes": requested_modes,
            "requested_gl_order": requested_gl,
            "selected_N_modes": requested_modes,
            "selected_gl_order": requested_gl,
            "attempts": [],
            "passed": True,
        }
        return requested, report, model, solution

    target_modes = max(requested_modes, CONVERGENCE_MIN_MODES)
    target_gl = max(requested_gl, CONVERGENCE_MIN_GL_ORDER)
    if target_modes > CONVERGENCE_MAX_MODES or target_gl > CONVERGENCE_MAX_GL_ORDER:
        raise ValueError(
            "Convergence-controlled publication requires N_modes <= 3840 "
            "and gl_order <= 512"
        )

    cache: dict[tuple[int, int], tuple[LinearPBModel, dict[str, Any]]] = {}

    def solve_resolution(modes: int, gl_order: int) -> tuple[LinearPBModel, dict[str, Any]]:
        key = (int(modes), int(gl_order))
        if key not in cache:
            candidate = apply_param_overrides(
                requested,
                {"N_modes": key[0], "gl_order": key[1]},
                reset_lambda_D=False,
            )
            model = LinearPBModel(candidate)
            cache[key] = (model, solve_case(use_edl=True, model=model))
        return cache[key]

    attempts: list[dict[str, Any]] = []
    while True:
        lower_modes = max(1, target_modes // 2)
        lower_gl = max(2, target_gl // 2)
        _, modes_lower = solve_resolution(lower_modes, target_gl)
        selected_model, selected_solution = solve_resolution(target_modes, target_gl)
        _, gl_lower = solve_resolution(target_modes, lower_gl)
        modes_check = _convergence_delta(modes_lower, selected_solution)
        gl_check = _convergence_delta(gl_lower, selected_solution)
        attempt_passed = bool(modes_check["passed"] and gl_check["passed"])
        attempts.append(
            {
                "attempt": len(attempts) + 1,
                "mode_comparison": {
                    "lower_N_modes": lower_modes,
                    "upper_N_modes": target_modes,
                    "fixed_gl_order": target_gl,
                    **modes_check,
                },
                "quadrature_comparison": {
                    "fixed_N_modes": target_modes,
                    "lower_gl_order": lower_gl,
                    "upper_gl_order": target_gl,
                    **gl_check,
                },
                "passed": attempt_passed,
            }
        )
        if attempt_passed:
            selected = apply_param_overrides(
                requested,
                {"N_modes": target_modes, "gl_order": target_gl},
                reset_lambda_D=False,
            )
            report = {
                "performed": True,
                "mode": "forced" if perform is True else "auto",
                "requested_N_modes": requested_modes,
                "requested_gl_order": requested_gl,
                "selected_N_modes": target_modes,
                "selected_gl_order": target_gl,
                "escalated": (
                    target_modes != requested_modes or target_gl != requested_gl
                ),
                "maximum_N_modes": CONVERGENCE_MAX_MODES,
                "maximum_gl_order": CONVERGENCE_MAX_GL_ORDER,
                "attempts": attempts,
                "passed": True,
            }
            return selected, report, selected_model, selected_solution

        mode_failed = not bool(modes_check["passed"])
        gl_failed = not bool(gl_check["passed"])
        if mode_failed:
            if target_modes >= CONVERGENCE_MAX_MODES:
                break
            target_modes = min(2 * target_modes, CONVERGENCE_MAX_MODES)
        if gl_failed:
            if target_gl >= CONVERGENCE_MAX_GL_ORDER:
                break
            target_gl = min(2 * target_gl, CONVERGENCE_MAX_GL_ORDER)

    failed_report = {
        "performed": True,
        "mode": "forced" if perform is True else "auto",
        "requested_N_modes": requested_modes,
        "requested_gl_order": requested_gl,
        "selected_N_modes": target_modes,
        "selected_gl_order": target_gl,
        "maximum_N_modes": CONVERGENCE_MAX_MODES,
        "maximum_gl_order": CONVERGENCE_MAX_GL_ORDER,
        "attempts": attempts,
        "passed": False,
    }
    raise RuntimeError(
        "N_modes/GL convergence failed at the allowed maxima; result will not "
        f"be published: {failed_report}"
    )


def _boundary_aware_grid(boundaries_m: Iterable[float], point_count: int) -> np.ndarray:
    """Return exactly ``point_count`` ordered points containing every edge."""

    boundaries = np.unique(np.asarray(tuple(boundaries_m), dtype=float))
    if boundaries.ndim != 1 or boundaries.size < 2:
        raise ValueError("At least two finite boundaries are required")
    if not np.all(np.isfinite(boundaries)) or np.any(np.diff(boundaries) <= 0.0):
        raise ValueError("Boundaries must be finite and strictly increasing")
    n = int(point_count)
    interval_count = boundaries.size - 1
    if n < interval_count + 1:
        raise ValueError(
            f"point_count={n} cannot contain all {boundaries.size} boundaries"
        )

    # Give every geometrical interval at least one grid interval, then
    # distribute the remainder by physical length with a largest-remainder
    # allocation.  Concatenation drops repeated internal endpoints, retaining
    # an exact total of n points.
    lengths = np.diff(boundaries)
    allocations = np.ones(interval_count, dtype=int)
    remaining = (n - 1) - interval_count
    if remaining:
        ideal = remaining * lengths / float(np.sum(lengths))
        whole = np.floor(ideal).astype(int)
        allocations += whole
        leftover = remaining - int(np.sum(whole))
        if leftover:
            order = np.argsort(-(ideal - whole), kind="stable")
            allocations[order[:leftover]] += 1

    pieces = [np.array([boundaries[0]], dtype=float)]
    for left, right, intervals in zip(
        boundaries[:-1], boundaries[1:], allocations, strict=True
    ):
        pieces.append(np.linspace(left, right, int(intervals) + 1)[1:])
    grid = np.concatenate(pieces)
    if grid.size != n or not np.all(np.diff(grid) > 0.0):
        raise RuntimeError("Boundary-aware grid construction failed")
    return grid


def _segment_arrays(
    x_m: np.ndarray,
    segments: tuple[Segment, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign each surface point to one half-open segment."""

    indices = np.full(np.asarray(x_m).shape, -1, dtype=int)
    positive = [index for index, segment in enumerate(segments) if segment.length_m > 0.0]
    for position, index in enumerate(positive):
        segment = segments[index]
        if position == len(positive) - 1:
            mask = (x_m >= segment.x0_m) & (x_m <= segment.x1_m)
        else:
            mask = (x_m >= segment.x0_m) & (x_m < segment.x1_m)
        indices[mask] = index
    if np.any(indices < 0):
        raise RuntimeError("Surface grid contains points not owned by a positive segment")
    segment_id = np.asarray([segments[index].name for index in indices], dtype="U16")
    material = np.asarray([segments[index].material for index in indices], dtype="U4")
    return indices, segment_id, material


def _exp_clipped(exponent: np.ndarray | float) -> np.ndarray:
    return np.exp(np.clip(np.asarray(exponent, dtype=float), -700.0, 700.0))


def _kinetic_vector(
    E_V: float,
    phi_tilde: np.ndarray,
    material: str,
    params: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    eta = np.asarray(
        local_overpotential_V(E_V, phi_tilde, material, params),
        dtype=float,
    )
    current = np.asarray(
        local_current_density_A_per_m2(E_V, phi_tilde, material, params),
        dtype=float,
    )
    if eta.ndim == 0:
        eta = np.full(np.asarray(phi_tilde).shape, float(eta))
    if current.ndim == 0:
        current = np.full(np.asarray(phi_tilde).shape, float(current))
    return eta, current


def _build_surface_data(
    model: LinearPBModel,
    params: Mapping[str, Any],
    with_edl: Mapping[str, Any],
    without_edl: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    edges = [model.segments[0].x0_m, *[segment.x1_m for segment in model.segments]]
    x_m = _boundary_aware_grid(edges, int(params["Nx"]))
    indices, segment_id, material = _segment_arrays(x_m, model.segments)
    E_with = _case_value(with_edl, "E_mix_V", "E_mix")
    E_no = _case_value(without_edl, "E_mix_V", "E_mix")
    phi_with = np.asarray(model.phi_tilde_surface(E_with, x_m), dtype=float)
    phi_no = np.zeros_like(phi_with)
    thermal_voltage = float(model.derived["thermal_voltage_V"])

    data: dict[str, np.ndarray] = {
        "x_m": x_m,
        "x_nm": x_m * 1.0e9,
        "segment_index": indices,
        "segment_id": segment_id,
        "material": material,
        "phi_tilde_with": phi_with,
        "phi_tilde_no": phi_no,
        "phi_rp_with_V": phi_with * thermal_voltage,
        "phi_rp_no_V": phi_no,
        "c_R1_with": _exp_clipped(-float(params["z_R1"]) * phi_with),
        "c_O2_with": _exp_clipped(-float(params["z_O2"]) * phi_with),
        "c_R1_no": np.ones_like(phi_with),
        "c_O2_no": np.ones_like(phi_with),
    }

    for suffix in ("with", "no"):
        E = E_with if suffix == "with" else E_no
        phi = phi_with if suffix == "with" else phi_no
        for reaction_material, lower in (("Au", "au"), ("Pd", "pd")):
            eta = np.full(x_m.shape, np.nan, dtype=float)
            current = np.full(x_m.shape, np.nan, dtype=float)
            mask = material == reaction_material
            eta_values, current_values = _kinetic_vector(
                E,
                phi[mask],
                reaction_material,
                params,
            )
            eta[mask] = eta_values
            current[mask] = current_values
            data[f"eta_{lower}_{suffix}_V"] = eta
            data[f"j_{lower}_{suffix}"] = current
    return data


def _quadrature_on_segment(
    model: LinearPBModel,
    params: Mapping[str, Any],
    segment: Segment,
    E_V: float,
    *,
    use_edl: bool,
) -> dict[str, np.ndarray]:
    nodes, weights = leggauss(int(params["gl_order"]))
    x_m = 0.5 * (segment.x1_m - segment.x0_m) * nodes + 0.5 * (
        segment.x1_m + segment.x0_m
    )
    weights_m = 0.5 * (segment.x1_m - segment.x0_m) * weights
    phi_tilde = (
        np.asarray(model.phi_tilde_surface(E_V, x_m), dtype=float)
        if use_edl
        else np.zeros_like(x_m)
    )
    eta, current = _kinetic_vector(E_V, phi_tilde, segment.material, params)
    return {
        "x_m": x_m,
        "weights_m": weights_m,
        "phi_tilde": phi_tilde,
        "eta_V": eta,
        "current_density_A_per_m2": current,
        "integrated_current_A": np.asarray(
            float(params["out_of_plane_width"]) * np.dot(weights_m, current)
        ),
    }


def _build_quadrature_data(
    model: LinearPBModel,
    params: Mapping[str, Any],
    with_edl: Mapping[str, Any],
    without_edl: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    active = {segment.material: segment for segment in model.segments if segment.material in {"Au", "Pd"}}
    result: dict[str, dict[str, Any]] = {}
    for condition, case, use_edl in (
        ("with_edl", with_edl, True),
        ("without_edl", without_edl, False),
    ):
        E_V = _case_value(case, "E_mix_V", "E_mix")
        au = _quadrature_on_segment(model, params, active["Au"], E_V, use_edl=use_edl)
        pd = _quadrature_on_segment(model, params, active["Pd"], E_V, use_edl=use_edl)
        I_au = float(au["integrated_current_A"])
        I_pd = float(pd["integrated_current_A"])
        denominator = abs(I_au) + abs(I_pd)
        result[condition] = {
            "Au": au,
            "Pd": pd,
            "I_Au_A": I_au,
            "I_Pd_A": I_pd,
            "residual_A": I_au + I_pd,
            "relative_balance_residual": (
                abs(I_au + I_pd) / denominator if denominator else 0.0
            ),
            "i_mix_abs_halfcell_A": 0.5 * (abs(I_au) + abs(I_pd)),
            "i_mix_abs_full_period_A": abs(I_au) + abs(I_pd),
            "i_mix_avg_A_per_m2": (
                0.5
                * (abs(I_au) + abs(I_pd))
                / float(model.derived["reactive_area_halfcell_m2"])
            ),
        }
    return result


def _build_surface_charge_segments(
    model: LinearPBModel,
    params: Mapping[str, Any],
    with_edl: Mapping[str, Any],
    surface: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    E_with = _case_value(with_edl, "E_mix_V", "E_mix")
    x_m = np.asarray(surface["x_m"])
    segment_ids = np.asarray(surface["segment_id"])
    output: list[dict[str, Any]] = []
    for segment in model.segments:
        mask = segment_ids == segment.name
        segment_x = x_m[mask]
        if not segment_x.size:
            continue
        phi_tilde = np.asarray(model.phi_tilde_surface(E_with, segment_x), dtype=float)
        phi_V = phi_tilde * float(model.derived["thermal_voltage_V"])
        capacitance = float(segment.C_H_F_per_m2)
        pzc = float(params[f"pzc_{segment.material}"])
        sigma = capacitance * (E_with - pzc - phi_V)
        output.append(
            {
                "segment_id": segment.name,
                "material": segment.material,
                "x_m": segment_x,
                "x_nm": segment_x * 1.0e9,
                "phi_rp_V": phi_V,
                "sigma_C_per_m2": sigma,
            }
        )
    return output


def _build_grid_2d(
    model: LinearPBModel,
    params: Mapping[str, Any],
    with_edl: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    edges = [model.segments[0].x0_m, *[segment.x1_m for segment in model.segments]]
    x_m = _boundary_aware_grid(edges, DISPLAY_NX)
    lambda_D = float(model.derived["lambda_D_m"])
    y_m = np.linspace(
        0.0,
        float(params["y_max_lambda_D"]) * lambda_D,
        int(params["Ny_2d"]),
    )
    E_with = _case_value(with_edl, "E_mix_V", "E_mix")
    phi_tilde = np.asarray(model.phi_tilde(E_with, x_m, y_m), dtype=float)
    expected_shape = (y_m.size, x_m.size)
    if phi_tilde.shape == expected_shape[::-1]:
        phi_tilde = phi_tilde.T
    if phi_tilde.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected 2D potential shape {phi_tilde.shape}; expected {expected_shape}"
        )
    return {
        "x_m": x_m,
        "x_nm": x_m * 1.0e9,
        "y_m": y_m,
        "y_nm": y_m * 1.0e9,
        "phi_tilde": phi_tilde,
        "phi_s_mV": phi_tilde * float(model.derived["thermal_voltage_V"]) * 1.0e3,
        "c_R1_norm": _exp_clipped(-float(params["z_R1"]) * phi_tilde),
        "c_O2_norm": _exp_clipped(-float(params["z_O2"]) * phi_tilde),
    }


def _geometry_validation(model: LinearPBModel, params: Mapping[str, Any]) -> dict[str, Any]:
    segments = model.segments
    names = [segment.name for segment in segments]
    materials = [segment.material for segment in segments]
    boundaries = [segments[0].x0_m, *[segment.x1_m for segment in segments]]
    chain_ok = all(
        math.isclose(left.x1_m, right.x0_m, rel_tol=0.0, abs_tol=1.0e-18)
        for left, right in zip(segments[:-1], segments[1:], strict=True)
    )
    valid_lengths = all(
        segment.length_m > 0.0 if segment.material in {"Au", "Pd"} else segment.length_m >= 0.0
        for segment in segments
    )
    expected_default = np.asarray([0.0, 5.0e-9, 9.0e-9, 13.0e-9, 18.0e-9])
    default_lengths = default_params()
    is_default_geometry = all(
        math.isclose(float(params[name]), float(default_lengths[name]), rel_tol=0.0, abs_tol=1.0e-18)
        for name in ("L_C_left", "L_Au", "L_Pd", "L_C_right")
    )
    exact_default_edges = bool(
        np.allclose(boundaries, expected_default, rtol=0.0, atol=1.0e-18)
    )
    passed = (
        names == ["C_left", "Au", "Pd", "C_right"]
        and materials == ["C", "Au", "Pd", "C"]
        and chain_ok
        and valid_lengths
        and (not is_default_geometry or exact_default_edges)
    )
    return {
        "segment_names": names,
        "materials": materials,
        "boundaries_nm": [float(value * 1.0e9) for value in boundaries],
        "covers_without_gaps_or_overlap": chain_ok,
        "reactive_segments_positive_and_C_segments_nonnegative": valid_lengths,
        "default_geometry_requested": is_default_geometry,
        "default_edges_0_5_9_13_18_nm": exact_default_edges,
        "mirror_material_sequence": [
            "C", "Au", "Pd", "C",
            "C", "Pd", "Au", "C",
            "C", "Au", "Pd", "C",
        ],
        "full_material_period_nm": float(2.0e9 * boundaries[-1]),
        "passed": passed,
    }


def _numerical_validation(
    model: LinearPBModel,
    params: Mapping[str, Any],
    with_edl: Mapping[str, Any],
    without_edl: Mapping[str, Any],
    surface: Mapping[str, np.ndarray],
    quadrature: Mapping[str, Mapping[str, Any]],
    charge_segments: list[dict[str, Any]],
    grid_2d: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    current_checks: dict[str, Any] = {}
    currents_passed = True
    for name, case in (("with_edl", with_edl), ("without_edl", without_edl)):
        q = quadrature[name]
        reported_residual = float(case.get("relative_balance_residual", float("inf")))
        recomputed_residual = float(q["relative_balance_residual"])
        reported_I_au = _case_value(case, "I_Au_A")
        reported_I_pd = _case_value(case, "I_Pd_A")
        scale = max(abs(reported_I_au), abs(reported_I_pd), np.finfo(float).tiny)
        quadrature_match = max(
            abs(float(q["I_Au_A"]) - reported_I_au),
            abs(float(q["I_Pd_A"]) - reported_I_pd),
        ) / scale
        passed = (
            reported_residual <= BALANCE_TOLERANCE
            and recomputed_residual <= BALANCE_TOLERANCE
            and quadrature_match <= 1.0e-8
        )
        currents_passed = currents_passed and passed
        current_checks[name] = {
            "reported_relative_balance_residual": reported_residual,
            "recomputed_relative_balance_residual": recomputed_residual,
            "quadrature_vs_solver_relative_difference": quadrature_match,
            "tolerance": BALANCE_TOLERANCE,
            "passed": passed,
        }

    x_2d = np.asarray(grid_2d["x_m"], dtype=float)
    phi_surface = np.asarray(
        model.phi_tilde_surface(_case_value(with_edl, "E_mix_V", "E_mix"), x_2d),
        dtype=float,
    )
    y0_difference = float(
        np.max(np.abs(np.asarray(grid_2d["phi_tilde"])[0, :] - phi_surface))
    )

    charge_difference = 0.0
    for item in charge_segments:
        material = str(item["material"])
        expected = float(
            model.derived[f"C_H_{material}_effective_F_per_m2"]
        ) * (
            _case_value(with_edl, "E_mix_V", "E_mix")
            - float(params[f"pzc_{material}"])
            - np.asarray(item["phi_rp_V"], dtype=float)
        )
        charge_difference = max(
            charge_difference,
            float(np.max(np.abs(np.asarray(item["sigma_C_per_m2"]) - expected))),
        )

    # Independently compare the Stern charge with the diffuse-layer normal
    # displacement.  The pointwise residual of a truncated cosine series is
    # sampled away from discontinuous material junctions; Gibbs ringing at the
    # junction itself is tracked separately as a documented caveat.
    E_with = _case_value(with_edl, "E_mix_V", "E_mix")
    diffuse_prefactor = (
        float(model.derived["epsilon_s_F_per_m"])
        * float(model.derived["thermal_voltage_V"])
        / float(model.derived["lambda_D_m"])
    )
    robin_segments: dict[str, Any] = {}
    robin_max_rms_relative = 0.0
    robin_segment_checks: list[bool] = []
    for segment in model.segments:
        if segment.length_m <= 0.0:
            continue
        margin = 0.02 * segment.length_m
        x_interior = np.linspace(
            segment.x0_m + margin,
            segment.x1_m - margin,
            257,
        )
        sigma_stern = np.asarray(
            model.surface_charge_C_per_m2(
                E_with,
                x_interior,
                material=segment.material,
            ),
            dtype=float,
        )
        sigma_diffuse = -diffuse_prefactor * np.asarray(
            model.derivative_y_tilde_surface(E_with, x_interior),
            dtype=float,
        )
        difference = sigma_stern - sigma_diffuse
        rms_difference = float(np.sqrt(np.mean(difference**2)))
        rms_stern = float(np.sqrt(np.mean(sigma_stern**2)))
        near_zero_charge = rms_stern <= ROBIN_ZERO_CHARGE_RMS_THRESHOLD_C_PER_M2
        rms_relative = None if near_zero_charge else rms_difference / rms_stern
        if rms_relative is not None:
            robin_max_rms_relative = max(robin_max_rms_relative, rms_relative)
        segment_passed = (
            rms_difference < ROBIN_ZERO_CHARGE_ABSOLUTE_RMS_TOLERANCE_C_PER_M2
            if near_zero_charge
            else bool(rms_relative < ROBIN_INTERIOR_RMS_RELATIVE_TOLERANCE)
        )
        robin_segment_checks.append(segment_passed)
        robin_segments[segment.name] = {
            "material": segment.material,
            "interior_trim_fraction_each_side": 0.02,
            "sample_count": int(x_interior.size),
            "max_abs_sigma_difference_C_per_m2": float(
                np.max(np.abs(difference))
            ),
            "rms_sigma_difference_C_per_m2": rms_difference,
            "rms_relative_to_stern_charge": rms_relative,
            "near_zero_stern_charge": near_zero_charge,
            "criterion": (
                "absolute_rms" if near_zero_charge else "relative_rms"
            ),
            "passed": segment_passed,
        }
    robin_required = int(params["N_modes"]) >= CONVERGENCE_MIN_MODES
    robin_passed = all(robin_segment_checks)

    x_boundaries = np.asarray(
        [model.segments[0].x0_m, model.segments[-1].x1_m], dtype=float
    )
    neumann_max = float(
        np.max(np.abs(model.derivative_x_tilde(E_with, x_boundaries)))
    )
    neumann_passed = neumann_max <= NEUMANN_ABSOLUTE_TOLERANCE

    c_mask = np.asarray(surface["material"]) == "C"
    c_kinetics_nan = all(
        bool(np.all(np.isnan(np.asarray(surface[key])[c_mask])))
        for key in (
            "eta_au_with_V", "eta_pd_with_V", "eta_au_no_V", "eta_pd_no_V",
            "j_au_with", "j_pd_with", "j_au_no", "j_pd_no",
        )
    )
    y0_passed = y0_difference <= Y0_ABSOLUTE_TOLERANCE
    charge_passed = charge_difference <= CHARGE_ABSOLUTE_TOLERANCE_C_PER_M2
    passed = (
        currents_passed
        and y0_passed
        and charge_passed
        and c_kinetics_nan
        and neumann_passed
        and (robin_passed or not robin_required)
    )
    return {
        "current_balance": current_checks,
        "y0_surface_back_substitution": {
            "max_abs_phi_tilde_difference": y0_difference,
            "absolute_tolerance": Y0_ABSOLUTE_TOLERANCE,
            "passed": y0_passed,
        },
        "pointwise_stern_charge_relation": {
            "max_abs_difference_C_per_m2": charge_difference,
            "absolute_tolerance_C_per_m2": CHARGE_ABSOLUTE_TOLERANCE_C_PER_M2,
            "passed": charge_passed,
        },
        "stern_charge_vs_diffuse_normal_displacement": {
            "relation": (
                "sigma_Stern = -epsilon_s*(R*T/F)/lambda_D*"
                "d(phi_tilde)/d(y_tilde)"
            ),
            "segments": robin_segments,
            "maximum_segment_rms_relative_difference": robin_max_rms_relative,
            "rms_relative_tolerance_strict": ROBIN_INTERIOR_RMS_RELATIVE_TOLERANCE,
            "near_zero_charge_threshold_C_per_m2": (
                ROBIN_ZERO_CHARGE_RMS_THRESHOLD_C_PER_M2
            ),
            "near_zero_charge_absolute_rms_tolerance_C_per_m2": (
                ROBIN_ZERO_CHARGE_ABSOLUTE_RMS_TOLERANCE_C_PER_M2
            ),
            "required_for_this_resolution": robin_required,
            "passed": robin_passed,
            "caveat": (
                "The pointwise check excludes 2% of each segment at both ends "
                "because discontinuous Robin data produce truncated-series Gibbs ringing."
            ),
        },
        "lateral_neumann_boundaries": {
            "max_abs_dphi_tilde_dx_tilde": neumann_max,
            "absolute_tolerance": NEUMANN_ABSOLUTE_TOLERANCE,
            "passed": neumann_passed,
        },
        "C_regions_have_no_faradaic_kinetics": c_kinetics_nan,
        "debye_huckel": {
            "max_abs_phi_tilde": _case_value(with_edl, "max_abs_phi_tilde"),
            "warning_threshold": float(params["dh_warn_threshold"]),
            "threshold_exceeded": (
                _case_value(with_edl, "max_abs_phi_tilde")
                > float(params["dh_warn_threshold"])
            ),
            "caveat": (
                "Linearized PB convergence does not establish strong-field "
                "physical validity."
            ),
        },
        "fourier_caveat": (
            "Piecewise Stern data can show Gibbs ringing near material boundaries; "
            "interpret boundary-scale oscillations only after mode convergence."
        ),
        "passed": passed,
    }


def _summary(
    params: Mapping[str, Any],
    derived: Mapping[str, Any],
    with_edl: Mapping[str, Any],
    without_edl: Mapping[str, Any],
) -> dict[str, Any]:
    E_with = _case_value(with_edl, "E_mix_V", "E_mix")
    E_no = _case_value(without_edl, "E_mix_V", "E_mix")
    i_with = _case_value(with_edl, "i_mix_avg_A_per_m2")
    i_no = _case_value(without_edl, "i_mix_avg_A_per_m2")
    return {
        "model_name": MODEL_NAME,
        "geometry_name": GEOMETRY_NAME,
        "topology_id": TOPOLOGY_ID,
        "electrostatic_backend": ELECTROSTATIC_BACKEND,
        "condition_results": {
            "with_edl": {
                "E_mix_V": E_with,
                "i_mix_avg_A_per_m2": i_with,
                "i_mix_abs_halfcell_A": _case_value(with_edl, "i_mix_abs_halfcell_A"),
                "i_mix_abs_full_period_A": _case_value(with_edl, "i_mix_abs_full_period_A"),
                "I_Au_A": _case_value(with_edl, "I_Au_A"),
                "I_Pd_A": _case_value(with_edl, "I_Pd_A"),
                "relative_balance_residual": _case_value(
                    with_edl, "relative_balance_residual"
                ),
            },
            "without_edl": {
                "E_mix_V": E_no,
                "i_mix_avg_A_per_m2": i_no,
                "i_mix_abs_halfcell_A": _case_value(
                    without_edl, "i_mix_abs_halfcell_A"
                ),
                "i_mix_abs_full_period_A": _case_value(
                    without_edl, "i_mix_abs_full_period_A"
                ),
                "I_Au_A": _case_value(without_edl, "I_Au_A"),
                "I_Pd_A": _case_value(without_edl, "I_Pd_A"),
                "relative_balance_residual": _case_value(
                    without_edl, "relative_balance_residual"
                ),
            },
        },
        "delta_E_with_minus_without_V": E_with - E_no,
        "delta_i_with_minus_without_A_per_m2": i_with - i_no,
        "geometry": {
            "segments": derived["segments"],
            "halfcell_length_nm": float(derived["L_halfcell_m"]) * 1.0e9,
            "full_mirror_period_nm": float(derived["L_full_period_m"]) * 1.0e9,
            "reactive_area_halfcell_m2": float(derived["reactive_area_halfcell_m2"]),
            "reactive_area_full_period_m2": float(derived["reactive_area_full_period_m2"]),
        },
        "numerical": {
            "N_modes": int(params["N_modes"]),
            "Nx": int(params["Nx"]),
            "gl_order": int(params["gl_order"]),
            "Ny_2d": int(params["Ny_2d"]),
            "Nx_2d": DISPLAY_NX,
            "y_max_lambda_D": float(params["y_max_lambda_D"]),
        },
    }


def _surface_rows(surface: Mapping[str, np.ndarray]) -> Iterable[dict[str, Any]]:
    keys = (
        "x_nm", "segment_id", "material", "phi_tilde_with", "phi_tilde_no",
        "phi_rp_with_V", "phi_rp_no_V", "c_R1_with", "c_O2_with",
        "c_R1_no", "c_O2_no", "eta_au_with_V", "eta_pd_with_V",
        "eta_au_no_V", "eta_pd_no_V", "j_au_with", "j_pd_with",
        "j_au_no", "j_pd_no",
    )
    arrays = {key: np.asarray(surface[key]) for key in keys}
    for index in range(arrays["x_nm"].size):
        yield {key: arrays[key][index] for key in keys}


def _charge_rows(items: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for item in items:
        for x_nm, phi_V, sigma in zip(
            item["x_nm"], item["phi_rp_V"], item["sigma_C_per_m2"], strict=True
        ):
            yield {
                "segment_id": item["segment_id"],
                "material": item["material"],
                "x_nm": x_nm,
                "phi_rp_V": phi_V,
                "sigma_C_per_m2": sigma,
            }


def _quadrature_rows(quadrature: Mapping[str, Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    for condition, result in quadrature.items():
        for material in ("Au", "Pd"):
            values = result[material]
            for x_m, weight, phi, eta, current in zip(
                values["x_m"],
                values["weights_m"],
                values["phi_tilde"],
                values["eta_V"],
                values["current_density_A_per_m2"],
                strict=True,
            ):
                yield {
                    "condition": condition,
                    "material": material,
                    "x_nm": float(x_m) * 1.0e9,
                    "weight_m": weight,
                    "phi_tilde": phi,
                    "eta_V": eta,
                    "current_density_A_per_m2": current,
                }


def _evenly_spaced_indices(size: int, maximum_count: int) -> np.ndarray:
    if size <= maximum_count:
        return np.arange(size, dtype=int)
    return np.unique(
        np.rint(np.linspace(0, size - 1, maximum_count)).astype(int)
    )


def _grid_rows(
    grid: Mapping[str, np.ndarray],
    x_indices: np.ndarray,
    y_indices: np.ndarray,
) -> Iterable[dict[str, Any]]:
    x = np.asarray(grid["x_nm"])
    y = np.asarray(grid["y_nm"])
    phi = np.asarray(grid["phi_s_mV"])
    c_r1 = np.asarray(grid["c_R1_norm"])
    c_o2 = np.asarray(grid["c_O2_norm"])
    for y_index in y_indices:
        y_nm = y[y_index]
        for x_index in x_indices:
            x_nm = x[x_index]
            yield {
                "x_nm": x_nm,
                "y_nm": y_nm,
                "phi_s_mV": phi[y_index, x_index],
                "c_R1_over_bulk": c_r1[y_index, x_index],
                "c_O2_over_bulk": c_o2[y_index, x_index],
            }


def _write_result_data(
    run_directory: Path,
    bundle: Mapping[str, Any],
    baseline: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> list[Path]:
    written = [
        write_json(run_directory / "params.json", bundle["params"]),
        write_json(run_directory / "derived.json", bundle["derived"]),
        write_json(run_directory / "summary.json", bundle["summary"]),
        write_json(run_directory / "validation.json", bundle["validation"]),
        write_json(run_directory / "environment.json", environment_snapshot()),
        write_json(run_directory / "inputs" / "baseline_params.json", baseline),
        write_json(run_directory / "inputs" / "overrides.json", overrides),
        write_json(
            run_directory / "inputs" / "numerical_resolution.json",
            bundle["validation"]["convergence"],
        ),
    ]

    summary_rows = []
    for condition, values in bundle["summary"]["condition_results"].items():
        summary_rows.append({"condition": condition, **values})
    summary_fields = ["condition", *bundle["summary"]["condition_results"]["with_edl"].keys()]
    written.append(
        write_csv(run_directory / "csv" / "summary_compare.csv", summary_fields, summary_rows)
    )

    convergence_rows = []
    for attempt in bundle["validation"]["convergence"].get("attempts", []):
        mode = attempt["mode_comparison"]
        gl = attempt["quadrature_comparison"]
        convergence_rows.append(
            {
                "attempt": attempt["attempt"],
                "lower_N_modes": mode["lower_N_modes"],
                "upper_N_modes": mode["upper_N_modes"],
                "fixed_gl_order_for_modes": mode["fixed_gl_order"],
                "mode_delta_E_mix_mV": mode["delta_E_mix_mV"],
                "mode_delta_i_mix_percent": mode["delta_i_mix_percent"],
                "mode_passed": mode["passed"],
                "fixed_N_modes_for_GL": gl["fixed_N_modes"],
                "lower_gl_order": gl["lower_gl_order"],
                "upper_gl_order": gl["upper_gl_order"],
                "GL_delta_E_mix_mV": gl["delta_E_mix_mV"],
                "GL_delta_i_mix_percent": gl["delta_i_mix_percent"],
                "GL_passed": gl["passed"],
                "attempt_passed": attempt["passed"],
            }
        )
    if convergence_rows:
        written.append(
            write_csv(
                run_directory / "csv" / "convergence.csv",
                list(convergence_rows[0]),
                convergence_rows,
            )
        )

    surface_fields = list(next(iter(_surface_rows(bundle["surface"]))).keys())
    written.append(
        write_csv(
            run_directory / "csv" / "surface_profiles.csv",
            surface_fields,
            _surface_rows(bundle["surface"]),
        )
    )
    charge_fields = ["segment_id", "material", "x_nm", "phi_rp_V", "sigma_C_per_m2"]
    written.append(
        write_csv(
            run_directory / "csv" / "surface_charge_distribution.csv",
            charge_fields,
            _charge_rows(bundle["surface_charge_segments"]),
        )
    )
    quadrature_fields = [
        "condition", "material", "x_nm", "weight_m", "phi_tilde", "eta_V",
        "current_density_A_per_m2",
    ]
    written.append(
        write_csv(
            run_directory / "csv" / "gl_current_quadrature.csv",
            quadrature_fields,
            _quadrature_rows(bundle["quadrature"]),
        )
    )
    written.append(
        write_csv(
            run_directory / "csv" / "display_grid_2d_preview.csv",
            ["x_nm", "y_nm", "phi_s_mV", "c_R1_over_bulk", "c_O2_over_bulk"],
            _grid_rows(
                bundle["grid_2d"],
                _evenly_spaced_indices(
                    np.asarray(bundle["grid_2d"]["x_nm"]).size,
                    DISPLAY_CSV_MAX_NX,
                ),
                _evenly_spaced_indices(
                    np.asarray(bundle["grid_2d"]["y_nm"]).size,
                    DISPLAY_CSV_MAX_NY,
                ),
            ),
        )
    )

    full_nx = int(np.asarray(bundle["grid_2d"]["x_nm"]).size)
    full_ny = int(np.asarray(bundle["grid_2d"]["y_nm"]).size)
    csv_x_indices = _evenly_spaced_indices(full_nx, DISPLAY_CSV_MAX_NX)
    csv_y_indices = _evenly_spaced_indices(full_ny, DISPLAY_CSV_MAX_NY)
    written.append(
        write_json(
            run_directory / "grid_2d_metadata.json",
            {
                "full_resolution": {
                    "shape_Ny_Nx": [full_ny, full_nx],
                    "artifact": "npz/display_grid_2d.npz",
                    "description": "Full figure/display grid used for the 2D plots",
                },
                "human_readable_preview": {
                    "shape_Ny_Nx": [
                        int(csv_y_indices.size),
                        int(csv_x_indices.size),
                    ],
                    "artifact": "csv/display_grid_2d_preview.csv",
                    "sampling": "evenly spaced indices including both endpoints",
                    "x_source_indices": csv_x_indices.tolist(),
                    "y_source_indices": csv_y_indices.tolist(),
                },
            },
        )
    )

    model: LinearPBModel = bundle["model"]
    E_with = _case_value(bundle["with_edl"], "E_mix_V", "E_mix")
    written.append(
        write_npz(
            run_directory / "npz" / "spectral_coefficients.npz",
            rho=model.rho,
            gamma=model.gamma,
            A_M=model.A_M,
            A_pzc=model.A_pzc,
            A_with_edl=model.coefficients(E_with),
            A_without_edl=np.zeros_like(model.A_M),
        )
    )
    grid = bundle["grid_2d"]
    written.append(
        write_npz(
            run_directory / "npz" / "display_grid_2d.npz",
            x_nm=grid["x_nm"],
            y_nm=grid["y_nm"],
            phi_tilde=grid["phi_tilde"],
            phi_s_mV=grid["phi_s_mV"],
            c_R1_norm=grid["c_R1_norm"],
            c_O2_norm=grid["c_O2_norm"],
        )
    )
    return written


def build_result_bundle(
    params: Mapping[str, Any] | None = None,
    output_root: str | Path | None = None,
    run_id: str | None = None,
    publish: bool = True,
    perform_convergence: bool | None = None,
) -> dict[str, Any]:
    """Compute, validate, and optionally publish one complete result bundle.

    ``params`` is an override mapping on top of :func:`default_params`.  A
    published run always performs the production convergence study and
    receives a fresh ``results/<run_id>`` directory.  An existing explicit
    destination raises :class:`FileExistsError` before the numerical solve.
    """

    if publish and perform_convergence is False:
        raise ValueError(
            "Publishing an unconverged/debug result is disabled; use publish=False"
        )
    if publish and run_id is not None:
        preflight = Path(output_root or DEFAULT_OUTPUT_ROOT).expanduser().resolve() / run_id
        if preflight.exists():
            raise FileExistsError(f"Result directory already exists: {preflight}")

    baseline = default_params()
    overrides = dict(params or {})
    requested = apply_param_overrides(baseline, overrides)
    validate_params(requested)
    if publish and any(
        not math.isclose(
            float(requested[name]),
            float(baseline[name]),
            rel_tol=0.0,
            abs_tol=1.0e-18,
        )
        for name in ("L_C_left", "L_Au", "L_Pd", "L_C_right")
    ):
        raise ValueError(
            "This package publishes only the canonical C(5)|Au(4)|Pd(4)|C(5) "
            "geometry; use publish=False for geometry regressions"
        )
    canonical, convergence, model, with_edl = _resolve_converged_solution(
        requested,
        perform=True if publish else perform_convergence,
    )
    derived = compute_derived_params(canonical)
    without_edl = solve_case(canonical, use_edl=False)

    surface = _build_surface_data(model, canonical, with_edl, without_edl)
    quadrature = _build_quadrature_data(model, canonical, with_edl, without_edl)
    surface_charge_segments = _build_surface_charge_segments(
        model, canonical, with_edl, surface
    )
    grid_2d = _build_grid_2d(model, canonical, with_edl)
    summary = _summary(canonical, derived, with_edl, without_edl)
    geometry_validation = _geometry_validation(model, canonical)
    numerical_validation = _numerical_validation(
        model,
        canonical,
        with_edl,
        without_edl,
        surface,
        quadrature,
        surface_charge_segments,
        grid_2d,
    )
    validation: dict[str, Any] = {
        "geometry": geometry_validation,
        "numerical": numerical_validation,
        "convergence": convergence,
        "artifact_checks": {"published": False, "passed": True},
        "passed": bool(
            geometry_validation["passed"]
            and numerical_validation["passed"]
            and convergence["passed"]
        ),
    }
    if not validation["passed"]:
        raise RuntimeError(f"Numerical result validation failed: {validation}")

    bundle: dict[str, Any] = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "model_name": MODEL_NAME,
        "geometry_name": GEOMETRY_NAME,
        "topology_id": TOPOLOGY_ID,
        "params": canonical,
        "derived": derived,
        "with_edl": with_edl,
        "without_edl": without_edl,
        "surface": surface,
        "quadrature": quadrature,
        "surface_charge_segments": surface_charge_segments,
        "grid_2d": grid_2d,
        "summary": summary,
        "validation": validation,
        "model": model,
        "output": None,
        "run_id": run_id,
        "figure_paths": [],
        "checksums": {},
    }
    if not publish:
        return bundle

    destination, identifier = make_run_directory(
        output_root or DEFAULT_OUTPUT_ROOT,
        run_id,
    )
    bundle["output"] = str(destination)
    bundle["run_id"] = identifier

    figure_paths = [Path(path) for path in generate_all_figures(bundle, destination)]
    figure_relatives = relative_paths(figure_paths, destination)
    pngs = sorted(destination.glob("figures/**/*.png"))
    svgs = sorted(destination.glob("figures/**/*.svg"))
    pdfs = sorted(destination.glob("**/*.pdf"))
    editable_svg = all("<text" in path.read_text(encoding="utf-8") for path in svgs)
    figure3_pngs = sorted((destination / "figures" / "Figure_3").glob("*.png"))
    figure_rp_pngs = sorted((destination / "figures" / "Figure_RP").glob("*.png"))
    artifact_passed = (
        len(figure_paths) == 18
        and len(set(figure_relatives)) == 18
        and len(pngs) == 9
        and len(svgs) == 9
        and len(figure3_pngs) == 6
        and len(figure_rp_pngs) == 3
        and not pdfs
        and editable_svg
    )
    validation["artifact_checks"] = {
        "published": True,
        "generate_all_figures_return_count": len(figure_paths),
        "figure3_png_count": len(figure3_pngs),
        "figure_rp_png_count": len(figure_rp_pngs),
        "png_count": len(pngs),
        "svg_count": len(svgs),
        "pdf_count": len(pdfs),
        "all_svg_text_editable": editable_svg,
        "expected": "9 PNG + 9 SVG + 0 PDF",
        "passed": artifact_passed,
    }
    validation["passed"] = bool(validation["passed"] and artifact_passed)
    if not validation["passed"]:
        raise RuntimeError(f"Published artifact validation failed: {validation}")

    data_paths = _write_result_data(destination, bundle, baseline, overrides)
    source = source_hashes(PACKAGE_ROOT)
    figure_metadata = {
        "dpi_png": FIGURE_DPI,
        "formats": ["png", "svg"],
        "svg_fonttype": "none",
        "pdf_enabled": False,
        "font_stack": "Helvetica, Nimbus Sans, Arial, DejaVu Sans, sans-serif",
        "figure_files": figure_relatives,
        "figure_count": 18,
        "generated_by": "janus_on_c_support.figures.generate_all_figures",
    }
    figure_metadata_path = write_json(
        destination / "figure_metadata.json", figure_metadata
    )
    manifest = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
        "model_name": MODEL_NAME,
        "geometry_name": GEOMETRY_NAME,
        "topology_id": TOPOLOGY_ID,
        "electrostatic_backend": ELECTROSTATIC_BACKEND,
        "run_id": identifier,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "output_directory": str(destination),
        "parameter_provenance": "default_params plus explicit inputs/overrides.json",
        "source": source,
        "formats": ["png", "svg", "csv", "npz", "json"],
        "checksums_file": CHECKSUM_FILENAME,
        "validation_passed": True,
    }
    manifest_path = write_json(destination / "run_manifest.json", manifest)

    unhashed_paths = [*figure_paths, *data_paths, figure_metadata_path, manifest_path]
    artifact_relatives = relative_paths(unhashed_paths, destination)
    hashes = artifact_hashes(
        destination,
        exclude=(CHECKSUM_FILENAME, "artifacts.json"),
    )
    artifacts = {
        "files": artifact_relatives,
        "file_count": len(artifact_relatives),
        "sha256": hashes,
        "checksums_file": CHECKSUM_FILENAME,
    }
    artifacts_path = write_json(destination / "artifacts.json", artifacts)
    checksum_path, checksums = write_checksums(destination)
    bundle["figure_paths"] = figure_relatives
    bundle["artifacts"] = artifacts
    bundle["checksums"] = checksums
    bundle["metadata_paths"] = relative_paths(
        [figure_metadata_path, manifest_path, artifacts_path, checksum_path], destination
    )
    return bundle


__all__ = ["DEFAULT_OUTPUT_ROOT", "DISPLAY_NX", "build_result_bundle"]
