"""Regression and boundary-condition tests for the four-segment core."""

from __future__ import annotations

import json

import numpy as np
import pytest

from janus_on_c_support.electrostatics import LinearPBModel
from janus_on_c_support.parameters import (
    apply_param_overrides,
    default_params,
    validate_params,
)
from janus_on_c_support.solver import solve_case


@pytest.fixture(scope="module")
def default_model() -> LinearPBModel:
    return LinearPBModel(default_params())


@pytest.fixture(scope="module")
def default_with_edl(default_model: LinearPBModel) -> dict[str, object]:
    return solve_case(use_edl=True, model=default_model)


@pytest.fixture(scope="module")
def two_nm_janus_result() -> dict[str, object]:
    params = apply_param_overrides(
        default_params(),
        {
            "L_C_left": 0.0,
            "L_Au": 2.0e-9,
            "L_Pd": 2.0e-9,
            "L_C_right": 0.0,
        },
    )
    return solve_case(params, use_edl=True)


def test_default_segment_edges_and_mirror_period() -> None:
    params = apply_param_overrides(default_params(), {"N_modes": 16, "Nx": 41})
    model = LinearPBModel(params)
    assert [segment.name for segment in model.segments] == [
        "C_left",
        "Au",
        "Pd",
        "C_right",
    ]
    assert [segment.material for segment in model.segments] == ["C", "Au", "Pd", "C"]
    edges_nm = [model.segments[0].x0_m * 1.0e9]
    edges_nm.extend(segment.x1_m * 1.0e9 for segment in model.segments)
    assert edges_nm == pytest.approx([0.0, 5.0, 9.0, 13.0, 18.0], abs=1.0e-12)
    assert model.derived["L_full_period_nm"] == pytest.approx(36.0, abs=1.0e-12)
    assert all(
        left.x1_m == pytest.approx(right.x0_m, abs=1.0e-24)
        for left, right in zip(model.segments[:-1], model.segments[1:], strict=True)
    )


def test_surface_resolution_must_retain_all_distinct_edges() -> None:
    with pytest.raises(ValueError, match="Nx must be >= 5"):
        validate_params(apply_param_overrides(default_params(), {"Nx": 4}))
    zero_c = apply_param_overrides(
        default_params(),
        {
            "L_C_left": 0.0,
            "L_Au": 2.0e-9,
            "L_Pd": 2.0e-9,
            "L_C_right": 0.0,
            "Nx": 3,
        },
    )
    validate_params(zero_c)


def test_uniform_robin_surface_reduces_to_analytic_constant() -> None:
    params = apply_param_overrides(
        default_params(),
        {
            "N_modes": 32,
            "Nx": 151,
            "C_H_Au": 0.30,
            "C_H_C": 0.30,
            "C_H_Pd": 0.30,
            "pzc_Au": 0.60,
            "pzc_C": 0.60,
            "pzc_Pd": 0.60,
        },
    )
    model = LinearPBModel(params)
    E_V = 0.40
    g = float(model.derived["g_C"])
    expected = (
        g
        / (1.0 + g)
        * float(model.derived["beta_per_V"])
        * (E_V - float(params["pzc_C"]))
    )
    actual = model.phi_tilde_surface(E_V, model.surface_grid_m())
    assert actual == pytest.approx(expected, rel=0.0, abs=2.0e-12)
    assert np.max(np.abs(model.A_M[1:])) < 2.0e-13
    assert np.max(np.abs(model.A_pzc[1:])) < 2.0e-13


def test_explicit_g_controls_effective_capacitance_and_surface_charge() -> None:
    explicit_g = 2.75
    params = apply_param_overrides(
        default_params(),
        {
            "N_modes": 32,
            "Nx": 101,
            "g_Au": explicit_g,
            # Deliberately inconsistent input: an explicit g must take priority.
            "C_H_Au": 0.01,
        },
    )
    model = LinearPBModel(params)
    expected_capacitance = (
        explicit_g
        * float(model.derived["epsilon_s_F_per_m"])
        / float(model.derived["lambda_D_m"])
    )
    assert float(model.params["C_H_Au"]) == pytest.approx(expected_capacitance)
    assert float(
        model.derived["C_H_Au_effective_F_per_m2"]
    ) == pytest.approx(expected_capacitance)
    au = next(segment for segment in model.segments if segment.name == "Au")
    assert au.C_H_F_per_m2 == pytest.approx(expected_capacitance)

    E_V = 0.61
    x_m = np.array([0.25 * au.x_start_m + 0.75 * au.x_end_m])
    phi_rp = model.phi_rp_V(E_V, x_m)
    sigma = model.surface_charge_C_per_m2(E_V, x_m, material="Au")
    expected_sigma = expected_capacitance * (
        E_V - float(model.params["pzc_Au"]) - phi_rp
    )
    assert sigma == pytest.approx(expected_sigma, rel=0.0, abs=1.0e-14)


def test_zero_C_two_nm_case_regresses_existing_cosine_janus(
    two_nm_janus_result: dict[str, object],
) -> None:
    # Traceable anchor: Figure_Au2nm_Pd2nm/Au_C_Pd, L_support=0 nm,
    # N_modes=960 and GL128.
    assert two_nm_janus_result["E_mix_V"] == pytest.approx(
        0.6249104327869085, rel=0.0, abs=5.0e-12
    )
    assert two_nm_janus_result["i_mix_avg_A_per_m2"] == pytest.approx(
        0.06649993987830802, rel=0.0, abs=5.0e-11
    )
    derived = two_nm_janus_result["derived"]
    assert isinstance(derived, dict)
    assert derived["edges_nm"] == pytest.approx([0.0, 0.0, 2.0, 4.0, 4.0])


def test_zero_width_C_segments_never_own_domain_endpoints() -> None:
    params = apply_param_overrides(
        default_params(),
        {
            "L_C_left": 0.0,
            "L_Au": 2.0e-9,
            "L_Pd": 2.0e-9,
            "L_C_right": 0.0,
            "N_modes": 16,
            "Nx": 41,
        },
    )
    model = LinearPBModel(params)
    endpoints_m = np.asarray([0.0, 2.0e-9, 4.0e-9])
    assert model.material_profile(endpoints_m).tolist() == ["Au", "Pd", "Pd"]
    assert not np.any(model.segment_mask(endpoints_m, "C_left"))
    assert not np.any(model.segment_mask(endpoints_m, "C_right"))
    assert model.segment_mask(endpoints_m, "Au").tolist() == [True, False, False]
    assert model.segment_mask(endpoints_m, "Pd").tolist() == [False, True, True]

    E_V = 0.61
    automatic = model.surface_charge_C_per_m2(E_V, endpoints_m[-1:])
    explicit_pd = model.surface_charge_C_per_m2(
        E_V, endpoints_m[-1:], material="Pd"
    )
    assert automatic == pytest.approx(explicit_pd, rel=0.0, abs=1.0e-14)


def test_without_edl_equal_length_equal_kinetics_regression() -> None:
    result = solve_case(default_params(), use_edl=False)
    assert result["E_mix_V"] == pytest.approx(0.467, rel=0.0, abs=5.0e-13)
    assert result["i_mix_avg_A_per_m2"] == pytest.approx(
        0.1175603541, rel=0.0, abs=5.0e-11
    )
    assert result["max_abs_phi_tilde"] == 0.0
    assert result["I_C_A"] == 0.0


def test_default_absolute_current_balance_and_json_safety(
    default_with_edl: dict[str, object],
) -> None:
    I_au = float(default_with_edl["I_Au_A"])
    I_pd = float(default_with_edl["I_Pd_A"])
    relative = abs(I_au + I_pd) / (abs(I_au) + abs(I_pd))
    assert relative < 1.0e-10
    assert float(default_with_edl["relative_balance_residual"]) < 1.0e-10
    assert float(default_with_edl["i_mix_abs_full_period_A"]) == pytest.approx(
        2.0 * float(default_with_edl["i_mix_abs_halfcell_A"]),
        rel=1.0e-14,
    )
    # ``allow_nan=False`` makes NaN/Infinity a hard failure.
    json.dumps(default_with_edl, allow_nan=False)


def test_cosine_basis_enforces_both_neumann_boundaries(
    default_model: LinearPBModel,
    default_with_edl: dict[str, object],
) -> None:
    E_mix = float(default_with_edl["E_mix_V"])
    boundaries = np.array([0.0, float(default_model.derived["L_total_m"])])
    for height in (0.0, 2.0 * float(default_model.derived["lambda_D_m"])):
        derivative = default_model.derivative_x_tilde(
            E_mix, boundaries, y_m=height
        )
        assert np.max(np.abs(derivative)) < 1.0e-10
