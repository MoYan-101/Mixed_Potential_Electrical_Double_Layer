import numpy as np
import pytest

from au_pd_facing_gap.model import FacingElectrolyteSlitModel, solve_cases
from au_pd_facing_gap.nonlinear_model import (
    NonlinearFacingElectrolyteSlitModel,
    solve_planar_gouy_chapman_stern_limit,
)


def test_rejects_zero_or_negative_gap() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        FacingElectrolyteSlitModel({}, 0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        FacingElectrolyteSlitModel({}, -1.0)


@pytest.mark.parametrize(
    ("gap_nm", "expected_E", "expected_au", "expected_pd"),
    [
        (3.0, 0.6242900341334946, -6.862962706236852, -5.387844893598346),
        (10.0, 0.6035802257161148, -6.104229721401284, -4.533558466566083),
        (1000.0, 0.6008249905513604, -5.982661115191934, -4.4405307947489385),
    ],
)
def test_baseline_regression(gap_nm: float, expected_E: float, expected_au: float, expected_pd: float) -> None:
    result = FacingElectrolyteSlitModel({}, gap_nm).solve()
    assert result["E_mix_V"] == pytest.approx(expected_E, abs=3e-12)
    assert result["phi_RP_Au_tilde"] == pytest.approx(expected_au, abs=3e-11)
    assert result["phi_RP_Pd_tilde"] == pytest.approx(expected_pd, abs=3e-11)
    assert result["relative_balance_residual"] < 1e-12
    assert max(abs(value) for value in result["robin_residuals"].values()) < 1e-12


def test_profile_satisfies_linear_pb_ode() -> None:
    model = FacingElectrolyteSlitModel({}, 10.0)
    E = model.solve()["E_mix_V"]
    x = np.linspace(0.0, model.gap_m, 2001)
    phi = model.phi_tilde(E, x)
    dx_tilde = (x[1] - x[0]) / model.derived["lambda_D"]
    second = (phi[:-2] - 2 * phi[1:-1] + phi[2:]) / dx_tilde**2
    residual = -second + phi[1:-1]
    assert np.max(np.abs(residual)) < 3e-5


def test_large_gap_matches_independent_planar_limit() -> None:
    model = FacingElectrolyteSlitModel({}, 1000.0)
    result = model.solve()
    beta = model.derived["beta_per_V"]
    E = result["E_mix_V"]
    expected_au = model.derived["g_Au"] / (1 + model.derived["g_Au"]) * beta * (E - model.params["pzc_Au"])
    expected_pd = model.derived["g_Pd"] / (1 + model.derived["g_Pd"]) * beta * (E - model.params["pzc_Pd"])
    assert result["phi_RP_Au_tilde"] == pytest.approx(expected_au, abs=1e-12)
    assert result["phi_RP_Pd_tilde"] == pytest.approx(expected_pd, abs=1e-12)


def test_standard_case_list_is_strictly_positive() -> None:
    result = solve_cases()
    assert [case["gap_nm"] for case in result["cases"]] == [3.0, 10.0, 1000.0]


@pytest.mark.parametrize(
    ("gap_nm", "expected_E", "expected_au", "expected_pd"),
    [
        (1.0, 0.586742517611, -4.798214655678, -4.528139383702),
        (2.0, 0.579720896054, -4.541462102339, -4.238000740344),
        (3.0, 0.576929358660, -4.436602319618, -4.125436781824),
        (5.0, 0.574708203478, -4.351618221417, -4.037422348421),
        (10.0, 0.573364616546, -4.299443193048, -3.984949774934),
        (1000.0, 0.573051674540, -4.287200952299, -3.972817983945),
    ],
)
def test_nonlinear_pb_regression(
    gap_nm: float,
    expected_E: float,
    expected_au: float,
    expected_pd: float,
) -> None:
    params = {
        "gap_distances_nm": [gap_nm],
        "root_xtol_V": 1.0e-12,
        "nonlinear_pb_tol": 1.0e-8,
        "nonlinear_pb_initial_points_small": 601,
    }
    result = NonlinearFacingElectrolyteSlitModel(params, gap_nm).solve()
    assert result["E_mix_V"] == pytest.approx(expected_E, abs=3e-10)
    assert result["phi_RP_Au_tilde"] == pytest.approx(expected_au, abs=3e-8)
    assert result["phi_RP_Pd_tilde"] == pytest.approx(expected_pd, abs=3e-8)
    assert result["relative_balance_residual"] < 1e-9
    assert max(abs(value) for value in result["robin_residuals"].values()) < 1e-9
    diagnostics = result["nonlinear_pb_diagnostics"]
    assert diagnostics["max_rms_residual"] < 2e-7
    assert abs(diagnostics["integrated_charge_residual"]) < 1e-5


def test_nonlinear_large_gap_matches_gouy_chapman_asinh_limit() -> None:
    params = {"gap_distances_nm": [1000.0], "root_xtol_V": 1.0e-12}
    result = NonlinearFacingElectrolyteSlitModel(params, 1000.0).solve()
    reference = solve_planar_gouy_chapman_stern_limit(params)
    assert result["E_mix_V"] == pytest.approx(reference["E_mix_V"], abs=1e-11)
    assert result["phi_RP_Au_tilde"] == pytest.approx(
        reference["phi_RP_Au_tilde"], abs=1e-10
    )
    assert result["phi_RP_Pd_tilde"] == pytest.approx(
        reference["phi_RP_Pd_tilde"], abs=1e-10
    )


def test_nonlinear_pb_reduces_to_linear_pb_at_small_potential() -> None:
    params = {
        "gap_distances_nm": [3.0],
        "pzc_Au": 0.5,
        "pzc_Pd": 0.5,
        "nonlinear_pb_tol": 1.0e-9,
    }
    linear = FacingElectrolyteSlitModel(params, 3.0)
    nonlinear = NonlinearFacingElectrolyteSlitModel(params, 3.0)
    E_V = 0.50001
    x_m = np.linspace(0.0, linear.gap_m, 401)
    difference = nonlinear.phi_tilde(E_V, x_m) - linear.phi_tilde(E_V, x_m)
    assert np.max(np.abs(difference)) < 2e-10
