import math

import numpy as np
import pytest

from au_pd_independent_edls.model import IndependentPlanarEDLModel, solve_comparison


def test_baseline_regression_and_balance() -> None:
    result = solve_comparison()
    with_edl = result["with_edl"]
    without = result["without_edl"]
    assert with_edl["E_mix_V"] == pytest.approx(0.6008249905513604, abs=2e-12)
    assert without["E_mix_V"] == pytest.approx(0.467, abs=2e-12)
    assert with_edl["i_mix_avg_A_per_m2"] == pytest.approx(0.0799514258453, rel=2e-11)
    assert with_edl["relative_balance_residual"] < 1e-12
    assert abs(with_edl["closed_form_minus_brent_V"]) < 5e-11


def test_half_space_profile_and_robin_residual() -> None:
    model = IndependentPlanarEDLModel()
    E = model.solve()["E_mix_V"]
    for material in ("Au", "Pd"):
        phi0 = model.surface_phi_tilde(E, material)
        y = np.array([0.0, model.derived["lambda_D"], 20 * model.derived["lambda_D"]])
        profile = model.phi_tilde_profile(E, material, y)
        assert profile[0] == pytest.approx(phi0)
        assert profile[1] == pytest.approx(phi0 / math.e)
        assert abs(profile[-1]) < abs(phi0) * 3e-9
        g = model.derived[f"g_{material}"]
        pzc = model.params[f"pzc_{material}"]
        driving = model.derived["beta_per_V"] * (E - pzc)
        outward_derivative = phi0
        assert outward_derivative + g * phi0 == pytest.approx(g * driving, abs=1e-12)


def test_area_scaling_keeps_emix_and_scales_current() -> None:
    one = IndependentPlanarEDLModel().solve()
    two = IndependentPlanarEDLModel(
        {"active_faces_Au": 2, "active_faces_Pd": 2}
    ).solve()
    assert two["E_mix_V"] == pytest.approx(one["E_mix_V"], abs=1e-12)
    assert two["I_Au_A"] == pytest.approx(2 * one["I_Au_A"], rel=1e-12)


def test_g_zero_reduces_to_no_edl() -> None:
    result = solve_comparison({"g_Au": 0.0, "g_Pd": 0.0})
    assert result["with_edl"]["E_mix_V"] == pytest.approx(
        result["without_edl"]["E_mix_V"], abs=1e-12
    )
