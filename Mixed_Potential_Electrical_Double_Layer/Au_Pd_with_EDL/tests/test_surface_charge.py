from __future__ import annotations

import json

import numpy as np
import pytest

from au_pd_edl.parameters import apply_param_overrides
from au_pd_edl.solver import solve_case
from au_pd_edl.surface_charge import (
    _summarize_material,
    compute_gouy_chapman_diagnostics,
)


@pytest.mark.integration
def test_solve_case_includes_quadrature_based_gc_diagnostics(
    coarse_params: dict[str, object],
) -> None:
    params = apply_param_overrides(
        coarse_params,
        {
            "d_Au_Pd": 0.0,
            # Exercise the effective-capacitance definition rather than the
            # nominal C_H value.
            "g_Au": 0.75,
        },
    )
    result, model = solve_case(params, use_edl=True, return_model=True)
    assert model is not None
    diagnostic = result["gouy_chapman"]

    assert diagnostic["applicable"] is True
    assert diagnostic["status"] == "computed_from_stern_surface_charge"
    assert set(diagnostic["metals"]) == {"Au", "Pd"}

    au = diagnostic["metals"]["Au"]
    derived = result["derived"]
    expected_C_H = (
        float(derived["g_Au"])
        * float(derived["epsilon_s"])
        / float(derived["lambda_D"])
    )
    assert au["effective_C_H_F_per_m2"] == pytest.approx(expected_C_H)
    assert au["effective_C_H_F_per_m2"] != pytest.approx(
        float(params["C_H_Au"])
    )

    phi_s = float(derived["thermal_voltage_V"]) * model.boundary_field(
        "Au", float(result["E_mix_V"])
    )
    sigma = expected_C_H * (
        float(result["E_mix_V"]) - float(params["pzc_Au"]) - phi_s
    )
    weights = model.boundary_quadrature["Au"].weights_tilde
    expected_mean_abs = float(np.sum(np.abs(sigma) * weights) / np.sum(weights))
    expected_length = (
        2.0
        * float(derived["epsilon_s"])
        * float(derived["R"])
        * float(derived["T"])
        / float(derived["F"])
        / expected_mean_abs
    )
    assert au["mean_abs_sigma_C_per_m2"] == pytest.approx(expected_mean_abs)
    assert au["mean_gouy_chapman_length_m"] == pytest.approx(expected_length)
    assert au["boundary_area_m2"] == pytest.approx(
        float(derived["L_Au"]) * float(derived["out_of_plane_width"])
    )
    assert au["n_boundary_samples"] >= 2

    # The entire result remains strict-JSON serializable (no NaN/Infinity).
    json.dumps(result, allow_nan=False)


def test_paper_charge_benchmark_and_sign_symmetry(
    coarse_params: dict[str, object],
) -> None:
    params = apply_param_overrides(coarse_params)
    no_edl_result = solve_case(params, use_edl=False)
    derived = no_edl_result["derived"]
    numerator = (
        2.0
        * float(derived["epsilon_s"])
        * float(derived["R"])
        * float(derived["T"])
        / float(derived["F"])
    )
    # 10 microC/cm2 = 0.1 C/m2.  The absolute value makes the length
    # invariant to the sign of the surface charge.
    common = {
        "weights_tilde": np.asarray([1.0]),
        "effective_C_H": 0.2,
        "gc_numerator_C_per_m": numerator,
        "lambda_D": float(derived["lambda_D"]),
        "out_of_plane_width": float(derived["out_of_plane_width"]),
    }
    positive = _summarize_material(
        sigma_quadrature=np.asarray([0.1]),
        sigma_samples=np.asarray([0.1, 0.1]),
        **common,
    )
    negative = _summarize_material(
        sigma_quadrature=np.asarray([-0.1]),
        sigma_samples=np.asarray([-0.1, -0.1]),
        **common,
    )
    positive_nm = float(positive["mean_gouy_chapman_length_nm"])
    negative_nm = float(negative["mean_gouy_chapman_length_nm"])
    assert positive_nm == pytest.approx(0.35696, abs=5.0e-5)
    assert negative_nm == pytest.approx(positive_nm, rel=0.0, abs=0.0)
    assert positive["charge_sign"] == "positive"
    assert negative["charge_sign"] == "negative"

    mixed = _summarize_material(
        sigma_quadrature=np.asarray([0.1]),
        sigma_samples=np.asarray([0.1, 0.0, -0.1]),
        **common,
    )
    assert mixed["charge_sign"] == "mixed"
    assert mixed["has_sign_change"] is True
    assert mixed["has_zero_charge"] is True
    assert mixed["has_zero_crossing"] is True
    assert mixed["local_max_gouy_chapman_length_m"] is None
    assert mixed["local_max_length_infinite"] is True


def test_without_edl_diagnostic_is_explicitly_not_applicable_and_json_safe(
    coarse_params: dict[str, object],
) -> None:
    result = solve_case(coarse_params, use_edl=False)
    diagnostic = result["gouy_chapman"]

    assert diagnostic["applicable"] is False
    assert diagnostic["status"] == "not_applicable_without_edl"
    assert diagnostic["metals"] is None
    json.dumps(diagnostic, allow_nan=False)


def test_with_edl_diagnostic_requires_matching_model(
    coarse_params: dict[str, object],
) -> None:
    result, model = solve_case(coarse_params, use_edl=True, return_model=True)
    assert model is not None
    with pytest.raises(ValueError, match="required"):
        compute_gouy_chapman_diagnostics(result, None)

    changed = dict(result)
    changed["params"] = apply_param_overrides(
        coarse_params, {"d_Au_Pd": 1.0e-9}
    )
    with pytest.raises(ValueError, match="does not match"):
        compute_gouy_chapman_diagnostics(changed, model)


def test_zero_charge_uses_null_lengths_and_explicit_infinite_flags(
    coarse_params: dict[str, object],
) -> None:
    # C_H=0 is the exact zero-charge boundary case.  It remains a with-EDL
    # solve, but its charge-derived GC length is infinite rather than NaN/Inf.
    params = apply_param_overrides(
        coarse_params,
        {"C_H_Au": 0.0, "g_Au": None},
    )
    result = solve_case(params, use_edl=True)
    au = result["gouy_chapman"]["metals"]["Au"]

    assert au["charge_sign"] == "zero"
    assert au["has_zero_charge"] is True
    assert au["has_zero_crossing"] is True
    assert au["mean_abs_sigma_C_per_m2"] == 0.0
    assert au["mean_gouy_chapman_length_m"] is None
    assert au["mean_gouy_chapman_length_nm"] is None
    assert au["mean_length_infinite"] is True
    assert au["local_min_gouy_chapman_length_m"] is None
    assert au["local_max_gouy_chapman_length_m"] is None
    assert au["local_max_length_infinite"] is True
    json.dumps(result["gouy_chapman"], allow_nan=False)
