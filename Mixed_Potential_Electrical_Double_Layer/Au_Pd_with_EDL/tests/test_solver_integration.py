from __future__ import annotations

import math

import pytest

from au_pd_edl.parameters import apply_param_overrides
from au_pd_edl.solver import solve_case


@pytest.mark.integration
def test_mixed_potential_balances_absolute_currents(
    coarse_params: dict[str, object],
) -> None:
    params = apply_param_overrides(coarse_params, {"d_Au_Pd": 3.0e-9})
    result = solve_case(params, use_edl=True)

    assert result["root"]["converged"] is True
    assert math.isfinite(float(result["E_mix_V"]))
    assert float(result["I_Au_A"]) > 0.0
    assert float(result["I_Pd_A"]) < 0.0
    assert float(result["relative_balance_residual"]) < 1.0e-10
    assert abs(float(result["I_Au_A"])) == pytest.approx(
        abs(float(result["I_Pd_A"])), rel=1.0e-10
    )


@pytest.mark.integration
def test_adjacent_limit_matches_high_resolution_legacy_reference(
    coarse_params: dict[str, object],
) -> None:
    """Cross-discretization d=0 regression against N_modes=960/Nx=5000 legacy."""

    params = apply_param_overrides(coarse_params, {"d_Au_Pd": 0.0})
    result = solve_case(params, use_edl=True)

    assert float(result["E_mix_V"]) == pytest.approx(
        0.6006949941170235, abs=1.0e-4
    )
    assert float(result["i_mix_avg_A_per_m2"]) == pytest.approx(
        0.0815880733807085, rel=2.0e-3
    )
