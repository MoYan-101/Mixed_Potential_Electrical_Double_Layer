from __future__ import annotations

import numpy as np
import pytest

from au_pd_edl.parameters import compute_derived_params, default_params
from au_pd_edl.scan import default_separations, run_separation_scan


def test_default_scan_grid_contains_required_points() -> None:
    params = default_params()
    values = default_separations(params)
    lambda_D = float(compute_derived_params(params)["lambda_D"])

    assert len(values) == 31
    assert values[0] == pytest.approx(0.0, abs=0.0)
    assert values[-1] == pytest.approx(100.0e-9, abs=1.0e-20)
    for multiple in (1.0, 2.0, 3.0, 5.0):
        assert np.any(np.isclose(values, multiple * lambda_D, rtol=0.0, atol=1.0e-20))


@pytest.mark.integration
def test_plateau_and_without_edl_invariance(
    coarse_params: dict[str, object],
) -> None:
    scan = run_separation_scan(
        coarse_params,
        [0.0, 75.0e-9, 100.0e-9],
        retain_models_at_nm=(),
    )
    rows = scan["rows"]

    assert scan["plateau_check"]["passed"] is True
    assert len({float(row["E_mix_without_EDL_V"]) for row in rows}) == 1
    assert len({float(row["i_mix_avg_without_EDL_A_per_m2"]) for row in rows}) == 1
    assert max(float(row["relative_balance_residual"]) for row in rows) <= 1.0e-8
