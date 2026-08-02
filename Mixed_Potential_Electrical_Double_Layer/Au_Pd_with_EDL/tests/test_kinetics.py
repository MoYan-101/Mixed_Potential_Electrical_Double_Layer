from __future__ import annotations

import numpy as np
import pytest

from au_pd_edl.kinetics import (
    au_local_current_density,
    emix_closed_form_no_edl,
    no_edl_absolute_currents,
    pd_local_current_density,
)
from au_pd_edl.parameters import apply_param_overrides, default_params


@pytest.mark.parametrize("separation_nm", [0.0, 3.0, 100.0])
def test_without_edl_is_independent_of_separation(separation_nm: float) -> None:
    params = apply_param_overrides(
        default_params(), {"d_Au_Pd": separation_nm * 1.0e-9}
    )
    E_mix = emix_closed_form_no_edl(params)
    currents = no_edl_absolute_currents(E_mix, params)

    assert E_mix == pytest.approx(0.4670000000000001, abs=1.0e-14)
    assert currents["i_mix_avg_A_per_m2"] == pytest.approx(
        0.11756035407872442, rel=1.0e-13
    )
    assert currents["I_Au_A"] > 0.0
    assert currents["I_Pd_A"] < 0.0
    assert currents["relative_balance_residual"] < 1.0e-13


def test_local_current_signs_and_finiteness() -> None:
    params = default_params()
    phi = np.linspace(-8.0, 8.0, 33)
    j_au = np.asarray(au_local_current_density(0.60, phi, params), dtype=float)
    j_pd = np.asarray(pd_local_current_density(0.60, phi, params), dtype=float)

    assert np.all(np.isfinite(j_au))
    assert np.all(np.isfinite(j_pd))
    assert np.all(j_au > 0.0)
    assert np.all(j_pd < 0.0)
