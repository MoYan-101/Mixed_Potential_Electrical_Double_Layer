from __future__ import annotations

import math

import pytest

from au_pd_edl.parameters import (
    FORBIDDEN_LEGACY_KEYS,
    MAX_SEPARATION_M,
    apply_param_overrides,
    default_params,
    validate_params,
)


@pytest.mark.parametrize("legacy_key", sorted(FORBIDDEN_LEGACY_KEYS))
def test_rejects_legacy_carbon_and_support_parameters(legacy_key: str) -> None:
    with pytest.raises(ValueError, match=r"Forbidden Au\|C\|Pd parameter"):
        apply_param_overrides(default_params(), {legacy_key: 0.0})


def test_rejects_unknown_parameter() -> None:
    with pytest.raises(KeyError, match=r"Unknown Au\|Pd parameter"):
        apply_param_overrides(default_params(), {"not_a_model_parameter": 1.0})


@pytest.mark.parametrize("separation", [-1.0e-12, MAX_SEPARATION_M + 1.0e-12])
def test_rejects_separation_outside_closed_interval(separation: float) -> None:
    max_separation_nm = MAX_SEPARATION_M * 1.0e9
    with pytest.raises(
        ValueError,
        match=rf"within 0-{max_separation_nm:g} nm",
    ):
        apply_param_overrides(default_params(), {"d_Au_Pd": separation})


def test_accepts_maximum_standalone_separation() -> None:
    params = apply_param_overrides(
        default_params(), {"d_Au_Pd": MAX_SEPARATION_M}
    )
    assert params["d_Au_Pd"] == pytest.approx(MAX_SEPARATION_M)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("epsilon_s", math.nan),
        ("lambda_D", math.inf),
        ("g_Au", -1.0),
        ("g_Pd", math.nan),
        ("L_Au", 0.0),
        ("L_Pd", -1.0e-9),
        ("N_modes", 0),
        ("Nx", 2),
    ],
)
def test_rejects_invalid_physical_and_numerical_values(name: str, value: float) -> None:
    params = default_params()
    params[name] = value
    with pytest.raises(ValueError):
        validate_params(params)


@pytest.mark.parametrize(
    "removed_finite_domain_key",
    ["farfield_lambda_D", "h_bulk_lambda_D", "h_corner_lambda_D", "gap_min_cells"],
)
def test_rejects_removed_finite_domain_controls(
    removed_finite_domain_key: str,
) -> None:
    with pytest.raises(KeyError, match=r"Unknown Au\|Pd parameter"):
        apply_param_overrides(
            default_params(), {removed_finite_domain_key: 1.0}
        )
