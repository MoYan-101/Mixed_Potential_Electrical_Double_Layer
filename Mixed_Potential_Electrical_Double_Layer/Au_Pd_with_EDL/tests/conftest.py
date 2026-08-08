from __future__ import annotations

import pytest

from au_pd_edl.parameters import apply_param_overrides, default_params


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: exercises the spectral PB model and mixed-potential root solve",
    )


@pytest.fixture
def coarse_params() -> dict[str, object]:
    """Fast, deterministic spectral settings shared by model-level tests."""

    return apply_param_overrides(
        default_params(),
        {
            "N_modes": 80,
            "Nx": 1201,
            "dh_violation_action": "ignore",
        },
    )
