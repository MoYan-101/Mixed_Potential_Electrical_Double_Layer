from __future__ import annotations

import pytest

from au_pd_edl.parameters import default_params
from au_pd_edl.validation import run_numerical_validation


@pytest.mark.integration
def test_required_numerical_validation_suite_passes() -> None:
    report = run_numerical_validation(default_params())

    assert report["result_schema_version"] == 2
    assert report["electrostatic_backend"] == "semi_infinite_cosine_fourier"
    assert report["passed"] is True
    assert report["adjacent_legacy_reference"]["passed"] is True
    assert report["homogeneous_robin"]["passed"] is True
    assert report["affine_direct_solve"]["passed"] is True
    assert report["semi_infinite_farfield"]["passed"] is True
    assert report["semi_infinite_farfield"]["finite_H_control_applicable"] is False
    assert report["spectral_mode_convergence"]["passed"] is True
    assert report["surface_quadrature_convergence"]["passed"] is True
    assert report["cache"]["unique_mixed_potential_spectral_solves"] == 12
