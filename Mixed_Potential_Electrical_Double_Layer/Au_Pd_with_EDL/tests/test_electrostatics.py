from __future__ import annotations

import math

import numpy as np
import pytest

from au_pd_edl.electrostatics import I_mn, J_n, LinearPBModel
from au_pd_edl.kinetics import au_local_current_density, pd_local_current_density
from au_pd_edl.parameters import apply_param_overrides


def test_legacy_cosine_integrals_include_zero_mode_limits() -> None:
    a, b = 0.7, 2.9
    assert J_n(0, a, b, 0.0) == pytest.approx(b - a)
    assert I_mn(0, 0, a, b, 0.0, 0.0) == pytest.approx(b - a)

    rm, rn = 0.8, 1.7
    x = np.linspace(a, b, 100001)
    numerical = np.trapezoid(np.cos(rm * x) * np.cos(rn * x), x)
    assert I_mn(1, 2, a, b, rm, rn) == pytest.approx(
        numerical, rel=2.0e-10
    )


@pytest.mark.parametrize("separation_nm", [0.0, 3.0])
def test_spectral_domain_and_metadata(
    coarse_params: dict[str, object], separation_nm: float
) -> None:
    model = LinearPBModel(
        apply_param_overrides(
            coarse_params, {"d_Au_Pd": separation_nm * 1.0e-9}
        )
    )
    metadata = model.spectral_metadata()

    assert metadata["electrostatic_backend"] == "semi_infinite_cosine_fourier"
    assert metadata["electrolyte_domain_id"] == (
        "semi_infinite_upper_half_strip_y_ge_0"
    )
    assert metadata["electrolyte_domain_description"] == (
        "semi-infinite strip 0 <= x_tilde <= L_tilde, y_tilde >= 0"
    )
    assert metadata["farfield_condition"] == (
        "phi_tilde -> 0 as y_tilde -> infinity"
    )
    assert "farfield_top_over_lambda" not in metadata
    assert metadata["n_coefficients"] == int(coarse_params["N_modes"]) + 1
    assert metadata["diagnostic_surface_n_x_coordinates"] == int(
        model.surface_x_tilde.size
    )
    assert "saved_surface_n_x_coordinates" not in metadata
    assert "n_x_coordinates" not in metadata
    assert metadata["geometry"] == (
        "flush_coplanar_on_uncharged_insulating_substrate"
    )
    assert metadata["topology_checks"]["substrate_gap_is_boundary"] is (
        separation_nm > 0.0
    )
    assert metadata["topology_checks"]["reactive_boundary_interiors_disjoint"]
    assert metadata["topology_checks"]["reactive_boundary_overlap_measure_zero"]
    assert not hasattr(model, "free_dofs")
    assert not hasattr(model, "dirichlet_dofs")
    assert not hasattr(model, "nodal_field")
    assert not hasattr(model, "boundary_node_field")
    assert model.surface_field(0.60).shape == model.surface_x_tilde.shape
    assert model.boundary_sample_field("Au", 0.60).shape == (
        model.boundary_quadrature["Au"].x_tilde.shape
    )


def test_reactive_quadrature_lengths_and_gap_is_neumann(
    coarse_params: dict[str, object],
) -> None:
    model = LinearPBModel(
        apply_param_overrides(coarse_params, {"d_Au_Pd": 3.0e-9})
    )
    derived = model.derived
    au_length = float(np.sum(model.boundary_quadrature["Au"].weights_tilde))
    pd_length = float(np.sum(model.boundary_quadrature["Pd"].weights_tilde))

    assert au_length == pytest.approx(float(derived["L_Au_tilde"]), rel=1.0e-13)
    assert pd_length == pytest.approx(float(derived["L_Pd_tilde"]), rel=1.0e-13)
    metadata = model.spectral_metadata()
    assert metadata["substrate_gap_electrostatic_bc"] == "homogeneous Neumann"
    assert metadata["topology_checks"]["boundary_lengths_tilde"][
        "substrate_gap"
    ] == pytest.approx(float(derived["d_Au_Pd_tilde"]), rel=1.0e-13)


def test_projected_matrix_matches_legacy_Au_gap0_Pd_assembly(
    coarse_params: dict[str, object],
) -> None:
    params = apply_param_overrides(
        coarse_params, {"d_Au_Pd": 3.0e-9, "N_modes": 12}
    )
    model = LinearPBModel(params)
    d = model.derived
    L = float(d["L_total_tilde"])
    L_Au = float(d["L_Au_tilde"])
    x_Pd = L_Au + float(d["d_Au_Pd_tilde"])
    expected_S = np.zeros_like(model.S)
    expected_rhs_m = np.zeros_like(model.rhs_m)
    expected_rhs_pzc = np.zeros_like(model.rhs_pzc)
    segments = (
        (0.0, L_Au, float(d["g_Au"]), float(d["pzc_Au_tilde"])),
        # The exposed substrate gap is exactly g=0.  Including it explicitly
        # proves that it contributes neither matrix nor PZC load.
        (L_Au, x_Pd, 0.0, 0.0),
        (x_Pd, L, float(d["g_Pd"]), float(d["pzc_Pd_tilde"])),
    )
    for m, rm in enumerate(model.rho):
        row_weight = 1.0 / L if m == 0 else 2.0 / L
        for n, rn in enumerate(model.rho):
            expected_S[m, n] = row_weight * sum(
                g * I_mn(m, n, a, b, float(rm), float(rn))
                for a, b, g, _pzc in segments
            )
        expected_rhs_m[m] = row_weight * sum(
            g * J_n(m, a, b, float(rm)) for a, b, g, _pzc in segments
        )
        expected_rhs_pzc[m] = row_weight * sum(
            g * pzc * J_n(m, a, b, float(rm))
            for a, b, g, pzc in segments
        )

    assert np.allclose(model.S, expected_S, rtol=0.0, atol=2.0e-14)
    assert np.allclose(model.rhs_m, expected_rhs_m, rtol=0.0, atol=2.0e-14)
    assert np.allclose(
        model.rhs_pzc, expected_rhs_pzc, rtol=0.0, atol=2.0e-13
    )


def test_interpolation_rejects_points_below_flush_substrate(
    coarse_params: dict[str, object],
) -> None:
    model = LinearPBModel(
        apply_param_overrides(coarse_params, {"d_Au_Pd": 3.0e-9})
    )
    with pytest.raises(ValueError, match=r"only y >= 0"):
        model.interpolate(0.60, np.asarray([1.0]), np.asarray([-1.0e-6]))


@pytest.mark.parametrize("separation_nm", [0.0, 3.0])
def test_affine_spectral_residual_is_small(
    coarse_params: dict[str, object], separation_nm: float
) -> None:
    params = apply_param_overrides(
        coarse_params, {"d_Au_Pd": separation_nm * 1.0e-9}
    )
    residuals = LinearPBModel(params).affine_residuals()
    assert residuals["phi_m_relative_l2"] < 1.0e-10
    assert residuals["phi_pzc_relative_l2"] < 1.0e-10


def test_homogeneous_surface_matches_semi_infinite_closed_form(
    coarse_params: dict[str, object],
) -> None:
    base = LinearPBModel(coarse_params)
    common_g = float(base.derived["g_Au"])
    common_pzc = float(base.derived["pzc_Au"])
    params = apply_param_overrides(
        coarse_params,
        {
            "d_Au_Pd": 0.0,
            "g_Pd": common_g,
            "pzc_Pd": common_pzc,
        },
    )
    model = LinearPBModel(params)
    E = 0.60
    expected = common_g * float(model.derived["beta"]) * (E - common_pzc) / (
        1.0 + common_g
    )
    field = np.concatenate(
        [model.boundary_field("Au", E), model.boundary_field("Pd", E)]
    )
    assert np.max(np.abs(field - expected)) < 1.0e-10


def test_interpolation_concentrations_and_currents_are_finite(
    coarse_params: dict[str, object],
) -> None:
    params = apply_param_overrides(coarse_params, {"d_Au_Pd": 3.0e-9})
    model = LinearPBModel(params)
    field = model.upper_grid(0.60, n_x=41, n_y=25, y_max_over_lambda=3.0)[
        "phi_tilde"
    ]
    c_r1 = np.exp(np.clip(-float(params["z_R1"]) * field, -700.0, 700.0))
    c_o2 = np.exp(np.clip(-float(params["z_O2"]) * field, -700.0, 700.0))

    assert np.all(np.isfinite(field))
    assert np.all(np.isfinite(c_r1)) and np.all(c_r1 > 0.0)
    assert np.all(np.isfinite(c_o2)) and np.all(c_o2 > 0.0)

    phi_au = model.boundary_field("Au", 0.60)
    phi_pd = model.boundary_field("Pd", 0.60)
    j_au = np.asarray(au_local_current_density(0.60, phi_au, params), dtype=float)
    j_pd = np.asarray(pd_local_current_density(0.60, phi_pd, params), dtype=float)
    assert np.all(np.isfinite(j_au)) and np.all(j_au > 0.0)
    assert np.all(np.isfinite(j_pd)) and np.all(j_pd < 0.0)


def test_interpolation_matches_direct_cosine_sum(
    coarse_params: dict[str, object],
) -> None:
    model = LinearPBModel(
        apply_param_overrides(coarse_params, {"d_Au_Pd": 3.0e-9})
    )
    rng = np.random.default_rng(20260802)
    x = rng.uniform(0.0, float(model.derived["L_total_tilde"]), 80)
    y = rng.uniform(0.0, 5.0, 80)
    E = 0.60
    coefficients = model.coefficient_field(E)
    direct = np.sum(
        np.cos(np.outer(x, model.rho))
        * np.exp(-np.outer(y, model.gamma))
        * coefficients[None, :],
        axis=1,
    )
    assert np.allclose(model.interpolate(E, x, y), direct, rtol=0.0, atol=1.0e-12)


def test_every_mode_decays_in_the_far_field(
    coarse_params: dict[str, object],
) -> None:
    model = LinearPBModel(coarse_params)
    assert np.all(model.gamma >= 1.0)
    near = np.max(np.abs(model.upper_grid(0.60, 101, 2, 1.0)["phi_tilde"][-1]))
    far = np.max(np.abs(model.upper_grid(0.60, 101, 2, 20.0)["phi_tilde"][-1]))
    assert far < near * math.exp(-18.0)
