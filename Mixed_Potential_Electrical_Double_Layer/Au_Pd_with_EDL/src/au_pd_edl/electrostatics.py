"""Semi-infinite cosine-spectral linearized-PB electrostatics.

The implementation follows the Galerkin/cosine construction used by the
legacy :mod:`Solve_Emix_updating` solver, but the middle Au--Pd span is an
ideal uncharged insulating substrate.  Consequently its charging parameter is
identically zero and it contributes neither a Robin matrix term nor a PZC
load.

``LinearPBModel`` is the public solver class for the semi-infinite strip

    0 <= x_tilde <= L_tilde,  y_tilde >= 0,

with reflecting lateral boundaries and ``phi_tilde -> 0`` as
``y_tilde -> infinity``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .parameters import (
    ELECTROLYTE_DOMAIN_DESCRIPTION,
    ELECTROLYTE_DOMAIN_ID,
    ELECTROSTATIC_BACKEND,
    GEOMETRY_NAME,
    compute_derived_params,
    validate_params,
)


@dataclass(frozen=True)
class BoundaryQuadrature:
    """One-dimensional trapezoidal quadrature on a reactive segment."""

    x_tilde: np.ndarray
    weights_tilde: np.ndarray
    phi_m: np.ndarray
    phi_pzc: np.ndarray


def _cosine_integral(wavenumber: np.ndarray | float, a: float, b: float) -> np.ndarray:
    """Return ``integral_a^b cos(k x) dx`` with a stable ``k=0`` limit."""

    k = np.asarray(wavenumber, dtype=float)
    width = float(b) - float(a)
    midpoint = 0.5 * (float(a) + float(b))
    # np.sinc(q) = sin(pi q)/(pi q), hence the argument below gives
    # sin(k width / 2)/(k width / 2) without a removable singularity.
    return width * np.cos(k * midpoint) * np.sinc(k * width / (2.0 * math.pi))


def J_n(n: int, a: float, b: float, rho_n: float) -> float:
    """Legacy-compatible ``J_n(a,b) = integral cos(rho_n x) dx``."""

    del n  # rho_n contains all information; retained for API/equation parity.
    return float(_cosine_integral(float(rho_n), float(a), float(b)))


def I_mn(
    m: int,
    n: int,
    a: float,
    b: float,
    rho_m: float,
    rho_n: float,
) -> float:
    """Legacy-compatible cosine-product integral on ``[a,b]``."""

    del m, n
    return float(
        0.5
        * (
            _cosine_integral(float(rho_m) - float(rho_n), a, b)
            + _cosine_integral(float(rho_m) + float(rho_n), a, b)
        )
    )


def _cosine_product_integrals(rho: np.ndarray, a: float, b: float) -> np.ndarray:
    """Vectorized matrix of ``integral cos(rho_m x) cos(rho_n x) dx``."""

    difference = rho[:, None] - rho[None, :]
    total = rho[:, None] + rho[None, :]
    return 0.5 * (
        _cosine_integral(difference, a, b) + _cosine_integral(total, a, b)
    )


def _surface_grid_with_boundaries(
    L_total: float,
    L_Au: float,
    x_Pd_start: float,
    n_x: int,
) -> np.ndarray:
    """Uniform surface grid with exact material junctions inserted."""

    x = np.linspace(0.0, float(L_total), int(n_x), dtype=float)
    x = np.concatenate(
        [x, np.asarray([0.0, L_Au, x_Pd_start, L_total], dtype=float)]
    )
    x = np.unique(x)
    x.sort()
    return x


def _segment_trapezoid_grid(
    a: float,
    b: float,
    *,
    target_spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return samples and weights whose measure is exactly ``b-a``."""

    width = float(b) - float(a)
    if width <= 0.0:
        raise ValueError("Reactive boundary length must be positive")
    n_cells = max(1, int(math.ceil(width / float(target_spacing))))
    x = np.linspace(float(a), float(b), n_cells + 1, dtype=float)
    dx = width / n_cells
    weights = np.full(n_cells + 1, dx, dtype=float)
    weights[[0, -1]] *= 0.5
    return x, weights


class LinearPBModel:
    """Cosine/Fourier linearized-PB solver for flush Au/Pd electrodes.

    The spectral solution is

    ``phi_tilde(x,y) = sum_n A_n cos(rho_n x) exp(-gamma_n y)``,

    where ``rho_n=n*pi/L`` and ``gamma_n=sqrt(1+rho_n**2)``.  Projecting the
    piecewise Robin boundary condition gives ``(diag(gamma)+S) A = R``.
    Because the PDE and charging law are linear, the coefficient vector is
    reused affinely as ``A(E)=beta*E*A_M-A_PZC``.
    """

    backend_name = ELECTROSTATIC_BACKEND

    def __init__(self, params: dict[str, Any]):
        self.params = dict(params)
        validate_params(self.params)
        self.derived = compute_derived_params(self.params)
        self._upper_grid_cache: dict[
            tuple[float, int, int, float], dict[str, np.ndarray]
        ] = {}
        self._assemble_affine_fields()
        self._prepare_surface_fields()
        self.boundary_quadrature = {
            "Au": self._build_boundary_quadrature("Au"),
            "Pd": self._build_boundary_quadrature("Pd"),
        }

    def _assemble_affine_fields(self) -> None:
        d = self.derived
        n_modes = int(self.params["N_modes"])
        L_total = float(d["L_total_tilde"])
        L_Au = float(d["L_Au_tilde"])
        x_Pd_start = L_Au + float(d["d_Au_Pd_tilde"])

        mode_index = np.arange(n_modes + 1, dtype=float)
        rho = mode_index * math.pi / L_total
        gamma = np.sqrt(1.0 + rho**2)
        projection_weight = np.full(n_modes + 1, 2.0 / L_total, dtype=float)
        projection_weight[0] = 1.0 / L_total

        au_product = _cosine_product_integrals(rho, 0.0, L_Au)
        pd_product = _cosine_product_integrals(rho, x_Pd_start, L_total)
        charging_product = (
            float(d["g_Au"]) * au_product + float(d["g_Pd"]) * pd_product
        )
        S = projection_weight[:, None] * charging_product
        matrix = np.diag(gamma) + S

        au_load = _cosine_integral(rho, 0.0, L_Au)
        pd_load = _cosine_integral(rho, x_Pd_start, L_total)
        rhs_m = projection_weight * (
            float(d["g_Au"]) * au_load + float(d["g_Pd"]) * pd_load
        )
        rhs_pzc = projection_weight * (
            float(d["g_Au"]) * float(d["pzc_Au_tilde"]) * au_load
            + float(d["g_Pd"]) * float(d["pzc_Pd_tilde"]) * pd_load
        )

        lu, piv = lu_factor(matrix, check_finite=True)
        affine = lu_solve(
            (lu, piv), np.column_stack([rhs_m, rhs_pzc]), check_finite=True
        )

        self.mode_index = mode_index
        self.rho = rho
        self.gamma = gamma
        self.S = S
        self.matrix = matrix
        self.rhs_m = rhs_m
        self.rhs_pzc = rhs_pzc
        # Affine spectral coefficient vectors.
        self.phi_m = np.asarray(affine[:, 0], dtype=float)
        self.phi_pzc = np.asarray(affine[:, 1], dtype=float)

    def _prepare_surface_fields(self) -> None:
        d = self.derived
        self.surface_x_tilde = _surface_grid_with_boundaries(
            float(d["L_total_tilde"]),
            float(d["L_Au_tilde"]),
            float(d["L_Au_tilde"]) + float(d["d_Au_Pd_tilde"]),
            int(self.params["Nx"]),
        )
        cosine = np.cos(np.outer(self.surface_x_tilde, self.rho))
        self.surface_phi_m = cosine @ self.phi_m
        self.surface_phi_pzc = cosine @ self.phi_pzc

    def _evaluate_affine_components(
        self,
        x_tilde: np.ndarray,
        y_tilde: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_arr, y_arr = np.broadcast_arrays(
            np.asarray(x_tilde, dtype=float), np.asarray(y_tilde, dtype=float)
        )
        basis = np.cos(np.outer(x_arr.ravel(), self.rho))
        basis *= np.exp(-np.outer(y_arr.ravel(), self.gamma))
        phi_m = (basis @ self.phi_m).reshape(x_arr.shape)
        phi_pzc = (basis @ self.phi_pzc).reshape(x_arr.shape)
        return phi_m, phi_pzc

    def _build_boundary_quadrature(self, material: str) -> BoundaryQuadrature:
        d = self.derived
        L_total = float(d["L_total_tilde"])
        target_spacing = L_total / max(int(self.params["Nx"]) - 1, 1)
        if material == "Au":
            a, b = 0.0, float(d["L_Au_tilde"])
        elif material == "Pd":
            a = float(d["L_Au_tilde"]) + float(d["d_Au_Pd_tilde"])
            b = L_total
        else:
            raise KeyError(f"Unknown reactive material: {material}")
        x, weights = _segment_trapezoid_grid(a, b, target_spacing=target_spacing)
        cosine = np.cos(np.outer(x, self.rho))
        return BoundaryQuadrature(
            x_tilde=x,
            weights_tilde=weights,
            phi_m=cosine @ self.phi_m,
            phi_pzc=cosine @ self.phi_pzc,
        )

    def coefficient_field(self, E_mix: float) -> np.ndarray:
        """Return the spectral coefficient vector ``A(E_mix)``."""

        phi_metal_tilde = float(self.derived["beta"]) * float(E_mix)
        return phi_metal_tilde * self.phi_m - self.phi_pzc

    def surface_field(self, E_mix: float) -> np.ndarray:
        """Return the ordered surface samples used for field diagnostics."""

        phi_metal_tilde = float(self.derived["beta"]) * float(E_mix)
        return phi_metal_tilde * self.surface_phi_m - self.surface_phi_pzc

    def boundary_field(self, material: str, E_mix: float) -> np.ndarray:
        data = self.boundary_quadrature[material]
        phi_metal_tilde = float(self.derived["beta"]) * float(E_mix)
        return phi_metal_tilde * data.phi_m - data.phi_pzc

    def boundary_sample_field(self, material: str, E_mix: float) -> np.ndarray:
        """Return ordered boundary samples for charge sign/zero diagnostics."""

        return self.boundary_field(material, E_mix)

    def integrate_boundary_exponential(
        self,
        material: str,
        E_mix: float,
        exponent_coefficient: float,
    ) -> float:
        data = self.boundary_quadrature[material]
        phi = self.boundary_field(material, E_mix)
        values = np.exp(np.clip(exponent_coefficient * phi, -700.0, 700.0))
        return float(np.sum(values * data.weights_tilde))

    def interpolate(
        self,
        E_mix: float,
        x_tilde: np.ndarray,
        y_tilde: np.ndarray,
    ) -> np.ndarray:
        x_arr, y_arr = np.broadcast_arrays(
            np.asarray(x_tilde, dtype=float), np.asarray(y_tilde, dtype=float)
        )
        L_total = float(self.derived["L_total_tilde"])
        tolerance = 128.0 * np.finfo(float).eps * max(1.0, L_total)
        if np.any(y_arr < -tolerance):
            raise ValueError(
                "The flush coplanar electrolyte domain contains only y >= 0"
            )
        if np.any(x_arr < -tolerance) or np.any(x_arr > L_total + tolerance):
            raise ValueError("x_tilde lies outside the spectral strip")
        phi_m, phi_pzc = self._evaluate_affine_components(
            np.clip(x_arr, 0.0, L_total), np.maximum(y_arr, 0.0)
        )
        phi_metal_tilde = float(self.derived["beta"]) * float(E_mix)
        return phi_metal_tilde * phi_m - phi_pzc

    def top_profile(self, E_mix: float, n_x: int = 1600) -> dict[str, np.ndarray]:
        x_tilde = np.linspace(
            0.0, float(self.derived["L_total_tilde"]), int(n_x), dtype=float
        )
        phi_tilde = self.interpolate(E_mix, x_tilde, np.zeros_like(x_tilde))
        return {"x_tilde": x_tilde, "phi_tilde": phi_tilde}

    def upper_grid(
        self,
        E_mix: float,
        n_x: int = 600,
        n_y: int = 320,
        y_max_over_lambda: float = 5.0,
    ) -> dict[str, np.ndarray]:
        key = (float(E_mix), int(n_x), int(n_y), float(y_max_over_lambda))
        if key in self._upper_grid_cache:
            return self._upper_grid_cache[key]
        if int(n_x) < 2 or int(n_y) < 2 or float(y_max_over_lambda) <= 0.0:
            raise ValueError("upper_grid requires n_x,n_y >= 2 and y_max_over_lambda > 0")
        x = np.linspace(0.0, float(self.derived["L_total_tilde"]), int(n_x))
        y = np.linspace(0.0, float(y_max_over_lambda), int(n_y))
        coefficients = self.coefficient_field(float(E_mix))
        cosine = np.cos(np.outer(x, self.rho))
        decay = np.exp(-np.outer(y, self.gamma))
        phi = (decay * coefficients[None, :]) @ cosine.T
        result = {"x_tilde": x, "y_tilde": y, "phi_tilde": phi}
        self._upper_grid_cache[key] = result
        return result

    def affine_residuals(self) -> dict[str, float]:
        rhs_m_res = self.matrix @ self.phi_m - self.rhs_m
        rhs_pzc_res = self.matrix @ self.phi_pzc - self.rhs_pzc

        def relative(residual: np.ndarray, rhs: np.ndarray) -> float:
            numerator = float(np.linalg.norm(residual))
            denominator = float(np.linalg.norm(rhs)) + 1.0e-30
            return numerator / denominator

        return {
            "phi_m_relative_l2": relative(rhs_m_res, self.rhs_m),
            "phi_pzc_relative_l2": relative(rhs_pzc_res, self.rhs_pzc),
        }

    def spectral_metadata(self) -> dict[str, Any]:
        """Return metadata describing the spectral discretization."""

        d = self.derived
        gap_exists = bool(float(d["d_Au_Pd_tilde"]) > 0.0)
        boundary_lengths = {
            "au_top": float(d["L_Au_tilde"]),
            "pd_top": float(d["L_Pd_tilde"]),
        }
        if gap_exists:
            boundary_lengths["substrate_gap"] = float(d["d_Au_Pd_tilde"])
        return {
            "geometry": GEOMETRY_NAME,
            "electrostatic_backend": self.backend_name,
            "electrolyte_domain_id": ELECTROLYTE_DOMAIN_ID,
            "electrolyte_domain_description": ELECTROLYTE_DOMAIN_DESCRIPTION,
            "basis": "cos(rho_n x_tilde) exp(-gamma_n y_tilde)",
            "n_modes": int(self.params["N_modes"]),
            "n_coefficients": int(self.phi_m.size),
            "surface_quadrature_target_points": int(self.params["Nx"]),
            "diagnostic_surface_n_x_coordinates": int(
                self.surface_x_tilde.size
            ),
            "n_y_coordinates": None,
            "farfield_condition": "phi_tilde -> 0 as y_tilde -> infinity",
            "substrate_gap_electrostatic_bc": "homogeneous Neumann",
            "lateral_electrostatic_bc": "homogeneous Neumann (reflection)",
            "boundary_quadrature_points": {
                material: int(data.x_tilde.size)
                for material, data in self.boundary_quadrature.items()
            },
            "topology_checks": {
                "boundary_partition_complete": True,
                "reactive_boundary_interiors_disjoint": True,
                "reactive_boundary_overlap_measure_zero": True,
                "electrolyte_domain_is_semi_infinite_strip": True,
                "substrate_gap_is_boundary": gap_exists,
                "substrate_gap_electrostatic_bc": "homogeneous Neumann",
                "boundary_lengths_tilde": boundary_lengths,
            },
        }

    electrostatics_metadata = spectral_metadata


# New descriptive name while retaining all existing imports and type hints.
ElectrostaticSystem = LinearPBModel


__all__ = [
    "BoundaryQuadrature",
    "ElectrostaticSystem",
    "I_mn",
    "J_n",
    "LinearPBModel",
]
