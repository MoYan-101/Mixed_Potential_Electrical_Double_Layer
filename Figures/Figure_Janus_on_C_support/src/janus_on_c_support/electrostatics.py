"""Four-segment cosine-Galerkin linear Poisson--Boltzmann model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
import scipy.linalg as la

from .parameters import canonical_params, compute_derived_params


@dataclass(frozen=True)
class Segment:
    """One constant-property interval of the mirror half-cell."""

    name: str
    material: str
    x_start_m: float
    x_end_m: float
    C_H_F_per_m2: float
    g: float
    pzc_V: float
    faradaic: bool

    @property
    def x0_m(self) -> float:
        return self.x_start_m

    @property
    def x1_m(self) -> float:
        return self.x_end_m

    @property
    def length_m(self) -> float:
        return self.x_end_m - self.x_start_m

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "material": self.material,
            "x_start_m": float(self.x_start_m),
            "x_end_m": float(self.x_end_m),
            "length_m": float(self.length_m),
            "C_H_F_per_m2": float(self.C_H_F_per_m2),
            "g": float(self.g),
            "pzc_V": float(self.pzc_V),
            "faradaic": bool(self.faradaic),
        }


def _build_segments(
    params: Mapping[str, Any], derived: Mapping[str, Any]
) -> tuple[Segment, ...]:
    edges = [float(value) for value in derived["edges_m"]]
    definitions = (
        ("C_left", "C", False),
        ("Au", "Au", True),
        ("Pd", "Pd", True),
        ("C_right", "C", False),
    )
    return tuple(
        Segment(
            name=name,
            material=material,
            x_start_m=edges[index],
            x_end_m=edges[index + 1],
            C_H_F_per_m2=float(
                derived[f"C_H_{material}_effective_F_per_m2"]
            ),
            g=float(derived[f"g_{material}"]),
            pzc_V=float(params[f"pzc_{material}"]),
            faradaic=faradaic,
        )
        for index, (name, material, faradaic) in enumerate(definitions)
    )


class LinearPBModel:
    r"""Linear-PB solution for ``C|Au|Pd|C`` with mirrored side boundaries.

    The dimensionless solution is

    .. math::

       \tilde\phi(x,y)=\sum_n A_n\cos(\rho_n x)e^{-\gamma_n y},

    where ``rho_n=n*pi/L`` and ``gamma_n=sqrt(1+rho_n**2)``.
    The cosine basis enforces homogeneous Neumann conditions at both lateral
    boundaries, while every vertical mode decays to zero in the bulk.  A
    piecewise Stern--Robin condition on all four surface segments determines
    the coefficients by Galerkin projection.
    """

    def __init__(self, params: Mapping[str, Any] | None = None):
        self.params = canonical_params(params)
        self.derived = compute_derived_params(self.params)
        self.segments = _build_segments(self.params, self.derived)
        self.N_modes = int(self.params["N_modes"])
        self.rho = np.arange(self.N_modes + 1, dtype=float) * math.pi / float(
            self.derived["L_tilde"]
        )
        self.gamma = np.sqrt(1.0 + self.rho**2)
        self._assemble_affine_coefficients()

    @property
    def rho_tilde(self) -> np.ndarray:
        return self.rho

    @property
    def A_M(self) -> np.ndarray:
        return self._A_M

    @property
    def A_pzc(self) -> np.ndarray:
        return self._A_pzc

    def _piecewise_integrals(self, *, include_pzc: bool) -> np.ndarray:
        """Return ``int f(x) cos(k*pi*x/L) dx`` through order ``2N``."""

        count = 2 * self.N_modes + 1
        result = np.zeros(count, dtype=float)
        k = np.arange(1, count, dtype=float)
        rho_k = k * math.pi / float(self.derived["L_tilde"])
        beta = float(self.derived["beta_per_V"])
        lambda_D = float(self.derived["lambda_D_m"])
        for segment in self.segments:
            a = segment.x_start_m / lambda_D
            b = segment.x_end_m / lambda_D
            if b <= a:
                continue
            value = segment.g
            if include_pzc:
                value *= beta * segment.pzc_V
            result[0] += value * (b - a)
            result[1:] += value * (
                np.sin(rho_k * b) - np.sin(rho_k * a)
            ) / rho_k
        return result

    def _assemble_affine_coefficients(self) -> None:
        """Assemble ``(gamma + S) A = r`` and solve two affine RHSs."""

        nmax = self.N_modes
        size = nmax + 1
        length = float(self.derived["L_tilde"])
        g_integrals = self._piecewise_integrals(include_pzc=False)
        gpzc_integrals = self._piecewise_integrals(include_pzc=True)

        # Allocate Fortran-contiguous storage so LAPACK can factor it in place.
        matrix = np.empty((size, size), dtype=float, order="F")
        matrix[0, :] = g_integrals[:size] / length
        if nmax:
            n = np.arange(1, size, dtype=int)
            for m in range(1, size):
                matrix[m, 0] = 2.0 * g_integrals[m] / length
                matrix[m, 1:] = (
                    g_integrals[np.abs(m - n)] + g_integrals[m + n]
                ) / length
        diagonal = np.diag_indices(size)
        matrix[diagonal] += self.gamma

        rhs_m = np.empty(size, dtype=float)
        rhs_pzc = np.empty(size, dtype=float)
        rhs_m[0] = g_integrals[0] / length
        rhs_pzc[0] = gpzc_integrals[0] / length
        rhs_m[1:] = 2.0 * g_integrals[1:size] / length
        rhs_pzc[1:] = 2.0 * gpzc_integrals[1:size] / length
        rhs = np.column_stack((rhs_m, rhs_pzc))

        lu, piv = la.lu_factor(matrix, overwrite_a=True, check_finite=False)
        solved = la.lu_solve((lu, piv), rhs, check_finite=False)
        self._A_M = np.asarray(solved[:, 0], dtype=float)
        self._A_pzc = np.asarray(solved[:, 1], dtype=float)
        self._r_M = rhs_m
        self._r_pzc = rhs_pzc

    def surface_grid_m(self, nx: int | None = None) -> np.ndarray:
        """Return a surface grid with all four material edges inserted."""

        count = int(self.params["Nx"] if nx is None else nx)
        if count < 2:
            raise ValueError("nx must be at least two")
        total = float(self.derived["L_total_m"])
        grid = np.concatenate(
            (
                np.linspace(0.0, total, count, dtype=float),
                np.asarray(self.derived["edges_m"], dtype=float),
            )
        )
        return np.unique(grid)

    def _validated_x(self, x_m: np.ndarray | float) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x_m, dtype=float))
        total = float(self.derived["L_total_m"])
        tolerance = max(1.0e-18, total * 1.0e-12)
        if not np.all(np.isfinite(x)):
            raise ValueError("x_m must contain only finite values")
        if np.any(x < -tolerance) or np.any(x > total + tolerance):
            raise ValueError(f"x_m must lie within [0, {total:.12g}] m")
        return np.clip(x, 0.0, total)

    @staticmethod
    def _validated_y(y_m: np.ndarray | float) -> np.ndarray:
        y = np.atleast_1d(np.asarray(y_m, dtype=float))
        if not np.all(np.isfinite(y)) or np.any(y < 0.0):
            raise ValueError("y_m must contain only finite non-negative values")
        return y

    def coefficients(self, E_V: float) -> np.ndarray:
        """Return ``A_n`` at one shared metal potential."""

        value = float(E_V)
        if not math.isfinite(value):
            raise ValueError("E_V must be finite")
        return self._A_M * float(self.derived["beta_per_V"]) * value - self._A_pzc

    def affine_surface_components(
        self, x_m: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the two surface fields in ``A=A_M*beta*E-A_pzc``."""

        x = self._validated_x(x_m)
        x_tilde = x / float(self.derived["lambda_D_m"])
        return (
            self._evaluate_surface_series(self._A_M, x_tilde),
            self._evaluate_surface_series(self._A_pzc, x_tilde),
        )

    def _evaluate_surface_series(
        self, coefficients: np.ndarray, x_tilde: np.ndarray
    ) -> np.ndarray:
        output = np.empty(x_tilde.size, dtype=float)
        block_size = int(self.params["evaluation_block_size"])
        for start in range(0, x_tilde.size, block_size):
            stop = min(start + block_size, x_tilde.size)
            basis = np.cos(np.outer(x_tilde[start:stop], self.rho))
            output[start:stop] = basis @ coefficients
        return output

    def phi_tilde_surface(
        self, E_V: float, x_m: np.ndarray | float | None = None
    ) -> np.ndarray:
        """Evaluate dimensionless reaction-plane potential at ``y=0``."""

        x = self.surface_grid_m() if x_m is None else self._validated_x(x_m)
        components_m, components_pzc = self.affine_surface_components(x)
        return (
            components_m * float(self.derived["beta_per_V"]) * float(E_V)
            - components_pzc
        )

    def phi_tilde(
        self,
        E_V: float,
        x_m: np.ndarray | float,
        y_m: np.ndarray | float,
    ) -> np.ndarray:
        """Evaluate the 2-D field; 1-D ``x``/``y`` produce ``(Ny, Nx)``."""

        x = self._validated_x(x_m)
        y = self._validated_y(y_m)
        x_tilde = x / float(self.derived["lambda_D_m"])
        y_tilde = y / float(self.derived["lambda_D_m"])
        coefficients = self.coefficients(E_V)
        result = np.empty((y.size, x.size), dtype=float)
        x_block_size = int(self.params["evaluation_block_size"])
        mode_block_size = int(self.params["mode_block_size"])
        for x_start in range(0, x.size, x_block_size):
            x_stop = min(x_start + x_block_size, x.size)
            block = np.zeros((y.size, x_stop - x_start), dtype=float)
            for mode_start in range(0, coefficients.size, mode_block_size):
                mode_stop = min(mode_start + mode_block_size, coefficients.size)
                decay = np.exp(
                    -np.outer(y_tilde, self.gamma[mode_start:mode_stop])
                ) * coefficients[None, mode_start:mode_stop]
                cosine = np.cos(
                    np.outer(
                        self.rho[mode_start:mode_stop],
                        x_tilde[x_start:x_stop],
                    )
                )
                block += decay @ cosine
            result[:, x_start:x_stop] = block
        return result

    def phi_rp_V(
        self,
        E_V: float,
        x_m: np.ndarray | float | None = None,
        y_m: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """Return solution potential in volts at the RP or over a 2-D grid."""

        thermal = float(self.derived["thermal_voltage_V"])
        if y_m is None:
            return thermal * self.phi_tilde_surface(E_V, x_m)
        if x_m is None:
            x_m = self.surface_grid_m()
        return thermal * self.phi_tilde(E_V, x_m, y_m)

    def material_profile(self, x_m: np.ndarray | float) -> np.ndarray:
        """Return right-continuous material identities on positive intervals.

        Zero-width segments never own a grid point.  In particular, when
        ``L_C_right=0`` the domain endpoint belongs to the final positive
        segment (Pd), rather than to the collapsed C interval.
        """

        x = self._validated_x(x_m)
        positive = tuple(segment for segment in self.segments if segment.length_m > 0.0)
        if not positive:
            raise RuntimeError("Geometry has no positive-length segment")
        right_edges = np.asarray([segment.x_end_m for segment in positive], dtype=float)
        indices = np.searchsorted(right_edges, x, side="right")
        indices = np.clip(indices, 0, len(positive) - 1)
        materials = np.asarray([segment.material for segment in positive])
        return materials[indices]

    def segment_mask(self, x_m: np.ndarray | float, name: str) -> np.ndarray:
        """Return a half-open interval mask (the final interval is closed)."""

        x = self._validated_x(x_m)
        matching = [segment for segment in self.segments if segment.name == name]
        if len(matching) != 1:
            raise KeyError(f"Unknown segment name: {name}")
        segment = matching[0]
        if segment.length_m == 0.0:
            return np.zeros(x.shape, dtype=bool)
        positive = tuple(item for item in self.segments if item.length_m > 0.0)
        if segment is positive[-1]:
            return (x >= segment.x_start_m) & (x <= segment.x_end_m)
        return (x >= segment.x_start_m) & (x < segment.x_end_m)

    def surface_charge_C_per_m2(
        self,
        E_V: float,
        x_m: np.ndarray | float | None = None,
        *,
        material: str | None = None,
    ) -> np.ndarray:
        """Evaluate ``sigma=C_H*(E-PZC-phi_RP)`` at the surface."""

        x = self.surface_grid_m() if x_m is None else self._validated_x(x_m)
        phi_rp = self.phi_rp_V(E_V, x)
        if material is not None:
            if material not in {"Au", "C", "Pd"}:
                raise ValueError("material must be 'Au', 'C', or 'Pd'")
            capacitance = float(
                self.derived[f"C_H_{material}_effective_F_per_m2"]
            )
            pzc = float(self.params[f"pzc_{material}"])
            return capacitance * (float(E_V) - pzc - phi_rp)

        identities = self.material_profile(x)
        capacitance = np.empty(x.size, dtype=float)
        pzc = np.empty(x.size, dtype=float)
        for identity in ("C", "Au", "Pd"):
            mask = identities == identity
            capacitance[mask] = float(
                self.derived[f"C_H_{identity}_effective_F_per_m2"]
            )
            pzc[mask] = float(self.params[f"pzc_{identity}"])
        return capacitance * (float(E_V) - pzc - phi_rp)

    def derivative_x_tilde(
        self,
        E_V: float,
        x_m: np.ndarray | float,
        y_m: float = 0.0,
    ) -> np.ndarray:
        """Return ``d(phi_tilde)/d(x_tilde)`` at one height."""

        x = self._validated_x(x_m)
        y = float(y_m)
        if not math.isfinite(y) or y < 0.0:
            raise ValueError("y_m must be finite and non-negative")
        x_tilde = x / float(self.derived["lambda_D_m"])
        y_tilde = y / float(self.derived["lambda_D_m"])
        coefficients = -self.rho * self.coefficients(E_V) * np.exp(
            -self.gamma * y_tilde
        )
        output = np.empty(x.size, dtype=float)
        block_size = int(self.params["evaluation_block_size"])
        for start in range(0, x.size, block_size):
            stop = min(start + block_size, x.size)
            output[start:stop] = np.sin(
                np.outer(x_tilde[start:stop], self.rho)
            ) @ coefficients
        return output

    def derivative_y_tilde_surface(
        self, E_V: float, x_m: np.ndarray | float
    ) -> np.ndarray:
        """Return ``d(phi_tilde)/d(y_tilde)`` at ``y=0``."""

        x = self._validated_x(x_m)
        x_tilde = x / float(self.derived["lambda_D_m"])
        return self._evaluate_surface_series(
            -self.gamma * self.coefficients(E_V), x_tilde
        )

    def robin_residual(
        self,
        E_V: float,
        x_m: np.ndarray | float,
        *,
        material: str | None = None,
    ) -> np.ndarray:
        """Pointwise truncated-series Stern--Robin boundary residual."""

        x = self._validated_x(x_m)
        phi = self.phi_tilde_surface(E_V, x)
        derivative = self.derivative_y_tilde_surface(E_V, x)
        beta = float(self.derived["beta_per_V"])
        if material is not None:
            if material not in {"Au", "C", "Pd"}:
                raise ValueError("material must be 'Au', 'C', or 'Pd'")
            g = float(self.derived[f"g_{material}"])
            pzc_tilde = beta * float(self.params[f"pzc_{material}"])
            return derivative + g * (beta * float(E_V) - phi - pzc_tilde)

        identities = self.material_profile(x)
        g_values = np.empty(x.size, dtype=float)
        pzc_values = np.empty(x.size, dtype=float)
        for identity in ("C", "Au", "Pd"):
            mask = identities == identity
            g_values[mask] = float(self.derived[f"g_{identity}"])
            pzc_values[mask] = beta * float(self.params[f"pzc_{identity}"])
        return derivative + g_values * (beta * float(E_V) - phi - pzc_values)

    def spectral_coefficients_json(self, E_V: float) -> dict[str, list[float]]:
        """Return the traceable spectral solution using JSON-native lists."""

        return {
            "mode_n": list(range(self.N_modes + 1)),
            "rho_tilde": [float(value) for value in self.rho],
            "gamma": [float(value) for value in self.gamma],
            "A_M": [float(value) for value in self._A_M],
            "A_pzc": [float(value) for value in self._A_pzc],
            "A_at_E_mix": [float(value) for value in self.coefficients(E_V)],
        }
