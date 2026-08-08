"""Numerical engine for the 2 nm Au | support | Pd figure collection.

This module deliberately contains no plotting or output-file code.  It keeps
the dense cosine-coefficient solve from :mod:`Solve_Emix_updating`, but uses
Gauss--Legendre quadrature on the two reactive segments for the FULL mixed-
potential balance.  The quadrature grid is therefore independent of the
surface/2-D grids used by the figure scripts.

The public entry points are :func:`build_case`, :func:`build_cases`,
:func:`build_scan_case`, :func:`build_support_pzc_scan_case`,
:func:`build_support_pzc_scan_family`, and :func:`run_convergence_checks`.
The regular ``build_case`` API deliberately keeps the six publication-case
lengths whitelisted, while the scan APIs accept any finite non-negative
support length for parameter-study work.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOLVER_DIR = ROOT / "Mixed_Potential_Electrical_Double_Layer"
BASE_RESULT_ID = "20260528_111255"
BASE_PARAMS_PATH = SOLVER_DIR / "results" / BASE_RESULT_ID / "params.json"

if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

import Solve_Emix_updating as solver  # noqa: E402


# Physical cases and shared baseline overrides.
L_SUPPORT_NM_VALUES = (0.0, 1.0, 2.0, 3.0, 10.0, 1000.0)
L_AU_M = 2.0e-9
L_PD_M = 2.0e-9
I0_EQUAL_A_PER_M2 = 1.852573885166257e-4
ALPHA_EQUAL = 0.5
OUT_OF_PLANE_WIDTH_M = 0.01
C_H_AU_F_PER_M2 = 0.50
C_H_SUPPORT_F_PER_M2 = 0.20
C_H_PD_F_PER_M2 = 0.50
PZC_AU_SHE_V = 0.51
PZC_PD_SHE_V = 0.21
PZC_PH_SLOPE_V_PER_PH = 0.059126500015747985
PZC_REFERENCE_PH = 7.0
PZC_AU_RHE_V = PZC_AU_SHE_V + PZC_PH_SLOPE_V_PER_PH * PZC_REFERENCE_PH
PZC_PD_RHE_V = PZC_PD_SHE_V + PZC_PH_SLOPE_V_PER_PH * PZC_REFERENCE_PH

# Numerical settings fixed by the requested plan.
SHORT_N_MODES = 960
LONG_N_MODES = 7680
SHORT_CONVERGENCE_N_MODES = 480
LONG_CONVERGENCE_N_MODES = 5760
DEFAULT_GL_ORDER = 128
CONVERGENCE_GL_ORDER = 64
NY_2D = 320
Y_MAX_LAMBDA_D = 5.0

# EDLModel only needs a tiny internal grid because all production integration
# and plotting grids below are explicit and independent of params["Nx"].
COEFFICIENT_GRID_NX = 4
SHORT_ACTIVE_POINTS = 401
LONG_ACTIVE_POINTS = 241
LONG_SUPPORT_EDGE_POINTS = 361
LONG_SUPPORT_MIDDLE_POINTS = 321
LONG_EDGE_LAMBDA_D = 5.0

X_BLOCK_SIZE = 192
MODE_BLOCK_SIZE = 256
Y_BLOCK_SIZE = 8
DECAY_BLOCK_ABS_TOL = 1.0e-13
SURFACE_RECONSTRUCTION_TOL = 5.0e-10
ROOT_XTOL_V = 1.0e-13
RELATIVE_BALANCE_TOL = 1.0e-10


@dataclass(frozen=True)
class SigmaSegment:
    """Surface-charge values for one material segment."""

    material: str
    x_nm: np.ndarray
    phi_rp_V: np.ndarray
    sigma_C_per_m2: np.ndarray

    @property
    def sigma_uC_per_cm2(self) -> np.ndarray:
        return 100.0 * self.sigma_C_per_m2


@dataclass(frozen=True)
class LegacyCase:
    """All scalar and array data needed by the legacy-case plot/export layer."""

    value_nm: float
    params: dict[str, Any]
    derived: dict[str, Any]
    summary: dict[str, Any]
    res_edl: dict[str, Any]
    res_no: dict[str, Any]
    gl_order: int

    x_nm: np.ndarray
    phi_rp_with_V: np.ndarray
    phi_rp_no_V: np.ndarray
    i1_with: np.ndarray
    i2_with: np.ndarray
    i1_no: np.ndarray
    i2_no: np.ndarray
    mask_Au: np.ndarray
    mask_support: np.ndarray
    mask_Pd: np.ndarray
    c_R1_norm: np.ndarray
    c_O2_norm: np.ndarray

    x_2d_nm: np.ndarray
    y_2d_nm: np.ndarray
    phi_s_mV: np.ndarray
    phi_2d_surface_max_error: float
    sigma_segments: tuple[SigmaSegment, ...]

    L_Au_nm: float
    L_support_nm: float
    L_C_nm: float
    L_Pd_nm: float
    L_total_nm: float
    active_zoom_windows_nm: tuple[tuple[float, float], ...]
    convergence: dict[str, Any]


@dataclass(frozen=True)
class ConvergenceResult:
    """Scalar convergence comparison for one support length."""

    L_support_nm: float
    low_N_modes: int
    high_N_modes: int
    low_gl_order: int
    high_gl_order: int
    E_mix_low_modes_V: float
    E_mix_high_modes_V: float
    i_mix_avg_low_modes_A_per_m2: float
    i_mix_avg_high_modes_A_per_m2: float
    mode_delta_E_mV: float
    mode_delta_i_percent: float
    E_mix_gl64_V: float
    E_mix_gl128_V: float
    i_mix_avg_gl64_A_per_m2: float
    i_mix_avg_gl128_A_per_m2: float
    quadrature_delta_E_mV: float
    quadrature_delta_i_percent: float
    relative_balance_high: float
    E_tolerance_mV: float
    i_tolerance_percent: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _GaussData:
    order: int
    x_Au_tilde: np.ndarray
    w_Au_tilde: np.ndarray
    phi_M_Au: np.ndarray
    phi_pzc_Au: np.ndarray
    x_Pd_tilde: np.ndarray
    w_Pd_tilde: np.ndarray
    phi_M_Pd: np.ndarray
    phi_pzc_Pd: np.ndarray


@dataclass(frozen=True)
class _ProfileData:
    x_nm: np.ndarray
    x_tilde: np.ndarray
    phi_tilde: np.ndarray
    phi_rp_V: np.ndarray
    i1_with: np.ndarray
    i2_with: np.ndarray
    i1_no: np.ndarray
    i2_no: np.ndarray
    mask_Au: np.ndarray
    mask_support: np.ndarray
    mask_Pd: np.ndarray
    c_R1_norm: np.ndarray
    c_O2_norm: np.ndarray


def _canonical_support_nm(value_nm: float) -> float:
    value = float(value_nm)
    for allowed in L_SUPPORT_NM_VALUES:
        if math.isclose(value, allowed, rel_tol=0.0, abs_tol=1.0e-9):
            return float(allowed)
    allowed_text = ", ".join(f"{item:g}" for item in L_SUPPORT_NM_VALUES)
    raise ValueError(f"L_support must be one of {allowed_text} nm; got {value_nm!r}")


def _scan_support_nm(value_nm: float) -> float:
    """Validate a continuous OFAT support length without changing case policy."""

    value = float(value_nm)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"OFAT L_support must be finite and non-negative; got {value_nm!r}"
        )
    return value


def resolution_for_case(value_nm: float) -> int:
    """Return the production cosine-mode count for a support length."""

    value = _canonical_support_nm(value_nm)
    return LONG_N_MODES if value == 1000.0 else SHORT_N_MODES


def resolution_for_scan(value_nm: float) -> int:
    """Return the OFAT mode count for an arbitrary validated support length."""

    value = _scan_support_nm(value_nm)
    return LONG_N_MODES if math.isclose(value, 1000.0, rel_tol=0.0, abs_tol=1.0e-9) else SHORT_N_MODES


def load_baseline_params() -> dict[str, Any]:
    """Load the traceable equal-i0/alpha baseline before geometry overrides."""

    with BASE_PARAMS_PATH.open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    params = solver.apply_param_overrides(
        solver.default_params(),
        saved,
        reset_lambda_D=False,
    )
    return params


def _params_for_support(support_nm: float, selected_modes: int) -> dict[str, Any]:
    """Build the shared Au=Pd=2 nm parameter set for a validated support."""

    if selected_modes < 1:
        raise ValueError("n_modes must be positive")

    params = load_baseline_params()
    overrides = {
        "L_Au": L_AU_M,
        "L_gap": support_nm * 1.0e-9,
        "L_Pd_len": L_PD_M,
        "Cdl_Au": C_H_AU_F_PER_M2,
        "Cdl_C": C_H_SUPPORT_F_PER_M2,
        "Cdl_Pd": C_H_PD_F_PER_M2,
        "pzc_Au": PZC_AU_RHE_V,
        "pzc_Pd": PZC_PD_RHE_V,
        "g_Au": None,
        "g_C": None,
        "g_Pd": None,
        "it0_1": I0_EQUAL_A_PER_M2,
        "it0_2": I0_EQUAL_A_PER_M2,
        "alpha1": ALPHA_EQUAL,
        "alpha2": ALPHA_EQUAL,
        "out_of_plane_width": OUT_OF_PLANE_WIDTH_M,
        "N_modes": selected_modes,
        "Nx": COEFFICIENT_GRID_NX,
        "xtol": min(float(params.get("xtol", ROOT_XTOL_V)), ROOT_XTOL_V),
    }
    params = solver.apply_param_overrides(params, overrides, reset_lambda_D=False)
    solver.validate_params(params)
    _validate_fixed_case_params(params, support_nm, selected_modes)
    return params


def params_for_case(value_nm: float, n_modes: int | None = None) -> dict[str, Any]:
    """Build parameters for one of the six formal publication cases."""

    support_nm = _canonical_support_nm(value_nm)
    selected_modes = resolution_for_case(support_nm) if n_modes is None else int(n_modes)
    return _params_for_support(support_nm, selected_modes)


def params_for_support_scan(
    value_nm: float,
    n_modes: int | None = None,
) -> dict[str, Any]:
    """Build parameters for any finite non-negative OFAT support length."""

    support_nm = _scan_support_nm(value_nm)
    selected_modes = resolution_for_scan(support_nm) if n_modes is None else int(n_modes)
    return _params_for_support(support_nm, selected_modes)


def _validate_fixed_case_params(
    params: Mapping[str, Any],
    support_nm: float,
    n_modes: int,
) -> None:
    expected = {
        "L_Au": L_AU_M,
        "L_gap": support_nm * 1.0e-9,
        "L_Pd_len": L_PD_M,
        "Cdl_Au": C_H_AU_F_PER_M2,
        "Cdl_C": C_H_SUPPORT_F_PER_M2,
        "Cdl_Pd": C_H_PD_F_PER_M2,
        "it0_1": I0_EQUAL_A_PER_M2,
        "it0_2": I0_EQUAL_A_PER_M2,
        "alpha1": ALPHA_EQUAL,
        "alpha2": ALPHA_EQUAL,
        "out_of_plane_width": OUT_OF_PLANE_WIDTH_M,
    }
    for key, target in expected.items():
        actual = float(params[key])
        atol = max(1.0e-15, abs(target) * 1.0e-12)
        if not math.isclose(actual, target, rel_tol=0.0, abs_tol=atol):
            raise ValueError(f"{key} should be {target:.15g}, got {actual:.15g}")
    if int(params["N_modes"]) != n_modes:
        raise ValueError(f"N_modes should be {n_modes}, got {params['N_modes']}")
    if not math.isclose(float(params["C_tot"]), 10.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("The selected baseline must remain at C_tot=10 mol/m^3 (10 mM)")


def _build_coefficient_model(params: Mapping[str, Any]) -> Any:
    """Build EDLModel coefficients and release unused dense work matrices."""

    edl = solver.EDLModel(dict(params))
    # S and M are retained by the general-purpose model for diagnostics, but
    # production here only needs the solved affine coefficients.  Releasing
    # them is important for the N=7680 case and does not alter A_M/A_pzc.
    edl.pre.pop("S", None)
    edl.pre.pop("M", None)
    edl.pre.pop("rM", None)
    edl.pre.pop("r_pzc", None)
    gc.collect()
    return edl


def _evaluate_affine_components(edl: Any, x_tilde: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the two affine surface bases with x/mode blocking."""

    x = np.asarray(x_tilde, dtype=float)
    rho = np.asarray(edl.pre["rho"], dtype=float)
    a_m = np.asarray(edl.pre["A_M"], dtype=float)
    a_pzc = np.asarray(edl.pre["A_pzc"], dtype=float)
    out_m = np.zeros(x.size, dtype=float)
    out_pzc = np.zeros(x.size, dtype=float)

    for x_start in range(0, x.size, X_BLOCK_SIZE):
        x_stop = min(x_start + X_BLOCK_SIZE, x.size)
        x_block = x[x_start:x_stop]
        block_m = np.zeros(x_block.size, dtype=float)
        block_pzc = np.zeros(x_block.size, dtype=float)
        for m_start in range(0, rho.size, MODE_BLOCK_SIZE):
            m_stop = min(m_start + MODE_BLOCK_SIZE, rho.size)
            basis = np.cos(np.outer(x_block, rho[m_start:m_stop]))
            block_m += basis @ a_m[m_start:m_stop]
            block_pzc += basis @ a_pzc[m_start:m_stop]
        out_m[x_start:x_stop] = block_m
        out_pzc[x_start:x_stop] = block_pzc
    return out_m, out_pzc


def _evaluate_coefficient_series(
    rho: np.ndarray,
    coefficients: np.ndarray,
    x_tilde: np.ndarray,
) -> np.ndarray:
    """Evaluate one cosine-coefficient vector with bounded work arrays."""

    x = np.asarray(x_tilde, dtype=float)
    modes = np.asarray(rho, dtype=float)
    values = np.asarray(coefficients, dtype=float)
    if modes.shape != values.shape:
        raise ValueError(
            "rho and coefficients must have identical shapes; "
            f"got {modes.shape} and {values.shape}"
        )

    out = np.zeros(x.size, dtype=float)
    for x_start in range(0, x.size, X_BLOCK_SIZE):
        x_stop = min(x_start + X_BLOCK_SIZE, x.size)
        x_block = x[x_start:x_stop]
        block = np.zeros(x_block.size, dtype=float)
        for m_start in range(0, modes.size, MODE_BLOCK_SIZE):
            m_stop = min(m_start + MODE_BLOCK_SIZE, modes.size)
            basis = np.cos(np.outer(x_block, modes[m_start:m_stop]))
            block += basis @ values[m_start:m_stop]
        out[x_start:x_stop] = block
    return out


def _evaluate_surface_series(edl: Any, E_mix: float, x_tilde: np.ndarray) -> np.ndarray:
    phi_m, phi_pzc = _evaluate_affine_components(edl, x_tilde)
    return phi_m * float(edl.derived["beta"]) * float(E_mix) - phi_pzc


def _mapped_legendre_interval(a: float, b: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    half = 0.5 * (b - a)
    center = 0.5 * (a + b)
    return center + half * nodes, half * weights


def _build_gauss_data(edl: Any, order: int) -> _GaussData:
    if int(order) < 2:
        raise ValueError("Gauss-Legendre order must be at least two")
    derived = edl.derived
    x_au, w_au = _mapped_legendre_interval(
        0.0,
        float(derived["L_Au_tilde"]),
        int(order),
    )
    x_pd, w_pd = _mapped_legendre_interval(
        float(derived["L_C_tilde"]),
        float(derived["L_tilde"]),
        int(order),
    )
    combined = np.concatenate((x_au, x_pd))
    phi_m, phi_pzc = _evaluate_affine_components(edl, combined)
    split = x_au.size
    return _GaussData(
        order=int(order),
        x_Au_tilde=x_au,
        w_Au_tilde=w_au,
        phi_M_Au=phi_m[:split],
        phi_pzc_Au=phi_pzc[:split],
        x_Pd_tilde=x_pd,
        w_Pd_tilde=w_pd,
        phi_M_Pd=phi_m[split:],
        phi_pzc_Pd=phi_pzc[split:],
    )


def _gauss_currents(
    E: float,
    gauss: _GaussData,
    edl: Any,
    params: Mapping[str, Any],
) -> dict[str, float]:
    """FULL local-current integrals on separate Au and Pd GL grids."""

    ctx = solver._kinetics_context(float(E), dict(params))
    beta_e = float(edl.derived["beta"]) * float(E)
    phi_au = gauss.phi_M_Au * beta_e - gauss.phi_pzc_Au
    phi_pd = gauss.phi_M_Pd * beta_e - gauss.phi_pzc_Pd
    k_au = float(np.dot(gauss.w_Au_tilde, solver.safe_exp(-ctx["Gamma1"] * phi_au)))
    k_pd = float(np.dot(gauss.w_Pd_tilde, solver.safe_exp(ctx["Gamma2"] * phi_pd)))
    pref1 = float(
        ctx["it0_1"]
        * solver.safe_exp((1.0 - ctx["alpha1"]) * ctx["beta"] * ctx["eta1"])
    )
    pref2 = float(
        -ctx["it0_2"]
        * solver.safe_exp(-ctx["alpha2"] * ctx["beta"] * ctx["eta2"])
    )
    i_au = pref1 * k_au
    i_pd = pref2 * k_pd
    return {
        "I_Au": float(i_au),
        "I_Pd": float(i_pd),
        "residual": float(i_au + i_pd),
        "i_mix": float(abs(i_au)),
        "K_Au": k_au,
        "K_Pd": k_pd,
    }


def _relative_balance(result: Mapping[str, Any]) -> float:
    i_au = float(result["I_Au"])
    i_pd = float(result["I_Pd"])
    return float(abs(i_au + i_pd) / (abs(i_au) + abs(i_pd) + 1.0e-300))


def _solve_with_edl(
    edl: Any,
    params: Mapping[str, Any],
    gauss: _GaussData,
) -> dict[str, Any]:
    rxn = solver._effective_reaction_params(dict(params))

    def residual(e_value: float) -> float:
        return float(_gauss_currents(e_value, gauss, edl, params)["residual"])

    e_mix, info = solver._solve_root_problem(
        f=residual,
        E1_eq=float(rxn["E1_eq_eff"]),
        E2_eq=float(rxn["E2_eq_eff"]),
        xtol=ROOT_XTOL_V,
        max_bracket_expands=int(params.get("max_bracket_expands", 12)),
        info={"mode": "FULL", "quadrature": "Gauss-Legendre", "gl_order": gauss.order},
    )
    if not bool(info.get("converged", False)) or not math.isfinite(e_mix):
        raise RuntimeError(f"with-EDL GL root solve did not converge: {info}")
    currents = _gauss_currents(e_mix, gauss, edl, params)
    info["residual_at_root"] = float(currents["residual"])
    out = solver._build_run_output(
        mode="FULL",
        E_mix=float(e_mix),
        i_mix=float(currents["i_mix"]),
        info=info,
        derived=edl.derived,
        a1=float(edl.pre["a1"]),
        b1=float(edl.pre["b1"]),
        a2=float(edl.pre["a2"]),
        b2=float(edl.pre["b2"]),
    )
    out.update(currents)
    solver._attach_current_unit_outputs(
        out,
        float(edl.derived["lambda_D"]),
        float(params["out_of_plane_width"]),
        float(edl.derived["L_Au"] + edl.derived["L_Pd_len"]),
    )
    out.update(rxn)
    out["quadrature"] = "Gauss-Legendre"
    out["gl_order"] = int(gauss.order)
    out["relative_balance_residual"] = _relative_balance(out)
    if float(out["relative_balance_residual"]) >= RELATIVE_BALANCE_TOL:
        raise RuntimeError(
            "with-EDL absolute-current balance failed: "
            f"relative residual={out['relative_balance_residual']:.3g}"
        )
    return out


def _solve_without_edl(params: Mapping[str, Any]) -> dict[str, Any]:
    out = solver.run_case(dict(params), mode="FULL", return_profiles=False, use_edl=False)
    out["relative_balance_residual"] = _relative_balance(out)
    if float(out["relative_balance_residual"]) >= RELATIVE_BALANCE_TOL:
        raise RuntimeError(
            "w/o-EDL absolute-current balance failed: "
            f"relative residual={out['relative_balance_residual']:.3g}"
        )
    return out


def _unique_sorted(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return array
    return np.unique(np.round(array, decimals=12))


def _short_surface_grid_nm(derived: Mapping[str, Any]) -> np.ndarray:
    l_au = float(derived["L_Au"]) * 1.0e9
    l_support = float(derived["L_gap"]) * 1.0e9
    l_c = float(derived["L_C"]) * 1.0e9
    l_total = float(derived["L_total"]) * 1.0e9
    au = np.linspace(0.0, l_au, SHORT_ACTIVE_POINTS, dtype=float)
    pd = np.linspace(l_c, l_total, SHORT_ACTIVE_POINTS, dtype=float)
    parts: list[np.ndarray] = [au, pd]
    if l_support > 0.0:
        # Approximately 0.01 nm over short supports, while retaining exact
        # material boundaries and a manageable upper bound.
        n_support = min(1601, max(101, int(math.ceil(l_support / 0.01)) + 1))
        parts.append(np.linspace(l_au, l_c, n_support, dtype=float))
    return _unique_sorted(np.concatenate(parts))


def _long_surface_grid_nm(derived: Mapping[str, Any]) -> np.ndarray:
    """Nonuniform 1000 nm grid: dense active/boundary layers, sparse middle."""

    l_au = float(derived["L_Au"]) * 1.0e9
    l_support = float(derived["L_gap"]) * 1.0e9
    l_c = float(derived["L_C"]) * 1.0e9
    l_total = float(derived["L_total"]) * 1.0e9
    lambda_nm = float(derived["lambda_D"]) * 1.0e9
    edge_width = min(LONG_EDGE_LAMBDA_D * lambda_nm, 0.5 * l_support)

    au = np.linspace(0.0, l_au, LONG_ACTIVE_POINTS, dtype=float)
    pd = np.linspace(l_c, l_total, LONG_ACTIVE_POINTS, dtype=float)
    t_edge = np.linspace(0.0, 1.0, LONG_SUPPORT_EDGE_POINTS, dtype=float)
    left_edge = l_au + edge_width * t_edge**1.7
    right_edge = l_c - edge_width * t_edge**1.7
    middle_left = l_au + edge_width
    middle_right = l_c - edge_width
    if middle_right > middle_left:
        middle = np.linspace(
            middle_left,
            middle_right,
            LONG_SUPPORT_MIDDLE_POINTS,
            dtype=float,
        )
    else:
        middle = np.array([middle_left], dtype=float)
    grid = _unique_sorted(np.concatenate((au, left_edge, middle, right_edge, pd)))

    au_count = int(np.count_nonzero(grid <= l_au + 1.0e-10))
    pd_count = int(np.count_nonzero(grid >= l_c - 1.0e-10))
    if au_count < 201 or pd_count < 201:
        raise RuntimeError(
            f"Long-grid active resolution is too low: Au={au_count}, Pd={pd_count}"
        )
    return grid


def surface_grid_nm(derived: Mapping[str, Any]) -> np.ndarray:
    support_nm = float(derived["L_gap"]) * 1.0e9
    if math.isclose(support_nm, 1000.0, rel_tol=0.0, abs_tol=1.0e-6):
        return _long_surface_grid_nm(derived)
    return _short_surface_grid_nm(derived)


def _segment_masks_nm(
    x_nm: np.ndarray,
    l_au_nm: float,
    l_c_nm: float,
    l_total_nm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tol = 1.0e-10
    mask_au = (x_nm >= -tol) & (x_nm <= l_au_nm + tol)
    mask_pd = (x_nm >= l_c_nm - tol) & (x_nm <= l_total_nm + tol)
    if l_c_nm - l_au_nm > tol:
        mask_support = (x_nm >= l_au_nm - tol) & (x_nm <= l_c_nm + tol)
    else:
        mask_support = np.zeros(x_nm.shape, dtype=bool)
    return mask_au, mask_support, mask_pd


def _build_profile_data(
    edl: Any,
    params: Mapping[str, Any],
    res_edl: Mapping[str, Any],
    res_no: Mapping[str, Any],
) -> _ProfileData:
    derived = edl.derived
    x_nm = surface_grid_nm(derived)
    lambda_nm = float(derived["lambda_D"]) * 1.0e9
    x_tilde = x_nm / lambda_nm
    phi_tilde = _evaluate_surface_series(edl, float(res_edl["E_mix"]), x_tilde)
    thermal_v = float(derived["R"]) * float(derived["T"]) / float(derived["F"])
    phi_rp = thermal_v * phi_tilde
    l_au_nm = float(derived["L_Au"]) * 1.0e9
    l_c_nm = float(derived["L_C"]) * 1.0e9
    l_total_nm = float(derived["L_total"]) * 1.0e9
    mask_au, mask_support, mask_pd = _segment_masks_nm(
        x_nm,
        l_au_nm,
        l_c_nm,
        l_total_nm,
    )

    ctx_with = solver._kinetics_context(float(res_edl["E_mix"]), dict(params))
    pref1_with = float(
        ctx_with["it0_1"]
        * solver.safe_exp(
            (1.0 - ctx_with["alpha1"]) * ctx_with["beta"] * ctx_with["eta1"]
        )
    )
    pref2_with = float(
        -ctx_with["it0_2"]
        * solver.safe_exp(-ctx_with["alpha2"] * ctx_with["beta"] * ctx_with["eta2"])
    )
    i1_with = np.zeros(x_nm.shape, dtype=float)
    i2_with = np.zeros(x_nm.shape, dtype=float)
    i1_with[mask_au] = pref1_with * solver.safe_exp(
        -ctx_with["Gamma1"] * phi_tilde[mask_au]
    )
    i2_with[mask_pd] = pref2_with * solver.safe_exp(
        ctx_with["Gamma2"] * phi_tilde[mask_pd]
    )

    ctx_no = solver._kinetics_context(float(res_no["E_mix"]), dict(params))
    pref1_no = float(
        ctx_no["it0_1"]
        * solver.safe_exp((1.0 - ctx_no["alpha1"]) * ctx_no["beta"] * ctx_no["eta1"])
    )
    pref2_no = float(
        -ctx_no["it0_2"]
        * solver.safe_exp(-ctx_no["alpha2"] * ctx_no["beta"] * ctx_no["eta2"])
    )
    i1_no = np.zeros(x_nm.shape, dtype=float)
    i2_no = np.zeros(x_nm.shape, dtype=float)
    i1_no[mask_au] = pref1_no
    i2_no[mask_pd] = pref2_no

    return _ProfileData(
        x_nm=x_nm,
        x_tilde=x_tilde,
        phi_tilde=phi_tilde,
        phi_rp_V=phi_rp,
        i1_with=i1_with,
        i2_with=i2_with,
        i1_no=i1_no,
        i2_no=i2_no,
        mask_Au=mask_au,
        mask_support=mask_support,
        mask_Pd=mask_pd,
        c_R1_norm=np.asarray(
            solver.safe_exp(-float(params["z_R1"]) * phi_tilde),
            dtype=float,
        ),
        c_O2_norm=np.asarray(
            solver.safe_exp(-float(params["z_O2"]) * phi_tilde),
            dtype=float,
        ),
    )


def _y_slices(n_y: int) -> list[slice]:
    slices = [slice(0, 1)]
    for start in range(1, n_y, Y_BLOCK_SIZE):
        slices.append(slice(start, min(start + Y_BLOCK_SIZE, n_y)))
    return slices


def _evaluate_2d_series(
    edl: Any,
    E_mix: float,
    x_tilde: np.ndarray,
    y_tilde: np.ndarray,
) -> np.ndarray:
    """Evaluate phi_tilde(x,y) with bounded mode/x/y work arrays."""

    rho = np.asarray(edl.pre["rho"], dtype=float)
    gamma = np.asarray(edl.pre["gamma"], dtype=float)
    coefficients = (
        np.asarray(edl.pre["A_M"], dtype=float)
        * float(edl.derived["beta"])
        * float(E_mix)
        - np.asarray(edl.pre["A_pzc"], dtype=float)
    )
    x = np.asarray(x_tilde, dtype=float)
    y = np.asarray(y_tilde, dtype=float)
    result = np.zeros((y.size, x.size), dtype=float)
    y_blocks = _y_slices(y.size)

    for x_start in range(0, x.size, X_BLOCK_SIZE):
        x_stop = min(x_start + X_BLOCK_SIZE, x.size)
        x_block = x[x_start:x_stop]
        for m_start in range(0, rho.size, MODE_BLOCK_SIZE):
            m_stop = min(m_start + MODE_BLOCK_SIZE, rho.size)
            rho_block = rho[m_start:m_stop]
            gamma_block = gamma[m_start:m_stop]
            coefficient_block = coefficients[m_start:m_stop]
            cosine = np.cos(np.outer(rho_block, x_block))
            abs_coefficients = np.abs(coefficient_block)
            for y_slice in y_blocks:
                y_block = y[y_slice]
                y_min = float(y_block[0])
                contribution_bound = float(
                    np.dot(abs_coefficients, np.exp(-gamma_block * y_min))
                )
                if y_min > 0.0 and contribution_bound < DECAY_BLOCK_ABS_TOL:
                    continue
                decay = (
                    np.exp(-np.outer(y_block, gamma_block))
                    * coefficient_block[np.newaxis, :]
                )
                result[y_slice, x_start:x_stop] += decay @ cosine
    return result


def _build_2d_data(
    edl: Any,
    res_edl: Mapping[str, Any],
    profile: _ProfileData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    lambda_nm = float(edl.derived["lambda_D"]) * 1.0e9
    y_nm = np.linspace(0.0, Y_MAX_LAMBDA_D * lambda_nm, NY_2D, dtype=float)
    y_tilde = y_nm / lambda_nm
    phi_tilde_2d = _evaluate_2d_series(
        edl,
        float(res_edl["E_mix"]),
        profile.x_tilde,
        y_tilde,
    )
    surface_error = float(np.max(np.abs(phi_tilde_2d[0] - profile.phi_tilde)))
    if surface_error > SURFACE_RECONSTRUCTION_TOL:
        raise RuntimeError(
            "2D y=0 field does not reproduce phi_RP: "
            f"max |delta phi_tilde|={surface_error:.3g}"
        )
    thermal_v = (
        float(edl.derived["R"])
        * float(edl.derived["T"])
        / float(edl.derived["F"])
    )
    phi_s_mV = 1000.0 * thermal_v * phi_tilde_2d
    if not np.all(np.isfinite(phi_s_mV)):
        raise RuntimeError("2D solution potential contains non-finite values")
    return profile.x_nm.copy(), y_nm, phi_s_mV, surface_error


def _build_sigma_segments(
    params: Mapping[str, Any],
    derived: Mapping[str, Any],
    e_mix: float,
    profile: _ProfileData,
) -> tuple[SigmaSegment, ...]:
    specs = (
        ("Au", profile.mask_Au, "Cdl_Au", "pzc_Au"),
        ("support", profile.mask_support, "Cdl_C", "pzc_C"),
        ("Pd", profile.mask_Pd, "Cdl_Pd", "pzc_Pd"),
    )
    segments: list[SigmaSegment] = []
    for material, mask, c_h_key, pzc_key in specs:
        if not np.any(mask):
            continue
        sigma = float(params[c_h_key]) * (
            float(e_mix) - float(params[pzc_key]) - profile.phi_rp_V[mask]
        )
        segments.append(
            SigmaSegment(
                material=material,
                x_nm=profile.x_nm[mask].copy(),
                phi_rp_V=profile.phi_rp_V[mask].copy(),
                sigma_C_per_m2=np.asarray(sigma, dtype=float),
            )
        )
    return tuple(segments)


def _attach_profile_diagnostics(
    result: dict[str, Any],
    params: Mapping[str, Any],
    edl: Any,
    gauss: _GaussData,
    profile: _ProfileData,
) -> None:
    status = solver._compute_dh_status_from_phi(profile.phi_tilde, dict(params))
    result.update(status)
    e_mix = float(result["E_mix"])
    beta_e = float(edl.derived["beta"]) * e_mix
    phi_au = gauss.phi_M_Au * beta_e - gauss.phi_pzc_Au
    phi_pd = gauss.phi_M_Pd * beta_e - gauss.phi_pzc_Pd
    thermal_v = (
        float(edl.derived["R"])
        * float(edl.derived["T"])
        / float(edl.derived["F"])
    )
    result["phi2_1_meanV"] = float(
        thermal_v
        * np.dot(gauss.w_Au_tilde, phi_au)
        / float(edl.derived["L_Au_tilde"])
    )
    result["phi2_2_meanV"] = float(
        thermal_v
        * np.dot(gauss.w_Pd_tilde, phi_pd)
        / (float(edl.derived["L_tilde"]) - float(edl.derived["L_C_tilde"]))
    )
    solver._handle_dh_violation(result, dict(params), mode="FULL", use_edl=True)


def _case_summary(
    value_nm: float,
    params: Mapping[str, Any],
    res_edl: Mapping[str, Any],
    res_no: Mapping[str, Any],
) -> dict[str, Any]:
    comparison = solver._make_edl_comparison_metrics(dict(res_edl), dict(res_no), mode="FULL")
    return {
        "model": "Au|C|Pd linear-PB piecewise-Robin",
        "L_Au_nm": 2.0,
        "L_support_nm": float(value_nm),
        "L_Pd_nm": 2.0,
        "C_H_Au_uF_per_cm2": 100.0 * float(params["Cdl_Au"]),
        "C_H_support_uF_per_cm2": 100.0 * float(params["Cdl_C"]),
        "C_H_Pd_uF_per_cm2": 100.0 * float(params["Cdl_Pd"]),
        "N_modes": int(params["N_modes"]),
        "gl_order": int(res_edl["gl_order"]),
        "E_mix_with": float(res_edl["E_mix"]),
        "E_mix_no": float(res_no["E_mix"]),
        "i_mix_abs_with": float(res_edl["i_mix_abs_A"]),
        "i_mix_abs_no": float(res_no["i_mix_abs_A"]),
        "i_mix_avg_with": float(res_edl["i_mix_avg_A_per_m2"]),
        "i_mix_avg_no": float(res_no["i_mix_avg_A_per_m2"]),
        "delta_E_mix": float(comparison["delta_E_mix"]),
        "ratio_i_mix_avg": float(comparison["ratio_i_mix_avg"]),
        "relative_balance_with": float(res_edl["relative_balance_residual"]),
        "relative_balance_no": float(res_no["relative_balance_residual"]),
        "max_abs_phi_tilde_with_edl": float(res_edl["max_abs_phi_tilde"]),
        "debye_huckel_ok_with_edl": bool(res_edl["debye_huckel_ok"]),
    }


def _active_windows_nm(derived: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    support_nm = float(derived["L_gap"]) * 1.0e9
    if not math.isclose(support_nm, 1000.0, rel_tol=0.0, abs_tol=1.0e-6):
        return ()
    lambda_nm = float(derived["lambda_D"]) * 1.0e9
    l_au_nm = float(derived["L_Au"]) * 1.0e9
    l_c_nm = float(derived["L_C"]) * 1.0e9
    l_total_nm = float(derived["L_total"]) * 1.0e9
    return (
        (0.0, min(l_total_nm, l_au_nm + 5.0 * lambda_nm)),
        (max(0.0, l_c_nm - 5.0 * lambda_nm), l_total_nm),
    )


def _build_case_from_existing_edl(
    support_nm: float,
    params: dict[str, Any],
    edl: Any,
    *,
    gl_order: int = DEFAULT_GL_ORDER,
    include_2d: bool = True,
    res_no_cached: Mapping[str, Any] | None = None,
) -> LegacyCase:
    """Assemble one case from an already-factorized coefficient model."""

    derived = copy.deepcopy(edl.derived)
    gauss = _build_gauss_data(edl, int(gl_order))
    res_edl = _solve_with_edl(edl, params, gauss)
    res_no = (
        _solve_without_edl(params)
        if res_no_cached is None
        else copy.deepcopy(dict(res_no_cached))
    )
    profile = _build_profile_data(edl, params, res_edl, res_no)
    _attach_profile_diagnostics(res_edl, params, edl, gauss, profile)

    if include_2d:
        x_2d_nm, y_2d_nm, phi_s_mV, surface_error = _build_2d_data(
            edl,
            res_edl,
            profile,
        )
    else:
        x_2d_nm = np.empty(0, dtype=float)
        y_2d_nm = np.empty(0, dtype=float)
        phi_s_mV = np.empty((0, 0), dtype=float)
        surface_error = float("nan")

    sigma_segments = _build_sigma_segments(
        params,
        derived,
        float(res_edl["E_mix"]),
        profile,
    )
    summary = _case_summary(support_nm, params, res_edl, res_no)
    l_au_nm = float(derived["L_Au"]) * 1.0e9
    l_c_nm = float(derived["L_C"]) * 1.0e9
    l_pd_nm = float(derived["L_Pd_len"]) * 1.0e9
    l_total_nm = float(derived["L_total"]) * 1.0e9
    numerical_metadata = {
        "N_modes": int(params["N_modes"]),
        "gl_order": int(gl_order),
        "root_xtol_V": ROOT_XTOL_V,
        "relative_balance_with": float(res_edl["relative_balance_residual"]),
        "relative_balance_no": float(res_no["relative_balance_residual"]),
        "profile_grid_kind": "nonuniform-boundary-layer" if support_nm == 1000.0 else "piecewise-uniform",
        "profile_grid_points": int(profile.x_nm.size),
        "Ny_2d": int(y_2d_nm.size),
        "phi_2d_surface_max_error": surface_error,
    }

    return LegacyCase(
        value_nm=support_nm,
        params=copy.deepcopy(params),
        derived=derived,
        summary=summary,
        res_edl=res_edl,
        res_no=res_no,
        gl_order=int(gl_order),
        x_nm=profile.x_nm,
        phi_rp_with_V=profile.phi_rp_V,
        phi_rp_no_V=np.zeros(profile.x_nm.shape, dtype=float),
        i1_with=profile.i1_with,
        i2_with=profile.i2_with,
        i1_no=profile.i1_no,
        i2_no=profile.i2_no,
        mask_Au=profile.mask_Au,
        mask_support=profile.mask_support,
        mask_Pd=profile.mask_Pd,
        c_R1_norm=profile.c_R1_norm,
        c_O2_norm=profile.c_O2_norm,
        x_2d_nm=x_2d_nm,
        y_2d_nm=y_2d_nm,
        phi_s_mV=phi_s_mV,
        phi_2d_surface_max_error=surface_error,
        sigma_segments=sigma_segments,
        L_Au_nm=l_au_nm,
        L_support_nm=support_nm,
        L_C_nm=l_c_nm,
        L_Pd_nm=l_pd_nm,
        L_total_nm=l_total_nm,
        active_zoom_windows_nm=_active_windows_nm(derived),
        convergence=numerical_metadata,
    )


def _build_case_from_params(
    support_nm: float,
    params: dict[str, Any],
    *,
    gl_order: int = DEFAULT_GL_ORDER,
    include_2d: bool = True,
) -> LegacyCase:
    """Compute a validated support case without creating files or figures."""

    edl = _build_coefficient_model(params)
    try:
        return _build_case_from_existing_edl(
            support_nm,
            params,
            edl,
            gl_order=gl_order,
            include_2d=include_2d,
        )
    finally:
        del edl
        gc.collect()


def build_case(
    value_nm: float,
    *,
    n_modes: int | None = None,
    gl_order: int = DEFAULT_GL_ORDER,
    include_2d: bool = True,
) -> LegacyCase:
    """Compute one whitelisted legacy publication case."""

    support_nm = _canonical_support_nm(value_nm)
    params = params_for_case(support_nm, n_modes=n_modes)
    return _build_case_from_params(
        support_nm,
        params,
        gl_order=gl_order,
        include_2d=include_2d,
    )


def build_scan_case(
    value_nm: float,
    *,
    n_modes: int | None = None,
    gl_order: int = DEFAULT_GL_ORDER,
    include_2d: bool = False,
) -> LegacyCase:
    """Compute an arbitrary finite non-negative support length for OFAT use.

    Unlike :func:`build_case`, this API does not add the supplied length to the
    formal six-case collection.  Its default deliberately skips the 2-D field
    because the support-length OFAT only needs scalar and surface diagnostics.
    """

    support_nm = _scan_support_nm(value_nm)
    params = params_for_support_scan(support_nm, n_modes=n_modes)
    return _build_case_from_params(
        support_nm,
        params,
        gl_order=gl_order,
        include_2d=include_2d,
    )


def build_support_pzc_scan_case(
    L_support_nm: float,
    pzc_C_V: float,
    *,
    n_modes: int | None = None,
    gl_order: int = DEFAULT_GL_ORDER,
    include_2d: bool = False,
) -> LegacyCase:
    """Compute one support-length/PZC point for a parameter study.

    The geometry and all locked Au=Pd=2 nm baseline parameters come from
    :func:`params_for_support_scan`; only the support PZC is overridden.  The
    default skips the 2-D field because PZC scans use scalar and surface
    diagnostics.
    """

    support_nm = _scan_support_nm(L_support_nm)
    pzc_c = float(pzc_C_V)
    if not math.isfinite(pzc_c):
        raise ValueError(f"pzc_C_V must be finite; got {pzc_C_V!r}")

    params = params_for_support_scan(support_nm, n_modes=n_modes)
    params = solver.apply_param_overrides(
        params,
        {"pzc_C": pzc_c},
        reset_lambda_D=False,
    )
    solver.validate_params(params)
    return _build_case_from_params(
        support_nm,
        params,
        gl_order=gl_order,
        include_2d=include_2d,
    )


def build_support_pzc_scan_family(
    L_support_nm: float,
    pzc_C_values_V: Sequence[float],
    *,
    n_modes: int | None = None,
    gl_orders: Sequence[int] = (DEFAULT_GL_ORDER,),
    include_2d: bool = False,
) -> dict[tuple[float, int], LegacyCase]:
    """Build a support-PZC family with one dense EDL coefficient solve.

    For the locked Au=Pd=2 nm geometry, the Au and Pd Robin coefficients are
    mirror symmetric.  The matrix ``M``, the applied-potential response
    ``A_M``, and all spatial/quadrature grids are therefore independent of
    ``pzc_C``.  Reflection in the cosine basis maps coefficient ``n`` to
    ``(-1)**n`` times itself, which isolates the centered-support response from
    one reference solution.  Each requested PZC then needs only an affine
    coefficient update and a FULL absolute-current root solve.

    The returned mapping is keyed by ``(pzc_C_V, gl_order)``.  It is deliberately
    separate from :func:`build_support_pzc_scan_case`, which remains the simple
    independent single-point reference path.
    """

    support_nm = _scan_support_nm(L_support_nm)
    pzc_values: list[float] = []
    for raw_value in pzc_C_values_V:
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"Every pzc_C value must be finite; got {raw_value!r}")
        if value not in pzc_values:
            pzc_values.append(value)

    orders: list[int] = []
    for raw_order in gl_orders:
        order_float = float(raw_order)
        order = int(raw_order)
        if (
            not math.isfinite(order_float)
            or order_float != float(order)
            or order < 2
        ):
            raise ValueError(
                "Every Gauss-Legendre order must be a finite integer >= 2; "
                f"got {raw_order!r}"
            )
        if order not in orders:
            orders.append(order)
    if not orders:
        raise ValueError("gl_orders must contain at least one order")
    if not pzc_values:
        return {}

    params_reference = params_for_support_scan(support_nm, n_modes=n_modes)
    pzc_reference = float(params_reference["pzc_C"])
    edl = _build_coefficient_model(params_reference)
    try:
        derived = edl.derived
        l_au = float(derived["L_Au_tilde"])
        l_pd = float(derived["L_tilde"] - derived["L_C_tilde"])
        g_au = float(derived["g_Au"])
        g_pd = float(derived["g_Pd"])
        if not math.isclose(l_au, l_pd, rel_tol=1.0e-12, abs_tol=1.0e-14):
            raise ValueError(
                "Support-PZC family reuse requires mirror-symmetric Au/Pd "
                f"lengths; got {l_au:.15g} and {l_pd:.15g}"
            )
        if not math.isclose(g_au, g_pd, rel_tol=1.0e-12, abs_tol=1.0e-14):
            raise ValueError(
                "Support-PZC family reuse requires mirror-symmetric Au/Pd "
                f"Robin coefficients; got {g_au:.15g} and {g_pd:.15g}"
            )

        pzc_active_mean = 0.5 * (
            float(params_reference["pzc_Au"])
            + float(params_reference["pzc_Pd"])
        )
        pzc_denominator = pzc_active_mean - pzc_reference
        if abs(pzc_denominator) <= 1.0e-12:
            raise ValueError(
                "Mirror/parity support-response isolation is singular when "
                "pzc_C equals the mean Au/Pd PZC"
            )

        a_m = np.asarray(edl.pre["A_M"], dtype=float)
        a_pzc_reference = np.asarray(edl.pre["A_pzc"], dtype=float).copy()
        parity = np.where(np.arange(a_m.size) % 2 == 0, 1.0, -1.0)
        symmetric_a_pzc = 0.5 * (
            a_pzc_reference + parity * a_pzc_reference
        )
        beta = float(derived["beta"])
        d_a_pzc_d_v = (
            beta * pzc_active_mean * a_m - symmetric_a_pzc
        ) / pzc_denominator
        if math.isclose(support_nm, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
            # A zero-width segment has exactly zero source measure.  Enforce
            # that mathematical negative control rather than retaining roundoff
            # from subtracting two nearly equal symmetric coefficient vectors.
            d_a_pzc_d_v = np.zeros_like(d_a_pzc_d_v)

        a_m_reflection_error = float(
            np.max(np.abs(a_m - parity * a_m))
            / (np.max(np.abs(a_m)) + 1.0e-300)
        )
        support_response_reflection_error = float(
            np.max(np.abs(d_a_pzc_d_v - parity * d_a_pzc_d_v))
            / (np.max(np.abs(d_a_pzc_d_v)) + 1.0e-300)
        ) if np.any(d_a_pzc_d_v) else 0.0
        if a_m_reflection_error > 5.0e-10:
            raise RuntimeError(
                "Applied-potential coefficient response is not mirror "
                f"symmetric: relative error={a_m_reflection_error:.3g}"
            )
        if support_response_reflection_error > 5.0e-10:
            raise RuntimeError(
                "Isolated support-PZC response is not mirror symmetric: "
                f"relative error={support_response_reflection_error:.3g}"
            )

        rho = np.asarray(edl.pre["rho"], dtype=float)
        x_internal = np.asarray(edl.pre["x_tilde"], dtype=float)
        phi_pzc_reference_internal = np.asarray(
            edl.pre["phi_tilde_pzc"], dtype=float
        ).copy()
        d_phi_pzc_internal_d_v = _evaluate_coefficient_series(
            rho,
            d_a_pzc_d_v,
            x_internal,
        )
        c_au = np.asarray(edl.pre["c_Au"], dtype=float)
        c_pd = np.asarray(edl.pre["c_Pd"], dtype=float)
        thermal_v = 1.0 / beta
        base_segs = tuple(edl.pre["segs"])
        res_no_reference = _solve_without_edl(params_reference)
        cases: dict[tuple[float, int], LegacyCase] = {}

        for pzc_c in pzc_values:
            params = solver.apply_param_overrides(
                params_reference,
                {"pzc_C": pzc_c},
                reset_lambda_D=False,
            )
            solver.validate_params(params)
            a_pzc = (
                a_pzc_reference
                + (pzc_c - pzc_reference) * d_a_pzc_d_v
            )

            edl.params = copy.deepcopy(params)
            edl.derived["pzc_C"] = pzc_c
            edl.derived["pzc_C_tilde"] = beta * pzc_c
            edl.pre["A_pzc"] = a_pzc
            edl.pre["b1"] = -thermal_v * float(np.dot(c_au, a_pzc))
            edl.pre["b2"] = -thermal_v * float(np.dot(c_pd, a_pzc))
            edl.pre["phi_tilde_pzc"] = (
                phi_pzc_reference_internal
                + (pzc_c - pzc_reference) * d_phi_pzc_internal_d_v
            )
            edl.pre["segs"] = [
                (
                    name,
                    a,
                    b,
                    gseg,
                    beta * pzc_c if name == "C" else pzc_tilde,
                )
                for name, a, b, gseg, pzc_tilde in base_segs
            ]

            for order in orders:
                cases[(pzc_c, order)] = _build_case_from_existing_edl(
                    support_nm,
                    params,
                    edl,
                    gl_order=order,
                    include_2d=include_2d,
                    res_no_cached=res_no_reference,
                )
        return cases
    finally:
        del edl
        gc.collect()


def build_cases(
    values_nm: Sequence[float] = L_SUPPORT_NM_VALUES,
    *,
    include_2d: bool = True,
) -> list[LegacyCase]:
    """Build all requested support-length cases in the supplied order."""

    return [build_case(value, include_2d=include_2d) for value in values_nm]


def _scalar_solution(
    value_nm: float,
    n_modes: int,
    gl_orders: Sequence[int],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    params = params_for_case(value_nm, n_modes=n_modes)
    edl = _build_coefficient_model(params)
    results: dict[int, dict[str, Any]] = {}
    for order in gl_orders:
        gauss = _build_gauss_data(edl, int(order))
        results[int(order)] = _solve_with_edl(edl, params, gauss)
    derived = copy.deepcopy(edl.derived)
    del edl
    gc.collect()
    return results, derived


def convergence_check(value_nm: float) -> ConvergenceResult:
    """Run the requested mode and GL64->128 checks for one case."""

    support_nm = _canonical_support_nm(value_nm)
    is_long = support_nm == 1000.0
    low_modes = LONG_CONVERGENCE_N_MODES if is_long else SHORT_CONVERGENCE_N_MODES
    high_modes = LONG_N_MODES if is_long else SHORT_N_MODES
    low_results, _ = _scalar_solution(support_nm, low_modes, (DEFAULT_GL_ORDER,))
    high_results, _ = _scalar_solution(
        support_nm,
        high_modes,
        (CONVERGENCE_GL_ORDER, DEFAULT_GL_ORDER),
    )
    low = low_results[DEFAULT_GL_ORDER]
    high64 = high_results[CONVERGENCE_GL_ORDER]
    high128 = high_results[DEFAULT_GL_ORDER]

    def relative_percent(a: float, b: float) -> float:
        return float(100.0 * abs(a - b) / max(abs(b), 1.0e-300))

    mode_delta_e_mv = 1000.0 * abs(float(high128["E_mix"]) - float(low["E_mix"]))
    mode_delta_i_pct = relative_percent(
        float(high128["i_mix_avg_A_per_m2"]),
        float(low["i_mix_avg_A_per_m2"]),
    )
    gl_delta_e_mv = 1000.0 * abs(float(high128["E_mix"]) - float(high64["E_mix"]))
    gl_delta_i_pct = relative_percent(
        float(high128["i_mix_avg_A_per_m2"]),
        float(high64["i_mix_avg_A_per_m2"]),
    )
    e_tolerance_mv = 0.03 if is_long else 0.002
    i_tolerance_percent = 0.1 if is_long else 0.005
    balance = float(high128["relative_balance_residual"])
    passed = bool(
        mode_delta_e_mv < e_tolerance_mv
        and mode_delta_i_pct < i_tolerance_percent
        and gl_delta_e_mv < e_tolerance_mv
        and gl_delta_i_pct < i_tolerance_percent
        and balance < RELATIVE_BALANCE_TOL
    )
    return ConvergenceResult(
        L_support_nm=support_nm,
        low_N_modes=low_modes,
        high_N_modes=high_modes,
        low_gl_order=CONVERGENCE_GL_ORDER,
        high_gl_order=DEFAULT_GL_ORDER,
        E_mix_low_modes_V=float(low["E_mix"]),
        E_mix_high_modes_V=float(high128["E_mix"]),
        i_mix_avg_low_modes_A_per_m2=float(low["i_mix_avg_A_per_m2"]),
        i_mix_avg_high_modes_A_per_m2=float(high128["i_mix_avg_A_per_m2"]),
        mode_delta_E_mV=mode_delta_e_mv,
        mode_delta_i_percent=mode_delta_i_pct,
        E_mix_gl64_V=float(high64["E_mix"]),
        E_mix_gl128_V=float(high128["E_mix"]),
        i_mix_avg_gl64_A_per_m2=float(high64["i_mix_avg_A_per_m2"]),
        i_mix_avg_gl128_A_per_m2=float(high128["i_mix_avg_A_per_m2"]),
        quadrature_delta_E_mV=gl_delta_e_mv,
        quadrature_delta_i_percent=gl_delta_i_pct,
        relative_balance_high=balance,
        E_tolerance_mV=e_tolerance_mv,
        i_tolerance_percent=i_tolerance_percent,
        passed=passed,
    )


def run_convergence_checks(
    values_nm: Sequence[float] = L_SUPPORT_NM_VALUES,
    *,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Run and optionally enforce all plan-specified numerical checks."""

    results = [convergence_check(value) for value in values_nm]
    failures = [result for result in results if not result.passed]
    if strict and failures:
        details = "; ".join(
            f"C={item.L_support_nm:g} nm: "
            f"mode dE={item.mode_delta_E_mV:.4g} mV, "
            f"mode di={item.mode_delta_i_percent:.4g}%, "
            f"GL dE={item.quadrature_delta_E_mV:.4g} mV, "
            f"GL di={item.quadrature_delta_i_percent:.4g}%"
            for item in failures
        )
        raise RuntimeError(f"Numerical convergence thresholds were not met: {details}")
    return [result.to_dict() for result in results]


def profile_columns(case: LegacyCase) -> dict[str, np.ndarray]:
    """Return aligned one-dimensional columns for CSV/export helpers."""

    return {
        "x_nm": case.x_nm,
        "phi_RP_with_V": case.phi_rp_with_V,
        "phi_RP_no_V": case.phi_rp_no_V,
        "i1_with_A_per_m2": case.i1_with,
        "i2_with_A_per_m2": case.i2_with,
        "i1_no_A_per_m2": case.i1_no,
        "i2_no_A_per_m2": case.i2_no,
        "mask_Au": case.mask_Au,
        "mask_support": case.mask_support,
        "mask_Pd": case.mask_Pd,
        "c_R1_over_bulk": case.c_R1_norm,
        "c_O2_over_bulk": case.c_O2_norm,
    }


__all__ = [
    "BASE_RESULT_ID",
    "BASE_PARAMS_PATH",
    "L_SUPPORT_NM_VALUES",
    "SHORT_N_MODES",
    "LONG_N_MODES",
    "DEFAULT_GL_ORDER",
    "NY_2D",
    "SigmaSegment",
    "LegacyCase",
    "ConvergenceResult",
    "load_baseline_params",
    "params_for_case",
    "params_for_support_scan",
    "resolution_for_case",
    "resolution_for_scan",
    "surface_grid_nm",
    "build_case",
    "build_scan_case",
    "build_support_pzc_scan_case",
    "build_support_pzc_scan_family",
    "build_cases",
    "convergence_check",
    "run_convergence_checks",
    "profile_columns",
]
