"""
EDL ↔ mixed potential ↔ kinetics self-consistent solver (PDF-faithful)
======================================================================

Implements the equations in:
- Derivation_260115_for me.pdf (equation numbers like S1-xx, S3-xx, D5-xx)
- Mixed potential _Potential distribution_251230.pdf (main narrative; repeats key relations)

Requirements:
    pip install numpy scipy pandas matplotlib

Outputs (auto-created):
    ./results/<timestamp>/
        results_summary.csv
        sensitivities.csv
        profiles.npz
        figures/*.png
        ofat_<param>.csv
        heatmap_*.csv

Core assumptions (match the PDFs):
- Debye–Hückel / linearized Poisson–Boltzmann (Eq. (S1-4))
- Linear charging boundary condition (Eq. (S-12a))
- Irreversible Frumkin-corrected Butler–Volmer kinetics (Eqs. (S3-4),(S3-5) or (S3-8),(S3-9))
- Mixed potential defined by net-current = 0 (Eq. (S3-7b))

If you later switch to nonlinear PB or potential-dependent C_dl, the affine EDL decomposition
(Eqs. (D1-4)–(D3-2)) no longer holds (see D4 "Scope and limitations").
"""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
except Exception:
    go = None

import scipy.linalg as la
from scipy.optimize import root_scalar


# -----------------------------
# Utilities
# -----------------------------

def safe_exp(x: np.ndarray | float, clip: float = 700.0) -> np.ndarray | float:
    """Safe exp() with clipping to avoid overflow."""
    return np.exp(np.clip(x, -clip, clip))


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def J_n(n: int, a: float, b: float, rho_n: float) -> float:
    """Eq. (S1-26): J_n(a,b) = ∫_a^b cos(rho_n x̃) dx̃"""
    if n == 0:
        return b - a
    return (math.sin(rho_n * b) - math.sin(rho_n * a)) / rho_n


def I_mn(m: int, n: int, a: float, b: float, rho_m: float, rho_n: float) -> float:
    """Eq. (S1-27): I_mn(a,b) = ∫_a^b cos(rho_m x̃) cos(rho_n x̃) dx̃"""
    if m == n:
        rm = rho_m
        return 0.5 * (b - a) + (math.sin(2 * rm * b) - math.sin(2 * rm * a)) / (4 * rm)
    denom1 = rho_m - rho_n
    denom2 = rho_m + rho_n
    term1 = (math.sin(denom1 * b) - math.sin(denom1 * a)) / denom1
    term2 = (math.sin(denom2 * b) - math.sin(denom2 * a)) / denom2
    return 0.5 * (term1 + term2)


def make_edges(vals: np.ndarray, scale: str) -> np.ndarray:
    """Build bin edges from centers for pcolormesh heatmaps."""
    v = np.asarray(vals, dtype=float)
    if v.ndim != 1 or len(v) < 2:
        raise ValueError("make_edges needs a 1D array of length >= 2")
    if scale == "log":
        if np.any(v <= 0):
            raise ValueError("log-scale edges require positive values")
        lv = np.log(v)
        edges = np.empty(len(v) + 1, dtype=float)
        edges[1:-1] = 0.5 * (lv[:-1] + lv[1:])
        edges[0] = lv[0] - (edges[1] - lv[0])
        edges[-1] = lv[-1] + (lv[-1] - edges[-2])
        return np.exp(edges)
    edges = np.empty(len(v) + 1, dtype=float)
    edges[1:-1] = 0.5 * (v[:-1] + v[1:])
    edges[0] = v[0] - (edges[1] - v[0])
    edges[-1] = v[-1] + (v[-1] - edges[-2])
    return edges


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a (possibly nested) dict for CSV saving."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        kk = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_dict(v, prefix=kk + "."))
        else:
            out[kk] = v
    return out


HOVER_PARAM_KEYS = [
    "C_tot", "lambda_D", "epsilon_r", "T",
    "L_Au", "L_gap", "L_Pd_len",
    "Cdl_Au", "Cdl_C", "Cdl_Pd",
    "pzc_Au", "pzc_C", "pzc_Pd",
    "it0_1", "it0_2", "alpha1", "alpha2",
    "z_R1", "z_O2", "E1_eq", "E2_eq",
]

_PLOTLY_WARNED = False


def format_hover_params(params: Dict[str, Any], keys: Optional[List[str]] = None) -> str:
    """Format parameter dict for Plotly hover text."""
    flat = flatten_dict(params)
    if keys is None:
        keys = sorted(flat.keys())
    lines: List[str] = []
    for key in keys:
        if key not in flat:
            continue
        val = flat[key]
        if isinstance(val, float):
            lines.append(f"{key}={val:.6g}")
        else:
            lines.append(f"{key}={val}")
    return "<br>".join(lines)


def _maybe_warn_plotly() -> None:
    global _PLOTLY_WARNED
    if go is None and not _PLOTLY_WARNED:
        print("Plotly not installed; skipping interactive HTML. Install with: pip install plotly")
        _PLOTLY_WARNED = True


def write_plotly_html(fig, path: Path) -> None:
    if go is None:
        _maybe_warn_plotly()
        return
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)


# -----------------------------
# Default parameters (edit here)
# -----------------------------

def default_params() -> Dict[str, Any]:
    """
    Parameter dictionary `params` (requested by the user).

    Units:
    - Lengths: meters
    - Potentials: volts (E_mix, E_eq, pzc must share the same reference)
    - C_tot: mol/m^3
    - Capacitance Cdl: F/m^2 (or set dimensionless g_* directly)
    - Exchange current densities it0_*: A/m^2

    Dimensionless (PDF definitions):
    - g_i = (λ_D/ε_s) C_dl,i (Eq. (S-11c))
    - x̃ = x/λ_D, L̃ = L/λ_D (Eq. (S1-3))
    - φ̃ = (F/RT)(φ - φ_b) (Eq. (S1-1))
    """
    return dict(
        # constants
        R=8.314,
        F=96485.0,
        T=298.0,

        # permittivity
        epsilon0=8.8541878128e-12,  # F/m (TODO check PDF table if needed)
        epsilon_r=78.5,
        epsilon_s=None,             # override if desired

        # electrolyte
        C_tot=100.0,                # mol/m^3
        lambda_D=None,              # optional override

        # geometry
        L_Au=11e-9,                 # m
        L_gap=10e-9,                # m
        L_Pd_len=37e-9,             # m

        # interfacial electrostatics (Cdl or g)
        Cdl_Au=120e-6,              # F/m^2
        Cdl_C=0.2,                  # F/m^2
        Cdl_Pd=37.7e-6,             # F/m^2
        g_Au=None, g_C=None, g_Pd=None,  # if set, overrides Cdl_* via Eq. (S-11c)

        # pzc (V)
        pzc_Au=0.513,
        pzc_C=0.361,
        pzc_Pd=0.371,

        # kinetics
        it0_1=8.85e-5,              # A/m^2
        it0_2=3.878e-4,             # A/m^2
        alpha1=0.3,
        alpha2=0.37,
        z_R1=-1.0,
        z_O2=1.0,
        E1_eq=0.0,                  # V
        E2_eq=0.834,                # V

        # numerics
        N_modes=80,
        Nx=1200,
        xtol=1e-10,
        max_bracket_expands=12,

        # switches
        use_edl=True,                       # True=with EDL, False=no EDL
        use_affine_phi2=True,              # Eq. (D5-3)
        use_closed_form_when_affine=True,  # Eq. (D5-6)/(D5-8)

        # what to run
        do_ofat=True,
        do_heatmaps=True,
        do_sensitivities=True,

        # scan sizes
        ofat_n=15,
        heatmap_nx=25,
        heatmap_ny=25,
        scan_mode="MEAN",  # "MEAN" or "BOTH"
    )


# -----------------------------
# EDL model: cosine expansion + matrix solve
# -----------------------------

def compute_derived_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Derived quantities used by both EDL and no-EDL paths."""
    p = params
    R_gas = float(p["R"]); F = float(p["F"]); T = float(p["T"])
    beta = F / (R_gas * T)

    # epsilon_s
    if p.get("epsilon_s") is not None:
        eps_s = float(p["epsilon_s"])
    else:
        eps_s = float(p["epsilon_r"]) * float(p["epsilon0"])

    # lambda_D (Eq. (S1-2))
    if p.get("lambda_D") is not None:
        lambda_D = float(p["lambda_D"])
    else:
        C_tot = float(p["C_tot"])
        lambda_D = math.sqrt(eps_s * R_gas * T / (2.0 * F**2 * C_tot))

    # geometry (m)
    L_Au = float(p["L_Au"]); L_gap = float(p["L_gap"]); L_Pd_len = float(p["L_Pd_len"])
    if not (L_Au > 0 and L_gap > 0 and L_Pd_len > 0):
        raise ValueError("Geometry lengths must be positive")
    L_C = L_Au + L_gap
    L_total = L_C + L_Pd_len

    # dimensionless lengths (Eq. (S1-3))
    L_tilde = L_total / lambda_D
    L_Au_tilde = L_Au / lambda_D
    L_C_tilde = L_C / lambda_D

    # g_i = (lambda_D/epsilon_s) Cdl_i (Eq. (S-11c)), unless overridden
    def g_from_Cdl(Cdl: float) -> float:
        return (lambda_D / eps_s) * Cdl

    g_Au = p.get("g_Au"); g_C = p.get("g_C"); g_Pd = p.get("g_Pd")
    if g_Au is None: g_Au = g_from_Cdl(float(p["Cdl_Au"]))
    if g_C is None:  g_C  = g_from_Cdl(float(p["Cdl_C"]))
    if g_Pd is None: g_Pd = g_from_Cdl(float(p["Cdl_Pd"]))

    # pzc (V) -> phi_pzc_tilde (Eq. (S1-1), phi_b=0)
    pzc_Au = float(p["pzc_Au"]); pzc_C = float(p["pzc_C"]); pzc_Pd = float(p["pzc_Pd"])
    pzc_Au_tilde = beta * pzc_Au
    pzc_C_tilde = beta * pzc_C
    pzc_Pd_tilde = beta * pzc_Pd

    return dict(
        R=R_gas, F=F, T=T, beta=beta,
        epsilon_s=eps_s, lambda_D=lambda_D,
        L_Au=L_Au, L_gap=L_gap, L_Pd_len=L_Pd_len,
        L_C=L_C, L_total=L_total,
        L_tilde=L_tilde, L_Au_tilde=L_Au_tilde, L_C_tilde=L_C_tilde,
        g_Au=g_Au, g_C=g_C, g_Pd=g_Pd,
        pzc_Au=pzc_Au, pzc_C=pzc_C, pzc_Pd=pzc_Pd,
        pzc_Au_tilde=pzc_Au_tilde, pzc_C_tilde=pzc_C_tilde, pzc_Pd_tilde=pzc_Pd_tilde,
    )

class EDLModel:
    """
    Linear EDL model (Debye–Hückel) with lateral heterogeneity.

    Governing PDE:
        ∂²φ̃_s/∂x̃² + ∂²φ̃_s/∂ỹ² = φ̃_s                      (Eq. (S1-4))

    BCs:
        ∂φ̃_s/∂x̃|_{x̃=0} = 0,  ∂φ̃_s/∂x̃|_{x̃=L̃} = 0          (Eq. (S1-5))
        φ̃_s(x̃,ỹ→∞) = 0                                      (Eq. (S1-6))
        ∂φ̃_s/∂ỹ|_{ỹ=0} = -g_i(φ̃_M - φ̃_s(x̃,0) - φ̃_pzc,i)   (Eq. (S-12a))
        with piecewise constants g_i, φ̃_pzc,i on Au/C/Pd       (Eq. (S-12b))

    Cosine expansion:
        φ̃_s(x̃,ỹ) = Σ A_n cos(ρ_n x̃) exp(-γ_n ỹ)            (Eq. (S1-13))
        ρ_n = nπ/L̃, γ_n = √(1+ρ_n²)                           (Eq. (S1-14))
        φ̃_s(x̃,0) = Σ A_n cos(ρ_n x̃)                          (Eq. (S1-15))

    Coefficients from matrix system:
        [γ + S]A = R                                           (Eq. (S1-24))
        with explicit S and R elements (Eqs. (S1-28)–(S1-32)).

    Affine reuse:
        A(φ̃_M) = A_M φ̃_M - A_pzc                              (Eq. (D3-1)),
        enabling fast updates during outer mixed-potential iterations.
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = copy.deepcopy(params)
        self.derived: Dict[str, Any] = {}
        self.pre: Dict[str, Any] = {}
        self._build()

    def _compute_derived(self) -> None:
        p = self.params
        R_gas = float(p["R"]); F = float(p["F"]); T = float(p["T"])
        beta = F / (R_gas * T)

        # ε_s
        if p.get("epsilon_s") is not None:
            eps_s = float(p["epsilon_s"])
        else:
            eps_s = float(p["epsilon_r"]) * float(p["epsilon0"])

        # λ_D (Eq. (S1-2))
        if p.get("lambda_D") is not None:
            lambda_D = float(p["lambda_D"])
        else:
            C_tot = float(p["C_tot"])
            lambda_D = math.sqrt(eps_s * R_gas * T / (2.0 * F**2 * C_tot))

        # geometry (m)
        L_Au = float(p["L_Au"]); L_gap = float(p["L_gap"]); L_Pd_len = float(p["L_Pd_len"])
        if not (L_Au > 0 and L_gap > 0 and L_Pd_len > 0):
            raise ValueError("Geometry lengths must be positive")
        L_C = L_Au + L_gap
        L_total = L_C + L_Pd_len

        # dimensionless lengths (Eq. (S1-3))
        L_tilde = L_total / lambda_D
        L_Au_tilde = L_Au / lambda_D
        L_C_tilde = L_C / lambda_D

        # g_i = (λ_D/ε_s) Cdl_i (Eq. (S-11c)), unless overridden
        def g_from_Cdl(Cdl: float) -> float:
            return (lambda_D / eps_s) * Cdl

        g_Au = p.get("g_Au"); g_C = p.get("g_C"); g_Pd = p.get("g_Pd")
        if g_Au is None: g_Au = g_from_Cdl(float(p["Cdl_Au"]))
        if g_C is None:  g_C  = g_from_Cdl(float(p["Cdl_C"]))
        if g_Pd is None: g_Pd = g_from_Cdl(float(p["Cdl_Pd"]))

        # pzc (V) -> φ̃_pzc (Eq. (S1-1), φ_b=0)
        pzc_Au = float(p["pzc_Au"]); pzc_C = float(p["pzc_C"]); pzc_Pd = float(p["pzc_Pd"])
        pzc_Au_tilde = beta * pzc_Au
        pzc_C_tilde  = beta * pzc_C
        pzc_Pd_tilde = beta * pzc_Pd

        self.derived = dict(
            R=R_gas, F=F, T=T, beta=beta,
            epsilon_s=eps_s, lambda_D=lambda_D,
            L_Au=L_Au, L_gap=L_gap, L_Pd_len=L_Pd_len,
            L_C=L_C, L_total=L_total,
            L_tilde=L_tilde, L_Au_tilde=L_Au_tilde, L_C_tilde=L_C_tilde,
            g_Au=g_Au, g_C=g_C, g_Pd=g_Pd,
            pzc_Au=pzc_Au, pzc_C=pzc_C, pzc_Pd=pzc_Pd,
            pzc_Au_tilde=pzc_Au_tilde, pzc_C_tilde=pzc_C_tilde, pzc_Pd_tilde=pzc_Pd_tilde,
        )

    def _build(self) -> None:
        self._compute_derived()
        p = self.params; d = self.derived

        N = int(p["N_modes"])
        Nx = int(p["Nx"])

        L = d["L_tilde"]
        L_Au = d["L_Au_tilde"]
        L_C = d["L_C_tilde"]

        n = np.arange(N + 1, dtype=float)
        rho = n * math.pi / L                 # Eq. (S1-14)
        gamma = np.sqrt(1.0 + rho**2)         # Eq. (S1-14)

        segs = [
            ("Au", 0.0,  L_Au, d["g_Au"], d["pzc_Au_tilde"]),
            ("C",  L_Au, L_C,  d["g_C"],  d["pzc_C_tilde"]),
            ("Pd", L_C,  L,    d["g_Pd"], d["pzc_Pd_tilde"]),
        ]

        # Build S matrix (Eqs. (S1-28)–(S1-31))
        S = np.zeros((N + 1, N + 1), dtype=float)

        for nn in range(N + 1):
            acc = 0.0
            for _, a, b, gseg, _ in segs:
                acc += gseg * J_n(nn, a, b, float(rho[nn]))
            S[0, nn] = acc / L  # Eq. (S1-28)

        for mm in range(1, N + 1):
            acc0 = 0.0
            for _, a, b, gseg, _ in segs:
                acc0 += gseg * J_n(mm, a, b, float(rho[mm]))
            S[mm, 0] = 2.0 * acc0 / L  # Eq. (S1-30)

            for nn in range(1, N + 1):
                acc = 0.0
                for _, a, b, gseg, _ in segs:
                    acc += gseg * I_mn(mm, nn, a, b, float(rho[mm]), float(rho[nn]))
                S[mm, nn] = 2.0 * acc / L  # Eq. (S1-31)

        # Matrix system: [γ + S]A = R (Eq. (S1-24))
        M = np.diag(gamma) + S
        lu, piv = la.lu_factor(M)

        # R = rM φ̃_M - r_pzc, using R0 (S1-29) and Rm (S1-32); see also Eq. (D1-4)–(D1-5)
        rM = np.zeros(N + 1, dtype=float)
        r_pzc = np.zeros(N + 1, dtype=float)

        accM0 = 0.0; accp0 = 0.0
        for _, a, b, gseg, pzc_t in segs:
            dx = b - a
            accM0 += gseg * dx
            accp0 += gseg * pzc_t * dx
        rM[0] = accM0 / L
        r_pzc[0] = accp0 / L

        for mm in range(1, N + 1):
            accM = 0.0; accp = 0.0
            for _, a, b, gseg, pzc_t in segs:
                Jm = J_n(mm, a, b, float(rho[mm]))
                accM += gseg * Jm
                accp += gseg * pzc_t * Jm
            rM[mm] = 2.0 * accM / L
            r_pzc[mm] = 2.0 * accp / L

        # Affine decomposition A = A_M φ̃_M - A_pzc (Eq. (D3-1))
        A_M = la.lu_solve((lu, piv), rM)
        A_pzc = la.lu_solve((lu, piv), r_pzc)

        # Segment-average weights (Eqs. (S4-6) and (S4-9))
        c_Au = np.zeros(N + 1, dtype=float)
        c_Pd = np.zeros(N + 1, dtype=float)
        c_Au[0] = 1.0; c_Pd[0] = 1.0
        L_Pd = L - L_C
        for nn in range(1, N + 1):
            c_Au[nn] = math.sin(float(rho[nn]) * L_Au) / (L_Au * float(rho[nn]))
            c_Pd[nn] = -math.sin(float(rho[nn]) * L_C) / (L_Pd * float(rho[nn]))

        # Affine φ2,i = a_i E + b_i (Eq. (D5-3))
        a1 = float(np.dot(c_Au, A_M))
        a2 = float(np.dot(c_Pd, A_M))
        b1 = -(d["R"] * d["T"] / d["F"]) * float(np.dot(c_Au, A_pzc))
        b2 = -(d["R"] * d["T"] / d["F"]) * float(np.dot(c_Pd, A_pzc))

        # Precompute surface basis φ̃_M(x̃) and φ̃_pzc(x̃) for FULL mode (Eq. (S1-15) + affine (D3-1))
        x = np.linspace(0.0, L, Nx)
        cos_mat = np.cos(np.outer(x, rho))
        phi_tilde_M = cos_mat @ A_M
        phi_tilde_pzc = cos_mat @ A_pzc

        self.pre = dict(
            N=N, Nx=Nx,
            rho=rho, gamma=gamma,
            segs=segs,
            S=S, M=M,
            A_M=A_M, A_pzc=A_pzc,
            c_Au=c_Au, c_Pd=c_Pd,
            a1=a1, a2=a2, b1=b1, b2=b2,
            x_tilde=x, phi_tilde_M=phi_tilde_M, phi_tilde_pzc=phi_tilde_pzc,
        )

    def phi_tilde_surface(self, E_mix: float) -> Tuple[np.ndarray, np.ndarray]:
        """φ̃_s(x̃,0) using affine decomposition (Eq. (S1-15) + Eq. (D3-1))."""
        beta = self.derived["beta"]
        phiM_tilde = beta * E_mix
        x = self.pre["x_tilde"]
        phi = self.pre["phi_tilde_M"] * phiM_tilde - self.pre["phi_tilde_pzc"]
        return x, phi

    def segment_mean_phi2(self, E_mix: float, use_affine_phi2: bool) -> Tuple[float, float]:
        """Segment-mean φ2 (V). Either affine (D5-3) or direct (S3-16)."""
        if use_affine_phi2:
            return self.pre["a1"] * E_mix + self.pre["b1"], self.pre["a2"] * E_mix + self.pre["b2"]
        Au_t, Pd_t = self.segment_mean_phi_tilde(E_mix)
        scale = self.derived["R"] * self.derived["T"] / self.derived["F"]
        return scale * Au_t, scale * Pd_t

    def segment_mean_phi_tilde(self, E_mix: float) -> Tuple[float, float]:
        """Segment-mean φ̃ (Eq. (S4-5))."""
        beta = self.derived["beta"]
        phiM_tilde = beta * E_mix
        A = self.pre["A_M"] * phiM_tilde - self.pre["A_pzc"]
        return float(np.dot(self.pre["c_Au"], A)), float(np.dot(self.pre["c_Pd"], A))


# -----------------------------
# Kinetics and mixed potential
# -----------------------------

def currents_mean_field(E: float, phi2_1: float, phi2_2: float, params: Dict[str, Any]) -> Tuple[float, float]:
    """Mean-field irreversible Frumkin-BV currents, Eqs. (S3-4) & (S3-5)."""
    R_gas = float(params["R"]); F = float(params["F"]); T = float(params["T"])
    beta = F / (R_gas * T)

    it0_1 = float(params["it0_1"]); it0_2 = float(params["it0_2"])
    alpha1 = float(params["alpha1"]); alpha2 = float(params["alpha2"])
    z_R1 = float(params["z_R1"]); z_O2 = float(params["z_O2"])
    E1_eq = float(params["E1_eq"]); E2_eq = float(params["E2_eq"])

    eta1 = E - E1_eq  # Eq. (S3-6)
    eta2 = E - E2_eq

    Gamma1 = (1.0 - alpha1) + z_R1  # Eq. (S3-4)
    Gamma2 = alpha2 - z_O2          # Eq. (S3-5)

    log_i1 = math.log(it0_1) + (1.0 - alpha1) * beta * eta1 - Gamma1 * beta * phi2_1
    log_i2 = math.log(it0_2) - alpha2 * beta * eta2 + Gamma2 * beta * phi2_2

    i1 = float(safe_exp(log_i1))
    i2 = -float(safe_exp(log_i2))
    return i1, i2


def full_mode_currents(E: float, edl: EDLModel, params: Dict[str, Any], return_profiles: bool) -> Dict[str, Any]:
    """
    FULL mode:
    - local i1(x̃), i2(x̃): Eqs. (S3-8),(S3-9)
    - K integrals: Eqs. (S3-11),(S3-12)
    - net current: Eq. (S3-7b)
    """
    R_gas = float(params["R"]); F = float(params["F"]); T = float(params["T"])
    beta = F / (R_gas * T)

    it0_1 = float(params["it0_1"]); it0_2 = float(params["it0_2"])
    alpha1 = float(params["alpha1"]); alpha2 = float(params["alpha2"])
    z_R1 = float(params["z_R1"]); z_O2 = float(params["z_O2"])
    E1_eq = float(params["E1_eq"]); E2_eq = float(params["E2_eq"])

    eta1 = E - E1_eq
    eta2 = E - E2_eq

    Gamma1 = (1.0 - alpha1) + z_R1
    Gamma2 = alpha2 - z_O2

    x, phi_tilde = edl.phi_tilde_surface(E)

    L_Au = edl.derived["L_Au_tilde"]
    L_C = edl.derived["L_C_tilde"]
    L = edl.derived["L_tilde"]

    mask_Au = (x >= 0.0) & (x <= L_Au + 1e-12)
    mask_Pd = (x >= L_C - 1e-12) & (x <= L + 1e-12)

    K_Au = float(np.trapezoid(safe_exp(-Gamma1 * phi_tilde[mask_Au]), x[mask_Au]))  # Eq. (S3-12)
    K_Pd = float(np.trapezoid(safe_exp(Gamma2 * phi_tilde[mask_Pd]), x[mask_Pd]))   # Eq. (S3-11)

    # Note: integration is over d x̃ as in the PDFs. If you need per-depth current (A/m),
    # multiply these by λ_D (m): I_per_depth = λ_D * I_here. (TODO check your exact experimental mapping.)
    I_Au = float(it0_1 * safe_exp((1.0 - alpha1) * beta * eta1) * K_Au)
    I_Pd = float(-it0_2 * safe_exp(-alpha2 * beta * eta2) * K_Pd)

    residual = I_Au + I_Pd  # Eq. (S3-7b)
    i_mix = abs(I_Au)       # definition used here: |∫_Au i1 d x̃| (at Emix equals |∫_Pd i2 d x̃|)

    out: Dict[str, Any] = dict(I_Au=I_Au, I_Pd=I_Pd, residual=residual, i_mix=i_mix, K_Au=K_Au, K_Pd=K_Pd)

    if return_profiles:
        i1 = np.zeros_like(x)
        i2 = np.zeros_like(x)
        pref1 = it0_1 * safe_exp((1.0 - alpha1) * beta * eta1)
        pref2 = -it0_2 * safe_exp(-alpha2 * beta * eta2)
        i1[mask_Au] = pref1 * safe_exp(-Gamma1 * phi_tilde[mask_Au])
        i2[mask_Pd] = pref2 * safe_exp(Gamma2 * phi_tilde[mask_Pd])
        out.update(dict(x_tilde=x, phi_tilde=phi_tilde, i1=i1, i2=i2, mask_Au=mask_Au, mask_Pd=mask_Pd))
    return out


def full_mode_currents_no_edl(
    E: float,
    derived: Dict[str, Any],
    params: Dict[str, Any],
    return_profiles: bool,
) -> Dict[str, Any]:
    """
    FULL mode with EDL disabled:
    - phi_tilde(x) = 0 everywhere
    - K_Au = L_Au_tilde, K_Pd = L_Pd_tilde
    - local i1/i2 are uniform within segments
    """
    R_gas = float(params["R"]); F = float(params["F"]); T = float(params["T"])
    beta = F / (R_gas * T)

    it0_1 = float(params["it0_1"]); it0_2 = float(params["it0_2"])
    alpha1 = float(params["alpha1"]); alpha2 = float(params["alpha2"])
    E1_eq = float(params["E1_eq"]); E2_eq = float(params["E2_eq"])

    eta1 = E - E1_eq
    eta2 = E - E2_eq

    L_Au = float(derived["L_Au_tilde"])
    L_C = float(derived["L_C_tilde"])
    L = float(derived["L_tilde"])
    L_Pd = L - L_C

    K_Au = L_Au
    K_Pd = L_Pd

    I_Au = float(it0_1 * safe_exp((1.0 - alpha1) * beta * eta1) * K_Au)
    I_Pd = float(-it0_2 * safe_exp(-alpha2 * beta * eta2) * K_Pd)

    residual = I_Au + I_Pd
    i_mix = abs(I_Au)

    out: Dict[str, Any] = dict(I_Au=I_Au, I_Pd=I_Pd, residual=residual, i_mix=i_mix, K_Au=K_Au, K_Pd=K_Pd)

    if return_profiles:
        Nx = int(params["Nx"])
        x = np.linspace(0.0, L, Nx)
        phi_tilde = np.zeros_like(x)
        mask_Au = (x >= 0.0) & (x <= L_Au + 1e-12)
        mask_Pd = (x >= L_C - 1e-12) & (x <= L + 1e-12)

        i1 = np.zeros_like(x)
        i2 = np.zeros_like(x)
        pref1 = it0_1 * safe_exp((1.0 - alpha1) * beta * eta1)
        pref2 = -it0_2 * safe_exp(-alpha2 * beta * eta2)
        i1[mask_Au] = pref1
        i2[mask_Pd] = pref2
        out.update(dict(x_tilde=x, phi_tilde=phi_tilde, i1=i1, i2=i2, mask_Au=mask_Au, mask_Pd=mask_Pd))
    return out


def mean_mode_residual_no_edl(E: float, derived: Dict[str, Any], params: Dict[str, Any]) -> float:
    """Mean-field residual with EDL disabled (phi2_1=phi2_2=0)."""
    i1, i2 = currents_mean_field(E, 0.0, 0.0, params)
    L_Au = float(derived["L_Au_tilde"])
    L_Pd = float(derived["L_tilde"] - derived["L_C_tilde"])
    return float(L_Au * i1 + L_Pd * i2)


def emix_closed_form_no_edl(derived: Dict[str, Any], params: Dict[str, Any]) -> float:
    """Closed-form Emix for no-EDL mean-field model (phi2_1=phi2_2=0)."""
    R_gas = float(params["R"]); F = float(params["F"]); T = float(params["T"])
    it0_1 = float(params["it0_1"]); it0_2 = float(params["it0_2"])
    alpha1 = float(params["alpha1"]); alpha2 = float(params["alpha2"])
    E1_eq = float(params["E1_eq"]); E2_eq = float(params["E2_eq"])

    kappa = 1.0 - alpha1 + alpha2
    L_Au = float(derived["L_Au_tilde"])
    L_Pd = float(derived["L_tilde"] - derived["L_C_tilde"])

    E_base = ((1.0 - alpha1) * E1_eq + alpha2 * E2_eq) / kappa
    E_base += (R_gas * T / (F * kappa)) * math.log((L_Pd * it0_2) / (L_Au * it0_1))
    return float(E_base)


def solve_emix_no_edl(
    params: Dict[str, Any],
    derived: Dict[str, Any],
    mode: str,
    xtol: float,
    max_bracket_expands: int,
    E_guess: Optional[float] = None,
    bracket: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """Root-find Emix for no-EDL kinetics (phi2_1=phi2_2=0)."""
    mode = mode.upper()
    E1_eq = float(params["E1_eq"]); E2_eq = float(params["E2_eq"])
    if E_guess is None:
        E_guess = 0.5 * (E1_eq + E2_eq)
    if bracket is None:
        bracket = (min(E1_eq, E2_eq) - 0.5, max(E1_eq, E2_eq) + 0.5)

    if mode == "FULL":
        f = lambda E: float(full_mode_currents_no_edl(E, derived, params, return_profiles=False)["residual"])
    elif mode == "MEAN":
        f = lambda E: float(mean_mode_residual_no_edl(E, derived, params))
    else:
        raise ValueError("mode must be FULL or MEAN")

    a, b = float(bracket[0]), float(bracket[1])
    fa, fb = f(a), f(b)
    expands = 0
    while np.sign(fa) == np.sign(fb) and expands < max_bracket_expands:
        mid = 0.5 * (a + b)
        span = b - a
        a = mid - 1.5 * span
        b = mid + 1.5 * span
        fa, fb = f(a), f(b)
        expands += 1

    info: Dict[str, Any] = dict(mode=mode, bracket_a=a, bracket_b=b, f_a=float(fa), f_b=float(fb), expands=expands)

    try:
        if np.isfinite(fa) and np.isfinite(fb) and np.sign(fa) != np.sign(fb):
            sol = root_scalar(f, bracket=(a, b), method="brentq", xtol=xtol)
            method = "brentq"
        else:
            sol = root_scalar(f, x0=E_guess, x1=E_guess + 0.05, method="secant", xtol=xtol, maxiter=200)
            method = "secant"

        info.update(method=method, converged=bool(sol.converged), iterations=int(sol.iterations))
        if not sol.converged:
            return float("nan"), float("nan"), info

        E_mix = float(sol.root)

        if mode == "FULL":
            cur = full_mode_currents_no_edl(E_mix, derived, params, return_profiles=False)
            i_mix = float(cur["i_mix"])
            resid = float(cur["residual"])
        else:
            i1, _ = currents_mean_field(E_mix, 0.0, 0.0, params)
            i_mix = abs(float(derived["L_Au_tilde"]) * i1)
            resid = mean_mode_residual_no_edl(E_mix, derived, params)

        info["residual_at_root"] = float(resid)
        return E_mix, i_mix, info

    except Exception as e:
        info.update(converged=False, error=repr(e))
        return float("nan"), float("nan"), info


def mean_mode_residual(E: float, edl: EDLModel, params: Dict[str, Any], use_affine_phi2: bool) -> float:
    """Mean-field residual: L̃_Au i1 + (L̃-L̃_C) i2 = 0 (Eq. (S3-13b))."""
    phi2_1, phi2_2 = edl.segment_mean_phi2(E, use_affine_phi2=use_affine_phi2)
    i1, i2 = currents_mean_field(E, phi2_1, phi2_2, params)
    L_Au = edl.derived["L_Au_tilde"]
    L_Pd = edl.derived["L_tilde"] - edl.derived["L_C_tilde"]
    return float(L_Au * i1 + L_Pd * i2)


def emix_closed_form_affine(edl: EDLModel, params: Dict[str, Any]) -> float:
    """
    Closed-form Emix when φ2,i = a_i E + b_i (Eq. (D5-3)),
    equivalent to Eq. (D5-6)/(D5-8) but with base term consistent with Eq. (S3-14).
    TODO: verify whether your SI's E0 definition already includes the length ratio term.
    """
    R_gas = float(params["R"]); F = float(params["F"]); T = float(params["T"])
    it0_1 = float(params["it0_1"]); it0_2 = float(params["it0_2"])
    alpha1 = float(params["alpha1"]); alpha2 = float(params["alpha2"])
    z_R1 = float(params["z_R1"]); z_O2 = float(params["z_O2"])
    E1_eq = float(params["E1_eq"]); E2_eq = float(params["E2_eq"])

    kappa = 1.0 - alpha1 + alpha2
    rho = (1.0 - alpha1) + z_R1
    chi = alpha2 - z_O2

    a1 = float(edl.pre["a1"]); a2 = float(edl.pre["a2"])
    b1 = float(edl.pre["b1"]); b2 = float(edl.pre["b2"])

    L_Au = edl.derived["L_Au_tilde"]
    L_Pd = edl.derived["L_tilde"] - edl.derived["L_C_tilde"]

    E_base = ((1.0 - alpha1) * E1_eq + alpha2 * E2_eq) / kappa
    E_base += (R_gas * T / (F * kappa)) * math.log((L_Pd * it0_2) / (L_Au * it0_1))

    return float((kappa * E_base + chi * b2 + rho * b1) / (kappa - rho * a1 - chi * a2))


def solve_emix(
    edl: EDLModel,
    params: Dict[str, Any],
    mode: str,
    use_affine_phi2: bool,
    xtol: float,
    max_bracket_expands: int,
    E_guess: Optional[float] = None,
    bracket: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """Root-find Emix from zero-net-current condition (Eq. (S3-7b)/(S3-13b))."""
    mode = mode.upper()
    E1_eq = float(params["E1_eq"]); E2_eq = float(params["E2_eq"])
    if E_guess is None:
        E_guess = 0.5 * (E1_eq + E2_eq)
    if bracket is None:
        bracket = (min(E1_eq, E2_eq) - 0.5, max(E1_eq, E2_eq) + 0.5)

    if mode == "FULL":
        f = lambda E: float(full_mode_currents(E, edl, params, return_profiles=False)["residual"])
    elif mode == "MEAN":
        f = lambda E: float(mean_mode_residual(E, edl, params, use_affine_phi2=use_affine_phi2))
    else:
        raise ValueError("mode must be FULL or MEAN")

    a, b = float(bracket[0]), float(bracket[1])
    fa, fb = f(a), f(b)
    expands = 0
    while np.sign(fa) == np.sign(fb) and expands < max_bracket_expands:
        mid = 0.5 * (a + b)
        span = b - a
        a = mid - 1.5 * span
        b = mid + 1.5 * span
        fa, fb = f(a), f(b)
        expands += 1

    info: Dict[str, Any] = dict(mode=mode, use_affine_phi2=use_affine_phi2,
                               bracket_a=a, bracket_b=b, f_a=float(fa), f_b=float(fb), expands=expands)

    try:
        if np.isfinite(fa) and np.isfinite(fb) and np.sign(fa) != np.sign(fb):
            sol = root_scalar(f, bracket=(a, b), method="brentq", xtol=xtol)
            method = "brentq"
        else:
            sol = root_scalar(f, x0=E_guess, x1=E_guess + 0.05, method="secant", xtol=xtol, maxiter=200)
            method = "secant"

        info.update(method=method, converged=bool(sol.converged), iterations=int(sol.iterations))
        if not sol.converged:
            return float("nan"), float("nan"), info

        E_mix = float(sol.root)

        if mode == "FULL":
            cur = full_mode_currents(E_mix, edl, params, return_profiles=False)
            i_mix = float(cur["i_mix"])
            resid = float(cur["residual"])
        else:
            phi2_1, phi2_2 = edl.segment_mean_phi2(E_mix, use_affine_phi2=use_affine_phi2)
            i1, _ = currents_mean_field(E_mix, phi2_1, phi2_2, params)
            i_mix = abs(edl.derived["L_Au_tilde"] * i1)
            resid = mean_mode_residual(E_mix, edl, params, use_affine_phi2=use_affine_phi2)

        info["residual_at_root"] = float(resid)
        return E_mix, i_mix, info

    except Exception as e:
        info.update(converged=False, error=repr(e))
        return float("nan"), float("nan"), info


# -----------------------------
# Single-case runner
# -----------------------------

def run_case(
    params: Dict[str, Any],
    mode: str,
    return_profiles: bool,
    use_edl: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run one parameter set; returns Emix, imix, and (FULL) profiles."""
    mode = mode.upper()
    p = copy.deepcopy(params)

    if use_edl is None:
        use_edl = bool(p.get("use_edl", True))
    else:
        use_edl = bool(use_edl)
        p["use_edl"] = use_edl

    # EDL entry points:
    # - currents_mean_field(...): phi2_1/phi2_2 in the Frumkin term (-Gamma*beta*phi2)
    # - full_mode_currents(...): phi_tilde(x) in Boltzmann factors exp(+/-Gamma*phi_tilde)
    # no-EDL handling: set phi2_1/phi2_2 = 0, phi_tilde = 0, K_Au/L_Au_tilde, K_Pd/L_Pd_tilde, and skip EDL solve.
    use_affine_phi2 = bool(p.get("use_affine_phi2", True))
    use_closed = bool(p.get("use_closed_form_when_affine", True))

    if use_edl:
        edl = EDLModel(p)

        if mode == "MEAN" and use_affine_phi2 and use_closed:
            E_mix = emix_closed_form_affine(edl, p)
            resid = mean_mode_residual(E_mix, edl, p, use_affine_phi2=True)
            info = dict(converged=True, method="closed_form(D5-6/D5-8)", iterations=0, residual_at_root=float(resid))
            # i_mix
            phi2_1, phi2_2 = edl.segment_mean_phi2(E_mix, use_affine_phi2=True)
            i1, _ = currents_mean_field(E_mix, phi2_1, phi2_2, p)
            i_mix = abs(edl.derived["L_Au_tilde"] * i1)
        else:
            E_mix, i_mix, info = solve_emix(
                edl=edl,
                params=p,
                mode=mode,
                use_affine_phi2=use_affine_phi2,
                xtol=float(p.get("xtol", 1e-10)),
                max_bracket_expands=int(p.get("max_bracket_expands", 12)),
            )

        out: Dict[str, Any] = dict(
            mode=mode,
            E_mix=float(E_mix),
            i_mix=float(i_mix),
            residual=float(info.get("residual_at_root", float("nan"))),
            converged=bool(info.get("converged", False)),
            method=str(info.get("method", "")),
            iterations=int(info.get("iterations", -1)),
            # EDL affine coefficients
            a1=float(edl.pre["a1"]), b1=float(edl.pre["b1"]),
            a2=float(edl.pre["a2"]), b2=float(edl.pre["b2"]),
            # derived EDL params
            lambda_D=float(edl.derived["lambda_D"]),
            g_Au=float(edl.derived["g_Au"]), g_C=float(edl.derived["g_C"]), g_Pd=float(edl.derived["g_Pd"]),
            L_tilde=float(edl.derived["L_tilde"]),
            L_Au_tilde=float(edl.derived["L_Au_tilde"]),
            L_C_tilde=float(edl.derived["L_C_tilde"]),
        )

        if return_profiles:
            if mode != "FULL":
                raise ValueError("return_profiles=True only supported for FULL mode")
            prof = full_mode_currents(float(E_mix), edl, p, return_profiles=True)
            out.update(prof)
            # segment-mean phi2 values for reporting
            phi2_1_m, phi2_2_m = edl.segment_mean_phi2(float(E_mix), use_affine_phi2=False)
            out["phi2_1_meanV"] = float(phi2_1_m)
            out["phi2_2_meanV"] = float(phi2_2_m)
    else:
        derived = compute_derived_params(p)

        if mode == "MEAN" and use_closed:
            E_mix = emix_closed_form_no_edl(derived, p)
            resid = mean_mode_residual_no_edl(E_mix, derived, p)
            info = dict(converged=True, method="closed_form(no_edl)", iterations=0, residual_at_root=float(resid))
            i1, _ = currents_mean_field(E_mix, 0.0, 0.0, p)
            i_mix = abs(derived["L_Au_tilde"] * i1)
        else:
            E_mix, i_mix, info = solve_emix_no_edl(
                params=p,
                derived=derived,
                mode=mode,
                xtol=float(p.get("xtol", 1e-10)),
                max_bracket_expands=int(p.get("max_bracket_expands", 12)),
            )

        # a1/b1/a2/b2 are EDL-specific; keep zeros for no-EDL outputs.
        out = dict(
            mode=mode,
            E_mix=float(E_mix),
            i_mix=float(i_mix),
            residual=float(info.get("residual_at_root", float("nan"))),
            converged=bool(info.get("converged", False)),
            method=str(info.get("method", "")),
            iterations=int(info.get("iterations", -1)),
            a1=0.0, b1=0.0,
            a2=0.0, b2=0.0,
            lambda_D=float(derived["lambda_D"]),
            g_Au=float(derived["g_Au"]), g_C=float(derived["g_C"]), g_Pd=float(derived["g_Pd"]),
            L_tilde=float(derived["L_tilde"]),
            L_Au_tilde=float(derived["L_Au_tilde"]),
            L_C_tilde=float(derived["L_C_tilde"]),
        )

        if return_profiles:
            if mode != "FULL":
                raise ValueError("return_profiles=True only supported for FULL mode")
            prof = full_mode_currents_no_edl(float(E_mix), derived, p, return_profiles=True)
            out.update(prof)
            out["phi2_1_meanV"] = 0.0
            out["phi2_2_meanV"] = 0.0

    return out


# -----------------------------
# Saving + plotting
# -----------------------------

def plot_baseline_profiles(case_full: Dict[str, Any], params: Dict[str, Any], out_dir: Path) -> None:
    """Save baseline FULL-mode potential and current profiles as PNG."""
    fig_dir = ensure_dir(out_dir / "figures")

    R_gas = float(params["R"]); F = float(params["F"]); T = float(params["T"])
    scale = R_gas * T / F

    x_tilde = case_full["x_tilde"]
    x_nm = x_tilde * case_full["lambda_D"] * 1e9
    phi2 = scale * case_full["phi_tilde"]

    i1 = case_full["i1"]
    i2 = case_full["i2"]

    L_Au_nm = case_full["L_Au_tilde"] * case_full["lambda_D"] * 1e9
    L_C_nm = case_full["L_C_tilde"] * case_full["lambda_D"] * 1e9

    plt.figure()
    plt.plot(x_nm, phi2)
    plt.axvline(L_Au_nm, linestyle="--")
    plt.axvline(L_C_nm, linestyle="--")
    plt.xlabel("x [nm]")
    plt.ylabel(r"$\phi_2(x)$ [V]")
    plt.title("Reaction-plane potential along surface (FULL mode)")
    plt.tight_layout()
    plt.savefig(fig_dir / "baseline_phi2.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(x_nm, i1, label="i1 (Au)")
    plt.plot(x_nm, i2, label="i2 (Pd)")
    plt.axvline(L_Au_nm, linestyle="--")
    plt.axvline(L_C_nm, linestyle="--")
    plt.xlabel("x [nm]")
    plt.ylabel(r"$i(x)$ [A/m$^2$]")
    plt.title("Local current density profiles (FULL mode)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "baseline_currents.png", dpi=300)
    plt.close()

    plot_baseline_profiles_html(
        x_nm=x_nm,
        phi2=phi2,
        i1=i1,
        i2=i2,
        L_Au_nm=L_Au_nm,
        L_C_nm=L_C_nm,
        params=params,
        fig_dir=fig_dir,
    )


def plot_baseline_profiles_html(
    x_nm: np.ndarray,
    phi2: np.ndarray,
    i1: np.ndarray,
    i2: np.ndarray,
    L_Au_nm: float,
    L_C_nm: float,
    params: Dict[str, Any],
    fig_dir: Path,
) -> None:
    if go is None:
        _maybe_warn_plotly()
        return

    hover_params = format_hover_params(params, keys=HOVER_PARAM_KEYS)
    hover_text = [hover_params] * len(x_nm)

    fig_phi = go.Figure()
    fig_phi.add_trace(
        go.Scatter(
            x=x_nm,
            y=phi2,
            mode="lines",
            name="phi2",
            text=hover_text,
            hovertemplate="x=%{x:.6g} nm<br>phi2=%{y:.6g} V<br>%{text}<extra></extra>",
        )
    )
    fig_phi.add_shape(
        type="line",
        x0=L_Au_nm,
        x1=L_Au_nm,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(dash="dash", color="gray"),
    )
    fig_phi.add_shape(
        type="line",
        x0=L_C_nm,
        x1=L_C_nm,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(dash="dash", color="gray"),
    )
    fig_phi.update_layout(
        title="Reaction-plane potential along surface (FULL mode)",
        xaxis_title="x [nm]",
        yaxis_title="phi2(x) [V]",
        hovermode="closest",
    )
    write_plotly_html(fig_phi, fig_dir / "baseline_phi2.html")

    fig_i = go.Figure()
    fig_i.add_trace(
        go.Scatter(
            x=x_nm,
            y=i1,
            mode="lines",
            name="i1 (Au)",
            text=hover_text,
            hovertemplate="x=%{x:.6g} nm<br>%{fullData.name}=%{y:.6g} A/m^2<br>%{text}<extra></extra>",
        )
    )
    fig_i.add_trace(
        go.Scatter(
            x=x_nm,
            y=i2,
            mode="lines",
            name="i2 (Pd)",
            text=hover_text,
            hovertemplate="x=%{x:.6g} nm<br>%{fullData.name}=%{y:.6g} A/m^2<br>%{text}<extra></extra>",
        )
    )
    fig_i.add_shape(
        type="line",
        x0=L_Au_nm,
        x1=L_Au_nm,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(dash="dash", color="gray"),
    )
    fig_i.add_shape(
        type="line",
        x0=L_C_nm,
        x1=L_C_nm,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(dash="dash", color="gray"),
    )
    fig_i.update_layout(
        title="Local current density profiles (FULL mode)",
        xaxis_title="x [nm]",
        yaxis_title="i(x) [A/m^2]",
        hovermode="closest",
    )
    write_plotly_html(fig_i, fig_dir / "baseline_currents.html")


def plot_ofat_html(
    dfp_plot: pd.DataFrame,
    pname: str,
    metric: str,
    ylab: str,
    xscale: str,
    modes: List[str],
    fig_dir: Path,
) -> None:
    if go is None:
        _maybe_warn_plotly()
        return
    if dfp_plot.empty:
        return

    fig = go.Figure()
    for mode in modes:
        sub = dfp_plot[dfp_plot["mode"] == mode]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["value"],
                y=sub[metric],
                mode="lines+markers",
                name=mode,
                text=sub["hover_params"],
                hovertemplate=f"{pname}=%{{x:.6g}}<br>{metric}=%{{y:.6g}}<br>%{{text}}<extra>{mode}</extra>",
            )
        )

    fig.update_layout(
        title=f"OFAT: {metric} vs {pname}",
        xaxis_title=pname,
        yaxis_title=ylab,
        hovermode="closest",
    )
    if xscale == "log":
        fig.update_xaxes(type="log")
    write_plotly_html(fig, fig_dir / f"ofat_{pname}_{metric}.html")


def _format_nm_tag(val_m: float) -> str:
    nm = float(val_m) * 1e9
    if abs(nm - round(nm)) < 1e-6:
        return str(int(round(nm)))
    return f"{nm:.3g}".replace(".", "p")


def _case_tag_from_params(params: Dict[str, Any]) -> str:
    return "LAu{0}_Lgap{1}_LPd{2}".format(
        _format_nm_tag(params["L_Au"]),
        _format_nm_tag(params["L_gap"]),
        _format_nm_tag(params["L_Pd_len"]),
    )


def build_profiles_for_emix(
    params: Dict[str, Any],
    E_mix: float,
    use_edl: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if use_edl:
        edl = EDLModel(params)
        prof = full_mode_currents(float(E_mix), edl, params, return_profiles=True)
        derived = edl.derived
        return prof, derived
    derived = compute_derived_params(params)
    prof = full_mode_currents_no_edl(float(E_mix), derived, params, return_profiles=True)
    return prof, derived


def _save_profiles_npz(path: Path, prof: Dict[str, Any], derived: Dict[str, Any]) -> None:
    scale = float(derived["R"]) * float(derived["T"]) / float(derived["F"])
    x_tilde = prof["x_tilde"]
    x_m = x_tilde * float(derived["lambda_D"])
    phi2_V = scale * prof["phi_tilde"]

    np.savez(
        path,
        x_tilde=x_tilde,
        x_m=x_m,
        phi_tilde=prof["phi_tilde"],
        phi2_V=phi2_V,
        i1=prof["i1"],
        i2=prof["i2"],
        mask_Au=prof["mask_Au"],
        mask_Pd=prof["mask_Pd"],
        L_Au_tilde=float(derived["L_Au_tilde"]),
        L_C_tilde=float(derived["L_C_tilde"]),
        lambda_D=float(derived["lambda_D"]),
    )


def compute_polarization_curve(
    params: Dict[str, Any],
    mode: str,
    use_edl: bool,
    E_values: np.ndarray,
    use_affine_phi2: bool,
) -> Dict[str, np.ndarray]:
    mode = mode.upper()
    I_Au = np.zeros_like(E_values, dtype=float)
    I_Pd = np.zeros_like(E_values, dtype=float)
    I_total = np.zeros_like(E_values, dtype=float)

    if use_edl:
        edl = EDLModel(params)
        L_Au = float(edl.derived["L_Au_tilde"])
        L_Pd = float(edl.derived["L_tilde"] - edl.derived["L_C_tilde"])
        for i, E in enumerate(E_values):
            if mode == "FULL":
                cur = full_mode_currents(float(E), edl, params, return_profiles=False)
                I_Au[i] = float(cur["I_Au"])
                I_Pd[i] = float(cur["I_Pd"])
                I_total[i] = float(cur["residual"])
            elif mode == "MEAN":
                phi2_1, phi2_2 = edl.segment_mean_phi2(float(E), use_affine_phi2=use_affine_phi2)
                i1, i2 = currents_mean_field(float(E), phi2_1, phi2_2, params)
                I_Au[i] = L_Au * i1
                I_Pd[i] = L_Pd * i2
                I_total[i] = I_Au[i] + I_Pd[i]
            else:
                raise ValueError("mode must be FULL or MEAN")
    else:
        derived = compute_derived_params(params)
        L_Au = float(derived["L_Au_tilde"])
        L_Pd = float(derived["L_tilde"] - derived["L_C_tilde"])
        for i, E in enumerate(E_values):
            if mode == "FULL":
                cur = full_mode_currents_no_edl(float(E), derived, params, return_profiles=False)
                I_Au[i] = float(cur["I_Au"])
                I_Pd[i] = float(cur["I_Pd"])
                I_total[i] = float(cur["residual"])
            elif mode == "MEAN":
                i1, i2 = currents_mean_field(float(E), 0.0, 0.0, params)
                I_Au[i] = L_Au * i1
                I_Pd[i] = L_Pd * i2
                I_total[i] = I_Au[i] + I_Pd[i]
            else:
                raise ValueError("mode must be FULL or MEAN")

    return dict(E=E_values, I_total=I_total, I_Au=I_Au, I_Pd=I_Pd)


def plot_compare_polarization_curve(
    curve_edl: Dict[str, np.ndarray],
    curve_no: Dict[str, np.ndarray],
    E_mix_edl: float,
    E_mix_no: float,
    out_path: Path,
    title: str,
) -> None:
    plt.figure()
    plt.plot(curve_edl["E"], curve_edl["I_total"], label="use_edl=True")
    plt.plot(curve_no["E"], curve_no["I_total"], label="use_edl=False")
    plt.axhline(0.0, color="k", linewidth=0.8)
    plt.axvline(E_mix_edl, linestyle="--", color="C0")
    plt.axvline(E_mix_no, linestyle="--", color="C1")
    plt.xlabel("E [V]")
    plt.ylabel("I_total [A/m^2]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_compare_emix_imix(
    E_mix_edl: float,
    i_mix_edl: float,
    E_mix_no: float,
    i_mix_no: float,
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].bar(["with_edl", "no_edl"], [E_mix_edl, E_mix_no])
    axes[0].set_ylabel("E_mix [V]")
    axes[0].set_title("E_mix")
    axes[1].bar(["with_edl", "no_edl"], [i_mix_edl, i_mix_no])
    axes[1].set_ylabel("i_mix [A/m^2]")
    axes[1].set_title("i_mix")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_compare_phi2(
    prof_edl: Dict[str, Any],
    derived_edl: Dict[str, Any],
    prof_no: Dict[str, Any],
    derived_no: Dict[str, Any],
    out_path: Path,
    title: str,
) -> None:
    scale_edl = float(derived_edl["R"]) * float(derived_edl["T"]) / float(derived_edl["F"])
    scale_no = float(derived_no["R"]) * float(derived_no["T"]) / float(derived_no["F"])
    x_nm = prof_edl["x_tilde"] * float(derived_edl["lambda_D"]) * 1e9
    phi2_edl = scale_edl * prof_edl["phi_tilde"]
    phi2_no = scale_no * prof_no["phi_tilde"]

    L_Au_nm = float(derived_edl["L_Au_tilde"]) * float(derived_edl["lambda_D"]) * 1e9
    L_C_nm = float(derived_edl["L_C_tilde"]) * float(derived_edl["lambda_D"]) * 1e9

    plt.figure()
    plt.plot(x_nm, phi2_edl, label="use_edl=True")
    plt.plot(x_nm, phi2_no, label="use_edl=False")
    plt.axvline(L_Au_nm, linestyle="--")
    plt.axvline(L_C_nm, linestyle="--")
    plt.xlabel("x [nm]")
    plt.ylabel("phi2(x) [V]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def compare_edl_effects(
    base_params: Dict[str, Any],
    outputs: Tuple[str, ...] = ("E_mix", "i_mix", "phi2_vs_x", "polarization_curve"),
    save_dir: str | Path = "results/compare_case1",
    solver_settings: Optional[Dict[str, Any]] = None,
    save_data: bool = True,
    save_fig: bool = True,
) -> Dict[str, Any]:
    p0 = copy.deepcopy(base_params)
    outputs_set = {o.lower() for o in outputs}
    solver_settings = solver_settings or {}

    mode = str(solver_settings.get("mode", "FULL")).upper()
    use_affine_phi2 = bool(p0.get("use_affine_phi2", True))

    res_edl = run_case(p0, mode=mode, return_profiles=False, use_edl=True)
    res_no = run_case(p0, mode=mode, return_profiles=False, use_edl=False)

    delta_E_mix = res_edl["E_mix"] - res_no["E_mix"]
    if res_no["i_mix"] != 0:
        ratio_i_mix = res_edl["i_mix"] / res_no["i_mix"]
        pct_i_mix = 100.0 * (res_edl["i_mix"] - res_no["i_mix"]) / res_no["i_mix"]
    else:
        ratio_i_mix = float("nan")
        pct_i_mix = float("nan")

    comparison = dict(delta_E_mix=float(delta_E_mix), ratio_i_mix=float(ratio_i_mix), pct_i_mix=float(pct_i_mix))
    out: Dict[str, Any] = dict(with_edl=res_edl, no_edl=res_no, comparison=comparison, paths={})

    prof_edl: Optional[Dict[str, Any]] = None
    prof_no: Optional[Dict[str, Any]] = None
    derived_edl: Optional[Dict[str, Any]] = None
    derived_no: Optional[Dict[str, Any]] = None

    if save_data or ("phi2_vs_x" in outputs_set):
        prof_edl, derived_edl = build_profiles_for_emix(p0, float(res_edl["E_mix"]), use_edl=True)
        prof_no, derived_no = build_profiles_for_emix(p0, float(res_no["E_mix"]), use_edl=False)

        if "phi2_vs_x" in outputs_set:
            scale_edl = float(derived_edl["R"]) * float(derived_edl["T"]) / float(derived_edl["F"])
            scale_no = float(derived_no["R"]) * float(derived_no["T"]) / float(derived_no["F"])
            out["with_edl"]["phi2_vs_x"] = dict(
                x_m=prof_edl["x_tilde"] * float(derived_edl["lambda_D"]),
                phi2_V=scale_edl * prof_edl["phi_tilde"],
            )
            out["no_edl"]["phi2_vs_x"] = dict(
                x_m=prof_no["x_tilde"] * float(derived_no["lambda_D"]),
                phi2_V=scale_no * prof_no["phi_tilde"],
            )

    curve_edl: Optional[Dict[str, np.ndarray]] = None
    curve_no: Optional[Dict[str, np.ndarray]] = None
    if "polarization_curve" in outputs_set:
        E_min = solver_settings.get("E_min")
        E_max = solver_settings.get("E_max")
        n_E = int(solver_settings.get("n_E", 200))
        if E_min is None or E_max is None:
            span = float(solver_settings.get("E_span", 0.5))
            E1_eq = float(p0["E1_eq"])
            E2_eq = float(p0["E2_eq"])
            E_min = min(E1_eq, E2_eq) - span
            E_max = max(E1_eq, E2_eq) + span
        E_values = np.linspace(float(E_min), float(E_max), n_E)
        curve_edl = compute_polarization_curve(p0, mode=mode, use_edl=True, E_values=E_values, use_affine_phi2=use_affine_phi2)
        curve_no = compute_polarization_curve(p0, mode=mode, use_edl=False, E_values=E_values, use_affine_phi2=use_affine_phi2)
        out["with_edl"]["polarization_curve"] = curve_edl
        out["no_edl"]["polarization_curve"] = curve_no

    if save_data or save_fig:
        out_dir = ensure_dir(Path(save_dir))
        paths: Dict[str, str] = {"save_dir": str(out_dir)}

        if save_data:
            with_dir = ensure_dir(out_dir / "with_edl")
            no_dir = ensure_dir(out_dir / "no_edl")
            ensure_dir(with_dir / "figures")
            ensure_dir(no_dir / "figures")

            params_path = out_dir / "params.json"
            with params_path.open("w", encoding="utf-8") as f:
                json.dump(p0, f, indent=2, sort_keys=True, default=str)

            summary_path = out_dir / "summary_compare.csv"
            df_sum = pd.DataFrame([dict(
                E_mix_with=res_edl["E_mix"],
                i_mix_with=res_edl["i_mix"],
                E_mix_no=res_no["E_mix"],
                i_mix_no=res_no["i_mix"],
                delta_E_mix=comparison["delta_E_mix"],
                ratio_i_mix=comparison["ratio_i_mix"],
                pct_i_mix=comparison["pct_i_mix"],
                mode=mode,
            )])
            df_sum.to_csv(summary_path, index=False)

            if prof_edl is None or prof_no is None or derived_edl is None or derived_no is None:
                prof_edl, derived_edl = build_profiles_for_emix(p0, float(res_edl["E_mix"]), use_edl=True)
                prof_no, derived_no = build_profiles_for_emix(p0, float(res_no["E_mix"]), use_edl=False)

            _save_profiles_npz(with_dir / "profiles.npz", prof_edl, derived_edl)
            _save_profiles_npz(no_dir / "profiles.npz", prof_no, derived_no)

            paths.update(
                params_json=str(params_path),
                summary_compare=str(summary_path),
                with_edl_profiles=str(with_dir / "profiles.npz"),
                no_edl_profiles=str(no_dir / "profiles.npz"),
            )

        if save_fig:
            fig_dir = ensure_dir(out_dir / "figures")
            tag = _case_tag_from_params(p0)
            title = f"EDL compare ({mode})"

            if curve_edl is None or curve_no is None:
                E_min = solver_settings.get("E_min")
                E_max = solver_settings.get("E_max")
                n_E = int(solver_settings.get("n_E", 200))
                if E_min is None or E_max is None:
                    span = float(solver_settings.get("E_span", 0.5))
                    E1_eq = float(p0["E1_eq"])
                    E2_eq = float(p0["E2_eq"])
                    E_min = min(E1_eq, E2_eq) - span
                    E_max = max(E1_eq, E2_eq) + span
                E_values = np.linspace(float(E_min), float(E_max), n_E)
                curve_edl = compute_polarization_curve(p0, mode=mode, use_edl=True, E_values=E_values, use_affine_phi2=use_affine_phi2)
                curve_no = compute_polarization_curve(p0, mode=mode, use_edl=False, E_values=E_values, use_affine_phi2=use_affine_phi2)

            plot_compare_polarization_curve(
                curve_edl=curve_edl,
                curve_no=curve_no,
                E_mix_edl=float(res_edl["E_mix"]),
                E_mix_no=float(res_no["E_mix"]),
                out_path=fig_dir / f"compare_polcurve_{tag}.png",
                title=title,
            )

            plot_compare_emix_imix(
                E_mix_edl=float(res_edl["E_mix"]),
                i_mix_edl=float(res_edl["i_mix"]),
                E_mix_no=float(res_no["E_mix"]),
                i_mix_no=float(res_no["i_mix"]),
                out_path=fig_dir / f"compare_emix_imix_{tag}.png",
                title=title,
            )

            if "phi2_vs_x" in outputs_set:
                if prof_edl is None or prof_no is None or derived_edl is None or derived_no is None:
                    prof_edl, derived_edl = build_profiles_for_emix(p0, float(res_edl["E_mix"]), use_edl=True)
                    prof_no, derived_no = build_profiles_for_emix(p0, float(res_no["E_mix"]), use_edl=False)
                plot_compare_phi2(
                    prof_edl=prof_edl,
                    derived_edl=derived_edl,
                    prof_no=prof_no,
                    derived_no=derived_no,
                    out_path=fig_dir / f"compare_phi2_{tag}.png",
                    title=title,
                )

            paths["figures_dir"] = str(fig_dir)

        out["paths"] = paths

    return out


def self_check_with_reference(
    params: Dict[str, Any],
    ref: Dict[str, float],
    mode: str = "FULL",
    atol_E: float = 1e-10,
    atol_I: float = 1e-8,
) -> Dict[str, float]:
    """Compare current with-EDL results to a saved reference (from older code)."""
    res = run_case(params, mode=mode, return_profiles=False, use_edl=True)
    dE = float(abs(res["E_mix"] - float(ref["E_mix"])))
    dI = float(abs(res["i_mix"] - float(ref["i_mix"])))
    ok = (dE <= atol_E) and (dI <= atol_I)
    print(f"self_check_with_reference: dE={dE:.6e}, dI={dI:.6e}, ok={ok}")
    return dict(dE=dE, dI=dI, ok=bool(ok))


def make_summary_row(run_tag: str, params: Dict[str, Any], result: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Row for results_summary.csv (includes full parameters + key outputs)."""
    row = dict(run=run_tag, **flatten_dict(params))
    row.update(
        mode=result.get("mode"),
        E_mix=result.get("E_mix"),
        i_mix=result.get("i_mix"),
        residual=result.get("residual"),
        converged=result.get("converged"),
        method=result.get("method"),
        iterations=result.get("iterations"),
        lambda_D=result.get("lambda_D"),
        g_Au=result.get("g_Au"), g_C=result.get("g_C"), g_Pd=result.get("g_Pd"),
        a1=result.get("a1"), b1=result.get("b1"), a2=result.get("a2"), b2=result.get("b2"),
    )
    if extra:
        row.update(extra)
    if "phi2_1_meanV" in result:
        row["phi2_1_meanV"] = result["phi2_1_meanV"]
        row["phi2_2_meanV"] = result["phi2_2_meanV"]
    return row


# -----------------------------
# Scans and sensitivities
# -----------------------------

def make_ofat_specs(p0: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Default OFAT scan specs (editable)."""
    n = int(p0["ofat_n"])
    return {
        "C_tot":      {"type": "log",    "span": 10.0, "n": n},
        "lambda_D":   {"type": "log",    "span": 10.0, "n": n},
        "epsilon_r":  {"type": "linear", "span": 20.0, "n": n},
        "T":          {"type": "linear", "span": 30.0, "n": n},
        "L_Au":       {"type": "log",    "span": 10.0, "n": n},
        "L_gap":      {"type": "log",    "span": 10.0, "n": n},
        "L_Pd_len":   {"type": "log",    "span": 10.0, "n": n},
        "Cdl_Au":     {"type": "log",    "span": 10.0, "n": n},
        "Cdl_C":      {"type": "log",    "span": 10.0, "n": n},
        "Cdl_Pd":     {"type": "log",    "span": 10.0, "n": n},
        "pzc_Au":     {"type": "linear", "span": 0.2,  "n": n},
        "pzc_C":      {"type": "linear", "span": 0.2,  "n": n},
        "pzc_Pd":     {"type": "linear", "span": 0.2,  "n": n},
        "it0_1":      {"type": "log",    "span": 10.0, "n": n},
        "it0_2":      {"type": "log",    "span": 10.0, "n": n},
        "alpha1":     {"type": "linear", "span": 0.2,  "n": n},
        "alpha2":     {"type": "linear", "span": 0.2,  "n": n},
        "E1_eq":      {"type": "linear", "span": 0.3,  "n": n},
        "E2_eq":      {"type": "linear", "span": 0.3,  "n": n},
        "z_R1":       {"type": "linear", "span": 1.0,  "n": n},
        "z_O2":       {"type": "linear", "span": 1.0,  "n": n},
    }


def make_scan_values(p0_val: float, spec: Dict[str, Any]) -> np.ndarray:
    kind = spec["type"]; span = float(spec["span"]); n = int(spec["n"])
    if kind == "log":
        if p0_val <= 0:
            raise ValueError("log scan requires positive baseline value")
        return np.logspace(np.log10(p0_val / span), np.log10(p0_val * span), n)
    if kind == "linear":
        return np.linspace(p0_val - span, p0_val + span, n)
    raise ValueError(f"Unknown scan type: {kind}")


def run_ofat(base_params: Dict[str, Any], out_dir: Path, modes: List[str], summary_rows: List[Dict[str, Any]]) -> None:
    """OFAT scan: saves per-parameter CSV + PNG, and appends every run to results_summary rows."""
    fig_dir = ensure_dir(out_dir / "figures")
    specs = make_ofat_specs(base_params)

    for pname, spec in specs.items():
        p0 = base_params.get(pname)
        if pname == "lambda_D" and (p0 is None):
            p0 = EDLModel(base_params).derived["lambda_D"]

        vals = make_scan_values(float(p0), spec)
        if pname in ("alpha1", "alpha2"):
            vals = np.clip(vals, 0.01, 0.99)

        rows_param: List[Dict[str, Any]] = []
        rows_param_plot: List[Dict[str, Any]] = []

        for v in vals:
            pvar = copy.deepcopy(base_params)

            if pname == "lambda_D":
                pvar["lambda_D"] = float(v)
            else:
                pvar[pname] = float(v)
                if base_params.get("lambda_D") is None:
                    pvar["lambda_D"] = None

            for mode in modes:
                res = run_case(pvar, mode=mode, return_profiles=False)
                row = dict(param=pname, value=float(v), mode=mode,
                           E_mix=res["E_mix"], i_mix=res["i_mix"],
                           residual=res["residual"], converged=res["converged"], method=res["method"])
                rows_param.append(row)
                rows_param_plot.append(dict(**row, hover_params=format_hover_params(pvar, keys=HOVER_PARAM_KEYS)))
                summary_rows.append(make_summary_row(run_tag=f"ofat:{pname}", params=pvar, result=res))

        dfp = pd.DataFrame(rows_param)
        dfp.to_csv(out_dir / f"ofat_{pname}.csv", index=False)
        dfp_plot = pd.DataFrame(rows_param_plot)

        # plots
        for metric, ylab in [("E_mix", "E_mix [V]"), ("i_mix", "i_mix [A/m^2]")]:
            plt.figure()
            for mode in modes:
                sub = dfp[dfp["mode"] == mode]
                plt.plot(sub["value"], sub[metric], marker="o", linestyle="-", label=mode)
            plt.xlabel(pname)
            plt.ylabel(ylab)
            plt.title(f"OFAT: {metric} vs {pname}")
            if spec["type"] == "log":
                plt.xscale("log")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / f"ofat_{pname}_{metric}.png", dpi=300)
            plt.close()
            plot_ofat_html(dfp_plot, pname, metric, ylab, spec["type"], modes, fig_dir)


def run_heatmaps(base_params: Dict[str, Any], out_dir: Path, mode: str, summary_rows: List[Dict[str, Any]]) -> None:
    """2D heatmaps (>=2 pairs); saves CSV+PNG and appends each grid point to results_summary rows."""
    fig_dir = ensure_dir(out_dir / "figures")
    mode = mode.upper()
    nx = int(base_params["heatmap_nx"])
    ny = int(base_params["heatmap_ny"])

    # --- Pair 1: C_tot vs Δpzc ---
    C0 = float(base_params["C_tot"])
    C_vals = np.logspace(np.log10(C0 / 10.0), np.log10(C0 * 10.0), nx)
    delta0 = float(base_params["pzc_Au"]) - float(base_params["pzc_Pd"])
    delta_vals = np.linspace(delta0 - 0.2, delta0 + 0.2, ny)

    Emix = np.full((ny, nx), np.nan)
    imix = np.full((ny, nx), np.nan)

    for iy, dlt in enumerate(delta_vals):
        for ix, C in enumerate(C_vals):
            pvar = copy.deepcopy(base_params)
            pvar["C_tot"] = float(C)
            pvar["lambda_D"] = None
            pvar["pzc_Au"] = float(pvar["pzc_Pd"]) + float(dlt)
            res = run_case(pvar, mode=mode, return_profiles=False)
            Emix[iy, ix] = res["E_mix"]
            imix[iy, ix] = res["i_mix"]
            summary_rows.append(make_summary_row("heatmap:Ctot_vs_deltapzc", pvar, res,
                                                extra={"C_tot": float(C), "delta_pzc": float(dlt)}))

    pd.DataFrame(Emix, index=delta_vals, columns=C_vals).to_csv(out_dir / f"heatmap_Ctot_vs_deltapzc_Emix_{mode}.csv")
    pd.DataFrame(imix, index=delta_vals, columns=C_vals).to_csv(out_dir / f"heatmap_Ctot_vs_deltapzc_imix_{mode}.csv")

    x_edges = make_edges(C_vals, "log"); y_edges = make_edges(delta_vals, "linear")
    X, Y = np.meshgrid(x_edges, y_edges)

    for Z, name, cbarlab in [(Emix, "Emix", "E_mix [V]"), (imix, "imix", "i_mix [A/m^2]")]:
        plt.figure()
        plt.pcolormesh(X, Y, Z, shading="auto")
        plt.xscale("log")
        plt.xlabel("C_tot [mol/m^3]")
        plt.ylabel("Δpzc = pzc_Au - pzc_Pd [V]")
        plt.title(f"Heatmap ({mode}): {name}(C_tot, Δpzc)")
        plt.colorbar(label=cbarlab)
        plt.tight_layout()
        plt.savefig(fig_dir / f"heatmap_Ctot_vs_deltapzc_{name}_{mode}.png", dpi=300)
        plt.close()

    # --- Pair 2: Cdl_Au multiplier vs it0_1 multiplier ---
    gfac_vals = np.logspace(-1, 1, nx)
    i0fac_vals = np.logspace(-1, 1, ny)

    Emix2 = np.full((ny, nx), np.nan)
    imix2 = np.full((ny, nx), np.nan)

    for iy, i0fac in enumerate(i0fac_vals):
        for ix, gfac in enumerate(gfac_vals):
            pvar = copy.deepcopy(base_params)
            pvar["Cdl_Au"] = float(base_params["Cdl_Au"]) * float(gfac)
            pvar["it0_1"] = float(base_params["it0_1"]) * float(i0fac)
            pvar["g_Au"] = None; pvar["g_C"] = None; pvar["g_Pd"] = None
            res = run_case(pvar, mode=mode, return_profiles=False)
            Emix2[iy, ix] = res["E_mix"]
            imix2[iy, ix] = res["i_mix"]
            summary_rows.append(make_summary_row("heatmap:gfac_vs_i0fac", pvar, res,
                                                extra={"Cdl_Au_multiplier": float(gfac), "it0_1_multiplier": float(i0fac)}))

    pd.DataFrame(Emix2, index=i0fac_vals, columns=gfac_vals).to_csv(out_dir / f"heatmap_gfac_vs_i0fac_Emix_{mode}.csv")
    pd.DataFrame(imix2, index=i0fac_vals, columns=gfac_vals).to_csv(out_dir / f"heatmap_gfac_vs_i0fac_imix_{mode}.csv")

    x_edges = make_edges(gfac_vals, "log"); y_edges = make_edges(i0fac_vals, "log")
    X, Y = np.meshgrid(x_edges, y_edges)

    for Z, name, cbarlab in [(Emix2, "Emix", "E_mix [V]"), (imix2, "imix", "i_mix [A/m^2]")]:
        plt.figure()
        plt.pcolormesh(X, Y, Z, shading="auto")
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("Cdl_Au multiplier (∝ g_Au/g_Pd)")
        plt.ylabel("it0_1 multiplier (∝ it0_1/it0_2)")
        plt.title(f"Heatmap ({mode}): {name}(g_Au/g_Pd, it0 ratio)")
        plt.colorbar(label=cbarlab)
        plt.tight_layout()
        plt.savefig(fig_dir / f"heatmap_gfac_vs_i0fac_{name}_{mode}.png", dpi=300)
        plt.close()


def compute_sensitivities(base_params: Dict[str, Any], out_dir: Path, mode: str, summary_rows: List[Dict[str, Any]], rel_step: float = 0.01) -> pd.DataFrame:
    """
    Normalized sensitivities:
        S_p^E = ∂ln(|E_mix|)/∂ln(p)
        S_p^I = ∂ln(i_mix)/∂ln(p)
    Also appends the +/- perturbed runs to results_summary rows.
    """
    mode = mode.upper()
    specs = make_ofat_specs(base_params)

    base_res = run_case(base_params, mode=mode, return_profiles=False)
    E0 = float(base_res["E_mix"]); I0 = float(base_res["i_mix"])

    rows: List[Dict[str, Any]] = []

    for pname in specs.keys():
        p0 = base_params.get(pname)
        if pname == "lambda_D" and (p0 is None):
            p0 = EDLModel(base_params).derived["lambda_D"]
        if p0 is None:
            continue
        p0 = float(p0)
        if p0 == 0.0:
            continue

        p_plus = p0 * (1.0 + rel_step)
        p_minus = p0 * (1.0 - rel_step)
        if pname in ("alpha1", "alpha2"):
            p_plus = float(np.clip(p_plus, 0.01, 0.99))
            p_minus = float(np.clip(p_minus, 0.01, 0.99))

        def eval_case(tag: str, pval: float) -> Tuple[float, float]:
            pvar = copy.deepcopy(base_params)
            if pname == "lambda_D":
                pvar["lambda_D"] = float(pval)
            else:
                pvar[pname] = float(pval)
                if base_params.get("lambda_D") is None:
                    pvar["lambda_D"] = None
            res = run_case(pvar, mode=mode, return_profiles=False)
            summary_rows.append(make_summary_row(run_tag=tag, params=pvar, result=res, extra={"sens_param": pname}))
            return float(res["E_mix"]), float(res["i_mix"])

        E_plus, I_plus = eval_case(f"sens:+:{pname}", p_plus)
        E_minus, I_minus = eval_case(f"sens:-:{pname}", p_minus)

        def ln_abs(x: float) -> float:
            if not np.isfinite(x) or x == 0.0:
                return float("nan")
            return math.log(abs(x))

        denom = math.log(abs(p_plus)) - math.log(abs(p_minus))
        S_E = (ln_abs(E_plus) - ln_abs(E_minus)) / denom if denom != 0 else float("nan")
        S_I = (math.log(I_plus) - math.log(I_minus)) / denom if (I_plus > 0 and I_minus > 0 and denom != 0) else float("nan")

        rows.append(dict(param=pname, p0=p0, E_mix_0=E0, i_mix_0=I0, S_E=S_E, S_I=S_I))

    df = pd.DataFrame(rows)
    df["abs_S_E"] = df["S_E"].abs()
    df["abs_S_I"] = df["S_I"].abs()
    df = df.sort_values(by=["abs_S_E", "abs_S_I"], ascending=False)

    df.to_csv(out_dir / "sensitivities.csv", index=False)

    fig_dir = ensure_dir(out_dir / "figures")
    for col in ("S_E", "S_I"):
        plt.figure(figsize=(10, 4))
        sub = df.head(20)
        plt.bar(sub["param"], sub[col])
        plt.xticks(rotation=60, ha="right")
        plt.ylabel(col)
        plt.title(f"Top-20 normalized sensitivities ({col}, mode={mode})")
        plt.tight_layout()
        plt.savefig(fig_dir / f"sensitivity_{col}_{mode}.png", dpi=300)
        plt.close()

    return df


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    params = default_params()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(Path("./results") / timestamp)
    ensure_dir(out_dir / "figures")

    summary_rows: List[Dict[str, Any]] = []

    # Baseline FULL + MEAN (required)
    case_full = run_case(params, mode="FULL", return_profiles=True)
    case_mean = run_case(params, mode="MEAN", return_profiles=False)

    summary_rows.append(make_summary_row("baseline:FULL", params, case_full))
    summary_rows.append(make_summary_row("baseline:MEAN", params, case_mean))

    # Save baseline profiles (FULL)
    R_gas = float(params["R"]); F = float(params["F"]); T = float(params["T"])
    scale = R_gas * T / F
    x_tilde = case_full["x_tilde"]
    x_m = x_tilde * case_full["lambda_D"]
    phi2_V = scale * case_full["phi_tilde"]

    np.savez(
        out_dir / "profiles.npz",
        x_tilde=x_tilde,
        x_m=x_m,
        phi_tilde=case_full["phi_tilde"],
        phi2_V=phi2_V,
        i1=case_full["i1"],
        i2=case_full["i2"],
        mask_Au=case_full["mask_Au"],
        mask_Pd=case_full["mask_Pd"],
        L_Au_tilde=case_full["L_Au_tilde"],
        L_C_tilde=case_full["L_C_tilde"],
        lambda_D=case_full["lambda_D"],
    )

    plot_baseline_profiles(case_full, params, out_dir)

    # Print comparison (required)
    df_cmp = pd.DataFrame([
        dict(mode="FULL", E_mix=case_full["E_mix"], i_mix=case_full["i_mix"], residual=case_full["residual"], method=case_full["method"]),
        dict(mode="MEAN", E_mix=case_mean["E_mix"], i_mix=case_mean["i_mix"], residual=case_mean["residual"], method=case_mean["method"]),
    ])
    print("\n=== Baseline comparison (FULL vs MEAN) ===")
    print(df_cmp.to_string(index=False))

    # OFAT
    if bool(params.get("do_ofat", True)):
        scan_mode = str(params.get("scan_mode", "MEAN")).upper()
        modes = ["MEAN", "FULL"] if scan_mode == "BOTH" else ["MEAN"]
        run_ofat(params, out_dir=out_dir, modes=modes, summary_rows=summary_rows)

    # Heatmaps
    if bool(params.get("do_heatmaps", True)):
        run_heatmaps(params, out_dir=out_dir, mode="MEAN", summary_rows=summary_rows)

    # Sensitivities
    if bool(params.get("do_sensitivities", True)):
        compute_sensitivities(params, out_dir=out_dir, mode="MEAN", summary_rows=summary_rows, rel_step=0.01)

    # Save master summary (required)
    pd.DataFrame(summary_rows).to_csv(out_dir / "results_summary.csv", index=False)

    print(f"\nAll results saved under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
