# Flush coplanar Au | insulating substrate | Pd model

This package implements Au and Pd electrodes embedded flush in the same ideal
insulating substrate.  It does **not** obtain this topology by assigning zero
capacitance or zero PZC to the carbon segment of the legacy Au|C|Pd solver.

Au and Pd are connected by an ideal wire outside the electrolyte.  They share
one electronic potential, `E_mix`, while their signed absolute currents obey

```text
I_Au + I_Pd = 0.
```

The reported common-area current density is
`abs(I_Au) / [W * (L_Au + L_Pd)]`; the substrate gap never contributes
reaction area.

The electrolyte occupies the semi-infinite strip above the coplanar surface,
`0 <= x <= L` and `y >= 0`.  At `y = 0`, the boundary is partitioned into a flush Au electrode,
an exposed ideal insulating-substrate gap, and a flush Pd electrode.  The Au
and Pd segments use Stern--Robin charging conditions and support their
respective reactions.  The substrate segment is uncharged and nonreactive,
with a homogeneous Neumann electrostatic condition.  The normal far-field
condition is `phi_tilde -> 0` as `y -> infinity`; the left and right
boundaries are reflecting.  There is no
electrolyte slot below the substrate plane and no vertical metal sidewall.
When `d_Au_Pd = 0`, the exposed substrate segment disappears exactly.

The reflecting lateral boundaries represent a symmetry/periodic stripe-cell
idealization.  An isolated finite electrode pair would require added lateral
electrolyte padding and a different outer-boundary study.

## First-version physics

- linearized Poisson--Boltzmann / Debye--Huckel electrostatics;
- Stern charging on the coplanar Au and Pd interfaces;
- an ideal, uncharged insulating-substrate segment between the electrodes;
- Frumkin-corrected irreversible Butler--Volmer kinetics on Au and Pd only;
- an exact semi-infinite far field and reflecting lateral boundaries;
- no nonlinear PB, ion transport, diffusion limit, solution ohmic drop,
  wire resistance, dielectric polarization, fixed substrate charge, or
  substrate reaction.

The dimensionless electrolyte problem is

```text
-Laplacian(phi_tilde) + phi_tilde = 0,
```

solved on the semi-infinite strip with the same cosine/Fourier Galerkin
construction used by the legacy `Solve_Emix_updating.py` solver,

```text
phi_tilde(x,y) = sum_n A_n cos(rho_n x) exp(-gamma_n y),
rho_n = n*pi/L,  gamma_n = sqrt(1 + rho_n^2).
```

The insulating gap enters the projected Robin system as an exact `g(x)=0`
segment and therefore contributes neither a charging-matrix term nor a PZC
load.  The projected matrix is factorized once for each separation.  Its two
affine coefficient vectors give
`A(E) = beta*E*A_M - A_PZC`, so mixed-potential iterations only recombine
precomputed spectral fields.  There is no finite-height top boundary in the
production model; the earlier finite-height `H = 10 lambda_D` Q1 domain was a truncation
approximation, not the present physical far-field condition.

## Install and run

From this directory, using the project macOS environment, the package can be
run directly from its source tree:

```bash
PYTHONPATH=src ../.venv_macos/bin/python -m au_pd_edl.cli single --d-nm 2
PYTHONPATH=src ../.venv_macos/bin/python -m au_pd_edl.cli scan
```

The dedicated, non-overwriting 1000 nm high-resolution figure bundle is
generated from the saved `20260802_144648` physical parameters with:

```bash
MPLCONFIGDIR=/private/tmp/au_pd_mpl PYTHONPATH=src \
  ../.venv_macos/bin/python make_d1000nm_figure_set.py
```

The default output is
`results/20260802_144648/figures/d1000nm_high_resolution/`. It uses
`N_modes=3840`, `Nx=20001`, records the refinement/ringing audit, and renders
canonical Figure 3 spatial panels b--e with a broken x axis spanning
`0--60 nm` and `990--1050 nm`; panels a/f are unchanged. The 2D maps use the
same two windows.

To redraw only Figure 3 b--e in that existing bundle without rerunning the
solver, use the same command with `--refresh-figure3-only`.

An editable install with `python -m pip install -e .` is optional when the
active environment contains the declared build requirements.

The full scan includes 0--10 nm in 0.5 nm steps; 15, 20, 30, 50, 75, and
100 nm; and exact `lambda_D`, `2 lambda_D`, `3 lambda_D`, and `5 lambda_D`
locations.  It retains the full fields at 0, 2, 3, and 10 nm and exports the
four comparison-safe RP figures.

Public Python interfaces are:

```python
from au_pd_edl import (
    solve_case,
    run_edl_comparison_pair,
    compute_polarization_curve,
    compute_gouy_chapman_diagnostics,
    build_gouy_chapman_analysis,
    run_separation_scan,
    save_gouy_chapman_outputs,
    save_run,
    run_numerical_validation,
)
```

## Parameter topology

Use `L_Au`, `d_Au_Pd`, `L_Pd`, `C_H_Au`, and `C_H_Pd`.
`d_Au_Pd` is the width of exposed insulating substrate between the two flush
metal electrodes, not the width of a lower electrolyte slot.  The package
rejects `L_gap`, `Cdl_C`, `C_H_C`, `g_C`, and `pzc_C` explicitly, even if
their value is zero.  SI units are used throughout; `C_tot = 10` means
10 mol/m3 = 10 mM.

Standalone solves accept `0 <= d_Au_Pd <= 1000 nm`.  The standard overlap
scan intentionally remains restricted to 0--100 nm so its validated 100 nm
plateau reference and output contract do not change.  Long-gap standalone
calculations require an explicit spectral-resolution check because the same
number of cosine modes spans a wider domain.

The production spectral controls are `N_modes = 960` and `Nx = 5000`.
`N_modes` truncates the coefficient vector at `n = N_modes`; `Nx` controls
surface quadrature and saved profile resolution.  Finite-top and Q1 spacing
controls are intentionally absent because the physical far field is imposed
analytically at infinity.

The baseline is traced to
`../../Figures/Figure_same_length_i0_alpha/inputs/params_same_length_i0_alpha050_au25_pd25_20260528_111255.json`
and uses 25 nm Au/Pd, equal exchange currents, alpha = 0.5, 10 mM electrolyte,
and 1 cm out-of-plane width.  Carbon/support charging keys are deliberately
absent from `params_template.json` because the exposed substrate is treated
as an ideal uncharged insulator.

## Outputs

Each new scan result contains parameter/source snapshots, derived quantities,
spectral and software metadata, JSON/CSV summaries, the complete separation
CSV, full spectral-coefficient NPZ fields, regular display-grid NPZ/CSV data,
RP/local-current
profiles, and PNG/SVG figures.  No PDF is generated.

`run_manifest.json`, `summary.json`, and `artifacts.json` use result schema
version 2 and identify the electrostatic backend as
`semi_infinite_cosine_fourier`.  `save_run(...)` accepts only a new or empty
output directory and refuses to overwrite any non-empty directory.  Standalone
Figure 3 export writes its own figure metadata but never edits an existing run
manifest; its loader rejects legacy or mismatched result-schema/backend
metadata before reading numerical CSV files.  Validation and Gouy--Chapman
analysis JSON files also carry the same result-schema/backend identity (while
the latter retains its separate analysis-document `schema_version = 1`).

Each `fields/spectral_metadata_*.json` distinguishes the internally augmented
diagnostic surface sampling (`diagnostic_surface_n_x_coordinates`) from the
uniform surface arrays saved in the matching `spectral_field_*.npz`
(`saved_surface_n_x_coordinates`).  The retained `n_x_coordinates` field refers
only to those saved arrays and equals the lengths of `surface_x_tilde` and
`surface_phi_tilde`.

`delta_E_total_EDL_V` and `delta_i_total_EDL_A_per_m2` compare with EDL against
the w/o-EDL reference.  `delta_E_overlap_vs_100nm_V` and
`delta_i_overlap_vs_100nm_A_per_m2` isolate the additional proximity effect
relative to the 100 nm with-EDL field.

Every full scan exports both the original 0--100 nm trend figure and a
same-style 0--20 nm zoom.  The zoom preserves the original y scales and the
w/o-EDL and 100 nm reference lines.  Scan output also includes
`csv/gouy_chapman_length_vs_separation.csv` and
`gouy_chapman_analysis.json`.

Each full scan also exports six Figure-3-style PNG/SVG panels under
`figures/Figure_3/`.  These panels use `d_Au_Pd = 10 nm`, giving a
length-matched 25|10|25 nm comparison against the legacy Au|C|Pd Figure 3.
Panels b/c show the solution potential and concentration along the coplanar
`y = 0` metal/substrate boundary.  Reaction-only quantities in panels d/e
are undefined and left blank over the insulating substrate.  The potential
reference map contains only the Au and Pd PZCs and never introduces a
substrate/support PZC.

## Gouy--Chapman diagnostic

For each metal interface, the metal-side Stern charge is evaluated from the
same Robin boundary condition used by the spectral model,

```text
sigma_M(x) = C_H,M_eff * [E_mix - PZC_M - phi_s(x, 0)],
C_H,M_eff = g_M * epsilon_s / lambda_D.
```

Positive `sigma_M` denotes positive charge on the metal.  For the symmetric
1:1 electrolyte, the reported Gouy--Chapman diagnostic is

```text
l_GC = 2 * epsilon_s * R * T / (F * abs(sigma)).
```

Because charge is laterally nonuniform, the primary material value uses the
boundary-length average of `abs(sigma)`.  Local minimum/maximum lengths and
zero-crossing flags are reported separately.  A zero surface charge is stored
as a null length plus an explicit infinite flag; the w/o-EDL control is marked
not applicable rather than being assigned a fictitious Stern charge.

The source paper applies `l_GC` to the charge density of a planar support in
the space between supported nanoparticles (`../../MS/cs5c06754.pdf`).  In the
present model the substrate gap is prescribed as an ideal uncharged boundary,
so the Gouy--Chapman calculation applies only to Au and Pd and does not define
a substrate `l_GC`.  Moreover, the linearized-PB operator contains `lambda_D`
as its explicit screening length.  Therefore the calculated metal `l_GC` is
a surface-charge-derived diagnostic for comparison with the separation trend,
not an independent or unique overlap cutoff in this first-version model.

## Interpretation caveat

Every result reports `max_abs_phi_tilde`.  If it exceeds 1, the linearized PB
assumption is outside its nominal small-potential regime.  Such output is
labelled as an internal geometry-sensitivity calculation and is not claimed
as an absolute quantitative prediction.  Making the electrodes flush changes
the geometric boundary-value problem but does not, by itself, cure this
linearization limitation.

`results/20260802_024804/` is a frozen v0.2.0 finite-height Q1/H=`10 lambda_D`
result.  It is not overwritten or silently relabelled by the v0.3.0 spectral
backend.  A newly generated result directory is required before spectral and
finite-height Q1 numbers are compared in a controlled archival regression.

`results/20260801_175012/` is retained only as a legacy T-domain/open-slot
historical result.  It is not the current flush-substrate result and is not
modified by this geometry change; see `result_parameter_memory.md` for its
archived numerical record.
