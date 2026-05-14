# Methods Note for `Solve_Emix_updating.py`

This note summarizes the current linear-workflow implementation in
`Solve_Emix_updating.py` and is written as a manuscript-facing methods record
for the present compare, OFAT, and heatmap figures. It reflects the current
baseline parameter set, output conventions, and figure-generation logic in the
active code.

## Model Scope

The model couples:

1. a laterally heterogeneous Au-support-Pd electrostatic problem,
2. Frumkin-corrected interfacial kinetics on the reactive Au and Pd segments,
3. a mixed-potential condition obtained by enforcing zero net current.

The support region is treated as an electrostatic segment with its own
capacitance and potential of zero charge, but it does not carry a Faradaic
reaction in the current implementation. Accordingly, the mixed potential is
determined by the Au and Pd current balance, while the support modifies the
solution-side reaction-plane potential through lateral electrostatic coupling.

The current shipped solver is the linear Debye-Huckel implementation. The
`with EDL` result always refers to the fully spatially resolved `FULL`
numerical solution, whereas the `without EDL` result refers to the comparison
model in which the solution-side electrostatic potential is set to zero.

## Baseline Parameter Set

### Electrolyte and medium

| Quantity | Code key | Value |
| --- | --- | ---: |
| Electrolyte concentration | `C_tot` | `10.0 mol m^-3 = 10 mM = 0.010 M` |
| pH | `pH` | `7.0` |
| Reference pH | `pH_ref` | `7.0` |
| Temperature | `T` | `298.0 K` |
| Relative permittivity | `epsilon_r` | `78.5` |
| Out-of-plane width | `out_of_plane_width` | `1.0 m` |

In the current code convention, `C_tot` denotes the concentration of each ionic
species in a symmetric 1:1 electrolyte. Thus an experimental electrolyte
concentration of `0.1 M` corresponds to `C_tot = 100 mol m^-3` in the solver
input.

### Geometry

| Quantity | Code key | Value |
| --- | --- | ---: |
| Au length | `L_Au` | `11 nm` |
| Support length | `L_gap` | `10 nm` |
| Pd length | `L_Pd_len` | `37 nm` |

### Interfacial capacitances

Capacitances are stored internally in `F m^-2` but are reported throughout the
figures and tables in `uF cm^-2`.

| Quantity | Code key | Internal value | Display value |
| --- | --- | ---: | ---: |
| Au double-layer capacitance | `Cdl_Au` | `0.23 F m^-2` | `23.0 uF cm^-2` |
| Support capacitance | `Cdl_C` | `0.10 F m^-2` | `10.0 uF cm^-2` |
| Pd double-layer capacitance | `Cdl_Pd` | `0.377 F m^-2` | `37.7 uF cm^-2` |

Useful conversion:

- `1 F m^-2 = 100 uF cm^-2`
- `1 uF cm^-2 = 0.01 F m^-2`

### Potentials and kinetic parameters

| Quantity | Code key | Value |
| --- | --- | ---: |
| Au potential of zero charge | `pzc_Au` | `0.513 V` |
| Support potential of zero charge | `pzc_C` | `0.361 V` |
| Pd potential of zero charge | `pzc_Pd` | `0.371 V` |
| Au equilibrium potential | `E1_eq` | `0.100 V` |
| Pd equilibrium potential | `E2_eq` | `0.834 V` |
| Au exchange current density | `it0_1` | `8.85e-5 A m^-2` |
| Pd exchange current density | `it0_2` | `3.878e-4 A m^-2` |
| Au transfer coefficient | `alpha1` | `0.5` |
| Pd transfer coefficient | `alpha2` | `0.37` |

The baseline pH-dependent parameterization is referenced to `pH_ref = 7.0`.
Reaction 2 is treated as a proton-coupled half reaction of the form
`H+ + e- -> 1/2 H2`, with:

- `E2_eq_pH_slope_V_per_pH = -(2.303 RT / F)`
- `it0_2_pH_order = 1.0`

Reaction 1 is currently pH-independent in the shipped defaults.

## Derived Quantities

The code computes:

- the solution permittivity `epsilon_s = epsilon_r * epsilon0`,
- the Debye length `lambda_D`,
- the dimensionless lateral lengths `L_tilde = L / lambda_D`,
- the dimensionless electrostatic coupling parameters
  `g_i = (lambda_D / epsilon_s) Cdl_i`,
- the dimensionless potentials of zero charge
  `pzc_tilde = (F / RT) * pzc`.

These quantities are assembled in `compute_derived_params()` before the
electrostatic solve.

## Electrostatic Model

The `with EDL` solution uses a linear Debye-Huckel treatment of the
electrostatic field in a laterally heterogeneous Au-support-Pd domain. The
problem is solved in two dimensions using a cosine spectral expansion in the
lateral coordinate and Robin boundary conditions at the reaction plane. The
three segments differ through their geometry, capacitance, and potential of
zero charge.

The solution-side reaction-plane potential is written in the code as
`phi_tilde(x)` in dimensionless form and is converted to volts as:

`phi_RP(x) = (RT/F) * phi_tilde(x)`

In the 2026 main solver, this reaction-plane potential is evaluated at the
surface boundary (`y_tilde = 0`). The root-level prototype
`edl_mixed_potential_model.py` supports a finite reaction-plane offset, but
that offset is not part of the present production workflow.

In the `without EDL` comparison model:

- `phi_tilde(x) = 0`
- `phi_RP(x) = 0`

everywhere by construction.

## Kinetic Model and Mixed-Potential Condition

Local Faradaic kinetics are evaluated only on the Au and Pd segments. The
support segment does not contribute a Faradaic current in the current model.

For a given trial metal potential `E`, the `FULL` model evaluates the local
currents from the spatially resolved reaction-plane potential and integrates
them over the reactive Au and Pd segments to obtain:

- `I_Au(E)`
- `I_Pd(E)`

The mixed potential is then defined by the zero-net-current condition:

`I_Au(E_mix) + I_Pd(E_mix) = 0`

For the `without EDL` comparison case, the same current-balance condition is
solved with `phi_RP(x) = 0`.

Closed-form mixed-potential shortcuts are not used in the present compare
workflow. The active compare solution uses the spatially resolved `FULL`
numerical branch.

## pH Handling

When pH dependence is enabled, the current code applies:

`E_eq(pH) = E_eq + slope * (pH - pH_ref)`

`i0(pH) = i0 * 10^(-order * (pH - pH_ref))`

The default pH scans are therefore fixed-ionic-strength scans: changing `pH`
does not automatically change `C_tot` unless `C_tot` is explicitly scanned at
the same time.

## Current Definition and Reporting Convention

The internal mixed-current balance is first evaluated as the integrated current
over the dimensionless lateral coordinate:

`I_norm = int i(x) d x_tilde`

The code also retains two auxiliary derived quantities:

- `I_phys = lambda_D * I_norm` with units `A m^-1`
- `I_abs = out_of_plane_width * I_phys` with units `A`

However, the figures in the current workflow do not use absolute current as the
primary reported quantity. Instead, they use the average mixed current density
over the total reactive Au+Pd area:

`A_reactive = (L_Au + L_Pd_len) * out_of_plane_width`

`bar(i_mix) = i_mix_abs / A_reactive`

with units `A m^-2`.

This choice is physically preferable when `L_Au` or `L_Pd` are varied, because
it removes the trivial increase in absolute current that would arise from
increasing reactive length alone.

The same normalization is used for the compare polarization curves:

`bar(I_net)(E) = I_net_abs(E) / A_reactive`

## Baseline Compare Figures

The current compare workflow generates:

- `compare_polcurve_<tag>.png`
- `compare_emix_imix_<tag>.png`
- `compare_phi2_<tag>.png`
- `compare_potentials_overpotential_<tag>.png`

and the publication-style summary panel is generated separately from the same
underlying compare data.

For the baseline compare workflow:

1. the baseline parameter dictionary is assembled,
2. the `with EDL` case is solved using `mode="FULL", use_edl=True`,
3. the `without EDL` case is solved using `mode="FULL", use_edl=False`,
4. both solutions are compared at the same geometry and kinetic parameter set.

The default compare polarization curve is sampled in a local potential window:

- `min(E_mix_with, E_mix_without) - 0.10 V`
- to `max(E_mix_with, E_mix_without) + 0.10 V`

using `200` points.

The `compare_emix_imix` figure reports:

- `E_mix [V]`
- `bar(i_mix) [A m^-2]`

The `compare_phi2` figure reports `phi_RP(x)`, not the metal potential.
The metal potential itself is shown separately in the multi-panel compare
figure. The local overpotential is defined piecewise on the reactive segments
as:

`eta_Au(x) = E_mix - E1_eq_eff - phi_RP(x)`

`eta_Pd(x) = E_mix - E2_eq_eff - phi_RP(x)`

where `E1_eq_eff` and `E2_eq_eff` denote the pH-adjusted effective equilibrium
potentials when pH dependence is active.

## OFAT Workflow

The OFAT workflow is one-factor-at-a-time. At each scan point, a single
parameter is changed while all other baseline inputs are held fixed, and the
same `with EDL` versus `without EDL` comparison is repeated.

Current default settings:

- `ofat_n = 15`
- `ofat_L_gap_n = 15`

The current code uses:

- logarithmic scans for `C_tot`, `L_Au`, `L_Pd_len`, `lambda_D`, `Cdl_*`,
  and `it0_*`,
- linear scans for `pH`, `epsilon_r`, `T`, `pzc_*`, `alpha*`, `E*_eq`,
  `z_*`, and `L_gap`.

Current default OFAT ranges include:

- `C_tot`: `0.1 mM` to `10 M`
- `pH`: `0` to `14`
- `L_gap` (displayed as support length): `0` to `1000 nm`

Each OFAT parameter writes:

- `csv/ofat_compare_<pname>.csv`
- `figures/ofat_compare_<pname>_E_mix.png`
- `figures/ofat_compare_<pname>_i_mix_avg_A_per_m2.png`

The OFAT CSV stores:

- scanned parameter value,
- effective pH-adjusted kinetic and thermodynamic quantities,
- `E_mix` with and without EDL,
- average mixed current density with and without EDL,
- `delta_E_mix`,
- `delta_i_mix_avg_A_per_m2`,
- `ratio_i_mix_avg`,
- `pct_i_mix_avg`,
- `max_abs_phi_tilde`,
- Debye-Huckel validity flag.

The plotted axes use chemistry-style display units where appropriate:

- concentration in `M`,
- lengths in `nm`,
- capacitances in `uF cm^-2`.

## Heatmap Workflow

Only the new Au-Pd coupled heatmaps are active in the current code. The legacy
support-focused heatmaps are intentionally disabled in the active workflow.

The current heatmap families are:

1. `Cdl_Au × Cdl_Pd`
2. `L_Au × L_Pd`
3. `pzc_Au × pzc_Pd`

The `Cdl_Au × Cdl_Pd` heatmap uses `5` to `100 uF cm^-2` on both axes.
The log combined panel uses logarithmic sampling/display; the linear combined
panel uses linear sampling/display.

The `L_Au × L_Pd` heatmap now uses:

- `2 nm` to `1000 nm` on both axes,
- logarithmic sampling/display in the log combined panel,
- linear sampling/display in the linear combined panel.

The `pzc_Au × pzc_Pd` heatmap uses linear scans around the baseline values.

The active heatmap figure outputs are:

- `heatmap_combined_panel_log.png`
- `heatmap_combined_panel_linear.png`

The heatmap potential panels are plotted in `mV`; the CSV files remain in
solver units.

The corresponding CSV files are written under `csv/` and include:

- `Emix_with_edl_FULL`
- `Emix_without_edl`
- `delta_Emix`
- `imix_avg_with_edl_FULL`
- `imix_avg_without_edl`
- `delta_i_mix_avg`

Supplementary CSV-only quantities retained for downstream analysis are:

- `log10_imix_avg_with_edl_FULL`
- `log10_imix_avg_without_edl`
- `delta_log10_i_mix_avg`
- `ratio_i_mix_avg`

Each heatmap marks the baseline parameter set explicitly in the plotted domain.
The current implementation uses per-figure color scales rather than forcing a
single shared color scale across all heatmap families.

## Output Structure

The main workflow writes results under:

`results/<timestamp>/`

with the following structure:

- `figures/` for PNG, PDF, HTML, or SVG figures
- `csv/` for CSV exports
- `profiles.npz` for baseline profiles
- `params.json` or case-level parameter snapshots where applicable

For the compare helper workflow:

`results/compare_demo/<timestamp>/`

contains:

- `figures/`
- `csv/summary_compare.csv`
- `with_edl/`
- `no_edl/`
- `params.json`

For the main workflow:

`results/<timestamp>/csv/results_summary.csv`

collects the baseline, OFAT, heatmap, and sensitivity summary rows into one
master table.

## Reproducibility

The baseline linear workflow can be regenerated with:

```bash
python3 -m pip install -r requirements.txt
python3 Solve_Emix_updating.py
```

The compare helper workflow can be regenerated with:

```bash
python3 run_compare.py
```

The publication-style compare panel can be regenerated from the latest compare
run with:

```bash
python3 make_latest_publication_compare_panels.py
```

## Model-Validity Note

The current electrostatic solver is the linear Debye-Huckel version. For the
present baseline parameter set, the `with EDL` solution yields:

- `max_abs_phi_tilde > 1`

and therefore exceeds the nominal strict small-potential regime associated with
the linear Debye-Huckel approximation. The current figures should therefore be
interpreted as outputs of the present linearized model under the specified
parameter set, rather than as a strictly small-potential Debye-Huckel limit.
