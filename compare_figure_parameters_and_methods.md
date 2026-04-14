# Compare Figure Parameters and Methods

This document records the effective parameter set and plotting procedure used by the current compare workflow in `run_compare.py`. The purpose of this note is to provide a manuscript-ready description for the compare figures generated from the present code configuration.

## Scope

The current compare workflow generates the following figure set for a given parameter set:

- `compare_polcurve_<tag>.png`
- `compare_emix_imix_<tag>.png`
- `compare_phi2_<tag>.png`
- `compare_potentials_overpotential_<tag>.png`

For the present baseline configuration, the geometry tag is:

- `LAu11_Lgap10_LPd37`

These figures are produced by the following workflow:

1. `default_params()` defines the baseline parameter dictionary.
2. `params_template.json` is loaded and applied as an override.
3. `compare_edl_effects(..., solver_settings={"mode": "FULL"})` is executed.
4. The `with EDL` result is always the fully spatially resolved numerical solution (`mode="FULL", use_edl=True`).
5. The `without EDL` result is the corresponding no-EDL comparison model (`mode="FULL", use_edl=False`).

## Effective Input Parameters Used for the Compare Figures

The following values are the actual input parameters used by the current compare workflow after applying `params_template.json` on top of `default_params()`.

### Physical Constants and Medium

| Code key | Value | Unit | Meaning |
| --- | ---: | --- | --- |
| `R` | 8.314 | J mol^-1 K^-1 | Gas constant |
| `F` | 96485.0 | C mol^-1 | Faraday constant |
| `T` | 298.0 | K | Temperature |
| `epsilon0` | 8.8541878128e-12 | F m^-1 | Vacuum permittivity |
| `epsilon_r` | 78.5 | - | Relative permittivity |
| `epsilon_s` | auto | F m^-1 | Solution permittivity, computed as `epsilon_r * epsilon0` |
| `C_tot` | 0.1 | mol m^-3 | Concentration parameter used in the Debye-length expression |

### Geometry

| Code key | Value | Unit | Meaning |
| --- | ---: | --- | --- |
| `L_Au` | 1.1e-08 | m | Au segment length |
| `L_gap` | 1.0e-08 | m | Gap length |
| `L_Pd_len` | 3.7e-08 | m | Pd segment length |
| `out_of_plane_width` | 1.0 | m | Assumed out-of-plane width used to convert line current to absolute current |

### Interfacial Electrostatics

| Code key | Value | Unit | Meaning |
| --- | ---: | --- | --- |
| `Cdl_Au` | 1.20e-4 | F m^-2 | Au double-layer capacitance |
| `Cdl_C` | 1.0 | F m^-2 | Gap-region capacitance proxy used in the current model |
| `Cdl_Pd` | 3.77e-5 | F m^-2 | Pd double-layer capacitance |
| `g_Au` | auto | - | Computed from `Cdl_Au` |
| `g_C` | auto | - | Computed from `Cdl_C` |
| `g_Pd` | auto | - | Computed from `Cdl_Pd` |
| `pzc_Au` | 0.513 | V | Potential of zero charge on Au |
| `pzc_C` | 0.361 | V | Potential of zero charge on the gap region |
| `pzc_Pd` | 0.371 | V | Potential of zero charge on Pd |

### Kinetics

| Code key | Value | Unit | Meaning |
| --- | ---: | --- | --- |
| `it0_1` | 8.85e-5 | A m^-2 | Exchange current density for the Au half-reaction |
| `it0_2` | 3.878e-4 | A m^-2 | Exchange current density for the Pd half-reaction |
| `alpha1` | 0.5 | - | Charge-transfer coefficient for the Au half-reaction |
| `alpha2` | 0.37 | - | Charge-transfer coefficient for the Pd half-reaction |
| `z_R1` | -1.0 | - | Effective Frumkin charge factor for the Au reaction |
| `z_O2` | 1.0 | - | Effective Frumkin charge factor for the Pd reaction |
| `E1_eq` | 0.1 | V | Equilibrium potential for the Au half-reaction |
| `E2_eq` | 0.834 | V | Equilibrium potential for the Pd half-reaction |

### Numerical and Compare-Plot Controls

| Code key | Value | Unit | Meaning |
| --- | ---: | --- | --- |
| `N_modes` | 80 | - | Number of cosine modes in the EDL spectral expansion |
| `Nx` | 1200 | - | Number of surface grid points used for profiles and segment integration |
| `xtol` | 1.0e-10 | V | Root-finding tolerance for `E_mix` |
| `max_bracket_expands` | 12 | - | Maximum bracket expansion count in the root finder |
| `use_affine_phi2` | `true` | - | Enabled in the parameter set, but inactive for the `with EDL (FULL)` compare solution |
| `dh_warn_threshold` | 1.0 | - | Warning threshold for `max |phi_tilde|` |
| `dh_violation_action` | `warn` | - | Action used when the Debye-Huckel warning threshold is exceeded |
| `mode` | `FULL` | - | Compare figures are generated in `FULL` mode |
| `n_E` | 200 | - | Number of sampled points for the compare polarization curve |
| `E_window_halfspan` | 0.10 | V | Half-width of the default local polarization window around the two `E_mix` values |

## Derived Quantities Used by the Current Baseline Compare Figures

The following values are computed internally from the input parameter set before the compare figures are generated.

| Quantity | Value | Unit | Note |
| --- | ---: | --- | --- |
| `epsilon_s` | 6.950537433048001e-10 | F m^-1 | `epsilon_r * epsilon0` |
| `lambda_D` | 3.041217889017721e-08 | m | Debye length used in the current run |
| `g_Au` | 5.250617670899842e-03 | - | Dimensionless interfacial coupling on Au |
| `g_C` | 4.375514725749868e+01 | - | Dimensionless interfacial coupling in the gap region |
| `g_Pd` | 1.6495690516077001e-03 | - | Dimensionless interfacial coupling on Pd |
| `pzc_Au_tilde` | 19.977948168610236 | - | Dimensionless PZC on Au |
| `pzc_C_tilde` | 14.058556118651646 | - | Dimensionless PZC in the gap region |
| `pzc_Pd_tilde` | 14.447989806148923 | - | Dimensionless PZC on Pd |
| `L_Au_tilde` | 0.3616972016284198 | - | Dimensionless Au length |
| `L_gap_tilde` | 0.328815637844018 | - | Dimensionless gap length |
| `L_Pd_tilde` | 1.2166178600228668 | - | Dimensionless Pd length |
| `L_tilde` | 1.9071306994953048 | - | Dimensionless total length |

## Solved Quantities Used by the Present Compare Figures

The compare figures are drawn from the following baseline solutions:

| Quantity | With EDL (`FULL`) | Without EDL | Unit |
| --- | ---: | ---: | --- |
| `E_mix` | 0.44349454423603574 | 0.49157212957749513 | V |
| `i_mix_norm` | 0.07038215501102128 | 0.06555361211259984 | A m^-2 |
| `i_mix_phys` | 2.140474688871362e-09 | 1.993628178465674e-09 | A m^-1 |
| `i_mix_abs` | 2.140474688871362e-09 | 1.993628178465674e-09 | A |

Additional compare metrics used in the current baseline:

| Quantity | Value | Unit |
| --- | ---: | --- |
| `delta_E_mix` (`with EDL - without EDL`) | -4.8077585341459395e-02 | V |
| `delta_i_mix_abs` | 1.4684651040568797e-10 | A |
| `ratio_i_mix_abs` | 1.0736579227720904 | - |
| `max_abs_phi_tilde` in the `with EDL (FULL)` solution | 3.0596557118175287 | - |

For the default local compare polarization plot, the sampled potential window is:

- `E = 0.34349454423603576` to `0.5915721295774952 V`

This window is constructed as:

- `min(E_mix_with, E_mix_without) - 0.10 V`
- `max(E_mix_with, E_mix_without) + 0.10 V`

with `200` uniformly spaced points.

## Figure Generation Method

### 1. With-EDL solution

The `with EDL` result is obtained from the fully spatially resolved EDL model (`mode="FULL"`). The code solves the linear Debye-Huckel electrostatic boundary-value problem over a laterally heterogeneous Au-gap-Pd domain using a cosine spectral expansion. The reaction-plane potential field `phi2(x)` is then converted from the dimensionless electrostatic potential as:

`phi2(x) = (RT/F) * phi_tilde(x)`

Local kinetic currents are evaluated with the spatially resolved Frumkin-corrected kinetic expressions on the reactive Au and Pd segments, and the mixed potential is obtained by solving the current-balance condition:

`I_Au(E) + I_Pd(E) = 0`

where `I_Au` and `I_Pd` are the segment-integrated currents in the current code.

### 2. Without-EDL comparison solution

The `without EDL` result uses the same geometry and kinetic parameters, but disables the EDL electrostatic solve. In this comparison model:

- `phi_tilde(x) = 0` everywhere
- `phi2(x) = 0` everywhere

The no-EDL mixed potential is then obtained from the same current-balance condition, but without the EDL-induced electrostatic correction.

### 3. Current definitions and unit conversion

The internal `FULL` current balance is performed using the x-integrated current written in terms of the dimensionless lateral coordinate `x_tilde = x / lambda_D`:

`I_norm = int i(x) d x_tilde`

This quantity carries units of `A m^-2` in the current code output. It is then converted as follows:

- `I_phys = lambda_D * I_norm`, with units `A m^-1`
- `I_abs = out_of_plane_width * I_phys`, with units `A`

For the present compare figures:

- `out_of_plane_width = 1.0 m`

Therefore, the plotted `i_mix [A]` and `I_net [A]` correspond numerically to the line current obtained under a unit-width out-of-plane assumption. If a physical device width `W_phys` is to be used, the plotted currents should be rescaled linearly by `W_phys / 1.0`.

## Figure-Specific Definitions

### `compare_polcurve_<tag>.png`

This figure plots the net current:

`I_net(E) = I_Au(E) + I_Pd(E)`

for `with EDL (FULL)` and `without EDL` over the local compare window described above. The dashed vertical lines mark the corresponding `E_mix` values of the two models.

### `compare_emix_imix_<tag>.png`

This figure is a bar comparison of:

- `E_mix [V]`
- `i_mix [A]`

where `i_mix` is the absolute current reported after conversion with `out_of_plane_width = 1.0 m`.

### `compare_phi2_<tag>.png`

This figure plots the solution-side reaction-plane potential `phi2(x)` at the converged `E_mix` for each model. The x-axis is expressed in nanometres. In the no-EDL comparison model, `phi2(x)` is zero by construction; it is not equal to `E_mix`, because `E_mix` is the metal potential whereas `phi2(x)` is the solution-side reaction-plane electrostatic potential.

### `compare_potentials_overpotential_<tag>.png`

This figure contains three panels:

1. Metal potential, plotted as a constant `E_m(x) = E_mix` for each model.
2. Reaction-plane potential, plotted as `phi2(x)`.
3. Local overpotential, defined piecewise on the reactive segments as:

`eta_Au(x) = E_mix - E1_eq - phi2(x)`

`eta_Pd(x) = E_mix - E2_eq - phi2(x)`

The non-reactive gap segment is left undefined in the overpotential panel.

## Units and Reporting Conventions

- All input lengths are in metres.
- All plotted spatial coordinates are converted to nanometres.
- All potentials are reported in volts.
- Local interfacial current densities `i(x)` are in `A m^-2`.
- The compare polarization and mixed-current plots report `A`, based on `out_of_plane_width = 1.0 m`.
- In the current code, `C_tot` is interpreted as the concentration of each ionic species in a symmetric 1:1 electrolyte.
- Accordingly, an experimental electrolyte concentration of `0.1 M` would correspond to `C_tot = 100 mol m^-3` in the present code convention, not `0.1 mol m^-3`.

## Model-Validity Note for the Present Baseline

The current compare figures are generated with a linear Debye-Huckel EDL model. For the present baseline parameter set, the `with EDL (FULL)` solution gives:

- `max_abs_phi_tilde = 3.0596557118175287`

which exceeds the nominal small-potential condition usually associated with the linear Debye-Huckel approximation. Accordingly, the present compare figures should be described as the outputs of the current linearized model under the above parameter set, rather than as a strictly small-potential Debye-Huckel limit.

## Reproducibility

The current compare figure set can be regenerated with:

```bash
.venv/bin/python 2026/Mixed_Potential_Electrical_Double_Layer/run_compare.py
```

The outputs are written to:

`2026/Mixed_Potential_Electrical_Double_Layer/results/compare_demo/<timestamp>/`

including the parameter snapshot (`params.json`), the numeric summary (`summary_compare.csv`), and the compare figures listed above.
