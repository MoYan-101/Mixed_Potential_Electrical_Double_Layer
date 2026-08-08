# Result Parameter Memory

Purpose: record which saved result folder was generated from which parameter set.
Append a new section whenever a new result folder is used for figures, tables, or manuscript comparisons.

Last updated: 2026-06-22.

## 20260528_111255

- Result folder: [results/20260528_111255](results/20260528_111255/)
- Source parameter file: [results/20260528_111255/params.json](results/20260528_111255/params.json)
- Solver file: [Solve_Emix_updating.py](Solve_Emix_updating.py)
- Model family: linear Debye-Huckel EDL with mixed-potential kinetics
- Outputs enabled: baseline comparison, OFAT scans, heatmaps
- Outputs disabled: self-checks, convergence check, sensitivities

### Unit Conventions

| Quantity | Internal unit | Display/helper unit |
| --- | --- | --- |
| Lengths | m | nm |
| Potentials | V | V |
| Concentration `C_tot` | mol/m^3 | M, where 1 M = 1000 mol/m^3 |
| Capacitance `Cdl_*` | F/m^2 | uF/cm^2, where 1 F/m^2 = 100 uF/cm^2 |
| Current density `it0_*` | A/m^2 | A/m^2 |
| `lambda_D` | m | nm |
| Dimensionless `g_*` | dimensionless | dimensionless |

### Physical Constants And Electrolyte

| Parameter | Value in file | Unit | Human-readable value | Note |
| --- | ---: | --- | ---: | --- |
| `R` | 8.314 | J/mol/K | 8.314 J/mol/K | gas constant |
| `F` | 96485.0 | C/mol | 96485.0 C/mol | Faraday constant |
| `T` | 298.0 | K | 298.0 K | temperature |
| `epsilon0` | 8.8541878128e-12 | F/m | 8.8541878128e-12 F/m | vacuum permittivity |
| `epsilon_r` | 78.5 | dimensionless | 78.5 | relative permittivity |
| `epsilon_s` | null | F/m | auto = 6.950537433048001e-10 F/m | `epsilon_r * epsilon0` |
| `C_tot` | 10.0 | mol/m^3 | 0.010 M = 10 mM | symmetric 1:1 electrolyte, each ion |
| `lambda_D` | null | m | auto = 3.041217889017721e-9 m = 3.041217889 nm | computed from `C_tot`, `epsilon_s`, `R`, `T`, `F` |

### Geometry

| Parameter | Value in file | Unit | Human-readable value |
| --- | ---: | --- | ---: |
| `L_Au` | 1.1e-08 | m | 11.0 nm |
| `L_gap` | 1e-08 | m | 10.0 nm |
| `L_Pd_len` | 3.7e-08 | m | 37.0 nm |
| `L_total` | derived | m | 58.0 nm |
| `out_of_plane_width` | 0.01 | m | 1.0 cm |
| `L_Au_tilde` | derived | dimensionless | 3.616972016284198 |
| `L_C_tilde` | derived | dimensionless | 6.905128394724378 |
| `L_tilde` | derived | dimensionless | 19.071306994953044 |

### Interfacial EDL Parameters

| Parameter | Value in file | Unit | Human-readable value | Note |
| --- | ---: | --- | ---: | --- |
| `Cdl_Au` | 0.2 | F/m^2 | 20.0 uF/cm^2 | input capacitance |
| `Cdl_C` | 0.1 | F/m^2 | 10.0 uF/cm^2 | support capacitance |
| `Cdl_Pd` | 0.4 | F/m^2 | 40.0 uF/cm^2 | input capacitance |
| `g_Au` | null | dimensionless | auto = 0.8751029451499737 | from `lambda_D / epsilon_s * Cdl_Au` |
| `g_C` | null | dimensionless | auto = 0.43755147257498683 | from `lambda_D / epsilon_s * Cdl_C` |
| `g_Pd` | null | dimensionless | auto = 1.7502058902999473 | from `lambda_D / epsilon_s * Cdl_Pd` |
| `pzc_Au` | 0.93 | V | 0.93 V | same reference as `E_mix` and `E_eq` |
| `pzc_C` | 0.5 | V | 0.50 V | support PZC |
| `pzc_Pd` | 0.78 | V | 0.78 V | same reference as `E_mix` and `E_eq` |

### Kinetic Parameters

| Parameter | Value | Unit | Note |
| --- | ---: | --- | --- |
| `it0_1` | 8.85e-05 | A/m^2 | exchange current density for reaction 1 |
| `it0_2` | 0.0003878 | A/m^2 | exchange current density for reaction 2 |
| `alpha1` | 0.5 | dimensionless | transfer coefficient |
| `alpha2` | 0.37 | dimensionless | transfer coefficient |
| `z_R1` | -1.0 | dimensionless | reactant charge parameter |
| `z_O2` | 1.0 | dimensionless | reactant charge parameter |
| `E1_eq` | 0.1 | V | equilibrium potential for reaction 1 |
| `E2_eq` | 0.834 | V | equilibrium potential for reaction 2 |
| `pH` | 7.0 | dimensionless | baseline pH |
| `pH_ref` | 7.0 | dimensionless | pH reference for `E*_eq` and `it0_*` |
| `E1_eq_pH_slope_V_per_pH` | 0.0 | V/pH | pH correction for `E1_eq` |
| `E2_eq_pH_slope_V_per_pH` | -0.059126500015747985 | V/pH | pH correction for `E2_eq` |
| `it0_1_pH_order` | 0.0 | dimensionless | pH order for `it0_1` |
| `it0_2_pH_order` | 1.0 | dimensionless | pH order for `it0_2` |

### Solver And Model Switches

| Parameter | Value | Unit/allowed values | Meaning |
| --- | ---: | --- | --- |
| `N_modes` | 80 | count | cosine modes for linear EDL expansion |
| `Nx` | 1200 | count | surface grid points before exact boundary insertion |
| `xtol` | 1e-10 | V/root tolerance | mixed-potential root tolerance |
| `max_bracket_expands` | 12 | count | maximum bracket expansion attempts |
| `use_edl` | true | boolean | EDL-enabled baseline path |
| `use_affine_phi2` | true | boolean | use affine segment-mean phi2 shortcut for MEAN mode |
| `use_closed_form_when_affine` | true | boolean | allow closed-form MEAN-mode mixed potential |
| `dh_warn_threshold` | 1.0 | dimensionless | warning threshold for max abs(phi_tilde) |
| `dh_violation_action` | warn | ignore/warn/raise | action when Debye-Huckel threshold is exceeded |
| `do_self_checks` | false | boolean | optional self-check suite |
| `do_convergence_check` | false | boolean | optional mode/grid convergence check |
| `do_ofat` | true | boolean | one-factor-at-a-time scans enabled |
| `do_heatmaps` | true | boolean | heatmap scans enabled |
| `do_sensitivities` | false | boolean | local sensitivity scan disabled |
| `scan_mode` | BOTH | MEAN/FULL/BOTH | scan mode selection |
| `heatmap_mode` | BOTH | MEAN/FULL/BOTH | heatmap mode selection |
| `sensitivity_mode` | BOTH | MEAN/FULL/BOTH | sensitivity mode selection |

### OFAT Scan Settings

| Parameter | Value in file | Unit | Human-readable value |
| --- | ---: | --- | ---: |
| `ofat_n` | 15 | count | 15 |
| `ofat_L_gap_n` | 15 | count | 15 |
| `ofat_pH_min` | 0.0 | pH | 0.0 |
| `ofat_pH_max` | 14.0 | pH | 14.0 |
| `ofat_C_tot_min` | 0.1 | mol/m^3 | 0.0001 M = 0.1 mM |
| `ofat_C_tot_max` | 1000.0 | mol/m^3 | 1.0 M |
| `ofat_L_gap_min` | 0.0 | m | 0.0 nm |
| `ofat_L_gap_max` | 1e-06 | m | 1000.0 nm |

### Heatmap Scan Settings

| Parameter | Value in file | Unit | Human-readable value |
| --- | ---: | --- | ---: |
| `heatmap_nx` | 25 | count | 25 |
| `heatmap_ny` | 25 | count | 25 |
| `heatmap_C_tot_min` | 0.1 | mol/m^3 | 0.0001 M = 0.1 mM |
| `heatmap_C_tot_max` | 10000.0 | mol/m^3 | 10.0 M |
| `heatmap_L_min` | 2e-09 | m | 2.0 nm |
| `heatmap_L_max` | 1e-06 | m | 1000.0 nm |
| `heatmap_Cdl_C_min` | 0.05 | F/m^2 | 5.0 uF/cm^2 |
| `heatmap_Cdl_C_max` | 1.0 | F/m^2 | 100.0 uF/cm^2 |
| `heatmap_pzc_span` | 0.3 | V | +/- 0.3 V around baseline PZC |
| `heatmap_pzc_C_offsets` | [-0.1, 0.0, 0.1] | V | support PZC offsets |

### Quick Reminder

The figures and CSV files under [results/20260528_111255](results/20260528_111255/) correspond to:

- `C_tot = 10 mM`
- `lambda_D = 3.041 nm` auto-calculated
- `L_Au / L_gap / L_Pd = 11 / 10 / 37 nm`
- `out_of_plane_width = 1 cm`
- `Cdl_Au / Cdl_support / Cdl_Pd = 20 / 10 / 40 uF/cm^2`
- `pzc_Au / pzc_support / pzc_Pd = 0.93 / 0.50 / 0.78 V`
- `E1_eq / E2_eq = 0.10 / 0.834 V`
- `pH = 7`
- linear Debye-Huckel EDL, OFAT and heatmaps enabled
