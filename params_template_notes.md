# Parameter Notes (English)

Each key below matches `params_template.json`. Units are the same as in the code.

- `R`: Gas constant (J/mol/K).
- `F`: Faraday constant (C/mol).
- `T`: Temperature (K).

- `epsilon0`: Vacuum permittivity (F/m).
- `epsilon_r`: Relative permittivity of the electrolyte (dimensionless).
- `epsilon_s`: Absolute permittivity override (F/m); set `null` to use `epsilon0 * epsilon_r`.

- `C_tot`: Concentration of each ionic species for a symmetric 1:1 electrolyte (mol/m^3), matching the Debye-length formula used in the code. The shipped baseline `10.0` corresponds to `10 mM`. If you want to model `0.1 M`, write `100.0`, not `0.1`.
- Concentration-related calculations stay in `mol/m^3`, but `C_tot` scan axes are converted to `M` in the final figures.
- `lambda_D`: Debye length (m); set `null` to auto-calculate from `C_tot`, `epsilon_r`, and `T`.

- `pH`: Working pH used by the pH-dependent kinetic parameterization.
- `pH_ref`: Reference pH at which `E1_eq`, `E2_eq`, `it0_1`, and `it0_2` are defined.
- `E1_eq_pH_slope_V_per_pH`, `E2_eq_pH_slope_V_per_pH`: Linear pH slopes for the equilibrium potentials. The code uses `E_eq(pH) = E_eq + slope * (pH - pH_ref)`.
- `it0_1_pH_order`, `it0_2_pH_order`: Base-10 pH orders for exchange current densities. The code uses `it0(pH) = it0 * 10^(-order * (pH - pH_ref))`.
- In the shipped defaults, reaction 2 is treated as `H+ + e- -> 1/2 H2` for the pH scan, so `E2_eq_pH_slope_V_per_pH` is set to `-(2.303RT/F)` and `it0_2_pH_order = 1`.

- `L_Au`: Length of Au segment (m).
- `L_gap`: Length of gap/central segment (m).
- `ofat_pH_min`, `ofat_pH_max`: Default pH scan bounds. The shipped defaults are `0` to `14`.
- `ofat_C_tot_min`, `ofat_C_tot_max`: Internal lower/upper bounds for the default OFAT concentration scan (`mol/m^3`). The shipped defaults correspond to `0.1 mM` and `10 M`.
- `heatmap_C_tot_min`, `heatmap_C_tot_max`: Internal lower/upper bounds for the default `C_tot` heatmap range (`mol/m^3`). The shipped defaults also correspond to `0.1 mM` and `10 M`.
- `heatmap_L_min`, `heatmap_L_max`: Internal lower/upper bounds for the
  default `L_Au × L_Pd` heatmap range. The shipped defaults are `2e-9` to
  `1000e-9 m` (2 to 1000 nm) on both axes.
- `ofat_L_gap_min`, `ofat_L_gap_max`: Default OFAT scan bounds for `L_gap`.
  The shipped defaults are `0` to `1000e-9 m` (0 to 1000 nm).
- `L_Pd_len`: Length of Pd segment (m).
- `Cdl_Au`: Double-layer capacitance at Au (stored internally in `F/m^2`;
  displayed in figures as `uF/cm^2`).
- `Cdl_C`: Double-layer capacitance at the support segment (stored internally
  in `F/m^2`; displayed in figures as `uF/cm^2`).
- `Cdl_Pd`: Double-layer capacitance at Pd (stored internally in `F/m^2`;
  displayed in figures as `uF/cm^2`).

- `g_Au`: Dimensionless g for Au; set `null` to compute from `Cdl_Au`.
- `g_C`: Dimensionless g for C; set `null` to compute from `Cdl_C`.
- `g_Pd`: Dimensionless g for Pd; set `null` to compute from `Cdl_Pd`.

- `pzc_Au`: Potential of zero charge for Au (V).
- `pzc_C`: Potential of zero charge for C (V).
- `pzc_Pd`: Potential of zero charge for Pd (V).

- `it0_1`: Exchange current density for reaction 1 (A/m^2).
- `it0_2`: Exchange current density for reaction 2 (A/m^2).

- `alpha1`: Charge transfer coefficient for reaction 1 (dimensionless).
- `alpha2`: Charge transfer coefficient for reaction 2 (dimensionless).

- `z_R1`: Charge number of reactant in reaction 1 (dimensionless).
- `z_O2`: Charge number of oxidant in reaction 2 (dimensionless).

- `E1_eq`: Equilibrium potential for reaction 1 (V).
- `E2_eq`: Equilibrium potential for reaction 2 (V).

- `N_modes`: Number of cosine modes in EDL expansion (integer).
- `Nx`: Number of spatial grid points along x (integer).
- `xtol`: Root solver tolerance for E_mix (V).
- `max_bracket_expands`: Max bracket expansion count in root finding (integer).
- `use_edl`: Enable EDL effect (true/false).
- `use_affine_phi2`: Use affine phi2 approximation (true/false).
- `use_closed_form_when_affine`: Use closed-form E_mix when affine is enabled (true/false).
- `do_self_checks`: Run built-in self-checks (true/false).
- `do_convergence_check`: Run optional convergence scan (true/false; slower).
- `dh_warn_threshold`: Warn/guard threshold for `max(|phi_tilde|)` in the linear Debye-Huckel model.
- `dh_violation_action`: `"ignore"`, `"warn"`, or `"raise"` when the Debye-Huckel threshold is exceeded.
- `do_ofat`: Run OFAT scan (true/false).
- `do_heatmaps`: Run 2D heatmaps (true/false).
- `do_sensitivities`: Run sensitivity analysis (true/false).
- `ofat_n`: Number of points per OFAT scan (integer).
- `heatmap_nx`: Heatmap grid size in x (integer).
- `heatmap_ny`: Heatmap grid size in y (integer).
- `scan_mode`: `"MEAN"` or `"BOTH"` for OFAT.
- `heatmap_mode`: `"MEAN"`, `"FULL"`, or `"BOTH"`.
- `sensitivity_mode`: `"MEAN"`, `"FULL"`, or `"BOTH"`.

Common unit pitfalls:
- Internal capacitance units are `F/m^2`. Figure axes and summary tables now
  display capacitance in `uF/cm^2`.
- Conversion reminder: `100 uF/cm^2 = 1.0 F/m^2`, so `1 uF/cm^2 = 0.01 F/m^2`.
- As a practical default, the code now avoids scanning below `1 uF/cm^2`
  unless you explicitly force smaller capacitance values.
- Internal concentration units are `mol/m^3`. For reference, `0.1 M = 100 mol/m^3`.
- `1 mM = 1 mol/m^3`, so the default lower concentration bound `0.1 mol/m^3` is `0.1 mM`.
- In the current pH implementation, changing pH alone does not change ionic strength. If you want pH and ionic strength to co-vary, you must also change `C_tot` explicitly.
