# Parameter Notes (English)

Each key below matches `params_template.json`. Units are the same as in the code.

- `R`: Gas constant (J/mol/K).
- `F`: Faraday constant (C/mol).
- `T`: Temperature (K).

- `epsilon0`: Vacuum permittivity (F/m).
- `epsilon_r`: Relative permittivity of the electrolyte (dimensionless).
- `epsilon_s`: Absolute permittivity override (F/m); set `null` to use `epsilon0 * epsilon_r`.

- `C_tot`: Total electrolyte concentration (mol/m^3).
- `lambda_D`: Debye length (m); set `null` to auto-calculate from `C_tot`, `epsilon_r`, and `T`.

- `L_Au`: Length of Au segment (m).
- `L_gap`: Length of gap/central segment (m).
- `L_Pd_len`: Length of Pd segment (m).
- `Cdl_Au`: Double-layer capacitance at Au (F/m^2).
- `Cdl_C`: Double-layer capacitance at C segment (F/m^2).
- `Cdl_Pd`: Double-layer capacitance at Pd (F/m^2).

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
- `do_ofat`: Run OFAT scan (true/false).
- `do_heatmaps`: Run 2D heatmaps (true/false).
- `do_sensitivities`: Run sensitivity analysis (true/false).
- `ofat_n`: Number of points per OFAT scan (integer).
- `heatmap_nx`: Heatmap grid size in x (integer).
- `heatmap_ny`: Heatmap grid size in y (integer).
- `scan_mode`: "MEAN" or "BOTH".
