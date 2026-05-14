# Mixed_Potential_Electrical_Double_Layer
构建“EDL ↔ mixed potential ↔ kinetics”的自洽求解器，数值求出 E_mix, i_mix

## 参数修改
我尝试把可能的改变的参数都纳入进去：Au-C-Pd的几何(长度)、Cdl、E_pzc; alpha; exchange current density; electrolyte concentration; E_eq

## W/O EDL
尝试输出在相同参数的情况下，with and without EDL的E_mix和i_mix的对比


## How To Run (Beginner Steps)

### 0) Install dependencies
From this folder, install the Python packages used by the workflow:

```
python3 -m pip install -r requirements.txt
```

The core solver uses `numpy`, `scipy`, `pandas`, and `matplotlib`. `plotly` is
used only for optional interactive HTML figures:

```
python3 -m pip install -r requirements-optional.txt
```

### 1) Run the main script
This runs the default parameters and saves results under this script folder's
`results/<timestamp>/`.

```
python3 Solve_Emix_updating.py
```

You will get:
- `results/<timestamp>/results_summary.csv`
- `results/<timestamp>/summary_compare.csv`
- `results/<timestamp>/profiles.npz`
- `results/<timestamp>/figures/*.png`
- OFAT comparison outputs `ofat_compare_*.csv` and `figures/ofat_compare_*.png`
- heatmap comparison outputs `heatmap_compare_*.csv` and `figures/heatmap_compare_*.png` if enabled
- sensitivity outputs if enabled in parameters

### 2) Compare with-EDL vs no-EDL (single run)
Use the helper script below. It compares with/without EDL and saves results.

```
python3 run_compare.py
```

Outputs are saved to:
- `results/compare_demo/<timestamp>/summary_compare.csv`
- `results/compare_demo/<timestamp>/figures/*.png`
- compare figures include:
  `compare_polcurve_*.png`, `compare_emix_imix_*.png`, `compare_phi2_*.png`,
  and `compare_potentials_overpotential_*.png`

### 3) Batch run multiple cases from JSON
This is the easiest way to change parameters without editing the source code.

Step A: edit `params_template.json` (only change numbers).

Step B: run:
```
python3 run_cases.py
```

This will run:
- `case1`: uses `params_template.json` as-is
- `case2`: only overrides `L_gap = 100e-9` (support width)

Outputs:
- `results/case1/`
- `results/case2/`
- each case now saves both `params_overrides_input.json` and `params_used.json`

Notes:
- `run_cases.py` and `run_compare.py` resolve `params_template.json` relative to
  their own directory, so they work even if you launch them from another folder.
- `i(x)` in baseline profiles is the local current density in `A/m^2`.
- `i_mix`, `I_Au`, `I_Pd`, and `I_total` are the code's `int i d x_tilde`
  quantities. Since `x_tilde` is dimensionless, they still carry `A/m^2`.
- `run_case()` / summaries / scans save both line-current outputs such as
  `i_mix_phys_A_per_m = lambda_D * i_mix` and absolute-current outputs such as
  `i_mix_abs_A = out_of_plane_width * i_mix_phys_A_per_m`.
- The model is 2D, so the natural integrated-current quantity is `A/m`. We now
  set `out_of_plane_width = 1.0 m` by default, so the main comparison figures
  and summaries report `A` under this unit-width assumption.
- If your real electrode has a different out-of-plane width `W`, change
  `out_of_plane_width` and the absolute current will scale automatically as
  `I_abs = W * i_mix_phys_A_per_m`.
- Current-related comparison figures now default to average mixed current
  density in `A/m^2`, normalized by the reactive Au+Pd area. Absolute-current
  outputs are still saved in summaries for downstream use.
  The baseline `i(x)` profile remains a local current density in `A/m^2`.
- `run_case()` also returns `max_abs_phi_tilde` and `debye_huckel_ok`, and you
  can control threshold handling with `dh_warn_threshold` and
  `dh_violation_action = ignore|warn|raise`.
- Scan and heatmap axes now label the parameter units explicitly (`nm`, `V`,
  `uF/cm^2`, `M`, `K`, or `-` for dimensionless quantities).
- The main workflow now uses `with EDL` vs `without EDL` as the primary
  comparison. The with-EDL result is always the FULL numerical solution.
- The default compare polarization curve is now a local window around the two
  `E_mix` values (`±0.10 V` margin by default). If you need the old full-range
  curve, pass explicit `E_min` and `E_max` in `solver_settings`.
- The default OFAT range for `L_gap` is now `0` to `1000e-9 m` (0 to 1000 nm),
  which is intended for nanoscale to sub-micron support-width scans.
- The default OFAT `L_gap` scan is plotted in `nm` and now uses the same
  sampling count as the generic OFAT default (`ofat_L_gap_n = ofat_n = 15`).
- The `L_Au × L_Pd` heatmap now uses a fixed `2` to `1000 nm` range on both
  axes, instead of scaling the range around the baseline lengths.
- The default concentration ranges for OFAT are `0.1 mM` to `10 M`.
  Internal calculations still use `mol/m^3`, but the plotted concentration
  axis is converted to `M` for chemistry-style figures.
- The active default 2D heatmap families are material-coupled scans:
  `Cdl_Au × Cdl_Pd`, `L_Au × L_Pd`, and `pzc_Au × pzc_Pd`.
- Each active heatmap family writes standalone maps for `E_mix` with EDL,
  `delta_Emix`, average mixed current density with EDL, and
  `delta_i_mix_avg`.
- The shipped baseline now uses `C_tot = 10.0 mol/m^3`, which corresponds to
  `10 mM` for a symmetric 1:1 electrolyte.
- Displayed capacitances now use `uF/cm^2` throughout figures and summary
  tables, although the solver still stores and computes them internally in
  `F/m^2`.
- Default OFAT capacitance scans are now floored at `1 uF/cm^2`
  (`0.01 F/m^2`) to avoid exploring obviously nonphysical sub-`1 uF/cm^2`
  values by default.
- In the current Debye-length formula, `C_tot` means the concentration of each
  ion for a symmetric 1:1 electrolyte. If you intend `0.1 M`, enter `100.0`,
  not `0.1`.
- In the 2026 main solver, the kinetic reaction-plane potential is evaluated
  at the surface (`y_tilde = 0`). The older root-level prototype includes a
  finite reaction-plane distance; do not mix those two conventions without
  adding the same offset to the 2026 solver.
- `pH` is now available as an explicit scan variable. In the current model, pH
  acts only through reaction thermodynamics/kinetics, not through ionic
  strength, unless you explicitly also change `C_tot`.
- By default, reaction 2 is treated as a proton-coupled reduction of the form
  `H+ + e- -> 1/2 H2` for pH scans. Accordingly, the shipped defaults use
  `E2_eq_pH_slope_V_per_pH = -59.16 mV/pH` at `298 K` and `it0_2_pH_order = 1`.
- The pH dependence is parameterized by:
  `E1_eq_pH_slope_V_per_pH`, `E2_eq_pH_slope_V_per_pH`,
  `it0_1_pH_order`, and `it0_2_pH_order`, all referenced to `pH_ref`.
- The default pH scan range is `0` to `14`, and the main workflow will export
  `ofat_compare_pH_E_mix.png` and `ofat_compare_pH_i_mix_avg_A_per_m2.png` when
  `do_ofat=true`.
- Common unit reminder: `100 uF/cm^2 = 1.0 F/m^2`, so `1 uF/cm^2 = 0.01 F/m^2`.

### 4) Common pitfall: lambda_D
If you manually set `lambda_D`, it overrides the auto-calculated Debye length.
So if you change `C_tot`, `epsilon_r`, or `T`, you should set `lambda_D = null`
in your JSON so it can be recalculated.

The helper `apply_param_overrides(...)` already does this automatically unless
you explicitly set `lambda_D` in your overrides.
