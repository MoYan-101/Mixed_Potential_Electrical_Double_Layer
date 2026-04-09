# Mixed_Potential_Electrical_Double_Layer
构建“EDL ↔ mixed potential ↔ kinetics”的自洽求解器，数值求出 E_mix, i_mix

## 参数修改
我尝试把可能的改变的参数都纳入进去：Au-C-Pd的几何(长度)、Cdl、E_pzc; alpha; exchange current density; electrolyte concentration; E_eq

## W/O EDL
尝试输出在相同参数的情况下，with and without EDL的E_mix和i_mix的对比


## How To Run (Beginner Steps)

### 1) Run the main script (original workflow)
This runs the default parameters and saves results under this script folder's
`results/<timestamp>/`.

```
python Solve_Emix_updating.py
```

You will get:
- `results/<timestamp>/results_summary.csv`
- `results/<timestamp>/profiles.npz`
- `results/<timestamp>/figures/*.png`
- OFAT / heatmap / sensitivity outputs if enabled in parameters

### 2) Compare with-EDL vs no-EDL (single run)
Use the helper script below. It compares with/without EDL and saves results.

```
python run_compare.py
```

Outputs are saved to:
- `results/compare_demo/summary_compare.csv`
- `results/compare_demo/figures/*.png`

### 3) Batch run multiple cases from JSON
This is the easiest way to change parameters without editing the source code.

Step A: edit `params_template.json` (only change numbers).

Step B: run:
```
python run_cases.py
```

This will run:
- `case1`: uses `params_template.json` as-is
- `case2`: only overrides `L_gap = 100e-9`

Outputs:
- `results/case1/`
- `results/case2/`
- each case now saves both `params_overrides_input.json` and `params_used.json`

Notes:
- `run_cases.py` and `run_compare.py` resolve `params_template.json` relative to
  their own directory, so they work even if you launch them from another folder.
- `i_mix` and `I_total` in scans/compare plots are the code's
  `int i dx_tilde` quantities. Multiply by `lambda_D` if you need current per
  unit depth in `A/m`.

### 4) Common pitfall: lambda_D
If you manually set `lambda_D`, it overrides the auto-calculated Debye length.
So if you change `C_tot`, `epsilon_r`, or `T`, you should set `lambda_D = null`
in your JSON so it can be recalculated.

The helper `apply_param_overrides(...)` already does this automatically unless
you explicitly set `lambda_D` in your overrides.
