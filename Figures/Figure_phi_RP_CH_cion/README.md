# Figure_phi_RP_CH_cion

This folder contains a homogeneous single-surface EDL figure set for explaining how
Helmholtz capacitance `C_H` and ion concentration `c_ion` affect the reaction-plane
potential `phi_RP` when `E < PZC`.

Baseline demonstration:

- `E = 0.40 V`
- `PZC = 0.60 V`
- Homogeneous surface: `Cdl_Au = Cdl_C = Cdl_Pd = C_H` and `pzc_Au = pzc_C = pzc_Pd = PZC`
- Linear Debye-Huckel EDL with the repository solver

Key relation for this homogeneous limit:

```text
phi_RP = [g / (1 + g)] * (E - PZC)
g = lambda_D * C_H / epsilon_s = C_H / C_D
sigma = C_H * (E - PZC - phi_RP)
phi(x) = phi_RP * exp(-x / lambda_D)
```

For the fixed `C_H = 0.20 F/m^2` sweep, increasing `c_ion` makes `sigma` more
negative while `phi_RP` shifts toward 0. This reflects the shorter diffuse-layer
length at high salt: the solution-side potential is less negative at the
reaction plane, but it decays over a much shorter distance. The `phi(x)` profile
comparison uses `c_ion = 10^-4, 10^-3, 10^-2, 10^-1, and 1 M` to show the trend.

Generated outputs:

- `phi_rp_CH_sweep_homogeneous_E040_PZC060.png/svg`
- `phi_rp_cion_sweep_homogeneous_E040_PZC060.png/svg`
- `phi_rp_CH_cion_heatmap_homogeneous_E040_PZC060.png/svg`
- `phi_rp_CH_cion_explanation_homogeneous_E040_PZC060.png/svg`
- `sigma_cion_sweep_homogeneous_E040_PZC060.png/svg`
- `phi_profile_cion_comparison_homogeneous_E040_PZC060.png/svg`
- `sigma_cion_phi_profile_homogeneous_E040_PZC060.png/svg`

Traceability:

- `inputs/inputs_homogeneous_E040_PZC060.json`
- `csv/phi_rp_CH_sweep_homogeneous_E040_PZC060.csv`
- `csv/phi_rp_cion_sweep_homogeneous_E040_PZC060.csv`
- `csv/phi_rp_CH_cion_heatmap_homogeneous_E040_PZC060.csv`
- `csv/sigma_cion_sweep_homogeneous_E040_PZC060.csv`
- `csv/phi_profile_cion_comparison_homogeneous_E040_PZC060.csv`
