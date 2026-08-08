# Facing Au/Pd electrodes with an electrolyte-filled slit

This package contains two parallel, facing planar electrodes. Au is the left
wall, Pd is the right wall, and a symmetric 1:1 electrolyte fills the finite
slit between them. The metals are connected by an ideal external wire and
therefore share one `E_mix`; their Faradaic currents satisfy the absolute
balance

```text
I_Au + I_Pd = 0.
```

The package now retains two electrostatic backends. Existing timestamped
linear results are not overwritten.

## Analytic linearized-PB backend

With `x_tilde = x/lambda_D` and `psi = F phi_s/(RT)`, the original electrolyte
problem is

```text
-d2(psi)/d(x_tilde)2 + psi = 0,  0 < x < d,
-psi'(0) + g_Au psi(0) = g_Au beta(E_mix - PZC_Au),
+psi'(d) + g_Pd psi(d) = g_Pd beta(E_mix - PZC_Pd).
```

It uses a stable two-exponential analytic representation. This backend is
kept for reproducibility and remains the CLI default.

## Full nonlinear-PB backend

The nonlinear backend solves the complete finite-slit boundary-value problem

```text
d2(psi)/d(x_tilde)2 = sinh(psi),  0 < x < d,
-psi'(0) + g_Au psi(0) = g_Au beta(E_mix - PZC_Au),
+psi'(d) + g_Pd psi(d) = g_Pd beta(E_mix - PZC_Pd),
I_Au + I_Pd = 0.
```

For finite overlapping gaps, `scipy.solve_bvp` resolves the spatial nonlinear
PB field and an outer Brent solve enforces the log-current balance. The
Gouy--Chapman relation

```text
psi_s = 2 asinh(q_D / 2)
```

is not imposed independently at the two walls, because that relation assumes
each interface has its own `psi -> 0`, `psi' -> 0` bulk far field. Instead, it
is used only for the independent-planar large-gap asymptote. The production
configuration switches to this asymptote above `50 lambda_D`; at 1000 nm,
`d/lambda_D = 328.816` and the cross-gap tail is of order `1.6e-143`, so the
asymptote and the finite-slit solution are identical at double precision.

The production nonlinear parameter file is `params_nonlinear_pb.json` and
contains `d = 1, 2, 3, 5, 10, 1000 nm`. Each run also repeats the calculation
with a five-times tighter PB tolerance and a denser initial mesh; differences
are recorded in `nonlinear_pb_convergence.json` and the corresponding CSV.

## Geometry and figures

The solution is translationally invariant along the electrode face. Exported
2D maps therefore repeat the exact one-dimensional normal profile along a
representative 25 nm face coordinate. Spatial axes are explicitly marked as
not drawn to equal scale.

For gaps below 100 nm, the full positive-width electrolyte slit is shown. A
zero-width electrolyte gap is rejected. For 1000 nm, figures show two
`5 lambda_D` edge windows and omit the bulk middle interval using a broken
axis. Au/Pd boundary labels are drawn without colored electrode bars over the
axes.

The six-gap nonlinear run exports:

- 12 RP 2D PNG and 12 matching SVG files;
- six Figure-3-style PNG and six matching SVG panels;
- zero PDF files;
- full profiles, endpoint metrics, surface charge, solver diagnostics, and
  convergence data in CSV/JSON.

## Run commands

From this directory, the existing linear backend is

```bash
MPLCONFIGDIR=/private/tmp/au_pd_mpl PYTHONPATH=src \
  ../.venv_macos/bin/python -m au_pd_facing_gap.cli
```

The nonlinear six-gap run is

```bash
MPLCONFIGDIR=/private/tmp/au_pd_mpl PYTHONPATH=src \
  ../.venv_macos/bin/python -m au_pd_facing_gap.cli \
  --backend nonlinear --params params_nonlinear_pb.json
```

Runs are non-overwriting and create timestamped directories under `results/`.

## Scope and caveats

Huang, Chen, and Eikerling (PNAS 2023, DOI 10.1073/pnas.2307307120) is used as
a conceptual reference for simultaneous two-electrode charging and steady
Faradaic balance. Their model is a lumped cell-voltage/Stern--Gouy--Chapman/
solution-resistance model and is not a spatial nanometre-gap PB-overlap solver.

Nonlinear PB removes the Debye--Huckel small-potential approximation, but it
does not make the 1 nm result automatically quantitative. The retained model
is a mean-field point-ion continuum description with a grand-canonical bulk
chemical-potential reference. Finite ion size, dielectric saturation, ion
correlations, ion-number conservation/Donnan shifts, explicit Stern-layer
thickness, Nernst--Planck transport, diffusion limitation, and solution
resistance are not included. These limitations are strongest at 1 nm.
