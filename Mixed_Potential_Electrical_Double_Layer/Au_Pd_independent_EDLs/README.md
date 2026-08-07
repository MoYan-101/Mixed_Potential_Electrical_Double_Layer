# Independent planar Au/Pd EDLs

This model represents two spatially independent, locally planar
metal/electrolyte interfaces. Au and Pd share one electronic potential through
an ideal external wire, while their electrostatic problems have separate local
coordinates and no spatial EDL overlap. The two local bulk electrolytes are
assumed to share the same solution-potential reference (for example through a
common reservoir or an ideal salt bridge).

For each material `M = Au, Pd`, the dimensionless linearized-PB problem is

```text
-d2(phi_tilde_M)/d(y_tilde_M)2 + phi_tilde_M = 0,
phi_tilde_M -> 0 as y_tilde_M -> infinity,
-d(phi_tilde_M)/d(y_tilde_M) + g_M phi_tilde_M
    = g_M beta (E_mix - PZC_M) at y_tilde_M = 0.
```

It has the exact solution

```text
phi_tilde_RP,M = [g_M/(1 + g_M)] beta (E_mix - PZC_M),
phi_tilde_M(y) = phi_tilde_RP,M exp(-y/lambda_D).
```

The only coupling between Au and Pd is

```text
I_Au(E_mix) + I_Pd(E_mix) = 0.
```

One active face per electrode is used by default, matching the former
25 nm x 1 cm reactive area convention. Set `active_faces_Au` or
`active_faces_Pd` to 2 only for a genuinely two-sided wetted foil. This model
does not include finite-particle edge/fringing fields.

Run from this directory with the shared project environment:

```bash
MPLCONFIGDIR=/private/tmp/au_pd_mpl PYTHONPATH=src \
  ../.venv_macos/bin/python -m au_pd_independent_edls.cli
```

Each run creates a non-overwriting timestamped directory under `results/` with
six Figure-3-style PNG/SVG panels and two RP 2D PNG/SVG figures. No PDF is
generated. The linearized-PB validity diagnostic is retained; solving the
electrostatics analytically removes Fourier ringing but does not make a large
`max |phi_tilde|` quantitatively valid.

## Electrolyte-concentration study

Run the traceable `C_tot` study with:

```bash
MPLCONFIGDIR=/private/tmp/au_pd_mpl PYTHONPATH=src \
  ../.venv_macos/bin/python -m au_pd_independent_edls.ctot_study
```

The study creates a separate non-overwriting
`results/<timestamp>_ctot_study/` directory. It scans `10^-4` to `10^3 M`,
retains exact `0.01`, `1`, `10`, and `10^3 M` checkpoints, and releases any
explicit `lambda_D`/`g_Au`/`g_Pd` overrides at every point so those quantities
are recalculated from concentration. The segment above `10 M` is labelled as a
formal high-salt extension; `10^3 M` is not treated as physically realizable.

Outputs include Figure-4-style `E_mix`, average mixed-current-density, and
polarization plots; an Au/Pd mechanism figure; and calculated `0.01/1 M`
independent EDL profiles with a schematic `w/o EDL` zero-potential line.
Surface charge, dimensionless and dimensional reaction-plane potential,
overpotential, dimensionless BV activation factor, normalized reactant
concentration, their kinetic product, currents, root diagnostics, and
the final average mixed current density are saved in CSV/JSON form. Figures are
PNG/SVG only; no PDF is generated. The run is staged in a temporary sibling directory and
renamed into place only after validation. `run_manifest.json` records source,
parameter, configuration, and software hashes/versions; `artifacts.json`
records artifact sizes and SHA-256 checksums.
