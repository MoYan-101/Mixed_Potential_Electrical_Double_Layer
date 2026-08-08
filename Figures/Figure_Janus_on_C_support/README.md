# C-supported Au–Pd Janus mirror-cell model

This standalone package models the flush, coplanar surface sequence

`C (5 nm) | Au (4 nm) | Pd (4 nm) | C (5 nm)`

in a semi-infinite electrolyte. The lateral boundaries are homogeneous
Neumann boundaries, so the cosine solution is an even reflection of the
18 nm computational half-cell. Its material continuation is

`C|Au|Pd|C | C|Pd|Au|C | C|Au|Pd|C ...`

and the complete translational material period is 36 nm. Adjacent 5 nm carbon
end segments join into a 10 nm exposed-carbon region. This is an alternating-
orientation stripe surrogate, not a same-orientation 18 nm periodic array.

## Physics

- Semi-infinite linearized Poisson–Boltzmann/Debye–Hückel electrolyte.
- Piecewise Stern–Robin charging on all four exposed segments.
- Au, Pd, and conductive C share one ideal electronic mixed potential.
- Carbon charges through its own Helmholtz capacitance and PZC but carries no
  Faradaic reaction current.
- The mixed potential is determined only by absolute-current balance,
  `I_Au + I_Pd = 0`.
- Mixed current density is normalized by the total reactive Au+Pd area; carbon
  area is excluded.

The model is a flush coplanar boundary surrogate. It does not include particle
curvature, height, sidewalls, a buried particle–carbon contact, ion transport,
solution resistance, or nonlinear Poisson–Boltzmann physics.

## Canonical parameters

- `L_C_left/L_Au/L_Pd/L_C_right = 5/4/4/5 nm`
- `C_H,Au/C/Pd = 0.50/0.20/0.50 F m^-2`
- `PZC_Au/C/Pd = 0.93/0.50/0.78 V vs. RHE`
- `i0,1 = i0,2 = 1.852573885166257e-4 A m^-2`
- `alpha1 = alpha2 = 0.5`
- `C_tot = 10 mol m^-3 = 10 mM`
- out-of-plane width `W = 0.01 m`
- production resolution `N_modes = 960`, `Nx = 5000`, `GL128`, `Ny = 320`

The fixed parameter entry is `janus_on_c_support.default_params()`. The public
calculation/export API also provides `apply_param_overrides()`, `solve_case()`,
and `build_result_bundle()`. Geometry overrides may be used for non-publishing
regression calculations; published bundles are deliberately restricted to the
canonical 5/4/4/5 nm geometry so their model name and artifact tags cannot be
misleading.

## Run

From the `2026/` workspace root:

```bash
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/private/tmp/janus_on_c_mpl \
Mixed_Potential_Electrical_Double_Layer/.venv_macos/bin/python \
Figures/Figure_Janus_on_C_support/make_all_janus_on_c_support.py
```

Each run is written without overwriting an existing run under `results/<run_id>/`.
The bundle contains six Figure 3 panels, three Figure RP outputs, numerical
fields and profiles, parameters, validation, manifests, and SHA-256 checksums.
Figures are exported as 600 dpi PNG and editable-text SVG; PDF is not produced.

A published run must pass the production refinement gate. The default check is
`N_modes: 480 -> 960` and `GL: 64 -> 128`, with strict requirements
`delta E_mix < 0.1 mV` and `delta i_mix < 0.1%`. A failed dimension is doubled
up to `N_modes = 3840` or `GL512`; failure at either maximum prevents
publication. The full 2D grid is stored in compressed NPZ. A smaller,
human-readable CSV preview and explicit sampling metadata are saved alongside
it to keep each run practical to inspect and archive.

## Latest published result

The current no-overwrite reference bundle is
[`results/20260808_163633/`](results/20260808_163633/). It was regenerated after
the Figure 3/Figure RP typography audit. The preceding bundle
`results/20260808_162015/` remains intact and was not overwritten.

For the canonical parameter set, the reference result is:

- with EDL: `E_mix = 0.610999760140 V`,
  `i_mix_avg = 0.0645804682433 A m^-2`
- w/o EDL: `E_mix = 0.467000000000 V`,
  `i_mix_avg = 0.117560354079 A m^-2`
- with-EDL half-cell/full-period absolute mixed current:
  `5.16643745947e-12 / 1.03328749189e-11 A`
- w/o-EDL half-cell/full-period absolute mixed current:
  `9.40482832630e-12 / 1.88096566526e-11 A`

The run contains 6 Figure 3 PNG/SVG pairs and 3 Figure RP PNG/SVG pairs:
9 PNG + 9 editable-text SVG + 0 PDF. All 38 entries in
`checksums.sha256` were verified, all JSON files parse strictly, and the source
aggregate SHA-256 recorded in the manifest is
`d0e1a7e212f36097fceb505e247a472a26a778832a9ec1fa71c2a4626b27c517`.
The complete package test suite reports `19 passed`.

Production convergence passed without escalation:

- `N_modes: 480 -> 960`: `delta E_mix = 0.0009691984 mV`,
  `delta i_mix = 0.00520068%`
- `GL: 64 -> 128`: `delta E_mix = 0.0002903232 mV`,
  `delta i_mix = 0.00139755%`

## Four-case EDL comparison and Hutchings 2022

The dedicated synthesis package is
[`ALL/Explantion_Hutchings`](../../ALL/Explantion_Hutchings/). It keeps the
user-specified spelling `Explantion_Hutchings`, reads the four upstream models
without changing them, and regenerates the common comparison bars plus the
nine Figure 3/Figure RP figure pairs for each case.

Absolute-current comparison uses one physical reporting area throughout:

```text
out-of-plane width W = 0.01 m
Au reactive width = 4 nm
Pd reactive width = 4 nm
common reactive area = W * (4 + 4) nm = 8.0e-11 m2
I_mix = |I_Au| = |I_Pd| at I_Au + I_Pd = 0
```

The geometry-to-area mapping is important:

- Legacy case 2 uses the computational half-cell `Au(2)|Pd(2) nm`. Its
  Neumann even continuation contains `Au(4)|Pd(4)` in a complete material
  period. The ALL case folder now displays the original 0--4 nm half-cell;
  only the Summary current is twice the saved half-cell current.
- Legacy case 3 similarly starts from `Au(2)|C(10)|Pd(2) nm`. After even
  continuation, its complete period is 28 nm. The ALL case folder displays the
  original 0--14 nm half-cell; only the Summary current is twice the saved
  half-cell current.
- Case 4 already contains Au(4) and Pd(4) in its 18 nm computational half-cell.
  The common-area comparison therefore uses its half-cell current, not the
  doubled 36 nm full-period current.
- Case 1 has independent uniform half-spaces and no lateral Neumann reflection.
  Its original Au(2)+Pd(2) current is multiplied by two only as a linear area
  equivalent to Au(4)+Pd(4); this must not be described as boundary extension.

Thus the Summary uses the same effective Au/Pd reporting widths, 4/4 nm, once
the boundary/area mapping is handled consistently. The individual ALL case
folders do not use this width for their spatial axes: cases 1--3 retain native
2 nm Au/Pd domains, whereas case 4 retains its native 4 nm Au/Pd geometry.
`I_mix` below is a positive half-reaction magnitude, not the zero net current
and not current density.

| Case | Effective electrostatic geometry | `E_mix`, w/o EDL (V) | `E_mix`, with EDL (V) | `I_mix`, w/o EDL (pA) | `I_mix`, with EDL (pA) |
|---|---|---:|---:|---:|---:|
| 1 | Independent uniform Au/Pd; area-equivalent Au4/Pd4 | 0.467000000 | 0.624910433 | 9.404828326 | 3.452046250 |
| 2 | Touching `Au(4)|Pd(4)` Janus period | 0.467000000 | 0.624910433 | 9.404828326 | 5.319995190 |
| 3 | Separated `Au(4)|C(10)|Pd(4)|C(10)` period | 0.467000000 | 0.597600469 | 9.404828326 | 4.521278834 |
| 4 | `C(5)|Au(4)|Pd(4)|C(5)` supported-Janus half-cell | 0.467000000 | 0.610999760 | 9.404828326 | 5.166437459 |

The table and Summary figures use the common 4 nm+4 nm area. In each case
folder, Figure 3 panel a instead reports the native computational-cell current.
For cases 1--4, the native with-EDL values are respectively
`1.726023/2.659998/2.260639/5.166437 pA`; the native w/o-EDL values are
`4.702414/4.702414/4.702414/9.404828 pA`. The ALL JSON files store both bases
explicitly.

The source summaries are
[`Au_Pd_independent/summary.json`](../Figure_Au2nm_Pd2nm/Au_Pd_independent/summary.json),
[`L_support=0 nm`](../Figure_Au2nm_Pd2nm/Au_C_Pd/inputs/summary_compare_L_support_0nm_au2_pd2_20260528_111255.json),
[`L_support=10 nm`](../Figure_Au2nm_Pd2nm/Au_C_Pd/inputs/summary_compare_L_support_10nm_au2_pd2_20260528_111255.json),
and [`results/20260808_163633/summary.json`](results/20260808_163633/summary.json).

For the Hutchings comparison, the primary presentation set is cases 2--4:

- case 2 is displayed as the touching `Au(2)|Pd(2)` Neumann half-cell; its
  complete even extension gives the effective `Au(4)|Pd(4)` Janus-interface
  reference used for the Summary-area interpretation;
- case 4 is the C-supported Janus surrogate, with a touching Au--Pd boundary
  inside the particle and exposed C on both outer sides;
- case 3 is the physical-mixture surrogate, with Au and Pd spatially separated
  by 10 nm of conductive, chargeable C.

Case 1 should remain a secondary electrostatic reference rather than being
presented as the experimental physical mixture. It has two independent
uniform solution half-spaces that share an ideal `E_mix`, but contains neither
a carbon network nor finite-particle edge fields.

Within this primary three-case set, the with-EDL current ordering is

`touching Janus (5.319995 pA) > C-supported Janus (5.166437 pA) >`
`C-separated physical-mixture surrogate (4.521279 pA)`.

Relative to the C-separated surrogate, the touching Janus has a 27.31 mV
higher `E_mix` and a 17.67% higher `I_mix`; the C-supported Janus has a
13.40 mV higher `E_mix` and a 14.27% higher `I_mix`. Their effective Au/Pd
widths are matched. The two Janus models are nevertheless not a pure-overlap
matched pair because the exposed-C environment, mirror arrangement, and
boundary topology differ. These percentages describe the present model
geometries; they are not predictions of the experimental rate ratios.

The controlled observations are:

- Turning on the EDL shifts `E_mix` positively by 130.6--157.9 mV in all four
  models, but lowers common-area `I_mix` by 43.4--63.3%. Thus these parameters do not
  support the statement that the EDL itself enhances the current.
- Cases 1 and 2 provide the most direct independent-versus-contact
  electrostatic reference at common area, capacitances, PZCs, and kinetics.
  Their `E_mix` values are identical to numerical precision, but the with-EDL
  `I_mix` is 54.11% higher in case 2. Contact-induced lateral EDL coupling
  therefore reduces part of the EDL current suppression without moving the
  symmetric mixed-potential root. Case 1 is still an analytic uniform
  half-space reference rather than a finite-width geometry.
- Inserting 10 nm of charged C between Au and Pd (case 3 versus case 2) lowers
  `E_mix` by 27.31 mV and the with-EDL current by 15.01%. This is a combined
  C-charging, separation, and overlap-attenuation effect, not a pure distance
  effect. Within the matched Au--C--Pd length scan, 10 nm is `3.29 lambda_D`
  and differs from the 1000 nm reference by
  `Delta E_mix = +0.239082 mV` and `Delta I_mix/I_mix = +0.02613%`. C10 and
  C1000 can therefore be treated as the same far-field plateau at the present
  explanation/plot precision, but they are not bitwise or mathematically
  identical.
- Case 4 combines a touching intra-particle Au--Pd boundary with a charged C
  environment and a 10 nm inter-particle C gap. Its current lies between cases
  2 and 3. Its native Au/Pd widths are 4/4 nm; cases 2/3 reach the same effective
  Summary widths only through their Neumann/area mapping. Common-area current
  differences therefore do not contain a reporting-area change, but the
  underlying fields still come from different half-cells and retain exposed-C
  and mirror-cell topology differences; they are not pure overlap measurements.

The paper used for interpretation is Huang et al., *Nature* 603, 271--275
(2022), `Au--Pd separation enhances bimetallic catalysis of alcohol oxidation`
([local PDF](../../MS/s41586-022-04397-7.pdf)). It experimentally establishes
cooperative redox enhancement (CORE): alcohol oxidation on Au supplies
electrons, conductive carbon transfers them to spatially distinct Pd sites,
and Pd consumes them through oxygen reduction. The reported initial HMF rates
were `3.8e-5 M s^-1` for alloy Au--Pd/C, `4.9e-5 M s^-1` for the physical
Au/C+Pd/C mixture, and `6.7e-5 M s^-1` for Janus-like Au@Pd/C.

These calculations add only a solution-side hypothesis to that electronic
mechanism: after ideal electronic coupling has already imposed a common
`E_mix`, nanoscale EDL topology can further change reaction-plane potentials,
reactant concentrations, and the coupled current. They do **not** reproduce or
prove the paper's alloy-versus-separated activity enhancement. None of the
four cases is an Au--Pd alloy model, and all assume zero electronic resistance.
They also omit alloy electronic structure, adsorption, oxygen/HMF transport,
particle curvature and size, and support-network resistance. The paper used
0.4 M NaHCO3 at 80 degrees C for thermocatalysis and about 0.1 M NaOH for the
electrochemical tests, whereas these models use 10 mM electrolyte, 298 K,
pH 7, equal Au/Pd lengths, equal `i0`, and generic redox species. Therefore
`I_mix` must not be equated numerically with the paper's reaction rate or
TOF.

The defensible interpretation is: Hutchings et al. demonstrate the electronic
CORE benefit of keeping Au and Pd as distinct, conductively connected redox
sites; this model suggests that, once that connection exists, a touching Janus
interface can additionally lessen EDL-induced current suppression relative to
electrostatically independent interfaces. Under the present linear-PB model,
EDL overlap alone does not explain the experimental benefit of separation over
alloying.

## Figure typography and visible-label contract

Starting with run `20260808_163633`, the visible Figure 3 titles are
`Reactant concentration at RP`, `Overpotential at RP`, and
`Current density at RP`; the word `Local` is not displayed anywhere in the
Figure 3 SVG/PNG artwork. Existing artifact filenames retain `_local_` only to
preserve traceability and filename stability.

Figure 3 and Figure RP follow the scientific typography conventions in the
[IUPAC italic/roman guidance](https://media.iupac.org/standing/idcns/italic_roman.html)
and the [BIPM SI Brochure](https://www.bipm.org/en/si-brochure-9):

- quantity and coordinate symbols are italic (`E`, `i`, `x`, `y`, `phi`,
  `Phi`, `eta`, `c`, and `sigma`);
- descriptive subscripts are upright (`mix`, `RP`, `bulk`, `eq`, and `s`);
- chemical identities, material names, abbreviations, unit symbols, and SI
  prefixes are upright (`Red`, `Ox`, `Au`, `Pd`, `C`, `PZC`, `EDL`, `RHE`);
- chemical species are displayed as `Red_1^-` and `Ox_2^+` with upright names,
  numeric subscripts, and charge superscripts;
- compound units use multiplication spacing and negative powers, including
  `A m^-2` and `µC cm^-2`;
- dimensionless concentration ratios are shown without the non-SI `(-)`
  suffix;
- the solution-potential colorbar uses italic `Phi` with upright descriptive
  subscript `s`, and spatial axes use italic `x` and `y` with upright `nm`.

## Tests

From this package directory:

```bash
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/private/tmp/janus_on_c_test_mpl \
../../Mixed_Potential_Electrical_Double_Layer/.venv_macos/bin/python \
-m pytest -p no:cacheprovider -q
```

## Applicability

The result bundle reports `max |phi_tilde|`. Values above one exceed the weak-
potential range of Debye–Hückel linearization. Passing root, charge, spectral,
and grid-convergence checks does not remove that physical applicability limit.
For `results/20260808_163633/`, `max |phi_tilde| = 7.590510604486912 > 1`, so
this caveat is active even though all numerical validation checks pass.
Piecewise material boundaries are represented by a truncated cosine series, so
boundary-local Fourier/Gibbs features must be assessed through mode refinement
and must not be interpreted automatically as physical oscillations.
