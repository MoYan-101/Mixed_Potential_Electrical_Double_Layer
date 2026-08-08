# Au | Pd with EDL result memory

## Solver backend after the v0.3.0 switch

- The active source now solves the linearized-PB problem on the exact
  semi-infinite strip with the cosine/Fourier expansion
  `sum A_n cos(rho_n x_tilde) exp(-gamma_n y_tilde)`.
- The exposed insulating substrate enters as an exact `g(x)=0` Neumann span.
- Formal defaults are `N_modes=960` and `Nx=5000`, matching the established
  high-resolution legacy Fourier comparison.
- Finite-top and Q1 spacing controls have been removed from the active
  parameter schema; the physical far field is `phi_tilde -> 0` as
  `y_tilde -> infinity`.
- `results/20260802_144648/` is the active v0.3.0 spectral result.
- `results/20260802_024804/` remains an unchanged historical v0.2.0
  finite-height-Q1 result; its stored numbers and `fem_*` artifact names are
  not relabelled as v0.3.0 outputs.

## Current model identity

- Au and Pd are zero-thickness coplanar electrodes embedded flush in the same
  ideal insulating substrate.
- The electrolyte domain is the semi-infinite strip
  `0 <= x_tilde <= L_tilde`, `y_tilde >= 0`; there is no lower electrolyte
  slot and no vertical metal sidewall.
- The `y = 0` boundary is split into Au, exposed insulating substrate, and Pd.
- Au/Pd use Stern--Robin electrostatic conditions and carry their respective
  reactions.
- The substrate gap is ideal, uncharged, and nonreactive, with homogeneous
  Neumann electrostatic condition.
- The normal far field is imposed analytically by
  `exp(-gamma_n y_tilde) -> 0` as `y_tilde -> infinity`; there is no finite
  top boundary. Left/right boundaries are reflecting.
- `d_Au_Pd` is the exposed substrate width between the flush electrodes.
- Au and Pd share `E_mix` through an ideal external wire.
- The mixed-potential condition uses signed absolute currents:
  `I_Au + I_Pd = 0`.
- First version: linearized PB/Debye--Huckel + Stern + Frumkin BV.

## Baseline

- Source:
  `../../Figures/Figure_same_length_i0_alpha/inputs/params_same_length_i0_alpha050_au25_pd25_20260528_111255.json`
- `L_Au = L_Pd = 25 nm`
- `d_Au_Pd = 0--100 nm`
- `C_tot = 10 mM`
- `C_H_Au / C_H_Pd = 20 / 40 uF/cm2`
- `pzc_Au / pzc_Pd = 0.93 / 0.78 V`
- `it0_1 = it0_2 = 1.852573885166257e-4 A/m2`
- `alpha1 = alpha2 = 0.5`
- `out_of_plane_width = 0.01 m`
- No C/support charging parameters exist in this schema.

## Active v0.3 numerical baseline

- Backend: `semi_infinite_cosine_fourier`.
- Result schema: version `2`.
- Formal production defaults: `N_modes = 960`, `Nx = 5000`.
- The active discretization has `N_modes + 1` spectral coefficients. `Nx`
  controls boundary quadrature and saved surface sampling; it is not a count
  of finite elements or degrees of freedom.
- The far-field condition is exact in the basis; no finite-height convergence
  parameter exists.
- Validation includes the homogeneous Robin closed form, affine-system
  residual/direct-solve checks, the adjacent d=0 regression, spectral
  convergence at `N_modes = 240/480/960`, and independent `Nx` convergence.
- New outputs use `spectral_field_*.npz` and
  `spectral_metadata_*.json`; `run_manifest.json`, `summary.json`, and
  `artifacts.json` must all identify schema version 2 and the spectral backend.
- Spectral artifact metadata records internal boundary-augmented diagnostic
  surface samples separately from the uniform surface arrays saved in NPZ;
  `n_x_coordinates` always denotes the saved array length.
- Active result directory: `results/20260802_144648/`.
- Package/backend/schema: v0.3.0,
  `semi_infinite_cosine_fourier`, result schema `2`.
- `lambda_D = 3.0412178890177213 nm`.
- The scan contains 31 separation rows. The w/o-EDL reference is exactly
  separation independent:
  - `E_mix = 0.4670000000000000 V`;
  - `i_mix_avg = 0.1175603540787242 A/m2`.
- Representative with-EDL results:

| d (nm) | E_mix (V) | i_mix_avg (A/m2) | max abs phi_tilde |
| ---: | ---: | ---: | ---: |
| 0 | 0.6006949941 | 0.0815880736 | 5.985007 |
| 2 | 0.5996672634 | 0.0817905770 | 6.003685 |
| 3 | 0.5993920655 | 0.0819852829 | 6.008671 |
| 10 | 0.5988338456 | 0.0824895117 | 6.018786 |
| 100 | 0.5987867357 | 0.0825382481 | 6.019778 |

- `validation.json` passed the exact homogeneous Robin limit, affine vs.
  direct solve, `N_modes = 240/480/960` convergence, independent
  `Nx = 2501/5000` convergence, d=0 legacy-Fourier regression, and the
  75/100 nm plateau check.
- At d=0, the new result differs from the legacy N=960/Nx=5000 Fourier
  reference by `7.7335e-12 V` in `E_mix` and `2.6233e-9` relative in
  `i_mix_avg`.
- Maximum mixed-current relative balance residual over the scan:
  `2.0182e-11`.
- The saved spectral metadata distinguishes the boundary-enhanced diagnostic
  grid from the 5000-point saved profile grid. Every
  `saved_surface_n_x_coordinates` / `n_x_coordinates` value matches the
  corresponding `spectral_field_*.npz` arrays.
- Output contract: 57 files total = 11 CSV + 14 JSON + 8 NPZ + 12 PNG +
  12 SVG + 0 PDF. The figures comprise 4 RP PNG/SVG pairs, 2 trend PNG/SVG
  pairs, and 6 Figure-3 PNG/SVG pairs.
- The linearized-PB caveat remains active because the scan maximum is
  `max |phi_tilde| = 6.0197783533 > 1`; the result is an internal
  geometry-sensitivity calculation, not an absolute nonlinear-PB prediction.

## d = 1000 nm polarization schematic (post-processing extension)

- Source result: `results/20260802_144648/`; the original scan manifest,
  artifacts registry, and 0--100 nm plateau contract remain unchanged.
- Standalone `d_Au_Pd` validation now permits 0--1000 nm.  The standardized
  overlap scan in `scan.py` intentionally remains restricted to 0--100 nm.
- The 1000 nm domain uses explicit long-gap numerical overrides
  `N_modes = 4800` and `Nx = 20000`, instead of the source 960/5000.  The
  adjacent convergence point `N_modes = 3840, Nx = 20000` differs by
  `1.021 uV` in `E_mix` and `0.00291%` in `i_mix_avg`.
- Production values:
  - with EDL: `E_mix = 0.5987851675838214 V`,
    `I_mix = 4.127089323472036e-11 A`,
    `i_mix_avg = 0.08254178646944071 A/m2`;
  - w/o EDL: `E_mix = 0.46699999999999997 V`,
    `I_mix = 5.878017703936211e-11 A`,
    `i_mix_avg = 0.11756035407872421 A/m2`;
  - `delta E_mix = +0.13178516758382142 V` and
    `I_mix` drops by `29.7877357411%`.
- The 1000 nm value differs from the saved 100 nm plateau by only
  `-1.568109 uV` in `E_mix` and `+0.004287%` in `i_mix_avg`.
- Script and outputs are under
  `results/20260802_144648/figures/Polarization_Scheme/`:
  - `make_polarization_scheme_d1000nm.py`;
  - one `polarization_scheme_d1000nm_*.png` and matching editable-text SVG;
  - `*_curves.csv`, `*_convergence.csv`, `*_effective_params.json`, and
    `*_metadata.json` for traceability;
  - zero PDF and zero `__pycache__` deliverables.
- The figure uses signed absolute half-reaction currents (`I_Au > 0`,
  `I_Pd < 0`) and the exact balance `I_Au + I_Pd = 0`; display current is
  `I_A * 1e9` in units of `10^-3 uA`.
- The Debye--Huckel caveat still applies:
  `max |phi_tilde| = 6.0198315890 > 1`.

## d = 1000 nm high-resolution 2D and Figure 3 bundle

- Output: `results/20260802_144648/figures/d1000nm_high_resolution/`; the
  parent `20260802_144648` artifacts and manifest are not overwritten.
- Generator: `make_d1000nm_figure_set.py`.
- Production resolution: `N_modes = 3840`, `Nx = 20001`; refinement rows for
  960/5001, 1920/10001, and 3840/20001 are saved under `convergence/`.
- Production values: `E_mix_with = 0.5987841470159301 V`,
  `i_mix_avg_with = 0.0825441910814434 A/m2`, and
  `max |phi_tilde| = 6.019730740675451`.
- Over the central surface interval `x = 75--975 nm`, at least 50 nm from
  either metal/substrate junction, residual peak-to-peak potential is
  `0.0094654522 mV` (`< 0.02 mV` audit threshold).
- Figure outputs: 12 PNG + 12 SVG + 0 PDF. In the canonical Figure 3 set,
  spatial panels b--e use a broken x axis with windows `0--60 nm` and
  `990--1050 nm`; panels a/f remain unchanged. The `active_zoom/` copies are
  retained for path compatibility, and the two 2D maps use the same windows.
- The omitted interval is `60--990 nm` (930 nm); the visible middle lanes are
  explicitly labelled `insulating substrate`, not legacy charged support.

## Required RP figures

- `d_Au_Pd = 0, 2, 3, 10 nm`
- Shared Phi_s and concentration color limits across all four plots
- Solution view is `0 <= y <= 5 lambda_D`
- Semi-infinite spectral coefficients and the finite display grid are saved
  separately in NPZ files.
- Exactly four PNG and four SVG files; no PDF
- Figure-3/RP labels identify the middle segment as insulating substrate;
  reaction-only profiles are blank there.

## Archived v0.2.0 finite-height Q1 flush-substrate run

Everything in this section describes the frozen historical directory below,
not the active v0.3 spectral backend. Its old metadata, numerical controls,
artifact names, and values are preserved only for traceability.

- Historical package/backend: v0.2.0, `scikit-fem = 12.0.2`, quadrilateral
  Q1 discretization.
- Historical finite top: `H = 10 lambda_D`; nominal full-domain cell size
  `0.2 lambda_D`, with `0.05 lambda_D` junction refinement and at least four
  cells across each nonzero gap.
- Historical validation included 8/10/12-`lambda_D` finite-top checks and
  base/refined discretization checks.
- Historical field artifacts retain their original `fem_*` names. They are
  not part of the active result schema.

- Result directory: `results/20260802_024804/`
- Run tag: `flush_au25_pd25_equal_i0_alpha050_ctot10mM`
- Source package version: `0.2.0`
- Geometry metadata:
  `flush_coplanar_on_uncharged_insulating_substrate`
- The archived d=10 nm discretization metadata confirms one complete upper rectangle with
  `10692` nodes and `10465` Q1 elements.  Its named lower boundary contains
  Au, `substrate_gap`, and Pd facets; `substrate_gap` has homogeneous Neumann
  electrostatic BC.  No `far_slot`, `slot_walls`, or lower far field exists.
- `lambda_D = 3.0412178890177213 nm`
- separation rows: 31, including 0--10 nm by 0.5 nm, the required larger
  values, and exact 1/2/3/5 `lambda_D` positions.
- w/o EDL is exactly separation independent:
  - `E_mix = 0.4670000000000000 V`
  - `i_mix_avg = 0.1175603540787242 A/m2`
- with-EDL representative results:

| d (nm) | E_mix (V) | i_mix_avg (A/m2) | E overlap vs 100 nm (mV) |
| ---: | ---: | ---: | ---: |
| 0 | 0.6006820512 | 0.0816052765 | 1.912548 |
| 2 | 0.5996510953 | 0.0818090324 | 0.881592 |
| 3 | 0.5993752357 | 0.0820053036 | 0.605732 |
| 10 | 0.5988155273 | 0.0825130276 | 0.046024 |
| 100 | 0.5987695033 | 0.0825594159 | 0 |

- The d=0 value is unchanged because the former slot already vanished in that
  limit.  At d=10 nm, replacing the old T-domain slot by the flush substrate
  shifts `E_mix` by `+0.726540 mV` and changes `i_mix_avg` by `-1.5193%`
  relative to the legacy run.
- 75/100 nm plateau check passed:
  - `|delta E| = 1.1168e-9 V`, below `0.1 mV`.
  - relative `delta i = 2.0614e-8`, below `0.5%`.
- maximum mixed-current relative balance residual over all scan rows:
  `2.0394e-11`.
- `validation.json` passed all d=0 regression, homogeneous Robin,
  affine-vs-direct, 8/10/12-lambda far-field, and base/refined mesh checks.
- Package tests recorded for that archived implementation: `45 passed`.
- Output contract passed:
  - RP: 4 PNG + 4 SVG + 0 PDF;
  - trends: 2 PNG + 2 SVG + 0 PDF;
  - Figure 3: 6 PNG + 6 SVG + 0 PDF;
  - total: 12 PNG + 12 SVG + 0 PDF.
- The RP lane and Figure-3 substrate span are light gray and labelled
  `insulating substrate`; panel d/e reaction quantities remain blank there.
- Linearized-PB caveat remains active: scan maximum
  `max |phi_tilde| = 6.0189321091 > 1`.  The flush geometry is more faithful
  to a coplanar device, but these values remain internal linear-model
  sensitivity results rather than absolute quantitative predictions.

### Archived v0.2.0 Gouy--Chapman diagnostic

- On Au/Pd, the metal-side Stern charge is
  `sigma_M = C_H,M_eff (E_mix - PZC_M - phi_s)` with
  `C_H,M_eff = g_M epsilon_s / lambda_D`.
- Positive `sigma_M` means positive metal charge.
- The primary material diagnostic is
  `l_GC = 2 epsilon_s R T / (F <|sigma_M|>_boundary)`; local ranges and
  zero crossings are reported separately.
- The insulating substrate is prescribed uncharged and has no Stern/PZC
  model, so no substrate `sigma` or `l_GC` is reported.
- `l_GC` remains a metal-charge-derived diagnostic.  `lambda_D` is the
  explicit screening length in the linearized-PB operator, so `l_GC` is not
  treated as an independent or unique overlap cutoff.
- In the archived `results/20260802_024804/`, the paper benchmark
  `10 uC/cm2 -> l_GC` is `0.3569561472 nm`.
- Across that archived 31-point scan:
  - Au mean `l_GC = 0.9899886239--1.0095096043 nm`;
  - Pd mean `l_GC = 1.3149730290--1.3849585992 nm`;
  - both metals remain negatively charged without a zero crossing.
- Zero-offset 0--20 nm fits relative to the 100 nm plateau give:
  - `E_mix` decay length `2.6353733506 nm`, `R2 = 0.9998858859`;
  - `i_mix_avg` decay length `3.7516865913 nm`, `R2 = 0.9714265676`.

## Legacy T-domain/open-slot run: `results/20260801_175012/`

This directory is preserved unchanged as a historical result from the former
Au|open electrolyte slot|Pd topology.  It is **not** the latest/current result
for the flush-substrate model.  In that former topology, electrolyte occupied
a lower slot for `d > 0`, while the vertical metal walls were insulating and
nonreactive.  All values and output descriptions in the remainder of this
section apply only to that legacy topology.

- Result directory: `results/20260801_175012/`
- Run tag: `au25_pd25_equal_i0_alpha050_ctot10mM`
- `lambda_D = 3.0412178890177213 nm`
- separation rows: 31, including 0--10 nm by 0.5 nm, the required larger
  values, and exact 1/2/3/5 `lambda_D` positions.
- w/o EDL is exactly separation independent:
  - `E_mix = 0.4670000000000000 V`
  - `i_mix_avg = 0.1175603540787242 A/m2`
- with EDL representative results:

| d (nm) | E_mix (V) | i_mix_avg (A/m2) | E overlap vs 100 nm (mV) |
| ---: | ---: | ---: | ---: |
| 0 | 0.6006820512 | 0.0816052765 | 2.633558 |
| 2 | 0.5990159461 | 0.0826771560 | 0.967453 |
| 3 | 0.5986736721 | 0.0830596250 | 0.625179 |
| 10 | 0.5980889875 | 0.0837859855 | 0.040494 |
| 100 | 0.5980484934 | 0.0838383467 | 0 |

- 75/100 nm plateau check passed:
  - `|delta E| = 1.0674e-9 V`, below `0.1 mV`.
  - relative `delta i = 2.1758e-8`, below `0.5%`.
  - That legacy run therefore labelled 100 nm as its validated no-overlap
    plateau.
- maximum mixed-current relative balance residual over all scan rows:
  `2.563e-11`, below `1e-8`.
- d=0 legacy high-resolution regression passed:
  - `|delta E_mix| = 1.2943e-5 V = 0.01294 mV`.
  - relative `delta i_mix_avg = 2.1085e-4 = 0.0211%`.
- homogeneous Robin, affine-vs-direct, 8/10/12-lambda far-field, and
  base-vs-refined mesh validations all passed; full report is
  `results/20260801_175012/validation.json`.
- Required RP outputs passed visual/data QA:
  - exactly 4 PNG + 4 SVG + 0 PDF in `figures/rp_2d/`;
  - PNG resolution 600 dpi export, display grid 600 x 320;
  - SVG files retain editable text;
  - shared limits are `Phi_s = +/-160 mV`, Red1 `1e-3--1`, Ox2 `1--1e3`;
  - each full FEM field includes the lower slot for d>0; d=0 has no slot.
- linearized-PB caveat was active: representative
  `max |phi_tilde| = 5.984--6.031 > 1`.  Those values are an internal
  geometry-sensitivity calculation, not an absolute quantitative prediction.

### Legacy Gouy--Chapman diagnostic and 0--20 nm trend zoom

- The source definition is
  `l_GC = 2 epsilon_s R T / (F |sigma|)` from `../../MS/cs5c06754.pdf`.
  The paper applies `sigma` to a charged support between supported particles;
  the legacy open-slot model had no charged gap support.
- The legacy model reports metal-side Stern charge
  `sigma_M = C_H,M_eff (E_mix - PZC_M - phi_s)` with
  `C_H,M_eff = g_M epsilon_s / lambda_D`.  Positive `sigma` means positive
  metal charge.
- The primary material diagnostic uses
  `2 epsilon_s R T / (F <|sigma_M|>_boundary)`.  Boundary means use the FEM
  facet quadrature; local ranges and zero crossings use Q1 boundary nodes.
- For the paper benchmark `|sigma| = 10 uC/cm2`, that run's `epsilon_s` and
  `T = 298 K` give `l_GC = 0.3569561472 nm`, reproducing the quoted `0.36 nm`.
- At `d = 0 nm`:
  - Au: `mean sigma = -0.03535936118 C/m2`,
    `mean l_GC = 1.0095096043 nm`.
  - Pd: `mean sigma = -0.02577377745 C/m2`,
    `mean l_GC = 1.3849585992 nm`.
- Across all 31 separations:
  - Au mean `l_GC = 0.9827548013--1.0095096043 nm`; complete local envelope
    `0.6615111583--1.0160831953 nm`.
  - Pd mean `l_GC = 1.2990527003--1.3849585992 nm`; complete local envelope
    `0.6838468075--1.8852439585 nm`.
  - Both metals remain negatively charged with no charge-zero crossing.
- The bulk Debye length is `lambda_D = 3.0412178890 nm`.  Zero-offset fits
  over the 27 points at `0 <= d <= 20 nm`, relative to the validated 100 nm
  plateau, gave:
  - `E_mix`: decay length `2.1366866178 nm`, `R2 = 0.9977427919`.
  - `i_mix_avg`: decay length `2.7232315614 nm`, `R2 = 0.9970263764`.
- Therefore a single Gouy--Chapman length did not determine the legacy
  model's overlap range; the fitted mixed-potential/current lengths also
  included geometry and kinetics.
- Added outputs for `results/20260801_175012/`:
  - `figures/trends/emix_imix_vs_separation_0_20nm_au25_pd25_equal_i0_alpha050_ctot10mM.png/svg`;
  - `csv/gouy_chapman_length_vs_separation.csv` with 62 rows;
  - `gouy_chapman_analysis.json` and corresponding `run_manifest.json`
    metadata.
- The trend directory contains 2 PNG + 2 SVG + 0 PDF.  The original
  0--100 nm PNG/SVG and `separation_scan.csv` hashes are unchanged.  The
  31 diagnostic recomputations match the original `E_mix` and `i_mix_avg`
  values exactly within stored floating-point precision.
- Package verification after this addition was `41 passed`.

### Legacy Figure-3-style direct comparison at d = 10 nm

- Six matched panels are retained under
  `results/20260801_175012/figures/Figure_3/`, with exactly 6 PNG + 6 SVG
  + 0 PDF and editable SVG text.
- The historical geometry was Au|open electrolyte gap|Pd = 25|10|25 nm.  It
  was length-matched against
  `../../Figures/Figure_same_length_i0_alpha/Figure_3/`, whose middle 10 nm
  region was a carbon boundary.
- The legacy panels were:
  - a: `E_mix` and common-area `i_mix` for with EDL versus w/o EDL;
  - b: solution potential along `y = 0` through the open gap;
  - c: Red1/Ox2 normalized concentrations along `y = 0`;
  - d: local overpotential on the Au/Pd reaction planes;
  - e: signed local current density on Au/Pd;
  - f: equilibrium, mixed, and Au/Pd PZC reference potentials.
- The legacy open gap was shaded pale blue.  Panels d/e left it blank because
  it had no reaction plane.  No support/C or support PZC appeared in that set.
- Values used in the legacy panel a/f were
  `E_mix_with = 0.5980889874770864 V`, `E_mix_no = 0.467 V`,
  `i_mix_with = 0.08378598550976511 A/m2`, and
  `i_mix_no = 0.11756035407872421 A/m2`.
- Historical traceability is in
  `figures/Figure_3/figure_3_metadata.json` and `run_manifest.json` inside
  `results/20260801_175012/`.
- Full package verification at the time was `41 passed`.
