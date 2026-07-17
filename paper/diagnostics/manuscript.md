# Explaining structural ambiguity across the powder-diffraction information ladder

**Working diagnostics-paper manuscript — benchmark version 1.0.0**

## Abstract

Agreement between a calculated and measured powder pattern does not establish
that a structural model is unique. Complex phases are lost when structure
factors become intensities, reciprocal directions collapse during powder
averaging, and experimental broadening can merge the remaining differences.
We present a relationship-aware diagnostic framework that compares candidate
models at successive information levels and reports where their distinctions
disappear. Its central complex-valued diagnostic maps each matched reflection
to a bounded amplitude--phase disk and decomposes total mismatch into amplitude
and phase components. A frozen synthetic benchmark tests crystallographic
invariance, an exact periodic homometric pair, a controlled near-homometric
perturbation, lattice-compatible multi-element candidates, and a
resolution-limited strained-cell pair. Conventional cosine, Pearson,
Jensen--Shannon and Gaussian-weighted cross-correlation profile measures all
report essentially perfect agreement for the exact homometric pair, whereas
the shell-balanced complex metric reports phase mismatch 0.374 with amplitude
mismatch below 6e-16. For the strained pair, profile cosine changes from
0.999665 at FWHM 0.20 inverse angstrom to 0.937674 at FWHM 0.015 inverse
angstrom. The results demonstrate how a diagnostic can distinguish phase-loss
ambiguity from resolution-limited ambiguity without claiming experimentally
unobserved phases. Numerical gates pass; external blinded crystallographer
review of the explanations remains pending.

## 1. Scope and scientific question

The intended use is candidate-guided characterization: scientists already
have one or more plausible periodic structures and need to know whether a
powder experiment can distinguish them, why their patterns agree, and what
measurement or refinement operation is justified next. The method is not an
arbitrary structure-solution algorithm.

The central question is not only

> How similar are the final powder profiles?

but

> At which information-losing transformation do structurally different models
> become similar?

The calculation follows

\[
\rho(\mathbf r)\rightarrow F(\mathbf G)\rightarrow |F(\mathbf G)|^2
\rightarrow I_{\mathrm{stick}}(Q)\rightarrow I_{\mathrm{profile}}(Q).
\]

The complex structure factor retains calculated amplitude and phase. Squared
amplitudes discard phase. Powder averaging discards reciprocal direction.
Broadening and noise can then erase differences that survive the earlier
steps. This framing follows the general inverse-problem warning that distinct
structures can be homometric and that strong powder agreement need not imply
structural accuracy.

## 2. Methods

### 2.1 Relationship-aware comparison

Candidate pairs are classified before diagnostics are selected:

- regime I: equivalent or lattice-compatible structures;
- regime II: commensurate structures with a valid common reciprocal mapping;
- regime III: unrelated lattices without an intrinsic hkl-phase
  correspondence.

Complex reflection-wise diagnostics are disabled in regime III. Such pairs
are compared using powder profiles, Q windows, pair distributions and
experimental discriminability. This prevents a plotted phase difference from
being invented where crystallography supplies no mapping.

### 2.2 Bounded amplitude--phase mismatch

After valid setting and relative-origin alignment, let
\(a_h=|F_A(h)|\), \(b_h=|F_B(h)|\), and \(\Delta\phi_h\) be the wrapped phase
difference. Each reflection is represented by

\[
x_h=\frac{b_h-a_h}{a_h+b_h+\epsilon},\qquad
y_h=\frac{2\sqrt{a_hb_h}}{a_h+b_h+\epsilon}
\sin\left(\frac{\Delta\phi_h}{2}\right).
\]

The radius obeys

\[
r_h^2=x_h^2+y_h^2=
\frac{|F_B(h)-F_A(h)|^2}{(|F_A(h)|+|F_B(h)|+\epsilon)^2}\leq1.
\]

For normalized reflection weights \(w_h\),

\[
D_{\mathrm{SF}}^2=D_{\mathrm{amp}}^2+D_{\mathrm{phase}}^2,
\]

with each component the weighted RMS of its disk coordinate. Calculated phases
are used only to explain model-to-model differences. They are never described
as phases measured by a conventional powder experiment.

### 2.3 Reflection weighting benchmark

Four explicit weighting choices are retained:

1. uniform reflection weights;
2. mean candidate intensity;
3. square root of mean candidate intensity;
4. shell-balanced mean intensity.

For shell-balanced intensity weighting, every 0.5 inverse-angstrom shell has
equal total influence, while reflections inside a shell are weighted by mean
candidate intensity. The benchmark reports every result rather than selecting
a favorable weight after looking at the answer.

Uniform weighting gives large populations of weak and numerically extinct
high-Q reflections substantial influence. Pure intensity weighting suppresses
that instability but can concentrate the score in a few strong low-Q
reflections. Shell balancing is the declared primary visualization because it
suppresses numerical extinctions while preserving resolution coverage. It is
not claimed to be universally optimal.

### 2.4 Profile-similarity baselines

Profiles on the same Q grid are compared by:

- cosine similarity;
- centered Pearson correlation;
- Jensen--Shannon similarity after non-negative area normalization;
- normalized Gaussian-weighted cross-correlation.

The final measure is a transparent implementation in the generalized
weighted cross-correlation family used in powder-pattern comparison. Its
Gaussian lag kernel tolerates small peak shifts. It is named by the exact
implemented kernel and is not presented as a reproduction of a proprietary
program.

### 2.5 Frozen synthetic cases

The exact homometric case embeds two non-congruent subsets of the cyclic group
Z8 in a periodic P1 cell:

\[
A=\{0,3,4,5\},\qquad B=\{0,1,3,4\}.
\]

The analysis verifies that their directed periodic-difference multisets are
equal and that no translation or inversion plus translation maps one set to
the other. Identical Si scatterers at these positions therefore have equal
kinematic intensities while retaining different complex factors.

The near-homometric case moves one B site from fractional x=0.500 to 0.503.
The compatible case uses two synthetic NaSiO2 motifs with shifted oxygen
positions. The resolution case changes one 5.000 angstrom cell length to
5.025 angstrom and compares Gaussian Q-profile FWHM values 0.20 and 0.015
inverse angstrom.

All CIFs are synthetic, versioned and SHA-256 checked. They are mathematical
and software tests rather than independent experimental validation.

### 2.6 Invariance and release gates

Equivalent structures are generated by atom permutation, relative origin
shift, integer coordinate wrapping and rigid Cartesian rotation of the lattice
representation. The complex score must remain below 1e-10. Further gates test
the cyclic construction, exact intensity equality, nonzero phase detection,
extinction-stable amplitude, resolution transition and metric bounds.

Explanation quality is treated separately. A blinded packet asks external
crystallographers to score correctness, usefulness and whether the claimed
information-loss mechanism follows from the supplied evidence. The software
cannot pass this human gate itself.

## 3. Results

### 3.1 Exact and near homometry

For the exact pair, all profile baselines are 1.000000 to displayed precision.
With shell-balanced intensity weights, amplitude mismatch is 5.8e-16 and phase
mismatch is 0.373986, giving the same total complex mismatch. The disk is
therefore vertical in its physically weighted component: the structures have
equal amplitudes but different calculated phases.

Uniform weights instead give amplitude mismatch 0.115774. Inspection shows
that the contribution comes from near-zero numerical extinctions, whose tiny
denominators amplify floating-point differences. Mean-intensity and
shell-balanced intensity weights reduce the amplitude component below 7e-16.
This is why weighting is part of the reported scientific result rather than a
hidden plotting choice.

The near-homometric perturbation retains profile cosine 0.999936. Its
shell-balanced amplitude component rises to 0.025847 while phase mismatch
remains 0.374107. Ordinary profile agreement therefore remains saturated even
though exact intensity equality has been broken.

### 3.2 Compatible multi-element candidates

The synthetic NaSiO2 pair has profile cosine 0.999365 and Gaussian-weighted
cross-correlation 0.999438. Its shell-balanced complex mismatch is 0.038464,
decomposed into amplitude 0.027762 and phase 0.026622. Both types of model
difference are present, but broad profile agreement remains high.

### 3.3 Resolution-limited ambiguity

The strained-cell pair has profile cosine 0.999665 at FWHM 0.20 inverse
angstrom. At FWHM 0.015 inverse angstrom, cosine falls to 0.937674 and
Jensen--Shannon similarity to 0.713411. The same structural difference is
therefore practically hidden by the broad experiment and exposed by the
narrow one.

### 3.4 Robustness

All representation-equivalence tests produce complex mismatches below 1e-14,
comfortably inside the 1e-10 gate. Across Qmax from 3 to 6 inverse angstrom,
the coefficient of variation in the near-homometric total score ranges from
0.0177 to 0.0264 across the four schemes. No single weighting is declared
range-invariant; the variation is published for sensitivity analysis.

All numerical gates pass. The external-review gate is unsigned, so the
publication package status is `pending_external_review`, not `passed`.

## 4. Discussion

The exact construction demonstrates why an intensity-only score cannot be a
general structural metric. A score of one can mean either structural identity
or homometric ambiguity. The phase-aware disk separates these cases for model
diagnosis without implying phase measurement.

The resolution case demonstrates a different mechanism. Its ambiguity is not
fundamental phase loss: a higher-resolution powder experiment exposes the
distinction. A useful diagnostic should therefore recommend measurement
improvement for this case but complementary phase- or local-structure-sensitive
evidence for exact homometry.

Recent optimization studies report that powder-similarity landscapes can be
rough and contain spurious peak-overlap minima even for moderate structural
perturbations. The present results support using powder agreement for
candidate-guided, symmetry-aware refinement and explanation, not unrestricted
coordinate descent from arbitrary structures.

Published examples of several chemically plausible structures fitting the
same powder pattern further show why fit statistics alone should not force a
winner. Complementary PDF, solid-state NMR, energetic or alternative-scattering
evidence may be necessary.

## 5. Limitations

- The benchmark structures are synthetic and small.
- The exact homometric construction uses identical scatterers and ideal
  kinematic diffraction.
- Weighting conclusions may change with composition, Q range and experimental
  uncertainty.
- Gaussian broadening is a controlled resolution proxy, not a full instrument
  model.
- Profile baselines are implemented transparently but have not been validated
  against every external crystallographic program.
- External crystallographer review remains pending.
- No claim is made about arbitrary structure solution, quantitative phase
  analysis or certification-quality uncertainty.

## 6. Reproducibility

Run:

```bash
python scripts/run_diagnostics_publication.py --verify
```

The command verifies the input manifest, regenerates `results.json`,
`metric_table.csv`, the review packet and both figures in PNG, PDF and SVG.
`environment.json` records the generating Python and dependency versions.

## References

1. de Gelder, R., Wehrens, R. & Hageman, J. A. A generalized expression for
   the similarity of spectra: application to powder diffraction pattern
   classification. *J. Comput. Chem.* **22**, 273–289 (2001).
   DOI: 10.1002/1096-987X(200102)22:3<273::AID-JCC1001>3.0.CO;2-0.
2. Baake, M. & Grimm, U. Kinematic diffraction is insufficient to distinguish
   order from disorder. *Phys. Rev. B* **79**, 020203 (2009).
   DOI: 10.1103/PhysRevB.79.020203.
3. Schlesinger, C. et al. Ambiguous structure determination from powder data:
   four different structural models of 4,11-difluoroquinacridone with similar
   X-ray powder patterns. *IUCrJ* **9**, 406–424 (2022).
   DOI: 10.1107/S2052252522004237.
4. Segal, N. et al. The loss landscape of powder X-ray diffraction-based
   structure optimization is too rough for gradient descent. *Digital
   Discovery* **5**, 1590–1599 (2026). DOI: 10.1039/D6DD00017G.
