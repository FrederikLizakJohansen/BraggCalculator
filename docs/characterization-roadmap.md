# Materials Characterization and Refinement Roadmap

**Status:** active  
**Started:** 2026-07-17  
**Working branch:** `feature/reflection-coefficient-plane`

This document tracks the work needed to turn the differentiable diffraction
kernel and diagnostic prototypes into a tool that materials chemists can use
for candidate-guided powder characterization. It is updated with implementation
status, executable evidence, measured results and remaining limitations as work
proceeds.

The product remains deliberately narrower than arbitrary structure solution:
it starts from scientifically plausible structural models, refines only
declared parameter groups and reports when the supplied experiment does not
support a conclusion.

## Status key

- **Done:** implemented, tested and demonstrated.
- **In progress:** current implementation milestone.
- **Planned:** accepted scope, not yet implemented.
- **Research:** requires benchmarking or a scientific design decision before a
  production implementation is justified.

## Baseline at the start of this roadmap

The repository already contains:

- complex structure factors and the bounded amplitude--phase mismatch disk;
- profile discriminability and Jacobian/identifiability diagnostics;
- symmetry-compatible coordinate displacements;
- differentiable nuisance parameters and staged optimization;
- XY/XYE ingestion, provenance, candidate comparison, a CLI and an HTML report;
- a public NIST SRM 660c LaB6 real-data regression.

Baseline NIST scan `100a`, 20.3--50.0 degrees 2-theta:

| Quantity | Baseline |
|---|---:|
| Profile residual, Rwp | 0.19877 |
| Refined cubic lattice parameter | 4.1549091 A |
| Certified lattice parameter | 4.1568260 A |
| Lattice difference | -0.001917 A |

This baseline validates the data-to-report pipeline, but not a
certification-quality physical model.

## Milestone 1 -- Instrument-aware lattice refinement

**Status: Done**

### 1.1 Symmetry-aware lattice parameterization

Implement independent metric degrees of freedom determined by the loaded
crystal system rather than refining one uniform lattice scale.

Acceptance criteria:

- cubic, tetragonal, orthorhombic, hexagonal/trigonal, monoclinic and triclinic
  metric constraints are represented;
- the zero-displacement value exactly reconstructs the starting lattice;
- Torch gradients propagate through the parameterization;
- synthetic perturbed-cell data recover the allowed lattice variables;
- experimental results report named physical cell parameters.

### 1.2 Instrument and specimen profile

Add a documented, differentiable profile layer with the minimum corrections
needed for conventional Bragg--Brentano laboratory data.

Initial scope:

- separate Gaussian and Lorentzian Caglioti-like broadening contributions;
- a differentiable asymmetric peak component for axial-divergence-dominated
  low-angle tails;
- specimen-displacement peak shifts for Bragg--Brentano geometry;
- fixed or refined wavelength components;
- declared instrument geometry and parameter units in provenance.

Acceptance criteria:

- every profile component is area normalized on a sufficiently wide grid;
- symmetric limits reproduce the existing pseudo-Voigt behavior;
- analytical/autograd results agree with finite differences;
- synthetic profile parameters are recoverable;
- corrections can be disabled independently;
- the NIST example improves over the recorded baseline and still reports any
  scientifically important residual-model warning.

### 1.3 NIST validation

Use the public NIST SRM 660c scan and its supplied metadata as an external
regression, not as training data silently encoded into the model.

Acceptance criteria:

- dataset checksum and instrument assumptions appear in provenance;
- the example prints baseline and improved metrics;
- the refined cell is compared with the certified value and uncertainty;
- no claim of certification equivalence is made unless uncertainty and the
  full fundamental-parameters analysis have actually been reproduced.

### Milestone 1 measured result

The validation now uses all 5332 measured rows in NIST scan `100a`, including
the high-angle reflections needed to distinguish cell scale from angular
offsets. The session uses the reported 217.5 mm Bragg--Brentano radius and
-0.07877 mm specimen displacement, the NIST six-line Cu K-alpha spectrum and
the fixed calibrated zero shift.

| Quantity | Historical prototype | Milestone 1 |
|---|---:|---:|
| Data range | 20.3--50.0 degrees | 20.3--150.908 degrees |
| Measured points | 1045 | 5332 |
| Profile residual, Rwp | 0.19877 | 0.12073 |
| Refined cubic lattice parameter | 4.1549091 A | 4.1566837 A |
| Difference from certified value | -0.001917 A | -0.000142 A |

The lattice error improved by more than an order of magnitude, but remains
about 1.78 times the certificate's 95% expanded uncertainty. The milestone is
therefore an instrument-aware characterization result, not a reproduction of
the certification analysis. The largest known missing term is the full
graphite-analyzer passband/fundamental-parameters convolution.

Implemented evidence:

- point-group-invariant log-strain gives 1/2/3/4/6 cell metric modes and exact
  zero-displacement reconstruction;
- a synthetic tetragonal example recovers its two allowed modes with a maximum
  cell-parameter error of approximately 1.7e-6;
- split pseudo-Voigt normalization, symmetric limits, specimen-shift formula,
  autograd/finite-difference agreement and synthetic profile recovery are
  tested;
- six emission components reuse one exact structure-factor calculation;
- symmetry-equivalent reciprocal points are aggregated before profile
  rendering, reducing the full-scan differentiable run from multi-gigabyte
  graphs to a routine CPU calculation;
- CIF Uiso/Biso values are preserved and used by the scattering calculation;
- the full experimental assumptions and declared limitations are recorded in
  result provenance and the HTML report.

## Milestone 2 -- Full crystallographic refinement parameters

**Status: Done**

- symmetry-constrained site occupancies and shared-site simplexes; **done**
- positive isotropic displacement factors; **done**
- positive-semidefinite anisotropic displacement tensors; **done**
- composition, bond-length, angle and minimum-distance restraints; **done**
- rigid-body translations and rotations; **done**
- multiple phases and simplex-constrained phase fractions; **done**

Each parameter family requires a synthetic recovery example, physical-domain
tests and an identifiability warning for a deliberately correlated case.

The occupancy implementation distinguishes:

1. **composition mode**, which redistributes species on a shared site while
   keeping that site's total occupancy fixed; and
2. **vacancy mode**, which adds vacancy as a simplex component and can refine
   total site occupancy.

Occupancies and isotropic displacement factors are shared across every member
of a crystallographic orbit. The first executable gate is a mixed-site
synthetic recovery with a figure comparing target and recovered composition,
orbit Biso values, calculated profile and residual.

### Milestone 2A measured result

The executable mixed-site perovskite example starts from Sr0.70Ca0.30 and
generates a target with Ca=0.55 and orbit Biso values of 0.60, 0.35 and 1.10
square angstrom for the A, Ti and O sites. A controlled staged refinement
recovers all four target quantities to better than 1e-5.

An unrestricted whole-profile run deliberately exposes the identifiability
problem: it reaches Rwp=0.01340 but returns Ca=0.51572 while the displacement
parameters drift. The largest local occupancy--Biso correlation is about 0.90,
and the session emits an explicit warning. This is a successful diagnostic,
not a failed optimizer: several parameter combinations explain almost the same
powder signal.

Evidence artifacts:

- `demo/occupancy_adp_refinement.png` compares profiles, residuals, physical
  parameters, loss history and the local correlation matrix;
- `demo/occupancy_adp_report.html` contains the ordinary session report;
- `demo/refine_occupancy_adp.py` regenerates both artifacts;
- unit tests cover composition and vacancy simplexes, orbit sharing, positivity,
  autograd and the deliberately ambiguous whole-profile result.

### Milestone 2B acceptance gate

Anisotropic displacement tensors use the Cartesian convention

\[
T_{hj}=\exp\!\left(-\tfrac12\,\mathbf G_h^\mathsf T
\mathbf U_j\mathbf G_h\right),
\]

with \(\mathbf U\) in square angstrom. The isotropic limit must exactly match
the existing \(B_{iso}\) implementation through
\(\mathbf U=B_{iso}\mathbf I/(8\pi^2)\). Independent tensor modes must obey
site symmetry, propagate across crystallographic orbits and remain positive
definite throughout optimization.

Structural restraints must be differentiable, use a fixed periodic-image
topology during one refinement and report their standardized contributions
separately from the diffraction-data objective. The executable gate shows
tensor recovery and a deliberately weak structural refinement with and
without bond/angle restraints in one generated figure and HTML report.

### Milestone 2B measured result

A tetragonal synthetic case reduces a Cartesian symmetric tensor to its two
site-symmetry-allowed modes. Starting from isotropic U=0.006 square angstrom,
the session recovers target eigenvalues (0.004, 0.004, 0.014) square angstrom
with a maximum tensor-component error of 5.6e-5 square angstrom and
Rwp=0.000338. The matrix-exponential parameterization keeps every eigenvalue
positive, and its isotropic limit agrees with the established Biso path to
floating-point precision.

The deliberately sparse geometry example uses only eight strong reflections.
An unrestrained model reaches a weighted intensity loss of 6.7e-11 but returns
Si--O distances of 2.038 and 1.492 A and an O--Si--O angle of 140.50 degrees.
With declared bond, angle and minimum-distance information, refinement returns
1.620, 1.620 A and 109.50 degrees while retaining a near-exact data loss of
1.3e-8. This is a diffraction null space rather than an optimizer failure; the
chemical information is reported as a separate prior penalty.

Evidence artifacts:

- `demo/anisotropic_restraint_refinement.png` shows the profile, residual,
  displacement ellipse/eigenvalues and restrained versus unrestrained geometry;
- `demo/anisotropic_restraint_report.html` embeds the figure and numerical
  restraint evidence;
- CIF Uij/Bij ingestion, site-symmetry mode counts, positivity, autograd,
  isotropic equivalence and every restraint family have regression tests.

### Milestone 2C acceptance gate

Rigid bodies are explicitly declared, non-overlapping site groups. Their
internal Cartesian coordinates and pair distances must remain invariant while
three translation and three rotation coordinates remain differentiable. The
prepared reciprocal topology is complete, so intentionally symmetry-breaking
body motion remains calculable, but it must be reported as such.

Multi-phase refinement must distinguish candidate comparison from a physical
mixture. Phase contributions are combined in one calculated profile using a
positive simplex. The first implementation reports integrated profile-area
fractions, not uncalibrated mass fractions; conversion to quantitative weight
fractions requires a validated Rietveld scale convention. A deliberately weak
minor-phase example must emit a detectability/identifiability warning rather
than overstate the recovered fraction.

### Milestone 2C measured result

A triclinic four-site synthetic model declares three sites as one rigid SiO2
group and leaves a Na site fixed. From diffraction intensities alone, the six
pose coordinates recover a target translation of (0.06, -0.04, 0.02) A and a
rotation vector of (4.0, -2.5, 3.0) degrees to floating-point precision. All
three internal distances change by less than 9e-16 A. This demonstrates the
parameterization invariant and the differentiable recovery path; it is not a
claim that arbitrary powder data determine six pose coordinates uniquely.

A fixed-structure NaCl/CsCl physical mixture starts at 50/50 and is generated
at profile-area fractions 0.72/0.28. The shared-profile refinement returns
0.719914/0.280086 with Rwp=0.001035, while the softmax simplex sums to one to
machine precision. In a separate deliberately weak case, a 0.03% CsCl trace
component has standardized component norm 2.006, below the approximate
three-sigma threshold. The session emits an explicit unsupported-fraction
warning.

Evidence artifacts:

- `demo/rigid_multiphase_refinement.png` shows pose recovery, exact internal-
  distance invariance, the two-phase profile, fraction recovery and the trace-
  phase detection gate;
- `demo/rigid_multiphase_report.html` embeds the figure and numerical evidence;
- unit tests cover overlap rejection, exact rigid geometry, autograd, simplex
  positivity/sum, phase-fraction recovery and the weak-phase warning;
- the CLI separates alternative-candidate comparison from an explicitly
  requested `--mixture` run.

## Milestone 3 -- Calibrated uncertainty and identifiability

**Status: Done**

- physical parameter scales in the session Jacobian;
- restraint and prior contributions to the normal matrix;
- rank-aware covariance and null-direction reporting;
- correlated observation/background uncertainty;
- bounds-aware intervals, profile likelihood or bootstrap validation;
- coverage tests on repeated synthetic datasets.

Until this milestone passes, local covariance output remains diagnostic and
must not be presented as a certification uncertainty.

### Milestone 3 acceptance gate

The implementation must distinguish three sources of apparent certainty:

1. information supplied by the diffraction observations;
2. rank supplied only by restraints or priors; and
3. numerical regularization used to compute a generalized inverse.

A prior may make a posterior normal matrix invertible, but it must never cause
the corresponding direction to be labeled data-identifiable. Session
Jacobians must use declared characteristic steps with units, report data and
posterior ranks separately, and expose the dominant null-space combinations in
machine-readable form.

An optional positive-definite observation covariance must be honored by both
the refinement objective and diagnostics; marginal sigma values alone are not
an acceptable substitute when covariance is supplied. Bounds-aware parametric
bootstrap intervals will be the first non-linear interval method. They must
report boundary hits, failed replicates, random seed and empirical coverage in
a repeated-synthetic gate. These intervals remain conditional on the supplied
forward model and noise model.

The executable figure must include a well-identified case, a data-null
direction made finite only by a prior, correlated observations, bootstrap
coverage and a boundary-limited parameter. The HTML report must state which
uncertainty claims are local Gaussian approximations and which are empirical
bootstrap results.

### Milestone 3 measured result

The session now whitens residuals, Jacobians and pairwise model differences by
an optional full positive-definite observation covariance. The covariance
diagonal must agree with the supplied marginal sigma values, and a SHA-256 of
the matrix is recorded in provenance. Subsampled session Jacobians are scaled
back to the population information level and use declared characteristic steps
with physical descriptions.

Data, prior and posterior information are reported separately. In the
executable rank gate, identical occupancy and Biso response columns give data
rank 1/2 and one explicit null combination. A Biso prior raises posterior rank
to 2/2, while data rank remains 1/2 and the warning states that the finite
direction is prior-supplied. Standard errors are withheld when even the
posterior remains rank deficient.

The bounds-aware parametric-bootstrap gate uses fixed NaCl/CsCl phase profiles.
With correlated Gaussian observations, the 95% interval for a generating CsCl
profile-area fraction of 0.28 is [0.237836, 0.321471]. Across 200 independently
generated experiments, nominal 90% intervals cover the target 88.0% of the
time. In a boundary-limited 0.003 trace-phase case, 388 of 900 replicates hit
the zero bound, visibly invalidating a symmetric Gaussian error-bar summary.

Evidence artifacts:

- `demo/uncertainty_identifiability.png` shows covariance, singular spectra,
  the null direction, correlated bootstrap, repeated coverage and boundary
  pile-up;
- `demo/uncertainty_identifiability_report.html` embeds the figure and records
  assumptions, seeds, draw counts and limitations;
- `braggcalculator/uncertainty.py` provides the reusable bounds-aware
  parametric bootstrap;
- regression tests cover prior-supplied posterior rank, null vectors,
  correlated covariance whitening, covariance cropping/checksums, bootstrap
  standard errors, boundary hits and repeated-synthetic coverage.

## Milestone 4 -- Robust refinement mechanics

**Status: Planned**

- L-BFGS and damped Gauss--Newton/trust-region local solvers;
- Poisson likelihood for raw count data;
- coarse-to-fine peak-width continuation;
- adaptive parameter release based on residual support and correlations;
- rollback when a release step worsens validation;
- deterministic multistart policies and explicit convergence classifications.

## Milestone 5 -- General structural diagnostics

**Status: Planned / Research**

- automatic diffraction information-loss ladder classification;
- peak-group-to-site/orbit attribution;
- counterfactual site and motif substitutions;
- commensurate-cell and supercell comparison;
- superstructure reflection analysis;
- Patterson/PDF comparison;
- unrelated-polymorph powder and motif comparison;
- experimental-design recommendations across wavelength, radiation and
  resolution choices.

## Milestone 6 -- Reference validation

**Status: Planned**

- profile and refined-parameter comparisons against established refinement
  software;
- public datasets spanning different instruments and material classes;
- synthetic recovery matrices for all refinable parameter families;
- difficult cases including overlap, weak scatterers, preferred orientation,
  multiple phases and occupancy/displacement correlation;
- expert review of the generated diagnostic conclusions.

## Milestone 7 -- Scientist and agent interfaces

**Status: Planned**

- interactive linked structure, profile, peak-group and mismatch views;
- parameter tables with bounds, restraints, release state and provenance;
- saveable refinement projects and resumable optimization traces;
- versioned structured JSON schemas;
- REST/service operations and MCP tools;
- CIF, profile, table and audit-trail exports.

## Milestone 8 -- Publication package

**Status: Research**

- benchmark mismatch-disk weighting choices and invariance;
- curate homometric, near-homometric and resolution-limited examples;
- compare diagnostic scores with existing powder-similarity metrics;
- evaluate whether explanations agree with expert crystallographic reasoning;
- freeze versioned data, environments, figures and analysis scripts.

The first diagnostics paper should focus on explanation and experimental
discriminability. A full refinement paper should remain separate unless the
instrument, uncertainty and reference-validation milestones are complete.

## Progress log

### 2026-07-17

- Recorded the starting implementation and NIST quantitative baseline.
- Confirmed from the NIST pdCIF that scan `100a` used Cu K-alpha radiation, a
  post-specimen graphite analyzer, Bragg--Brentano geometry, a 217.5 mm
  specimen-to-detector distance and a reported vertical specimen displacement.
- Began Milestone 1.
- Completed Milestone 1 with symmetry-aware lattice refinement, TCH Gaussian/
  Lorentzian broadening, compact axial asymmetry, specimen displacement, the
  six-line Cu spectrum and full-scan NIST validation.
- Improved the NIST lattice error from -0.001917 A to -0.000142 A while keeping
  an explicit warning that the result lies outside certification uncertainty.
- Next implementation target: Milestone 2, beginning with symmetry-constrained
  occupancies and positive isotropic displacement parameters.
- Began Milestone 2 and fixed the distinction between composition-only and
  vacancy-enabled occupancy refinement before implementing the parameter API.
- Completed Milestone 2A: symmetry-constrained composition/vacancy simplexes,
  positive orbit-shared Biso parameters, session/CLI integration, physical
  reporting and occupancy--displacement correlation warnings.
- Demonstrated exact controlled recovery and a deliberately underdetermined
  joint fit in a generated six-panel figure and HTML diagnostic report.
- Began Milestone 2B by fixing the Cartesian anisotropic-displacement
  convention and separate reporting requirements for structural restraints.
- Completed Milestone 2B with site-symmetry-compatible positive-definite U
  tensors, CIF anisotropic ingestion, four differentiable restraint families,
  session/CLI integration and separate penalty provenance.
- Demonstrated that sparse diffraction can fit chemically incorrect geometry
  essentially exactly and that explicit restraints resolve the null direction
  without being misreported as diffraction evidence.
- Began Milestone 2C by defining exact rigid-body invariance and separating
  profile-area phase fractions from unvalidated quantitative weight fractions.
- Completed Milestone 2C with differentiable Cartesian rigid-body poses,
  fixed-structure physical mixtures, exact positive phase simplexes, CLI/API
  integration and profile-level trace-phase detectability warnings.
- Demonstrated six-mode rigid-pose recovery with sub-femtometre numerical
  distance invariance, 72/28 mixture recovery and an intentionally unsupported
  0.03% trace component in a generated figure and HTML report.
- Completed Milestone 3 with full observation-covariance whitening, physical
  Jacobian step metadata, separate data/prior/posterior ranks, explicit null
  combinations and restraint-aware posterior curvature.
- Added bounds-aware parametric bootstrap intervals and demonstrated 88.0%
  empirical coverage for nominal 90% intervals over 200 synthetic experiments,
  plus a trace-phase boundary pile-up that rules out symmetric error bars.
