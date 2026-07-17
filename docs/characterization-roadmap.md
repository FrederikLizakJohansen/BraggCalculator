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

**Status: Done**

- L-BFGS and damped Gauss--Newton/trust-region local solvers;
- Poisson likelihood for raw count data;
- coarse-to-fine peak-width continuation;
- adaptive parameter release based on residual support and correlations;
- rollback when a release step worsens validation;
- deterministic multistart policies and explicit convergence classifications.

### Milestone 4 acceptance gate

- one staged API must run Adam and L-BFGS while preserving declared parameter
  groups, and an independently tested damped Gauss--Newton solver must expose
  trust radius, damping, accepted steps and gain ratios;
- convergence must be classified from gradient norm, relative loss change,
  accepted steps or exhaustion, rather than from an optimizer return flag;
- the Poisson objective must reject negative observations, keep expected counts
  positive and report the mean deviance separately from Gaussian fit statistics;
- continuation must record the peak-width multiplier used by every stage and
  end at the physical multiplier of 1.0;
- adaptive release must record sensitivity, residual support, correlations and
  an explicit accepted/rejected reason for each candidate parameter group;
- every guarded stage must snapshot its parameters and restore them if held-out
  validation worsens beyond the declared tolerance;
- multistart runs must use reproducible seed provenance and retain the
  convergence classification and score of every attempt;
- an executable figure and HTML report must show a low-count Poisson example,
  continuation, damping/trust behavior, adaptive release, rollback and
  deterministic restart classifications.

### Milestone 4 measured result

The optimization layer now runs declared Adam and strong-Wolfe L-BFGS stages,
and provides a separate damped Gauss--Newton solver with gain-ratio trust
updates. Stage results retain training/validation evidence, the declared width
multiplier, gradient norm, acceptance reason and a convergence classification.
Rejected stages restore an exact tensor snapshot. The refinement session adds
a smooth-positive Poisson count objective, reports mean Poisson deviance,
supports coarse-to-fine profile stages, records adaptive structural-group
release decisions, and preserves the seed, score and classification of every
restart.

In the deterministic low-count gate, the generating peak amplitude is 3.0
counts: Poisson deviance estimates 2.82 while observed-count-weighted Gaussian
least squares estimates 1.86. Direct physical-width optimization remains at
the deliberately distant starting peak centre of -1.80, whereas the 8x to 3x
to 1x continuation reaches 2.00 and drives profile error below
\(10^{-9}\). Damped Gauss--Newton recovers the nonlinear target
\(a=1.000, p=2.000\) with a `gradient_converged` classification. The release
gate accepts the supported lattice direction and explicitly rejects an
unsupported occupancy, an insensitive ADP and a 0.995-correlated duplicate
mode. A deliberately overfit training step increases held-out loss from zero
to approximately four and is restored exactly. Six seeded non-convex restarts
retain both local basins and distinguish gradient convergence from step-budget
exhaustion.

Evidence artifacts:

- `demo/robust_refinement_mechanics.png` shows all six executable gates;
- `demo/robust_refinement_mechanics_report.html` embeds the figure and the
  machine-readable release/restart tables;
- `demo/robust_refinement_mechanics.py` regenerates both artifacts
  deterministically;
- regression tests cover L-BFGS, Gauss--Newton, continuation callbacks,
  validation restoration, adaptive release and a complete low-count Poisson
  diffraction session.

## Milestone 5 -- General structural diagnostics

**Status: Done / Research implementation**

- automatic diffraction information-loss ladder classification;
- peak-group-to-site/orbit attribution;
- counterfactual site and motif substitutions;
- commensurate-cell and supercell comparison;
- superstructure reflection analysis;
- Patterson/PDF comparison;
- unrelated-polymorph powder and motif comparison;
- experimental-design recommendations across wavelength, radiation and
  resolution choices.

### Milestone 5 acceptance gate

- every comparison must declare whether the prepared structures are
  equivalent, lattice-compatible, integer/rationally commensurate, or
  unrelated, and must never expose an `hkl` phase comparison for an unrelated
  pair;
- the information-loss ladder must report separately normalized complex,
  intensity, ideal-powder, broadened-profile and radial-pair similarities, then
  identify the largest information-hiding transition with an explanation;
- commensurate comparisons must record the integer cell transformation where
  one exists and identify supercell reflections that cannot be indexed in the
  parent cell, including their calculated intensity fraction;
- peak groups must be formed using a declared resolution width and report their
  reflection mixture entropy/effective count plus site-orbit counterfactual
  effects; counterfactual effects must be labelled non-additive because of
  interference;
- the radial Patterson/PDF diagnostic must include periodic neighbors,
  scattering weights, radiation type, broadening and a bounded similarity;
- unrelated structures must still receive powder-profile and radial-pair
  comparison, while complex and direct reflection-phase fields remain absent;
- experiment suggestions must recompute expected count discrimination for
  declared wavelength, radiation, Q range and resolution configurations and
  retain assumptions instead of returning an unexplained rank;
- deterministic tests, one multi-regime figure and a self-contained HTML
  report must cover equivalent, compatible, commensurate and unrelated pairs,
  superstructure peaks, counterfactual attribution, Patterson/PDF similarity
  and measurement ranking.

### Milestone 5 measured result

The new relationship gate classifies exact equivalent/origin-shifted models,
same-lattice decorated variants, integer or bounded-rational commensurate cells,
and unrelated lattices. Reciprocal vectors rather than raw Miller labels are
matched after a valid cell relationship. Regime III results set the complex and
intensity similarities to `None`, while retaining ideal-powder, broadened
profile, expected-count discrimination and periodic radial-pair comparisons.

In the executable compatible SiO2 example, similarity rises from 0.856869 for
aligned complex factors to 0.983362 after phase removal, 0.996230 after powder
projection and 0.997803 after the declared broadening. The diagnostic therefore
identifies phase loss as the dominant information-hiding transition. Its
radial-pair similarity is 0.603583 and is reported as an alternative real-space
view rather than another step in the diffraction ladder.

For an ordered 2x Si/P cell, the integer cell transformation is recovered and
118 calculated non-parent reciprocal points carry 0.174574% of the integrated
intensity in the evaluated Q range. Resolution-defined peak groups report the
reflection-mixture effective count and recalculated site-removal effects. In a
two-oxygen counterfactual, replacing O(1) alone has relative profile-effect norm
0.721116 and local alignment 0.417202; O(2) gives 0.763888 and 0.471699; replacing
both reproduces the full A-to-B profile direction with norm and alignment 1.0.
The separate effects are deliberately not added because interference makes the
decomposition non-additive.

The unrelated triclinic/hexagonal pair emits no complex metric, but reports
profile similarity 0.302389, radial-pair similarity 0.190419 and an expected
count separation under the declared measurement model. The experiment gate
ranks four configurations while preserving radiation, wavelength, Q range,
resolution, exposure/count scale, background and variance assumptions. In the
declared example the neutron exposure ranks first; among like-for-like Cu X-ray
setups, reducing FWHM from 0.25 to 0.04 inverse angstrom raises expected
separation from 5.88e6 to 2.53e8.

Evidence artifacts:

- `demo/general_structural_diagnostics.png` shows the four relationship regimes,
  information ladder, superstructure reflections, peak/site effects,
  counterfactuals, periodic radial PDF, experiment ranking and unrelated-pair
  powder evidence;
- `demo/general_structural_diagnostics_report.html` embeds the figure and the
  numerical relationship, similarity, attribution and measurement tables;
- `braggcalculator/structural_diagnostics.py` provides the reusable API;
- regression tests cover all regimes, the no-invented-phase gate,
  information-loss classification, site counterfactuals, peak groups,
  superstructure intensity, periodic PDFs and resolution recommendations.

Current research limitations are explicit: rational cell searches use a
bounded denominator rather than a full crystallographic common-cell search;
counterfactual substitution requires a contribution-wise site/species mapping;
the radial signal is a kinematic scattering-weighted periodic pair diagnostic,
not a fully corrected experimental total-scattering PDF; measurement ranking is
conditional on the declared count and background models.

## Milestone 6 -- Reference validation

**Status: Engineering matrix implemented; external gates open**

- profile and refined-parameter comparisons against established refinement
  software;
- public datasets spanning different instruments and material classes;
- synthetic recovery matrices for all refinable parameter families;
- difficult cases including overlap, weak scatterers, preferred orientation,
  multiple phases and occupancy/displacement correlation;
- expert review of the generated diagnostic conclusions.

Acceptance is now executable rather than narrative. `ValidationMetric` applies
explicit maximum/minimum pass and warning limits; `ValidationCase` retains every
metric; and `ValidationMatrix` fails missing required categories and never
averages a failed or unsupported case into a successful aggregate. The JSON
artifact records source provenance, units, thresholds, assumptions, warnings
and case status.

Implemented evidence:

- 12 X-ray/neutron line-pattern cases across six structures match pymatgen with
  maximum position and scaled-intensity errors below the declared tolerances;
- five checksummed public profiles cover NIST laboratory X-ray data, GSAS-II
  laboratory X-ray PbSO4 and fluoroapatite data, and constant-wavelength neutron
  PbSO4 and yttrium iron garnet data;
- the NIST SRM 660c full scan refines to `Rwp=0.120729` and a lattice error of
  `+0.000142 A`; the fit-statistic gate passes but the lattice gate correctly
  warns because the result remains outside the `0.000080 A` certified expanded
  uncertainty;
- synthetic gates pass for symmetry-allowed lattice and coordinate modes,
  scale, background, zero shift, width, composition, Biso, anisotropic U,
  rigid-body pose and positive phase fractions;
- difficult-case gates demonstrate a 42.94-fold resolution-dependent
  discrimination increase, 5.07-fold greater normalized hydrogen sensitivity
  for the declared neutron example, a sub-threshold 0.03% trace phase, and a
  detected 0.899 occupancy--Biso cross-group correlation with a warning;
- preferred orientation and time-of-flight physics remain visibly
  `unsupported`, rather than being approximated by the constant-wavelength
  model.

Current matrix: 28 pass, one warning, one pending software-review case, two
unsupported cases and zero failures. Overall status remains `unsupported`
because capability gaps are intentionally stronger than the passing numerical
cases.

Evidence artifacts:

- `demo/reference_validation.png` is the six-panel visual matrix;
- `demo/reference_validation_report.html` embeds the figure, all case gates,
  public sources, SHA-256 values and the expert-review checklist;
- `demo/reference_validation_results.json` is the machine-readable frozen run;
- `data/reference_validation/manifest.json` is the immutable public-data
  manifest;
- `braggcalculator/validation.py` is the reusable validation API.

Two release gates remain open. A frozen final GSAS-II project has not yet been
reproduced profile-for-profile with covariance comparison, and no external
crystallographer has signed the generated conclusion checklist. These are
reported as pending; neither is claimed complete.

## Milestone 7 -- Scientist and agent interfaces

**Status: Implemented**

- interactive linked structure, profile, peak-group and mismatch views;
- parameter tables with bounds, restraints, release state and provenance;
- saveable refinement projects and resumable optimization traces;
- versioned structured JSON schemas;
- REST/service operations and MCP tools;
- CIF, profile, table and audit-trail exports.

The interface layer now shares one versioned project and service model across
Python, the command line, HTTP and MCP. A project bundle copies and checksums
its pattern and CIF inputs, freezes the refinement policy, records an append-only
run lineage and writes one structured result per run. Continuation restores the
exact raw parameter groups from the previous run and starts a fresh optimizer;
optimizer moments are not serialized and every continuation is therefore a new,
explicit trace segment rather than an invisible extension.

The self-contained HTML workspace links:

- candidate selection and fractional structure projections;
- observed, calculated and residual profiles;
- resolution-defined peak groups and their contributing reflections;
- an origin-aligned mismatch disk when the starting models share a valid
  lattice representation;
- optimizer traces, informative regions and recommendations;
- parameter values, physical bounds, restraint weights, active/fixed state and
  provenance;
- JSON, profile CSV, parameter CSV, refined CIF and audit exports.

`ProjectStore` supplies create/run/resume/read/export behavior. The
`bragg-project` command exposes that lifecycle to scientists. `DiagnosticService`
provides versioned `simulate_pattern`, `compare_models`, `suggest_measurement`,
project create/run/resume/status/result and sensitivity-analysis operations; `bragg-service` exposes
them at `/v1/operations/<name>`. `bragg-mcp` provides the same operations as
structured MCP tools. MCP refuses structural release unless the caller supplies
`release_policy_acknowledged=true`.

Versioned contracts are checked into `braggcalculator/schemas/v1/`, while every
runtime document also carries a schema identifier. Project paths are confined
to the declared service root and all copied inputs are rechecked before a run.

The generated example contains two compatible candidate models and two linked
runs. Run 2 names run 1 as its parent and resumes the stored raw parameters. The
final candidates reach Rwp values of about 0.067, but expected pairwise
separation remains below the discrimination threshold; the UI therefore reports
that the synthetic experiment does not distinguish them instead of selecting a
winner. Its starting-model mismatch disk has `D_SF=0.1354`.

Evidence artifacts:

- `demo/scientist_agent_interface.png` summarizes fit, mismatch, run lineage,
  checkpoint trace segments, exports and shared operations;
- `demo/scientist_workspace_project/runs/run-0002/workspace.html` is the linked
  offline workspace;
- `demo/scientist_workspace_project/project.json` and `audit.json` demonstrate
  the portable project and audit formats;
- both run directories contain structured results, tables and refined CIFs;
- interface regression tests cover policy round trips, raw-parameter resume,
  checksums, exports, workspace content, service scope and the MCP release gate.

Current limitations are explicit. The workspace mismatch disk describes the
origin-aligned starting models, because a general refined-structure alignment
export is not yet available. Peak groups use a declared fixed `0.08 A^-1`
resolution proxy. The dependency-free HTTP transport is intended for trusted
local use and does not provide authentication or TLS. The MCP implementation is
a compact stdio JSON-RPC server rather than a hosted multi-user service.

### Milestone 7.5 -- Guided end-to-end characterization UI

**Status: Implemented**

Before beginning the publication benchmark, the scientist interface now also
provides a complete local browser workflow through `bragg-ui`:

- upload one XY/XYE pattern and one or more CIF candidates;
- declare wavelength, radiation, uncertainty-column meaning and a staged
  release policy;
- require explicit acknowledgement before structural coordinates,
  occupancies or displacement factors are released;
- create a portable, checksummed project before any optimization occurs;
- run every candidate fairly, or continue from the latest raw-parameter
  checkpoint as an explicit child run;
- inspect fit, standardized residual, structure relationship, complex
  amplitude--phase mismatch, pair distribution, resolution-defined peak
  groups, identifiability, physical parameters, experiment design, run lineage
  and exports in separate linked tabs;
- read a plain-language explanation, mathematical basis and concrete
  interpretation checklist beside every diagnostic family.

The application is dependency-free in the browser: plots are linked SVG and
all calculations are performed by the existing project/refinement/diagnostic
layers. It binds to localhost by default and remains a trusted-local tool
without authentication or TLS.

A bundled synthetic NaSiO2 dataset and two lattice-compatible candidates form
an executable tutorial. The example intentionally teaches an honest
non-discrimination conclusion: a good profile fit and a non-zero calculated
complex mismatch can coexist when the supplied powder experiment does not
contain enough candidate-separating information. The dataset is labeled as
synthetic throughout and is not counted as independent validation.

Evidence artifacts:

- `braggcalculator/ui/index.html` is the complete guided application;
- `braggcalculator/tutorial_data/` contains the copied/checksummed tutorial
  inputs and their scope statement;
- `demo/end_to_end_tutorial_project/` is a completed portable UI project;
- `demo/end_to_end_ui.png` records the running application with calculated
  tutorial results;
- `demo/end_to_end_ui_mismatch.png` and
  `demo/end_to_end_ui_identifiability.png` record the complex and local-rank
  interpretations;
- UI regression tests cover upload policy acknowledgement, tutorial creation,
  diagnostics, local HTTP routes and artifact path confinement.

## Milestone 8 -- Publication package

**Status: Implemented; external review pending**

- benchmark mismatch-disk weighting choices and invariance;
- curate homometric, near-homometric and resolution-limited examples;
- compare diagnostic scores with existing powder-similarity metrics;
- evaluate whether explanations agree with expert crystallographic reasoning;
- freeze versioned data, environments, figures and analysis scripts.

The first diagnostics paper should focus on explanation and experimental
discriminability. A full refinement paper should remain separate unless the
instrument, uncertainty and reference-validation milestones are complete.

The first diagnostics-paper package is now frozen under `paper/diagnostics/`.
One deterministic command verifies all input hashes, recalculates the full
benchmark, applies numerical release gates and regenerates machine-readable
results, tables, blinded review forms and publication figures:

```bash
python scripts/run_diagnostics_publication.py --verify
```

The curated synthetic matrix contains:

- a verified exact periodic homometric pair constructed from non-congruent
  cyclic Z8 subsets with equal directed difference multisets;
- a controlled near-homometric site perturbation;
- the lattice-compatible multi-element UI candidates;
- a 0.5% strained-cell pair evaluated under broad and high-resolution profile
  models;
- representation-equivalent atom permutations, origin shifts, coordinate
  wrapping and rotated Cartesian settings.

Four mismatch weighting declarations are reported: uniform, mean intensity,
square-root mean intensity and shell-balanced intensity. In the exact
homometric case, uniform weights produce a spurious amplitude component of
0.1158 from numerical extinctions. Mean-intensity and shell-balanced weights
reduce that component below 7e-16. The declared shell-balanced score retains a
phase component of 0.3740 while the ideal profile cosine is 1.000000. The
weighting choice is therefore visible scientific metadata rather than a hidden
plotting option.

Baseline profile comparisons include cosine, Pearson, Jensen--Shannon and a
transparent Gaussian-weighted cross-correlation metric. They all saturate for
the exact homometric pair. The resolution case changes from cosine 0.999665 at
FWHM 0.20 inverse angstrom to 0.937674 at FWHM 0.015 inverse angstrom, correctly
identifying an experimentally recoverable distinction rather than fundamental
phase-loss ambiguity.

All software and mathematical gates pass, including the input manifest,
homometric construction, representation invariance, phase detection,
extinction-stable amplitude decomposition, resolution transition and metric
bounds. The package status remains `pending_external_review`: no external
crystallographer has signed the blinded explanation review. The review
protocol requires at least two independent reviewers, per-case correctness and
usefulness medians of at least four out of five, no unresolved unsupported
mechanism judgment and a frozen response log.

Evidence artifacts:

- `paper/diagnostics/manuscript.md` is the working paper draft;
- `paper/diagnostics/results.json` and `metric_table.csv` contain the complete
  numerical result;
- `paper/diagnostics/figures/diagnostic_benchmark.*` compares information
  levels and profile baselines;
- `paper/diagnostics/figures/weighting_invariance.*` reports invariance,
  Q-range sensitivity and every release gate;
- `paper/diagnostics/expert-review.md` and the generated blinded packet define
  the remaining human gate;
- `data/publication_diagnostics/manifest.json` freezes all case inputs and
  provenance.

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
- Implemented the Milestone 6 validation record/gate API and a fixed public-data
  manifest with native GSAS constant-step STD/ESD ingestion.
- Ran 32 reference-validation cases: 28 pass, the NIST lattice comparison warns,
  direct frozen GSAS-II profile reproduction is pending, preferred orientation
  and TOF remain unsupported, and no case fails.
- Generated the reference-validation figure, self-contained HTML report and
  machine-readable JSON; retained external crystallographer review as an
  explicit unsigned release gate.
- Implemented Milestone 7 portable project bundles, versioned project/result
  schemas, exact raw-parameter continuation with separate trace segments, and
  refined-CIF/profile/parameter/audit exports.
- Added the linked offline scientist workspace, local REST operations,
  `bragg-project` lifecycle CLI and agent-safe MCP tools with an explicit
  structural-release acknowledgement gate.
- Generated a two-model, two-run interface project and companion six-panel
  figure; the example correctly concludes that its candidate profiles are not
  experimentally discriminated.
- Added the guided `bragg-ui` application with upload, declared release policy,
  run/resume controls, nine diagnostic tabs, theory/lay explanations, a bundled
  two-candidate tutorial and portable export links.
- Implemented the Milestone 8 diagnostics publication package with frozen
  homometric, near-homometric, compatible and resolution-limited cases; four
  mismatch weighting policies; four profile-metric baselines; invariance and
  Q-range sensitivity matrices; a working manuscript; one-command figure/data
  regeneration; and an explicitly unsigned external-review gate.
