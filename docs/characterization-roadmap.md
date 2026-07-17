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

**Status: In progress**

- symmetry-constrained site occupancies and shared-site simplexes; **done**
- positive isotropic displacement factors; **done**
- positive-semidefinite anisotropic displacement tensors;
- composition, bond-length, angle and minimum-distance restraints;
- rigid-body translations and rotations;
- multiple phases and simplex-constrained phase fractions.

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

## Milestone 3 -- Calibrated uncertainty and identifiability

**Status: Planned**

- physical parameter scales in the session Jacobian;
- restraint and prior contributions to the normal matrix;
- rank-aware covariance and null-direction reporting;
- correlated observation/background uncertainty;
- bounds-aware intervals, profile likelihood or bootstrap validation;
- coverage tests on repeated synthetic datasets.

Until this milestone passes, local covariance output remains diagnostic and
must not be presented as a certification uncertainty.

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
