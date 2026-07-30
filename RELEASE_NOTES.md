# BraggCalculator 0.4.0

Release date: 30 July 2026

## Refinement workflow

- `refine_structure()` loads observed XY/XYE data and a candidate structure,
  runs a declared staged refinement, and returns one structured result.
- Two-theta and Q pattern coordinates are accepted. Q data use the supplied
  wavelength for an exact constant-wavelength coordinate conversion.
- `RefinementResult` carries the refined structure and CIF, calculated profile,
  residual, objective trace, convergence evidence, parameter values and
  physical bounds, fit statistics, identifiability diagnostics, warnings, and
  provenance.
- `RefinementPolicy` covers scale, background, calibration, profile, lattice,
  fractional coordinates, occupancies, isotropic displacement, anisotropic
  displacement, rigid bodies, restraints, robust objectives, restarts, and
  held-out checks.
- `PhaseMixtureSession` supports profile-area phase-fraction refinement.

## Atom-site permutation

- `SpeciesAssignmentConfig` declares fixed sites, allowed species, site groups,
  search limits, composition rules, mixed-occupancy handling, displacement
  transfer, optional oxidation states, and ambiguity tolerance.
- Pairwise, complete, bounded, and seeded-random searches operate on
  asymmetric-unit sites.
- The search checks composition and Wyckoff multiplicity, removes
  symmetry-expanded duplicates, screens every retained structure, and runs
  continuous refinement on the best candidates.
- Ranked candidates record site changes, screening and refined scores,
  convergence, refined CIFs, truncation, and experimental ambiguity.

## Compatibility

- BraggCalculator 0.3.0 simulation, artifact, and batched Torch APIs remain
  available.
- Python 3.12 and 3.13 are supported.
- The `refinement` installation extra installs PyTorch.

## Scientific scope

The focused refinement session accepts two-theta and constant-wavelength Q
data. Q coordinates are converted with the supplied wavelength for the angular
instrument model. Local covariance and identifiability describe the declared
forward model near the refined solution. The species-assignment result reports
a group of indistinguishable candidates when the supplied PXRD pattern does
not select one assignment within the configured tolerance.
