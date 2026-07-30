# BraggCalculator 0.4.1

Release date: 30 July 2026

## Documentation

- The README now introduces pattern simulation, Q-space output, experimental
  effects, and batched pattern generation before the refinement workflow.
- The refinement example and species-assignment example remain unchanged.
- The API reference identifies the current package version as 0.4.1.

## Refinement

- `refine_structure()` accepts a CIF path, CIF text, or pymatgen structure.
- Observed patterns can use two-theta or constant-wavelength Q coordinates.
- `RefinementResult` contains the refined structure and CIF, calculated
  profile, residual, objective trace, convergence evidence, parameter bounds,
  fit statistics, identifiability diagnostics, warnings, and provenance.
- Asymmetric-unit species-assignment screening supports constrained and bounded
  searches followed by continuous refinement of the leading candidates.

## Compatibility

- The numerical simulation and refinement APIs are unchanged from 0.4.0.
- BraggCalculator supports Python 3.12 and 3.13.
- The `refinement` installation extra installs PyTorch.
