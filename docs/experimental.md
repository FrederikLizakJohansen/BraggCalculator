# Experimental characterization workflow

BraggCalculator's experimental layer is deliberately scoped to candidate-guided,
single-phase powder characterization. It is not a general structure-solution or
certification-grade Rietveld package.

## Python workflow

```python
from braggcalculator import DiffractionDataset, RefinementPolicy, RefinementSession

dataset = DiffractionDataset.from_xye(
    "sample.xye",
    wavelength=1.5405929,
    third_column="sigma",
)
dataset = dataset.exclude([(27.8, 28.1)])

session = RefinementSession(
    dataset,
    ["candidate-a.cif", "candidate-b.cif"],
    names=["candidate A", "candidate B"],
)
result = session.run(RefinementPolicy.cautious())
session.write_html(result, "report.html")
```

The session reports observed/calculated/residual profiles, Rwp, weighted
chi-squared, held-out Rwp, large standardized-residual regions, candidate
ranking, pairwise expected discrimination, restart sensitivity, provenance and
a recommended next action.

## CLI

```bash
bragg-diagnose sample.xye \
  --model candidate-a.cif \
  --model candidate-b.cif \
  --wavelength 1.5405929 \
  --output report.html
```

Add `--weight-column` when the third input column is a least-squares weight
rather than a standard deviation. Add `--coordinates` only for scientifically
plausible starting models and inspect the resulting warning.

## Current experimental model

- X-ray or neutron kinematic intensities from the existing forward engine;
- optional wavelength components and normalized component weights;
- area-normalized pseudo-Voigt peaks;
- positive Caglioti U, V and W terms;
- positive scale and a polynomial background;
- zero shift and uniform lattice scaling;
- optional symmetry-compatible coordinate displacements;
- weighted least squares, held-out bins, restraints and declared Adam stages.

## Important limitations

- The polynomial background is not a physical scattering model.
- Axial divergence, transparency, absorption, preferred orientation,
  crystallite size and microstrain are not implemented.
- Uniform lattice scaling is currently the only session-level lattice model.
- Covariance tools exist, but session-level experimental uncertainties are not
  yet calibrated and must not be reported as certification uncertainties.
- Rwp ranking is accompanied by discrimination and robustness diagnostics, but
  is not evidence that the winning structure is correct.

The NIST SRM 660c demonstration intentionally emits an incomplete-model warning.
It verifies ingestion, wavelength components, refinement, provenance and
reporting against real measurements; it does not reproduce NIST's fundamental
parameters analysis.
