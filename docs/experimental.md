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

The session lattice is parameterized by point-group-invariant Cartesian
log-strain modes. This gives one metric degree of freedom for cubic structures,
two for tetragonal/hexagonal/trigonal structures, three for orthorhombic, four
for monoclinic and six for triclinic structures. Reports include the physical
`a`, `b`, `c`, `alpha`, `beta` and `gamma` values as well as the internal mode
coordinates.

### Occupancy and isotropic displacement refinement

Occupancy refinement is opt-in and has two explicit meanings:

- `occupancy_mode="composition"` redistributes species on each shared site
  with a softmax while preserving that site's initial total occupancy;
- `occupancy_mode="vacancy"` adds vacancy as another simplex component, so the
  total site occupancy may change but cannot exceed one.

All symmetry-equivalent sites in a crystallographic orbit share the same
composition. Isotropic displacement refinement is also orbit-shared and uses a
softplus transform, so every refined `B_iso` remains positive. If a CIF has a
zero or absent displacement value, `default_b_iso` supplies the positive
starting value (0.5 square angstrom by default) and is recorded in provenance.

```python
policy = RefinementPolicy.cautious(
    occupancy_mode="composition",
    refine_b_iso=True,
)
result = session.run(policy)
```

The defaults restrain both families toward their starting values. Setting
`occupancy_restraint=0` or `b_iso_restraint=0` removes that protection and
should normally be reserved for controlled tests or independently informative
data. The report warns when local occupancy and displacement directions are
strongly correlated. A low Rwp does not override that warning.

## CLI

```bash
bragg-diagnose sample.xye \
  --model candidate-a.cif \
  --model candidate-b.cif \
  --wavelength 1.5405929 \
  --output report.html
```

For a conventional Cu-source Bragg--Brentano measurement with known geometry:

```bash
bragg-diagnose sample.xye \
  --model candidate.cif \
  --wavelength 1.5405925 \
  --copper-ka-spectrum \
  --goniometer-radius-mm 217.5 \
  --specimen-displacement-mm -0.07877 \
  --output report.html
```

Add `--weight-column` when the third input column is a least-squares weight
rather than a standard deviation. Add `--coordinates` only for scientifically
plausible starting models and inspect the resulting warning.

To release a fixed-composition shared site and positive isotropic displacement
parameters:

```bash
bragg-diagnose sample.xye \
  --model candidate.cif \
  --wavelength 1.5405929 \
  --occupancy-mode composition \
  --b-iso \
  --output report.html
```

Use `--occupancy-mode vacancy` only when vacancy is scientifically plausible.
The `--occupancy-restraint` and `--b-iso-restraint` options control the raw-
parameter quadratic restraints.

## Current experimental model

- X-ray or neutron kinematic intensities from the existing forward engine;
- optional wavelength components, natural line widths and normalized weights;
- the reusable NIST six-line Cu K-alpha spectrum;
- an area-normalized Thompson--Cox--Hastings split pseudo-Voigt;
- separate Gaussian U/V/W and Lorentzian X/Y broadening terms;
- an empirical cotangent-scaled low-angle tail;
- Bragg--Brentano specimen-displacement shifts with declared goniometer radius;
- positive scale and a polynomial background;
- zero shift and symmetry-aware lattice strain;
- optional symmetry-compatible coordinate displacements;
- optional symmetry-constrained composition or vacancy occupancies;
- optional positive, orbit-shared isotropic displacement parameters;
- weighted least squares, held-out bins, restraints and declared Adam stages.

## Important limitations

- The polynomial background is not a physical scattering model.
- The current axial-divergence term is a compact empirical split-width model,
  not the full Finger--Cox--Jephcoat or fundamental-parameters convolution.
- Transparency, absorption, preferred orientation and explicit crystallite
  size/microstrain models are not implemented.
- Covariance tools exist, but session-level experimental uncertainties are not
  yet calibrated and must not be reported as certification uncertainties.
- Occupancy and displacement parameters often attenuate the same reflections;
  they must not be interpreted independently when the reported Jacobian
  correlation is large.
- Rwp ranking is accompanied by discrimination and robustness diagnostics, but
  is not evidence that the winning structure is correct.

The NIST SRM 660c demonstration uses the complete public scan, its reported
Bragg--Brentano radius and specimen displacement, the six-line Cu spectrum and
the symmetry-aware cubic cell. It intentionally emits an incomplete-model
warning: the result is substantially closer to the certified cell than the
historical prototype, but it does not reproduce NIST's graphite-analyzer
passband or full fundamental-parameters analysis.
