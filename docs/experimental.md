# Experimental characterization workflow

BraggCalculator's experimental layer is deliberately scoped to candidate-guided
powder characterization and declared fixed-structure phase mixtures. It is not
a general structure-solution or certification-grade Rietveld package.

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

### Observation covariance and uncertainty

Independent `sigma` values remain the default. For correlated observations,
provide a positive-definite covariance whose diagonal agrees with `sigma**2`:

```python
dataset = DiffractionDataset(
    coordinate=two_theta,
    intensity=intensity,
    sigma=np.sqrt(np.diag(covariance)),
    mask=np.ones(len(two_theta), dtype=bool),
    domain="two_theta",
    wavelength=1.5406,
    observation_covariance=covariance,
)
```

The full covariance is used to whiten the refinement residual, local Jacobian
and model-discrimination norm. Its SHA-256 checksum is written to provenance.
Marginal standardized residual plots are retained for localization because a
whitened correlated residual is a mixture of profile bins.

Session identifiability output now distinguishes `data_rank` from
`posterior_rank`. Quadratic parameter priors and declared structural restraints
contribute to the posterior normal matrix. If they repair a data-null
direction, the report says so explicitly; posterior invertibility is not
reported as diffraction identifiability. Characteristic Jacobian steps include
physical descriptions such as 0.001 log strain, 0.01 mm specimen displacement,
0.1 square-angstrom local Biso change and the declared rigid-body translation
or rotation scale.

For non-linear or bounded parameters, use the bounds-aware parametric
bootstrap utility:

```python
from braggcalculator import parametric_bootstrap

interval = parametric_bootstrap(
    observed,
    fitted_expected,
    estimator,
    covariance=covariance,
    draws=1000,
    bounds=[(0.0, 1.0)],
    parameter_names=["minor profile-area fraction"],
    seed=1729,
)
```

The result records percentile bounds, empirical standard errors, failed draws,
boundary hits, seed and noise model. The estimator must reproduce the intended
refinement policy for every bootstrap replicate. These intervals are
conditional on the supplied forward, noise, prior and bounds models.

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

### Anisotropic displacement tensors

Set `refine_u_aniso=True` to refine Cartesian anisotropic displacement tensors
in square angstrom. A matrix exponential acts within the site-symmetry-
invariant tensor subspace, preserving positive eigenvalues, special-position
restrictions and crystallographic-orbit relationships at every step. Isotropic
and anisotropic displacement parameters cannot be released simultaneously.

```python
policy = RefinementPolicy.cautious(refine_u_aniso=True)
```

The forward attenuation is `exp(-0.5 * G.T @ U_cart @ G)`. CIF anisotropic
`Uij` and `Bij` loops are read and converted to this Cartesian convention.
Missing or zero tensors start from `default_u_iso` (0.006 square angstrom by
default), which is recorded in provenance.

### Structural restraints

Composition, bond-length, angle and minimum-distance restraints are supplied
explicitly. Site indices refer to the prepared structure. Periodic images are
resolved once at the start of refinement and then kept fixed so optimization
remains continuous.

```python
from dataclasses import replace

restraints = {
    "composition": [{"species": "O", "target": 2.0, "sigma": 0.02}],
    "bonds": [{"sites": [0, 1], "target": 1.62, "sigma": 0.02}],
    "angles": [{
        "sites": [0, 1, 2],
        "target_degrees": 109.5,
        "sigma_degrees": 1.5,
    }],
    "minimum_distances": [{
        "sites": [0, 2], "minimum": 2.5, "sigma": 0.05,
    }],
}
policy = RefinementPolicy.cautious(refine_coordinates=True)
policy = replace(policy, structural_restraints=restraints)
```

Every standardized squared contribution and their mean are reported separately
from the diffraction statistics. These terms encode prior chemical
information; they are not additional diffraction observations.

### Rigid-body refinement

Rigid bodies are opt-in, explicitly named, non-overlapping groups of prepared-
structure site indices. Each group has three Cartesian translation modes and a
three-component exponential-map rotation about its Cartesian centroid (or a
declared pivot). Internal Cartesian distances are invariant by construction.

```python
rigid_bodies = [{"name": "tetrahedron", "sites": [0, 1, 2, 3, 4]}]
policy = RefinementPolicy.cautious(rigid_bodies=rigid_bodies)
result = session.run(policy)
```

Free symmetry-coordinate refinement and rigid-body refinement cannot be
released together. A rigid body can intentionally leave the starting space
group; the fixed complete reflection topology still evaluates that model, and
the result records a symmetry-breaking warning. The current implementation
does not automatically discover molecules, coordination polyhedra or symmetry-
related rigid-body copies: the scientist must declare the groups.

### Physical phase mixtures

`PhaseMixtureSession` is distinct from candidate comparison. Candidate
comparison fits alternative models independently; a phase mixture sums their
contributions in one observed profile. A softmax parameterization keeps every
fraction positive and makes the fractions sum exactly to one.

```python
from braggcalculator import PhaseMixturePolicy, PhaseMixtureSession

mixture = PhaseMixtureSession(
    dataset,
    ["major-phase.cif", "minor-phase.cif"],
    names=["major", "minor"],
)
result = mixture.run(PhaseMixturePolicy(initial_fractions=(0.9, 0.1)))
mixture.write_html(result, "mixture-report.html")
```

Each fixed phase profile is normalized to unit integrated area over the fitted
range before mixing. Consequently, the returned values are **profile-area
fractions**, not weight, mole or volume fractions. Quantitative phase analysis
requires a separately validated Rietveld scale convention, phase scattering
powers and sample corrections. The result also reports each component's norm
relative to the supplied uncertainty and warns below an approximate three-
sigma profile-detectability threshold.

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

Use `--u-aniso` for anisotropic tensors. A JSON file containing the dictionary
shown above can be supplied with `--restraints restraints.json`; its global
multiplier is controlled by `--structural-restraint-weight`.

Rigid-body declarations can be supplied as JSON:

```json
[{"name": "tetrahedron", "sites": [0, 1, 2, 3, 4]}]
```

```bash
bragg-diagnose sample.xye --model candidate.cif --wavelength 1.5405929 \
  --rigid-bodies rigid-bodies.json --output rigid-report.html
```

Repeated models become physical phases only when `--mixture` is explicit:

```bash
bragg-diagnose sample.xye \
  --model major.cif --name major \
  --model minor.cif --name minor \
  --mixture --initial-phase-fraction 0.9 --initial-phase-fraction 0.1 \
  --wavelength 1.5405929 --output mixture-report.html
```

The initial mixture implementation keeps the phase structures fixed while
refining their profile-area fractions and shared nuisance/profile parameters.

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
- optional positive-definite, site-symmetry-compatible anisotropic tensors;
- optional explicitly declared Cartesian rigid-body translations and rotations;
- fixed-structure physical mixtures with simplex-constrained profile-area fractions;
- explicit composition, bond, angle and minimum-distance restraints with
  separate contribution reporting;
- independent-sigma or full-covariance Gaussian whitening;
- separate diffraction-data and posterior ranks with null-direction reporting;
- bounds-aware parametric-bootstrap intervals with boundary/failure provenance;
- weighted least squares, held-out bins, restraints and declared Adam stages.

## Important limitations

- The polynomial background is not a physical scattering model.
- The current axial-divergence term is a compact empirical split-width model,
  not the full Finger--Cox--Jephcoat or fundamental-parameters convolution.
- Transparency, absorption, preferred orientation and explicit crystallite
  size/microstrain models are not implemented.
- Local Gaussian covariance is conditional on the supplied model and remains a
  diagnostic approximation. Use repeated-synthetic or bootstrap coverage
  before reporting parameter intervals, and do not label either result as a
  certification uncertainty without an external validation chain.
- Occupancy and displacement parameters often attenuate the same reflections;
  they must not be interpreted independently when the reported Jacobian
  correlation is large.
- Periodic images used by geometry restraints are fixed from the starting
  topology. Large coordinate changes that alter bonding require rebuilding the
  restraint set rather than continuing the same local refinement.
- Rigid-body membership and pivots are fixed from the starting model; rigid
  bodies are not discovered automatically and may intentionally break its
  starting crystallographic symmetry.
- Multi-phase values are profile-area fractions over the fitted range. They
  must not be labeled as quantitative weight fractions, and phase structures
  are not yet refined jointly in a mixture run.
- Rwp ranking is accompanied by discrimination and robustness diagnostics, but
  is not evidence that the winning structure is correct.

The NIST SRM 660c demonstration uses the complete public scan, its reported
Bragg--Brentano radius and specimen displacement, the six-line Cu spectrum and
the symmetry-aware cubic cell. It intentionally emits an incomplete-model
warning: the result is substantially closer to the certified cell than the
historical prototype, but it does not reproduce NIST's graphite-analyzer
passband or full fundamental-parameters analysis.
