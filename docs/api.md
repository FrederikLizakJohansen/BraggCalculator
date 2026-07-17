# API reference

The public API consists of `BraggCalculator`, `ReflectionTable`, and the NumPy
and optional PyTorch backends. Angles supplied as configuration are in degrees,
lengths are in angstroms, and reciprocal quantities are in inverse angstroms.

## `BraggCalculator`

```python
BraggCalculator(
    mode="xray",
    wavelength=1.5406,
    two_theta_range=(10.0, 80.0),
    two_theta_step=0.01,
    q_range=(0.0, 10.0),
    q_step=0.005,
    qmax=None,
    profile=GaussianProfile(),
    profile_q=GaussianProfileQ(),
    backend=NumpyBackend(),
    symprec=1e-3,
    angle_tolerance=5.0,
    primitive=True,
    debye_waller_factors={},
    neutron_scattering_lengths={},
    intensity_tolerance=1e-5,
    phase_chunk_entries=4_194_304,
)
```

`mode` is `"xray"` or `"neutron"`. `wavelength` accepts a positive number or a
pymatgen radiation name such as `"CuKa1"`. The angular and Q ranges configure
the respective outputs. If `qmax` is omitted, the calculator derives the
smallest value that covers both ranges; an explicit value that would truncate
either range is rejected.

`profile` and `profile_q` provide the broadening model for angular and Q-space
profiles. The included Gaussian models use integrated intensities as areas and
are configured by `fwhm_deg` or `fwhm_q`. `backend` is `NumpyBackend()` by
default or a `TorchBackend` instance for autograd and device execution.

```python
from braggcalculator.profiles import GaussianProfile, GaussianProfileQ
```

`symprec`, `angle_tolerance`, and `primitive` control pymatgen/spglib symmetry
preprocessing. `debye_waller_factors` maps element symbols to isotropic B values
in square angstroms. `neutron_scattering_lengths` maps symbols or atomic numbers
to a measured length or a pymatgen isotope key. `intensity_tolerance` is the
relative line-reporting cutoff. `phase_chunk_entries` bounds the size of each
structure-factor phase matrix.

### Structure preparation

```python
calculator.load(structure_like)
```

`structure_like` may be a CIF path, a pymatgen `Structure`, or, when ASE is
installed, an ASE `Atoms`. The method returns the calculator. Symmetry analysis
and reflection enumeration happen here and define a fixed discrete topology.

```python
parameters = calculator.tensor_parameters(requires_grad=False)
```

Returns backend arrays named `lattice`, `frac_coords`, `occupancies`, and
`b_iso`. For a Torch backend, `requires_grad` can be `True` or an iterable of
names. These arrays may be edited or optimized and passed through the
`parameters` argument of the calculation methods. Rebuild the calculator if a
lattice change can add or remove reflections from the configured range.

```python
coordinate_model = calculator.symmetry_coordinate_parameterization()
independent = coordinate_model.initial_values(
    calculator.backend, requires_grad=True
)
parameters = coordinate_model.forward_parameters(calculator, independent)
```

This parameterization derives the allowed local displacement subspace of each
prepared Wyckoff orbit from its site stabilizer. Independent displacements are
expanded through fixed symmetry rotations to every orbit member and then to
every scattering contribution. A special position therefore exposes only its
symmetry-allowed coordinate degrees of freedom. The topology is intentionally
local and fixed: refinements that change space group, Wyckoff multiplicity or
site assignment require rebuilding the calculator and parameterization.

### Calculations

```python
complex_f = calculator.structure_factors(parameters=None, indices=None)
```

Returns the complex structure factor \(F(hkl)\) for every selected reciprocal
point. With the Torch backend, real-valued losses constructed from these
complex tensors remain differentiable with respect to the continuous
structural parameters.

```python
f_squared = calculator.fq(parameters=None, indices=None)
```

Returns one uncorrected $|F|^2$ value for every selected reciprocal point.

```python
position, intensity = calculator.iq(domain="two_theta", parameters=None)
```

Returns individual reciprocal-point positions and Lorentz-polarization-corrected
intensities. `domain` is `"two_theta"` or `"q"`.

```python
position, intensity = calculator.line_pattern(
    domain="two_theta", parameters=None, scaled=False
)
```

Merges coincident reciprocal points into powder lines and removes lines below
the configured reporting tolerance. `scaled=True` normalizes the maximum line
intensity to 100. Prepared metric degeneracies must remain unchanged when
passing a modified lattice; use `pattern()` for differentiable symmetry-breaking
changes.

```python
grid, profile = calculator.pattern(domain="two_theta", parameters=None)
```

Broadens all individual reciprocal-point intensities onto the configured
regular grid. The profile is area-normalized, not maximum-normalized.
`experiment_parameters` may provide differentiable physical `scale`,
`background`, `zero_shift`, and `fwhm` values.

```python
from braggcalculator import ProfileNuisanceParameterization

nuisance_model = ProfileNuisanceParameterization.from_calculator(
    calculator, domain="q", initial_background=1.0
)
raw_nuisance = nuisance_model.initial_values(
    calculator.backend, requires_grad=True
)
physical_nuisance = nuisance_model.physical(raw_nuisance, calculator.backend)
grid, profile = calculator.pattern(
    domain="q", experiment_parameters=physical_nuisance
)
```

Scale, FWHM and background use exponential transforms and remain positive;
zero shift uses a declared characteristic step. The raw values are separate
scalar leaves so an optimization schedule can release them in stages.

```python
from braggcalculator import OptimizationStage, staged_adam

trace = staged_adam(
    objective,
    raw_nuisance,
    [
        OptimizationStage("scale/background", ("scale", "background"), 100, 0.03),
        OptimizationStage("position/width", ("zero_shift", "fwhm"), 100, 0.02),
        OptimizationStage("joint", tuple(raw_nuisance), 100, 0.01),
    ],
)
```

Only the named Torch leaf tensors are passed to each stage's optimizer. This
makes the release policy explicit and records the stage associated with every
loss value.

For guarded refinement, `staged_optimize` additionally supports
`optimizer="lbfgs"`, a per-stage `width_multiplier`, a stage-preparation
callback and a held-out validation objective. Its result contains a
`StageOutcome` for every stage and an explicit convergence classification.
`damped_gauss_newton` is the residual-vector local solver; it reports damping,
trust radius, gain ratio and accepted steps. `recommend_parameter_groups`
returns machine-readable release decisions from sensitivity, residual support
and cross-group correlation evidence.

```python
table = calculator.reflection_table(domain="two_theta", parameters=None)
```

Returns a `ReflectionTable` with `hkl`, `d_spacing`, `q`, `two_theta`, complex
`structure_factor`, `f_squared`, and corrected `intensity` columns. HKLs are a NumPy integer array;
numerical columns use the configured backend.

## Lattice-compatible diagnostics

```python
from braggcalculator.diagnostics import compare_calculators

result = compare_calculators(calculator_a, calculator_b, optimize_origin=True)
```

The calculators must currently use the same lattice representation, radiation
mode and wavelength. The result contains exact matched HKLs, the fitted
relative-origin correction, disk coordinates, the per-reflection radius,
amplitude and phase dissimilarities, and thresholds governing weak-reflection
phase interpretation. Automatic equivalent-setting and supercell mappings are
deliberately deferred.

```python
from braggcalculator.diagnostics import compare_profile_counts

result = compare_profile_counts(
    calculator_a,
    calculator_b,
    count_scale=100.0,
    background_density=10.0,
)
```

This converts the area-normalized calculated profile density into expected
measured-bin counts before computing the expected separation. Consequently,
refining the plotting grid does not spuriously multiply the available
information. For externally supplied expected bin values,
`profile_discrimination` accepts either independent variances or a full
positive-definite covariance matrix.

## Scaled Jacobian diagnostics

```python
from braggcalculator.sensitivity import analyze_jacobian

diagnostics = analyze_jacobian(
    jacobian,
    residual=observed - calculated,
    weights=1.0 / variance,
    parameter_scales=[0.01, 0.02],
    parameter_names=["oxygen x", "oxygen y"],
)
```

The scales are characteristic physical changes, not optimization bounds. The
reported sensitivity therefore means pattern response per declared meaningful
step and can be compared across parameters with different units. Rank-deficient
normal matrices are explicitly flagged; their generalized inverse is not
presented as an identifiable covariance estimate.

An optional prior precision or standardized prior Jacobian can be included:

```python
diagnostics = analyze_jacobian(
    jacobian,
    covariance=observation_covariance,
    parameter_scales=[0.05, 0.1],
    parameter_names=["occupancy", "Biso"],
    prior_precision=np.diag([0.0, 25.0]),
)
```

`rank` and `covariance_is_identifiable` always describe diffraction data alone.
`posterior_rank` and `posterior_covariance_is_identifiable` include the prior.
The result also exposes scaled null-space vectors, prior precision, posterior
normal matrix and standard errors in both characteristic-step and input-
parameter coordinates.

## Parametric bootstrap

`parametric_bootstrap` reapplies a user-supplied estimator to independent or
correlated Gaussian simulations from a fitted expected profile. Percentile
intervals respect declared bounds, while lower/upper boundary-hit counts make
truncation visible. The result includes all successful estimates, failed-draw
count, seed and noise-model provenance.

## Backends

```python
import torch

from braggcalculator.backends import NumpyBackend, TorchBackend

numpy_backend = NumpyBackend()
torch_backend = TorchBackend(device="cuda", dtype=torch.float64)
```

Double precision is the default for both backends because systematic absences
depend on accurate cancellation of complex amplitudes. `TorchBackend` requires
the `torch` extra and preserves autograd through the continuous calculation.
