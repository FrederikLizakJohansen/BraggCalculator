# API reference

The public API consists of `BraggCalculator`, `ReflectionTable`,
`SimulationArtifacts`, and the NumPy and optional PyTorch backends. Angles
supplied as configuration are in degrees, lengths are in angstroms, and
reciprocal quantities are in inverse angstroms.

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
grid, profile = calculator.pattern(
    domain="two_theta", parameters=None, artifacts=None
)
```

Broadens all individual reciprocal-point intensities onto the configured
regular grid. The default profile is area-normalized. The method accepts
differentiable refinement controls and controlled synthetic artifacts.
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

Pass a `SimulationArtifacts` instance to perturb peak positions and
intensities, override the broadening with a pseudo-Voigt profile, and add
background, impurity peaks, noise, or missing observations while rendering.

## Synthetic simulation artifacts

Artifacts are opt-in. A calculation without `artifacts=` is unchanged. Each
effect family has its own immutable configuration object:

```python
from braggcalculator import (
    BackgroundArtifacts,
    BraggCalculator,
    CalibrationArtifacts,
    DetectorArtifacts,
    IntensityArtifacts,
    NoiseArtifacts,
    PeakProfileArtifacts,
    PreferredOrientation,
    SimulationArtifacts,
    SpuriousPeakArtifacts,
)

calculator = BraggCalculator(q_range=(0.0, 10.0), q_step=0.01).load(
    "structure.cif"
)
artifacts = SimulationArtifacts(
    calibration=CalibrationArtifacts(zero_shift=(-0.005, 0.005)),
    profile=PeakProfileArtifacts(
        model="tch",
        caglioti_u=0.002,
        caglioti_v=0.0,
        caglioti_w=0.004,
        lorentzian_x=0.002,
        lorentzian_y=0.003,
        crystallite_size_nm=(30.0, 80.0),
        microstrain=(0.0002, 0.001),
    ),
    intensity=IntensityArtifacts(
        peak_jitter=(0.98, 1.02),
        preferred_orientation=PreferredOrientation(
            axis=(0, 0, 1), ratio=(0.8, 1.2), fraction=0.7
        ),
    ),
    background=BackgroundArtifacts(constant=(0.0, 0.01)),
    noise=NoiseArtifacts(
        gaussian_std=(0.0, 0.002), poisson_count_scale=10_000
    ),
    detector=DetectorArtifacts(excluded_ranges=((4.2, 4.3),)),
    spurious_peaks=SpuriousPeakArtifacts(
        count=(0, 2), intensity=(0.001, 0.01), fwhm=(0.03, 0.08)
    ),
    normalize_signal=True,
    final_normalize=True,
    domain="q",
    seed=7,
)
q, simulated = calculator.pattern(domain="q", artifacts=artifacts)
```

A scalar fixes a parameter. A `(minimum, maximum)` tuple samples uniformly once
per call; reflection-wise jitter and random masks sample once per affected
point. `seed` controls all random choices. A fixed seed repeats the same
realization across calls and numerical backends; `seed=None` produces a new
realization each time.

### Batched Torch artifact simulation

Install the optional Torch dependency for device-native batch augmentation:

```bash
python -m pip install "braggcalculator[torch]"
```

The batch API operates on cached powder lines rather than loaded calculator
objects. This separates one-time crystallographic work from stochastic
augmentation:

```text
CIF → BraggCalculator line calculation → cached padded powder lines
                                            ↓
                               batched Torch artifacts
                                            ↓
                              peak batch or dense profiles
```

Use `apply_peak_artifact_batch()` when a model consumes discrete peak
positions and intensities:

```python
from braggcalculator import (
    CalibrationArtifacts,
    IntensityArtifacts,
    SimulationArtifacts,
    apply_peak_artifact_batch,
)

artifacts = SimulationArtifacts(
    calibration=CalibrationArtifacts(
        zero_shift=(-0.01, 0.01),
        axis_scale=(0.995, 1.005),
        peak_jitter_std=(0.0, 0.003),
    ),
    intensity=IntensityArtifacts(
        scale=(0.8, 1.2),
        peak_jitter=(0.9, 1.1),
        peak_dropout_probability=0.05,
    ),
)

q_augmented, intensity_augmented, augmented_mask = apply_peak_artifact_batch(
    q_lines,                    # [batch, padded_peaks]
    intensities,                # [batch, padded_peaks]
    peak_mask=peak_mask,        # boolean [batch, padded_peaks]
    artifacts=artifacts,
    domain="q",
    generator=device_generator,
)
```

This function intentionally stops after position and intensity effects. A
model that consumes only peak lists cannot observe a continuous background,
rendered profile shape, channel noise, detector gaps, or spurious rendered
peaks.

Use `render_artifact_batch()` for dense or hybrid model inputs:

```python
from braggcalculator import (
    BackgroundArtifacts,
    NoiseArtifacts,
    PeakProfileArtifacts,
    SimulationArtifacts,
    render_artifact_batch,
)

artifacts = SimulationArtifacts(
    profile=PeakProfileArtifacts(
        model="tch",
        caglioti_u=(0.001, 0.004),
        caglioti_w=(0.002, 0.006),
        crystallite_size_nm=(20.0, 100.0),
        microstrain=(0.0, 0.001),
    ),
    background=BackgroundArtifacts(constant=(0.0, 0.03)),
    noise=NoiseArtifacts(
        gaussian_std=(0.0, 0.003),
        correlated_std=(0.0, 0.003),
        correlation_length=(0.02, 0.1),
        poisson_count_scale=(5_000, 20_000),
    ),
    normalize_signal=True,
    final_normalize=True,
    domain="q",
)

patterns = render_artifact_batch(
    q_lines,
    intensities,
    peak_mask=peak_mask,
    grid=q_grid,                 # [grid] or [batch, grid]
    artifacts=artifacts,
    wavelength=1.5406,
    measured_background=background_on_q_grid,
    generator=device_generator,
)
# patterns has shape [batch, grid]
```

The input contract is:

| Argument | Shape | Required for |
| --- | --- | --- |
| `positions`, `intensities` | `[batch, padded_peaks]` | All calls |
| `peak_mask` | Boolean `[batch, padded_peaks]` | Ragged reflection lists; omitted means all valid |
| `grid` | `[grid]` or `[batch, grid]` | Dense rendering |
| `hkl` | `[batch, padded_peaks, 3]` | Preferred orientation |
| `lattice` | `[batch, 3, 3]` | Preferred orientation |
| `wavelength` | Scalar or `[batch]` | Q-domain TCH and coherent-domain-size broadening |
| `measured_background` | `[grid]` or `[batch, grid]` | Pre-interpolated measured background |

Every tensor must share the position tensor's device; floating-point inputs
must also share its dtype. The renderer preserves both, processes all samples
without a Python batch loop, and chunks the reflection dimension according to
`max_entries`. A pre-interpolated `measured_background` is recommended in a
training loop because it avoids repeatedly transferring and interpolating a
`BackgroundPattern`.

`profile.model="calculator"` cannot inspect a calculator in this stateless
API. Supply `profile_fwhm` and optional `profile_eta` as scalars, `[batch]`, or
`[batch, padded_peaks]`, or select the explicit `"pseudo_voigt"` or `"tch"`
model.

For repeatable calls, either set `SimulationArtifacts.seed` or pass a
device-local `torch.Generator`, but not both. A generator is the fast option
for a training stream; its realization depends on batch order and shape.
Independent per-sample generator loops are deliberately not hidden inside the
API because they would serialize the GPU workload. Poisson sampling,
quantization, masking, and dropout are discrete; the remaining continuous
Torch operations retain autograd.

Run the standalone throughput benchmark with, for example:

```bash
python benchmarks/benchmark_batched_artifacts.py \
  --device cuda --batch-size 256 --peaks 512 --grid-points 2048

python benchmarks/benchmark_batched_artifacts.py \
  --device cuda --batch-size 256 --peaks 512 --peak-only
```

### Artifact components

`CalibrationArtifacts` provides a domain-native `zero_shift`, an `axis_scale`,
an independent zero-mean per-reflection `peak_jitter_std`, and the conventional
Bragg--Brentano specimen-displacement shift. Position jitter is expressed in
the selected domain's coordinate units. Specimen displacement is only valid in
the two-theta domain and requires the goniometer radius in millimetres.

`PeakProfileArtifacts` has three profile modes:

- `"calculator"` preserves the calculator's configured Gaussian profile.
- `"pseudo_voigt"` uses an area-normalized pseudo-Voigt with a fixed or sampled
  FWHM and mixing value `eta`.
- `"tch"` combines Caglioti Gaussian coefficients `U`, `V`, and `W` with
  Lorentzian `X` and `Y` terms using the Thompson--Cox--Hastings approximation.
  It can additionally include Scherrer coherent-domain-size broadening and
  isotropic microstrain broadening.

Caglioti and Lorentzian coefficients are specified in degrees of two-theta even
when the output domain is Q; the resulting local widths are converted to
inverse angstroms. `crystallite_size_nm` is a coherent diffracting-domain size,
not a particle-size measurement. `axial_asymmetry` is a compact split-width
low-angle-tail approximation, not a full Finger--Cox--Jephcoat convolution.

`IntensityArtifacts` provides overall scale, independent reflection jitter,
reflection dropout, and a modified March--Dollase preferred-orientation
correction. The texture axis is an HKL direction, the ratio must be positive,
and a ratio of one recovers random powder intensities.

`BackgroundArtifacts` combines a constant, a linear term in intensity per
coordinate unit, Chebyshev coefficients on `[-1, 1]`, broad
`AmorphousHump` contributions, and an optional measured background. Hump
amplitudes are peak heights. `SpuriousPeakArtifacts` instead adds unindexed
pseudo-Voigt peaks whose intensities are integrated areas.

`NoiseArtifacts` provides independent Gaussian noise, Gaussian-correlated
noise with a correlation length in coordinate units, and Poisson counting
noise. `poisson_count_scale` is the number of expected counts per output
intensity unit. Poisson sampling is discrete and therefore intentionally ends
the autograd path through the simulated expectation; continuous profile,
background, and additive-noise operations retain it.

`DetectorArtifacts` provides random missing channels, explicit excluded
coordinate ranges, saturation, and quantization. Excluded or randomly masked
channels are returned as zero intensity.

### Measured backgrounds

Load a whitespace- or comma-separated `.xy` or `.xye` file and interpolate it
onto the simulation grid:

```python
from braggcalculator import BackgroundArtifacts, BackgroundPattern

blank = BackgroundPattern.from_file(
    "empty_capillary.xye",
    domain="two_theta",
    third_column="sigma",
    source="beamline blank measurement, DOI or archive record",
)
artifacts = SimulationArtifacts(
    background=BackgroundArtifacts(
        measured=blank,
        measured_scale=0.8,
        measured_offset=0.0,
        extrapolation="error",
    )
)
two_theta, simulated = calculator.pattern(artifacts=artifacts)
```

The first two columns are coordinate and intensity. For an `.xye` file the
third column is interpreted as standard uncertainty by default; use
`third_column="weight"` for inverse-variance weights or `"ignore"` when the
third column should not be retained. Coordinates must be finite, unique, and
strictly increasing. Intensities must be non-negative. The object records the
source label and SHA-256 digest of the input bytes.

`extrapolation="error"` is the default and requires the measurement to cover
the entire output grid. `"zero"` or `"edge"` must be selected explicitly when
zero or constant-end extrapolation is scientifically appropriate. Background
and simulation domains must match; the library does not silently reinterpret a
two-theta trace as Q.

Curated collections use `BackgroundLibrary`:

```python
from braggcalculator import BackgroundLibrary

library = BackgroundLibrary("backgrounds/manifest.json")
print(library.names)
blank = library.load("instrument-a-empty-capillary")
```

The JSON manifest has this form:

```json
{
  "schema_version": 1,
  "backgrounds": {
    "instrument-a-empty-capillary": {
      "path": "instrument-a-empty-capillary.xye",
      "domain": "two_theta",
      "third_column": "sigma",
      "source": "stable archive URL or DOI",
      "sha256": "full lowercase SHA-256 digest"
    }
  }
}
```

Loading rejects missing provenance, path traversal, malformed data, and digest
mismatches. `BackgroundLibrary.bundled()` opens the package manifest. It is
currently empty: measured curves will only be bundled when their redistribution
terms, instrument/sample geometry, uncertainty convention, stable source, and
checksum can all be recorded.

### Model scope and references

These components are controlled simulation models, not a fundamental-parameters
instrument description or a replacement for calibration against a suitable
standard. The profile equations follow the Caglioti angular-width convention
and the area-normalized Thompson--Cox--Hastings pseudo-Voigt approximation
summarized in the
[IUCr peak-profile review](https://journals.iucr.org/j/issues/2021/06/00/gj5272/).
Preferred orientation uses the
[March--Dollase model](https://doi.org/10.1107/S0021889886089458).
The coherent-domain-size term uses the Scherrer relation and therefore inherits
its shape-factor and applicability limitations; see the
[IUCr review of crystallite-size determination](https://journals.iucr.org/j/issues/2024/05/00/oc5037/).
The bundled background-library policy is deliberately stricter than the direct
file loader because a portable reference trace must retain enough geometry and
provenance to be scientifically interpretable.

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

This low-level operation requires the same lattice representation, radiation
mode and wavelength. The result contains exact matched HKLs, the fitted
relative-origin correction, disk coordinates, the per-reflection radius,
amplitude and phase dissimilarities, and thresholds governing weak-reflection
phase interpretation.

For arbitrary periodic structures, use the relationship-aware operation:

```python
from braggcalculator import diagnose_structures

diagnostic = diagnose_structures(
    structure_a,
    structure_b,
    radiation="xray",
    wavelength=1.5406,
    q_range=(0.5, 8.0),
    profile_fwhm_q=0.08,
    site_groups={"framework": [0, 1, 2]},
    counterfactual_groups={"framework": [0, 1, 2]},
)
```

The result begins with an `equivalent`, `lattice_compatible`, `commensurate`
or `unrelated` relationship. Compatible and commensurate pairs receive a
reciprocal-vector-matched complex comparison. Unrelated pairs return
`mismatch=None` and retain only powder-profile and periodic radial-pair
comparisons; the API does not invent a phase correspondence.

`similarities` contains the complex, intensity, ideal-powder, broadened-profile
and radial-pair levels. `dominant_information_loss` identifies phase loss,
powder averaging or peak overlap only when the corresponding similarity jump
exceeds the declared 0.03 gate. Peak-group site-removal and structural
counterfactual effects retain interference and therefore are explicitly
non-additive.

Commensurate integer parent/supercell pairs expose a `superstructure` result.
Its reflections are those whose supercell indices do not transform to integer
parent indices; their calculated intensity fraction is reported.

Candidate experiments can be compared under explicit count assumptions:

```python
from braggcalculator import suggest_measurements

recommendations = suggest_measurements(
    structure_a,
    structure_b,
    [
        {"name": "laboratory", "radiation": "xray", "wavelength": 1.5406,
         "q_range": (0.5, 6.0), "fwhm_q": 0.20, "count_scale": 1000},
        {"name": "high resolution", "radiation": "xray", "wavelength": 1.5406,
         "q_range": (0.5, 6.0), "fwhm_q": 0.04, "count_scale": 1000},
    ],
)
```

The score is expected measured-count separation under a symmetric Poisson
variance approximation. Radiation sources are comparable only when their
declared count scales and backgrounds represent credible exposure conditions.

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

## Reference validation

The validation API keeps numerical evidence separate from capability and review
gates:

```python
from pathlib import Path

from benchmarks.reference_cases import reference_structures
from braggcalculator import (
    ValidationMatrix,
    load_reference_sources,
    validate_line_oracle,
    validate_public_sources,
)

root = Path.cwd()
sources = load_reference_sources(root / "data/reference_validation/manifest.json")
matrix = ValidationMatrix(
    cases=(
        *validate_line_oracle(reference_structures()),
        *validate_public_sources(root, sources),
    ),
    sources=sources,
    required_categories=("line_oracle", "public_data"),
    expert_review_status="pending_review",
)
matrix.write_json("validation.json")
```

Metrics use explicit `maximum`, `minimum` or `informational` directions with
pass and warning limits. A missing required category fails the matrix. Case
statuses include `pass`, `warn`, `fail`, `unsupported` and `pending_review`;
the last two are not converted into success by passing numerical cases.

`DiffractionDataset.from_gsas_constant_step` reads 80-column constant-step GSAS
`STD` and ESD-style banks. It uses the declared point count to remove record
padding, reconstructs centidegree coordinates from the `CONST` header and
records how uncertainties were obtained.

## Portable projects and linked workspaces

```python
from braggcalculator import ProjectStore, RefinementPolicy

project = ProjectStore.create(
    "my-project",
    dataset_path="pattern.xye",
    model_paths=["model-a.cif", "model-b.cif"],
    names=["A", "B"],
    wavelength=1.5406,
    policy=RefinementPolicy.cautious(refine_coordinates=True),
)
document, result = project.run()
continued_document, continued_result = project.run(resume=True)
```

Creation copies and checksums every input. A resumed run restores all raw
parameter groups from the latest result, starts a fresh optimizer and records a
new trace segment with `parent_run_id`. It does not claim to preserve Adam or
L-BFGS internal optimizer state.

Each run exports:

- `result.json` using `braggcalculator.session-result/v1`;
- observed/calculated/residual `profiles.csv`;
- flattened `parameters.csv`;
- one refined CIF per candidate;
- a linked self-contained `workspace.html`;
- project-level `audit.json`.

Project, result, audit and service-response JSON Schemas are distributed under
`braggcalculator/schemas/v1/`.

## Service and MCP operations

`DiagnosticService(root).dispatch(operation, payload)` is the shared local API.
Available operations are project create/run/resume/status/result, stored-run
sensitivity analysis, pattern simulation, relationship-aware model comparison
and measurement ranking.

```bash
bragg-service --root bragg-projects --port 8765
bragg-mcp --root bragg-projects
```

The HTTP transport accepts JSON POST requests at
`/v1/operations/<operation>`. The MCP stdio server publishes structured tools
for the same scientific operations. MCP project creation requires
`release_policy_acknowledged=true` whenever the declared policy releases any
structural family; an agent cannot silently opt into broad structural
refinement.

## Guided local web application

`bragg-ui` serves a dependency-free browser client and a small local JSON API
on top of `ProjectStore`:

```bash
bragg-ui --root bragg-ui-projects --host 127.0.0.1 --port 8766
```

The application supports browser-side XY/XYE and CIF upload, checksummed
project creation, staged run/resume, a bundled synthetic tutorial, linked SVG
diagnostics and project artifact download. Existing completed projects can be
opened by identifier or with `?project=<identifier>`.

The UI endpoints are:

- `POST /api/examples/tutorial`;
- `POST /api/projects`;
- `POST /api/projects/<id>/run` and `/resume`;
- `GET /api/projects/<id>/diagnostics`;
- `GET /api/projects/<id>/artifacts/<project-relative-path>`.

Structural parameter release through upload requires
`release_policy_acknowledged=true`. Artifact and project paths are confined to
the declared root. This is a trusted-local interface without authentication or
TLS; it must not be exposed directly as a multi-user network service.

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
