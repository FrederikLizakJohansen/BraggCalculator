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

### Calculations

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
regular grid. The default profile is area-normalized, not maximum-normalized.
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

### Artifact components

`CalibrationArtifacts` provides a domain-native `zero_shift`, an `axis_scale`,
and the conventional Bragg--Brentano specimen-displacement shift. The latter is
only valid in the two-theta domain and requires the goniometer radius in
millimetres.

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

Returns a `ReflectionTable` with `hkl`, `d_spacing`, `q`, `two_theta`,
`f_squared`, and corrected `intensity` columns. HKLs are a NumPy integer array;
numerical columns use the configured backend.

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
