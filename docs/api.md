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
