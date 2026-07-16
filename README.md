# BraggCalculator

BraggCalculator is a fast, validated powder X-ray and neutron diffraction
engine for ideal periodic crystals. It uses a crystallographic reciprocal-cell
calculation rather than the pairwise Debye equation and provides NumPy and
optional PyTorch kernels.

The current scientific scope is monochromatic, kinematic powder diffraction.
It includes neutral-atom X-ray form factors, coherent elemental neutron
scattering lengths, occupancies, isotropic Debye-Waller factors, the standard
unpolarized powder Lorentz-polarization correction, and area-normalized
Gaussian profiles. It does not model diffuse scattering, finite-particle shape,
preferred orientation, microstrain, absorption, background, anomalous X-ray
terms, or instrumental wavelength distributions.

## Installation

```bash
python -m pip install .
python -m pip install ".[torch]"  # optional Torch/autograd backend
python -m pip install ".[ase]"    # optional ASE input
```

Python 3.12 and 3.13 are supported.

## Quick start

```python
from braggcalculator import BraggCalculator

calculator = BraggCalculator(mode="xray", wavelength="CuKa1")
calculator.load("examples/NaCl.cif")

two_theta, integrated_intensity = calculator.line_pattern(scaled=True)
grid, profile = calculator.pattern()
```

`iq()` returns one line per reciprocal-lattice point. `line_pattern()` merges
coincident powder lines and applies the conventional relative reporting
threshold. `pattern()` broadens the individual reciprocal-point intensities so
no intensity is lost through early grouping. When differentiating a lattice
change that breaks its prepared metric symmetry, use `iq()` or `pattern()`;
`line_pattern()` assumes the original coincident groups remain degenerate.
`reflection_table()` exposes each point's integer HKL, d-spacing, Q, 2-theta,
`|F|^2`, and corrected intensity without relying on private calculator state.

The Q-space API uses inverse angstroms:

```python
q, intensity = calculator.line_pattern(domain="q")
q_grid, profile_q = calculator.pattern(domain="q")
```

## Torch and autograd

Symmetry detection and HKL enumeration are discrete preprocessing operations.
Autograd therefore operates on a fixed reflection topology. Rebuild the
calculator if a lattice change is large enough that reflections can enter or
leave the configured Q range.

```python
from braggcalculator import BraggCalculator
from braggcalculator.backends import TorchBackend

calculator = BraggCalculator(backend=TorchBackend(device="cpu")).load(
    "examples/NaCl.cif"
)
parameters = calculator.tensor_parameters(
    requires_grad=["lattice", "frac_coords", "occupancies", "b_iso"]
)
grid, profile = calculator.pattern(parameters=parameters)
loss = profile.square().sum()
loss.backward()
```

Species identities and reflection indices are intentionally not differentiable.
Isotope-specific neutron samples can select a tabulated isotope through
`neutron_scattering_lengths={"H": "2H"}` (or supply a measured/custom length)
because pymatgen structures do not retain isotope identity.

By default, `qmax` is derived from the requested 2-theta and Q ranges and the
physical Ewald limit. An explicit `qmax` that would truncate either output range
is rejected instead of silently dropping reflections.

## Scientific conventions

- Direct lattice vectors are rows in angstroms.
- `g = 1 / d`, `Q = 2 pi / d`, and `s = sin(theta) / wavelength = g / 2`.
- Isotropic displacement amplitudes use `exp(-B s^2)`.
- Line intensities are `|F|^2` times the powder Lorentz-polarization factor.
- Gaussian profile amplitudes are integrated areas, not peak heights.
- Every reciprocal point is evaluated explicitly. This makes systematic
  absences emerge from the full structure factor and avoids multiplicity
  double-counting.

X-ray coefficients, radiation wavelengths, and coherent elemental neutron
lengths are read from the required pymatgen dependency. This keeps the source
of physical values explicit and versioned rather than duplicating an
unmaintained local table.

## Validation and performance

Run the unit and analytical test suite:

```bash
python -m pytest -q
```

Validate X-ray and neutron peak positions and normalized intensities against
pymatgen across cubic, diamond, perovskite, triclinic, disordered, and 40-atom
P1 cells:

```bash
python scripts/validate_against_pymatgen.py
```

Run the reproducible performance comparison. The command fails if either the
cached or end-to-end calculation is not faster for every case:

```bash
python benchmarks/benchmark_against_pymatgen.py \
    --number 20 --repeat 7 --require-speedup 1 --json benchmark.json
```

On the development environment (Python 3.13, NumPy 2.4.2, pymatgen 2026.5.4),
all reference cases match to floating-point precision. On the original five
small-cell cases, cached line calculations are 23–41 times faster and
end-to-end calculations are 6–10 times faster than pymatgen; the larger P1 case
is 21 times faster cached and 14 times faster end-to-end. Performance is
machine- and dependency-version-specific, so the script records exact environment
metadata with each JSON result.

## Data and model references

- P. A. Doyle and P. S. Turner, *Acta Crystallographica A* **24**, 390–397
  (1968), DOI: [10.1107/S0567739468000756](https://doi.org/10.1107/S0567739468000756).
- V. F. Sears, *Neutron News* **3**(3), 26–37 (1992), DOI:
  [10.1080/10448639208218770](https://doi.org/10.1080/10448639208218770).
- The [pymatgen diffraction documentation](https://pymatgen.org/pymatgen.analysis.diffraction.html)
  describes the independent oracle implementation used in validation.

## License

Apache License 2.0. See `LICENSE`.
