<p align="center">
  <img src="https://raw.githubusercontent.com/FrederikLizakJohansen/BraggCalculator/main/assets/braggcalculator-logo.png" width="650" alt="BraggCalculator logo">
</p>

# BraggCalculator

BraggCalculator is a fast, validated powder X-ray and neutron diffraction
engine for ideal periodic crystals. It evaluates reciprocal-space Bragg
diffraction with NumPy or optional PyTorch kernels.

The current scientific scope is monochromatic, kinematic powder diffraction.
It includes neutral-atom X-ray form factors, coherent elemental neutron
scattering lengths, occupancies, isotropic Debye-Waller factors, the standard
powder Lorentz or Lorentz-polarization correction, and area-normalized Gaussian
profiles. It does not model diffuse scattering, finite-particle shape, preferred
orientation, microstrain, absorption, background, anomalous X-ray terms, or
instrumental wavelength distributions.

## Installation

Install the latest release from [PyPI](https://pypi.org/project/braggcalculator/):

```bash
python -m pip install braggcalculator
```

Optional dependencies are available as extras:

```bash
python -m pip install "braggcalculator[torch]"  # Torch/autograd/CUDA backend
python -m pip install "braggcalculator[ase]"    # ASE structure input
python -m pip install "braggcalculator[all]"    # All optional dependencies
```

Python 3.12 and 3.13 are supported. To work from a source checkout, use
`python -m pip install -e ".[all]"`.

The [API reference](https://github.com/FrederikLizakJohansen/BraggCalculator/blob/main/docs/api.md)
documents the public configuration, methods, and result conventions.

## Quick start

```python
from braggcalculator import BraggCalculator

calculator = BraggCalculator(mode="xray", wavelength="CuKa1")
calculator.load("structure.cif")

two_theta, integrated_intensity = calculator.line_pattern(scaled=True)
grid, profile = calculator.pattern()
```

`line_pattern()` returns the conventional merged powder lines. `pattern()`
returns an area-normalized Gaussian profile on a regular grid.
`reflection_table()` provides the corresponding HKLs, d-spacings, Q values,
scattering angles, structure factors, and corrected intensities. The
[API reference](https://github.com/FrederikLizakJohansen/BraggCalculator/blob/main/docs/api.md)
documents lower-level reciprocal-point output and the rules for differentiable
lattice changes.

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
    "structure.cif"
)
parameters = calculator.tensor_parameters(
    requires_grad=["lattice", "frac_coords", "occupancies", "b_iso"]
)
grid, profile = calculator.pattern(parameters=parameters)
loss = profile.square().sum()
loss.backward()
```

Use `TorchBackend(device="cuda")` with a CUDA-enabled PyTorch installation to
run the continuous diffraction and profile kernels on a GPU.

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

BraggCalculator evaluates the same kinematic equations as pymatgen and does
not prune the reciprocal set. In pymatgen 2026.5.4, each pattern rebuilds the
reciprocal points and flattened site arrays, then a Python loop processes one
reflection at a time; the site sum inside that reflection is vectorized.
BraggCalculator reduces the primitive cell and constructs the exact reflection
topology during `load()`. Its numerical kernel processes reflection-by-site
chunks and merges equal-spacing lines with indexed reductions. Repeated calls
reuse the topology, and reducible supercells perform numerical work on the
primitive sites. These two savings are reported separately as cached and
end-to-end timings.

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

Performance is machine- and dependency-version-specific, so benchmark JSON
records the exact environment and all timing samples. The versioned scaling
data, plotting commands, and CPU/CUDA protocol are documented in
[the paper README](https://github.com/FrederikLizakJohansen/BraggCalculator/blob/main/paper/README.md).

## Demonstration

The [NaCl demonstration](https://github.com/FrederikLizakJohansen/BraggCalculator/tree/main/demo)
loads a CIF, verifies the calculated powder lines against pymatgen, and writes
an overlay with a residual panel:

```bash
python -m pip install -e . matplotlib
python demo/compare_with_pymatgen.py
```

The script stops if either implementation departs from the stated numerical
tolerances.

An experimental candidate-guided workflow is also available. It reads XY/XYE
data with uncertainties, refines plausible CIF models under a declared policy,
compares candidates and writes a self-contained HTML diagnostic report. The
current instrument-aware layer supports point-group-constrained lattice modes,
six-line Cu K-alpha emission, natural line widths, Thompson--Cox--Hastings
broadening, a compact low-angle asymmetry term and Bragg--Brentano specimen
displacement. Structural refinement additionally supports symmetry-constrained
occupancies, isotropic or anisotropic displacement parameters, and explicitly
reported composition/bond/angle/minimum-distance restraints. Declared rigid
bodies preserve internal Cartesian geometry while refining pose, and a separate
mixture session combines fixed phase structures with positive profile-area
fractions and trace-phase detectability warnings. See
[`docs/experimental.md`](docs/experimental.md) and the public NIST SRM 660c
example:

```bash
python demo/characterize_nist_lab6.py
```

The rigid-body and physical-mixture visual gate is generated with:

```bash
python demo/refine_rigid_multiphase.py
```

Rank-aware data/posterior information, correlated observation covariance,
bounds-aware bootstrap intervals and repeated-synthetic coverage are
demonstrated with:

```bash
python demo/uncertainty_identifiability.py
```

The complete implementation and validation sequence is tracked in
[`docs/characterization-roadmap.md`](docs/characterization-roadmap.md).

Portable refinement projects and a linked offline workspace are also available:

```bash
bragg-project create my-project \
    --data pattern.xye --model model-a.cif --model model-b.cif \
    --wavelength 1.5406
bragg-project run my-project
bragg-project resume my-project
```

Projects copy and checksum their inputs, freeze the policy, retain parent-linked
run records and export result JSON, profile/parameter CSV, refined CIF, audit
JSON and an interactive HTML workspace. Local REST and MCP transports use the
same versioned operations through `bragg-service` and `bragg-mcp`.

For a complete guided browser workflow, launch the trusted-local application:

```bash
bragg-ui --root bragg-ui-projects
```

Then open `http://127.0.0.1:8766`. The UI can load the bundled two-candidate
teaching project or ingest XY/XYE and CIF uploads, declare structural release,
run/resume refinement, and explain fit, residual, structural, complex
amplitude/phase, peak-overlap, identifiability and experiment-design
diagnostics in separate tabs. The bundled data are synthetic and explicitly
labeled as a tutorial, not independent validation.

The executed, code-first companion is
[`notebooks/complete_characterization_tutorial.ipynb`](notebooks/complete_characterization_tutorial.ipynb).
It follows the same example through input provenance, simulation, structure
comparison, information-loss diagnostics, staged two-model refinement,
identifiability, measurement design and auditable exports, with mathematical
and plain-language explanations beside every figure.

The reproducible diagnostics-paper benchmark is also included:

```bash
python scripts/run_diagnostics_publication.py --verify
```

It freezes exact/near homometric and resolution-limited cases, compares four
reflection-weighting declarations with four powder-profile similarity
baselines, tests representation invariance and regenerates the manuscript
figures and tables under `paper/diagnostics/`. Numerical gates pass; external
crystallographer review remains explicitly pending.

## Data and model references

- P. A. Doyle and P. S. Turner, *Acta Crystallographica A* **24**, 390–397
  (1968), DOI: [10.1107/S0567739468000756](https://doi.org/10.1107/S0567739468000756).
- V. F. Sears, *Neutron News* **3**(3), 26–37 (1992), DOI:
  [10.1080/10448639208218770](https://doi.org/10.1080/10448639208218770).
- The [pymatgen diffraction documentation](https://pymatgen.org/pymatgen.analysis.diffraction.html)
  describes the independent oracle implementation used in validation.

## License

Apache License 2.0. See `LICENSE`.
