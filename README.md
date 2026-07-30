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
profiles. Optional experimental-effect models can add angle-dependent
instrument and sample broadening, preferred orientation, background, counting
noise, missing channels, and spurious peaks to simulated profiles. These
models do not cover diffuse scattering, absorption, anomalous X-ray terms, or
instrumental wavelength distributions.

The accompanying paper describes the implementation, validation across the
Crystallography Open Database, and CPU/GPU performance:
[read the ChemRxiv preprint](https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15006388/v1).

## Installation

Install the latest release from [PyPI](https://pypi.org/project/braggcalculator/):

```bash
python -m pip install braggcalculator
```

Optional dependencies are available as extras:

```bash
python -m pip install "braggcalculator[torch]"  # Torch/autograd/CUDA backend
python -m pip install "braggcalculator[refinement]"  # PXRD refinement workflow
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

## Refinement

Version 0.4.0 adds generated-CIF refinement, structured fit results, and
asymmetric-unit species-assignment screening. See the
[0.4.0 release notes](RELEASE_NOTES.md) for the complete review summary.

Install the Torch refinement extra:

```bash
python -m pip install "braggcalculator[refinement]"
```

Refine an observed XYE pattern and a generated CIF through one entry point:

```python
from braggcalculator import RefinementPolicy, refine_generated_cif

result = refine_generated_cif(
    pattern="observed.xye",
    cif="generated-candidate.cif",
    wavelength=1.5406,
    radiation="xray",
    policy=RefinementPolicy.cautious(refine_coordinates=True),
)

print(result.fit_statistics["r_wp"])
result.write_cif("refined.cif")
```

The result includes the refined structure, calculated profile, residual,
objective history, convergence record, parameter values and bounds,
identifiability diagnostics, warnings, and provenance.

A discrete species-assignment stage can screen element swaps across independent
crystallographic sites:

```python
from braggcalculator import SpeciesAssignmentConfig

assignment = SpeciesAssignmentConfig(
    search="auto",
    fixed_sites=(2,),
    max_candidates=128,
    continuous_top_k=4,
)

result = refine_generated_cif(
    pattern="observed.xye",
    cif="generated-candidate.cif",
    wavelength=1.5406,
    policy=RefinementPolicy.cautious(refine_coordinates=True),
    species_assignment=assignment,
)
```

The search keeps composition fixed by default, checks Wyckoff multiplicities,
and marks assignments with experimentally indistinguishable refined scores.
See the [refinement guide](docs/refinement.md) for parameter selection, site
rules, result interpretation, and performance guidance.

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

Experimental effects are opt-in and independently configurable:

```python
from braggcalculator import (
    BackgroundArtifacts,
    NoiseArtifacts,
    PeakProfileArtifacts,
    SimulationArtifacts,
)

artifacts = SimulationArtifacts(
    profile=PeakProfileArtifacts(
        model="tch",
        caglioti_u=0.002,
        caglioti_w=0.004,
        crystallite_size_nm=40.0,
        microstrain=0.001,
    ),
    background=BackgroundArtifacts(constant=0.01),
    noise=NoiseArtifacts(poisson_count_scale=10_000),
    seed=7,
)
q_grid, augmented = calculator.pattern(domain="q", artifacts=artifacts)
```

`SimulationArtifacts` also supports calibration shifts, Bragg--Brentano
specimen displacement, March--Dollase preferred orientation, amorphous humps,
measured `.xy`/`.xye` backgrounds, correlated noise, detector gaps,
saturation, quantization, and unindexed peaks. See the
[artifact API](https://github.com/FrederikLizakJohansen/BraggCalculator/blob/main/docs/api.md#synthetic-simulation-artifacts)
for the component objects, units, sampled-range controls, and background
library format, and the
[artifact gallery](https://github.com/FrederikLizakJohansen/BraggCalculator/blob/main/paper/figures/artifact_gallery.png)
for isolated examples of every effect family.

For ML workloads, ideal powder lines can be cached once and augmented as a
complete Torch batch without rebuilding structures:

```python
from braggcalculator import (
    PeakProfileArtifacts,
    SimulationArtifacts,
    apply_peak_artifact_batch,
    render_artifact_batch,
)

artifacts = SimulationArtifacts(
    profile=PeakProfileArtifacts(
        model="pseudo_voigt", fwhm=(0.03, 0.08), eta=(0.2, 0.8)
    )
)

# q_lines, intensities and peak_mask have shape [batch, padded_peaks].
q_augmented, intensity_augmented, peak_mask = apply_peak_artifact_batch(
    q_lines, intensities, peak_mask=peak_mask, artifacts=artifacts
)

# Full profiles and all dense artifacts require a dense or hybrid model input.
patterns = render_artifact_batch(
    q_lines,
    intensities,
    peak_mask=peak_mask,
    grid=q_grid,
    artifacts=artifacts,
)
```

The peak-only function applies calibration and intensity effects. The dense
renderer additionally applies profiles, backgrounds, noise, detector effects,
and spurious peaks entirely with device-local Torch operations. See
[batched artifact simulation](https://github.com/FrederikLizakJohansen/BraggCalculator/blob/main/docs/api.md#batched-torch-artifact-simulation)
for tensor shapes, metadata requirements, random generators, and measured
background handling.

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

The frozen publication corpus extends this check to 70 CC0 CIFs from the
Crystallography Open Database, balanced across all seven crystal systems and
covering 62 declared space groups:

```bash
python scripts/validate_cif_corpus.py \
    --output paper/data/cif_validation_results.json
```

The manifest pins every COD revision and SHA-256 digest, and the result records
all parser warnings and failures rather than dropping difficult cases.

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

The executable
[artifact-simulation notebook](https://github.com/FrederikLizakJohansen/BraggCalculator/blob/main/notebooks/artifact_simulation.ipynb)
continues from a CIF through an ideal pattern, individual artifact components,
an imported `.xye` background, and a reproducible combined simulation:

```bash
jupyter notebook notebooks/artifact_simulation.ipynb
```

## Related software

BraggCalculator and
[DebyeCalculator](https://github.com/FrederikLizakJohansen/DebyeCalculator)
provide complementary scattering models: BraggCalculator uses reciprocal-space
translational symmetry for periodic crystals, whereas DebyeCalculator evaluates
the real-space Debye scattering equation for finite, disordered, or
non-crystalline structures.

## Data and model references

- P. A. Doyle and P. S. Turner, *Acta Crystallographica A* **24**, 390–397
  (1968), DOI: [10.1107/S0567739468000756](https://doi.org/10.1107/S0567739468000756).
- V. F. Sears, *Neutron News* **3**(3), 26–37 (1992), DOI:
  [10.1080/10448639208218770](https://doi.org/10.1080/10448639208218770).
- The [pymatgen diffraction documentation](https://pymatgen.org/pymatgen.analysis.diffraction.html)
  describes the independent oracle implementation used in validation.

## License

Apache License 2.0. See `LICENSE`.
