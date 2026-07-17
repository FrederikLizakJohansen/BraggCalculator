# NaCl demonstration

`compare_with_pymatgen.py` loads `NaCl.cif`, calculates its normalized Cu
Kα₁ powder lines with BraggCalculator and pymatgen, verifies their numerical
agreement, and plots the overlaid patterns with a residual panel.

From the repository root, install the package and plotting dependency, then run:

```bash
python -m pip install -e . matplotlib
python demo/compare_with_pymatgen.py
```

The default output is `demo/nacl_vs_pymatgen.png`. Use `--show` to open the
figure interactively or `--output path/to/figure.pdf` to write a vector PDF.

## Compatible-model mismatch disk

`diagnose_compatible_models.py` constructs two same-cell models. The second
uses a different origin, reversed atom order, and one genuine oxygen-coordinate
perturbation. It recovers the arbitrary origin shift and plots the remaining
bounded amplitude-phase mismatch:

```bash
python demo/diagnose_compatible_models.py
```

The script prints the unaligned and aligned dissimilarities, their amplitude
and phase components, the recovered origin correction, the disk-identity
error, and the five most mismatched reflections. Its default figure is
`demo/mismatch_disk.png`.

## Profile discrimination and parameter information

`analyze_profile_information.py` moves one oxygen coordinate, simulates a
bin-level counting experiment, calculates where the two profiles are
distinguishable, and asks which declared parameter direction is supported by
that difference:

```bash
python demo/analyze_profile_information.py
```

The default figure `demo/profile_information.png` connects the two calculated
profiles, standardized bin residuals, local discriminating information, and
scaled Jacobian information for the candidate parameters.

## Symmetry-constrained coordinate refinement

`refine_symmetry_coordinates.py` constructs a centrosymmetric P-1 model,
creates synthetic data after changing its three independent general-position
coordinates, and refines only those three variables. The inversion mate is
generated automatically and cannot move independently:

```bash
python demo/refine_symmetry_coordinates.py
```

The default `demo/symmetry_refinement.png` shows convergence and recovery of
the known independent displacements.

## Joint staged refinement

`refine_staged.py` combines symmetry-compatible coordinates with positive
scale, background and peak width plus a free zero shift. It creates an exact
synthetic target, deliberately starts every group at the wrong value, and
releases groups in a declared sequence before a final joint stage:

```bash
python demo/refine_staged.py
```

The default `demo/staged_refinement.png` shows the stage-wise loss and final
target/recovered parameter ratios.

## Symmetry-aware lattice refinement

`refine_symmetry_lattice.py` generates a tetragonal target and recovers it
through exactly two point-group-invariant metric modes:

```bash
python demo/refine_symmetry_lattice.py
```

## Instrument-aware real-data refinement

`characterize_nist_lab6.py` refines the complete public NIST SRM 660c scan
using the six-line Cu spectrum, the reported Bragg--Brentano geometry and the
symmetry-aware cubic cell:

```bash
python demo/characterize_nist_lab6.py
```

It writes `demo/nist_lab6_report.html` and prints both the historical
limited-scan baseline and the current full-scan result. The example is an
instrument-aware validation, not a reproduction of NIST's certification fit.
