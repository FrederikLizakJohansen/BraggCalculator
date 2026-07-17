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

## Occupancy and isotropic displacement diagnostics

`refine_occupancy_adp.py` uses a mixed Sr/Ca perovskite to show two regimes:
a controlled staged recovery where correlated parameter families are released
separately, and a whole-profile joint refinement where the correlation matrix
correctly warns that composition and displacement factors are not independently
determined.

```bash
python demo/refine_occupancy_adp.py
```

It writes `demo/occupancy_adp_refinement.png` and
`demo/occupancy_adp_report.html`.

## Anisotropic displacement and structural restraints

`refine_anisotropic_restraints.py` recovers a site-symmetry-compatible
anisotropic displacement tensor and then demonstrates why chemical restraints
must be reported separately from diffraction agreement. Eight strong synthetic
reflections admit a nearly exact but chemically distorted three-atom model;
bond-length, angle and minimum-distance restraints select the plausible local
geometry while preserving the diffraction fit.

```bash
python demo/refine_anisotropic_restraints.py
```

It writes `demo/anisotropic_restraint_refinement.png` and a self-contained
`demo/anisotropic_restraint_report.html`.

## Rigid-body and physical phase-mixture diagnostics

`refine_rigid_multiphase.py` recovers a known three-dimensional translation
and rotation of a declared SiO2 group while demonstrating that all internal
distances remain invariant. It then refines a synthetic 72/28 NaCl/CsCl
physical mixture with an exact positive simplex and evaluates a 0.03% trace
phase whose signal falls below the declared approximate detectability threshold.

```bash
python demo/refine_rigid_multiphase.py
```

It writes `demo/rigid_multiphase_refinement.png` and the self-contained
`demo/rigid_multiphase_report.html`. Fractions in this example are integrated
profile-area fractions, not quantitative weight fractions.

## Calibrated uncertainty and identifiability

`uncertainty_identifiability.py` separates diffraction-data rank from rank
supplied by a prior, visualizes a data-null occupancy/Biso combination, honors
a correlated observation covariance, and tests bounded parametric-bootstrap
intervals across repeated synthetic phase-mixture experiments.

```bash
python demo/uncertainty_identifiability.py
```

It writes `demo/uncertainty_identifiability.png` and the self-contained
`demo/uncertainty_identifiability_report.html`. The report explicitly limits
all intervals to the supplied forward, Gaussian noise, prior and bounds models.

## Robust refinement mechanics

`robust_refinement_mechanics.py` exercises the guarded optimization layer on
small deterministic problems: low-count Poisson deviance, Adam-to-L-BFGS
peak-width continuation, damped Gauss--Newton trust steps, evidence-based
parameter release, validation rollback and seeded multistart classification.

```bash
python demo/robust_refinement_mechanics.py
```

It writes `demo/robust_refinement_mechanics.png` and the self-contained
`demo/robust_refinement_mechanics_report.html`. These are numerical-mechanics
gates; they do not claim that optimizer convergence proves structural truth.

## General structural diagnostics

`general_structural_diagnostics.py` demonstrates the relationship gate and the
diagnostics it enables: the diffraction information-loss ladder, ordered-cell
superstructure reflections, resolution-defined peak/site effects,
counterfactual motif substitutions, periodic scattering-weighted PDFs,
unrelated-polymorph powder comparison and declared-experiment ranking.

```bash
python demo/general_structural_diagnostics.py
```

It writes `demo/general_structural_diagnostics.png` and the self-contained
`demo/general_structural_diagnostics_report.html`. The unrelated example
deliberately reports no complex or `hkl`-phase metric.

## Reference-validation matrix

`reference_validation.py` runs the independent line oracle, verifies and loads
the fixed public X-ray/neutron corpus, reruns the NIST certified-lattice check,
recovers every implemented refinable parameter family, and exercises difficult
overlap, weak-hydrogen, trace-phase and occupancy/Biso-correlation cases.

```bash
python demo/reference_validation.py
```

It writes `demo/reference_validation.png`, a self-contained
`demo/reference_validation_report.html`, and the fully structured
`demo/reference_validation_results.json`. The report deliberately remains
`unsupported` overall while preferred orientation and TOF are absent, and it
keeps direct GSAS-II final-profile reproduction and external expert sign-off as
pending gates.

## Scientist workspace and agent-safe project lifecycle

`scientist_agent_interface.py` builds a portable two-candidate project, runs it,
continues from the first run's exact raw parameter checkpoint, and generates a
linked offline workspace.

```bash
python demo/scientist_agent_interface.py
```

It writes `demo/scientist_agent_interface.png` and the complete
`demo/scientist_workspace_project/` bundle. Open
`runs/run-0002/workspace.html` to click between the structure projection,
profile, resolution-defined peak groups, mismatch disk and optimizer trace.
The right-hand table reports bounds, restraint weights, release state and
provenance; adjacent JSON, CSV, CIF and audit artifacts are directly linked.

The example is intentionally non-discriminating: the lower-Rwp model is not
presented as structurally proven. Continuation restores raw parameter values but
starts a fresh optimizer, so each run remains a separate trace segment.
