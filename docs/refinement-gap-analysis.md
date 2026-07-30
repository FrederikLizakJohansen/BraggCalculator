# Refinement extraction and gap analysis

This review branch starts from BraggCalculator 0.3.0 at `d546e1f`. The
characterization branch at `112c138` supplies the refinement implementation.

## What the existing engine already provides

The current `RefinementSession` accepts one measured powder pattern and one or
more candidate structures. It already provides:

- XY/XYE and constant-step GSAS data loading, uncertainties, masks, and input
  checks;
- X-ray and neutron calculations through the Torch backend;
- scale, polynomial background, zero shift, peak-profile, specimen-displacement,
  and symmetry-constrained lattice refinement;
- symmetry-constrained fractional coordinates, occupancies, isotropic
  displacement, and anisotropic displacement;
- bond-length, bond-angle, minimum-distance, and composition restraints;
- rigid-body pose refinement and profile-area phase mixtures;
- Gaussian weighted least squares and Poisson deviance;
- staged Adam and L-BFGS refinement, profile-width continuation, holdout checks,
  deterministic restarts, and convergence records;
- local sensitivity, covariance, correlation, rank, and informative-region
  diagnostics.

The calculator keeps the reciprocal topology fixed during continuous
refinement. This makes the Torch derivatives stable while parameters move
within the prepared reflection range.

## Gaps to close for 0.4.0

1. The public workflow begins with several low-level objects. A candidate CIF
   user needs one function that loads the pattern and CIF, applies a declared
   policy, and returns one structured result.
2. `CandidateRefinementResult.structure` stores the starting structure. The
   project layer can rebuild the refined structure from a checkpoint, yet that
   capability lives inside broader workspace infrastructure.
3. Parameter bounds and final parameter paths are spread across policy,
   parameterization, and provenance fields. The focused result needs one
   parameter table with values, bounds, units, and release state.
4. Session construction supports two-theta data. Q-domain input needs either
   direct support or a clear validation message in the 0.4.0 API.
5. Continuous refinement has no species-assignment search. Element identities
   require a discrete outer search over asymmetric-unit sites.
6. Exceptions describe invalid input well. The high-level result also needs a
   stable failure status for optimizer outcomes and truncated discrete searches.
7. The broad characterization branch carries UI, MCP, service, project,
   publication, validation-corpus, generated-report, and tutorial systems. The
   focused branch needs a compact demonstration and concise API documentation.

## Module classification

| Extensive-branch module | Classification | Focused-branch decision |
| --- | --- | --- |
| `core.py` | Direct dependency | Transfer its differentiable structural parameter support and retain the 0.3.0 artifact API. |
| `backends/numpy_backend.py` | Direct dependency | Transfer the extra numerical operations used by refinement and retain batched-artifact operations. |
| `backends/torch_backend.py` | Direct dependency | Transfer the extra differentiable operations and retain batched-artifact operations. |
| `dataset.py` | Required refinement core | Transfer. |
| `experiment.py` | Required refinement core | Transfer. |
| `experimental_profile.py` | Required refinement core | Transfer. |
| `optimization.py` | Required refinement core | Transfer. |
| `parameters.py` | Required refinement core | Transfer. |
| `restraints.py` | Required refinement core | Transfer. |
| `sensitivity.py` | Required refinement core | Transfer. |
| `session.py` | Required refinement core | Transfer and place the high-level API above it. |
| `uncertainty.py` | Direct user diagnostic | Transfer. |
| `mixture.py` | Optional focused refinement | Transfer because it shares the same small dependency set. |
| `io.py` | Direct dependency | Transfer CIF displacement parsing. |
| `profiles.py` | Direct dependency | Transfer differentiable width control. |
| `renderer.py` | Direct dependency | Transfer the profile-width call path. |
| `results.py` | Direct dependency | Transfer the Jacobian result type while retaining 0.3.0 reflection results. |
| `structure_factor.py` | Direct dependency | Transfer anisotropic displacement support. |
| `symmetry.py` | Direct dependency | Transfer site mappings and displacement extraction. |
| `radiation.py` | Compact refinement helper | Transfer the documented Cu K-alpha spectrum helper. |
| `diagnostics.py` | Broader characterization | Keep on the extensive branch. |
| `structural_diagnostics.py` | Broader characterization | Keep on the extensive branch. |
| `validation.py` | Validation platform | Keep on the extensive branch. Focused regression tests remain. |
| `cli.py` | Broader command interface | Keep on the extensive branch. |
| `project.py`, `project_cli.py`, `workspace.py` | Project and workspace infrastructure | Keep on the extensive branch. |
| `service.py`, `mcp.py` | Service and agent infrastructure | Keep on the extensive branch. |
| `ui.py`, `ui/index.html` | Browser UI | Keep on the extensive branch. |
| `publication.py` | Publication infrastructure | Keep on the extensive branch. |
| `schemas/v1/*` | Service and project schemas | Keep on the extensive branch. |
| `tutorial_data/*` | UI tutorial artifact | Keep on the extensive branch. |

The focused branch will add `refinement.py` for the structure-refinement entry point
and `species_assignment.py` for the discrete asymmetric-unit search.

## Commit classification

| Commit | Main contribution | Classification |
| --- | --- | --- |
| `f248b9d` | Complex structure-factor mismatch | Broader diagnostics |
| `eca2c6b` | Profile information diagnostics | Direct diagnostic concepts; extract sensitivity dependencies only |
| `12bd29b` | Symmetry-constrained coordinates | Required refinement core |
| `82ed658` | Staged profile refinement | Required refinement core |
| `ad1d981` | Experimental session | Required refinement core |
| `4ebbd30` | Instrument and lattice refinement | Required refinement core |
| `5324f06` | Occupancy and isotropic displacement | Required refinement core |
| `b0ba3f2` | Anisotropic displacement and restraints | Required refinement core |
| `83d983f` | Rigid body and phase mixture | Focused optional refinement |
| `6f531a6` | Uncertainty and identifiability | Direct user diagnostic |
| `1088967` | Robust optimization | Required refinement core |
| `0aa6d8f` | General structural diagnostics | Broader diagnostics |
| `24cc2fd` | Reference validation matrix | Validation platform |
| `e804c83` | Project and agent interfaces | Project, service, and workspace infrastructure |
| `395f6e7` | Guided UI | Browser UI |
| `1b8fa44` | Publication benchmark | Publication output |
| `8de3b69` | Frozen publication revision | Publication output |
| `c6ec7c2` | Complete characterization notebook | Tutorial artifact |
| `2d5a002` | Refinement tutorial notebooks | Tutorial artifact; replace with a compact demonstration |
| `45a38e6` | Main 0.3.0 integration | Integration history retained on the extensive branch |
| `112c138` | Refreshed tutorials and scaling results | Generated tutorial and benchmark artifacts |

## Non-module file classification

- Focused tests will come from the parameter, session, optimization, restraint,
  uncertainty, mixture, Torch-gradient, I/O, and simulation regression tests.
- `data/`, `paper/`, large notebooks, binary figures, generated HTML/JSON/CSV
  reports, and validation corpora stay on the extensive branch.
- The focused branch will contain one generated test structure, one small
  generated pattern example, and text documentation.
- README and API documentation will describe simulation and refinement together.

## Planned public entry point

```python
result = refine_structure(
    pattern="observed.xye",
    cif="candidate.cif",
    wavelength=1.5406,
    radiation="xray",
    policy=RefinementPolicy.cautious(refine_coordinates=True),
)
```

The result will contain the refined structure, refined CIF text, calculated
profile, residual, objective history, convergence record, parameter table,
identifiability summary, warnings, and provenance.

Species screening will use the same entry point through a
`SpeciesAssignmentConfig`. The search will rank composition-preserving
asymmetric-unit assignments, refine the best candidates continuously, and mark
assignments that the measured pattern cannot distinguish.
