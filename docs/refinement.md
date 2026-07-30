# Refining a generated CIF against PXRD data

BraggCalculator can take an observed powder pattern and a generated CIF through
one complete refinement call. The workflow loads both inputs, calculates the
candidate pattern, changes the selected parameters, and returns the refined
structure with fit and uncertainty information.

## Installation

Refinement uses PyTorch for automatic derivatives:

```bash
python -m pip install "braggcalculator[refinement]"
```

From a source checkout:

```bash
python -m pip install -e ".[refinement]"
```

## Smallest working example

An XYE file contains two-theta, intensity, and uncertainty:

```python
from braggcalculator import RefinementPolicy, refine_generated_cif

result = refine_generated_cif(
    pattern="observed.xye",
    cif="generated-candidate.cif",
    wavelength=1.5406,
    radiation="xray",
    policy=RefinementPolicy.quick(),
)

print(result.fit_statistics["r_wp"])
print(result.convergence["classification"])
result.write_cif("refined.cif")
```

The first two XYE columns are the coordinate and intensity. BraggCalculator
reads the third column as `sigma` by default. Set `third_column="weight"` when
the third column stores \(w=1/\sigma^2\).

`pattern` can also be a NumPy array with two or three columns. Separate
`sigma=` and `weights=` arrays are available for two-column input.

## Choosing parameters

A refinement policy says which physical quantities may move. Start with the
profile and unit cell:

```python
policy = RefinementPolicy.cautious()
```

Release atomic coordinates when the starting model has a plausible atom
assignment and the residual supports structural movement:

```python
policy = RefinementPolicy.cautious(refine_coordinates=True)
```

Occupancy and displacement options are explicit:

```python
occupancy_policy = RefinementPolicy.cautious(
    occupancy_mode="composition",
    refine_b_iso=True,
)

anisotropic_policy = RefinementPolicy.cautious(
    refine_u_aniso=True,
)
```

`occupancy_mode="composition"` keeps the declared species total on a mixed
site. `occupancy_mode="vacancy"` also permits an empty fraction. Isotropic
`B_iso` and anisotropic `U` use separate policies.

`RefinementPolicy.robust()` adds broad-to-narrow profile stages, deterministic
restarts, a held-out validation check, and an L-BFGS polish:

```python
policy = RefinementPolicy.robust(
    refine_coordinates=True,
    restarts=3,
)
```

Advanced users can build `OptimizationStage` objects and pass them through
`RefinementPolicy(stages=...)`.

## Bounds and structural constraints

BraggCalculator builds physical bounds into each parameterization:

- lattice lengths stay positive and lattice symmetry selects the independent
  metric modes;
- coordinates follow the starting space-group orbits and wrap into the unit
  cell;
- occupancies stay in the interval from zero to one and follow a simplex when
  several species share a site;
- isotropic displacement values stay positive;
- anisotropic displacement tensors stay positive semidefinite;
- scale and positive peak-width terms stay positive.

`result.parameters` lists every reported scalar with its value, unit, physical
range, constraint, and release state.

Chemical geometry can add measured or expected structure information:

```python
policy = RefinementPolicy(
    refine_coordinates=True,
    structural_restraints={
        "bonds": [
            {"sites": [0, 1], "target": 1.95, "sigma": 0.03},
        ],
        "minimum_distances": [
            {"sites": [0, 2], "minimum": 1.6, "sigma": 0.05},
        ],
    },
)
```

The result reports diffraction loss and restraint contributions separately.

## Atom-site permutation

A generated CIF may place the correct elements on the wrong crystallographic
sites. Element identity is a discrete choice. BraggCalculator therefore
screens complete structures first and runs continuous refinement on the best
assignments.

```python
from braggcalculator import SpeciesAssignmentConfig

assignment = SpeciesAssignmentConfig(
    search="auto",
    fixed_sites=(2,),
    allowed_species={
        0: ("Fe", "Mn"),
        1: ("Fe", "Mn"),
    },
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

The site index refers to one independent site in the asymmetric unit. The
space group expands that site to every symmetry-equivalent atom in the crystal.
Inspect these sites before configuring a search:

```python
from braggcalculator import asymmetric_unit_sites
from braggcalculator.io import to_pmg_structure

sites = asymmetric_unit_sites(to_pmg_structure("generated-candidate.cif"))
for site in sites:
    print(
        site.site_index,
        site.label,
        site.wyckoff_symbol,
        site.multiplicity,
        site.original_species,
    )
```

Named groups make repeated rules easier to read:

```python
assignment = SpeciesAssignmentConfig(
    site_groups={
        "transition_metals": (0, 1, 3),
        "oxygen_framework": (2, 4),
    },
    fixed_sites=("oxygen_framework",),
    allowed_species={
        "transition_metals": ("Fe", "Mn", "Co"),
    },
)
```

## Composition and symmetry rules

Composition stays fixed by default. A direct swap preserves composition when
both sites have the same multiplicity. Assignments across different
multiplicities pass an exact composition check.

The search operates on asymmetric-unit sites. This removes duplicates created
by assigning the same species to symmetry-equivalent full-cell atoms.

Mixed occupancies stay attached to their starting site with
`mixed_occupancy_policy="fixed"`. Use `"reject"` to stop when a mixed site is
present.

Displacement values follow the crystallographic site by default:

```python
SpeciesAssignmentConfig(displacement_policy="site")
```

Set `displacement_policy="species"` to move site properties with the assigned
species. This policy requires a source site with the same species and
multiplicity.

User-supplied oxidation states can add a charge-balance check:

```python
SpeciesAssignmentConfig(
    oxidation_states={"Na": 1, "Ti": 4, "O": -2},
    target_charge=0,
)
```

## Search modes

- `"pairwise"` checks the original assignment and every allowed direct swap.
- `"complete"` checks all allowed combinations up to `max_candidates`.
- `"bounded"` walks the allowed combinations in deterministic order and stops
  at `max_candidates`.
- `"random"` explores multiplicity-compatible permutations with a reproducible
  `seed`.
- `"auto"` selects complete search for a small space and bounded search for a
  larger space.

Every valid assignment receives a fast screening score. Continuous refinement
runs on `continuous_top_k` assignments. Increase this value when screening and
continuous profile models rank close candidates differently.

The result records `truncated=True` when the candidate limit stops the search.
Its warning includes the configured limit.

## Reading the result

The main result contains:

- `refined_structure` and `refined_cif`;
- `coordinate`, `observed`, `calculated`, and `residual`;
- `objective_history` and `stage_history`;
- `status` and the detailed `convergence` record;
- `parameters` with values, bounds, units, and release state;
- `fit_statistics`, including \(R_\mathrm{wp}\), \(\chi^2\), and held-out
  \(R_\mathrm{wp}\);
- `diagnostics`, including identifiability, informative regions, and the next
  recommended action;
- `warnings` and input/optimizer `provenance`.

When species screening is active, `result.species_assignments.candidates`
contains the ranked assignments. Each candidate records:

- original and proposed species at every independent site;
- site index, Wyckoff symbol, and multiplicity;
- screening and continuous scores;
- convergence details;
- refined structure and CIF;
- `indistinguishable=True` when its score falls within
  `ambiguity_tolerance` of the best refined assignment.

Several assignments can produce the same powder intensities. The ambiguity
flag means the supplied experiment supports those assignments equally within
the declared tolerance.

## Performance guidance

Start with pairwise search when one swapped pair is likely. Use complete search
for a few independent sites. Set `max_candidates` before opening a larger
allowed-species space.

Fast screening fits scale and polynomial background for every assignment.
Continuous refinement carries most of the compute cost. A practical first pass
uses `continuous_top_k=3` to `5`, followed by a larger value when several
screening scores cluster together.

X-ray and neutron patterns can rank assignments differently because the
elements have different scattering contrast. Run both experiments when both
datasets are available.

## Compact demonstration

The repository demonstration generates an observed pattern, loads a swapped
CIF, screens the assignments, refines the best two, prints the ranking, and
writes the best refined CIF:

```bash
python demo/refine_generated_cif.py --output refined-demo.cif
```

## Current scientific limits

The focused session uses two-theta input. The uncertainty estimates describe
the local model around the refined solution. Species screening keeps the
reflection topology and crystallographic symmetry inferred from each complete
candidate. Powder diffraction may leave element assignments unresolved; the
ranked ambiguity report carries that information into the result.
