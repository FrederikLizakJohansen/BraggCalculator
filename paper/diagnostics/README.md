# Diffraction diagnostics publication package

This directory is the reproducible Milestone 8 package for a proposed paper on
explaining structural ambiguity in powder diffraction. It is separate from the
forward-kernel JOSS manuscript in `paper/`.

## Release status

`pending_external_review`

All ten frozen software/mathematical gates pass except the deliberately human
gate: no external crystallographer has yet signed the blinded explanation
review. The package must not be described as expert-validated until completed
review packets meet the protocol in `expert-review.md`.

## Reproduce everything

From the repository root:

```bash
python scripts/run_diagnostics_publication.py --verify
```

The command verifies every input SHA-256, recomputes all structure factors,
profiles, metrics, invariance transformations and release gates, then rewrites
the machine-readable results, tables, review packet and PNG/PDF/SVG figures.
It exits non-zero if a numerical gate fails. A pending human-review gate is
reported but is not converted into a software failure.

## Frozen benchmark result

| Case | Key observation |
|---|---|
| Exact homometric | profile cosine 1.000000; shell-balanced amplitude mismatch below 6e-16; phase mismatch 0.3740 |
| Near homometric | profile cosine 0.999936; shell-balanced complex mismatch 0.3750 |
| Compatible NaSiO2 candidates | profile cosine 0.999365; shell-balanced complex mismatch 0.03846 |
| Strained cell, broad FWHM 0.20 A^-1 | profile cosine 0.999665 |
| Strained cell, high-resolution FWHM 0.015 A^-1 | profile cosine 0.937674 |

The exact construction uses the non-congruent cyclic Z8 subsets
`{0,3,4,5}` and `{0,1,3,4}`. The analysis independently verifies their equal
directed periodic-difference multisets before evaluating diffraction.

## Contents

- `manuscript.md`: working diagnostics-paper draft;
- `results.json`: complete numerical record and release gates;
- `artifact_manifest.json`: byte hashes for every frozen generated artifact
  and the analysis sources;
- `metric_table.csv`: compact comparison table;
- `environment.json`: generation environment;
- `requirements-lock.txt`: exact key package versions used for the frozen run;
- `expert_review_packet.json`: unsigned blinded review form;
- `expert_review_key.json`: blind-ID mapping, kept separate during review;
- `figures/diagnostic_benchmark.*`: information-loss and metric comparison;
- `figures/weighting_invariance.*`: invariance, reflection-range sensitivity
  and release gates.

Inputs and their digests are frozen under
`data/publication_diagnostics/manifest.json`. All cases are synthetic. They
test mathematical behavior and explanatory logic; they do not substitute for
independent experimental validation or establish arbitrary structure-solution
performance.
