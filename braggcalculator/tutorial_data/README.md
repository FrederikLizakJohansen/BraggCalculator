# Guided UI tutorial data

This directory contains the portable teaching example loaded by `bragg-ui`.

- `pattern.xye` is a synthetic Cu K-alpha powder pattern with columns
  `2theta`, intensity and standard uncertainty.
- `model-a.cif` is the reference NaSiO2 motif used to generate the teaching
  pattern.
- `model-b.cif` is a lattice-compatible candidate with shifted oxygen sites.

The example is deliberately ambiguous after profile broadening and refinement.
Its purpose is to teach how fit quality, complex structure-factor mismatch,
peak overlap, parameter identifiability and expected experimental
discriminability answer different questions. It is not an independent
experimental validation dataset.

The application copies and checksums all three files into each new tutorial
project, so the original package resources remain unchanged.
