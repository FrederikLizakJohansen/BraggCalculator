# Frozen diffraction-diagnostics publication inputs

These CIFs are synthetic, deterministic benchmark cases. They are not a
collection of experimentally solved structures.

- `homometric-a.cif` and `homometric-b.cif` place identical Si scatterers on
  two non-congruent subsets of the cyclic group Z8. Their periodic difference
  multisets, and therefore ideal kinematic intensities, are identical.
- `near-homometric-b.cif` moves one site by 0.003 fractional units to break the
  exact equality in a controlled way.
- `resolution-cubic.cif` and `resolution-strained.cif` differ by 0.5% in one
  cell length. Broad profiles hide much of the resulting splitting; narrow
  profiles expose it.
- `realistic-compatible-a.cif` and `realistic-compatible-b.cif` are the
  NaSiO2 teaching candidates also used by the guided UI.

`manifest.json` freezes file hashes, roles and generation assumptions. The
publication script refuses to run when an input digest changes.
