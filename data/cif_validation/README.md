# Frozen COD validation corpus

This directory contains version 1.0.0 of the BraggCalculator powder-pattern
validation corpus: 70 CIFs from the Crystallography Open Database (COD), with
ten structures from each of the seven crystal systems. The records cover 62
declared space groups, 3--992 supplied sites, and 31 structures with partial
occupancy or disorder.

The corpus is selected before comparing either diffraction implementation.
Selection excludes retracted, database-error, theoretical, duplicate,
unparseable, and unsupported-element records. It does **not** depend on whether
BraggCalculator agrees with pymatgen. The complete rule, fixed seed, COD
revision, source URL, and SHA-256 digest for every file are recorded in
`manifest.json`.

Rebuild a new corpus release from the current COD state with:

```bash
python scripts/build_cif_validation_corpus.py --accessed-date YYYY-MM-DD
```

Run the frozen X-ray and neutron comparison with:

```bash
python scripts/validate_cif_corpus.py \
  --output paper/data/cif_validation_results.json
```

Both calculators receive the same pymatgen-parsed `Structure`. Parser warnings
are retained in the result artifact because some deposited formulae omit
hydrogen or otherwise differ from the coordinate model. This benchmark tests
implementation agreement for the parsed periodic structure; it is not an
experimental validation of any deposition or of the kinematic model.

COD data are dedicated to the public domain under CC0. Each CIF retains its
original authorship, publication, COD identifier, and revision metadata. Users
of individual structures should acknowledge the original authors recorded in
that CIF as requested by COD.
