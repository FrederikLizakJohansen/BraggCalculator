# Reference-validation powder data

Four unmodified constant-step banks are vendored from the official
[GSAS-II tutorial corpus](https://advancedphotonsource.github.io/GSAS-II-tutorials/tutorials.html).
Their URLs, SHA-256 checksums, instrument classes, wavelengths and expected
point counts are frozen in `manifest.json`. The manifest also covers the
complete NIST SRM 660c scan vendored one directory above.

They are validation inputs, not claims of a successful Rietveld refinement.
The current milestone checks immutable provenance, format ingestion and
instrument/material coverage. Direct reproduction of a GSAS-II final profile
requires a frozen GSAS-II project and is tracked separately from the pymatgen
line-pattern oracle and NIST certified lattice comparison.

The original 80-column records and CRLF line endings are deliberately retained
so the recorded checksums continue to identify the downloaded source files.
