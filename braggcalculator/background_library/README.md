# Background library

This directory is reserved for measured background traces that may be
redistributed with BraggCalculator. `manifest.json` is intentionally empty
until a trace satisfies all of the following requirements:

- a stable primary source or archive record;
- permission to redistribute the numerical trace;
- an identified instrument, radiation, holder/environment and scan geometry;
- declared coordinate and intensity units;
- a declared third-column uncertainty or weight convention;
- the unmodified source file and its SHA-256 digest.

Each manifest entry must contain `path`, `domain`, `source`, and `sha256`.
Optional `third_column` values are `sigma`, `weight`, or `ignore`. The
`BackgroundLibrary` loader confines paths to this directory and verifies the
digest before parsing a trace.
