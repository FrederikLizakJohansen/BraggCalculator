# Contributing

Scientific changes must include an analytical test or an independently
reviewable reference result. Changes to line positions or intensities must also
run `python scripts/validate_against_pymatgen.py` and explain any intentional
difference from the oracle.

Before opening a pull request, run:

```bash
python -m pytest -q
ruff check .
python scripts/validate_against_pymatgen.py
python benchmarks/benchmark_against_pymatgen.py --require-speedup 1
```

Performance changes should report the JSON output from the benchmark script,
including dependency versions and platform metadata. Do not replace physical
constants with fitted or approximate placeholder values.
