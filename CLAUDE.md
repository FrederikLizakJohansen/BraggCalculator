# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BraggCalculator is a symmetry- and hkl-based powder X-ray diffraction (PXRD) engine, a sister project to DebyeCalculator. It computes diffraction patterns from crystal structures provided as `.cif` files, ASE `Atoms`, or `pymatgen.Structure` objects. It supports both X-ray and neutron modes.

## Commands

### Install dependencies
```bash
poetry install
```

### Run all tests
```bash
poetry run pytest tests/
```

### Run a single test
```bash
poetry run pytest tests/test_factor.py::test_fq_runs_smoke
```

## Architecture

The main entry point is the `BraggCalculator` dataclass in `braggcalculator/core.py`, exposed via `__init__.py`. The calculation pipeline follows this flow:

1. **`io.py`** — `to_pmg_structure()` converts input (CIF path, pymatgen Structure, ASE Atoms) into a `pymatgen.Structure`.
2. **`symmetry.py`** — `SymmetryEngine.reduce()` uses spglib (via pymatgen's `SpacegroupAnalyzer`) to get the primitive cell, unique sites, orbit mappings, and symmetry operations.
3. **`hkl.py`** — `HKLEnumerator.enumerate()` generates candidate (h,k,l) indices up to `hkl_max`, computes d-spacings via the reciprocal metric tensor, filters by `qmax`, and estimates multiplicities from Laue group operations.
4. **`structure_factor.py`** — `compute_F2()` calculates |F(hkl)|² using scattering factors, phases from fractional coordinates, occupancies, and Debye-Waller factors.
5. **`factors.py`** — Provides scattering factor lookup (`xray_form_factors`, `neutron_b_coherent`). Currently placeholder implementations using Z as proxy.
6. **`renderer.py`** — Applies Lorentz-polarization correction and multiplicity (`apply_lp_and_multiplicity`), then convolves delta-line peaks onto a grid via profile functions.
7. **`profiles.py`** — Gaussian broadening profiles for both 2θ and Q domains.

### Backend system

All numerical operations go through a backend abstraction (`braggcalculator/backends/`):
- **`NumpyBackend`** (default) — wraps NumPy.
- **`TorchBackend`** (optional) — wraps PyTorch for GPU acceleration and autograd compatibility. Only imported if torch is available.

Backends expose a common API (`asarray`, `exp`, `sin`, `einsum`, `linspace`, etc.). New backends must implement this same interface.

### Key output methods on `BraggCalculator`

- `load(structure_like)` — parses structure, runs symmetry reduction, enumerates hkl.
- `fq()` — per-hkl |F|² (no corrections).
- `iq(domain)` — delta-line intensities after LP & multiplicity.
- `pattern(domain)` — gridded intensity profile. `domain` is `"two_theta"` (default) or `"q"`.

## Known Placeholders / TODOs

Several modules have placeholder implementations marked with TODO comments:
- `factors.py`: X-ray form factors use Z as a flat proxy (should use Waasmaier-Kirfel tables); neutron b_coh is a flat 5.0 fm.
- `hkl.py`: Laue group operations are identity-only placeholders; systematic extinction rules are not yet implemented.
- `renderer.py`: LP factor is a simplified approximation.
