# Publication figures

Every figure is generated from versioned structures and records the software
environment and numerical results in JSON. The left column of the pattern
comparison mirrors the unbroadened powder lines. The other columns apply the
same area-normalized Gaussian broadening to both sets of lines at FWHM 0.1° and
at a deliberately stronger FWHM 0.5°, then overlay the resulting profiles.
This separates agreement in the diffraction calculation from agreement after
an explicitly identical instrument broadening step and checks that stronger
peak overlap does not conceal a discrepancy.

Generate the initial X-ray oracle-comparison figure:

```bash
python scripts/plot_pattern_comparison.py
```

This writes PNG and vector PDF figures plus exact error metrics to
`paper/figures/`. To inspect every deterministic reference case, run:

```bash
python scripts/plot_pattern_comparison.py \
  --cases NaCl Si SrTiO3 triclinic-SiO2 NaKCl-disordered P1-40-atom
```

The script exits with an error if the line positions, line intensities, or
broadened profiles exceed their stated tolerances.

## Scaling and multi-hardware benchmark

The scaling benchmark contains two controlled series. Triclinic P1 cells
increase the number of irreducible sites at approximately fixed density. NaCl
supercells increase the input from 8 to 512 sites while retaining the same
two-site primitive cell, exposing the effect of symmetry reduction.

Run this exact command on each machine and return the resulting JSON file:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python benchmarks/benchmark_scaling.py \
  --p1-sites 4 8 16 32 64 128 256 \
  --symmetry-factors 1 2 3 4 \
  --number 10 --repeat 7 \
  --output scaling_results.json
```

The hardware label defaults to the detected CPU model. Use
`--hardware-label` only when two machines report indistinguishable CPU names.
Every case is checked against pymatgen before timing, and the JSON retains all
repeat samples rather than only summary values.

Combine any number of returned runs into the scaling figure:

```bash
python scripts/plot_scaling_benchmark.py machine_a.json machine_b.json
```
