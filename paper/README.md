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

This writes a 450 dpi PNG preview, editable vector PDF and SVG figures, plus
exact error metrics to `paper/figures/`. Both implementations use the same
0.9 pt stroke weight; solid orange and dashed blue profiles remain visible
even when their values overlap exactly. To inspect every deterministic
reference case, run:

```bash
python scripts/plot_pattern_comparison.py \
  --cases NaCl Si SrTiO3 triclinic-SiO2 NaKCl-disordered P1-40-atom
```

The script exits with an error if the line positions, line intensities, or
broadened profiles exceed their stated tolerances.

## Scaling and multi-hardware benchmark

The scaling benchmark contains two controlled series. “Supplied sites” means
`len(structure)` before any BraggCalculator symmetry processing; for these
ordered structures it is also the supplied atom count. Triclinic P1 cells
increase from 4 to 256 supplied and irreducible sites at approximately fixed
density. Conventional NaCl supercells contain 8, 64, 216, and 512 supplied
sites, while BraggCalculator reduces every member to the same two-site
primitive cell. This separates irreducible atom-count scaling from the benefit
and preprocessing cost of symmetry reduction.

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

The primary publication outputs are editable vector PDF and SVG at Nature's
183 mm two-column width. The PNG is a 450 dpi review preview. Runtime points
are medians and error bars span the interquartile range of the seven retained
repeat samples.

### CUDA machine

The CPU command does not use a GPU even when one is installed. On the GPU
machine, install the CUDA-enabled PyTorch build and run this separate command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python benchmarks/benchmark_scaling.py \
  --backend torch --device cuda \
  --p1-sites 4 8 16 32 64 128 256 \
  --symmetry-factors 1 2 3 4 \
  --number 10 --repeat 7 \
  --output scaling_gpu.json
```

The collector uses float64 for oracle-quality parity and synchronizes CUDA
immediately before and after every timed block. “Cached” measures repeated GPU
diffraction with a prepared topology. “End-to-end” includes CPU symmetry and
HKL preprocessing, host-to-device transfers, and synchronized GPU execution.
pymatgen remains on the same machine's CPU, which is recorded explicitly in
the JSON and figure legend.
