# JOSS paper and publication figures

`paper.md` and `paper.bib` are the canonical JOSS manuscript sources. This is
required by the current JOSS publishing pipeline; the paper workflow builds
them with the official Open Journals action. `paper.tex` is a matching LaTeX
companion for local review and is not the submission source.

Build the local LaTeX copy from this directory with:

```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

The two manuscript figures are generated from versioned structures and records
of the software environment and numerical results in JSON.

The left column of the pattern
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

## Frozen CIF corpus validation

The larger oracle check uses 70 CC0 structures from the Crystallography Open
Database: ten from every crystal system, 62 declared space groups, 3--992
supplied sites, and 31 structures containing partial occupancy or disorder.
The fixed selection rule does not inspect BraggCalculator agreement. Every COD
revision and file digest is recorded in `data/cif_validation/manifest.json`.

```bash
python scripts/validate_cif_corpus.py \
  --output paper/data/cif_validation_results.json
```

The command compares both X-ray and neutron line patterns, writes every result
and CIF parser warning to JSON, and exits nonzero if any comparison fails.

## Scaling and multi-hardware benchmark

The scaling benchmark contains two controlled series. “Supplied sites” means
`len(structure)` before any BraggCalculator symmetry processing; for these
ordered structures it is also the supplied atom count. Triclinic P1 cells
increase from 4 to 256 supplied and irreducible sites at approximately fixed
density. Conventional NaCl supercells contain 8, 64, 216, and 512 supplied
sites, while BraggCalculator reduces every member to the same two-site
primitive cell. This separates irreducible atom-count scaling from the benefit
and preprocessing cost of symmetry reduction.

Run both CPU backends on the comparison host:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python benchmarks/benchmark_scaling.py \
  --backend numpy --device cpu \
  --p1-sites 4 8 16 32 64 128 256 \
  --symmetry-factors 1 2 3 4 \
  --number 10 --repeat 7 \
  --output scaling_cpu_numpy.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python benchmarks/benchmark_scaling.py \
  --backend torch --device cpu \
  --p1-sites 4 8 16 32 64 128 256 \
  --symmetry-factors 1 2 3 4 \
  --number 10 --repeat 7 \
  --output scaling_cpu_torch.json
```

The hardware label defaults to the detected CPU model. Use
`--hardware-label` only when two machines report indistinguishable CPU names.
Every case is checked against pymatgen before timing, and the JSON retains all
repeat samples together with summary values.

Combine one record per execution path into the scaling figure. The versioned
paper figure uses the NumPy CPU, PyTorch CPU, and PyTorch CUDA measurements from
the same A3000 WSL2 host:

```bash
python scripts/plot_scaling_benchmark.py \
  paper/data/scaling_cpu_numpy_A3000.json \
  paper/data/scaling_cpu_torch_A3000.json \
  paper/data/scaling_nvidia_rtx_a3000_laptop.json
```

The primary publication outputs are editable vector PDF and SVG at Nature's
183 mm two-column width. The PNG is a 450 dpi review preview. Color distinguishes
the NumPy CPU, PyTorch CPU, and PyTorch CUDA paths; line style distinguishes
cached from end-to-end timing. Runtime lines show medians, speedup lines show
ratios of medians, and bands show interquartile ranges over the raw timings or
paired speedup ratios. The pymatgen runtime trace pools all same-host repeats,
while each speedup series uses the pymatgen repeats in its own timing record.
The bottom row plots the direct ratio of PyTorch CPU and CUDA median runtimes.

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
the JSON. Consequently, each speedup series compares BraggCalculator with the
pymatgen samples in its own record. All three versioned records share the same
WSL2 host, Git revision, dependencies, cases, dtype, thread limits, and timing
protocol, so the NumPy CPU, PyTorch CPU, and CUDA runtimes form a controlled
same-machine comparison. The NaCl series becomes the same two-site, 410-HKL
workload after reduction, so CUDA launch, transfer, and synchronization costs
dominate its cached time.
