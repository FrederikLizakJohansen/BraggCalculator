# NaCl demonstration

`compare_with_pymatgen.py` loads `NaCl.cif`, calculates its normalized Cu
Kα₁ powder lines with BraggCalculator and pymatgen, verifies their numerical
agreement, and plots the overlaid patterns with a residual panel.

From the repository root, install the package and plotting dependency, then run:

```bash
python -m pip install -e . matplotlib
python demo/compare_with_pymatgen.py
```

The default output is `demo/nacl_vs_pymatgen.png`. Use `--show` to open the
figure interactively or `--output path/to/figure.pdf` to write a vector PDF.

## Compatible-model mismatch disk

`diagnose_compatible_models.py` constructs two same-cell models. The second
uses a different origin, reversed atom order, and one genuine oxygen-coordinate
perturbation. It recovers the arbitrary origin shift and plots the remaining
bounded amplitude-phase mismatch:

```bash
python demo/diagnose_compatible_models.py
```

The script prints the unaligned and aligned dissimilarities, their amplitude
and phase components, the recovered origin correction, the disk-identity
error, and the five most mismatched reflections. Its default figure is
`demo/mismatch_disk.png`.
