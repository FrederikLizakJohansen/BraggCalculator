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

For a complete CIF-to-simulated-data workflow, open
[`notebooks/artifact_simulation.ipynb`](../notebooks/artifact_simulation.ipynb).
It starts from the same NaCl CIF, constructs each experimental-effect component,
loads `example_background.xye`, and combines the effects with a reproducible
seed. The `.xye` file is a synthetic tutorial trace, not measured reference
data.
