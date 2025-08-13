# BraggCalculator

A symmetry- and hkl-based PXRD engine — sister to DebyeCalculator.
Inputs: `.cif`, ASE `Atoms`, or `pymatgen.Structure`.
Backends: NumPy (default), Torch (optional, for GPU & autograd).

## Quickstart
```python
from braggcalculator import BraggCalculator

calc = BraggCalculator(mode="xray", wavelength=1.5406)
calc.load("NaCl.cif")
tt, I = calc.pattern()
