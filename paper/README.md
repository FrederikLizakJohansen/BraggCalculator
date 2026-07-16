# Publication figures

Every figure is generated from versioned structures and records the software
environment and numerical results in JSON. The plotted continuous profiles use
the same area-normalized Gaussian broadening for BraggCalculator and pymatgen,
so the comparison tests the diffraction calculation rather than two different
instrument models.

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
