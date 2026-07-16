#!/usr/bin/env python3
"""Validate line positions and intensities against pymatgen XRD and ND.

This script is intentionally independent of pytest so its table can be used in
continuous integration and JOSS review artifacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from pymatgen.analysis.diffraction.neutron import NDCalculator
from pymatgen.analysis.diffraction.xrd import XRDCalculator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.reference_cases import reference_structures  # noqa: E402
from braggcalculator import BraggCalculator  # noqa: E402


def validate(mode: str, position_atol: float, intensity_atol: float) -> None:
    oracle_type = XRDCalculator if mode == "xray" else NDCalculator
    print(f"{'case':<20} {'mode':<8} {'peaks':>7} {'max |d2theta|':>16} {'max |dI|':>14}")
    for name, structure in reference_structures().items():
        calculator = BraggCalculator(mode=mode).load(structure)
        actual_x, actual_y = calculator.line_pattern(scaled=True)
        oracle = oracle_type(wavelength=calculator.wavelength).get_pattern(
            structure,
            two_theta_range=calculator.two_theta_range,
            scaled=True,
        )
        actual_x = np.asarray(actual_x)
        actual_y = np.asarray(actual_y)
        np.testing.assert_allclose(actual_x, oracle.x, rtol=0, atol=position_atol)
        np.testing.assert_allclose(actual_y, oracle.y, rtol=0, atol=intensity_atol)
        dx = float(np.max(np.abs(actual_x - oracle.x), initial=0.0))
        dy = float(np.max(np.abs(actual_y - oracle.y), initial=0.0))
        print(f"{name:<20} {mode:<8} {len(actual_x):>7d} {dx:>16.3e} {dy:>14.3e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--position-atol", type=float, default=1e-10)
    parser.add_argument("--intensity-atol", type=float, default=1e-9)
    parser.add_argument("--mode", choices=("xray", "neutron", "both"), default="both")
    args = parser.parse_args()
    modes = ("xray", "neutron") if args.mode == "both" else (args.mode,)
    for mode in modes:
        validate(mode, args.position_atol, args.intensity_atol)


if __name__ == "__main__":
    main()
