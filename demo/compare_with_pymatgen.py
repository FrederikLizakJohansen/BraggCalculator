#!/usr/bin/env python3
"""Calculate the NaCl powder pattern and compare it with pymatgen."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure

from braggcalculator import BraggCalculator


HERE = Path(__file__).resolve().parent
CIF_PATH = HERE / "NaCl.cif"
DEFAULT_OUTPUT = HERE / "nacl_vs_pymatgen.png"


def calculate_patterns() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized NaCl powder lines from both implementations."""
    structure = Structure.from_file(CIF_PATH)
    calculator = BraggCalculator(wavelength="CuKa1").load(CIF_PATH)
    actual_x, actual_y = calculator.line_pattern(scaled=True)
    expected = XRDCalculator(wavelength=calculator.wavelength).get_pattern(
        structure,
        two_theta_range=calculator.two_theta_range,
        scaled=True,
    )

    actual_x = np.asarray(actual_x)
    actual_y = np.asarray(actual_y)
    expected_x = np.asarray(expected.x)
    expected_y = np.asarray(expected.y)
    np.testing.assert_allclose(actual_x, expected_x, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(actual_y, expected_y, rtol=1e-10, atol=1e-10)
    return actual_x, actual_y, expected_x, expected_y


def plot_comparison(output: Path, *, show: bool = False) -> tuple[float, float]:
    """Write the oracle comparison plot and return the two maximum errors."""
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    actual_x, actual_y, expected_x, expected_y = calculate_patterns()
    position_error = float(np.max(np.abs(actual_x - expected_x), initial=0.0))
    intensity_error = float(np.max(np.abs(actual_y - expected_y), initial=0.0))

    plt.rcParams.update(
        {
            "axes.spines.right": False,
            "axes.spines.top": False,
            "font.size": 9,
            "pdf.fonttype": 42,
        }
    )
    figure, (pattern_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(7.0, 4.6),
        gridspec_kw={"height_ratios": (4, 1), "hspace": 0.08},
        sharex=True,
        layout="constrained",
    )
    pattern_axis.vlines(
        expected_x,
        0.0,
        expected_y,
        color="#D55E00",
        linewidth=2.0,
        label="pymatgen",
    )
    pattern_axis.vlines(
        actual_x,
        0.0,
        actual_y,
        color="#0072B2",
        linewidth=1.0,
        linestyles="dashed",
        label="BraggCalculator",
    )
    residual_axis.axhline(0.0, color="0.55", linewidth=0.8)
    residual_axis.plot(actual_x, actual_y - expected_y, "o", color="#0072B2", markersize=3)

    pattern_axis.set_title("NaCl powder X-ray diffraction (Cu Kα₁)")
    pattern_axis.set_ylabel("Relative intensity (%)")
    pattern_axis.legend(frameon=False)
    residual_axis.set_xlabel(r"Scattering angle $2\theta$ (degrees)")
    residual_axis.set_ylabel("Difference\n(% points)")
    pattern_axis.grid(axis="x", color="0.92", linewidth=0.6)
    residual_axis.grid(axis="x", color="0.92", linewidth=0.6)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    if show:
        plt.show()
    plt.close(figure)
    return position_error, intensity_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    position_error, intensity_error = plot_comparison(args.output, show=args.show)
    print(f"wrote {args.output}")
    print(f"maximum position error: {position_error:.3e} degrees")
    print(f"maximum intensity error: {intensity_error:.3e} percentage points")


if __name__ == "__main__":
    main()
