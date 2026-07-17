#!/usr/bin/env python3
"""Demonstrate origin-aligned mismatch-disk diagnostics for two models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure

from braggcalculator import BraggCalculator, MismatchDiskResult
from braggcalculator.diagnostics import compare_calculators


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "mismatch_disk.png"


def calculate_diagnostics() -> tuple[MismatchDiskResult, MismatchDiskResult, np.ndarray]:
    """Return unaligned/aligned comparisons and matched Q values."""
    lattice = Lattice.from_parameters(4.6, 5.3, 6.1, 78.0, 84.0, 72.0)
    species = ["Si", "O", "O", "N"]
    coordinates = np.array(
        [
            [0.13, 0.21, 0.34],
            [0.31, 0.47, 0.11],
            [0.72, 0.08, 0.59],
            [0.44, 0.76, 0.27],
        ]
    )
    model_a = Structure(lattice, species, coordinates)

    origin_change = np.array([0.125, 0.25, 0.375])
    model_b_coordinates = (coordinates + origin_change) % 1.0
    model_b_coordinates[1, 0] += 0.018  # The genuine structural difference.
    model_b = Structure(
        lattice,
        species[::-1],
        model_b_coordinates[::-1],
        coords_are_cartesian=False,
    )

    settings = dict(wavelength="CuKa1", q_range=(0.4, 7.0), q_step=0.01)
    calculator_a = BraggCalculator(**settings).load(model_a)
    calculator_b = BraggCalculator(**settings).load(model_b)
    unaligned = compare_calculators(calculator_a, calculator_b, optimize_origin=False)
    aligned = compare_calculators(calculator_a, calculator_b, optimize_origin=True)
    table_a = calculator_a.reflection_table(domain="q")
    matched_q = np.asarray(table_a.q)[aligned.match.indices_a]
    return unaligned, aligned, matched_q


def plot_disk(output: Path, *, show: bool = False) -> MismatchDiskResult:
    """Write the mismatch disk and return the aligned numerical result."""
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    _, result, q = calculate_diagnostics()
    average_amplitude = 0.5 * (result.amplitude_a + result.amplitude_b)
    point_size = 12.0 + 75.0 * average_amplitude / max(float(average_amplitude.max()), 1.0)

    figure, axis = plt.subplots(figsize=(6.2, 5.4), layout="constrained")
    angle = np.linspace(0.0, 2.0 * np.pi, 500)
    axis.plot(np.cos(angle), np.sin(angle), color="0.35", linewidth=1.0)
    points = axis.scatter(
        result.x,
        result.y,
        c=q,
        s=point_size,
        cmap="viridis",
        alpha=0.72,
        edgecolors="none",
    )
    axis.axhline(0.0, color="0.82", linewidth=0.7)
    axis.axvline(0.0, color="0.82", linewidth=0.7)
    axis.set(
        aspect="equal",
        xlim=(-1.04, 1.04),
        ylim=(-1.04, 1.04),
        xlabel="Normalized amplitude mismatch",
        ylabel="Signed phase mismatch",
        title=(
            "Origin-aligned structure-factor mismatch\n"
            rf"$D_{{\mathrm{{SF}}}}$={result.d_sf:.4f} "
            f"(amplitude={result.d_amplitude:.4f}, phase={result.d_phase:.4f})"
        ),
    )
    colorbar = figure.colorbar(points, ax=axis, shrink=0.82)
    colorbar.set_label(r"$Q$ ($\AA^{-1}$)")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    if show:
        plt.show()
    plt.close(figure)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unaligned, aligned, q = calculate_diagnostics()
    plot_disk(args.output, show=args.show)
    print(f"wrote {args.output}")
    print(f"matched reflections: {len(aligned.match)}")
    print(f"unaligned D_SF: {unaligned.d_sf:.6f}")
    print(f"aligned D_SF:   {aligned.d_sf:.6f}")
    print(f"  amplitude:    {aligned.d_amplitude:.6f}")
    print(f"  phase:        {aligned.d_phase:.6f}")
    print("origin correction: " + np.array2string(aligned.alignment.shift, precision=6))
    print(f"disk identity error: {aligned.identity_error:.3e}")
    print("largest remaining mismatches:")
    for index in np.argsort(aligned.radius)[-5:][::-1]:
        h, k, ell = aligned.match.hkl[index]
        print(
            f"  ({h:2d} {k:2d} {ell:2d})  Q={q[index]:.3f}  "
            f"r={aligned.radius[index]:.4f}  "
            f"x={aligned.x[index]:+.4f}  y={aligned.y[index]:+.4f}"
        )


if __name__ == "__main__":
    main()
