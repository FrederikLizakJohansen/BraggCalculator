#!/usr/bin/env python3
"""Demonstrate bin-aware discrimination and scaled Jacobian guidance."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure

from braggcalculator import BraggCalculator, JacobianDiagnostics, ProfileDiscriminationResult
from braggcalculator.backends import TorchBackend
from braggcalculator.diagnostics import compare_profile_counts
from braggcalculator.sensitivity import ParameterPath, analyze_jacobian, torch_profile_jacobian


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "profile_information.png"
COUNT_SCALE = 0.1
BACKGROUND_DENSITY = 100.0


def calculate_information() -> tuple[ProfileDiscriminationResult, JacobianDiagnostics]:
    """Return expected profile separation and local parameter diagnostics."""
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
    model_b = model_a.copy()
    model_b.translate_sites([1], [0.018, 0.0, 0.0], frac_coords=True)

    settings = dict(
        backend=TorchBackend(),
        wavelength="CuKa1",
        q_range=(0.4, 7.0),
        q_step=0.02,
    )
    calculator_a = BraggCalculator(**settings).load(model_a)
    calculator_b = BraggCalculator(**settings).load(model_b)
    comparison = compare_profile_counts(
        calculator_a,
        calculator_b,
        count_scale=COUNT_SCALE,
        background_density=BACKGROUND_DENSITY,
    )

    paths = (
        ParameterPath("frac_coords", (1, 0), 0.01, "O1 x"),
        ParameterPath("frac_coords", (1, 1), 0.01, "O1 y"),
        ParameterPath("frac_coords", (2, 0), 0.01, "O2 x"),
        ParameterPath("frac_coords", (0, 0), 0.01, "Si x"),
    )
    parameters = calculator_a.tensor_parameters()
    grid, _, density_jacobian = torch_profile_jacobian(
        calculator_a, parameters, paths, domain="q"
    )
    np.testing.assert_allclose(grid, comparison.coordinate, rtol=0.0, atol=1e-12)
    count_jacobian = (
        density_jacobian * comparison.bin_widths[:, None] * COUNT_SCALE
    )
    diagnostics = analyze_jacobian(
        count_jacobian,
        residual=comparison.expected_b - comparison.expected_a,
        weights=1.0 / comparison.variance,
        parameter_scales=[path.scale for path in paths],
        parameter_names=[path.display_name for path in paths],
    )
    return comparison, diagnostics


def plot_information(output: Path, *, show: bool = False):
    """Write the connected profile/discrimination/information figure."""
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    comparison, diagnostics = calculate_information()
    q = comparison.coordinate
    sigma = np.sqrt(comparison.variance)
    standardized_b_minus_a = -comparison.difference / sigma

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(8.0, 7.0),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": (2.2, 1.2, 1.5)},
    )
    axes[0].plot(q, comparison.expected_a, color="#0072B2", label="model A")
    axes[0].plot(q, comparison.expected_b, color="#D55E00", alpha=0.8, label="model B")
    axes[0].set_ylabel("Expected counts/bin")
    axes[0].legend(frameon=False)

    axes[1].plot(q, standardized_b_minus_a, color="0.25", linewidth=0.9)
    discrimination_axis = axes[1].twinx()
    discrimination_axis.fill_between(
        q,
        0.0,
        comparison.pointwise_discrimination,
        color="#CC79A7",
        alpha=0.45,
        label=r"local $\mathcal{D}_i$",
    )
    axes[1].axhline(0.0, color="0.7", linewidth=0.7)
    axes[1].set_ylabel("Standardized\ndifference")
    discrimination_axis.set_ylabel(r"Local $\mathcal{D}_i$")
    discrimination_axis.legend(frameon=False, loc="upper right")

    for index, name in enumerate(diagnostics.parameter_names):
        axes[2].plot(q, diagnostics.local_information[:, index], label=name, linewidth=1.0)
    axes[2].set_yscale("symlog", linthresh=1e-5)
    axes[2].set_xlabel(r"$Q$ ($\AA^{-1}$)")
    axes[2].set_ylabel("Information per bin")
    axes[2].legend(frameon=False, ncols=2)
    axes[0].set_title(
        "One oxygen-coordinate perturbation: "
        rf"expected $\Delta\chi^2={comparison.total_discrimination:.1f}$"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    if show:
        plt.show()
    plt.close(figure)
    return comparison, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison, diagnostics = plot_information(args.output, show=args.show)
    print(f"wrote {args.output}")
    print(f"expected Delta chi^2: {comparison.total_discrimination:.3f}")
    print(f"expected separation: {np.sqrt(comparison.total_discrimination):.3f} sigma")
    print(
        f"Jacobian rank: {diagnostics.rank}/{len(diagnostics.parameter_names)}; "
        f"condition number: {diagnostics.condition_number:.3g}"
    )
    print("parameter guidance per declared 0.01 fractional-coordinate step:")
    order = np.argsort(np.abs(diagnostics.residual_support))[::-1]
    for index in order:
        print(
            f"  {diagnostics.parameter_names[index]:5s}  "
            f"sensitivity={diagnostics.sensitivity[index]:8.3f}  "
            f"residual_support={diagnostics.residual_support[index]:+8.3f}"
        )
    print("most discriminating Q bins:")
    for index in np.argsort(comparison.pointwise_discrimination)[-3:][::-1]:
        print(
            f"  Q={comparison.coordinate[index]:.3f}  "
            f"D_i={comparison.pointwise_discrimination[index]:.3f}"
        )


if __name__ == "__main__":
    main()
