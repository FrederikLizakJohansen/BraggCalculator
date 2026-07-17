#!/usr/bin/env python3
"""Recover a synthetic displacement with a symmetry-constrained parameterization."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure

from braggcalculator import BraggCalculator
from braggcalculator.backends import TorchBackend


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "symmetry_refinement.png"
TARGET_DISPLACEMENT = np.array([0.015, -0.010, 0.008])


def run_refinement(steps: int = 220):
    """Return target/recovered displacements, loss history, and orbit changes."""
    import torch

    lattice = Lattice.from_parameters(4.2, 5.1, 6.3, 78.0, 82.0, 73.0)
    structure = Structure.from_spacegroup(
        "P-1",
        lattice,
        ["Si", "O"],
        [[0.13, 0.21, 0.34], [0.0, 0.0, 0.0]],
    )
    calculator = BraggCalculator(
        primitive=False,
        backend=TorchBackend(),
        q_range=(0.5, 5.0),
        q_step=0.03,
    ).load(structure)
    coordinate_model = calculator.symmetry_coordinate_parameterization()
    if coordinate_model.independent_count != 3:
        raise RuntimeError("the example expects one three-coordinate general-position orbit")

    target_values = torch.as_tensor(TARGET_DISPLACEMENT, dtype=torch.float64)
    target_parameters = coordinate_model.forward_parameters(calculator, target_values)
    _, target_profile = calculator.pattern(domain="q", parameters=target_parameters)
    target_profile = target_profile.detach()

    values = coordinate_model.initial_values(calculator.backend, requires_grad=True)
    optimizer = torch.optim.Adam([values], lr=0.003)
    profile_scale = torch.mean(target_profile**2)
    history = []
    for _ in range(steps):
        optimizer.zero_grad()
        parameters = coordinate_model.forward_parameters(calculator, values)
        _, profile = calculator.pattern(domain="q", parameters=parameters)
        loss = torch.mean((profile - target_profile) ** 2) / profile_scale
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))

    initial_coordinates = np.asarray(calculator._symm["frac_coords"])
    refined_coordinates = (
        coordinate_model.expand(values.detach(), calculator.backend).cpu().numpy()
    )
    orbit_change = refined_coordinates[:2] - initial_coordinates[:2]
    return (
        TARGET_DISPLACEMENT.copy(),
        values.detach().cpu().numpy(),
        np.asarray(history),
        orbit_change,
    )


def plot_refinement(output: Path, *, show: bool = False):
    """Write a convergence/recovery figure and return the numerical results."""
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    target, recovered, history, orbit_change = run_refinement()
    figure, (loss_axis, coordinate_axis) = plt.subplots(
        1, 2, figsize=(8.0, 3.6), layout="constrained"
    )
    loss_axis.semilogy(history, color="#0072B2")
    loss_axis.set(xlabel="Adam step", ylabel="Normalized profile loss", title="Convergence")

    positions = np.arange(3)
    coordinate_axis.bar(positions - 0.18, target, 0.36, label="target", color="#D55E00")
    coordinate_axis.bar(positions + 0.18, recovered, 0.36, label="recovered", color="#0072B2")
    coordinate_axis.axhline(0.0, color="0.65", linewidth=0.7)
    coordinate_axis.set(
        xticks=positions,
        xticklabels=("u₀", "u₁", "u₂"),
        ylabel="Fractional displacement",
        title="Independent coordinates",
    )
    coordinate_axis.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    if show:
        plt.show()
    plt.close(figure)
    return target, recovered, history, orbit_change


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target, recovered, history, orbit_change = plot_refinement(args.output, show=args.show)
    print(f"wrote {args.output}")
    print("target displacement:    " + np.array2string(target, precision=7))
    print("recovered displacement: " + np.array2string(recovered, precision=7))
    print(f"maximum parameter error: {np.max(np.abs(recovered - target)):.3e}")
    print(f"initial loss: {history[0]:.3e}; final loss: {history[-1]:.3e}")
    print("general-position orbit changes:")
    print(np.array2string(orbit_change, precision=7))


if __name__ == "__main__":
    main()
