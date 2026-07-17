#!/usr/bin/env python3
"""Jointly recover structural and profile parameters with declared stages."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure

from braggcalculator import (
    BraggCalculator,
    OptimizationStage,
    ProfileNuisanceParameterization,
    staged_adam,
)
from braggcalculator.backends import TorchBackend


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "staged_refinement.png"


def run_staged_example():
    """Run the synthetic joint refinement and return numerical results."""
    import torch

    lattice = Lattice.from_parameters(4.2, 5.1, 6.3, 78.0, 82.0, 73.0)
    structure = Structure.from_spacegroup(
        "P-1", lattice, ["Si", "O"], [[0.13, 0.21, 0.34], [0.0, 0.0, 0.0]]
    )
    calculator = BraggCalculator(
        primitive=False,
        backend=TorchBackend(),
        q_range=(0.5, 5.0),
        q_step=0.03,
    ).load(structure)
    coordinate_model = calculator.symmetry_coordinate_parameterization()
    nuisance_model = ProfileNuisanceParameterization.from_calculator(
        calculator,
        domain="q",
        initial_background=20.0,
        zero_shift_scale=0.01,
    )

    target_coordinates = torch.tensor([0.012, -0.007, 0.006], dtype=torch.float64)
    target_raw = {
        "scale": torch.tensor(np.log(1.25), dtype=torch.float64),
        "background": torch.tensor(np.log(2.5), dtype=torch.float64),
        "zero_shift": torch.tensor(1.2, dtype=torch.float64),
        "fwhm": torch.tensor(np.log(1.5), dtype=torch.float64),
    }
    target_nuisance = nuisance_model.physical(target_raw, calculator.backend)
    _, target_profile = calculator.pattern(
        domain="q",
        parameters=coordinate_model.forward_parameters(calculator, target_coordinates),
        experiment_parameters=target_nuisance,
    )
    target_profile = target_profile.detach()

    coordinates = coordinate_model.initial_values(calculator.backend, requires_grad=True)
    raw_nuisance = nuisance_model.initial_values(calculator.backend, requires_grad=True)
    groups = {"coordinates": coordinates, **raw_nuisance}
    weights = 1.0 / (target_profile + 100.0)

    def objective():
        _, profile = calculator.pattern(
            domain="q",
            parameters=coordinate_model.forward_parameters(calculator, coordinates),
            experiment_parameters=nuisance_model.physical(raw_nuisance, calculator.backend),
        )
        return torch.mean(weights * (profile - target_profile) ** 2)

    stages = (
        OptimizationStage("scale/background", ("scale", "background"), 120, 0.03),
        OptimizationStage("position/width", ("zero_shift", "fwhm"), 180, 0.02),
        OptimizationStage("coordinates", ("coordinates",), 220, 0.006),
        OptimizationStage("joint", tuple(groups), 1000, 0.003),
        OptimizationStage("joint polish", tuple(groups), 1000, 0.001),
        OptimizationStage("background cleanup", ("background",), 250, 0.02),
    )
    trace = staged_adam(objective, groups, stages)
    recovered_nuisance_tensors = nuisance_model.physical(raw_nuisance, calculator.backend)
    target_physical = {name: float(value) for name, value in target_nuisance.items()}
    recovered_physical = {
        name: float(value.detach()) for name, value in recovered_nuisance_tensors.items()
    }
    return {
        "trace": trace,
        "target_coordinates": target_coordinates.numpy(),
        "recovered_coordinates": coordinates.detach().numpy(),
        "target_physical": target_physical,
        "recovered_physical": recovered_physical,
    }


def plot_staged_refinement(output: Path, *, result=None, show: bool = False):
    """Write the stage trace and target/recovered parameter comparison."""
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    result = run_staged_example() if result is None else result
    trace = result["trace"]
    figure, (loss_axis, parameter_axis) = plt.subplots(
        1, 2, figsize=(9.0, 3.8), layout="constrained"
    )
    loss_axis.semilogy(trace.loss, color="#0072B2", linewidth=1.0)
    boundaries = np.flatnonzero(np.asarray(trace.stage[1:]) != np.asarray(trace.stage[:-1])) + 1
    for boundary in boundaries:
        loss_axis.axvline(boundary, color="0.75", linewidth=0.7)
    loss_axis.set(xlabel="Optimization step", ylabel="Weighted profile loss")

    names = ("u₀", "u₁", "u₂", "scale", "background", "zero shift", "FWHM")
    target = np.r_[
        result["target_coordinates"],
        [result["target_physical"][name] for name in ("scale", "background", "zero_shift", "fwhm")],
    ]
    recovered = np.r_[
        result["recovered_coordinates"],
        [
            result["recovered_physical"][name]
            for name in ("scale", "background", "zero_shift", "fwhm")
        ],
    ]
    ratio = np.divide(recovered, target, out=np.ones_like(recovered), where=target != 0)
    positions = np.arange(len(names))
    parameter_axis.bar(positions, ratio, color="#009E73")
    parameter_axis.axhline(1.0, color="0.25", linewidth=0.8)
    parameter_axis.set(
        xticks=positions,
        xticklabels=names,
        ylabel="Recovered / target",
        ylim=(0.9, 1.1),
        title="Final parameter recovery",
    )
    parameter_axis.tick_params(axis="x", rotation=40)
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
    result = plot_staged_refinement(args.output, show=args.show)
    print(f"wrote {args.output}")
    print(
        "coordinates target/recovered: "
        f"{np.array2string(result['target_coordinates'], precision=6)} / "
        f"{np.array2string(result['recovered_coordinates'], precision=6)}"
    )
    for name in ("scale", "background", "zero_shift", "fwhm"):
        print(
            f"{name:11s}: target={result['target_physical'][name]:.7g}  "
            f"recovered={result['recovered_physical'][name]:.7g}"
        )
    print(
        f"weighted loss: {result['trace'].loss[0]:.3e} -> "
        f"{result['trace'].loss[-1]:.3e}"
    )


if __name__ == "__main__":
    main()
