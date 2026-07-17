#!/usr/bin/env python3
"""Recover a synthetic tetragonal cell through two symmetry-allowed modes."""

from __future__ import annotations

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from braggcalculator import BraggCalculator, lattice_parameters
from braggcalculator.backends import TorchBackend


def run_lattice_refinement(steps: int = 180):
    """Recover two tetragonal metric modes and return numerical evidence."""
    structure = Structure(Lattice.tetragonal(4.0, 5.0), ["Si"], [[0, 0, 0]])
    calculator = BraggCalculator(
        primitive=False,
        backend=TorchBackend(),
        two_theta_range=(15.0, 75.0),
    ).load(structure)
    model = calculator.symmetry_lattice_parameterization()
    target_modes = torch.tensor([0.45, -0.30], dtype=torch.float64)
    target_lattice = model.expand(target_modes, calculator.backend)
    target_parameters = calculator.tensor_parameters()
    target_parameters["lattice"] = target_lattice
    target_positions = calculator.iq(parameters=target_parameters)[0].detach()

    modes = model.initial_values(calculator.backend, requires_grad=True)
    optimizer = torch.optim.Adam([modes], lr=0.08)
    history = []
    for _ in range(steps):
        optimizer.zero_grad()
        parameters = calculator.tensor_parameters()
        parameters["lattice"] = model.expand(modes, calculator.backend)
        positions = calculator.iq(parameters=parameters)[0]
        loss = torch.mean((positions - target_positions) ** 2)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))

    recovered_lattice = model.expand(modes, calculator.backend).detach().cpu().numpy()
    target_cell = lattice_parameters(target_lattice.detach().cpu().numpy())
    recovered_cell = lattice_parameters(recovered_lattice)
    maximum_error = max(abs(target_cell[key] - recovered_cell[key]) for key in target_cell)
    assert model.crystal_system == "tetragonal"
    assert model.independent_count == 2
    assert maximum_error < 5e-5

    return {
        "crystal_system": model.crystal_system,
        "independent_count": model.independent_count,
        "target_modes": target_modes.numpy(),
        "recovered_modes": modes.detach().numpy(),
        "target_cell": target_cell,
        "recovered_cell": recovered_cell,
        "maximum_error": maximum_error,
        "history": np.asarray(history),
    }


def main():
    result = run_lattice_refinement()

    print(f"crystal system: {result['crystal_system']}")
    print(f"independent lattice modes: {result['independent_count']}")
    print(f"target cell: {result['target_cell']}")
    print(f"recovered cell: {result['recovered_cell']}")
    print(f"maximum cell-parameter error: {result['maximum_error']:.3e}")
    print(
        "mode error norm: "
        f"{np.linalg.norm(result['recovered_modes'] - result['target_modes']):.3e}"
    )


if __name__ == "__main__":
    main()
