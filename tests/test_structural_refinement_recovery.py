import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from braggcalculator import BraggCalculator
from braggcalculator.backends import TorchBackend


def _match_f_squared(calculator, parameterization, target_values, parameter_name, *, steps, rate):
    torch = pytest.importorskip("torch")
    target_parameters = calculator.tensor_parameters()
    target_parameters[parameter_name] = parameterization.expand(
        target_values,
        calculator.backend,
    )
    target = calculator.fq(target_parameters).detach()
    values = parameterization.initial_values(
        calculator.backend,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([values], lr=rate)
    for _ in range(steps):
        optimizer.zero_grad()
        parameters = calculator.tensor_parameters()
        parameters[parameter_name] = parameterization.expand(
            values,
            calculator.backend,
        )
        predicted = calculator.fq(parameters)
        loss = torch.mean(((predicted - target) / target.max()) ** 2)
        loss.backward()
        optimizer.step()
    return values.detach(), loss.detach()


def test_fractional_coordinate_perturbation_is_recovered():
    torch = pytest.importorskip("torch")
    structure = Structure.from_spacegroup(
        "P-1",
        Lattice.from_parameters(4.2, 5.1, 6.3, 78, 82, 73),
        ["Si", "O"],
        [[0.13, 0.21, 0.34], [0, 0, 0]],
    )
    calculator = BraggCalculator(
        primitive=False,
        backend=TorchBackend(),
        q_range=(0.5, 5.0),
    ).load(structure)
    model = calculator.symmetry_coordinate_parameterization()
    target = torch.tensor([0.012, -0.018, 0.021], dtype=torch.float64)
    recovered, loss = _match_f_squared(
        calculator,
        model,
        target,
        "frac_coords",
        steps=250,
        rate=0.03,
    )
    torch.testing.assert_close(recovered, target, atol=5e-7, rtol=0)
    assert loss < 1e-12


def test_occupancy_and_isotropic_displacement_perturbations_are_recovered():
    torch = pytest.importorskip("torch")
    lattice = Lattice.from_parameters(4.2, 5.1, 6.3, 78, 82, 73)
    occupancy_structure = Structure(
        lattice,
        [{"Sr": 0.7, "Ca": 0.3}, "Ti", "O"],
        [[0.13, 0.21, 0.34], [0.37, 0.42, 0.11], [0.67, 0.12, 0.45]],
    )
    occupancy_calculator = BraggCalculator(
        primitive=False,
        backend=TorchBackend(),
        q_range=(0.5, 7.0),
    ).load(occupancy_structure)
    occupancy_model = occupancy_calculator.symmetry_occupancy_parameterization(
        mode="composition"
    )
    occupancy_target = occupancy_model.initial_values(occupancy_calculator.backend) + 0.5
    recovered_occupancy, occupancy_loss = _match_f_squared(
        occupancy_calculator,
        occupancy_model,
        occupancy_target,
        "occupancies",
        steps=250,
        rate=0.08,
    )
    torch.testing.assert_close(recovered_occupancy, occupancy_target, atol=2e-6, rtol=0)
    assert occupancy_loss < 1e-12

    displacement_structure = Structure(
        lattice,
        ["Si", "O"],
        [[0.13, 0.21, 0.34], [0.37, 0.42, 0.11]],
        site_properties={"B_iso": [0.3, 0.7]},
    )
    displacement_calculator = BraggCalculator(
        primitive=False,
        backend=TorchBackend(),
        q_range=(0.5, 7.0),
    ).load(displacement_structure)
    displacement_model = displacement_calculator.symmetry_b_iso_parameterization()
    displacement_target = displacement_model.initial_values(
        displacement_calculator.backend
    ) + torch.tensor([0.4, -0.3], dtype=torch.float64)
    recovered_displacement, displacement_loss = _match_f_squared(
        displacement_calculator,
        displacement_model,
        displacement_target,
        "b_iso",
        steps=300,
        rate=0.05,
    )
    torch.testing.assert_close(
        recovered_displacement,
        displacement_target,
        atol=2e-6,
        rtol=0,
    )
    assert displacement_loss < 1e-12


def test_anisotropic_displacement_perturbation_is_recovered():
    torch = pytest.importorskip("torch")
    structure = Structure(
        Lattice.from_parameters(4.2, 5.1, 6.3, 78, 82, 73),
        ["Si", "O"],
        [[0.13, 0.21, 0.34], [0.37, 0.42, 0.11]],
        site_properties={"U_cart": [np.eye(3) * 0.006, np.eye(3) * 0.009]},
    )
    calculator = BraggCalculator(
        primitive=False,
        backend=TorchBackend(),
        q_range=(0.5, 8.0),
    ).load(structure)
    model = calculator.symmetry_u_aniso_parameterization()
    initial = model.initial_values(calculator.backend)
    target = initial + torch.linspace(-0.15, 0.15, len(initial), dtype=torch.float64)
    recovered, loss = _match_f_squared(
        calculator,
        model,
        target,
        "u_cart",
        steps=600,
        rate=0.03,
    )
    torch.testing.assert_close(recovered, target, atol=2e-6, rtol=0)
    assert loss < 1e-12
