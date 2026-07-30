import pytest
from pymatgen.core import Lattice, Structure

from braggcalculator import BraggCalculator, StructuralRestraintSet
from braggcalculator.backends import TorchBackend


def _three_atom_structure():
    return Structure(
        Lattice.cubic(10.0),
        ["O", "Si", "O"],
        [[0.30, 0.50, 0.50], [0.50, 0.50, 0.50], [0.45, 0.65, 0.50]],
    )


def test_structural_restraints_report_separate_standardized_contributions():
    calculator = BraggCalculator(primitive=False).load(_three_atom_structure())
    restraints = StructuralRestraintSet.from_dict(
        calculator,
        {
            "composition": [{"species": "O", "target": 2.0, "sigma": 0.1}],
            "bonds": [{"sites": [0, 1], "target": 2.0, "sigma": 0.1}],
            "angles": [
                {
                    "sites": [0, 1, 2],
                    "target_degrees": 90.0,
                    "sigma_degrees": 2.0,
                }
            ],
            "minimum_distances": [{"sites": [0, 2], "minimum": 2.0, "sigma": 0.1}],
        },
    )
    parameters = calculator.tensor_parameters()
    loss, contributions = restraints.loss(
        parameters["lattice"],
        parameters["frac_coords"],
        parameters["occupancies"],
        calculator.backend,
    )
    assert restraints.count == 4
    assert set(contributions) == {
        "composition[0].O",
        "bond[0]",
        "angle[0]",
        "minimum_distance[0]",
    }
    assert loss >= 0
    assert contributions["composition[0].O"] == pytest.approx(0.0)
    assert contributions["bond[0]"] == pytest.approx(0.0)
    assert contributions["minimum_distance[0]"] == pytest.approx(0.0)


def test_bond_and_angle_restraints_are_differentiable_and_restore_geometry():
    torch = pytest.importorskip("torch")
    structure = _three_atom_structure()
    calculator = BraggCalculator(primitive=False, backend=TorchBackend()).load(structure)
    restraints = StructuralRestraintSet.from_dict(
        calculator,
        {
            "bonds": [
                {"sites": [0, 1], "target": 1.7, "sigma": 0.02},
                {"sites": [1, 2], "target": 1.7, "sigma": 0.02},
            ],
            "angles": [
                {
                    "sites": [0, 1, 2],
                    "target_degrees": 105.0,
                    "sigma_degrees": 1.0,
                }
            ],
        },
    )
    coordinates = calculator.tensor_parameters()["frac_coords"].clone().requires_grad_(True)
    optimizer = torch.optim.Adam([coordinates], lr=0.01)
    for _ in range(350):
        optimizer.zero_grad()
        parameters = calculator.tensor_parameters()
        loss, _ = restraints.loss(
            parameters["lattice"], coordinates, parameters["occupancies"], calculator.backend
        )
        loss.backward()
        optimizer.step()
    assert loss.item() < 1e-8
    assert torch.all(torch.isfinite(coordinates.grad))


def test_minimum_distance_penalty_activates_only_below_limit():
    calculator = BraggCalculator(primitive=False).load(_three_atom_structure())
    parameters = calculator.tensor_parameters()
    inactive = StructuralRestraintSet.from_dict(
        calculator,
        {"minimum_distances": [{"sites": [0, 1], "minimum": 1.5, "sigma": 0.1}]},
    )
    active = StructuralRestraintSet.from_dict(
        calculator,
        {"minimum_distances": [{"sites": [0, 1], "minimum": 2.5, "sigma": 0.1}]},
    )
    inactive_loss, _ = inactive.loss(
        parameters["lattice"],
        parameters["frac_coords"],
        parameters["occupancies"],
        calculator.backend,
    )
    active_loss, _ = active.loss(
        parameters["lattice"],
        parameters["frac_coords"],
        parameters["occupancies"],
        calculator.backend,
    )
    assert inactive_loss == pytest.approx(0.0)
    assert active_loss == pytest.approx(25.0)
