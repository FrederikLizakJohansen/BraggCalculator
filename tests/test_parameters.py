import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from braggcalculator import BraggCalculator
from braggcalculator.backends import TorchBackend


def test_p1_has_three_independent_coordinates_per_site(triclinic_structure):
    calculator = BraggCalculator(primitive=False).load(triclinic_structure)
    model = calculator.symmetry_coordinate_parameterization()
    assert calculator._symm["spacegroup_symbol"] == "P1"
    assert model.independent_count == 3 * len(triclinic_structure)
    values = model.initial_values(calculator.backend)
    np.testing.assert_allclose(model.expand(values, calculator.backend), calculator._symm["frac_coords"])


def test_inversion_mates_receive_opposite_displacements():
    lattice = Lattice.from_parameters(4.2, 5.1, 6.3, 78, 82, 73)
    structure = Structure.from_spacegroup("P-1", lattice, ["Si"], [[0.13, 0.21, 0.34]])
    calculator = BraggCalculator(primitive=False).load(structure)
    model = calculator.symmetry_coordinate_parameterization()
    assert model.independent_count == 3
    displacement = np.array([0.01, -0.02, 0.03])
    expanded = model.expand(displacement, calculator.backend)
    change = expanded - calculator._symm["frac_coords"]
    np.testing.assert_allclose(change[0], displacement)
    np.testing.assert_allclose(change[1], -displacement)


def test_fixed_special_positions_have_no_coordinate_degrees_of_freedom(strontium_titanate):
    calculator = BraggCalculator(primitive=False).load(strontium_titanate)
    model = calculator.symmetry_coordinate_parameterization()
    assert calculator._symm["spacegroup_symbol"] == "Pm-3m"
    assert model.independent_count == 0
    expanded = model.expand(model.initial_values(calculator.backend), calculator.backend)
    np.testing.assert_allclose(expanded, calculator._symm["frac_coords"])


def test_symmetry_expansion_preserves_torch_gradient():
    torch = pytest.importorskip("torch")
    lattice = Lattice.from_parameters(4.2, 5.1, 6.3, 78, 82, 73)
    structure = Structure.from_spacegroup(
        "P-1", lattice, ["Si", "O"], [[0.13, 0.21, 0.34], [0.0, 0.0, 0.0]]
    )
    calculator = BraggCalculator(
        primitive=False,
        backend=TorchBackend(),
        q_range=(0.5, 4.0),
        q_step=0.05,
    ).load(structure)
    model = calculator.symmetry_coordinate_parameterization()
    values = model.initial_values(calculator.backend, requires_grad=True)
    parameters = model.forward_parameters(calculator, values)
    _, profile = calculator.pattern(domain="q", parameters=parameters)
    weights = torch.linspace(0.5, 1.5, len(profile), dtype=profile.dtype)
    loss = torch.sum(weights * profile)
    loss.backward()
    assert values.grad is not None
    assert torch.all(torch.isfinite(values.grad))
    assert torch.linalg.vector_norm(values.grad) > 0
