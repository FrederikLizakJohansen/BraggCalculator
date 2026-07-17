import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from braggcalculator import BraggCalculator
from braggcalculator.backends import TorchBackend


def _mixed_perovskite():
    structure = Structure.from_spacegroup(
        "Pm-3m",
        Lattice.cubic(3.9),
        [{"Sr": 0.7, "Ca": 0.3}, "Ti", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]],
    )
    structure.add_site_property(
        "B",
        [
            0.4 if "Sr" in site.species else (0.3 if "Ti" in site.species else 0.7)
            for site in structure
        ],
    )
    return structure


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


@pytest.mark.parametrize(
    ("lattice", "crystal_system", "degrees_of_freedom"),
    [
        (Lattice.cubic(4.0), "cubic", 1),
        (Lattice.tetragonal(4.0, 5.0), "tetragonal", 2),
        (Lattice.hexagonal(4.0, 6.0), "hexagonal", 2),
        (Lattice.orthorhombic(4.0, 5.0, 6.0), "orthorhombic", 3),
        (Lattice.monoclinic(4.0, 5.0, 6.0, 105.0), "monoclinic", 4),
        (Lattice.from_parameters(4.0, 5.0, 6.0, 75.0, 85.0, 95.0), "triclinic", 6),
    ],
)
def test_lattice_parameterization_has_crystal_system_metric_dofs(
    lattice, crystal_system, degrees_of_freedom
):
    calculator = BraggCalculator(primitive=False).load(
        Structure(lattice, ["Si"], [[0, 0, 0]])
    )
    model = calculator.symmetry_lattice_parameterization()
    assert model.crystal_system == crystal_system
    assert model.independent_count == degrees_of_freedom
    np.testing.assert_allclose(
        model.expand(model.initial_values(calculator.backend), calculator.backend),
        lattice.matrix,
        atol=1e-12,
    )
    assert np.linalg.det(model.expand(np.ones(model.independent_count), calculator.backend)) > 0


def test_lattice_modes_recover_synthetic_reflection_positions():
    torch = pytest.importorskip("torch")
    structure = Structure(Lattice.tetragonal(4.0, 5.0), ["Si"], [[0, 0, 0]])
    calculator = BraggCalculator(
        primitive=False,
        backend=TorchBackend(),
        two_theta_range=(15.0, 75.0),
    ).load(structure)
    model = calculator.symmetry_lattice_parameterization()
    target_values = torch.tensor([0.45, -0.3], dtype=torch.float64)
    target_parameters = calculator.tensor_parameters()
    target_parameters["lattice"] = model.expand(target_values, calculator.backend)
    target_positions = calculator.iq(parameters=target_parameters)[0].detach()

    values = model.initial_values(calculator.backend, requires_grad=True)
    optimizer = torch.optim.Adam([values], lr=0.08)
    for _ in range(180):
        optimizer.zero_grad()
        parameters = calculator.tensor_parameters()
        parameters["lattice"] = model.expand(values, calculator.backend)
        positions = calculator.iq(parameters=parameters)[0]
        loss = torch.mean((positions - target_positions) ** 2)
        loss.backward()
        optimizer.step()
    torch.testing.assert_close(values, target_values, atol=2e-3, rtol=0)


def test_shared_site_composition_is_a_symmetry_shared_simplex():
    calculator = BraggCalculator(primitive=False).load(_mixed_perovskite())
    model = calculator.symmetry_occupancy_parameterization(mode="composition")
    assert model.independent_count == 1
    assert model.labels == ("orbit_0.Ca_vs_Sr",)
    values = model.initial_values(calculator.backend)
    np.testing.assert_allclose(model.expand(values, calculator.backend), calculator._symm["occ"])

    changed = values + 0.8
    expanded = model.expand(changed, calculator.backend)
    assert np.all(expanded >= 0)
    assert expanded[0] + expanded[1] == pytest.approx(1.0)
    np.testing.assert_allclose(expanded[2:], 1.0)


def test_vacancy_mode_bounds_every_orbit_occupancy_and_has_gradient():
    torch = pytest.importorskip("torch")
    calculator = BraggCalculator(
        primitive=False, backend=TorchBackend(), two_theta_range=(20, 70)
    ).load(_mixed_perovskite())
    model = calculator.symmetry_occupancy_parameterization(mode="vacancy")
    values = model.initial_values(calculator.backend, requires_grad=True)
    expanded = model.expand(values, calculator.backend)
    assert torch.all(expanded >= 0)
    for group in model.physical_groups(values.detach().numpy()):
        assert sum(group["species"].values()) <= 1.0
    parameters = calculator.tensor_parameters()
    parameters["occupancies"] = expanded
    loss = torch.sum(calculator.fq(parameters=parameters))
    loss.backward()
    assert torch.all(torch.isfinite(values.grad))
    assert torch.linalg.vector_norm(values.grad) > 0


def test_b_iso_is_positive_and_shared_across_symmetry_orbits():
    torch = pytest.importorskip("torch")
    calculator = BraggCalculator(
        primitive=False, backend=TorchBackend(), two_theta_range=(20, 70)
    ).load(_mixed_perovskite())
    model = calculator.symmetry_b_iso_parameterization()
    assert model.independent_count == 3
    values = model.initial_values(calculator.backend, requires_grad=True)
    expanded = model.expand(values, calculator.backend)
    torch.testing.assert_close(
        expanded,
        torch.as_tensor(calculator._symm["B"], dtype=torch.float64),
    )
    assert torch.all(expanded > 0)
    torch.testing.assert_close(expanded[3:], expanded[3].expand(3))
    parameters = calculator.tensor_parameters()
    parameters["b_iso"] = expanded
    loss = torch.sum(calculator.fq(parameters=parameters))
    loss.backward()
    assert torch.all(torch.isfinite(values.grad))
    assert torch.linalg.vector_norm(values.grad) > 0
