import numpy as np
import pytest

from braggcalculator import BraggCalculator
from braggcalculator.backends import NumpyBackend, TorchBackend

torch = pytest.importorskip("torch")


def test_numpy_torch_line_parity(strontium_titanate):
    numpy_calc = BraggCalculator(backend=NumpyBackend()).load(strontium_titanate)
    torch_calc = BraggCalculator(backend=TorchBackend()).load(strontium_titanate)
    nx, ny = numpy_calc.line_pattern(scaled=True)
    tx, ty = torch_calc.line_pattern(scaled=True)
    np.testing.assert_allclose(tx.cpu(), nx, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ty.cpu(), ny, rtol=1e-12, atol=1e-12)


def test_torch_profile_has_finite_parameter_gradients(triclinic_structure):
    calculator = BraggCalculator(
        backend=TorchBackend(),
        two_theta_step=0.05,
    ).load(triclinic_structure)
    parameters = calculator.tensor_parameters(requires_grad=True)
    grid, pattern = calculator.pattern(parameters=parameters)
    loss = torch.sum(pattern * torch.linspace(0.5, 1.5, len(grid)))
    loss.backward()
    for name, value in parameters.items():
        assert value.grad is not None, name
        assert torch.all(torch.isfinite(value.grad)), name


def test_coordinate_gradient_matches_central_difference(triclinic_structure):
    calculator = BraggCalculator(backend=TorchBackend()).load(triclinic_structure)
    parameters = calculator.tensor_parameters(requires_grad=["frac_coords"])
    _, intensities = calculator.iq(parameters=parameters)
    weights = torch.linspace(0.7, 1.3, len(intensities), dtype=torch.float64)
    loss = torch.sum(weights * intensities)
    loss.backward()
    analytical = parameters["frac_coords"].grad[0, 0].item()

    epsilon = 1e-6
    finite_values = []
    for delta in (-epsilon, epsilon):
        shifted = calculator.tensor_parameters()
        shifted["frac_coords"][0, 0] += delta
        finite_values.append(torch.sum(weights * calculator.iq(parameters=shifted)[1]).item())
    finite_difference = (finite_values[1] - finite_values[0]) / (2 * epsilon)
    assert analytical == pytest.approx(finite_difference, rel=2e-6, abs=1e-5)
