import numpy as np
import pytest

from braggcalculator import BraggCalculator
from braggcalculator.backends import TorchBackend
from braggcalculator.sensitivity import (
    ParameterPath,
    analyze_jacobian,
    torch_profile_jacobian,
)

torch = pytest.importorskip("torch")


def test_scaled_jacobian_diagnostics_have_expected_sensitivity_and_support():
    jacobian = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 2.0]])
    weights = np.array([1.0, 4.0, 1.0])
    scales = np.array([0.5, 2.0])
    residual = np.array([1.0, -1.0, 0.5])
    result = analyze_jacobian(
        jacobian,
        residual=residual,
        weights=weights,
        parameter_scales=scales,
        parameter_names=["x", "y"],
    )
    scaled = jacobian * scales
    whitened = np.sqrt(weights)[:, None] * scaled
    expected_sensitivity = np.sqrt(np.sum(whitened**2, axis=0))
    expected_support = whitened.T @ (np.sqrt(weights) * residual) / expected_sensitivity
    np.testing.assert_allclose(result.sensitivity, expected_sensitivity)
    np.testing.assert_allclose(result.residual_support, expected_support)
    np.testing.assert_allclose(result.local_information, weights[:, None] * scaled**2)
    assert result.parameter_names == ("x", "y")
    assert result.covariance_is_identifiable


def test_rank_deficiency_is_reported_instead_of_claiming_covariance():
    jacobian = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    result = analyze_jacobian(jacobian)
    assert result.rank == 1
    assert np.isinf(result.condition_number)
    assert not result.covariance_is_identifiable
    assert abs(result.column_cosine[0, 1]) == pytest.approx(1.0)


def test_full_covariance_jacobian_matches_manual_whitening():
    jacobian = np.array([[1.0, 0.2], [0.3, 2.0], [0.5, -0.1]])
    covariance = np.array([[2.0, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 1.5]])
    result = analyze_jacobian(jacobian, covariance=covariance)
    cholesky = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(cholesky, jacobian)
    np.testing.assert_allclose(result.normal_matrix, whitened.T @ whitened)
    assert result.local_information is None


def test_torch_profile_jacobian_matches_central_difference(triclinic_structure):
    calculator = BraggCalculator(
        backend=TorchBackend(), q_range=(0.5, 4.0), q_step=0.05
    ).load(triclinic_structure)
    parameters = calculator.tensor_parameters()
    path = ParameterPath("frac_coords", (1, 0), scale=0.01, label="O1 x")
    grid, profile, jacobian = torch_profile_jacobian(
        calculator, parameters, [path], domain="q"
    )
    assert jacobian.shape == (len(grid), 1)
    assert profile.shape == grid.shape

    epsilon = 1e-6
    finite_profiles = []
    for delta in (-epsilon, epsilon):
        shifted = calculator.tensor_parameters()
        shifted["frac_coords"][1, 0] += delta
        finite_profiles.append(calculator.pattern(domain="q", parameters=shifted)[1].cpu().numpy())
    finite_difference = (finite_profiles[1] - finite_profiles[0]) / (2 * epsilon)
    np.testing.assert_allclose(jacobian[:, 0], finite_difference, rtol=2e-5, atol=1e-5)
