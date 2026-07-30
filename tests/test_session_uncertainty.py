import numpy as np
import pytest

from braggcalculator.session import _local_identifiability, _weighted_squared_norm


def test_session_identifiability_separates_data_and_prior_rank():
    torch = pytest.importorskip("torch")
    parameters = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    coordinate = torch.linspace(0.5, 2.0, 12, dtype=torch.float64)

    def calculate():
        return coordinate * (parameters[0] + parameters[1])

    def prior_residuals():
        return parameters[1:2] / 0.2

    diagnostics = _local_identifiability(
        calculate,
        {"structural": parameters},
        np.ones(len(coordinate)),
        np.ones(len(coordinate), dtype=bool),
        max_points=len(coordinate),
        group_labels={"structural": ("occupancy", "Biso")},
        group_scales={"structural": (0.05, 0.1)},
        group_step_descriptions={"structural": ("0.05 occupancy", "0.1 square-angstrom Biso")},
        prior_residuals=prior_residuals,
    )

    assert diagnostics["data_rank"] == 1
    assert diagnostics["posterior_rank"] == 2
    assert not diagnostics["data_covariance_is_identifiable"]
    assert diagnostics["posterior_covariance_is_identifiable"]
    assert diagnostics["null_directions"]
    assert diagnostics["characteristic_step_descriptions"][0] == "0.05 occupancy"


def test_weighted_squared_norm_uses_full_covariance():
    values = np.array([1.0, -0.5, 0.25])
    covariance = np.array([[2.0, 0.4, 0.0], [0.4, 1.5, 0.2], [0.0, 0.2, 1.0]])
    sigma = np.sqrt(np.diag(covariance))
    selected = np.array([True, True, True])

    result = _weighted_squared_norm(values, selected, sigma, covariance)
    assert result == pytest.approx(values @ np.linalg.solve(covariance, values))
