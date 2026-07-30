import numpy as np
import pytest

from braggcalculator import parametric_bootstrap


def test_correlated_parametric_bootstrap_matches_linear_standard_error():
    coordinate = np.linspace(-1.0, 1.0, 20)
    design = np.column_stack([np.ones(len(coordinate)), coordinate])
    truth = np.array([3.0, -0.7])
    indices = np.arange(len(coordinate))
    covariance = 0.04 * 0.55 ** np.abs(indices[:, None] - indices[None, :])
    precision_design = np.linalg.solve(covariance, design)
    normal = design.T @ precision_design

    def estimator(values):
        return np.linalg.solve(normal, precision_design.T @ values)

    expected = design @ truth
    observed = expected + np.linspace(-0.05, 0.05, len(expected))
    result = parametric_bootstrap(
        observed,
        expected,
        estimator,
        covariance=covariance,
        draws=1200,
        parameter_names=("intercept", "slope"),
        seed=1729,
    )
    analytic = np.sqrt(np.diag(np.linalg.inv(normal)))

    np.testing.assert_allclose(result.standard_error, analytic, rtol=0.08)
    assert np.all(result.contains(truth))
    assert result.failed_draws == 0
    assert result.noise_model == "correlated Gaussian covariance"


def test_bootstrap_reports_boundary_pileup():
    expected = np.full(12, 0.03)
    observed = expected.copy()

    def nonnegative_mean(values):
        return np.array([np.mean(values)])

    result = parametric_bootstrap(
        observed,
        expected,
        nonnegative_mean,
        sigma=np.full(len(expected), 0.2),
        draws=600,
        bounds=[(0.0, np.inf)],
        parameter_names=("trace fraction",),
        seed=81,
    )

    assert result.boundary_hits[0, 0] > 100
    assert result.lower[0] == pytest.approx(0.0)
    assert result.upper[0] > result.point_estimate[0]


def test_bootstrap_records_failed_estimator_replicates():
    def conditionally_failing_estimator(values):
        if values[0] < 0:
            raise RuntimeError("synthetic optimizer failure")
        return np.array([np.mean(values)])

    result = parametric_bootstrap(
        np.ones(8),
        np.zeros(8),
        conditionally_failing_estimator,
        sigma=np.ones(8),
        draws=200,
        seed=99,
    )
    assert 50 < result.failed_draws < 150
    assert result.successful_draws + result.failed_draws == result.requested_draws


def test_repeated_synthetic_bootstrap_has_expected_coverage():
    rng = np.random.default_rng(20260717)
    truth = 0.4
    sigma = np.full(16, 0.3)

    def estimator(values):
        return np.array([np.mean(values)])

    covered = []
    for repeat in range(40):
        observed = truth + sigma * rng.standard_normal(len(sigma))
        fitted = np.full(len(sigma), np.mean(observed))
        result = parametric_bootstrap(
            observed,
            fitted,
            estimator,
            sigma=sigma,
            draws=160,
            confidence_level=0.9,
            seed=1000 + repeat,
        )
        covered.append(bool(result.contains([truth])[0]))

    coverage = np.mean(covered)
    assert 0.78 <= coverage <= 1.0
