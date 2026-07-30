"""Bounds-aware parametric-bootstrap uncertainty utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    """Empirical parameter intervals conditional on a fitted observation model."""

    parameter_names: tuple[str, ...]
    point_estimate: np.ndarray
    bootstrap_estimates: np.ndarray
    standard_error: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    confidence_level: float
    bounds: np.ndarray
    boundary_hits: np.ndarray
    requested_draws: int
    successful_draws: int
    failed_draws: int
    seed: int
    noise_model: str

    def contains(self, values) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.shape != self.point_estimate.shape:
            raise ValueError("values must match the bootstrap parameter shape")
        return (self.lower <= values) & (values <= self.upper)


def parametric_bootstrap(
    observed,
    fitted_expected,
    estimator: Callable[[np.ndarray], np.ndarray],
    *,
    sigma=None,
    covariance=None,
    draws: int = 400,
    confidence_level: float = 0.95,
    bounds=None,
    parameter_names=None,
    seed: int = 0,
) -> BootstrapResult:
    """Calculate percentile intervals from a declared Gaussian noise model.

    The estimator is reapplied to every simulated observation. Bounds are
    enforced by clipping the estimator output and every clipping event is
    counted; a pile-up at a bound is therefore visible rather than hidden.
    """
    observed = _observation_vector("observed", observed)
    expected = _observation_vector("fitted_expected", fitted_expected)
    if expected.shape != observed.shape:
        raise ValueError("observed and fitted_expected must have equal shapes")
    if (sigma is None) == (covariance is None):
        raise ValueError("provide exactly one of sigma or covariance")
    if not isinstance(draws, int) or draws < 2:
        raise ValueError("draws must be an integer of at least two")
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")

    point = _parameter_vector(estimator(observed))
    parameter_count = len(point)
    names = (
        tuple(f"p{index}" for index in range(parameter_count))
        if parameter_names is None
        else tuple(str(name) for name in parameter_names)
    )
    if len(names) != parameter_count or any(not name for name in names):
        raise ValueError("parameter_names must match the estimator output")
    limits = _parameter_bounds(bounds, parameter_count)
    point, _ = _apply_bounds(point, limits)

    rng = np.random.default_rng(int(seed))
    if covariance is not None:
        covariance_array = np.asarray(covariance, dtype=np.float64)
        expected_shape = (len(observed), len(observed))
        if covariance_array.shape != expected_shape or not np.all(np.isfinite(covariance_array)):
            raise ValueError(f"covariance must be a finite matrix with shape {expected_shape}")
        if not np.allclose(covariance_array, covariance_array.T, rtol=1e-12, atol=1e-14):
            raise ValueError("covariance must be symmetric")
        try:
            cholesky = np.linalg.cholesky(covariance_array)
        except np.linalg.LinAlgError as error:
            raise ValueError("covariance must be positive definite") from error

        def simulate():
            return expected + cholesky @ rng.standard_normal(len(expected))

        noise_model = "correlated Gaussian covariance"
    else:
        sigma_array = np.asarray(sigma, dtype=np.float64)
        if (
            sigma_array.shape != observed.shape
            or np.any(sigma_array <= 0)
            or not np.all(np.isfinite(sigma_array))
        ):
            raise ValueError("sigma must be one positive finite value per observation")

        def simulate():
            return expected + sigma_array * rng.standard_normal(len(expected))

        noise_model = "independent Gaussian sigma"

    estimates = []
    boundary_hits = np.zeros((parameter_count, 2), dtype=np.int64)
    failures = 0
    for _ in range(draws):
        try:
            estimate = _parameter_vector(estimator(simulate()))
            if estimate.shape != point.shape:
                raise ValueError("estimator output shape changed during bootstrap")
            estimate, hits = _apply_bounds(estimate, limits)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
            failures += 1
            continue
        estimates.append(estimate)
        boundary_hits += hits
    if len(estimates) < 2:
        raise RuntimeError("fewer than two bootstrap replicates succeeded")
    samples = np.asarray(estimates)
    tail = 0.5 * (1.0 - confidence_level)
    lower, upper = np.quantile(samples, [tail, 1.0 - tail], axis=0)
    return BootstrapResult(
        parameter_names=names,
        point_estimate=point,
        bootstrap_estimates=samples,
        standard_error=np.std(samples, axis=0, ddof=1),
        lower=lower,
        upper=upper,
        confidence_level=float(confidence_level),
        bounds=limits,
        boundary_hits=boundary_hits,
        requested_draws=draws,
        successful_draws=len(samples),
        failed_draws=failures,
        seed=int(seed),
        noise_model=noise_model,
    )


def _observation_vector(name, values):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or len(result) == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return result


def _parameter_vector(values):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or len(result) == 0 or not np.all(np.isfinite(result)):
        raise ValueError("estimator must return a non-empty finite vector")
    return result


def _parameter_bounds(bounds, parameter_count):
    if bounds is None:
        return np.tile([-np.inf, np.inf], (parameter_count, 1))
    result = np.asarray(bounds, dtype=np.float64)
    if result.shape != (parameter_count, 2) or np.any(result[:, 0] >= result[:, 1]):
        raise ValueError("bounds must contain one increasing [lower, upper] pair per parameter")
    if np.any(np.isnan(result)):
        raise ValueError("bounds cannot contain NaN")
    return result


def _apply_bounds(values, bounds):
    below = values <= bounds[:, 0]
    above = values >= bounds[:, 1]
    hits = np.stack([below, above], axis=1).astype(np.int64)
    return np.clip(values, bounds[:, 0], bounds[:, 1]), hits
